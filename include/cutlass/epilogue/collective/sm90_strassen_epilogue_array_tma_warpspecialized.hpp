/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"
#include "cutlass/detail/collective.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/trace.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/gemm/strassen_group_array_problem_shape.hpp"

#include "cute/tensor.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace MmaStrassen;

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class StrassenMiGroup_,
  int StagesC_,
  int StagesD_,
  int FragmentSize_,
  bool ReuseSmemC_,
  bool DelayTmaStore_,
  int NumEpilogueWarpGroups_,
  class CtaTileMNK_,   //     (CTA_M,CTA_N,CTA_K)
  class EpilogueTile_, // (EPI_TILE_M,EPI_TILE_N)
  class ElementC_,
  class StrideC_,
  class ElementD_,
  class StrideD_,
  class FusionCallbacks_,
  class CopyOpG2S_,
  class SmemLayoutAtomC_,
  class CopyOpS2R_,
  class CopyOpS2G_,
  class SmemLayoutAtomD_,
  class CopyOpR2S_,
  class CopyAtomC_,
  class CopyOpR2R_,
  class ProblemShape_,
  class SubMatLayoutC_
>
class CollectiveStrassenEpilogue<
    StrassenMiGroup_,
    Sm90PtrArrayTmaWarpSpecialized<StagesC_,
                                   StagesD_,
                                   FragmentSize_,
                                   ReuseSmemC_,
                                   DelayTmaStore_,
                                   NumEpilogueWarpGroups_
                                  >,
    CtaTileMNK_,
    EpilogueTile_,
    ElementC_,
    StrideC_,
    ElementD_,
    StrideD_,
    FusionCallbacks_,
    CopyOpG2S_,
    SmemLayoutAtomC_,
    CopyOpS2R_,
    CopyOpS2G_,
    SmemLayoutAtomD_,
    CopyOpR2S_,
    CopyAtomC_,
    CopyOpR2R_,
    ProblemShape_,
    SubMatLayoutC_
> {
public:
  //
  // Type Aliases
  //
  using StrassenMiGroup = StrassenMiGroup_;
  using ProblemShape = ProblemShape_;
  using SubMatLayoutC = SubMatLayoutC_;
  using DispatchPolicy = Sm90PtrArrayTmaWarpSpecialized<StagesC_,
                                                        StagesD_,
                                                        FragmentSize_,
                                                        ReuseSmemC_,
                                                        DelayTmaStore_, 
                                                        NumEpilogueWarpGroups_
                                                       >;
  using CtaTileMNK = CtaTileMNK_;
  using EpilogueTile = EpilogueTile_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;
  using CopyOpG2S = CopyOpG2S_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyAtomC = CopyAtomC_;
  using CopyOpR2R = CopyOpR2R_;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyC = CopyOpG2S;
  using GmemTiledCopyD = CopyOpS2G;

  static_assert(!is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");
  static_assert(size<0>(CtaTileMNK{}) % size<0>(shape(EpilogueTile{})) == 0, "EPI_TILE_M must divide CTA_M");
  static_assert(size<1>(CtaTileMNK{}) % size<1>(shape(EpilogueTile{})) == 0, "EPI_TILE_N must divide CTA_N");
  static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

private:
  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;
  using NonVoidElementD = cute::conditional_t<not is_destination_supported,fusion::get_element_aux_t<FusionCallbacks>, ElementD>;
  static_assert(not cute::is_void_v<NonVoidElementD>, "SmemElementD is void");
  using NonVoidElementC = cute::conditional_t<not is_source_supported,NonVoidElementD,ElementC>; // prevents void ref breakages

  using SmemElementC = typename cutlass::detail::get_unpacked_element_type<NonVoidElementC>::type;
  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<NonVoidElementD>::type;

  constexpr static int StagesC = StagesC_;
  constexpr static int StagesD = StagesD_;
  constexpr static bool ReuseSmemC = ReuseSmemC_ and is_destination_supported;
  constexpr static bool DelayTmaStore = DelayTmaStore_;

  constexpr static bool is_m_major_C = detail::is_m_major<InternalStrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<InternalStrideD>();

  constexpr static bool is_im2col_C = cute::is_same_v<CopyOpG2S, SM90_TMA_LOAD_IM2COL>;
  constexpr static bool is_im2col_D = cute::is_same_v<CopyOpS2G, SM90_TMA_STORE_IM2COL>;

  // Check if register transformation is needed before copying register to shared memory.
  constexpr static bool IsUseR2R = !cute::is_void_v<CopyOpR2R>;

  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<StagesC>{}),
      cute::conditional_t<is_m_major_C, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{},
      make_shape(size<0>(EpilogueTile{}), size<1>(EpilogueTile{}), Int<ReuseSmemC ? StagesC : StagesD>{}),
      cute::conditional_t<is_m_major_D, Step<_2,_1,_3>, Step<_1,_2,_3>>{} ));

  static const bool IsFusedM2M3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();
  static const bool IsFusedM4M5 = StrassenMiGroup::hasM4() && StrassenMiGroup::hasM5();

  constexpr static bool support_smem_reuse = is_source_supported && is_destination_supported && StagesD <= StagesC
                                            && cosize(take<0,2>(SmemLayoutC{})) == cosize(take<0,2>(SmemLayoutD{}));
  static_assert(not (ReuseSmemC && not support_smem_reuse), "Smem reuse requirements not met");

  constexpr static size_t SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{});
  constexpr static size_t SmemAlignmentC = cutlass::detail::alignment_for_swizzle(SmemLayoutC{});
  constexpr static size_t MaxSmemAlignment = cute::max(SmemAlignmentC, SmemAlignmentD);

  using SmemArrayTypeC = cute::ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>>;
  using SmemArrayTypeD = cute::ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>>;

  using EmptyType = cute::tuple<>;
  using SmemCStorage = cute::conditional_t<is_source_supported and (not ReuseSmemC),
                         SmemArrayTypeC,
                         EmptyType>;
  using SmemDStorage = cute::conditional_t<is_destination_supported,
                         SmemArrayTypeD,
                         EmptyType>;

  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageReuseC {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

    using RWCTypes = typename StrassenMiGroup::RWCTypes;

public:
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;
  using LoadPipelineState = cutlass::PipelineState<StagesC>;
  constexpr static uint32_t TmaTransactionBytes =
    (size(take<0,2>(SmemLayoutC{})) * static_cast<uint32_t>(sizeof_bits<SmemElementC>::value)) / 8;
  constexpr static bool RequiresTransactionBytes = true;

  constexpr static int NumEpilogueWarpGroups = NumEpilogueWarpGroups_;
  constexpr static uint32_t MinTensorMapWorkspaceAlignment = 64;

  // TMA pipeline for storing D
  using StorePipeline = cute::conditional_t<ReuseSmemC,
                          cutlass::PipelineTmaStore<StagesC, StagesD-1>,
                          cutlass::PipelineTmaStore<StagesD>>;
  using StorePipelineState = cutlass::PipelineState<ReuseSmemC ? StagesC : StagesD>;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideC, StrideC>;
  static constexpr bool IsMoEGemmKernel = cute::is_same_v<
    ProblemShape,
    cutlass::gemm::StrassenMoEProblemShape<typename ProblemShape::UnderlyingProblemShape>
  >;

  struct SharedStorage {
    struct TensorStorage {
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;

      using FusionStorage = typename FusionCallbacks::SharedStorage;
      FusionStorage thread;
    } tensors;

    struct NonEmptyTensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_C;
      cute::TmaDescriptor smem_tensormap_postsum_load;
      cute::array<cute::TmaDescriptor, NumEpilogueWarpGroups> smem_tensormap_D;
      cute::array<cute::TmaDescriptor, NumEpilogueWarpGroups> smem_tensormap_postsum_store;
    };

    struct EmptyTensorMapStorage {};
    using TensorMapStorage = cute::conditional_t<IsMoEGemmKernel, EmptyTensorMapStorage, NonEmptyTensorMapStorage>;
    TensorMapStorage tensormaps;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;
  using PointerC = cute::conditional_t<IsMoEGemmKernel, ElementC const*, ElementC const**>;
  using PointerD = cute::conditional_t<IsMoEGemmKernel, ElementD*, ElementD**>;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    PointerC ptr_C = nullptr;
    StrideC dC;
    PointerD ptr_D = nullptr;
    StrideD dD;
    uint64_t const* ptr_C_batch_indices = nullptr;
    uint64_t const* ptr_D_batch_indices = nullptr;

    Arguments() {}

    Arguments(typename FusionCallbacks::Arguments thread, PointerC ptr_C, StrideC dC,
          PointerD ptr_D, StrideD dD,
          uint64_t const* ptr_C_batch_indices = nullptr,
          uint64_t const* ptr_D_batch_indices = nullptr) :
              thread(thread), ptr_C(ptr_C), dC(dC), ptr_D(ptr_D), dD(dD),
          ptr_C_batch_indices(ptr_C_batch_indices), ptr_D_batch_indices(ptr_D_batch_indices)
              {
                thread.beta = 0;
              }

    template<typename Other>
    Arguments(const Other& other) : ptr_C(other.ptr_C), dC(other.dC),
                                     ptr_D(other.ptr_D), dD(other.dD)
              {
                thread.alpha = other.thread.alpha;
                thread.beta = other.thread.beta;
                if constexpr (requires { other.ptr_C_batch_indices; other.ptr_D_batch_indices; }) {
                  ptr_C_batch_indices = other.ptr_C_batch_indices;
                  ptr_D_batch_indices = other.ptr_D_batch_indices;
                }
              }
  };

  // Device side epilogue params
  struct Params {
    using TMA_C = decltype(make_tma_copy(
        CopyOpG2S{},
        make_tensor(make_gmem_ptr(static_cast<NonVoidElementC const*>(nullptr)),
            repeat_like(InternalStrideC{}, int32_t(0)), InternalStrideC{}),
        take<0,2>(SmemLayoutC{}),
        EpilogueTile{},
        _1{}));

    using TMA_D = decltype(make_tma_copy(
        CopyOpS2G{},
        make_tensor(make_gmem_ptr(static_cast<NonVoidElementD const*>(nullptr)),
            repeat_like(InternalStrideD{}, int32_t(0)), InternalStrideD{}),
        take<0,2>(SmemLayoutD{}),
        EpilogueTile{},
        _1{}));
    
    using TMA_D_ADD = decltype(make_tma_copy(
      SM90_TMA_REDUCE_ADD{},
      make_tensor(make_gmem_ptr(static_cast<NonVoidElementD const*>(nullptr)),
        repeat_like(InternalStrideD{}, int32_t(0)), InternalStrideD{}),
      take<0,2>(SmemLayoutD{}),
      EpilogueTile{},
      _1{}));

    typename FusionCallbacks::Params thread{};
    TMA_C tma_load_c;
    TMA_C tma_load_postsum_m;
    TMA_D tma_store_postsum_m;
    TMA_D tma_store_d;
    TMA_D_ADD tma_add_postsum_m;
    TMA_D_ADD tma_add_d;
    cute::TmaDescriptor* tensormaps;
    PointerC ptr_C;
    uint64_t const* ptr_C_batch_indices;
    StrideC dC;
    PointerD ptr_D;
    uint64_t const* ptr_D_batch_indices;
    StrideD dD;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* ptr_postsum_m;
    uint64_t* postsum_m_batch_indices;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      ElementD* postsum_m,
      uint64_t* postsum_m_batch_indices,
      [[maybe_unused]] void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_M = int32_t(size<0>(CtaTileMNK{}));
    auto init_N = int32_t(size<1>(CtaTileMNK{}));
    auto init_L = 1;
    auto moe_M = init_M;

    static_assert(!is_im2col_C and !is_im2col_D, "Im2Col not supported on C or D");

    InternalStrideC stride_c;
    InternalStrideD stride_d;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_c = InternalStrideC{};
      stride_d = InternalStrideD{};
      if constexpr (IsMoEGemmKernel) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(0), 1);
        init_M = get<0>(problem_shape_MNKL);
        init_N = get<1>(problem_shape_MNKL);
        moe_M = 0;
        for (int group_idx = 0; group_idx < problem_shape.groups(); ++group_idx) {
          moe_M += get<0>(problem_shape.get_host_problem_shape(group_idx));
        }
        if constexpr (cute::is_same_v<cute::remove_cvref_t<decltype(get<0>(stride_c))>, cute::Int<1>>) {
          get<1>(stride_c) = init_M;
          get<1>(stride_d) = init_M;
        }
        else {
          get<0>(stride_c) = init_N;
          get<0>(stride_d) = init_N;
        }
      }
    } 
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(0), 1);
      init_M = get<0>(problem_shape_MNKL);
      init_N = get<1>(problem_shape_MNKL);
      moe_M = init_M;
      stride_c = args.dC;
      stride_d = args.dD;
    }

    uint32_t transaction_bytes = TmaTransactionBytes;
    typename Params::TMA_C tma_load_c{};
    if constexpr (is_source_supported) {
    // NOTE: Since TMA desc creation with nullptr not possible until 12.6, we use an initial address even when tensor addresses are on device. This address is never used.
      ElementC const* ptr_C_first_batch;
      if constexpr (IsMoEGemmKernel) {
        ptr_C_first_batch = args.ptr_C;
      }
      else {
        ptr_C_first_batch = reinterpret_cast<ElementC const*>(reinterpret_cast<uint64_t>(args.ptr_C) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned
      }
      Tensor tensor_c = make_tensor(ptr_C_first_batch, make_layout(make_shape(moe_M,init_N,init_L), append<3>(stride_c, _0{})));
      tma_load_c = make_tma_copy(
          CopyOpG2S{},
          tensor_c,
          take<0,2>(SmemLayoutC{}),
          EpilogueTile{},
          _1{});
    }

    typename Params::TMA_D tma_store_d{};
    typename Params::TMA_D_ADD tma_add_d{};

    if constexpr (is_destination_supported) {
    // NOTE: Since TMA desc creation with nullptr not possible until 12.6, we use an initial address even when tensor addresses are on device. This address is never used.
      ElementD const* ptr_D_first_batch;
      if constexpr (IsMoEGemmKernel) {
        ptr_D_first_batch = args.ptr_D;
      }
      else {
        ptr_D_first_batch = reinterpret_cast<ElementD const*>(reinterpret_cast<uint64_t>(args.ptr_D) & 0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned
      }
      Tensor tensor_d = make_tensor(ptr_D_first_batch, make_layout(make_shape(moe_M,init_N,init_L), append<3>(stride_d, _0{})));
      tma_store_d = make_tma_copy(
          CopyOpS2G{},
          tensor_d,
          take<0,2>(SmemLayoutD{}),
          EpilogueTile{},
          _1{});
      tma_add_d = make_tma_copy_C_sm90(
          SM90_TMA_REDUCE_ADD{},
          tensor_d,
          take<0,2>(SmemLayoutD{}),
          EpilogueTile{});
    }

    typename Params::TMA_C tma_load_postsum_m{};

    auto postsum_m_extent = IsMoEGemmKernel ? (7 * moe_M) / 4 : (4 * moe_M) / 2;
    Tensor tensor_load_m = make_tensor(make_gmem_ptr<NonVoidElementC const>(postsum_m),
                       make_layout(make_shape(postsum_m_extent,max(init_N/2,1),init_L),
                                                   make_stride(max(get<0>(stride_d)/2, get<0>(stride_d)), get<1>(stride_d), get<2>(stride_d))));
    tma_load_postsum_m = make_tma_copy_C_sm90(
        CopyOpG2S{},
        tensor_load_m,
        take<0,2>(SmemLayoutD{}),
        EpilogueTile{});

    typename Params::TMA_D tma_store_postsum_m{};
    Tensor tensor_store_m = make_tensor(make_gmem_ptr<NonVoidElementD>(postsum_m),
                                       make_layout(make_shape(postsum_m_extent,max(init_N/2,1),init_L),
                                                   make_stride(max(get<0>(stride_d)/2, get<0>(stride_d)), get<1>(stride_d), get<2>(stride_d))));

    tma_store_postsum_m = make_tma_copy_C_sm90(
        CopyOpS2G{},
        tensor_store_m,
        take<0,2>(SmemLayoutD{}),
        EpilogueTile{});

    typename Params::TMA_D_ADD tma_add_postsum_m{};
    tma_add_postsum_m = make_tma_copy_C_sm90(
      SM90_TMA_REDUCE_ADD{},
      tensor_store_m,
      take<0,2>(SmemLayoutD{}),
      EpilogueTile{});

    auto fusion_workspace = static_cast<char*>(workspace);
    auto fusion_workspace_size = round_nearest(FusionCallbacks::get_workspace_size(problem_shape, args.thread), MinTensorMapWorkspaceAlignment);
    auto tma_descriptor_workspace = IsMoEGemmKernel ? nullptr : reinterpret_cast<cute::TmaDescriptor*>(
                      static_cast<char*>(workspace) + fusion_workspace_size);

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, fusion_workspace),
      tma_load_c,
      tma_load_postsum_m,
      tma_store_postsum_m,
      tma_store_d,
      tma_add_postsum_m,
      tma_add_d,
      tma_descriptor_workspace,
      args.ptr_C,
      args.ptr_C_batch_indices,
      args.dC,
      args.ptr_D,
      args.ptr_D_batch_indices,
      args.dD,
      transaction_bytes,
      postsum_m,
      postsum_m_batch_indices
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    constexpr uint32_t NumTensorMaps = 2 * NumEpilogueWarpGroups +
      (cute::is_void_v<ElementC> ? 0 : 2 * NumEpilogueWarpGroups);
    auto descriptors_shape = cute::make_shape(sm_count, Int<NumTensorMaps>{});
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B tensormap copies
    return (IsMoEGemmKernel ? 0 : size(descriptors_shape) * SizeOfCuTensorMap) +
        (round_nearest(FusionCallbacks::get_workspace_size(problem_shape, args.thread), MinTensorMapWorkspaceAlignment));
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return FusionCallbacks::initialize_workspace(problem_shape, args.thread, workspace, stream, cuda_adapter);
  }

  template <class ProblemShape>
  static bool
  can_implement(
      ProblemShape problem_shape,
      [[maybe_unused]] Arguments const& args) {

    bool implementable = true;
    bool fusion_implementable = true;

    if constexpr (IsMoEGemmKernel) {
      implementable &= !is_destination_supported || (args.ptr_D != nullptr && args.ptr_D_batch_indices != nullptr);
      implementable &= !is_source_supported || args.ptr_C == nullptr || args.ptr_C_batch_indices != nullptr;
    }

    if (problem_shape.is_host_problem_shape_available()) {
      for (int i = 0; i < problem_shape.groups(); ++i) {
        auto problem_shape_MNKL = append<4>(problem_shape.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;

        if constexpr (is_destination_supported) {
          constexpr int tma_alignment_bits_D = cutlass::detail::get_output_alignment_bits<ElementD>();
          constexpr int min_tma_aligned_elements_D = tma_alignment_bits_D / cutlass::sizeof_bits<ElementD>::value;
          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_D>(cute::make_shape(M,N,L), InternalStrideD{});
        }

        if constexpr (is_source_supported) {
          constexpr int tma_alignment_bits_C = cutlass::detail::get_input_alignment_bits<ElementC>();
          constexpr int min_tma_aligned_elements_C = tma_alignment_bits_C / cutlass::sizeof_bits<ElementC>::value;
          implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(cute::make_shape(M,N,L), InternalStrideC{});
        }

        fusion_implementable = fusion_implementable && FusionCallbacks::can_implement(problem_shape_MNKL, args.thread);
      }
    }
    else {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Ignoring check to can implement because host problem shape is not available.\n");
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    if (!fusion_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for FusionCallbacks.\n");
    }

    bool beta_implementable = true;

    if (cute::is_void_v<ElementC> || args.ptr_C == nullptr) {
      if constexpr (detail::has_beta<Arguments>::value) {
        beta_implementable = args.thread.beta == 0.0;
      }
      if constexpr (detail::has_beta_ptr<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr == nullptr;
      }
      if constexpr (detail::has_beta_ptr_array<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr_array == nullptr;
      }
    }

    if (!beta_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Beta/beta pointer was set, but epilogue is sourceless (void-C).\n");
    }

    return implementable && fusion_implementable && beta_implementable;
  }

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_load_pipe_increment(TileShapeMNK tile_shape_MNK) {
    // Compute number of epilogue subtiles
    return size<1>(zipped_divide(make_layout(take<0,2>(tile_shape_MNK)), EpilogueTile{}));
  }

  template<class TileShapeMNK>
  CUTLASS_HOST_DEVICE
  static constexpr int
  get_store_pipe_increment(TileShapeMNK tile_shape_MNK) {
    return get_load_pipe_increment(tile_shape_MNK);
  }

  CUTLASS_HOST_DEVICE
  CollectiveStrassenEpilogue(Params const& params_, TensorStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return StrassenMiGroup::NumMisWithGLLoads() > 0;
  }

  /*template<
    class ProblemShapeMNKL
  >
  CUTLASS_DEVICE decltype(auto)
  get_load_tma(ProblemShapeMNKL problem_shape_mnkl, int sub_m_idx, PostsumOp global_src) {
    auto [M, N, K, L] = problem_shape_mnkl;

    Tensor d = params.tma_store_d.get_tma_tensor(make_shape(M, N, L));
    auto d0_ptr = d.data() + make_coord(0,0,_);
    auto d0 = make_tensor(d0_ptr, d.layout());

    if (global_src.valid() && global_src.is_mem_global()) {
      if (global_src.is_layout_final()) {
        auto d_ptr = d.data() + make_coord((global_src.get_op()%2)*N/2,
                                           (global_src.get_op()/2)*M/2,_);
        return cute::tuple(make_tensor(d_ptr, d.layout()), false);
      } else if (global_src.is_layout_interim_matrix()) {
        auto m_ptr = postsum_m.data() + make_coord(0,global_src.get_op()*M/2,_);
        return cute::tuple(make_tensor(m_ptr, postsum_m.layout()), true);
      }
    }

    return cute::tuple(d0, false);
  }*/

  CUTLASS_DEVICE auto
  load_init(
      Params const& params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx,
      int32_t warp_group_idx = 0) {
    // Initialize tma for loading
    constexpr bool IsLoad = true;
    auto load_tensormaps = tensormaps_init<IsLoad>(params, shared_tensormaps, sm_count, sm_idx, warp_group_idx);
    return load_tensormaps;
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class TiledMma,
    class TensorMapC,
    __CUTE_REQUIRES(std::is_pointer_v<TensorMapC>)
  >
  CUTLASS_DEVICE auto
  load(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl, const int sub_m_idx,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      TensorStorage& shared_tensors2,
      TensorMapC const& load_tensormap,
      TensorMapC const& load_postsum_tensormap,
      PostsumOp src_global_ops[4],
      int subtile_idx=-1,
      bool wait_until_load_finishes = false) {
    using namespace cute;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    static_assert(!is_im2col_D, "Do not support im2col");

    int c_m_coord = m_coord;
    if (l_coord != 0 && params.ptr_C != nullptr && src_global_ops[0].is_layout_final()) {
      if constexpr (IsMoEGemmKernel) {
        c_m_coord += int(params.ptr_C_batch_indices[l_coord] / size<0>(CtaTileMNK{}));
      } else {
        printf("696 TODO\n");
      }
    } else if (src_global_ops[0].is_layout_interim_matrix()) {
      c_m_coord += int(src_global_ops[0].get_op()*(M/2)/size<0>(CtaTileMNK{}));
      if constexpr (IsMoEGemmKernel) {
        if (l_coord > 0)
          c_m_coord += int((params.ptr_D_batch_indices[l_coord] * 7 / 4) /
                          size<0>(CtaTileMNK{}));
      }
    }

    auto coord_shape = append<3>(make_shape(c_m_coord, n_coord), Int<0>{});
    // auto first_load_tuple = get_load_tma(problem_shape_mnkl, sub_m_idx, src_global_ops[0]);

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    Tensor mC_mn = params.tma_load_postsum_m.get_tma_tensor(append<3>(make_shape(4*(M/2),N/2), Int<1>{}));             //       (M,N,L)
    Tensor mC = coalesce(mC_mn, take<0,2>(CtaTileMNK{}));
    Tensor gC = local_tile(mC, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)

    // Apply epilogue subtile, get matching smem tensor
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});           //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s = params.tma_load_postsum_m.get_slice(Int<0>{});
    Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);                                    // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);                                    // (G2S,G2S_M,G2S_N,PIPE_C)

    // Get the fusion callbacks for the producer load warp
    auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs{
                      problem_shape_mnkl,
                      CtaTileMNK{},
                      tile_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      thread_idx
                    };
    auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
    bool is_C_load_needed = is_source_supported && (fusion_callbacks.is_C_load_needed() || src_global_ops[0].valid());

    LoadPipelineState last_load_producer_state = load_pipe_producer_state;

    // Predication for TMA load (one thread issues TMA load)
    bool issue_tma_load = cute::elect_one_sync();

    // Pre-loop fusion callback entry point
    pld_callbacks.begin();

    LoadPipelineState prior_state = load_pipe_producer_state;

    bool did_load = false;

    CUTLASS_PRAGMA_UNROLL
    for (int epi_m = 0; epi_m < size<2>(gC_epi); ++epi_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_n = 0; epi_n < size<3>(gC_epi); ++epi_n) {
        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gC_epi)) + epi_m) != subtile_idx) {
          continue;
        }

        // Acquire the lock for this stage
        constexpr uint16_t mcast_mask = 0;
        uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);

        load_pipeline.producer_acquire(load_pipe_producer_state);

        // Loop fusion callback entry point
        pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(), issue_tma_load);

        // Execute the TMA load for C if needed
        if (is_C_load_needed) {
          if (issue_tma_load) {
            if constexpr (IsMoEGemmKernel) {
              copy(params.tma_load_postsum_m.with(*tma_barrier, mcast_mask),
                  bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
            }
            else {
              copy(params.tma_load_postsum_m.with(load_postsum_tensormap, *tma_barrier, mcast_mask),
                  bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
            }
            load_pipeline.producer_expect_transaction(load_pipe_producer_state);
          }
          last_load_producer_state = load_pipe_producer_state;
          did_load = true;
        }

        // Commit TMA loads for this stage and release the lock
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();

    if (wait_until_load_finishes && did_load) {
      typename CollectiveStrassenEpilogue::LoadPipelineState epi_load_pipe_tma_consumer_state =
        {last_load_producer_state.index(), !last_load_producer_state.phase(), last_load_producer_state.count()};
      load_pipeline.consumer_wait(epi_load_pipe_tma_consumer_state);
    }

    return load_pipe_producer_state;
  }

   template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  load_m0(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl, const int sub_m_idx,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& epilogue_tensors,
      TensorStorage& epilogue_tensors2,
      PostsumOp src_global_ops[4],
      int subtile_idx=-1) {
    using namespace cute;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    ElementD* postsum_m0 = (ElementD*)params.ptr_postsum_m + (l_coord == 0 ? 0 : params.postsum_m_batch_indices[l_coord]) + (m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{});
    uint STAGE_ELEMS = size<0>(EpilogueTile{}) * size<1>(EpilogueTile{});
    uint NUM_STAGES = (size<0>(TileShapeMNK{}) * size<1>(TileShapeMNK{}))/STAGE_ELEMS;
    cutlass::Array<ElementD, 8>* ptr_smem = (cutlass::Array<ElementD, 8>*)epilogue_tensors.collective.smem_C.begin();

    bool issue_tma_load = thread_idx == 0;//cute::elect_one_sync();
    bool is_M_load_needed = src_global_ops[0].valid();

    int num_src_ops = 0;
    #pragma unroll 4
    for (; num_src_ops < 4; num_src_ops++)
      if (!src_global_ops[num_src_ops].valid()) break;

    load_pipeline.params_.transaction_bytes = STAGE_ELEMS * 2 * num_src_ops;

    if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
      MY_PRINTF("1030 %d %d: %d %d\n", threadIdx.x, is_M_load_needed, m_coord, n_coord);

    if (is_M_load_needed)
    for (uint stage = 0; stage < size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{}); stage += STAGE_ELEMS) {
      uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);
      if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        MY_PRINTF("731 epi load %d %d: %d %d : phase %d %d %d\n", threadIdx.x, stage, m_coord, n_coord,
                load_pipe_producer_state.phase_, load_pipe_producer_state.index(), load_pipe_producer_state.count());
      load_pipeline.producer_acquire(load_pipe_producer_state);
      if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        MY_PRINTF("734 epi load %d %d: %d %d\n", threadIdx.x, stage, m_coord, n_coord);
      if (issue_tma_load && is_M_load_needed) {
        #pragma unroll 
        for (int src_op_i = 0; src_op_i < num_src_ops; src_op_i++) {
          auto src_op = src_global_ops[src_op_i];
          auto postsum_m_ptr = postsum_m0 + src_op.get_op()*(M/2)*(N/2);
          uint smem_ld_index = load_pipe_producer_state.index()*STAGE_ELEMS;

          SM90_BULK_COPY_G2S().copy(postsum_m_ptr, tma_barrier, &ptr_smem[(smem_ld_index + src_op_i*STAGE_ELEMS*StagesC)/8],
                                    STAGE_ELEMS * sizeof(ElementD));
          if (num_src_ops == 1 && stage + STAGE_ELEMS < size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{}))
            SM90_BULK_COPY_G2S::PREFETCH().copy(postsum_m_ptr + STAGE_ELEMS,
                                      STAGE_ELEMS * sizeof(ElementD));
        }

        load_pipeline.producer_expect_transaction(load_pipe_producer_state);
      }
      load_pipeline.producer_commit(load_pipe_producer_state);
      ++load_pipe_producer_state;
      postsum_m0 += STAGE_ELEMS;
    }

    if (thread_idx == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
      MY_PRINTF("1044 %d %d: %d %d\n", 0/*stage*/, threadIdx.x, m_coord, n_coord);

    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE auto
  load_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state) {

    if (!is_producer_load_needed()){//fusion_callbacks.is_producer_load_needed()) {
      return load_pipe_producer_state; 
    }

    bool issue_tma_load = cute::elect_one_sync();
    if (issue_tma_load) {
      load_pipeline.producer_tail(load_pipe_producer_state);
    }

    return load_pipe_producer_state;
  }

    template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  store_m0(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      cute::Tensor<AccEngine,AccLayout> accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& epilogue_tensors,
      TensorStorage& epilogue_tensors2,
      int subtile_idx=-1) {
    NumericArrayConverter<ElementD, float, 8> converter;
    cutlass::Array<ElementD, 8>* ptr_smem = (cutlass::Array<ElementD, 8>*)epilogue_tensors.collective.smem_D.begin();
    ElementD* postsum_m0 = (ElementD*)params.ptr_postsum_m;
    cutlass::Array<float, 8>* arrs = (cutlass::Array<float, 8>*)&accumulators;

    auto [M, N, K, L] = problem_shape_mnkl;

    uint NumMMAThreads = size(TiledMma{});

    uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}))/NumMMAThreads;
    uint VECTOR_ELEMS = 8;

    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    postsum_m0 = postsum_m0 + (l_coord == 0 ? 0 : params.postsum_m_batch_indices[l_coord]) + (m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{});

    if (StagesD > 1) {
      //Multiple stage using TMA
      uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{})) / NumMMAThreads;
      bool issue_tma_store = thread_idx == 0;
      uint stage = 0;
      for (int e = 0; e < STAGE_ELEMS; e += VECTOR_ELEMS) {
        cutlass::Array<ElementD, 8> arr = converter(arrs[(stage + e)/VECTOR_ELEMS]);
        ptr_smem[store_pipe_producer_state.index() * STAGE_ELEMS * NumMMAThreads/VECTOR_ELEMS +
                 thread_idx + (e/VECTOR_ELEMS)*NumMMAThreads] = arr;
      }

      for (; stage < accumulators.size(); stage += STAGE_ELEMS) {
        cutlass::arch::fence_view_async_shared();
        asm volatile("bar.cta.sync %0, %1;" : : "r"(5), "r"(NumMMAThreads));

        if (issue_tma_store) {
          auto smem = &ptr_smem[store_pipe_producer_state.index() * STAGE_ELEMS * NumMMAThreads/VECTOR_ELEMS];
          // cutlass::Array<ElementD, 8>* st_ptr = (cutlass::Array<ElementD, 8>*)&postsum_m0[write_stage*STAGE_ELEMS*128 + (stage - STAGE_ELEMS)*128];
          auto st_ptr = &postsum_m0[stage*NumMMAThreads];
          SM90_BULK_COPY_S2G().copy(smem, st_ptr, STAGE_ELEMS * NumMMAThreads * sizeof(ElementD));
          store_pipeline.producer_commit(store_pipe_producer_state);
        }

        ++store_pipe_producer_state;

        if (issue_tma_store)
          store_pipeline.producer_acquire(store_pipe_producer_state);
        asm volatile("bar.cta.sync %0, %1;" : : "r"(6), "r"(NumMMAThreads));

        if (stage + STAGE_ELEMS < accumulators.size())
          for (int e = 0; e < STAGE_ELEMS; e += VECTOR_ELEMS) {
            cutlass::Array<ElementD, 8> arr = converter(arrs[(stage + STAGE_ELEMS + e)/VECTOR_ELEMS]);
            ptr_smem[store_pipe_producer_state.index() * STAGE_ELEMS * NumMMAThreads/VECTOR_ELEMS +
                    thread_idx + (e/VECTOR_ELEMS)*NumMMAThreads] = arr;
          }
      }

      return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
    } else {
      //Single stage using TMA
      uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{})) / NumMMAThreads;
      bool issue_tma_store = thread_idx == 0;
      uint stage = 0;

      for (; stage < accumulators.size(); stage += STAGE_ELEMS) {
        for (int e = 0; e < STAGE_ELEMS; e += VECTOR_ELEMS) {
          cutlass::Array<ElementD, 8> arr = converter(arrs[(stage + e)/VECTOR_ELEMS]);
          ptr_smem[thread_idx + (e/VECTOR_ELEMS)*NumMMAThreads] = arr;
        }

        cutlass::arch::fence_view_async_shared();
        asm volatile("bar.cta.sync %0, %1;" : : "r"(5), "r"(NumMMAThreads));

        if (issue_tma_store) {
          auto smem = &ptr_smem[0];
          // cutlass::Array<ElementD, 8>* st_ptr = (cutlass::Array<ElementD, 8>*)&postsum_m0[write_stage*STAGE_ELEMS*128 + (stage - STAGE_ELEMS)*128];
          auto st_ptr = &postsum_m0[stage*NumMMAThreads];
          SM90_BULK_COPY_S2G().copy(smem, st_ptr, STAGE_ELEMS * NumMMAThreads * sizeof(ElementD));
          asm volatile("cp.async.bulk.commit_group;");
          cute::tma_store_wait<0>();
        }

        asm volatile("bar.cta.sync %0, %1;" : : "r"(6), "r"(NumMMAThreads));
      }

      return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
    }
  }

    template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma
  >
  CUTLASS_DEVICE auto
  store_m2(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl, int sub_m_idx,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl,
      cute::Tensor<AccEngine,AccLayout>& accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& epilogue_tensors,
      TensorStorage& epilogue_tensors2,
      int subtile_idx=-1) {
    NumericArrayConverter<ElementD, float, 8> converter;
    NumericArrayConverter<float, ElementD, 8> conv_half_to_float;

    cutlass::Array<ElementD, 8>* ptr_smem_ld = (cutlass::Array<ElementD, 8>*)epilogue_tensors.collective.smem_C.begin();
    ElementD* postsum_m0 = (ElementD*)params.ptr_postsum_m;
    cutlass::Array<float, 8>* arrs = (cutlass::Array<float, 8>*)&accumulators;

    auto dest_global_op = RWCTypes::PostsumGlobalDestByOutputIndex(sub_m_idx);
    auto src_global_op = RWCTypes::PostsumSrcByOutputIndex(sub_m_idx, 0);

    auto [M, N, K, L] = problem_shape_mnkl;

    constexpr uint NumMMAThreads = size(TiledMma{});

    uint VECTOR_ELEMS = 8;

    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    postsum_m0 = postsum_m0 + (l_coord == 0 ? 0 : params.postsum_m_batch_indices[l_coord]) + (m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{});
    postsum_m0 = postsum_m0 + (dest_global_op.get_op())*(M/2)*(N/2);

    //Multiple stage using TMA
    constexpr uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{})) / NumMMAThreads;
    bool issue_tma_store = thread_idx == 0;
    
    const bool is_producer_load_needed = src_global_op.valid() && src_global_op.is_mem_global() && src_global_op.is_layout_interim_linear();

    cutlass::Array<ElementD, 8>* ptr_smem_st = (cutlass::Array<ElementD, 8>*)epilogue_tensors.collective.smem_D.begin();
                                            // ((is_producer_load_needed) ?
                                                  // epilogue_tensors.collective.smem_C.begin() :
                                                  // epilogue_tensors.collective.smem_D.begin());

    if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
      MY_PRINTF("936 EpiStore %d: %d %d : %d %d : %ld\n", threadIdx.x, m_coord, n_coord, is_producer_load_needed, l_coord, params.postsum_m_batch_indices[l_coord]);

    LoadPipelineState load_wait_state = load_pipe_consumer_state;
    if (is_producer_load_needed && ReuseSmemC) {
      load_wait_state = store_pipe_producer_state;
      load_wait_state.phase_ ^= 1;
    }
    
    #pragma unroll (128/STAGE_ELEMS)
    for (uint stage = 0; stage < accumulators.size(); stage += STAGE_ELEMS) {
      if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        MY_PRINTF("1050 EpiStore %d: %d %d : %d phase %d %d %d\n", threadIdx.x, m_coord, n_coord, stage, load_wait_state.phase_, load_wait_state.index(), load_wait_state.count());
      if (is_producer_load_needed) load_pipeline.consumer_wait(load_wait_state);
      if (thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1071 EpiStore %d: %d %d : %d\n", threadIdx.x, m_coord, n_coord, stage);
      //Load smem here
      asm volatile("bar.cta.sync %0, %1;" : : "r"(6), "r"(NumMMAThreads));
      uint smem_st_index =  (store_pipe_producer_state.index()* STAGE_ELEMS);
      uint smem_ld_index =  (load_wait_state.index()* STAGE_ELEMS);

      #pragma unroll STAGE_ELEMS
      for (int e = 0; e < STAGE_ELEMS; e += VECTOR_ELEMS) {
        cutlass::Array<ElementD, 8> arr = converter(arrs[(stage + e)/VECTOR_ELEMS]);

        auto e_idx = thread_idx + (e/VECTOR_ELEMS)*NumMMAThreads;
        auto st_idx = smem_st_index * NumMMAThreads/VECTOR_ELEMS + e_idx;
        auto ld_idx = smem_ld_index * NumMMAThreads/VECTOR_ELEMS + e_idx;
        auto src_val = ptr_smem_ld[ld_idx];
        // if (StrassenMiGroup::hasM0() && thread_idx == 0 && stage == 0 && m_coord == 0 && n_coord == 0)
        //   MY_PRINTF("1107 EpiStore %d %d: %d %d : %d : %f\n", threadIdx.x, NumMMAThreads, m_coord, n_coord, stage, arrs[stage + 0][0]);
        if (is_producer_load_needed) {
          arrs[(stage + e)/VECTOR_ELEMS] = conv_half_to_float(src_val) + arrs[(stage + e)/VECTOR_ELEMS];
          // if (StrassenMiGroup::hasM0() && thread_idx == 0 && stage == 0 && m_coord == 0 && n_coord == 0)
          //   MY_PRINTF("1107 EpiStore %d: %d %d : %d : %f %f\n", threadIdx.x, m_coord, n_coord, stage, arrs[stage + 0][0], float(src_val[0]));
        }
        ptr_smem_st[st_idx] = converter(arrs[(stage + e)/VECTOR_ELEMS]);
      }

      cutlass::arch::fence_view_async_shared();
      
      asm volatile("bar.cta.sync %0, %1;" : : "r"(5), "r"(NumMMAThreads));
      if (issue_tma_store) {
        auto smem = &ptr_smem_st[(smem_st_index * NumMMAThreads)/VECTOR_ELEMS];
        // cutlass::Array<ElementD, 8>* st_ptr = (cutlass::Array<ElementD, 8>*)&postsum_m0[write_stage*STAGE_ELEMS*128 + (stage - STAGE_ELEMS)*128];
        auto st_ptr = &postsum_m0[stage*NumMMAThreads];
        SM90_BULK_COPY_S2G().copy(smem, st_ptr, STAGE_ELEMS * NumMMAThreads * sizeof(ElementD));
        store_pipeline.producer_commit(store_pipe_producer_state);
      }

      // store_uses_load_stage[store_pipe_producer_state.index()] = is_producer_load_needed;
      ++issued_stores;

      if (is_producer_load_needed) {
        if (not ReuseSmemC) {
          load_pipeline.consumer_release(load_pipe_consumer_state);
          ++load_pipe_consumer_state;
        }
        ++load_wait_state;
      }

      ++store_pipe_producer_state;
      if (issue_tma_store)
        store_pipeline.producer_acquire(store_pipe_producer_state);
      asm volatile("bar.cta.sync %0, %1;" : : "r"(5), "r"(NumMMAThreads));
      if constexpr (ReuseSmemC) {
        if (issued_stores > StorePipeline::UnacquiredStages) {
          if (is_producer_load_needed) {
            load_pipeline.consumer_release(load_pipe_consumer_state);
          }
          ++load_pipe_consumer_state;
          // store_uses_load_stage[store_pipe_producer_state.index()] = false;
        }
      }
    }

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  template<
    class ProblemShapeMNKL
  >
  CUTLASS_DEVICE decltype(auto)
  get_second_load_tma(ProblemShapeMNKL problem_shape_mnkl) {
    auto [M, N, K, L] = problem_shape_mnkl;
    Tensor postsum_m = params.tma_load_postsum_m.get_tma_tensor(make_shape(M/2,N/2,L));
    auto m0_ptr = postsum_m.data() + make_coord(0,0,_);
    auto m0 = make_tensor(m0_ptr, postsum_m.layout());

    if (StrassenMiGroup::hasM5() && !IsFusedM4M5) {
      //Load M2
      auto m2_ptr = postsum_m.data() + make_coord(0,1*M/2,_);
      return cute::tuple(make_tensor(m2_ptr, postsum_m.layout()), true);
    }

    return cute::tuple(m0, false);
  }

  template<
    class ProblemShapeMNKL
  >
  CUTLASS_DEVICE decltype(auto)
  get_store_tma(ProblemShapeMNKL problem_shape_mnkl, int sub_m_idx, bool matrix_or_linear, PostsumOp global_srcs[4], int batch, bool& use_tma_reduce) {
    auto [M, N, K, L] = problem_shape_mnkl;
    Tensor postsum_m = params.tma_store_postsum_m.get_tma_tensor(append<3>(make_shape(M/2,N/2), Int<1>{}));
    auto m0_ptr = postsum_m.data() + make_coord(0,0,_);
    auto m0 = make_tensor(m0_ptr, postsum_m.layout());
  
    Tensor d = params.tma_store_d.get_tma_tensor(append<3>(make_shape(M,N), Int<1>{}));
    auto d0_ptr = d.data() + make_coord(0,0,_);
    auto d0 = make_tensor(d0_ptr, d.layout());
    bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();
    global_srcs[0] = MmaStrassen::PostsumOp(); global_srcs[1] = MmaStrassen::PostsumOp();
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
      auto global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(i);
      int signMi = RWCTypes::MiSignByOutputIndex(i, StrassenMiGroup::getMi(sub_m_idx));
      const bool satisfy_layout = matrix_or_linear == true ?
                                  global_dest.is_layout_final() || global_dest.is_layout_interim_matrix() :
                                  global_dest.is_layout_interim_linear();
      if (global_dest.valid() && global_dest.is_mem_global() && satisfy_layout && signMi != 0) {
        PostsumOp src0 = RWCTypes::PostsumSrcByOutputIndex(i, 0);
        PostsumOp src1 = RWCTypes::PostsumSrcByOutputIndex(i, 1);
        if (global_dest.is_layout_final()) {
          auto coord = make_coord((global_dest.get_op()%2)*N/2,
                                             (global_dest.get_op()/2)*M/2,_);
          int idx = 0;
          //If the src op and dest op are same then use TMA Reduce instead of Load, Add and Store 
          bool use_tma = src0.is_mem_global() && src0.is_layout_final() && src0.get_op() == global_dest.get_op();
          if (src0.valid() && src0.is_mem_global() && !use_tma)
            //Only Global Memory are added in epilogue's store()
            global_srcs[idx++] = src0;
          use_tma_reduce = use_tma;
          use_tma = idx == 0 && src1.is_mem_global() && src1.is_layout_final() && src1.get_op() == global_dest.get_op();
          if (src1.valid() && src1.is_mem_global() && !use_tma)
            global_srcs[idx++] = src1;
          use_tma_reduce = use_tma_reduce || use_tma;
          return cute::tuple(coord, global_dest, false);
        } else if (global_dest.is_layout_interim_matrix() || global_dest.is_layout_interim_linear()) {
          auto coord = make_coord(int(batch == 0 ? 0 : params.postsum_m_batch_indices[batch]),global_dest.get_op()*M/2,_);
          int idx = 0;
          use_tma_reduce = false;
          if (src0.valid() && src0.is_mem_global())// && src0.get_op() != global_dest.get_op() && src0.is_layout_interim_linear()
            global_srcs[idx++] = src0;
          if (src1.valid() && src1.is_mem_global())// && src1.get_op() != global_dest.get_op() && src1.is_layout_interim_linear()
            global_srcs[idx++] = src1;

          return cute::tuple(coord, global_dest, true);
        }
      }
    }

    return cute::tuple(make_coord(0,0,_), MmaStrassen::PostsumOp(), false);
  }


  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma,
    class TensorMapD
  >
  CUTLASS_DEVICE auto
  store(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      ProblemShapeMNKL problem_shape_mnkl,
      TileShapeMNK tile_shape_MNK,
      TileCoordMNKL tile_coord_mnkl, int sub_m_idx,
      cute::Tensor<AccEngine,AccLayout>& accumulators_ref,
      TiledMma tiled_mma,
      int thread_idx,
      TensorStorage& shared_tensors,
      TensorMapD const& store_tensormap,
      TensorMapD const& store_postsum_tensormap,
      int subtile_idx=-1) {

    using namespace cute;
    using ElementAccumulator = typename AccEngine::value_type;
    using ElementCompute_ = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
    using ElementCompute = cute::conditional_t<cute::is_void_v<ElementCompute_>,ElementAccumulator,ElementCompute_>;
    constexpr bool IsFusedM2M3M6 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3() && StrassenMiGroup::hasM6();
    cute::Tensor<AccEngine,AccLayout> accumulators_copy = accumulators_ref;
    cute::Tensor<AccEngine,AccLayout>& accumulators = IsFusedM4M5 ? accumulators_copy : accumulators_ref;

    static_assert(is_rmem<AccEngine>::value, "Accumulator must be RF resident.");
    static_assert(rank(AccLayout{}) == 3, "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N)");
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<TileShapeMNK>::value, "TileShapeMNK must be static");
    static_assert(rank(TileShapeMNK{}) == 3, "TileShapeMNK must be rank 3");
    static_assert(rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;


    static_assert(!is_im2col_D, "Do not support im2col");
    const uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}));
    constexpr uint NumMMAThreads = size(TiledMma{});

    int d_m_coord = m_coord;
    if constexpr (IsMoEGemmKernel) {
      if (l_coord != 0) {
        d_m_coord += int(params.ptr_D_batch_indices[l_coord] / size<0>(CtaTileMNK{}));
      }
    }
    auto coord_shape = append<3>(make_shape(d_m_coord, n_coord), Int<0>{});

    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    PostsumOp first_store_srcs[4], second_store_srcs[4];
    bool use_tma_first_store = false;
    auto first_store_tuple = get_store_tma(problem_shape_mnkl, sub_m_idx, true, first_store_srcs, l_coord, use_tma_first_store);
    PostsumOp first_store_dest = get<1>(first_store_tuple);
    auto& tma_store_dorm = get<2>(first_store_tuple) ? params.tma_store_postsum_m : params.tma_store_d ;
    auto& tma_add_dorm = get<2>(first_store_tuple) ? params.tma_add_postsum_m : params.tma_add_d;
    int output_idx = first_store_dest.get_op();
    if (get<2>(first_store_tuple) == false) {
      coord_shape = append<3>(
        make_shape(
          d_m_coord + (output_idx / 2) * (M / 2 / size<0>(CtaTileMNK{})),
          n_coord + (output_idx % 2) * (N / 2 / size<1>(CtaTileMNK{}))),
        Int<0>{});
    } else {
      coord_shape = append<3>(
        make_shape(
          m_coord + (first_store_dest.get_op() * (M/2))/size<0>(CtaTileMNK{}) +
            (IsMoEGemmKernel && l_coord > 0 ? int((params.ptr_D_batch_indices[l_coord] * 7 / 4) /
              size<0>(CtaTileMNK{})) : 0),
          n_coord),
        Int<0>{});
    }

    Tensor mD_mn = get<2>(first_store_tuple) ? params.tma_store_postsum_m.get_tma_tensor(append<3>(make_shape(M/2,N/2), Int<1>{})) :
                                               params.tma_store_d.get_tma_tensor(append<3>(make_shape(M,N), Int<1>{}));            //       (M,N,L)

    Tensor mD = coalesce(mD_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD = local_tile(mD, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)

    bool use_tma_second_store = false;
    auto second_store_tuple = get_store_tma(problem_shape_mnkl, sub_m_idx, false, second_store_srcs, l_coord, use_tma_second_store);
    bool has_second_store = get<1>(second_store_tuple).valid();
    PostsumOp second_store_dest = get<1>(second_store_tuple);
    Tensor mD2_mn = mD_mn;//get<0>(second_store_tuple);
    Tensor mD2 = coalesce(mD2_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD2 = local_tile(mD2, take<0,2>(CtaTileMNK{}), coord_shape);

    bool has_second_load = get<1>(get_second_load_tma(problem_shape_mnkl)); //TODO: second load postsums should be passed

    if (second_store_dest.valid() && thread_idx == 0) {
      printf("1149: Two store Epilogue not implemented.\n");
    }
    
    // Apply epilogue subtiling
    Tensor gD_epi = flat_divide(gD, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor gD2_epi = flat_divide(gD2, EpilogueTile{});

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sC2 = shared_tensors.collective.smem_C.begin() + cosize_v<SmemLayoutC>;//StagesC*size<0>(EpilogueTile{})*size<1>(EpilogueTile{});
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    auto ptr_sD2 = shared_tensors.collective.smem_C.begin();//StagesD*size<0>(EpilogueTile{})*size<1>(EpilogueTile{});
    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sC2_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC2), SmemLayoutC{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
    Tensor sD2_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sD2), SmemLayoutD{}));

    TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);

    // (t)hread-partition for (r)egister to (r)egister copy (tRR_)
    TiledCopy tiled_r2r = [&]() {
      if constexpr (IsUseR2R) {
        return make_tiled_copy_S(Copy_Atom<CopyOpR2R, ElementCompute>{}, tiled_copy_C_atom);
      }
      else {
        return make_tiled_copy_S(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>,
          ElementCompute>{}, tiled_copy_C_atom);
      }
    }();
    ThrCopy thread_r2r = tiled_r2r.get_slice(thread_idx);

    // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
    TiledCopy tiled_r2s = [&]() {
      if constexpr (IsUseR2R) {
        return make_tiled_copy_D(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_r2r);
      }
      else {
        return make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_copy_C_atom);
      }
    }();
    ThrCopy thread_r2s = tiled_r2s.get_slice(thread_idx);
    Tensor tRS_rAcc = thread_r2s.retile_S(accumulators);                                   // ((R2S,R2S_V),MMA_M,MMA_N)
    Tensor tRS_sD   = thread_r2s.partition_D(sD_epi);                                       // (R2S,R2S_M,R2S_N,PIPE_D)
    Tensor tRS_sD2   = thread_r2s.partition_D(sD2_epi);                                       // (R2S,R2S_M,R2S_N,PIPE_D)

    auto mma_tile_m = size<0>(TileShapeMNK{}) / size<1>(tRS_rAcc);
    auto mma_tile_n = size<1>(TileShapeMNK{}) / size<2>(tRS_rAcc);
    auto epi_tile_m = size<0>(EpilogueTile{});
    auto epi_tile_n = size<1>(EpilogueTile{});

    // Allocate D registers
    Layout tRS_rD_layout = make_layout(take<0,3>(shape(thread_r2s.partition_S(sD_epi))));
    Tensor tRS_rD = make_tensor<SmemElementD>(tRS_rD_layout);                                          // (R2S,R2S_M,R2S_N)

    Layout tRS_rD2_layout = make_layout(take<0,3>(shape(thread_r2s.partition_S(sD2_epi))));
    Tensor tRS_rD2 = make_tensor<SmemElementD>(tRS_rD2_layout);

    // Vectorized fragment view
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    Tensor tRS_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(tRS_rAcc);
    Tensor tRS_rD_frg   = recast<Array<SmemElementD      , FragmentSize>>(tRS_rD);
    Tensor tRS_rD2_frg   = recast<Array<SmemElementD      , FragmentSize>>(tRS_rD2);
    CUTE_STATIC_ASSERT(size<0>(tRS_rAcc) % FragmentSize == 0, "Fragment size does not vectorize properly");

    // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
    TiledCopy tiled_s2r = make_tiled_copy_S(Copy_Atom<CopyOpS2R, SmemElementC>{}, tiled_copy_C_atom);
    ThrCopy thread_s2r = tiled_s2r.get_slice(thread_idx);
    Tensor tSR_sC        = thread_s2r.partition_S(sC_epi);                                  // (S2R,S2R_M,S2R_N,PIPE_C)
    Layout tSR_rC_layout = thread_s2r.retile_D(tRS_rD).layout();                            // (S2R,S2R_M,S2R_N)

    Tensor tSR_sC2        = thread_s2r.partition_S(sC2_epi);                                  // (S2R,S2R_M,S2R_N,PIPE_C)

    // Allocate C registers
    // If C smem load is a non-vectorized dst(i) = src(i) then we can allocate C registers directly in the compute type
    // to eliminate some redundant pack+unpack instruction sequences for sub-word types
    constexpr bool IsDirectS2R = cute::is_same_v<CopyOpS2R, AutoVectorizingCopyWithAssumedAlignment<128>>
                                && decltype(max_common_vector(tSR_rC_layout, tSR_sC.layout()))::value <= 1;
    using RegisterElementC = cute::conditional_t<IsDirectS2R, ElementCompute, SmemElementC>;
    Tensor tRS_rC = make_tensor<RegisterElementC>(tRS_rD_layout);                                  // (R2S,R2S_M,R2S_N)
    Tensor tSR_rC = thread_s2r.retile_D(tRS_rC);                                                   // (S2R,S2R_M,S2R_N)

    Tensor tRS_rC2 = make_tensor<RegisterElementC>(tRS_rD2_layout);                                  // (R2S,R2S_M,R2S_N)
    Tensor tSR_rC2 = thread_s2r.retile_D(tRS_rC2);

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = tma_store_dorm.get_slice(Int<0>{});
    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    Tensor bSG_sD2 = thrblk_s2g.partition_S(sD2_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD2 = thrblk_s2g.partition_D(gD2_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    Tensor tRS_cD_mn = thread_r2s.partition_S(flat_divide(cD_mn, EpilogueTile{}));     // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
    // Relative coordinate tensors (static)
    Tensor cD = make_coord_tensor(cD_mn.layout());                                                  // (CTA_M,CTA_N)
    Tensor tRS_cD = make_coord_tensor(tRS_cD_mn.layout());                          // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
    // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
    auto residue_cD = make_coord(M,N) - cD_mn(_0{});                                                           // (m,n)
    auto residue_tRS_cD = make_coord(M,N) - tRS_cD_mn(_0{});                                                   // (m,n)

    CUTE_STATIC_ASSERT(epi_tile_m % mma_tile_m == 0, "MMA_TILE_M must divide EPI_TILE_M");

    if constexpr (epi_tile_m * epi_tile_n > mma_tile_m * mma_tile_n) {
      // When the epilogue subtile is larger than the MMA tiles, loop over multiple MMA tiles
      CUTE_STATIC_ASSERT(epi_tile_n % mma_tile_n == 0, "MMA_TILE_N must divide EPI_TILE_N");
    }
    else {
    CUTE_STATIC_ASSERT(mma_tile_n % epi_tile_n == 0, "EPI_TILE_N must divide MMA_TILE_N");
    }
    // if (IsFusedM2M3M6 && sub_m_idx == 1 && thread_idx == 0 && m_coord*128 == 1024){//} && n_coord*256 == 512) {
    //   printf("1293 epi store %d : %d %d : %f\n", threadIdx.x, m_coord*128, n_coord*256, float(accumulators[0]));
    // }
    // Get TiledCopy for partition reference when consumer store.
    TiledCopy tiled_copy_partition_ref = make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_copy_C_atom);
    // Get the fusion callbacks for the consumer store warps
    constexpr bool RefSrc = true; // Register tensors reference R2S copy src layout
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs{
                      problem_shape_mnkl,
                      CtaTileMNK{},
                      tile_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      tiled_copy_partition_ref,
                      cD,
                      residue_cD,
                      tRS_cD,
                      residue_tRS_cD,
                      tRS_rC,
                      thread_idx
                    };
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    bool is_producer_load_needed = first_store_srcs[0].valid() || second_store_srcs[0].valid();//fusion_callbacks.is_producer_load_needed();
    bool is_C_load_needed = is_source_supported && is_producer_load_needed;//fusion_callbacks.is_C_load_needed();

    using FragmentVisit = decltype(cst_callbacks.visit(tRS_rAcc_frg(0), 0, 0, 0));
    constexpr bool IsDirectR2S = cute::is_same_v<FragmentVisit, Array<SmemElementD, FragmentSize>>;
    using RegisterElementD = cute::conditional_t<!IsDirectR2S, ElementCompute, SmemElementD>;
    Tensor tRS_rCompute = make_tensor<RegisterElementD>(tRS_rD_layout);                         // (R2S,R2S_M,R2S_N)
    Tensor tRS_rCompute_frg = recast<Array<RegisterElementD, FragmentSize>>(tRS_rCompute);

    Tensor tRS_rCompute2 = make_tensor<RegisterElementD>(tRS_rD_layout);                            // (R2S,R2S_M,R2S_N)
    Tensor tRS_rCompute2_frg = recast<Array<RegisterElementD, FragmentSize>>(tRS_rCompute2);

    // Thread synchronizer for previously issued waits or fences
    // to ensure visibility of smem reads/writes to threads or TMA unit
    auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    // Predication for TMA store (a single thread from one warp issues TMA store)
    bool issue_tma_store = ((thread_idx / NumThreadsPerWarp) == 0) && cute::elect_one_sync();

    // In the reuse smem configuration we have StagesC smem buffers and at most StagesD committed TMA stores in flight.
    // The TMA store pipeline producer acquire returns when at most StagesD-1 committed stores are in-flight, so we can
    // only guarantee store completion after StagesD iterations, then we can begin issuing releases on the smem buffer locks.
    // store_pipe_producer_state tracks the acquire and load_pipe_consumer_state tracks the release, in circular buffer fashion.
    LoadPipelineState load_wait_state = load_pipe_consumer_state;
    if constexpr (ReuseSmemC) {
      load_wait_state = store_pipe_producer_state;
      load_wait_state.phase_ ^= 1;
    }

    // We can delay issue of TMA store by one iteration to achieve better interleaving of non-TMA instructions
    // Sync requirements of smem reuse may preclude this optimization
    // Delayed stores cause delayed stage releases which causes deadlock when StagesC == StagesD
    int epi_m_prev = 0, epi_n_prev = 0;
    static_assert(not (DelayTmaStore and ReuseSmemC and StagesC <= StagesD), "This TMA epilogue configuration will deadlock");

    // The TMA store sequence for one subtile iteration
    auto tma_store_fn = [&] (int epi_m, int epi_n) {
      // Write the tile from smem to gmem with TMA
      cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
      synchronize(); // ensure all threads have issued their async fence
      if constexpr (is_destination_supported) {
        if (issue_tma_store) {
          auto active_store_tensormap = get<2>(first_store_tuple) ? store_postsum_tensormap : store_tensormap;
          if (use_tma_first_store) {
            if (thread_idx == 0) {
              //TODO: This tensormap probably needs to be fixed for multiple groups
              Tensor sD_tile = group_modes<0,2>(sD_epi(_,_,store_pipe_producer_state.index()));
              Tensor gD_tile = group_modes<0,2>(gD_epi(_,_,epi_m,epi_n));
              if constexpr (IsMoEGemmKernel) {
                copy(tma_add_dorm, sD_tile, gD_tile);
              }
              else {
                copy(tma_add_dorm.with(active_store_tensormap), sD_tile, gD_tile);
              }
            }
            // copy(tma_store_dorm, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
          } else {
            if constexpr (IsMoEGemmKernel) {
              copy(tma_store_dorm, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
            }
            else {
              copy(tma_store_dorm.with(active_store_tensormap), bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
            }
          }
        }
      }

      // Post async fence, pre TMA commit callback entry point
      cst_callbacks.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);

      // Commit the TMA stores for this stage
      if (issue_tma_store) {
        store_pipeline.producer_commit(store_pipe_producer_state);
      }
      ++store_pipe_producer_state;
      ++issued_stores;

      // Wait for the next smem buffer to be available
      if (issue_tma_store) {
        store_pipeline.producer_acquire(store_pipe_producer_state);
      }
      synchronize();

      if constexpr (ReuseSmemC) {
        // producer_acquire returns when at most StagesD-1 committed stores are pending
        bool store_finished = issued_stores > StorePipeline::UnacquiredStages;
        // Let dma warp know earliest smem buffer is consumed and empty after StagesD producer commits
        if (store_finished) {
          if (is_producer_load_needed) {
            load_pipeline.consumer_release(load_pipe_consumer_state);
          }
          ++load_pipe_consumer_state;
        }
      }
    };

    //
    // BEGIN EPILOGUE
    //

    // Pre-loop fusion callback entry point
    cst_callbacks.begin();
    if (cst_callbacks.begin_sync_needed()) {
      synchronize();
    }

    // For each output tile
    int linear_store_stage = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n, ++linear_store_stage) {
        bool is_first_iteration = epi_m == 0 && epi_n == 0;
        bool is_last_iteration = epi_m == size<2>(gD_epi)-1 && epi_n == size<3>(gD_epi)-1;

        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gD_epi)) + epi_m) != subtile_idx) {
          continue;
        }

        cst_callbacks.begin_loop(epi_m, epi_n);

        if (is_producer_load_needed) {
          // Wait for the producer load to fill smem
          load_pipeline.consumer_wait(load_wait_state);

          if (is_C_load_needed) {
            // Copy source tile from smem to register
            const uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}));
            if (first_store_srcs[0].valid()) {
              // copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
              if (first_store_srcs[0].is_layout_interim_linear()) {
                cutlass::Array<ElementD, 8>* ptr_smem = (cutlass::Array<ElementD, 8>*)((ElementD*)ptr_sC + load_wait_state.index()*STAGE_ELEMS);
                for (int i = 0; i < size(tSR_rC); i += 8) {
                  auto frg = ptr_smem[thread_idx + (i/8)*NumMMAThreads];
                  for (int j = 0; j < 8; j++)
                    tSR_rC(i + j) = frg[j];
                }
              }
              else {
                copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
              }
            }
            if (first_store_srcs[1].valid()) {
              if (first_store_srcs[1].is_layout_interim_linear()) {
                cutlass::Array<ElementD, 8>* ptr_smem = (cutlass::Array<ElementD, 8>*)((ElementD*)ptr_sC2 + load_wait_state.index()*(STAGE_ELEMS));
                for (int i = 0; i < size(tSR_rC2); i += 8) {
                  auto frg = ptr_smem[thread_idx + (i/8)*NumMMAThreads];
                  for (int j = 0; j < 8; j++) {
                    tSR_rC2(i + j) = frg[j];
                  }
                }
              } else {
                copy(tiled_s2r, tSR_sC2(_,_,_,load_wait_state.index()), tSR_rC2);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(tSR_rC); ++i) {
                  tSR_rC(i) = tSR_rC(i) + tSR_rC2(i);
                }
              }
            }
            if (ReuseSmemC) //TODO: Optimize this it is only needed when output is
            // layoutfinal and input is layoutintermediate
              synchronize();
            // Ensure smem loads are complete before reusing smem for mixed types/layouts
            if constexpr (ReuseSmemC && not (SmemLayoutC{} == SmemLayoutD{})) {
              synchronize();
            }
          }
        }

        // First loop fusion callback entry point
        cst_callbacks.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);

        if (is_producer_load_needed) {
          if constexpr (not ReuseSmemC) {
            // Let producer load warp know smem buffers are consumed and empty
            cutlass::arch::fence_view_async_shared();
            load_pipeline.consumer_release(load_pipe_consumer_state);
            ++load_pipe_consumer_state;
          }
          ++load_wait_state;
        }

        if constexpr (epi_tile_m * epi_tile_n > mma_tile_m * mma_tile_n) {
          // When the epilogue subtile is larger than the MMA tiles, loop over multiple
          // MMA tiles
          static constexpr int MmaMPerEpiM = epi_tile_m / mma_tile_m;
          static constexpr int MmaNPerEpiN = epi_tile_n / mma_tile_n;

          CUTLASS_PRAGMA_UNROLL
          for (int mma_n_in_epi = 0; mma_n_in_epi < MmaNPerEpiN; ++mma_n_in_epi) {
            int mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi;

            CUTLASS_PRAGMA_UNROLL
            for (int mma_m_in_epi = 0; mma_m_in_epi < MmaMPerEpiM; ++mma_m_in_epi) {
              int mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi;
              Tensor tRS_rAcc_frg_mn = tRS_rAcc_frg(_,mma_m,mma_n);
              int idx_in_epi_subtile = (mma_n_in_epi * MmaMPerEpiM + mma_m_in_epi);

              tRS_rCompute_frg(idx_in_epi_subtile) = cst_callbacks.visit(
                tRS_rAcc_frg_mn(0), idx_in_epi_subtile, epi_m, epi_n);
            }
          }
        }
        else {
          int mma_m = epi_m;
          int mma_n = (epi_n * size<1>(EpilogueTile{})) / mma_tile_n;
          Tensor tRS_rAcc_frg_mn = tRS_rAcc_frg(_,mma_m,mma_n);

          // Vectorized fragment loop with visitor callback entry point
          int epi_n_in_mma = epi_n % (mma_tile_n / epi_tile_n);
          int r2s_v = epi_n_in_mma * size(tRS_rCompute_frg);

          if (second_store_dest.valid() && second_store_dest.is_mem_global() && second_store_dest.is_layout_interim_linear()) {
            //LayoutInterim
            cutlass::Array<float, 8>* arrs = (cutlass::Array<float, 8>*)&accumulators[0];
            const uint PER_THREAD_ELEMS = (size<0>(EpilogueTile{})*size<1>(EpilogueTile{}))/NumMMAThreads;
            const uint VECTOR_ELEMS = 8;
            NumericArrayConverter<ElementD, float, 8> converter;
            auto ptr_smem = (cutlass::Array<ElementD, 8>*)ptr_sD2;
            // if (thread_idx == 1 && m_coord == 0 && n_coord == 0 && linear_store_stage == 0)
            //   MY_PRINTF("1630 %f %d\n", float(arrs[(linear_store_stage*PER_THREAD_ELEMS)/VECTOR_ELEMS][0]),
            //           linear_store_stage);
            for (int e = 0; e < PER_THREAD_ELEMS; e += VECTOR_ELEMS) {
              cutlass::Array<ElementD, 8> arr = converter(arrs[(linear_store_stage*PER_THREAD_ELEMS + e)/VECTOR_ELEMS]);
              ptr_smem[store_pipe_producer_state.index() * STAGE_ELEMS/VECTOR_ELEMS +
                 thread_idx + (e/VECTOR_ELEMS)*NumMMAThreads] = arr;
            }
          }

          //TODO: Here addition with source C happens
          if (first_store_srcs[0].valid() and (first_store_srcs[0].is_layout_interim_linear() || first_store_srcs[0].is_layout_interim_matrix())) {
            cutlass::Array<ElementD, FragmentSize> frg;
            for (int i = 0; i < size(tSR_rC); i++) {
              frg[i] = tSR_rC(i);
            }

            // if (StrassenMiGroup::hasM2() && thread_idx == 0 && m_coord == 0 && n_coord == 0 && epi_m == 0 && epi_n == 0)
            //   MY_PRINTF("1548 %d : %f %f\n", sub_m_idx, float(tRS_rAcc_frg_mn(r2s_v)[0]), float(frg[0]));

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tRS_rCompute_frg); ++i) {
              auto conv = cutlass::NumericArrayConverter<float, SmemElementD, FragmentSize>{}(frg);
              tRS_rAcc_frg_mn(r2s_v + i) = (StrassenMiGroup::hasM6() && !IsFusedM2M3M6) ?
                                              (conv - tRS_rAcc_frg_mn(r2s_v + i)) :
                                              (tRS_rAcc_frg_mn(r2s_v + i) + conv);
            }
          }

          if (first_store_srcs[1].valid() and (first_store_srcs[1].is_layout_interim_linear() || first_store_srcs[1].is_layout_interim_matrix())) {
            // typename decltype(tRS_rCompute_frg)::x y;
            cutlass::Array<ElementD, FragmentSize> frg;
            for (int i = 0; i < size(tSR_rC2); i++) {
              frg[i] = tSR_rC2(i);
            }
            // if (thread_idx == 1 && m_coord == 0 && n_coord == 0 && epi_m == 0 && epi_n == 0)
              // MY_PRINTF("1562 %f %f\n", float(tRS_rAcc_frg_mn(r2s_v)[0]), float(frg[0]));
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tRS_rCompute_frg); ++i) {
              tRS_rAcc_frg_mn(r2s_v + i) = tRS_rAcc_frg_mn(r2s_v + i) +
                                           cutlass::NumericArrayConverter<float, SmemElementD, FragmentSize>{}(frg);
            }
          }
          
          CUTLASS_PRAGMA_UNROLL
          for (int epi_v = 0; epi_v < size(tRS_rCompute_frg); ++epi_v) {
            tRS_rCompute_frg(epi_v) = cst_callbacks.visit(tRS_rAcc_frg_mn(r2s_v + epi_v), epi_v, epi_m, epi_n);
          }
        }

        // The latest we can delay the TMA store is right before the smem store of the next iteration
        // since the current TMA store needs to be committed before we can acquire the next smem buffer
        if constexpr (DelayTmaStore) {
          // Issue TMA stores for the previous subtile
          if (not is_first_iteration and subtile_idx == -1) {
            tma_store_fn(epi_m_prev, epi_n_prev);
          }
          epi_m_prev = epi_m;
          epi_n_prev = epi_n;
        }

        // Smem reduction callback entry point using current store buffer for workspace
        cst_callbacks.reduce(sD_epi(_,_,store_pipe_producer_state.index()),
                              synchronize, epi_m, epi_n, is_last_iteration, tRS_rCompute_frg);

        // Copy tile from register to regiser if needed
        if constexpr (IsUseR2R) {
          // retile source and destination for tiled_r2r
          Tensor tRR_rD_src = thread_r2r.retile_S(tRS_rD);                             // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)
          Tensor tRR_rD_dst = thread_r2r.retile_D(tRS_rD);                             // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)

          // Output needs register shuffling before copying to shared memory.
          copy(tiled_r2r, tRR_rD_src, tRR_rD_dst);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tRS_rD_frg); ++i) {
          tRS_rD_frg(i) = cutlass::NumericArrayConverter<SmemElementD, RegisterElementD, FragmentSize>{}(tRS_rCompute_frg(i));
        }
        
        // Copy tile from register to smem
        if constexpr (is_destination_supported) {
          copy(tiled_r2s, tRS_rD, tRS_sD(_,_,_,store_pipe_producer_state.index()));
        }

        // Post reduction, pre TMA store callback entry point
        constexpr bool issue_smem_store = true; // No smem store predication
        cst_callbacks.postreduce(epi_m, epi_n, store_pipe_producer_state.count(), issue_smem_store);

        if constexpr (not DelayTmaStore) {
          // Issue TMA stores for this subtile
          tma_store_fn(epi_m, epi_n);
        }

        cst_callbacks.end_loop(epi_m, epi_n);

      } // for epi_m
    } // for epi_n


    if constexpr (DelayTmaStore) {
      // Issue TMA stores for the last subtile
      tma_store_fn(epi_m_prev, epi_n_prev);
    }

    // Post-loop fusion callback entry point
    cst_callbacks.end();

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  CUTLASS_DEVICE auto
  store_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_consumer_state,
      StorePipeline store_pipeline,
      StorePipelineState store_pipe_producer_state,
      const bool has_global_src) {
    // wait for all TMA stores to complete
    store_pipeline.producer_tail(store_pipe_producer_state);
    // reset store counter
    issued_stores = 0;

    if constexpr (ReuseSmemC) {
      if (is_producer_load_needed()){//} || fusion_callbacks.is_producer_load_needed()) {
        // Issue releases on up to StagesD-1 previously issued TMA stores
        constexpr int release_stages = cute::min(StorePipeline::UnacquiredStages, get_load_pipe_increment(CtaTileMNK{}));
        CUTLASS_PRAGMA_UNROLL
        for (int stage = 0; stage < release_stages; ++stage) {
          load_pipeline.consumer_release(load_pipe_consumer_state);
          ++load_pipe_consumer_state;
        }
      }
    }

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  CUTLASS_DEVICE auto
  store_init(
      Params const& params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx,
      int32_t warp_group_idx) {
    int warp_idx_in_warp_group = canonical_warp_idx_sync() % NumWarpsPerWarpGroup;
    // Since only one warp issues TMA store, we only need that one warp to initialize tensormaps
    if (warp_idx_in_warp_group == 0) {
      // Initialize tma
      constexpr bool IsLoad = false;
      auto store_tensormaps = tensormaps_init<IsLoad>(params, shared_tensormaps, sm_count, sm_idx, warp_group_idx);
      return store_tensormaps;
    }
    TmaDescriptor* null_tma_desc = nullptr;
    return cute::make_tuple(null_tma_desc, null_tma_desc);
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //

  template <bool IsLoad>
  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx,
      int32_t warp_group_idx) {

    if constexpr (IsMoEGemmKernel) {
      TmaDescriptor* null_tma_desc = nullptr;
      return cute::make_tuple(null_tma_desc, null_tma_desc);
    }
    else {

    constexpr uint32_t NumTensorMaps = 2 * NumEpilogueWarpGroups +
      (cute::is_void_v<ElementC> ? 0 : 2 * NumEpilogueWarpGroups);
    Layout desc_layout = make_layout(make_shape(sm_count, Int<NumTensorMaps>{}));

    Tensor gmem_tensormap = make_tensor(params.tensormaps, desc_layout);                      // (SMs, NumInputTensors)

    if constexpr (IsLoad) {
      if (is_source_supported) {
        constexpr int C_tensormap_index = 2 * NumEpilogueWarpGroups;
        constexpr int Postsum_tensormap_index = 3 * NumEpilogueWarpGroups;
        Tensor pC_tensormap = make_tensor(params.tma_load_c.get_tma_descriptor(), Int<1>{}, Int<1>{});
        Tensor sC_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_C), Int<1>{}, Int<1>{});
        Tensor pPostsum_tensormap = make_tensor(params.tma_load_postsum_m.get_tma_descriptor(), Int<1>{}, Int<1>{});
        Tensor sPostsum_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_postsum_load), Int<1>{}, Int<1>{});

        if (cute::elect_one_sync()) {
          // Bringing tensormaps from params to smem for modification later
          copy(recast<uint128_t>(pC_tensormap), recast<uint128_t>(sC_tensormap));
          copy(recast<uint128_t>(pPostsum_tensormap), recast<uint128_t>(sPostsum_tensormap));
        }
        __syncwarp();
        return cute::make_tuple(&gmem_tensormap(sm_idx, C_tensormap_index + warp_group_idx),
              &gmem_tensormap(sm_idx, Postsum_tensormap_index + warp_group_idx));

      }
      TmaDescriptor* null_tma_desc = nullptr;
      return cute::make_tuple(null_tma_desc, null_tma_desc);
    }
    else {
      constexpr int Postsum_tensormap_index = NumEpilogueWarpGroups;
      Tensor pD_tensormap = make_tensor(params.tma_store_d.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sD_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_D[warp_group_idx]), Int<1>{}, Int<1>{});
      Tensor pPostsum_tensormap = make_tensor(params.tma_store_postsum_m.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sPostsum_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_postsum_store[warp_group_idx]), Int<1>{}, Int<1>{});

      if (cute::elect_one_sync()) {
        // Bringing tensormaps from params to smem for modification later
        copy(recast<uint128_t>(pD_tensormap), recast<uint128_t>(sD_tensormap));
        copy(recast<uint128_t>(pPostsum_tensormap), recast<uint128_t>(sPostsum_tensormap));
      }
      __syncwarp();
      return cute::make_tuple(&gmem_tensormap(sm_idx, warp_group_idx),
                              &gmem_tensormap(sm_idx, Postsum_tensormap_index + warp_group_idx));
    }
    }
  }

  // Replace address for the global tensor (to be done by single thread)
  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      int32_t next_batch,
      int32_t warp_group_idx) {
    // Replacing global_address for the next batch
    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        if (params.ptr_C != nullptr) {
          cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_C,
                                                          params.ptr_C[next_batch]);
        }
      }
      auto postsum_ptr = static_cast<ElementD*>(params.ptr_postsum_m) + params.postsum_m_batch_indices[next_batch];
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_postsum_load,
                                                      postsum_ptr);
    }
    else if constexpr (is_destination_supported) {
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_D[warp_group_idx],
                                                      params.ptr_D[next_batch]);
      auto postsum_ptr = static_cast<ElementD*>(params.ptr_postsum_m) + params.postsum_m_batch_indices[next_batch];
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_postsum_store[warp_group_idx],
                                                      postsum_ptr);
    }
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <bool IsLoad, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t warp_group_idx) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> postsum_shape  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> postsum_stride = {0,0,0,0,0};

    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        if (params.dC != nullptr) {
          ElementC const* ptr_C = nullptr;
          Tensor tensor_c = make_tensor(ptr_C, make_layout(make_shape(M,N,Int<1>{}), params.dC[next_group]));

          cute::detail::fill_tma_gmem_shape_stride(params.tma_load_c, tensor_c, 
                                                  prob_shape, prob_stride);
          // Convert strides to byte strides
          for (uint64_t& stride : prob_stride) {
            stride = (stride * sizeof_bits_v<ElementC>) / 8;
          }
          cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_C,
                                                                  prob_shape,
                                                                  prob_stride);
        }
      }
      NonVoidElementC const* ptr_postsum = nullptr;
      using PostsumStride = cute::conditional_t<is_m_major_D,
        decltype(make_stride(Int<1>{}, uint32_t{}, Int<0>{})),
        decltype(make_stride(uint32_t{}, Int<1>{}, Int<0>{}))>;
      PostsumStride dPostsum{};
      if constexpr (is_m_major_D) {
        get<1>(dPostsum) = M/2;
      }
      else {
        get<0>(dPostsum) = N/2;
      }
      Tensor tensor_postsum = make_tensor(ptr_postsum, make_layout(make_shape(4*M/2,N/2,Int<1>{}), dPostsum));
      cute::detail::fill_tma_gmem_shape_stride(params.tma_load_postsum_m, tensor_postsum,
                                               postsum_shape, postsum_stride);
      for (uint64_t& stride : postsum_stride) {
        stride = (stride * sizeof_bits_v<NonVoidElementC>) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_postsum_load,
                                                              postsum_shape,
                                                              postsum_stride);
    }
    else if constexpr (is_destination_supported) {
      ElementD const* ptr_D = nullptr;
      Tensor tensor_d = make_tensor(ptr_D, make_layout(make_shape(M,N,Int<1>{}), params.dD[next_group]));

      cute::detail::fill_tma_gmem_shape_stride(params.tma_store_d, tensor_d, 
                                               prob_shape, prob_stride);
      // Convert strides to byte strides
      for (uint64_t& stride : prob_stride) {
        stride = (stride * sizeof_bits_v<ElementD>) / 8;
      }

      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_D[warp_group_idx],
                                                              prob_shape,
                                                              prob_stride);

      NonVoidElementD const* ptr_postsum = nullptr;
      using PostsumStride = cute::conditional_t<is_m_major_D,
        decltype(make_stride(Int<1>{}, uint32_t{}, Int<0>{})),
        decltype(make_stride(uint32_t{}, Int<1>{}, Int<0>{}))>;
      PostsumStride dPostsum{};
      if constexpr (is_m_major_D) {
        get<1>(dPostsum) = M/2;
      }
      else {
        get<0>(dPostsum) = N/2;
      }
      Tensor tensor_postsum = make_tensor(ptr_postsum, make_layout(make_shape(4*M/2,N/2,Int<1>{}), dPostsum));
      cute::detail::fill_tma_gmem_shape_stride(params.tma_store_postsum_m, tensor_postsum,
                                               postsum_shape, postsum_stride);
      for (uint64_t& stride : postsum_stride) {
        stride = (stride * sizeof_bits_v<NonVoidElementD>) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_postsum_store[warp_group_idx],
                                                              postsum_shape,
                                                              postsum_stride);
    }
  }

  template <bool IsLoad, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& params,
      cute::TmaDescriptor const* tensormap,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch,
      int32_t warp_group_idx) {
    if constexpr (!IsMoEGemmKernel) {
      if (cute::elect_one_sync()) {
        // Replacing global_address for the next batch
        tensormaps_replace_global_address<IsLoad>(shared_tensormaps, params, next_batch, warp_group_idx);

        if constexpr (IsGroupedGemmKernel) {
          // Replacing global dims and strides for the next batch
          tensormaps_replace_global_tensor_properties<IsLoad>(
              shared_tensormaps, params, next_batch, problem_shape_mnkl, warp_group_idx);
        }

      }
    }
  }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release(
      TensorMapStorage& shared_tensormaps,
      cute::TmaDescriptor const* tensormap,
      cute::TmaDescriptor const* postsum_tensormap,
      const int32_t warp_group_idx = 0) {
    if constexpr (!IsMoEGemmKernel) {
    // Commit and wait for all TMA load/store instructions before updating the tensormap in gmem.
    // This operation only happens when the group/batch changes between consecutive tiles.
    // If there are no uncommitted instructions then tma_desc_commit_group results in an empty bulk async-group.
    auto tma_desc_wait_all_fn = [] () CUTLASS_LAMBDA_FUNC_INLINE {
      if (cute::elect_one_sync()) {
        cute::tma_desc_commit_group();
        cute::tma_desc_wait_group();
      }
    };
    // Entire warp must do this (ie its aligned)
    if constexpr (IsLoad) {
      if constexpr (is_source_supported) {
        tma_desc_wait_all_fn();
        tma_descriptor_cp_fence_release(tensormap, shared_tensormaps.smem_tensormap_C);
        tma_descriptor_cp_fence_release(postsum_tensormap, shared_tensormaps.smem_tensormap_postsum_load);
      }
    }
    else if constexpr (is_destination_supported) {
      tma_desc_wait_all_fn();
      tma_descriptor_cp_fence_release(tensormap, shared_tensormaps.smem_tensormap_D[warp_group_idx]);
      tma_descriptor_cp_fence_release(postsum_tensormap, shared_tensormaps.smem_tensormap_postsum_store[warp_group_idx]);
    }
    }
  }

  template <bool IsLoad>
  CUTLASS_DEVICE
  void tensormaps_fence_acquire(
      cute::TmaDescriptor const* tensormap,
      cute::TmaDescriptor const* postsum_tensormap) {
    if constexpr (!IsMoEGemmKernel) {
      if constexpr (IsLoad) {
        if constexpr (is_source_supported) {
          cute::tma_descriptor_fence_acquire(tensormap);
          cute::tma_descriptor_fence_acquire(postsum_tensormap);
        }
      }
      else {
        cute::tma_descriptor_fence_acquire(tensormap);
        cute::tma_descriptor_fence_acquire(postsum_tensormap);
      }
    }
  }

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
  cute::array<bool, ReuseSmemC ? StagesC : 1> store_uses_load_stage{};
  int issued_stores = 0;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
