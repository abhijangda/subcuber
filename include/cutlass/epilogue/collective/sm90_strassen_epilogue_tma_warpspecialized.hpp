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
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/device/strassen_decls.h"

#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"

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
    Sm90TmaWarpSpecialized<StagesC_,StagesD_,FragmentSize_,ReuseSmemC_,DelayTmaStore_>,
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
  using DispatchPolicy = Sm90TmaWarpSpecialized<StagesC_,StagesD_,FragmentSize_,ReuseSmemC_,DelayTmaStore_>;
  using CtaTileMNK = CtaTileMNK_;
  using EpilogueTile = EpilogueTile_;
  using FusionCallbacks = FusionCallbacks_;
  using ElementC = ElementC_;
  using StrideC = StrideC_;
  using ElementD = ElementD_;
  using StrideD = StrideD_;
  using CopyOpG2S = CopyOpG2S_;
  using SmemLayoutAtomC = SmemLayoutAtomC_;
  using CopyOpS2R = CopyOpS2R_;
  using CopyOpS2G = CopyOpS2G_;
  using SmemLayoutAtomD = SmemLayoutAtomD_;
  using CopyOpR2S = CopyOpR2S_;
  using CopyAtomC = CopyAtomC_;
  using CopyOpR2R = CopyOpR2R_;
  using ProblemShape = ProblemShape_;
  using SubMatLayoutC = SubMatLayoutC_;

  using ThreadEpilogueOp = typename epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
  using GmemTiledCopyC = CopyOpG2S;
  using GmemTiledCopyD = CopyOpS2G;

  static_assert(!is_layout<EpilogueTile>::value && is_tuple<EpilogueTile>::value, "EpilogueTile must be a cute::Tile or cute::Shape");
  static_assert(cute::rank(CtaTileMNK{}) == 3, "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
  static_assert(cute::rank(EpilogueTile{}) == 2, "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");
  static_assert(size<0>(CtaTileMNK{}) % size<0>(shape(EpilogueTile{})) == 0, "EPI_TILE_M must divide CTA_M");
  static_assert(size<1>(CtaTileMNK{}) % size<1>(shape(EpilogueTile{})) == 0, "EPI_TILE_N must divide CTA_N");
  static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
  static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");
  constexpr static bool UseLinearStoreLoads = true;
  constexpr static int StagesC = StagesC_;
  
private:
  constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;
  using NonVoidElementD = cute::conditional_t<not is_destination_supported,fusion::get_element_aux_t<FusionCallbacks>, ElementD>;
  static_assert(not cute::is_void_v<NonVoidElementD>, "SmemElementD is void");
  using NonVoidElementC = cute::conditional_t<not is_source_supported,NonVoidElementD,ElementC>; // prevents void ref breakages

  using TmaElementD = cute::conditional_t<cute::is_same_v<NonVoidElementD, cutlass::complex<float>>, uint64_t, NonVoidElementD>;
  using TmaElementC = cute::conditional_t<cute::is_same_v<NonVoidElementC, cutlass::complex<float>>, uint64_t, NonVoidElementC>;

  using SmemElementC = typename cutlass::detail::get_unpacked_element_type<NonVoidElementC>::type;
  using SmemElementD = typename cutlass::detail::get_unpacked_element_type<NonVoidElementD>::type;

  constexpr static int StagesD = StagesD_;
  constexpr static bool ReuseSmemC = ReuseSmemC_ and is_destination_supported;
  constexpr static bool DelayTmaStore = DelayTmaStore_;

  constexpr static bool is_m_major_C = detail::is_m_major<StrideC>();
  constexpr static bool is_m_major_D = detail::is_m_major<StrideD>();

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

  using SmemLayout1dM = decltype(make_layout(
    make_shape(size<0>(EpilogueTile{})), //*size<1>(EpilogueTile{})
    LayoutRight{}));
  
  static const bool IsFusedM2M3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();
  static const bool IsFusedM4M5 = StrassenMiGroup::hasM4() && StrassenMiGroup::hasM5();

  constexpr static bool support_smem_reuse = is_destination_supported && StagesD <= StagesC
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

  const static int PresumStages = 1;
  using PresumTileShapeA = Shape<_2,_128>;
  using PresumTileShapeB = Shape<_2,_128>;
  using PresumSmemLayoutA = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));
  using PresumSmemLayoutB = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));

  using PresumSmemShapeA__ = decltype(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{}), Int<PresumStages>{}));
  using PresumSmemLayoutA__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{})), LayoutRight{}));

  static constexpr uint32_t TmaTransactionBytesPresum =
        cutlass::bits_to_bytes(size<0>(PresumSmemLayoutA{}) * size<1>(PresumSmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementD>::value));

  // using EpilogueTile1D = cute::tuple<cute::Int<64*16>>;
  using EpilogueTile1D = decltype(make_shape(shape<0>(EpilogueTile{}) * shape<1>(EpilogueTile{})));

  static constexpr uint32_t TmaTransactionLoadM1DBytes = 
        cutlass::bits_to_bytes(get<0>(EpilogueTile1D{}) * static_cast<uint32_t>(sizeof_bits<ElementD>::value));

  struct CollectiveStorageWithC {
    alignas(SmemAlignmentC) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
  };

  union CollectiveStorageWithoutC {
    cute::array<SmemElementC, 0> smem_C;
    ArrayEngine<SmemElementD, 0> smem_C2;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_Postsum;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
    alignas(SmemAlignmentD) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D2;
  };

  template<bool HasD2>
  union CollectiveStorageReuseCWithoutPostsumAndD2_ {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutC>> smem_Postsum;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutC>> smem_C2;
    struct {
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
      alignas(int(HasD2)*MaxSmemAlignment) ArrayEngine<SmemElementD, int(HasD2)*cosize_v<SmemLayoutD>> smem_D2;
    };
  };

  template<bool HasD2>
  struct CollectiveStorageReuseCWithPostsum_ {
    union {
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutC>> smem_Postsum;
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutC>> smem_C2;
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D2;
    };
    union {
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
    };
  };

  struct CollectiveStorageReuseCWithPostsumAndC2_ {
    union {
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutC>> smem_Postsum;
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D2;
    };
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C2;
    union {
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, cosize_v<SmemLayoutC>> smem_C;
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
    };
  };

  union CollectiveStorageReuseCFusedM2M3Linear {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, 128*128> smem_C;
    struct {
      alignas(MaxSmemAlignment) SmemElementC f[128*128 - cosize_v<SmemLayoutD>*StrassenMiGroup::RWCTypes::NumGlobalStores()];
      alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
    };
  };

  union CollectiveStorageReuseCFusedM4M5Linear {
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, 128*128> smem_C;
    alignas(MaxSmemAlignment) ArrayEngine<SmemElementD, cosize_v<SmemLayoutD>> smem_D;
    // alignas(MaxSmemAlignment) ArrayEngine<SmemElementC, 2*8192> smem_reg_transfer;
  };

  static constexpr int MaxNumGLLoads = StrassenMiGroup::template MaxNumGlobalLoadsPerMi<ElementC>();

  //In 2 Level outer M4's inner M1,M5,M6 and M4 does 2 stores and 2 loads 
  static constexpr bool Level2StoreM4_ = StrassenMiGroup::Level == 1 && StrassenMiGroup::Level1Idx == 4;
  //In 2 Level outer M5's, inner M1 & M6 does 1 store and 2 loads (C and C2); M5 and M4 does 1 store and 3 loads (Postsum, C, and C2)
  static constexpr bool Level2LoadsC2 = StrassenMiGroup::Level == 1 && StrassenMiGroup::Level1Idx == 5;

#if defined(L2_SCHED_TMA_REDUCE_ADD)
  static const bool l2_sched_tma_reduce_add = true;
#else
  static const bool l2_sched_tma_reduce_add = false;
#endif

  using CollectiveStorageReuseC = cute::conditional_t<(Level2StoreM4_ || Level2LoadsC2) && l2_sched_tma_reduce_add,
                                                      CollectiveStorageReuseCWithoutPostsumAndD2_<Level2StoreM4_>,
                                                      cute::conditional_t<MaxNumGLLoads + int(Level2LoadsC2) >= 3,
                                                                          CollectiveStorageReuseCWithPostsumAndC2_,
                                                                          cute::conditional_t<MaxNumGLLoads + int(Level2LoadsC2) >= 2  || Level2StoreM4_,
                                                                                              CollectiveStorageReuseCWithPostsum_<Level2StoreM4_>,
                                                                                              CollectiveStorageReuseCWithoutPostsumAndD2_<Level2StoreM4_>>>
                                                     >;
  using RWCTypes = typename StrassenMiGroup::RWCTypes;

public:
  static constexpr bool Level2StoreM4 = Level2StoreM4_;
  static const size_t PresumSmemSize = size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{})*PresumStages;
  struct PresumTensorStorage : cute::aligned_struct<128, _0> {
  #if 0
    cute::array_aligned<SmemElementD, PresumSmemSize> smem_A0;
    cute::array_aligned<SmemElementD, PresumSmemSize> smem_A1;
    cute::array_aligned<SmemElementD, PresumSmemSize> smem_A2;
    cute::array_aligned<SmemElementD, PresumSmemSize> smem_A3;
  #endif
  };
  // TMA pipeline for loading C
  using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;
  using LoadPipelineState = cutlass::PipelineState<StagesC>;
  constexpr static uint32_t TmaTransactionBytes =
    (size(take<0,2>(SmemLayoutC{})) * static_cast<uint32_t>(sizeof_bits<SmemElementC>::value)) / 8;
  constexpr static bool RequiresTransactionBytes = true;

  // TMA pipeline for storing D
  using StorePipeline = cute::conditional_t<ReuseSmemC,
                          cutlass::PipelineTmaStore<StagesC, StagesD-1>,
                          cutlass::PipelineTmaStore<StagesD>>;
  using StorePipelineState = cutlass::PipelineState<ReuseSmemC ? StagesC : StagesD>;

  struct SharedStorage {
    struct TensorStorage {
      using FusionStorage = typename FusionCallbacks::SharedStorage;
      using CollectiveStorage = cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                  cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
      CollectiveStorage collective;
      FusionStorage thread;
    } tensors;

    using PipelineStorage = typename LoadPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename FusionCallbacks::Arguments thread{};
    ElementC const* ptr_C;
    StrideC dC;
    ElementD const* ptr_D;
    StrideD dD;
    ElementC const* ptr_C2 = nullptr;
    StrideC dC2;
    ElementD const* ptr_D2 = nullptr;
    StrideD dD2;

    Arguments(typename FusionCallbacks::Arguments thread, ElementC const* ptr_C, StrideC dC,
              ElementD const* ptr_D, StrideD dD,
              ElementC const* ptr_C2 = nullptr, ElementD const* ptr_D2 = nullptr)
              : thread(thread), ptr_C(ptr_C), dC(dC), ptr_D(ptr_D), dD(dD),
                ptr_C2(ptr_C2), dC2(dD), ptr_D2(ptr_D2), dD2(dD)
              {
                thread.beta = 0;
              }

    Arguments(typename FusionCallbacks::Arguments thread, ElementC const* ptr_C, StrideC dC,
              ElementD const* ptr_D, StrideD dD,
              ElementC const* ptr_C2, StrideC dC2, ElementD const* ptr_D2, StrideD dD2)
              : Arguments(thread, ptr_C, dC, ptr_D, dD, ptr_C2, ptr_D2)
              {
                this->dC2 = dC2;
                this->dD2 = dD2;
              }

    template<typename Other>
    Arguments(const Other& other) : ptr_C(other.ptr_C), dC(other.dC),
                                     ptr_D(other.ptr_D), dD(other.dD), dC2(other.dD), dD2(other.dD)
              {
                thread.alpha = other.thread.alpha;
                thread.beta = other.thread.beta;
                if constexpr (requires { other.ptr_C2; other.dC2; }) {
                  ptr_C2 = other.ptr_C2;
                  dC2 = other.dC2;
                }
                if constexpr (requires { other.ptr_D2; other.dD2; }) {
                  ptr_D2 = other.ptr_D2;
                  dD2 = other.dD2;
                }
              }
  };

  using StrideM1D = cute::tuple<_1>;

  // Device side epilogue params
  struct Params {
    using TMA_C = decltype(make_tma_copy(
        CopyOpG2S{},
        make_tensor(make_gmem_ptr<TmaElementC const>(nullptr),
            repeat_like(StrideC{}, int32_t(0)), StrideC{}),
        take<0,2>(SmemLayoutC{}),
        EpilogueTile{},
        _1{}));
    
    using TMA_C_1D = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(make_gmem_ptr<TmaElementC const>(nullptr),
            repeat_like(StrideM1D{}, int32_t(0)), StrideM1D{}),
        SmemLayout1dM{}));

    using TMA_D = decltype(make_tma_copy(
        CopyOpS2G{},
        make_tensor(make_gmem_ptr<TmaElementD>(nullptr),
            repeat_like(StrideD{}, int32_t(0)), StrideD{}),
        take<0,2>(SmemLayoutD{}),
        EpilogueTile{},
        _1{}));

    using TMA_D_ADD = decltype(make_tma_copy(
      SM90_TMA_REDUCE_ADD{},
      make_tensor(make_gmem_ptr<TmaElementD>(nullptr),
        repeat_like(StrideD{}, int32_t(0)), StrideD{}),
      take<0,2>(SmemLayoutD{}),
      EpilogueTile{},
      _1{}));
    
    // using TMA_D = decltype(make_tma_copy(
    //     CopyOpS2G{},
    //     make_tensor(make_gmem_ptr<TmaElementD>(nullptr),
    //         repeat_like(StrideD{}, int32_t(0)), StrideD{}),
    //     take<0,2>(SmemLayoutD{}),
    //     EpilogueTile{},
    //     _1{}));

    typename FusionCallbacks::Params thread{};
    TMA_C tma_load_c;
    TMA_C tma_load_postsum_m;
    TMA_D tma_store_postsum_m;
    TMA_D tma_store_d;
    TMA_D_ADD tma_add_postsum_m;
    TMA_D_ADD tma_add_d;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;

    using TMA_PresumLoad_A = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<cutlass::half_t const*>(nullptr),
            repeat_like(StrideD{}, int32_t(0)), StrideD{}),
          PresumSmemLayoutA__{}));

    TMA_PresumLoad_A tma_load_presumld_a;
    TMA_C_1D tma_load_linear_m;
    uint32_t tma_transaction_bytes_presum_mk = TmaTransactionBytesPresum;
    void* ptr_postsum_m;
    TMA_C tma_load_c2{};
    TMA_D tma_store_d2{};
    TMA_D_ADD tma_add_d2{};
    ElementC const* ptr_C2 = nullptr;
    ElementD const* ptr_D2 = nullptr;
  };

  //
  // Methods  
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const& problem_shape,
      Arguments const& args,
      ElementD* ptr_presum_load_A,
      ElementD* postsum_m,
      [[maybe_unused]] void* workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    uint32_t transaction_bytes = TmaTransactionBytes;
    typename Params::TMA_C tma_load_c{};
    typename Params::TMA_C tma_load_c2{};
    if constexpr (is_source_supported) {
      Tensor tensor_c = make_tensor(make_gmem_ptr<TmaElementC const>(args.ptr_C), make_layout(make_shape(M,N,L), args.dC));
      tma_load_c = make_tma_copy_C_sm90(
          CopyOpG2S{},
          tensor_c,
          take<0,2>(SmemLayoutC{}),
          EpilogueTile{});
      if (args.ptr_C2 != nullptr) {
        Tensor tensor_c2 = make_tensor(make_gmem_ptr<TmaElementC const>(args.ptr_C2), make_layout(make_shape(M,N,L), args.dC2));
        tma_load_c2 = make_tma_copy_C_sm90(
            CopyOpG2S{},
            tensor_c2,
            take<0,2>(SmemLayoutC{}),
            EpilogueTile{});
      }
    }

    typename Params::TMA_C tma_load_postsum_m{};

    Tensor tensor_load_m = make_tensor(make_gmem_ptr<TmaElementC const>(postsum_m), make_layout(make_shape(4*M/2,N/2,L), make_stride(N/2, get<1>(args.dD), get<2>(args.dD))));
    tma_load_postsum_m = make_tma_copy_C_sm90(
        CopyOpG2S{},
        tensor_load_m,
        take<0,2>(SmemLayoutD{}),
        EpilogueTile{});

    typename Params::TMA_D tma_store_postsum_m{};
    Tensor tensor_store_m = make_tensor(make_gmem_ptr<TmaElementD>(postsum_m), make_layout(make_shape(4*M/2,N/2,L), make_stride(N/2, get<1>(args.dD), get<2>(args.dD))));

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
    
    typename Params::TMA_PresumLoad_A tma_load_presumld_a{};
    Tensor tensor_presum_ld_a = make_tensor(make_gmem_ptr<TmaElementD>(ptr_presum_load_A), make_layout(make_shape(M,N,L), make_stride(get<0>(args.dD), get<1>(args.dD), get<2>(args.dD))));

    tma_load_presumld_a = make_tma_copy(
        SM90_TMA_LOAD{},
        tensor_presum_ld_a,
        PresumSmemLayoutA__{});

    typename Params::TMA_D tma_store_d{};
    typename Params::TMA_D tma_store_d2{};
    typename Params::TMA_D_ADD tma_add_d{};
    typename Params::TMA_D_ADD tma_add_d2{};
    if constexpr (is_destination_supported) {
      Tensor tensor_d = make_tensor(make_gmem_ptr<TmaElementD>(args.ptr_D), make_layout(make_shape(M,N,L), args.dD));
      tma_store_d = make_tma_copy_C_sm90(
          CopyOpS2G{},
          tensor_d,
          take<0,2>(SmemLayoutD{}),
          EpilogueTile{});
      tma_add_d = make_tma_copy_C_sm90(
          SM90_TMA_REDUCE_ADD{},
          tensor_d,
          take<0,2>(SmemLayoutD{}),
          EpilogueTile{});
      if (args.ptr_D2 != nullptr) {
        Tensor tensor_d2 = make_tensor(make_gmem_ptr<TmaElementD>(args.ptr_D2), make_layout(make_shape(M,N,L), args.dD2));
        tma_store_d2 = make_tma_copy_C_sm90(
            CopyOpS2G{},
            tensor_d2,
            take<0,2>(SmemLayoutD{}),
            EpilogueTile{});

        tma_add_d2 = make_tma_copy_C_sm90(
            SM90_TMA_REDUCE_ADD{},
            tensor_d2,
            take<0,2>(SmemLayoutD{}),
            EpilogueTile{});
      }
    }

    // typename decltype(make_stride(1))::x y;

    Tensor tensor_postsum_1d = make_tensor(make_gmem_ptr<TmaElementD>(postsum_m),
                                           make_layout(make_shape(4*M/2*N/2), StrideM1D{}));

    typename Params::TMA_C_1D tma_load_linear_m{};
    tma_load_linear_m = make_tma_copy(SM90_TMA_LOAD{},
                                      tensor_postsum_1d,
                                      SmemLayout1dM{});

    return {
      FusionCallbacks::to_underlying_arguments(problem_shape, args.thread, workspace),
      tma_load_c,
      tma_load_postsum_m,
      tma_store_postsum_m,
      tma_store_d,
      tma_add_postsum_m,
      tma_add_d,
      transaction_bytes,
      tma_load_presumld_a,
      tma_load_linear_m,
      TmaTransactionBytesPresum,
      postsum_m,
      tma_load_c2,
      tma_store_d2,
      tma_add_d2,
      args.ptr_C2,
      args.ptr_D2
    };
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return FusionCallbacks::get_workspace_size(problem_shape, args.thread);
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
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    auto shape = cute::make_shape(M,N,L);

    bool implementable = true;
    if constexpr (is_destination_supported) {
      constexpr int tma_alignment_bits_D = cutlass::detail::get_output_alignment_bits<ElementD>();
      constexpr int min_tma_aligned_elements_D = tma_alignment_bits_D / cutlass::sizeof_bits<ElementD>::value;
      if constexpr (cute::is_same_v<CopyOpS2G, SM90_TMA_STORE_IM2COL>) { // ignore L stride for implicit gemm
        implementable = cutlass::detail::check_alignment<min_tma_aligned_elements_D>(take<0,2>(shape), take<0,2>(StrideD{}));
      }
      else {
        implementable = cutlass::detail::check_alignment<min_tma_aligned_elements_D>(shape, StrideD{});
      }
    }

    if constexpr (not cute::is_void_v<ElementC>) {
      constexpr int tma_alignment_bits_C = cutlass::detail::get_input_alignment_bits<ElementC>();
      constexpr int min_tma_aligned_elements_C = tma_alignment_bits_C / cutlass::sizeof_bits<ElementC>::value;
      if constexpr (cute::is_same_v<CopyOpG2S, SM90_TMA_LOAD_IM2COL>) { // ignore L stride for implicit gemm
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(take<0,2>(shape), take<0,2>(StrideC{}));
      }
      else {
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(shape, StrideC{});
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }

    bool fusion_implementable = FusionCallbacks::can_implement(problem_shape, args.thread);

    if (!fusion_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for FusionCallbacks.\n");
    }

    bool beta_implementable = true;

    if constexpr (not is_source_supported) {
      if constexpr (detail::has_beta<Arguments>::value) {
        beta_implementable = args.thread.beta == 0.0;
      }
      if constexpr (detail::has_beta_ptr<Arguments>::value) {
        beta_implementable = beta_implementable && args.thread.beta_ptr == nullptr;
      }
    }

    if (!beta_implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Beta must be zero when ElementC is void.\n");
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

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void
  prefetch_tma_descriptors(Params const& epilogue_params) {
    if constexpr (is_source_supported) {
      cute::prefetch_tma_descriptor(epilogue_params.tma_load_c.get_tma_descriptor());
      if (epilogue_params.ptr_C2 != nullptr) {
        cute::prefetch_tma_descriptor(epilogue_params.tma_load_c2.get_tma_descriptor());
      }
    }
    if constexpr (is_destination_supported) {
      cute::prefetch_tma_descriptor(epilogue_params.tma_store_d.get_tma_descriptor());
      if (epilogue_params.ptr_D2 != nullptr) {
        cute::prefetch_tma_descriptor(epilogue_params.tma_store_d2.get_tma_descriptor());
      }
    }

    cute::prefetch_tma_descriptor(epilogue_params.tma_store_postsum_m.get_tma_descriptor());
    cute::prefetch_tma_descriptor(epilogue_params.tma_load_postsum_m.get_tma_descriptor());
  }

  CUTLASS_HOST_DEVICE
  CollectiveStrassenEpilogue(Params const& params_, TensorStorage& shared_tensors)
      : params(params_), fusion_callbacks(params_.thread, shared_tensors.thread) {}

  CUTLASS_DEVICE
  bool
  is_producer_load_needed() const {
    return StrassenMiGroup::NumMisWithGLLoads() > 0 ||
           (is_source_supported && fusion_callbacks.is_C_load_needed());
  }

  CUTLASS_DEVICE
  bool
  is_C_load_needed() const {
    return is_source_supported && fusion_callbacks.is_C_load_needed();
  }

  template<
    class ProblemShapeMNKL
  >
  CUTLASS_DEVICE decltype(auto)
  get_load_tma(ProblemShapeMNKL problem_shape_mnkl, int sub_m_idx, PostsumOp global_src) {
    auto [M, N, K, L] = problem_shape_mnkl;
    Tensor postsum_m = params.tma_load_postsum_m.get_tma_tensor(make_shape(M/2,N/2,L));
    auto m0_ptr = postsum_m.data() + make_coord(0,0,_);
    auto m0 = make_tensor(m0_ptr, postsum_m.layout());

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
  }

  template<
    bool SecondSource = false,
    class ProblemShapeMNKL
  >
  CUTLASS_DEVICE decltype(auto)
  get_c_load_tma(ProblemShapeMNKL problem_shape_mnkl, int sub_m_idx) {
    auto [M, N, K, L] = problem_shape_mnkl;
    auto const& tma_load = SecondSource ? params.tma_load_c2 : params.tma_load_c;
    Tensor c = tma_load.get_tma_tensor(make_shape(M, N, L));
    auto c0_ptr = c.data() + make_coord(0, 0, _);
    auto c0 = make_tensor(c0_ptr, c.layout());

    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
      auto global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(i);
      int sign_mi = RWCTypes::MiSignByOutputIndex(i, StrassenMiGroup::getMi(sub_m_idx));
      if (global_dest.valid() && global_dest.is_mem_global() &&
          global_dest.is_layout_final() && sign_mi != 0) {
        auto c_ptr = c.data() + make_coord((global_dest.get_op() % 2) * N / 2,
                                           (global_dest.get_op() / 2) * M / 2, _);
        return make_tensor(c_ptr, c.layout());
      }
    }

    return c0;
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
  get_store_tma(ProblemShapeMNKL problem_shape_mnkl, int sub_m_idx, bool matrix_or_linear, PostsumOp global_srcs[4], bool& use_tma_reduce) {
    auto [M, N, K, L] = problem_shape_mnkl;
    Tensor postsum_m = params.tma_store_postsum_m.get_tma_tensor(make_shape(M/2,N/2,L));
    auto m0_ptr = postsum_m.data() + make_coord(0,0,_);
    auto m0 = make_tensor(m0_ptr, postsum_m.layout());
  
    Tensor d = params.tma_store_d.get_tma_tensor(make_shape(M, N, L));
    auto d0_ptr = d.data() + make_coord(0,0,_);
    auto d0 = make_tensor(d0_ptr, d.layout());
    bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();
    global_srcs[0] = MmaStrassen::PostsumOp(); global_srcs[1] = MmaStrassen::PostsumOp();
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
      auto global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(i);
      int signMi = RWCTypes::MiSignByOutputIndex(i, StrassenMiGroup::getMi(sub_m_idx));
      const bool satisfy_layout = matrix_or_linear == true ? global_dest.is_layout_final() || global_dest.is_layout_interim_matrix() : global_dest.is_layout_interim_linear();
      if (global_dest.valid() && global_dest.is_mem_global() && satisfy_layout && signMi != 0) {
        PostsumOp src0 = RWCTypes::PostsumSrcByOutputIndex(i, 0);
        PostsumOp src1 = RWCTypes::PostsumSrcByOutputIndex(i, 1);
        if (global_dest.is_layout_final()) {
          auto d_ptr = d.data() + make_coord((global_dest.get_op()%2)*N/2,
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
          return cute::tuple(make_tensor(d_ptr, d.layout()), global_dest, false);
        } else if (global_dest.is_layout_interim_matrix() || global_dest.is_layout_interim_linear()) {
          auto m_ptr = postsum_m.data() + make_coord(0,global_dest.get_op()*M/2,_);
          int idx = 0;
          use_tma_reduce = false;
          if (src0.valid() && src0.is_mem_global())// && src0.get_op() != global_dest.get_op() && src0.is_layout_interim_linear()
            global_srcs[idx++] = src0;
          if (src1.valid() && src1.is_mem_global())// && src1.get_op() != global_dest.get_op() && src1.is_layout_interim_linear()
            global_srcs[idx++] = src1;

          return cute::tuple(make_tensor(m_ptr, postsum_m.layout()), global_dest, true);
        }
      }
    }

    return cute::tuple(d0, MmaStrassen::PostsumOp(), false);
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class TiledMma
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
      PostsumOp src_global_ops[4],
      bool is_dest_final,
      int subtile_idx=-1) {
    using namespace cute;

    // Indexing variables
    auto [M, N, K, L] = problem_shape_mnkl;
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

    // The tma tensor C under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape = conditional_return<is_im2col_C>(
      make_coord(m_coord, n_coord),
      make_coord(m_coord, n_coord, l_coord));

    // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
    bool OutputDorM = StrassenMiGroup::hasM0(); //true for D and false for M
    
    auto first_load_tuple = get_load_tma(problem_shape_mnkl, sub_m_idx, src_global_ops[0]);
    bool has_first_load = src_global_ops[0].valid();
    Tensor mPostsum_mn = get<0>(first_load_tuple);                        //       (M,N,L)
    auto& tma_load_postsum = params.tma_load_postsum_m;
    Tensor mPostsum = coalesce(mPostsum_mn, take<0,2>(CtaTileMNK{}));
    Tensor gPostsum = local_tile(mPostsum, take<0,2>(CtaTileMNK{}), coord_shape);                       // (CTA_M,CTA_N)

    Tensor mC_mn = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (is_source_supported) {
        return get_c_load_tma(problem_shape_mnkl, sub_m_idx);
      }
      else {
        return mPostsum_mn;
      }
    }();
    Tensor mC = coalesce(mC_mn, take<0,2>(CtaTileMNK{}));
    Tensor gC = local_tile(mC, take<0,2>(CtaTileMNK{}), coord_shape);

    Tensor mC2_mn = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (Level2LoadsC2 && is_source_supported) {
        if (params.ptr_C2 != nullptr) {
          return get_c_load_tma<true>(problem_shape_mnkl, sub_m_idx);
        }
      }
      return mC_mn;
    }();
    Tensor mC2 = coalesce(mC2_mn, take<0,2>(CtaTileMNK{}));
    Tensor gC2 = local_tile(mC2, take<0,2>(CtaTileMNK{}), coord_shape);

    auto second_load_tuple = get_load_tma(problem_shape_mnkl, sub_m_idx, src_global_ops[1]);
    bool has_second_load = is_source_supported && src_global_ops[1].valid();
    Tensor mPostsum2_mn = get<0>(get_second_load_tma(problem_shape_mnkl));
    Tensor mPostsum2 = coalesce(mPostsum2_mn, take<0,2>(CtaTileMNK{}));
    Tensor gPostsum2 = local_tile(mPostsum2, take<0,2>(CtaTileMNK{}), coord_shape);

    // Apply epilogue subtile, get matching smem tensor
    auto ptr_sPostsum = shared_tensors.collective.smem_Postsum.begin();
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    Tensor gPostsum_epi = flat_divide(gPostsum, EpilogueTile{});                 // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor gC_epi = flat_divide(gC, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sPostsum_epi = make_tensor(make_smem_ptr(ptr_sPostsum), SmemLayoutC{}); //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sC_epi = make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{});           //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    auto ptr_sC2 = shared_tensors.collective.smem_C2.begin();
    Tensor gC2_epi = flat_divide(gC2, EpilogueTile{});
    Tensor sC2_epi = make_tensor(make_smem_ptr(ptr_sC2), SmemLayoutC{});

    auto ptr_sPostsum2 = ptr_sPostsum + cosize_v<SmemLayoutC>;
    Tensor gPostsum2_epi = flat_divide(gPostsum2, EpilogueTile{});                // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sPostsum2_epi = make_tensor(make_smem_ptr(ptr_sPostsum2), SmemLayoutC{}); //      (EPI_TILE_M,EPI_TILE_N,PIPE_C)

    // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
    ThrCopy thrblk_g2s_postsum = tma_load_postsum.get_slice(Int<0>{});
    Tensor bGS_gPostsum = thrblk_g2s_postsum.partition_S(gPostsum_epi);                  // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sPostsum = thrblk_g2s_postsum.partition_D(sPostsum_epi);                   // (G2S,G2S_M,G2S_N,PIPE_C)
    Tensor bGS_sPostsumC = thrblk_g2s_postsum.partition_D(sC_epi);                       // (G2S,G2S_M,G2S_N,PIPE_C)

    ThrCopy thrblk_g2s_c = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (is_source_supported) {
        return params.tma_load_c.get_slice(Int<0>{});
      }
      else {
        return tma_load_postsum.get_slice(Int<0>{});
      }
    }();
    Tensor bGS_gC = thrblk_g2s_c.partition_S(gC_epi);                                   // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sC = thrblk_g2s_c.partition_D(sC_epi);                                   // (G2S,G2S_M,G2S_N,PIPE_C)
    Tensor bGS_gC2 = thrblk_g2s_c.partition_S(gC2_epi);
    Tensor bGS_sC2 = thrblk_g2s_c.partition_D(sC2_epi);

    ThrCopy thrblk_g2s_postsum2 = tma_load_postsum.get_slice(Int<0>{});
    Tensor bGS_gPostsum2 = thrblk_g2s_postsum2.partition_S(gPostsum2_epi);                // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
    Tensor bGS_sPostsum2 = thrblk_g2s_postsum2.partition_D(sPostsum2_epi);                 // (G2S,G2S_M,G2S_N,PIPE_C)
    
    cutlass::Array<ElementD, 8>* ptr_smem2 = (cutlass::Array<ElementD, 8>*)ptr_sPostsum2;

    // Get the fusion callbacks for the producer load warp
    auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs(
                      problem_shape_mnkl,
                      CtaTileMNK{},
                      tile_coord_mnkl,
                      tiled_mma,
                      EpilogueTile{},
                      thread_idx
                    );
    auto pld_callbacks = fusion_callbacks.get_producer_load_callbacks(pld_args);
    //TODO: Has to hard code for M2 & M3 with TMA Reduce
    //If uses tma and M0, M2, or M3, then do not is_C_load_needed should be false
    const bool mi_can_load_C = StrassenMiGroup::getMi(sub_m_idx) != 0 &&
                   StrassenMiGroup::getMi(sub_m_idx) != 2 && StrassenMiGroup::getMi(sub_m_idx) != 3;
    bool is_C_load_needed = is_source_supported && is_dest_final && fusion_callbacks.is_C_load_needed() && mi_can_load_C;

    // Predication for TMA load (one thread issues TMA load)
    bool issue_tma_load = cute::elect_one_sync();

    // Pre-loop fusion callback entry point
    pld_callbacks.begin();

    ElementD* postsum_m0 = (ElementD*)params.ptr_postsum_m + (m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{});
    uint STAGE_ELEMS = size<0>(EpilogueTile{}) * size<1>(EpilogueTile{});

    if (StrassenMiGroup::hasM5()) {
      //M5 loads M2
      postsum_m0 = postsum_m0 + (M/2)*(N/2);
    } 
    //TODO: These two loops are swapped from original case same for store
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
        if (issue_tma_load && (has_first_load || is_C_load_needed)) {
          if (has_first_load) {
            copy(tma_load_postsum.with(*tma_barrier, mcast_mask),
                bGS_gPostsum(_,_,_,epi_m,epi_n), bGS_sPostsum(_,_,_,load_pipe_producer_state.index()));
            load_pipeline.producer_expect_transaction(load_pipe_producer_state);
          }
          if constexpr (is_source_supported) {
            if (is_C_load_needed) {
              copy(params.tma_load_c.with(*tma_barrier, mcast_mask),
                bGS_gC(_,_,_,epi_m,epi_n), bGS_sC(_,_,_,load_pipe_producer_state.index()));
              load_pipeline.producer_expect_transaction(load_pipe_producer_state);
              if constexpr (Level2LoadsC2) {
                if (params.ptr_C2 != nullptr) {
                  copy(params.tma_load_c2.with(*tma_barrier, mcast_mask),
                    bGS_gC2(_,_,_,epi_m,epi_n), bGS_sC2(_,_,_,load_pipe_producer_state.index()));
                  load_pipeline.producer_expect_transaction(load_pipe_producer_state);
                }
              }
            }
          }
          if (has_second_load) {
            if (StrassenMiGroup::hasM5() and UseLinearStoreLoads) {
              SM90_BULK_COPY_G2S().copy(postsum_m0, tma_barrier, &ptr_smem2[load_pipe_producer_state.index()*STAGE_ELEMS/8],
                                        STAGE_ELEMS * sizeof(ElementD));
            } else {
              copy(tma_load_postsum.with(*tma_barrier, mcast_mask),
                bGS_gPostsum2(_,_,_,epi_m,epi_n), bGS_sPostsum2(_,_,_,load_pipe_producer_state.index()));
            }
            load_pipeline.producer_expect_transaction(load_pipe_producer_state);
          }
        }

        // Commit TMA loads for this stage and release the lock
        load_pipeline.producer_commit(load_pipe_producer_state);
        ++load_pipe_producer_state;
        postsum_m0 += STAGE_ELEMS;
      }
    }

    // Post-loop fusion callback entry point
    pld_callbacks.end();

    return load_pipe_producer_state;
  }

  CUTLASS_DEVICE auto
  load_tail(
      LoadPipeline load_pipeline,
      LoadPipelineState load_pipe_producer_state) {
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
    ElementD* postsum_m0 = (ElementD*)params.ptr_postsum_m + (m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{});
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
    CUTLASS_PRAGMA_NO_UNROLL
    for (uint stage = 0; stage < size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{}); stage += STAGE_ELEMS) {
      uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);
      load_pipeline.producer_acquire(load_pipe_producer_state);
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
    postsum_m0 = postsum_m0 + (m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{});

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

struct SM90_BULK_TMA_ADD_S2G
{
  CUTE_HOST_DEVICE static void
  copy(void const* smem_ptr,
       void      * gmem_ptr, int32_t store_bytes)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], %2;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_COPY without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};


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

    postsum_m0 = postsum_m0 + (m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{});
    postsum_m0 = postsum_m0 + (dest_global_op.get_op())*(M/2)*(N/2);

    //Multiple stage using TMA
    constexpr uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{})) / NumMMAThreads;
    bool issue_tma_store = thread_idx == 0;
    
    const bool is_producer_load_needed = src_global_op.valid() && src_global_op.is_mem_global() && src_global_op.is_layout_interim_linear();

    cutlass::Array<ElementD, 8>* ptr_smem_st = (cutlass::Array<ElementD, 8>*)epilogue_tensors.collective.smem_D.begin();
                                            // ((is_producer_load_needed) ?
                                                  // epilogue_tensors.collective.smem_C.begin() :
                                                  // epilogue_tensors.collective.smem_D.begin());

    LoadPipelineState load_wait_state = load_pipe_consumer_state;
    if constexpr (ReuseSmemC) {
      load_wait_state = store_pipe_producer_state;
      load_wait_state.phase_ ^= 1;
    }
    
    #pragma unroll (128/STAGE_ELEMS)
    for (uint stage = 0; stage < accumulators.size(); stage += STAGE_ELEMS) {
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
      if (is_producer_load_needed && stage/STAGE_ELEMS + 1 > StorePipeline::UnacquiredStages) {
        if (ReuseSmemC) {
          load_pipeline.consumer_release(load_pipe_consumer_state);
          ++load_pipe_consumer_state;
        }
      }
    }

    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

  template<
    class ProblemShapeMNKL,
    class TileShapeMNK,
    class TileCoordMNKL,
    class AccEngine, class AccLayout,
    class TiledMma
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
      TensorStorage& shared_tensors2,
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
    const uint STAGE_ELEMS = size<0>(EpilogueTile{}) * size<1>(EpilogueTile{});
    constexpr uint NumMMAThreads = size(TiledMma{});
    // The tma tensor D under im2col mode only has two modes (M, N) which
    // should be local tiled with only (m_coord, n_coord).
    auto coord_shape = conditional_return<is_im2col_D>( 
        make_coord(m_coord, n_coord),
        make_coord(m_coord, n_coord, l_coord));
    // Represent the full output tensor, slice to get the tile this CTA is responsible for
    PostsumOp first_store_srcs[4], second_store_srcs[4];
    bool use_tma_first_store = false;
    auto first_store_tuple = get_store_tma(problem_shape_mnkl, sub_m_idx, true, first_store_srcs, use_tma_first_store);
    PostsumOp first_store_dest = get<1>(first_store_tuple);
    if constexpr (Level2StoreM4 || Level2LoadsC2) {
      //always do TMA Add for Level2's M4 and Level2's M5
      use_tma_first_store = use_tma_first_store ||
        (l2_sched_tma_reduce_add && first_store_dest.is_layout_final());
    }

    bool store_level2_m4 = Level2StoreM4 && params.ptr_D2 != nullptr && first_store_dest.is_layout_final();
    auto& tma_store_dorm = get<2>(first_store_tuple) ? params.tma_store_postsum_m : params.tma_store_d ;
    auto& tma_add_dorm = get<2>(first_store_tuple) ? params.tma_add_postsum_m : params.tma_add_d;
    Tensor mD_mn = get<0>(first_store_tuple);
    Tensor mD = coalesce(mD_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD = local_tile(mD, take<0,2>(CtaTileMNK{}), coord_shape);                                  // (CTA_M,CTA_N)
    
    bool use_tma_second_store = false;
    auto second_store_tuple = get_store_tma(problem_shape_mnkl, sub_m_idx, false, second_store_srcs, use_tma_second_store);
    bool has_second_store = get<1>(second_store_tuple).valid();
    PostsumOp second_store_dest = get<1>(second_store_tuple);
    Tensor mD2_mn = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (Level2StoreM4) {
        if (store_level2_m4) {
          Tensor output_d2 = params.tma_store_d2.get_tma_tensor(make_shape(M, N, L));
          auto output_d2_ptr = output_d2.data() + make_coord((first_store_dest.get_op() % 2) * N / 2,
                                                           (first_store_dest.get_op() / 2) * M / 2, _);
          return make_tensor(output_d2_ptr, output_d2.layout());
        }
      }
      return get<0>(second_store_tuple);
    }();
    Tensor mD2 = coalesce(mD2_mn, take<0,2>(CtaTileMNK{}));
    Tensor gD2 = local_tile(mD2, take<0,2>(CtaTileMNK{}), coord_shape);

    bool has_second_load = get<1>(get_second_load_tma(problem_shape_mnkl)); //TODO: second load postsums should be passed

    // Apply epilogue subtiling
    Tensor gD_epi = flat_divide(gD, EpilogueTile{});                             // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor gD2_epi = flat_divide(gD2, EpilogueTile{});

    // Construct the corresponding pipelined smem tensors
    auto ptr_sC = shared_tensors.collective.smem_C.begin();
    auto ptr_sPostsum = shared_tensors.collective.smem_Postsum.begin();
    auto ptr_sPostsum2 = ptr_sPostsum + cosize_v<SmemLayoutC>;
    auto ptr_sD = shared_tensors.collective.smem_D.begin();
    auto ptr_sD2 = shared_tensors.collective.smem_D2.begin();

    Tensor sC_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC), SmemLayoutC{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    auto ptr_sC2 = shared_tensors.collective.smem_C2.begin();
    Tensor sC2_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sC2), SmemLayoutC{}));
    Tensor sPostsum_epi = cute::as_position_independent_swizzle_tensor(
              make_tensor(make_smem_ptr(ptr_sPostsum), SmemLayoutC{}));        // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sPostsum2_epi = cute::as_position_independent_swizzle_tensor(
              make_tensor(make_smem_ptr(ptr_sPostsum2), SmemLayoutC{}));       // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
    Tensor sD_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sD), SmemLayoutD{}));             // (EPI_TILE_M,EPI_TILE_N,PIPE_D)
    Tensor sD2_epi = cute::as_position_independent_swizzle_tensor(
                      make_tensor(make_smem_ptr(ptr_sD2), SmemLayoutD{}));

    TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);

    // (t)hread-partition for (r)egister to (r)egister copy (tRR_)
    TiledCopy tiled_r2r = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
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
    TiledCopy tiled_r2s = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
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
    Tensor tRS_rD = make_tensor<SmemElementD>(tRS_rD_layout);                                      // (R2S,R2S_M,R2S_N)

    Layout tRS_rD2_layout = make_layout(take<0,3>(shape(thread_r2s.partition_S(sD2_epi))));
    Tensor tRS_rD2 = make_tensor<SmemElementD>(tRS_rD2_layout);

    // Vectorized fragment view
    constexpr int FragmentSize = DispatchPolicy::FragmentSize;
    Tensor tRS_rAcc_frg = recast<Array<ElementAccumulator, FragmentSize>>(tRS_rAcc);
    Tensor tRS_rD_frg   = recast<Array<SmemElementD      , FragmentSize>>(tRS_rD);                                      // (R2S,R2S_M,R2S_N)
    Tensor tRS_rD2_frg   = recast<Array<SmemElementD      , FragmentSize>>(tRS_rD2);

    CUTE_STATIC_ASSERT(size<0>(tRS_rAcc) % FragmentSize == 0, "Fragment size does not vectorize properly");

    // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
    TiledCopy tiled_s2r = make_tiled_copy_S(Copy_Atom<CopyOpS2R, SmemElementC>{}, tiled_copy_C_atom);
    ThrCopy thread_s2r = tiled_s2r.get_slice(thread_idx);
    Tensor tSR_sC        = thread_s2r.partition_S(sC_epi);                                  // (S2R,S2R_M,S2R_N,PIPE_C)
    Tensor tSR_sC2       = thread_s2r.partition_S(sC2_epi);
    Layout tSR_rC_layout = thread_s2r.retile_D(tRS_rD).layout();                            // (S2R,S2R_M,S2R_N)

    Tensor tSR_sPostsum  = thread_s2r.partition_S(sPostsum_epi);                            // (S2R,S2R_M,S2R_N,PIPE_C)
    Tensor tSR_sPostsum2 = thread_s2r.partition_S(sPostsum2_epi);                           // (S2R,S2R_M,S2R_N,PIPE_C)

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
    Tensor tRS_rPostsum = make_tensor<RegisterElementC>(tRS_rD_layout);
    Tensor tSR_rPostsum = thread_s2r.retile_D(tRS_rPostsum);
    Tensor tRS_rPostsum2 = make_tensor<RegisterElementC>(tRS_rD2_layout);
    Tensor tSR_rPostsum2 = thread_s2r.retile_D(tRS_rPostsum2);

    // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
    ThrCopy thrblk_s2g = tma_store_dorm.get_slice(Int<0>{});
    Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)
    
    ThrCopy thrblk_s2g_d2 = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (Level2StoreM4) {
        if (store_level2_m4) {
          return params.tma_store_d2.get_slice(Int<0>{});
        }
      }
      return thrblk_s2g;
    }();
    Tensor bSG_sD2 = thrblk_s2g_d2.partition_S(sD2_epi);                                    // (S2G,S2G_M,S2G_N,PIPE_D)
    Tensor bSG_gD2 = thrblk_s2g_d2.partition_D(gD2_epi);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    Tensor mD_crd = make_identity_tensor(make_shape(M,N));                                                     // (M,N)
    Tensor cD_mn = local_tile(mD_crd, take<0,2>(CtaTileMNK{}), make_coord(m_coord, n_coord));          // (CTA_M,CTA_N)
    Tensor tRS_cD_mn = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
      if constexpr (IsUseR2R) {
        // (t)hread-partition for ConsumerStoreCallbacks. 
        TiledCopy tiled_cst = make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementC>{}, tiled_copy_C_atom);
        ThrCopy thread_cst = tiled_cst.get_slice(thread_idx);

        return thread_cst.partition_S(flat_divide(cD_mn, EpilogueTile{}));             // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
      }
      else {
        return thread_r2s.partition_S(flat_divide(cD_mn, EpilogueTile{}));             // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
      }
    }();
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

    // Get TiledCopy for partition reference when consumer store.
    TiledCopy tiled_copy_partition_ref = make_tiled_copy_S(Copy_Atom<CopyOpR2S,SmemElementD>{}, tiled_copy_C_atom);
    // Get the fusion callbacks for the consumer store warps
    constexpr bool RefSrc = true; // Register tensors reference tiled copy src layout
    auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs(
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
                    );
    auto cst_callbacks = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args);
    bool is_M_load_needed = first_store_srcs[0].valid() || second_store_srcs[0].valid();

    // if (StrassenMiGroup::hasM2() && sub_m_idx==0 && threadIdx.x%128==0 && blockIdx.x==0&&blockIdx.y == 0)
    //   printf("1396 %d : %d %d %d\n", sub_m_idx, is_producer_load_needed, first_store_srcs[0].valid(), second_store_srcs[0].valid());
    // StrassenMiGroup::hasM1() || StrassenMiGroup::hasM2() || StrassenMiGroup::hasM3() || StrassenMiGroup::hasM4() || StrassenMiGroup::hasM6();
        //TODO: Has to hard code for M2 & M3 with TMA Reduce
    //If uses tma and M0, M2, or M3, then do not is_C_load_needed should be false
    const bool mi_can_load_C = StrassenMiGroup::getMi(sub_m_idx) != 0 &&
                   StrassenMiGroup::getMi(sub_m_idx) != 2 && StrassenMiGroup::getMi(sub_m_idx) != 3;
    bool is_C_load_needed = is_source_supported && first_store_dest.is_layout_final() &&
                            fusion_callbacks.is_C_load_needed() && mi_can_load_C;
    bool is_producer_load_needed = is_M_load_needed || is_C_load_needed;
    if (StrassenMiGroup::hasM2() && m_coord == 0 && n_coord == 0 && thread_idx == 0)
      MY_PRINTF("788 %d %d %d\n", is_C_load_needed, first_store_dest.is_layout_final(), fusion_callbacks.is_C_load_needed());

    auto cst_args2 = cutlass::epilogue::fusion::detail::ConsumerStoreArgs(
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
                      tRS_rC2,
                      thread_idx
                    );
    auto cst_callbacks2 = fusion_callbacks.template get_consumer_store_callbacks<RefSrc>(cst_args2);

    using FragmentVisit = decltype(cst_callbacks.visit(tRS_rAcc_frg(0), 0, 0, 0));
    constexpr bool IsDirectR2S = cute::is_same_v<FragmentVisit, Array<SmemElementD, FragmentSize>>;
    using RegisterElementD = cute::conditional_t<!IsDirectR2S, ElementCompute, SmemElementD>;
    Tensor tRS_rCompute = make_tensor<RegisterElementD>(tRS_rD_layout);                            // (R2S,R2S_M,R2S_N)
    Tensor tRS_rCompute_frg = recast<Array<RegisterElementD, FragmentSize>>(tRS_rCompute);

    Tensor tRS_rCompute2 = make_tensor<RegisterElementD>(tRS_rD_layout);                            // (R2S,R2S_M,R2S_N)
    Tensor tRS_rCompute2_frg = recast<Array<RegisterElementD, FragmentSize>>(tRS_rCompute2);

    // Thread synchronizer for previously issued waits or fences
    // to ensure visibility of smem reads/writes to threads or TMA unit
    auto synchronize = [&] () CUTLASS_LAMBDA_FUNC_INLINE { cutlass::arch::NamedBarrier::sync(size(TiledMma{}), cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };

    // Predication for TMA store (one warp issues TMA store)
    bool issue_tma_store = (thread_idx / NumThreadsPerWarp) == 0;

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
    [[maybe_unused]] int epi_m_prev = 0;
    [[maybe_unused]] int epi_n_prev = 0;
    static_assert(not (DelayTmaStore and ReuseSmemC and StagesC <= StagesD), "This TMA epilogue configuration will deadlock");

    // The TMA store sequence for one subtile iteration
    auto tma_store_fn = [&] (int epi_m, int epi_n, int linear_store_stage) CUTLASS_LAMBDA_FUNC_INLINE {
      // Write the tile from smem to gmem with TMA
      cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA
      synchronize(); // ensure all threads have issued their async fence
      if constexpr (is_destination_supported) {
        if (issue_tma_store) {
          if (use_tma_first_store) {
            if (thread_idx == 0) {
              Tensor sD_tile = group_modes<0,2>(sD_epi(_,_,store_pipe_producer_state.index()));
              Tensor gD_tile = group_modes<0,2>(gD_epi(_,_,epi_m,epi_n));
              copy(tma_add_dorm,
                   sD_tile,
                   gD_tile);
            }
            // copy(tma_store_dorm, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
          } else {
            copy(tma_store_dorm, bSG_sD(_,_,_,store_pipe_producer_state.index()), bSG_gD(_,_,_,epi_m,epi_n));
          }

          if constexpr (Level2StoreM4) {
            if (store_level2_m4) {
              if (use_tma_first_store) {
                if (thread_idx == 0) {
                  Tensor sD2_tile = group_modes<0,2>(sD2_epi(_,_,store_pipe_producer_state.index()));
                  Tensor gD2_tile = group_modes<0,2>(gD2_epi(_,_,epi_m,epi_n));
                  copy(params.tma_add_d2, sD2_tile, gD2_tile);
                }
              }
              else {
                copy(params.tma_store_d2, bSG_sD2(_,_,_,store_pipe_producer_state.index()), bSG_gD2(_,_,_,epi_m,epi_n));
              }
            }
          }

          if (second_store_dest.valid() && second_store_dest.is_mem_global() && second_store_dest.is_layout_interim_linear() && thread_idx == 0) {
            //TODO: Can distribute writes over all threads a warp similar to layoutfinal copy
            ElementD* postsum_m0 = reinterpret_cast<ElementD*>(params.ptr_postsum_m) + second_store_dest.get_op()*(N/2)*(M/2);
            postsum_m0 = &postsum_m0[(m_coord*((N/2)/size<1>(TileShapeMNK{})) + n_coord)*size<0>(TileShapeMNK{})*size<1>(TileShapeMNK{})];
            ElementD* st_ptr = &postsum_m0[linear_store_stage * STAGE_ELEMS];
            auto smem = &ptr_sD2[store_pipe_producer_state.index() * STAGE_ELEMS];
            SM90_BULK_COPY_S2G().copy(smem, st_ptr, STAGE_ELEMS * sizeof(ElementD));
          } else if (second_store_dest.is_layout_final()) {
            // copy(params.tma_store_postsum_m, bSG_sD2(_,_,_,store_pipe_producer_state.index()), bSG_gD2(_,_,_,epi_m,epi_n));
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
        if (store_finished && is_producer_load_needed) {
          load_pipeline.consumer_release(load_pipe_consumer_state);
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

    // if (StrassenMiGroup::hasM1()) {
    //   // load_pipeline.consumer_wait(load_wait_state);
    //   cutlass::Array<ElementD, 8>* ptr_smem = (cutlass::Array<ElementD, 8>*)shared_tensors.collective.smem_C.begin();
    //   for (int i = 0; i < 128/8; i++) {
    //     auto frg = *(ptr_smem + thread_idx + i);
    //     tRS_rAcc_frg(i) = tRS_rAcc_frg(i) + cutlass::NumericArrayConverter<float, SmemElementD, FragmentSize>{}(frg);
    //   }
    //   cutlass::arch::fence_view_async_shared();
    //   // ++load_wait_state;
    //   // load_pipeline.consumer_release(load_pipe_consumer_state);
    //   // ++load_pipe_consumer_state;
    // }
    // if (StrassenMiGroup::hasM4() && thread_idx == 1 && m_coord == 0 && n_coord == 0)
    //   MY_PRINTF("1630 %f\n", float(accumulators[0]));
    // if (thread_idx == 0 && StrassenMiGroup::hasM5() && m_coord == 0 && n_coord == 0)
    //   MY_PRINTF("1432 %d %d ; %d %d ; %d\n", src_global_ops[0][0].valid(), src_global_ops[0][0].is_layout_interim_linear(),
    // src_global_ops[0][1].valid(), src_global_ops[0][1].is_layout_interim_linear(), is_C_load_needed);
    // For each output tile
    int linear_store_stage = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int epi_m = 0; epi_m < size<2>(gD_epi); ++epi_m) {
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gD_epi); ++epi_n,++linear_store_stage) {
        [[maybe_unused]] bool is_first_iteration = epi_m == 0 && epi_n == 0;
        bool is_last_iteration = epi_m == size<2>(gD_epi)-1 && epi_n == size<3>(gD_epi)-1;

        if (subtile_idx != -1 && (epi_n * static_cast<int>(size<2>(gD_epi)) + epi_m) != subtile_idx) {
          continue;
        }

        cst_callbacks.begin_loop(epi_m, epi_n);

        if (is_producer_load_needed) {
          // Wait for the producer load to fill smem
          load_pipeline.consumer_wait(load_wait_state);

          if (is_M_load_needed || is_C_load_needed) {
            // Copy source tile from smem to register
            const uint STAGE_ELEMS = (size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}));
            if (first_store_srcs[0].valid()) {
              if (first_store_srcs[0].is_layout_interim_linear()) {
                cutlass::Array<ElementD, 8>* ptr_smem = (cutlass::Array<ElementD, 8>*)((ElementD*)ptr_sPostsum + load_wait_state.index()*STAGE_ELEMS);
                for (int i = 0; i < size(tSR_rPostsum); i += 8) {
                  auto frg = ptr_smem[thread_idx + (i/8)*NumMMAThreads];
                  for (int j = 0; j < 8; j++)
                    tSR_rPostsum(i + j) = frg[j];
                }
              }
              else {
                copy(tiled_s2r, tSR_sPostsum(_,_,_,load_wait_state.index()), tSR_rPostsum);
              }
            }
            if (is_C_load_needed) {
              if constexpr (Level2LoadsC2) {
                Tensor source_fragment = make_tensor<SmemElementC>(tRS_rD_layout);
                Tensor source_fragment_s2r = thread_s2r.retile_D(source_fragment);
                copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), source_fragment_s2r);
                CUTLASS_PRAGMA_UNROLL
                for (int element = 0; element < size(tSR_rC); ++element) {
                  tSR_rC(element) = source_fragment_s2r(element);
                }
                if (params.ptr_C2 != nullptr) {
                  copy(tiled_s2r, tSR_sC2(_,_,_,load_wait_state.index()), source_fragment_s2r);
                  CUTLASS_PRAGMA_UNROLL
                  for (int element = 0; element < size(tSR_rC); ++element) {
                    tSR_rC(element) += source_fragment_s2r(element);
                  }
                }
              }
              else {
                copy(tiled_s2r, tSR_sC(_,_,_,load_wait_state.index()), tSR_rC);
              }
            }
            if (first_store_srcs[1].valid()) {
              if (first_store_srcs[1].is_layout_interim_linear()) {
                cutlass::Array<ElementD, 8>* ptr_src;
                if constexpr (is_source_supported) {
                  ptr_src = (cutlass::Array<ElementD, 8>*)((ElementD*)ptr_sPostsum2 + load_wait_state.index()*(STAGE_ELEMS));
                }
                else {
                  ElementD* postsum_src = reinterpret_cast<ElementD*>(params.ptr_postsum_m) +
                    first_store_srcs[1].get_op() * (M/2) * (N/2);
                  postsum_src += (m_coord * ((N/2) / size<1>(TileShapeMNK{})) + n_coord) *
                                 size<0>(TileShapeMNK{}) * size<1>(TileShapeMNK{});
                  postsum_src += linear_store_stage * STAGE_ELEMS;
                  ptr_src = reinterpret_cast<cutlass::Array<ElementD, 8>*>(postsum_src);
                }
                for (int i = 0; i < size(tSR_rPostsum2); i += 8) {
                  auto frg = ptr_src[thread_idx + (i/8)*NumMMAThreads];
                  for (int j = 0; j < 8; j++) {
                    tSR_rPostsum2(i + j) = frg[j];
                  }
                }
              } else {
                copy(tiled_s2r, tSR_sPostsum2(_,_,_,load_wait_state.index()), tSR_rPostsum2);
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
          assert(false);
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
          Tensor tRS_rAcc_frg_mn2 = tRS_rAcc_frg_mn;

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

          if constexpr (!Level2StoreM4) {
            CUTLASS_PRAGMA_UNROLL
            for (int epi_v = 0; epi_v < size(tRS_rCompute_frg); ++epi_v) {
              tRS_rCompute2_frg(epi_v) = cst_callbacks2.visit(tRS_rAcc_frg_mn(r2s_v + epi_v), epi_v, epi_m, epi_n);
            }
          }

          //TODO: Here addition with source C happens
          if (first_store_srcs[0].valid() and (first_store_srcs[0].is_layout_interim_linear() || first_store_srcs[0].is_layout_interim_matrix())) {
            cutlass::Array<ElementD, FragmentSize> frg;
            for (int i = 0; i < size(tSR_rPostsum); i++) {
              frg[i] = tSR_rPostsum(i);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tRS_rCompute_frg); ++i) {
              auto conv = cutlass::NumericArrayConverter<float, SmemElementD, FragmentSize>{}(frg);
              tRS_rAcc_frg_mn(r2s_v + i) = (StrassenMiGroup::hasM6() && !IsFusedM2M3M6) ?
                                              (conv - tRS_rAcc_frg_mn(r2s_v + i)) :
                                              (tRS_rAcc_frg_mn(r2s_v + i) + conv);
            }
          }

          if (first_store_srcs[1].valid() and (first_store_srcs[1].is_layout_interim_linear() || first_store_srcs[1].is_layout_interim_matrix())) {
            cutlass::Array<ElementD, FragmentSize> frg;
            for (int i = 0; i < size(tSR_rPostsum2); i++) {
              frg[i] = tSR_rPostsum2(i);
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
            if constexpr (Level2StoreM4) {
              if (store_level2_m4) {
                tRS_rCompute2_frg(epi_v) = cutlass::NumericArrayConverter<RegisterElementD, ElementAccumulator, FragmentSize>{}(
                  tRS_rAcc_frg_mn(r2s_v + epi_v));
              }
            }
            if (first_store_dest.is_layout_final()) {
              tRS_rCompute_frg(epi_v) = cst_callbacks.visit(tRS_rAcc_frg_mn(r2s_v + epi_v), epi_v, epi_m, epi_n);
            } else {
              tRS_rCompute_frg(epi_v) = cutlass::NumericArrayConverter<RegisterElementD, ElementAccumulator, FragmentSize>{}(
                tRS_rAcc_frg_mn(r2s_v + epi_v));
            }
          }
        }

        // The latest we can delay the TMA store is right before the smem store of the next iteration
        // since the current TMA store needs to be committed before we can acquire the next smem buffer
        if constexpr (DelayTmaStore) {
          // Issue TMA stores for the previous subtile
          if (not is_first_iteration and subtile_idx == -1) {
            tma_store_fn(epi_m_prev, epi_n_prev, linear_store_stage - 1);
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
          Tensor tRR_rD_src = thread_r2r.retile_S(tRS_rCompute);                             // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)
          Tensor tRR_rD_dst = thread_r2r.retile_D(tRS_rCompute);                             // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)

          // Output register transformation before copying to shared memory.
          copy(tiled_r2r, tRR_rD_src, tRR_rD_dst);
          if constexpr (Level2StoreM4) {
            if (store_level2_m4) {
              Tensor tRR_rD2_src = thread_r2r.retile_S(tRS_rCompute2);
              Tensor tRR_rD2_dst = thread_r2r.retile_D(tRS_rCompute2);
              copy(tiled_r2r, tRR_rD2_src, tRR_rD2_dst);
            }
          }
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tRS_rD_frg); ++i) {
          tRS_rD_frg(i) = cutlass::NumericArrayConverter<SmemElementD, RegisterElementD, FragmentSize>{}(tRS_rCompute_frg(i));
          if (!Level2StoreM4 || store_level2_m4) {
            tRS_rD2_frg(i) = cutlass::NumericArrayConverter<SmemElementD, RegisterElementD, FragmentSize>{}(tRS_rCompute2_frg(i));
          }
        }
        
        if (store_level2_m4 || (second_store_dest.valid() && second_store_dest.is_mem_global() && second_store_dest.is_layout_final())) {
          copy(tiled_r2s, tRS_rD2, tRS_sD2(_,_,_,store_pipe_producer_state.index()));
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
          tma_store_fn(epi_m, epi_n, linear_store_stage);
        }

        cst_callbacks.end_loop(epi_m, epi_n);

      } // for epi_m
    } // for epi_n

    if constexpr (DelayTmaStore) {
      // Issue TMA stores for the last subtile
      tma_store_fn(epi_m_prev, epi_n_prev, linear_store_stage - 1);
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
      const bool sync_load_pipe,
      uint sub_m_idx) {
    // wait for all TMA stores to complete
    store_pipeline.producer_tail(store_pipe_producer_state);
    // reset store counter
    issued_stores = 0;

    if constexpr (ReuseSmemC) {
      if (sync_load_pipe) {
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

private:
  Params const& params;
  FusionCallbacks fusion_callbacks;
  int issued_stores = 0;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
