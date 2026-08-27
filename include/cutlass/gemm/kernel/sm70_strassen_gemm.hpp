/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/strassen_gemm_universal_decl.h"

#include "cute/tensor.hpp"

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class StrassenMiGroup_,
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class StrassenGemmUniversal<
  StrassenMiGroup_,
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //  
  using StrassenMiGroup = StrassenMiGroup_;
  using Mma = CollectiveMainloop_;

  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  static constexpr bool IsGdcEnabled = false;

  static constexpr bool is_valid_tile_scheduler =
  cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>;
static_assert(is_valid_tile_scheduler, "SM70 kernel does not support specializing the tile scheduler.");

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = static_cast<int>(cute::max(
      sizeof(typename CollectiveMainloop::SharedStorage),
      sizeof(typename CollectiveEpilogue::SharedStorage)));

  struct SharedStorage {
    union {
      typename CollectiveMainloop::SharedStorage mainloop;
      typename CollectiveEpilogue::SharedStorage epilogue;
    };
  };

  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(cute::size(TiledMma{}));
  static constexpr uint32_t kThreadCount = MaxThreadsPerBlock;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};

    Arguments(GemmUniversalMode mode, ProblemShape problem_shape,
              MainloopArguments mainloop, EpilogueArguments epilogue,
              KernelHardwareInfo hw_info, TileSchedulerArguments scheduler = TileSchedulerArguments()) :
      mode(mode), problem_shape(problem_shape), mainloop(mainloop), epilogue(epilogue),
      hw_info(hw_info), scheduler(scheduler) 
    {}

    template<typename Other>
    Arguments(const Other& other) :
      mode(other.mode), problem_shape(other.problem_shape), mainloop(other.mainloop),
      epilogue(other.epilogue), hw_info(other.hw_info), scheduler(other.scheduler)
      {}
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};

    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementD* ptr_D;
    ElementA* presum_m_a_workspace;
    ElementB* presum_m_b_workspace;
    ElementD* postsum_m_workspace;

    int run = 0;

    CUTLASS_HOST_DEVICE
    int get_problem_shape_m(int idx = 0) const {
      return get<0>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_n(int idx = 0) const {
      return get<1>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_k(int idx = 0) const {
      return get<2>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_A(int idx = 0) const {
      return get_problem_shape_k(idx);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_B(int idx = 0) const {
      return get_problem_shape_n(idx);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MA(int idx = 0) const {
      return get_problem_shape_k(idx)/2;
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MB(int idx = 0) const {
      return get_problem_shape_n(idx)/2;
    }

    CUTLASS_HOST_DEVICE
    int get_presum_log_tile_multiplier_a() const {
      return mainloop.get_presum_tile_log_multiplier_a();
    }
  
    CUTLASS_HOST_DEVICE
    int get_presum_log_tile_multiplier_b() const {
      return mainloop.get_presum_tile_log_multiplier_b();
    }

    CUTLASS_HOST_DEVICE
    ProblemShape get_half_problem_shape(int idx = 0) const {
      return ProblemShape{get_problem_shape_m(idx)/2, get_problem_shape_n(idx)/2,
                          get_problem_shape_k(idx)/2};
    }

    CUTLASS_HOST_DEVICE
    ElementA* get_ptr_A(int idx = 0) const {
      return ptr_A;
    }

    CUTLASS_HOST_DEVICE
    ElementB* get_ptr_B(int idx = 0) const {
      return ptr_B;
    }

    CUTLASS_HOST_DEVICE
    ElementD* get_ptr_D(int idx = 0) const {
      return ptr_D;
    }

    CUTLASS_HOST_DEVICE
    ElementA* get_ptr_presum_A(int problem_idx) const {
      return presum_m_a_workspace;
    }

    CUTLASS_HOST_DEVICE
    ElementB* get_ptr_presum_B(int problem_idx) const {
      return presum_m_b_workspace;
    }

    CUTLASS_HOST_DEVICE
    ElementD* get_postsum_ptr(int problem_idx) const {
      return postsum_m_workspace;
    }
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, ElementA* presum_m_a, ElementB* presum_m_b, ElementD* postsum_m, void* workspace) {
    (void) workspace;

    KernelHardwareInfo hw_info{args.hw_info.device_id, args.hw_info.sm_count};
    auto problem_shape_MNKL = append<4>(args.problem_shape, Int<1>{});

    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, presum_m_a, presum_m_b, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, postsum_m, args.epilogue, workspace),
      const_cast<ElementA*>(args.mainloop.ptr_A),
      const_cast<ElementB*>(args.mainloop.ptr_B),
      const_cast<ElementD*>(args.epilogue.ptr_D),
      presum_m_a, presum_m_b, postsum_m,
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool mode_implementable = args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    return mode_implementable && TileScheduler::can_implement(args.scheduler);
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    return workspace_size;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr, 
    CudaHostAdapter* cuda_adapter = nullptr) {
    cutlass::Status status = Status::kSuccess;

    return status;
  }

  static dim3
  get_grid_shape(Params const& params) {
    int batch_count = 1;
    if constexpr (cute::rank(ProblemShape{}) == 4) {
      batch_count = cute::size<3>(params.problem_shape);
    }

    return dim3(
      cute::size(cute::ceil_div(cute::shape<0>(params.problem_shape)/2, cute::shape<0>(TileShape{}))),
      cute::size(cute::ceil_div(cute::shape<1>(params.problem_shape)/2, cute::shape<1>(TileShape{}))),
      batch_count
    );
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf, char* accums_store = nullptr, dim3 base_block = {0,0,0}) {
    operator()(params, *(SharedStorage*)smem_buf, accums_store, base_block);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage, char* accums_store = nullptr, dim3 base_block = {0,0,0}) {
    char* smem_buf = reinterpret_cast<char*>(&shared_storage);
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto [M,N,K,L] = problem_shape_MNKL;

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    auto [m_coord, n_coord, l_coord] = static_cast<uint3>(blockIdx);
    auto blk_coord_mnkl = make_coord(int(m_coord), int(n_coord), _, int(l_coord));                         // (m,n,k,l)

    CollectiveMainloop collective_mma;

    auto all_inputs = collective_mma.load_init(problem_shape_MNKL, params.mainloop);
    auto load_inputs = collective_mma.get_inputs(all_inputs, 0);

    Tensor gA = get<0>(load_inputs)(_, _, get<0>(blk_coord_mnkl), _, get<3>(blk_coord_mnkl)); // (BLK_M,BLK_K,k)
    Tensor gB = get<1>(load_inputs)(_, _, get<1>(blk_coord_mnkl), _, get<3>(blk_coord_mnkl)); // (BLK_N,BLK_K,k)

    // Compute tile residues for predication
    auto m_max_coord = M / 2 - size<0>(gA) * get<0>(blk_coord_mnkl);                         // M/2 - BLK_M * m_coord
    auto n_max_coord = N / 2 - size<0>(gB) * get<1>(blk_coord_mnkl);                         // N/2 - BLK_N * n_coord
    auto k_residue   = K / 2 - size<2>(TileShape{}) * cute::ceil_div(K / 2, size<2>(TileShape{}));
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
    clear(accumulators);

    using RWMTypes = typename StrassenMiGroup::RWMTypes;

    {
      int myRW = RWMTypes::MyVal;
      int sign = MmaStrassen::SIGN(myRW);
      myRW = MmaStrassen::ABS(myRW);

      if (myRW > 0) {
        switch(myRW) {
          case MmaStrassen::ContinueAccums: {
            typename Mma::FragmentC& accumStore = reinterpret_cast<typename Mma::FragmentC&>(*accums_store);
            accumulators = accumStore;
            break;
          }
        }
      }
    }

    auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
    int  k_tile_count = cute::ceil_div(K / 2, size<2>(TileShape{}));

    bool is_neg = false;
    bool any_global_dst_matrix = false;
    bool any_global_dst_valid = false;

    const uint sub_m_idx = 0;
    for (int c = 0; c < 4; c++) {
      using RWCTypes = typename StrassenMiGroup::RWCTypes;

      const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(c);
      const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDestByOutputIndex(c);
      uint mi = StrassenMiGroup::getMi(sub_m_idx);

      int misign = RWCTypes::MiSignByOutputIndex(c, mi);

      if (misign == 0 || (!postsum_global_dest.valid())) continue;

      is_neg = misign == -1;

      any_global_dst_matrix = any_global_dst_matrix || postsum_global_dest.is_layout_final() ||
                                                      postsum_global_dest.is_layout_interim_matrix();
      any_global_dst_valid = any_global_dst_valid || postsum_global_dest.valid();
    }

    // Perform the collective scoped MMA
    collective_mma(
      params.mainloop,
      blk_coord_mnkl,
      problem_shape_MNKL,
      accumulators,
      gA,
      gB,
      accumulators,
      k_tile_iter, k_tile_count,
      residue_mnk,
      thread_idx,
      smem_buf
    );

    if (is_neg)
      for (int i = 0; i < accumulators.size(); i++)
        accumulators[i] = -1 * accumulators[i];

    // Epilogue and write to gD
    CollectiveEpilogue epilogue{params.epilogue};
    if (any_global_dst_valid) {
      if (any_global_dst_matrix) {
        epilogue.store(
          problem_shape_MNKL,
          blk_shape,
          blk_coord_mnkl,
          accumulators,
          tiled_mma,
          residue_mnk,
          thread_idx,
          smem_buf
        );
      } else {
        epilogue.store_m0(
          problem_shape_MNKL,
          blk_shape,
          blk_coord_mnkl,
          accumulators,
          tiled_mma,
          residue_mnk,
          thread_idx,
          smem_buf
        );
      }
    }

    {
      int myRW = RWMTypes::MyVal;
      int sign = MmaStrassen::SIGN(myRW);
      myRW = MmaStrassen::ABS(myRW);

      if (myRW > 0) {
        switch(myRW) {
          case MmaStrassen::KeepAccums:{
            typename Mma::FragmentC& accumStore = reinterpret_cast<typename Mma::FragmentC&>(*accums_store);
            accumStore = accumulators;
            break;
          }
          case MmaStrassen::ContinueAccums:
          {
            break;
          }
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
