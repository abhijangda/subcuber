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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/strassen_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/arch/grid_dependency_control.h"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

template <
  typename StrassenMiGroup_,
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileSchedulerTag_
>
class StrassenGemmUniversal<
  StrassenMiGroup_,
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  using StrassenMiGroup = StrassenMiGroup_;
  using Mma = CollectiveMainloop_;

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
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr uint32_t TileSchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  using TileSchedulerTag = TileSchedulerTag_;

  using TileScheduler = typename detail::StrassenTileSchedulerSelector<
                                          TileSchedulerTag, 
                                          ArchTag, 
                                          TileShape,
                                          ClusterShape,
                                          TileSchedulerPipelineStageCount
                                          >::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  
  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;      // 1 warp       
  static constexpr uint32_t NumMMAThreads          = size(TiledMma{});       // 8 warps
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;      // 1 warp for C

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = NumMMAThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t kThreadCount = MaxThreadsPerBlock;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumFixupBarriers = NumMmaWarpGroups;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;
  static constexpr bool     IsMainloopAuxiliaryLoadNeeded = detail::HasAuxiliaryLoad_v<typename CollectiveMainloop::DispatchPolicy>;
  static constexpr uint32_t ConsumerSubMIterations = StrassenMiGroup::numMs();

  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    size<0>(TileShape{}) * size<1>(TileShape{}) / NumMMAThreads *
    sizeof(ElementAccumulator) / sizeof(uint32_t);
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;
  using StoreWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;
  static constexpr bool UseM0M1StoreOrderBarrier = CollectiveMainloop::PresumStages == 4 &&
                                                   ((StrassenMiGroup::hasM0() && size<0>(typename CollectiveMainloop::PresumTileShapeA{}) > 2) ||
                                                    (StrassenMiGroup::hasM1() && size<0>(typename CollectiveMainloop::PresumTileShapeB{}) > 2));

  using TileSchedulerPipeline = typename TileScheduler::Pipeline;
  using TileSchedulerPipelineState = typename TileSchedulerPipeline::PipelineState;
  using TileSchedulerStorage = typename TileScheduler::SharedStorage;
  using TileSchedulerThrottlePipeline = typename TileScheduler::ThrottlePipeline;
  using TileSchedulerThrottlePipelineState = typename TileSchedulerThrottlePipeline::PipelineState;
  
  static const bool DoesPresum = (StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
                                 (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum());
  using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
  using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

  struct TensorStorage1 : cute::aligned_struct<128, _1> {
    MainloopTensorStorage mainloop;
    typename CollectiveMainloop::PresumTensorStorage presum_tensors;
    EpilogueTensorStorage epilogue;
  };

  struct TensorStorage2 : cute::aligned_struct<128, _1> {
    MainloopTensorStorage mainloop;
    static constexpr size_t ReusedEpilogueStorageOffset = 16384;

    union {
      struct {
        typename CollectiveMainloop::PresumTensorStorage presum_tensors;
      };
      struct {
        char padding[ReusedEpilogueStorageOffset];
        EpilogueTensorStorage epilogue;
      };
    };
  };
  using TensorStorage = cute::conditional_t<DoesPresum,
                                          cute::conditional_t<UseM0M1StoreOrderBarrier,
                                                              TensorStorage2,
                                                              TensorStorage1>,
                                          TensorStorage1>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
      alignas(16) typename StoreWarpOrderBarrier::SharedStorage store_order;
    } pipelines;

    alignas(16) TileSchedulerStorage scheduler;
    TensorStorage tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

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
    ProblemShape get_half_problem_shape(int idx = 0) const {
      return ProblemShape{get_problem_shape_m(idx)/2, get_problem_shape_n(idx)/2,
                          get_problem_shape_k(idx)/2};
    }

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
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementD* ptr_D;
    ElementA* presum_m_a_workspace;
    ElementB* presum_m_b_workspace;
    ElementD* postsum_m_workspace;

    void* workspace{nullptr};

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
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto half_problem_shape_MNKL = append<4>(args.get_half_problem_shape(), 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
      half_problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles
      );

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, presum_m_a, presum_m_b, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, const_cast<ElementA*>(args.mainloop.ptr_A), postsum_m, epilogue_workspace),
      hw_info,
      scheduler,
      const_cast<ElementA*>(args.mainloop.ptr_A),
      const_cast<ElementB*>(args.mainloop.ptr_B),
      const_cast<ElementD*>(args.epilogue.ptr_D),
      presum_m_a, presum_m_b, postsum_m,
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);
    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.get_half_problem_shape(), TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf, char* in_accums = nullptr, dim3 base_block = {0,0,0}) {
    operator()(params, *reinterpret_cast<SharedStorage*>(smem_buf), in_accums, base_block);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, SharedStorage& shared_storage, char* in_accums = nullptr, dim3 base_block = {0,0,0}) {
    using namespace cute;
    using X = Underscore;
    (void) in_accums;
    (void) base_block;

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif

// Any Tensor Op MMA Atom in the ISA is arch conditional.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    printf("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(NumMMAThreads == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");

    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      MainloopAux = 3
    };



    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    // TileScheduler pipeline
    typename TileSchedulerPipeline::Params scheduler_pipeline_params;
    typename TileSchedulerThrottlePipeline::Params scheduler_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) { 
      if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::ProducerConsumer;
      }
      else {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Consumer;
      }
      scheduler_pipeline_params.producer_blockid = 0;
      scheduler_pipeline_params.producer_arv_count = 1;
      scheduler_pipeline_params.consumer_arv_count = NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads;

      if (is_epi_load_needed) {
        scheduler_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      } 
      scheduler_pipeline_params.transaction_bytes = sizeof(typename TileScheduler::CLCResponse);
      
      scheduler_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
      scheduler_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      scheduler_throttle_pipeline_params.dst_blockid = 0;
      scheduler_throttle_pipeline_params.initializing_warp = 3;
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Consumer;
      }
      // set role when it is for DMA warp in Mainloop
      else if (warp_group_role == WarpGroupRole::Producer &&
               producer_warp_role == ProducerWarpRole::Mainloop) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Producer;
      }
    }
    TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(), scheduler_pipeline_params);
    TileSchedulerPipelineState scheduler_pipe_consumer_state;

    TileSchedulerThrottlePipeline scheduler_throttle_pipeline(shared_storage.scheduler.throttle_pipeline(), scheduler_throttle_pipeline_params);
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state = cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && (producer_warp_role == ProducerWarpRole::Mainloop || 
        producer_warp_role == ProducerWarpRole::MainloopAux)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumMMAThreads;
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    } 
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumMMAThreads;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = (warp_group_role == WarpGroupRole::Consumer0 ||
                                          warp_group_role == WarpGroupRole::Consumer1) ? 0 : 1;
    params_load_order_barrier.group_size = NumMMAThreads;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    typename StoreWarpOrderBarrier::Params params_store_order_barrier;
    params_store_order_barrier.group_id = (warp_group_role == WarpGroupRole::Consumer0 ||
                                           warp_group_role == WarpGroupRole::Consumer1) ? 0 : 1;
    params_store_order_barrier.group_size = NumMMAThreads;
    StoreWarpOrderBarrier store_order_barrier(shared_storage.pipelines.store_order, params_store_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    using RWCTypes = typename StrassenMiGroup::RWCTypes;


    auto cluster_wait_fn = [] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto half_problem_shape_MNKL = append<4>(params.get_half_problem_shape(), Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    TileScheduler scheduler{params.scheduler};
    if constexpr (IsSchedDynamicPersistent) {
      scheduler.set_data_ptr(shared_storage.scheduler.data());
    }
    // Declare work_tile_info, then define it in each of warps that use it.
    typename TileScheduler::WorkTileInfo work_tile_info;

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto all_inputs = collective_mainloop.load_init(problem_shape_MNKL, half_problem_shape_MNKL, params.mainloop);
    auto all_presumld_inputs = collective_mainloop.presumld_inputs(problem_shape_MNKL, half_problem_shape_MNKL, params.mainloop);
    auto load_inputs = collective_mainloop.get_inputs(all_inputs);
    auto load_inputs2 = collective_mainloop.get_inputs(all_inputs, 1);
    auto load_inputs3 = collective_mainloop.get_inputs(all_inputs, 2);

    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    Tensor gA2_mkl = get<0>(load_inputs2);
    Tensor gB2_nkl = get<1>(load_inputs2);

    constexpr bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();

    auto k_tile_count = (params.get_problem_shape_k()/2)/decltype(size<2>(blk_shape))::value;

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      // Scheduler Producer Warp
      if (producer_warp_role == ProducerWarpRole::Warp1) {
        if constexpr (IsSchedDynamicPersistent) { 
          bool requires_clc_query = true;
          TileSchedulerPipelineState scheduler_pipe_producer_state = cutlass::make_producer_start_state<TileSchedulerPipeline>();

          cutlass::arch::wait_on_dependent_grids();
          while (work_tile_info.is_valid()) {

            if (requires_clc_query) {
              // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
              scheduler_throttle_pipeline.consumer_wait(scheduler_pipe_throttle_consumer_state);
              scheduler_throttle_pipeline.consumer_release(scheduler_pipe_throttle_consumer_state);
              ++scheduler_pipe_throttle_consumer_state;

              // Query next work tile
              scheduler_pipe_producer_state = scheduler.advance_to_next_work(scheduler_pipeline, scheduler_pipe_producer_state);
            }

            // Fetch next work tile
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
              work_tile_info,
              scheduler_pipeline,
              scheduler_pipe_consumer_state
            );
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }

            work_tile_info = next_work_tile_info;
          }
          scheduler_pipeline.producer_tail(scheduler_pipe_producer_state);
        } 
      } // Scheduler Producer Warp End  
      else

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        bool requires_clc_query = true;
        int sub_m_idx = 0;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;   
            continue;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
          auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
          sub_m_idx = 0;

          if (requires_clc_query) {
            scheduler_throttle_pipeline.producer_acquire(scheduler_pipe_throttle_producer_state);
            scheduler_throttle_pipeline.producer_commit(scheduler_pipe_throttle_producer_state);
            ++scheduler_pipe_throttle_producer_state;
          }

          if (StrassenMiGroup::numMs() > 1) {
            auto load_work_tile = [&] (decltype(work_tile_info) load_work_tile_info, int load_sub_m_idx) {
              auto load_m_coord = idx2crd(load_work_tile_info.M_idx, shape<2>(gA_mkl));
              auto load_n_coord = idx2crd(load_work_tile_info.N_idx, shape<2>(gB_nkl));
              auto load_l_coord = idx2crd(load_work_tile_info.L_idx, shape<4>(gB_nkl));
              auto load_blk_coord = make_coord(load_m_coord, load_n_coord, _, load_l_coord);
              auto load_k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
              load_k_tile_iter.coord += (is_fused && load_sub_m_idx == 1) ? k_tile_count : 0;

              collective_mainloop.load(
                params.mainloop, half_problem_shape_MNKL,
                mainloop_pipeline,
                mainloop_pipe_producer_state,
                (is_fused || load_sub_m_idx == 0) ? load_inputs :
                  ((load_sub_m_idx == 1) ? load_inputs2 : load_inputs3),
                load_blk_coord, load_sub_m_idx,
                load_k_tile_iter, k_tile_count,
                lane_idx,
                block_rank_in_cluster,
                shared_storage.tensors.mainloop,
                all_presumld_inputs,
                shared_storage.tensors.presum_tensors,
                (UseM0M1StoreOrderBarrier) ? &store_order_barrier : (StoreWarpOrderBarrier*)nullptr
              );
              mainloop_pipe_producer_state.advance(k_tile_count);
            };

            load_work_tile(work_tile_info, 0);
            load_work_tile(work_tile_info, 1);
            if (StrassenMiGroup::numMs() == 3) {
              load_work_tile(work_tile_info, 2);
            }
          } else {
            collective_mainloop.load(
              params.mainloop, half_problem_shape_MNKL,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              blk_coord, sub_m_idx,
              k_tile_iter, k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop,
              all_presumld_inputs,
              shared_storage.tensors.presum_tensors,
              (StoreWarpOrderBarrier*)nullptr
            );
            mainloop_pipe_producer_state.advance(k_tile_count);
          }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info,
                                                                            scheduler_pipeline,             
                                                                            scheduler_pipe_consumer_state
                                                                           );

          work_tile_info = next_work_tile_info;
          if constexpr (IsSchedDynamicPersistent) { 
            requires_clc_query = increment_pipe; 
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
        } // Scheduler work fetch loop
        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

      }
      else if (producer_warp_role == ProducerWarpRole::MainloopAux) {
        if constexpr (IsMainloopAuxiliaryLoadNeeded) {
          while (work_tile_info.is_valid()) {
            if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
              auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
              work_tile_info = next_work_tile_info;
              continue;
            }

            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
            auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
            auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
            auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

            collective_mainloop.load_auxiliary(
              params.mainloop,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              blk_coord,
              k_tile_iter, work_k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop
            );
            // Update starting pipeline state for the next tile
            mainloop_pipe_producer_state.advance(work_k_tile_count);

            // Get next work tile
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
              work_tile_info,
              scheduler_pipeline,
              scheduler_pipe_consumer_state
            );

            work_tile_info = next_work_tile_info;
          } // Scheduler work fetch loop

        }
      }

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue &&
               collective_epilogue.is_producer_load_needed()) {

        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();

        CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

        while (work_tile_info.is_valid()) {
          #pragma unroll (StrassenMiGroup::numMs())
          for (int fused_mi = 0; fused_mi < StrassenMiGroup::numMs(); fused_mi++) {
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            #pragma unroll 4
            for (int c = 0; c < 4; c++) {
              const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(c);
              const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDestByOutputIndex(c);

              uint mi = StrassenMiGroup::getMi(fused_mi);
              int misign = RWCTypes::MiSignByOutputIndex(c, mi);

              if (misign == 0 || (!postsum_shared_dest.valid() && !postsum_global_dest.valid())) continue;

              MmaStrassen::PostsumOp postsum_srcs[4] = {MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp()};
              int postsum_src_len = 0;
              #pragma unroll 4
              for (int read_c = 0; read_c < 4; read_c++) {
                auto postsum_src = RWCTypes::PostsumSrcByOutputIndex(c, read_c);
                if (postsum_src.valid() && postsum_src.is_mem_global() && postsum_src.is_layout_interim()) {
                  postsum_srcs[postsum_src_len++] = postsum_src;
                }
              }

              if (postsum_src_len > 0) {
                if (lane_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                  MY_PRINTF("944 %d : %d %d\n", fused_mi, m_coord, n_coord);
                load_order_barrier.wait();
                load_order_barrier.advance();
                if (lane_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                  MY_PRINTF("948 %d : %d %d\n", fused_mi, m_coord, n_coord);
                epi_load_pipe_producer_state =
                collective_epilogue.load_m0(
                  epi_load_pipeline,
                  epi_load_pipe_producer_state,
                  problem_shape_MNKL,
                  blk_shape,
                  blk_coord, fused_mi,
                  tiled_mma,
                  lane_idx,
                  shared_storage.tensors.epilogue,
                  shared_storage.tensors.epilogue,
                  postsum_srcs
                );
                if (lane_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                  MY_PRINTF("963 %d : %d %d\n", fused_mi, m_coord, n_coord);
              }
            }
          }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info,
                                                                            scheduler_pipeline,     
                                                                            scheduler_pipe_consumer_state
                                                                           );
          work_tile_info = next_work_tile_info;
          if constexpr (IsSchedDynamicPersistent) { 
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
      if constexpr (UseM0M1StoreOrderBarrier) store_order_barrier.arrive();
      // if constexpr (UseM0M1StoreOrderBarrier) {asm volatile("bar.cta.sync %0, %1;" : : "r"(7), "r"(NumMMAThreads + 32));}
      CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);
      const int mma_thread_idx = (threadIdx.x - 128) % NumMMAThreads;

      while (work_tile_info.is_valid()) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                 // (MMA,MMA_M,MMA_N)

        #pragma unroll
        for (int consumer_sub_m_iter = 0; consumer_sub_m_iter < ConsumerSubMIterations; ++consumer_sub_m_iter) {
        auto sub_m_idx = consumer_sub_m_iter;
 
        if (sub_m_idx == 0 ||
            (StrassenMiGroup::numMs() >= 2 && sub_m_idx == 1) ||
            (StrassenMiGroup::numMs() >= 3 && sub_m_idx == 2)) {}
        else CUTE_GCC_UNREACHABLE;

        bool is_neg = false;
        bool has_global_src = false;
        bool any_global_dst_final = false;
        bool any_global_dst_valid = false;

        for (int c = 0; c < 4; c++) {
          const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(c);
          const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDestByOutputIndex(c);
          uint mi = StrassenMiGroup::getMi(sub_m_idx);
          int misign = RWCTypes::MiSignByOutputIndex(c, mi);

          if (misign == 0 || (!postsum_global_dest.valid())) continue;

          is_neg = misign == -1;

          any_global_dst_final = any_global_dst_final || postsum_global_dest.is_layout_final();
          any_global_dst_valid = any_global_dst_valid || postsum_global_dest.valid();

          #pragma unroll 4
          for (int read_c = 0; read_c < 4; read_c++) {
            auto postsum_src = RWCTypes::PostsumSrcByOutputIndex(c, read_c);
            if (postsum_src.valid() && postsum_src.is_mem_global()) {
              has_global_src = true;
            }
          }
        }

        if (is_neg)
          for (int i = 0; i < accumulators.size(); i++)
              accumulators[i] = -1 * accumulators[i];

        if (has_global_src) {
          load_order_barrier.arrive();
        }

        if (mma_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1044 %d : %d %d\n", sub_m_idx, m_coord, n_coord);
        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          collective_mainloop.mma(
            blk_coord, sub_m_idx, problem_shape_MNKL, half_problem_shape_MNKL,
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            shared_storage.tensors.presum_tensors,
            params.mainloop
          );

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            k_tile_count
          );

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(k_tile_count);
        }

        #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
        if (scheduler.is_last_tile(work_tile_info)) {
          // Hint on an early release of global memory resources.
          // The timing of calling this function only influences performance,
          // not functional correctness.
          cutlass::arch::launch_dependent_grids();

        }
        #endif

        // Index of warp group within consumer warp groups
        int consumer_warp_group_idx = canonical_warp_group_idx() - NumLoadWarpGroups;

        // Perform reduction across splits, if needed
        // TileScheduler::fixup(
        //   params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

        if (is_neg)
          for (int i = 0; i < accumulators.size(); i++)
            accumulators[i] = -1 * accumulators[i];

        decltype(epi_load_pipe_consumer_state) epi_load_pipe_consumer_state_next = epi_load_pipe_consumer_state;
        decltype(epi_store_pipe_producer_state) epi_store_pipe_producer_state_next = epi_store_pipe_producer_state;

        if (mma_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1094 %d:  %d %d\n", sub_m_idx, m_coord, n_coord);

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler) && any_global_dst_valid) {
          // Epilogue and write to gD
          if (!any_global_dst_final) {
            auto ret = collective_epilogue.store_m2(
              epi_load_pipeline,
              epi_load_pipe_consumer_state,
              epi_store_pipeline,
              epi_store_pipe_producer_state,
              problem_shape_MNKL, sub_m_idx,
              blk_shape,
              blk_coord,
              accumulators,
              tiled_mma,
              mma_thread_idx,
              shared_storage.tensors.epilogue,
              shared_storage.tensors.epilogue
            );
            epi_load_pipe_consumer_state_next = get<0>(ret);
            epi_store_pipe_producer_state_next = get<1>(ret);
          } else {
            auto ret =
            collective_epilogue.store(
              epi_load_pipeline,
              epi_load_pipe_consumer_state,
              epi_store_pipeline,
              epi_store_pipe_producer_state,
              problem_shape_MNKL,
              blk_shape,
              blk_coord, sub_m_idx,
              accumulators,
              tiled_mma,
              mma_thread_idx,
              shared_storage.tensors.epilogue,
              shared_storage.tensors.epilogue,
              work_tile_info.reduction_subtile_idx()
            );
            epi_load_pipe_consumer_state_next = get<0>(ret);
            epi_store_pipe_producer_state_next = get<1>(ret);
          }
        }

        auto ret =
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state_next,
          epi_store_pipeline,
          epi_store_pipe_producer_state_next,
          has_global_src,
          sub_m_idx
        );

        if (mma_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1148 %d: %d %d ; %d\n", sub_m_idx, m_coord, n_coord, has_global_src);

        epi_load_pipe_consumer_state = get<0>(ret);
        epi_store_pipe_producer_state = get<1>(ret);

        if constexpr (UseM0M1StoreOrderBarrier) store_order_barrier.arrive();
        }

        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info,
                                                                          scheduler_pipeline,
                                                                          scheduler_pipe_consumer_state
                                                                          );
        work_tile_info = next_work_tile_info;
        if constexpr (IsSchedDynamicPersistent) { 
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
          }
        }
      } // Scheduler work fetch loop
      // if constexpr (UseM0M1StoreOrderBarrier) {asm volatile("bar.cta.sync %0, %1;" : : "r"(7), "r"(NumMMAThreads + 32));}
    } // Consumer Warp Groups End
#endif
  }

};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
