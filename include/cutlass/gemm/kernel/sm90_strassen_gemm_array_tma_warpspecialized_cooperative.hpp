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
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/strassen_group_array_problem_shape.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  typename StrassenMiGroup_,
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
  cute::enable_if_t<cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, typename CollectiveMainloop_::DispatchPolicy::Schedule>>
>
{
  // Get the type of the scheduler response.
  template<typename TileScheduler, typename = void>
  struct TileSchedulerResponseGetter {
    using Type = typename TileScheduler::CLCResponse;
  };

  template<typename TileScheduler>
  struct TileSchedulerResponseGetter<TileScheduler, void_t<typename TileScheduler::SchedulerResponse>> {
    using Type = typename TileScheduler::SchedulerResponse;
  };

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(typename ProblemShape::UnderlyingProblemShape{}) == 3 or rank(typename ProblemShape::UnderlyingProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  static_assert(cute::is_base_of_v<KernelPtrArrayTmaWarpSpecializedCooperative, typename CollectiveMainloop_::DispatchPolicy::Schedule>);

  static constexpr bool IsGdcEnabled = false;

  using StrassenMiGroup = StrassenMiGroup_;
  using Mma = CollectiveMainloop_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using InternalStrideA = typename CollectiveMainloop::InternalStrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using InternalStrideB = typename CollectiveMainloop::InternalStrideB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using Schedule = typename DispatchPolicy::Schedule;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using InternalStrideC = typename CollectiveEpilogue::InternalStrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using InternalStrideD = typename CollectiveEpilogue::InternalStrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;
  static constexpr uint32_t MinTensorMapWorkspaceAlignment = 64;

  static_assert(
    cute::is_void_v<TileScheduler_>
    or (
      IsGroupedGemmKernel
      and cute::is_any_of_v<TileScheduler_, GroupScheduler>
    ),
    "Ptr-Array Cooperative and Grouped Gemm Cooperative kernel only supports the default scheduler.");

  using SchedulerTag = cute::conditional_t<
    cute::is_void_v<TileScheduler_>,
    cute::conditional_t<
      IsGroupedGemmKernel,
      GroupScheduler,     // Special grouped gemm scheduler
      void                // Default scheduler for non-grouped kernels
    >,
    TileScheduler_
  >;

  using TileScheduler = typename detail::StrassenTileSchedulerSelector<
                                          SchedulerTag, 
                                          ArchTag, 
                                          TileShape,
                                          ClusterShape,
                                          8, // SchedulerPipelineStageCount -- Grouped GEMM scheduler will benefit from a larger number of stages
                                          cute::conditional_t<cute::is_same_v<SchedulerTag, void>, void, ProblemShape>
                                          >::Scheduler;

  // using TileScheduler = typename detail::TileSchedulerSelector<
  //   SchedulerTag,
  //   ArchTag,
  //   TileShape,
  //   ClusterShape,
  //   8, // SchedulerPipelineStageCount -- Grouped GEMM scheduler will benefit from a larger number of stages.
  //   cute::conditional_t<cute::is_same_v<SchedulerTag, void>, void, ProblemShape> // Use void for default scheduler.
  // >::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using TileSchedulerResponse = typename TileSchedulerResponseGetter<TileScheduler>::Type;

  static constexpr auto TileSchedulerStages = 8;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaThreads = size(TiledMma{});
  static constexpr uint32_t NumMmaWarpGroups = NumMmaThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = NumMmaThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t kThreadCount = MaxThreadsPerBlock;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;
  static constexpr bool     IsMainloopAuxiliaryLoadNeeded = detail::HasAuxiliaryLoad_v<typename CollectiveMainloop::DispatchPolicy>;
  static constexpr uint32_t ConsumerSubMIterations = StrassenMiGroup::numMs();

  /// Register requirement for Load and Math WGs
  static constexpr uint32_t LoadRegisterRequirement = 40;
  static constexpr uint32_t MmaRegisterRequirement = 232;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  static const bool DoesPresum = (StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
                                 (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum());

  // Kernel level shared memory storage
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      typename CollectiveMainloop::PresumTensorStorage presum_tensors;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using TileSchedulerPipelineStorage = typename TileScheduler::PipelineStorage;
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) TileSchedulerPipelineStorage scheduler;
      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;

    alignas(16) TileSchedulerResponse scheduler_response[TileSchedulerStages];

    struct TensorMapStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorMapStorage = typename CollectiveMainloop::TensorMapStorage;
      using EpilogueTensorMapStorage = typename CollectiveEpilogue::TensorMapStorage;

      alignas(128) MainloopTensorMapStorage mainloop;
      alignas(128) EpilogueTensorMapStorage epilogue;
    } tensormaps;
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

    Arguments() :
      mode(), problem_shape(), mainloop(), epilogue(),
      hw_info(), scheduler() 
    {}

    Arguments(GemmUniversalMode mode, ProblemShape problem_shape,
              MainloopArguments mainloop, EpilogueArguments epilogue,
              KernelHardwareInfo hw_info, TileSchedulerArguments scheduler = TileSchedulerArguments()) :
      mode(mode), problem_shape(problem_shape), mainloop(mainloop), epilogue(epilogue),
      hw_info(hw_info), scheduler(scheduler) 
    {}

    CUTLASS_HOST_DEVICE
    int get_problem_shape_m(int group_idx) const {
      return get<0>(problem_shape.get_problem_shape(group_idx));
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_n(int group_idx = 0) const {
      return get<1>(problem_shape.get_problem_shape(group_idx));
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_k(int group_idx = 0) const {
      return get<2>(problem_shape.get_problem_shape(group_idx));
    }

    CUTLASS_HOST_DEVICE
    ProblemShape get_half_problem_shape(int group_idx = 0) const {
      return ProblemShape{get_problem_shape_m(group_idx)/2, get_problem_shape_n(group_idx)/2,
                          get_problem_shape_k(group_idx)/2};
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
    ElementA** ptr_A;
    ElementB** ptr_B;
    ElementD** ptr_D;
    ElementA* presum_m_a_workspace;
    uint64_t* presum_a_batch_indices;
    ElementB* presum_m_b_workspace;
    uint64_t* presum_b_batch_indices;
    ElementD* postsum_m_workspace;
    uint64_t* postsum_m_batch_indices;

    void* workspace{nullptr};

    int run = 0;

     CUTLASS_HOST_DEVICE
    int get_problem_shape_m(int problem_idx) const {
      return get<0>(problem_shape.get_problem_shape(problem_idx));
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_n(int problem_idx = 0) const {
      return get<1>(problem_shape.get_problem_shape(problem_idx));
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_k(int problem_idx = 0) const {
      return get<2>(problem_shape.get_problem_shape(problem_idx));
    }

    CUTLASS_HOST_DEVICE
    int get_stride_A(int problem_idx = 0) const {
      return get_problem_shape_k(problem_idx);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_B(int problem_idx = 0) const {
      return get_problem_shape_n(problem_idx);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MA(int problem_idx = 0) const {
      return get_problem_shape_k(problem_idx)/2;
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MB(int problem_idx = 0) const {
      return get_problem_shape_n(problem_idx)/2;
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
    std::remove_cvref_t<ProblemShape> get_half_problem_shape(int problem_idx) const {
      return std::remove_cvref_t<ProblemShape>{get_problem_shape_m(problem_idx)/2,
                                               get_problem_shape_n(problem_idx)/2,
                                               get_problem_shape_k(problem_idx)/2};
    }

    CUTLASS_HOST_DEVICE
    ElementA* get_ptr_A(int problem_idx) const {
      return ptr_A[problem_idx];
    }

    CUTLASS_HOST_DEVICE
    ElementB* get_ptr_B(int problem_idx) const {
      return ptr_B[problem_idx];
    }

    CUTLASS_HOST_DEVICE
    ElementD* get_ptr_D(int problem_idx) const {
      return ptr_D[problem_idx];
    }

    CUTLASS_HOST_DEVICE
    ElementA* get_ptr_presum_A(int problem_idx) const {
      return presum_m_a_workspace + presum_a_batch_indices[problem_idx];
    }

    CUTLASS_HOST_DEVICE
    ElementB* get_ptr_presum_B(int problem_idx) const {
      return presum_m_b_workspace + presum_b_batch_indices[problem_idx];
    }

    CUTLASS_HOST_DEVICE
    ElementD* get_postsum_ptr(int problem_idx) const {
      return postsum_m_workspace ;//+ postsum_m_batch_indices[problem_idx];
    }
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, ElementA* presum_m_a, uint64_t* presum_a_batch_indices, ElementB* presum_m_b, uint64_t* presum_b_batch_indices, ElementD* postsum_m, uint64_t* postsum_m_batch_indices, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    ProblemShape problem_shapes = args.problem_shape;

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
    workspace_offset += CollectiveEpilogue::get_workspace_size(problem_shapes, args.epilogue, sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);

    void* mainloop_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveMainloop::get_workspace_size(problem_shapes, args.mainloop, sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);

    TileSchedulerParams scheduler;
    if constexpr (IsGroupedGemmKernel) {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace);
    }
    else {
      scheduler = TileScheduler::to_underlying_arguments(
      problem_shapes.get_host_problem_shape(), TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace);
    }

    return {
      args.mode,
      problem_shapes,
      CollectiveMainloop::to_underlying_arguments(problem_shapes, presum_m_a, presum_a_batch_indices, presum_m_b, presum_b_batch_indices, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(problem_shapes, args.epilogue, postsum_m, postsum_m_batch_indices, epilogue_workspace),
      hw_info,
      scheduler,
      const_cast<ElementA**>(args.mainloop.ptr_A),
      const_cast<ElementB**>(args.mainloop.ptr_B),
      const_cast<ElementD**>(args.epilogue.ptr_D),
      presum_m_a, presum_a_batch_indices,
      presum_m_b, presum_b_batch_indices,
      postsum_m, postsum_m_batch_indices,
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    if constexpr (IsGroupedGemmKernel) {
      // Group GEMM currently only supports rank-3 problem shapes
      implementable &= (args.mode == GemmUniversalMode::kGrouped && rank(typename ProblemShape::UnderlyingProblemShape{}) == 3);
    }
    else {
      implementable &= (args.mode == GemmUniversalMode::kArray && rank(typename ProblemShape::UnderlyingProblemShape{}) == 4);
    }
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements for Ptr Array Gemm or Grouped Gemm.\n");
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

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, sm_count);
    workspace_size = round_nearest(workspace_size, MinTensorMapWorkspaceAlignment);

    workspace_size += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, sm_count);
    workspace_size = round_nearest(workspace_size, MinTensorMapWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size, MinTensorMapWorkspaceAlignment);

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
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = CollectiveMainloop::initialize_workspace(args.problem_shape, args.mainloop, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveMainloop::get_workspace_size(args.problem_shape, args.mainloop, args.hw_info.sm_count);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<typename ProblemShape::UnderlyingProblemShape, ElementAccumulator>(
      args.scheduler, typename ProblemShape::UnderlyingProblemShape{}, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset, MinTensorMapWorkspaceAlignment);
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
    dim3 grid_shape;
    if constexpr (IsGroupedGemmKernel) {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    else {
      grid_shape = TileScheduler::get_grid_shape(params.scheduler, params.problem_shape.get_host_problem_shape(), TileShape{}, ClusterShape{}, params.hw_info, args);
    }
    return grid_shape;
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

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif

// Any Tensor Op MMA Atom in the ISA is arch conditional.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    printf("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(size(TiledMma{}) == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");
    static_assert(NumMmaWarpGroups == 2, "Cooperative kernels currently only support NumMmaWarpGroups == 2");

    if constexpr (cutlass::epilogue::collective::detail::sm90_is_ptr_array_tma_dispatch_policy_v<typename CollectiveEpilogue::DispatchPolicy>) {
      static_assert(NumMmaWarpGroups == CollectiveEpilogue::NumEpilogueWarpGroups,
                    "Tiled MmA does not match expected warp groups performing the epilogue");
    }

    static_assert(cute::rank(InternalStrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(InternalStrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      MainloopAux = 1,
      Epilogue = 2,
      Scheduler = 3
    };

    auto scheduler = [&] () {
      // Group scheduler requires a different constructor that takes a response ptr
      if constexpr (cute::is_same_v<SchedulerTag, GroupScheduler>) {
        return TileScheduler{params.scheduler, shared_storage.scheduler_response};
      }
      else {
        return TileScheduler{params.scheduler};
      }
    } ();

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % size(TiledMma{});
    auto warp_group_idx = canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Note: Tma Descriptor Prefetch (from either const or param) is not applicable here

    // TileScheduler pipeline
    using TileSchedulerPipeline = typename TileScheduler::Pipeline;
    typename TileSchedulerPipeline::Params tile_scheduler_pipeline_params;
    if constexpr (cute::is_same_v<SchedulerTag, GroupScheduler>) {
      if (warp_group_role == WarpGroupRole::Producer
        && producer_warp_role == ProducerWarpRole::Scheduler) {
        tile_scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Producer;
      }
      else {
        tile_scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Consumer;
      }
      tile_scheduler_pipeline_params.consumer_arv_count = NumMmaThreads
                                                        + NumThreadsPerWarp * (
                                                          1                                                           // Main DMA warp
                                                          + (collective_epilogue.is_producer_load_needed() ? 1 : 0)   // Epilog DMA warp
                                                          + (IsMainloopAuxiliaryLoadNeeded ? 1 : 0)                   // Aux DMA warp
                                                        );
      tile_scheduler_pipeline_params.producer_arv_count = 1;
    }
    TileSchedulerPipeline tile_scheduler_pipeline(shared_storage.pipelines.scheduler, tile_scheduler_pipeline_params);
    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer
      && (producer_warp_role == ProducerWarpRole::Mainloop
       || producer_warp_role == ProducerWarpRole::MainloopAux)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumMmaThreads;
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
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = size(TiledMma{});
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
    params_load_order_barrier.group_size = NumMmaThreads;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename TileSchedulerPipeline::PipelineState tile_scheduler_pipe_consumer_state;
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState tile_scheduler_pipe_producer_state = cutlass::make_producer_start_state<TileSchedulerPipeline>();
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

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    const auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    const auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    const auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

    if (not work_tile_info.is_valid()) {
      // When problem shapes are only on device, the grid launched may be larger than the total number of blocks across groups
      return;
    }

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
    // auto half_problem_shape_MNKL = append<4>(params.get_half_problem_shape(work_tile_info.L_idx), 1);

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
  //  auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);

    auto all_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    // auto all_presumld_inputs = collective_mainloop.presumld_inputs(problem_shape_MNKL, half_problem_shape_MNKL, params.mainloop);
    auto load_inputs = collective_mainloop.get_inputs(all_inputs);
    auto load_inputs2 = collective_mainloop.get_inputs(all_inputs, 1);
    auto load_inputs3 = collective_mainloop.get_inputs(all_inputs, 2);

    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    constexpr bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();
    if (threadIdx.x == 0)
      MY_PRINTF("746 %d %d\n", blockIdx.x, blockIdx.y);

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = (params.get_problem_shape_k(work_tile_info.L_idx)/2)/decltype(size<2>(blk_shape))::value;

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

      if (producer_warp_role == ProducerWarpRole::Scheduler) {
        // GroupScheduler requires a producer warp to iterate over the group infos and push
        // the work tile infos to the downstream pipelines.
        if constexpr (cute::is_same_v<SchedulerTag, GroupScheduler>) {
          do {
            auto [next_work_tile_info, increment_pipe] = scheduler.advance_to_next_work(tile_scheduler_pipeline, tile_scheduler_pipe_producer_state);
            work_tile_info = next_work_tile_info;
            if (increment_pipe) {
              ++tile_scheduler_pipe_producer_state;
            }
          } while (work_tile_info.is_valid());
          tile_scheduler_pipeline.producer_tail(tile_scheduler_pipe_producer_state);
        }
      }
      // Mainloop Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Mainloop) {
        if (threadIdx.x%32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            MY_PRINTF("769 %d %d\n", work_tile_info.M_idx, work_tile_info.N_idx);
        int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx;
        int32_t const mock_l_coord = 0;
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;
                   if (threadIdx.x%32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
                MY_PRINTF("775 %d %d\n", work_tile_info.M_idx, work_tile_info.N_idx);
        // Fetch a copy ofcurr_batch tensormaps for the CTA
        auto input_tensormaps = collective_mainloop.tensormaps_init(params.mainloop, shared_storage.tensormaps.mainloop, sm_count, sm_idx);

        // Update tensormap for the initial batch for the CTA
        collective_mainloop.tensormaps_perform_update(
          shared_storage.tensormaps.mainloop,
          params.mainloop,
          input_tensormaps,
          problem_shape_MNKL,
          curr_batch
        );
        // Ensure warp is converged before issuing tensormap fence release
        __syncwarp();
        // Entire warp must do this (i.e. it's aligned)
        collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);

        // bool do_load_order_arrive = true;
        bool did_batch_change = true;
        do {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
                work_tile_info, tile_scheduler_pipeline, tile_scheduler_pipe_consumer_state);
            work_tile_info = next_work_tile_info;
            if (increment_pipe) {
              ++tile_scheduler_pipe_consumer_state;
            }
            continue;
          }

          if (threadIdx.x%32 == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
            MY_PRINTF("803 %d %d\n", work_tile_info.M_idx, work_tile_info.N_idx);
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          if (threadIdx.x%32 == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
            MY_PRINTF("810 %d %d\n", work_tile_info.M_idx, work_tile_info.N_idx);
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, mock_l_coord);

          // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape)/2;
          auto work_k_tile_start = 0;//TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          if (did_batch_change) {
            load_inputs = collective_mainloop.tensors_perform_update(load_inputs, params.mainloop, problem_shape_MNKL, curr_batch);
            collective_mainloop.tensormaps_fence_acquire(shared_storage.tensormaps.mainloop, input_tensormaps);
          }
          if (threadIdx.x%32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            MY_PRINTF("817 %d : %d %d\n", StrassenMiGroup::getMi(), m_coord, n_coord);

          if (StrassenMiGroup::numMs() > 1) {
            if (threadIdx.x%32 == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
                MY_PRINTF("826 %d %d\n", m_coord, n_coord);
            auto load_work_tile = [&] (decltype(work_tile_info) load_work_tile_info, int load_sub_m_idx) {
              if (threadIdx.x%32 == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
                MY_PRINTF("829 %d %d\n", m_coord, n_coord);
              auto load_m_coord = idx2crd(load_work_tile_info.M_idx, shape<2>(gA_mkl));
              auto load_n_coord = idx2crd(load_work_tile_info.N_idx, shape<2>(gB_nkl));
              auto load_l_coord = idx2crd(load_work_tile_info.L_idx, shape<4>(gB_nkl));
              auto load_blk_coord = make_coord(load_m_coord, load_n_coord, _, load_l_coord);
              auto load_k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
              load_k_tile_iter.coord += (is_fused && load_sub_m_idx == 1) ? work_k_tile_count : 0;

              if (threadIdx.x%32 == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
                MY_PRINTF("1133 %d : %d : %d %d\n", work_k_tile_count, load_sub_m_idx, load_m_coord, load_n_coord);

              collective_mainloop.load(
                params.mainloop,
                problem_shape_MNKL,
                mainloop_pipeline,
                mainloop_pipe_producer_state,
                (is_fused || load_sub_m_idx == 0) ? load_inputs :
                  ((load_sub_m_idx == 1) ? load_inputs2 : load_inputs3),
                input_tensormaps,
                load_blk_coord, load_sub_m_idx,
                load_k_tile_iter, work_k_tile_count,
                lane_idx,
                block_rank_in_cluster,
                shared_storage.tensors.presum_tensors,
                shared_storage.tensors.mainloop
              );
              // Pipeline state is only advanced if there are K tiles to compute
              mainloop_pipe_producer_state.advance(work_k_tile_count);
            };

            load_work_tile(work_tile_info, 0);
            load_work_tile(work_tile_info, 1);
            if (StrassenMiGroup::numMs() == 3) {
              load_work_tile(work_tile_info, 2);
            }
          } else {
            if (threadIdx.x%32 == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
                MY_PRINTF("863 %d %d\n", m_coord, n_coord);
            collective_mainloop.load(
              params.mainloop,
              problem_shape_MNKL,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              input_tensormaps,
              blk_coord, 0,
              k_tile_iter, work_k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.presum_tensors,
              shared_storage.tensors.mainloop
            );
            // Pipeline state is only advanced if there are K tiles to compute
            mainloop_pipe_producer_state.advance(work_k_tile_count);
          }

          // // Signal for the epilogue load warp to begin
          // if (do_load_order_arrive) {
          //   load_order_barrier.arrive();
          //   do_load_order_arrive = false;
          // }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, tile_scheduler_pipeline, tile_scheduler_pipe_consumer_state);
          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++tile_scheduler_pipe_consumer_state;
          }
          auto next_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx
          did_batch_change = next_batch != curr_batch;
          if (work_tile_info.is_valid() && did_batch_change) {
            curr_batch = next_batch;
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(curr_batch), 1);
            }
            collective_mainloop.tensormaps_perform_update(
              shared_storage.tensormaps.mainloop,
              params.mainloop,
              input_tensormaps,
              problem_shape_MNKL,
              curr_batch
            );
            // Ensure warp is converged before issuing tensor replace
            __syncwarp();
            // Entire warp must do this (i.e. it's aligned)
            collective_mainloop.tensormaps_cp_fence_release(shared_storage.tensormaps.mainloop, input_tensormaps);
          }
        } while (work_tile_info.is_valid()); // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);
      } // Mainloop Producer Warp End
      else if (producer_warp_role == ProducerWarpRole::MainloopAux) {
        if constexpr (IsMainloopAuxiliaryLoadNeeded) {
          int32_t curr_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx;
          int32_t const mock_l_coord = 0;

          bool did_batch_change = true;
          do {
            if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
              auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, tile_scheduler_pipeline, tile_scheduler_pipe_consumer_state);
              work_tile_info = next_work_tile_info;
              if (increment_pipe) {
                ++tile_scheduler_pipe_consumer_state;
              }
              continue;
            }

            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, mock_l_coord);

            // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
            auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape)/2;
            auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
            auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

            if (did_batch_change) {
              load_inputs = collective_mainloop.tensors_perform_update(load_inputs, params.mainloop, problem_shape_MNKL, curr_batch);
            }

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
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, tile_scheduler_pipeline, tile_scheduler_pipe_consumer_state);
            work_tile_info = next_work_tile_info;
            if (increment_pipe) {
              ++tile_scheduler_pipe_consumer_state;
            }
            auto next_batch = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl)); // Usually just returns work_tile_info.L_idx
            did_batch_change = next_batch != curr_batch;
            if (work_tile_info.is_valid() && did_batch_change) {
              curr_batch = next_batch;
              if constexpr (IsGroupedGemmKernel) {
                problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(curr_batch), 1);
              }
            }
          } while (work_tile_info.is_valid()); // Scheduler work fetch loop
        } // End of auxiliary load needed check
      } // Mainloop Auxiliary Load Producer Warp End
      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && collective_epilogue.is_producer_load_needed()) {
        int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
        int32_t const sm_count = params.hw_info.sm_count;

        // Since load requires linear we do not need to do this
        auto epi_load_tensormap = get<0>(collective_epilogue.load_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx));

        bool did_batch_change = true;
        constexpr bool IsEpiLoad = true;

        collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
          shared_storage.tensormaps.epilogue,
          params.epilogue,
          epi_load_tensormap,
          problem_shape_MNKL,
          work_tile_info.L_idx,
          0
        );

        // Converge before issuing tensormap fence release since fence is aligned
        __syncwarp();
        collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);

        // load_order_barrier.wait();
        do {
          int32_t curr_batch = work_tile_info.L_idx;

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, tile_scheduler_pipeline, tile_scheduler_pipe_consumer_state);

          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            #pragma unroll (StrassenMiGroup::numMs())
            for (int fused_mi = 0; fused_mi < StrassenMiGroup::numMs(); fused_mi++) {
              // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
              auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
              auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
              auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
              auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

              if (did_batch_change) {
                collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_load_tensormap);
              }

              bool wait = work_tile_info.is_valid() && curr_batch != next_work_tile_info.L_idx;

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
              // epi_load_pipe_producer_state = collective_epilogue.load(
              //   epi_load_pipeline,
              //   epi_load_pipe_producer_state,
              //   problem_shape_MNKL,
              //   blk_shape,
              //   blk_coord,
              //   tiled_mma,
              //   lane_idx,
              //   shared_storage.tensors.epilogue,
              //   epi_load_tensormap,
              //   work_tile_info.reduction_subtile_idx(),
              //   wait
              // );
            }
          }

          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++tile_scheduler_pipe_consumer_state;
          }
          did_batch_change = curr_batch != work_tile_info.L_idx;

          if (work_tile_info.is_valid() && did_batch_change) {
            if constexpr (IsGroupedGemmKernel) {
              problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
            }

            // tensormap update
            {
              collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
                shared_storage.tensormaps.epilogue,
                params.epilogue,
                epi_load_tensormap,
                problem_shape_MNKL,
                work_tile_info.L_idx,
                0
              );

              // Converge before issuing tensormap fence release since fence is aligned
              __syncwarp();
              collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue, epi_load_tensormap, 0);
            }
          }

        } while (work_tile_info.is_valid()); // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      // Index of warp group within consumer warp groups
      int consumer_warp_group_idx = warp_group_role == WarpGroupRole::Consumer0 ? 0 : 1;
      const int mma_thread_idx = (threadIdx.x - 128) % NumMmaThreads;

      int32_t const sm_idx = blockIdx.x + (blockIdx.y * gridDim.x);
      int32_t const sm_count = params.hw_info.sm_count;
      // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
      bool do_store_tail = false;
      // Get a copy of tensormaps
      auto epi_store_tensormap = get<0>(collective_epilogue.store_init(params.epilogue, shared_storage.tensormaps.epilogue, sm_count, sm_idx, consumer_warp_group_idx));

      bool did_batch_change = true;
      constexpr bool IsEpiLoad = false;

      if (warp_idx_in_warp_group == 0) {
        collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
          shared_storage.tensormaps.epilogue,
          params.epilogue,
          epi_store_tensormap,
          problem_shape_MNKL,
          work_tile_info.L_idx,
          consumer_warp_group_idx
        );

        // Converge before issuing tensormap fence release since fence is aligned
        __syncwarp();
        collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue,
                                                                    epi_store_tensormap,
                                                                    consumer_warp_group_idx);
      }

      do {
        if constexpr (IsGroupedGemmKernel) {
          problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
        }

        int32_t curr_batch = work_tile_info.L_idx;

        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape)/2;

        // if (mma_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        //     printf("817 %d : %d %d\n", StrassenMiGroup::getMi(), m_coord, n_coord);

        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)

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

        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {

          collective_mainloop.mma(
            problem_shape_MNKL,
            blk_coord,
            sub_m_idx,
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            work_k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            shared_storage.tensors.presum_tensors,
            params.mainloop
          );

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            work_k_tile_count
          );

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
        }

        // Perform reduction across splits, if needed
        // TileScheduler::fixup(
        //   params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

        if (did_batch_change) {
          collective_epilogue.template tensormaps_fence_acquire<IsEpiLoad>(epi_store_tensormap);
        }

        if (is_neg)
          for (int i = 0; i < accumulators.size(); i++)
            accumulators[i] = -1 * accumulators[i];

        decltype(epi_load_pipe_consumer_state) epi_load_pipe_consumer_state_next = epi_load_pipe_consumer_state;
        decltype(epi_store_pipe_producer_state) epi_store_pipe_producer_state_next = epi_store_pipe_producer_state;

        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler) && any_global_dst_valid) {
          if (mma_thread_idx == 0 && StrassenMiGroup::hasM2() && sub_m_idx == 0 && n_coord == 2)
            MY_PRINTF("1278 %d %d %d %d ; %d %d\n", threadIdx.x, StrassenMiGroup::getMi(sub_m_idx), m_coord, n_coord, work_tile_info.M_idx, work_tile_info.N_idx);
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
              epi_store_tensormap,
              work_tile_info.reduction_subtile_idx()
            );

            epi_load_pipe_consumer_state_next = get<0>(ret);
            epi_store_pipe_producer_state_next = get<1>(ret);
          }
          if (mma_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            MY_PRINTF("1315 %d : %d %d %d\n", threadIdx.x, StrassenMiGroup::getMi(sub_m_idx), m_coord, n_coord);
          do_store_tail = true;
        }

        // auto ret = collective_epilogue.store_tail(
        //   epi_load_pipeline,
        //   epi_load_pipe_consumer_state_next,
        //   epi_store_pipeline,
        //   epi_store_pipe_producer_state_next,
        //   has_global_src
        // );

        // epi_load_pipe_consumer_state = get<0>(ret);
        // epi_store_pipe_producer_state = get<1>(ret);
        epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;

        if (threadIdx.x%256 == 0)// && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1174 %d : %d %d %d\n", work_k_tile_count, sub_m_idx, m_coord, n_coord);
        }

        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info, tile_scheduler_pipeline, tile_scheduler_pipe_consumer_state);
        work_tile_info = next_work_tile_info;
        if (increment_pipe) {
          ++tile_scheduler_pipe_consumer_state;
        }

        did_batch_change = curr_batch != work_tile_info.L_idx;
        if (work_tile_info.is_valid() && did_batch_change) {
          if constexpr (IsGroupedGemmKernel) {
            problem_shape_MNKL = append<4>(params.problem_shape.get_problem_shape(work_tile_info.L_idx), 1);
          }
          if (warp_idx_in_warp_group == 0) {
            collective_epilogue.template tensormaps_perform_update<IsEpiLoad>(
              shared_storage.tensormaps.epilogue,
              params.epilogue,
              epi_store_tensormap,
              problem_shape_MNKL,
              work_tile_info.L_idx,
              consumer_warp_group_idx
            );

            // Converge before issuing tensormap fence release since fence is aligned
            __syncwarp();
            collective_epilogue.template tensormaps_cp_fence_release<IsEpiLoad>(shared_storage.tensormaps.epilogue,
                                                                       epi_store_tensormap,
                                                                       consumer_warp_group_idx);
          }
        }

      } while (work_tile_info.is_valid()); // Scheduler work fetch loop

      // Cooperative only needs TMA to complete at the very end of the kernel
      if (do_store_tail) {
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state,
          collective_epilogue.is_producer_load_needed()
        );
      }
    } // Consumer Warp Groups End
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
