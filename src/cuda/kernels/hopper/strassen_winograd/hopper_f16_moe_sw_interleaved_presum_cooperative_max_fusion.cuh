#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoeCooperativeKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoeCooperativeKernelSchedule, MoeCooperativeEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoeCooperativeKernelSchedule, MoeCooperativeEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoeCooperativeKernelSchedule, MoeCooperativeEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template <typename ClusterShape_, int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoeCooperativeStrassenGemmKernelsForCluster = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroupsForCluster<ClusterShape_, StageCount>, MoeCooperativeScheduleStrassenGroups, MoeProblemShape,
    ArchTag, OperatorClass,
    ElementA, LayoutA *, cutlass::layout::OriginalLayout,
    ElementB, LayoutB *, cutlass::layout::OriginalLayout,
    ElementC, LayoutC *, cutlass::layout::OriginalLayout, ElementAccumulator,
    ClusterShape_, cute::Int<StageCount>, PresumTileShapeA, PresumTileShapeB,
    cutlass::gemm::device::PresumOpt<0,0,0,0>,
    AlignmentA, AlignmentB, AlignmentC, ElementD>;
template <int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoeCooperativeStrassenGemmKernels = MoeCooperativeStrassenGemmKernelsForCluster<ClusterShape, StageCount, PresumTileShapeA, PresumTileShapeB>;

template <typename ClusterShape_>
using HopperF16MoeInterleavedPresumCooperativeMaxFusion_2x256ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativeStrassenGemmKernelsForCluster<ClusterShape_, 4, Shape<_2,_256>, Shape<_2,_256>>>;
using HopperF16MoeInterleavedPresumCooperativeMaxFusion_2x256 = HopperF16MoeInterleavedPresumCooperativeMaxFusion_2x256ForCluster<ClusterShape>;
using HopperF16MoeInterleavedPresumCooperativeMaxFusion_2x256_1x2 = HopperF16MoeInterleavedPresumCooperativeMaxFusion_2x256ForCluster<Shape<_1,_2,_1>>;