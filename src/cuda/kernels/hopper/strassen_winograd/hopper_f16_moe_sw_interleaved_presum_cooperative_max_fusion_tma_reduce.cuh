#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion_tma_reduce.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoeCooperativeTmaReduceKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeTmaReduceEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeTmaReduceScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoeCooperativeTmaReduceKernelSchedule, MoeCooperativeTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoeCooperativeTmaReduceKernelSchedule, MoeCooperativeTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoeCooperativeTmaReduceKernelSchedule, MoeCooperativeTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoeCooperativeTmaReduceStrassenGemmKernelsForCluster = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroupsTmaReduceForCluster<ClusterShape_, StageCount>, MoeCooperativeTmaReduceScheduleStrassenGroups,
    MoeProblemShape, ArchTag, OperatorClass,
    ElementA, cutlass::layout::RowMajor *, cutlass::layout::OriginalLayout,
    ElementB, LayoutB_ *, cutlass::layout::OriginalLayout,
    ElementC, LayoutCD_ *, cutlass::layout::OriginalLayout,
    ElementAccumulator, ClusterShape_, cute::Int<StageCount>, PresumTileShapeA,
    PresumTileShapeB, cutlass::gemm::device::PresumOpt<0,0,0,0>,
    AlignmentA, AlignmentB, AlignmentC, ElementD>;
template <typename LayoutB_, typename LayoutCD_, int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoeCooperativeTmaReduceStrassenGemmKernels = MoeCooperativeTmaReduceStrassenGemmKernelsForCluster<LayoutB_, LayoutCD_, ClusterShape, StageCount, PresumTileShapeA, PresumTileShapeB>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_>
using HopperF16MoeInterleavedPresumCooperativeMaxFusionTmaReduce_2x256ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativeTmaReduceStrassenGemmKernelsForCluster<LayoutB_, LayoutCD_, ClusterShape_, 4, Shape<_2,_256>, Shape<_2,_256>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumCooperativeMaxFusionTmaReduce_2x256 = HopperF16MoeInterleavedPresumCooperativeMaxFusionTmaReduce_2x256ForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumCooperativeMaxFusionTmaReduce_2x256_1x2 = HopperF16MoeInterleavedPresumCooperativeMaxFusionTmaReduce_2x256ForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;