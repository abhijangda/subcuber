#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoePingpongTmaReduceKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using MoePingpongTmaReduceEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using MoePingpongTmaReduceScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule, MoePingpongTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule, MoePingpongTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule, MoePingpongTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template <typename ClusterShape_, int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoePingpongTmaReduceStrassenGemmKernelsForCluster = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroupsTmaReduceForCluster<ClusterShape_, StageCount>, MoePingpongTmaReduceScheduleStrassenGroups,
    MoeProblemShape, ArchTag, OperatorClass,
    ElementA, LayoutA *, cutlass::layout::OriginalLayout,
    ElementB, LayoutB *, cutlass::layout::OriginalLayout,
    ElementC, LayoutC *, cutlass::layout::OriginalLayout,
    ElementAccumulator, ClusterShape_, cute::Int<StageCount>, PresumTileShapeA,
    PresumTileShapeB, cutlass::gemm::device::PresumOpt<0,0,0,0>,
    AlignmentA, AlignmentB, AlignmentC, ElementD>;
template <int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoePingpongTmaReduceStrassenGemmKernels = MoePingpongTmaReduceStrassenGemmKernelsForCluster<ClusterShape, StageCount, PresumTileShapeA, PresumTileShapeB>;

template <typename ClusterShape_>
using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_2x128ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongTmaReduceStrassenGemmKernelsForCluster<ClusterShape_, 6, Shape<_2,_128>, Shape<_2,_128>>>;
using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_2x128 = HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_2x128ForCluster<ClusterShape>;
using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_2x128_1x2 = HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_2x128ForCluster<Shape<_1,_2,_1>>;
template <typename ClusterShape_>
using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_4x128ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongTmaReduceStrassenGemmKernelsForCluster<ClusterShape_, 6, Shape<_4,_128>, Shape<_4,_128>>>;
using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_4x128 = HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_4x128ForCluster<ClusterShape>;
using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_4x128_1x2 = HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_4x128ForCluster<Shape<_1,_2,_1>>;