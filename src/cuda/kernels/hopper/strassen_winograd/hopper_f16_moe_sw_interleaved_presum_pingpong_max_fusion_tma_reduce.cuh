#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoePingpongTmaReduceKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using MoePingpongTmaReduceEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using MoePingpongTmaReduceScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule, MoePingpongTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule, MoePingpongTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule, MoePingpongTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template<int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoePingpongTmaReduceStrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroupsTmaReduce<StageCount>, MoePingpongTmaReduceScheduleStrassenGroups,
    MoeProblemShape, ElementA, LayoutA *, cutlass::layout::OriginalLayout,
    ElementB, LayoutB *, cutlass::layout::OriginalLayout,
    ElementC, LayoutC *, cutlass::layout::OriginalLayout,
    ElementAccumulator, ClusterShape, cute::Int<StageCount>, PresumTileShapeA,
    PresumTileShapeB, cutlass::gemm::device::PresumOpt<0,0,0,0>>;

using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_2x128 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongTmaReduceStrassenGemmKernels<6, Shape<_2,_128>, Shape<_2,_128>>>;
using HopperF16MoeInterleavedPresumPingpongMaxFusionTmaReduce_4x128 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongTmaReduceStrassenGemmKernels<6, Shape<_4,_128>, Shape<_4,_128>>>;