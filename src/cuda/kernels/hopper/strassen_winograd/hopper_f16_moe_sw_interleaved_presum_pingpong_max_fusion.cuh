#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoePingpongKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using MoePingpongEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using MoePingpongScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template<int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoePingpongStrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroups<StageCount>, MoePingpongScheduleStrassenGroups, MoeProblemShape,
    ElementA, LayoutA *, cutlass::layout::OriginalLayout,
    ElementB, LayoutB *, cutlass::layout::OriginalLayout,
    ElementC, LayoutC *, cutlass::layout::OriginalLayout, ElementAccumulator,
    ClusterShape, cute::Int<StageCount>, PresumTileShapeA, PresumTileShapeB,
    cutlass::gemm::device::PresumOpt<0,0,0,0>>;

using HopperF16MoeInterleavedPresumPingpongMaxFusion_2x128 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongStrassenGemmKernels<6, Shape<_2,_128>, Shape<_2,_128>>>;
using HopperF16MoeInterleavedPresumPingpongMaxFusion_4x128 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongStrassenGemmKernels<6, Shape<_4,_128>, Shape<_4,_128>>>;