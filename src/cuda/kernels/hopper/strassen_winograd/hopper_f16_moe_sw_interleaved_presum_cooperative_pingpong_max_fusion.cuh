#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_pingpong_max_fusion.cuh"

using MoeCooperativePingpongProblemShape =
    cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoeCooperativeKernelSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeEpilogueSchedule =
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using MoePingpongKernelSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using MoePingpongEpilogueSchedule =
    cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using MoeCooperativePingpongSchedule = ScheduleStrassenGroups<
    ParallelMiGroups<MoeCooperativeKernelSchedule, MoeCooperativeEpilogueSchedule,
                     false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule,
                     false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule,
                     false, FusedMiGroup<7, 4>>>;

using MoeCooperativePingpongStrassenKernels =
    cutlass::gemm::device::StrassenGemmKernels<
        CooperativePingpongStrassenGroups<4>, MoeCooperativePingpongSchedule,
        MoeCooperativePingpongProblemShape,
        ElementA, LayoutA *, cutlass::layout::OriginalLayout,
        ElementB, LayoutB *, cutlass::layout::OriginalLayout,
        ElementC, LayoutC *, cutlass::layout::OriginalLayout,
        ElementAccumulator, ClusterShape, cute::Int<4>,
        Shape<_2,_256>, Shape<_2,_256>,
        cutlass::gemm::device::PresumOpt<0,0,0,0>>;

using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusion_2x256 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativePingpongStrassenKernels>;