#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_pingpong_max_fusion_tma_reduce.cuh"

using MoeCooperativePingpongTmaReduceProblemShape =
    cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoeCooperativeTmaReduceKernelSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeTmaReduceEpilogueSchedule =
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using MoePingpongTmaReduceKernelSchedule =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using MoePingpongTmaReduceEpilogueSchedule =
    cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using MoeCooperativePingpongTmaReduceSchedule = ScheduleStrassenGroups<
    ParallelMiGroups<MoeCooperativeTmaReduceKernelSchedule,
                     MoeCooperativeTmaReduceEpilogueSchedule,
                     false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule,
                     MoePingpongTmaReduceEpilogueSchedule,
                     false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoePingpongTmaReduceKernelSchedule,
                     MoePingpongTmaReduceEpilogueSchedule,
                     false, FusedMiGroup<7, 4>>>;

using MoeCooperativePingpongTmaReduceStrassenKernels =
    cutlass::gemm::device::StrassenGemmKernels<
        CooperativePingpongStrassenGroupsTmaReduce<4>,
        MoeCooperativePingpongTmaReduceSchedule,
        MoeCooperativePingpongTmaReduceProblemShape,
        ElementA, LayoutA *, cutlass::layout::OriginalLayout,
        ElementB, LayoutB *, cutlass::layout::OriginalLayout,
        ElementC, LayoutC *, cutlass::layout::OriginalLayout,
        ElementAccumulator, ClusterShape, cute::Int<4>,
        Shape<_2,_256>, Shape<_2,_256>,
        cutlass::gemm::device::PresumOpt<0,0,0,0>>;

using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativePingpongTmaReduceStrassenKernels>;