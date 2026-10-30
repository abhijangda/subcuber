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

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_>
using MoeCooperativePingpongTmaReduceStrassenKernelsForCluster =
    cutlass::gemm::device::StrassenGemmKernels<
        CooperativePingpongStrassenGroupsTmaReduceForCluster<ClusterShape_, 4>,
        MoeCooperativePingpongTmaReduceSchedule,
        MoeCooperativePingpongTmaReduceProblemShape,
        ArchTag, OperatorClass,
        ElementA, cutlass::layout::RowMajor *, cutlass::layout::OriginalLayout,
        ElementB, LayoutB_ *, cutlass::layout::OriginalLayout,
        ElementC, LayoutCD_ *, cutlass::layout::OriginalLayout,
        ElementAccumulator, ClusterShape_, cute::Int<4>,
        Shape<_2,_256>, Shape<_2,_256>,
        cutlass::gemm::device::PresumOpt<0,0,0,0>,
        AlignmentA, AlignmentB, AlignmentC, ElementD>;
template <typename LayoutB_, typename LayoutCD_>
using MoeCooperativePingpongTmaReduceStrassenKernels = MoeCooperativePingpongTmaReduceStrassenKernelsForCluster<LayoutB_, LayoutCD_, ClusterShape>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_>
using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativePingpongTmaReduceStrassenKernelsForCluster<LayoutB_, LayoutCD_, ClusterShape_>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256 = HopperF16MoeInterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256ForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256_1x2 = HopperF16MoeInterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256ForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;