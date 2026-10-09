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

template <typename ClusterShape_>
using MoeCooperativePingpongStrassenKernelsForCluster =
    cutlass::gemm::device::StrassenGemmKernels<
        CooperativePingpongStrassenGroupsForCluster<ClusterShape_, 4>, MoeCooperativePingpongSchedule,
        MoeCooperativePingpongProblemShape,
        ArchTag, OperatorClass,
        ElementA, LayoutA *, cutlass::layout::OriginalLayout,
        ElementB, LayoutB *, cutlass::layout::OriginalLayout,
        ElementC, LayoutC *, cutlass::layout::OriginalLayout,
        ElementAccumulator, ClusterShape_, cute::Int<4>,
        Shape<_2,_256>, Shape<_2,_256>,
        cutlass::gemm::device::PresumOpt<0,0,0,0>,
        AlignmentA, AlignmentB, AlignmentC, ElementD>;
using MoeCooperativePingpongStrassenKernels = MoeCooperativePingpongStrassenKernelsForCluster<ClusterShape>;

template <typename ClusterShape_>
using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusion_2x256ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativePingpongStrassenKernelsForCluster<ClusterShape_>>;
using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusion_2x256 = HopperF16MoeInterleavedPresumCooperativePingpongMaxFusion_2x256ForCluster<ClusterShape>;
using HopperF16MoeInterleavedPresumCooperativePingpongMaxFusion_2x256_1x2 = HopperF16MoeInterleavedPresumCooperativePingpongMaxFusion_2x256ForCluster<Shape<_1,_2,_1>>;