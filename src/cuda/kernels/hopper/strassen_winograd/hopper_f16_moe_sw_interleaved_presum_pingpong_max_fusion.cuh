#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoePingpongKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using MoePingpongEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using MoePingpongScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoePingpongKernelSchedule, MoePingpongEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoePingpongStrassenGemmKernelsForCluster = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroupsForCluster<ClusterShape_, StageCount>, MoePingpongScheduleStrassenGroups, MoeProblemShape,
    ArchTag, OperatorClass,
    ElementA, cutlass::layout::RowMajor *, cutlass::layout::OriginalLayout,
    ElementB, LayoutB_ *, cutlass::layout::OriginalLayout,
    ElementC, LayoutCD_ *, cutlass::layout::OriginalLayout, ElementAccumulator,
    ClusterShape_, cute::Int<StageCount>, PresumTileShapeA, PresumTileShapeB,
    cutlass::gemm::device::PresumOpt<0,0,0,0>,
    AlignmentA, AlignmentB, AlignmentC, ElementD>;
template <typename LayoutB_, typename LayoutCD_, int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoePingpongStrassenGemmKernels = MoePingpongStrassenGemmKernelsForCluster<LayoutB_, LayoutCD_, ClusterShape, StageCount, PresumTileShapeA, PresumTileShapeB>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_>
using HopperF16MoeInterleavedPresumPingpongMaxFusion_2x128ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongStrassenGemmKernelsForCluster<LayoutB_, LayoutCD_, ClusterShape_, 6, Shape<_2,_128>, Shape<_2,_128>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumPingpongMaxFusion_2x128 = HopperF16MoeInterleavedPresumPingpongMaxFusion_2x128ForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumPingpongMaxFusion_2x128_1x2 = HopperF16MoeInterleavedPresumPingpongMaxFusion_2x128ForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_>
using HopperF16MoeInterleavedPresumPingpongMaxFusion_4x128ForCluster =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoePingpongStrassenGemmKernelsForCluster<LayoutB_, LayoutCD_, ClusterShape_, 6, Shape<_4,_128>, Shape<_4,_128>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumPingpongMaxFusion_4x128 = HopperF16MoeInterleavedPresumPingpongMaxFusion_4x128ForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16MoeInterleavedPresumPingpongMaxFusion_4x128_1x2 = HopperF16MoeInterleavedPresumPingpongMaxFusion_4x128ForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;