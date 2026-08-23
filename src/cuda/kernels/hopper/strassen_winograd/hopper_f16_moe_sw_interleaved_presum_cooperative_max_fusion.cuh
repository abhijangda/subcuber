#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoeCooperativeKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoeCooperativeKernelSchedule, MoeCooperativeEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoeCooperativeKernelSchedule, MoeCooperativeEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoeCooperativeKernelSchedule, MoeCooperativeEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template<int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoeCooperativeStrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroups<StageCount>, MoeCooperativeScheduleStrassenGroups, MoeProblemShape,
    ElementA, LayoutA *, cutlass::layout::OriginalLayout,
    ElementB, LayoutB *, cutlass::layout::OriginalLayout,
    ElementC, LayoutC *, cutlass::layout::OriginalLayout, ElementAccumulator,
    ClusterShape, cute::Int<StageCount>, PresumTileShapeA, PresumTileShapeB,
    cutlass::gemm::device::PresumOpt<0,0,0,0>>;

using HopperF16MoeInterleavedPresumCooperativeMaxFusion_2x256 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativeStrassenGemmKernels<4, Shape<_2,_256>, Shape<_2,_256>>>;