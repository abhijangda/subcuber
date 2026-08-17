#pragma once

#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion_tma_reduce.cuh"

using MoeProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>;
using MoeCooperativeTmaReduceKernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeTmaReduceEpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using MoeCooperativeTmaReduceScheduleStrassenGroups = ScheduleStrassenGroups<
    ParallelMiGroups<MoeCooperativeTmaReduceKernelSchedule, MoeCooperativeTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<MoeCooperativeTmaReduceKernelSchedule, MoeCooperativeTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<MoeCooperativeTmaReduceKernelSchedule, MoeCooperativeTmaReduceEpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template<int StageCount, typename PresumTileShapeA, typename PresumTileShapeB>
using MoeCooperativeTmaReduceStrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroupsTmaReduce<StageCount>, MoeCooperativeTmaReduceScheduleStrassenGroups,
    MoeProblemShape, ElementA, LayoutA *, ElementB, LayoutB *, ElementC, LayoutC *,
    ElementAccumulator, ClusterShape, cute::Int<StageCount>, PresumTileShapeA,
    PresumTileShapeB, cutlass::gemm::device::PresumOpt<0,0,0,0>>;

using HopperF16MoeInterleavedPresumCooperativeMaxFusionTmaReduce_2x256 =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<
        MoeCooperativeTmaReduceStrassenGemmKernels<4, Shape<_2,_256>, Shape<_2,_256>>>;