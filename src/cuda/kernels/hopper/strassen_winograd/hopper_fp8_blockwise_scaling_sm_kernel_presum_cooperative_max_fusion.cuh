#pragma once

#define MY_PRINTF(...) ;

#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cute/tensor.hpp"
#include "cutlass/layout/strassen_layout.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_strassen_gemm_builder.hpp"
#include "cutlass/epilogue/collective/collective_strassen_builder.hpp"
#include "cutlass/gemm/device/strassen_gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/strassen_gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/device/strassen_decls.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"

using namespace MmaStrassen;
using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
using SubMatLayoutA = cutlass::layout::StrassenLayout;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
using SubMatLayoutB = cutlass::layout::StrassenLayout;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
using ElementC = void;
using ElementD = cutlass::half_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ProblemShape = Shape<int,int,int>;
using TileShape = Shape<_128,_128,_128>;
using ClusterShape = Shape<_1,_2,_1>;
const uint StageCountTypeM0 = 6;
const uint StageCountTypeM2M6 = 5;
const uint StageCountTypeM4M5 = 5;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using AllPresumsM0 = AllPresums<PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel,
                                PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
                                   PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

template <bool UseTmaReduce>
using StrassenGroupsForReduce = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, AllPresumsM0>,
    StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM0,
        RWMTypes<>,
        RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,
                 CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
        AllPresumsM0, 0, 0, 1>,
    StrassenLevel1M1Group<1, 0, TileShape, ClusterShape, StageCountTypeM0,
        RWMTypes<>,
        RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
        AllPresumsM0>,
    StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
        RWMTypes<>,
        conditional_t<UseTmaReduce,
            RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>>>,
                     CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>,
                     CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>>>,
            RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>>>,
                     CUW<2, LayoutInterim, LayoutNone, Expr<Plus<3>>>,
                     CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>>>>,
        AllPresumsM1To6, 0, 2, 3, 6>,
    StrassenLevel1M3Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
        RWMTypes<>,
        RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
        AllPresumsM1To6>,
    StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM4M5,
        RWMTypes<>,
        conditional_t<UseTmaReduce,
            RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<3, MemGlobal, LayoutFinal>>>,
                     CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutFinal>>>>,
            RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim>>>,
                     CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim>>>>>,
        AllPresumsM1To6, 0, 4, 5>,
    StrassenLevel1M5Group<1, 0, TileShape, ClusterShape, StageCountTypeM4M5,
        RWMTypes<>,
        RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                     Plus<0, MemGlobal, LayoutInterim1D>>>>,
        AllPresumsM1To6>,
    StrassenLevel1M6Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
        RWMTypes<>,
        RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
        AllPresumsM1To6>>;

using ScheduleStrassenGroups1 = ScheduleStrassenGroups<
    ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 0>>,
    ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 2>>,
    ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 4>>>;
using ScheduleStrassenGroupsPar = ScheduleStrassenGroups<
    ParallelMiGroups<KernelSchedule, EpilogueSchedule, false,
        FusedMiGroup<7, 0>, FusedMiGroup<7, 2>, FusedMiGroup<7, 4>>>;

template <bool UseTmaReduce, typename Schedule = ScheduleStrassenGroups1>
using StrassenGemmKernelsForReduce = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroupsForReduce<UseTmaReduce>, Schedule,
    ProblemShape, ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, SubMatLayoutA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, SubMatLayoutB,
    ElementC, LayoutC, cutlass::layout::OriginalLayout,
    ElementAccumulator, ClusterShape, cute::Int<StageCountTypeM0>,
    Shape<_4,_128>, Shape<_4,_128>, cutlass::gemm::device::PresumOpt<>,
    AlignmentA, AlignmentB, AlignmentC, ElementD>;

template <bool UseTmaReduce, typename Schedule = ScheduleStrassenGroups1>
using HopperFp8BlockwiseScalingSmKernelPresumCooperativeMaxFusionForReduce =
    cutlass::gemm::device::StrassenGemmUniversalAdapter<StrassenGemmKernelsForReduce<UseTmaReduce, Schedule>>;

using HopperFp8BlockwiseScalingSmKernelPresumCooperativeMaxFusion_4x128_4x128_OptNo_1x2 =
    HopperFp8BlockwiseScalingSmKernelPresumCooperativeMaxFusionForReduce<false>;
using HopperFp8BlockwiseScalingSmKernelPresumCooperativeMaxFusion_4x128_4x128_OptNo_1x2_Par =
    HopperFp8BlockwiseScalingSmKernelPresumCooperativeMaxFusionForReduce<false, ScheduleStrassenGroupsPar>;