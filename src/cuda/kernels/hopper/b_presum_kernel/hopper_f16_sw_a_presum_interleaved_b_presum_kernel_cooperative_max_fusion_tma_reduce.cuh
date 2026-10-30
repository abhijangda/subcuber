#include "cuda/kernels/hopper/b_presum_kernel/hopper_f16_sw_a_presum_interleaved_b_presum_kernel_cooperative_max_fusion.cuh"

template <typename ClusterShape_, int StageCountTypeM0, typename PresumPolicy = AllPresumsM0>
using StrassenGroupsTmaReduceForCluster = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, PresumPolicy>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
                                                                  PresumPolicy, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  PresumPolicy>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>>>,
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>,
                                                                           CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>>>,
                                                                  AllPresumsM1To6, 0, 2, 3, 6>,
                                            StrassenLevel1M3Group<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<3, MemGlobal, LayoutFinal>>>,
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutFinal>>>>,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>>;
template <int StageCountTypeM0>
using StrassenGroupsTmaReduce = StrassenGroupsTmaReduceForCluster<ClusterShape, StageCountTypeM0>;

using ScheduleStrassenGroupsTmaReduce = ScheduleStrassenGroups<ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 0>>,
                                                               ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 2>,
                                                               /*ParallelMiGroups<KernelSchedule, EpilogueSchedule, false,*/ FusedMiGroup<7, 4>>>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, int StageCountTypeM0, typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using StrassenGemmKernelsTmaReduceForCluster = cutlass::gemm::device::StrassenGemmKernels<StrassenGroupsTmaReduceForCluster<ClusterShape_, StageCountTypeM0, AllPresumsM0ForLayout<LayoutB_>>,
                                                                       Schedule,
                                                                       ProblemShape,
                                                                       ArchTag, OperatorClass,
                                                                       ElementA, cutlass::layout::RowMajor, cutlass::layout::StrassenLayout,
                                                                       ElementB, LayoutB_, cutlass::layout::StrassenLayout,
                                                                       ElementC, LayoutCD_, cutlass::layout::OriginalLayout,
                                                                       ElementAccumulator, ClusterShape_,
                                                                       cute::Int<StageCountTypeM0>,
                                                                       PresumTileShapeA, PresumTileShapeB,
                                                                       PresumOpts,
                                                                       AlignmentA, AlignmentB, AlignmentC,
                                                                       ElementD>;
template <typename LayoutB_, typename LayoutCD_, int StageCountTypeM0, typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using StrassenGemmKernelsTmaReduce = StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape, StageCountTypeM0, PresumTileShapeA, PresumTileShapeB, PresumOpts>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 4, Shape<_2,_256>, cute::conditional_t<cute::is_same_v<LayoutB_, cutlass::layout::ColumnMajor>, Shape<_4,_128>, Shape<_2,_256>>, cutlass::gemm::device::PresumOpt<>, Schedule>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNo = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNo_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNo_1x2 = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNo_1x2_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 4, Shape<_2,_256>, cute::conditional_t<cute::is_same_v<LayoutB_, cutlass::layout::ColumnMajor>, Shape<_4,_128>, Shape<_2,_256>>, cutlass::gemm::device::PresumOpt<0,0,0,0>, Schedule>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000 = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000_1x2 = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000_1x2_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 4, Shape<_4,_256>, cute::conditional_t<cute::is_same_v<LayoutB_, cutlass::layout::ColumnMajor>, Shape<_8,_128>, Shape<_4,_256>>, cutlass::gemm::device::PresumOpt<>, Schedule>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNo = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNo_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNo_1x2 = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNo_1x2_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 4, Shape<_8,_256>, Shape<_8,_256>, cutlass::gemm::device::PresumOpt<>, Schedule>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNo = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNo_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNo_1x2 = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNo_1x2_Par = HopperF16APresumInterleavedBPresumKernelCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;