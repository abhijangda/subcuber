#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion.cuh"

template <typename ClusterShape_, int StageCountTypeM0>
using StrassenGroupsTmaReduceForCluster = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM0>,
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
                                                               ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 2>>,
                                                               ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 4>>>;

template <typename ClusterShape_, int StageCountTypeM0, typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using StrassenGemmKernelsTmaReduceForCluster = cutlass::gemm::device::StrassenGemmKernels<StrassenGroupsTmaReduceForCluster<ClusterShape_, StageCountTypeM0>,
                                                                       ScheduleStrassenGroupsTmaReduce,
                                                                       ProblemShape,
                                                                       ArchTag, OperatorClass,
                                                                       ElementA, LayoutA, cutlass::layout::OriginalLayout,
                                                                       ElementB, LayoutB, cutlass::layout::OriginalLayout,
                                                                       ElementC, LayoutC, cutlass::layout::OriginalLayout,
                                                                       ElementAccumulator, ClusterShape_,
                                                                       cute::Int<StageCountTypeM0>,
                                                                       PresumTileShapeA, PresumTileShapeB,
                                                                       PresumOpts,
                                                                       AlignmentA, AlignmentB, AlignmentC,
                                                                       ElementD>;
template <int StageCountTypeM0, typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using StrassenGemmKernelsTmaReduce = StrassenGemmKernelsTmaReduceForCluster<ClusterShape, StageCountTypeM0, PresumTileShapeA, PresumTileShapeB, PresumOpts>;

template <typename ClusterShape_>
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster = HopperF16InterleavedPresumCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 4, Shape<_2,_256>, Shape<_2,_256>>>>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_OptNo = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster<ClusterShape>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_OptNo_1x2 = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_OptNoForCluster<Shape<_1,_2,_1>>;
template <typename ClusterShape_>
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster = HopperF16InterleavedPresumCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 4, Shape<_2,_256>, Shape<_2,_256>, cutlass::gemm::device::PresumOpt<0,0,0,0>>>>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000 = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster<ClusterShape>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000_1x2 = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_2x256_2x256_Opt_0000ForCluster<Shape<_1,_2,_1>>;
template <typename ClusterShape_>
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster = HopperF16InterleavedPresumCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 4, Shape<_4,_256>, Shape<_4,_256>>>>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_4x256_4x256_OptNo = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster<ClusterShape>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_4x256_4x256_OptNo_1x2 = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_4x256_4x256_OptNoForCluster<Shape<_1,_2,_1>>;
template <typename ClusterShape_>
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster = HopperF16InterleavedPresumCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 4, Shape<_8,_256>, Shape<_8,_256>>>>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_8x256_8x256_OptNo = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster<ClusterShape>;
using HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_8x256_8x256_OptNo_1x2 = HopperF16InterleavedPresumCooperativeMaxFusionTmaReduce_8x256_8x256_OptNoForCluster<Shape<_1,_2,_1>>;