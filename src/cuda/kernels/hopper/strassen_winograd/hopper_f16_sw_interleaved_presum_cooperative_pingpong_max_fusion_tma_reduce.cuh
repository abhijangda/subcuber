#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_pingpong_max_fusion.cuh"

template<int StageCountTypeM0>
using CooperativePingpongStrassenGroupsTmaReduce = StrassenLevel1Groups<StrassenPresum<1, 0, CooperativePingpongTileShapeM0, AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, CooperativePingpongTileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, CooperativePingpongTileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, CooperativePingpongTileShapeM2To6, ClusterShape, CooperativePingpongStageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>>>,
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>,
                                                                           CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>>>,
                                                                  AllPresumsM1To6, 0, 2, 3, 6>,
                                            StrassenLevel1M3Group<1, 0, CooperativePingpongTileShapeM2To6, ClusterShape, CooperativePingpongStageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, 0, CooperativePingpongTileShapeM2To6, ClusterShape, CooperativePingpongStageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<3, MemGlobal, LayoutFinal>>>,
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutFinal>>>>,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, CooperativePingpongTileShapeM2To6, ClusterShape, CooperativePingpongStageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, CooperativePingpongTileShapeM2To6, ClusterShape, CooperativePingpongStageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>>;

using CooperativePingpongScheduleStrassenGroupsTmaReduce = ScheduleStrassenGroups<ParallelMiGroups<CooperativePingpongKernelScheduleM0, CooperativePingpongEpilogueScheduleM0, false, FusedMiGroup<7, 0>>,
                                                                                   ParallelMiGroups<CooperativePingpongKernelScheduleM2To6, CooperativePingpongEpilogueScheduleM2To6, false, FusedMiGroup<7, 2>>,
                                                                                   ParallelMiGroups<CooperativePingpongKernelScheduleM2To6, CooperativePingpongEpilogueScheduleM2To6, false, FusedMiGroup<7, 4>>>;

template<int StageCountTypeM0, typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using CooperativePingpongStrassenGemmKernelsTmaReduce = cutlass::gemm::device::StrassenGemmKernels<CooperativePingpongStrassenGroupsTmaReduce<StageCountTypeM0>,
                                                                                                   CooperativePingpongScheduleStrassenGroupsTmaReduce,
                                                                                                   ProblemShape,
                                                                                                   ElementA, LayoutA, ElementB, LayoutB,
                                                                                                   ElementC, LayoutC,
                                                                                                   ElementAccumulator, ClusterShape,
                                                                                                   cute::Int<StageCountTypeM0>,
                                                                                                   PresumTileShapeA, PresumTileShapeB,
                                                                                                   PresumOpts>;

using HopperF16InterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256_2x256_OptNo = HopperF16InterleavedPresumCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<CooperativePingpongStrassenGemmKernelsTmaReduce<4, Shape<_2,_256>, Shape<_2,_256>>>>;
using HopperF16InterleavedPresumCooperativePingpongMaxFusionTmaReduce_2x256_2x256_Opt_0000 = HopperF16InterleavedPresumCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<CooperativePingpongStrassenGemmKernelsTmaReduce<4, Shape<_2,_256>, Shape<_2,_256>, cutlass::gemm::device::PresumOpt<0,0,0,0>>>>;
using HopperF16InterleavedPresumCooperativePingpongMaxFusionTmaReduce_4x256_4x256_OptNo = HopperF16InterleavedPresumCooperativeMaxFusionBase<StrassenGemmUniversalAdapter<CooperativePingpongStrassenGemmKernelsTmaReduce<4, Shape<_4,_256>, Shape<_4,_256>>>>;