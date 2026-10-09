#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cuh"

template <typename ClusterShape_, int StageCountTypeM0>
using StrassenGroupsTmaReduceForCluster = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>> >, //C1 = Sh = C1+M2 ; Reg = C1
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>/*, Expr<Plus<1, MemShared, LayoutInterim1D>>*/ >, //C2 = C1Sh+M3
                                                                           CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>> >,
                                                                  AllPresumsM1To6, 0, 2, 3, 6>,
                                            StrassenLevel1M3Group<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes</*CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>,*/ //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<3, MemGlobal, LayoutFinal>>>,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutFinal>>//,
                                                                                                                               /*Plus<0, MemShared, LayoutInterim1D>*/
                                                                                                                               >
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShape, ClusterShape_, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresumsM1To6>
                                            >;
template <int StageCountTypeM0>
using StrassenGroupsTmaReduce = StrassenGroupsTmaReduceForCluster<ClusterShape, StageCountTypeM0>;
using ScheduleStrassenGroupsTmaReduce = ScheduleStrassenGroups<ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 0>>,
                                                               ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 2>>,
                                                               ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 4>>
                                                              >;

template <typename ClusterShape_, int StageCountTypeM0, typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using StrassenGemmKernelsTmaReduceForCluster = cutlass::gemm::device::StrassenGemmKernels<StrassenGroupsTmaReduceForCluster<ClusterShape_, StageCountTypeM0>,
                                                                       Schedule,
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

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 6, Shape<_2,_128>, Shape<_2, _128>, cutlass::gemm::device::PresumOpt<>, Schedule>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernelForCluster<ClusterShape>;

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT {
public:
  using StrassenGemmKernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernelForCluster<ClusterShape_, Schedule>;
  using GemmKernel = typename StrassenGemmKernel::GemmKernel;
  using Arguments = typename StrassenGemmKernel::Arguments;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

  static cutlass::Status can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  static size_t get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);

  cutlass::Status init(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                       cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0,
                         cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

  cutlass::Status initialize(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0,
                      cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);
  cutlass::Status operator()(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0,
                             cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

private:
  StrassenGemmKernel gemm_;
};

using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<ClusterShape>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<ClusterShape, ScheduleStrassenGroupsPar>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_1x2 = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<Shape<_1,_2,_1>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_1x2_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000KernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 6, Shape<_2,_128>, Shape<_2, _128>,
                                                                                                        cutlass::gemm::device::PresumOpt<0,0,0,0>, Schedule>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000Kernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000KernelForCluster<ClusterShape>;

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T {
public:
  using StrassenGemmKernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000KernelForCluster<ClusterShape_, Schedule>;
  using GemmKernel = typename StrassenGemmKernel::GemmKernel;
  using Arguments = typename StrassenGemmKernel::Arguments;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

  static cutlass::Status can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  static size_t get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);

  cutlass::Status init(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                       cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0,
                         cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

  cutlass::Status initialize(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0,
                      cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);
  cutlass::Status operator()(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0,
                             cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

private:
  StrassenGemmKernel gemm_;
};

using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000 = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<ClusterShape>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<ClusterShape, ScheduleStrassenGroupsPar>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_1x2 = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<Shape<_1,_2,_1>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_1x2_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 6, Shape<_4,_128>, Shape<_4, _128>, cutlass::gemm::device::PresumOpt<>, Schedule>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernelForCluster<ClusterShape>;

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT {
public:
  using StrassenGemmKernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernelForCluster<ClusterShape_, Schedule>;
  using GemmKernel = typename StrassenGemmKernel::GemmKernel;
  using Arguments = typename StrassenGemmKernel::Arguments;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

  static cutlass::Status can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  static size_t get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);

  cutlass::Status init(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                       cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0,
                         cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

  cutlass::Status initialize(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0,
                      cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);
  cutlass::Status operator()(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0,
                             cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

private:
  StrassenGemmKernel gemm_;
};

using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<ClusterShape>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<ClusterShape, ScheduleStrassenGroupsPar>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_1x2 = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<Shape<_1,_2,_1>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_1x2_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<ClusterShape_, 5, Shape<_8,_128>, Shape<_8, _128>, cutlass::gemm::device::PresumOpt<>, Schedule>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernelForCluster<ClusterShape>;

template <typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT {
public:
  using StrassenGemmKernel = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernelForCluster<ClusterShape_, Schedule>;
  using GemmKernel = typename StrassenGemmKernel::GemmKernel;
  using Arguments = typename StrassenGemmKernel::Arguments;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

  static cutlass::Status can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  static size_t get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr);

  cutlass::Status init(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                       cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0,
                         cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

  cutlass::Status initialize(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr);
  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0,
                      cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);
  cutlass::Status operator()(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0,
                             cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false);

private:
  StrassenGemmKernel gemm_;
};

using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<ClusterShape>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<ClusterShape, ScheduleStrassenGroupsPar>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_1x2 = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<Shape<_1,_2,_1>>;
using HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_1x2_Par = HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;