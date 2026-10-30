#include "cuda/kernels/hopper/b_presum_kernel/hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion.cuh"

template <typename ClusterShape_, int StageCountTypeM0, typename PresumPolicy = AllPresumsM0>
using StrassenGroupsTmaReduceForCluster = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, PresumPolicy>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  PresumPolicy, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShape, ClusterShape_, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  PresumPolicy>,
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
                                                               ParallelMiGroups<KernelSchedule, EpilogueSchedule, false, FusedMiGroup<7, 2>,
                                                               /*ParallelMiGroups<KernelSchedule, EpilogueSchedule, false,*/ FusedMiGroup<7, 4>>
                                                              >;

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
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 6, Shape<_2,_128>, Shape<_2, _128>, cutlass::gemm::device::PresumOpt<>, Schedule>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernelForCluster<LayoutB_, LayoutCD_, ClusterShape>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT {
public:
  using StrassenGemmKernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoKernelForCluster<LayoutB_, LayoutCD_, ClusterShape_, Schedule>;
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

template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNo = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_1x2 = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_1x2_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000KernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 6, Shape<_2,_128>, Shape<_2, _128>,
                                                                                                        cutlass::gemm::device::PresumOpt<0,0,0,0>, Schedule>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000Kernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000KernelForCluster<LayoutB_, LayoutCD_, ClusterShape>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T {
public:
  using StrassenGemmKernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000KernelForCluster<LayoutB_, LayoutCD_, ClusterShape_, Schedule>;
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

template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000 = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_1x2 = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_1x2_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 6, Shape<_4,_128>, Shape<_4, _128>, cutlass::gemm::device::PresumOpt<>, Schedule>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernelForCluster<LayoutB_, LayoutCD_, ClusterShape>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT {
public:
  using StrassenGemmKernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoKernelForCluster<LayoutB_, LayoutCD_, ClusterShape_, Schedule>;
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

template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNo = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_1x2 = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_1x2_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernelForCluster = StrassenGemmUniversalAdapter<
                                                                           StrassenGemmKernelsTmaReduceForCluster<LayoutB_, LayoutCD_, ClusterShape_, 5, Shape<_8,_128>, Shape<_8, _128>, cutlass::gemm::device::PresumOpt<>, Schedule>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernelForCluster<LayoutB_, LayoutCD_, ClusterShape>;

template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule = ScheduleStrassenGroupsTmaReduce>
class HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT {
public:
  using StrassenGemmKernel = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoKernelForCluster<LayoutB_, LayoutCD_, ClusterShape_, Schedule>;
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

template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNo = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<LayoutB_, LayoutCD_, ClusterShape>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<LayoutB_, LayoutCD_, ClusterShape, ScheduleStrassenGroupsPar>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_1x2 = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<LayoutB_, LayoutCD_, Shape<_1,_2,_1>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_1x2_Par = HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT<LayoutB_, LayoutCD_, Shape<_1,_2,_1>, ScheduleStrassenGroupsPar>;

#define DEFINE_TMA_REDUCE_METHODS(class_name) \
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<LayoutB_, LayoutCD_, ClusterShape_, Schedule>::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::can_implement(args); \
} \
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule> \
size_t class_name<LayoutB_, LayoutCD_, ClusterShape_, Schedule>::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::get_workspace_size(args); \
} \
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<LayoutB_, LayoutCD_, ClusterShape_, Schedule>::init(Arguments const &args, int swizzles[7], void *workspace, \
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter); \
} \
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<LayoutB_, LayoutCD_, ClusterShape_, Schedule>::launch(cudaStream_t *streams, int num_streams, \
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<LayoutB_, LayoutCD_, ClusterShape_, Schedule>::initialize(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return init(args, swizzles, workspace, stream, cuda_adapter); \
} \
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<LayoutB_, LayoutCD_, ClusterShape_, Schedule>::run(cudaStream_t *streams, int num_streams, \
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
template <typename LayoutB_, typename LayoutCD_, typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<LayoutB_, LayoutCD_, ClusterShape_, Schedule>::operator()(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t *streams, int num_streams, \
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  cutlass::Status status = init(args, swizzles, workspace, \
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter); \
  if (status == cutlass::Status::kSuccess) { \
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
  } \
  return status; \
}

DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT)
DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T)
DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT)
DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT)

#undef DEFINE_TMA_REDUCE_METHODS
