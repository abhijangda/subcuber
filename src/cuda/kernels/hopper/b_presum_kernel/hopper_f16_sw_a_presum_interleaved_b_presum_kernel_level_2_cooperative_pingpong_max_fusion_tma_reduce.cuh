#define MY_PRINTF(...) ;

#define L2_SCHED_TMA_REDUCE_ADD

#include "cutlass/cutlass.h"
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

using namespace MmaStrassen;
using namespace cute;

using         ElementA    = cutlass::half_t;
using         LayoutA     = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;
using         ElementB    = cutlass::half_t;
using         LayoutB     = cutlass::layout::RowMajor;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;
using         ElementD    = cutlass::half_t;
using         LayoutC     = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm90;
using OperatorClass       = cutlass::arch::OpClassTensorOp;
using ProblemShape        = Shape<int,int,int>;
using TileShapeM0         = Shape<_128,_256,_64>;
using TileShapeM2To6      = Shape<_128,_128,_64>;
using ClusterShape        = Shape<_2,_1,_1>;
const uint StageCountTypeM0 = 4;
const uint StageCountTypeM2M6 = 6;
using KernelScheduleM0 = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueScheduleM0 = cutlass::epilogue::TmaWarpSpecializedCooperative;
using KernelScheduleM2To6 = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using EpilogueScheduleM2To6 = cutlass::epilogue::TmaWarpSpecialized;

template <typename LayoutB_>
using AllPresumsM0ForLayout = cute::conditional_t<
    cute::is_same_v<LayoutB_, cutlass::layout::ColumnMajor>,
    AllPresums<PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel,
               PresumCompute, PresumCompute, PresumCompute, PresumCompute>,
    AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
               PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel>>;
using AllPresumsM0 = AllPresumsM0ForLayout<LayoutB>;//PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
                                   PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

template <typename PresumPolicy = AllPresumsM0>
using StrassenGroupsM0ForPolicy = StrassenLevel1Groups<StrassenPresum<2, 0, TileShapeM0, PresumPolicy>,
                                              StrassenLevel1MiGroup<2, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>>,
                                                                    PresumPolicy, 0, 0>>;
using StrassenGroupsM0 = StrassenGroupsM0ForPolicy<>;

template<int Level1Idx, typename PresumPolicy = AllPresumsM0>
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShapeM0, PresumPolicy>,
                                            StrassenLevel1MiGroup<1, Level1Idx, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
                                                                  PresumPolicy, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, Level1Idx, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  PresumPolicy>,
                                            StrassenLevel1MiGroup<1, Level1Idx, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>>>,
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>,
                                                                           CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>>>,
                                                                  AllPresumsM1To6, 0, 2, 3, 6>,
                                            StrassenLevel1M3Group<1, Level1Idx, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, Level1Idx, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<3, MemGlobal, LayoutFinal>>>,
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutFinal>>>>,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, Level1Idx, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, Level1Idx, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM1To6>>;

using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<KernelScheduleM0, EpilogueScheduleM0, false, FusedMiGroup<7, 0>>,
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 2>,
                                                       /*ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, */FusedMiGroup<7, 4>>>;

template <typename LayoutB_, typename LayoutCD_, typename Groups, typename ElementL2Postsum, typename PresumTileShapeA, typename PresumTileShapeB,
         typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using StrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<Groups,
                                                                       ScheduleStrassenGroups1,
                                                                       ProblemShape,
                                                                       ArchTag, OperatorClass,
                                                                       ElementA, cutlass::layout::RowMajor, cutlass::layout::OriginalLayout,
                                                                       ElementB, LayoutB_, cutlass::layout::OriginalLayout,
                                                                       ElementL2Postsum, LayoutCD_, cutlass::layout::OriginalLayout,
                                                                       ElementAccumulator, ClusterShape,
                                                                       cute::Int<StageCountTypeM0>,
                                                                       PresumTileShapeA, PresumTileShapeB,
                                                                       PresumOpts,
                                                                       AlignmentA, AlignmentB, AlignmentC,
                                                                       ElementD>;

template <typename LayoutB_, typename LayoutCD_, typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using StrassenGemmLevel2UniversalAdapter = cutlass::gemm::device::StrassenGemmLevel2UniversalAdapter<
    StrassenGemmKernels<LayoutB_, LayoutCD_, StrassenGroupsM0ForPolicy<AllPresumsM0ForLayout<LayoutB_>>, void, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<LayoutB_, LayoutCD_, StrassenGroups<1, AllPresumsM0ForLayout<LayoutB_>>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<LayoutB_, LayoutCD_, StrassenGroups<2, AllPresumsM0ForLayout<LayoutB_>>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<LayoutB_, LayoutCD_, StrassenGroups<3, AllPresumsM0ForLayout<LayoutB_>>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<LayoutB_, LayoutCD_, StrassenGroups<4, AllPresumsM0ForLayout<LayoutB_>>, void, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<LayoutB_, LayoutCD_, StrassenGroups<5, AllPresumsM0ForLayout<LayoutB_>>, void, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<LayoutB_, LayoutCD_, StrassenGroups<6, AllPresumsM0ForLayout<LayoutB_>>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    true>;

template <typename StrassenGemmKernel_>
class HopperF16APresumInterleavedBPresumKernelLevel2CooperativePingpongMaxFusionTmaReduceBase {
public:
  using StrassenGemmKernel = StrassenGemmKernel_;
  using GemmKernel = typename StrassenGemmKernel::ChildStrassenGemmM0::GemmKernel;
  using Arguments = typename StrassenGemmKernel::Arguments;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

  static cutlass::Status can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr) {
    if (get<0>(args.problem_shape) % (4 * size<0>(TileShapeM0{})) != 0 ||
        get<1>(args.problem_shape) % (4 * size<1>(TileShapeM0{})) != 0 ||
        get<2>(args.problem_shape) % (4 * size<2>(TileShapeM0{})) != 0) {
      return cutlass::Status::kErrorInvalidProblem;
    }
    return StrassenGemmKernel::can_implement(args);
  }

  static size_t get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr) {
    return StrassenGemmKernel::get_workspace_size(args);
  }

  cutlass::Status init(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                       cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr) {
    return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
  }

  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0,
                         cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false) {
    return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
  }

  cutlass::Status initialize(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t stream = nullptr, cutlass::CudaHostAdapter *cuda_adapter = nullptr) {
    return init(args, swizzles, workspace, stream, cuda_adapter);
  }

  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0,
                      cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false) {
    return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }

  cutlass::Status operator()(Arguments const &args, int swizzles[7], void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0,
                             cutlass::CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false) {
    cutlass::Status status = init(args, swizzles, workspace,
                                  streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
    if (status == cutlass::Status::kSuccess) {
      status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
    }
    return status;
  }

private:
  StrassenGemmKernel gemm_;
};

template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelLevel2CooperativePingpongMaxFusionTmaReduce_2x256_2x256_OptNo = HopperF16APresumInterleavedBPresumKernelLevel2CooperativePingpongMaxFusionTmaReduceBase<StrassenGemmLevel2UniversalAdapter<LayoutB_, LayoutCD_, Shape<_2,_256>, cute::conditional_t<cute::is_same_v<LayoutB_, cutlass::layout::ColumnMajor>, Shape<_4,_128>, Shape<_2,_256>>>>;
template <typename LayoutB_, typename LayoutCD_>
using HopperF16APresumInterleavedBPresumKernelLevel2CooperativePingpongMaxFusionTmaReduce_4x256_4x256_OptNo = HopperF16APresumInterleavedBPresumKernelLevel2CooperativePingpongMaxFusionTmaReduceBase<StrassenGemmLevel2UniversalAdapter<LayoutB_, LayoutCD_, Shape<_4,_256>, cute::conditional_t<cute::is_same_v<LayoutB_, cutlass::layout::ColumnMajor>, Shape<_8,_128>, Shape<_4,_256>>>>;