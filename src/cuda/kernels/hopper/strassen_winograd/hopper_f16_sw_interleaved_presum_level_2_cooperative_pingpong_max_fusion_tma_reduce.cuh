#define MY_PRINTF(...) ;

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

using AllPresumsM0 = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel, PresumGlobalKernel>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
                                   PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

using StrassenGroupsM0 = StrassenLevel1Groups<StrassenPresum<2, 0, TileShapeM0, AllPresumsM0>,
                                              StrassenLevel1MiGroup<2, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>>,
                                                                    AllPresumsM0, 0, 0>>;

template<int Level1Idx>
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShapeM0, AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, Level1Idx, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, Level1Idx, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                  AllPresumsM0>,
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
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 2>>,
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 4>>>;

template<typename Groups, typename ElementL2Postsum, typename PresumTileShapeA, typename PresumTileShapeB,
         typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using StrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<Groups,
                                                                       ScheduleStrassenGroups1,
                                                                       ProblemShape,
                                                                       ArchTag, OperatorClass,
                                                                       ElementA, LayoutA, cutlass::layout::OriginalLayout,
                                                                       ElementB, LayoutB, cutlass::layout::OriginalLayout,
                                                                       ElementL2Postsum, LayoutC, cutlass::layout::OriginalLayout,
                                                                       ElementAccumulator, ClusterShape,
                                                                       cute::Int<StageCountTypeM0>,
                                                                       PresumTileShapeA, PresumTileShapeB,
                                                                       PresumOpts,
                                                                       AlignmentA, AlignmentB, AlignmentC,
                                                                       ElementD>;

template<typename PresumTileShapeA, typename PresumTileShapeB, typename PresumOpts = cutlass::gemm::device::PresumOpt<>>
using StrassenGemmLevel2UniversalAdapter = cutlass::gemm::device::StrassenGemmLevel2UniversalAdapter<
    StrassenGemmKernels<StrassenGroupsM0, void, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<StrassenGroups<1>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<StrassenGroups<2>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<StrassenGroups<3>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<StrassenGroups<4>, void, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<StrassenGroups<5>, void, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    StrassenGemmKernels<StrassenGroups<6>, ElementD, PresumTileShapeA, PresumTileShapeB, PresumOpts>,
    true>;

template <typename StrassenGemmKernel_>
class HopperF16InterleavedPresumLevel2CooperativePingpongMaxFusionTmaReduceBase {
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

using HopperF16InterleavedPresumLevel2CooperativePingpongMaxFusionTmaReduce_2x256_2x256_OptNo = HopperF16InterleavedPresumLevel2CooperativePingpongMaxFusionTmaReduceBase<StrassenGemmLevel2UniversalAdapter<Shape<_2,_256>, Shape<_2,_256>>>;