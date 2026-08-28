#pragma once

#define MY_PRINTF(...) ;//printf(__VA_ARGS__)

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma_sm100.h"
#include "cutlass/epilogue/collective/collective_strassen_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/strassen_linear_combination.h"
#include "cutlass/gemm/collective/collective_strassen_gemm_builder.hpp"
#include "cutlass/gemm/device/strassen_decls.h"
#include "cutlass/gemm/device/strassen_gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/strassen_gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cute/tensor.hpp"

using namespace MmaStrassen;

using BlackwellF32EpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
  float, 1, float, float>;
using BlackwellF32InterimEpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
  float, 4, float, float>;

using BlackwellF32Layout = cutlass::layout::RowMajor;
using BlackwellF32ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
using BlackwellF32KernelSchedule = cutlass::gemm::KernelMultistage;
using BlackwellF32EpilogueSchedule = cutlass::epilogue::EpilogueSimtVectorized;
using BlackwellF32ProblemShape = cute::Shape<int, int, int, int>;
using BlackwellF32PresumOpts = cutlass::gemm::device::PresumOpt<>;//0, 0, 0, 0>;

using BlackwellF32AllPresumsM0 = AllPresums<
  PresumCompute, PresumCompute, PresumCompute, PresumCompute,
  PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
using BlackwellF32AllPresumsM1To6 = AllPresums<
  PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
  PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

template <typename TileShape, typename M0M1TileShape, int Stages,
          typename PresumTileShapeA, typename PresumTileShapeB>
struct BlackwellF32SWInterleavedPresumConfig {
  static constexpr auto LayoutM0 =
    cute::get<0>(M0M1TileShape{}) == cute::get<0>(TileShape{}) ? LayoutInterim1D : LayoutInterim;

  using StrassenGroups = StrassenLevel1Groups<
    StrassenPresum<1, 0, M0M1TileShape, BlackwellF32AllPresumsM0>,
    StrassenLevel1M0Group<1, 0, M0M1TileShape, BlackwellF32ClusterShape, Stages,
      RWMTypes<KeepAccums>,
      RWCTypes<CUW<1, LayoutM0, LayoutNone, Expr<Plus<0>>>>,
      BlackwellF32AllPresumsM0>,
    StrassenLevel1M1Group<1, 0, M0M1TileShape, BlackwellF32ClusterShape, Stages,
      RWMTypes<ContinueAccums>,
      RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
      BlackwellF32AllPresumsM1To6>,
    StrassenLevel1M2Group<1, 0, TileShape, BlackwellF32ClusterShape, Stages,
      RWMTypes<>,
      RWCTypes<CUW<1, LayoutM0, LayoutNone, Expr<Plus<2>>,
                       Expr<Plus<1, MemGlobal, LayoutM0>>>>,
      BlackwellF32AllPresumsM1To6>,
    StrassenLevel1M3Group<1, 0, TileShape, BlackwellF32ClusterShape, Stages,
      RWMTypes<>,
      RWCTypes<CUW<2, LayoutM0, LayoutNone, Expr<Plus<3>>,
                       Expr<Plus<1, MemGlobal, LayoutM0>>>>,
      BlackwellF32AllPresumsM1To6>,
    StrassenLevel1M4Group<1, 0, TileShape, BlackwellF32ClusterShape, Stages,
      RWMTypes<>,
      RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>,
               CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>,
                   Expr<Plus<2, MemGlobal, LayoutM0>>>>,
      BlackwellF32AllPresumsM1To6>,
    StrassenLevel1M5Group<1, 0, TileShape, BlackwellF32ClusterShape, Stages,
      RWMTypes<>,
      RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>,
                   Expr<Plus<1, MemGlobal, LayoutM0>,
                        Plus<0, MemGlobal, LayoutInterim1D>>>>,
      BlackwellF32AllPresumsM1To6>,
    StrassenLevel1M6Group<1, 0, TileShape, BlackwellF32ClusterShape, Stages,
      RWMTypes<>,
      RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>,
                   Expr<Plus<2, MemGlobal, LayoutM0>>>>,
      BlackwellF32AllPresumsM1To6>>;

  using ScheduleStrassenGroups = MmaStrassen::ScheduleStrassenGroups<
    ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                     FusedMiGroup<7, 0, 1>>,
    ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                     FusedMiGroup<7, 2>>,
    ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                     FusedMiGroup<7, 3>>,
    ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                     FusedMiGroup<7, 4>>,
    ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                     FusedMiGroup<7, 5>>,
    ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                     FusedMiGroup<7, 6>>>;

  using Kernels = cutlass::gemm::device::StrassenGemmKernels<
    StrassenGroups, ScheduleStrassenGroups, BlackwellF32ProblemShape,
    cutlass::arch::Sm100, cutlass::arch::OpClassSimt,
    float, BlackwellF32Layout, float, BlackwellF32Layout,
    float, BlackwellF32Layout,
    float, BlackwellF32ClusterShape, cute::Int<Stages>,
    PresumTileShapeA, PresumTileShapeB,
    BlackwellF32PresumOpts, 1, 4, 4>;

  using Adapter = cutlass::gemm::device::StrassenGemmUniversalAdapter<Kernels>;
};

template <typename StrassenGemmKernel_>
class BlackwellF32SWInterleavedPresumBase {
public:
  using StrassenGemmKernel = StrassenGemmKernel_;
  using GemmKernel = typename StrassenGemmKernel::GemmKernel;
  using Arguments = typename StrassenGemmKernel::Arguments;
  using RasterOrderOptions = typename GemmKernel::TileScheduler::RasterOrderOptions;

  static cutlass::Status can_implement(
      Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  static size_t get_workspace_size(
      Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter = nullptr) {
    return StrassenGemmKernel::get_workspace_size(args);
  }

  cutlass::Status initialize(Arguments const &args, int swizzles[7],
                             void *workspace = nullptr, cudaStream_t stream = nullptr,
                             cutlass::CudaHostAdapter *cuda_adapter = nullptr) {
    return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
  }

  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0,
                      cutlass::CudaHostAdapter *cuda_adapter = nullptr,
                      bool launch_with_pdl = false) {
    return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
  }

private:
  StrassenGemmKernel gemm_;
};

template <typename TileShape, typename M0M1TileShape, int Stages,
          typename PresumTileShapeA, typename PresumTileShapeB>
using BlackwellF32SWInterleavedPresum = BlackwellF32SWInterleavedPresumBase<
  typename BlackwellF32SWInterleavedPresumConfig<
    TileShape, M0M1TileShape, Stages, PresumTileShapeA, PresumTileShapeB>::Adapter>;
