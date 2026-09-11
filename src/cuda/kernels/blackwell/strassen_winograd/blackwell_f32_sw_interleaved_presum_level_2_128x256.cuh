#pragma once

#include "cuda/kernels/blackwell/strassen_winograd/blackwell_f32_sw_interleaved_presum.cuh"

using BlackwellF32Level2TileShape = cute::Shape<cute::_128, cute::_256, cute::_16>;
using BlackwellF32Level2ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
using BlackwellF32Level2PresumTileShape = cute::Shape<cute::_4, cute::_256>;

constexpr int kBlackwellF32Level2StagesM0M1 = 3;
constexpr int kBlackwellF32Level2StagesM2M6 = 5;

using BlackwellF32Level2AllPresumsKernel = AllPresums<>;

using BlackwellF32Level2StrassenGroupsM0 = StrassenLevel1Groups<
  StrassenPresum<2, 0, BlackwellF32Level2TileShape, BlackwellF32Level2AllPresumsKernel>,
  StrassenLevel1M0Group<
    2, 0, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM0M1, RWMTypes<KeepAccums>,
    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,
    BlackwellF32AllPresumsM0>>;

template <int Level1Index>
using BlackwellF32Level2StrassenGroups = StrassenLevel1Groups<
  StrassenPresum<1, 0, BlackwellF32Level2TileShape, BlackwellF32Level2AllPresumsKernel>,
  StrassenLevel1M0Group<
    1, Level1Index, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM0M1, RWMTypes<>,
    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,
    BlackwellF32AllPresumsM0>,
  StrassenLevel1M1Group<
    1, Level1Index, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM0M1, RWMTypes<>,
    RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>,
                 Expr<Plus<0, MemGlobal, LayoutInterim1D>>>>,
    BlackwellF32AllPresumsM1To6>,
  StrassenLevel1M2Group<
    1, Level1Index, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM2M6, RWMTypes<>,
    RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>,
                 Expr<Plus<0, MemGlobal, LayoutInterim1D>>>>,
    BlackwellF32AllPresumsM1To6>,
  StrassenLevel1M3Group<
    1, Level1Index, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM2M6, RWMTypes<>,
    RWCTypes<CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>,
                 Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
    BlackwellF32AllPresumsM1To6>,
  StrassenLevel1M4Group<
    1, Level1Index, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM2M6, RWMTypes<>,
    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>,
             CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>,
                 Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
    BlackwellF32AllPresumsM1To6>,
  StrassenLevel1M5Group<
    1, Level1Index, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM2M6, RWMTypes<>,
    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>,
                 Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                      Plus<0, MemGlobal, LayoutInterim1D>>>>,
    BlackwellF32AllPresumsM1To6>,
  StrassenLevel1M6Group<
    1, Level1Index, BlackwellF32Level2TileShape, BlackwellF32Level2ClusterShape,
    kBlackwellF32Level2StagesM2M6, RWMTypes<>,
    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>,
                 Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
    BlackwellF32AllPresumsM1To6>>;

using BlackwellF32Level2ScheduleStrassenGroups = ScheduleStrassenGroups<
  ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                   FusedMiGroup<7, 0>>,
  ParallelMiGroups<BlackwellF32KernelSchedule, BlackwellF32EpilogueSchedule, false,
                   FusedMiGroup<7, 1>>,
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

template <typename StrassenGroups>
using BlackwellF32Level2StrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<
  StrassenGroups, BlackwellF32Level2ScheduleStrassenGroups, BlackwellF32ProblemShape,
  cutlass::arch::Sm100, cutlass::arch::OpClassSimt,
  float, BlackwellF32Layout, BlackwellF32SubMatLayout,
  float, BlackwellF32Layout, BlackwellF32SubMatLayout,
  float, BlackwellF32Layout, BlackwellF32SubMatLayout,
  float, BlackwellF32Level2ClusterShape, cute::Int<kBlackwellF32Level2StagesM0M1>,
  BlackwellF32Level2PresumTileShape, BlackwellF32Level2PresumTileShape,
  cutlass::gemm::device::PresumOpt<>, 1, 4, 4>;

using BlackwellF32SWInterleavedPresumLevel2_128x256Adapter =
  cutlass::gemm::device::StrassenGemmLevel2UniversalAdapter<
    BlackwellF32Level2StrassenGemmKernels<BlackwellF32Level2StrassenGroupsM0>,
    BlackwellF32Level2StrassenGemmKernels<BlackwellF32Level2StrassenGroups<1>>,
    BlackwellF32Level2StrassenGemmKernels<BlackwellF32Level2StrassenGroups<2>>,
    BlackwellF32Level2StrassenGemmKernels<BlackwellF32Level2StrassenGroups<3>>,
    BlackwellF32Level2StrassenGemmKernels<BlackwellF32Level2StrassenGroups<4>>,
    BlackwellF32Level2StrassenGemmKernels<BlackwellF32Level2StrassenGroups<5>>,
    BlackwellF32Level2StrassenGemmKernels<BlackwellF32Level2StrassenGroups<6>>>;

class BlackwellF32SWInterleavedPresumLevel2_128x256 {
public:
  using StrassenGemmKernel = BlackwellF32SWInterleavedPresumLevel2_128x256Adapter;
  using GemmKernel = typename StrassenGemmKernel::ChildStrassenGemmM0::GemmKernel;
  using Arguments = typename StrassenGemmKernel::Arguments;
  using RasterOrderOptions = typename GemmKernel::TileScheduler::RasterOrderOptions;

  static cutlass::Status can_implement(Arguments const &args);
  static size_t get_workspace_size(Arguments const &args);

  cutlass::Status initialize(Arguments const &args, int swizzles[7],
                             void *workspace = nullptr, cudaStream_t stream = nullptr);
  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0);

private:
  StrassenGemmKernel gemm_;
};