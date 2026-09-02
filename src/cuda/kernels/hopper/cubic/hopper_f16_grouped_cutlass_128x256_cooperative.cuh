#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

using namespace cute;

class HopperF16GroupedCutlass128x256Cooperative {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
public:
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  using ElementAccumulator = float;
  using TileShape = Shape<_128,_256,_64>;
  using ClusterShape = Shape<_2,_1,_1>;
  using StageCountType = _4;
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator, ElementC, LayoutC *, AlignmentC,
      ElementC, LayoutC *, AlignmentC, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape, StageCountType, KernelSchedule>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Arguments = typename CutlassGemm::Arguments;

  static cutlass::Status can_implement(Arguments const &args);
  static size_t get_workspace_size(Arguments const &args);
  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr);
  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0);

private:
  CutlassGemm gemm_;
#endif
};