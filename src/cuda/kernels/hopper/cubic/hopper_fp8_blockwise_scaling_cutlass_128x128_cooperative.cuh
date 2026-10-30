#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

template <typename LayoutD_>
class HopperFp8BlockwiseScalingCutlass128x128Cooperative {
public:
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementD = cutlass::bfloat16_t;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;
  using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, LayoutD_, 8, ElementD, LayoutD_, 8,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementA, cute::tuple<cutlass::layout::RowMajor, LayoutSFA>, 16,
      ElementB, cute::tuple<cutlass::layout::ColumnMajor, LayoutSFB>, 16,
      float, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename CollectiveEpilogue::SharedStorage)>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using Arguments = typename CutlassGemm::Arguments;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;

  static cutlass::Status can_implement(Arguments const &args) {
    return CutlassGemm::can_implement(args);
  }

  static size_t get_workspace_size(Arguments const &args) {
    return CutlassGemm::get_workspace_size(args);
  }

  cutlass::Status initialize(Arguments const &args, int swizzles[7], void *workspace,
                             cudaStream_t stream = nullptr) {
    auto configured = args;
    configured.scheduler.max_swizzle_size = swizzles[0];
    return gemm_.initialize(configured, workspace, stream);
  }

  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0) {
    return gemm_.run(streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  }

private:
  CutlassGemm gemm_;
};