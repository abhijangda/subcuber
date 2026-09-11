#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma_sm100.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cute/tensor.hpp"

class BlackwellF32Cutlass256x128 {
  using Layout = cutlass::layout::RowMajor;
  using TileShape = cute::Shape<cute::_256, cute::_128, cute::_16>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassSimt,
      float, Layout, 1,
      float, Layout, 4,
      float,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCount<3>,
      cutlass::gemm::KernelMultistage>::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassSimt,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      float, Layout, 4,
      float, Layout, 4,
      cutlass::epilogue::EpilogueSimtVectorized>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;

public:
  using CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using Arguments = typename CutlassGemm::Arguments;

  static cutlass::Status can_implement(Arguments const &args);

  static size_t get_workspace_size(Arguments const &args);

  cutlass::Status init(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr);

  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0);

  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr);

  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0);

  cutlass::Status operator()(Arguments const &args, void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0);

private:
  CutlassGemm gemm_;
};