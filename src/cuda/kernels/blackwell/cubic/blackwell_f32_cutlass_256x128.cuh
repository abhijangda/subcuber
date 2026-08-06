#pragma once

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#ifndef SPLIT_K
#define SPLIT_K 1
#endif

class BlackwellF32Cutlass256x128 {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    float,
    1,
    float,
    float>;

  using RowMajor = cutlass::layout::RowMajor;
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static constexpr bool splitK = SPLIT_K;

public:
  using CutlassGemm = cutlass::gemm::device::Gemm<float,
                                                  RowMajor,
                                                  float,
                                                  RowMajor,
                                                  float,
                                                  RowMajor,
                                                  float,
                                                  cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm80,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  EpilogueOp,
                                                  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                  3, 1, 1, splitK>;

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