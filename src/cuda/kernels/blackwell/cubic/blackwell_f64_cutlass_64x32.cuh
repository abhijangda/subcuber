#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

class BlackwellF64Cutlass64x32 {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      double,
      2,
      double,
      double>;

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using ThreadBlockShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<16, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
  static constexpr int AlignmentA = 1;
  static constexpr int AlignmentB = 1;

public:
  using CutlassGemm = cutlass::gemm::device::Gemm<double,
                                                  ColumnMajor,
                                                  double,
                                                  ColumnMajor,
                                                  double,
                                                  ColumnMajor,
                                                  double,
                                                  cutlass::arch::OpClassTensorOp,
                                                  cutlass::arch::Sm80,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  EpilogueOp,
                                                  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                  4, AlignmentA, AlignmentB, true>;

  static constexpr bool kTransposeProblem = true;

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