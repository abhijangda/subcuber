#include "cuda/kernels/hopper/cubic/hopper_fp8_blockwise_scaling_cutlass_128x128_cooperative.cuh"
#include "cuda/kernel_runner_support.cuh"

using Gemm = HopperFp8BlockwiseScalingCutlass128x128Cooperative<cutlass::layout::ColumnMajor>;
STRASSEN_RUNNER_EXPORT_FP8_BLOCKWISE_CUTLASS3(run_hopper_fp8_blockwise_scaling_cutlass_128x128_cooperative_1x2_TNN, Gemm, Gemm::ScaleConfig)