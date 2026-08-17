#include "cuda/kernels/hopper/cubic/hopper_f16_grouped_cutlass_128x128_pingpong.cuh"
#include "cuda/kernel_runner_support.cuh"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

cutlass::Status HopperF16GroupedCutlass128x128Pingpong::can_implement(Arguments const &args) {
  return CutlassGemm::can_implement(args);
}

size_t HopperF16GroupedCutlass128x128Pingpong::get_workspace_size(Arguments const &args) {
  return CutlassGemm::get_workspace_size(args);
}

cutlass::Status HopperF16GroupedCutlass128x128Pingpong::initialize(
    Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status HopperF16GroupedCutlass128x128Pingpong::run(cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
}

STRASSEN_RUNNER_EXPORT_GROUPED_GEMM_CUTLASS3(
    run_hopper_f16_grouped_cutlass_128x128_pingpong,
    HopperF16GroupedCutlass128x128Pingpong)

#endif