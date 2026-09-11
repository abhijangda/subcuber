#include "cuda/kernels/blackwell/strassen_winograd/blackwell_f32_sw_interleaved_presum_level_2_128x256.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status BlackwellF32SWInterleavedPresumLevel2_128x256::can_implement(
    Arguments const &args) {
  return StrassenGemmKernel::can_implement(args);
}

size_t BlackwellF32SWInterleavedPresumLevel2_128x256::get_workspace_size(
    Arguments const &args) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status BlackwellF32SWInterleavedPresumLevel2_128x256::initialize(
    Arguments const &args, int swizzles[7], void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, swizzles, workspace, stream);
}

cutlass::Status BlackwellF32SWInterleavedPresumLevel2_128x256::run(
    cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams, num_streams);
}

STRASSEN_RUNNER_EXPORT_CUTLASS3(
  run_blackwell_f32_sw_interleaved_presum_level_2_128x256,
  BlackwellF32SWInterleavedPresumLevel2_128x256)