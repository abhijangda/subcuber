#include "cuda/kernels/hopper/b_presum_kernel/hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion.cuh"
#include "cuda/kernels/hopper/b_presum_kernel/kernel_runner_support.cuh"

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_2x128_2x128_opt_no, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_2x128_2x128_opt_0000, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_4x128_4x128_opt_no, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_8x128_8x128_opt_no, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusion_8x128_8x128_OptNo)
