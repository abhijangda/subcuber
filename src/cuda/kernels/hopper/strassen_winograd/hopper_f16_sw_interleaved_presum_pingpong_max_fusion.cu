#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cuh"
#include "cuda/kernel_runner_support.cuh"
#include "cutlass/util/reference/device/tensor_fill.h"

extern "C" int fill_hopper_f16_runner_operand(void *ptr, size_t capacity, uint64_t seed) {
  cutlass::reference::device::BlockFillRandomUniform(
      static_cast<cutlass::half_t *>(ptr), capacity, seed,
      cutlass::half_t(4), cutlass::half_t(-4), 0);
  return static_cast<int>(cudaGetLastError());
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNoT<ClusterShape_>::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

template <typename ClusterShape_>
size_t HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNoT<ClusterShape_>::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNoT<ClusterShape_>::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNoT<ClusterShape_>::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNoT<ClusterShape_>::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNoT<ClusterShape_>::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNoT<ClusterShape_>::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000T<ClusterShape_>::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

template <typename ClusterShape_>
size_t HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000T<ClusterShape_>::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000T<ClusterShape_>::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000T<ClusterShape_>::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000T<ClusterShape_>::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000T<ClusterShape_>::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000T<ClusterShape_>::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNoT<ClusterShape_>::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

template <typename ClusterShape_>
size_t HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNoT<ClusterShape_>::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNoT<ClusterShape_>::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNoT<ClusterShape_>::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNoT<ClusterShape_>::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNoT<ClusterShape_>::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNoT<ClusterShape_>::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNoT<ClusterShape_>::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

template <typename ClusterShape_>
size_t HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNoT<ClusterShape_>::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNoT<ClusterShape_>::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNoT<ClusterShape_>::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNoT<ClusterShape_>::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNoT<ClusterShape_>::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

template <typename ClusterShape_>
cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNoT<ClusterShape_>::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_no_2x1, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_no_1x2, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo_1x2)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_0000, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_0000_2x1, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_0000_1x2, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000_1x2)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_4x128_4x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_4x128_4x128_opt_no_2x1, HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_4x128_4x128_opt_no_1x2, HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo_1x2)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_8x128_8x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_8x128_8x128_opt_no_2x1, HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_8x128_8x128_opt_no_1x2, HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo_1x2)
