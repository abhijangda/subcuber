#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce.cuh"
#include "cuda/kernel_runner_support.cuh"

#define DEFINE_TMA_REDUCE_METHODS(class_name) \
cutlass::Status class_name::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::can_implement(args); \
} \
size_t class_name::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::get_workspace_size(args); \
} \
cutlass::Status class_name::init(Arguments const &args, int swizzles[7], void *workspace, \
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter); \
} \
cutlass::Status class_name::launch(cudaStream_t *streams, int num_streams, \
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
cutlass::Status class_name::initialize(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return init(args, swizzles, workspace, stream, cuda_adapter); \
} \
cutlass::Status class_name::run(cudaStream_t *streams, int num_streams, \
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
cutlass::Status class_name::operator()(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t *streams, int num_streams, \
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  cutlass::Status status = init(args, swizzles, workspace, \
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter); \
  if (status == cutlass::Status::kSuccess) { \
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
  } \
  return status; \
}

DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo)
DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000)
DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo)
DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo)

#undef DEFINE_TMA_REDUCE_METHODS

STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo)