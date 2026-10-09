#include "cuda/kernels/hopper/b_presum_kernel/hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce.cuh"
#include "cuda/kernels/hopper/b_presum_kernel/kernel_runner_support.cuh"

#define DEFINE_TMA_REDUCE_METHODS(class_name) \
template <typename ClusterShape_> \
cutlass::Status class_name<ClusterShape_>::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::can_implement(args); \
} \
template <typename ClusterShape_> \
size_t class_name<ClusterShape_>::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::get_workspace_size(args); \
} \
template <typename ClusterShape_> \
cutlass::Status class_name<ClusterShape_>::init(Arguments const &args, int swizzles[7], void *workspace, \
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter); \
} \
template <typename ClusterShape_> \
cutlass::Status class_name<ClusterShape_>::launch(cudaStream_t *streams, int num_streams, \
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
template <typename ClusterShape_> \
cutlass::Status class_name<ClusterShape_>::initialize(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return init(args, swizzles, workspace, stream, cuda_adapter); \
} \
template <typename ClusterShape_> \
cutlass::Status class_name<ClusterShape_>::run(cudaStream_t *streams, int num_streams, \
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
template <typename ClusterShape_> \
cutlass::Status class_name<ClusterShape_>::operator()(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t *streams, int num_streams, \
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  cutlass::Status status = init(args, swizzles, workspace, \
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter); \
  if (status == cutlass::Status::kSuccess) { \
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
  } \
  return status; \
}

DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT)
DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T)
DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT)
DEFINE_TMA_REDUCE_METHODS(HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT)

#undef DEFINE_TMA_REDUCE_METHODS

STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_2x1, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_1x2, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_1x2)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_2x1, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_1x2, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_1x2)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_2x1, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_1x2, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_1x2)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_2x1, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNo)
STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(run_hopper_f16_sw_a_presum_interleaved_b_presum_kernel_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_1x2, HopperF16APresumInterleavedBPresumKernelPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_1x2)