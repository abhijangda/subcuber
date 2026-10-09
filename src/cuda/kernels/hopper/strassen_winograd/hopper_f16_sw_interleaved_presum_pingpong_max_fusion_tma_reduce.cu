#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce.cuh"
#include "cuda/kernel_runner_support.cuh"

#define DEFINE_TMA_REDUCE_METHODS(class_name) \
template <typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<ClusterShape_, Schedule>::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::can_implement(args); \
} \
template <typename ClusterShape_, typename Schedule> \
size_t class_name<ClusterShape_, Schedule>::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) { \
  return StrassenGemmKernel::get_workspace_size(args); \
} \
template <typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<ClusterShape_, Schedule>::init(Arguments const &args, int swizzles[7], void *workspace, \
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter); \
} \
template <typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<ClusterShape_, Schedule>::launch(cudaStream_t *streams, int num_streams, \
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
template <typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<ClusterShape_, Schedule>::initialize(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) { \
  return init(args, swizzles, workspace, stream, cuda_adapter); \
} \
template <typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<ClusterShape_, Schedule>::run(cudaStream_t *streams, int num_streams, \
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
} \
template <typename ClusterShape_, typename Schedule> \
cutlass::Status class_name<ClusterShape_, Schedule>::operator()(Arguments const &args, int swizzles[7], void *workspace, \
                                    cudaStream_t *streams, int num_streams, \
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) { \
  cutlass::Status status = init(args, swizzles, workspace, \
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter); \
  if (status == cutlass::Status::kSuccess) { \
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl); \
  } \
  return status; \
}

DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNoT)
DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000T)
DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNoT)
DEFINE_TMA_REDUCE_METHODS(HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNoT)

#undef DEFINE_TMA_REDUCE_METHODS

STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_2x1_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_2x1_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_1x2_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_1x2)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_no_1x2_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_OptNo_1x2_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_2x1_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_2x1_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_1x2_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_1x2)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_2x128_2x128_opt_0000_1x2_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_2x128_2x128_Opt_0000_1x2_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_2x1_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_2x1_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_1x2_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_1x2)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_4x128_4x128_opt_no_1x2_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_4x128_4x128_OptNo_1x2_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_2x1_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_2x1_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_Par)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_1x2_seq, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_1x2)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_tma_reduce_8x128_8x128_opt_no_1x2_par, HopperF16InterleavedPresumPingpongMaxFusionTmaReduce_8x128_8x128_OptNo_1x2_Par)