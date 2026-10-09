#include "cuda/kernels/hopper/cubic/hopper_f16_cutlass_128x256_cooperative.cuh"
#include "cuda/kernel_runner_support.cuh"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <typename ClusterShape_>
cutlass::Status HopperF16Cutlass128x256CooperativeT<ClusterShape_>::can_implement(Arguments const &args) {
  return CutlassGemm::can_implement(args);
}

template <typename ClusterShape_>
size_t HopperF16Cutlass128x256CooperativeT<ClusterShape_>::get_workspace_size(Arguments const &args) {
  return CutlassGemm::get_workspace_size(args);
}

template <typename ClusterShape_>
cutlass::Status HopperF16Cutlass128x256CooperativeT<ClusterShape_>::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

template <typename ClusterShape_>
cutlass::Status HopperF16Cutlass128x256CooperativeT<ClusterShape_>::launch(cudaStream_t *streams, int num_streams) {
  cudaStream_t stream = streams == nullptr || num_streams == 0 ? nullptr : streams[0];
  return gemm_.run(stream);
}

template <typename ClusterShape_>
cutlass::Status HopperF16Cutlass128x256CooperativeT<ClusterShape_>::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

template <typename ClusterShape_>
cutlass::Status HopperF16Cutlass128x256CooperativeT<ClusterShape_>::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

template <typename ClusterShape_>
cutlass::Status HopperF16Cutlass128x256CooperativeT<ClusterShape_>::operator()(Arguments const &args, void *workspace,
                                                               cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS3(run_hopper_f16_cutlass_128x256_cooperative, HopperF16Cutlass128x256Cooperative)
STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS3(run_hopper_f16_cutlass_128x256_cooperative_2x1, HopperF16Cutlass128x256Cooperative)
STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS3(run_hopper_f16_cutlass_128x256_cooperative_1x2, HopperF16Cutlass128x256Cooperative_1x2)

#endif
