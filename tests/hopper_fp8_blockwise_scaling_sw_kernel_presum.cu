#include <cmath>
#include <cstring>

#include "cuda/kernels/hopper/strassen_winograd/hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_tma_reduce.cuh"
#include "cuda/kernel_runner_support.cuh"

using RunFn = int (*)(KernelRunnerBuffers, int, int, int, int, int,
                     cudaStream_t *, int, int, float *);

#define DECLARE_RUNNER(name) \
  extern "C" int name(KernelRunnerBuffers, int, int, int, int, int, \
                     cudaStream_t *, int, int, float *)

DECLARE_RUNNER(run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_4x128_4x128_opt_no_1x2_seq);
DECLARE_RUNNER(run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_4x128_4x128_opt_no_1x2_par);
DECLARE_RUNNER(run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_tma_reduce_4x128_4x128_opt_no_1x2_seq);
DECLARE_RUNNER(run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_tma_reduce_4x128_4x128_opt_no_1x2_par);

#undef DECLARE_RUNNER

static void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

static void check_cutlass(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS error: %s\n", cutlassGetStatusString(status));
    std::exit(EXIT_FAILURE);
  }
}

template <typename Gemm>
static void run_case(char const *name, RunFn runner, int m, int n, int k,
                     cudaStream_t *streams, int num_streams) {
  static_assert(std::is_same_v<typename Gemm::SubmatrixLayoutA,
                               cutlass::layout::OriginalLayout>);
  static_assert(std::is_same_v<typename Gemm::SubmatrixLayoutB,
                               cutlass::layout::OriginalLayout>);
  using Kernel = typename Gemm::GemmKernel;
  using StrideA = typename Gemm::StrideA;
  using StrideB = typename Gemm::StrideB;
  using StrideC = typename Gemm::StrideC;
  using StrideD = typename Gemm::StrideD;
  cudaStream_t launch_streams[7];
  std::copy_n(streams, 7, launch_streams);

  std::vector<ElementA> host_a(size_t(m) * k);
  std::vector<ElementB> host_b(size_t(n) * k);
  cutlass::reference::host::BlockFillRandomUniform(
      host_a.data(), host_a.size(), 1234, 2.0, -2.0);
  cutlass::reference::host::BlockFillRandomUniform(
      host_b.data(), host_b.size(), 5678, 2.0, -2.0);
  cutlass::device_memory::allocation<ElementA> a(host_a.size());
  cutlass::device_memory::allocation<ElementB> b(host_b.size());
  cutlass::device_memory::allocation<ElementD> actual(size_t(m) * n);
  cutlass::device_memory::allocation<ElementD> expected(size_t(m) * n);
  check_cuda(cudaMemcpy(a.get(), host_a.data(), host_a.size() * sizeof(ElementA),
                        cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(b.get(), host_b.data(), host_b.size() * sizeof(ElementB),
                        cudaMemcpyHostToDevice));

  float avg_ms = 0.0f;
  int status = runner({a.get(), b.get(), nullptr, actual.get()},
                      m, n, k, 2, 1, launch_streams, num_streams, 1, &avg_ms);
  if (status != 0) {
    std::fprintf(stderr, "%s runner failed: %d\n", name, status);
    std::exit(EXIT_FAILURE);
  }

  auto shape = cute::make_shape(m, n, k, 1);
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(shape);
  std::vector<float> host_scale_a(cute::size(cute::filter_zeros(layout_sfa)));
  std::vector<float> host_scale_b(cute::size(cute::filter_zeros(layout_sfb)));
  cutlass::reference::host::BlockFillRandomUniform(
      host_scale_a.data(), host_scale_a.size(), 2025, 4.0, -4.0);
  cutlass::reference::host::BlockFillRandomUniform(
      host_scale_b.data(), host_scale_b.size(), 2026, 4.0, -4.0);
  cutlass::device_memory::allocation<float> scale_a(host_scale_a.size());
  cutlass::device_memory::allocation<float> scale_b(host_scale_b.size());
  check_cuda(cudaMemcpy(scale_a.get(), host_scale_a.data(),
                        host_scale_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(scale_b.get(), host_scale_b.data(),
                        host_scale_b.size() * sizeof(float), cudaMemcpyHostToDevice));

  int device_id = 0;
  check_cuda(cudaGetDevice(&device_id));
  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(device_id);
  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k},
      {a.get(), stride_a, b.get(), stride_b,
       scale_a.get(), layout_sfa, scale_b.get(), layout_sfb},
      {{}, nullptr, StrideC{}, expected.get(), stride_d}, hw_info};
  args.scheduler.raster_order = kernel_runner_raster_order<Kernel>();
  check_cutlass(Gemm::can_implement(args));
  cutlass::device_memory::allocation<unsigned char> workspace(Gemm::get_workspace_size(args));
  Gemm gemm;
  int swizzles[] = {4, 4, 2, 2, 2, 2, 2};
  check_cutlass(gemm.initialize(args, swizzles, workspace.get()));
  for (int iteration = 0; iteration < 3; ++iteration) {
    check_cutlass(gemm.run(launch_streams, num_streams));
  }
  check_cuda(cudaDeviceSynchronize());

  std::vector<ElementD> host_actual(size_t(m) * n);
  std::vector<ElementD> host_expected(size_t(m) * n);
  check_cuda(cudaMemcpy(host_actual.data(), actual.get(),
                        host_actual.size() * sizeof(ElementD), cudaMemcpyDeviceToHost));
  check_cuda(cudaMemcpy(host_expected.data(), expected.get(),
                        host_expected.size() * sizeof(ElementD), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < host_actual.size(); ++i) {
    if (!std::isfinite(float(host_actual[i])) ||
        std::memcmp(&host_actual[i], &host_expected[i], sizeof(ElementD)) != 0) {
      std::fprintf(stderr, "%s %dx%dx%d mismatch at %zu: %g vs %g\n",
                   name, m, n, k, i, float(host_actual[i]), float(host_expected[i]));
      std::exit(EXIT_FAILURE);
    }
  }
  std::printf("PASS %s %dx%dx%d streams=%d\n", name, m, n, k, num_streams);
}

int main() {
  cudaStream_t streams[7];
  check_cuda(kernel_runner_create_streams(streams, 7));
  {
    constexpr int num_streams = 3;
    constexpr int m = 16384;
    constexpr int n = 16384;
    constexpr int k = 8192;
    run_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusion_4x128_4x128_OptNo_1x2>(
        "seq", run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_4x128_4x128_opt_no_1x2_seq,
        m, n, k, streams, num_streams);
    run_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusion_4x128_4x128_OptNo_1x2_Par>(
        "par", run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_4x128_4x128_opt_no_1x2_par,
        m, n, k, streams, num_streams);
    run_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusionTmaReduce_4x128_4x128_OptNo_1x2>(
        "tma_reduce_seq", run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_tma_reduce_4x128_4x128_opt_no_1x2_seq,
        m, n, k, streams, num_streams);
    run_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusionTmaReduce_4x128_4x128_OptNo_1x2_Par>(
        "tma_reduce_par", run_hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_tma_reduce_4x128_4x128_opt_no_1x2_par,
        m, n, k, streams, num_streams);
  }
  kernel_runner_destroy_streams(streams, 7);
}
