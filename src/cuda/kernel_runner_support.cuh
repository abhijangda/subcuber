#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

struct KernelRunnerBuffers {
  void const *a;
  void const *b;
  void const *c;
  void *d;
};

constexpr int kKernelRunnerMaxStreams = 49;

inline cudaError_t kernel_runner_create_streams(cudaStream_t *streams, int num_streams) {
  for (int i = 0; i < num_streams; ++i) {
    cudaError_t err = cudaStreamCreate(&streams[i]);
    if (err != cudaSuccess) {
      for (int j = 0; j < i; ++j) {
        cudaStreamDestroy(streams[j]);
      }
      return err;
    }
  }
  return cudaSuccess;
}

inline void kernel_runner_destroy_streams(cudaStream_t *streams, int num_streams) {
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}

inline int kernel_runner_status_to_error(cutlass::Status status) {
  return status == cutlass::Status::kSuccess ? 0 : static_cast<int>(status);
}

template <typename Gemm>
int kernel_runner_run_cutlass2(KernelRunnerBuffers buffers, int m, int n, int k,
                               int warmup_iterations, int iterations,
                               cudaStream_t *streams, int num_streams,
                               int split_k_slices, float *avg_ms) {
  using Kernel = typename Gemm::StrassenGemmKernel;
  using ElementA = typename Kernel::ElementA;
  using ElementB = typename Kernel::ElementB;
  using ElementC = typename Kernel::ElementC;
  using LayoutA = typename Kernel::LayoutA;
  using LayoutB = typename Kernel::LayoutB;
  using LayoutC = typename Kernel::LayoutC;
  using Arguments = typename Gemm::Arguments;

  if (m < 2 * Kernel::ThreadblockShape::kM ||
      n < 2 * Kernel::ThreadblockShape::kN ||
      k < 2 * Kernel::ThreadblockShape::kK) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  if constexpr (requires { typename Kernel::ChildGemmM0; }) {
    if (m % (4 * Kernel::ThreadblockShape::kM) != 0 ||
        n % (4 * Kernel::ThreadblockShape::kN) != 0 ||
        k % (4 * Kernel::ThreadblockShape::kK) != 0) {
      return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
    }
  }

  Arguments args(
      cutlass::gemm::GemmCoord(m, n, k),
      {reinterpret_cast<ElementA const *>(buffers.a), LayoutA::packed({m, k})},
      {reinterpret_cast<ElementB const *>(buffers.b), LayoutB::packed({k, n})},
      {reinterpret_cast<ElementC const *>(buffers.c), LayoutC::packed({m, n})},
      {reinterpret_cast<ElementC *>(buffers.d), LayoutC::packed({m, n})},
      {1.0f, 0.0f},
      split_k_slices);

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  Gemm gemm;
  status = gemm.initialize(args, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_gemm_cutlass2(KernelRunnerBuffers buffers, int m, int n, int k,
                                    int warmup_iterations, int iterations,
                                    cudaStream_t *streams, int num_streams,
                                    int split_k_slices, float *avg_ms) {
  using Kernel = typename Gemm::CutlassGemm;
  using ElementA = typename Kernel::ElementA;
  using ElementB = typename Kernel::ElementB;
  using ElementC = typename Kernel::ElementC;
  using LayoutA = typename Kernel::LayoutA;
  using LayoutB = typename Kernel::LayoutB;
  using LayoutC = typename Kernel::LayoutC;
  using Arguments = typename Gemm::Arguments;

  int problem_m = m;
  int problem_n = n;
  void const *operand_a = buffers.a;
  void const *operand_b = buffers.b;
  if constexpr (requires { Gemm::kTransposeProblem; }) {
    if constexpr (Gemm::kTransposeProblem) {
      std::swap(problem_m, problem_n);
      std::swap(operand_a, operand_b);
    }
  }
  Arguments args(
      cutlass::gemm::GemmCoord(problem_m, problem_n, k),
      {reinterpret_cast<ElementA const *>(operand_a), LayoutA::packed({problem_m, k})},
      {reinterpret_cast<ElementB const *>(operand_b), LayoutB::packed({k, problem_n})},
      {reinterpret_cast<ElementC const *>(buffers.c), LayoutC::packed({problem_m, problem_n})},
      {reinterpret_cast<ElementC *>(buffers.d), LayoutC::packed({problem_m, problem_n})},
      {1.0f, 0.0f},
      split_k_slices);

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  Gemm gemm;
  status = gemm.initialize(args, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_cutlass3(KernelRunnerBuffers buffers, int m, int n, int k,
                               int warmup_iterations, int iterations,
                               cudaStream_t *streams, int num_streams, int,
                               float *avg_ms) {
  using Kernel = typename Gemm::GemmKernel;
  using ElementA = typename Gemm::StrassenGemmKernel::ElementA;
  using ElementB = typename Gemm::StrassenGemmKernel::ElementB;
  using ElementC = typename Gemm::StrassenGemmKernel::ElementC;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using Arguments = typename Gemm::Arguments;
  using RasterOrderOptions = typename Gemm::RasterOrderOptions;

  if (m < 2 * int(cute::size<0>(typename Kernel::TileShape{})) ||
      n < 2 * int(cute::size<1>(typename Kernel::TileShape{})) ||
      k < 2 * int(cute::size<2>(typename Kernel::TileShape{}))) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  auto problem_shape = [&] {
    if constexpr (cute::rank(typename Kernel::ProblemShape{}) == 4) {
      return typename Kernel::ProblemShape{m, n, k, 1};
    } else {
      return typename Kernel::ProblemShape{m, n, k};
    }
  }();

  Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {reinterpret_cast<ElementA const *>(buffers.a), stride_a,
       reinterpret_cast<ElementB const *>(buffers.b), stride_b},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC const *>(buffers.c), stride_c,
       reinterpret_cast<ElementC *>(buffers.d), stride_d},
      hw_info);

  args.scheduler.raster_order = RasterOrderOptions::AlongN;

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  int swizzles[7] = {2, 2, 1, 1, 1, 1, 1};
  Gemm gemm;
  status = gemm.initialize(args, swizzles, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_moe_cutlass3(KernelRunnerBuffers buffers, int m, int n, int k,
                                   int warmup_iterations, int iterations,
                                   cudaStream_t *streams, int num_streams,
                                   int expert_count, float *avg_ms) {
  using Kernel = typename Gemm::GemmKernel;
  using ProblemShape = typename Kernel::ProblemShape::UnderlyingProblemShape;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
  using Arguments = typename Gemm::Arguments;

  if (expert_count <= 0 ||
      m < 2 * int(cute::size<0>(typename Kernel::TileShape{})) ||
      n < 2 * int(cute::size<1>(typename Kernel::TileShape{})) ||
      k < 2 * int(cute::size<2>(typename Kernel::TileShape{}))) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  std::vector<ProblemShape> problem_shapes_host(expert_count, ProblemShape{m, n, k});
  std::vector<StrideA> stride_a_host(expert_count,
      cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
  std::vector<StrideB> stride_b_host(expert_count,
      cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
  std::vector<StrideD> stride_d_host(expert_count,
      cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));
  std::vector<uint64_t> batch_a_host(expert_count);
  std::vector<uint64_t> batch_b_host(expert_count);
  std::vector<uint64_t> batch_d_host(expert_count);
  for (int expert = 0; expert < expert_count; ++expert) {
    batch_a_host[expert] = uint64_t(expert) * uint64_t(m);
    batch_b_host[expert] = uint64_t(expert) * uint64_t(k);
    batch_d_host[expert] = uint64_t(expert) * uint64_t(m);
  }

  cutlass::device_memory::allocation<ProblemShape> problem_shapes(expert_count);
  cutlass::device_memory::allocation<StrideA> stride_a(expert_count);
  cutlass::device_memory::allocation<StrideB> stride_b(expert_count);
  cutlass::device_memory::allocation<StrideD> stride_d(expert_count);
  cutlass::device_memory::allocation<uint64_t> batch_a(expert_count);
  cutlass::device_memory::allocation<uint64_t> batch_b(expert_count);
  cutlass::device_memory::allocation<uint64_t> batch_d(expert_count);
  cutlass::device_memory::allocation<ElementB> expert_weights(
      size_t(expert_count) * size_t(k) * size_t(n));

  auto copy_host_to_device = [](auto *dst, auto const *src, size_t count) {
    return cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyHostToDevice);
  };
  cudaError_t err = copy_host_to_device(problem_shapes.get(), problem_shapes_host.data(), expert_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_a.get(), stride_a_host.data(), expert_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_b.get(), stride_b_host.data(), expert_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_d.get(), stride_d_host.data(), expert_count);
  if (err == cudaSuccess) err = copy_host_to_device(batch_a.get(), batch_a_host.data(), expert_count);
  if (err == cudaSuccess) err = copy_host_to_device(batch_b.get(), batch_b_host.data(), expert_count);
  if (err == cudaSuccess) err = copy_host_to_device(batch_d.get(), batch_d_host.data(), expert_count);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  size_t weight_elements = size_t(k) * size_t(n);
  for (int expert = 0; expert < expert_count; ++expert) {
    err = cudaMemcpy(expert_weights.get() + size_t(expert) * weight_elements,
                     buffers.b, weight_elements * sizeof(ElementB), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      return static_cast<int>(err);
    }
  }

  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(device_id);
  Arguments args(
      cutlass::gemm::GemmUniversalMode::kMoE,
      {expert_count, problem_shapes.get(), problem_shapes_host.data()},
      {reinterpret_cast<ElementA const *>(buffers.a), stride_a.get(),
       expert_weights.get(), stride_b.get(), batch_a.get(), batch_b.get()},
      {{1.0f, 0.0f}, nullptr, nullptr, reinterpret_cast<ElementD *>(buffers.d),
       stride_d.get(), nullptr, batch_d.get()},
      hw_info);
  args.scheduler.raster_order = decltype(args.scheduler.raster_order)::AlongN;

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);
  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  int swizzles[7] = {2, 2, 1, 1, 1, 1, 1};
  Gemm gemm;
  status = gemm.initialize(args, swizzles, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) return static_cast<int>(err);
    }
    if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return static_cast<int>(err);

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) return static_cast<int>(err);
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }
  err = cudaEventRecord(start);
  if (err == cudaSuccess) {
    for (int i = 0; i < iterations; ++i) {
      status = gemm.run(streams, num_streams);
      if (num_streams > 1) err = cudaDeviceSynchronize();
      if (err != cudaSuccess || status != cutlass::Status::kSuccess) break;
    }
  }
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) err = cudaEventRecord(stop);
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) err = cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) {
    err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (err != cudaSuccess) return static_cast<int>(err);
  if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  *avg_ms = elapsed_ms / float(iterations);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_gemm_cutlass3(KernelRunnerBuffers buffers, int m, int n, int k,
                                    int warmup_iterations, int iterations,
                                    cudaStream_t *streams, int num_streams, int,
                                    float *avg_ms) {
  using Adapter = typename Gemm::CutlassGemm;
  using Kernel = typename Adapter::GemmKernel;
  using ElementA = typename Adapter::ElementA;
  using ElementB = typename Adapter::ElementB;
  using ElementC = typename Adapter::ElementC;
  using ElementD = typename Adapter::ElementD;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using Arguments = typename Gemm::Arguments;
  using RasterOrderOptions = typename Kernel::TileScheduler::RasterOrderOptions;

  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device_id;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  typename Kernel::ProblemShape problem_shape;
  if constexpr (cute::rank(typename Kernel::ProblemShape{}) == 4) {
    problem_shape = {m, n, k, 1};
  } else {
    problem_shape = {m, n, k};
  }

  Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {reinterpret_cast<ElementA const *>(buffers.a), stride_a,
       reinterpret_cast<ElementB const *>(buffers.b), stride_b},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC const *>(buffers.c), stride_c,
       reinterpret_cast<ElementD *>(buffers.d), stride_d},
      hw_info);

  args.scheduler.raster_order = RasterOrderOptions::AlongN;
  args.scheduler.max_swizzle_size = 1;

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  Gemm gemm;
  status = gemm.initialize(args, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_grouped_cutlass3(KernelRunnerBuffers buffers, int m, int n, int k,
                                       int warmup_iterations, int iterations,
                                       cudaStream_t *streams, int num_streams,
                                       int group_count, float *avg_ms) {
  using Kernel = typename Gemm::GemmKernel;
  using ProblemShape = typename Kernel::ProblemShape::UnderlyingProblemShape;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementD = typename Gemm::ElementD;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideC = typename Kernel::InternalStrideC;
  using StrideD = typename Kernel::InternalStrideD;
  using Arguments = typename Gemm::Arguments;

  if (group_count <= 0) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  std::vector<ProblemShape> problem_shapes_host(group_count, ProblemShape{m, n, k});
  std::vector<StrideA> stride_a_host(group_count,
      cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
  std::vector<StrideB> stride_b_host(group_count,
      cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
  std::vector<StrideC> stride_c_host(group_count,
      cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
  std::vector<StrideD> stride_d_host(group_count,
      cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));

  auto const *input = reinterpret_cast<ElementA const *>(buffers.a);
  auto *output = reinterpret_cast<ElementD *>(buffers.d);
  cutlass::device_memory::allocation<ElementB> group_weights(
      size_t(group_count) * size_t(k) * size_t(n));
  std::vector<ElementA const *> ptr_a_host(group_count);
  std::vector<ElementB const *> ptr_b_host(group_count);
  std::vector<ElementC const *> ptr_c_host(group_count, nullptr);
  std::vector<ElementD *> ptr_d_host(group_count);
  size_t weight_elements = size_t(k) * size_t(n);
  for (int group = 0; group < group_count; ++group) {
    ptr_a_host[group] = input + size_t(group) * size_t(m) * size_t(k);
    ptr_b_host[group] = group_weights.get() + size_t(group) * weight_elements;
    ptr_d_host[group] = output + size_t(group) * size_t(m) * size_t(n);
    cudaError_t err = cudaMemcpy(group_weights.get() + size_t(group) * weight_elements,
                                 buffers.b, weight_elements * sizeof(ElementB),
                                 cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return static_cast<int>(err);
  }

  cutlass::device_memory::allocation<ProblemShape> problem_shapes(group_count);
  cutlass::device_memory::allocation<StrideA> stride_a(group_count);
  cutlass::device_memory::allocation<StrideB> stride_b(group_count);
  cutlass::device_memory::allocation<StrideC> stride_c(group_count);
  cutlass::device_memory::allocation<StrideD> stride_d(group_count);
  cutlass::device_memory::allocation<ElementA const *> ptr_a(group_count);
  cutlass::device_memory::allocation<ElementB const *> ptr_b(group_count);
  cutlass::device_memory::allocation<ElementC const *> ptr_c(group_count);
  cutlass::device_memory::allocation<ElementD *> ptr_d(group_count);
  auto copy_host_to_device = [](auto *dst, auto const *src, size_t count) {
    return cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyHostToDevice);
  };
  cudaError_t err = copy_host_to_device(problem_shapes.get(), problem_shapes_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_a.get(), stride_a_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_b.get(), stride_b_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_c.get(), stride_c_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_d.get(), stride_d_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_a.get(), ptr_a_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_b.get(), ptr_b_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_c.get(), ptr_c_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_d.get(), ptr_d_host.data(), group_count);
  if (err != cudaSuccess) return static_cast<int>(err);

  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(device_id);
  Arguments args(
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_shapes.get(), problem_shapes_host.data()},
      {ptr_a.get(), stride_a.get(), ptr_b.get(), stride_b.get()},
      {{1.0f, 0.0f}, ptr_c.get(), stride_c.get(), ptr_d.get(), stride_d.get()},
      hw_info);
  args.scheduler.raster_order = decltype(args.scheduler.raster_order)::AlongN;

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  cutlass::device_memory::allocation<unsigned char> workspace(Gemm::get_workspace_size(args));
  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  int swizzles[7] = {2, 2, 1, 1, 1, 1, 1};
  Gemm gemm;
  status = gemm.initialize(args, swizzles, workspace.get());
  if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return static_cast<int>(err);
    if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return static_cast<int>(err);

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) return static_cast<int>(err);
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }
  err = cudaEventRecord(start);
  if (err == cudaSuccess) {
    for (int i = 0; i < iterations; ++i) {
      status = gemm.run(streams, num_streams);
      if (num_streams > 1) err = cudaDeviceSynchronize();
      if (err != cudaSuccess || status != cutlass::Status::kSuccess) break;
    }
  }
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) err = cudaEventRecord(stop);
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) err = cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) {
    err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (err != cudaSuccess) return static_cast<int>(err);
  if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  *avg_ms = elapsed_ms / float(iterations);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_grouped_gemm_cutlass3(KernelRunnerBuffers buffers, int m, int n, int k,
                                            int warmup_iterations, int iterations,
                                            cudaStream_t *streams, int num_streams,
                                            int group_count, float *avg_ms) {
  using Adapter = typename Gemm::CutlassGemm;
  using Kernel = typename Adapter::GemmKernel;
  using ProblemShape = typename Kernel::ProblemShape::UnderlyingProblemShape;
  using ElementA = typename Adapter::ElementA;
  using ElementB = typename Adapter::ElementB;
  using ElementC = typename Adapter::ElementC;
  using ElementD = typename Adapter::ElementD;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideC = typename Kernel::InternalStrideC;
  using StrideD = typename Kernel::InternalStrideD;
  using Arguments = typename Gemm::Arguments;

  if (group_count <= 0) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  std::vector<ProblemShape> problem_shapes_host(group_count, ProblemShape{m, n, k});
  std::vector<StrideA> stride_a_host(group_count,
      cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
  std::vector<StrideB> stride_b_host(group_count,
      cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
  std::vector<StrideC> stride_c_host(group_count,
      cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
  std::vector<StrideD> stride_d_host(group_count,
      cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));

  auto const *input = reinterpret_cast<ElementA const *>(buffers.a);
  auto *output = reinterpret_cast<ElementD *>(buffers.d);
  cutlass::device_memory::allocation<ElementB> group_weights(
      size_t(group_count) * size_t(k) * size_t(n));
  std::vector<ElementA const *> ptr_a_host(group_count);
  std::vector<ElementB const *> ptr_b_host(group_count);
  std::vector<ElementC const *> ptr_c_host(group_count, nullptr);
  std::vector<ElementD *> ptr_d_host(group_count);
  size_t weight_elements = size_t(k) * size_t(n);
  for (int group = 0; group < group_count; ++group) {
    ptr_a_host[group] = input + size_t(group) * size_t(m) * size_t(k);
    ptr_b_host[group] = group_weights.get() + size_t(group) * weight_elements;
    ptr_d_host[group] = output + size_t(group) * size_t(m) * size_t(n);
    cudaError_t err = cudaMemcpy(group_weights.get() + size_t(group) * weight_elements, buffers.b,
                                 weight_elements * sizeof(ElementB), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return static_cast<int>(err);
  }

  cutlass::device_memory::allocation<ProblemShape> problem_shapes(group_count);
  cutlass::device_memory::allocation<StrideA> stride_a(group_count);
  cutlass::device_memory::allocation<StrideB> stride_b(group_count);
  cutlass::device_memory::allocation<StrideC> stride_c(group_count);
  cutlass::device_memory::allocation<StrideD> stride_d(group_count);
  cutlass::device_memory::allocation<ElementA const *> ptr_a(group_count);
  cutlass::device_memory::allocation<ElementB const *> ptr_b(group_count);
  cutlass::device_memory::allocation<ElementC const *> ptr_c(group_count);
  cutlass::device_memory::allocation<ElementD *> ptr_d(group_count);

  auto copy_host_to_device = [](auto *dst, auto const *src, size_t count) {
    return cudaMemcpy(dst, src, count * sizeof(*src), cudaMemcpyHostToDevice);
  };
  cudaError_t err = copy_host_to_device(problem_shapes.get(), problem_shapes_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_a.get(), stride_a_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_b.get(), stride_b_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_c.get(), stride_c_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(stride_d.get(), stride_d_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_a.get(), ptr_a_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_b.get(), ptr_b_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_c.get(), ptr_c_host.data(), group_count);
  if (err == cudaSuccess) err = copy_host_to_device(ptr_d.get(), ptr_d_host.data(), group_count);
  if (err != cudaSuccess) return static_cast<int>(err);

  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(device_id);
  Arguments args(
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {group_count, problem_shapes.get(), problem_shapes_host.data()},
      {ptr_a.get(), stride_a.get(), ptr_b.get(), stride_b.get()},
      {{1.0f, 0.0f}, ptr_c.get(), stride_c.get(), ptr_d.get(), stride_d.get()},
      hw_info);
  args.scheduler.raster_order = decltype(args.scheduler.raster_order)::AlongN;
  args.scheduler.max_swizzle_size = 1;

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  cutlass::device_memory::allocation<unsigned char> workspace(Gemm::get_workspace_size(args));
  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  Gemm gemm;
  status = gemm.initialize(args, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return static_cast<int>(err);

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) return static_cast<int>(err);
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }
  err = cudaEventRecord(start);
  if (err == cudaSuccess) {
    for (int i = 0; i < iterations; ++i) {
      status = gemm.run(streams, num_streams);
      if (status != cutlass::Status::kSuccess) break;
    }
  }
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) err = cudaEventRecord(stop);
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) err = cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  if (err == cudaSuccess && status == cutlass::Status::kSuccess) {
    err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (err != cudaSuccess) return static_cast<int>(err);
  if (status != cutlass::Status::kSuccess) return kernel_runner_status_to_error(status);
  *avg_ms = elapsed_ms / float(iterations);
  return 0;
}

#define STRASSEN_RUNNER_EXPORT_CUTLASS2(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_cutlass2<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                iterations, streams, num_streams, split_k_slices, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS2(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_gemm_cutlass2<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                     iterations, streams, num_streams, split_k_slices, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_CUTLASS3(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_cutlass3<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                iterations, streams, num_streams, split_k_slices, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_MOE_CUTLASS3(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int expert_count, float *avg_ms) { \
    return kernel_runner_run_moe_cutlass3<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                    iterations, streams, num_streams, expert_count, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_GROUPED_CUTLASS3(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int group_count, float *avg_ms) { \
    return kernel_runner_run_grouped_cutlass3<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                        iterations, streams, num_streams, group_count, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_GROUPED_GEMM_CUTLASS3(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int group_count, float *avg_ms) { \
    return kernel_runner_run_grouped_gemm_cutlass3<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                             iterations, streams, num_streams, group_count, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS3(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_gemm_cutlass3<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                     iterations, streams, num_streams, split_k_slices, avg_ms); \
  }
