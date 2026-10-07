#pragma once

#include "cuda/kernel_runner_support.cuh"

static __global__ void kernel_runner_pack_strassen_f16(
    cutlass::half_t const *source, cutlass::half_t *destination,
    int rows, int columns) {
  size_t matrix_elements = size_t(rows) * size_t(columns);
  size_t half_rows = size_t(rows) / 2;
  size_t half_columns = size_t(columns) / 2;
  for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < matrix_elements; index += size_t(blockDim.x) * gridDim.x) {
    size_t row = index / size_t(columns);
    size_t column = index % size_t(columns);
    size_t quadrant = (row / half_rows) * 2 + column / half_columns;
    size_t packed_index = quadrant * half_rows * half_columns +
                          (row % half_rows) * half_columns + column % half_columns;
    destination[packed_index] = source[index];
  }
}

template <typename Gemm>
int kernel_runner_run_b_presum_kernel(
    KernelRunnerBuffers buffers, int rows, int columns, int reduction,
    int warmup_iterations, int iterations, cudaStream_t *streams,
    int num_streams, int split_k_slices, float *avg_ms) {
  if (rows <= 0 || columns <= 0 || reduction <= 0 ||
      rows % 2 != 0 || columns % 2 != 0 || reduction % 2 != 0) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  size_t a_elements = size_t(rows) * size_t(reduction);
  size_t b_elements = size_t(reduction) * size_t(columns);
  cutlass::device_memory::allocation<cutlass::half_t> packed_a(a_elements);
  cutlass::device_memory::allocation<cutlass::half_t> packed_b(b_elements);
  cudaStream_t stream = streams != nullptr && num_streams > 0 ? streams[0] : nullptr;
  unsigned blocks_a = unsigned(std::min<size_t>((a_elements + 255) / 256, 65535));
  unsigned blocks_b = unsigned(std::min<size_t>((b_elements + 255) / 256, 65535));
  kernel_runner_pack_strassen_f16<<<blocks_a, 256, 0, stream>>>(
      static_cast<cutlass::half_t const *>(buffers.a), packed_a.get(),
      rows, reduction);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  kernel_runner_pack_strassen_f16<<<blocks_b, 256, 0, stream>>>(
      static_cast<cutlass::half_t const *>(buffers.b), packed_b.get(),
      reduction, columns);
  error = cudaGetLastError();
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) return static_cast<int>(error);

  KernelRunnerBuffers packed_buffers{packed_a.get(), packed_b.get(), buffers.c, buffers.d};
    return kernel_runner_run_cutlass3<Gemm>(
      packed_buffers, rows, columns, reduction, warmup_iterations, iterations,
      streams, num_streams, split_k_slices, avg_ms);
}

  #define STRASSEN_RUNNER_EXPORT_B_PRESUM_KERNEL(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int rows, int columns, int reduction, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                   int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_b_presum_kernel<gemm_type>( \
        buffers, rows, columns, reduction, warmup_iterations, iterations, \
      streams, num_streams, split_k_slices, avg_ms); \
  }