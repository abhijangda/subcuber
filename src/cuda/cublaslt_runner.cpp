#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include "cutlass/numeric_types.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cuda/kernel_runner_support.cuh"

template <typename Element>
struct CublasLtType;

template <>
struct CublasLtType<float> {
  using Scale = float;
  static constexpr cudaDataType_t data_type = CUDA_R_32F;
  static constexpr cudaDataType_t output_type = CUDA_R_32F;
  static constexpr cudaDataType_t scale_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};

template <>
struct CublasLtType<double> {
  using Scale = double;
  static constexpr cudaDataType_t data_type = CUDA_R_64F;
  static constexpr cudaDataType_t output_type = CUDA_R_64F;
  static constexpr cudaDataType_t scale_type = CUDA_R_64F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
};

template <>
struct CublasLtType<cutlass::half_t> {
  using Scale = float;
  static constexpr cudaDataType_t data_type = CUDA_R_16F;
  static constexpr cudaDataType_t output_type = CUDA_R_16F;
  static constexpr cudaDataType_t scale_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
};

template <>
struct CublasLtType<cutlass::bfloat16_t> {
  using Scale = float;
  static constexpr cudaDataType_t data_type = CUDA_R_16BF;
  static constexpr cudaDataType_t output_type = CUDA_R_16BF;
  static constexpr cudaDataType_t scale_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};

template <>
struct CublasLtType<cutlass::float_e4m3_t> {
  using Scale = float;
  static constexpr cudaDataType_t data_type = CUDA_R_8F_E4M3;
  static constexpr cudaDataType_t output_type = CUDA_R_16BF;
  static constexpr cudaDataType_t scale_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};

static int cublaslt_status_to_error(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + static_cast<int>(status);
}

template <typename Element, bool ColumnMajorB = false,
          bool ColumnMajorD = std::is_same_v<Element, cutlass::float_e4m3_t>>
int run_cublaslt_gemm(KernelRunnerBuffers buffers, int m, int n, int k,
                      int warmup_iterations, int iterations, cudaStream_t *streams,
                      int num_streams, int, float *avg_ms) {
  using Type = CublasLtType<Element>;

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cudaStream_t stream = streams == nullptr || num_streams == 0 ? nullptr : streams[0];
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  void *workspace = nullptr;
  float *scale_a = nullptr;
  float *scale_b = nullptr;
  constexpr size_t kWorkspaceSize = 32ULL * 1024ULL * 1024ULL;

  auto cleanup = [&]() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (scale_b != nullptr) cudaFree(scale_b);
    if (scale_a != nullptr) cudaFree(scale_a);
    if (workspace != nullptr) cudaFree(workspace);
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_d != nullptr) cublasLtMatrixLayoutDestroy(layout_d);
    if (layout_c != nullptr) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_b != nullptr) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_a != nullptr) cublasLtMatrixLayoutDestroy(layout_a);
    if (matmul_desc != nullptr) cublasLtMatmulDescDestroy(matmul_desc);
    if (handle != nullptr) cublasLtDestroy(handle);
  };

  cublasStatus_t cublas_status = cublasLtCreate(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cublas_status = cublasLtMatmulDescCreate(&matmul_desc, Type::compute_type, Type::scale_type);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

    cublasOperation_t op_a = std::is_same_v<Element, cutlass::float_e4m3_t>
                   ? CUBLAS_OP_T
                   : CUBLAS_OP_N;
    cublasOperation_t op_b = CUBLAS_OP_N;
  cublas_status = cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cudaError_t cuda_status = cudaSuccess;
  if constexpr (std::is_same_v<Element, cutlass::float_e4m3_t>) {
    cublasLtMatmulMatrixScale_t scale_mode_a =
      CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
    cublasLtMatmulMatrixScale_t scale_mode_b =
      CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
    cublas_status = cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a,
      sizeof(scale_mode_a));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublaslt_status_to_error(cublas_status);
    }
    cublas_status = cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b,
      sizeof(scale_mode_b));
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublaslt_status_to_error(cublas_status);
    }

    size_t k_blocks = (size_t(k) + 127) / 128;
    size_t n_blocks = (size_t(n) + 127) / 128;
    size_t scale_a_count = size_t(m) * k_blocks;
    size_t scale_b_inner = ((k_blocks + 3) / 4) * 4;
    size_t scale_b_count = scale_b_inner * n_blocks;
    std::vector<float> host_scale_a(scale_a_count);
    std::vector<float> host_scale_b(scale_b_count);
    cutlass::reference::host::BlockFillRandomUniform(
        host_scale_a.data(), host_scale_a.size(), 2025, 4.0, -4.0);
    cutlass::reference::host::BlockFillRandomUniform(
        host_scale_b.data(), host_scale_b.size(), 2026, 4.0, -4.0);
    cuda_status = cudaMalloc(&scale_a, scale_a_count * sizeof(float));
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMalloc(&scale_b, scale_b_count * sizeof(float));
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpy(scale_a, host_scale_a.data(),
                               scale_a_count * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaMemcpy(scale_b, host_scale_b.data(),
                               scale_b_count * sizeof(float),
                               cudaMemcpyHostToDevice);
    }
    if (cuda_status != cudaSuccess) {
      cleanup();
      return static_cast<int>(cuda_status);
    }
    cublas_status = cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a,
        sizeof(scale_a));
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
      cublas_status = cublasLtMatmulDescSetAttribute(
          matmul_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b,
          sizeof(scale_b));
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublaslt_status_to_error(cublas_status);
    }
  }

  int layout_a_rows = std::is_same_v<Element, cutlass::float_e4m3_t> ? k : m;
  int layout_a_cols = std::is_same_v<Element, cutlass::float_e4m3_t> ? m : k;
  int layout_a_ld = k;
  int layout_b_ld = std::is_same_v<Element, cutlass::float_e4m3_t> || ColumnMajorB ? k : n;
  int layout_d_ld = ColumnMajorD ? m : n;
  cublas_status = cublasLtMatrixLayoutCreate(
      &layout_a, Type::data_type, layout_a_rows, layout_a_cols, layout_a_ld);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutCreate(
      &layout_b, Type::data_type, k, n, layout_b_ld);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutCreate(
      &layout_c, Type::output_type, m, n, layout_d_ld);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutCreate(
      &layout_d, Type::output_type, m, n, layout_d_ld);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  if constexpr (!std::is_same_v<Element, cutlass::float_e4m3_t>) {
    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    cublasLtOrder_t order_b = ColumnMajorB ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
    cublas_status = cublasLtMatrixLayoutSetAttribute(
        layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
      cublas_status = cublasLtMatrixLayoutSetAttribute(
          layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_b, sizeof(order_b));
    }
  }
  cublasLtOrder_t order_d = ColumnMajorD ? CUBLASLT_ORDER_COL : CUBLASLT_ORDER_ROW;
  if (cublas_status == CUBLAS_STATUS_SUCCESS) {
    cublas_status = cublasLtMatrixLayoutSetAttribute(
        layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_d, sizeof(order_d));
  }
  if (cublas_status == CUBLAS_STATUS_SUCCESS) {
    cublas_status = cublasLtMatrixLayoutSetAttribute(
        layout_d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_d, sizeof(order_d));
  }
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cublas_status = cublasLtMatmulPreferenceCreate(&preference);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceSize,
      sizeof(kWorkspaceSize));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cuda_status = cudaMalloc(&workspace, kWorkspaceSize);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  cublasLtMatmulHeuristicResult_t heuristic_result{};
  int returned_results = 0;
  cublas_status = cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, layout_a,
                                                 layout_b, layout_c, layout_d,
                                                 preference, 1, &heuristic_result,
                                                 &returned_results);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  if (returned_results == 0) {
    cleanup();
    return cublaslt_status_to_error(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  using Scale = typename Type::Scale;
  Scale alpha = Scale(1.0);
  Scale beta = Scale(0.0);
  auto launch = [&]() {
    return cublasLtMatmul(handle, matmul_desc, &alpha, buffers.a, layout_a,
                          buffers.b, layout_b, &beta, buffers.c, layout_c,
                          buffers.d, layout_d, &heuristic_result.algo, workspace,
                          kWorkspaceSize, stream);
  };

  for (int i = 0; i < warmup_iterations; ++i) {
    cublas_status = launch();
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublaslt_status_to_error(cublas_status);
    }
  }
  cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  cuda_status = cudaEventCreate(&start);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaEventCreate(&stop);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaEventRecord(start, stream);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  for (int i = 0; i < iterations; ++i) {
    cublas_status = launch();
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublaslt_status_to_error(cublas_status);
    }
  }

  cuda_status = cudaEventRecord(stop, stream);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaEventSynchronize(stop);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  float elapsed_ms = 0.0f;
  cuda_status = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cleanup();
  return 0;
}

extern "C" int run_cublaslt_f32(KernelRunnerBuffers buffers, int m, int n, int k,
                                 int warmup_iterations, int iterations,
                                 cudaStream_t *streams, int num_streams,
                                 int split_k_slices, float *avg_ms) {
  return run_cublaslt_gemm<float>(buffers, m, n, k, warmup_iterations,
                                  iterations, streams, num_streams, split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_f16(KernelRunnerBuffers buffers, int m, int n, int k,
                                 int warmup_iterations, int iterations,
                                 cudaStream_t *streams, int num_streams,
                                 int split_k_slices, float *avg_ms) {
  return run_cublaslt_gemm<cutlass::half_t>(buffers, m, n, k, warmup_iterations,
                                            iterations, streams, num_streams, split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_bf16(KernelRunnerBuffers buffers, int m, int n, int k,
                  int warmup_iterations, int iterations,
                  cudaStream_t *streams, int num_streams,
                  int split_k_slices, float *avg_ms) {
  return run_cublaslt_gemm<cutlass::bfloat16_t>(
    buffers, m, n, k, warmup_iterations, iterations, streams, num_streams,
    split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_fp8_blockwise_128x128(
    KernelRunnerBuffers buffers, int m, int n, int k, int warmup_iterations,
    int iterations, cudaStream_t *streams, int num_streams, int split_k_slices,
    float *avg_ms) {
  return run_cublaslt_gemm<cutlass::float_e4m3_t>(
      buffers, m, n, k, warmup_iterations, iterations, streams, num_streams,
      split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_f64(KernelRunnerBuffers buffers, int m, int n, int k,
                                 int warmup_iterations, int iterations,
                                 cudaStream_t *streams, int num_streams,
                                 int split_k_slices, float *avg_ms) {
  return run_cublaslt_gemm<double>(buffers, m, n, k, warmup_iterations,
                                   iterations, streams, num_streams, split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_f16_TNT(
    KernelRunnerBuffers buffers, int m, int n, int k, int warmup_iterations,
    int iterations, cudaStream_t *streams, int num_streams, int split_k_slices,
    float *avg_ms) {
  return run_cublaslt_gemm<cutlass::half_t, true>(
      buffers, m, n, k, warmup_iterations, iterations, streams, num_streams,
      split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_f16_TNN(
    KernelRunnerBuffers buffers, int m, int n, int k, int warmup_iterations,
    int iterations, cudaStream_t *streams, int num_streams, int split_k_slices,
    float *avg_ms) {
  return run_cublaslt_gemm<cutlass::half_t, true, true>(
      buffers, m, n, k, warmup_iterations, iterations, streams, num_streams,
      split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_fp8_blockwise_128x128_TNT(
    KernelRunnerBuffers buffers, int m, int n, int k, int warmup_iterations,
    int iterations, cudaStream_t *streams, int num_streams, int split_k_slices,
    float *avg_ms) {
  return run_cublaslt_gemm<cutlass::float_e4m3_t, true, false>(
      buffers, m, n, k, warmup_iterations, iterations, streams, num_streams,
      split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_fp8_blockwise_128x128_TNN(
    KernelRunnerBuffers buffers, int m, int n, int k, int warmup_iterations,
    int iterations, cudaStream_t *streams, int num_streams, int split_k_slices,
    float *avg_ms) {
  return run_cublaslt_gemm<cutlass::float_e4m3_t, true, true>(
      buffers, m, n, k, warmup_iterations, iterations, streams, num_streams,
      split_k_slices, avg_ms);
}