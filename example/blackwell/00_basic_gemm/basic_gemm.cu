/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have row-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  The kernel uses the CUTLASS 3 collective API and targets NVIDIA Blackwell GPUs.
*/

// Standard Library includes
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

#include "cutlass/cutlass.h"
#include "cutlass/arch/mma_sm100.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////


using Layout = cutlass::layout::RowMajor;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

#ifdef TILE_SIZE_256
using TileShape = cute::Shape<cute::_256, cute::_128, cute::_16>;
static const Stages = 3;
#elif defined(TILE_SIZE_128)
using TileShape = cute::Shape<cute::_64, cute::_128, cute::_16>;
static const Stages = 2;
#endif

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm100,
  cutlass::arch::OpClassSimt,
  float, Layout, 1,
  float, Layout, 4,
  float,
  TileShape,
  ClusterShape,
  cutlass::gemm::collective::StageCount<Stages>,
  cutlass::gemm::KernelMultistage>::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm100,
  cutlass::arch::OpClassSimt,
  TileShape,
  ClusterShape,
  cutlass::epilogue::collective::EpilogueTileAuto,
  float,
  float,
  float, Layout, 4,
  float, Layout, 4,
  cutlass::epilogue::EpilogueSimtVectorized>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  cute::Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue>;

using CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
// Notes:
// Files: output_tile_thread_map.h, default_thread_map_simt.h, epilogue/threadblock/predicated_tile_iterator.h

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  int runs = 1) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  

  using Kernel = typename CutlassGemm::GemmKernel;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using RasterOrderOptions = typename Kernel::TileScheduler::RasterOrderOptions;

  (void)lda;
  (void)ldb;
  (void)ldc;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  CutlassGemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      typename Kernel::ProblemShape{M, N, K, 1},
      {A, stride_a, B, stride_b},
      {{alpha, beta}, C, stride_c, C, stride_d},
      hw_info);

  args.scheduler.raster_order = RasterOrderOptions::AlongN;
  args.scheduler.max_swizzle_size = 1;

  cutlass::Status status = CutlassGemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }

  size_t workspace_size = CutlassGemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  CutlassGemm gemm_operator;
  status = gemm_operator.initialize(args, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  for (int r = 0; r < runs; r++) {
    status = gemm_operator.run();

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ float MatrixValue(int row, int column, int columns, int seed) {
  int64_t offset = int64_t(row) * columns + column;
  int64_t const multiplier = 16807;
  int64_t const modulus = 31;
  return float(((offset + seed) * multiplier % modulus) - modulus / 2);
}

/// Kernel to initialize a matrix with deterministic small integers.
__global__ void InitializeMatrix_kernel(
  float *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    matrix[int64_t(i) * columns + j] = MatrixValue(i, j, columns, seed);
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(float **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

cudaError_t VerifyResult(int M, int N, int K, float alpha, float beta, float const *C) {
  int const sample_rows[] = {0, M / 3, M / 2, M - 1};
  int const sample_columns[] = {0, N / 3, N / 2, N - 1};

  for (int row : sample_rows) {
    for (int column : sample_columns) {
      float actual = 0.0f;
      cudaError_t result = cudaMemcpy(
          &actual, C + int64_t(row) * N + column, sizeof(actual), cudaMemcpyDeviceToHost);
      if (result != cudaSuccess) {
        return result;
      }

      float accumulator = 0.0f;
      for (int inner = 0; inner < K; ++inner) {
        accumulator += MatrixValue(row, inner, K, 0) * MatrixValue(inner, column, N, 17);
      }
      float expected = alpha * accumulator + beta * MatrixValue(row, column, N, 101);
      float tolerance = 1.0e-4f * std::fmax(1.0f, std::fabs(expected));
      if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "Verification failed at (" << row << ", " << column
                  << "): expected " << expected << ", got " << actual << std::endl;
        return cudaErrorUnknown;
      }
    }
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, float alpha, float beta, int runs) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = K;
  int ldb = N;
  int ldc = N;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = VerifyResult(M, N, K, alpha, beta, C_cutlass);
  if (result != cudaSuccess) {
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);
    return result;
  }

  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, 10);
  if (result != cudaSuccess) {
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);
    return result;
  }
  cudaDeviceSynchronize();

  cudaEvent_t start;
  cudaEvent_t end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  
  result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, runs);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsedTime = 0;
  cudaEventElapsedTime(&elapsedTime, start, end);
  elapsedTime = elapsedTime/runs;
  std::cout << "Time elapsed " << (elapsedTime) << " ms" << std::endl;
  std::cout << "GFLOPS " << ((2L*((long)M)*((long)N)*K)/(elapsedTime/1e3))/1e9 << std::endl;
  //
  // Free device memory allocations.
  //

  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   basic_gemm_128 <M> <N> <K> [runs]
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 128, 128, 128 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  float scalars[2] = { 1, 0 };
  int runs = 10;
  if (argc > 4) {
    std::stringstream ss(arg[4]);
    ss >> runs;
  }
  // {
  //   std::stringstream ss(arg[6]);
  //   ss >> scalars[1];
  // }
  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1],      // beta
    runs
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
