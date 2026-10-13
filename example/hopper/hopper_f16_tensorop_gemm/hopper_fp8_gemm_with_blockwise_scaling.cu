/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief Blocked scale Hopper FP8 GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture
    This example demonstrate a blocked scaled FP8 GEMM using the new CUTLASS 3.0.
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:
    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.
    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster.
    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).
    4. This example applies block scale factors to A and B during accumulation and stores D directly.
    5. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can
    improve performance.
    Examples:
      $ ./examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling  \
        --m=2816 --n=3072 --k=16384 \
        --raster=h --swizzle=2
*/

#include <iostream>
#include <omp.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

// Includes from examples directory
#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                          // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                          // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// D matrix configuration
using         ElementD    = cutlass::bfloat16_t;
using         LayoutD     = cutlass::layout::RowMajor;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;
using         ElementC    = void;
using         LayoutC     = LayoutD;
constexpr int AlignmentC  = AlignmentD;

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ElementCompute      = float;                                          // Element type for epilogue computation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_128,_128>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster

using ScaleConfig =     cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;

using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand


using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum; 
using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecializedCooperative;

// Pingpong always generates stack spills.
// using KernelSchedule      = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8BlockScaledAccum; 
// using EpilogueSchedule    = cutlass::epilogue::TmaWarpSpecialized;

using EpilogueTileType    = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    cutlass::epilogue::fusion::LinearCombination<
      ElementD,
      float,
      ElementD,
      float
    >
  >::CollectiveOp;

using CollectiveMainloopWithBlockWiseScaling = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, // Indicates ProblemShape
    CollectiveMainloopWithBlockWiseScaling,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideD stride_D;
LayoutSFA layout_SFA;
LayoutSFB layout_SFB;
uint64_t seed = 0;

using LayoutScalar = cutlass::layout::PackedVectorLayout;
cutlass::HostTensor<ElementA  , LayoutA  > tensor_A;
cutlass::HostTensor<ElementB  , LayoutB  > tensor_B;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_D;
cutlass::HostTensor<ElementBlockScale, LayoutScalar> blockscale_tensor_A;
cutlass::HostTensor<ElementBlockScale, LayoutScalar> blockscale_tensor_B;
cutlass::HostTensor<ElementD  , LayoutD  > tensor_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

template<typename RasterOrderOptions>
struct Options {

  bool help = false;
  bool check = true;

  int iterations = 1000;
  int warmup = 100;
  int m = 1024, n = 512, k = 1024, l = 1;
  RasterOrderOptions raster;
  int swizzle;
  float epsilon = 0.02f;
  float non_zero_floor = 1.f;

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("check", check);
    cmd.get_cmd_line_argument("epsilon", epsilon);
    cmd.get_cmd_line_argument("non-zero-floor", non_zero_floor);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }

    cmd.get_cmd_line_argument("swizzle", swizzle, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "54_fp8_hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP8 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the l extent (batch) of the GEMM\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --warmup=<int>              Number of warmup iterations (default: 1000)\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --check=<bool>              Verify on the CPU using half the available CPU cores (default: true)\n"
      << "  --epsilon=<float>           Relative comparison tolerance (default: 0.02)\n"
      << "  --non-zero-floor=<float>    Comparison floor (default: 1)\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "54_fp8_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};


#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <typename Element, typename Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    double scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<Element>::value;
    int bits_output = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
      scope_max = 2;
      scope_min = 0;
    } else if (bits_input <= 8) {
      scope_max = 4;
      scope_min = -4;
    } else if (bits_output == 16) {
      scope_max = 4;
      scope_min = -4;
    } else {
      scope_max = 4;
      scope_min = -4;
    }

    cutlass::reference::host::BlockFillRandomUniform(
      view.data(), view.capacity(), seed, scope_max, scope_min, bits_input);
  }
  else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == cutlass::Distribution::Identity) {

    cutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }

  return true;
}

/// Helper to initialize a block of device data (scale_tensors)
template <typename Element, typename Layout>
bool initialize_scale_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    double scope_max, scope_min;

    scope_min = -4;
    scope_max = 4;

    cutlass::reference::host::BlockFillRandomUniform(
      view.data(), view.capacity(), seed, scope_max, scope_min);
  }
  else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view);
  }
  else if (dist_kind == cutlass::Distribution::Identity) {

    cutlass::reference::host::TensorFillIdentity(view);
  }
  else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
  }
  else if (dist_kind == cutlass::Distribution::Sequential) {
    cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
  }
  else {
    throw std::runtime_error("Not implementated.");
  }

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options<RasterOrderOptions> &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(options.m, options.k, options.l));
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(options.n, options.k, options.l));
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));

  // Layout SFA and SFB represent logically broadcasting data in CuTe.
  // E.g., if Layout SFA has shape ((ScaleGranularityM, M / ScaleGranularityM), (ScaleGraunularityK, K / ScaleGranularityK))
  // and strides ((0, 1), (0, M / ScaleGraunuarlityM)), then each collection of ScaleGranularityM x ScaleGranularityK
  // indices in the tensor map to the same offset.

  layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, options.l));
  layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto d_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto blockscale_a_coord = cutlass::make_Coord(size(filter_zeros(layout_SFA)));
  auto blockscale_b_coord = cutlass::make_Coord(size(filter_zeros(layout_SFB)));

  tensor_A.resize(a_coord);
  blockscale_tensor_A.resize(blockscale_a_coord);
  tensor_B.resize(b_coord);
  blockscale_tensor_B.resize(blockscale_b_coord);
  tensor_D.resize(d_coord);
  tensor_ref_D.resize(d_coord);

  cutlass::Distribution::Kind dist_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleA = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleB = cutlass::Distribution::Uniform;

  initialize_tensor(tensor_A.host_view(), dist_A, seed + 2022);
  initialize_tensor(tensor_B.host_view(), dist_B, seed + 2023);
  initialize_scale_tensor(blockscale_tensor_A.host_view(), dist_scaleA, seed + 2025);
  initialize_scale_tensor(blockscale_tensor_B.host_view(), dist_scaleB, seed + 2026);
  cutlass::reference::host::TensorFill(tensor_D.host_view());

#if 0 // Dump blockscaled tensors
  std::cout << "blockscale_tensor_A: " << blockscale_a_coord << std::endl;
  std::cout << blockscale_tensor_A.host_view() << "\n";
  std::cout << "blockscale_tensor_B: " << blockscale_b_coord << std::endl;
  std::cout << blockscale_tensor_B.host_view() << "\n";
#endif

  // Print block scaling tensors on the host side.
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_D.sync_device();
  blockscale_tensor_A.sync_device();
  blockscale_tensor_B.sync_device();
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options<RasterOrderOptions> &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, options.l},
    {tensor_A.device_data(),
     stride_A,
     tensor_B.device_data(),
     stride_B,
     blockscale_tensor_A.device_data(),
     layout_SFA,
     blockscale_tensor_B.device_data(),
     layout_SFB
     },
    {
      {}, // epilogue.thread
      nullptr, StrideC{},
      tensor_D.device_data(), stride_D
    }
  };

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options<RasterOrderOptions> &options) {
  auto A = cute::make_tensor(tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto B = cute::make_tensor(tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto D = cute::make_tensor(tensor_ref_D.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.n, options.l), stride_D));
  auto SFA = cute::make_tensor(blockscale_tensor_A.host_data(), layout_SFA);
  auto SFB = cute::make_tensor(blockscale_tensor_B.host_data(), layout_SFB);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator, decltype(A), decltype(SFA), decltype(B), decltype(SFB)
  > mainloop_params{A, SFA, B, SFB};
  cutlass::reference::host::GettEpilogueParams<
      ElementCompute, ElementCompute, ElementAccumulator, ElementCompute,
      decltype(D), decltype(D)
  > epilogue_params;

  epilogue_params.D = D;

  int available_cpus = omp_get_num_procs();
  int reference_threads = available_cpus > 1 ? available_cpus / 2 : 1;
  int previous_threads = omp_get_max_threads();
  int previous_dynamic = omp_get_dynamic();
  omp_set_dynamic(0);
  omp_set_num_threads(reference_threads);
  std::cout << "  Host reference: " << reference_threads << " threads ("
            << available_cpus << " available CPUs)" << std::endl;
  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
  omp_set_num_threads(previous_threads);
  omp_set_dynamic(previous_dynamic);

  bool passed = true;
  tensor_D.sync_host();
  passed &= cutlass::reference::host::TensorRelativelyEquals(tensor_D.host_view(), tensor_ref_D.host_view(), ElementD(options.epsilon), ElementD(options.non_zero_floor));
  double mse = cutlass::reference::host::TensorMSE(tensor_D.host_view(), tensor_ref_D.host_view());
  double mre = cutlass::reference::host::TensorMRE(tensor_D.host_view(), tensor_ref_D.host_view());
  double max_error = cutlass::reference::host::TensorGreatestError(tensor_D.host_view(), tensor_ref_D.host_view());
  std::cout << "  Result MSE: " << mse << ", MRE: " << mre << ", greatest error: " << max_error << std::endl;
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options<RasterOrderOptions> &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  if (options.check) {
    result.passed = verify(options);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;
  }
  else {
    result.passed = true;
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    for (int iter = 0; iter < options.warmup + options.iterations; ++iter) {
      if (iter == options.warmup)
        timer.start();
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of " << options.swizzle << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
  }

  return result.passed;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }
  //
  // Parse options
  //

  Options<RasterOrderOptions> options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  bool passed = run<Gemm>(options);
  if (!passed)
    return -1;
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
