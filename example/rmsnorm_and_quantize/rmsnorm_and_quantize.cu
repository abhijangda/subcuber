#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <math_constants.h>

#include "src/cuda/kernels/elementwise/rmsnorm_and_quantize.cuh"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"

namespace rmsnorm_example {

void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << cudaGetErrorString(result) << '\n';
    std::exit(1);
  }
}

__global__ void initialize(__nv_bfloat16* input, __nv_bfloat16* weight,
                           int rows, int hidden, int stride) {
  size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size_t(rows) * stride) {
    size_t const row = index / stride;
    size_t const column = index % stride;
    uint32_t random = uint32_t(index) * 747796405u + 2891336453u;
    random = ((random >> ((random >> 28) + 4)) ^ random) * 277803737u;
    random = (random >> 22) ^ random;
    float value = float(int(random % 65) - 32) / 8.0f;
    if (row == 0 || (row == 2 && column < 128)) value = 0.0f;
    if (row == 1) value *= 1.0e-6f;
    if (row == 3) value = column % 2 == 0 ? 1.0f : -1.0f;
    input[index] = __float2bfloat16_rn(value);
  }
  if (weight != nullptr && index < size_t(hidden)) {
    float value = float(int(index % 65) - 32) / 32.0f;
    weight[index] = __float2bfloat16_rn(value);
  }
}

__global__ void reference_rms(float* inverse_rms, __nv_bfloat16 const* input,
                              int rows, int hidden, int stride, float epsilon) {
  int const row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  double sum_squares = 0.0;
  for (int column = 0; column < hidden; ++column) {
    double value = __bfloat162float(input[size_t(row) * stride + column]);
    sum_squares += value * value;
  }
  inverse_rms[row] = rsqrtf(float(sum_squares) / hidden + epsilon);
}

__global__ void reference_normalize(__nv_bfloat16* normalized,
                                    __nv_bfloat16 const* input,
                                    __nv_bfloat16 const* weight,
                                    float const* inverse_rms,
                                    int rows, int hidden, int stride) {
  size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= size_t(rows) * hidden) return;
  size_t const row = index / hidden;
  int const column = index % hidden;
  float value = __bfloat162float(input[row * stride + column]);
  float norm = __bfloat162float(__float2bfloat16_rn(value * inverse_rms[row]));
  normalized[index] = __float2bfloat16_rn(norm * __bfloat162float(weight[column]));
}

__global__ void reference_quantize(__nv_fp8_e4m3* output, float* scales,
                                   __nv_bfloat16 const* normalized,
                                   int rows, int hidden, int scale_stride) {
  size_t const group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int const groups_per_row = hidden / 128;
  if (group >= size_t(rows) * groups_per_row) return;
  float absmax = 0.0f;
  for (int element = 0; element < 128; ++element) {
    absmax = fmaxf(absmax, fabsf(__bfloat162float(normalized[group * 128 + element])));
  }
  float const scale = fmaxf(absmax / 448.0f, 1.0f / (448.0f * 512.0f));
  scales[(group % groups_per_row) * scale_stride + group / groups_per_row] = scale;
  for (int element = 0; element < 128; ++element) {
    output[group * 128 + element] =
        __nv_fp8_e4m3(__bfloat162float(normalized[group * 128 + element]) / scale);
  }
}

__global__ void reference_strassen_presums(float* operands,
                                          __nv_bfloat16 const* normalized,
                                          int rows, int hidden) {
  size_t const index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t const matrix_elements = size_t(rows / 2) * (hidden / 2);
  if (index >= matrix_elements) return;
  size_t const top_left = (index / (hidden / 2)) * hidden + index % (hidden / 2);
  size_t const bottom_left = top_left + size_t(rows / 2) * hidden;
  float const top_left_value = __bfloat162float(normalized[top_left]);
  float const top_right_value = __bfloat162float(normalized[top_left + hidden / 2]);
  float const bottom_left_value = __bfloat162float(normalized[bottom_left]);
  float const bottom_right_value = __bfloat162float(normalized[bottom_left + hidden / 2]);
  float const bottom_sum = bottom_left_value + bottom_right_value;
  float const difference = bottom_sum - top_left_value;
  operands[index] = top_left_value;
  operands[matrix_elements + index] = top_right_value;
  operands[2 * matrix_elements + index] = bottom_right_value;
  operands[3 * matrix_elements + index] = top_left_value - bottom_left_value;
  operands[4 * matrix_elements + index] = bottom_sum;
  operands[5 * matrix_elements + index] = difference;
  operands[6 * matrix_elements + index] = top_right_value - difference;
}

__global__ void reference_strassen_quantize(__nv_fp8_e4m3* output, float* scales,
                                           float const* operands,
                                           int half_rows, int half_hidden, int scale_stride) {
  size_t const block = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int const groups_per_row = half_hidden / 128;
  size_t const matrix_blocks = size_t(half_rows) * groups_per_row;
  if (block >= 7 * matrix_blocks) return;
  int const matrix = block / matrix_blocks;
  size_t const row = block % matrix_blocks / groups_per_row;
  int const group = block % groups_per_row;
  float absmax = 0.0f;
  for (int element = 0; element < 128; ++element) {
    absmax = fmaxf(absmax, fabsf(operands[block * 128 + element]));
  }
  float const scale = matrix < 3 ? fmaxf(absmax / 448.0f, 1.0f / (448.0f * 512.0f)) :
      (absmax == 0.0f ? 1.0f : fmaxf(absmax / 448.0f, FLT_MIN));
  size_t const scale_plane = matrix < 3 ? size_t(matrix) * groups_per_row + group :
      size_t(3) * groups_per_row + group * 4 + matrix - 3;
  scales[scale_plane * scale_stride + row] = scale;
  for (int element = 0; element < 128; ++element) {
    output[block * 128 + element] = __nv_fp8_e4m3(operands[block * 128 + element] / scale);
  }
}

__global__ void verify_fp8_quad_division(unsigned* mismatches) {
  unsigned const index = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned const scale_index = index / 128;
  unsigned const code = index % 128;
  unsigned random = scale_index * 747796405u + 2891336453u;
  random = ((random >> ((random >> 28) + 4)) ^ random) * 277803737u;
  float scale = __uint_as_float(((scale_index % 254 + 1) << 23) | (random & 0x7fffff));
  if (scale_index < 8) {
    float const edge_scales[8] = {0x1p-60f, 0x1p60f,
        nextafterf(0x1p-60f, 0.0f), nextafterf(0x1p-60f, CUDART_INF_F),
        nextafterf(0x1p60f, 0.0f), nextafterf(0x1p60f, CUDART_INF_F),
        FLT_MIN, FLT_MAX};
    scale = edge_scales[scale_index];
  }
  __nv_fp8_e4m3 lower, upper;
  lower.__x = code < 126 ? code : 126;
  upper.__x = code < 126 ? code + 1 : 126;
  float const midpoint = (float(lower) + float(upper)) * 0.5f * scale;
  float values[4] = {nextafterf(midpoint, 0.0f), midpoint,
      nextafterf(midpoint, CUDART_INF_F), -midpoint};
  if (code == 127) {
    values[0] = 0.0f;
    values[1] = -0.0f;
    values[2] = __uint_as_float(1);
    values[3] = -__uint_as_float(1);
  }
  if (scale_index % 2 != 0) {
    for (int element = 0; element < 4; ++element) values[element] = -values[element];
  }
  __nv_fp8x4_e4m3 actual(strassen::elementwise::detail::divide_fp8_quad(values, scale));
  __nv_fp8x4_e4m3 expected(make_float4(
      __fdiv_rn(values[0], scale), __fdiv_rn(values[1], scale),
      __fdiv_rn(values[2], scale), __fdiv_rn(values[3], scale)));
  if (actual.__x != expected.__x && atomicAdd(mismatches, 1u) == 0) {
    printf("FP8 division mismatch: scale=%g code=%u actual=%08x expected=%08x\n",
           scale, code, actual.__x, expected.__x);
  }
  float octet_values[8] = {values[0], values[1], values[2], values[3],
      -values[3], -values[2], -values[1], -values[0]};
  uint2 const actual_octet = strassen::elementwise::detail::quantize_fp8_octet(octet_values, scale);
  __nv_fp8x4_e4m3 expected_upper(make_float4(
      __fdiv_rn(octet_values[4], scale), __fdiv_rn(octet_values[5], scale),
      __fdiv_rn(octet_values[6], scale), __fdiv_rn(octet_values[7], scale)));
  if ((actual_octet.x != expected.__x || actual_octet.y != expected_upper.__x) &&
      atomicAdd(mismatches, 1u) == 0) {
    printf("FP8 octet division mismatch: scale=%g code=%u\n", scale, code);
  }
}

bool verify_strassen_scale_layout(int half_rows, int half_hidden, int scale_stride) {
  using Config = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
  auto raw_layout = Config::tile_atom_to_shape_SFA(
      cute::make_shape(scale_stride, half_hidden, half_hidden, 1));
  auto presum_layout = Config::tile_atom_to_shape_SFA(
      cute::make_shape(4 * scale_stride, half_hidden, half_hidden, 1));
  int const groups = half_hidden / 128;
  int64_t const scale_matrix_elements = int64_t(groups) * scale_stride;
  for (int matrix = 0; matrix < 7; ++matrix) {
    for (int group = 0; group < groups; ++group) {
      for (int row = 0; row < half_rows; ++row) {
        int64_t const expected = matrix < 3 ? matrix * scale_matrix_elements +
            raw_layout(cute::make_coord(row, group * 128, 0)) :
            3 * scale_matrix_elements + presum_layout(
                cute::make_coord((matrix - 3) * scale_stride + row, group * 128, 0));
        int64_t const actual = strassen::elementwise::rmsnorm_strassen_a_scale_offset(
            static_cast<strassen::elementwise::StrassenAOperand>(matrix), row, group, groups, scale_stride);
        if (actual != expected) {
          std::cerr << "CUTLASS scale layout mismatch\n";
          return false;
        }
      }
    }
  }
  std::cout << "PASS: all seven scale views match the CUTLASS blockwise layouts\n";
  return true;
}

int run(int argc, char const** argv) {
  cutlass::CommandLine command(argc, argv);
  if (command.check_cmd_line_flag("help")) {
    std::cout << "Usage: rmsnorm_and_quantize [--rows=8192] [--hidden=8192] "
                 "[--iterations=1000] [--warmup=100] [--buffers=10] [--check=1] [--device=0] "
                 "[--epsilon=1e-6] [--input-padding=0] [--scale-padding=0] [--strassen] "
                 "[--quantize-only] [--zero-input]\n";
    return 0;
  }
  int rows = 8192, hidden = 8192, iterations = 1000, warmup = 100;
  int buffers = 10;
  int verify = 1, device = 0, input_padding = 0, scale_padding = 0;
  float epsilon = 1.0e-6f;
  bool const strassen = command.check_cmd_line_flag("strassen");
  bool const quantize_only = command.check_cmd_line_flag("quantize-only");
  command.get_cmd_line_argument("rows", rows);
  command.get_cmd_line_argument("hidden", hidden);
  command.get_cmd_line_argument("iterations", iterations);
  command.get_cmd_line_argument("warmup", warmup);
  command.get_cmd_line_argument("buffers", buffers);
  command.get_cmd_line_argument("check", verify);
  command.get_cmd_line_argument("device", device);
  command.get_cmd_line_argument("epsilon", epsilon);
  command.get_cmd_line_argument("input-padding", input_padding);
  command.get_cmd_line_argument("scale-padding", scale_padding);
  if (rows <= 0 || hidden <= 0 || hidden % 128 != 0 || iterations < 0 || warmup < 0 || buffers <= 0 ||
      (strassen && (rows % 2 != 0 || hidden % 256 != 0)) ||
      input_padding < 0 || input_padding % 4 != 0 || scale_padding < 0 ||
      (!quantize_only && (!std::isfinite(epsilon) || epsilon <= 0.0f))) {
    std::cerr << "Invalid shape, padding, epsilon, buffer count, or iteration count\n";
    return 1;
  }
  check(cudaSetDevice(device));
  cudaDeviceProp properties;
  check(cudaGetDeviceProperties(&properties, device));
  int const input_stride = hidden + input_padding;
  int const output_rows = strassen ? rows / 2 : rows;
  int const output_hidden = strassen ? hidden / 2 : hidden;
  int const matrices = strassen ? 7 : 1;
  int const scale_stride = output_rows + scale_padding;
  size_t const elements = size_t(rows) * hidden;
  size_t const input_elements = size_t(rows) * input_stride;
  size_t const output_elements = size_t(matrices) * output_rows * output_hidden;
  size_t const scale_elements = size_t(matrices) * scale_stride * (output_hidden / 128);
  cutlass::device_memory::allocation<__nv_bfloat16> input(input_elements * buffers);
  size_t const weight_elements = quantize_only ? 0 : size_t(hidden);
  cutlass::device_memory::allocation<__nv_bfloat16> weight(weight_elements * buffers);
  cutlass::device_memory::allocation<__nv_fp8_e4m3> output(output_elements * buffers);
  cutlass::device_memory::allocation<float> scales(scale_elements * buffers);
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  for (int buffer = 0; buffer < buffers; ++buffer) {
    initialize<<<(input_elements + 255) / 256, 256, 0, stream>>>(
        input.get() + size_t(buffer) * input_elements,
      quantize_only ? nullptr : weight.get() + size_t(buffer) * hidden, rows, hidden, input_stride);
    check(cudaGetLastError());
  }
  if (command.check_cmd_line_flag("zero-input")) {
    check(cudaMemsetAsync(input.get(), 0, input_elements * buffers * sizeof(__nv_bfloat16), stream));
  }
  check(cudaMemsetAsync(output.get(), 0x7f, output_elements * buffers, stream));
  check(cudaMemsetAsync(scales.get(), 0xff, scale_elements * buffers * sizeof(float), stream));
  auto launch = [&](int buffer) {
    auto* buffer_output = output.get() + size_t(buffer) * output_elements;
    auto* buffer_scales = scales.get() + size_t(buffer) * scale_elements;
    auto const* buffer_input = input.get() + size_t(buffer) * input_elements;
    auto const* buffer_weight = quantize_only ? nullptr : weight.get() + size_t(buffer) * hidden;
    if (quantize_only) {
      if (strassen) {
        check(strassen::elementwise::launch_quantize_bf16_fp8_strassen_a(
            buffer_output, buffer_scales, buffer_input, rows, hidden,
            stream, input_stride, scale_stride));
      } else {
        check(strassen::elementwise::launch_quantize_bf16_fp8(
            buffer_output, buffer_scales, buffer_input, rows, hidden,
            stream, input_stride, scale_stride));
      }
    } else if (strassen) {
      check(strassen::elementwise::launch_rmsnorm_and_quantize_strassen_a(
          buffer_output, buffer_scales, buffer_input, buffer_weight, rows, hidden,
          epsilon, stream, input_stride, scale_stride));
    } else {
      check(strassen::elementwise::launch_rmsnorm_and_quantize(
          buffer_output, buffer_scales, buffer_input, buffer_weight, rows, hidden,
          epsilon, stream, input_stride, scale_stride));
    }
  };
  for (int buffer = 0; buffer < buffers; ++buffer) launch(buffer);
  check(cudaStreamSynchronize(stream));
  std::cout << properties.name << (quantize_only ? ": quantization only, BF16 [" : ": RMSNorm, BF16 [")
            << rows << ", " << hidden
            << "] -> " << matrices << " E4M3 matrices [" << output_rows << ", "
            << output_hidden << "], " << output_elements / 128 << " FP32 scales\n";
  double const buffer_bytes = input_elements * 2.0 + weight_elements * 2.0 +
      output_elements + scale_elements * sizeof(float);
  std::cout << "Rotating buffers: " << buffers << "; footprint: "
            << buffer_bytes * buffers / (1024.0 * 1024.0) << " MiB; L2: "
            << properties.l2CacheSize / (1024.0 * 1024.0) << " MiB\n";

  if (verify) {
    if (strassen && !verify_strassen_scale_layout(rows / 2, hidden / 2, scale_stride)) return 1;
    {
      cutlass::device_memory::allocation<unsigned> mismatches(1);
      check(cudaMemsetAsync(mismatches.get(), 0, sizeof(unsigned), stream));
      verify_fp8_quad_division<<<4096 * 128 / 256, 256, 0, stream>>>(mismatches.get());
      check(cudaGetLastError());
      check(cudaStreamSynchronize(stream));
      unsigned mismatch_count = 0;
      check(cudaMemcpy(&mismatch_count, mismatches.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
      if (mismatch_count != 0) {
        std::cerr << "FP8 division mismatches: " << mismatch_count << '\n';
        return 1;
      }
      std::cout << "PASS: 2097152 quad and 4194304 octet FP8 boundary/extreme cases match precise division\n";
    }
    cutlass::device_memory::allocation<float> inverse_rms(quantize_only ? 0 : rows);
    cutlass::device_memory::allocation<__nv_bfloat16> normalized(elements);
    cutlass::device_memory::allocation<__nv_fp8_e4m3> expected(output_elements);
    cutlass::device_memory::allocation<float> expected_scales(scale_elements);
    cutlass::device_memory::allocation<float> operands(strassen ? output_elements : 0);
    check(cudaMemsetAsync(expected_scales.get(), 0xff, scale_elements * sizeof(float), stream));
    if (quantize_only) {
      check(cudaMemcpy2DAsync(normalized.get(), size_t(hidden) * sizeof(__nv_bfloat16),
        input.get(), size_t(input_stride) * sizeof(__nv_bfloat16),
        size_t(hidden) * sizeof(__nv_bfloat16), rows, cudaMemcpyDeviceToDevice, stream));
    } else {
      reference_rms<<<(rows + 127) / 128, 128, 0, stream>>>(
        inverse_rms.get(), input.get(), rows, hidden, input_stride, epsilon);
      reference_normalize<<<(elements + 255) / 256, 256, 0, stream>>>(
        normalized.get(), input.get(), weight.get(), inverse_rms.get(), rows, hidden, input_stride);
    }
    if (strassen) {
      reference_strassen_presums<<<(elements / 4 + 255) / 256, 256, 0, stream>>>(
        operands.get(), normalized.get(), rows, hidden);
      reference_strassen_quantize<<<(output_elements / 128 + 127) / 128, 128, 0, stream>>>(
        expected.get(), expected_scales.get(), operands.get(), rows / 2, hidden / 2, scale_stride);
    } else {
      reference_quantize<<<(elements / 128 + 127) / 128, 128, 0, stream>>>(
        expected.get(), expected_scales.get(), normalized.get(), rows, hidden, scale_stride);
    }
    check(cudaGetLastError());
    check(cudaStreamSynchronize(stream));
    std::vector<__nv_fp8_e4m3> actual_host(output_elements), expected_host(output_elements);
    std::vector<float> actual_scales(scale_elements), reference_scales(scale_elements);
    check(cudaMemcpy(expected_host.data(), expected.get(), output_elements, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(reference_scales.data(), expected_scales.get(), scale_elements * sizeof(float), cudaMemcpyDeviceToHost));
    for (int buffer = 0; buffer < buffers; ++buffer) {
      check(cudaMemcpy(actual_host.data(), output.get() + size_t(buffer) * output_elements,
                       output_elements, cudaMemcpyDeviceToHost));
      check(cudaMemcpy(actual_scales.data(), scales.get() + size_t(buffer) * scale_elements,
                       scale_elements * sizeof(float), cudaMemcpyDeviceToHost));
      for (size_t index = 0; index < output_elements; ++index) {
        if (actual_host[index].__x != expected_host[index].__x) {
          std::cerr << "FP8 mismatch in buffer " << buffer << " at output index " << index << ": "
                    << float(actual_host[index]) << " != " << float(expected_host[index]) << '\n';
          return 1;
        }
      }
      for (size_t index = 0; index < scale_elements; ++index) {
        bool const padding = index % scale_stride >= size_t(output_rows);
        if (padding ? !std::isnan(actual_scales[index]) : actual_scales[index] != reference_scales[index]) {
          std::cerr << "Scale mismatch in buffer " << buffer << " at " << index << '\n';
          return 1;
        }
      }
    }
    std::cout << "PASS: all " << buffers << " buffers, each with " << output_elements
              << " FP8 values and " << output_elements / 128
              << " scales match the unfused reference exactly\n";
  }

  if (iterations > 0) {
    for (int iteration = 0; iteration < warmup; ++iteration) launch(iteration % buffers);
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    check(cudaEventRecord(begin, stream));
    for (int iteration = 0; iteration < iterations; ++iteration) {
      launch((size_t(warmup) + iteration) % buffers);
    }
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float elapsed;
    check(cudaEventElapsedTime(&elapsed, begin, end));
    double const milliseconds = elapsed / iterations;
    double const minimum_bytes = elements * 2.0 + weight_elements * 2.0 + output_elements * (1.0 + 4.0 / 128);
    std::cout << std::fixed << std::setprecision(3) << "Average: " << milliseconds * 1000.0
              << " us; effective bandwidth: " << minimum_bytes / (milliseconds * 1.0e9)
              << " TB/s (" << iterations << " iterations, " << warmup << " warmups, "
              << buffers << " buffers)\n";
    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));
  }
  check(cudaStreamDestroy(stream));
  return 0;
}

}

int main(int argc, char const** argv) {
  return rmsnorm_example::run(argc, argv);
}