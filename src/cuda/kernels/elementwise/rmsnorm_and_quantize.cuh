#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace strassen::elementwise {

inline constexpr int RmsnormQuantizeGroupSize = 128;
inline constexpr float RmsnormQuantizeMinScale = 1.0f / (448.0f * 512.0f);

enum class StrassenAOperand : int {
	A0, A1, A3, A02, S1, S2, A1S2, Count
};

__host__ __device__ inline int64_t rmsnorm_strassen_a_scale_offset(
		StrassenAOperand operand, int row, int group, int groups_per_row,
		int64_t scale_stride) {
	int const matrix = static_cast<int>(operand);
	int64_t const plane = matrix < 3 ? int64_t(matrix) * groups_per_row + group :
			int64_t(3) * groups_per_row + int64_t(group) * 4 + matrix - 3;
	return plane * scale_stride + row;
}

namespace detail {

struct alignas(8) Bf16x4 {
	__nv_bfloat16 values[4];
};

struct alignas(8) Bf16Pairs4 {
	__nv_bfloat162 values[2];
};

struct alignas(16) Bf16Pairs8 {
	__nv_bfloat162 values[4];
};

struct alignas(4) Fp8x4 {
	__nv_fp8_e4m3 values[4];
};

__device__ __forceinline__ float4 divide_fp8_quad(float const (&values)[4], float scale) {
	if (scale >= 0x1p-60f && scale <= 0x1p60f) {
		float reciprocal;
		asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(scale));
		float const reciprocal_error = __fmaf_rn(-scale, reciprocal, 1.0f);
		reciprocal = __fmaf_rn(reciprocal, reciprocal_error, reciprocal);
		float quotients[4];
#pragma unroll
		for (int element = 0; element < 4; ++element) {
			float const quotient = __fmul_rn(values[element], reciprocal);
			float const residual = __fmaf_rn(-scale, quotient, values[element]);
			quotients[element] = copysignf(__fmaf_rn(reciprocal, residual, quotient), values[element]);
		}
		return make_float4(quotients[0], quotients[1], quotients[2], quotients[3]);
	}
	return make_float4(values[0] / scale, values[1] / scale,
			values[2] / scale, values[3] / scale);
}

__device__ __forceinline__ uint2 quantize_fp8_octet(float const (&values)[8], float scale) {
	float quotients[8];
	if (scale >= 0x1p-60f && scale <= 0x1p60f) {
		float reciprocal;
		asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(scale));
		float const reciprocal_error = __fmaf_rn(-scale, reciprocal, 1.0f);
		reciprocal = __fmaf_rn(reciprocal, reciprocal_error, reciprocal);
#pragma unroll
		for (int element = 0; element < 8; ++element) {
			float const quotient = __fmul_rn(values[element], reciprocal);
			float const residual = __fmaf_rn(-scale, quotient, values[element]);
			quotients[element] = copysignf(__fmaf_rn(reciprocal, residual, quotient), values[element]);
		}
	} else {
#pragma unroll
		for (int element = 0; element < 8; ++element) quotients[element] = values[element] / scale;
	}
	__nv_fp8x4_e4m3 lower(make_float4(quotients[0], quotients[1], quotients[2], quotients[3]));
	__nv_fp8x4_e4m3 upper(make_float4(quotients[4], quotients[5], quotients[6], quotients[7]));
	return make_uint2(lower.__x, upper.__x);
}

template <int Threads, int VectorSize = 4>
__global__ void quantize_bf16_fp8_kernel(
		__nv_fp8_e4m3* __restrict__ output, float* __restrict__ scales,
		__nv_bfloat16 const* __restrict__ input, int rows, int hidden_size,
		int64_t input_stride, int64_t scale_stride) {
	int const groups_per_row = hidden_size / RmsnormQuantizeGroupSize;
	int const row = blockIdx.x;
	constexpr int ThreadsPerGroup = RmsnormQuantizeGroupSize / VectorSize;
	using InputVector = std::conditional_t<VectorSize == 8, Bf16Pairs8, Bf16Pairs4>;
	int const lane = threadIdx.x % ThreadsPerGroup;
	unsigned const mask = VectorSize == 8 ? 0xffffu << (threadIdx.x % 32 / 16 * 16) : 0xffffffffu;
	auto const* input_row = reinterpret_cast<InputVector const*>(input + int64_t(row) * input_stride);
	auto* output_row = reinterpret_cast<__nv_fp8x4_e4m3*>(output + int64_t(row) * hidden_size);
	for (int group = threadIdx.x / ThreadsPerGroup; group < groups_per_row; group += Threads / ThreadsPerGroup) {
		int const vector = group * ThreadsPerGroup + lane;
		InputVector packed = input_row[vector];
		float values[VectorSize];
		float absmax = 0.0f;
#pragma unroll
		for (int pair = 0; pair < VectorSize / 2; ++pair) {
			float2 converted = __bfloat1622float2(packed.values[pair]);
			values[2 * pair] = converted.x;
			values[2 * pair + 1] = converted.y;
			absmax = fmaxf(absmax, fmaxf(fabsf(converted.x), fabsf(converted.y)));
		}
		absmax = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(absmax)));
		float const scale = fmaxf(absmax / 448.0f, RmsnormQuantizeMinScale);
		if (lane == 0) scales[int64_t(group) * scale_stride + row] = scale;
		if constexpr (VectorSize == 8) {
			reinterpret_cast<uint2*>(output_row)[vector] = quantize_fp8_octet(values, scale);
		} else {
			output_row[vector] = __nv_fp8x4_e4m3(divide_fp8_quad(values, scale));
		}
	}
}

template <int Threads, int VectorSize = 4>
__global__ __launch_bounds__(Threads, 4) void quantize_bf16_fp8_strassen_a_kernel(
		__nv_fp8_e4m3* __restrict__ output, float* __restrict__ scales,
		__nv_bfloat16 const* __restrict__ input, int half_rows, int hidden_size,
		int64_t input_stride, int64_t scale_stride) {
	constexpr int ThreadsPerGroup = RmsnormQuantizeGroupSize / VectorSize;
	using InputVector = std::conditional_t<VectorSize == 8, Bf16Pairs8, Bf16Pairs4>;
	using OutputVector = std::conditional_t<VectorSize == 8, uint2, __nv_fp8x4_e4m3>;
	int const groups_per_row = hidden_size / (2 * RmsnormQuantizeGroupSize);
	int const row = blockIdx.x;
	int const lane = threadIdx.x % ThreadsPerGroup;
	unsigned const mask = VectorSize == 8 ? 0xffffu << (threadIdx.x % 32 / 16 * 16) : 0xffffffffu;
	int const half_vectors = hidden_size / (2 * VectorSize);
	int64_t const matrix_vectors = int64_t(half_rows) * half_vectors;
	auto const* top = reinterpret_cast<InputVector const*>(input + int64_t(row) * input_stride);
	auto const* bottom = reinterpret_cast<InputVector const*>(input + (int64_t(row) + half_rows) * input_stride);
	auto* output_vectors = reinterpret_cast<OutputVector*>(output);
	for (int group = threadIdx.x / ThreadsPerGroup; group < groups_per_row; group += Threads / ThreadsPerGroup) {
		int const vector = group * ThreadsPerGroup + lane;
		InputVector quadrants[4] = {top[vector], top[half_vectors + vector],
				bottom[vector], bottom[half_vectors + vector]};
#pragma unroll
		for (int matrix = 0; matrix < 7; ++matrix) {
			float values[VectorSize];
			float absmax = 0.0f;
#pragma unroll
			for (int pair = 0; pair < VectorSize / 2; ++pair) {
				float2 converted;
				if (matrix < 3) {
					converted = __bfloat1622float2(quadrants[matrix < 2 ? matrix : 3].values[pair]);
				} else {
					float2 const top_left = __bfloat1622float2(quadrants[0].values[pair]);
					float2 const bottom_left = __bfloat1622float2(quadrants[2].values[pair]);
					if (matrix == 3) {
						converted = make_float2(top_left.x - bottom_left.x, top_left.y - bottom_left.y);
					} else {
						float2 const bottom_right = __bfloat1622float2(quadrants[3].values[pair]);
						converted = make_float2(bottom_left.x + bottom_right.x, bottom_left.y + bottom_right.y);
						if (matrix >= 5) converted = make_float2(converted.x - top_left.x, converted.y - top_left.y);
						if (matrix == 6) {
							float2 const top_right = __bfloat1622float2(quadrants[1].values[pair]);
							converted = make_float2(top_right.x - converted.x, top_right.y - converted.y);
						}
					}
				}
				values[2 * pair] = converted.x;
				values[2 * pair + 1] = converted.y;
				absmax = fmaxf(absmax, fmaxf(fabsf(converted.x), fabsf(converted.y)));
			}
			absmax = __uint_as_float(__reduce_max_sync(mask, __float_as_uint(absmax)));
			float const scale = matrix < 3 ? fmaxf(absmax / 448.0f, RmsnormQuantizeMinScale) :
					(absmax > 0.0f ? fmaxf(absmax / 448.0f, FLT_MIN) : 1.0f);
			if (lane == 0) {
				scales[rmsnorm_strassen_a_scale_offset(static_cast<StrassenAOperand>(matrix),
						row, group, groups_per_row, scale_stride)] = scale;
			}
			int64_t const output_offset = matrix * matrix_vectors + int64_t(row) * half_vectors + vector;
			if constexpr (VectorSize == 8) {
				output_vectors[output_offset] = quantize_fp8_octet(values, scale);
			} else {
				output_vectors[output_offset] = __nv_fp8x4_e4m3(divide_fp8_quad(values, scale));
			}
		}
	}
}

template <bool Strassen>
inline cudaError_t launch_quantize_bf16_fp8_impl(
		__nv_fp8_e4m3* output, float* scales, __nv_bfloat16 const* input,
		int rows, int hidden_size, cudaStream_t stream,
		int64_t input_stride, int64_t scale_stride) {
	constexpr int RowDivisor = Strassen ? 2 : 1;
	if (rows < 0 || rows % RowDivisor != 0 || hidden_size <= 0 ||
			hidden_size % (RowDivisor * RmsnormQuantizeGroupSize) != 0) {
		return cudaErrorInvalidValue;
	}
	if (input_stride == 0) input_stride = hidden_size;
	if (scale_stride == 0) scale_stride = rows / RowDivisor;
	if (input_stride < hidden_size || input_stride % 4 != 0 || scale_stride < rows / RowDivisor) {
		return cudaErrorInvalidValue;
	}
	if (rows == 0) return cudaSuccess;
	if (output == nullptr || scales == nullptr || input == nullptr ||
			reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) != 0 ||
			reinterpret_cast<uintptr_t>(output) % alignof(Fp8x4) != 0 ||
			reinterpret_cast<uintptr_t>(scales) % alignof(float) != 0) {
		return cudaErrorInvalidValue;
	}
	constexpr int Threads = 256;
	int const blocks = rows / RowDivisor;
	bool const vectorize8 = input_stride % 8 == 0 && reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
			reinterpret_cast<uintptr_t>(output) % 8 == 0;
	if constexpr (Strassen) {
		if (vectorize8) {
			quantize_bf16_fp8_strassen_a_kernel<Threads, 8><<<static_cast<unsigned>(blocks), Threads, 0, stream>>>(
					output, scales, input, rows / 2, hidden_size, input_stride, scale_stride);
		} else {
			quantize_bf16_fp8_strassen_a_kernel<Threads><<<static_cast<unsigned>(blocks), Threads, 0, stream>>>(
					output, scales, input, rows / 2, hidden_size, input_stride, scale_stride);
		}
	} else {
		if (vectorize8) {
			quantize_bf16_fp8_kernel<Threads, 8><<<static_cast<unsigned>(blocks), Threads, 0, stream>>>(
					output, scales, input, rows, hidden_size, input_stride, scale_stride);
		} else {
			quantize_bf16_fp8_kernel<Threads><<<static_cast<unsigned>(blocks), Threads, 0, stream>>>(
					output, scales, input, rows, hidden_size, input_stride, scale_stride);
		}
	}
	return cudaGetLastError();
}

template <int Threads>
__global__ void rmsnorm_and_quantize_kernel(
		__nv_fp8_e4m3* __restrict__ output, float* __restrict__ scales,
		__nv_bfloat16 const* __restrict__ input,
		__nv_bfloat16 const* __restrict__ weight, int hidden_size,
		int64_t input_stride, int64_t scale_stride, float epsilon) {
	using BlockReduce = cub::BlockReduce<float, Threads>;
	__shared__ typename BlockReduce::TempStorage reduce_storage;
	__shared__ float inverse_rms;

	auto const* input_row = reinterpret_cast<Bf16x4 const*>(
			input + int64_t(blockIdx.x) * input_stride);
	auto const* weight_vectors = reinterpret_cast<Bf16x4 const*>(weight);
	auto* output_row = reinterpret_cast<__nv_fp8x4_e4m3*>(
			output + int64_t(blockIdx.x) * hidden_size);

	float sum_squares = 0.0f;
#pragma unroll 4
	for (int vector = threadIdx.x; vector < hidden_size / 4; vector += Threads) {
		Bf16x4 values = input_row[vector];
#pragma unroll
		for (int element = 0; element < 4; ++element) {
			float value = __bfloat162float(values.values[element]);
			sum_squares += value * value;
		}
	}
	sum_squares = BlockReduce(reduce_storage).Sum(sum_squares);
	if (threadIdx.x == 0) {
		inverse_rms = rsqrtf(sum_squares / hidden_size + epsilon);
	}
	__syncthreads();

	int const lane = threadIdx.x % 32;
	int const warp = threadIdx.x / 32;
	for (int group = warp; group < hidden_size / RmsnormQuantizeGroupSize;
			 group += Threads / 32) {
		int const vector = group * 32 + lane;
		Bf16x4 values = input_row[vector];
		Bf16x4 weights = weight_vectors[vector];
		float normalized[4];
		float absmax = 0.0f;
#pragma unroll
		for (int element = 0; element < 4; element += 2) {
			__nv_bfloat162 norm = __floats2bfloat162_rn(
					__bfloat162float(values.values[element]) * inverse_rms,
					__bfloat162float(values.values[element + 1]) * inverse_rms);
			__nv_bfloat162 weight_pair = __halves2bfloat162(
					weights.values[element], weights.values[element + 1]);
			float2 product = __bfloat1622float2(__hmul2(norm, weight_pair));
			normalized[element] = product.x;
			normalized[element + 1] = product.y;
			absmax = fmaxf(absmax, fmaxf(fabsf(product.x), fabsf(product.y)));
		}
		absmax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(absmax)));
		float const scale = fmaxf(absmax / 448.0f, RmsnormQuantizeMinScale);
		if (lane == 0) {
			scales[int64_t(group) * scale_stride + blockIdx.x] = scale;
		}
		output_row[vector] = __nv_fp8x4_e4m3(divide_fp8_quad(normalized, scale));
	}
}

struct RmsPairSum {
	__device__ float2 operator()(float2 left, float2 right) const {
		return make_float2(left.x + right.x, left.y + right.y);
	}
};

template <int Threads, bool CacheRows = false>
__global__ void rmsnorm_and_quantize_strassen_a_kernel(
		__nv_fp8_e4m3* __restrict__ output, float* __restrict__ scales,
		__nv_bfloat16 const* __restrict__ input,
		__nv_bfloat16 const* __restrict__ weight, int half_rows, int hidden_size,
		int64_t input_stride, int64_t scale_stride, float epsilon) {
	using BlockReduce = cub::BlockReduce<float2, Threads>;
	__shared__ typename BlockReduce::TempStorage reduce_storage;
	__shared__ float2 inverse_rms;
	extern __shared__ Bf16x4 cached_rows[];

	auto const* top = reinterpret_cast<Bf16x4 const*>(input + int64_t(blockIdx.x) * input_stride);
	auto const* bottom = reinterpret_cast<Bf16x4 const*>(
			input + (int64_t(blockIdx.x) + half_rows) * input_stride);
	auto const* weights = reinterpret_cast<Bf16x4 const*>(weight);
	auto* output_vectors = reinterpret_cast<__nv_fp8x4_e4m3*>(output);
	float2 sum_squares = make_float2(0.0f, 0.0f);
#pragma unroll 4
	for (int vector = threadIdx.x; vector < hidden_size / 4; vector += Threads) {
		Bf16x4 top_values = top[vector];
		Bf16x4 bottom_values = bottom[vector];
		if constexpr (CacheRows) {
			cached_rows[vector] = top_values;
			cached_rows[hidden_size / 4 + vector] = bottom_values;
		}
#pragma unroll
		for (int element = 0; element < 4; ++element) {
			float top_value = __bfloat162float(top_values.values[element]);
			float bottom_value = __bfloat162float(bottom_values.values[element]);
			sum_squares.x += top_value * top_value;
			sum_squares.y += bottom_value * bottom_value;
		}
	}
	sum_squares = BlockReduce(reduce_storage).Reduce(sum_squares, RmsPairSum{});
	if (threadIdx.x == 0) {
		inverse_rms = make_float2(rsqrtf(sum_squares.x / hidden_size + epsilon),
				rsqrtf(sum_squares.y / hidden_size + epsilon));
	}
	__syncthreads();

	int const half_vectors = hidden_size / 8;
	int const groups_per_row = hidden_size / (2 * RmsnormQuantizeGroupSize);
	int const lane = threadIdx.x % 32;
	int const warp = threadIdx.x / 32;
	int64_t const matrix_vectors = int64_t(half_rows) * half_vectors;
	auto const* quantize_top = CacheRows ? cached_rows : top;
	auto const* quantize_bottom = CacheRows ? cached_rows + hidden_size / 4 : bottom;
	for (int group = warp; group < groups_per_row; group += Threads / 32) {
		int const vector = group * 32 + lane;
		Bf16x4 quadrants[4] = {quantize_top[vector], quantize_top[half_vectors + vector],
				quantize_bottom[vector], quantize_bottom[half_vectors + vector]};
		Bf16x4 weight_halves[2] = {weights[vector], weights[half_vectors + vector]};
		float normalized[4][4];
#pragma unroll
		for (int quadrant = 0; quadrant < 4; ++quadrant) {
			float const rms = quadrant < 2 ? inverse_rms.x : inverse_rms.y;
#pragma unroll
			for (int element = 0; element < 4; element += 2) {
				__nv_bfloat162 norm = __floats2bfloat162_rn(
						__bfloat162float(quadrants[quadrant].values[element]) * rms,
						__bfloat162float(quadrants[quadrant].values[element + 1]) * rms);
				__nv_bfloat162 weight_pair = __halves2bfloat162(
						weight_halves[quadrant % 2].values[element],
						weight_halves[quadrant % 2].values[element + 1]);
				float2 product = __bfloat1622float2(__hmul2(norm, weight_pair));
				normalized[quadrant][element] = product.x;
				normalized[quadrant][element + 1] = product.y;
			}
		}
		float operands[7][4];
#pragma unroll
		for (int element = 0; element < 4; ++element) {
			operands[0][element] = normalized[0][element];
			operands[1][element] = normalized[1][element];
			operands[2][element] = normalized[3][element];
			operands[3][element] = normalized[0][element] - normalized[2][element];
			operands[4][element] = normalized[2][element] + normalized[3][element];
			operands[5][element] = operands[4][element] - normalized[0][element];
			operands[6][element] = normalized[1][element] - operands[5][element];
		}
#pragma unroll
		for (int matrix = 0; matrix < 7; ++matrix) {
			float absmax = 0.0f;
#pragma unroll
			for (int element = 0; element < 4; ++element) {
				absmax = fmaxf(absmax, fabsf(operands[matrix][element]));
			}
			absmax = __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(absmax)));
			float const scale = matrix < 3 ? fmaxf(absmax / 448.0f, RmsnormQuantizeMinScale) :
					(absmax > 0.0f ? fmaxf(absmax / 448.0f, FLT_MIN) : 1.0f);
			if (lane == 0) {
				scales[rmsnorm_strassen_a_scale_offset(static_cast<StrassenAOperand>(matrix),
						blockIdx.x, group, groups_per_row, scale_stride)] = scale;
			}
			__nv_fp8x4_e4m3 quantized(divide_fp8_quad(operands[matrix], scale));
			output_vectors[matrix * matrix_vectors + int64_t(blockIdx.x) * half_vectors + vector] = quantized;
		}
	}
}

template <int Threads>
inline cudaError_t launch_rmsnorm_strassen_a_with_cache(
		__nv_fp8_e4m3* output, float* scales, __nv_bfloat16 const* input,
		__nv_bfloat16 const* weight, int rows, int hidden_size, float epsilon,
		cudaStream_t stream, int64_t input_stride, int64_t scale_stride) {
	static thread_local int planned_device = -1;
	static thread_local int planned_hidden_size = 0;
	static thread_local size_t shared_bytes = 0;
	int device;
	cudaError_t status = cudaGetDevice(&device);
	if (status != cudaSuccess) return status;
	if (device != planned_device || hidden_size != planned_hidden_size) {
		size_t const requested_bytes = size_t(hidden_size) * 2 * sizeof(__nv_bfloat16);
		int baseline_blocks;
		status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&baseline_blocks,
				rmsnorm_and_quantize_strassen_a_kernel<Threads, false>, Threads, 0);
		if (status != cudaSuccess) return status;
		cudaFuncAttributes attributes;
		status = cudaFuncGetAttributes(&attributes, rmsnorm_and_quantize_strassen_a_kernel<Threads, true>);
		if (status != cudaSuccess) return status;
		int max_shared_bytes;
		status = cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
		if (status != cudaSuccess) return status;
		size_t selected_bytes = 0;
		if (requested_bytes + attributes.sharedSizeBytes <= size_t(max_shared_bytes)) {
			if (requested_bytes > size_t(attributes.maxDynamicSharedSizeBytes)) {
				status = cudaFuncSetAttribute(rmsnorm_and_quantize_strassen_a_kernel<Threads, true>,
						cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_bytes - int(attributes.sharedSizeBytes));
				if (status != cudaSuccess) return status;
			}
			int cached_blocks;
			status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cached_blocks,
					rmsnorm_and_quantize_strassen_a_kernel<Threads, true>, Threads, requested_bytes);
			if (status != cudaSuccess) return status;
			if (cached_blocks >= baseline_blocks && cached_blocks > 0) selected_bytes = requested_bytes;
		}
		shared_bytes = selected_bytes;
		planned_device = device;
		planned_hidden_size = hidden_size;
	}
	if (shared_bytes != 0) {
		rmsnorm_and_quantize_strassen_a_kernel<Threads, true><<<rows / 2, Threads, shared_bytes, stream>>>(
				output, scales, input, weight, rows / 2, hidden_size, input_stride, scale_stride, epsilon);
	} else {
		rmsnorm_and_quantize_strassen_a_kernel<Threads, false><<<rows / 2, Threads, 0, stream>>>(
				output, scales, input, weight, rows / 2, hidden_size, input_stride, scale_stride, epsilon);
	}
	return cudaGetLastError();
}

}

inline cudaError_t launch_rmsnorm_and_quantize(
		__nv_fp8_e4m3* output, float* scales, __nv_bfloat16 const* input,
		__nv_bfloat16 const* weight, int rows, int hidden_size,
		float epsilon = 1.0e-6f, cudaStream_t stream = nullptr,
		int64_t input_stride = 0, int64_t scale_stride = 0) {
	if (rows < 0 || hidden_size <= 0 || hidden_size % RmsnormQuantizeGroupSize != 0 ||
			!std::isfinite(epsilon) || epsilon <= 0.0f) {
		return cudaErrorInvalidValue;
	}
	if (input_stride == 0) input_stride = hidden_size;
	if (scale_stride == 0) scale_stride = rows;
	if (input_stride < hidden_size || input_stride % 4 != 0 || scale_stride < rows) {
		return cudaErrorInvalidValue;
	}
	if (rows == 0) return cudaSuccess;
	if (output == nullptr || scales == nullptr || input == nullptr || weight == nullptr ||
			reinterpret_cast<uintptr_t>(input) % alignof(detail::Bf16x4) != 0 ||
			reinterpret_cast<uintptr_t>(weight) % alignof(detail::Bf16x4) != 0 ||
			reinterpret_cast<uintptr_t>(output) % alignof(detail::Fp8x4) != 0 ||
			reinterpret_cast<uintptr_t>(scales) % alignof(float) != 0) {
		return cudaErrorInvalidValue;
	}
	if (rows <= 256) {
		detail::rmsnorm_and_quantize_kernel<512><<<rows, 512, 0, stream>>>(
				output, scales, input, weight, hidden_size, input_stride, scale_stride, epsilon);
	} else {
		detail::rmsnorm_and_quantize_kernel<256><<<rows, 256, 0, stream>>>(
				output, scales, input, weight, hidden_size, input_stride, scale_stride, epsilon);
	}
	return cudaGetLastError();
}

inline cudaError_t launch_rmsnorm_and_quantize_strassen_a(
		__nv_fp8_e4m3* output, float* scales, __nv_bfloat16 const* input,
		__nv_bfloat16 const* weight, int rows, int hidden_size,
		float epsilon = 1.0e-6f, cudaStream_t stream = nullptr,
		int64_t input_stride = 0, int64_t scale_stride = 0) {
	if (rows < 0 || rows % 2 != 0 || hidden_size <= 0 ||
			hidden_size % (2 * RmsnormQuantizeGroupSize) != 0 ||
			!std::isfinite(epsilon) || epsilon <= 0.0f) {
		return cudaErrorInvalidValue;
	}
	if (input_stride == 0) input_stride = hidden_size;
	if (scale_stride == 0) scale_stride = rows / 2;
	if (input_stride < hidden_size || input_stride % 4 != 0 || scale_stride < rows / 2) {
		return cudaErrorInvalidValue;
	}
	if (rows == 0) return cudaSuccess;
	if (output == nullptr || scales == nullptr || input == nullptr || weight == nullptr ||
			reinterpret_cast<uintptr_t>(input) % alignof(detail::Bf16x4) != 0 ||
			reinterpret_cast<uintptr_t>(weight) % alignof(detail::Bf16x4) != 0 ||
			reinterpret_cast<uintptr_t>(output) % alignof(detail::Fp8x4) != 0 ||
			reinterpret_cast<uintptr_t>(scales) % alignof(float) != 0) {
		return cudaErrorInvalidValue;
	}
	if (rows <= 256) {
		return detail::launch_rmsnorm_strassen_a_with_cache<512>(
				output, scales, input, weight, rows, hidden_size, epsilon, stream, input_stride, scale_stride);
	}
	return detail::launch_rmsnorm_strassen_a_with_cache<256>(
			output, scales, input, weight, rows, hidden_size, epsilon, stream, input_stride, scale_stride);
}

inline cudaError_t launch_quantize_bf16_fp8(
		__nv_fp8_e4m3* output, float* scales, __nv_bfloat16 const* input,
		int rows, int hidden_size, cudaStream_t stream = nullptr,
		int64_t input_stride = 0, int64_t scale_stride = 0) {
	return detail::launch_quantize_bf16_fp8_impl<false>(
			output, scales, input, rows, hidden_size, stream, input_stride, scale_stride);
}

inline cudaError_t launch_quantize_bf16_fp8_strassen_a(
		__nv_fp8_e4m3* output, float* scales, __nv_bfloat16 const* input,
		int rows, int hidden_size, cudaStream_t stream = nullptr,
		int64_t input_stride = 0, int64_t scale_stride = 0) {
	return detail::launch_quantize_bf16_fp8_impl<true>(
			output, scales, input, rows, hidden_size, stream, input_stride, scale_stride);
}

}
