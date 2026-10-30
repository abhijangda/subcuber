#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "src/cuda/allocate_float_matrices.cuh"
#include "src/cuda/cubic_reference.cuh"
#include "src/cuda/strassen_reference_l1.cuh"

#if defined(CUTLASS_API_v3)
#include <omp.h>

#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#endif

namespace strassen_tests {

struct TestCase {
	cutlass::gemm::GemmCoord problem_size;
	int split_k_slices;
	int level;
	const char *name;
	int num_streams = 1;
};

struct BlockScales {
	std::vector<float> a;
	std::vector<float> b;
};

inline void PrintTo(TestCase const &test_case, std::ostream *os) {
	*os << test_case.name;
}

enum KernelType {
	Threadblock,
	Presum
};

enum GPUArch {
	Ampere,
	Hopper
};

enum DataType {
	F32,
	F16
};

inline std::string gtest_case_name(testing::TestParamInfo<TestCase> const &info) {
	std::string name = info.param.name;
	for (char &ch : name) {
		bool valid = (ch >= 'a' && ch <= 'z') ||
								 (ch >= 'A' && ch <= 'Z') ||
								 (ch >= '0' && ch <= '9');
		if (!valid) {
			ch = '_';
		}
	}

	return name;
}

inline bool device_supports_tests(int minimum_capability = 80) {
	cudaDeviceProp props;
	cudaError_t error = cudaGetDeviceProperties(&props, 0);
	if (error != cudaSuccess) {
		std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(error) << std::endl;
		return false;
	}

	if ((props.major * 10 + props.minor) < minimum_capability) {
		std::cout << "Skipping Strassen-Winograd tests: compute capability "
							<< props.major << "." << props.minor << " is below sm"
							<< minimum_capability << "." << std::endl;
		return false;
	}

	return true;
}

template<typename ElementOutput, typename ElementInput = ElementOutput>
inline bool compare_results(
	std::vector<ElementOutput> const &actual,
	std::vector<ElementOutput> const &expected,
	int n, int level) {
	static_assert(std::is_same<ElementOutput, float>::value || std::is_same<ElementOutput, cutlass::half_t>::value);

	constexpr bool is_fp8 = std::is_same_v<ElementInput, cutlass::float_e4m3_t> ||
		std::is_same_v<ElementInput, cutlass::float_e5m2_t>;
	float max_rel_tol = is_fp8 ? 10*1e-2f:
		(std::is_same<ElementOutput, float>::value ? 1e-4f : (level == 1 ? 1e-2f : 1e-1f + 1e-2f));
	float max_abs_tol = is_fp8 ? 20.0f :
		(std::is_same<ElementOutput, float>::value ? 1e-4f : (level == 1 ? 4 : 14));
	for (std::size_t idx = 0; idx < actual.size(); ++idx) {
		float cutlass_value = static_cast<float>(actual[idx]);
		float reference_value = static_cast<float>(expected[idx]);
		float rel_diff;
		float abs_diff;
		bool equal = false;
		if ((reference_value == 0.0f && cutlass_value == 0.0f) ||
				(std::isnan(reference_value) && std::isnan(cutlass_value))) {
			equal = true;
		} else if (!std::isnan(reference_value) && !std::isnan(cutlass_value)) {
			abs_diff = std::fabs(reference_value - cutlass_value);
			rel_diff = (std::fabs(reference_value - cutlass_value) / std::fabs(reference_value + 1.0e-6f));
			equal = (abs_diff <= max_abs_tol || rel_diff <= max_rel_tol);
		}

		if (!equal) {
			std::cerr << "Mismatch at (" << (idx / n) << ", " << (idx % n) << "): "
								<< cutlass_value << " != " << reference_value << " rel_diff = " << rel_diff << " abs_diff = " << abs_diff << std::endl;
			return false;
		}
	}

	return true;
}

template<typename CutlassGemm,
				 typename TensorRefA, typename TensorRefB,
		 		 typename TensorRefC>
inline cudaError_t run_cutlass_gemm(
	TestCase const &test_case,
	TensorRefA ref_a,
	TensorRefB ref_b,
	TensorRefC ref_c,
	TensorRefC ref_d,
	std::array<cudaStream_t, 49> &streams,
	BlockScales *block_scales = nullptr) {

	auto problem_size = test_case.problem_size;
#ifdef CUTLASS_API_v2
	int ldc = problem_size.n();

	CutlassGemm gemm_operator;
	typename CutlassGemm::Arguments args(
		problem_size,
		ref_a,
		ref_b,
		ref_c,
		ref_d,
		{1.0f, 0.0f},
		test_case.split_k_slices);

	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	cutlass::Status status = gemm_operator.initialize(args, workspace.get());
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS initialize failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	auto run_streams = streams;
	status = gemm_operator.run(run_streams.data(), test_case.num_streams);
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS run failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	cudaError_t result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		std::cerr << "CUTLASS synchronization failed: " << cudaGetErrorString(result) << std::endl;
	}

	return result;

#elif CUTLASS_API_v3
	using CollectiveMainloop = typename CutlassGemm::GemmKernel::CollectiveMainloop;
	using StrideA = typename CutlassGemm::GemmKernel::StrideA;
	using StrideB = typename CutlassGemm::GemmKernel::StrideB;
	using StrideC = typename CutlassGemm::GemmKernel::StrideC;
	using StrideD = typename CutlassGemm::GemmKernel::StrideD;

	StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {problem_size.m(), problem_size.k(), 1});
	StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {problem_size.n(), problem_size.k(), 1});
	StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {problem_size.m(), problem_size.n(), 1});
	StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {problem_size.m(), problem_size.n(), 1});

	int device_id = 0;
	cutlass::KernelHardwareInfo kernel_hw_info;
	kernel_hw_info.device_id = device_id;
	kernel_hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
	auto kernel_problem_size = [&] {
		if constexpr (cute::rank(typename CutlassGemm::GemmKernel::ProblemShape{}) == 4) {
			return typename CutlassGemm::GemmKernel::ProblemShape{
				problem_size.m(), problem_size.n(), problem_size.k(), 1};
		} else {
			return typename CutlassGemm::GemmKernel::ProblemShape{
				problem_size.m(), problem_size.n(), problem_size.k()};
		}
	}();

	typename CutlassGemm::GemmKernel::ElementC const *ptr_c = nullptr;
	if constexpr (!std::is_void_v<typename CutlassGemm::GemmKernel::ElementC>) {
		ptr_c = ref_c.data();
	}

	cutlass::device_memory::allocation<float> scale_a;
	cutlass::device_memory::allocation<float> scale_b;
	BlockScales local_scales;
	BlockScales &host_scales = block_scales ? *block_scales : local_scales;
	cudaError_t scale_status = cudaSuccess;
	auto args = [&] {
		if constexpr (requires { CollectiveMainloop::ScaleGranularityK; }) {
			using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
				CollectiveMainloop::ScaleGranularityM,
				CollectiveMainloop::ScaleGranularityN,
				CollectiveMainloop::ScaleGranularityK>;
			auto shape = cute::make_shape(problem_size.m(), problem_size.n(), problem_size.k(), 1);
			auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(shape);
			auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(shape);
			auto &host_scale_a = host_scales.a;
			auto &host_scale_b = host_scales.b;
			host_scale_a.resize(cute::size(cute::filter_zeros(layout_sfa)));
			host_scale_b.resize(cute::size(cute::filter_zeros(layout_sfb)));
			cutlass::reference::host::BlockFillRandomUniform(
				host_scale_a.data(), host_scale_a.size(), 2025, 4.0, -4.0);
			cutlass::reference::host::BlockFillRandomUniform(
				host_scale_b.data(), host_scale_b.size(), 2026, 4.0, -4.0);
			scale_a.reset(host_scale_a.size());
			scale_b.reset(host_scale_b.size());
			scale_status = cudaMemcpy(scale_a.get(), host_scale_a.data(),
				host_scale_a.size() * sizeof(float), cudaMemcpyHostToDevice);
			if (scale_status == cudaSuccess) {
				scale_status = cudaMemcpy(scale_b.get(), host_scale_b.data(),
					host_scale_b.size() * sizeof(float), cudaMemcpyHostToDevice);
			}
			typename CutlassGemm::Arguments arguments{
				cutlass::gemm::GemmUniversalMode::kGemm, kernel_problem_size,
				{ref_a.data(), stride_a, ref_b.data(), stride_b,
				 scale_a.get(), layout_sfa, scale_b.get(), layout_sfb},
				{{}, ptr_c, stride_c, ref_d.data(), stride_d}, kernel_hw_info};
			arguments.scheduler.raster_order = decltype(arguments.scheduler.raster_order)::AlongN;
			arguments.scheduler.max_swizzle_size = 1;
			return arguments;
		} else {
			return typename CutlassGemm::Arguments{
				cutlass::gemm::GemmUniversalMode::kGemm, kernel_problem_size,
				{ref_a.data(), stride_a, ref_b.data(), stride_b},
				{{1.0f, 0.0f}, ptr_c, stride_c, ref_d.data(), stride_d}, kernel_hw_info};
		}
	}();
	if (scale_status != cudaSuccess) {
		return scale_status;
	}

	CutlassGemm gemm_operator;
	cutlass::Status status = CutlassGemm::can_implement(args);
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS can_implement failed with status " << int(status) << std::endl;
		return cudaErrorNotSupported;
	}

	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
	int swizzles[7] = {1, 1, 1, 1, 1, 1, 1};

	status = gemm_operator.initialize(args, swizzles, workspace.get());
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS initialize failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	auto run_streams = streams;
	status = gemm_operator.run(run_streams.data(), test_case.num_streams);
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS run failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	cudaError_t result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		std::cerr << "CUTLASS synchronization failed: " << cudaGetErrorString(result) << std::endl;
	}

	if constexpr (requires { CollectiveMainloop::ScaleGranularityK; }) {
		if (result == cudaSuccess) {
			std::vector<float> device_scale_a(host_scales.a.size());
			std::vector<float> device_scale_b(host_scales.b.size());
			result = cudaMemcpy(device_scale_a.data(), scale_a.get(),
				device_scale_a.size() * sizeof(float), cudaMemcpyDeviceToHost);
			if (result == cudaSuccess) {
				result = cudaMemcpy(device_scale_b.data(), scale_b.get(),
					device_scale_b.size() * sizeof(float), cudaMemcpyDeviceToHost);
			}
			if (result == cudaSuccess &&
					(device_scale_a != host_scales.a || device_scale_b != host_scales.b)) {
				std::cerr << "FP8 device scales differ from the host reference scales." << std::endl;
				return cudaErrorUnknown;
			}
		}
	}

	return result;

#else
	return cudaErrorNotSupported;
#endif
}

#if defined(CUTLASS_API_v3)
template<typename CutlassGemm, typename TensorRefA, typename TensorRefB, typename TensorRefD>
inline cudaError_t run_block_scaled_reference(
	TestCase const &test_case, TensorRefA ref_a, TensorRefB ref_b, TensorRefD ref_d,
	BlockScales const &block_scales) {
	using namespace cute;
	using Kernel = typename CutlassGemm::GemmKernel;
	using Mainloop = typename Kernel::CollectiveMainloop;
	using ElementA = typename Kernel::ElementA;
	using ElementB = typename Kernel::ElementB;
	using ElementD = typename Kernel::ElementD;
	using ElementAccumulator = float;
	using ElementBlockScale = float;
	using StrideA = typename Kernel::StrideA;
	using StrideB = typename Kernel::StrideB;
	using StrideD = typename Kernel::StrideD;
	constexpr int ScaleK = Mainloop::ScaleGranularityK;
	constexpr int ScaleN = Mainloop::ScaleGranularityN;
	static_assert(Mainloop::ScaleGranularityM == 1);
	using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, ScaleN, ScaleK>;
	auto problem = test_case.problem_size;
	int m = problem.m();
	int n = problem.n();
	int k = problem.k();
	const int halfM = m/2;
	const int halfN = n/2;
	const int halfK = k/2;
	if (test_case.level != 1 || m % 2 || n % 2 || halfN % ScaleN || halfK % ScaleK || k % 2) {
		return cudaErrorNotSupported;
	}
	auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
	auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
	auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
	auto problem_shape = make_shape(m, n, k, 1);
	auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
	auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);
	auto A = make_tensor(ref_a.data(), make_layout(make_shape(m,k,1), stride_A));
	auto B = make_tensor(ref_b.data(), make_layout(make_shape(n,k,1), stride_B));
	auto SFA = make_tensor(block_scales.a.data(), layout_sfa);
	auto SFB = make_tensor(block_scales.b.data(), layout_sfb);

	auto half_shape = make_shape(halfM,halfN,halfK,1);
	auto presum_layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(half_shape);
	auto presum_layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(half_shape);
	auto presum_stride_a = cutlass::make_cute_packed_stride(StrideA{}, {halfM,halfK,1});
	auto presum_stride_b = cutlass::make_cute_packed_stride(StrideB{}, {halfN,halfK,1});
	auto presum_layout_a = make_layout(make_shape(halfM,halfK,1), presum_stride_a);
	auto presum_layout_b = make_layout(make_shape(halfN,halfK,1), presum_stride_b);
	std::array<std::vector<ElementA>,4> reference_presum_A;
	std::array<std::vector<ElementB>,4> presum_b;
	std::array<std::vector<ElementBlockScale>,4> presum_scale_a;
	std::array<std::vector<ElementBlockScale>,4> presum_scale_b;
	for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
		reference_presum_A[presum_idx].resize(size_t(halfM)*halfK);
		presum_b[presum_idx].resize(size_t(halfN)*halfK);
		presum_scale_a[presum_idx].resize(size(filter_zeros(presum_layout_sfa)));
		presum_scale_b[presum_idx].resize(size(filter_zeros(presum_layout_sfb)));
	}

	int available_cpus = omp_get_num_procs();
	int reference_threads = available_cpus > 1 ? available_cpus / 2 : 1;
	int previous_threads = omp_get_max_threads();
	int previous_dynamic = omp_get_dynamic();
	omp_set_dynamic(0);
	omp_set_num_threads(reference_threads);

	#pragma omp parallel for collapse(2)
	for (int row = 0; row < halfM; ++row) {
		for (int block = 0; block < halfK/ScaleK; ++block) {
			std::array<std::array<ElementAccumulator,ScaleK>,4> values;
			std::array<ElementAccumulator,4> max_abs{};
			for (int element_idx = 0; element_idx < ScaleK; ++element_idx) {
				int inner = block*ScaleK + element_idx;
				float a0 = float(A(row,inner,0))*SFA(row,inner,0);
				float a1 = float(A(row,inner+halfK,0))*SFA(row,inner+halfK,0);
				float a2 = float(A(row+halfM,inner,0))*SFA(row+halfM,inner,0);
				float a3 = float(A(row+halfM,inner+halfK,0))*SFA(row+halfM,inner+halfK,0);
				values[0][element_idx] = a2 + a3;
				values[1][element_idx] = values[0][element_idx] - a0;
				values[2][element_idx] = a0 - a2;
				values[3][element_idx] = a1 - values[1][element_idx];
				for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
					max_abs[presum_idx] = std::max(max_abs[presum_idx], std::fabs(values[presum_idx][element_idx]));
				}
			}
			for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
				float scale = max_abs[presum_idx] > 0.0f ?
					std::max(max_abs[presum_idx]/float(cutlass::platform::numeric_limits<ElementA>::max()), FLT_MIN) : 1.0f;
				presum_scale_a[presum_idx][presum_layout_sfa(make_coord(row,block*ScaleK,0))] = scale;
				for (int element_idx = 0; element_idx < ScaleK; ++element_idx) {
					auto coord = make_coord(row,block*ScaleK+element_idx,0);
					reference_presum_A[presum_idx][presum_layout_a(coord)] = ElementA(values[presum_idx][element_idx]/scale);
				}
			}
		}
	}

	#pragma omp parallel for collapse(2)
	for (int block_n = 0; block_n < halfN/ScaleN; ++block_n) {
		for (int block_k = 0; block_k < halfK/ScaleK; ++block_k) {
			std::array<std::array<ElementAccumulator,ScaleN*ScaleK>,4> values;
			std::array<ElementAccumulator,4> max_abs{};
			for (int column_idx = 0; column_idx < ScaleN; ++column_idx) {
				for (int inner_idx = 0; inner_idx < ScaleK; ++inner_idx) {
					int column = block_n*ScaleN + column_idx;
					int inner = block_k*ScaleK + inner_idx;
					int element_idx = column_idx*ScaleK + inner_idx;
					float b0 = float(B(column,inner,0))*SFB(column,inner,0);
					float b1 = float(B(column+halfN,inner,0))*SFB(column+halfN,inner,0);
					float b2 = float(B(column,inner+halfK,0))*SFB(column,inner+halfK,0);
					float b3 = float(B(column+halfN,inner+halfK,0))*SFB(column+halfN,inner+halfK,0);
					values[0][element_idx] = b1 - b0;
					values[1][element_idx] = b3 - b1;
					values[2][element_idx] = values[1][element_idx] + b0;
					values[3][element_idx] = values[2][element_idx] - b2;
					for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
						max_abs[presum_idx] = std::max(max_abs[presum_idx], std::fabs(values[presum_idx][element_idx]));
					}
				}
			}
			for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
				float scale = max_abs[presum_idx] > 0.0f ?
					std::max(max_abs[presum_idx]/float(cutlass::platform::numeric_limits<ElementB>::max()), FLT_MIN) : 1.0f;
				presum_scale_b[presum_idx][presum_layout_sfb(make_coord(block_n*ScaleN,block_k*ScaleK,0))] = scale;
				for (int column_idx = 0; column_idx < ScaleN; ++column_idx) {
					for (int inner_idx = 0; inner_idx < ScaleK; ++inner_idx) {
						auto coord = make_coord(block_n*ScaleN+column_idx,block_k*ScaleK+inner_idx,0);
						presum_b[presum_idx][presum_layout_b(coord)] = ElementB(values[presum_idx][column_idx*ScaleK+inner_idx]/scale);
					}
				}
			}
		}
	}

	auto layout_a = make_layout(make_shape(halfM,halfK,1), stride_A);
	auto layout_b = make_layout(make_shape(halfN,halfK,1), stride_B);
	auto quadrant_layout_sfa = make_layout(shape(presum_layout_sfa), stride(layout_sfa));
	auto quadrant_layout_sfb = make_layout(shape(presum_layout_sfb), stride(layout_sfb));
	auto A0 = make_tensor(A.data(), layout_a);
	auto A1 = make_tensor(A.data()+A.layout()(make_coord(0,halfK,0)), layout_a);
	auto A3 = make_tensor(A.data()+A.layout()(make_coord(halfM,halfK,0)), layout_a);
	auto B0 = make_tensor(B.data(), layout_b);
	auto B2 = make_tensor(B.data()+B.layout()(make_coord(0,halfK,0)), layout_b);
	auto B3 = make_tensor(B.data()+B.layout()(make_coord(halfN,halfK,0)), layout_b);
	auto SFA0 = make_tensor(SFA.data(), quadrant_layout_sfa);
	auto SFA1 = make_tensor(SFA.data()+layout_sfa(make_coord(0,halfK,0)), quadrant_layout_sfa);
	auto SFA3 = make_tensor(SFA.data()+layout_sfa(make_coord(halfM,halfK,0)), quadrant_layout_sfa);
	auto SFB0 = make_tensor(SFB.data(), quadrant_layout_sfb);
	auto SFB2 = make_tensor(SFB.data()+layout_sfb(make_coord(0,halfK,0)), quadrant_layout_sfb);
	auto SFB3 = make_tensor(SFB.data()+layout_sfb(make_coord(halfN,halfK,0)), quadrant_layout_sfb);
	auto input_presum_a = [&](int presum_idx) { return make_tensor(reference_presum_A[presum_idx].data(), presum_layout_a); };
	auto input_presum_b = [&](int presum_idx) { return make_tensor(presum_b[presum_idx].data(), presum_layout_b); };
	auto scale_presum_a = [&](int presum_idx) { return make_tensor(presum_scale_a[presum_idx].data(), presum_layout_sfa); };
	auto scale_presum_b = [&](int presum_idx) { return make_tensor(presum_scale_b[presum_idx].data(), presum_layout_sfb); };

	std::array<std::vector<ElementAccumulator>,7> products;
	auto product_layout = make_layout(make_shape(halfM,halfN,1), make_stride(int64_t(halfN),_1{},int64_t(halfM)*halfN));
	auto reference_gemm = [&](int product_idx, auto input_a, auto scale_a, auto input_b, auto scale_b) {
		products[product_idx].resize(size_t(halfM)*halfN);
		auto product = make_tensor(products[product_idx].data(), product_layout);
		cutlass::reference::host::GettBlockScalingMainloopParams<
			ElementAccumulator, decltype(input_a), decltype(scale_a), decltype(input_b), decltype(scale_b)
		> mainloop_params{input_a, scale_a, input_b, scale_b};
		cutlass::reference::host::GettEpilogueParams<
			float, float, ElementAccumulator, float, decltype(product), decltype(product)
		> epilogue_params;
		epilogue_params.D = product;
		cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
	};

	reference_gemm(0, A0, SFA0, B0, SFB0);
	reference_gemm(1, A1, SFA1, B2, SFB2);
	reference_gemm(2, input_presum_a(1), scale_presum_a(1), input_presum_b(2), scale_presum_b(2));
	reference_gemm(3, input_presum_a(2), scale_presum_a(2), input_presum_b(1), scale_presum_b(1));
	reference_gemm(4, input_presum_a(0), scale_presum_a(0), input_presum_b(0), scale_presum_b(0));
	reference_gemm(5, input_presum_a(3), scale_presum_a(3), B3, SFB3);
	reference_gemm(6, A3, SFA3, input_presum_b(3), scale_presum_b(3));

	std::vector<ElementD> host_output(size_t(m)*n);
	auto D = make_tensor(host_output.data(), make_layout(make_shape(m,n,1), stride_D));
	#pragma omp parallel for
	for (int row = 0; row < halfM; ++row) {
		for (int column = 0; column < halfN; ++column) {
			size_t offset = size_t(row)*halfN + column;
			float m0 = products[0][offset];
			float m1 = products[1][offset];
			float m2 = products[2][offset];
			float m3 = products[3][offset];
			float m4 = products[4][offset];
			float m5 = products[5][offset];
			float m6 = products[6][offset];
			float interim_m0 = float(ElementD(m0));
			float sum_m2 = interim_m0 + m2;
			float sum_m3 = sum_m2 + m3;
			D(row,column,0) = ElementD(m0 + m1);
			D(row,column+halfN,0) = ElementD(float(ElementD(sum_m2)) + m4 + m5);
			D(row+halfM,column,0) = ElementD(sum_m3 - m6);
			D(row+halfM,column+halfN,0) = ElementD(float(ElementD(sum_m3)) + m4);
		}
	}
	omp_set_num_threads(previous_threads);
	omp_set_dynamic(previous_dynamic);
	return cudaMemcpy(ref_d.data(), host_output.data(), host_output.size() * sizeof(ElementD), cudaMemcpyHostToDevice);
}
#endif

template<typename CutlassGemm, typename ElementInputA, typename ElementInputB, typename LayoutInputB>
inline void initialize_inputs(
	cutlass::TensorView<ElementInputA, cutlass::layout::RowMajor> view_a,
	cutlass::TensorView<ElementInputB, LayoutInputB> view_b) {
#if defined(CUTLASS_API_v3)
	using Mainloop = typename CutlassGemm::GemmKernel::CollectiveMainloop;
	if constexpr (requires { Mainloop::ScaleGranularityK; }) {
		cutlass::reference::host::BlockFillRandomUniform(
			view_a.data(), view_a.capacity(), 2022, 4, -4, 0);
		cutlass::reference::host::BlockFillRandomUniform(
			view_b.data(), view_b.capacity(), 2023, 4, -4, 0);
	} else
#endif
	{
		int range_end = std::is_same<ElementInputA, float>::value ? 16 : 4;
		int range_start = std::is_same<ElementInputA, float>::value ? -16 : -4;
		cutlass::reference::host::TensorFillRandomUniform(
			view_a, 1, ElementInputA(range_end), ElementInputA(range_start), 2);
		cutlass::reference::host::TensorFillRandomUniform(
			view_b, 1, ElementInputB(range_end), ElementInputB(range_start), 2);
	}
}

template<typename CutlassGemm,
				 typename ElementInputA,
		 		 typename ElementInputB,
		 		 typename ElementOutput_>
inline cudaError_t run_case(TestCase const &test_case) {
#if defined(CUTLASS_API_v3)
	using ElementOutput = typename CutlassGemm::GemmKernel::ElementD;
	using LayoutInputB = cute::conditional_t<
		cutlass::gemm::detail::is_major<1, typename CutlassGemm::GemmKernel::StrideB>(),
		cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
#else
	using ElementOutput = ElementOutput_;
	using LayoutInputB = cutlass::layout::RowMajor;
#endif

	auto problem_size = test_case.problem_size;
	int lda = problem_size.k();
	int ldb = problem_size.n();
	int ldc = problem_size.n();
	size_t sizeof_c = sizeof(ElementOutput) * size_t(problem_size.m()) * size_t(problem_size.n());

	cutlass::HostTensor<ElementInputA, cutlass::layout::RowMajor> tensor_a(
		problem_size.mk());
	cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
		problem_size.kn());  // <- Create matrix B with dimensions K x N
	cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_c(
		problem_size.mn());  // <- Create matrix C with dimensions M x N
	cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_d(
		problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
							// CUTLASS kernel
	cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_ref_d(
		problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
							// reference kernel
	
	initialize_inputs<CutlassGemm>(tensor_a.host_view(), tensor_b.host_view());

	cutlass::reference::host::TensorFill(tensor_c.host_view());
	cutlass::reference::host::TensorFill(tensor_d.host_view());
	cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

	tensor_a.sync_device(); tensor_b.sync_device(); tensor_c.sync_device();
	tensor_d.sync_device(); tensor_ref_d.sync_device();

	cudaError_t result;
	std::array<cudaStream_t, 49> streams{};
	for (cudaStream_t &stream : streams) {
		result = cudaStreamCreate(&stream);
		if (result != cudaSuccess) {
			std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(result) << std::endl;
			for (cudaStream_t created_stream : streams) {
				if (created_stream) {
					cudaStreamDestroy(created_stream);
				}
			}
			return result;
		}
	}

	BlockScales block_scales;
	result = run_cutlass_gemm<CutlassGemm>(
		test_case, tensor_a.device_ref(), tensor_b.device_ref(),
		tensor_c.device_ref(), tensor_d.device_ref(), streams, &block_scales);

	if (result == cudaSuccess) {
		auto run_reference = [&] {
#if defined(CUTLASS_API_v3)
			using Mainloop = typename CutlassGemm::GemmKernel::CollectiveMainloop;
			if constexpr (requires { Mainloop::ScaleGranularityK; }) {
				return run_block_scaled_reference<CutlassGemm>(
					test_case, tensor_a.host_ref(), tensor_b.host_ref(), tensor_ref_d.device_ref(), block_scales);
			} else
#endif
			{
				auto run_row_major_reference = [&](auto ref_b) {
					return ReferenceStrassenWinogradGemm<ElementInputA, ElementInputB, ElementOutput, float, float>(
						problem_size.m(),
						problem_size.n(),
						problem_size.k(),
						test_case.level,
						1.0f,
						tensor_a.device_ref().data(),
						lda,
						ref_b.data(),
						ldb,
						0.0f,
						tensor_ref_d.device_ref().data(),
						ldc);
				};
				if constexpr (!std::is_same_v<LayoutInputB, cutlass::layout::RowMajor>) {
					cutlass::HostTensor<ElementInputB, cutlass::layout::RowMajor> reference_b(problem_size.kn());
					cutlass::reference::host::TensorCopy(reference_b.host_view(), tensor_b.host_view());
					reference_b.sync_device();
					return run_row_major_reference(reference_b.device_ref());
				} else {
					return run_row_major_reference(tensor_b.device_ref());
				}
			}
		};
		result = run_reference();
		if (result != cudaSuccess) {
			std::cerr << "Reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
		}
	}

	if (result == cudaSuccess) {
		std::vector<ElementOutput> host_cutlass(size_t(problem_size.m()) * size_t(problem_size.n()));
		std::vector<ElementOutput> host_reference(size_t(problem_size.m()) * size_t(problem_size.n()));

		result = cudaMemcpy(host_cutlass.data(), tensor_d.device_ref().data(), sizeof_c, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			std::cerr << "Copying CUTLASS result failed: " << cudaGetErrorString(result) << std::endl;
		}

		if (result == cudaSuccess) {
			result = cudaMemcpy(host_reference.data(), tensor_ref_d.device_ref().data(), sizeof_c, cudaMemcpyDeviceToHost);
			if (result != cudaSuccess) {
				std::cerr << "Copying reference result failed: " << cudaGetErrorString(result) << std::endl;
			}
		}

		if (result == cudaSuccess &&
			!compare_results<ElementOutput, ElementInputA>(
				host_cutlass, host_reference, problem_size.n(), test_case.level)) {
			result = cudaErrorUnknown;
		}
	}

	for (cudaStream_t stream : streams) {
		cudaStreamDestroy(stream);
	}

	return result;
}

template<typename CutlassGemm,
				 typename ElementInputA,
		 		 typename ElementInputB,
		 		 typename ElementOutput>
inline void run_gtest_case(TestCase const &test_case) {
	int minimum_capability = 80;
#if defined(CUTLASS_API_v3)
	using Mainloop = typename CutlassGemm::GemmKernel::CollectiveMainloop;
	if constexpr (requires { Mainloop::ScaleGranularityK; }) {
		minimum_capability = 90;
	}
#endif
	if (!device_supports_tests(minimum_capability)) {
		GTEST_SKIP() << "Device does not support these sm" << minimum_capability << " Strassen-Winograd tests.";
	}

	cudaError_t result = run_case<CutlassGemm, ElementInputA, ElementInputB, ElementOutput>(test_case);
	if (result == cudaErrorNotSupported) {
		GTEST_SKIP() << "CUTLASS kernel does not support this problem shape.";
	}
	ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);
}

}  // namespace strassen_tests
