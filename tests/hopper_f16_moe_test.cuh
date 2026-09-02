#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "cuda/kernel_runner_support.cuh"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "base_test.cuh"

namespace strassen_tests {

using MoeRunner = int (*)(
	KernelRunnerBuffers, int, int, int, int, int,
	cudaStream_t *, int, int, float *);

struct MoeTestCase {
	cutlass::gemm::GemmCoord problem_size;
	int groups;
	char const *name;
};

inline std::vector<MoeTestCase> moe_test_cases() {
	return {
		{{8192, 8192, 8192}, 1, "8192x8192x8192_groups_1"},
		{{8192, 8192, 8192}, 3, "8192x8192x8192_groups_3"},
		{{12288, 12288, 8192}, 1, "12288x12288x8192_groups_1"},
		{{12288, 12288, 8192}, 3, "12288x12288x8192_groups_3"},
	};
}

inline std::string moe_gtest_case_name(testing::TestParamInfo<MoeTestCase> const &info) {
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

inline cudaError_t compare_moe_group_results(
	ElementC const *actual,
	ElementC const *expected,
	std::size_t element_count,
	int n) {
	constexpr std::size_t kComparisonChunkElements = 1 << 20;
	std::vector<ElementC> host_actual(kComparisonChunkElements);
	std::vector<ElementC> host_expected(kComparisonChunkElements);

	for (std::size_t offset = 0; offset < element_count; offset += kComparisonChunkElements) {
		std::size_t count = std::min(kComparisonChunkElements, element_count - offset);
		cudaError_t result = cudaMemcpy(
			host_actual.data(), actual + offset, count * sizeof(ElementC), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			return result;
		}
		result = cudaMemcpy(
			host_expected.data(), expected + offset, count * sizeof(ElementC), cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			return result;
		}

		host_actual.resize(count);
		host_expected.resize(count);
		if (!compare_results(host_actual, host_expected, n, 1)) {
			return cudaErrorUnknown;
		}
		host_actual.resize(kComparisonChunkElements);
		host_expected.resize(kComparisonChunkElements);
	}

	return cudaSuccess;
}

inline cudaError_t run_moe_case(MoeTestCase const &test_case, MoeRunner runner) {
	using DeviceGemmReference = cutlass::reference::device::Gemm<
		ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
		ElementAccumulator, ElementAccumulator>;

	int m = test_case.problem_size.m();
	int n = test_case.problem_size.n();
	int k = test_case.problem_size.k();
	std::size_t elements_a = std::size_t(test_case.groups) * std::size_t(m) * std::size_t(k);
	std::size_t elements_b = std::size_t(k) * std::size_t(n);
	std::size_t elements_d_per_group = std::size_t(m) * std::size_t(n);
	std::size_t elements_d = std::size_t(test_case.groups) * elements_d_per_group;

	cutlass::device_memory::allocation<ElementA> tensor_a(elements_a);
	cutlass::device_memory::allocation<ElementB> tensor_b(elements_b);
	cutlass::device_memory::allocation<ElementC> tensor_d(elements_d);
	cutlass::device_memory::allocation<ElementC> tensor_ref_d(elements_d_per_group);

	cutlass::reference::device::BlockFillRandomUniform(
		tensor_a.get(), elements_a, 2021, ElementA(4), ElementA(-4), 0);
	cutlass::reference::device::BlockFillRandomUniform(
		tensor_b.get(), elements_b, 2022, ElementB(4), ElementB(-4), 0);
	cudaError_t result = cudaGetLastError();
	if (result != cudaSuccess) {
		return result;
	}

	cudaStream_t stream = nullptr;
	result = cudaStreamCreate(&stream);
	if (result != cudaSuccess) {
		return result;
	}

	float average_ms = 0.0f;
	int runner_result = runner(
		{tensor_a.get(), tensor_b.get(), nullptr, tensor_d.get()},
		m, n, k, 0, 1, &stream, 1, test_case.groups, &average_ms);
	cudaError_t destroy_result = cudaStreamDestroy(stream);
	if (runner_result != 0) {
		std::cerr << "MoE CUTLASS runner failed with status " << runner_result << std::endl;
		return cudaErrorUnknown;
	}
	if (destroy_result != cudaSuccess) {
		return destroy_result;
	}

	DeviceGemmReference gemm_reference;
	for (int group = 0; group < test_case.groups; ++group) {
		result = cudaMemset(tensor_ref_d.get(), 0, elements_d_per_group * sizeof(ElementC));
		if (result != cudaSuccess) {
			return result;
		}

		ElementA *group_a = tensor_a.get() + std::size_t(group) * std::size_t(m) * std::size_t(k);
		cutlass::TensorRef ref_a(group_a, LayoutA::packed({m, k}));
		cutlass::TensorRef ref_b(tensor_b.get(), LayoutB::packed({k, n}));
		cutlass::TensorRef ref_d(tensor_ref_d.get(), LayoutC::packed({m, n}));
		gemm_reference(
			{m, n, k}, ElementAccumulator(1), ref_a, ref_b,
			ElementAccumulator(0), ref_d, ref_d);

		result = cudaDeviceSynchronize();
		if (result != cudaSuccess) {
			return result;
		}

		ElementC const *group_d = tensor_d.get() + std::size_t(group) * elements_d_per_group;
		result = compare_moe_group_results(group_d, tensor_ref_d.get(), elements_d_per_group, n);
		if (result != cudaSuccess) {
			std::cerr << "MoE output mismatch in group " << group << std::endl;
			return result;
		}
	}

	return cudaSuccess;
}

inline void run_moe_gtest_case(MoeTestCase const &test_case, MoeRunner runner) {
	cudaDeviceProp properties;
	cudaError_t result = cudaGetDeviceProperties(&properties, 0);
	ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);
	if (properties.major < 9) {
		GTEST_SKIP() << "Device does not support Hopper Strassen-Winograd tests.";
	}

	result = run_moe_case(test_case, runner);
	ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);
}

}  // namespace strassen_tests