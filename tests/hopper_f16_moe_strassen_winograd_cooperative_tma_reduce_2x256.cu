#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_moe_sw_interleaved_presum_cooperative_max_fusion_tma_reduce.cuh"

#include "hopper_f16_moe_test.cuh"

extern "C" int run_hopper_f16_moe_sw_interleaved_presum_cooperative_max_fusion_tma_reduce_2x256(
	KernelRunnerBuffers, int, int, int, int, int, cudaStream_t *, int, int, float *);

class F16HopperMoeCooperativeTmaReduce2x256Test
	: public testing::TestWithParam<strassen_tests::MoeTestCase> {};

TEST_P(F16HopperMoeCooperativeTmaReduce2x256Test, MatchesReference) {
	strassen_tests::run_moe_gtest_case(
		GetParam(), run_hopper_f16_moe_sw_interleaved_presum_cooperative_max_fusion_tma_reduce_2x256);
}

INSTANTIATE_TEST_SUITE_P(
	,
	F16HopperMoeCooperativeTmaReduce2x256Test,
	testing::ValuesIn(strassen_tests::moe_test_cases()),
	strassen_tests::moe_gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}