#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_level_2_cooperative_pingpong_max_fusion_tma_reduce.cuh"

#include "base_test.cuh"

class F16HopperStrassenWinogradLevel2CooperativePingpongTmaReduceTest : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{16384, 16384, 16384}, 1, 2, "16384x16384x16384 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradLevel2CooperativePingpongTmaReduceTest, MatchesReference) {
	strassen_tests::run_gtest_case<HopperF16InterleavedPresumLevel2CooperativePingpongMaxFusionTmaReduce_2x256_2x256_OptNo, ElementA, ElementB, ElementD>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16HopperStrassenWinogradLevel2CooperativePingpongTmaReduceTest,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}