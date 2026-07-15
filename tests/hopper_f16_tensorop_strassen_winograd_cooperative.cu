#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion.cuh"

#include "base_test.cuh"

class F16HopperStrassenWinogradCooperativeTest : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{8192, 8192, 8192}, 1, 1, "8192x8192x8192 split_k=1"},
		{{12288, 12288, 8192}, 1, 1, "12288x12288x8192 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradCooperativeTest, MatchesReference) {
	strassen_tests::run_gtest_case<HopperF16InterleavedPresumCooperativeMaxFusion_2x256_2x256_OptNo, ElementA, ElementB, ElementC>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16HopperStrassenWinogradCooperativeTest,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}