#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_cooperative_max_fusion.cuh"

#include "base_test.cuh"

class F16HopperStrassenWinogradCooperativeTest : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{7168, 7168, 7168}, 1, 1, "7168x7168x7168 split_k=1"},
		{{6144, 6144, 4096}, 1, 1, "6144x6144x4096 split_k=1"},
		{{8192, 4096, 8192}, 1, 1, "8192x4096x8192 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradCooperativeTest, MatchesReference) {
	strassen_tests::run_gtest_case<HopperF16InterleavedPresumCooperativeMaxFusion_4x256_4x256_OptNo, ElementA, ElementB, ElementC>(GetParam());
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