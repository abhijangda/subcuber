#include "hopper_f16_tensorop_strassen_winograd_level_2_cooperative_tma_reduce.cuh"

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{16384, 16384, 8192}, 1, 2, "16384x16384x8192 split_k=1"},
		{{16384, 16384, 16384}, 1, 2, "16384x16384x16384 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradLevel2CooperativeTmaReduceTest, MatchesReference_TTT) {
	strassen_tests::run_gtest_case<HopperF16InterleavedPresumLevel2CooperativeMaxFusionTmaReduce_2x256_2x256_OptNo<cutlass::layout::RowMajor, cutlass::layout::RowMajor>, ElementA, ElementB, ElementD>(GetParam());
}

TEST_P(F16HopperStrassenWinogradLevel2CooperativeTmaReduceTest, MatchesReference4x256_TTT) {
	strassen_tests::run_gtest_case<HopperF16InterleavedPresumLevel2CooperativeMaxFusionTmaReduce_4x256_4x256_OptNo<cutlass::layout::RowMajor, cutlass::layout::RowMajor>, ElementA, ElementB, ElementD>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16HopperStrassenWinogradLevel2CooperativeTmaReduceTest,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}