#include "cuda/kernels/blackwell/strassen_winograd/blackwell_f32_sw_interleaved_presum_level_2_128x256.cuh"
#include "base_test.cuh"

static std::vector<strassen_tests::TestCase> test_cases() {
	return {
		{{8192, 8192, 8192}, 1, 2, "8192x8192x8192"},
		{{2048, 2048, 2048}, 1, 2, "2048x2048x2048"},
		{{6144, 3072, 2048}, 1, 2, "6144x3072x2048"},
		{{2048, 4096, 6144}, 1, 2, "2048x4096x6144"},
	};
}

class BlackwellF32StrassenWinogradPresumLevel2Test
    : public testing::TestWithParam<strassen_tests::TestCase> {};

TEST_P(BlackwellF32StrassenWinogradPresumLevel2Test, MatchesReference) {
	strassen_tests::run_gtest_case<
		BlackwellF32SWInterleavedPresumLevel2_128x256, float, float, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  ,
  BlackwellF32StrassenWinogradPresumLevel2Test,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}