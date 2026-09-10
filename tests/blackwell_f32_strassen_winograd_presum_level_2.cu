#include "cuda/kernels/blackwell/strassen_winograd/blackwell_f32_sw_interleaved_presum_level_2_128x256.cuh"
#include "base_test.cuh"

static std::vector<strassen_tests::TestCase> test_cases() {
	return {
		{{8192, 8192, 8192}, 1, 2, "8192x8192x8192"},
		{{4096, 4096, 4096}, 1, 2, "4096x4096x4096"},
		{{6144, 8192, 4096}, 1, 2, "6144x8192x4096"},
		{{5120, 5120, 9216}, 1, 2, "5120x5120x9216"},
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