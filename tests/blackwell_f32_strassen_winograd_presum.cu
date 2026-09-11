#include "cuda/kernels/blackwell/strassen_winograd/blackwell_f32_sw_interleaved_presum.cuh"
#include "base_test.cuh"

using BlackwellF32StrassenWinogradPresum = BlackwellF32SWInterleavedPresum<
  cute::Shape<cute::_128, cute::_256, cute::_16>,
  cute::Shape<cute::_128, cute::_256, cute::_16>,
  5, cute::Shape<cute::_4, cute::_256>, cute::Shape<cute::_4, cute::_256>>;

static std::vector<strassen_tests::TestCase> test_cases() {
	return {
		{{8192, 8192, 8192}, 1, 1, "8192x8192x8192"},
		{{2048, 2048, 2048}, 1, 1, "2048x2048x2048"},
		{{6144, 2560, 2048}, 1, 1, "6144x2560x2048"},
		{{2048, 4096, 6144}, 1, 1, "2048x4096x6144"},
	};
}

class BlackwellF32StrassenWinogradPresumTest
    : public testing::TestWithParam<strassen_tests::TestCase> {};

TEST_P(BlackwellF32StrassenWinogradPresumTest, MatchesReference) {
	strassen_tests::run_gtest_case<
		BlackwellF32StrassenWinogradPresum, float, float, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  ,
  BlackwellF32StrassenWinogradPresumTest,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}