#include "hopper_f16_tensorop_strassen_winograd_level_2_cooperative_tma_reduce.cuh"

TEST_P(F16HopperStrassenWinogradLevel2CooperativeTmaReduceTest, MatchesReference_TNT) {
	strassen_tests::run_gtest_case<HopperF16InterleavedPresumLevel2CooperativeMaxFusionTmaReduce_2x256_2x256_OptNo<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>, ElementA, ElementB, ElementD>(GetParam());
}

TEST_P(F16HopperStrassenWinogradLevel2CooperativeTmaReduceTest, MatchesReference4x256_TNT) {
	strassen_tests::run_gtest_case<HopperF16InterleavedPresumLevel2CooperativeMaxFusionTmaReduce_4x256_4x256_OptNo<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>, ElementA, ElementB, ElementD>(GetParam());
}