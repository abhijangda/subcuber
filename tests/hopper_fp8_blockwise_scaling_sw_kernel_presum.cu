#include "cuda/kernels/hopper/strassen_winograd/hopper_fp8_blockwise_scaling_sw_kernel_presum_cooperative_max_fusion_tma_reduce.cuh"

#include "base_test.cuh"

class Fp8HopperBlockwiseScalingSwKernelPresumTest
    : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases() {
  return {
      {{16384, 16384, 8192}, 1, 1, "16384x16384x8192", 1},
  };
}

TEST_P(Fp8HopperBlockwiseScalingSwKernelPresumTest, SequentialMatchesReference) {
  strassen_tests::run_gtest_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusion_4x128_4x128_OptNo_1x2,
                                ElementA, ElementB, ElementC>(GetParam());
}

// TEST_P(Fp8HopperBlockwiseScalingSwKernelPresumTest, ParallelMatchesReference) {
//   strassen_tests::run_gtest_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusion_4x128_4x128_OptNo_1x2_Par,
//                                 ElementA, ElementB, ElementC>(GetParam());
// }

// TEST_P(Fp8HopperBlockwiseScalingSwKernelPresumTest, TmaReduceSequentialMatchesReference) {
//   strassen_tests::run_gtest_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusionTmaReduce_4x128_4x128_OptNo_1x2,
//                                 ElementA, ElementB, ElementC>(GetParam());
// }

// TEST_P(Fp8HopperBlockwiseScalingSwKernelPresumTest, TmaReduceParallelMatchesReference) {
//   strassen_tests::run_gtest_case<HopperFp8BlockwiseScalingSwKernelPresumCooperativeMaxFusionTmaReduce_4x128_4x128_OptNo_1x2_Par,
//                                 ElementA, ElementB, ElementC>(GetParam());
// }

INSTANTIATE_TEST_SUITE_P(
    ,
    Fp8HopperBlockwiseScalingSwKernelPresumTest,
    testing::ValuesIn(test_cases()),
    strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
