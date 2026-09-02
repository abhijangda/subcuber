#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_moe_sw_interleaved_presum_cooperative_max_fusion.cuh"
#include "cuda/kernel_runner_support.cuh"

STRASSEN_RUNNER_EXPORT_MOE_CUTLASS3(run_hopper_f16_moe_sw_interleaved_presum_cooperative_max_fusion_2x256,
                                    HopperF16MoeInterleavedPresumCooperativeMaxFusion_2x256)