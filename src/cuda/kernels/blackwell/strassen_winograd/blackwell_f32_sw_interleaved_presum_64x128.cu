#include "cuda/kernels/blackwell/strassen_winograd/blackwell_f32_sw_interleaved_presum.cuh"
#include "cuda/kernel_runner_support.cuh"

using BlackwellF32SWInterleavedPresum_64x128 = BlackwellF32SWInterleavedPresum<
  cute::Shape<cute::_64, cute::_128, cute::_16>,
  cute::Shape<cute::_64, cute::_128, cute::_16>,
  3, cute::Shape<cute::_4, cute::_128>, cute::Shape<cute::_4, cute::_128>>;

STRASSEN_RUNNER_EXPORT_CUTLASS3(run_blackwell_f32_sw_interleaved_presum_64x128,
                                BlackwellF32SWInterleavedPresum_64x128)
