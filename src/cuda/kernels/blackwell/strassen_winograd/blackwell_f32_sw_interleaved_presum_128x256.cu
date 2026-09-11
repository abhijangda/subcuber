#include "cuda/kernels/blackwell/strassen_winograd/blackwell_f32_sw_interleaved_presum.cuh"
#include "cuda/kernel_runner_support.cuh"

using BlackwellF32SWInterleavedPresum_128x256 = BlackwellF32SWInterleavedPresum<
  cute::Shape<cute::_128, cute::_256, cute::_16>,
  cute::Shape<cute::_128, cute::_256, cute::_16>,
  5, cute::Shape<cute::_4, cute::_256>, cute::Shape<cute::_4, cute::_256>>;

STRASSEN_RUNNER_EXPORT_CUTLASS3(run_blackwell_f32_sw_interleaved_presum_128x256,
                                BlackwellF32SWInterleavedPresum_128x256)
