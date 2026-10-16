#pragma once

#include <cfloat>

#include "cutlass/arch/memory.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/presum_detail.h"

namespace cutlass {
namespace gemm {
namespace device {

template <typename Element>
CUTLASS_GLOBAL
void KernelFill(Element* ptr, size_t count, Element value) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < count;
       idx += size_t(blockDim.x) * gridDim.x) {
    ptr[idx] = value;
  }
}

template <typename GemmKernel0>
CUTLASS_GLOBAL
__launch_bounds__(GemmKernel0::kThreadCount, 1)
void print_kernel(typename GemmKernel0::Params params) {
  using AllPresums = typename GemmKernel0::StrassenMiGroup::AllPresums;

}

template <typename PresumGroup, typename GemmKernel0, uint kThreadCount>
CUTLASS_GLOBAL
__launch_bounds__(kThreadCount, 1)
void KernelPresumGlobalCompute(typename GemmKernel0::Params params, int problem_idx) {
  using ElementA = typename GemmKernel0::ElementA;
  using ElementB = typename GemmKernel0::ElementB;

  using AllPresums = typename PresumGroup::AllPresums;
  using PresumVecType = typename GemmKernel0::Mma::PresumVecTypeA;
  using PresumComputeType = typename GemmKernel0::Mma::PresumComputeType;
  using PresumComputeToIOType = NumericArrayConverter<ElementA, PresumComputeType, PresumVecType::kElements>;
  using PresumIOToComputeType = NumericArrayConverter<PresumComputeType, ElementA, PresumVecType::kElements>;
  const bool IsStrassenLayout = GemmKernel0::Mma::IsStrassenLayout;

  using PresumShapeA = typename GemmKernel0::Mma::PresumShapeA;
  using PresumShapeB = typename GemmKernel0::Mma::PresumShapeB;
  using ElementBlockScale = float;

  constexpr bool is_B_row_or_col_major = params.get_layout_B();
  using PresumGlobalIteratorA = threadblock::PresumDetail::GlobalIterator<ElementA, kThreadCount, PresumShapeA, PresumVecType, false, true, false>;
  using PresumGlobalIteratorB = threadblock::PresumDetail::GlobalIterator<ElementB, kThreadCount, PresumShapeB, PresumVecType, true, is_B_row_or_col_major, false>;

  cutlass::gemm::GemmCoord threadblock_tile_offset = {(int)blockIdx.y, (int)blockIdx.x, 0};

  const int M = params.get_problem_shape_m(problem_idx);
  const int N = params.get_problem_shape_n(problem_idx);
  const int K = params.get_problem_shape_k(problem_idx);

  const int halfM = M/2;
  const int halfN = N/2;
  const int halfK = K/2;
  
  int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * gridDim.x;
  uint thread_idx = threadIdx.x;
#if defined(CUTLASS_API_v2)
  const int presum_multiplier_a = 1 << params.get_presum_log_tile_multiplier_a();
  const int presum_multiplier_b = 1 << params.get_presum_log_tile_multiplier_b();
#elif defined(CUTLASS_API_v3)
  const int presum_multiplier_a = 1 << GemmKernel0::Mma::get_presum_log_multiplier(K, N);
  const int presum_multiplier_b = 1 << GemmKernel0::Mma::get_presum_log_multiplier(K, M);
#endif

  auto a1_coord = IsStrassenLayout ? MatrixCoord(1*halfM, 0) : MatrixCoord(0, halfK);
  auto a2_coord = IsStrassenLayout ? MatrixCoord(2*halfM, 0) : MatrixCoord(halfM, 0);
  auto a3_coord = IsStrassenLayout ? MatrixCoord(3*halfM, 0) : MatrixCoord(halfM, halfK);

  PresumGlobalIteratorA iter_PresumA(
    params.get_ptr_A(problem_idx), params.get_stride_A(problem_idx) / (IsStrassenLayout ? 2 : 1),
    {M, K},
    {threadblock_tile_offset.m() * PresumShapeA::kM, threadblock_tile_offset.n() * PresumShapeA::kN * presum_multiplier_a},
    block_idx, {0, 0}, thread_idx, a1_coord, a2_coord, a3_coord
  );

  auto packed_b1_coord = is_B_row_or_col_major ? MatrixCoord(halfK, 0) : MatrixCoord(0, halfN);
  auto packed_b2_coord = is_B_row_or_col_major ? MatrixCoord(2*halfK, 0) : MatrixCoord(0, 2*halfN);
  auto packed_b3_coord = is_B_row_or_col_major ? MatrixCoord(3*halfK, 0) : MatrixCoord(0, 3*halfN);
  auto b1_coord = IsStrassenLayout ? packed_b1_coord : MatrixCoord(0, halfN);
  auto b2_coord = IsStrassenLayout ? packed_b2_coord : MatrixCoord(halfK, 0);
  auto b3_coord = IsStrassenLayout ? packed_b3_coord : MatrixCoord(halfK, halfN);

  PresumGlobalIteratorB iter_PresumB(
    params.get_ptr_B(problem_idx), params.get_stride_B(problem_idx) / (IsStrassenLayout ? 2 : 1),
    {K, N},
    {threadblock_tile_offset.m() * PresumShapeB::kM * presum_multiplier_b, threadblock_tile_offset.n() * PresumShapeB::kN},
    block_idx, {0, 0}, thread_idx, b1_coord, b2_coord, b3_coord
  );

  PresumGlobalIteratorA iter_PresumA_M(
    params.get_ptr_presum_A(problem_idx), params.get_stride_MA(problem_idx),
    {halfM, halfK},
    {threadblock_tile_offset.m() * PresumShapeA::kM, threadblock_tile_offset.n() * PresumShapeA::kN * presum_multiplier_a},
    block_idx, {0, 0}, thread_idx, {1*halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
  );

  auto bm1_coord = is_B_row_or_col_major ? MatrixCoord{1*halfK, 0} : MatrixCoord{0, 1*halfN};
  auto bm2_coord = is_B_row_or_col_major ? MatrixCoord{2*halfK, 0} : MatrixCoord{0, 2*halfN};
  auto bm3_coord = is_B_row_or_col_major ? MatrixCoord{3*halfK, 0} : MatrixCoord{0, 3*halfN};

  PresumGlobalIteratorB iter_PresumB_M(
    params.get_ptr_presum_B(problem_idx), params.get_stride_MB(problem_idx),
    {halfK, halfN},
    {threadblock_tile_offset.m() * PresumShapeB::kM * presum_multiplier_b, threadblock_tile_offset.n() * PresumShapeB::kN},
    block_idx, {0, 0}, thread_idx, bm1_coord, bm2_coord, bm3_coord
  );

  auto presumAComputeLoads = AllPresums::APresumComputeLoads(PresumGlobalKernel);
  auto presumBComputeLoads = AllPresums::BPresumComputeLoads(PresumGlobalKernel);
  PresumVecType a0; a0.clear(); ElementBlockScale scale_a0 = 1.0f;
  PresumVecType a1; a1.clear(); ElementBlockScale scale_a1 = 1.0f;
  PresumVecType a2; a2.clear(); ElementBlockScale scale_a2 = 1.0f;
  PresumVecType a3; a3.clear(); ElementBlockScale scale_a3 = 1.0f;

  PresumVecType b0; b0.clear(); ElementBlockScale scale_b0 = 1.0f;
  PresumVecType b1; b1.clear(); ElementBlockScale scale_b1 = 1.0f;
  PresumVecType b2; b2.clear(); ElementBlockScale scale_b2 = 1.0f;
  PresumVecType b3; b3.clear(); ElementBlockScale scale_b3 = 1.0f;

  PresumIOToComputeType presum_io_to_compute_type;
  PresumComputeToIOType presum_compute_to_io_type;

  const uint IterationsA = (PresumShapeA::kM*PresumShapeA::kN*sizeof(ElementA))/(sizeof(PresumVecType)*kThreadCount);
  //TODO: Separate IterationsA and IterationsB
  for (int presum_iter = 0; presum_iter < IterationsA * presum_multiplier_a; presum_iter++) {
    const uint presum_tile = presum_iter / IterationsA;
    iter_PresumA.reset(presum_tile);
    iter_PresumA_M.reset(presum_tile);
    iter_PresumA.set_iteration(presum_iter - presum_tile * IterationsA);
    iter_PresumA_M.set_iteration(presum_iter - presum_tile * IterationsA);

    if (!(iter_PresumA_M.validTB() && iter_PresumA_M.valid())) continue;
    if (presumAComputeLoads.numAccess() > 0) {
      if constexpr (requires { params.mainloop.ptr_SFA; params.mainloop.layout_SFA; }) {
        constexpr uint ScaleGranularityK = GemmKernel0::Mma::ScaleGranularityK;
        using PresumShapeScaleA = cutlass::gemm::GemmShape<PresumShapeA::kM, ScaleGranularityK, 1>;
        using PresumGlobalScaleIteratorA = threadblock::PresumDetail::GlobalIterator<ElementBlockScale const, kThreadCount, PresumShapeScaleA, Array<float, 1>, false, false, false>;

        int scale_row = iter_PresumA.tb_offset.row() + iter_PresumA.row;
        int scale_column = iter_PresumA.tb_offset.column() + iter_PresumA.col;
        auto const& layout_SFA = params.mainloop.layout_SFA;
        auto const* ptr_SFA = params.mainloop.ptr_SFA;

        PresumGlobalScaleIteratorA iter_PresumScaleA(
          ptr_SFA, cute::get<1, 1>(layout_SFA.stride()),
          {M, K/ScaleGranularityK},
          {scale_row, scale_column/ScaleGranularityK},
          block_idx, {0, 0}, 0,
          MatrixCoord(0, halfK/ScaleGranularityK),
          MatrixCoord(halfM, 0),
          MatrixCoord(halfM, halfK/ScaleGranularityK)
        );
        bool valid_scale = iter_PresumScaleA.validTB() && iter_PresumScaleA.valid() &&
                           scale_row < halfM && scale_column < halfK;

        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0)) {
          arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_a0, iter_PresumScaleA.get(0), valid_scale);
        }
        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1)) {
          arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_a1, iter_PresumScaleA.get(1), valid_scale);
        }
        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2)) {
          arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_a2, iter_PresumScaleA.get(2), valid_scale);
        }
        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3)) {
          arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_a3, iter_PresumScaleA.get(3), valid_scale);
        }
      }

      //Do loads needed for Computing Global Presums
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a0, iter_PresumA.get(0),
                                                                iter_PresumA_M.validTB());
      }
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a1, iter_PresumA.get(1),
                                                                iter_PresumA_M.validTB());
      }
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a2, iter_PresumA.get(2),
                                                                iter_PresumA_M.validTB());
      }
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a3, iter_PresumA.get(3),
                                                                iter_PresumA_M.validTB());
      }
      // iter_PresumA.inc();
    }

    auto compute_a0 = presum_io_to_compute_type(a0);
    auto compute_a1 = presum_io_to_compute_type(a1);
    auto compute_a2 = presum_io_to_compute_type(a2);
    auto compute_a3 = presum_io_to_compute_type(a3);
    if constexpr (requires { params.mainloop.ptr_SFA; }) {
      CUTLASS_PRAGMA_UNROLL
      for (int element_idx = 0; element_idx < PresumVecType::kElements; ++element_idx) {
        compute_a0[element_idx] = __fmul_rn(compute_a0[element_idx], scale_a0);
        compute_a1[element_idx] = __fmul_rn(compute_a1[element_idx], scale_a1);
        compute_a2[element_idx] = __fmul_rn(compute_a2[element_idx], scale_a2);
        compute_a3[element_idx] = __fmul_rn(compute_a3[element_idx], scale_a3);
      }
    }

    auto s1   = compute_a2 + compute_a3;
    auto s2   = s1 - compute_a0;
    auto a02  = compute_a0 - compute_a2;
    auto a1s2 = compute_a1 - s2;

    ElementBlockScale scale_s1 = 1.0f;
    ElementBlockScale scale_s2 = 1.0f;
    ElementBlockScale scale_a02 = 1.0f;
    ElementBlockScale scale_a1s2 = 1.0f;
    if constexpr (requires { params.mainloop.ptr_SFA; }) {
      constexpr int ScaleGranularityK = GemmKernel0::Mma::ScaleGranularityK;
      constexpr int ScaleThreads = ScaleGranularityK / PresumVecType::kElements;
      static_assert(GemmKernel0::Mma::ScaleGranularityM == 1);
      static_assert(ScaleGranularityK % PresumVecType::kElements == 0);
      static_assert(ScaleThreads > 0 && ScaleThreads <= 32 && (ScaleThreads & (ScaleThreads - 1)) == 0);
      static_assert(PresumShapeA::kN % ScaleGranularityK == 0);

      auto quantize_presum = [&](auto& presum, bool check = false) {
        ElementBlockScale max_abs = 0.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int element_idx = 0; element_idx < PresumVecType::kElements; ++element_idx) {
          if (check == true && iter_PresumA_M.col < 128)
            printf("228 %d %d : %f\n", iter_PresumA_M.col, element_idx, float(presum[element_idx]));
          max_abs = fmaxf(max_abs, fabsf(float(presum[element_idx])));
        }
        unsigned active_mask = __activemask();
        CUTLASS_PRAGMA_UNROLL
        for (int offset = ScaleThreads / 2; offset > 0; offset /= 2) {
          max_abs = fmaxf(max_abs, __shfl_xor_sync(active_mask, max_abs, offset, ScaleThreads));
        }
        ElementBlockScale scale = max_abs > 0.0f ?
          fmaxf(__fdiv_rn(max_abs, float(cutlass::platform::numeric_limits<ElementA>::max())),
                FLT_MIN) : 1.0f;
        CUTLASS_PRAGMA_UNROLL
        for (int element_idx = 0; element_idx < PresumVecType::kElements; ++element_idx) {
          presum[element_idx] = __fdiv_rn(presum[element_idx], scale);
        }
        return scale;
      };

      scale_s1 = quantize_presum(s1);
      scale_s2 = quantize_presum(s2);// presum_iter == 0 && iter_PresumA_M.tb_offset.row() + iter_PresumA_M.row == 0);
      scale_a02 = quantize_presum(a02);
      scale_a1s2 = quantize_presum(a1s2);
    }

    auto store_presum_scale = [&](ElementBlockScale scale, int presum_idx) {
      if constexpr (requires { params.mainloop.ptr_SFA; }) {
        constexpr int ScaleGranularityK = GemmKernel0::Mma::ScaleGranularityK;
        using PresumShapeScaleA = cutlass::gemm::GemmShape<PresumShapeA::kM, ScaleGranularityK, 1>;
        using PresumGlobalScaleIteratorA = threadblock::PresumDetail::GlobalIterator<ElementBlockScale, kThreadCount, PresumShapeScaleA, Array<float, 1>, false, false, false>;
        int scale_row = iter_PresumA_M.tb_offset.row() + iter_PresumA_M.row;
        int scale_column = iter_PresumA_M.tb_offset.column() + iter_PresumA_M.col;
        int scale_rows = AllPresums::numAPresumYes() * halfM;
        PresumGlobalScaleIteratorA iter_PresumScaleA_M(
          params.mainloop.ptr_presum_scale_A, scale_rows,
          {scale_rows, halfK/ScaleGranularityK},
          {scale_row, scale_column/ScaleGranularityK},
          block_idx, {0, 0}, 0,
          {halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
        );
        arch::global_store<ElementBlockScale, sizeof(ElementBlockScale)>(scale, iter_PresumScaleA_M.get(presum_idx),
          iter_PresumScaleA_M.validTB() && iter_PresumScaleA_M.valid() &&
          scale_row < halfM && scale_column < halfK && scale_column % ScaleGranularityK == 0);
      }
    };

    if (AllPresums::doesComputeA(MmaStrassen::APresums::A02, PresumGlobalKernel)) {
      store_presum_scale(scale_a02, AllPresums::indexAPresum(MmaStrassen::APresums::A02));
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(a02), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A02)),
                                                               iter_PresumA_M.validTB());
    }
    if (AllPresums::doesComputeA(MmaStrassen::APresums::S1, PresumGlobalKernel)) {
      store_presum_scale(scale_s1, AllPresums::indexAPresum(MmaStrassen::APresums::S1));
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s1), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S1)),
                                                               iter_PresumA_M.validTB());
    }
    if (AllPresums::doesComputeA(MmaStrassen::APresums::S2, PresumGlobalKernel)) {
      store_presum_scale(scale_s2, AllPresums::indexAPresum(MmaStrassen::APresums::S2));
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S2)),
                                                               iter_PresumA_M.validTB());
    }
    if (AllPresums::doesComputeA(MmaStrassen::APresums::A1S2, PresumGlobalKernel)) {
      store_presum_scale(scale_a1s2, AllPresums::indexAPresum(MmaStrassen::APresums::A1S2));
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(a1s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A1S2)),
                                                               iter_PresumA_M.validTB());
    }

    // iter_PresumA_M.inc();
  }

  const uint IterationsB = (PresumShapeB::kM*PresumShapeB::kN*sizeof(ElementB))/(sizeof(PresumVecType)*kThreadCount);
  constexpr bool HasBScale = requires { params.mainloop.ptr_SFB; params.mainloop.layout_SFB; };
  for (int presum_tile = 0; presum_tile < presum_multiplier_b; ++presum_tile) {
    Array<ElementBlockScale,4> presum_b_max_abs;
    presum_b_max_abs.clear();
    Array<ElementBlockScale,4> presum_b_scale;
    for (int presum_idx = 0; presum_idx < 4; ++presum_idx) presum_b_scale[presum_idx] = 1.0f;
    for (int presum_pass = 0; presum_pass < (HasBScale ? 2 : 1); ++presum_pass) {
      for (int presum_iter = presum_tile * IterationsB; presum_iter < (presum_tile + 1) * IterationsB; presum_iter++) {
        iter_PresumB.reset(presum_tile);
        iter_PresumB_M.reset(presum_tile);
        iter_PresumB.set_iteration(presum_iter - presum_tile * IterationsB);
        iter_PresumB_M.set_iteration(presum_iter - presum_tile * IterationsB);

        if (!(iter_PresumB_M.validTB() && iter_PresumB_M.valid())) continue;

        if (presumBComputeLoads.numAccess() > 0) {
          if constexpr (HasBScale) {
            constexpr int ScaleGranularityN = GemmKernel0::Mma::ScaleGranularityN;
            constexpr int ScaleGranularityK = GemmKernel0::Mma::ScaleGranularityK;
            using PresumShapeScaleB = cutlass::gemm::GemmShape<ScaleGranularityK, ScaleGranularityN, 1>;
            using PresumGlobalScaleIteratorB = threadblock::PresumDetail::GlobalIterator<ElementBlockScale const, kThreadCount, PresumShapeScaleB, Array<float, 1>, true, true, false>;

            int scale_row = iter_PresumB.tb_offset.row() + iter_PresumB.row;
            int scale_column = iter_PresumB.tb_offset.column() + iter_PresumB.col;
            auto const& layout_SFB = params.mainloop.layout_SFB;
            auto const* ptr_SFB = params.mainloop.ptr_SFB;

            PresumGlobalScaleIteratorB iter_PresumScaleB(
              ptr_SFB, cute::get<1, 1>(layout_SFB.stride()),
              {K/ScaleGranularityK, N/ScaleGranularityN},
              {scale_row/ScaleGranularityK, scale_column/ScaleGranularityN},
              block_idx, {0, 0}, 0,
              MatrixCoord(0, halfN/ScaleGranularityN),
              MatrixCoord(halfK/ScaleGranularityK, 0),
              MatrixCoord(halfK/ScaleGranularityK, halfN/ScaleGranularityN)
            );
            bool valid_scale = iter_PresumScaleB.validTB() && iter_PresumScaleB.valid() &&
                               scale_row < halfK && scale_column < halfN;

            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0)) {
              arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_b0, iter_PresumScaleB.get(0), valid_scale);
            }
            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1)) {
              arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_b1, iter_PresumScaleB.get(1), valid_scale);
            }
            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2)) {
              arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_b2, iter_PresumScaleB.get(2), valid_scale);
            }
            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3)) {
              arch::global_load<ElementBlockScale, sizeof(ElementBlockScale)>(scale_b3, iter_PresumScaleB.get(3), valid_scale);
            }
          }

          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0)) {
            arch::global_load<PresumVecType, sizeof(PresumVecType)>(b0, iter_PresumB.get(0),
                                                                    iter_PresumB_M.validTB());
          }
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1)) {
            arch::global_load<PresumVecType, sizeof(PresumVecType)>(b1, iter_PresumB.get(1),
                                                                    iter_PresumB_M.validTB());
          }
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2)) {
            arch::global_load<PresumVecType, sizeof(PresumVecType)>(b2, iter_PresumB.get(2),
                                                                    iter_PresumB_M.validTB());
          }
          if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3)) {
            arch::global_load<PresumVecType, sizeof(PresumVecType)>(b3, iter_PresumB.get(3),
                                                                    iter_PresumB_M.validTB());
          }
          // iter_PresumB.inc();
        }

        auto compute_b0 = presum_io_to_compute_type(b0);
        auto compute_b1 = presum_io_to_compute_type(b1);
        auto compute_b2 = presum_io_to_compute_type(b2);
        auto compute_b3 = presum_io_to_compute_type(b3);
        if constexpr (HasBScale) {
          CUTLASS_PRAGMA_UNROLL
          for (int element_idx = 0; element_idx < PresumVecType::kElements; ++element_idx) {
            compute_b0[element_idx] = __fmul_rn(compute_b0[element_idx], scale_b0);
            compute_b1[element_idx] = __fmul_rn(compute_b1[element_idx], scale_b1);
            compute_b2[element_idx] = __fmul_rn(compute_b2[element_idx], scale_b2);
            compute_b3[element_idx] = __fmul_rn(compute_b3[element_idx], scale_b3);
          }
        }

        auto b10  = compute_b1 - compute_b0;
        auto b31  = compute_b3 - compute_b1;
        auto s3   = b31 + compute_b0;
        auto s3b2 = s3 - compute_b2;

        if constexpr (HasBScale) {
          auto quantize_presum = [&](auto& presum, int presum_idx) {
            CUTLASS_PRAGMA_UNROLL
            for (int element_idx = 0; element_idx < PresumVecType::kElements; ++element_idx) {
              if (presum_pass == 0) {
                presum_b_max_abs[presum_idx] = fmaxf(presum_b_max_abs[presum_idx], fabsf(float(presum[element_idx])));
              } else {
                presum[element_idx] = __fdiv_rn(presum[element_idx], presum_b_scale[presum_idx]);
              }
            }
          };
          quantize_presum(b10, 0);
          quantize_presum(b31, 1);
          quantize_presum(s3, 2);
          quantize_presum(s3b2, 3);
          if (presum_pass == 0) continue;
        }

        auto store_presum_scale = [&](ElementBlockScale scale, int presum_idx) {
          if constexpr (HasBScale) {
            constexpr int ScaleGranularityN = GemmKernel0::Mma::ScaleGranularityN;
            constexpr int ScaleGranularityK = GemmKernel0::Mma::ScaleGranularityK;
            using PresumShapeScaleB = cutlass::gemm::GemmShape<ScaleGranularityK, ScaleGranularityN, 1>;
            using PresumGlobalScaleIteratorB = threadblock::PresumDetail::GlobalIterator<ElementBlockScale, kThreadCount, PresumShapeScaleB, Array<float, 1>, true, true, false>;
            int scale_row = iter_PresumB_M.tb_offset.row() + iter_PresumB_M.row;
            int scale_column = iter_PresumB_M.tb_offset.column() + iter_PresumB_M.col;
            PresumGlobalScaleIteratorB iter_PresumScaleB_M(
              params.mainloop.ptr_presum_scale_B, halfN/ScaleGranularityN,
              {AllPresums::numBPresumYes()*halfK/ScaleGranularityK, halfN/ScaleGranularityN},
              {scale_row/ScaleGranularityK, scale_column/ScaleGranularityN},
              block_idx, {0, 0}, 0,
              {halfK/ScaleGranularityK, 0}, {2*halfK/ScaleGranularityK, 0}, {3*halfK/ScaleGranularityK, 0}
            );
            arch::global_store<ElementBlockScale, sizeof(ElementBlockScale)>(scale, iter_PresumScaleB_M.get(presum_idx),
              iter_PresumScaleB_M.validTB() && iter_PresumScaleB_M.valid() &&
              scale_row < halfK && scale_column < halfN &&
              scale_row % ScaleGranularityK == 0 && scale_column % ScaleGranularityN == 0);
          }
        };

        if (AllPresums::doesComputeB(MmaStrassen::BPresums::B10, PresumGlobalKernel)) {
          store_presum_scale(presum_b_scale[0], AllPresums::indexBPresum(MmaStrassen::BPresums::B10));
          arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(b10), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B10)),
                                                                   iter_PresumB_M.validTB());
        }
        if (AllPresums::doesComputeB(MmaStrassen::BPresums::B31, PresumGlobalKernel)) {
          store_presum_scale(presum_b_scale[1], AllPresums::indexBPresum(MmaStrassen::BPresums::B31));
          arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(b31), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)),
                                                                   iter_PresumB_M.validTB());
        }
        if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3, PresumGlobalKernel)) {
          store_presum_scale(presum_b_scale[2], AllPresums::indexBPresum(MmaStrassen::BPresums::S3));
          arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s3), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3)),
                                                                   iter_PresumB_M.validTB());
        }
        if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3B2, PresumGlobalKernel)) {
          store_presum_scale(presum_b_scale[3], AllPresums::indexBPresum(MmaStrassen::BPresums::S3B2));
          arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s3b2), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3B2)),
                                                                   iter_PresumB_M.validTB());
        }

        // iter_PresumB_M.inc();
      }
      if constexpr (HasBScale) {
        static_assert(PresumShapeB::kM == GemmKernel0::Mma::ScaleGranularityK);
        static_assert(PresumShapeB::kN == GemmKernel0::Mma::ScaleGranularityN);
        static_assert(kThreadCount % 32 == 0);
        if (presum_pass == 0) {
          __shared__ ElementBlockScale warp_max_abs[4][kThreadCount/32];
          CUTLASS_PRAGMA_UNROLL
          for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
            ElementBlockScale max_abs = presum_b_max_abs[presum_idx];
            CUTLASS_PRAGMA_UNROLL
            for (int offset = 16; offset > 0; offset /= 2) {
              max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, offset));
            }
            if (thread_idx % 32 == 0) warp_max_abs[presum_idx][thread_idx/32] = max_abs;
          }
          __syncthreads();
          CUTLASS_PRAGMA_UNROLL
          for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
            ElementBlockScale max_abs = 0.0f;
            CUTLASS_PRAGMA_UNROLL
            for (int warp_idx = 0; warp_idx < kThreadCount/32; ++warp_idx) {
              max_abs = fmaxf(max_abs, warp_max_abs[presum_idx][warp_idx]);
            }
            presum_b_scale[presum_idx] = max_abs > 0.0f ?
              fmaxf(__fdiv_rn(max_abs, float(cutlass::platform::numeric_limits<ElementB>::max())), FLT_MIN) : 1.0f;
          }
          __syncthreads();
        }
      }
    }
  }
}
}
}
}