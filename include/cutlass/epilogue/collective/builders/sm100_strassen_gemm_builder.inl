/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/numeric/numeric_types.hpp"

#include "cutlass/detail/layout.hpp"
#include "cutlass/gemm/collective/builders/sm100_common.inl"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_strassen_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/cutlass.h"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

///////////////////////////////////////////////////////////////////////////////

template <
  typename StrassenMiGroup,
  class MmaTileShape_MNK,
  class ClusterShape_MNK,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class ProblemShape,
  class EpilogueScheduleType,
  class FusionOp
>
struct CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm100,
    arch::OpClassSimt,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    ProblemShape,
    EpilogueScheduleType,
    FusionOp,
    cute::enable_if_t<
      cute::is_same_v<EpilogueScheduleType, EpilogueSimtVectorized> ||
      cute::is_same_v<EpilogueScheduleType, EpiloguePtrArraySimtVectorized> ||
      cute::is_same_v<EpilogueScheduleType, EpilogueScheduleAuto>>> {
  using CtaTileShape_MNK = MmaTileShape_MNK;

  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>;
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

  using ThreadOp = cute::conditional_t<
    IsDefaultFusionOp<FusionOp>::value,
    thread::LinearCombination<
      ElementD, AlignmentD, ElementAccumulator, ElementCompute,
      ScaleType, FloatRoundStyle::round_to_nearest, ElementC>,
    thread::LinearCombinationBiasElementwise<
      ElementC, ElementAccumulator, ElementCompute, ElementD, ElementD, AlignmentD,
      typename FusionOp::ActivationFn, cutlass::plus<ElementCompute>,
      false, typename FusionOp::ElementBias>>;
  static_assert(not (cute::is_same_v<EpilogueScheduleType, EpiloguePtrArraySimtVectorized> &&
                     not IsDefaultFusionOp<FusionOp>::value),
                "unsupported schedule + fusion");

  using WarpShape_MNK = decltype(
    cutlass::gemm::collective::detail::sm100_simt_f32_warp_shape_mnk_selector<CtaTileShape_MNK>());
  static constexpr int ThreadCount = cute::size(WarpShape_MNK{}) * NumThreadsPerWarp;
  static constexpr int WarpShape_M = cute::size<0>(WarpShape_MNK{});
  static constexpr int WarpShape_N = cute::size<1>(WarpShape_MNK{});

  using EpiTileM = cute::Int<WarpShape_M * 32>;
  using EpiTileN = cute::Int<WarpShape_N * 16>;

  using SmemLayout = cute::conditional_t<cutlass::detail::is_major<0>(GmemStrideTypeD{}),
                                         cute::Layout<cute::Shape<EpiTileM, EpiTileN>, cute::Stride<cute::_1, EpiTileM>>,
                                         cute::Layout<cute::Shape<EpiTileM, EpiTileN>, cute::Stride<EpiTileN, cute::_1>>>;

  using CopyAtomR2S = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccumulator>;
  using CopyAtomS2R = cute::Copy_Atom<
    cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentD * cute::sizeof_bits_v<ElementAccumulator>>,
    ElementAccumulator>;

  using TiledCopyS2R = decltype(
    cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
      CopyAtomS2R, ThreadCount, AlignmentD, GmemStrideTypeD, EpiTileM, EpiTileN>());

  using Schedule = cute::conditional_t<cute::is_same_v<EpilogueScheduleType, EpilogueScheduleAuto>,
                                       EpilogueSimtVectorized,
                                       EpilogueScheduleType>;
  using CopyAtomR2G = cute::Copy_Atom<
    cute::AutoVectorizingCopyWithAssumedAlignment<AlignmentD * cute::sizeof_bits_v<ElementD>>, ElementD>;
  using CollectiveOp = cutlass::epilogue::collective::StrassenEpilogue<
      StrassenMiGroup,
      GmemStrideTypeC,
      GmemStrideTypeD,
      ThreadOp,
      SmemLayout,
      CopyAtomR2S,
      TiledCopyS2R,
      CopyAtomR2G,
      ProblemShape,
      Schedule>;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective