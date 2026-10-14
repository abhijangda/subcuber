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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_strassen_mma_decl.hpp"
#include "cutlass/gemm/collective/strassen_fp8_accumulation.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/strassen_layout.hpp"
#include "cutlass/gemm/device/strassen_decls.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/detail/blockwise_scale_layout.hpp"

#include "cutlass/gemm/threadblock/presum_detail.h"

#define MAX(x,y) (((x)<(y)) ? (y) : (x))

using namespace cutlass::gemm::threadblock;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  class StrassenMiGroup_,
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StridePairA_,
  class ElementB_,
  class StridePairB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class PresumDispatchPolicy_,
  class PresumTileShapeA_,
  class PresumGmemTiledCopyA_,
  class PresumSmemLayoutAtomA_,
  class PresumSmemCopyAtomA_,
  class PresumTileShapeB_,
  class PresumGmemTiledCopyB_,
  class PresumSmemLayoutAtomB_,
  class PresumSmemCopyAtomB_,
  class PresumOpt_,
  class ProblemShape_,
  class SubMatLayoutA_,
  class SubMatLayoutB_>
struct CollectiveStrassenMma<
    StrassenMiGroup_,
    MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    StridePairA_,
    ElementB_,
    StridePairB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_,
    PresumDispatchPolicy_,
    PresumTileShapeA_,
    PresumGmemTiledCopyA_,
    PresumSmemLayoutAtomA_,
    PresumSmemCopyAtomA_,
    PresumTileShapeB_,
    PresumGmemTiledCopyB_,
    PresumSmemLayoutAtomB_,
    PresumSmemCopyAtomB_,
    PresumOpt_,
    ProblemShape_,
    SubMatLayoutA_,
    SubMatLayoutB_>
  {
  static constexpr bool UsesStrassenLayoutA = cute::is_same_v<SubMatLayoutA_, cutlass::layout::StrassenLayout>;
  static constexpr bool UsesStrassenLayoutB = cute::is_same_v<SubMatLayoutB_, cutlass::layout::StrassenLayout>;

  //
  // Type Aliases
  //
  using StrassenMiGroup = StrassenMiGroup_;
  using PresumOpt = PresumOpt_;
  using ProblemShape = ProblemShape_;
  using SubMatLayoutA = SubMatLayoutA_;
  using SubMatLayoutB = SubMatLayoutB_;
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecializedBlockScalingFP8<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = cute::tuple_element_t<0,StridePairA_>;
  using LayoutSFA = cute::tuple_element_t<1,StridePairA_>;
  using ElementB = ElementB_;
  using StrideB = cute::tuple_element_t<0,StridePairB_>;
  using LayoutSFB = cute::tuple_element_t<1,StridePairB_>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using FragmentC = decltype(partition_fragment_C(TiledMma(), take<0,2>(TileShape{})));
  using ElementBlockScale = ElementAccumulator;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScaleTMA = cute::SM90_TMA_LOAD;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using PresumShape = GemmShape<size<0>(TileShape{}), size<1>(TileShape{}), 1>;
  using PresumShapeA = PresumShape;
  using PresumShapeB = PresumShape;
  using PresumTileShapeA = PresumTileShapeA_;
  using PresumTileShapeB = PresumTileShapeB_;
  using PresumVecTypeA = Array<ElementA, (size<0>(PresumTileShapeA{})*sizeof(ElementA))/sizeof(ElementA)>;
  using PresumVecTypeB = Array<ElementA, (size<0>(PresumTileShapeB{})*sizeof(ElementA))/sizeof(ElementA)>;
  using PresumStoreVecType = Array<ElementA, 16/sizeof(ElementA)>;
  using PresumComputeType = ElementAccumulator;
  using PresumComputeToIOTypeA = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeA::kElements>;
  using PresumIOToComputeTypeA = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeA::kElements>;
  using PresumComputeToIOTypeB = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeB::kElements>;
  using PresumIOToComputeTypeB = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeB::kElements>;
  static const bool IsStrassenLayout = cute::is_same_v<SubMatLayoutA, cutlass::layout::StrassenLayout> &&
                                       cute::is_same_v<SubMatLayoutB, cutlass::layout::StrassenLayout>;

  static const int NumMMAThreads = size(TiledMma{});
  static constexpr int PresumStoreWarpSize = NumMMAThreads/4;
  static const int kPresumThreads = NumMMAThreads/4 * (size<0>(PresumTileShapeA{})/2);//TODO: Pass this from StrassenGemmKernel
  using PresumGlobalIteratorA = PresumDetail::GlobalIterator<ElementA, kPresumThreads,  
                                                             PresumShape, PresumStoreVecType, false, false>;
  using PresumGlobalIteratorB = PresumDetail::GlobalIterator<ElementB, kPresumThreads, 
                                                             PresumShape, PresumStoreVecType, true, false>;
  // using PresumSharedIterator = PresumDetail::SharedIterator<ElementA, PresumVecType, NumThreads,
  //                                                           Base::SharedStorage::PresumBuffer::NumLoadsForPresum,
  //                                                           Base::SharedStorage::PresumBuffer::SingleStageSize,
  //                                                           Stages, PresumAccessPerIter>;
  static const uint kPresumComputeIterationsA = size<0>(TileShape{})/size<0>(PresumTileShapeA{});
  static const uint kPresumComputeIterationsB = size<0>(TileShape{})/size<0>(PresumTileShapeB{});

  static const bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();
  static const bool is_fused_m2_m3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  static const int PresumStages = DispatchPolicy::Stages;
  using PresumLoadPipeline = cutlass::PipelineTmaAsync<PresumStages>;
  using PresumLoadPipelineState = cutlass::PipelineState<PresumStages>;
  using PresumLoadPipelineParams = typename PresumLoadPipeline::Params;

  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int ScaleGranularityM = size<0,0>(LayoutSFA{});
  static constexpr int ScaleGranularityN = size<0,0>(LayoutSFB{});
  static constexpr int ScaleGranularityK = size<1,0>(LayoutSFA{});

  static_assert(size<2>(TileShape{}) % ScaleGranularityK == 0);
  static_assert(ScaleGranularityK % size<2>(typename TiledMma::AtomShape_MNK{}) == 0);

  static constexpr int ScalePromotionInterval = ScaleGranularityK / size<2>(typename TiledMma::AtomShape_MNK{});
  static_assert(ScalePromotionInterval % 4 == 0, "ScalePromotionInterval must be a multiple of 4.");
  static_assert(ScalePromotionInterval >= size<2>(TileShape{}) / tile_size<2>(TiledMma{}),
    "ScalePromotionInterval must be greater than or equal to the number of stages of the MMA atom.");
  static_assert(ScalePromotionInterval % (size<2>(TileShape{}) / tile_size<2>(TiledMma{})) == 0,
    "ScalePromotionInterval must be a multiple of the number of stages of the MMA atom.");
  static constexpr int ScaleMsPerTile = size<0>(TileShape{}) / ScaleGranularityM;
  static constexpr int ScaleNsPerTile = size<1>(TileShape{}) / ScaleGranularityN;

  static constexpr bool MMajorSFA = size<0,1>(LayoutSFA{}.stride()) == 1;
  static constexpr bool NMajorSFB = size<0,1>(LayoutSFB{}.stride()) == 1;

  static constexpr int ScaleTmaThreshold = 32;
  static constexpr bool IsTmaLoadSFA = ScaleMsPerTile >= ScaleTmaThreshold && ScaleNsPerTile < ScaleTmaThreshold && MMajorSFA;
  static constexpr bool IsTmaLoadSFB = ScaleNsPerTile >= ScaleTmaThreshold && ScaleMsPerTile < ScaleTmaThreshold && NMajorSFB;
  // Two threads per CTA are producers (1 for operand tile `tma`, and 32 for scales `cp.async`)
  static constexpr int NumProducerThreadEvents = ((IsTmaLoadSFA && IsTmaLoadSFB)? 1 : 33);

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert((size<0>(TileShape{}) % ScaleGranularityM) == 0, "FP8 scaling granularity must evenly divide tile shape along M.");
  static_assert((size<1>(TileShape{}) % ScaleGranularityN) == 0, "FP8 scaling granularity must evenly divide tile shape along N.");

  using ScaleConfig = ::cutlass::detail::Sm90BlockwiseScaleConfig<
      ScaleGranularityM, 
      ScaleGranularityN, 
      ScaleGranularityK,
      MMajorSFA ? cute::GMMA::Major::MN : cute::GMMA::Major::K,
      NMajorSFB ? cute::GMMA::Major::MN : cute::GMMA::Major::K>;
  using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(TileShape{}));
  using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(TileShape{}));

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  using SmemLayoutStageA = decltype(SmemLayoutA{}(_,_,cute::Int<0>{}));
  using SmemLayoutStageB = decltype(SmemLayoutB{}(_,_,cute::Int<0>{}));

  using PresumSmemShapeA__ = decltype(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}));
  using PresumSmemLayoutA__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{})), LayoutRight{}));
  using PresumSmemLayoutB__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{})), LayoutRight{}));

  // Block scaling gmem-to-smem copy atom
  //  we can have partial tiles in M or N, so don't vectorize those loads
  using CopyAtomSFA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
  using CopyAtomSFB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;

  static constexpr int AlignmentSFA = IsTmaLoadSFA ? 128 / cutlass::sizeof_bits<ElementBlockScale>::value : 1;
  static constexpr int AlignmentSFB = IsTmaLoadSFB ? 128 / cutlass::sizeof_bits<ElementBlockScale>::value : 1;

  // Block scaling smem layout
  using SmemLayoutSFA = decltype(make_layout(
    append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
  ));
  using SmemLayoutSFB = decltype(make_layout(
    append(shape(SmemLayoutAtomSFB{}), Int<DispatchPolicy::Stages>{}),
    append(stride(SmemLayoutAtomSFB{}), size(filter_zeros(SmemLayoutAtomSFB{})))
  ));


  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 1 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<ElementAccumulator, ElementBlockScale>,
             "ElementAccumulator and ElementBlockScale should be same datatype");

  struct SharedStorage
  {
    struct alignas(128) StageStorage {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutStageA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutStageB>> smem_B;
    };

    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array<StageStorage, DispatchPolicy::Stages> stages;
      CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
      CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  static const size_t PresumSingleStageSizeA = size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{});
  static const size_t PresumSingleStageSizeB = size<0>(PresumTileShapeB{})*size<1>(PresumTileShapeB{});

  static const size_t PresumSingleStageSize = is_fused ? MAX(PresumSingleStageSizeA, PresumSingleStageSizeB) : // && StrassenMiGroup::AllPresums::computeAnyAPresum() && StrassenMiGroup::AllPresums::computeAnyBPresum()
                                                  (StrassenMiGroup::hasM0() ? PresumSingleStageSizeA : PresumSingleStageSizeB);
  static const size_t PresumSmemSize = PresumSingleStageSize*PresumStages;
  static const uint PresumWriteStages = 1;
  static const size_t PresumOutSmemSize = 1;//size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{}) * PresumWriteStages;

  struct PresumTensorStorageEmpty : cute::aligned_struct<128, _0> {
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A0;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A1;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A2;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A3;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A0;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A1;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A2;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A3;
  };

  struct PresumTensorStorageNonEmpty : cute::aligned_struct<128, _0> {
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A0;
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A1;
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A2;
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A3;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A0;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A1;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A2;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A3;
    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A0;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B0;
    // };

    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A1;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B1;
    // };

    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A2;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B2;
    // };
    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A3;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B3;
    // };
  };

  using PresumTensorStorage = std::conditional_t<(StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
                                                 (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum()),
                                                 PresumTensorStorageNonEmpty,
                                                 PresumTensorStorageEmpty>;

  static const bool IsFusedM4M5 = StrassenMiGroup::hasM4() && StrassenMiGroup::hasM5();
  static const bool IsFusedM2M3M6 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3() && StrassenMiGroup::hasM6();

  using AllPresums = typename StrassenMiGroup::AllPresums;
  using TensorStorage = typename SharedStorage::TensorStorage;
  using StageStorage = typename SharedStorage::StageStorage;
  static_assert(sizeof(StageStorage) % sizeof(typename TiledMma::ValTypeA) == 0);
  static_assert(sizeof(StageStorage) % sizeof(typename TiledMma::ValTypeB) == 0);
  using InterleavedSmemLayoutA = decltype(make_composed_layout(
      SmemLayoutA{}.layout_a(),
      SmemLayoutA{}.offset(),
      replace<2>(SmemLayoutA{}.layout_b(),
                 make_layout(shape<2>(SmemLayoutA{}),
                             replace<1>(stride<2>(SmemLayoutA{}.layout_b()),
                                        Int<sizeof(StageStorage) / sizeof(typename TiledMma::ValTypeA)>{})))));
  using InterleavedSmemLayoutB = decltype(make_composed_layout(
      SmemLayoutB{}.layout_a(),
      SmemLayoutB{}.offset(),
      replace<2>(SmemLayoutB{}.layout_b(),
                 make_layout(shape<2>(SmemLayoutB{}),
                             replace<1>(stride<2>(SmemLayoutB{}.layout_b()),
                                        Int<sizeof(StageStorage) / sizeof(typename TiledMma::ValTypeB)>{})))));
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool ComputesPresum =
      (StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
      (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum());

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    ElementBlockScale const* ptr_SFA;
    LayoutSFA layout_SFA;
    ElementBlockScale const* ptr_SFB;
    LayoutSFB layout_SFB;

    Arguments() = default;

    Arguments(ElementA const* ptr_A, StrideA dA, ElementB const* ptr_B, StrideB dB,
              ElementBlockScale const* ptr_SFA, LayoutSFA layout_SFA,
              ElementBlockScale const* ptr_SFB, LayoutSFB layout_SFB) :
              ptr_A(ptr_A), dA(dA), ptr_B(ptr_B), dB(dB),
              ptr_SFA(ptr_SFA), layout_SFA(layout_SFA), ptr_SFB(ptr_SFB), layout_SFB(layout_SFB)
              {}

    template<typename Other>
    Arguments(const Other& other) : ptr_A(other.ptr_A), dA(other.dA), ptr_B(other.ptr_B), dB(other.dB),
                                   ptr_SFA(other.ptr_SFA), layout_SFA(other.layout_SFA), ptr_SFB(other.ptr_SFB), layout_SFB(other.layout_SFB)
    {}
  };

  // Device side kernel params
  struct Params {
    static auto getTmaSFA() {
      if constexpr (IsTmaLoadSFA) {
        return make_tma_copy(
          GmemTiledCopyScaleTMA{},
          make_tensor(static_cast<ElementBlockScale const*>(nullptr), filter_zeros(LayoutSFA{})),
          filter_zeros(SmemLayoutSFA{}(_,_,_0{})),
          Shape<Int<ScaleMsPerTile>, Int<1>>{},
          _1{});
      }
      else {
        return nullptr;
      }
    }
    static auto getTmaSFB() {
      if constexpr (IsTmaLoadSFB) {
        return make_tma_copy(
          GmemTiledCopyScaleTMA{},
          make_tensor(static_cast<ElementBlockScale const*>(nullptr), filter_zeros(LayoutSFB{})),
          filter_zeros(SmemLayoutSFB{}(_,_,_0{})),
          Shape<Int<ScaleNsPerTile>, Int<1>>{},
          _1{});
      }
      else {
        return nullptr;
      }
    }
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,_0{}),
        TileShape{},
        ClusterShape{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,_0{}),
        TileShape{},
        ClusterShape{}));
    // NOTE: Does make_tma_copy supports 0 stride?
    using TMA_SFA = decltype(getTmaSFA());
    using TMA_SFB = decltype(getTmaSFB());
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_SFA tma_load_sfa;
    TMA_SFB tma_load_sfb;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    // Block scaling factors for A and B
    ElementBlockScale const* ptr_SFA;
    ElementBlockScale const* ptr_SFB;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    TMA_A tma_load_presum_a{};
    TMA_B tma_load_presum_b{};

    // Assumption: StrideA is congruent with Problem_MK
    using TMA_PresumLoad_A = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(static_cast<ElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      PresumSmemLayoutA__{}));
    //Always Column Major
    using PresumStrideB = decltype(make_stride(get<0>(StrideB{}), get<1>(StrideB{}), get<2>(StrideB{})));
    using TMA_PresumLoad_B = decltype(make_tma_copy(
      SM90_TMA_LOAD{},
      make_tensor(static_cast<ElementB const*>(nullptr), repeat_like(PresumStrideB{}, int32_t(0)), PresumStrideB{}),
      PresumSmemLayoutB__{}));
    TMA_PresumLoad_A tma_load_presumld_a{};
    TMA_PresumLoad_B tma_load_presumld_b{};

    ElementA* ptr_presum_A = nullptr;
    ElementB* ptr_presum_B = nullptr;
    uint32_t presum_tile_log_multiplier_a_ = 0;
    uint32_t presum_tile_log_multiplier_b_ = 0;
    uint32_t presum_tile_log_divider_a_ = 0;
    uint32_t presum_tile_log_divider_b_ = 0;

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_multiplier_a() const {
      return (PresumOpt::FixedPresumTileMultilplierLogA != UINT32_MAX) ?
              PresumOpt::FixedPresumTileMultilplierLogA :
              presum_tile_log_multiplier_a_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_multiplier_b() const {
      return (PresumOpt::FixedPresumTileMultilplierLogB != UINT32_MAX) ?
              PresumOpt::FixedPresumTileMultilplierLogB :
              presum_tile_log_multiplier_b_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_divider_a() const {
      return (PresumOpt::FixedPresumTileDividerLogA != UINT32_MAX) ? 
              PresumOpt::FixedPresumTileDividerLogA :
              presum_tile_log_divider_a_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_divider_b() const {
      return (PresumOpt::FixedPresumTileDividerLogB != UINT32_MAX) ? 
              PresumOpt::FixedPresumTileDividerLogB :
              presum_tile_log_divider_b_;
    }
  };

  //
  // Methods
  //

  //TODO: collect these functions in one place.
  CUTLASS_HOST_DEVICE
  static uint32_t get_presum_log_multiplier(int k, int m_or_n) {
    //TODO: We want k/2 is a divisor of Number of threads (which is usually 256 for us)
    if (k <= m_or_n) return 0;
    int multiplier = (k + m_or_n - 1)/m_or_n;
    uint32_t log_multiplier = 0;
    for (uint64_t covered = 1; covered < uint64_t(multiplier); covered *= 2) {
      ++log_multiplier;
    }
    return log_multiplier;
  }

  CUTLASS_HOST_DEVICE
  static uint32_t get_presum_log_divider(int k, int m_or_n) {
    //TODO: We want k/2 is a divisor of Number of threads (which is usually 256 for us)
    if (k >= m_or_n) return 0;
    int divider = m_or_n/k;

    if (divider >= 8) {
      return 3; //divide with 8
    } else if (divider >= 4) {
      return 2; //divide with 4
    } else if (divider >= 2) {
      return 1; //divide with 2
    } else {
      return 0; //divide with 1
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, ElementA* ptr_presum_a, ElementB* ptr_presum_b, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto presum_tile_log_multiplier_a = get_presum_log_multiplier(K, N);
    auto presum_tile_log_multiplier_b = get_presum_log_multiplier(K, M);
    auto presum_tile_log_divider_a = get_presum_log_divider(K, N);
    auto presum_tile_log_divider_b = get_presum_log_divider(K, M);

    auto ptr_presum_A = reinterpret_cast<ElementA const*>(ptr_presum_a);
    auto ptr_presum_B = reinterpret_cast<ElementB const*>(ptr_presum_b);
    auto ptr_A = reinterpret_cast<ElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<ElementB const*>(args.ptr_B);
    auto ptr_SFA = reinterpret_cast<ElementBlockScale const*>(args.ptr_SFA);
    auto ptr_SFB = reinterpret_cast<ElementBlockScale const*>(args.ptr_SFB);

    Tensor tensor_sfa = make_tensor(ptr_SFA, filter_zeros(args.layout_SFA));
    Tensor tensor_sfb = make_tensor(ptr_SFB, filter_zeros(args.layout_SFB));
    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));
    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_SFA tma_load_sfa{};
    if constexpr (IsTmaLoadSFA) {
      tma_load_sfa = make_tma_copy(
          GmemTiledCopyScaleTMA{},
          tensor_sfa,
          filter_zeros(SmemLayoutSFA{})(_,_,cute::Int<0>{}),
          Shape<Int<ScaleMsPerTile>, Int<1>>{},
          _1{});
    }
    typename Params::TMA_SFB tma_load_sfb{};
    if constexpr (IsTmaLoadSFB) {
      tma_load_sfb = make_tma_copy(
          GmemTiledCopyScaleTMA{},
          tensor_sfb,
          filter_zeros(SmemLayoutSFB{})(_,_,cute::Int<0>{}),
          Shape<Int<ScaleNsPerTile>, Int<1>>{},
          _1{});
    }
    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes_sfa = TmaTransactionBytesSFA;
    uint32_t transaction_bytes_sfb = TmaTransactionBytesSFB;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk + transaction_bytes_sfa + transaction_bytes_sfb;

    auto presum_stride_a = args.dA;
    auto presum_stride_b = args.dB;
    if constexpr (::cutlass::gemm::detail::is_major<0,StrideA>()) {
      get<1>(presum_stride_a) = M/2;
    } else {
      get<0>(presum_stride_a) = K/2;
    }
    if constexpr (::cutlass::gemm::detail::is_major<0,StrideB>()) {
      get<1>(presum_stride_b) = N/2;
    } else {
      get<0>(presum_stride_b) = K/2;
    }

    Tensor tensor_presum_a = make_tensor(ptr_presum_A, make_layout(make_shape(4*M/2,K/2,L), presum_stride_a));
    Tensor tensor_presum_b = make_tensor(ptr_presum_B, make_layout(make_shape(4*N/2,K/2,L), presum_stride_b));

    typename Params::TMA_A tma_load_presum_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_presum_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_presum_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_presum_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});

    typename Params::TMA_PresumLoad_A tma_load_presumld_a = make_tma_copy(
      SM90_TMA_LOAD{},
      tensor_a,
      PresumSmemLayoutA__{});
    Tensor tensor_presumld_b = make_tensor(
      ptr_B,
      //Always-Column Major
      make_layout(make_shape(N,K,L), make_stride(get<0>(args.dB), get<1>(args.dB), get<2>(args.dB))));
    typename Params::TMA_PresumLoad_B tma_load_presumld_b = make_tma_copy(
      SM90_TMA_LOAD{},
      tensor_presumld_b,
      PresumSmemLayoutB__{});

    return {
      tma_load_a,
      tma_load_b,
      tma_load_sfa,
      tma_load_sfb,
      transaction_bytes,
      transaction_bytes_mk,
      transaction_bytes_nk,
      args.ptr_SFA,
      args.ptr_SFB,
      args.layout_SFA,
      args.layout_SFB,
      tma_load_presum_a,
      tma_load_presum_b,
      tma_load_presumld_a,
      tma_load_presumld_b,
      (ElementA*)ptr_presum_a, (ElementB*)ptr_presum_b,
      presum_tile_log_multiplier_a,
      presum_tile_log_multiplier_b,
      presum_tile_log_divider_a,
      presum_tile_log_divider_b
    };;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    if constexpr (ComputesPresum) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: FP8 in-kernel pre-sum computation is not implemented.\n");
      return false;
    }
    if constexpr (UsesStrassenLayoutA || UsesStrassenLayoutB) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: FP8 packed Strassen operand layouts are not implemented.\n");
      return false;
    }
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    if (!cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{})) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load tensor A.\n");
    }
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    if (!cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{})) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load tensor B.\n");
    }
    constexpr int min_tma_aligned_elements_S = tma_alignment_bits / cutlass::sizeof_bits<ElementBlockScale>::value;
    if (IsTmaLoadSFA && !cutlass::detail::check_alignment<min_tma_aligned_elements_S>(args.layout_SFA)) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load scale A.\n");
    }
    if (IsTmaLoadSFB && !cutlass::detail::check_alignment<min_tma_aligned_elements_S>(args.layout_SFB)) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size doesn't meet the minimum alignment requirements for using TMA to load scale B.\n");
    }

    // We expect full tiles in K
    if (K % size<2>(TileShape{}) != 0) {
      implementable = false;
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem size K is incompatible with tile size.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  static constexpr uint32_t TmaTransactionBytesSFA =
        (IsTmaLoadSFA? cutlass::bits_to_bytes(ScaleMsPerTile * static_cast<uint32_t>(sizeof_bits<ElementBlockScale>::value)): 0);
  static constexpr uint32_t TmaTransactionBytesSFB =
        (IsTmaLoadSFB? cutlass::bits_to_bytes(ScaleNsPerTile * static_cast<uint32_t>(sizeof_bits<ElementBlockScale>::value)): 0);
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesSFA + TmaTransactionBytesSFB;

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params)
  {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    if constexpr (IsTmaLoadSFA) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_sfa.get_tma_descriptor());
    }
    if constexpr (IsTmaLoadSFB) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_sfb.get_tma_descriptor());
    }
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  get_base_tensors(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto halfM = M/2; auto halfN = N/2; auto halfK = K/2;

    const int32_t init_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(IsStrassenLayout ? make_shape(4*halfM,halfK,init_L) : make_shape(M,K,init_L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(IsStrassenLayout ? make_shape(halfN,4*halfK,init_L) : make_shape(N,K,init_L));                            // (n,k,l)
    auto A00_data = mA_mkl.data() + make_coord(0,0,_);
    auto B00_data = mB_nkl.data() + make_coord(0,0,_);

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(make_tensor(A00_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(make_tensor(B00_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)

    Tensor presum_mA_mkl = mainloop_params.tma_load_presum_a.get_tma_tensor(make_shape(4*halfM,halfK,init_L));                            // (m,k,l)
    Tensor presum_mB_nkl = mainloop_params.tma_load_presum_b.get_tma_tensor(make_shape(halfN,4*halfK,init_L));                            // (n,k,l)
    auto presum_mA_data = presum_mA_mkl.data() + make_coord(0,0,_);
    auto presum_mB_data = presum_mB_nkl.data() + make_coord(0,0,_);

    // Make tiled views, defer the slice
    Tensor gPresumA02 = local_tile(make_tensor(presum_mA_data, presum_mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    Tensor gPresumB31 = local_tile(make_tensor(presum_mB_data, presum_mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    return cute::make_tuple(gA_mkl, gPresumA02, gB_nkl, gPresumB31);
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                             // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                             // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});         // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});         // (BLK_N,BLK_K,n,k,l)

    // Note that mSFA_mkl and mSFB_nkl are already blocked tiled in the `m` host and
    // gScaleA_mkl and gScaleB_nkl in `g` global memory are same as mSFA_mkl and mSFB_nkl.
    auto mSFA_mkl = [&]() {
      if constexpr (IsTmaLoadSFA) {
        return mainloop_params.tma_load_sfa.get_tma_tensor(shape(filter_zeros(mainloop_params.layout_SFA)));
      }
      else {
        return make_tensor(make_gmem_ptr(mainloop_params.ptr_SFA), mainloop_params.layout_SFA); // (scale_m,k,l)
      }
    }();
    auto mSFB_nkl = [&]() {
      if constexpr (IsTmaLoadSFB) {
        return mainloop_params.tma_load_sfb.get_tma_tensor(shape(filter_zeros(mainloop_params.layout_SFB)));
      }
      else {
        return make_tensor(make_gmem_ptr(mainloop_params.ptr_SFB), mainloop_params.layout_SFB); // (scale_n,k,l)
      }
    }();

    return cute::make_tuple(gA_mkl, gB_nkl, mSFA_mkl, mSFB_nkl);
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, ProblemShape_MNKL const& half_problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto [halfM, halfN, halfK, halfL] = half_problem_shape_MNKL;

    auto original_inputs = load_init(problem_shape_MNKL, mainloop_params);

    const bool is_original_a = cute::is_same_v<SubMatLayoutA, cutlass::layout::OriginalLayout>;
    // const bool is_strassen_a = cute::is_same_v<SubMatLayoutA, cutlass::layout::StrassenLayout>;

    const bool is_original_b = cute::is_same_v<SubMatLayoutB, cutlass::layout::OriginalLayout>;
    // const bool is_strassen_b = cute::is_same_v<SubMatLayoutB, cutlass::layout::StrassenLayout>;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(is_original_a ? make_shape(M,K,L) : make_shape(4*M/2,K/2,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                             // (n,k,l)

    Tensor presum_mA_mkl = mainloop_params.tma_load_presum_a.get_tma_tensor(make_shape(halfM,halfK,L));                            // (m,k,l)
    Tensor presum_mB_nkl = mainloop_params.tma_load_presum_b.get_tma_tensor(make_shape(halfN,halfK,L));                            // (n,k,l)
    auto presum_mA_ptr = presum_mA_mkl.data();

    auto presum_A02_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A02)*halfM,_);
    auto presum_S1_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S1)*halfM,_);
    auto presum_S2_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S2)*halfM,_);
    auto presum_A1S2_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A1S2)*halfM,_);

    auto presum_A02 = make_tensor(presum_A02_ptr, presum_mA_mkl.layout());
    auto presum_S1 = make_tensor(presum_S1_ptr, presum_mA_mkl.layout());
    auto presum_S2 = make_tensor(presum_S2_ptr, presum_mA_mkl.layout());
    auto presum_A1S2 = make_tensor(presum_A1S2_ptr, presum_mA_mkl.layout());

    auto presum_mB_ptr = presum_mB_nkl.data();

    auto presum_B31_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B31)*halfN,_);
    auto presum_B10_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B10)*halfN,_);
    auto presum_S3_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3)*halfN,_);
    auto presum_S3B2_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3B2)*halfN,_);

    auto presum_B31 = make_tensor(presum_B31_ptr, presum_mB_nkl.layout());
    auto presum_B10 = make_tensor(presum_B10_ptr, presum_mB_nkl.layout());
    auto presum_S3 = make_tensor(presum_S3_ptr, presum_mB_nkl.layout());
    auto presum_S3B2 = make_tensor(presum_S3B2_ptr, presum_mB_nkl.layout());

    auto A00_data = mA_mkl.data() + (is_original_a ? make_coord(0*halfK,0*halfM,_) : make_coord(0*halfK,0*halfM,_));
    auto A01_data = mA_mkl.data() + (is_original_a ? make_coord(1*halfK,0*halfM,_) : make_coord(0*halfK,1*halfM,_));
    auto A10_data = mA_mkl.data() + (is_original_a ? make_coord(0*halfK,1*halfM,_) : make_coord(0*halfK,2*halfM,_));
    auto A11_data = mA_mkl.data() + (is_original_a ? make_coord(1*halfK,1*halfM,_) : make_coord(0*halfK,3*halfM,_));

    auto B00_data = mB_nkl.data() + (is_original_b ? make_coord(0*halfK,0*halfN,_) : make_coord(0*halfK,0*halfN,_));
    auto B01_data = mB_nkl.data() + (is_original_b ? make_coord(0*halfK,1*halfN,_) : make_coord(0*halfK,1*halfN,_));
    auto B10_data = mB_nkl.data() + (is_original_b ? make_coord(1*halfK,0*halfN,_) : make_coord(0*halfK,2*halfN,_));
    auto B11_data = mB_nkl.data() + (is_original_b ? make_coord(1*halfK,1*halfN,_) : make_coord(0*halfK,3*halfN,_));

    // Make tiled views, defer the slice
    Tensor gA00_mkl = local_tile(make_tensor(A00_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gA01_mkl = local_tile(make_tensor(A01_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA10_mkl = local_tile(make_tensor(A10_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA11_mkl = local_tile(make_tensor(A11_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    Tensor gB00_nkl = local_tile(make_tensor(B00_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)
    Tensor gB01_nkl = local_tile(make_tensor(B01_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gB10_nkl = local_tile(make_tensor(B10_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gB11_nkl = local_tile(make_tensor(B11_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    Tensor gPresumA02 = local_tile(presum_A02, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gPresumS1  = local_tile(presum_S1,  TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gPresumS2  = local_tile(presum_S2,  TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gPresumA1S2 = local_tile(presum_A1S2, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    Tensor gPresumB31 = local_tile(presum_B31, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumB10 = local_tile(presum_B10, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumS3 = local_tile(presum_S3, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumS3B2 = local_tile(presum_S3B2, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    
    return cute::make_tuple(gA00_mkl, gA01_mkl, gA10_mkl, gA11_mkl,
                            gPresumA02, gPresumS1, gPresumS2, gPresumA1S2,
                            gB00_nkl, gB01_nkl, gB10_nkl, gB11_nkl,
                            gPresumB31, gPresumB10, gPresumS3, gPresumS3B2,
                            get<2>(original_inputs), get<3>(original_inputs));
  }

  CUTLASS_DEVICE auto
  get_strassen_layout_inputs(int sub_m_idx) {
    if ((StrassenMiGroup::hasM0() && !is_fused) || (is_fused && sub_m_idx == 0)) {
      return make_tuple(int(MmaStrassen::APresums::A0), int(MmaStrassen::BPresums::B0));
    }
    if ((StrassenMiGroup::hasM1() && !is_fused) || (is_fused && sub_m_idx == 1)) {
      return make_tuple(int(MmaStrassen::APresums::A1), int(MmaStrassen::BPresums::B2));
    }

    bool IsFusedM2M3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();

    if ((StrassenMiGroup::hasM2() && !IsFusedM2M3) || (IsFusedM2M3 && sub_m_idx == 0)) {
      return make_tuple(int(MmaStrassen::APresums::S2 - MmaStrassen::APresums::APresumStart), int(MmaStrassen::BPresums::S3 - MmaStrassen::BPresums::BPresumStart));
    }

    if ((StrassenMiGroup::hasM3() && !IsFusedM2M3) || (IsFusedM2M3 && sub_m_idx == 1)) {
      return make_tuple(int(StrassenMiGroup::APresums::A02 - MmaStrassen::APresums::APresumStart), int(StrassenMiGroup::BPresums::B31 - MmaStrassen::BPresums::BPresumStart));
    }
    if ((StrassenMiGroup::hasM4() && !IsFusedM4M5) || (IsFusedM4M5 && sub_m_idx == 0)) {
      return make_tuple(int(StrassenMiGroup::APresums::S1 - MmaStrassen::APresums::APresumStart), int(StrassenMiGroup::BPresums::B10 - MmaStrassen::BPresums::BPresumStart));
    }
    if ((StrassenMiGroup::hasM5() && !IsFusedM4M5) || (IsFusedM4M5 && sub_m_idx == 1)) {
      return make_tuple(int(StrassenMiGroup::APresums::A1S2 - MmaStrassen::APresums::APresumStart), int(StrassenMiGroup::BPresums::B3));
    }
    if ((StrassenMiGroup::hasM6() && !IsFusedM2M3M6) || (IsFusedM2M3M6 && sub_m_idx == 2)) {
      return make_tuple(int(StrassenMiGroup::APresums::A3), int(StrassenMiGroup::BPresums::S3B2 - MmaStrassen::BPresums::BPresumStart));
    }

    return make_tuple(0, 0);
  }

  template<typename Tuple>
  CUTLASS_DEVICE auto
  get_inputs(Tuple const& load_inputs, int sub_m_idx = 0) {
    auto load_inputs_a = take<0, 8  >(load_inputs);
    auto load_inputs_b = take<8, 8+8>(load_inputs);

    if ((StrassenMiGroup::hasM0() && !is_fused) || (is_fused && sub_m_idx == 0)) {
      return make_tuple(get<MmaStrassen::APresums::A0>(load_inputs_a),
                        get<MmaStrassen::BPresums::B0>(load_inputs_b),
                        get<16>(load_inputs), get<17>(load_inputs));
    }
    if ((StrassenMiGroup::hasM1() && !is_fused) || (is_fused && sub_m_idx == 1)) {
      return make_tuple(get<MmaStrassen::APresums::A1>(load_inputs_a),
                        get<MmaStrassen::BPresums::B2>(load_inputs_b),
                        get<16>(load_inputs), get<17>(load_inputs));
    }

    bool IsFusedM2M3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();

    if ((StrassenMiGroup::hasM2() && !IsFusedM2M3) || (IsFusedM2M3 && sub_m_idx == 0)) {
       return make_tuple(get<MmaStrassen::APresums::S2>(load_inputs_a),
                         get<MmaStrassen::BPresums::S3>(load_inputs_b),
                         get<16>(load_inputs), get<17>(load_inputs));
    }

    if ((StrassenMiGroup::hasM3() && !IsFusedM2M3) || (IsFusedM2M3 && sub_m_idx == 1)) {
      return make_tuple(get<StrassenMiGroup::APresums::A02>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::B31>(load_inputs_b),
                        get<16>(load_inputs), get<17>(load_inputs));
    }
    if ((StrassenMiGroup::hasM4() && !IsFusedM4M5) || (IsFusedM4M5 && sub_m_idx == 0)) {
      return make_tuple(get<StrassenMiGroup::APresums::S1>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::B10>(load_inputs_b),
                        get<16>(load_inputs), get<17>(load_inputs));
    }
    if ((StrassenMiGroup::hasM5() && !IsFusedM4M5) || (IsFusedM4M5 && sub_m_idx == 1)) {
      return make_tuple(get<StrassenMiGroup::APresums::A1S2>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::B3>(load_inputs_b),
                        get<16>(load_inputs), get<17>(load_inputs));
    }
    if ((StrassenMiGroup::hasM6() && !IsFusedM2M3M6) || (IsFusedM2M3M6 && sub_m_idx == 2)) {
      return make_tuple(get<StrassenMiGroup::APresums::A3>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::S3B2>(load_inputs_b),
                        get<16>(load_inputs), get<17>(load_inputs));
    }

    return make_tuple(get<0>(load_inputs_a), get<0>(load_inputs_b), get<16>(load_inputs), get<17>(load_inputs));
    // CUTE_GCC_UNREACHABLE;
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  presumld_inputs(ProblemShape_MNKL const& problem_shape_MNKL, ProblemShape_MNKL const& half_problem_shape_MNKL, Params const& mainloop_params) const {
    auto inputs = load_init(problem_shape_MNKL, half_problem_shape_MNKL, mainloop_params);
    return make_tuple(get<0>(inputs), get<1>(inputs), get<2>(inputs), get<3>(inputs),
                      get<8>(inputs), get<9>(inputs), get<10>(inputs), get<11>(inputs));
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class TensorScaleA, class TensorScaleB,
    class PresumLDInputs,
    class KTileIterator, class BlockCoord,
    class ProblemShape_MNKL
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      ProblemShape_MNKL const& problem_shape_mnkl,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorScaleA, TensorScaleB> const& load_inputs,
      cute::tuple<int, int> const& strassen_layout_load_idxs,
      BlockCoord const& blk_coord, int sub_m_idx,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors,
      PresumLDInputs const& all_presumld_inputs,
      PresumTensorStorage& shared_presum_tensors) {
    int lane_predicate = cute::elect_one_sync();
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    uint block_idx = 0;
    auto [M, N, K, L] = problem_shape_mnkl;
    auto halfM = M/2; auto halfN = N/2; auto halfK = K/2; auto halfL = L;
    const int n_coord_div = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) % (1 << mainloop_params.get_presum_tile_log_divider_a());
    const int new_n_coord = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) >> mainloop_params.get_presum_tile_log_divider_a();

    PresumGlobalIteratorA iter_PresumA_M(
      mainloop_params.ptr_presum_A, halfK,
      {halfM, halfK},
      {m_coord * size<0>(TileShape{}) + ((n_coord_div * size<0>(TileShape{})) >> mainloop_params.get_presum_tile_log_divider_a()),
        new_n_coord * size<1>(TileShape{})},
      block_idx, {0, 0}, 0, {1*halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
    );
    PresumGlobalIteratorB iter_PresumB_M(
      mainloop_params.ptr_presum_B, halfN,
      {halfK, halfN},
      {(m_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_b()) * (size<0>(TileShape{}))) >> mainloop_params.get_presum_tile_log_divider_b(),
        n_coord * size<1>(TileShape{})},
      block_idx, {0, 0}, 0, {1*halfK, 0}, {2*halfK, 0}, {3*halfK, 0}
    );

    // Blockscaling: Tma loads for load_input and CpAsync for load_scale
    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.stages[0].smem_A.data()), InterleavedSmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.stages[0].smem_B.data()), InterleavedSmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.data()), filter_zeros(SmemLayoutSFA{})); // (ScaleMsPerTile,PIPE)
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.data()), filter_zeros(SmemLayoutSFB{})); // (ScaleNsPerTile,PIPE)

    Tensor gA0 = get<0>(all_presumld_inputs);
    Tensor gA1 = get<1>(all_presumld_inputs);
    Tensor gA2 = get<2>(all_presumld_inputs);
    Tensor gA3 = get<3>(all_presumld_inputs);

    Tensor gB0 = get<4>(all_presumld_inputs);
    Tensor gB1 = get<5>(all_presumld_inputs);
    Tensor gB2 = get<6>(all_presumld_inputs);
    Tensor gB3 = get<7>(all_presumld_inputs);

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    bool const use_presum_a = IsFusedM4M5 ||
                              (is_fused_m2_m3 ? sub_m_idx != 2 :
                                StrassenMiGroup::APresumLoads().get_first_access_idx() >= MmaStrassen::APresums::APresumStart);
    bool const use_presum_b = IsFusedM4M5 ? sub_m_idx == 0 :
                              (is_fused_m2_m3 ||
                                StrassenMiGroup::BPresumLoads().get_first_access_idx() >= MmaStrassen::BPresums::BPresumStart);

    auto base_tensors = get_base_tensors(problem_shape_mnkl, mainloop_params);
    auto strassen_layout_inputs = make_tuple(use_presum_a ? get<1>(base_tensors) : get<0>(base_tensors),
                                             use_presum_b ? get<3>(base_tensors) : get<2>(base_tensors));

    auto selected_load_inputs = [&]() {
      if constexpr (IsStrassenLayout) {
        return strassen_layout_inputs;
      } else {
        return load_inputs;
      }
    }();
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);
    Tensor mSFA_mkl = get<2>(load_inputs);
    Tensor mSFB_nkl = get<3>(load_inputs);

    auto& tma_load_a = (IsFusedM4M5) ? mainloop_params.tma_load_presum_a :
                        (is_fused_m2_m3 ?
                          (sub_m_idx == 2 ? mainloop_params.tma_load_a : mainloop_params.tma_load_presum_a) :
                          ((StrassenMiGroup::APresumLoads().get_first_access_idx() < MmaStrassen::APresums::APresumStart) ?
                            mainloop_params.tma_load_a :
                            mainloop_params.tma_load_presum_a));
    auto& tma_load_b = (IsFusedM4M5) ? (sub_m_idx == 0 ? mainloop_params.tma_load_presum_b : mainloop_params.tma_load_b) :
                        (is_fused_m2_m3 ?
                          mainloop_params.tma_load_presum_b :
                          ((StrassenMiGroup::BPresumLoads().get_first_access_idx() < MmaStrassen::BPresums::BPresumStart) ?
                            mainloop_params.tma_load_b : 
                            mainloop_params.tma_load_presum_b));

    auto block_tma_a = tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = tma_load_b.get_slice(cluster_local_block_id.x);

    auto block_tma_presum_ld_a = mainloop_params.tma_load_presumld_a.get_slice(0);
    auto block_tma_presum_ld_b = mainloop_params.tma_load_presumld_b.get_slice(0);

    int const raw_sub_m_offset_a = IsStrassenLayout ? (get<0>(strassen_layout_load_idxs)*halfM / size<0>(TileShape{})) : 0;
    int const raw_sub_m_offset_b = IsStrassenLayout ? (get<1>(strassen_layout_load_idxs)*halfK / size<2>(TileShape{})) : 0;

    // Partition the inputs based on the current block coordinates.
    Tensor gA = gA_mkl(_,_,m_coord + raw_sub_m_offset_a,_,l_coord);                                                     // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)
    Tensor gSFA = local_tile(
      mSFA_mkl, make_tile(Int<ScaleMsPerTile>{}, Int<1>{}),
      make_coord(m_coord,_,l_coord));
    Tensor gSFB = local_tile(
      mSFB_nkl, make_tile(Int<ScaleNsPerTile>{}, Int<1>{}),
      make_coord(n_coord,_,l_coord));

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);                                                 // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                                 // (TMA,TMA_N,TMA_K,PIPE)

    auto [tAgA_SFA, tAsA_SFA] = [&]() {
      if constexpr (IsTmaLoadSFA) {
        auto block_tma_sfa = mainloop_params.tma_load_sfa.get_slice(cluster_local_block_id.y);
        Tensor tAgA_SFA_ = block_tma_sfa.partition_S(gSFA);
        Tensor tAsA_SFA_ = block_tma_sfa.partition_D(sSFA);
        return cute::make_tuple(tAgA_SFA_, tAsA_SFA_);
      }
      else {
        return cute::make_tuple(0, 0);
      }
    }();
    auto [tBgB_SFB, tBsB_SFB] = [&]() {
      if constexpr (IsTmaLoadSFB) {
        auto block_tma_sfb = mainloop_params.tma_load_sfb.get_slice(cluster_local_block_id.y);
        Tensor tBgB_SFB_ = block_tma_sfb.partition_S(gSFB);
        Tensor tBsB_SFB_ = block_tma_sfb.partition_D(sSFB);
        return cute::make_tuple(tBgB_SFB_, tBsB_SFB_);
      }
      else {
        return cute::make_tuple(0, 0);
      }
    }();

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;
    uint16_t mcast_mask_sf = 0;

    // Issue TmaLoads for GEMM operands A/B and CpAsync for scale tensors
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};                       // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
      }
    }

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //
      int write_stage = smem_pipe_write.index();
      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      // Copy operands A and B from global memory to shared memory
      if (lane_predicate) copy(tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
      if (lane_predicate) copy(tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter + raw_sub_m_offset_b), tBsB(_,_,_,write_stage));

      // Copy scale tensors from global memory to shared memory
      if constexpr (IsTmaLoadSFA) {
        if (lane_predicate) {
          copy(mainloop_params.tma_load_sfa.with(*tma_barrier, mcast_mask_sf), tAgA_SFA(_,_,_,*k_tile_iter), tAsA_SFA(_,_,_,write_stage));
        }
      }
      if constexpr (IsTmaLoadSFB) {
        if (lane_predicate) {
          copy(mainloop_params.tma_load_sfb.with(*tma_barrier, mcast_mask_sf), tBgB_SFB(_,_,_,*k_tile_iter), tBsB_SFB(_,_,_,write_stage));
        }
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  template <
    class TensorA, class TensorB,
    class TensorScaleA, class TensorScaleB,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load_auxiliary(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorScaleA, TensorScaleB> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    // Block scaling: load_scale has scaling tensors in global memory which are not tiled
    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.data()), SmemLayoutSFA{}); // (ScaleMsPerTile,k)
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.data()), SmemLayoutSFB{}); // (ScaleNsPerTile,k)

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

    Tensor mSFA_mkl = get<2>(load_inputs);
    Tensor mSFB_nkl = get<3>(load_inputs);

    Tensor iSFA_mkl = make_identity_tensor(shape(mainloop_params.layout_SFA));                                // (m,k,l)
    Tensor iSFB_nkl = make_identity_tensor(shape(mainloop_params.layout_SFB));                                // (n,k,l)

    Tensor gSFA_mkl = local_tile(mSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M,BLK_K,m,k,l)
    Tensor cSFA_mkl = local_tile(iSFA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});     // (BLK_M,BLK_K,m,k,l)
    Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)
    Tensor cSFB_nkl = local_tile(iSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)

    Tensor gSFA_k = gSFA_mkl(_,_,m_coord,_,l_coord);
    Tensor cSFA_k = cSFA_mkl(_,_,m_coord,_,l_coord);
    Tensor gSFB_k = gSFB_nkl(_,_,n_coord,_,l_coord);
    Tensor cSFB_k = cSFB_nkl(_,_,n_coord,_,l_coord);

    TiledCopy scale_copy_a = make_tiled_copy(CopyAtomSFA{},
      Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
    TiledCopy scale_copy_b = make_tiled_copy(CopyAtomSFB{},
      Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
    ThrCopy thr_scale_copy_a = scale_copy_a.get_slice(thread_idx);
    ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(thread_idx);

    Tensor tSFAgSFA_k = thr_scale_copy_a.partition_S(gSFA_k);
    Tensor tSFAcSFA_k = thr_scale_copy_a.partition_S(cSFA_k);
    Tensor tSFAsSFA   = thr_scale_copy_a.partition_D(sSFA);

    Tensor tSFBgSFB_k = thr_scale_copy_b.partition_S(gSFB_k);
    Tensor tSFBcSFB_k = thr_scale_copy_b.partition_S(cSFB_k);
    Tensor tSFBsSFB   = thr_scale_copy_b.partition_D(sSFB);

    Tensor tSFApSFA = make_tensor<bool>(shape(filter_zeros(tSFAsSFA(_,_,_,_0{}))));                 // (CPY,CPY_M,CPY_K)
    Tensor tSFBpSFB = make_tensor<bool>(shape(filter_zeros(tSFBsSFB(_,_,_,_0{}))));                 // (CPY,CPY_N,CPY_K)

    auto SFA_shape = shape(mainloop_params.layout_SFA);
    auto SFB_shape = shape(mainloop_params.layout_SFB);

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      // Since scale granularity K is multiple of BLK_K we do not have to consider if that is OOB
      bool load_sfa = thread_idx < ScaleMsPerTile;
      Tensor tSFAcSFA = tSFAcSFA_k(_,_,_,*k_tile_iter);
      Tensor tSFAcSFA_compact = filter_zeros(tSFAcSFA);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSFApSFA); ++i) {
        tSFApSFA(i) = load_sfa && elem_less(tSFAcSFA_compact(i), SFA_shape);
      }

      bool load_sfb = thread_idx < ScaleNsPerTile;
      Tensor tSFBcSFB = tSFBcSFB_k(_,_,_,*k_tile_iter);
      Tensor tSFBcSFB_compact = filter_zeros(tSFBcSFB);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tSFBpSFB); ++i) {
        tSFBpSFB(i) = load_sfb && elem_less(tSFBcSFB_compact(i), SFB_shape);
      }
      int write_stage = smem_pipe_write.index();
      // Copy scale tensors from global memory to shared memory
      if constexpr (!IsTmaLoadSFA) {
        copy_if(scale_copy_a, tSFApSFA, filter_zeros(tSFAgSFA_k(_,_,_,*k_tile_iter)), filter_zeros(tSFAsSFA(_,_,_,write_stage)));
      }
      if constexpr (!IsTmaLoadSFB) {
        copy_if(scale_copy_b, tSFBpSFB, filter_zeros(tSFBgSFB_k(_,_,_,*k_tile_iter)), filter_zeros(tSFBsSFB(_,_,_,write_stage)));
      }
      if constexpr (!IsTmaLoadSFA || !IsTmaLoadSFB) {
        pipeline.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive_noinc);
      }

      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor
  >
  CUTLASS_DEVICE
  void scale_if_needed(StrassenGmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor scaleFactor) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor);
    }
  }
  template<
    class EngineAccum,
    class LayoutAccum,
    class ScaleFactor1,
    class ScaleFactor2
  >
  CUTLASS_DEVICE
  void scale_if_needed(StrassenGmmaFP8Accumulation<EngineAccum, LayoutAccum>& accumulation, ScaleFactor1 scaleFactor1, ScaleFactor2 scaleFactor2) {
    if constexpr (ScalePromotionInterval != 4) {
      accumulation.scale_if_needed(scaleFactor1, scaleFactor2);
    }
    else {
      // avoid unnecessary tests when granularity is the finnest
      accumulation.scale(scaleFactor1, scaleFactor2);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class BlockCoord,
    class FrgTensorC,
    class ProblemShape_MNKL
  >
  CUTLASS_DEVICE void
  mma(BlockCoord const& blk_coord, int sub_m_idx, 
      ProblemShape_MNKL const& problem_shape,
      ProblemShape_MNKL const& half_problem_shape,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      PresumTensorStorage& shared_presum_tensors,
      Params const& mainloop_params) {

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.stages[0].smem_A.data()), InterleavedSmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.stages[0].smem_B.data()), InterleavedSmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    // Block scaling
    Tensor sSFA = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFA.data()), make_layout(
        make_shape(get<0>(shape(SmemLayoutSFA{})),
                   get<1>(TileShape{}),
                   make_shape(get<1>(shape(SmemLayoutSFA{})),
                   get<2>(shape(SmemLayoutSFA{})))),
        make_stride(get<0>(stride(SmemLayoutSFA{})), _0{},
                    make_stride(get<1>(stride(SmemLayoutSFA{})), get<2>(stride(SmemLayoutSFA{}))))
      ));                                                                                       // (BLK_M,BLK_N,(BLK_K,P))
    Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.data()), make_layout(
        make_shape(get<0>(TileShape{}),
                   get<0>(shape(SmemLayoutSFB{})),
                   make_shape(get<1>(shape(SmemLayoutSFB{})),
                   get<2>(shape(SmemLayoutSFB{})))),
        make_stride(_0{},
                    get<0>(stride(SmemLayoutSFB{})),
                    make_stride(get<1>(stride(SmemLayoutSFB{})),
                    get<2>(stride(SmemLayoutSFB{}))))
      ));                                                                                       // (BLK_M,BLK_N,(BLK_K,P))

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA);                 // (MMA,MMA_M,MMA_N,(MMA_K,PIPE))
    Tensor tCsSFB = tiled_mma.get_slice(thread_idx).partition_C(sSFB);                 // (MMA,MMA_M,MMA_N,(MMA_K,PIPE))

    Tensor tCsA = thread_mma.partition_A(sA);                                                  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                  // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                            // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                            // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                          // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                          // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                           // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                        // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                          // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                          // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Per block scale values for operand A and B
    // Since scale factors always broadcast across MMA_K we slice that away
    Tensor tCrSFA = make_tensor_like<ElementBlockScale>(tCsSFA(_, _, _, _0{}));                     // (MMA,MMA_M,MMA_N)
    Tensor tCrSFB = make_tensor_like<ElementBlockScale>(tCsSFB(_, _, _, _0{}));                     // (MMA,MMA_M,MMA_N)

    // Prologue GMMAs

    if ((is_fused_m2_m3 || is_fused || IsFusedM4M5) && sub_m_idx > 0) tiled_mma.accumulate_ = GMMA::ScaleOut::One;
    else tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
    StrassenGmmaFP8Accumulation accumulation(accum, ScalePromotionInterval, size<2>(tCrA));
    warpgroup_fence_operand(accumulation());
    {
      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers
      copy(tCsSFA(_,_,_,make_coord(_0{}, read_stage)), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, read_stage)), tCrSFB);

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();
      warpgroup_fence_operand(accumulation());

      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
      }
      warpgroup_wait<0>();
      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFA, tCrSFB);
      }
    }

    warpgroup_fence_operand(accumulation());
    // Mainloop GMMAs
    k_tile_count -= 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count)
    {
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers (at most twice per block along M and/or N)
      copy(tCsSFA(_,_,_,make_coord(_0{}, read_stage)), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, read_stage)), tCrSFB);

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_fence_operand(accumulation());

      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
      }
      warpgroup_wait<0>();
      pipeline.consumer_release(smem_pipe_release); // Unlock previous tile
      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFA, tCrSFB);
      }

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_release;
    }
    if (k_tile_count) {
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();

      // Load per block scale values from shared memory to registers (at most twice per block along M and/or N)
      copy(tCsSFA(_,_,_,make_coord(_0{}, read_stage)), tCrSFA);
      copy(tCsSFB(_,_,_,make_coord(_0{}, read_stage)), tCrSFB);

      if constexpr (ScalePromotionInterval != 4) {
        if (accumulation.prepare_if_needed()) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
        }
      }
      else {
        // Always zero out the accumulator for finest granularity
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      }

      warpgroup_fence_operand(accumulation());
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_fence_operand(accumulation());

      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_b = tCrSFB(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
          filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
        }
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        ElementBlockScale scale_a = tCrSFA(_0{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(filter_zeros(tCrSFB)); i++) {
          filter_zeros(tCrSFB)(i) = filter_zeros(tCrSFB)(i) * scale_a;
        }
      }
      warpgroup_wait<0>();
      pipeline.consumer_release(smem_pipe_release); // Unlock previous tile
      // Block scale the accumulators with reg tensor `tCrSFA` and `tCrSFB`
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        scale_if_needed(accumulation, scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        scale_if_needed(accumulation, tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        scale_if_needed(accumulation, tCrSFA, tCrSFB);
      }
    }
    if constexpr (ScalePromotionInterval != 4) {
      // residues only exists when granularity is not the finnest
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
        ElementBlockScale scale_ab = tCrSFA(_0{});
        accumulation.scale_residue_if_needed(scale_ab);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile == 1) {
        accumulation.scale_residue_if_needed(tCrSFA);
      }
      if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrSFB);
      }
      if constexpr (ScaleMsPerTile  > 1 && ScaleNsPerTile  > 1) {
        accumulation.scale_residue_if_needed(tCrSFA, tCrSFB);
      }
    }

    warpgroup_fence_operand(accumulation());
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // The pipeline is not released in the first iteration
    smem_pipe_release.advance(k_tile_count - 1);
    pipeline.consumer_release(smem_pipe_release);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
