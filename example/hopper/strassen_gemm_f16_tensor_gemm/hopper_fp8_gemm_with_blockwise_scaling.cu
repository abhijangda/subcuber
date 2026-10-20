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

/*! \file
    \brief Simple Hopper GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    This example demonstrate a simple way to instantiate and run a TF32 GEMM using the new CUTLASS 3.0
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:

    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.

    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. Another advantage is that TMA can load in FP32 data and
    convert them implicitly to TF32.

    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).

    4. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the 
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can 
    improve performance.

    Examples:

      $ ./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm --m=2048 --n=2048 --k=2048 --rasterization=N --swizzle=2
*/

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

#define MY_PRINTF(...) ;//printf(__VA_ARGS__)

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/layout/strassen_layout.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_strassen_gemm_builder.hpp"
#include "cutlass/epilogue/collective/collective_strassen_builder.hpp"
#include "cutlass/gemm/device/strassen_gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/strassen_gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gett.hpp"

#include "cutlass/gemm/device/strassen_decls.h"
using namespace MmaStrassen;

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::float_e4m3_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
using         SubMatLayoutA = cutlass::layout::StrassenLayout;
// using         SubMatLayoutA = cutlass::layout::OriginalLayout;
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::float_e4m3_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
using         SubMatLayoutB = cutlass::layout::StrassenLayout;
// using         SubMatLayoutB = cutlass::layout::OriginalLayout;
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = void;//TensorOpElem;                                           // No C source operand
using         ElementD    = cutlass::half_t;                                // Element type for D matrix operand
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
using         SubMatLayoutC = cutlass::layout::OriginalLayout;
using         LayoutD     = LayoutC;
using         LayoutScalar = cutlass::layout::PackedVectorLayout;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C/D matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ElementBlockScale   = float;                                          // Element type for blockscaling during accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag

using ProblemShape = Shape<int,int,int>; // M,N,K

#if defined(PINGPONG)
using TileShape           = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using TileShapeM0         = TileShape;
using TileShapeM2To6      = TileShape;
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
const uint StageCountTypeM0   = 6;                                          // Stage count maximized based on the tile size
const uint StageCountTypeM2M6 = 6;
const uint StageCountTypeM4M5 = 6;
using PresumTileShapeA    = Shape<_2, _128>;
using PresumTileShapeB    = Shape<_2, _128>;
using KernelScheduleM0 = cutlass::gemm::KernelTmaWarpSpecializedPingpong;       // Kernel to launch based on the default setting in the Collective Builder
using EpilogueScheduleM0 = cutlass::epilogue::TmaWarpSpecialized;
using KernelScheduleM2To6 = KernelScheduleM0;
using EpilogueScheduleM2To6 = EpilogueScheduleM0;
#elif defined(COOPERATIVE)
using TileShapeM0           = Shape<_128,_128,_128>;                           // Threadblock-level tile size
using TileShapeM2To6        = Shape<_128,_128,_128>;
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster
const uint StageCountTypeM0   = cute::is_void_v<ElementC> ? 6 : 6;          // Stage count maximized based on the tile size
const uint StageCountTypeM2M6 = cute::is_void_v<ElementC> ? 5 : 5;
const uint StageCountTypeM4M5 = cute::is_void_v<ElementC> ? 5 : 5;
using PresumTileShapeA    = Shape<_4, _128>;
using PresumTileShapeB    = Shape<_4, _128>;
using KernelScheduleM2To6 = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum;       // Kernel to launch based on the default setting in the Collective Builder
using EpilogueScheduleM2To6 = cutlass::epilogue::TmaWarpSpecializedCooperative;
using KernelScheduleM0 = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum;
using EpilogueScheduleM0 = cutlass::epilogue::TmaWarpSpecializedCooperative;
#elif defined(COOPERATIVE_PINGPONG)
using TileShapeM0         = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using TileShapeM2To6      = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
const uint StageCountTypeM0 = 4 ; //cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
const uint StageCountTypeM2M6 = 6 ;
const uint StageCountTypeM4M5 = 6;
using PresumTileShapeA    = Shape<_2, _256>;
using PresumTileShapeB    = Shape<_2, _256>;
using KernelScheduleM2To6 = cutlass::gemm::KernelTmaWarpSpecializedPingpong;       // Kernel to launch based on the default setting in the Collective Builder
using EpilogueScheduleM2To6 = cutlass::epilogue::TmaWarpSpecialized;
using KernelScheduleM0 = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueScheduleM0 = cutlass::epilogue::TmaWarpSpecializedCooperative;
#endif

using ScaleConfig =     cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;

using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand

using PresumOpts = cutlass::gemm::device::PresumOpt<>;//0,0,0,0>;
//StageCount = 6 is a little slower than this with swizzle = 8.
//TODO: Stages 5 produces wrong results for C2

using AllPresumsKernel = AllPresums<>;
using AllPresumsM0    =  AllPresums<PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel,  //A Presums
                                    PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel>; //B Presums
// using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                  //  PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel>;//PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
//TODO: Can also divide presum among M0 and M1 if K * K/N is not big enough
//TODO: If PresumShape and K/TK cannot cover all of A and B then report error

using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,    PresumAvailable,    PresumAvailable,    PresumAvailable>;

#if 1 //TMA Reduce
// #if defined(COOPERATIVE)
//   static_assert(StageCountTypeM4M5 == 4, "StageCountTypeM4M5 should be 4 for TMA Reduce");
// #endif

using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShapeM0, AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>> >, //C1 = Sh = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>/*, Expr<Plus<1, MemShared, LayoutInterim1D>>*/ >, //C2 = C1Sh+M3
                                                                           CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>> >,
                                                                  AllPresumsM1To6, 0, 2, 3, 6>,
                                            StrassenLevel1M3Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes</*CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>,*/ //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<3, MemGlobal, LayoutFinal>>>,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutFinal>>//,
                                                                                                                               /*Plus<0, MemShared, LayoutInterim1D>*/
                                                                                                                               >
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresumsM1To6>
                                            >;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<KernelScheduleM0, EpilogueScheduleM0, false, FusedMiGroup<7, 0>>, //FusedMiGroup<7, 2>, FusedMiGroup<7, 4>>
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 2>>, //TODO: Change this to true
                                                                                                                          //  FusedMiGroup<7, 4>>
                                                                                // FusedMiGroup<7, 6>>
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 2>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 4>>
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 6>>
                                                        >;
#elif 0
#if defined(COOPERATIVE_PINGPONG)
#error "This schedule do not work with mixed schedule"
#endif
#error "This is deprecated schedule"
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShapeM0, AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>> >, //C1 = Sh = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                           CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>/*, Expr<Plus<1, MemShared, LayoutInterim1D>>*/ >, //C2 = C1Sh+M3
                                                                           CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>> >,
                                                                  AllPresumsM1To6, 0, 2, 3, 6>,
                                            StrassenLevel1M3Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes</*CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>,*/ //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>> >,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>//,
                                                                                                                               /*Plus<0, MemShared, LayoutInterim1D>*/
                                                                                                                               >
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresumsM1To6>
                                            >;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<KernelScheduleM0, EpilogueScheduleM0, false, FusedMiGroup<7, 0>>,
                                                        ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 2>>, //TODO: Change this to true
                                                                                                                            // FusedMiGroup<7, 4>>
                                                                                // FusedMiGroup<7, 6>>
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 2>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 4>>
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 6>>
                                                        >;

#elif 1
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShapeM0, AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShapeM0, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim>> >, //C1 = Sh = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                           CUW<2, LayoutInterim, LayoutNone, Expr<Plus<3>>/*, Expr<Plus<1, MemShared, LayoutInterim1D>>*/ >, //C2 = C1Sh+M3
                                                                           CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>> >,
                                                                  AllPresumsM1To6, 0, 2, 3, 6>,
                                            StrassenLevel1M3Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes</*CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>,*/ //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim>> >,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim>>//,
                                                                                                                               /*Plus<0, MemShared, LayoutInterim1D>*/
                                                                                                                               >
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresumsM1To6>
                                            >;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<KernelScheduleM0, EpilogueScheduleM0, false, FusedMiGroup<7, 0>>,
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 2>>, //TODO: Change this to true
                                                                                                                          //  FusedMiGroup<7, 4>>
                                                                                // FusedMiGroup<7, 6>>
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 2>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 4>>
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 6>>
                                                        >;
#else

using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, AllPresumsKernel>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShape, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>, //C1 = Sh = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                           CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>/*, Expr<Plus<1, MemShared, LayoutInterim1D>>*/ >>, //C2 = C1Sh+M3
                                                                  AllPresumsM1To6, 0, 2, 3>,
                                            StrassenLevel1M3Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M4Group<1, 0, TileShape, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutInterim1D, LayoutNone,  Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>,//C3 = C2+M4
                                                                           >,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M5Group<1, 0, TileShape, ClusterShape, StageCountTypeM4M5,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresumsM1To6>
                                            >;
//For 8kx8kx8k. do not create a single kernel and run as
// strassen_winograd_presum4_hopper_f16_tensorop_gemm --m=$((8*1024)) --n=$((8*1024)) --k=$((8*1024)) --iterations=100 --check=0 --streams=7 --swizzles=2,2,1,1,1,1,1 --beta=0 --raster=N
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<false, FusedMiGroup<7, 0>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 2>>, //TODO: Change this to true
                                                                              //  FusedMiGroup<7, 4>,
                                                                              //  FusedMiGroup<7, 5>,
                                                                              //  FusedMiGroup<7, 6>>
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 4>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 5>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 6>>
                                                        >;
#endif

using StrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<StrassenGroups, ScheduleStrassenGroups1,
                                                                       ProblemShape,
                                                                       ArchTag, OperatorClass,
                                                                       ElementA, cute::tuple<LayoutA, LayoutSFA>, SubMatLayoutA,
                                                                       ElementB, cute::tuple<LayoutB, LayoutSFB>, SubMatLayoutB,
                                                                       ElementC, LayoutC, SubMatLayoutC,
                                                                       ElementAccumulator, ClusterShape,
                                                                       cute::Int<StageCountTypeM0>,
                                                                       PresumTileShapeA, PresumTileShapeB,
                                                                       PresumOpts,
                                                                       AlignmentA, AlignmentB, AlignmentC,
                                                                       ElementD>;

using Gemm = cutlass::gemm::device::StrassenGemmUniversalAdapter<StrassenGemmKernels>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
LayoutSFA layout_SFA;
LayoutSFB layout_SFB;
uint64_t seed;

cutlass::HostTensor<typename Gemm::ElementA, LayoutA> tensor_A;
cutlass::HostTensor<typename Gemm::ElementB, LayoutB> tensor_B;
cutlass::HostTensor<ElementAccumulator, LayoutScalar> blockscale_tensor_A;
cutlass::HostTensor<ElementAccumulator, LayoutScalar> blockscale_tensor_B;
cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, LayoutD> tensor_D;
cutlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, LayoutD> tensor_ref_D;
std::array<std::vector<ElementA>, 4> reference_presum_A;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  int l = 1;
  RasterOrderOptions raster;
  int swizzles[7];
  bool reference_check;
  int streams;

  Options():
    help(false),
    m(5120), n(4096), k(4096),
    alpha(1.f), beta(0.f),
    reference_check(true),
    iterations(1000),
    raster(RasterOrderOptions::Heuristic),
    streams(1)

  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("mnk")) {
      int mnk = 0;
      cmd.get_cmd_line_argument("mnk", mnk);
      m = n = k = mnk;
    } else {
      cmd.get_cmd_line_argument("m", m);
      cmd.get_cmd_line_argument("n", n);
      cmd.get_cmd_line_argument("k", k);
    }

    cmd.get_cmd_line_argument("streams", streams);
    cmd.get_cmd_line_argument("check", reference_check);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }
    
    std::string str_swizzles;
    std::string default_swizzle = "1,1,1,1,1,1,1";
    cmd.get_cmd_line_argument("swizzles", str_swizzles, default_swizzle);

    std::stringstream str_swizzles_stream(str_swizzles);

    std::string str_swizzle;
    if (cute::is_same_v<SubMatLayoutA, cutlass::layout::StrassenLayout> &&  
        cute::is_same_v<SubMatLayoutB, cutlass::layout::StrassenLayout>) {
      std::getline(str_swizzles_stream, str_swizzle, ',');
      for (int ii = 0; ii < 7; ii++) swizzles[ii] = stoi(str_swizzle);
    } else {
      int idx = 0;
      while(std::getline(str_swizzles_stream, str_swizzle, ','))
      {
        if (idx < 7)
          swizzles[idx] = stoi(str_swizzle);
        idx++; 
      }
      for (int ii = idx; ii < 7; ii++) swizzles[ii] = 1;
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "48_hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP16 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --streams=<int>             Number of overlapping streams.\n\n"
      << "  --check=<0|1>               Check results or not.\n\n";


    out
      << "\n\nExamples:\n\n"
      << "$ " << "48_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element, class Layout>
bool initialize_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed, bool ones = false) {
  if (dist_kind != cutlass::Distribution::Uniform) {
    throw std::runtime_error("Unsupported operand distribution.");
  }
  if (!ones) {
    cutlass::reference::host::BlockFillRandomUniform(
      view.data(), view.capacity(), seed, 4, -4, 0);
  } else {
    cutlass::reference::host::TensorFill(view, Element(1.f));    
  }

  return true;
}

/// Helper to initialize a block of device data (scale_tensors)
template <typename Element, typename Layout>
bool initialize_scale_tensor(
  cutlass::TensorView<Element, Layout> view,
  cutlass::Distribution::Kind dist_kind,
  uint64_t seed, bool ones = false) {

  if (!ones) {

    double scope_max, scope_min;

    scope_min = -4;
    scope_max = 4;

    cutlass::reference::host::BlockFillRandomUniform(
      view.data(), view.capacity(), seed, scope_max, scope_min);
  }
  else if (ones) {
    cutlass::reference::host::TensorFill(view, Element(1.f));
  }

  return true;
}

template <class Element>
void upload_strassen_layout(
    Element const* original, Element* packed, int outer, int inner, bool column_major) {
  int half_outer = outer / 2;
  int half_inner = inner / 2;
  size_t quadrant_elements = size_t(half_outer) * half_inner;
  for (int quadrant = 0; quadrant < 4; ++quadrant) {
    int outer_quadrant = column_major ? quadrant % 2 : quadrant / 2;
    int inner_quadrant = column_major ? quadrant / 2 : quadrant % 2;
    auto source = original + size_t(outer_quadrant * half_outer) * inner +
                  inner_quadrant * half_inner;
    CUDA_CHECK(cudaMemcpy2D(packed + quadrant * quadrant_elements,
                           size_t(half_inner) * sizeof(Element),
                           source, size_t(inner) * sizeof(Element),
                           size_t(half_inner) * sizeof(Element), half_outer,
                           cudaMemcpyHostToDevice));
  }
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, options.l));
  layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, options.l));

  auto a_coord = cutlass::make_Coord(options.m * options.l, options.k);
  auto d_coord = cutlass::make_Coord(options.m * options.l, options.n);
  auto b_coord = cutlass::make_Coord(options.k, options.n * options.l);
  auto blockscale_a_coord = cutlass::make_Coord(size(filter_zeros(layout_SFA)));
  auto blockscale_b_coord = cutlass::make_Coord(size(filter_zeros(layout_SFB)));

  tensor_A.resize(a_coord);
  blockscale_tensor_A.resize(blockscale_a_coord);
  tensor_B.resize(b_coord);
  blockscale_tensor_B.resize(blockscale_b_coord);
  tensor_D.resize(d_coord);
  tensor_ref_D.resize(d_coord);

  cutlass::Distribution::Kind dist_A = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_B = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleA = cutlass::Distribution::Uniform;
  cutlass::Distribution::Kind dist_scaleB = cutlass::Distribution::Uniform;

  initialize_tensor(tensor_A.host_view(), dist_A, seed + 2022, false);
  initialize_tensor(tensor_B.host_view(), dist_B, seed + 2023, false);
  initialize_scale_tensor(blockscale_tensor_A.host_view(), dist_scaleA, seed + 2025, false);
  initialize_scale_tensor(blockscale_tensor_B.host_view(), dist_scaleB, seed + 2026, false);
  cutlass::reference::host::TensorFill(tensor_D.host_view());

#if 0 // Dump blockscaled tensors
  std::cout << "blockscale_tensor_A: " << blockscale_a_coord << std::endl;
  std::cout << blockscale_tensor_A.host_view() << "\n";
  std::cout << "blockscale_tensor_B: " << blockscale_b_coord << std::endl;
  std::cout << blockscale_tensor_B.host_view() << "\n";
#endif

  // Print block scaling tensors on the host side.
  if constexpr (cute::is_same_v<SubMatLayoutA, cutlass::layout::StrassenLayout>) {
    upload_strassen_layout(tensor_A.host_data(), tensor_A.device_data(), options.m, options.k, false);
  } else {
    tensor_A.sync_device();
  }
  if constexpr (cute::is_same_v<SubMatLayoutB, cutlass::layout::StrassenLayout>) {
    upload_strassen_layout(tensor_B.host_data(), tensor_B.device_data(), options.n, options.k, true);
  } else {
    tensor_B.sync_device();
  }
  tensor_D.sync_device();
  blockscale_tensor_A.sync_device();
  blockscale_tensor_B.sync_device();
}


/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {tensor_A.device_data(),
     stride_A,
     tensor_B.device_data(),
     stride_B,
     blockscale_tensor_A.device_data(),
     layout_SFA,
     blockscale_tensor_B.device_data(),
     layout_SFB
     },
    {
      {}, // epilogue.thread
      nullptr, StrideC{},
      tensor_D.device_data(), stride_D
    },
    kernel_hw_info
  };

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8)
  arguments.scheduler.max_swizzle_size = options.swizzles[0];

  return arguments;
}

void strassen_winograd_reference(const Options &options) {
  constexpr int ScaleK = Gemm::CollectiveMainloop::ScaleGranularityK;
  constexpr int ScaleN = Gemm::CollectiveMainloop::ScaleGranularityN;
  static_assert(Gemm::CollectiveMainloop::ScaleGranularityM == 1);
  const int halfM = options.m/2;
  const int halfN = options.n/2;
  const int halfK = options.k/2;
  if (options.m % 2 || options.n % 2 || halfN % ScaleN || halfK % ScaleK || options.k % 2 || options.l != 1) {
    throw std::runtime_error("Strassen reference requires even M/N, N divisible by twice ScaleN, K divisible by twice ScaleK, and L=1.");
  }

  auto A = cute::make_tensor(tensor_A.host_data(),
      cute::make_layout(cute::make_shape(options.m, options.k, options.l), stride_A));
  auto B = cute::make_tensor(tensor_B.host_data(),
      cute::make_layout(cute::make_shape(options.n, options.k, options.l), stride_B));
  auto SFA = cute::make_tensor(blockscale_tensor_A.host_data(), layout_SFA);
  auto SFB = cute::make_tensor(blockscale_tensor_B.host_data(), layout_SFB);

  auto half_shape = make_shape(halfM,halfN,halfK,1);
  auto presum_layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(half_shape);
  auto presum_layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(half_shape);
  auto presum_stride_a = cutlass::make_cute_packed_stride(StrideA{}, {halfM,halfK,1});
  auto presum_stride_b = cutlass::make_cute_packed_stride(StrideB{}, {halfN,halfK,1});
  auto presum_layout_a = make_layout(make_shape(halfM,halfK,1), presum_stride_a);
  auto presum_layout_b = make_layout(make_shape(halfN,halfK,1), presum_stride_b);
  std::array<std::vector<ElementB>,4> presum_b;
  std::array<std::vector<ElementBlockScale>,4> presum_scale_a;
  std::array<std::vector<ElementBlockScale>,4> presum_scale_b;
  for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
    reference_presum_A[presum_idx].resize(size_t(halfM)*halfK);
    presum_b[presum_idx].resize(size_t(halfN)*halfK);
    presum_scale_a[presum_idx].resize(size(filter_zeros(presum_layout_sfa)));
    presum_scale_b[presum_idx].resize(size(filter_zeros(presum_layout_sfb)));
  }

  #pragma omp parallel for collapse(2)
  for (int row = 0; row < halfM; ++row) {
    for (int block = 0; block < halfK/ScaleK; ++block) {
      std::array<std::array<ElementAccumulator,ScaleK>,4> values;
      std::array<ElementAccumulator,4> max_abs{};
      for (int element_idx = 0; element_idx < ScaleK; ++element_idx) {
        int inner = block*ScaleK + element_idx;
        float a0 = float(A(row,inner,0))*SFA(row,inner,0);
        float a1 = float(A(row,inner+halfK,0))*SFA(row,inner+halfK,0);
        float a2 = float(A(row+halfM,inner,0))*SFA(row+halfM,inner,0);
        float a3 = float(A(row+halfM,inner+halfK,0))*SFA(row+halfM,inner+halfK,0);
        values[0][element_idx] = a2 + a3;
        values[1][element_idx] = values[0][element_idx] - a0;
        values[2][element_idx] = a0 - a2;
        values[3][element_idx] = a1 - values[1][element_idx];
        for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
          max_abs[presum_idx] = std::max(max_abs[presum_idx], std::fabs(values[presum_idx][element_idx]));
        }
      }
      for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
        float scale = max_abs[presum_idx] > 0.0f ?
          std::max(max_abs[presum_idx]/float(cutlass::platform::numeric_limits<ElementA>::max()), FLT_MIN) : 1.0f;
        presum_scale_a[presum_idx][presum_layout_sfa(make_coord(row,block*ScaleK,0))] = scale;
        for (int element_idx = 0; element_idx < ScaleK; ++element_idx) {
          auto coord = make_coord(row,block*ScaleK+element_idx,0);
          reference_presum_A[presum_idx][presum_layout_a(coord)] = ElementA(values[presum_idx][element_idx]/scale);
        }
      }
    }
  }

  #pragma omp parallel for collapse(2)
  for (int block_n = 0; block_n < halfN/ScaleN; ++block_n) {
    for (int block_k = 0; block_k < halfK/ScaleK; ++block_k) {
      std::array<std::array<ElementAccumulator,ScaleN*ScaleK>,4> values;
      std::array<ElementAccumulator,4> max_abs{};
      for (int column_idx = 0; column_idx < ScaleN; ++column_idx) {
        for (int inner_idx = 0; inner_idx < ScaleK; ++inner_idx) {
          int column = block_n*ScaleN + column_idx;
          int inner = block_k*ScaleK + inner_idx;
          int element_idx = column_idx*ScaleK + inner_idx;
          float b0 = float(B(column,inner,0))*SFB(column,inner,0);
          float b1 = float(B(column+halfN,inner,0))*SFB(column+halfN,inner,0);
          float b2 = float(B(column,inner+halfK,0))*SFB(column,inner+halfK,0);
          float b3 = float(B(column+halfN,inner+halfK,0))*SFB(column+halfN,inner+halfK,0);
          values[0][element_idx] = b1 - b0;
          values[1][element_idx] = b3 - b1;
          values[2][element_idx] = values[1][element_idx] + b0;
          values[3][element_idx] = values[2][element_idx] - b2;
          for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
            max_abs[presum_idx] = std::max(max_abs[presum_idx], std::fabs(values[presum_idx][element_idx]));
          }
        }
      }
      for (int presum_idx = 0; presum_idx < 4; ++presum_idx) {
        float scale = max_abs[presum_idx] > 0.0f ?
          std::max(max_abs[presum_idx]/float(cutlass::platform::numeric_limits<ElementB>::max()), FLT_MIN) : 1.0f;
        presum_scale_b[presum_idx][presum_layout_sfb(make_coord(block_n*ScaleN,block_k*ScaleK,0))] = scale;
        for (int column_idx = 0; column_idx < ScaleN; ++column_idx) {
          for (int inner_idx = 0; inner_idx < ScaleK; ++inner_idx) {
            auto coord = make_coord(block_n*ScaleN+column_idx,block_k*ScaleK+inner_idx,0);
            presum_b[presum_idx][presum_layout_b(coord)] = ElementB(values[presum_idx][column_idx*ScaleK+inner_idx]/scale);
          }
        }
      }
    }
  }

  auto layout_a = make_layout(make_shape(halfM,halfK,1), stride_A);
  auto layout_b = make_layout(make_shape(halfN,halfK,1), stride_B);
  auto quadrant_layout_sfa = make_layout(shape(presum_layout_sfa), stride(layout_SFA));
  auto quadrant_layout_sfb = make_layout(shape(presum_layout_sfb), stride(layout_SFB));
  auto A0 = make_tensor(A.data(), layout_a);
  auto A1 = make_tensor(A.data()+A.layout()(make_coord(0,halfK,0)), layout_a);
  auto A3 = make_tensor(A.data()+A.layout()(make_coord(halfM,halfK,0)), layout_a);
  auto B0 = make_tensor(B.data(), layout_b);
  auto B2 = make_tensor(B.data()+B.layout()(make_coord(0,halfK,0)), layout_b);
  auto B3 = make_tensor(B.data()+B.layout()(make_coord(halfN,halfK,0)), layout_b);
  auto SFA0 = make_tensor(SFA.data(), quadrant_layout_sfa);
  auto SFA1 = make_tensor(SFA.data()+layout_SFA(make_coord(0,halfK,0)), quadrant_layout_sfa);
  auto SFA3 = make_tensor(SFA.data()+layout_SFA(make_coord(halfM,halfK,0)), quadrant_layout_sfa);
  auto SFB0 = make_tensor(SFB.data(), quadrant_layout_sfb);
  auto SFB2 = make_tensor(SFB.data()+layout_SFB(make_coord(0,halfK,0)), quadrant_layout_sfb);
  auto SFB3 = make_tensor(SFB.data()+layout_SFB(make_coord(halfN,halfK,0)), quadrant_layout_sfb);
  auto input_presum_a = [&](int presum_idx) { return make_tensor(reference_presum_A[presum_idx].data(), presum_layout_a); };
  auto input_presum_b = [&](int presum_idx) { return make_tensor(presum_b[presum_idx].data(), presum_layout_b); };
  auto scale_presum_a = [&](int presum_idx) { return make_tensor(presum_scale_a[presum_idx].data(), presum_layout_sfa); };
  auto scale_presum_b = [&](int presum_idx) { return make_tensor(presum_scale_b[presum_idx].data(), presum_layout_sfb); };

  std::array<std::vector<ElementAccumulator>,7> products;
  auto product_layout = make_layout(make_shape(halfM,halfN,1), make_stride(int64_t(halfN),_1{},int64_t(halfM)*halfN));
  auto reference_gemm = [&](int product_idx, auto input_a, auto scale_a, auto input_b, auto scale_b) {
    products[product_idx].resize(size_t(halfM)*halfN);
    auto product = make_tensor(products[product_idx].data(), product_layout);
    cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator, decltype(input_a), decltype(scale_a), decltype(input_b), decltype(scale_b)
    > mainloop_params{input_a, scale_a, input_b, scale_b};
    cutlass::reference::host::GettEpilogueParams<
      float, float, ElementAccumulator, float, decltype(product), decltype(product)
    > epilogue_params;
    epilogue_params.D = product;
    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);
  };

  reference_gemm(0, A0, SFA0, B0, SFB0);
  reference_gemm(1, A1, SFA1, B2, SFB2);
  reference_gemm(2, input_presum_a(1), scale_presum_a(1), input_presum_b(2), scale_presum_b(2));
  reference_gemm(3, input_presum_a(2), scale_presum_a(2), input_presum_b(1), scale_presum_b(1));
  reference_gemm(4, input_presum_a(0), scale_presum_a(0), input_presum_b(0), scale_presum_b(0));
  reference_gemm(5, input_presum_a(3), scale_presum_a(3), B3, SFB3);
  reference_gemm(6, A3, SFA3, input_presum_b(3), scale_presum_b(3));

  auto D = make_tensor(tensor_ref_D.host_data(),
    make_layout(make_shape(options.m,options.n,options.l), stride_D));
  #pragma omp parallel for
  for (int row = 0; row < halfM; ++row) {
    for (int column = 0; column < halfN; ++column) {
      size_t offset = size_t(row)*halfN + column;
      float m0 = products[0][offset];
      float m1 = products[1][offset];
      float m2 = products[2][offset];
      float m3 = products[3][offset];
      float m4 = products[4][offset];
      float m5 = products[5][offset];
      float m6 = products[6][offset];
      float interim_m0 = float(ElementD(m0));
      float sum_m2 = interim_m0 + m2;
      float sum_m3 = sum_m2 + m3;
      D(row,column,0) = ElementD(m0 + m1);
      D(row,column+halfN,0) = ElementD(float(ElementD(sum_m2)) + m4 + m5);
      D(row+halfM,column,0) = ElementD(sum_m3 - m6);
      D(row+halfM,column+halfN,0) = ElementD(float(ElementD(sum_m3)) + m4);
    }
  }
}

bool verify_a_presums(const Options& options, ElementA const* device_presum_A) {
  const int halfM = options.m / 2;
  const int halfK = options.k / 2;
  const size_t presum_elements = size_t(halfM) * halfK;
  std::vector<ElementA> computed_presum_A(4 * presum_elements);
  CUDA_CHECK(cudaMemcpy(computed_presum_A.data(), device_presum_A,
                        computed_presum_A.size() * sizeof(ElementA), cudaMemcpyDeviceToHost));

  struct PresumCheck {
    char const* name;
    int reference_idx;
    APresums presum;
  };
  constexpr std::array<PresumCheck, 4> checks{{
    {"A02", 2, APresums::A02},
    {"S1", 0, APresums::S1},
    {"S2", 1, APresums::S2},
    {"A1S2", 3, APresums::A1S2},
  }};

  bool all_passed = true;
  for (auto const& check : checks) {
    int workspace_idx = AllPresumsM0::indexAPresum(check.presum);
    bool passed = true;
    for (size_t idx = 0; idx < presum_elements; ++idx) {
      ElementA reference = reference_presum_A[check.reference_idx][idx];
      ElementA computed = computed_presum_A[size_t(workspace_idx) * presum_elements + idx];
      if (float(reference) != float(computed)) {
        size_t row = idx / halfK;
        size_t column = idx % halfK;
        std::cout << "  A presum " << check.name << " mismatch at index " << idx
                  << " (" << row << ", " << column << "): reference " << float(reference)
                  << ", computed " << float(computed) << std::endl;
        passed = false;
        break;
      }
    }
    std::cout << "  A presum " << check.name << ": " << (passed ? "Passed" : "Failed") << std::endl;
    all_passed = all_passed && passed;
  }
  return all_passed;
}

bool verify(const Options &options, ElementA const* device_presum_A) {
  int available_cpus = omp_get_num_procs();
  int reference_threads = available_cpus > 1 ? available_cpus / 2 : 1;
  int previous_threads = omp_get_max_threads();
  int previous_dynamic = omp_get_dynamic();
  omp_set_dynamic(0);
  omp_set_num_threads(reference_threads);
  std::cout << "  Host Strassen-Winograd reference: " << reference_threads << " threads ("
            << available_cpus << " available CPUs)" << std::endl;
  strassen_winograd_reference(options);
  omp_set_num_threads(previous_threads);
  omp_set_dynamic(previous_dynamic);

  bool passed_A_presums = verify_a_presums(options, device_presum_A);

  tensor_D.sync_host();

  constexpr float MAX_REL_ERR = 1e-2f;
  constexpr float MAX_ABS_ERR = 5.0f;
  bool passed_C0 = true;
  bool passed_C1 = true;
  bool passed_C2 = true;
  bool passed_C3 = true;

  auto computed_view = tensor_D.host_view();
  auto reference_view = tensor_ref_D.host_view();
  for (int r = 0; r < options.m; r++) {
    for (int c = 0; c < options.n; c++) {
      auto coord = cutlass::make_Coord(r, c);
      float computed = float(computed_view.at(coord));
      float reference = float(reference_view.at(coord));
      float abs_err = std::fabs(reference - computed);
      float err = abs_err / std::fabs(reference + 1e-6);
      bool equal = reference == computed || err <= MAX_REL_ERR || abs_err <= MAX_ABS_ERR;

      // C0
      if (r < options.m/2 && c < options.n/2 && !equal && passed_C0) {
        printf("C0 mismatch at %d, %d: reference %f, computed %f\n",
               r, c, float(reference), float(computed));
        passed_C0 = false;
      }

      // C1
      if (r < options.m/2 && c >= options.n/2 && !equal && passed_C1) {
        printf("C1 mismatch at %d, %d: reference %f, computed %f\n",
               r, c, float(reference), float(computed));
        passed_C1 = false;
      }

      // C2
      if (r >= options.m/2 && c < options.n/2 && !equal && passed_C2) {
        printf("C2 mismatch at %d, %d: reference %f, computed %f\n",
               r, c, float(reference), float(computed));
        passed_C2 = false;
      }

      // C3
      if (r >= options.m/2 && c >= options.n/2 && !equal && passed_C3) {
        printf("C3 mismatch at %d, %d: reference %f, computed %f\n",
               r, c, float(reference), float(computed));
        passed_C3 = false;
      }
    }
  }

  std::cout << "  C0: " << (passed_C0 ? "Passed" : "Failed") << std::endl;
  std::cout << "  C1: " << (passed_C1 ? "Passed" : "Failed") << std::endl;
  std::cout << "  C2: " << (passed_C2 ? "Passed" : "Failed") << std::endl;
  std::cout << "  C3: " << (passed_C3 ? "Passed" : "Failed") << std::endl;

  return passed_A_presums && passed_C0 && passed_C1 && passed_C2 && passed_C3;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  cudaEvent_t events[7];

  cudaStream_t streams[7];
  for (int i = 0; i < 7; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, 0+i));
    CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
  }

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  std::cout << 464 << " " <<workspace_size << std::endl;
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, options.swizzles, workspace.get()));

  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run(streams, options.streams));

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;

  if (options.reference_check) {
    CUDA_CHECK(cudaDeviceSynchronize());
    result.passed = verify(options, Gemm::get_presum_a_ptr(arguments, workspace.get()));

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    if (!result.passed) {
      exit(-1);
    }
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    // CUTLASS_CHECK(gemm.initialize(arguments, options.swizzles, workspace.get()));
    for (int iter = 0; iter < options.iterations/10; ++iter) {
      CUTLASS_CHECK(gemm.run(streams, options.streams));
      if (options.streams > 1) result.error = cudaDeviceSynchronize();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run(streams, options.streams));
      if (options.streams > 1) {
        // for (int s1 = 0; s1 < options.streams; s1++) {
        //   CUDA_CHECK(cudaEventRecord(events[s1], streams[s1]));
        // }
        // for (int s1 = 0; s1 < options.streams; s1++) {
        //   // for (int s2 = 0; s2 < options.streams; s2++) {
        //     // if (s1 != s2) {
        //       CUDA_CHECK(cudaStreamWaitEvent(streams[s1], events[0]));
        //       CUDA_CHECK(cudaStreamWaitEvent(streams[0], events[s1]));
        //     // }
        //   // }
        // }
        result.error = cudaDeviceSynchronize();
      }
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of ";
    for (int i = 0; i < 7; i++) std::cout << options.swizzles[i] << ", ";
    std::cout <<std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;

    for (int s = 0; s < options.streams; s++) {
      CUDA_CHECK(cudaEventDestroy(events[s]));
      CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9 || props.minor != 0) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
    return 0;
  }

  
  

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if constexpr (!cute::is_void_v<ElementC>) {
    if (options.beta == 0.0f) {
      std::cerr << "Warning: beta=0 with non-void ElementC may reduce performance; "
                   "use ElementC=void when C is not needed.\n";
    }
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
