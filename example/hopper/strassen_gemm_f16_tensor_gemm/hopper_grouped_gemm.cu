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
    \brief Hopper Grouped GEMM example using CUTLASS 3 APIs for NVIDIA Hopper architecture.

    This example demonstrates an implementation of Grouped GEMM using a TMA + GMMA
    warp-specialized cooperative kernel.
    For this example all scheduling work is performed on the device.
    The new feature showcased in this example is on-the-fly modification of TMA descriptors
    to move between groups/problem_count (represented by groups).

    To run this example:

      $ ./examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm --m=2048 --n=2048 --k=2048 --groups=10

      The above example command makes all 10 groups to be sized at the given m, n, k sizes.
      Skipping any of the problem dimensions randomizes it across the different groups.
      Same applies for alpha and beta values that are randomized across the different groups.

    To run this example for a set of problems using the benchmark option:

      $ ./examples/57_hopper_grouped_gemm/57_hopper_grouped_gemm --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x128
        1 256x512x512
        2 512x256x128
        3 256x256x128
        4 256x512x1024
        5 1024x512x128 and so on
*/

#include <iostream>

#define MY_PRINTF(...) ;//printf(__VA_ARGS__)

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_strassen_gemm_builder.hpp"
#include "cutlass/epilogue/collective/collective_strassen_builder.hpp"
#include "cutlass/gemm/device/strassen_gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/strassen_gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass/gemm/device/strassen_decls.h"
using namespace MmaStrassen;

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag

#ifdef GROUPED
using ProblemShape = cutlass::gemm::StrassenGroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
constexpr auto GemmMode = cutlass::gemm::GemmUniversalMode::kGrouped;
#elif defined(MOE)
using ProblemShape = cutlass::gemm::StrassenMoEProblemShape<Shape<int,int,int>>; // <M,N,K> per group
constexpr auto GemmMode = cutlass::gemm::GemmUniversalMode::kMoE;
#endif

#if defined(PINGPONG)
using TileShape           = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using TileShapeM0         = TileShape;
using TileShapeM2To6      = TileShape;
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
const uint StageCountTypeM0 = 6 ; //cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
const uint StageCountTypeM2M6 = 6 ;
using PresumTileShapeA    = Shape<_2, _128>;
using PresumTileShapeB    = Shape<_2, _128>;
using KernelScheduleM0 = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;       // Kernel to launch based on the default setting in the Collective Builder
using EpilogueScheduleM0 = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using KernelScheduleM2To6 = KernelScheduleM0;
using EpilogueScheduleM2To6 = EpilogueScheduleM0;
#elif defined(COOPERATIVE)
using TileShapeM0           = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using TileShapeM2To6        = Shape<_128,_256,_64>;
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
const uint StageCountTypeM0 = 4 ; //cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
const uint StageCountTypeM2M6 = 4 ;
using PresumTileShapeA    = Shape<_2, _256>;
using PresumTileShapeB    = Shape<_2, _256>;
using KernelScheduleM0 = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueScheduleM0 = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using KernelScheduleM2To6 = KernelScheduleM0;       // Kernel to launch based on the default setting in the Collective Builder
using EpilogueScheduleM2To6 = EpilogueScheduleM0;
#elif defined(COOPERATIVE_PINGPONG)
using TileShapeM0         = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using TileShapeM2To6      = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
const uint StageCountTypeM0 = 4 ; //cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
const uint StageCountTypeM2M6 = 6 ;
using PresumTileShapeA    = Shape<_2, _256>;
using PresumTileShapeB    = Shape<_2, _256>;
using KernelScheduleM2To6 = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;       // Kernel to launch based on the default setting in the Collective Builder
using EpilogueScheduleM2To6 = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using KernelScheduleM0 = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueScheduleM0 = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
#endif

using PresumOpts = cutlass::gemm::device::PresumOpt<0,0,0,0>;
//StageCount = 6 is a little slower than this with swizzle = 8.
//TODO: Stages 5 produces wrong results for C2

using AllPresumsKernel = AllPresums<>;
                          // using AllPresumsM0    =  AllPresums<PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel,  //A Presums
                                        //  PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel>; //B Presums
using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                   PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
//TODO: Can also divide presum among M0 and M1 if K * K/N is not big enough
//TODO: If PresumShape and K/TK cannot cover all of A and B then report error

using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,    PresumAvailable,    PresumAvailable,    PresumAvailable>;

#if 0 //TMA Reduce
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
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes</*CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>,*/ //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<3, MemGlobal, LayoutFinal>>>,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutFinal>>//,
                                                                                                                               /*Plus<0, MemShared, LayoutInterim1D>*/
                                                                                                                               >
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
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
                                                       ParallelMiGroups<KernelScheduleM2To6, EpilogueScheduleM2To6, false, FusedMiGroup<7, 2>, //TODO: Change this to true
                                                                                                                           FusedMiGroup<7, 4>>
                                                                                // FusedMiGroup<7, 6>>
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 2>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 4>>
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 6>>
                                                        >;
#elif 0
#if defined(COOPERATIVE_PINGPONG)
#error "This schedule do not work with mixed schedule"
#endif
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
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes</*CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>,*/ //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>> >,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>//,
                                                                                                                               /*Plus<0, MemShared, LayoutInterim1D>*/
                                                                                                                               >
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
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
                                            StrassenLevel1MiGroup<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes</*CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>,*/ //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim>> >,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim>>//,
                                                                                                                               /*Plus<0, MemShared, LayoutInterim1D>*/
                                                                                                                               >
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShapeM2To6, ClusterShape, StageCountTypeM2M6,
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
                                            StrassenLevel1M4Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutInterim1D, LayoutNone,  Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>,//C3 = C2+M4
                                                                           >,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M5Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
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
                                                                       ElementA, LayoutA *, ElementB, LayoutB *,
                                                                       ElementC, LayoutC *,
                                                                       ElementAccumulator, ClusterShape,
                                                                       cute::Int<StageCountTypeM0>,
                                                                       PresumTileShapeA, PresumTileShapeB,
                                                                       PresumOpts>;

using Gemm = cutlass::gemm::device::StrassenGemmUniversalAdapter<StrassenGemmKernels>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideC = typename Gemm::GemmKernel::InternalStrideC;
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

//
// Data members
//

// Host-side allocations
std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;

std::vector<uint64_t> ptr_A_batch_indices_host;
std::vector<uint64_t> ptr_B_batch_indices_host;
std::vector<uint64_t> ptr_C_batch_indices_host;
std::vector<uint64_t> ptr_D_batch_indices_host;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_ref_D;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;
cutlass::DeviceAllocation<StrideD> stride_D;
cutlass::DeviceAllocation<uint64_t> ptr_A_batch_indices;
cutlass::DeviceAllocation<uint64_t> ptr_B_batch_indices;
cutlass::DeviceAllocation<uint64_t> ptr_C_batch_indices;
cutlass::DeviceAllocation<uint64_t> ptr_D_batch_indices;

// Note, this is an array of pointers to alpha and beta scaling values per group
cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
cutlass::DeviceAllocation<ElementAccumulator> block_beta;

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
  int m, n, k, groups;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  RasterOrderOptions raster;
  int const tma_alignment_bits = 128;
  int const alignment = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
  int swizzles[7];
  bool reference_check;
  int streams;

  Options():
    help(false),
    m(5120), n(4096), k(4096), groups(1),
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

    cmd.get_cmd_line_argument("groups", groups);
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
    int idx = 0;
    while(std::getline(str_swizzles_stream, str_swizzle, ','))
    {
      if (idx < 7)
        swizzles[idx] = stoi(str_swizzle);
      idx++; 
    }
    for (int ii = idx; ii < 7; ii++) swizzles[ii] = 1;

    if (groups > 0) {
      problem_sizes_host.assign(groups, {m, n, k});
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
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
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
template <class Element>
bool initialize_block(
  Element* block, int group,
  uint64_t size,
  uint64_t seed=2023, bool mod = false, bool ones = false) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(4);
    scope_min = Element(-4);
  }

  if (!mod and !ones) {
    cutlass::reference::device::BlockFillRandomUniform(
      block, size, seed, scope_max, scope_min, 0);
  } else if (mod) {
    Element* host = new Element[size];
    for (uint64_t i = 0; i < size; i++)
      host[i] = static_cast<Element>(static_cast<int>(i / (9 * 1024)));
    CUDA_CHECK(cudaMemcpy(block, host, size * sizeof(Element), cudaMemcpyHostToDevice));
    delete[] host;
  } else if (ones) {
    using Layout = cutlass::layout::PackedVectorLayout;
    Layout::TensorCoord extent(static_cast<Layout::Index>(size)); // -Wconversion
    Layout layout = Layout::packed(extent);
    cutlass::TensorView<Element, Layout> view(block, layout, extent);

    cutlass::reference::device::TensorFill(
      view, Element(1+int(group)));
  }

  return true;
}

/// Allocates device-side data
void allocate(const Options &options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;
  uint64_t total_rows_A = 0;
  uint64_t total_rows_B = 0;
  uint64_t total_rows_C = 0;
  uint64_t total_rows_D = 0;

  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);
    ptr_A_batch_indices_host.push_back(total_rows_A);
    ptr_B_batch_indices_host.push_back(total_rows_B);
    ptr_C_batch_indices_host.push_back(total_rows_C);
    ptr_D_batch_indices_host.push_back(total_rows_D);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
    total_rows_A += M;
    total_rows_B += K;
    total_rows_C += M;
    total_rows_D += M;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  uint64_t seed = 2020;

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(options.groups);
  std::vector<ElementB *> ptr_B_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementC *> ptr_D_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {
    // If the current group's matrix has size 0, set the pointer to nullptr
    if (i < options.groups - 1 && offset_A.at(i) == offset_A.at(i + 1)) {
      ptr_A_host.at(i) = nullptr;
    } else {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    }
    if (i < options.groups - 1 && offset_B.at(i) == offset_B.at(i + 1)) {
      ptr_B_host.at(i) = nullptr;
    } else {
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    }
    if (i < options.groups - 1 && offset_C.at(i) == offset_C.at(i + 1)) {
      ptr_C_host.at(i) = nullptr;
    } else {
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    }
    if (i < options.groups - 1 && offset_D.at(i) == offset_D.at(i + 1)) {
      ptr_D_host.at(i) = nullptr;
    } else {
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  ptr_A_batch_indices.reset(options.groups);
  ptr_A_batch_indices.copy_from_host(ptr_A_batch_indices_host.data());
  ptr_B_batch_indices.reset(options.groups);
  ptr_B_batch_indices.copy_from_host(ptr_B_batch_indices_host.data());
  ptr_C_batch_indices.reset(options.groups);
  ptr_C_batch_indices.copy_from_host(ptr_C_batch_indices_host.data());
  ptr_D_batch_indices.reset(options.groups);
  ptr_D_batch_indices.copy_from_host(ptr_D_batch_indices_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    uint64_t elements_A = uint64_t(get<0>(problem)) * uint64_t(get<2>(problem));
    uint64_t elements_B = uint64_t(get<2>(problem)) * uint64_t(get<1>(problem));

    initialize_block(block_A.get() + offset_A.at(i), i, elements_A, seed + 2021 + 2*i, false, true);
    initialize_block(block_B.get() + offset_B.at(i), i, elements_B, seed + 2022 + 2*i, false, true);
  }
  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);
  // kernel_hw_info.sm_count = ;

  typename Gemm::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;

  if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    fusion_args.alpha = options.alpha;
    fusion_args.beta = options.beta;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    // Single alpha and beta for all groups
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  }
  else {
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
    fusion_args.alpha = 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = alpha_device.get();
    fusion_args.beta_ptr_array = beta_device.get();
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
  }

  if (true) {
    arguments = typename Gemm::Arguments {
      GemmMode,
      {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
    #if defined(MOE)
      {block_A.get(), stride_A.get(), block_B.get(), stride_B.get(),
       ptr_A_batch_indices.get(), ptr_B_batch_indices.get()},
      {fusion_args, nullptr, nullptr, block_D.get(), stride_D.get(),
       nullptr, ptr_D_batch_indices.get()},
    #else
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {fusion_args, nullptr, nullptr, ptr_D.get(), stride_D.get()},
    #endif
      kernel_hw_info
    };
  }
  else {
    arguments = typename Gemm::Arguments {
      GemmMode,
      {options.groups, problem_sizes.get(), nullptr},
    #if defined(MOE)
      {block_A.get(), stride_A.get(), block_B.get(), stride_B.get(),
       ptr_A_batch_indices.get(), ptr_B_batch_indices.get()},
      {fusion_args, block_C.get(), stride_C.get(), block_D.get(), stride_D.get(),
       ptr_C_batch_indices.get(), ptr_D_batch_indices.get()},
    #else
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
    #endif
      kernel_hw_info
    };
  }

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8) 
  // arguments.scheduler.max_swizzle_size = options.swizzles;

  return arguments;
}

bool verify(const Options &options) {
  bool passed = true;
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    printf("Checking problem %d\n", i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);
    cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), Gemm::LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), Gemm::LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), Gemm::LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), Gemm::LayoutD::packed({M, N}));

    //
    // Compute reference output
    //

    // Create instantiation for device reference gemm kernel
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(
      {M, N, K},
      ElementAccumulator(alpha_host.at(i)),
      ref_A,
      ref_B,
      ElementAccumulator(beta_host.at(i)),
      ref_C,
      ref_D);

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    ElementC* host_ref_D = new ElementC[options.m*options.n];
    CUDA_CHECK(cudaMemcpy(host_ref_D, block_ref_D.get()  + offset_D.at(i), options.m*options.n*sizeof(ElementC), cudaMemcpyDeviceToHost));
    ElementC* host_D = new ElementC[options.m*options.n];
    CUDA_CHECK(cudaMemcpy(host_D, block_D.get() + offset_D.at(i), options.m*options.n*sizeof(ElementC), cudaMemcpyDeviceToHost));

    float MAX_REL_ERR = 1e-2;
    float MAX_ABS_ERR = 5;

    for (int r = 0; r < options.m; r++) {
    for (int c = 0; c < options.n; c++) {
      auto idx = r*options.n + c;
      cutlass::half_t e1 = host_ref_D[idx];
      cutlass::half_t e2 = host_D[idx];
      float err = fabs((float)e1 -(float)e2)/fabs((float)e1 + 1e-6);
      float abs_err = fabs((float)e1 -(float)e2);

      //C0
      if (r < options.m/2 && c < options.n/2) {
        if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
          printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
          passed = false;
        }
      }

      //C1
      if (r < options.m/2 && c >= options.n/2) {
        if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
          printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
          passed = false;
        }
      }

      //C2
      if (r >= options.m/2 && c < options.n/2) {
        if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
          printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
          passed = false;
        }
      }

      //C3
      if (r >= options.m/2 && c >= options.n/2) {
        if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
          printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
          passed = false;
        }
      }
      // if (r > 0 || c > 4112) 
      if (!passed) break;
    }
    if (!passed) break;
    }
    if(!passed) break;
  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  allocate(options);
  initialize(options);

  std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
  for (int32_t i = 0; i < options.groups; ++i) {
    std::cout << "    " << options.problem_sizes_host.at(i);
    std::cout << ", "   << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
  }
  std::cout << "  Groups      : " << options.groups  << std::endl;

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
    result.passed = verify(options);

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
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0, options.problem_sizes_host);

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

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
