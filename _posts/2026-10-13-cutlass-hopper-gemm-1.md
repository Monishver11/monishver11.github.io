---
layout: post
title: Hopper GEMM in CUTLASS C++ and the CuTe DSL, part 1, the layers and the builder [W4A8]
date: 2026-10-13 00:00:00-0400
featured: false
description: Part 1 of a series that reads the CUTLASS 3.x C++ Hopper GEMM stack from the source and puts each piece next to its CuTe DSL equivalent, as the base for porting vLLM's W4A8 kernel to the CuTe DSL. This part covers the five levels of CUTLASS 3.x, how a kernel is assembled from them, and what the CollectiveBuilder chooses
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This is part 1 of a series on the Hopper GEMM, written two times: once in CUTLASS 3.x C++ and once in the CuTe Python DSL. Each part reads one piece of the C++ kernel and then shows the same piece in the CuTe DSL. The series builds on the [CuTe DSL fundamentals post](/blog/2026/fa3-cute/). That post explains the Hopper primitives one at a time: layouts, TMA, mbarriers, pipelines, WGMMA and warp specialization. This series shows how the primitives fit together in a complete GEMM kernel, in both languages.

The reason for the series is a port. vLLM has a W4A8 GEMM for Hopper, with FP8 activations and INT4 weights, in `csrc/libtorch_stable/quantization/cutlass_w4a8/`. It is written with CUTLASS 3.x in C++. I want to write it again in the CuTe DSL. For that, I need two kinds of knowledge:

- **What the kernel does.** The warp roles, the mainloop, the INT4 to FP8 conversion, the scales, the epilogue and the tile configurations. The only source for this is the C++ code. The CuTe DSL has no Hopper example of a mixed-input GEMM.
- **How to write it in the CuTe DSL.** The DSL has no `CollectiveBuilder` and no collective classes, so every choice that the C++ templates make must be written by hand.

So each section has two halves. The first half reads the C++. The second half shows the matching CuTe DSL code.

The rule for the series is the same as for the DSL post. Every statement about the hardware or the library comes from a primary source: the CUTLASS source, the CUTLASS documentation, the CUDA PTX ISA, or NVIDIA's Hopper material. Each section ends with its references. If a statement has no source, the post marks it as unverified. I have no local GPU, so nothing in the series was compiled or run. Every code excerpt is read from the source, not from a run.

A note on versions. vLLM builds its C++ kernels against CUTLASS `v4.7.1`. Its `CMakeLists.txt` sets `CUTLASS_REVISION` to that tag. The CuTe DSL that vLLM pins, `nvidia-cutlass-dsl`, is version `4.8.0`. So the series quotes the C++ source from the `v4.7.1` tag, because that is the code that vLLM compiles. It quotes the CuTe DSL source and examples from the `v4.8.0` tag, because that is the DSL that the port will run on. The DSL post ran its kernels on version `4.5.1` and quoted the `v4.7.0` source, so some DSL names in this series are newer. I compared the files that the series uses against the `v4.8.0` tag with `diff`. The SM90 kernel, the SM90 mainloop collectives, the SM90 epilogue collective, the SM90 builder and the SM90 tile scheduler are identical in both tags. Seven shared headers are different, for example `gemm/dispatch_policy.hpp` and `gemm/device/gemm_universal_adapter.h`. In six of them, `v4.8.0` only adds lines and removes none. Only `pipeline/sm90_pipeline.hpp` changes existing code: `PipelineState::advance` has a new branch for a single stage. Where a difference matters, the section says so.

The running example is CUTLASS's example 49, `examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu`. It builds Hopper GEMMs with the `CollectiveBuilder` and runs several kernel schedules. Two of its variants use the kernel schedule `KernelTmaWarpSpecializedCooperative` and the epilogue schedule `TmaWarpSpecializedCooperative`. One adds the stream-K tile scheduler. The other uses the persistent tile scheduler and a custom epilogue visitor tree (lines 647 to 652). The series follows the second one, because W4A8 also uses the persistent scheduler and an epilogue visitor tree. The W4A8 kernel uses these two schedules too (`w4a8_mm_entry.cu`, lines 102 to 107). The large difference is the mainloop collective. The dense mainloop reads both operands of the tensor core instruction from shared memory. The W4A8 mainloop converts the INT4 weights in registers and gives them to the instruction from registers. The series explains the dense stack first, and then the W4A8 mainloop.

The CuTe DSL side uses one example: `examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py` in the `v4.8.0` tag. Its class `HopperWgmmaGemmPersistentKernel` has the same structure as the C++ cooperative kernel. It has one warpgroup for TMA loads and one or two warpgroups for the MMA. It sets the register counts to 40 for the load warpgroup and 232 for the MMA warpgroups (lines 306 and 307). The C++ cooperative kernel uses the same two values, unless the accumulator needs 208 or more registers per MMA thread. In that case it uses 24 and 240 (`sm90_gemm_tma_warpspecialized_cooperative.hpp`, lines 132 to 151). It loops over output tiles with a persistent tile scheduler, and it stores the output with TMA. The same folder has `dense_gemm.py`, but that kernel is not warp-specialized, so this series uses the persistent file.

The series. Every part covers the C++ code and the matching CuTe DSL code.

- **Part 1 (this post): the layers and the builder.** The five levels of CUTLASS 3.x (device, kernel, collective, tiled MMA and copy, atom) and what replaces each level in the DSL, how example 49 and the W4A8 kernel assemble a kernel, and what the `CollectiveBuilder` chooses that the DSL code must choose by hand.
- **Part 2: the kernel.** The warp-specialized cooperative kernel, the tile scheduler and the persistent loop, and clusters with TMA multicast.
- **Part 3: the mainloop.** The producer side with TMA loads, the pipeline classes, and the consumer side with WGMMA.
- **Part 4: the epilogue and the host.** The TMA store, the epilogue visitor tree, and the host side: arguments, TMA descriptors and the launch.
- **Part 5: W4A8.** The mixed-input mainloop that reads one operand from registers, and how to write it in the DSL, where no example exists.

The plan for this part:

- The five levels of CUTLASS 3.x, and how a level picks its implementation
- How example 49 and the W4A8 kernel assemble a kernel from these layers
- The `CollectiveBuilder`: what it chooses, and how

Let's get started.

---

#### **The five levels of CUTLASS 3.x** {#levels}

First, why the levels exist. A GEMM kernel is a set of choices: the hardware instruction, the arrangement of instructions over threads, the order of copies and MMAs over time, the order of output tiles over the grid, and the work that is done on the accumulator before the store. NVIDIA's [CUTLASS 3.x design post](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) calls these choices largely independent, but the best combination depends on the hardware and on the problem shape. So CUTLASS 3.x gives each choice its own level. You can change the choice at one level and keep the other levels as they are. The post calls this design a "hierarchical system of composable, orthogonal building blocks".

| Choice | Level that owns it | Example of a change at that level only |
|---|---|---|
| The hardware instruction | Atom | A BF16 WGMMA or an FP8 WGMMA |
| The arrangement of instructions over threads (space) | Tiled MMA and copy | One or two warpgroups for each tile |
| The order of work over time: pipeline stages, warp roles, barriers | Collective | The dense mainloop or the mixed-input mainloop |
| The output tile that each CTA computes next | Kernel, through its tile scheduler | A persistent scheduler or a stream-K scheduler |
| The work on the accumulator before the store | Epilogue collective | A plain store or a scaling epilogue |

There are two ways to use the levels. In the first way, you give the `CollectiveBuilder` the data types, the tile shape and the schedule, and it chooses the rest. In the second way, you write one level yourself and keep the others. The W4A8 kernel uses both ways. It uses the builder, and it changes only two things: the mainloop and the epilogue fusion. Its kernel, tile scheduler and device adapter are the same as for a dense GEMM.

The CUTLASS 3.x GEMM API has five levels. The GEMM API document lists them from the highest level to the lowest: device, kernel, collective, tiled MMA and copy, and atom. Each level has one main class, and each class is a template over the classes of the levels below it. The table gives the class at each level, the file where the Hopper version of it lives, and the job of the level. The text in quotation marks in the last column is quoted from the GEMM API document. The rest of that column is my summary.

| Level | Main class | Hopper source (v4.7.1) | Job |
|---|---|---|---|
| Device | `cutlass::gemm::device::GemmUniversalAdapter` | `gemm/device/gemm_universal_adapter.h` | A host-side handle. It converts the user's `Arguments` into the kernel's `Params`, and it launches the kernel. |
| Kernel | `cutlass::gemm::kernel::GemmUniversal` | `gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` | "A collection of all clusters in the grid." It gives each warp its role, and it schedules the collectives over output tiles. |
| Collective | `CollectiveMma` (the mainloop) and `CollectiveEpilogue` | `gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` and `epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` | "The largest collection of threads onto which mma atoms and copy atoms are tiled." The mainloop does the loop over K tiles. The epilogue writes the output tile. |
| Tiled MMA and copy | `cute::TiledMMA` and `cute::TiledCopy`, called through `cute::gemm()` and `cute::copy()` | `cute/atom/mma_atom.hpp` and `cute/atom/copy_atom.hpp` | One MMA or one copy over a whole tile, made from many atoms. |
| Atom | `cute::MMA_Atom` and `cute::Copy_Atom` | `cute/atom/mma_atom.hpp` and `cute/atom/copy_atom.hpp`. The Hopper instructions that they wrap are in `cute/arch/mma_sm90_gmma.hpp` and `cute/arch/copy_sm90_tma.hpp`. | "The smallest collection of threads and data that must participate in the execution of a hardware-accelerated math or copy operation." On Hopper, one WGMMA instruction or one TMA instruction. |

<div class="row justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/cutlass-hopper-gemm/five-levels.svg" title="The five levels of CUTLASS 3.x" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The five levels as nested templates. The device adapter wraps the kernel. The kernel holds a mainloop collective and an epilogue collective. Each collective uses tiled operations, and each tiled operation is made of atoms. On the right, the dense mainloop policy and the W4A8 mainloop policy both name the cooperative schedule, so both select the same cooperative kernel. Editable source: <a href="/assets/img/cutlass-hopper-gemm/five-levels.excalidraw">five-levels.excalidraw</a>.
</div>

The device level is the only level that runs on the host. The other four levels run on the GPU. The device level does not choose the grid shape. The document says that the kernel controls the grid shape and the other launch parameters, so that all CUTLASS 3.x kernels can share one launch path. Part 4 of this series shows this launch path.

##### **The same five levels in the CuTe DSL**

The two lowest levels are CuTe, and the CuTe DSL has the same two levels. The three highest levels are CUTLASS classes, and the CuTe DSL does not have them. In the DSL, one Python class does the work of all three. The table shows where each level's work is in `dense_gemm_persistent.py` (`v4.8.0`).

| Level | CuTe DSL equivalent in `dense_gemm_persistent.py` |
|---|---|
| Device | The method `__call__`, decorated with `@cute.jit` (line 404). It runs on the host when the kernel is compiled. It makes the TMA atoms, computes the grid, declares the shared memory struct, and calls `self.kernel(...).launch(grid=..., block=..., cluster=..., stream=...)` (lines 510 to 530). |
| Kernel | The method `kernel`, decorated with `@cute.kernel` (line 533). It splits the warps into roles by warpgroup index, sets the register counts with `setmaxregister_decrease` and `setmaxregister_increase`, and runs `while work_tile.is_valid_tile` over a `StaticPersistentTileScheduler`. The load side is at lines 723 to 735, and the MMA side at lines 783 to 849. |
| Collective | No separate class. The mainloop is two loops in `kernel`: the producer's loop over K tiles in the load warp (line 742), and the consumer's loop over K tiles in the MMA warpgroups (lines 860 and 882). The epilogue is a third loop in the MMA warpgroups (lines 933 to 969). It uses a `PipelineTmaStore` created at line 844. |
| Tiled MMA and copy | `sm90_utils.make_trivial_tiled_mma(...)` (line 348) returns a `TiledMma`. The calls are `cute.gemm` and `cute.copy`, the same as in C++. |
| Atom | `make_trivial_tiled_mma` builds the MMA atom with `cute.make_mma_atom(MmaF16BF16Op(...))` or `MmaF8Op` (`utils/hopper_helpers.py`, line 170). The TMA atoms come from `cute.nvgpu.cpasync.make_tiled_tma_atom` with `CopyBulkTensorTileG2SOp`, `CopyBulkTensorTileG2SMulticastOp` or `CopyBulkTensorTileS2GOp` (lines 1214 to 1251). |

So a `TiledMma` built with `make_trivial_tiled_mma` is the Python version of a `cute::TiledMMA` in C++. For the three highest levels, the port has no library class to call. The port's `kernel` method must contain the code that the C++ kernel class and the two collective classes contain.

##### **Where each level sits in the GEMM loop nest**

The GEMM API document maps the levels onto the three nested loops of a tiled GEMM. Here is its pseudocode, shortened:

```c++
// kernel::GemmUniversal: the cluster-tile loops. The hardware runs them as the grid,
// or a persistent kernel runs them as a while loop over the tile scheduler.
for (int cluster_m = 0; cluster_m < GemmM; cluster_m += ClusterTileM) {
  for (int cluster_n = 0; cluster_n < GemmN; cluster_n += ClusterTileN) {

    // collective::CollectiveMma: the mainloop over all K tiles.
    for (int k_tile = 0; k_tile < size<2>(gmem_tensor_A); k_tile++) {

      // inside cute::gemm(tiled_mma, a, b, c): fully static, fully unrolled.
      for (int tiled_mma_k = 0; tiled_mma_k < size<2>(A); tiled_mma_k++) {
        for (int tiled_mma_m = 0; tiled_mma_m < size<1>(A); tiled_mma_m++) {
          for (int tiled_mma_n = 0; tiled_mma_n < size<1>(B); tiled_mma_n++) {
            mma.call(d, a, b, c);   // the atom: one hardware instruction
          }
        }
      }
    }
  }
}
```

The two outer loops belong to the kernel level. The document says that a non-persistent kernel does not write these loops. The grid launch runs them, one CTA (or one cluster) for each output tile. A persistent kernel writes them as one `while` loop that asks the tile scheduler for the next tile. The cooperative kernel in this series is persistent. If `GemmUniversal` gets no tile scheduler, as in `w4a8_mm_entry.cu`, `TileSchedulerSelector` selects `PersistentTileSchedulerSm90` (`gemm/kernel/tile_scheduler.hpp`). Part 2 explains its loop.

The `k_tile` loop belongs to the collective. The document says that this loop is not unrolled. In the document's model, the three innermost loops are inside one call to `cute::gemm`. The document says that these loops are generally fully static and are fully unrolled. The Hopper mainloop follows this model with one exception. For the first K tile only, it writes the `k_block` loop around `cute::gemm` by hand. It sets the accumulate flag to zero before the first K block and to one after it, so the first WGMMA overwrites the accumulator (`sm90_mma_tma_gmma_ss_warpspecialized.hpp`, lines 491 to 497). For every later K tile, it calls `cute::gemm` once on the whole K-tile slice, `tCrA(_,_,_,read_stage)`, so the K-block, M and N loops are all inside the call (lines 508 to 552). The DSL post writes a `range_constexpr` loop over K blocks around `cute.gemm` in every K tile.

In `dense_gemm_persistent.py`, the same loop nest is written out in Python. This is the consumer side, shortened (lines 849 to 897):

```python
while work_tile.is_valid_tile:                        # kernel level: next output tile
    ...
    accumulators.fill(0.0)
    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
    ...                                               # prologue K tiles omitted
    for k_tile in range(prologue_mma_cnt, k_tile_cnt):        # collective: K tiles, runtime loop
        mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
        for k_block_idx in cutlass.range_constexpr(num_k_blocks):   # K blocks, unrolled
            k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
            cute.gemm(tiled_mma, accumulators,
                      tCrA[k_block_coord], tCrB[k_block_coord], accumulators)  # M, N inside
        ...
```

Each C++ loop has a Python twin:

| Loop | C++ (cooperative kernel and SS mainloop) | CuTe DSL (`dense_gemm_persistent.py`) |
|---|---|---|
| Output tiles | `while (work_tile_info.is_valid())` in the kernel class | `while work_tile.is_valid_tile` over `StaticPersistentTileScheduler` |
| K tiles | `CUTLASS_PRAGMA_NO_UNROLL for ( ; k_tile_count > 0; --k_tile_count)` in the collective's `mma()` | `for k_tile in range(...)`, a runtime loop |
| K blocks | First K tile: `CUTLASS_PRAGMA_UNROLL for (int k_block ...)` around `cute::gemm`. Other K tiles: inside `cute::gemm`. | `for k_block_idx in cutlass.range_constexpr(...)` around `cute.gemm` in every K tile, unrolled at compile time |
| M and N inside one K block | inside `cute::gemm` | inside `cute.gemm` |

The two versions also differ in how they start the accumulator. The C++ mainloop sets the accumulate flag to zero for the first K block, so the first WGMMA overwrites the accumulator. The DSL example instead fills the accumulator with zeros and sets the flag to accumulate for every K block (lines 856 and 857). Both give the same result. The C++ form does not need the zero fill.

##### **How a level picks its implementation: dispatch policies**

CUTLASS 3.x uses one name for many implementations. For example, every 3.x kernel is a `GemmUniversal`. A tag type selects the implementation. This tag type is the dispatch policy. The CUTLASS 3.x design document says that mainloop implementations, epilogue implementations and kernel schedules are selected with tag-dispatch policies, not with new type names.

The chain starts at the mainloop. The mainloop's dispatch policy is a small struct. For the dense Hopper mainloop, it is this (`gemm/dispatch_policy.hpp`, v4.7.1):

```c++
template<
  int Stages_,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule = KernelTmaWarpSpecializedCooperative
>
struct MainloopSm90TmaGmmaWarpSpecialized {
  constexpr static int Stages = Stages_;
  using ClusterShape = ClusterShape_;
  using ArchTag = arch::Sm90;
  using Schedule = KernelSchedule;
};
```

The policy carries four facts: the number of shared memory stages, the cluster shape, the architecture (`Sm90`), and the kernel schedule. The `CollectiveMma` template is specialized on this policy type, so the policy selects the mainloop implementation.

The kernel level then reads the schedule out of the mainloop's policy. The cooperative kernel is a partial specialization of `GemmUniversal` with this condition (`sm90_gemm_tma_warpspecialized_cooperative.hpp`, lines 60 to 65):

```c++
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<cute::is_base_of_v<KernelTmaWarpSpecializedCooperative, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
```

So the kernel template has no schedule parameter of its own. The compiler selects the cooperative kernel if the mainloop's `Schedule` is `KernelTmaWarpSpecializedCooperative`, or a type derived from it. The condition uses `is_base_of`, so derived tags match too. `dispatch_policy.hpp` derives several tags from the cooperative tag, for example `KernelTmaWarpSpecializedCooperativeFP8FastAccum` and `KernelTmaWarpSpecializedCooperativeFP8Blockwise`. Each of these selects the same cooperative kernel.

This chain is how W4A8 reuses the dense kernel. The `CollectiveBuilder` gives W4A8 a different mainloop policy, `MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput`, because its two operands have different widths, 4 bits and 8 bits. The builder tests this width difference at line 361 of `collective/builders/sm90_gmma_builder.inl` and selects the policy at lines 473 to 477. That policy's `Schedule` is the schedule that `w4a8_mm_entry.cu` passes, `KernelTmaWarpSpecializedCooperative`. So W4A8 has a different mainloop and the same kernel. Its epilogue collective is the same class too, with a different epilogue visitor tree. Parts 2 to 4 of this series explain this shared kernel, epilogue and host path. Part 5 explains the different mainloop.

##### **In the CuTe DSL: no dispatch policies**

The CuTe DSL has no tag dispatch. `dense_gemm_persistent.py` is one Python class with one kernel. Its choices are constructor arguments and Python attributes, for example the tile shape, the cluster shape and `atom_layout_mnk` (lines 251 to 306). If a choice changes the generated code, the kernel reads it as a compile-time value, and `cutlass.const_expr` removes the branches that are not used. The DSL post explains this mechanism in [the JIT model](/blog/2026/fa3-cute/#jit-specialization).

So the C++ dispatch chain becomes plain code in the port. The port does not select a cooperative kernel class and a mixed-input mainloop class. It is one class whose `kernel` method contains the cooperative warp roles and the mixed-input mainloop directly.

**References**
- NVIDIA Technical Blog, [CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) (why the levels exist)
- CUTLASS v4.7.1, [`media/docs/cpp/gemm_api_3x.md`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/media/docs/cpp/gemm_api_3x.md) (the five levels, the loop-nest pseudocode, the definitions of collective, kernel and atom, and the launch note) and [`media/docs/cpp/cutlass_3x_design.md`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/media/docs/cpp/cutlass_3x_design.md) (tag-dispatch policies)
- [`include/cutlass/gemm/dispatch_policy.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/dispatch_policy.hpp) (the kernel schedule tags and the mainloop policies)
- [`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) (the kernel specialization)
- [`include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl) (the mixed-input policy choice)
- vLLM, `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu` (the W4A8 schedules)
- CUTLASS v4.8.0, [`examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py) (the DSL kernel) and [`python/CuTeDSL/cutlass/utils/hopper_helpers.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/python/CuTeDSL/cutlass/utils/hopper_helpers.py) (`make_trivial_tiled_mma`)

---

#### **How a kernel is assembled** {#assembly}

This section reads the type definitions that build one kernel. It reads example 49 first, then the W4A8 kernel, and then the CuTe DSL code that does the same job.

##### **Example 49: four type definitions, in a fixed order**

Example 49 builds its kernel with four `using` declarations (`49_collective_builder.cu`, lines 307 to 337). Here they are, shortened:

```c++
// 1. The epilogue collective.
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128,_128,_64>, Shape<_1,_1,_1>,          // tile shape, cluster shape
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleType,
    cute::conditional_t<UseCustomEVT, CustomEVT, DefaultOperation>>::CollectiveOp;   // CustomEVT in our variant

// 2. The mainloop collective. Its stage count depends on the epilogue's shared memory.
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    Shape<_128,_128,_64>, Shape<_2,_1,_1>,          // tile shape, cluster shape
    cutlass::gemm::collective::StageCountAutoCarveout<                  // when StageCountType is the
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,   // default StageCountAuto
    MainloopScheduleType>::CollectiveOp;

// 3. The kernel: a problem shape plus the two collectives.
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, TileSchedulerType>;

// 4. The host-side handle.
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

The order is not free. The epilogue comes first because the mainloop builder needs its size. `StageCountAutoCarveout<N>` tells the mainloop builder to keep `N` bytes of shared memory free (`collective_builder_decl.hpp`, line 50). Here `N` is the size of the epilogue's shared storage. The builder then fills the rest of the shared memory with as many mainloop stages as fit. Section 3 shows this calculation.

The problem shape `Shape<int,int,int,int>` is (M, N, K, L), where L is the batch count. All four values are `int`, so they are runtime values, and one compiled kernel serves every problem size. The tile shape and the cluster shape use `_128` and `_2`, which are compile-time constants.

There is one surprise in the example. The epilogue builder gets the cluster shape `1x1x1`, and the mainloop builder gets `2x1x1`. In `v4.7.1`, every SM90 epilogue builder takes the cluster shape as a template parameter, but no SM90 epilogue builder uses it in its body. Some builders only pass it on to another builder (`epilogue/collective/builders/sm90_builder.inl`). The TMA warp-specialized builder and the implementation it uses do not read it. So the mismatch has no effect. I found this by a search of the builder source, not from a statement in the documentation.

##### **The W4A8 kernel: the same four definitions, with the operands swapped**

The W4A8 kernel uses the same four definitions in the same order (`w4a8_mm_entry.cu`, lines 113 to 156). Four things are different.

**The K tile.** The tile shape is a template parameter `TileShape_MN`, with the K extent added by `cute::append`:

```c++
static int constexpr TileShapeK = 128 * 8 / sizeof_bits<MmaType>::value;   // MmaType = float_e4m3_t
```

`sizeof_bits` of FP8 is 8, so `TileShapeK` is 128 elements, which is 128 bytes of FP8. In the INT4 operand, a 128-element K tile is 64 bytes.

**The operands are swapped.** vLLM computes D = A·B, where A is the FP8 activation matrix (M by K, row-major) and B is the INT4 weight matrix (K by N, column-major). But the mainloop builder gets B in its first slot and A in its second slot:

```c++
using CollectiveMainloopShuffled = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementB, cutlass::Array<ElementScale, ScalePackSize>>,   // slot A: INT4 weights + scales
    LayoutB_Reordered, AlignmentB,
    ElementA, LayoutA_Transpose, AlignmentA,                               // slot B: FP8 activations
    ElementAccumulator, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;
```

So the kernel computes the transpose, Dᵀ = Bᵀ·Aᵀ. Bᵀ is N by K, and Aᵀ is K by M. The epilogue builder gets `LayoutTranspose<LayoutC>` and `LayoutTranspose<LayoutD>`, which is column-major. A column-major N by M matrix has the same bytes as a row-major M by N matrix. So the kernel writes Dᵀ, and the memory holds D in the layout that vLLM expects.

<div class="row justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/cutlass-hopper-gemm/operand-swap.svg" title="The W4A8 operand swap" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Left: the product that vLLM asks for. Right: the product that the kernel computes. The INT4 weights and their scales go into the builder's A slot, because only the A operand of WGMMA can come from registers, where the conversion to FP8 happens. The FP8 activations go into the B slot. The kernel writes D transposed in column-major order, which is the same memory as D in row-major order. Editable source: <a href="/assets/img/cutlass-hopper-gemm/operand-swap.excalidraw">operand-swap.excalidraw</a>.
</div>

Three facts explain the swap. The first two come from the README of CUTLASS example 55, together with one PTX rule.

1. The README says that the narrow type must go through the register file, where it is converted to the wider type. The PTX ISA allows the A operand of WGMMA to come from registers, but the B operand must come from shared memory. So the INT4 weights must be the A operand.
2. The README says that if the narrow type is in the B slot, the collective swaps and transposes the operands itself, and that the TMA epilogues do not support this implicit swap. My inference from this: a kernel that wants a TMA epilogue must do the swap explicitly, as W4A8 does. The comment in `w4a8_mm_entry.cu` says the same: the cooperative TMA epilogue is "the only epi supporting the required swap + transpose".
3. The README says that for an 8-bit by 4-bit GEMM, both operands must be K-major. `LayoutA_Transpose` is column-major for a K by M matrix, which is K-major. The INT4 weights are column-major K by N, which is also K-major.

The builder code shows reason 2. It sets `SwapAB = IsMixedInput ? !IsATransformed : ...` (`sm90_gmma_builder.inl`, line 413). `IsATransformed` is true if the A slot holds a tuple. The builder also wraps a narrow A type without scales in a tuple (line 396), so it is true in that case too. W4A8 puts its tuple in the A slot, so `SwapAB` is false, and the builder does not swap a second time.

**The scale type is an array.** The first slot is `cute::tuple<int4b_t, cutlass::Array<float_e4m3_t, 8>>`. The tuple tells the builder that the INT4 operand has scales. The README says that the scale-only mode for FP8 by INT4 is "significantly slower" than direct conversion, and that a lookup-table method fixes this. To select the lookup-table method, the scale type must be `cutlass::Array<ElementScale, 8>`. The README adds that this method needs a different encoding of the weights and the scales. In vLLM, two C++ ops make this encoding when the weights are loaded (`w4a8_mm_entry.cu`, lines 380 to 429). `cutlass_encode_and_reorder_int4b` re-encodes the INT4 values with the nibble lookup table in `w4a8_utils.cu` and reorders them. `cutlass_pack_scale_fp8` stores each FP8 scale as an `Array` of 8 elements. The Python method `process_weights_after_loading` in `mixed_precision/cutlass.py` calls both ops (lines 79 and 86). Part 5 explains both.

The fourth difference, `LayoutB_Reordered`, is the weight layout that `compute_memory_reordering_atom` makes. Part 5 explains it too.

##### **In the CuTe DSL: a constructor, a host method and a setup method**

The CuTe DSL has no builder, so `dense_gemm_persistent.py` makes the same choices in three methods of one class.

1. **`__init__`** (line 251) takes the choices that the user makes: the accumulator type, the tile shape (M, N), the cluster shape (M, N), and two tile-order options, `swizzle_size` and `raster_along_m`. The K extent of the tile is not an argument.
2. **`__call__`** (line 405, with `@cute.jit` on line 404) reads the data types and the layouts from the input tensors (lines 433 to 438). It checks them, calls `_setup_attributes`, makes the TMA atoms, computes the grid, declares the shared memory struct, and launches the kernel.
3. **`_setup_attributes`** (line 328) does the work of the two builders. It makes the tiled MMA, sets the K extent of the tile, computes the cluster layout and the multicast counts, picks the epilogue tile, computes the stage counts, and makes the shared memory layouts.

Two of these choices follow the C++ code closely.

**The K extent of the tile** is four MMA instructions: `mma_inst_tile_k = 4` (line 358), and the tile K is the instruction's K times 4. The K of one WGMMA is 16 for 16-bit types and 32 for FP8, so the tile K is 64 for BF16 and 128 for FP8. For FP8, this is the same 128 as W4A8's `TileShapeK`.

**The stage count** follows the same principle as `StageCountAutoCarveout`. The epilogue's shared memory is reserved first, and the mainloop gets the rest (`_compute_stages`, lines 1008 to 1022):

```python
ab_bytes_per_stage = size(A tile) * a_bits // 8 + size(B tile) * b_bits // 8
epi_bytes = size(epi_tile) * c_bits // 8 * 4          # epi_stage = 4
ab_stage = (smem_capacity // occupancy - (1024 + epi_bytes)) // ab_bytes_per_stage
```

The 1024 bytes are kept for the mbarriers. The C++ builder differs in the details. It rounds the bytes of a stage and the carveout up to multiples of 128, and it adds the size of one stage's pipeline barriers to each stage (`sm90_gmma_builder.inl`, lines 71 to 85). The DSL code reserves a flat 1024 bytes, and it computes the epilogue's bytes from the epilogue tile and 4 epilogue stages. For a 128 by 128 BF16 tile with K = 64, one stage holds 128·64·2 + 128·64·2 = 32,768 bytes. Section 3 completes this calculation with the epilogue tile and the shared memory capacity.

##### **What the port must change here**

The DSL example accepts only operands of the same width. `__call__` raises an error if the widths are different (lines 444 to 447):

```python
if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
    raise TypeError(f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}")
```

So the port starts from this class, but it changes these parts:

| Part | Dense DSL example | W4A8 port |
|---|---|---|
| Operand widths | Equal, checked in `__call__` | 4 bits (weights) and 8 bits (activations) |
| Operand order | A and B as given | The weights in the A position and the activations in the B position, as in the C++ swap |
| Output | C as given | The output written as Dᵀ (N by M, column-major), which is D in memory |
| Extra inputs | None | The group scales, and the per-token and per-channel scales |
| Stage count | A and B bytes per stage | INT4 bytes plus FP8 bytes plus scale bytes per stage |
| K extent of the tile | Four MMA instructions | 128, the same value for FP8 |
| Weight and scale preparation | None | Two C++ ops run when the weights are loaded. The port must keep them or write them again. |

The table lists what must change. It does not say how. The later parts show how, one piece at a time.

**References**
- CUTLASS v4.7.1, [`examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu) (the four type definitions)
- CUTLASS v4.7.1, [`examples/55_hopper_mixed_dtype_gemm/README.md`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/examples/55_hopper_mixed_dtype_gemm/README.md) (the narrow type goes through registers, the implicit swap and the TMA epilogue, the K-major rule, the lookup-table scale mode)
- PTX ISA, [`wgmma.mma_async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma) (A may be in registers, B must be in shared memory)
- CUTLASS v4.7.1, [`include/cutlass/gemm/collective/collective_builder_decl.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/collective/collective_builder_decl.hpp) (`StageCountAutoCarveout`), [`include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl) (`SwapAB`), [`include/cutlass/epilogue/collective/builders/sm90_builder.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/epilogue/collective/builders/sm90_builder.inl) (the epilogue builder)
- vLLM, `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu` (the W4A8 type definitions)
- CUTLASS v4.8.0, [`examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py) (`__init__`, `__call__`, `_setup_attributes`, `_compute_stages`)

---

#### **What the CollectiveBuilder chooses** {#builder}

The mainloop builder turns about ten template arguments into a complete `CollectiveMma` type. This section opens the builder and lists each choice that it makes. Then it shows the CuTe DSL code that makes the same choice by hand. The numbers are for the example 49 variant: FP16 operands, a 128 by 128 by 64 tile, a 2x1x1 cluster and the cooperative schedule.

The builder is a set of partial specializations of `CollectiveBuilder`. Example 49 selects the one marked `GMMA_TMA_WS_SS` (`collective/builders/sm90_gmma_builder.inl`, lines 195 to 309). "WS" means warp-specialized, and "SS" means that WGMMA reads both operands from shared memory. W4A8 selects a different one, `GMMA_TMA_WS_RS`, which starts at line 313. The last subsection explains the differences.

##### **The six choices of the SS builder**

| # | Choice | C++ code in the builder | Result for example 49 |
|---|---|---|---|
| 1 | Warpgroups per tile | `AtomLayoutMNK` is `Shape<_2,_1,_1>` if the schedule is cooperative, else `Shape<_1,_1,_1>` | 2 MMA warpgroups. Each one computes 64 rows of the 128-row tile. |
| 2 | The tiled MMA | `make_tiled_mma(GMMA::ss_op_selector<...>(), AtomLayoutMNK{})` | A WGMMA atom of 64 by 128 by 16 (FP16), tiled 2 times along M |
| 3 | The TMA copy for A and for B | `sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape))` for A, and `shape<0>` for B | A: `SM90_TMA_LOAD` (cluster N is 1). B: `SM90_TMA_LOAD_MULTICAST` (cluster M is 2). |
| 4 | The shared memory layout atoms | `ss_smem_selector<Major, Element, BLK_MN, BLK_K>()` | `Layout_K_SW128_Atom` for both A and B |
| 5 | The number of pipeline stages | `compute_stage_count_or_override<...>(StageCountType{})` | 6 stages (calculation below) |
| 6 | The dispatch policy | `MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, Schedule>`, or the `FP8` version of it for FP8 inputs | `MainloopSm90TmaGmmaWarpSpecialized<6, Shape<_2,_1,_1>, KernelTmaWarpSpecializedCooperative>` |

The builder also sets `SmemCopyAtomA` and `SmemCopyAtomB` to `void`. In an SS mainloop, no thread copies an operand from shared memory to registers, because WGMMA reads both operands from shared memory through descriptors. This line changes for W4A8.

Choices 3, 4 and 5 need more explanation.

**Choice 3, the TMA copies.** In a 2x1x1 cluster, the two CTAs compute two tiles that are next to each other along M. They use different rows of A, but the same columns of B. So one TMA load of a B tile can write it into the shared memory of both CTAs. This is TMA multicast. The rule in the builder follows from this. The A tile is shared by the CTAs along N, so A uses multicast only if the cluster's N extent is more than 1. The B tile is shared by the CTAs along M, so B uses multicast only if the cluster's M extent is more than 1. Part 2 explains multicast in detail.

**Choice 4, the shared memory layout atom.** For a K-major operand, `ss_smem_selector` takes the widest of four layout atoms whose K width divides the tile's K extent (`collective/builders/sm90_common.inl`, lines 307 to 324). It tests the 128-byte swizzle atom first, then the 64-byte and the 32-byte swizzle atoms, and last the interleaved atom, which has no swizzle. For FP16 with K = 64, one row of the tile is 64 · 2 = 128 bytes, so the builder picks the 128-byte swizzle, `Layout_K_SW128_Atom`. The [DSL post](/blog/2026/fa3-cute/#swizzle) explains this swizzle.

**Choice 5, the stage count.** `StageCountAutoCarveout` selects this function (`sm90_gmma_builder.inl`, lines 68 to 85, abridged):

```c++
constexpr auto mainloop_pipeline_bytes = sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage);
constexpr int stage_bytes_ =
  cutlass::bits_to_bytes(a_bits * size<0>(TileShapeMNK{}) * size<2>(TileShapeMNK{})) +
  cutlass::bits_to_bytes(b_bits * size<1>(TileShapeMNK{}) * size<2>(TileShapeMNK{}));
constexpr int stage_bytes = cutlass::round_up(stage_bytes_, alignment) +
  static_cast<int>(mainloop_pipeline_bytes);
constexpr int carveout_bytes = cutlass::round_up(carveout_bytes_, alignment);
constexpr int capacity_bytes = capacity_bytes_ / alignment * alignment;
return (capacity_bytes - carveout_bytes) / stage_bytes;
```

Here are the numbers for example 49:

| Term | Value | Source |
|---|---|---|
| A tile, 128 · 64 FP16 | 16,384 bytes | |
| B tile, 128 · 64 FP16 | 16,384 bytes | |
| Pipeline barriers for one stage: one full barrier and one empty barrier, 8 bytes each | 16 bytes | `pipeline/sm90_pipeline.hpp`, `PipelineTmaAsync::SharedStorage` |
| `stage_bytes` | 32,784 bytes | |
| `capacity_bytes`, `sm90_smem_capacity_bytes` | 232,448 bytes (227 KB) | `sm90_common.inl`, line 57 |
| `carveout_bytes`, the epilogue's shared storage | 33,792 bytes (see below) | |
| Stages | floor((232,448 − 33,792) / 32,784) = 6 | |

The carveout is the size of the epilogue's `SharedStorage`. I calculated it from the epilogue source and did not compile it, so this value is unverified. The cooperative epilogue tile for a 128 by 128 tile is 128 by 32, which gives 4 epilogue subtiles. C and D are both FP16, so the epilogue reuses the C buffers for D, with 4 stages: 4 · 128 · 32 · 2 bytes = 32,768 bytes. The C load pipeline adds 4 · 16 = 64 bytes of barriers. The visitor nodes of example 49's custom tree have empty shared storage. The C buffer is aligned to 1,024 bytes for its 128-byte swizzle, so the struct size rounds up to 33,792 bytes. The result is robust: after the rounding to 128 bytes, any carveout from 3,072 to 35,712 bytes gives 6 stages.

##### **In the CuTe DSL: the same six choices, written by hand**

`dense_gemm_persistent.py` makes each choice in `__init__` or `_setup_attributes`. The results are the same as the builder's results for most choices, but not for all of them.

| # | Choice | CuTe DSL code | Same result as C++? |
|---|---|---|---|
| 1 | Warpgroups per tile | `atom_layout_mnk = (2, 1, 1) if tile M > 64 and tile N > 128 else (1, 1, 1)` (lines 283 to 287) | **No.** For a 128 by 128 tile, the DSL uses one MMA warpgroup, and that warpgroup computes the atom 2 times along M. The C++ cooperative builder always uses 2 warpgroups. |
| 2 | The tiled MMA | `sm90_utils.make_trivial_tiled_mma(a_dtype, b_dtype, a_major, b_major, acc_dtype, atom_layout_mnk, tiler_mn=(64, tile N))` (line 348). Its `a_source` argument defaults to `OperandSource.SMEM`. | Yes |
| 3 | The TMA copies | `CopyBulkTensorTileG2SOp()` if the multicast count is 1, else `CopyBulkTensorTileG2SMulticastOp()` (lines 1244 to 1248). The multicast count of A is the cluster's N extent, and of B the cluster's M extent (lines 366 and 367). | Yes |
| 4 | The shared memory layout atoms | `sm90_utils.get_smem_layout_atom(...)` picks the 128-byte, 64-byte or 32-byte swizzle if the major mode's size in bits is a multiple of 1024, 512 or 256, else the interleaved atom (`hopper_helpers.py`, lines 194 to 213). `cute.tile_to_shape` then tiles the atom to the staged shape. | Yes |
| 5 | The number of stages | `_compute_stages` (section 2) | Yes for this tile: 6 stages. See below. |
| 6 | The dispatch policy | None. The class is the implementation. | Not applicable |

For choice 5, the DSL code gets the same 6 stages, but its terms are different. Its capacity is `(228 - 1) * 1024` = 232,448 bytes (`cutlass_dsl/cutlass.py`, line 178), the same as C++. With one MMA warpgroup, the DSL uses the non-cooperative epilogue tile, 64 by 32. So it reserves 4 · 64 · 32 · 2 = 16,384 bytes for the epilogue, plus 1,024 bytes for barriers. Then (232,448 − 17,408) // 32,768 = 6.

##### **What changes in the W4A8 builder**

W4A8 selects the `GMMA_TMA_WS_RS` builder (`sm90_gmma_builder.inl`, lines 313 to 505). It makes the same six choices, with these differences:

1. **The tiled MMA** uses `GMMA::rs_op_selector` with both operands K-major. The atom reads A from registers. For FP8, the atom's K is 32.
2. **The smem-to-register copy.** `SmemCopyAtomA` is no longer `void`. It is `Copy_Atom<cute::AutoVectorizingCopy, ElementA>`, because the MMA warpgroups must copy the INT4 tile from shared memory into registers before they convert it. The builder sets it on the transformed operand's side, which is A for W4A8 (lines 479 and 480).
3. **The shared memory layout atoms** come from `rs_smem_selector`, not `ss_smem_selector`.
4. **The stage count** uses `compute_stage_count_or_override_single_affine_transformed_input` (lines 133 to 162). This version adds the scale bytes of one K tile to each stage: the scale width in bits, times the tile's M extent, times 1. The W4A8 scale type is `Array<float_e4m3_t, 8>`, which is 64 bits.
5. **The dispatch policy** is `MainloopSm90TmaGmmaRmemAWarpSpecializedMixedInput`, as section 1 showed.

After the swap, the tile shape means something new. The tile's M extent covers rows of the weight, which are output channels. The tile's N extent covers tokens. The W4A8 configurations have a tile M of 128 or 256 and a tile N from 16 to 256 (`w4a8_mm_entry.cu`, lines 255 to 265). So for a decode step with only a few tokens, the kernel can use a tile N of 16. A WGMMA with N = 16 is a legal instruction, because the PTX ISA allows N from 8 to 256 in steps of 8. Without the swap, the few tokens would be the M extent, and the WGMMA's M is fixed at 64.

Here is one stage of the `128x16` configuration, before the rounding and the barriers are added:

| Term | Value |
|---|---|
| INT4 weight tile, 128 · 128 · 4 bits | 8,192 bytes |
| FP8 activation tile, 16 · 128 · 8 bits | 2,048 bytes |
| Scales, 64 bits · 128 · 1 | 1,024 bytes |
| Sum | 11,264 bytes |

The full stage count needs the alignment of the RS smem atoms and the size of the W4A8 epilogue's storage. Part 5 computes both.

##### **What the port must choose by hand**

The port has no builder, so it must make all six choices in its own `_setup_attributes`:

| # | Choice | What the port must do |
|---|---|---|
| 1 | Warpgroups per tile | Always use `(2, 1, 1)`, as the cooperative C++ kernel does. The DSL example's rule would give one warpgroup for most W4A8 tiles. |
| 2 | The tiled MMA | Call `make_trivial_tiled_mma` with `a_source=OperandSource.RMEM`, the FP8 type for both operands, and both operands K-major. |
| 3 | The TMA copies | Use the same multicast rule. Only the `128x256_2x1x1` configuration has a cluster larger than 1. |
| 4 | The shared memory layout atoms | Choose an atom for the 4-bit weight tile and a layout for the scales. The DSL example has no 4-bit case. |
| 5 | The number of stages | Add the scale bytes to the stage formula. |
| 6 | The dispatch policy | Nothing. The port's class is the implementation. |

**References**
- CUTLASS v4.7.1, [`include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl) (the SS and RS builders and the stage count functions) and [`include/cutlass/gemm/collective/builders/sm90_common.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/collective/builders/sm90_common.inl) (`sm90_smem_capacity_bytes`, `sm90_cluster_shape_to_tma_atom`, `ss_smem_selector`)
- CUTLASS v4.7.1, [`include/cutlass/epilogue/collective/builders/sm90_builder.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/epilogue/collective/builders/sm90_builder.inl) and [`include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp) (the epilogue tile, its stages and its shared storage)
- CUTLASS v4.7.1, [`include/cutlass/pipeline/sm90_pipeline.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/pipeline/sm90_pipeline.hpp) (`PipelineTmaAsync::SharedStorage`)
- CUTLASS v4.8.0, [`examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py), [`python/CuTeDSL/cutlass/utils/hopper_helpers.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/python/CuTeDSL/cutlass/utils/hopper_helpers.py) and [`python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py) (the DSL choices and the capacity map)
- vLLM, `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu` (the ten tile configurations)
- PTX ISA, [`wgmma.mma_async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma) (the legal N values)

---

#### **Wrapping up**

CUTLASS 3.x gives each independent choice in a GEMM its own level: the instruction, the tiling over threads, the order of work over time, the order of tiles over the grid, and the epilogue. A dispatch policy connects the levels, so W4A8 can change only its mainloop and its epilogue fusion and keep the same kernel, scheduler and host adapter. The `CollectiveBuilder` makes six choices for the mainloop: the warpgroups per tile, the tiled MMA, the TMA copies, the shared memory atoms, the stage count and the policy. The CuTe DSL has no levels above CuTe and no builder. So the port writes the three highest levels as one Python class, and it makes the six choices by hand in its setup code.

Part 2 opens the kernel level: the cooperative warp-specialized kernel, its warp roles and register counts, the persistent tile scheduler, and clusters with TMA multicast, each next to the same code in `dense_gemm_persistent.py`.

If you find a mistake anywhere in here, please let me know and I'll fix it.
