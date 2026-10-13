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
- CUTLASS v4.7.1, [`media/docs/cpp/gemm_api_3x.md`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/media/docs/cpp/gemm_api_3x.md) (the five levels, the loop-nest pseudocode, the definitions of collective, kernel and atom, and the launch note) and [`media/docs/cpp/cutlass_3x_design.md`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/media/docs/cpp/cutlass_3x_design.md) (tag-dispatch policies)
- [`include/cutlass/gemm/dispatch_policy.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/dispatch_policy.hpp) (the kernel schedule tags and the mainloop policies)
- [`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) (the kernel specialization)
- [`include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl) (the mixed-input policy choice)
- vLLM, `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu` (the W4A8 schedules)
- CUTLASS v4.8.0, [`examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py) (the DSL kernel) and [`python/CuTeDSL/cutlass/utils/hopper_helpers.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/python/CuTeDSL/cutlass/utils/hopper_helpers.py) (`make_trivial_tiled_mma`)

---
