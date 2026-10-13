---
layout: post
title: CUTLASS C++ Hopper GEMM, part 1, the layers and the builder [W4A8]
date: 2026-10-13 00:00:00-0400
featured: false
description: Part 1 of a series that reads the CUTLASS 3.x C++ Hopper GEMM stack from the source, as the base for porting vLLM's W4A8 kernel to the CuTe DSL. This part covers the five levels of CUTLASS 3.x, how a kernel is assembled from them, and what the CollectiveBuilder chooses
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This is part 1 of a series on the CUTLASS 3.x C++ GEMM stack for Hopper. The series is the C++ companion to the [CuTe DSL fundamentals post](/blog/2026/fa3-cute/). That post explains the Hopper primitives through the CuTe Python DSL: layouts, TMA, mbarriers, pipelines, WGMMA and warp specialization. This series explains how CUTLASS 3.x puts the same primitives together in C++ to make a complete GEMM kernel.

The reason for the series is a port. vLLM has a W4A8 GEMM for Hopper, with FP8 activations and INT4 weights, in `csrc/libtorch_stable/quantization/cutlass_w4a8/`. It is written with CUTLASS 3.x in C++. I want to write it again in the CuTe DSL. To do that, I must first know what each layer of the C++ kernel does, because the CuTe DSL has no `CollectiveBuilder` that makes these choices for me.

The rule for the series is the same as for the DSL post. Every statement about the hardware or the library comes from a primary source: the CUTLASS source, the CUTLASS documentation, the CUDA PTX ISA, or NVIDIA's Hopper material. Each section ends with its references. If a statement has no source, the post marks it as unverified. I have no local GPU, so nothing in the series was compiled or run. Every code excerpt is read from the source, not from a run.

A note on versions. vLLM builds its C++ kernels against CUTLASS `v4.7.1`. Its `CMakeLists.txt` sets `CUTLASS_REVISION` to that tag. The CuTe DSL that vLLM pins, `nvidia-cutlass-dsl`, is version `4.8.0`. So the series quotes the C++ source from the `v4.7.1` tag, because that is the code that vLLM compiles. I compared the files that the series uses against the `v4.8.0` tag with `diff`. The SM90 kernel, the SM90 mainloop collectives, the SM90 epilogue collective, the SM90 builder and the SM90 tile scheduler are identical in both tags. Seven shared headers are different, for example `gemm/dispatch_policy.hpp` and `gemm/device/gemm_universal_adapter.h`. In six of them, `v4.8.0` only adds lines and removes none. Only `pipeline/sm90_pipeline.hpp` changes existing code: `PipelineState::advance` has a new branch for a single stage. Where a difference matters, the section says so.

The running example is CUTLASS's example 49, `examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu`. It builds Hopper GEMMs with the `CollectiveBuilder` and runs several kernel schedules. Two of its variants use the kernel schedule `KernelTmaWarpSpecializedCooperative` and the epilogue schedule `TmaWarpSpecializedCooperative`. One adds the stream-K tile scheduler. The other uses the persistent tile scheduler and a custom epilogue visitor tree (lines 647 to 652). The series follows the second one, because W4A8 also uses the persistent scheduler and an epilogue visitor tree. The W4A8 kernel uses these two schedules too (`w4a8_mm_entry.cu`, lines 102 to 107). The large difference is the mainloop collective. The dense mainloop reads both operands of the tensor core instruction from shared memory. The W4A8 mainloop converts the INT4 weights in registers and gives them to the instruction from registers. The series explains the dense stack first, and then the W4A8 mainloop.

The series:

- **Part 1 (this post): the layers and the builder.** The five levels of CUTLASS 3.x (device, kernel, collective, tiled MMA and copy, atom), how example 49 and the W4A8 kernel assemble a kernel from them, and what the `CollectiveBuilder` chooses.
- **Part 2: the kernel.** The warp-specialized cooperative kernel, the tile scheduler and the persistent loop, and clusters with TMA multicast.
- **Part 3: the mainloop.** The `load()` side, the C++ pipeline classes, and the `mma()` side.
- **Part 4: the epilogue and the host.** The TMA store, the epilogue visitor tree, and the host side: arguments, parameters, TMA descriptors and the launch.
- **Part 5: W4A8.** The mixed-input mainloop that reads one operand from registers, and a table that maps each C++ piece to the CuTe DSL.

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

The two lowest levels are CuTe, and the CuTe DSL has the same two levels. A `TiledMma` built with `make_trivial_tiled_mma` in the DSL post is the Python version of a `cute::TiledMMA` in C++. The three highest levels are CUTLASS, and the CuTe DSL does not have them. In the DSL, one class such as `dense_gemm.py`'s kernel class does the work of the kernel level and the collective level together. Its host function does the work of the device level. So for the port, I must write the code of the three highest levels by hand.

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

The `k_tile` loop belongs to the collective. The document says that this loop is not unrolled. In the document's model, the three innermost loops are inside one call to `cute::gemm`. The document says that these loops are generally fully static and are fully unrolled. The Hopper mainloop moves one of them out. It writes the `k_block` loop around `cute::gemm` by hand, so that it can set the accumulate flag for each K block (`sm90_mma_tma_gmma_ss_warpspecialized.hpp`, lines 492 to 497). The M and N loops stay inside the call. The DSL post does the same with its `range_constexpr` loop over K blocks.

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

**References**
- CUTLASS v4.7.1, [`media/docs/cpp/gemm_api_3x.md`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/media/docs/cpp/gemm_api_3x.md) (the five levels, the loop-nest pseudocode, the definitions of collective, kernel and atom, and the launch note) and [`media/docs/cpp/cutlass_3x_design.md`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/media/docs/cpp/cutlass_3x_design.md) (tag-dispatch policies)
- [`include/cutlass/gemm/dispatch_policy.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/dispatch_policy.hpp) (the kernel schedule tags and the mainloop policies)
- [`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) (the kernel specialization)
- [`include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/collective/builders/sm90_gmma_builder.inl) (the mixed-input policy choice)
- vLLM, `csrc/libtorch_stable/quantization/cutlass_w4a8/w4a8_mm_entry.cu` (the W4A8 schedules)

---
