---
layout: post
title: Hopper GEMM in CUTLASS C++ and the CuTe DSL, part 2, the kernel [W4A8]
date: 2026-10-20 00:00:00-0400
featured: false
description: Part 2 of a series that reads the CUTLASS 3.x C++ Hopper GEMM stack from the source and puts each piece next to its CuTe DSL equivalent, as the base for porting vLLM's W4A8 kernel to the CuTe DSL. This part covers the warp-specialized cooperative kernel, the persistent tile scheduler, and clusters with TMA multicast
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This is part 2 of the series on the Hopper GEMM in CUTLASS C++ and the CuTe DSL. [Part 1](/blog/2026/cutlass-hopper-gemm-1/) showed the five levels of CUTLASS 3.x, how a kernel is assembled from them, and what the `CollectiveBuilder` chooses. This part opens the kernel level. The kernel level decides which warps do which work, how many registers each warp gets, and which output tile each CTA computes next.

The rules and the versions are the same as in part 1. The C++ code is quoted from CUTLASS `v4.7.1`, which vLLM compiles. The CuTe DSL code is quoted from the `v4.8.0` tag, which vLLM pins. Nothing in this part was compiled or run. Every excerpt is read from the source.

The series:

- **[Part 1](/blog/2026/cutlass-hopper-gemm-1/): the layers and the builder.**
- **Part 2 (this post): the kernel.** The warp-specialized cooperative kernel, the tile scheduler and the persistent loop, and clusters with TMA multicast.
- **Part 3: the mainloop.** The producer side with TMA loads, the pipeline classes, and the consumer side with WGMMA.
- **Part 4: the epilogue and the host.** The TMA store, the epilogue visitor tree, and the host side: arguments, TMA descriptors and the launch.
- **Part 5: W4A8.** The mixed-input mainloop that reads one operand from registers, and how to write it in the DSL.

The plan for this part:

- The cooperative kernel: warpgroups, warp roles and register counts
- The tile scheduler and the persistent loop
- Clusters and TMA multicast

Let's get started.

---

#### **The cooperative kernel** {#cooperative}

The cooperative kernel is the `GemmUniversal` specialization in `gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`. Part 1 showed how the mainloop's dispatch policy selects it. This section reads what one CTA of this kernel does.

##### **Three warpgroups, three roles**

A CTA of the cooperative kernel has 384 threads, which are three warpgroups of 128 threads. The kernel computes the count from the tiled MMA (lines 116 to 125):

```c++
static constexpr uint32_t NumMMAThreads      = size(TiledMma{});      // 8 warps
static constexpr uint32_t NumLoadWarpGroups  = 1;
static constexpr uint32_t NumMmaWarpGroups   = NumMMAThreads / NumThreadsPerWarpGroup;
static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
```

The builder makes a tiled MMA of two warpgroups for the cooperative schedule (part 1, choice 1), so `NumMMAThreads` is 256. The kernel checks this with `static_assert(NumMMAThreads == 256, ...)` (line 373). It also requires a tile M extent of 128 or more (lines 374 and 375), because each of the two MMA warpgroups needs at least one 64-row WGMMA atom.

Each warpgroup gets a role from its index (lines 383 to 387 and 406):

```c++
enum class WarpGroupRole {
  Producer = 0,
  Consumer0 = 1,
  Consumer1 = 2
};
...
auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
```

`canonical_warp_group_idx()` is `threadIdx.x / 128`, broadcast from lane 0 with `__shfl_sync` so that the compiler knows it is the same in every lane of the warp (`cutlass/cutlass.h`, line 151). Threads 0 to 127 are the producer warpgroup. Threads 128 to 383 are the two consumer warpgroups.

"Cooperative" means that the two consumer warpgroups work on the same output tile. The tiled MMA splits the tile along M between them, so each one computes half of the rows of every tile. The tiled MMA places two 64-row WGMMA atoms along M, one for each warpgroup, so it covers 128 rows. If the tile M extent is 128, each warpgroup computes one block of 64 rows. If it is 256, the tiled MMA repeats two times. Then warpgroup 0 computes rows 0 to 63 and 128 to 191, and warpgroup 1 computes rows 64 to 127 and 192 to 255. The other Hopper schedule, ping-pong, gives each consumer warpgroup its own tile. The comment at line 382 states the cooperative rule: "Consumer0 and Consumer1 collaborate on the same tile".

<div class="row justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/cutlass-hopper-gemm/cooperative-cta.svg" title="One CTA of the cooperative kernel" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    One CTA of the cooperative kernel. The producer warpgroup keeps 40 registers per thread, and its warp 0 fills the stages of the mainloop pipeline with TMA loads. The two consumer warpgroups take 232 registers per thread, read the full stages, and compute the two halves of the rows of the same output tile. The bar at the bottom shows the register file of one SM. Editable source: <a href="/assets/img/cutlass-hopper-gemm/cooperative-cta.excalidraw">cooperative-cta.excalidraw</a>.
</div>

##### **The four warps of the producer warpgroup**

The producer warpgroup has four warps, and each warp has its own role (lines 388 to 393 and 407):

| Warp | Role | Work | Example 49 | W4A8 |
|---|---|---|---|---|
| 0 | `Mainloop` | Issues the TMA loads of the A and B tiles into the mainloop pipeline | Used | Used |
| 1 | `Warp1` | Asks a dynamic tile scheduler for the next tile | Not used: the SM90 persistent scheduler is static (`IsDynamicPersistent = false`, `sm90_tile_scheduler.hpp`, line 51) | Not used |
| 2 | `Epilogue` | Issues the TMA loads of the C tiles for the epilogue | Used, because example 49's custom tree reads C. This is a compile-time decision: `Sm90SrcFetch` needs the C load if the C element type is not `void`, whatever the value of beta. | Not used, if its epilogue does not read C. Part 4 checks this. |
| 3 | `MainloopAux` | Issues extra loads that some mainloop policies need | Not used | Not used |

So for a dense GEMM, one warp of the producer warpgroup issues all mainloop loads. The other three warps have little or no work. They exist because register reallocation works on a whole warpgroup, as the next subsection shows.

##### **Register reallocation: 40 and 232**

Each SM has 65,536 32-bit registers. The producer warpgroup needs few registers, because a TMA load needs only addresses and loop state, and one lane issues it. The consumer warpgroups need many registers, because they hold the accumulators. So the kernel moves registers from the producer to the consumers with the PTX instruction `setmaxnreg` (lines 132 to 151):

```c++
static constexpr int RegsPerThread =
  size<0>(TileShape{}) * size<1>(TileShape{}) / NumMMAThreads *
  sizeof(ElementAccumulator) / sizeof(uint32_t);
...
static constexpr bool HeavyRegisterPressure = ... (RegsPerThread >= 208);
static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
```

`RegsPerThread` is the number of accumulator registers that each consumer thread holds. The producer branch calls `cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>()` (line 562), and the consumer branch calls `cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>()` (line 772). These functions emit `setmaxnreg.dec.sync.aligned.u32` and `setmaxnreg.inc.sync.aligned.u32` (`arch/reg_reconfig.h`, lines 74 to 86). The [DSL post](/blog/2026/fa3-cute/#warp-specialization) gives the rules of this instruction.

The numbers fit the register file exactly:

| Warpgroup | Threads | Registers per thread | Registers |
|---|---|---|---|
| Producer | 128 | 40 | 5,120 |
| Consumer 0 and consumer 1 | 256 | 232 | 59,392 |
| Total | 384 | | 64,512 of 65,536 |

At launch, each thread gets the same register count. The kernel is compiled with `__launch_bounds__(384, 1)` (`device_kernel.h`, line 116), so the compiler limits each thread to 65,536 / 384 = 170 registers. I assume that this rounds down to 168, because registers are allocated in units of 8 per thread. 384 · 168 = 64,512, the same total as the table. Then the producer gives up 128 · (168 − 40) = 16,384 registers, and the consumers take 256 · (232 − 168) = 16,384 registers. The 168 is my inference from the allocation unit, not from a CUTLASS source line.

The accumulator registers decide which pair the kernel uses. For W4A8, the largest tiles are 256 by 128 and 128 by 256, so `RegsPerThread` = 256 · 128 / 256 · 4 / 4 = 128. That is below 208, so every W4A8 configuration uses 40 and 232.

##### **The life of one CTA**

The kernel's `operator()` has three parts: setup, the producer branch and the consumer branch.

1. **Setup (all 384 threads).** One thread prefetches the TMA descriptors (lines 411 to 415). The threads construct the mainloop pipeline, the epilogue load pipeline, the epilogue store pipeline and a small ordering barrier (lines 461 to 503). Then all CTAs of the cluster synchronize, so that every barrier is initialized before any CTA uses it (lines 517 to 528 and 558).
2. **The producer branch.** The producer warpgroup gives up registers. Warp 0 runs a loop over its output tiles. For each tile, it calls `collective_mainloop.load(...)`, which issues the TMA loads for all K tiles of that tile into the mainloop pipeline (lines 602 to 667). After its first tile, warp 0 signals the ordering barrier, and only then does warp 2 start to load C (lines 646 to 649 and 725).
3. **The consumer branch.** The two consumer warpgroups take registers. They run the same loop over output tiles. For each tile, they allocate the accumulators, call `collective_mainloop.mma(...)` over all K tiles, call `mma_tail`, and then call `collective_epilogue.store(...)` to write the tile (lines 770 to 885).

The producer and the consumers do not exchange tile coordinates. Each side asks its own copy of the tile scheduler for the next tile, and the static scheduler gives both sides the same sequence. The mainloop pipeline is their only connection: the producer fills shared memory stages, and the consumers empty them. Part 3 explains this pipeline.

The kernel also contains two PDL calls. PDL (programmatic dependent launch) lets the next kernel in the stream start before this kernel ends. The producer warps call `wait_on_dependent_grids()` before their first global memory read (lines 605 and 722). The consumers call `launch_dependent_grids()` when they start the epilogue of their last tile (lines 826 to 831). These functions emit `griddepcontrol.wait` and `griddepcontrol.launch_dependents`, but only if the build defines `CUTLASS_ENABLE_GDC_FOR_SM90` and compiles for `sm_90a` with CUDA 12 or newer (`arch/grid_dependency_control.h`, lines 43 to 48 and 85 to 99). vLLM's `CMakeLists.txt` does not define it. I checked upstream main at commit `b439421f11` (2026-10-19). So in vLLM's W4A8 kernel today, both calls compile to nothing.

##### **In the CuTe DSL: the same roles, fewer warps used**

`dense_gemm_persistent.py` has the same structure. The warpgroup roles are Python conditions in `kernel`:

```python
warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
...
is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
if is_dma_warp_group:
    cute.arch.setmaxregister_decrease(self.load_register_requirement)    # 40

if warp_idx == self.load_warp_id:                                         # warp 0
    ...  # producer loop: TMA loads of A and B

if not is_dma_warp_group:
    cute.arch.setmaxregister_increase(self.mma_register_requirement)     # 232
    ...  # consumer loop: WGMMA, then the epilogue
```

These lines are at 696 to 698 and 721 to 783. The table compares the two kernels.

| Item | C++ cooperative kernel | `dense_gemm_persistent.py` |
|---|---|---|
| Threads per CTA | 384 | (1 + MMA warpgroups) · 128 (lines 299 to 301): 384 with two MMA warpgroups, 256 with one |
| MMA warpgroups | Always 2 | 2 only if tile M > 64 and tile N > 128 (part 1, choice 1) |
| Register counts | 40 and 232, or 24 and 240 for large accumulators | Always 40 and 232 (lines 306 and 307) |
| Mainloop load warp | Warp 0 | Warp 0, `load_warp_id` (line 302) |
| Epilogue load warp | Warp 2, if the epilogue reads C | None: the example has no C input |
| TMA store | Issued inside the epilogue collective | Warp 4, `epi_store_warp_id`, the first warp of the first MMA warpgroup (lines 303 to 305 and 960) |
| Cluster barrier after setup | `cluster_arrive_relaxed` and `cluster_wait`, or `__syncthreads` for one CTA | `pipeline_init_arrive` and `pipeline_init_wait` (lines 639 and 719) |
| PDL | Calls present, compiled out in vLLM | No calls. The DSL has `cute.arch.griddepcontrol_wait` and `cute.arch.griddepcontrol_launch_dependents` (`cute/arch/nvvm_wrappers.py`, lines 3399 and 3421). |

For the port, three rows matter:

- The port must use two MMA warpgroups for every W4A8 tile, as part 1 said. So it always has 384 threads.
- The port does not need an epilogue load warp, if the W4A8 epilogue does not read C. Part 4 confirms this.
- The port can add PDL. Reviewers of the closed CuTe DSL port [vllm-project/vllm#48619](https://github.com/vllm-project/vllm/pull/48619) asked for PDL. The C++ kernel shows where the two calls go: the wait before the first global read, and the launch at the start of the last tile's epilogue. vLLM's own CuTe DSL kernels already have a `use_pdl()` switch (`vllm/model_executor/kernels/linear/cute_dsl/_ll_router_common.py`, line 53).

**References**
- CUTLASS v4.7.1, [`include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp) (the roles, the register counts, the producer and consumer branches)
- CUTLASS v4.7.1, [`include/cutlass/arch/reg_reconfig.h`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/arch/reg_reconfig.h) (`setmaxnreg`), [`include/cutlass/arch/grid_dependency_control.h`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/arch/grid_dependency_control.h) (`griddepcontrol`), [`include/cutlass/cutlass.h`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/cutlass.h) (`canonical_warp_group_idx`), [`include/cutlass/device_kernel.h`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/device_kernel.h) (`__launch_bounds__`), [`include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.7.1/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp) (`IsDynamicPersistent`)
- PTX ISA, [`setmaxnreg`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg) and [`griddepcontrol`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)
- CUTLASS v4.8.0, [`examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm_persistent.py) and [`python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py`](https://github.com/NVIDIA/cutlass/blob/v4.8.0/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py)
- vLLM, `CMakeLists.txt` (no `CUTLASS_ENABLE_GDC_FOR_SM90`, checked at upstream main `b439421f11`) and `vllm/model_executor/kernels/linear/cute_dsl/_ll_router_common.py` (`use_pdl`)

---
