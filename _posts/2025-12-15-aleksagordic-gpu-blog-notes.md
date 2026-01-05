---
layout: post
title: Reading Notes from Aleksa Gordic's GPU BlogPost
date: 2025-12-15 09:28:00-0400
featured: False
description: Reading notes for my reference from Aleksa Gordic's GPU BlogPost
tags: GPU
categories: 
giscus_comments: false
related_posts: false

---

[https://www.aleksagordic.com/blog/matmul](https://www.aleksagordic.com/blog/matmul)

**Fundamentals of NVIDIA GPU architecture(on H100):**

- Tensor cores: wgmma instrutions needed to fully exploit it.
- CUDA cores: arithmetic instructions usually have a latency of ~4-15 cycles.
- (Tensor cores, CUDA cores, warp scheduler, LD/ST and register file(16k 32 bit)) x4 of these per SM, aka quadrants.
- TMA(Tensor memory accelerator): For small requests, TMA loads have higher latency than regular async copies (due to address generation overhead). It handles load/store transfers between GMEM and SMEM (with swizzling).
- 1KiB of SMEM(Shared memory) goes for system use per block, so effectively we have: 228 - numblocks * 1kB kBs.
- Shared mem is faster than L1, as there is no need to tag stage comparisons for hit/miss.
- L1 cache line = 128B (=threads in warp fetching 4B floats)
- L1 *can* be used for register spill-over when register pressure is high
- Distributed shared memory(DSMEM): pooled shared memories (SMEM) of a physically close group of SMs (a GPC). Worse bandwidth/latency compared to shared mem, but better than L2. 
- No. of SM's: N=144(on the die), N=132(SXM5) and N=114(PCIe)
- L2: 
  - We can set the granularity of the data fetch size to 32, 64 or 128B using cudaDeviceSetLimit.
  - It is physically partitioned into two parts; each SM connects directly to only one partition and indirectly to the other through the crossbar.
  - Residency control: we can set a part of L2 cache for persistent data accesses and map it to a chunk of GMEM
  - It's possible to redirect power from L2 to SMs (demonstrated in MLPerf 2024)
  - L2 cache line = 128B, 4 sectors (sector==32B), same as L1
  - Contains data compression circuitry and does global atomics
  - 60 MiB on the die (50MiB for SXM/PCIe)
  - real read BW = 12-14 TB/s (near), far is significantly slower. Latency ~200 cycles.
- GPC: Graphics processing clusters. Each GPC contains 18 SMs, so there are 8 GPCs on the GPU. Four GPCs connect directly to one L2 partition and the other four to the second partition.
- VRAM/device memory (80 GB, common form factor):
  - GMEM - 32, 64, 128B mem transactions granularity
  - constant memory - very small ~64KiB
  - local memory - 512 KiB/thread (register spill space)
- To connect to other GPUs - nvlink v4. Bi-BW = 900 GB/s, Uni-BW = 450 GB/s (18 links with 25GB/s)
- To connect to x86 CPU, DPUs etc - PCIe Gen 5. Bi-BW = 128 GB/s and Uni-BW = 64 GB/s
- Note: There are a few other smaller caches for instructions.

- The memory system in a GPU is highly hierarchical, much like in CPU architectures.
- This hierarchy is dictated by physics and circuit design: SRAM cells are faster but larger (the control circuitry that enables their speed also increases their area), while DRAM cells are smaller/denser but slower. The result is that faster memory is lower capacity and expensive, while slower memory can be provided in much larger quantities.
- This trade-off between capacity and latency is exactly why cache hierarchies exist.
- Moving from device memory down to registers (levels 1-5), you see a clear trend: bandwidth increases by orders of magnitude, while both latency and capacity decrease by similar orders of magnitude.
- A few immediate implications follow:
  - Keep the most frequently accessed data as close as possible to the compute units.
  - Minimize accesses to the lower levels of the hierarchy, especially device memory (GMEM).
- One additional component worth noting is the Tensor Memory Accelerator (TMA), introduced with Hopper. TMA enables asynchronous data transfers between global memory and shared memory, as well as across shared memories within a cluster. It also supports swizzling to reduce bank conflicts.

Compute:
- The fundamental unit is the streaming multiprocessor (SM). Hopper H100 (SXM5) integrates 132 SMs in total.
- SMs are grouped into graphics processing clusters (GPCs): each GPC contains 18 SMs, and there are 8 GPCs on the GPU. Four GPCs connect directly to one L2 partition, and the other four to the second partition.
- Tensor Cores: Specialized units that execute matrix multiplications on small tiles (e.g., 64x16 @ 16x256) at high throughput. Large matrix multiplications are decomposed into many such tile operations, so leveraging them effectively is critical for reaching peak performance.
- CUDA cores and SFUs: The so-called "CUDA cores" (marketing speech) execute standard floating-point operations such as FMA (fused multiply-add: c = a * b + c). Special Function Units (SFUs) handle transcendental functions such as sin, cos, exp, log, but also algebraic functions such as sqrt, rsqrt, etc.
- Load/Store (LD/ST) units: Circuits that service load and store instructions, complementary to the TMA engine.
- Warp schedulers: Each SM contains schedulers that issue instructions for groups of 32 threads (called warps in CUDA). A warp scheduler can issue one warp instruction per cycle.
- Each SM is physically divided into four quadrants, each housing a subset of the compute units described above.
- Parallelism vs Concurrency (Imp.)
  - An SM can issue instructions from at most four warps simultaneously (i.e., 128 threads in true parallel execution at a given cycle).
  - However, an SM can host up to 2048 concurrent threads (64 warps). These warps are resident and scheduled in and out over time, allowing the hardware to hide memory/pipeline latency.
  - In other words, instruction parallelism (how many threads start executing an instruction on a given cycle) is limited to 128 threads per SM at once (4 32-wide warp instructions), while concurrency (how many threads are tracked in the scheduler and eligible to run) extends to 2048 threads.
- What is the ceiling—the maximum compute throughput of a GPU? This is often referred to as the "speed of light" (SoL) performance: the upper bound dictated by the physical characteristics of the chip.
- There are multiple ceilings depending on the data type. In LLM training workloads, bfloat16 (bf16) has been the dominant format in recent years, though fp8 and 4-bit formats are becoming increasingly important (for inference fp8 is fairly standard).
- The peak throughput is calculated as: `perf = freq_clk_max * num_tc * flop_per_tc_per_clk` or in words: maximum clock frequency × number of tensor cores × FLOPs per tensor core per cycle.
- The "speed of light" is not actually constant.
- In practice, the peak throughput depends on the actual clock frequency, which can vary under power or thermal throttling. If the GPU clock drops, so does the effective speed of light.
- Normally on H100 SXM the max clock freq is 1830 MHz => clock cycle takes ~0.55 ns
- But GPU might experience power throttling causing it to automatically drop the clock freq in order to reduce the transistor switching power. 


Doubts:
- In cache what is k-way set associative cache?
- What is transistor switching power and how its related to clock freq and power throttling?
  

Further reading: Horace He went into this phenomenon in more depth in [his blog post (3)](https://www.thonking.ai/p/strangely-matrix-multiplications).
  

**CUDA programming model**

- Thread has private registers synchronization: SMEM


GPU assembly languages: PTX and SASS

- (D) In fig 18, is the blockIdx.x and blockIdx.y mentioned pictorially right? In fig 19, its mentioned as "warp 1 from block (blockIdx.x, blockIdx.y) = (1,0)", with that as the case, then x goes from top to bottom right, but in the fig 18, x row of block dims span accross left to right.
- PTX code is generated for a thread block right? and will the PTX code shown executed per thread in the thread block in parallel?
- What is exposing ILP (in PTX code explanation)? - Instruction-Level Parallelism
- How loop unrolling exposes Instruction-Level Parallelism? So, does these instructions, run in parallel, as the warp takes and executes them? 

- SASS was a bit harder to understand, may be it could be due to the fact that its not explained in detail. For now, just got the gist and proceeding.

Designing near-SOTA synchronous matmul kernel

- (D) Loading A → As. This step is trickier because As is transposed. The reason for the transpose is that it enables vectorized loads (LDS.128) later during the compute phase.
- (D) The trade-off is that the stores cannot be vectorized: the 4 floats fetched from a row of A must now be scattered into a column of As, which maps into the same memory bank. That's acceptable because we prioritize fast loads — each element of As will be accessed multiple times during computation, while the stores happen only once.