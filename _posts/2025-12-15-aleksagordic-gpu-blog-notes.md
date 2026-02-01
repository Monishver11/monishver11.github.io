---
layout: post
title: Reading Notes from Aleksa Gordic's GPU BlogPost
date: 2025-12-15 09:28:00-0400
featured: False
description: Reading notes for my reference from Aleksa Gordic's GPU BlogPost
tags: GPU
categories: GPU-NYU
giscus_comments: false
related_posts: false

---

Credits: [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)

**Fundamentals of NVIDIA GPU architecture(on H100):**

- Tensor cores: wgmma instrutions needed to fully exploit it.
- CUDA cores: arithmetic instructions usually have a latency of ~4-15 cycles.
- (Tensor cores, CUDA cores, warp scheduler, LD/ST and register file(16k 32 bit)) x4 of these per SM, aka quadrants.
- TMA(Tensor memory accelerator): A dedicated hardware unit introduced in NVIDIA's Hopper architecture, designed to accelerate asynchronous data transfers between global memory (GMEM) and shared memory (SMEM) in GPUs.For small requests, TMA loads have higher latency than regular async copies (due to address generation overhead). It handles load/store transfers between GMEM and SMEM (with swizzling).
- SW dial(Shared(SMEM) & L1): size = 256kB, BW = 128B/cycle, latency = ~20-30 cycles(~15 ns). SMEM can be upto 228 KiB.
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
- VRAM/device memory (80 GB, common form factor. Extensions can offer 141 GB(H200)):
  - GMEM - 32, 64, 128B mem transactions granularity
  - constant memory - very small ~64KiB
  - local memory - 512 KiB/thread (register spill space)
- To connect to other GPUs - nvlink v4. Bi-BW = 900 GB/s, Uni-BW = 450 GB/s (18 links with 25GB/s)
- To connect to x86 CPU, DPUs etc - PCIe Gen 5. Bi-BW = 128 GB/s and Uni-BW = 64 GB/s
- Note: There are a few other smaller caches for instructions.

**Memory**

- The memory system in a GPU is highly hierarchical, much like in CPU architectures.
- This hierarchy is dictated by physics and circuit design: SRAM cells are faster but larger (the control circuitry that enables their speed also increases their area), while DRAM cells are smaller/denser but slower. The result is that faster memory is lower capacity and expensive, while slower memory can be provided in much larger quantities.
- This trade-off between capacity and latency is exactly why cache hierarchies exist.
- Moving from device memory down to registers (levels 1-5), you see a clear trend: bandwidth increases by orders of magnitude, while both latency and capacity decrease by similar orders of magnitude.
- A few immediate implications follow:
  - Keep the most frequently accessed data as close as possible to the compute units.
  - Minimize accesses to the lower levels of the hierarchy, especially device memory (GMEM).
- One additional component worth noting is the Tensor Memory Accelerator (TMA), introduced with Hopper. TMA enables asynchronous data transfers between global memory and shared memory, as well as across shared memories within a cluster. It also supports swizzling to reduce bank conflicts.

**Compute**

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

**Speed of light and power throttling**

- What is the ceiling—the maximum compute throughput of a GPU? This is often referred to as the "speed of light" (SoL) performance: the upper bound dictated by the physical characteristics of the chip.
- There are multiple ceilings depending on the data type. In LLM training workloads, bfloat16 (bf16) has been the dominant format in recent years, though fp8 and 4-bit formats are becoming increasingly important (for inference fp8 is fairly standard).
- The peak throughput is calculated as: `perf = freq_clk_max * num_tc * flop_per_tc_per_clk` or in words: maximum clock frequency × number of tensor cores × FLOPs per tensor core per cycle.
- The "speed of light" is not actually constant.
- In practice, the peak throughput depends on the actual clock frequency, which can vary under power or thermal throttling. If the GPU clock drops, so does the effective speed of light.
- Normally on H100 SXM the max clock freq is 1830 MHz => clock cycle takes ~0.55 ns
- But GPU might experience power throttling causing it to automatically drop the clock freq in order to reduce the transistor switching power. 
- In short, the number of clock cycles an instruction takes to execute directly determines its latency—the time from when an instruction begins until it completes. 


Doubts:
- In cache what is k-way set associative cache?
- What is transistor switching power and how its related to clock freq and power throttling?
  
Further reading: Horace He went into this phenomenon in more depth in [his blog post (3)](https://www.thonking.ai/p/strangely-matrix-multiplications).
  

**CUDA Programming Model**

- The key abstractions: Thread -> Warp(32 threads) -> Thread Block -> Thread Block Cluster -> Grid(of thread blocks or clusters)
- Thread 
  - has private registers 
  - sync: SMEM
- Warp
  - doesn't exist in the programming model, but because we understand that the hardware has decicated warp schedulers, and that e.g., SMEM row is designed to be fully covered by a single warp. And, we know as programmers that warp is important for perf;
- Thread block
  - a group of up to 1024 threads
  - guaranteed to be concurrently scheduled on a single SM
  - multiple thread blocks are independently scheduled across SMs
  - sync: SMEM
- Thread block cluster
  - a group of up to 8(with opt-in upto 16) thread blocks
  - guaranteed to be concurrently scheduled on a GPC
  - sync: DSMEM
- Grid of thread blocks or clusters
  - group of clusters or thread blocks
  - sync: L2 or GMEM
- by Sync, we mean the highest level of memory heirarchy where X can synchronize. Synchronization is also possible at lower levels than the one listed here(e.g., in GMEM)

- Every thread is "aware" of its position in the CUDA hierarchy through variables such as `gridDim`, `blockIdx`, `blockDim`, and `threadIdx`. Internally, these are stored in special registers and initialized by the CUDA runtime when a kernel launches.
- This positional information makes it easy to divide work across the GPU. For example, suppose we want to process a 1024×1024 image. We could partition it into 32×32 thread blocks, with each block containing a 32×32 arrangement of threads.

- Each thread can then compute its global coordinates, e.g. below and use those to fetch its assigned pixel from global memory (image[x][y]), perform some pointwise operation, and store the result back.
  
```
const int x = blockIdx.x * blockDim.x + threadIdx.x
const int y = blockIdx.y * blockDim.y + threadIdx.y
```

- The model supports a third dimension for {block/thread} Idx -> z, but it's almost never used in deep learning kernels.
- Note on PTX terminalogy:
  - Thread block cluster -> cooperative grid array(CGA)
  - Thread block -> cooperative thread arrays(CTAs)
- E.g., if `threadIdx.x` runs from 0-1023 (a 1D block of 1024 threads) we can split it into `x = threadIdx.x % 32` and `y = threadIdx.x / 32`, effectively reshaping the block into a 32×32 logical 2D layout.

- Connecting the CUDA model back to the hardware, one fact should now be clear: a thread block should contain at least 4 warps (i.e., 128 threads). Why?
  - A thread block is resident on a single SM.
  - Each SM has 4 warp schedulers—so to fully utilize the hardware, you don't want them sitting idle.
- Few more reasons for 4 warps:
  - On Hopper the warp-group (4 warps) is the unit of execution for WGMMA (matmul) tensor core instructions.
  - Also, with persistent kernels(one that runs indefinitely, like a producer consumer model), we often launch just one thread block per SM, so it's important to structure work so that all warp schedulers are kept busy.

**GMEM Model**

- It is implemented as a stack of DRAM layers with a logic layer at the bottom (HBM).
- So, GMEM is nothing but a matrix of DRAM cells and DRAM cells are glorified guarded capacitors.
- Access patterns matter because of the physics of DRAM cells
- The important conclusion is row access cost >> col access cost. So, traversing columns should be in the inner/fast for loop as that would lead to a single row read follwed by 256 column reads(for 256x256 matrix data) for one iteration of the inner for loop(256 row reads in total).
- So when people say “GMEM coalescing is very important”, this is what they mean: threads should access contiguous memory locations to minimize the number of DRAM rows touched.

**SMEM Model**

- Shared memory (SMEM) has very different properties from GMEM. It is built from SRAM cells rather than DRAM, which gives it fundamentally different speed and capacity trade-offs.
- It takes many more transistors to store a single bit of information.
- SMEM is organized into 32 banks, each bank 32 bits wide (4 bytes)
- SMEM can serve data from all 32 banks (128B) in a single cycle — but only if one rule is respected: **Threads in a warp must not access different addresses within the same bank. Otherwise, those requests are serialized across multiple cycles.**
- This situation is known as a bank conflict. If N threads access different addresses of the same bank, the result is an N-way bank conflict and the warp’s memory request takes N cycles to complete.
- In the worst case, all 32 threads target different addresses in the same bank, and throughput drops by a factor of 32.
- Importantly: if multiple threads in a warp access the same address within a bank, SMEM can broadcast (or multicast) that value to all of them.

- Just respect the "physics" of the hardware and it will reward you with performance!

**L1 Model**

- L1 and SMEM share the same physical storage, but L1 adds a hardware-managed scaffolding layer around that storage.

**The gradient across GPU generations:**

-  The biggest generational jump so far was from Ampere → Hopper, with the introduction of: 
   -  Distributed Shared Memory (DSMEM): direct SM-to-SM communication for loads, stores, and atomics across the SMEMs of an entire GPC.
   -  TMA: hardware unit for asynchronous tensor data movement (GMEM ↔ SMEM, SMEM ↔ SMEM).
   - Thread Block Clusters: a new CUDA programming model abstraction for grouping blocks across SMs.
   - Asynchronous transaction barriers: split barriers that count transactions (bytes) instead of just threads.

- Ampere (e.g. A100) itself introduced several key features:
  - tf32 and bf16 support in Tensor Cores.
  - Asynchronous copy (GMEM → SMEM) with two modes: bypass L1 and access L1.
  - Asynchronous barriers (hardware-accelerated in shared memory).
  - CUDA task graphs, which underpin CUDA graphs in PyTorch and reduce CPU launch + grid initialization overhead.
  - Warp-level reduction instructions exposed through CUDA Cooperative Groups (enabling warp-wide, integer dtype, reductions in a single step, without shuffle patterns).



**GPU assembly languages: PTX and SASS**

- One level above the hardware to its ISA (Instruction Set Architecture). An ISA is simply the set of instructions a processor (e.g., an NVIDIA GPU) can execute, along with their binary encodings (opcodes, operands, etc.) and behavioral semantics. Together, these define how programmers can direct the hardware to do useful work.
- The human-readable form of an ISA is known as the assembly: instead of writing raw binary like `0x1fff…3B`, a programmer uses mnemonics such as `FMA R12, R13, R14, R15` to express the same instruction.
- On NVIDIA GPUs, the native ISA is called SASS. Unfortunately, it is poorly documented—especially for the most recent GPU generations.
- PTX is NVIDIA's virtual ISA: an instruction set for an abstract GPU. PTX code is not executed directly; instead, it is compiled by ptxas into the native ISA (SASS).
- The key advantage of PTX is forward compatibility. A CUDA program compiled to PTX a decade ago can still run on a modern GPU like Blackwell. It may not exploit the latest hardware features efficiently, but it will execute correctly.
- This works because PTX is embedded into the CUDA binary alongside native SASS. When the binary runs on a future GPU, if matching SASS code is not already present, the PTX is JIT-compiled into SASS for the target architecture.

- ISA = the set of instructions a target processor can execute, with details of their binary encoding(format), and behaviour(semantics)
- Virtual ISA = ISA for abstract/virtual processor
- cubin(cuda binary) is in ELF format, contains SASS and ELF metadata(symbol table...)

- Why care about PTX/SASS? Because this is where the last few percent of performance can be found. On today's scale, those "few percent" are massive: if you're training an LLM across 30,000 H100s, improving a core kernel's performance by even 1% translates into millions of dollars saved.
- As [Aroun](https://github.com/ademeure) likes to put it: when writing large scale training/inference kernels, we care about `O(NR)`, not `O(N)`. (Here, NR = nuclear reactors.) In other words, there are likely no new asymptotic complexity classes waiting to be discovered — the big wins are (mostly) gone. But squeezing out ~1% efficiency across millions of GPUs is the equivalent of saving a few SMRs (small modular reactors) worth of energy.

- It's not that understanding SASS means you'll start writing CUDA kernels directly in SASS. Rather, when writing CUDA C++ you want to stay tightly coupled to the compiler's output (PTX/SASS). This lets you double-check that your hints (e.g., `#pragma unroll` to unroll a loop, or vectorized loads) are actually being lowered into the expected instructions (e.g., `LDG.128`).
- Note also that some instructions have no equivalent in CUDA C++; you simply have to write inline PTX!


Doubts:
- (D) In fig 18, is the blockIdx.x and blockIdx.y mentioned pictorially right? In fig 19, its mentioned as "warp 1 from block (blockIdx.x, blockIdx.y) = (1,0)", with that as the case, then x goes from top to bottom right, but in the fig 18, x row of block dims span accross left to right.
- PTX code is generated for a thread block right? and will the PTX code shown executed per thread in the thread block in parallel?
- What is exposing ILP (in PTX code explanation)? - Instruction-Level Parallelism
- How loop unrolling exposes Instruction-Level Parallelism? So, does these instructions, run in parallel, as the warp takes and executes them? 

- In general, three resources limit concurrency: Registers, Shared memory (SMEM) & Threads/warps
- In CUDA terminology, occupancy usually refers to the number of concurrent blocks that can run on an SM.
- Occupancy (warps): the ratio of active warps to the maximum number of warps per SM.
- Here, "active warps" means the warps of a thread block after they've been allocated resources (registers, SMEM, etc.) at launch.

- SASS was a bit harder to understand, may be it could be due to the fact that its not explained in detail. For now, just got the gist and proceeding.

Designing near-SOTA synchronous matmul kernel

**Roofline model**

- The roofline model plots performance (FLOP/s) on the y-axis against arithmetic intensity (AI) on the x-axis.
- Arithmetic intensity is defined as the number of FLOPs performed per byte loaded from device memory / GMEM (by default).

- The “ridge point” occurs at: `peak perf / GMEM bw`. For my H100 PCIe, this works out to ~410. Only once AI exceeds this value can the kernel enter the compute-bound regime.

- (D) Loading A → As. This step is trickier because As is transposed. The reason for the transpose is that it enables vectorized loads (LDS.128) later during the compute phase.
- (D) The trade-off is that the stores cannot be vectorized: the 4 floats fetched from a row of A must now be scattered into a column of As, which maps into the same memory bank. That's acceptable because we prioritize fast loads — each element of As will be accessed multiple times during computation, while the stores happen only once.