---
layout: post
title: GEMM Kernel Optimization Notes
date: 2026-03-28 21:49:00-0400
featured: false
description: My notes from Simon Boehm's CUDA GEMM optimization blog
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
---

#### **Introduction**

These are my notes from Simon Boehm's excellent [CUDA GEMM kernel optimization blog](https://siboehm.com/articles/22/CUDA-MMM). First and foremost, Simon really did a great job explaining the kernels and helping me internalize the intuition — hats off for the time and effort he put into it.

These notes exist as a reference to firmly hold these ideas and recall them later. Writing something in your own words takes effort, but pays off well down the line. I go through all the kernels one by one, add notes, and build a mental model. I refer a lot to Simon's blog for all the great drawings he made, and highly recommend you check that out first if possible.

A few housekeeping notes:

- I used Claude for rephrasing my own words — the understanding is mine, the polish is AI-assisted.
- This is a work in progress — I'll be adding one kernel at a time.
- As you read, try to **mentally visualize** the thread mappings, memory accesses, and data flow. It really helps solidify the intuition.

That's it on the intro — let's get in.

#### **Prerequisites**

I'm assuming readers are comfortable with CUDA basics — I won't go deep into them, but will touch upon what's needed briefly. If you need a refresher, here are my notes for reference: [GPU Programming Intro]({{ site.baseurl }}{% post_url 2025-09-23-GPU-Intro %}).

Also, since the visualizations Simon provides are excellent, I recommend having [his blog](https://siboehm.com/articles/22/CUDA-MMM) open in another tab while reading this.

---

#### **What is SGEMM?**

SGEMM stands for **S**ingle precision (FP32) **GE**neral **M**atrix **M**ultiply. It's one of the most basic yet important operations in all of deep learning training and inference. Its form is:
```
C = αAB + βC
```

For NVIDIA GPUs, cuBLAS provides highly optimized kernels for this. Matching cuBLAS-level performance will be our goal as we go through each kernel one by one.

#### **Quick CUDA Recap**

In CUDA, the hierarchy is: a kernel launch creates a **grid** → which contains **blocks** → which contain **threads**. All threads within a block share the same shared memory (SMEM) on the SM.

The number of threads in a block is configured via `blockDim` (a 3-int vector: `blockDim.x`, `blockDim.y`, `blockDim.z`). Similarly, the number of blocks in a grid is configured via `gridDim`. When we launch a kernel from the **host** (CPU), it creates a single grid on the **device** (GPU) with the specified blocks and threads. I'll use host/CPU and device/GPU interchangeably.

We work with matrices A (M×K), B (K×N), C (M×N). For simplicity, we assume square matrices throughout — handling non-square sizes involves extra boundary checks and optimizations to avoid thread wastage, which I haven't explored yet and won't cover here.

In CUDA, we write code from a **single thread's perspective**. The runtime handles parallelism and hardware mapping. The key questions for each kernel are:

- What work does each thread do?
- What is the memory layout, and how does it affect performance?
- Where do we store intermediate data, and how does data move before reaching the CUDA cores?

One more thing: all kernels here operate on **CUDA cores** (FP32 ALUs). With **tensor cores**, the mental model for how operations and data flow work changes significantly — that's next on my plate but not covered here.

---

#### **Kernel 1: Naive Implementation**

The simplest approach — just like how we learned matrix multiply in school. Take a row of A, a column of B, compute their dot product, and that gives one element of C. Three nested loops.

We launch the kernel like so:
```cuda
// create as many blocks as necessary to map all of C
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
// 32 * 32 = 1024 threads per block
dim3 blockDim(32, 32, 1);
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

This grid/block setup is mostly similar across kernels, so I won't repeat it each time.

The kernel itself:
```cuda
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```

Each thread computes **one element** of C. All threads work independently on their respective row of A and column of B — no synchronization needed. The data is loaded directly from **global memory (GMEM)**, which is off-chip with latencies in the range of 200–500 clock cycles — very expensive given how fast GPU compute units are.

**Tile quantization:** When the matrix dimensions aren't divisible by the block size, we still launch full blocks — some threads at the boundary end up with no elements to compute and go to waste. This is called tile quantization. There are techniques to mitigate this, but I haven't explored them yet — we'll save that for a later post.

> **Errata in Simon's blog (Note 8):** The note says `threadIdx.x` and `threadIdx.y` vary "based on the position of the thread in the **grid**." It should say **block** — `threadIdx` is the position within the block, not the grid.

##### **Lower Bounding the Fastest Possible Runtime**

For a GEMM of A (M×K) × B (K×N) + C (M×N):

- **Total FLOPs:** `2 × M × N × K + M × N`. For each element of C, we do a dot product of length K — that's a multiply and an add per step, so `2K` FLOPs (counted as FMA = 2 FLOPs). Then M×N additions for the `+ βC` term. (We're ignoring the α and β scalar multiplies for simplicity.)
- **Total data to read (minimum):** `(M×K + K×N + M×N) × 4B` (FP32)
- **Total data to store:** `M×N × 4B`

For M = K = N = 4092 (Simon's benchmark size):

- FLOPs: `2 × 4092³ + 4092² ≈ 137 GFLOPs`
- Data to read: `3 × 4092² × 4B ≈ 201 MB`
- Data to store: `4092² × 4B ≈ 67 MB`
- **Total memory traffic (minimum):** ~268 MB

On Simon's A6000 (30 TFLOPs/s FP32, 768 GB/s GMEM bandwidth):

- Compute time at peak: `137 GFLOPs / 30 TFLOPs/s ≈ 4.5 ms`
- Memory time at peak: `268 MB / 768 GB/s ≈ 0.34 ms`

Compute takes ~10× longer than memory — so an optimized kernel will be **compute-bound**, as long as total memory traffic stays under ~10× the minimum 268 MB.

##### **Memory Access Pattern of the Naive Kernel**

Assuming zero caching, each thread loads `2 × 4092 + 1` floats from GMEM. With 4092² threads total, that's ~548 GB of memory traffic — far above the 268 MB minimum.

**Thread-to-element mapping:**

With `blockDim = (32, 32)`, threads are grouped into warps based on linearized `threadId = threadIdx.x + 32 * threadIdx.y`. So warp 0 contains threads with `threadIdx.x = 0..31, threadIdx.y = 0`.

Now, from the kernel code:

- `x = blockIdx.x * 32 + threadIdx.x` → mapped to **rows** of A and C
- `y = blockIdx.y * 32 + threadIdx.y` → mapped to **columns** of B and C

For warp 0 (all threads have `threadIdx.y = 0`): each thread gets a **different row** (x = 0, 1, 2, ..., 31) but the **same column** (y = 0).

When these 32 threads access A in the inner loop — `A[x * K + i]` for a given `i` — they hit addresses `A[0*K+i], A[1*K+i], A[2*K+i], ...`. These are **K elements apart** in memory (row-major). That's a strided access — the worst case for coalescing.

Meanwhile, for B — `B[i * N + y]` — all 32 threads read the **same address** (y = 0 for all), so it's a broadcast.

**The core problem:** consecutive threads in a warp (varying `threadIdx.x`) are mapped to different **rows**. In row-major layout, different rows are far apart in memory. So every warp issues 32 separate memory transactions instead of one coalesced 128B transaction. This is why the naive kernel achieves only 15 GB/s GMEM throughput vs. a peak of 768 GB/s.

**A note on memory transactions:** Throughout these notes, B = bytes. The GPU GMEM subsystem operates in **32-byte sectors**. When a warp issues a memory instruction, the hardware serves it using the minimum number of 32B sectors needed. If all 32 threads access consecutive 4B floats (128B total, contiguous), it's served as a single 128B transaction (4 sectors). If addresses are scattered, each may require its own 32B sector access — up to 32 separate transactions in the worst case. More on this in the next kernel.

> We'll track this **thread → element mapping** for every kernel going forward — it's the most critical thing to get right, as it directly determines memory access patterns and coalescing behavior.


> **Errata in Simon's blog:** Simon writes two example threads as (0, 0) and (0, 1), and describes them as loading "the same column of B but different rows of A." But with his mapping (`x` from `threadIdx.x` = row, `y` from `threadIdx.y` = column), threads (0, 0) and (0, 1) share the same row of A and access different columns of B. For the description and diagram to be consistent, the second thread should be **(1, 0)**, not (0, 1). **[TODO: Confirm with Simon and update.]**

The naive kernel achieves ~300 GFLOPs on the A6000 — just 1% of the theoretical 30 TFLOPs.

So how do we make this faster? By optimizing memory access patterns so that global memory accesses can be **coalesced** (combined) into fewer transactions.

#### **Kernel 2: Global Memory Coalescing**

##### **Warps and Thread Grouping**

Before we dive in, let's formalize the concept of a **warp**. A warp is a hardware-level grouping of 32 threads within a block. All threads in a warp are issued the same instruction and executed by one of the **4 warp schedulers per SM**. This execution model is called **SIMT** (Single Instruction, Multiple Threads). It's similar to SIMD, but with a key difference: in SIMT, threads *can* diverge (take different branches), though divergence is expensive since the warp serializes the divergent paths. When all threads follow the same path, it's efficient.

Threads are grouped into warps based on a linearized thread ID:
```
threadId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
```

Threads with consecutive `threadId` values belong to the same warp.

##### **What is Global Memory Coalescing?**

When threads within a warp access **adjacent memory locations**, the hardware can **coalesce** these individual requests into a single bulk memory transaction. The GPU supports 32B, 64B, and 128B memory transactions. So if each of the 32 threads in a warp loads one 4B float from consecutive addresses, that's `32 × 4B = 128B` — which fits perfectly into a single 128B transaction.

If the accesses are **not** consecutive (strided or scattered), the hardware must issue **multiple** smaller transactions to satisfy all 32 threads — up to 32 separate 32B loads in the worst case. Each transaction costs cycles, so minimizing the number of transactions directly reduces latency.

> **Important (Simon's Note 20):** To allow coalescing, threads within a warp must access **consecutive addresses** — but the accesses don't have to be **in order** within the warp. Thread 5 can access address 100, thread 0 can access address 120, etc., as long as the set of addresses forms a contiguous block. The hardware handles the reordering.

##### **Why Kernel 1 Fails at Coalescing**

Recall Kernel 1's **thread → element mapping** (see Kernel 1 notes for full breakdown):

| | Warp 0 threads (threadIdx.x = 0..31) |
|---|---|
| Row (x) | 0, 1, 2, ..., 31 (all **different**) |
| Column (y) | 0, 0, 0, ..., 0 (all **same**) |

For `A[x * K + i]`: threads access rows 0, 1, 2, ... of A — addresses that are **K apart** in memory. Strided. Not coalesced.

This is the direct consequence of mapping `threadIdx.x → row`. Look at Simon's visualization for this — mentally place each thread and trace which memory addresses it touches. That's the key image to internalize.

##### **Fixing It: Remapping Threads to Elements**

To enable coalescing, we remap how threads are assigned to elements of C. The block becomes 1D (`blockDim = 1024`), and we derive row/column differently:
```cuda
const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // row
const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // column
```

**New thread → element mapping for warp 0** (threadIdx.x = 0..31):

| | Warp 0 threads (threadIdx.x = 0..31) |
|---|---|
| Row (x) | `threadIdx.x / 32` = 0 for all → **same row** |
| Column (y) | `threadIdx.x % 32` = 0, 1, 2, ..., 31 → **different columns** |

Now trace the memory accesses:

- **A:** `A[x * K + i]` — all threads have the same `x`, so they all read the **same address**. The hardware can **broadcast** this to all threads in one transaction.
- **B:** `B[i * N + y]` — threads access `B[i*N + 0], B[i*N + 1], ..., B[i*N + 31]`. These are **32 consecutive** 4B floats = 128B → perfectly **coalesced** into a single transaction.

The kernel:
```cuda
// blockDim is now 1D: 1024 threads (instead of 32x32)
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32 * 32);
sgemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```
```cuda
__global__ void sgemm_coalescing(int M, int N, int K, float alpha, const float *A,
                                  const float *B, float beta, float *C) {
  // derive row and column from 1D threadIdx.x
  const int x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);  // row
  const int y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);  // column

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```

##### **Results**

Just by changing the thread-to-element mapping, GMEM throughput jumps from **15 GB/s to 110 GB/s**. Performance goes from **~300 GFLOPs to ~2000 GFLOPs** — a ~6.5× improvement from a two-line code change.

We're still far from the 30 TFLOPs peak though. The next step: use the GPU's fast on-chip memory — **shared memory (SMEM)** — to cache data that gets reused, reducing the number of expensive GMEM accesses.

---

#### **Kernel 3: Shared Memory Cache-Blocking**

##### **SMEM vs GMEM**

Global memory (GMEM) is **off-chip** — far from the execution units, with high latency (200–500 cycles). Shared memory (SMEM) is **on-chip**, physically located on the SM, much closer to the cores. In terms of latency: registers < SMEM < L1/L2 cache < GMEM.

Key properties of SMEM:

- All threads within the same block share it — it's the primary mechanism for **intra-block communication**.
- Each block gets its own chunk of SMEM.
- On the SM, the L1 cache and SMEM share the same physical storage (the "unified data cache"), and the split is **configurable** — programmers can control how much to allocate to each.
- Bandwidth is dramatically higher. As Simon notes: Volta benchmarks report ~750 GiB/s for GMEM bandwidth vs. ~12,080 GiB/s for SMEM bandwidth (from [this paper](https://arxiv.org/abs/1804.06826)). Ampere numbers are in a similar range.

> **Errata in Simon's blog (Note 23):** The note says "it's possible to use more than 48KB of SMEM **per thread**." It should be **per block**.

##### **The Idea: Tiling**

Instead of having each thread read an entire row of A and column of B from GMEM, we load **2D tiles (chunks)** of A and B into SMEM, do the work from there, then slide the tiles forward.

Concretely:

- Take a tile of A (BLOCKSIZE × BLOCKSIZE) and slide it **horizontally** along A's columns.
- Take a tile of B (BLOCKSIZE × BLOCKSIZE) and slide it **vertically** down B's rows.
- At each step, load the current tiles into SMEM (`As` and `Bs`), compute partial dot products from SMEM, and accumulate into each thread's local result.

In terms of what each thread does:

1. **Load:** Each thread loads **one element** of the current tile of A and one element of B from GMEM into SMEM.
2. **Sync (`__syncthreads`):** Wait for all threads to finish loading — this is critical because the next step needs the full tiles.
3. **Compute:** Each thread computes the dot product of its row in `As` with its column in `Bs`, accumulating into a local variable.
4. **Sync again (`__syncthreads`):** Before moving to the next tile, we must ensure all threads are done reading the current `As` and `Bs` — otherwise fast threads could overwrite the SMEM with the next tile before slow threads finish.
5. **Advance:** Move the tile window forward (A shifts right by BLOCKSIZE columns, B shifts down by BLOCKSIZE rows) and repeat.

I highly recommend tracing through Simon's illustration of this mentally — visualize the sliding tiles and what each thread touches at each step.
```cuda
__global__ void sgemm_smem(int M, int N, int K, float alpha, const float *A,
                           const float *B, float beta, float *C) {
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;
  const uint threadCol = threadIdx.x % BLOCKSIZE;  // column within the tile
  const uint threadRow = threadIdx.x / BLOCKSIZE;  // row within the tile

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // advance pointers to the starting positions for this block
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // each thread loads one element of A and B into SMEM
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();  // wait for tile to be fully loaded

    // advance pointers to the next tile
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // dot product of this thread's row of As and column of Bs
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }

    __syncthreads();  // wait before overwriting SMEM with next tile
  }
  C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}
```

##### **Results**

This kernel achieves ~2980 GFLOPs — roughly a 50% improvement over Kernel 2. The improvement is modest partly because Kernel 2 already had decent L1 cache hit rates. We're still far from the ~30 TFLOPs the GPU can provide.

##### **Roofline Analysis**

The **roofline model** is a visual tool that shows the two fundamental ceilings on kernel performance:

1. **Compute ceiling (horizontal line):** The GPU's peak FLOPs/s — no kernel can exceed this regardless of how efficiently it uses memory. For Simon's A6000, this is ~30 TFLOPs/s.
2. **Memory bandwidth ceiling (diagonal line):** Performance limited by how fast data can be fed to the cores. This line has a slope equal to the peak memory bandwidth. A kernel operating at arithmetic intensity `I` (FLOPs per byte transferred) can achieve at most `I × peak_bandwidth` FLOPs/s.

The **x-axis** is arithmetic intensity (FLOPs/byte), and the **y-axis** is achieved FLOPs/s. The two ceilings form a "roof" shape:

- **Left of the ridge point** (where the diagonal meets the horizontal): the kernel is **memory-bound** — performance is limited by data transfer, not compute. Increasing arithmetic intensity (fewer bytes per FLOP) moves you right along the diagonal toward better performance.
- **Right of the ridge point:** the kernel is **compute-bound** — the cores are the bottleneck. You've saturated the compute units.

For Kernel 3: it sits on the diagonal (memory-bound region). It actually achieves *higher bandwidth* than cuBLAS, but because it does much *less work per byte loaded* (lower arithmetic intensity), overall FLOPs/s is worse. The path forward is clear: increase arithmetic intensity so we move right on the roofline, toward the compute ceiling.

##### **SMEM Usage and Occupancy**

At BLOCKSIZE = 32, the kernel uses `2 × 32 × 32 × 4B = 8 KB` of SMEM per block. (Obtainable via `--ptxas-options=-v`: `Used 37 registers, 8192 bytes smem, 400 bytes cmem[0]`.)

The A6000 allows up to 48 KB of SMEM per block, so we're well under the limit. But there's a trade-off: each SM has a total of ~100 KB of SMEM. If a kernel used the full 48 KB per block, only 2 blocks could be resident on an SM simultaneously. This reduces **occupancy** — the ratio of active warps to the maximum possible active warps on an SM.

Why does occupancy matter? Because of **zero-cost warp switching**. On a GPU, all resources (registers, SMEM) for every resident thread are **pre-allocated and stay resident** on the SM for the block's entire lifetime. When a warp stalls (waiting for a memory load, for example), the warp scheduler simply picks another ready warp and issues its instruction — **no save/restore overhead**. This is fundamentally different from CPU context switching, which requires saving and restoring register state to/from memory (costing cycles). Higher occupancy means a larger pool of resident warps, which means more chances to find a ready warp when one stalls, which means better latency hiding.

Three resources limit how many blocks can be resident on an SM: **register count**, **warp/thread count**, and **SMEM capacity**.

##### **Occupancy Calculation for Kernel 3**

Hardware limits for the A6000 (from `cudaGetDeviceProperties`):

| Metric | Value |
|---|---|
| Max threads per SM | 1536 |
| Max warps per SM | 48 |
| Max SMEM per SM | 102400 B |
| Max registers per SM | 65536 |
| Max SMEM per block | 48 KB |
| CUDA runtime SMEM overhead per block | 1024 B |
| Register allocation granularity | 256 regs, per warp |

Kernel resource demands:

| Metric | Value |
|---|---|
| Threads per block | 1024 |
| Registers per thread | 37 |
| SMEM per block | 8192 B |

A block can only be assigned to an SM if **all** of its requested resources can be satisfied. Now the calculation:

- **SMEM:** (8192 + 1024) B per block = 9216 B. 102400 / 9216 = 11.1 → **11 blocks** upper limit.
- **Threads:** 1024 threads per block, max 1536 per SM → **1 block** upper limit.
- **Registers:** 37 regs/thread × 32 threads/warp = 1184 regs/warp, rounded up to 1280 (allocation granularity is 256). 32 warps/block × 1280 = 40960 regs/block. Max 65536 per SM → **1 block** upper limit.

The bottleneck is threads and registers — only **1 block** fits per SM, giving 32 active warps out of a maximum 48 = **66% occupancy**.

66% occupancy isn't terrible, so occupancy alone doesn't explain the poor performance.

##### **The Real Bottleneck: Instruction Mix**

Profiling the kernel reveals that the majority of executed instructions are **LDS** (shared memory loads), not FMA (the actual compute). The inner loop in PTX looks like:
```
ld.shared.f32   %f91, [%r8+3456];   // SMEM load from As
ld.shared.f32   %f92, [%r7+108];    // SMEM load from Bs
fma.rn.f32      %f93, %f92, %f91, %f90;  // the actual compute
```

Two SMEM loads for every one FMA. Since SMEM loads have higher latency than an FMA, the compute units are starved. Looking at the profiler's warp stall breakdown, the dominant stall reason is **`Stall MIO Throttle`** — as Simon quotes from the [Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference):

> "Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions."

We're not using special math instructions or dynamic branches — so it's clear the kernel is bottlenecked on SMEM access throughput.

The fix: have each thread compute **more than one output element**, so we get more FMAs per SMEM load — shifting work into registers and reducing pressure on the SMEM pipeline. That's Kernel 4.

---

#### **Kernel 4: 1D Blocktiling — Multiple Results per Thread**

To be honest, this one took me some time to internalize, so I'll try to be as clear as possible.

##### **The Idea**

At a high level, it's simple: increase the work done by each thread by making it compute **multiple elements** of C, not just one. This reduces the ratio of memory instructions to compute instructions, which is exactly what we need. But the devil is in the details.

The kernel still uses the same outer loop as Kernel 3 — sliding tiles of A and B from GMEM into SMEM. The SMEM tile sizes are now `BM × BK` for A and `BK × BN` for B, with `BM = BN = 64, BK = 8`. Total SMEM: `(64×8 + 64×8) × 4B = 4 KB` per block.

The key change is in the **inner loops** — see Simon's illustration and follow along.

##### **Walking Through the Inner Loop**

Each thread now computes a **column of TM elements** in the output tile of C (not just one element). To accumulate these partial results, each thread allocates a small vector in **registers**:
```cuda
// thread-local accumulator, stored in registers
float threadResults[TM] = {0.0};
```

This is `TM` floats, local to each thread, living in the register file — the fastest memory on the GPU.

The inner loop structure:
```cuda
// outer loop: slide tiles along K dimension
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // load one element each of A and B from GMEM → SMEM (same as Kernel 3)
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  A += BK;
  B += BK * N;

  // inner loops: compute partial results from SMEM
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // cache one element of Bs (shared across all TM results)
    float Btmp = Bs[dotIdx * BN + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
  }
  __syncthreads();
}
```

Let me walk through what one thread does, step by step. Say this thread owns column `threadCol` in the output tile and rows `threadRow * TM` through `threadRow * TM + TM - 1`.

At `dotIdx = 0` (first column of the current `As` tile):
1. Cache `Bs[0 * BN + threadCol]` — that's one element from the first row of `Bs`, at this thread's column. Store it in `Btmp`.
2. Loop `resIdx = 0..TM-1`: multiply `As[row][0]` (first column value for each of the TM rows) by `Btmp`, and accumulate into `threadResults[resIdx]`.

At `dotIdx = 1` (second column of `As`):
1. Cache the new `Btmp = Bs[1 * BN + threadCol]`.
2. Again loop over all TM rows of `As`, multiply by `Btmp`, accumulate.

This continues for all `BK` columns. By the end, each `threadResults[resIdx]` holds the partial dot product contribution from this tile.

The key insight: `Btmp` is loaded **once** and reused across all TM rows — that's TM FMAs for just 1 SMEM load from `Bs`. Each `As` element is loaded once per `resIdx`. So per `dotIdx` step: `1 + TM` SMEM loads for `TM` FMAs.

Each thread works on its own column of the output tile, and all threads execute in parallel. **This is the core logic.** If you understand this, Kernel 5 is a natural extension — instead of each thread computing a column (1D), it computes a 2D block of C, using an outer product trick. We'll get to that next.

##### **A Point About the Outer Loop and Parallelism**

One thing worth making explicit: a thread block "owns" a fixed tile of C (determined by `blockIdx`). Its outer loop slides tiles of A horizontally and tiles of B vertically, accumulating partial results until the entire K dimension is traversed. Only then is the final result for that tile of C complete. Different thread blocks own different tiles of C and can execute this entire traversal **independently and in parallel** — there's no cross-block communication needed.

##### **Results and Memory Access Analysis**

This kernel achieves ~8600 GFLOPs — 2.2× faster than Kernel 3.

Let's compare the memory access patterns. K is the common dimension we tile over — it determines how many outer loop iterations we do. Each outer loop step processes one tile along K and accumulates into the result.

**Kernel 3** (1 result per thread, BLOCKSIZE = 32):

- GMEM: `K/32` outer iterations × 2 loads (one A, one B element per thread) = `K/16` per result
- SMEM: each dot product step loads one element from `As` and one from `Bs` → `K/32 × 32 × 2` total = `K×2` per result

**Kernel 4** (TM = 8 results per thread, BK = 8):

- GMEM: `K/8` outer iterations × 2 loads = `K/4` total, but shared across 8 results → `K/32` per result
- SMEM: each `dotIdx` step loads 1 from `Bs` + TM from `As` = 9 loads. Over `K/8 × BK = K` steps total → `K×9` total, across 8 results → `K×9/8` per result

Both GMEM and SMEM accesses per result are reduced. As expected, the profiler shows significantly fewer cycles spent stalling on memory pressure (see Simon's warp stall comparison chart).

##### **Sidenote: Compiler Optimizations**

Simon noted something interesting: if you swap the loop order (make `resIdx` the outer loop and `dotIdx` inner) and remove the explicit `Btmp` caching, performance doesn't change. The compiler is smart enough to unroll both loops (since loop counts are known at compile time) and eliminate redundant SMEM loads of `Bs` entries — arriving at the same instruction count.

Also, when PTX is lowered to SASS, the SMEM loads from `Bs` get **vectorized** into `LDS.128` (128-bit loads), loading 4 floats at once.

> **Simon's Note 39:** "This already hints at an optimization we'll perform later: transposing `As` such that we can also vectorize those loads." — we'll see this in Kernel 6.

##### **Why We Need More: Arithmetic Intensity**

**Arithmetic intensity** = FLOPs executed per byte transferred between GMEM and SMEM (counting both loads and stores).

This kernel still suffers from the same stalling-for-memory problem as Kernel 3, just to a lesser extent. The fix is the same: compute even more results per thread to increase arithmetic intensity.

Simon's visualization (Note 41) makes this clear: computing a **square** of results per thread is more efficient than a column, because a square lets you **share more inputs** across results. A TM×1 column reuses each `Btmp` across TM rows, but each `As` value is used only once. A TM×TN square reuses each `As` value across TN columns *and* each `Bs` value across TM rows — the outer product structure.

The fundamental point: all our kernels perform the **same total FLOPs**. The only thing we're changing is how many GMEM/SMEM accesses we need. By computing more results per thread, each loaded value gets reused more, arithmetic intensity goes up, and we push the kernel from memory-bound toward **compute-bound** — which is where we want to be, since the GPU has far more compute throughput than memory bandwidth. We'll keep optimizing arithmetic intensity as long as we remain memory-bound.

---

#### **Kernel 5: 2D Blocktiling — Even More Results per Thread**

As hinted at the end of Kernel 4, we now extend the idea: instead of each thread computing a **column** of results, each thread computes a **2D sub-tile** of 8×8 = 64 elements of C. This is the outer product approach.

##### **Stage 1: GMEM → SMEM Loading**

The first stage is the same idea as before — all threads cooperate to populate `As` and `Bs` in SMEM. But now each thread loads **multiple elements**, since the tiles are larger (BM=BN=128, BK=8) but we have fewer threads (256).

Within one tile of `As` (size BM×BK = 128×8), each thread loads one element per loop iteration, but the `loadOffset` loop makes each thread **traverse multiple rows** of the tile. With `strideA = numThreads / BK = 256/8 = 32`, and `BM = 128`, each thread loads `128/32 = 4` elements of A per outer iteration. Same logic applies to B.

This is a pattern of **chunked loading within a chunk** — the outer loop selects which GMEM tile to bring into SMEM, and within that tile, threads cooperatively load pieces across multiple iterations. This nested chunking becomes more layered as we go forward. I think of it like the movie Inception — a dream within a dream within a dream, and we do the actual compute in the innermost level.

See Simon's image (Note 42) for the visual.

##### **Stage 2: Two Separate Thread Mappings**

This is critical to understand. Each thread has **two independent mappings**, serving different purposes:

**Mapping 1 — for GMEM → SMEM loading (Step 5a in code):**
```cuda
const uint innerRowA = threadIdx.x / BK;   // row within BM×BK tile of A
const uint innerColA = threadIdx.x % BK;   // col within BM×BK tile of A
const uint innerRowB = threadIdx.x / BN;   // row within BK×BN tile of B
const uint innerColB = threadIdx.x % BN;   // col within BK×BN tile of B
const uint strideA = numThreads / BK;      // 256/8 = 32
const uint strideB = numThreads / BN;      // 256/128 = 2
```

This determines **which GMEM elements** this thread loads into SMEM. All 256 threads cooperate to fill the entire `As` and `Bs` tiles — each thread handles a few elements via the `loadOffset` loop.

**Mapping 2 — for SMEM → register computation (Step 5b) and writing results (Step 6):**
```cuda
const uint threadCol = threadIdx.x % (BN / TN);  // sub-tile column: 0..15
const uint threadRow = threadIdx.x / (BN / TN);  // sub-tile row: 0..15
```

With BN/TN = 128/8 = 16, the 256 threads form a **16×16 grid of sub-tiles**. Each thread owns one TM×TN = 8×8 piece of the output. For example, thread at `(threadRow=3, threadCol=5)` owns rows 24–31 of `As` and columns 40–47 of `Bs`, and writes its 64 results to the corresponding 8×8 region of C.

These mappings are **independent** — a thread might load elements at the top of `As` (Mapping 1) but compute using elements from the middle of `As` (Mapping 2).

##### **Stage 3: The Inner Loops — Outer Product**

Here's the full kernel with every step annotated:
```cuda
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {
  // Step 0: Block-level position in C
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // Step 1: Mapping 2 — which 8×8 sub-tile this thread computes
  const uint threadCol = threadIdx.x % (BN / TN);  // 0..15
  const uint threadRow = threadIdx.x / (BN / TN);  // 0..15

  // Step 2: Mapping 1 — which elements this thread loads into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideA = numThreads / BK;  // 32
  const uint strideB = numThreads / BN;  // 2

  // Step 3: Allocate SMEM and registers
  __shared__ float As[BM * BK];           // 128×8
  __shared__ float Bs[BK * BN];           // 8×128
  float threadResults[TM * TN] = {0.0};   // 64 partial results
  float regM[TM] = {0.0};                 // register cache: one col of As sub-tile
  float regN[TN] = {0.0};                 // register cache: one row of Bs sub-tile

  // Step 4: Advance pointers to this block's starting position
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Step 5: Outer loop — slide tiles along K
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // Step 5a: GMEM → SMEM (cooperative loading, uses Mapping 1)
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    A += BK;
    B += BK * N;

    // Step 5b: SMEM → Registers → Compute (uses Mapping 2)
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Load one column of this thread's As sub-tile into regM
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      // Load one row of this thread's Bs sub-tile into regN
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      // Outer product: accumulate into threadResults
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // Step 6: Write results to C (uses Mapping 2)
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}
```

Let me walk through one thread's execution of Step 5b. Say `threadRow = 3, threadCol = 5` — this thread owns rows 24–31 of `As` and columns 40–47 of `Bs`.

At `dotIdx = 0`:
1. Load `As[row 24..31][col 0]` into `regM[0..7]` — one column of this thread's vertical strip.
2. Load `Bs[row 0][col 40..47]` into `regN[0..7]` — one row of this thread's horizontal strip.
3. Compute the **outer product**: `regM[i] × regN[j]` for all i, j → an 8×8 matrix of partial products. Accumulate into `threadResults`.

At `dotIdx = 1`:
1. Load `As[row 24..31][col 1]` into `regM`, `Bs[row 1][col 40..47]` into `regN`.
2. Outer product → accumulate.

Continue for all BK = 8 steps. After all outer loop iterations finish traversing K, `threadResults` holds the final 64 values for this thread's 8×8 sub-tile of C.

I know this is getting complex. And to be honest, this isn't even close to the most optimized kernel — cuBLAS, CUTLASS, and CuTe push things much further with even more advanced tiling and scheduling strategies. Reading and understanding all this really makes me appreciate the work these engineers have put in. I really want to work alongside some of these people someday — wish me luck.

##### **Results and Memory Access Analysis**

Performance: **~16 TFLOPs** — another 2× improvement. Each thread now computes `TM × TN = 64` results.

**GMEM accesses per thread:**
- Each outer iteration (`K/BK = K/8` total), the thread loads `sizeSMEM / numThreads` elements per matrix.
- `sizeSMEM` per matrix = BM×BK = 128×8 = 1024 floats. With 256 threads: `1024/256 = 4` loads per matrix.
- Total: `K/8 × 2 × 4 = K` loads per thread → **K/64 per result**.

**SMEM accesses per thread:**
- Each `dotIdx` step (BK=8 per outer iteration): load TM=8 from `As` + TN=8 from `Bs` = 16 SMEM loads.
- Per outer iteration: `8 × 16 = 128` SMEM loads.
- Total: `K/8 × 128 = 16K` loads per thread → **K/4 per result**.

##### **What's Next**

Performance is reaching acceptable levels, but warp stalls from memory pipeline congestion are still too frequent. Kernel 6 addresses this with two measures: **transposing `As`** in SMEM to enable vectorized 128-bit SMEM loads (`LDS.128`), and **promising the compiler alignment** on GMEM accesses to enable wider GMEM transactions.

---
