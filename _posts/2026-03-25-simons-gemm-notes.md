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

#### **Intro**

These are my notes from Simon Boehm's excellent [CUDA GEMM kernel optimization blog](https://siboehm.com/articles/22/CUDA-MMM). First and foremost, Simon really did a great job explaining the kernels and helping me internalize the intuition — hats off for the time and effort he put into it.

These notes exist as a reference to firmly hold these ideas and recall them later. Writing something in your own words takes effort, but pays off well down the line. I go through all the kernels one by one, add notes, and build a mental model. I refer a lot to Simon's blog for all the great drawings he made, and highly recommend you check that out first if possible.

A few housekeeping notes:

- I used Claude for rephrasing my own words — the understanding is mine, the polish is AI-assisted.
- This is a work in progress — I'll be adding one kernel at a time.
- As you read, try to **mentally visualize** the thread mappings, memory accesses, and data flow. It really helps solidify the intuition.

That's it on the intro — let's get in.

#### **Prerequisites**

I'm assuming readers are comfortable with CUDA basics — I won't go deep into them, but will touch upon what's needed briefly. If you need a refresher, here are my notes for reference: [GPU Programming Intro](link-to-your-post).

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

> We'll track this **thread → element mapping** for every kernel going forward — it's the most critical thing to get right, as it directly determines memory access patterns and coalescing behavior.

Assuming zero caching, each thread loads `2 × 4092 + 1` floats from GMEM. With 4092² threads total, that's ~548 GB of memory traffic — far above the 268 MB minimum.

> **Errata in Simon's blog:** Simon writes two example threads as (0, 0) and (0, 1), and describes them as loading "the same column of B but different rows of A." But with his mapping (`x` from `threadIdx.x` = row, `y` from `threadIdx.y` = column), threads (0, 0) and (0, 1) share the same row of A and access different columns of B. For the description and diagram to be consistent, the second thread should be **(1, 0)**, not (0, 1). **[TODO: Confirm with Simon and update.]**

The naive kernel achieves ~300 GFLOPs on the A6000 — just 1% of the theoretical 30 TFLOPs.

So how do we make this faster? By optimizing memory access patterns so that global memory accesses can be **coalesced** (combined) into fewer transactions.

#### **Kernel 2: Global Memory Coalescing**

##### **Warps and Thread Grouping**

Before we dive in, let's formalize the concept of a **warp**. A warp is a hardware-level grouping of 32 threads within a block. All threads in a warp are issued the same instruction and executed by one of the **4 warp schedulers per SM**. This execution model is called **SIMT** (Single Instruction, Multiple Threads). It's similar to SIMD, but with a key difference: in SIMT, threads *can* diverge (take different branches), though divergence is expensive since the warp serializes the divergent paths. When all threads follow the same path, it's efficient — that's the happy case.

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

The rest of the kernel stays identical:
```cuda
if (x < M && y < N) {
  float tmp = 0.0;
  for (int i = 0; i < K; ++i) {
    tmp += A[x * K + i] * B[i * N + y];
  }
  C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}
```

##### **Results**

Just by changing the thread-to-element mapping, GMEM throughput jumps from **15 GB/s to 110 GB/s**. Performance goes from **~300 GFLOPs to ~2000 GFLOPs** — a ~6.5× improvement from a two-line code change.

We're still far from the 30 TFLOPs peak though. The next step: use the GPU's fast on-chip memory — **shared memory (SMEM)** — to cache data that gets reused, reducing the number of expensive GMEM accesses.

---


- note 8: In our example, threadIdx.x and threadIdx.y will vary from 0 to 31 based on the position of the thread in the grid. The last word should be block.
- img adjacent to note 19 seems off. the threadidx.z should be all zeros for the first 4*16 elements.
- in "Memory Access Pattern of the Naive Kernel", the ThreadId (0, 1), points to (1, 0) for the block, only then the image will be appropriate.
- note 23 - Also, it’s possible to use more than 48KB of SMEM per thread by utilizing dynamic shared memory. should be per block??
- smem loads are mentioned as 32b, 64b & 128b. is this b, bits/bytes. i think its bits, check.
- in kernel 5, why do we chunk the smem cache at this step: // populate the SMEM caches. in kernel 4, we're not doing this? got it, so its to make each thread load multiple elements, rather than just 1, as in kernel 4.
- in kernel 5, how we got GMEM loads calculation? specifically this part: (sizeSMEM/numThreads)
- note 49 - "as each thread in the block issues a 4-wide load during each iteration of the GMEM to SMEM loading loop" ??
- in warptiling, what is this: "There’s a register cache on recent GPUs, and tighter threadtiling gives us more register cache locality." ?? 
- in warptiling: should this be threadId and not threadIdx.x - "warpId=threadIdx.x % warpSize" ?? and shouldn't it be "/", with "%" even adjacent threads will be different warps right? this is my understanding, pls correct if i'm wrong
- note 55, link broken
- what is this part of the code: "// execute warptile matmul. Later this will map well to // warp-wide matrix instructions, executed on tensor cores."
- mental picture of warptiling is still not clear
- warp tiling gives alignment, so a group of threads exucute same instruction in 4 warp schedulers and context switch faster, thus making them a bit faster. right intuition??
- also, what relation to ILP in warptiling
- thread swizzling ??
- "It further means that all computation that happens inside the BK loop will be independent and can be parallelized (for example using ILP).", here is it because that addition is commutative and associative. and no need to have atomics, as order isn't important??
- kernel 11, cutlass & smem data layout link broken
- 


- a clear picture of roofline analysis and what it conveys and how to interpret
  

