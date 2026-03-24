---
layout: post
title: GEMM Kernel Optimizations
date: 2026-03-23 16:29:00-0400
featured: false
description: Building intuition for GPU matrix multiplication kernel optimizations, from naive to warp-tiled
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
---

#### **Goal**

This post walks through the standard sequence of optimizations applied to dense matrix multiplication (GEMM) kernels on NVIDIA GPUs. Starting from a naive implementation, each section identifies the current bottleneck, introduces the idea that addresses it, and shows the resulting pseudocode. We use a concrete running example — a $$16 \times 16$$ matrix multiply with $$4 \times 4$$ thread blocks — and scale up when a particular optimization requires it.

The primary references are: [Learn CUTLASS the Hard Way](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) (Kapil Sharma), [CUDA Matrix Multiplication](https://siboehm.com/articles/22/CUDA-MMM) (Simon Boehm), and [Matrix Multiplication](https://www.aleksagordic.com/blog/matmul) (Aleksa Gordic).

#### **Why GEMM**

Every linear layer in a Transformer is a matrix multiplication. In a [previous post]({{ '/blog/2026/transformer-block-accounting/' | relative_url }}), we derived that a single Transformer block's forward pass requires $$8 \cdot B \cdot S \cdot D^2 + 6 \cdot B \cdot S \cdot D \cdot F$$ FLOPs, nearly all of which arise from GEMMs: the $$Q, K, V, O$$ projections, the attention scores $$QK^T$$, the weighted sum with $$V$$, and the three FFN projections. The performance of the model is, to a close approximation, the performance of its GEMMs.

#### **Problem Setup**

We compute $$C = A \times B$$ where $$A$$ is $$[M \times K]$$, $$B$$ is $$[K \times N]$$, and $$C$$ is $$[M \times N]$$. Each element of $$C$$ is a dot product:

$$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$$

Throughout this post, we use $$M = N = K = 16$$ with a block size of $$4 \times 4$$ (16 threads per block, 16 blocks in the grid), unless stated otherwise.

#### **GPU Memory Hierarchy**

The central constraint in GPU kernel optimization is the disparity between compute throughput and memory bandwidth. The relevant levels of the memory hierarchy are:

| Level | Scope | Typical Size | Approximate Latency |
|-------|-------|----------------|---------|
| Global Memory (HBM) | All SMs | 16–80 GB | 400–600 cycles |
| L2 Cache | All SMs | 4–50 MB | ~200 cycles |
| Shared Memory (SMEM) | Per SM | 48–228 KB | 20–30 cycles |
| Registers | Per thread | 256 KB per SM | 1 cycle |

Every optimization in this post can be understood as moving data closer to compute (from global memory toward registers) and increasing the ratio of arithmetic operations to bytes transferred. This ratio — the **arithmetic intensity** — determines whether a kernel is compute-bound or memory-bound.

#### **Kernel Roadmap**

| # | Kernel | Key Idea |
|---|--------|----------|
| 1 | Naive | One thread computes one $$C[i][j]$$. Baseline. |
| 2 | Global Memory Coalescing | Reindex threads so adjacent threads access adjacent memory addresses. |
| 3 | Shared Memory Tiling | Load tiles of $$A$$ and $$B$$ into SMEM; reuse across threads in a block. |
| 4 | 1D Blocktiling | Each thread computes a column/row of results, increasing register reuse. |
| 5 | 2D Blocktiling | Each thread computes a $$T_M \times T_N$$ sub-tile of $$C$$ in registers. |
| 6 | Vectorized Loads | Use `float4` loads (128 bits per transaction) to saturate bus width. |
| 7 | Warptiling | Structure work at the warp level to maximize data reuse within a warp. |

Each section follows the same structure: (1) what bottleneck we are solving, (2) the idea with the running example, (3) pseudocode, and (4) what changed in terms of arithmetic intensity and memory traffic.

---

#### **Kernel 1: Naive**

##### **What bottleneck are we solving?**

This kernel establishes the baseline. There is no specific bottleneck being targeted; the goal is a correct, minimal implementation that makes the subsequent bottlenecks concrete.

##### **CUDA Indexing Convention**

A CUDA kernel launches a grid of thread blocks. Each thread is identified by two levels of coordinates: `blockIdx.x/y` (which block in the grid) and `threadIdx.x/y` (which thread within the block). The convention is that `x` corresponds to the horizontal (column) dimension and `y` to the vertical (row) dimension. This is the reverse of the standard matrix convention where the first index denotes the row. The mapping to matrix coordinates is:

$$\text{row } i = \texttt{blockIdx.y} \cdot \texttt{blockDim.y} + \texttt{threadIdx.y}$$

$$\text{col } j = \texttt{blockIdx.x} \cdot \texttt{blockDim.x} + \texttt{threadIdx.x}$$

The reason this convention matters is that threads within a block are linearized as $$\texttt{linear_id} = \texttt{threadIdx.y} \cdot \texttt{blockDim.x} + \texttt{threadIdx.x}$$, so `threadIdx.x` varies fastest. Adjacent values of `threadIdx.x` correspond to adjacent linear thread IDs. This becomes directly relevant in Kernel 2 when we discuss memory coalescing.

##### **The Idea**

One thread computes one element of $$C$$. The thread at global position $$(i, j)$$ reads row $$i$$ of $$A$$ and column $$j$$ of $$B$$, accumulates the dot product over the $$k$$-dimension, and writes the result to $$C[i][j]$$.

For the $$16 \times 16$$ example, the grid contains $$4 \times 4 = 16$$ blocks of $$4 \times 4 = 16$$ threads each, giving 256 threads total — one per element of $$C$$.

<div class="row justify-content-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/gemm/k1_thread_mapping.png" title="naive-thread-mapping" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Each color represents a distinct thread block. The label $$(t_y, t_x)$$ inside each cell denotes `(threadIdx.y, threadIdx.x)` of the thread responsible for that output element. Block labels are `(blockIdx.y, blockIdx.x)`.

##### **Pseudocode**
```
kernel naive_gemm(A[M][K], B[K][N], C[M][N]):
    row = blockIdx.y * blockDim.y + threadIdx.y
    col = blockIdx.x * blockDim.x + threadIdx.x

    accumulator = 0.0
    for k in 0 .. K-1:
        accumulator += A[row][k] * B[k][col]

    C[row][col] = accumulator

// Launch configuration:
blockDim = (4, 4)
gridDim  = (N/4, M/4)      // gridDim.x = columns, gridDim.y = rows
```

##### **Memory Access Pattern**

Consider the thread computing $$C[6][5]$$, which resides in Block$$(1,1)$$ with `threadIdx.y=2`, `threadIdx.x=1`. At each iteration of the $$k$$-loop, this thread reads one element from row 6 of $$A$$ and one element from column 5 of $$B$$.

In row-major layout, element $$(i, j)$$ of a matrix with $$N_{\text{cols}}$$ columns is stored at address $$i \cdot N_{\text{cols}} + j$$. The access pattern for this thread is therefore:

| Access | Address at $$k=0$$ | Address at $$k=1$$ | Stride |
|--------|-------|-------|--------|
| $$A[6][k]$$ | $$6 \cdot 16 + 0 = 96$$ | $$6 \cdot 16 + 1 = 97$$ | 1 (contiguous) |
| $$B[k][5]$$ | $$0 \cdot 16 + 5 = 5$$ | $$1 \cdot 16 + 5 = 21$$ | 16 (strided) |

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/gemm/k1_memory_access.png" title="naive-memory-access" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Reads from $$A$$ are contiguous, but reads from $$B$$ have stride $$N = 16$$. This strided access is one source of inefficiency, but it is not the dominant problem with this kernel.

##### **Redundant Global Memory Traffic**

The more significant issue is redundant loads. Consider the 16 threads in Block$$(0,0)$$. All four threads in the same row (e.g., those computing $$C[0][0]$$ through $$C[0][3]$$) read the entire row 0 of $$A$$ independently. Similarly, all four threads in the same column read the same column of $$B$$.

Per thread, there are $$2K = 32$$ global memory reads for $$K = 16$$ multiply-add operations. Per block, this amounts to $$16 \times 32 = 512$$ global reads, while the unique data required is only $$4 \times 16 + 16 \times 4 = 128$$ elements (4 rows of $$A$$ and 4 columns of $$B$$). Each element is loaded 4 times more than necessary.

This redundancy grows across the full grid. Row 0 of $$A$$ is read by every block in block-row 0 — four blocks, each with four threads reading the same row, yielding 16 independent loads of the same data from global memory.

Kernel 3 (Shared Memory Tiling) will eliminate within-block redundancy by loading shared data once into SMEM. Before that, Kernel 2 addresses the coalescing problem: ensuring that even redundant reads are issued in a pattern that the memory hardware can service efficiently.

##### **Arithmetic Intensity**

The arithmetic intensity of the naive kernel is:

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes loaded from global memory}}$$

The total work for the $$16 \times 16$$ GEMM is $$2 \cdot 16^3 = 8192$$ FLOPs. Each of the 256 threads loads $$2 \times 16 = 32$$ float32 values (128 bytes), giving total traffic of $$256 \times 128 = 32768$$ bytes.

$$\text{AI}_{\text{naive}} = \frac{8192}{32768} = 0.25 \text{ FLOPs/byte}$$

On an A100, the ratio of peak compute (19500 GFLOPS FP32) to peak memory bandwidth (2039 GB/s) gives a ridge point of approximately 9.6 FLOPs/byte. At 0.25 FLOPs/byte, this kernel is deeply memory-bound. The sequence of optimizations that follows is concerned with increasing this ratio.

---

#### **Kernel 2: Global Memory Coalescing**

##### **What bottleneck are we solving?**

In the naive kernel, each thread independently loads data from global memory (HBM). However, GPUs do not issue memory requests per-thread. Instead, threads in a **warp** (a group of 32 threads that execute in lockstep) have their memory requests combined at the hardware level. When consecutive threads in a warp access consecutive memory addresses, the hardware can service these as a single **memory transaction** (typically 32 or 128 bytes). This is called **coalescing**. When the addresses are scattered, each thread may require a separate transaction, reducing effective bandwidth by a factor proportional to the number of distinct cache lines touched.

##### **How Coalescing Works**

Threads within a block are grouped into warps based on their **linear thread ID**:

$$\texttt{linear\_id} = \texttt{threadIdx.y} \cdot \texttt{blockDim.x} + \texttt{threadIdx.x}$$

Threads 0–31 form warp 0, threads 32–63 form warp 1, and so on. Within a warp, thread $$n$$ and thread $$n+1$$ are adjacent. For coalescing, we need adjacent threads to access adjacent memory addresses.

In row-major layout, elements in the same row are contiguous in memory: element $$(i, j)$$ is at address $$i \cdot N + j$$. Therefore, stepping in the column dimension ($$j$$) gives stride 1, while stepping in the row dimension ($$i$$) gives stride $$N$$.

The coalescing requirement reduces to: **the fastest-varying thread index (`threadIdx.x`) must map to the fastest-varying memory dimension (columns, in row-major layout).**

##### **Two Possible Mappings**

There are two natural ways to assign thread indices to matrix coordinates:

**Mapping A** (coalesced): `threadIdx.x → col`, `threadIdx.y → row`

$$\text{row} = \texttt{blockIdx.y} \cdot \texttt{blockDim.y} + \texttt{threadIdx.y}, \quad \text{col} = \texttt{blockIdx.x} \cdot \texttt{blockDim.x} + \texttt{threadIdx.x}$$

**Mapping B** (non-coalesced): `threadIdx.x → row`, `threadIdx.y → col`

$$\text{row} = \texttt{blockIdx.x} \cdot \texttt{blockDim.x} + \texttt{threadIdx.x}, \quad \text{col} = \texttt{blockIdx.y} \cdot \texttt{blockDim.y} + \texttt{threadIdx.y}$$

Consider threads T0–T3 (the first four consecutive threads in a warp) within Block$$(0,0)$$ of our $$16 \times 16$$ example, at loop iteration $$k = 0$$:

| | Mapping A (coalesced) | Mapping B (non-coalesced) |
|---|---|---|
| Thread coordinates | T0:(row=0,col=0), T1:(row=0,col=1), T2:(row=0,col=2), T3:(row=0,col=3) | T0:(row=0,col=0), T1:(row=1,col=0), T2:(row=2,col=0), T3:(row=3,col=0) |
| B[k][col] addresses | 0, 1, 2, 3 (stride 1) | 0, 0, 0, 0 (broadcast) |
| A[row][k] addresses | 0, 0, 0, 0 (broadcast) | 0, 16, 32, 48 (stride 16) |
| C[row][col] addresses | 0, 1, 2, 3 (stride 1) | 0, 16, 32, 48 (stride 16) |

Under Mapping A, reads from $$B$$ and writes to $$C$$ are coalesced (stride 1). Under Mapping B, reads from $$A$$ and writes to $$C$$ have stride 16, meaning each access lands in a different cache line — up to 4× more memory transactions for the same data.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/gemm/k2_coalescing.png" title="coalescing-comparison" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The diagram shows the first four rows of linearized memory (addresses 0–63). In the coalesced case (top), threads T0–T3 land on consecutive addresses within row 0. In the non-coalesced case (bottom), threads T0–T3 land on the first element of four different rows, spaced 16 addresses apart.

##### **Pseudocode**

The kernel code is identical to Kernel 1. The only requirement is that the indexing convention uses Mapping A:
```
kernel coalesced_gemm(A[M][K], B[K][N], C[M][N]):
    row = blockIdx.y * blockDim.y + threadIdx.y   // threadIdx.y → row (slow index)
    col = blockIdx.x * blockDim.x + threadIdx.x   // threadIdx.x → col (fast index)

    accumulator = 0.0
    for k in 0 .. K-1:
        accumulator += A[row][k] * B[k][col]

    C[row][col] = accumulator
```

Our Kernel 1 already used this mapping, so the pseudocode is unchanged. The contribution of this section is the understanding of *why* this assignment is correct: it ensures that the fastest-varying thread dimension aligns with the fastest-varying memory dimension, enabling the hardware to coalesce accesses into minimal transactions.

##### **What Changed**

The arithmetic intensity is unchanged — we are still loading the same total number of bytes and performing the same FLOPs. What changes is the **effective memory bandwidth**: coalesced accesses utilize the bus efficiently, while non-coalesced accesses waste bandwidth on partially-filled cache line transfers.

In practice, the difference between Mapping A and Mapping B on the same naive kernel can be 2–4× in throughput, depending on matrix size and GPU architecture. But even with perfect coalescing, the fundamental problem from Kernel 1 remains: massive redundancy in global memory loads. Threads in the same block load overlapping rows of $$A$$ and columns of $$B$$ independently. The next kernel addresses this by introducing shared memory.

---

#### **Kernel 3: Shared Memory Tiling**

##### **What bottleneck are we solving?**

In Kernels 1 and 2, every thread independently loads its required elements from global memory. As established in Kernel 1's analysis, threads within the same block share significant data overlap: all threads in the same row of a block read the same row of $$A$$, and all threads in the same column read the same column of $$B$$. For Block$$(0,0)$$ with 16 threads and $$K=16$$, this results in 512 global memory reads when only 128 unique elements are needed — a 4× redundancy.

The idea in this kernel is to eliminate this within-block redundancy by using **shared memory** (SMEM), an on-chip memory space visible to all threads in a block, with approximately 20–30 cycle latency compared to 400–600 cycles for global memory.

##### **Tiling the K-Dimension**

The dot product $$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \cdot B[k][j]$$ can be decomposed into partial sums over tiles of size `TILE` along the $$k$$-dimension:

$$C[i][j] = \sum_{t=0}^{K/\text{TILE} - 1} \sum_{k'=0}^{\text{TILE}-1} A[i][t \cdot \text{TILE} + k'] \cdot B[t \cdot \text{TILE} + k'][j]$$

At each tile iteration $$t$$, a block needs a sub-matrix of $$A$$ of shape $$[\text{BLOCK} \times \text{TILE}]$$ and a sub-matrix of $$B$$ of shape $$[\text{TILE} \times \text{BLOCK}]$$. The block loads both sub-matrices **cooperatively** into shared memory, synchronizes, and then every thread computes its partial sum by reading from SMEM rather than global memory.

In our running example ($$M = N = K = 16$$, `BLOCK = TILE = 4`), Block$$(0,0)$$ computes $$C[0\!:\!4,\; 0\!:\!4]$$. The computation decomposes into $$K / \text{TILE} = 4$$ iterations. At iteration $$t$$, the block loads $$A[0\!:\!4,\; 4t\!:\!4(t\!+\!1)]$$ and $$B[4t\!:\!4(t\!+\!1),\; 0\!:\!4]$$ into two $$4 \times 4$$ shared memory arrays.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/gemm/k3_tiling_overview.png" title="smem-tiling-overview" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The colored regions in $$A$$ and $$B$$ correspond to the four tile iterations. At each iteration, one $$4 \times 4$$ tile from each matrix is loaded into SMEM. The accumulated product of all four tile-pairs yields the final $$4 \times 4$$ block of $$C$$.

##### **Cooperative Loading**

With a $$4 \times 4$$ block (16 threads) and a $$4 \times 4$$ tile (16 elements), each thread loads exactly one element into each of the two shared memory arrays:

$$\texttt{As}[\texttt{ty}][\texttt{tx}] = A[\text{row\_base} + \texttt{ty}][\; t \cdot \text{TILE} + \texttt{tx}]$$

$$\texttt{Bs}[\texttt{ty}][\texttt{tx}] = B[\; t \cdot \text{TILE} + \texttt{ty}][\text{col\_base} + \texttt{tx}]$$

where `ty = threadIdx.y` and `tx = threadIdx.x`. After loading, all threads execute a barrier synchronization (`__syncthreads()`) to ensure the shared memory is fully populated before any thread reads from it.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/gemm/k3_smem_detail.png" title="smem-loading-reuse" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The left two panels show which thread loads which element into `As` and `Bs`. The right panel illustrates the reuse: thread $$(1, 2)$$ computing $$C[1][2]$$ reads row 1 of `As` and column 2 of `Bs` — four elements from each, all served from SMEM at ~20 cycle latency instead of global memory.

##### **Synchronization**

Two barriers are required per tile iteration. The first, after the cooperative load, ensures all elements are in SMEM before any thread begins computing. The second, after the compute phase and before the next iteration's load, ensures no thread overwrites SMEM with the next tile's data while another thread is still reading the current tile. Without this second barrier, a fast thread could begin loading tile $$t+1$$ into `As`/`Bs` while a slow thread is still reading tile $$t$$, producing incorrect results.

##### **Pseudocode**
```
kernel smem_gemm(A[M][K], B[K][N], C[M][N]):
    ty = threadIdx.y
    tx = threadIdx.x
    row = blockIdx.y * BLOCK + ty
    col = blockIdx.x * BLOCK + tx

    __shared__ As[BLOCK][TILE]
    __shared__ Bs[TILE][BLOCK]

    accumulator = 0.0
    for t in 0 .. (K / TILE - 1):
        // Cooperative load: each thread loads one element into each tile
        As[ty][tx] = A[row][t * TILE + tx]
        Bs[ty][tx] = B[t * TILE + ty][col]
        __syncthreads()

        // Compute partial sum from SMEM
        for k in 0 .. TILE-1:
            accumulator += As[ty][k] * Bs[k][tx]
        __syncthreads()

    C[row][col] = accumulator

// Launch configuration:
BLOCK = 4, TILE = 4
blockDim = (BLOCK, BLOCK)
gridDim  = (N / BLOCK, M / BLOCK)
```

##### **Memory Traffic Analysis**

Consider Block$$(0,0)$$. At each tile iteration, the block issues $$2 \times \text{BLOCK} \times \text{TILE} = 2 \times 16 = 32$$ global memory reads (16 for `As`, 16 for `Bs`). Over $$K / \text{TILE} = 4$$ iterations, the total is $$4 \times 32 = 128$$ global reads. In the naive kernel, the same block issued 512 global reads. The reduction factor is exactly `BLOCK` (= 4): each element loaded into SMEM is reused by `BLOCK` threads instead of being independently loaded by each.

More generally, for a block of size $$\text{BLOCK} \times \text{BLOCK}$$ computing $$\text{BLOCK}^2$$ outputs, the global loads per block scale as $$2 \cdot \text{BLOCK} \cdot K$$ (one row-strip of $$A$$ and one column-strip of $$B$$), compared to $$\text{BLOCK}^2 \cdot 2K$$ in the naive kernel. The ratio is $$\text{BLOCK}$$: increasing block size directly reduces global memory traffic.

##### **Arithmetic Intensity**

Total FLOPs remain $$2 \cdot 16^3 = 8192$$. Total global loads are now $$16 \text{ blocks} \times 128 \text{ reads/block} = 2048$$ float32 values $$= 8192$$ bytes.

$$\text{AI}_{\text{smem}} = \frac{8192}{8192} = 1.0 \text{ FLOPs/byte}$$

This is a 4× improvement over the naive kernel's 0.25 FLOPs/byte, which matches the reduction factor of `BLOCK = 4`. For larger block sizes the improvement is proportionally greater: a $$32 \times 32$$ block yields a 32× reduction in global memory traffic per block.

However, even at 1.0 FLOPs/byte, the kernel remains well below the A100's compute-to-bandwidth ridge point of ~9.6 FLOPs/byte. The bottleneck is now that each thread still computes only a single output element, meaning the data loaded into SMEM per thread is not fully amortized. Kernel 4 addresses this by having each thread compute multiple output elements, increasing the compute-to-load ratio at the register level.

---

