---
layout: post
title: CUTLASS WGMMA on Hopper - Notes
date: 2026-04-18 12:30:00-0400
featured: false
description: My notes on WGMMA internals from the Colfax Research CUTLASS Hopper GEMM blog
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
---

#### **PreIntro**

These are my notes on WGMMA (Warp Group Matrix Multiply-Accumulate) on Hopper GPUs, primarily based on the Colfax Research blog [*CUTLASS Hopper GEMM — WGMMA*](https://research.colfax-intl.com/cutlass-hopper-gemm-wgmma/). Colfax did a fantastic job going deep into the internals of how CUTLASS maps tiled GEMMs to the `wgmma.mma_async` instruction — the level of detail is rare and much appreciated.

These notes exist as a reference to firmly hold these ideas and recall them later. Writing something in your own words takes effort, but pays off well down the line. I go through all the key concepts — the `TiledMMA` object, SMEM layout constraints, fragments and descriptors, the `cute::gemm` dispatch, synchronization, and core matrices — adding notes and building a mental model. I refer a lot to the Colfax blog for its excellent diagrams and worked examples, and highly recommend you check that out first if possible.

A few housekeeping notes:

- I used Claude for rephrasing my own words — the understanding is mine, the polish is AI-assisted.
- As you read, try to **mentally visualize** the core matrix layouts, descriptor addressing, thread-to-register mappings, and the async pipeline. It really helps solidify the intuition.

#### **Async WGMMA**

WGMMA stands for Warp Group Matrix Multiply-Accumulate. It is a primitive instruction used to dispatch tiled matrix multiplications to the Tensor Cores on Hopper (SM90) architecture GPUs. If you are already familiar with tiled GEMM, then the mental model is straightforward: we take the small tile-level matmuls and hand them off to Tensor Cores via this instruction. The full PTX mnemonic is `wgmma.mma_async`, but we will refer to it simply as WGMMA from here on.

In the Colfax blog, they lay out a "big picture" at the start of what necessitates the creation of efficient GEMM kernels — memory hierarchy exploitation, compute–memory overlap, and so on. Of those pieces, here we focus on WGMMA alone, as it is the core compute primitive that everything else is built around. The flow of this post is adapted from that Colfax blog, with additional notes to make the concepts easier to grasp and build up intuition.

Tensor Cores and the older `wmma` API existed before Hopper, but WGMMA introduces a critical difference: **asynchrony**. When you issue a WGMMA, you do not block and wait for the result. Instead, you submit work to the Tensor Cores, continue executing other instructions (e.g., issuing the next TMA load), and later synchronize to collect the results. This overlap of compute and data movement is central to why Hopper GEMMs achieve high utilization.

The **WG** in WGMMA stands for Warp Group. A warp group is a group of 4 contiguous warps, i.e., 128 contiguous threads. All threads in the warp group execute the WGMMA collectively. The operation it performs is:

$$C = A \times B + C$$

or, with accumulation disabled:

$$C = A \times B$$

If you are familiar with the classical BLAS form $$C = \alpha (A \times B) + \beta C$$, then in WGMMA both $$\alpha$$ and $$\beta$$ are fixed to 1.

##### **Operand Storage Requirements**

WGMMA imposes strict rules on where its operands must reside:

- **Operand B** must always be in shared memory (SMEM).
- **Operand A** can be in either SMEM or register memory (RMEM).
- **Accumulator C** is always held in RMEM.

> **TODO:** Explain *why* these placement constraints exist — ties into the Tensor Core datapath and the descriptor-based SMEM access model. Will address after covering matrix descriptors.

##### **Warp Rank and Warp Group Indexing**

A quick bit of terminology. The **warp rank** is the index of a warp within its CTA (thread block). For a given thread, it is simply $$\text{threadIdx} / 32$$. For warp groups, the first warp in each group always has a warp rank that is a multiple of four (i.e., 0, 4, 8, …). This convention exists for simple bookkeeping: you can obtain the warp group ID by masking off the lower 2 bits of the warp rank, which the hardware and CUTLASS both exploit for fixed-rule alignment.

##### **Roadmap**

With the background in place, we will go deep into the internals. The plan is:

1. **Invoking WGMMA in CUTLASS** — constructing the `TiledMMA` object, creating and partitioning SMEM tensors to be compatible with WGMMA.
2. **Handling asynchrony** — the synchronization mechanisms needed to ensure correctness when WGMMA executes out of step with the rest of the kernel.
3. **Layouts and descriptors** — the core matrix concept, the SMEM layouts WGMMA expects, and the matrix descriptors used when operands are sourced from SMEM.

---

#### **WGMMA Inside a CUTLASS Kernel**

To see where WGMMA fits, let us start from the standard tiled GEMM structure. We have input matrices $$A$$ ($$M \times K$$) and $$B$$ ($$K \times N$$), and we want to compute $$C = A \times B$$. The kernel fixes static tile sizes $$bM$$, $$bN$$, and $$bK$$, then launches a grid of $$\lceil M/bM \rceil \times \lceil N/bN \rceil$$ CTAs. Each CTA is responsible for computing one $$bM \times bN$$ output tile $$rC$$, which it holds in RMEM before writing back to global memory.

Inside each CTA, the **mainloop** iterates over $$\lceil K/bK \rceil$$ steps along the inner dimension. At each step, $$bM \times bK$$ and $$bN \times bK$$ tiles of $$A$$ and $$B$$ are loaded from global memory into shared memory as `sA` and `sB`. The `cute::gemm` call then computes the product of (stage-wise slices of) `sA` and `sB` and accumulates into `rC`. After the mainloop, the epilogue writes `rC` out to global memory.

What we are focusing on here is: when this `cute::gemm` call is made, how does it internally map to the WGMMA instruction primitive to exploit Tensor Cores on Hopper?

##### **Why B Is Stored Transposed**

Mathematically, a $$B$$-tile is $$bK \times bN$$. But CUTLASS stores `sB` as its transpose, so in SMEM the tile is laid out as $$bN \times bK$$. This makes `sA` ($$bM \times bK$$) and `sB` ($$bN \times bK$$) share the same shape pattern — $$K$$ lives in a predictable position for both operands, and the `cute::gemm` interface can treat them uniformly.

##### **Pipelined SMEM Buffers (Stages)**

There is a fundamental speed mismatch: TMA loads from global memory are slow relative to how fast WGMMA consumes data from SMEM. If you naively load one tile, compute on it, then load the next, the Tensor Cores stall on memory for most of their time.

The fix is to allocate $$P$$ SMEM slots — called **stages** (typically 2 or 3) — and keep the pipeline full. For $$k$$-tile iteration $$i$$, you write into slot $$i \bmod P$$, overwriting old data only once the consumer (WGMMA) has signaled it is done with that slot. The slots form a **circular buffer** that producer (TMA) and consumer (WGMMA) walk around at their own paces, synchronized via barriers. While WGMMA computes on slot $$i$$, TMA is already filling slots $$i+1$$, $$i+2$$, and so on.

This pattern is not a CUTLASS invention — it is the standard way GPU GEMMs have been written for years. Since the `cp.async` era on Ampere (and conceptually even earlier with manual double-buffering), serious GEMM kernels always overlap global-to-shared memory loads with tensor-core compute. The design here follows that well-established practice.

The stage count $$P$$ is a **compile-time integer** (e.g., `auto bP = Int<3>{};`) because CuTe needs it baked into the SMEM layout at compile time to statically size `SharedStorage` and let the compiler schedule accordingly.

In CuTe's type system, the circular buffer is represented as an extra outermost mode in the shape tuple. For a shape `(bM, bK, bP)`, mode-0 is $$bM$$, mode-1 is $$bK$$, and mode-2 is $$bP$$ (the stage count). So `sA` is not $$P$$ separate tensors — it is one 3D tensor where `sA(_, _, 0)` selects stage 0's tile, `sA(_, _, 1)` selects stage 1's, and so on.

> **TODO:** Add code showing `SharedStorage` definition with staged `sA`/`sB` tensors.

##### **The Four Key Pieces for `cute::gemm` with WGMMA**

When `cute::gemm` dispatches to WGMMA under the hood, four things must be set up correctly:

1. **`TiledMMA` object** — encapsulates everything `cute::gemm` needs to select and emit the correct `wgmma` PTX instruction.
2. **SMEM tensor layouts** — `sA` and `sB` must be laid out in shared memory in a way that is compatible with WGMMA's access pattern.
3. **Thread-level fragments** (`tCrA`, `tCrB`, `tCrC`) — constructed as partitions of the data using the `TiledMMA` object. Their layouts are WGMMA-specific and important to understand.
4. **Matrix descriptors** — `tCrA` (when A is sourced from SMEM) and `tCrB` are *not* register-backed tensors with values copied from SMEM. They are **matrix descriptors** constructed on top of SMEM, telling the hardware where to find the data.

Surrounding the `cute::gemm` call, warp-group synchronization primitives manage the asynchronous execution. We will cover each of these four pieces in depth in the following sections.

---

#### **TiledMMA Object for WGMMA**

Before diving into how the `TiledMMA` object is constructed, we need to establish two pieces of notation that appear throughout the WGMMA literature: **majorness** and **BLAS transpose flags**.

##### **Majorness: MN-Major vs K-Major**

A 2D matrix is stored in 1D memory, so one axis is contiguous (stride 1) and the other has a larger stride. "Majorness" names which logical axis is the stride-1 one. In GEMM, every operand touches two of the three logical dimensions $$\{M, N, K\}$$: $$A$$ is $$M \times K$$, $$B$$ is $$K \times N$$, $$C$$ is $$M \times N$$. CuTe names majorness by which of these logical dimensions is contiguous in memory:

- **K-major** — the $$K$$ axis is contiguous. Row-major $$A$$ is K-major (rows run along $$K$$). Column-major $$B$$ is also K-major (columns run along $$K$$).
- **MN-major** — the non-$$K$$ axis is contiguous. Column-major $$A$$ is M-major (hence MN-major). Row-major $$B$$ is N-major (hence MN-major).

The reason for grouping "not-K" under a single name "MN" is that $$K$$ is the reduction axis shared by $$A$$ and $$B$$, while $$M$$ and $$N$$ are the outer axes. The WGMMA instruction only cares about one distinction: is the contraction direction ($$K$$) the contiguous one, or is the outer direction? Calling both operands "MN-major" or "K-major" lets a single term describe them uniformly, rather than saying "M-major for A, N-major for B."

##### **BLAS Transpose Notation (NT, TN, NN, TT)**

BLAS GEMM is written `gemm(transA, transB, …)` where each flag is `N` (no transpose) or `T` (transpose). The convention assumes column-major storage, and the flags say what gets transposed before the multiply. The two-letter name $$XY$$ stands for `(transA, transB)`:

- **NT** — $$A$$ not transposed, $$B$$ transposed. Under column-major storage, both operands end up **MN-major**.
- **TN** — $$A$$ transposed, $$B$$ not. Both operands end up **K-major**.
- **NN** — $$A$$ is MN-major, $$B$$ is K-major (mixed).
- **TT** — $$A$$ is K-major, $$B$$ is MN-major (mixed the other way).

The clean cases for WGMMA are **NT** and **TN** because both operands share the same majorness, which is exactly why the reference code has `gemm_nt` and `gemm_tn` as its two setup paths, and why the blog's "A and B are MN-major" corresponds to an NT GEMM.

One caveat: for non-16-bit datatypes (FP8, INT8, TF32, etc.), WGMMA only supports K-major operands, so only **TN** works natively. For FP16/BF16 you get the flexibility of both NT and TN.

##### **The MMA Atom as a PTX Wrapper**

The `TiledMMA` object is built on the host from an **MMA Atom**, which is a CUTLASS struct that directly wraps a single `wgmma.mma_async` PTX instruction. The naming scheme lets you read the PTX call straight off the atom's name: `SM90_MxNxK_XYZ_SS<Major, Major>` corresponds to `wgmma.mma_async.sync.aligned.mMnNkK.z.x.y`. Concretely:

- **SM90** — Hopper architecture.
- **MxNxK** — the tile shape the instruction computes (the "wgmma atom" tile).
- **X, Y** — datatypes of operand $$A$$ and $$B$$.
- **Z** — datatype of the accumulator $$C$$.
- **SS vs RS** — where operand $$A$$ comes from. `S` = shared memory, `R` = registers. $$B$$ is always `S` (SMEM-only), which is why the second letter is never `R`.
- **The two template parameters** — majorness of $$A$$ and $$B$$ respectively, each either `GMMA::Major::MN` or `GMMA::Major::K`. In BLAS terms: both K-major → TN gemm, both MN-major → NT gemm.

##### **Allowed MxNxK Shapes**

Not every tile shape is legal; the hardware fixes certain dimensions:

- **M is always 64**, tied to the warpgroup size (128 threads collectively producing a 64-row output tile).
- **N is a multiple of 8**, ranging from 8 up to 256 — this is where you have flexibility.
- **K is fixed to 32 bytes** along the $$K$$ axis. $$K$$ is specified in bytes, not elements, so the element count depends on the operand datatype: FP16/BF16 (2 bytes) → $$K = 16$$, FP8/INT8 (1 byte) → $$K = 32$$, TF32 (4 bytes) → $$K = 8$$. The 32-byte figure is the Hopper Tensor Core's atomic K-width — the smallest K-stripe the hardware is wired to consume per wgmma issue. It also aligns naturally with the core-matrix structure (two 16-byte-contiguous core matrices span one K-atom).

##### **What "K = 32 Bytes" Means at Runtime**

The full $$M \times N \times K$$ tile is consumed by a single `wgmma` instruction — not streamed element-by-element. So "32 bytes along K" is the K-width of one atomic MMA issue, not a per-cycle load granularity. For FP16, a single instruction consumes a $$64 \times N \times 16$$ tile (the entire $$K = 16$$ slice of 32 bytes per row of $$A$$).

If your SMEM tile's $$bK$$ is larger than one atom's $$K$$, `cute::gemm` loops over multiple wgmma issues. The hierarchy is:

- Your SMEM tile: $$bM \times bK$$ (e.g., $$128 \times 64$$ for FP16).
- Split along $$K$$ into $$\text{MMA\_K} = bK / \text{atom\_K}$$ atoms (here $$64 / 16 = 4$$).
- Each atom issue = one `wgmma` instruction, consuming one 32-byte K-stripe.

The `cute::gemm` call hides the loop over $$\text{MMA\_M}$$, $$\text{MMA\_N}$$, $$\text{MMA\_K}$$ — it is multiple wgmma issues under the hood, not one giant operation.

##### **In-Instruction Transpose and Majorness Flexibility**

For 16-bit operand datatypes (FP16, BF16), both MN-major and K-major layouts are supported, so you can write NT, TN, or mixed gemms. For every other datatype (FP8, INT8, TF32, etc.), WGMMA requires K-major — only TN is natively possible.

The reason is **in-instruction transpose support**. The `wgmma.mma_async` PTX instruction has two flag bits, `imm-trans-a` and `imm-trans-b`, that tell the Tensor Core to read SMEM in one layout but internally swap the two axes before feeding data to the multiply unit — essentially a free on-the-fly transpose during the load step. This is what lets an MN-major operand work even though the natural consumption order inside the Tensor Core is K-major.

These transpose flag bits **only exist in the FP16/BF16 variants** of the instruction; for FP8 and narrower dtypes, the bits are absent from the PTX encoding entirely. The NVIDIA PTX docs state this explicitly: the transpose operation is only supported for the `wgmma.mma_async` variants with `.f16`/`.bf16` types on matrices accessed from shared memory using matrix descriptors.

Why the hardware makes this asymmetric distinction is not authoritatively documented. A plausible reason is that the transpose hardware path has die-area and power cost, so NVIDIA only wired it for the dtypes that most benefit (FP16/BF16 being the dominant training dtypes when Hopper shipped), while newer narrow dtypes (FP8, etc.) were added with a simpler, K-major-only data path. This is speculation — the Hopper architecture whitepaper or NVIDIA forums would be the place to get a grounded answer.

The practical consequence: if your operand is FP8 and stored MN-major in SMEM, you cannot just flip the transpose bit. You would have to physically re-lay-out the data (e.g., via a transposing TMA load or a shared-memory transpose pass) before calling `wgmma`. This is why non-16-bit WGMMA kernels are always TN.

##### **Thread Count and Launch Configuration**

Since WGMMA is a warpgroup-wide instruction, the `TiledMMA` object carries the number of threads that collectively execute it. You can read this off with `cute::size(tiled_mma)`, which for a single WGMMA atom returns **128**. The common pattern is to pass this directly as the block dimension at launch:

```cpp
dim3 dimBlock(cute::size(tiled_mma));
```

This stipulates that each CTA launches with exactly one warpgroup of 128 threads, matching the atom's participation requirement.

##### **Scaling to Multiple Warpgroups via AtomLayoutMNK**

The first argument to `make_tiled_mma` (the MMA atom) fixes the size of a single atomic wgmma issue — e.g., `SM90_64x64x16` commits one warpgroup (128 threads) to producing a $$64 \times 64$$ output tile. But a CTA's tile $$bM \times bN$$ is usually larger, so you need to specify how multiple warpgroups cover the CTA tile. That is what the optional second argument **`AtomLayoutMNK`** does: it is a three-mode CuTe layout $$(M, N, K)$$ whose sizes specify how many wgmma atoms are tiled along each axis, with one warpgroup per atom. The default `Layout<Shape<_1,_1,_1>>` means one atom, one warpgroup.

For example, `Layout<Shape<_2,_1,_1>>` says 2 atoms along $$M$$, 1 along $$N$$, 1 along $$K$$ — two warpgroups splitting the output tile along the $$M$$ axis. Warpgroup 0 takes the upper 64 rows, warpgroup 1 takes the lower 64 rows, each issuing its own independent wgmma instructions with no coordination between them. This is why the blog requires $$bM$$ to be a multiple of 128: two 64-row atoms stacked along $$M$$ cover 128 rows total.

`cute::size(tiled_mma)` now equals **256** because each atom needs 128 threads and there are two atoms. At launch, CuTe assigns threads 0–127 to warpgroup 0 and threads 128–255 to warpgroup 1.

##### **Why M/N Splits Are Natural but K Is Different**

Splits along $$M$$ or $$N$$ are embarrassingly parallel — each warpgroup writes a disjoint region of the output tile, so no coordination is needed. A split along $$K$$ (e.g., `Layout<Shape<_1,_1,_2>>`) would have both warpgroups compute the same $$(M, N)$$ output tile over different K-slices, which then requires a **reduction** to combine partial sums. That is a valid but more involved pattern (Stream-K–style decompositions use it) and carries extra cost, which is why standard GEMM kernels stick to $$M$$- or $$N$$-splits.

##### **PermutationMNK**

The third optional knob to `make_tiled_mma` controls how threads and warps are permuted within the atom layout. It is orthogonal to `AtomLayoutMNK` and mostly relevant for advanced tiling strategies. Safe to treat as out-of-scope for now, with Cris Cecka's writeup (linked in the Colfax blog) as the reference when you need it.

---

#### **SMEM Layout Constraints for WGMMA**

##### **Layout Atoms**

A **layout atom** is a small, fixed CuTe layout (shape + stride + swizzle) hand-designed by CUTLASS to satisfy WGMMA's core-matrix and swizzle requirements. The 8 canonical GMMA atoms — MN-major or K-major × {no-swizzle, SW32, SW64, SW128} — are the only WGMMA-blessed SMEM building blocks. You do not construct them yourself; you pick one and use `cute::tile_to_shape` to replicate it across the larger $$(bM, bK, bP)$$ shape. This replication preserves compatibility because `tile_to_shape` only adds outer modes — the atom's internal shape, stride, and swizzle stay intact.

##### **Reading the Full SMEM Layout Printout**

Consider the printout:

```
Sw<3,4,3> o smem_ptr o ((_64,_2),(_8,_8),_3):((_1,_512),(_64,_1024),_8192)
```

This is the result of tiling the atom `(64, 8):(1, 64)` over the shape `(128, 64, 3)` in column-major fashion. To derive it mechanically:

**Atom identification.** The `Layout_MN_SW128_Atom<half_t>` atom is `(64, 8):(1, 64)`: 64 elements contiguous in $$M$$ (stride 1), 8 elements in $$K$$ (stride 64). It covers a $$64 \times 8$$ sub-tile and occupies $$64 \times 8 = 512$$ half-elements of SMEM (its cosize).

**Tile counts.** Atom mode-0 (64) divides $$bM = 128$$ → 2 M-tiles. Atom mode-1 (8) divides $$bK = 64$$ → 8 K-tiles. The pipeline mode $$bP = 3$$ has no atom inner, so it is just 3 stages.

**New shape.** `tile_to_shape` puts the atom's structure inside each mode and tile counts outside: mode-0 becomes $$(64, 2)$$, mode-1 becomes $$(8, 8)$$, mode-2 is 3. Combined: $$((64, 2), (8, 8), 3)$$.

**New strides (column-major walk).** Inner strides come from the atom unchanged. Outer strides are placed so each successive tile starts where the previous tile's footprint ends, walking M-tiles first, then K-tiles, then stages:

- Inner $$M$$ stride = 1, outer $$M$$ stride = atom cosize = 512.
- Inner $$K$$ stride = 64, outer $$K$$ stride = atom cosize × M-tile-count = $$512 \times 2 = 1024$$.
- $$P$$ stride = single-stage cosize = $$1024 \times 8 = 8192$$.

Combined: $$((1, 512), (64, 1024), 8192)$$ — matches the printout.

The "column-major" intuition is that, when flattened to linear memory, M-tiles walk first (smallest outer stride 512), then K-tiles (1024), then pipeline stages (8192). The largest stride sits in the $$P$$ mode because different stages must live in completely separate SMEM regions with no interleaving — that is what makes circular buffering safe. The leading `Sw<3,4,3>` is the swizzle function applied on top of this raw layout, permuting addresses within each row to avoid SMEM bank conflicts; its internals are covered in the core-matrix section.

##### **The Rule: Atom Contiguous-Byte Span = Swizzle Mode's Name**

For the SW128 atom in our example, the contiguous direction is 64 elements, and $$64 \times \text{sizeof(half\_t)} = 128$$ bytes — exactly the "128" in the atom's name. This holds for every GMMA layout atom: the contiguous span in bytes equals the swizzle mode's number, with the four canonical values being:

- **No swizzle** → 16 bytes
- **SW32** → 32 bytes
- **SW64** → 64 bytes
- **SW128** → 128 bytes

The element count varies by dtype (FP16 / SW128 = 64 elements; FP8 / SW128 = 128 elements), but the byte width is fixed by the swizzle name.

##### **Why This Rule Exists**

It is a design constraint imposed by how core matrices and swizzling fit together. A **core matrix** is the smallest SMEM unit WGMMA operates on — 8 rows × 16 bytes contiguous. Swizzling permutes which 16-byte chunk each thread hits by XOR-ing address bits, spreading accesses across SMEM banks to avoid conflicts. The number after "SW" specifies how many 16-byte chunks the swizzle pattern interleaves: SW32 interleaves 2 chunks, SW64 interleaves 4, SW128 interleaves 8 (no-swizzle = 1, i.e., no permutation).

For the swizzle to produce a clean, well-defined permutation, the atom's contiguous-direction span must equal exactly one full swizzle period. A shorter span would cut the pattern mid-cycle, leaving some chunks unpermuted; a longer span would repeat the pattern with no benefit and break the atomicity. Locking the two together — atom contiguous bytes = swizzle bytes — makes the layout atom and its swizzle function a single self-consistent unit, which is why CUTLASS pre-defines exactly 4 swizzle modes × 2 majornesses = 8 GMMA atoms.

> **TODO:** The swizzle and core-matrix concepts are not yet fully clear here. They will be discussed in depth in the WGMMA core matrices section. After writing that section, revisit this one and add more bridging context.

##### **K-Major SW128 Derivation**

With `Layout_K_SW128_Atom<half_t>` = `(8, 64):(64, 1)`, $$K$$ becomes the contiguous direction (stride 1, span 64 elements = 128 bytes, matching the SW128 name) and $$M$$ is the strided direction (stride 64, span 8 elements). Atom cosize is $$8 \times 64 = 512$$ half-elements. Tiling over $$(bM, bK, bP) = (128, 64, 3)$$:

- **Tile counts:** $$128 / 8 = 16$$ along $$M$$, $$64 / 64 = 1$$ along $$K$$, 3 stages.
- **Pre-coalesce shape:** $$((8, 16), (64, 1), 3)$$.
- **Pre-coalesce strides (column-major walk):** inner $$M$$ = 64 (from atom), outer $$M$$ = atom cosize = 512; inner $$K$$ = 1 (from atom), outer $$K$$ = 0 (only one K-tile, degenerate); $$P$$ = single-stage cosize = $$512 \times 16 = 8192$$. Combined: $$((8, 16), (64, 1), 3) : ((64, 512), (1, 0), 8192)$$.

##### **Coalescing**

CuTe collapses adjacent inner/outer pairs $$(s_\text{inner}, s_\text{outer}) : (d_\text{inner}, d_\text{outer})$$ into $$s_\text{inner} \times s_\text{outer} : d_\text{inner}$$ whenever $$d_\text{outer} = d_\text{inner} \times s_\text{inner}$$ — i.e., when stepping the outer coordinate by 1 lands you exactly past $$s_\text{inner}$$ inner elements with no gap, making the two-level structure redundant. Applying this to the K-major case:

- **Mode-0 (M):** $$(8, 16) : (64, 512)$$ → check $$64 \times 8 = 512$$ ✓ → collapses to $$128 : 64$$.
- **Mode-1 (K):** $$(64, 1) : (1, 0)$$ → outer extent is 1, trivially removable → collapses to $$64 : 1$$.
- **Mode-2 (P):** already simple → $$3 : 8192$$.

Final layout: $$(128, 64, 3) : (64, 1, 8192)$$, matching the printout.

##### **Why Coalescing Happened Here but Not in the MN-Major Case**

In the MN-major SW128 example, the atom was `(64, 8):(1, 64)` and the outer $$M$$ stride (512) did not equal inner $$M$$ extent × inner $$M$$ stride ($$64 \times 1 = 64$$), so the two-level structure had to stay. K-major SW128 packs atoms tightly along $$M$$ — each new M-atom starts exactly where the previous one's M-span ends — so the layout simplifies. This is not a cosmetic difference: K-major SW128 is fully compact in $$M$$ for $$bM = 128$$, while MN-major SW128 is not, which has real implications for how core matrices are stacked in memory.

##### **The 8 Canonical GMMA Layout Atoms**

CUTLASS provides exactly 8 pre-defined layout atoms for WGMMA SMEM tiles, one for each combination of {MN-major, K-major} × {no-swizzle, SW32, SW64, SW128}:

- `GMMA::Layout_MN_INTER_Atom<T>` / `Layout_MN_SW32_Atom<T>` / `Layout_MN_SW64_Atom<T>` / `Layout_MN_SW128_Atom<T>`
- `GMMA::Layout_K_INTER_Atom<T>` / `Layout_K_SW32_Atom<T>` / `Layout_K_SW64_Atom<T>` / `Layout_K_SW128_Atom<T>`

The four swizzle modes differ in how many 16-byte segments they interleave when permuting SMEM addresses:

- **No swizzle (INTER)** — implicit 16-byte boundary, no permutation.
- **SW32** — 2 consecutive 16-byte segments interleaved.
- **SW64** — 4 consecutive 16-byte segments interleaved.
- **SW128** — 8 consecutive 16-byte segments interleaved.

These atoms are passed into `cute::tile_to_shape` with the SMEM shape `make_shape(bM, bK, bP)` for `sA` or `make_shape(bN, bK, bP)` for `sB` — modes given in that exact order so the atom's mode-0 lines up with $$M$$ (or $$N$$), mode-1 with $$K$$, and the pipeline mode is outermost. The tile sizes of the chosen atom must divide evenly into the corresponding modes of the SMEM shape.

##### **The Two Independent Constraints: MMA Atom vs Swizzle Mode**

The tile sizes $$bM$$, $$bN$$, $$bK$$ are squeezed by two different requirements at once, coming from two unrelated parts of the system:

**The MMA atom constraint.** The `wgmma` instruction itself fixes $$M = 64$$ and K-bytes = 32 (so $$K = 16$$ elements for FP16). Therefore $$bM$$ must be a multiple of 64, $$bN$$ a multiple of 8 (and ≤ 256), and $$bK$$ a multiple of 16 (for FP16). This comes from the Tensor Core hardware — what shapes the multiply unit can issue. It has nothing to do with how data sits in SMEM.

**The swizzle-mode (layout-atom) constraint.** Once you pick a layout atom — say `Layout_MN_SW128_Atom<half_t>` with internal shape `(64, 8)` — $$bM$$ must be a multiple of the atom's mode-0 (64), and $$bK$$ a multiple of mode-1 (8), so `tile_to_shape` can replicate the atom cleanly. If you switched to SW32, the atom would have a different shape (smaller contiguous span of 32 bytes = 16 FP16 elements), and the divisibility requirements on $$bM$$/$$bK$$ would change accordingly. This comes from the SMEM layout machinery — what shapes the atom can tile into without breaking its swizzle period.

The two constraints look superficially similar (both are divisibility conditions on $$bM$$, $$bN$$, $$bK$$), but they are imposed by different mechanisms and you must satisfy both simultaneously.

##### **Worked Example: How the Two Constraints Interact When You Change Swizzle Mode**

To make the independence concrete, take a fixed problem and walk through what happens as you swap the layout atom. Setup: FP16 GEMM, NT (both operands MN-major), MMA atom = `SM90_64x64x16_F16F16F16_SS`, choosing CTA tile sizes $$bM$$, $$bN$$, $$bK$$.

**The MMA atom constraint (constant across all cases).** The atom is $$64 \times 64 \times 16$$, so: $$bM$$ must be a multiple of 64, $$bN$$ a multiple of 64, $$bK$$ a multiple of 16. This constraint is fixed by the Tensor Core hardware and does not change as long as we keep the same MMA atom.

**Case A — SW128 atom (`Layout_MN_SW128_Atom<half_t>`).** The atom's internal shape is `(64, 8)`, with mode-0 ($$M$$, contiguous) spanning 64 elements = 128 bytes (matching the SW128 name) and mode-1 ($$K$$, strided) spanning 8 elements. Atom divisibility requires $$bM$$ multiple of 64 and $$bK$$ multiple of 8. Intersecting with the MMA constraint: $$bM$$ multiple of $$\text{lcm}(64, 64) = 64$$ and $$bK$$ multiple of $$\text{lcm}(16, 8) = 16$$. So a tile like $$bM = 128$$, $$bK = 64$$ works — $$128/64 = 2$$ atom tiles along $$M$$, $$64/8 = 8$$ atom tiles along $$K$$, and the MMA atom divides cleanly with $$128/64 = 2$$ along $$M$$ and $$64/16 = 4$$ wgmma issues along $$K$$. This is exactly the running example from the Colfax blog.

**Case B — SW32 atom (`Layout_MN_SW32_Atom<half_t>`).** The atom's contiguous direction must span 32 bytes = $$32 / 2 = 16$$ FP16 elements, so the atom shape is `(16, 8)`: mode-0 ($$M$$, contiguous) = 16 elements = 32 bytes, mode-1 ($$K$$, strided) = 8 elements. Atom divisibility now requires $$bM$$ multiple of 16 and $$bK$$ multiple of 8. Intersecting with MMA: $$bM$$ multiple of $$\text{lcm}(64, 16) = 64$$ and $$bK$$ multiple of $$\text{lcm}(16, 8) = 16$$. So $$bM = 128$$, $$bK = 64$$ still works — but notice what changed conceptually. With SW128, the swizzle atom's M-dimension (64) and the MMA atom's M-dimension (64) coincided exactly; the two constraints reinforced each other. With SW32, the swizzle atom's M-dimension (16) is more permissive, so the MMA atom is now the tighter constraint and you cannot infer it from the swizzle check.

This sets up a real failure mode: suppose someone picks $$bM = 32$$ thinking "the SW32 atom only needs 16-divisibility, and 32 satisfies that." It does pass the swizzle constraint ($$32/16 = 2$$ ✓) but fails the MMA atom constraint ($$32/64 < 1$$ ✗). The kernel would either fail to compile (`CuTe`'s `CUTE_STATIC_ASSERT` would fire) or — worse, in less defensive code paths — produce wrong results. This is precisely the failure mode the blog is warning about: the two constraints are independent and you cannot infer one from the other.

**Case C — No-swizzle (`Layout_MN_INTER_Atom<half_t>`).** The no-swizzle atom has contiguous span = 16 bytes = 8 FP16 elements, so the atom shape is `(8, 8)`: mode-0 ($$M$$) = 8 elements = 16 bytes, mode-1 ($$K$$) = 8 elements. Atom divisibility requires $$bM$$ multiple of 8 and $$bK$$ multiple of 8. Intersecting with MMA gives $$bM$$ multiple of $$\text{lcm}(64, 8) = 64$$ and $$bK$$ multiple of $$\text{lcm}(16, 8) = 16$$. Once again, $$bM = 128$$, $$bK = 64$$ works, and once again the swizzle constraint is even weaker than before — the MMA atom dominates on every dimension.

##### **The Pattern That Emerges**

For FP16 / MN-major, the swizzle atom's M-dimension is $$\text{swizzle\_bytes} / \text{sizeof}(T)$$, which gives $$\{8, 16, 32, 64\}$$ for {INTER, SW32, SW64, SW128} respectively. The K-dimension stays at 8 across all four atoms. The MMA atom requires $$M$$ divisible by 64 and $$K$$ divisible by 16. So:

- For **SW128** (M-atom = 64), the swizzle and MMA constraints coincide on $$M$$ and reinforce each other.
- For **SW64, SW32, INTER** (M-atoms = 32, 16, 8), the swizzle constraint on $$M$$ is weaker than MMA, so MMA dominates. The risk is that a careless $$bM$$ choice could pass the swizzle check but silently fail MMA.
- On $$K$$, MMA always dominates (16 vs swizzle's 8) regardless of swizzle mode.

##### **How You Would Adjust in Practice**

Starting from $$bM = 128$$, $$bN = 128$$, $$bK = 64$$ and changing one knob at a time:

- **SW128 → SW64:** same tile sizes still work, no adjustment needed. But you have given up some swizzle "tightness" — fewer 16-byte segments are interleaved, so for wide K-strides you might see more SMEM bank conflicts at runtime. Performance changes, not correctness.
- **SW128 → no-swizzle:** same tile sizes still work, but you have no bank-conflict avoidance at all. Typically only used for debugging or when tiles are small enough that conflicts do not matter.
- **Switch dtype from FP16 to FP8 while keeping SW128:** the SW128 atom's M-dimension becomes $$128 / \text{sizeof(fp8)} = 128$$ elements, not 64. Now $$bM = 128$$ is the minimum allowed by the swizzle atom, and $$bM = 64$$ would fail the swizzle constraint even though it satisfies the MMA atom (FP8 wgmma is `m64nNk32`, still $$M = 64$$). You would have to bump $$bM$$ to 128 minimum. This is the cleanest illustration of the two constraints pulling in different directions: the MMA atom got more permissive on $$K$$ (because FP8 packs more elements per byte) while the swizzle atom got more demanding on $$M$$.

The general lesson: when picking tile sizes, validate them against the MMA atom and the swizzle atom independently, and do not assume that satisfying one implies satisfying the other. The cases where they line up (SW128 + FP16) are convenient but not the rule.

---

#### **WGMMA Fragments and Descriptors**

##### **TiledMMA Lifecycle: Host Construction, Device Use**

The `tiled_mma` object is constructed on the host but is effectively a stateless type-level descriptor — it carries no runtime data, only template parameters and types encoding the MMA atom and atom layout. It is then passed into the kernel as an argument, which makes it available on the device. There is no "double creation" — the same object exists in both contexts; references to "host vs device" simply refer to where the surrounding code runs.

##### **`get_thread_slice` and What `partition_A`/`partition_B`/`partition_C` Produce**

On the device, `tiled_mma.get_thread_slice(threadIdx.x)` returns a `ThrMMA` object — essentially `tiled_mma` with the calling thread's index baked in. Calling `thr_mma.partition_A(sA)` then produces `tCsA`, a CuTe tensor that is a **view** over the same SMEM bytes as `sA` but reshaped into the layout `cute::gemm` needs: `(MMA, MMA_M, MMA_K, PIPE)`. No memory is allocated or copied — the underlying bytes are unchanged, only the indexing scheme is reinterpreted. The shape exposes the iteration axes of `cute::gemm`:

- **Mode-0 (MMA)** = one wgmma atom's worth of A-data ($$M = 64$$, $$K = 16$$ for FP16).
- **Mode-1 (MMA\_M)** = how many M-atoms across the SMEM tile ($$bM / 64 = 2$$).
- **Mode-2 (MMA\_K)** = how many K-atoms across the SMEM tile ($$bK / 16 = 4$$).
- **Mode-3 (PIPE)** = which pipeline stage.

So `tCsA(_, m, k, p)` selects the SMEM region for "the wgmma atom at M-atom index $$m$$, K-atom index $$k$$, in pipeline stage $$p$$," and `cute::gemm` walks $$m$$ and $$k$$ issuing one wgmma per $$(m, k)$$ pair.

##### **Separating Two Unrelated Splittings**

Two kinds of "splitting" of the SMEM tile happen, and they are orthogonal — easy to conflate, important to keep distinct:

- **Pipeline stages (the PIPE mode, \_3).** This is the circular SMEM buffer, used to overlap TMA prefetching with wgmma compute. The PIPE dimension is already baked into `sA` from `make_shape(bM, bK, bP)`. `partition_A` does not create or modify it — it just carries it through.
- **wgmma-atom iteration (the MMA\_M, MMA\_K modes).** Within one pipeline stage's SMEM tile ($$128 \times 64$$ for FP16), a single wgmma instruction only covers $$64 \times 16$$, so $$\text{MMA\_M} \times \text{MMA\_K} = 2 \times 4 = 8$$ wgmma issues are needed to consume the whole tile. This is what `partition_A` exposes — it restructures the layout so the inner mode is one wgmma atom's worth of data and the outer modes count how many atoms in each direction. `cute::gemm` then loops over those outer modes.

The PIPE mode is about producer/consumer concurrency. The MMA\_M/MMA\_K modes are about "tile is bigger than one wgmma atom, so loop." `partition_A` reshapes only the latter and leaves PIPE alone.

##### **Deriving the (\_8, \_2) Inner Structure of tCsA**

Recall `sA`'s layout was `((_64, _2), (_8, _8), _3) : ((_1, _512), (_64, _1024), _8192)` — atom-shape 64 with 2 M-atoms in mode-0, atom-shape 8 with 8 K-atoms in mode-1, 3 pipeline stages in mode-2. The MMA atom is $$64 \times 64 \times 16$$, so one wgmma issue consumes $$K = 16$$ elements, but the SMEM K-atom is only 8 elements wide. Therefore one wgmma atom spans 2 consecutive SMEM K-atoms ($$8 \times 2 = 16$$) — that is where the `(_8, _2)` comes from.

Concretely, mode-0 of `tCsA` represents "all the SMEM data in one wgmma atom" — 64 elements along $$M$$ and 16 along $$K$$. The $$K = 16$$ is built from the SMEM's existing K structure: 8 elements within one SMEM K-atom (stride 64), then jump to the next SMEM K-atom (stride 1024). That gives the nested `(8, 2) : (64, 1024)`. So mode-0 = `(64, (8, 2)) : (1, (64, 1024))`. The outer modes then count how many wgmma atoms fit: MMA\_M = 2 with stride 512 (each M-atom = 64 rows, next M-atom 512 elements later, inherited from `sA`); MMA\_K = 4 with stride 2048 (one wgmma atom consumes 2 SMEM K-atoms, so the next wgmma atom starts $$2 \times 1024 = 2048$$ away); PIPE = 3 with stride 8192. Combined: `((_64, (_8, _2)), _2, _4, _3) : ((_1, (_64, _1024)), _512, _2048, _8192)`, matching the printout.

##### **Reorganized Layout, Not a Thread-Level Slice**

In a non-WGMMA MMA (e.g., SM80 `mma.sync`), `partition_A` would return a different slice for each thread because each thread holds its own register fragment of $$A$$ — the "thread slice" name reflects that. For WGMMA in SS mode, $$A$$ is read directly from SMEM by the Tensor Core via the matrix descriptor, with no register fragmentation per thread. Every thread in the warpgroup needs the same view of "where in SMEM the data lives" because the descriptor encodes a base address and layout metadata that the hardware uses uniformly across all 128 threads. So `thr_mma.partition_A(sA)` returns the **same tensor view for every thread index 0–127**, and that view is the entire SMEM A-tile reshaped into the `(MMA, MMA_M, MMA_K, PIPE)` form `cute::gemm` needs.

The name `partition_A` is a slight misnomer here — it is a "partition" only in the formal CuTe sense (a layout transformation), not in the "give each thread a different chunk" sense. The $$C$$ tensor is different: `tCgC` and `tCrC` do give each thread its own register slice (the 32 accumulator values per thread arranged in the Z-pattern), because the accumulator genuinely lives in registers. The asymmetry is structural: $$A$$ and $$B$$ in SS mode are descriptor-based with a shared layout view, while $$C$$ is register-backed with a per-thread layout.

##### **Terminology Cleanup: "wgmma Atom" vs "SMEM Layout Atom"**

The word "atom" gets used for two unrelated things in the WGMMA discussion, and they need to be kept separate:

- **wgmma atom (MMA atom)** = the tile shape of one wgmma instruction, fixed by the chosen MMA atom (e.g., `SM90_64x64x16_F16F16F16_SS` → $$64 \times 64 \times 16$$). One wgmma issue consumes a $$64 \times 16$$ slab of $$A$$, a $$64 \times 16$$ slab of $$B$$, and produces a $$64 \times 64$$ slab of $$C$$. This is a **hardware concept** — the smallest unit the Tensor Core can issue.
- **SMEM layout atom** = the building block tiled by `tile_to_shape` to fill `sA`/`sB` (e.g., `Layout_MN_SW128_Atom<half_t>` = `(64, 8)`). The K-extent of this atom — 8 elements in our example — is the **SMEM K-atom width**. This is a **layout concept**, chosen by CUTLASS so swizzling lines up correctly with core matrices.

The K mismatch between the two is where the `(_8, _2)` nesting in `tCsA` comes from. WGMMA needs $$K = 16$$ elements per issue, but the SMEM layout atom only lays out $$K = 8$$ at a time. So one wgmma issue must span two consecutive SMEM layout atoms in the K direction:

```
SMEM K-atoms (laid out by tile_to_shape):
[ atom_0 (K=8) ][ atom_1 (K=8) ][ atom_2 (K=8) ][ atom_3 (K=8) ] ...
 \_________________________/   \_________________________/
   one wgmma issue (K=16)         next wgmma issue (K=16)
```

This is exactly the `(_8, _2)` in mode-0 of `tCsA`: 8 elements within one SMEM layout atom (stride 64), then ×2 to span two such atoms (stride 1024 to reach the next), totaling $$K = 16$$ for one wgmma issue. And $$\text{MMA\_K} = bK / 16 = 64 / 16 = 4$$ is the number of wgmma issues needed to consume the whole SMEM tile in $$K$$ — there are 8 SMEM layout atoms in $$K$$ total, paired up into 4 wgmma issues.

When reading layout printouts, mentally label which "atom" is in play:

- **MMA-atom-M, MMA-atom-N, MMA-atom-K** → hardware-defined wgmma tile shape ($$64, 64, 16$$ for FP16 here).
- **Layout-atom-M, Layout-atom-K** → CUTLASS swizzle-driven SMEM building block ($$64, 8$$ for `Layout_MN_SW128_Atom<half_t>`).

The two coincide on $$M$$ (both 64) for this specific atom + dtype combination, which is why the $$M$$ direction looks "clean" in the printout. They diverge on $$K$$ (16 vs 8), which is why mode-0 of `tCsA` has the nested `(_8, _2)` instead of a flat 16.

---

#### **Part 1: Inline PTX, Descriptor Sourcing, and the Iterator Concept**

##### **Inline PTX and Per-Thread Operand Sourcing**

The `wgmma.mma_async` PTX instruction is invoked through inline assembly inside CUTLASS's `fma` method. Inline PTX is a C++ feature that lets you embed a literal PTX string and bind C++ variables to its operands via constraint codes — `"l"(desc_a)` says "place the C++ variable `desc_a` (a `uint64_t`) in a 64-bit register, and substitute that register's name for the corresponding `%N` placeholder in the PTX string." Similarly `"r"(...)` binds a 32-bit register and `"+r"(...)` binds a read-write 32-bit register.

The crucial point for understanding wgmma is that **operand sourcing is per-thread**, even though instruction execution is warpgroup-collective. Every CUDA thread has its own register file. When the PTX instruction references `%16` (the `desc_a` slot), each of the 128 threads in the warpgroup substitutes that placeholder with its own register holding its own copy of `desc_a`. The hardware's "warpgroup collectivity" lives at the instruction-execution layer (the Tensor Core coordinates all 128 threads to do the matmul together), not at the operand-sourcing layer (CUDA has no shared-register mechanism).

For wgmma specifically, all 128 threads end up holding the **same** 64-bit descriptor value, because the descriptor encodes a SMEM base address (shared across the warpgroup) plus layout metadata (compile-time constants). The redundancy is by design — it is simply how the PTX/hardware contract is structured.

##### **CUTLASS Tensor Naming Convention `t<X><Y>`**

CUTLASS uses a two-letter suffix convention:

- The leading **`t`** marks "thread-partitioned" — produced by a partitioning operation (`partition_A`, `partition_C`, `tma_partition`, etc.).
- The **middle letter** identifies which partitioner produced the view. `C` is the MMA partitioner (named after the accumulator role conventionally taken by $$C$$). `A` and `B` in TMA-partition results refer to the TMA-copy atoms for $$A$$ and $$B$$.
- The **trailing letter** identifies the memory space the data lives in: `g` (gmem), `s` (smem), `r` (rmem).

So `tCsA` = thread-partitioned (by MMA partitioner) view of SMEM-A. `tCgC` = MMA-partitioned view of GMEM-C. `tCrA` = MMA-partitioned view of RMEM-A. The middle `C` is **not** the $$C$$ matrix — it is a label for the partitioner. (For TMA copies in reference code you will see `tAgA`, `tAsA`, etc., where the middle `A` means "TMA-copy partitioner for A.")

##### **tCrA vs tCsA: Descriptors vs Values**

In SS mode, both `tCrA` and `tCsA` describe the same SMEM bytes, but at different abstraction layers:

- **`tCsA`** is a tensor of `half_t` values. Indexing it gives you a half-precision element. This is the "what the data is" view.
- **`tCrA`** is a tensor of descriptors (`GMMA::DescriptorIterator`). Indexing it gives you a 64-bit descriptor that points at one wgmma atom's worth of SMEM. This is the "what wgmma needs to find the data" view.

The fragment is built by `make_fragment_A(tCsA)`, which constructs the descriptor-iterator on top of the SMEM view. The key insight of SS mode is: **values of SMEM are not copied into RMEM; rather, accessing the values of `tCrA` and `tCrB` accesses these 64-bit descriptors.** The operand never leaves SMEM; only a pointer-like descriptor sits in registers, and the Tensor Core reads SMEM directly.

In contrast, RS-mode wgmma would actually copy $$A$$ from SMEM into registers ahead of time, and `tCrA` would then be a true register-resident tensor of values rather than descriptors. That is a separate code path.

##### **The "Iterator" Semantics**

`DescriptorIterator` is the type because CUTLASS does not precompute and stash all 24 descriptors ($$\text{MMA\_M} \times \text{MMA\_K} \times \text{PIPE} = 2 \times 4 \times 3$$) in registers. That would burn $$24 \times 64 = 1536$$ bits per thread of register pressure, for no benefit. Instead, on each iteration of `cute::gemm`'s internal loop:

1. Read the current $$(m, k, p)$$ position.
2. Compute a fresh descriptor by adding the appropriate offset to the SMEM base.
3. Place it in a register.
4. Issue the wgmma instruction.
5. On the next iteration, overwrite the register with the next descriptor.

So at any moment only **one A-descriptor and one B-descriptor** live in registers per thread. The strides on `tCrA`'s layout are the iterator's address-arithmetic recipe — they tell it how to step from one descriptor to the next.

---

#### **Part 2: Decoding tCrA's Strides**

##### **tCrA's Shape: (\_1, \_2, \_4, \_3)**

Compare to `tCsA`'s shape `((_64, (_8, _2)), _2, _4, _3)`. Both have the same outer modes `_2, _4, _3` (MMA\_M, MMA\_K, PIPE), but the inner MMA mode differs dramatically:

- `tCsA` mode-0 = `(_64, (_8, _2))` — the $$64 \times 16$$ worth of half-precision elements that make up one wgmma atom's A-data.
- `tCrA` mode-0 = `_1` — a single descriptor that encodes that entire $$64 \times 16$$ atom in 64 bits.

This collapse is the whole point: **one wgmma atom = one descriptor**. There is nothing to iterate over within mode-0 because the atom is the unit the descriptor describes.

##### **The Descriptor's Address Encoding**

The 64-bit matrix descriptor packs five fields into 64 bits: SMEM base address, LBO, SBO, swizzle mode, and matrix base offset (covered later in the core-matrices section). The address fields are encoded in a packed format — specifically, SMEM addresses are stored in **16-byte units** rather than raw bytes. This is because wgmma's natural granularity is 16 bytes (the contiguous span of one core matrix), and storing addresses at finer resolution would waste bits.

The practical consequence is that the iterator's stride arithmetic operates in 16-byte units, not bytes and not elements. To compare against `tCsA`'s element-unit strides, you have to convert: $$\text{descriptor\_stride} \times 16 = \text{byte\_stride}$$, and $$\text{byte\_stride} / \text{sizeof(half\_t)} = \text{element\_stride}$$.

##### **The Strides: (\_0, \_64, \_256, \_1024)**

- **Mode-0 stride = 0.** Mode-0 has size 1, so there is no second position to step to — the stride is degenerate. Set to 0 by convention because "moving within a one-element axis doesn't change the address."
- **Mode-1 stride = 64 (MMA\_M).** Walking from M-atom 0 to M-atom 1 advances the descriptor by $$64 \times 16 = 1024$$ bytes = 512 `half_t`. Cross-check against `tCsA`'s MMA\_M stride: 512. ✓
- **Mode-2 stride = 256 (MMA\_K).** Walking from K-atom 0 to K-atom 1 advances by $$256 \times 16 = 4096$$ bytes = 2048 `half_t`. Cross-check against `tCsA`'s MMA\_K stride: 2048. ✓
- **Mode-3 stride = 1024 (PIPE).** Walking to the next pipeline stage advances by $$1024 \times 16 = 16384$$ bytes = 8192 `half_t`. Cross-check against `tCsA`'s PIPE stride: 8192. ✓

The pattern is clean: `tCrA`'s strides are exactly `tCsA`'s strides divided by 8 (since `half_t` is 2 bytes and the descriptor uses 16-byte units, the conversion factor is $$16 / 2 = 8$$).

##### **What the Iterator Actually Does at Runtime**

When `cute::gemm` indexes `tCrA(_, m, k, p)` for the current loop position, the iterator computes:

```
current_descriptor = base_descriptor + 64*m + 256*k + 1024*p
```

(in descriptor-units, all baked into the descriptor's address field). The resulting 64-bit value goes into a register and is fed as `desc_a` to the next wgmma issue. After the issue, the register is reused for the next position. This is what "the descriptor gets updated per wgmma" looks like in practice — a few integer adds, one register write, then the wgmma fires.

`tCrB` has identical structure for B-side descriptors. The two iterators advance independently as `cute::gemm` walks the $$(m, k)$$ loop nest.

---

#### **Part 3: Decoding tCgC and tCrC**

##### **The Asymmetry Between Operands and Accumulator**

$$A$$ and $$B$$ in SS mode are descriptor-based — `tCrA` and `tCrB` hold pointers, not data, and every thread sees the same view. $$C$$ is fundamentally different: the accumulator lives in registers, distributed across the warpgroup, with each thread owning a specific subset of the $$64 \times 64$$ atom's 4096 output values. With 128 threads, that is $$4096 / 128 = 32$$ values per thread per atom. This asymmetry is structural — wgmma's design makes the operands collective (read once from SMEM) but the result distributed (each thread keeps its slice in registers).

##### **Which 32 Values Each Thread Owns: The Z-Pattern**

The PTX documentation specifies the exact ownership pattern via the Z-pattern figure (figure 122 in the PTX docs, reproduced in the Colfax blog). For thread 0 in the $$64 \times 64$$ output atom:

```
(0,0)  (0,1)        (0,8)  (0,9)        (0,16) (0,17)        ...    (0,56) (0,57)
(8,0)  (8,1)        (8,8)  (8,9)        (8,16) (8,17)        ...    (8,56) (8,57)
```

That is 32 (row, col) positions, structured as three nested patterns:

- **Inner 2** = pair of adjacent columns $$(c, c+1)$$. Step = 1 column.
- **Middle 2** = pair of rows $$(r, r+8)$$. Step = 8 rows.
- **Outer 8** = 8 column-groups, each 8 columns apart, covering all 64 output columns. Step = 8 columns.

Total: $$2 \times 2 \times 8 = 32$$. ✓ This is why the inner mode has to be factored as `(2, 2, 8)` rather than a flat 32 — a flat shape can carry only one stride, but this ownership pattern needs three independent strides (for the three step patterns), which forces three modes.

##### **Decoding tCgC: ((\_2, \_2, \_8), \_2, \_2) : ((512, \_8, 4096), \_64, 32768)**

This is the GMEM view (per thread 0) of where its $$32 \times \text{MMA\_M} \times \text{MMA\_N}$$ values live in the output matrix. $$C$$ is column-major in GMEM with $$\text{ldC} = M = 512$$, so a row-step has linear stride 1 and a column-step has linear stride 512.

**Inner `(2, 2, 8) : (512, 8, 4096)`:**

- Inner 2 (column pair): 1-column step in column-major = $$1 \times \text{ldC} = 512$$. ✓
- Middle 2 (row pair separated by 8): 8-row step = $$8 \times 1 = 8$$. ✓
- Outer 8 (column-groups separated by 8): 8-column step = $$8 \times \text{ldC} = 4096$$. ✓

**Outer `(_2, _2) : (_64, 32768)`:**

- MMA\_M = 2 atoms along $$M$$, stride 64. Next M-atom = 64 rows down = $$64 \times 1 = 64$$. ✓
- MMA\_N = 2 atoms along $$N$$, stride 32768. Next N-atom = 64 columns over = $$64 \times \text{ldC} = 64 \times 512 = 32768$$. ✓

So `tCgC` is the precise map from "thread 0's logical position within its register slice" to "byte offset in the GMEM output." The strides encode both the Z-pattern within an atom and the atom-tiling across the CTA tile.

##### **Decoding tCrC: ((\_2, \_2, \_8), \_2, \_2) : ((\_1, \_2, \_4), \_32, \_64)**

Same shape as `tCgC` — same logical positions — but the strides now describe **register slot positions**, not memory addresses. Each thread allocates $$\text{MMA\_M} \times \text{MMA\_N} \times 32 = 2 \times 2 \times 32 = 128$$ register slots to hold its share of the full output tile, and the strides describe how those 128 slots are laid out within that register block.

**Inner `(2, 2, 8) : (1, 2, 4)`:**

- Stride 1 for mode-0, stride 2 for mode-1, stride 4 for mode-2.
- This is the standard column-major-compact layout for a `(2, 2, 8)` tensor: total cosize = $$2 \times 2 \times 8 = 32$$ register slots per atom. ✓
- The 32 values are packed contiguously and densely; no gaps.

**Outer `(2, 2) : (32, 64)`:**

- MMA\_M stride 32 — next M-atom's slots start 32 positions later (one atom = 32 slots). ✓
- MMA\_N stride 64 — next N-atom's slots start 64 positions later (after both M-atoms × 32 slots). ✓

So `tCrC` is "register storage layout": dense, compact, walking the 128 register slots in column-major order across (value-within-atom, MMA\_M, MMA\_N).

##### **How tCgC and tCrC Work Together in the Epilogue**

The shapes `((_2, _2, _8), _2, _2)` are identical between the two — same logical positions on both sides. The strides differ: `tCrC`'s describe register layout, `tCgC`'s describe GMEM addresses. When the epilogue runs `axpby(alpha, tCrC, beta, tCgC)`, it walks both tensors in lockstep, and for each shared logical index it copies one register's value to the corresponding GMEM address. This is exactly what shape-matching enables: the same iteration covers both views, the strides handle the source-vs-destination address arithmetic, and each thread independently writes its 128 values to its 128 GMEM positions. No cross-thread coordination needed — the Z-pattern guarantees that all 128 threads' positions tile the $$64 \times 64$$ atom exactly without overlap or gap.

---

#### **Part 4 (Bonus): How 128 Threads Map to the 64×64 Output Atom**

##### **The Basic Count**

One wgmma atom produces a $$64 \times 64$$ output tile = 4096 values. The warpgroup has 128 threads. Each thread holds 32 register values (per atom). $$128 \times 32 = 4096$$ ✓ — so the threads collectively own every output position exactly once, with no overlap and no gaps.

##### **Warp-Level Row Partitioning**

The 128 threads form 4 warps of 32. The PTX-defined Z-pattern partitions the atom into 4 horizontal bands of 16 rows each, one per warp:

- Warp 0 owns rows 0–15
- Warp 1 owns rows 16–31
- Warp 2 owns rows 32–47
- Warp 3 owns rows 48–63

So if you are warp $$w = \text{threadIdx.x} / 32$$, your warp owns rows $$16w$$ through $$16w + 15$$. The warp-to-band assignment is a clean horizontal stripe.

##### **Thread-Level Coverage Within a Warp's 16×64 Band**

The 32 threads of one warp tile their $$16 \times 64$$ band by repeating an $$8 \times 8$$ "core" pattern. In the first 8-column group (columns 0–7), the ownership map for warp 0 looks like:

```
       col0  col1  col2  col3  col4  col5  col6  col7
row 0:  T0    T0    T1    T1    T2    T2    T3    T3
row 1:  T4    T4    T5    T5    T6    T6    T7    T7
row 2:  T8    T8    T9    T9    T10   T10   T11   T11
row 3:  T12   T12   T13   T13   T14   T14   T15   T15
row 4:  T16   T16   T17   T17   T18   T18   T19   T19
row 5:  T20   T20   T21   T21   T22   T22   T23   T23
row 6:  T24   T24   T25   T25   T26   T26   T27   T27
row 7:  T28   T28   T29   T29   T30   T30   T31   T31
row 8:  T0    T0    T1    T1    T2    T2    T3    T3    ← rows 8–15 mirror rows 0–7
row 9:  T4    T4    T5    T5    T6    T6    T7    T7
...
row 15: T28   T28   T29   T29   T30   T30   T31   T31
```

Each thread $$t$$ owns 4 cells in this column-group: row pair $$\{t/4,\; t/4 + 8\}$$ × column pair $$\{2(t \bmod 4),\; 2(t \bmod 4) + 1\}$$. Together the 32 threads cover the $$16 \times 8$$ region exactly once: 4 cells per thread × 32 threads = 128 cells. ✓

##### **Column-Group Repetition**

This same $$16 \times 8$$ ownership pattern repeats 8 times across columns 0–7, 8–15, 16–23, …, 56–63. Each thread therefore owns $$4 \times 8 = 32$$ cells per warp's $$16 \times 64$$ band — which exactly matches the per-thread value count. The "outer 8" mode in the `(2, 2, 8)` factorization comes from these 8 repetitions of the column-group; the inner two 2s come from the row pair and column pair within one column-group.

##### **Scaling to a CTA Tile**

For our $$128 \times 128$$ CTA tile with $$\text{MMA\_M} = \text{MMA\_N} = 2$$, the CTA tile contains 4 wgmma atoms (2 along $$M$$, 2 along $$N$$). Each thread owns 32 cells per atom × 4 atoms = 128 cells total, spread across 4 disjoint regions of the GMEM output. That is why `tCrC`'s cosize per thread is 128 register slots and `tCgC` has 128 corresponding GMEM addresses.

##### **Why the Epilogue Needs No Synchronization**

The Z-pattern is a **partition** of the atom — every output cell has exactly one owner thread, with no overlap. So when the epilogue runs `axpby(alpha, tCrC, beta, tCgC)`, each of the 128 threads independently walks its 128 owned positions, writing each register value to its corresponding GMEM address. Since no two threads write to the same cell, there are no race conditions, no atomics, no synchronization needed — all $$128 \times 128 = 16384$$ writes happen in parallel across the warpgroup, and together they fill the full $$128 \times 128$$ CTA tile.

##### **The Indexing Chain**

Putting it all together, the per-thread mapping pipeline is:

1. `threadIdx.x` → warp index (`/32`) and lane index (`%32`).
2. Warp index → which 16-row horizontal band of each atom this thread participates in.
3. Lane index → which 4 cells (row pair × column pair) within each of the 8 column-groups of that band.
4. `tCgC` for this thread index → 128 GMEM byte offsets corresponding to those 128 owned cells across the CTA tile.
5. `tCrC` for this thread index → 128 register slots holding the computed values destined for those cells.

The shape of `tCgC` and `tCrC` being identical — `((2, 2, 8), 2, 2)` — is what lets a single iteration walk both views in lockstep, with the strides handling the source-vs-destination address arithmetic.

---

#### **The `cute::gemm` Call, Revisited**

##### **Part 1 of 4 — V Mode and `cute::gemm`'s Two-Level Dispatch**

###### **What V Means in the Dispatch Shape**

In the dispatch shape $$(V, M, K) \times (V, N, K) \Rightarrow (V, M, N)$$, **V** is CuTe shorthand for **value** — the inner mode of an MMA-partitioned tensor that holds one MMA atom's worth of data, per thread. It is *not* the number of wgmma issues. Mapping to the tensors we have already decoded:

- `tCrA` mode-0 `_1` is V (one descriptor per atom).
- `tCrB` mode-0 `_1` is V (same).
- `tCrC` mode-0 `(_2, _2, _8)` is V (32 register values per thread per atom).

The shape comment reads mode-0 of each tensor as V, and the outer modes as the iteration axes that `cute::gemm` loops over.

###### **Two-Level Dispatch**

Two layers of dispatch happen inside `cute::gemm`:

- **Outer-loop level.** The general overload `cute::gemm(tiled_mma, tCrA, tCrB, tCrC)` sees the full 4D shapes — `(V, MMA_M, MMA_K)` for A, `(V, MMA_N, MMA_K)` for B, `(V, MMA_M, MMA_N)` for C — and loops over MMA\_M, MMA\_N, MMA\_K. For our running problem, that is $$2 \times 2 \times 4 = 16$$ outer iterations.
- **Inner dispatch level.** At each outer iteration, the $$(m, n, k)$$ coordinates are fixed and what remains is just the V mode on each tensor — one wgmma atom's worth of data. This reduces to the $$(V) \times (V) \Rightarrow (V)$$ overload of `cute::gemm`, which calls `mma_unpack`. `mma_unpack` extracts the per-thread operand registers from the V modes and invokes the MMA atom's `fma` method (the inline PTX block).

So "the dispatch shape $$(V) \times (V) \Rightarrow (V)$$" means: once the loops over MMA\_M/N/K are stripped away, the remaining work is one wgmma issue, and V is the per-thread, per-atom data unit that gets passed into the asm.

To be explicit: V is **not** the number of wgmma issues. The number of wgmma issues per `cute::gemm` call is $$\text{MMA\_M} \times \text{MMA\_N} \times \text{MMA\_K} = 2 \times 2 \times 4 = 16$$. V is the per-thread data inside one issue — not a count of issues but the granularity of one issue.

---

##### **Part 2 of 4 — Reading the Inline PTX Block**

###### **Overall Structure**

The `fma` method wraps a single `wgmma.mma_async` instruction in inline PTX assembly. Inline asm has three colon-separated sections: the instruction string, the output operands, and the input operands. The instruction string contains `%N` placeholders that get substituted at compile time with register names (or immediate values), where the indexing follows the order in which operands are listed across the output section first, then the input section.

###### **The Instruction String, Line by Line**

```asm
"{\n"
```

Opens a PTX scope, like `{ }` in C. Variables declared inside are local to this block.

```asm
".reg .pred p;\n"
```

Declares a 1-bit predicate register named `p`. Predicates are PTX's true/false registers used for conditional execution of subsequent instructions.

```asm
"setp.ne.b32 p, %18, 0;\n"
```

Set predicate. Compares `%18` (which will be `scale_D`, a 32-bit value) against 0 using the not-equal comparison on byte-32. Result lands in `p`. So `p = (scale_D != 0)`. This predicate becomes the `enable_input_d` flag for wgmma — when `p` is true, the accumulator is read in ($$D = A \cdot B + D$$); when false, the accumulator starts at zero ($$D = A \cdot B$$).

```asm
"wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
```

The actual instruction. Decoding the qualifiers:

- `wgmma.mma_async` — the asynchronous warpgroup MMA.
- `.sync.aligned` — semantic constraints: the warpgroup must execute this collectively (`sync`) and the participating threads must be properly warp-aligned within the warpgroup (`aligned`).
- `.m64n64k16` — the atom shape.
- `.f16.f16.f16` — datatypes for $$D$$, $$A$$, $$B$$ respectively (with $$C$$'s type matching $$D$$).

```asm
"{ %0, %1, ..., %15 },"
```

The 16 $$D$$ operands (accumulator outputs). Each is a 32-bit register. $$16 \times 32$$ bits = 512 bits = 32 FP16 values per thread (each 32-bit register packs 2 FP16 values). This matches the 32 accumulator values per thread per atom from the Z-pattern analysis.

```asm
" %16,"
" %17,"
```

`desc_a` and `desc_b` — the 64-bit matrix descriptors for $$A$$ and $$B$$.

```asm
" p,   %19, %20, %21, %22;"
```

The remaining flag operands:

- `p` — the predicate from earlier (the `enable_input_d` / scale-d flag).
- `%19` = `scaleA` (1 or -1, negate $$A$$).
- `%20` = `scaleB` (1 or -1, negate $$B$$).
- `%21` = `tnspA` (0 or 1, transpose $$A$$ — only meaningful for FP16/BF16).
- `%22` = `tnspB` (0 or 1, transpose $$B$$).

```asm
"}\n"
```

Closes the PTX scope.

###### **The Output Operand Section**

```cpp
: "+r"(d00), ..., "+r"(d15)
```

`"+r"` means "32-bit register, read-and-write." Each `d##` is a `uint32_t&` reference passed in by the caller. The compiler allocates 16 32-bit registers, places the variables' current values in them before the instruction (so the accumulator can be read in), runs the wgmma, then copies the results back to the variables. These map to placeholders `%0` through `%15`.

###### **The Input Operand Section**

```cpp
: "l"(desc_a),
  "l"(desc_b),
  "r"(int32_t(scale_D)),
  "n"(int32_t(scaleA)),
  "n"(int32_t(scaleB)),
  "n"(int32_t(tnspA)),
  "n"(int32_t(tnspB))
```

Constraint codes:

- `"l"` = 64-bit register (descriptors live in `uint64_t` registers).
- `"r"` = 32-bit register (`scale_D` is a runtime value).
- `"n"` = 32-bit immediate constant (the four scale/transpose flags are compile-time template parameters baked into the instruction encoding, not register loads).

These map to placeholders `%16` through `%22`.

###### **Summary**

The whole block says: "Set predicate from `scale_D`, then issue one wgmma instruction that consumes 16 accumulator registers (read+write), 2 descriptor registers (read), 1 runtime scale register (read), and 4 immediate constants (compile-time)." That is the entire payload for one wgmma atom.

---

##### **Part 3 of 4 — The Call Chain from `cute::gemm` to PTX**

###### **The Layered Call Structure**

When user code calls `cute::gemm(tiled_mma, tCrA, tCrB, tCrC)`, the work flows through four layers before hitting the hardware:

1. **`cute::gemm(tiled_mma, tCrA, tCrB, tCrC)`** — top level. The user's call. Sees the full 4D tensor shapes.

2. **Outer-loop overload.** Loops over $$(m, n, k)$$ across $$\text{MMA\_M} \times \text{MMA\_N} \times \text{MMA\_K}$$. At each iteration it slices:
   - `tCrA(_, m, k)` → produces one `desc_a` for this iteration (the descriptor iterator computes it on the fly from the strides).
   - `tCrB(_, n, k)` → produces one `desc_b`.
   - `tCrC(_, m, n)` → produces 16 `uint32_t&` accumulator references for this $$(m, n)$$ atom slot.

   Passes these into the $$(V) \times (V) \Rightarrow (V)$$ dispatch overload.

3. **$$(V) \times (V) \Rightarrow (V)$$ overload** calls `mma_unpack(tiled_mma, slice_A, slice_B, slice_C)`. `mma_unpack` is the glue that:
   - Reads the descriptor values from `slice_A` and `slice_B`.
   - Reads the 16 `uint32_t` accumulator references from `slice_C`.
   - Calls `MMA_Atom::fma(desc_a, desc_b, d00, d01, ..., d15, scale_D)`.

4. **`MMA_Atom::fma(...)`** — the inline PTX block from Part 2. Takes the unpacked operands and dispatches the actual `wgmma.mma_async` instruction.

The hardware receives the PTX, the Tensor Core executes the matmul asynchronously, and writes results back to the accumulator registers.

###### **Runtime Versus Compile-Time Parameter Flow**

Two parallel paths feed the asm block, distinguished by when their values are determined:

- **Runtime values** flow `cute::gemm` → outer loop → `mma_unpack` → `fma`: the descriptors (`desc_a`, `desc_b`) and the accumulator register references (`d00`–`d15`). These change every iteration as the loop walks over $$(m, n, k)$$.
- **Compile-time values** are baked into the MMA Atom type from the start — `scaleA`, `scaleB`, `tnspA`, `tnspB`. They come from the template parameters of the MMA Atom struct (e.g., `SM90_64x64x16_F16F16F16_SS<MajorA, MajorB>` sets `tnspA` and `tnspB` based on the majorness flags). The `fma` method reads them as `static constexpr` values from its enclosing class scope and bakes them into the asm as immediate constants via the `"n"` constraint. Constant for the entire kernel.
- **`scale_D`** is technically runtime but typically constant within a `cute::gemm` call. Default `One` means accumulate ($$D = A \cdot B + D$$); the user can pass `Zero` for the first wgmma to clear the accumulator ($$D = A \cdot B$$).

###### **Why the Constraint Codes Matter**

The `"l"`, `"r"`, `"n"` distinction is not cosmetic — it tells the compiler how to materialize each operand:

- `"l"(desc_a)` says "this is a 64-bit value that needs to live in a register at the moment the wgmma issues." The compiler emits a load (or keeps it live in a register if already there).
- `"r"(scale_D)` says "this is a 32-bit runtime value, also in a register."
- `"n"(scaleA)` says "this is a compile-time integer constant, fold it directly into the instruction's binary encoding." No register is used; the value becomes part of the wgmma opcode itself.

That last point is why `scaleA`, `scaleB`, `tnspA`, `tnspB` cannot be runtime variables — wgmma's PTX encoding reserves dedicated bits for these flags, and they have to be known when the SASS is generated. This is also why the majorness of $$A$$ and $$B$$ is part of the MMA Atom's type (a template parameter) rather than a function argument.

---

##### **Part 4 of 4 — What `cute::gemm` Actually Loops Over Per Stage**

###### **Work Per Pipeline Stage**

Per stage, we have one `sA` ($$128 \times 64$$) and one `sB` ($$128 \times 64$$) sitting in SMEM. The work `cute::gemm` does for this stage is: compute the partial outer-product $$sA \cdot sB^T$$ and accumulate it into the $$128 \times 128$$ register tile `rC`. That is $$128 \times 128 \times 64 = 1{,}048{,}576$$ multiply-adds, distributed across the warpgroup.

One wgmma atom only covers $$64 \times 64 \times 16$$. So tiling the per-stage work over the atom shape gives:

- $$M$$ direction: $$128 / 64 = 2$$ atoms → MMA\_M = 2
- $$N$$ direction: $$128 / 64 = 2$$ atoms → MMA\_N = 2
- $$K$$ direction: $$64 / 16 = 4$$ atoms → MMA\_K = 4

Total wgmma issues per stage: $$2 \times 2 \times 4 = 16$$. This is the loop nest `cute::gemm` runs.

###### **The Loop Nest, Conceptually**

```python
for m in 0..MMA_M:           # 2 iterations
  for n in 0..MMA_N:         # 2 iterations
    # rC[m, n] is the (m, n)-th 64x64 sub-tile of the accumulator
    for k in 0..MMA_K:       # 4 iterations
      # one wgmma issue:
      #   reads desc_a from tCrA[m, k]   (64x16 of sA)
      #   reads desc_b from tCrB[n, k]   (64x16 of sB)
      #   accumulates into rC[m, n]      (64x64 of accumulator)
      wgmma(desc_a, desc_b, rC[m, n])
```

Each pass of the innermost K-loop issues one wgmma. After all 4 K-iterations, `rC[m, n]` holds the full inner product over the $$K = 64$$ slice for that $$(m, n)$$ accumulator slot.

###### **What Each Loop Level Means Physically**

- **MMA\_M loop (2 iterations):** which 64-row band of $$A$$ (and of the $$M$$-axis of $$C$$) we are working on.
- **MMA\_N loop (2 iterations):** which 64-column band of $$B$$ (and of the $$N$$-axis of $$C$$) we are working on.
- **MMA\_K loop (4 iterations):** within a fixed $$(m, n)$$ slot, sweep through the $$K$$-axis 16 elements at a time, accumulating partial products.

The MMA\_K loop is the **reduction loop** — same accumulator register slot, multiple K-slice contributions added in. The MMA\_M and MMA\_N loops are **independent output regions** — different accumulator slots, no interaction between iterations.

###### **A Concrete Iteration, Step by Step**

Take iteration $$(m=0, n=0, k=0)$$:

- `tCrA(_, 0, 0, read_pipe)` is a descriptor pointing at the $$64 \times 16$$ sub-tile of `sA` covering rows 0–63, cols 0–15.
- `tCrB(_, 0, 0, read_pipe)` is a descriptor for the $$64 \times 16$$ sub-tile of `sB` covering rows 0–63 of $$N$$, cols 0–15 of $$K$$.
- The wgmma fires. The Tensor Core reads both sub-tiles directly from SMEM via the descriptors, computes the $$64 \times 64$$ partial product, and adds it into `rC[0, 0]` — each thread's 32 register values for the $$(0, 0)$$ atom.

Next iteration $$(m=0, n=0, k=1)$$:

- Same `rC[0, 0]` accumulator (we are still reducing over $$K$$).
- New descriptors pointing 16 K-elements further into `sA` and `sB` (rows 0–63 of $$A$$, cols 16–31; rows 0–63 of $$B$$'s N-axis, cols 16–31 of $$K$$).
- Tensor Core computes another $$64 \times 64$$ partial product, accumulates into `rC[0, 0]`.

After $$k = 0, 1, 2, 3$$ the $$(m=0, n=0)$$ accumulator slot has the complete $$64 \times 64$$ partial product for the whole $$K = 64$$ slice. Move on to $$(m=0, n=1)$$ — different $$N$$ region of $$B$$, different accumulator slot.

###### **Where the Loop Physically Lives**

The loops are inside the templated `cute::gemm` overload. They get **fully unrolled at compile time** because MMA\_M, MMA\_N, MMA\_K are static integers (`Int<2>`, `Int<2>`, `Int<4>` in our case). The generated SASS is just 16 wgmma instructions back-to-back, with each instruction's operand registers and descriptor offsets precomputed by the compiler from the loop indices. No runtime loop overhead — what looks like a 3-deep loop in source is a flat sequence of 16 wgmma issues in machine code.

###### **Full Lifecycle for One Pipeline Stage**

1. TMA loads `sA` and `sB` for this stage from GMEM into SMEM (handled by the producer side, separately).
2. `cute::warpgroup_arrive()` — fence to make sure prior register writes are visible.
3. `cute::gemm(...)` — issues all 16 wgmma instructions in sequence. The Tensor Core handles them asynchronously; the 16 issues are queued up.
4. `cute::warpgroup_commit_batch()` — bundles those 16 issues into a wgmma group.
5. `cute::warpgroup_wait<0>()` — wait for that group to finish.
6. `rC` now holds the accumulated result for this K-slice. The kernel mainloop moves to the next pipeline stage and repeats.

###### **The Two Layers of K-Loop in the Kernel**

This is the place where it is easy to lose track of which loop is which. There are two nested K-loops:

- **Outer K-loop in the kernel mainloop** (in `gemm_device`): walks across the whole $$K$$-axis of the GMEM matrices, one pipeline stage at a time. This is the `while (k_tile_count > -K_PIPE_MAX)` loop in the reference code.
- **Inner K-loop inside `cute::gemm`** (the MMA\_K loop): walks within one stage's `sA`/`sB` (which has $$K = 64$$ here), issuing 4 wgmma's that span the 64 elements 16 at a time.

If $$K = 1024$$ and $$bK = 64$$, the outer loop runs 16 times and each iteration runs $$\text{MMA\_K} = 4$$ inner wgmma issues per $$(m, n)$$ atom slot — $$16 \times 16 = 256$$ wgmma's total per CTA, all accumulating into the same `rC`. After the outer K-loop finishes, `rC` holds the full $$M \times N$$ output tile for this CTA, ready for the epilogue.

---

#### **Synchronization for WGMMA**

##### **Part 1 of 4 — Proxies, the Memory Consistency Model, and What Runs Concurrently**

###### **Proxies: Async vs Generic**

A **proxy** in PTX terminology is a memory access pathway — a class of instructions that share the same coherence and ordering rules. NVIDIA introduced the term to distinguish traditional CUDA memory accesses from the new asynchronous hardware engines that bypass the SM's normal load/store path.

- **Generic proxy** is the conventional path: `ld.global`, `st.global`, `ld.shared`, `st.shared`, regular register reads and writes — everything CUDA has used since the start. Goes through the SM's main memory pipeline.
- **Async proxy** is the asynchronous path: operations executed by dedicated hardware engines that run in parallel with the SM's main pipeline. On Hopper this includes TMA (`cp.async.bulk` and friends) for async bulk GMEM↔SMEM copies, and `wgmma.mma_async` for async Tensor Core compute. These engines have their own internal queues and ordering, separate from the SM's instruction stream — a thread issues the instruction (just kicks it off), and the engine does the work in the background.

The reason this distinction matters is that the two proxies **do not automatically see each other's writes**. If a generic-proxy `st.shared` writes to SMEM and then a wgmma (async proxy) tries to read that SMEM, the wgmma may not see the store yet — even though in program order the store came first. Making a generic-proxy write visible to the async proxy requires an explicit `fence.proxy.async`. In the reference code this is not needed because SMEM is filled by TMA (also async proxy), so producer and consumer are both on the async side. Only `wgmma.fence` is needed, which handles ordering of accumulator register writes across the boundary into the async Tensor Core.

###### **The Memory Consistency Model**

A memory consistency model is the formal contract that says: given a multi-threaded program, what sequences of read/write outcomes is the hardware allowed to produce? It is how you reason about which orderings are guaranteed and which require explicit synchronization. Real hardware (including GPUs) uses weak models — they allow many surprising reorderings — and provides fence/barrier instructions you use to enforce ordering when you need it.

NVIDIA's PTX memory consistency model defines the proxy concept, the scopes (thread, warp, block, GPU, system), the fences needed for cross-proxy and cross-scope ordering, and the visibility semantics of async operations. For wgmma specifically, the relevant rule is that `wgmma.mma_async` executes in the async proxy and requires `wgmma.fence` to establish ordering with respect to prior generic-proxy register writes touching the same accumulator addresses. The cookbook recipe (fence → `mma_async` → commit → wait) is what satisfies the model's requirements without having to reason about the spec directly each time.

###### **What Runs Concurrently Due to WGMMA's Async Nature**

Several categories of work can overlap with an in-flight wgmma:

- **Other wgmma instructions** — multiple wgmma's can be in flight on the same Tensor Core simultaneously, as long as they are independent or share an accumulator with matching shape (the hardware orders the latter case naturally). This is what enables pipelining across the MMA\_K loop.
- **TMA loads** — the most important overlap. While the Tensor Core consumes stage $$i$$'s data, TMA can already be loading stages $$i+1$$ and $$i+2$$ from GMEM into SMEM. This is the entire point of the circular SMEM buffer.
- **Generic-proxy compute on the SM's main pipeline** — non-wgmma arithmetic, address computations, control flow, register copies. The CUDA cores can run their own work in parallel with the Tensor Core.
- **Other async-proxy operations** — async barriers, additional async copies.
- **Across warpgroups in the same CTA** — in warp-specialized kernels (one producer warpgroup, one consumer warpgroup), the two can do entirely independent work in parallel.

The async design is what lets the kernel hide memory and compute latency: issue the wgmma, return immediately, do something useful, and only wait when the result is actually needed.

---

##### **Part 2 of 4 — Fence, Commit, Wait: Roles and the Cost of Getting It Wrong**

###### **How an Async Result Actually Comes Back**

When a wgmma instruction issues, it specifies output registers (the accumulator slots `d00..d15`). The thread does not block waiting for the result — it returns immediately and moves on to the next instruction. The Tensor Core works in parallel and, when it finishes, writes the result directly into those same registers. There is no separate "fetch result" step.

The crucial constraint is that **during the in-flight period, the thread must not read or write the accumulator registers**. Those slots are owned by the Tensor Core until it finishes. Touching them produces undefined behavior. So the thread can do anything it wants in the meantime — issue more wgmma's, kick off TMA loads, do address arithmetic — as long as it stays away from those specific registers.

When the thread eventually reaches code that needs the result (e.g., the epilogue's `axpby` reading `tCrC`), it must first execute `wgmma.wait_group<0>`. This blocks until the Tensor Core has finished writing. After the wait returns, the registers contain valid data and can be read safely. "The wait happens only when you actually need the result" means: place the wait at the latest possible point in your code, just before the first instruction that consumes the accumulator. Everything between issue and wait runs in parallel with the Tensor Core — that is the latency you are hiding.

###### **The Role of Each Sync Primitive**

- **`cute::warpgroup_arrive()`** → `wgmma.fence.sync.aligned`. Establishes that prior register writes (especially to accumulator slots) are committed and visible before any subsequent wgmma reads or writes them. Without this fence, the Tensor Core might pick up stale values from registers that the issuing thread thought it had updated. In the kernel, the fence sits before `cute::gemm` to make sure things like `clear(rC)` (which writes zeros via generic-proxy register stores) have settled before the async Tensor Core touches `rC`.

- **`cute::warpgroup_commit_batch()`** → `wgmma.commit_group.sync.aligned`. Takes all wgmma's issued so far by this warpgroup that have not been placed into a group yet, and bundles them into a new "wgmma group." This is purely a bookkeeping step — it tells the hardware "treat these $$N$$ wgmma's as one trackable unit for the purpose of waiting." In the example, all 16 wgmma's from `cute::gemm` get bundled into one group.

- **`cute::warpgroup_wait<N>()`** → `wgmma.wait_group.sync.aligned N`. Blocks until at most $$N$$ wgmma groups remain pending — i.e., all groups except the most recent $$N$$ are complete. With $$N = 0$$, you wait for every committed group to finish. After this returns, all the accumulator registers from those groups hold valid final values.

The `<N>` parameter gives you a knob for double-buffering the wgmma groups themselves: you can issue group A, do other compute, issue group B, then `wait<1>` to drain only group A while B is still in flight. FlashAttention-3's GEMM-softmax overlap uses exactly this pattern.

###### **Why Missing Fences Cause Compiler Serialization**

If the compiler cannot prove that consecutive wgmma's are safely ordered, it has to assume the worst — that wgmma #2 might read stale accumulator values written by wgmma #1. To preserve correctness, it inserts a wait between every wgmma, forcing serial execution: issue → wait → issue → wait. This is a major performance hit because it collapses the Tensor Core's pipelining (Part 4 covers this in detail), but the compiler chooses correctness over speed by default.

The blog's warning is that bad sync code does not always crash — it might **silently degrade performance** instead. You write what you think is parallel code, but the compiler quietly serializes it because your fences are missing or wrong. The fix is to write the cookbook pattern explicitly so the compiler can confidently emit back-to-back wgmma issues without inserting safety waits.

###### **The Three Failure Modes for Incorrect Synchronization**

The blog lists them explicitly: (a) subtle race conditions producing wrong results in hard-to-debug ways, (b) compiler serialization wrecking performance, or (c) outright undefined behavior. Each comes from violating a different part of the memory consistency contract — missing the fence between non-wgmma writes and wgmma reads of the same registers (a, c), or letting the compiler hedge against possible races by adding implicit waits (b).

---

##### **Part 3 of 4 — The Same-Accumulator-Shape Exception, and Why the MMA\_K Loop Needs No Internal Fence**

###### **The General Fence Rule and Hopper's Exception**

The general rule is: if any prior code wrote to a register that an upcoming wgmma will read or write, you need `wgmma.fence` between them. Without it, the ordering is undefined, and the Tensor Core may pick up stale values.

Hopper carves out one specific exception. The hardware exception is: **multiple wgmma instructions can be in flight simultaneously without a fence between them, as long as they share the same accumulator (same register slots) AND have the same accumulator shape.** When this condition holds, the Tensor Core internally tracks the dependency between the wgmma's and orders them naturally — wgmma #2 sees wgmma #1's accumulator update without needing software help.

###### **Why "Same Accumulator Shape" Is the Key Qualifier**

The accumulator shape determines two things: the register footprint (how many registers per thread) and the per-thread layout (which register holds which logical output position, per the Z-pattern). Two wgmma's with the same atom shape (say, both $$64 \times 64$$) write to the same 16 registers per thread arranged the same way. The hardware can match them up slot-by-slot — the dependency is unambiguous.

If the accumulator shapes differed, the register footprints would not align cleanly. The hardware would have no clean way to know which register in wgmma #2 corresponds to which register in wgmma #1, so it cannot safely chain them. You would be back to needing an explicit fence to force the compiler/hardware into a serialized ordering.

###### **Why This Matters for the MMA\_K Loop**

Inside `cute::gemm`, the 4 wgmma's of the K-reduction loop (for fixed $$(m, n)$$) all write to the same `rC[m, n]` accumulator slot. Each iteration accumulates a new partial product into the same 16 registers per thread:

```
wgmma #1: rC[m,n] = A[m,k=0] · B[n,k=0] + rC[m,n]   (initially zero)
wgmma #2: rC[m,n] = A[m,k=1] · B[n,k=1] + rC[m,n]
wgmma #3: rC[m,n] = A[m,k=2] · B[n,k=2] + rC[m,n]
wgmma #4: rC[m,n] = A[m,k=3] · B[n,k=3] + rC[m,n]
```

All 4 use the same $$64 \times 64$$ atom shape, so they meet the exception condition. The Tensor Core chains them: wgmma #2 sees wgmma #1's accumulator output before it adds in its own contribution, and so on. No fence between them is needed — and crucially, no fence *should* be inserted, because that would defeat the pipeline overlap (covered in Part 4).

###### **wgmma's Across Different (m, n) Atoms**

What about wgmma's belonging to different $$(m, n)$$ slots — say, wgmma for `rC[0, 0]` followed by wgmma for `rC[0, 1]`? These also need no fence, but for a different reason: they touch **disjoint register sets**. There is no dependency at all to track. The hardware does not need to chain them; they can run fully in parallel as independent operations.

So the 16 wgmma's in `cute::gemm` fall into two categories:

- **wgmma's within the same $$(m, n)$$ slot but different $$k$$** — chained by the same-accumulator exception.
- **wgmma's across different $$(m, n)$$ slots** — independent, no dependency.

Either way, no fence is needed inside `cute::gemm`. The single `cute::warpgroup_arrive()` before `cute::gemm` is enough to handle the external dependency (prior generic-proxy writes like `clear(rC)` settling before the first wgmma reads `rC`).

###### **Where You Do Need a Fence**

The fence is needed at the boundary between non-wgmma register writes and the next wgmma that touches those registers. Common cases:

- After `clear(rC)` (regular CUDA register stores) and before the first wgmma of the mainloop. This is what `cute::warpgroup_arrive()` handles.
- After the epilogue reads `rC` and writes new values into it via non-wgmma code, before any subsequent wgmma. (Does not apply to standard GEMM since the epilogue is the end, but matters for back-to-back GEMM patterns like FlashAttention.)
- After any thread-side modification of the accumulator registers (rare in practice, but possible).

Inside the unrolled `cute::gemm` 16-wgmma sequence, all writes are wgmma → wgmma on either matched accumulators or disjoint registers. The exception covers everything; no internal fence required.

---

##### **Part 4 of 4 — Instruction-Level Parallelism via Tensor Core Pipelining**

###### **What "One wgmma Instruction" Actually Is**

First, clarify a potential confusion: one wgmma PTX instruction is one issue. It does not internally fan out to multiple sub-issues. When the warpgroup executes `wgmma.mma_async.m64n64k16 ...`, that is literally one unit of work handed to the Tensor Core. The "many wgmma's" we keep talking about come from `cute::gemm`'s outer loop unrolling 16 separate wgmma instructions back-to-back — not from one instruction expanding.

###### **Instruction-Level Parallelism via Pipelining**

"Multiple wgmma's in flight simultaneously" does not mean multiple wgmma's running at the same instant on the same multiply unit. It means multiple wgmma's are in **different stages of the Tensor Core's internal pipeline** at the same instant — exactly like how a CPU's instruction pipeline has fetch, decode, execute, and writeback all happening on different instructions in parallel.

Picture the Tensor Core as having internal pipeline stages: load operand → multiply → accumulate → write back. Each stage takes some cycles, and the whole instruction takes the sum of stage latencies (the full latency) to complete. But once stage 1 frees up, the next instruction can enter even while the previous one is still in stages 2, 3, 4.

Without pipelining (purely serial execution):

```
wgmma #1: |load|mul|acc|write|
wgmma #2:                    |load|mul|acc|write|
wgmma #3:                                       |load|mul|acc|write|
total time: 3 × full latency
```

With pipelining:

```
wgmma #1: |load|mul|acc|write|
wgmma #2:      |load|mul|acc|write|
wgmma #3:           |load|mul|acc|write|
total time: 1 full latency + 2 × stage latency  (much less)
```

This is what enables the Tensor Core to sustain near-peak throughput across many back-to-back wgmma issues — the steady-state rate is "one wgmma per stage interval" rather than "one wgmma per full latency," and the gap between those two numbers is exactly the win from pipelining.

###### **How Fences Interact with Pipelining**

If you put a `wgmma.fence` (or worse, `wait_group<0>`) between every wgmma, you force a full pipeline drain before the next instruction can enter:

```
wgmma #1: |load|mul|acc|write|
fence/wait:                  |drain|
wgmma #2:                          |load|mul|acc|write|
fence/wait:                                          |drain|
...
```

You collapse the pipeline back to serial execution and lose the throughput advantage. This is the "compiler serializing wgmma instructions" failure mode from Part 2 — incorrect synchronization forces the compiler to insert these waits as a safety measure, and you go from pipelined throughput to one-at-a-time latency.

###### **Why the Same-Accumulator-Shape Exception Is Critical Here**

Part 3's exception is what enables the pipelining to work for the K-reduction loop. Without it, you would need a fence between every K-step (since every K-step writes to the same accumulator), and each fence would drain the pipeline. The exception is the hardware's way of saying "I have handled this dependency internally — you do not need to drain me, just feed me the next instruction." That is how 4 wgmma's in the K-loop can flow through the pipeline back-to-back, rather than draining after each.

###### **The Broader Async Picture**

Pipelining inside the Tensor Core is one form of overlap. The other forms — TMA loading the next stage while the Tensor Core works on the current one, the SM's CUDA cores doing other compute in parallel — all stack on top. The async design is layered:

- **Within a wgmma group:** Tensor Core pipelining lets multiple wgmma's flow through hardware stages.
- **Across wgmma groups in one warpgroup:** `wait<N>` with $$N > 0$$ lets you keep group A in flight while issuing group B.
- **Across warpgroups in a CTA:** producer-consumer warp specialization runs entirely independent work in parallel.
- **Across the SM's pipelines:** the Tensor Core, the TMA engine, and the CUDA cores all run concurrently.

Each layer hides a different kind of latency. The `cute::warpgroup_arrive` / `commit_batch` / `wait` triplet around `cute::gemm` is what unlocks the innermost layer (Tensor Core pipelining) by giving the compiler enough information to emit back-to-back wgmma issues without inserting safety waits.

---

#### **WGMMA Core Matrices**

##### **References for Swizzling**

The swizzle mechanics are assumed background for this section. If you need to build up intuition for SMEM bank conflicts and the XOR-based address permutation, these are useful references:

- [CUDA Shared Memory Bank — Lei Mao](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)
- [Understanding CuTe Swizzling — Veitner](https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/)
- [SMEM Microbenchmarks — Feldmann](https://feldmann.nyc/blog/smem-microbenchmarks)
- [MatMul Swizzling Section — Aleksa Gordić](https://www.aleksagordic.com/blog/matmul#cpt4)

---

##### **Part 1 of 2 — Core Matrices, Descriptors, and What T Means**

###### **What Core Matrices Are**

A **core matrix** is the hardware's native "read unit" from SMEM — the smallest granule the Tensor Core can address via a matrix descriptor. Every core matrix has a fixed size: **8 elements in the strided direction × 16 bytes in the contiguous direction**. The Tensor Core does not see individual SMEM elements; it sees core matrices. Everything about how wgmma reads SMEM — the descriptor fields, the swizzle requirements, the layout constraints — traces back to core matrices as the fundamental unit.

###### **What T Is**

The blog defines $$T$$ implicitly via $$T \times \text{sizeof(dtype)} = 16$$ bytes. So $$T$$ is the number of elements that fit in one core matrix's contiguous span (= 16 bytes). It varies by dtype:

- FP16 (2 bytes): $$T = 16 / 2 = 8$$ elements
- FP8 (1 byte): $$T = 16 / 1 = 16$$ elements
- TF32 (4 bytes): $$T = 16 / 4 = 4$$ elements

$$T$$ shows up everywhere in the layout table as a stride unit because core matrices are always 16 bytes wide, and strides are measured in elements. One core matrix's contiguous span = $$T$$ elements = 16 bytes, always.

###### **How wA and wB Decompose into Core Matrices**

For K-major operands with FP16 (so $$T = 8$$, and the wgmma atom shape is $$M = 64$$, $$K = 16$$):

- **wA** is $$64 \times 16$$ ($$M \times K$$). The strided direction is $$M$$: 64 elements = 8 core matrices. The contiguous direction is $$K$$: 32 bytes = 2 core matrices (each 16 bytes wide). So wA = $$8 \times 2$$ core matrices.
- **wB** is $$N \times 16$$ ($$N \times K$$). The strided direction is $$N$$: $$N$$ elements = $$N/8$$ core matrices. The contiguous direction is $$K$$: 32 bytes = 2 core matrices. So wB = $$(N/8) \times 2$$ core matrices (though the blog writes it as $$2 \times (N/8)$$ transposed — same count).

You can label each core matrix by its (strided-group, contiguous-group) index. For wA:

```
         K=0..7    K=8..15
M= 0..7:  CM(0,0)   CM(0,1)
M= 8..15: CM(1,0)   CM(1,1)
M=16..23: CM(2,0)   CM(2,1)
...
M=56..63: CM(7,0)   CM(7,1)
```

8 rows × 2 columns = 16 core matrices total. Each is $$8 \times 8$$ elements (= $$8 \times T$$ for FP16).

###### **What LBO and SBO Mean Physically**

The matrix descriptor carries two stride fields that let the Tensor Core navigate between core matrices in the tile:

- **LBO (Leading dimension Byte Offset)** = byte distance between two adjacent core matrices along the $$K$$ (contiguous) direction. From CM(i, 0) to CM(i, 1) — same M-group, next K-group.
- **SBO (Stride dimension Byte Offset)** = byte distance between two adjacent core matrices along the $$M$$ or $$N$$ (strided) direction. From CM(i, j) to CM(i+1, j) — same K-group, next M-group.

With just {start address, LBO, SBO}, the Tensor Core can compute the address of any core matrix: $$\text{addr}(i, j) = \text{base} + i \times \text{SBO} + j \times \text{LBO}$$, where $$i$$ indexes along $$M/N$$ and $$j$$ indexes along $$K$$.

###### **The Five Descriptor Fields**

The 64-bit `GmmaDescriptor` packs:

- **Start address** — SMEM base of the tile (in 16-byte units).
- **LBO** — K-direction core-matrix stride.
- **SBO** — M/N-direction core-matrix stride.
- **Swizzle mode** — which of {none, SW32, SW64, SW128}.
- **Matrix base offset** — alignment correction for when the SMEM base is not aligned to the swizzle period boundary.

###### **What `make_gmma_desc` Does**

Given an SMEM tensor whose layout was built from a canonical GMMA atom + `tile_to_shape`, it reads the layout's strides to compute LBO and SBO in bytes, reads the swizzle function to determine the swizzle mode bits, computes the matrix base offset for alignment, and packs all five fields into a 64-bit `GmmaDescriptor`. This is what gets placed in a register as `desc_a` / `desc_b` and fed to the wgmma instruction. CUTLASS handles the construction automatically — you never manually fill these fields.

###### **Background the Blog Assumes**

The swizzle function `Swizzle<B, M, S>` (the XOR-based address permutation), the SMEM bank-conflict model (32 banks × 4 bytes) that motivates swizzling in the first place, and the bit-level mechanics of how swizzling remaps addresses are all assumed. The blog shows the end result (which layouts are admissible, what strides they have) without deriving why the swizzle works. Lei Mao's swizzle blog and the PTX docs on bank conflicts are the canonical references for the XOR mechanics.

---

##### **Part 2 of 2 — Interleaving, the Admissible Layout Table, and the Cross-Atom Subtlety**

###### **What "Interleaving" Means Concretely**

The question is: given the core matrices that make up wA, in what order do they sit in linear SMEM? The swizzle mode determines this by controlling how many K-direction core matrices are packed together before the next M-group starts.

###### **No-Swizzle (Non-Interleaved) Layout**

You walk all M-groups first, then K-groups. For wA's $$8 \times 2$$ core matrices:

```
Memory order:
CM(0,0), CM(1,0), CM(2,0), ..., CM(7,0),   ← all 8 M-groups for K=0
CM(0,1), CM(1,1), CM(2,1), ..., CM(7,1)    ← all 8 M-groups for K=1
```

To go from one row to the next within a core matrix (stepping +1 in $$M$$ within the same core matrix), you skip $$T$$ elements (because $$K$$ is contiguous, so the next row is $$T$$ elements later). M-inner stride = $$1T$$. To go from CM(0,0) to CM(0,1) (next K-group, same M-group), you skip over 8 core matrices worth of data — that distance is LBO and depends on how many M-groups exist. LBO is "free" (non-compact layouts are allowed — there can be padding). SBO is similarly free. This flexibility is why the blog writes the no-swizzle layout as `((8,m),(T,2)):((1T, SBO),(1, LBO))` with SBO and LBO as unspecified parameters.

###### **32-Byte Swizzle (SW32) — Interleave 2 Core Matrices per Row**

Now CM(i,0) and CM(i,1) sit side-by-side in memory for each M-group $$i$$:

```
Memory order:
[CM(0,0) CM(0,1)] [CM(1,0) CM(1,1)] [CM(2,0) CM(2,1)] ... [CM(7,0) CM(7,1)]
```

Within one row of the interleaved block, the data for one M-row spans both K-groups contiguously: $$T$$ elements from CM(i,0), then $$T$$ elements from CM(i,1), totaling $$2T$$ elements per row. Consequences for strides:

- **K-stride (LBO):** from CM(i,0) to CM(i,1) = $$T$$ elements (they are adjacent). Forced by the packing — no freedom here.
- **M-inner stride:** from row $$r$$ to row $$r+1$$ within a core matrix, you skip the full interleaved row width = $$2T$$ elements (because you have to jump past both CM(i,0)'s and CM(i,1)'s data for this row).

Layout: `((8,m),(T,2)):((2T, SBO),(1, T))`. The K-stride collapses to $$T$$, and the M-inner stride doubles to $$2T$$ compared to no-swizzle.

###### **64-Byte Swizzle (SW64) — Interleave 4 Core Matrices per Row**

SW64 packs 4 K-direction core matrices per M-row. But wA only has 2 K-groups per wgmma atom — so where do the other 2 come from?

This is the key subtlety: **the swizzle period extends across multiple wgmma atoms.** The 4 interleaved core matrices come from 2 consecutive wgmma atoms' K-slices packed together in SMEM. One row of core matrices looks like:

```
[atom0_CM(i,0), atom0_CM(i,1), atom1_CM(i,0), atom1_CM(i,1)]
 \___ wgmma atom 0's K ___/   \___ wgmma atom 1's K ___/
```

That is $$4 \times T = 4T$$ elements per row. Strides:

- **K-stride within one atom:** still $$T$$ (CM(i,0) to CM(i,1) are adjacent).
- **M-inner stride:** $$4T$$ (each row spans 4 interleaved core matrices).

Layout: `((8,m),(T,2)):((4T, SBO),(1, T))`.

This is what the blog means by "one has sets of 2 or 4 WGMMA atom operand tiles stacked side-by-side in the K-direction" and "these core matrices will belong to different WGMMA atoms for 64 and 128-byte swizzle." The layout atom's K-extent is larger than one wgmma atom's $$K$$ for SW64/SW128 because the interleave row spans multiple wgmma atoms. That is also why the blog notes these layouts are non-compact from a single wgmma atom's perspective — one atom's core matrices are not contiguous in memory; they are interleaved with another atom's.

###### **128-Byte Swizzle (SW128) — Interleave 8 Core Matrices per Row**

SW128 packs 8 K-direction core matrices per M-row. wA has 2 K-groups per atom, so this spans 4 consecutive wgmma atoms' K-slices packed together:

```
[atom0_CM(i,0), atom0_CM(i,1), atom1_CM(i,0), atom1_CM(i,1),
 atom2_CM(i,0), atom2_CM(i,1), atom3_CM(i,0), atom3_CM(i,1)]
```

That is $$8 \times T = 8T$$ elements per row. Strides:

- **K-stride within one atom:** $$T$$ (adjacent).
- **M-inner stride:** $$8T$$.

Layout: `((8,m),(T,2)):((8T, SBO),(1, T))`.

###### **The Pattern, Summarized**

| Swizzle | Core matrices interleaved per row | M-inner stride | K-stride (LBO, within atom) | Atoms sharing one interleave row |
|---------|-----------------------------------|----------------|-----------------------------|----------------------------------|
| None    | 1                                 | $$1T$$         | free (non-compact ok)       | 1                                |
| SW32    | 2                                 | $$2T$$         | $$T$$ (forced)              | 1                                |
| SW64    | 4                                 | $$4T$$         | $$T$$ (forced)              | 2                                |
| SW128   | 8                                 | $$8T$$         | $$T$$ (forced)              | 4                                |

The M-inner stride grows with swizzle size because each M-row has to skip over more interleaved K-direction core matrices before reaching the next row. The K-stride collapses to $$T$$ because the interleaving packs K-direction core matrices adjacently. And for SW64/SW128, the interleaving reaches across wgmma atom boundaries — multiple atoms' data is woven together in SMEM, which is why the SMEM layout atom's K-extent is larger than one wgmma atom's $$K$$.

###### **Why Interleaving Helps with Bank Conflicts (Intuition)**

SMEM has 32 banks, each 4 bytes wide. When 32 threads of a warp simultaneously read from SMEM, you want each thread to hit a different bank. Without swizzling, adjacent rows of a core matrix map to the same bank pattern — a warp reading consecutive rows hits massive conflicts. Swizzling applies an XOR between the column address and the row address, scattering each row's accesses across different banks. Interleaving more core matrices per row gives the XOR more address bits to work with, reducing conflicts further. That is why larger swizzle modes (SW128 > SW64 > SW32) generally perform better for wide tiles — more interleaving means more effective scattering.

###### **Connection Back to the GMMA Layout Atoms**

The 8 canonical GMMA layout atoms exist precisely to encode these interleaving patterns. Each atom's contiguous-direction span in bytes equals the swizzle mode's name (SW128 atom → 128 contiguous bytes), which we saw in the SMEM layout constraints section. Now we know why: the contiguous span has to cover one full interleave row — all the K-direction core matrices packed together — so the swizzle function's XOR can operate on a complete period. The atom's K-extent is larger than one wgmma atom's $$K$$ for SW64/SW128 because the interleave row spans multiple wgmma atoms. `tile_to_shape` then replicates this atom across the full SMEM tile, preserving the interleaving pattern at every scale.

###### **MN-Major Admissible Layouts**

The blog also provides the MN-major case for completeness. The same interleaving logic applies but with the roles of $$M$$ and $$K$$ swapped — the contiguous direction is now $$M$$ (or $$N$$) rather than $$K$$, so the interleaving packs M-direction (or N-direction) core matrices together, and the K-stride is what grows with swizzle size. The specific layout shapes differ but the principle is identical: swizzle mode determines how many contiguous-direction core matrices are packed per row, which forces certain strides and potentially interleaves across wgmma atom boundaries.

---

#### **Conclusion: The Full Picture — WGMMA on Hopper, End to End**

##### **The Problem**

You want to compute $$C = A \cdot B$$ on a GPU, where $$A$$ is $$M \times K$$, $$B$$ is $$K \times N$$, and the matrices are large. The computation is parallelized by tiling: launch a grid of CTAs, each responsible for a $$bM \times bN$$ tile of $$C$$, computed by iterating over the $$K$$-axis in $$bK$$-sized chunks. Within each CTA, tiles of $$A$$ and $$B$$ are loaded from GMEM into SMEM, and the Tensor Core does the actual multiply-accumulate. The challenge is: how do you talk to the Hopper Tensor Core correctly and efficiently?

##### **The Instruction**

Hopper's Tensor Core is accessed via `wgmma.mma_async` — an asynchronous, warpgroup-level instruction. A warpgroup is 4 contiguous warps (128 threads), aligned so the first warp's rank is a multiple of 4. The instruction computes $$D = A \cdot B + D$$ (or $$D = A \cdot B$$ with accumulator disabled) on a fixed-shape tile called the **wgmma atom**. The atom shape $$M \times N \times K$$ is constrained: $$M$$ is always 64, $$N$$ is a multiple of 8 from 8 to 256, and $$K$$ is fixed to 32 bytes (so $$K = 16$$ for FP16, 32 for FP8, 8 for TF32). Operand $$B$$ must always come from SMEM; $$A$$ can come from SMEM (SS mode) or registers (RS mode); the accumulator $$D$$ always lives in registers.

##### **The TiledMMA Object**

On the host, you construct a `TiledMMA` object from an MMA Atom — a CUTLASS struct that wraps the specific PTX instruction you want. The atom's name encodes everything: `SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>` wraps `wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16`, with both operands MN-major (NT gemm) and both sourced from SMEM (SS). The two template parameters set the transpose flags (`tnspA`, `tnspB`) that get baked into the PTX as compile-time immediates. For FP16/BF16 you can choose MN-major or K-major freely; for other dtypes (FP8, INT8, TF32) only K-major is supported because the hardware lacks in-instruction transpose for those types.

`size(tiled_mma)` gives the thread count (128 for one warpgroup). The optional `AtomLayoutMNK` argument lets you tile multiple warpgroups across the CTA's output tile — `Layout<Shape<_2,_1,_1>>` puts 2 warpgroups along $$M$$, each computing an independent $$64 \times 64$$ atom, giving 256 threads total.

##### **The SMEM Layouts**

The CTA tile sizes $$bM$$, $$bN$$, $$bK$$ must satisfy two independent constraints simultaneously:

- **The MMA atom constraint:** $$bM$$ divisible by 64 (atom $$M$$), $$bN$$ divisible by N-atom, $$bK$$ divisible by atom-$$K$$ (16 for FP16). This comes from the Tensor Core hardware.
- **The swizzle-mode constraint:** $$bM$$ and $$bK$$ must be divisible by the chosen layout atom's shape. This comes from the SMEM layout machinery.

CUTLASS provides 8 canonical GMMA layout atoms — {MN-major, K-major} × {no-swizzle, SW32, SW64, SW128}. Each is a small CuTe layout (shape + stride + swizzle function) hand-designed to satisfy WGMMA's core-matrix and swizzle requirements. You pick one and use `tile_to_shape` to replicate it across `make_shape(bM, bK, bP)` for `sA` (or `make_shape(bN, bK, bP)` for `sB`), where $$bP$$ is the pipeline stage count. `tile_to_shape` preserves the atom's internal structure and adds outer modes for tiling — column-major walk, M-tiles first, then K-tiles, then pipeline stages. The result is a single 3D SMEM tensor with the pipe dimension outermost, encoding the circular buffer directly in the layout.

The atom's contiguous-direction byte span always equals the swizzle mode's name (128 bytes for SW128, etc.) — this is by design, so one atom covers exactly one swizzle period. Larger swizzle modes interleave more core matrices per row, which scatters SMEM bank accesses more effectively but imposes wider M-strides.

##### **Core Matrices and Descriptors**

Under the hood, the Tensor Core reads SMEM in units of **core matrices** — the smallest addressable granule, always 8 rows (strided) × 16 bytes (contiguous). A wgmma atom's A-tile ($$64 \times 16$$ for FP16) decomposes into $$8 \times 2$$ core matrices; B-tile into $$(N/8) \times 2$$.

In SS mode, the wgmma instruction takes a 64-bit **matrix descriptor** for each operand — a packed register value encoding {SMEM base address, LBO (K-direction core-matrix stride), SBO (M/N-direction core-matrix stride), swizzle mode, alignment offset}. With these five fields, the Tensor Core can compute the address of any core matrix in the tile. CUTLASS's `make_gmma_desc` constructs this automatically from the SMEM layout.

The swizzle mode determines how core matrices are interleaved in memory. For SW128, 8 K-direction core matrices are packed per M-row; for SW64, 4; for SW32, 2; for no-swizzle, 1. This interleaving means that for SW64 and SW128, core matrices from different wgmma atoms are woven together in SMEM — the layout is compact at the tile level but non-compact from any single atom's perspective. The M-inner stride grows with swizzle size ($$1T$$, $$2T$$, $$4T$$, $$8T$$) because each M-row spans more interleaved K-data. The K-stride within an atom collapses to $$T$$ (= 16 bytes in elements) because K-direction core matrices are adjacent within the interleave row.

##### **Partitioning and Fragments**

On the device, `tiled_mma.get_thread_slice(threadIdx.x)` produces a `ThrMMA` object, which is then used to partition the SMEM and output tensors:

- **`tCsA = thr_mma.partition_A(sA)`** — reshapes `sA` into `(MMA, MMA_M, MMA_K, PIPE)`, where MMA is one wgmma atom's worth of data, MMA\_M and MMA\_K count how many atoms tile the SMEM, and PIPE is the stage count. In SS mode this is the same view for every thread (not a per-thread slice) because the Tensor Core reads SMEM via descriptors, not per-thread register fragments.

- **`tCrA = thr_mma.make_fragment_A(tCsA)`** — builds the descriptor-iterator view over the same SMEM. Shape `(_1, MMA_M, MMA_K, PIPE)` — mode-0 is 1 because one descriptor encodes one entire atom. The strides are in 16-byte descriptor-address units, exactly `tCsA`'s strides divided by 8 (for FP16). At runtime the iterator computes one 64-bit descriptor per wgmma issue on the fly, rather than precomputing all of them.

- **`tCgC` and `tCrC` — the output.** These are per-thread: each of 128 threads owns 32 values per $$64 \times 64$$ atom ($$4096 / 128 = 32$$), arranged in the PTX-specified Z-pattern. The shape `((_2,_2,_8), MMA_M, MMA_N)` reflects three nested step-patterns (column-pair, row-pair-by-8, 8-column-groups) that cannot be captured by a single stride, hence the `(2,2,8)` factorization. `tCgC` strides map these to GMEM addresses; `tCrC` strides map them to register slots. The epilogue walks both in lockstep.

The Z-pattern partitions the $$64 \times 64$$ atom exactly: 4 warps take 16-row bands each, and within a warp, 32 threads tile their $$16 \times 64$$ band with no overlap. The epilogue needs no cross-thread synchronization.

##### **The `cute::gemm` Dispatch**

`cute::gemm(tiled_mma, tCrA(_, _, _, pipe), tCrB(_, _, _, pipe), tCrC)` runs a compile-time-unrolled loop over $$\text{MMA\_M} \times \text{MMA\_N} \times \text{MMA\_K}$$ (e.g., $$2 \times 2 \times 4 = 16$$ iterations). At each $$(m, n, k)$$ position, it reduces to the $$(V) \times (V) \Rightarrow (V)$$ dispatch overload, which calls `mma_unpack` → `MMA_Atom::fma(desc_a, desc_b, d00..d15, scale_D)`. The `fma` method contains the inline PTX asm that issues one `wgmma.mma_async` instruction with 2 descriptor inputs, 16 accumulator read-write registers, and 4 compile-time immediate flags (`scaleA`, `scaleB`, `tnspA`, `tnspB`). The 16 asm instructions are emitted back-to-back with no waits between them.

##### **Synchronization**

Three primitives bracket the `cute::gemm` call:

- **`cute::warpgroup_arrive()`** → `wgmma.fence.sync.aligned`. Orders prior generic-proxy register writes (like `clear(rC)`) before the async Tensor Core reads those registers. Needed at the boundary between non-wgmma register writes and the first wgmma that touches those registers.
- **`cute::warpgroup_commit_batch()`** → `wgmma.commit_group.sync.aligned`. Bundles all 16 issued wgmma's into one "wgmma group" for tracking purposes.
- **`cute::warpgroup_wait<0>()`** → `wgmma.wait_group.sync.aligned 0`. Blocks until all committed groups are complete. After this, accumulator registers hold valid data.

No fence is needed within the 16 wgmma's because of Hopper's **same-accumulator-shape exception**: multiple wgmma's targeting the same accumulator registers with the same shape are ordered by the hardware automatically. The K-reduction loop (same `rC[m,n]`, different K-slices) and independent $$(m,n)$$ slots (disjoint registers) both fall under this exception.

The async nature means the Tensor Core pipelines internally: multiple wgmma's can be in different stages of the hardware pipeline simultaneously (load → multiply → accumulate → writeback). Inserting unnecessary fences or waits drains this pipeline, collapsing throughput from pipelined to serial — the "compiler serialization" failure mode.

No `fence.proxy.async` is needed because SMEM is filled by TMA (also async proxy), so there is no cross-proxy boundary to fence. If SMEM were filled by ordinary `ld.global`/`st.shared` (generic proxy), the fence would be required.

##### **The Two K-Loops**

A key structural point: there are two layers of K-iteration:

- **Inner K-loop inside `cute::gemm`:** walks within one stage's `sA`/`sB` ($$bK$$ elements), issuing MMA\_K wgmma's per $$(m,n)$$ slot. Fully unrolled at compile time.
- **Outer K-loop in the kernel mainloop:** walks across the full $$K$$-axis of the GMEM matrices, one pipeline stage ($$bK$$ elements) at a time. This is the `while (k_tile_count > ...)` loop. Each iteration loads a new `sA`/`sB` via TMA into the next circular-buffer slot and calls `cute::gemm` on it.

After the outer K-loop finishes, `rC` holds the complete $$M \times N$$ output tile for this CTA, and the epilogue writes it to GMEM.

---

#### **Final Clarifications — Descriptors and Cross-Atom Interleaving**

##### **What Descriptors Describe**

A matrix descriptor does not iterate individual core matrices one at a time. One 64-bit descriptor encodes an **entire wgmma atom's tile** of core matrices — all $$8 \times 2$$ of them for wA, or all $$2 \times (N/8)$$ for wB. The five fields (base address, LBO, SBO, swizzle mode, alignment offset) give the Tensor Core everything it needs to locate every core matrix in the tile on its own. The programmer never navigates core matrices manually.

What the `DescriptorIterator` in CUTLASS iterates over is **atoms** — one descriptor per atom, stepping through $$(MMA\_M, MMA\_K, PIPE)$$ positions. At each position it computes one fresh descriptor, hands it to the Tensor Core via the wgmma instruction, and the Tensor Core internally uses LBO and SBO to walk the core matrices within that atom's tile.

##### **How Different wgmma Atoms' Core Matrices End Up Interleaved in SMEM**

The key insight is that the SMEM layout is determined by `tile_to_shape` at the full SMEM-tile level ($$bM \times bK$$), not per wgmma atom. The swizzle period dictates how much K-direction data is packed into one contiguous memory row, and that period can be wider than one wgmma atom's K-span.

Take SW128 with FP16, K-major. The swizzle period is 128 bytes = 8 × 16-byte core matrices in $$K$$. But one wgmma atom only has 2 core matrices in $$K$$ ($$K = 16 = 2 \times T$$). The remaining 6 slots in the 128-byte row come from the next 3 wgmma atoms along $$K$$. With $$bK = 64$$ (4 wgmma atoms in $$K$$), one M-row in SMEM looks like:

```
One M-row in SMEM (128 bytes = 8 core matrices):
[atom0_CM(i,0)] [atom0_CM(i,1)] [atom1_CM(i,0)] [atom1_CM(i,1)] [atom2_CM(i,0)] [atom2_CM(i,1)] [atom3_CM(i,0)] [atom3_CM(i,1)]
 ← 16 bytes →    ← 16 bytes →    ← 16 bytes →    ← 16 bytes →    ← 16 bytes →    ← 16 bytes →    ← 16 bytes →    ← 16 bytes →
```

All 8 slots belong to 4 different wgmma atoms packed into one contiguous 128-byte row. The atoms are logically contiguous along $$K$$ (atom 0 covers $$K = 0..15$$, atom 1 covers $$K = 16..31$$, etc.), but physically their core matrices are packed side-by-side within each M-row to fill the swizzle period. From atom 0's perspective, its two core matrices CM(i,0) and CM(i,1) are adjacent (K-stride = $$T$$, which is good), but to reach atom 0's next M-row, you skip the entire 128-byte row — because atoms 1, 2, and 3's data sits in between. That is why the M-inner stride is $$8T$$: you are jumping over 6 other atoms' core matrices to get to your own next row.

##### **Cross-Atom Weaving by Swizzle Mode**

The cross-atom weaving is a function of the swizzle mode:

- **No swizzle:** 1 core matrix per row in $$K$$ = fits within one atom. No cross-atom weaving.
- **SW32:** 2 core matrices per row = exactly one atom's K-span. No cross-atom weaving.
- **SW64:** 4 core matrices per row = 2 atoms' K-spans woven together.
- **SW128:** 8 core matrices per row = 4 atoms' K-spans woven together.

The descriptor handles this transparently. Each atom's descriptor encodes the correct LBO and SBO for its core matrices within the interleaved layout. The Tensor Core follows those strides and picks out the right core matrices regardless of what other atoms' data is packed between them. The programmer never has to think about the weaving — the canonical GMMA layout atom + `tile_to_shape` pattern produces the right layout, and `make_gmma_desc` produces the right descriptor. The interleaving is a consequence of satisfying the swizzle period, and its benefit is better SMEM bank-conflict avoidance.

---

#### **Closing Thoughts**

Thanks to Colfax Research for making such in-depth content publicly available — this level of detail on CUTLASS internals is hard to come by and genuinely valuable for anyone trying to understand what happens under the hood of a Hopper GEMM kernel.

Next up, I'll be reading, learning, and understanding Part 2 of this series on **Pipelining** (multistage and warp specialization). Hopefully I'll add similar notes for my future reference and publish them like this one. Stay tuned.
