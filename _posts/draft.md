#### **K2: Tiled online softmax (FA2)**

##### **What's new**

K1 materialized $$S = QK^\top$$ in GMEM and ran three sequential kernels. K2 collapses everything into a single fused kernel that follows the FA2 algorithm directly: per Q-tile we stream KV tiles, recompute $$S$$ in SMEM, run online softmax across rows, and accumulate $$O$$ in registers. The algorithm is exactly the one derived in the FA2 section; the only thing this kernel adds is the threading and SMEM layout to make it run.

We are still on CUDA cores. The two GEMMs ($$QK^\top$$ and $$PV$$) are hand-rolled FP32 dot products. No tensor cores, no TMA, no async pipelining; those land in K3 and K4. The point of K2 is to verify the FA2 algorithm end-to-end in a form that reads top to bottom.

##### **Block sizes and the threading model**

```python
Br = 64
Bc = 64
THREADS = 256
THREADS_PER_ROW = 4
```

One CTA handles one Q-tile of $$B_r = 64$$ rows for one batch element. 256 threads split into 64 row groups of 4 threads each. Each row group owns one Q-row of the tile.

Why 4 threads per row? It ties three things together at the same constant:

- **Q-row ownership.** $$\text{THREADS} / \text{THREADS\_PER\_ROW} = 256 / 4 = 64 = B_r$$. Each row group is responsible for one row of the tile, no leftover.
- **Output sharding along $$d$$.** Each row's $$d$$ output channels are split across the 4 threads. For $$d = 128$$, each thread holds $$128 / 4 = 32$$ FP32 values of $$O$$ in registers. For $$d = 64$$, it's 16.
- **Row-scan partitioning across $$B_c$$.** When reducing the $$B_c = 64$$ columns of an $$S$$ row for max and sum, each of the 4 threads scans $$64 / 4 = 16$$ elements, then a sub-warp shuffle collapses the four partials.

The per-row softmax state $$(m_i, \ell_i)$$ lives in registers, replicated across the 4 threads of the group. They all compute the same reduced max and sum, so they all see the same values. The output accumulator $$O_\text{acc}$$ is the only thing sharded, along $$d$$.

```python
row         = tidx // THREADS_PER_ROW   # 0..63, which Q-row
col_group   = tidx %  THREADS_PER_ROW   # 0..3,  which 1/4 of the row
d_col_start = col_group * (d // THREADS_PER_ROW)
```

##### **SMEM layout**

```python
sQ_layout  = cute.make_layout((Br, d),  stride=(d, 1))    # BF16
sKV_layout = cute.make_layout((Bc, d),  stride=(d, 1))    # BF16, reused K then V
sS_layout  = cute.make_layout((Br, Bc), stride=(Bc, 1))   # FP32
```

For $$B_r = B_c = 64$$ and $$d = 128$$: 16 KB + 16 KB + 16 KB = 48 KB. Comfortably under H100's 228 KB SMEM/SM. We reuse one buffer for $$K_j$$ and $$V_j$$ because $$K_j$$ is finished with the moment $$S$$ has been exponentiated into $$P$$ (which now sits in `sS`); $$V_j$$ then overwrites the buffer.

Layouts are plain row-major. No swizzling, which means SMEM bank conflicts are not addressed here. K3 and K4 introduce swizzled layouts.

##### **Mapping the FA2 algorithm to code, phase by phase**

From the FA2 section: one CTA handles one Q-tile $$Q_i$$, initialises $$m_i = -\infty$$, $$\ell_i = 0$$, $$O_i = 0$$, then for each KV tile $$j$$ computes $$S_{ij}$$, runs the online softmax update, and accumulates into $$O_i$$. Finally divides by $$\ell_i$$. The kernel body follows that structure literally, in three phases.

**Phase 1: load Q once, init state.**

```python
m_i = -inf
l_i = 0.0
for c in range_constexpr(cols_per_thread_O):
    O_acc[c] = 0.0

for it in range_constexpr(0, Br * d, THREADS):
    ij = it + tidx
    sQ[ij // d, ij % d] = gQ[q_row_start + ij // d, ij % d]
cute.arch.sync_threads()
```

The Q-tile is loaded once and kept in SMEM for all KV iterations. The loop is a flat thread-strided scan: 256 threads collectively load $$B_r \cdot d$$ BF16 elements. For $$d = 128$$ each thread loads 32 elements; for $$d = 64$$ it loads 16. No coalescing optimisation; just one element per thread per inner-iteration. K3 replaces this whole loop with a single TMA call.

**Phase 2a: load $$K_j$$ into SMEM.** Same shape as the Q load:

```python
for it in range_constexpr(0, Bc * d, THREADS):
    ij = it + tidx
    sKV[ij // d, ij % d] = gK[kv_row_start + ij // d, ij % d]
cute.arch.sync_threads()
```

**Phase 2b: $$S = QK^\top \cdot \text{scale}$$ on CUDA cores.** Each thread takes a strided subset of the $$B_r \cdot B_c = 4096$$ output elements (16 elements per thread) and computes each as a $$d$$-length FP32 dot product. This is the most obviously inefficient piece of K2: scalar FMAs in registers with unswizzled SMEM reads, all on the FP32 pipe. K4 replaces these two nested loops with a single WGMMA call.

```python
for it in range_constexpr(0, Br * Bc, THREADS):
    ij = it + tidx
    si, sj = ij // Bc, ij % Bc
    dot = 0.0
    for k in range_constexpr(d):
        dot += sQ[si, k].to(f32) * sKV[sj, k].to(f32)
    val = dot * scale
    if const_expr(causal):
        if kv_row_start + sj > q_row_start + si:
            val = -inf
    sS[si, sj] = val
cute.arch.sync_threads()
```

The causal mask is per-element: `kv_row_start + sj` is the absolute KV position, `q_row_start + si` is the absolute Q position, and any future KV is set to $$-\infty$$ before the softmax sees it.

**Phase 2c: online softmax row update.** This is the FA2 recurrence verbatim. Compute the row max of $$S_{ij}$$, update $$m_i$$, rescale $$O_i$$, exponentiate $$S$$ in place into $$P$$, update $$\ell_i$$.

```python
local_max = -inf
for c_it in range_constexpr(cols_per_thread_S):    # 16 iters
    c = col_group + c_it * THREADS_PER_ROW
    local_max = fmax(local_max, sS[row, c])
row_max = cute.arch.warp_reduction_max(local_max,
                                       threads_in_group=THREADS_PER_ROW)

m_new = fmax(m_i, row_max)
alpha = exp(m_i - m_new)

for c in range_constexpr(cols_per_thread_O):
    O_acc[c] *= alpha

local_sum = 0.0
for c_it in range_constexpr(cols_per_thread_S):
    c = col_group + c_it * THREADS_PER_ROW
    p = exp(sS[row, c] - m_new)
    sS[row, c] = p
    local_sum += p
row_sum = cute.arch.warp_reduction_sum(local_sum,
                                       threads_in_group=THREADS_PER_ROW)

l_i = alpha * l_i + row_sum
m_i = m_new
cute.arch.sync_threads()
```

Tying back to the FA2 section's algorithm box: `row_max` is $$\tilde m_i^{(j)} = \max_c S_{ij}[c]$$, `m_new` is $$m_i^{(j)} = \max(m_i^{(j-1)}, \tilde m_i^{(j)})$$, `alpha` is the rescaling factor $$e^{m_i^{(j-1)} - m_i^{(j)}}$$, and `row_sum` is $$\sum_c P_{ij}[c]$$. After the update, `sS` no longer holds $$S$$; it holds $$P = \exp(S - m_\text{new})$$.

Note we exponentiate every element of the row, including the ones set to $$-\infty$$ by the causal mask. $$\exp(-\infty) = 0$$, so they contribute nothing to the sum and nothing to the subsequent $$PV$$ matmul. No branch needed.

**Phase 2d: load $$V_j$$, overwriting $$K_j$$.** Same shape as the K load.

**Phase 2e: $$O_\text{acc} \mathrel{+}= P V_j$$.** Each thread updates its `cols_per_thread_O` columns of $$O$$ by walking $$B_c$$.

```python
for c in range_constexpr(cols_per_thread_O):
    col = d_col_start + c
    pv = 0.0
    for k in range_constexpr(Bc):
        pv += sS[row, k] * sKV[k, col].to(f32)
    O_acc[c] += pv
```

For $$d = 128$$, each thread does $$32 \times 64 = 2048$$ mul-adds per KV iteration; for $$d = 64$$, $$16 \times 64 = 1024$$. The $$P$$ row is read redundantly by all 4 threads of the row group; that's wasted SMEM bandwidth, but in K2 we don't care.

**Phase 3: finalize.**

```python
inv_l = cute.arch.rcp_approx(l_i)
if l_i == 0.0 or l_i != l_i:               # fully-masked row
    inv_l = 1.0
for c in range_constexpr(cols_per_thread_O):
    gO[q_row_start + row, d_col_start + c] = (O_acc[c] * inv_l).to(bf16)
```

`rcp_approx` is the SFU reciprocal; one instruction. The guard handles rows fully masked out by the causal mask (the first row of a Q-tile that begins below the diagonal of a KV-tile sees no valid keys, so $$\ell = 0$$). Without the guard, $$0 \cdot \infty$$ would produce NaN.

##### **New CuTe primitives in K2**

A handful of primitives that did not appear in K1.

**`cute.struct` and `SmemAllocator`.** A `cute.struct` declares a fixed-size, aligned SMEM block. `Align[..., 1024]` enforces 1024-byte alignment; `MemRange[dtype, n]` reserves $$n$$ elements. The struct is materialised through `SmemAllocator`, and named regions are extracted with `get_tensor(layout)`. This is the standard CuTe pattern for several named SMEM tiles sharing one allocation:

```python
@cute.struct
class SharedStorage:
    sQ:  cute.struct.Align[cute.struct.MemRange[bf16,  cosize(sQ_layout)],  1024]
    sKV: cute.struct.Align[cute.struct.MemRange[bf16,  cosize(sKV_layout)], 1024]
    sS:  cute.struct.Align[cute.struct.MemRange[f32,   cosize(sS_layout)],  1024]
```

**`cute.arch.warp_reduction_max` / `_sum` with `threads_in_group`.** Sub-warp shuffle reduction. With `threads_in_group=4`, each group of 4 consecutive lanes does an independent reduction and the result is broadcast to all 4. Lowers to `__shfl_xor_sync` butterflies; the DSL hides the pattern. This is what makes the 4-thread row group work: after the reduction every thread of the group has the same `row_max` (and later `row_sum`), so the per-row state stays consistent without extra synchronisation.

**`cutlass.range` vs `cutlass.range_constexpr`.** `range_constexpr` is unrolled at JIT time and needs JIT-time-constant bounds; we use it everywhere the trip count is known. `cutlass.range(n_kv, unroll=1)` is a runtime loop, used only for the KV mainloop because $$n_\text{kv} = N / B_c$$ depends on the runtime sequence length.

**`cute.math.exp(..., fastmath=True)`.** Lowers to the SFU `ex2.approx.f32` (about 2 ULPs). The flag is what makes the softmax fast; without it we would get a much slower software exp.

**`cute.arch.rcp_approx`.** SFU reciprocal (about 1 ULP), used in the finalize divide.

The rest (`make_rmem_tensor`, `sync_threads`, `block_idx`, `thread_idx`, `.to(dtype)` casts) are direct analogues of CUDA C++ and don't need separate treatment.

##### **Running K2**

```bash
CHECK=1 SEQLEN=512 HEADDIM=64 CAUSAL=0 python bench.py k2
```

##### **What we expect**

K2 is the FA2 algorithm done correctly, with everything else still naive. Expectations:

- TFLOPS noticeably above K1 (no GMEM round-trip on $$S$$, fused softmax), still far below peak because tensor cores are unused and the scalar FP32 GEMMs dominate.
- HBM traffic drops by the K1 $$S$$-materialisation volume, on the order of $$2BHN^2$$ FP32 round-trip.
- Softmax is no longer a separate kernel pass; it overlaps with the QK GEMM in the same CTA.
- NCU will show heavy SMEM bank conflicts (unswizzled `sQ`/`sKV`/`sS`) and the vast majority of math issuing on the FMA pipe rather than the tensor pipe.

##### **Profiling**

<!-- TODO: NCU output for K2 -->

##### **What's next**

The scalar FP32 GEMM is the obvious bottleneck. K3 brings TMA to lift the GMEM->SMEM copies off the critical path with async bulk transfers; K4 then replaces the two GEMMs with WGMMA on the tensor cores.