#### **K1: Naive materialized attention**

##### **What's new**

This is the baseline. No fusion, no tiling, no online softmax, no Tensor Cores. Just the three operations of attention done as three separate kernels, with the full $$(B, N, N)$$ score matrix materialized in GMEM between them. The point isn't to be fast. The point is to have a "before" picture: a working implementation that we can correctness-check against SDPA, profile, and then start optimizing.

Reading this section, you should expect: the slowest kernel in the worklog, modest tile sizes, no special hardware features used, and a hard wall at large sequence lengths because the materialized matrix doesn't fit.

##### **The idea**

Attention is three operations. K1 implements them in the most direct way possible, as three separately-launched kernels with the intermediate $$S$$ matrix written to GMEM and read back:

$$
\begin{aligned}
S &= Q K^\top \cdot \text{scale} \quad (\text{plus causal mask if enabled}) \\
P &= \text{softmax}(S) \quad (\text{in place, S becomes P}) \\
O &= P V
\end{aligned}
$$

Each kernel uses one thread per output element (one thread per $$S_{i,j}$$, one thread per row for the softmax, one thread per $$O_{i,j}$$). No SMEM staging, no Tensor Cores. The whole computation reads from and writes to GMEM directly.

This is exactly the configuration we used to motivate FlashAttention earlier in the blog. From the attention-math section: at $$B=1, N=10^6, d$$ moderate, the materialized $$S$$ matrix alone needs about 4 TB. For the smaller sweep configs we actually run ($$N \le 16384$$, batch folded with heads), the matrix fits, but it dominates GMEM traffic and the arithmetic intensity is essentially $$I = 1$$ (every FLOP is preceded by a load). Naive GEMM, naive softmax, naive output projection, three times.

Why bother implementing it? Three reasons. First, it's the right "before" picture for the worklog. Every later kernel either removes a piece of K1's badness or replaces a piece with a hardware-specific fast path; you can't appreciate the gains without seeing the baseline. Second, this is the first time we run CuTe Python end to end, so it exercises the harness (`bench.py`, `ref_check.py`, the `compile_kernel` / `run_kernel` contract) on something simple. Third, even at this level you can verify the algorithm is right by running `CHECK=1` and comparing against SDPA; this gives us confidence the layout and indexing conventions are wired up correctly before we start adding optimizations.

##### **Implementation**

The kernel file `kernels/k1.py` contains three CuTe classes plus the `compile_kernel` / `run_kernel` dispatchers that the harness calls.

The three classes:

- **`K1Score`** computes $$S_{i,j,b} = \sum_k Q_{i,k,b} K_{j,k,b} \cdot \text{scale}$$, with the causal mask applied as a write of $$-\infty$$ when $$j > i$$. One thread per $$(i, j, b)$$ output element. Grid: $$(\lceil N / 16 \rceil, \lceil N / 16 \rceil, B)$$ blocks, each $$16 \times 16 \times 1$$ threads. Reads Q and K from GMEM in FP32, accumulates in FP32, writes the result to the $$S$$ buffer.

- **`K1Softmax`** does the per-row softmax. One thread per row. The thread reads the entire row of $$S$$ from GMEM to find the max, reads it again to compute exponentials and the sum, then reads it a third time to normalize. Grid: $$(\lceil N / 256 \rceil, B, 1)$$ blocks, each $$256 \times 1 \times 1$$ threads. The materialized $$S$$ is reused in place: the kernel reads $$S$$ and writes $$P$$ to the same buffer.

- **`K1Output`** computes $$O_{i,j,b} = \sum_k P_{i,k,b} V_{k,j,b}$$. One thread per $$(i, j, b)$$ output element. Grid: $$(\lceil d / 16 \rceil, \lceil N / 16 \rceil, B)$$ blocks, each $$16 \times 16 \times 1$$ threads. Reads $$P$$ (still in the $$S$$ buffer) and $$V$$ from GMEM, accumulates in FP32, writes the result to $$O$$ in BF16.

To keep this section anchored, here's the inner kernel for `K1Score`. The other two follow the same shape: compute the per-element coordinate, bounds-check, do the work, write the result.

```python
@cute.kernel
def kernel(self, mQ, mK, mS, scale: cutlass.Float32):
    bx, by, bz = cute.arch.block_idx()
    tx, ty, _ = cute.arch.thread_idx()

    j = bx * SCORE_BLOCK_X + tx
    i = by * SCORE_BLOCK_Y + ty
    b = bz

    if i < self.N and j < self.N:
        # Causal mask: kv position > query position => -inf.
        if cutlass.const_expr(self.causal) and j > i:
            mS[i, j, b] = -cutlass.Float32.inf
        else:
            gQ = mQ[(None, None, b)]
            gK = mK[(None, None, b)]
            dot = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(self.d):
                dot = dot + gQ[i, k].to(cutlass.Float32) * gK[j, k].to(cutlass.Float32)
            mS[i, j, b] = dot * scale
```

A few CuTe-specific things to read off this code:

- **`cute.arch.block_idx()` and `cute.arch.thread_idx()`** are the CuTe analogues of CUDA's `blockIdx` and `threadIdx`. They return tuples; we destructure them into `(bx, by, bz)` and `(tx, ty, _)`.
- **`mQ[(None, None, b)]`** is the CUTE slicing pattern from the foundations section: fix the batch coord at `b`, leave the other two free. The result `gQ` is a 2D view into thread block's batch.
- **`cutlass.const_expr(self.causal)`** is the JIT-time-constant marker. The compiler resolves this at compile time and emits a specialized kernel for either causal or non-causal, with no runtime branch.
- **`cutlass.range_constexpr(self.d)`** says "fully unroll this loop at compile time." Since `d` is a JIT-time constant (64 or 128), the K-axis dot product is emitted as 64 or 128 straight-line multiply-adds.

The dispatcher pattern is what the harness calls. `compile_kernel` builds the three CuTe objects, compiles each one once, allocates the $$S$$ buffer, and returns a handle:

```python
def compile_kernel(B, N, d, causal, tensors):
    score = K1Score(B, N, d, causal)
    softmax = K1Softmax(B, N)
    output = K1Output(B, N, d)
    scale = cutlass.Float32(1.0 / math.sqrt(d))
    stream = get_cuda_stream()

    s_storage, s_cute = _make_S_tensor(B, N)

    score_compiled = cute.compile(
        score, tensors["q_cute"], tensors["k_cute"], s_cute, scale, stream,
    )
    softmax_compiled = cute.compile(softmax, s_cute, stream)
    output_compiled = cute.compile(
        output, s_cute, tensors["v_cute"], tensors["o_cute"], stream,
    )

    return (score_compiled, softmax_compiled, output_compiled,
            scale, s_storage, s_cute)
```

The `S` buffer is allocated *once* inside `compile_kernel` and reused across all benchmark iterations. This is important: without it, every benchmark iteration would pay an OOM-prone $$\mathcal{O}(B N^2)$$ allocation cost, which would dominate the timing.

`run_kernel` is then trivial: launch the three compiled kernels in sequence on the current CUDA stream.

```python
def run_kernel(compiled_handle, tensors):
    (score_compiled, softmax_compiled, output_compiled,
     scale, s_storage, s_cute) = compiled_handle
    stream = get_cuda_stream()
    score_compiled(tensors["q_cute"], tensors["k_cute"], s_cute, scale, stream)
    softmax_compiled(s_cute, stream)
    output_compiled(s_cute, tensors["v_cute"], tensors["o_cute"], stream)
```

That's it. The full file is on GitHub: <!-- TODO: link to kernels/k1.py -->.

##### **Running K1**

From the project root:

```bash
CHECK=1 SEQLEN=512 HEADDIM=64 CAUSAL=0 python bench.py k1
```

This compiles K1, runs the correctness check against SDPA, then 10 warmup + 30 timed iterations.

<!-- TODO: actual K1 output, including the SDPA check result and the per-config timing/TFLOPS. -->

##### **What we expect to see**

Three things, all bad:

- **Low TFLOPS.** No Tensor Cores. Every multiply-add uses CUDA cores in FP32. The peak we can hit is bounded by the CUDA-core FP32 throughput, which on H100 is on the order of $$67$$ TFLOPS, far below the $$\sim 990$$ TFLOPS BF16 Tensor-Core peak we'd target later.
- **OOM at large $$N$$.** The $$S$$ buffer is $$B \times N \times N \times 4$$ bytes. For $$N = 16384$$ and the larger batch values in the sweep, this exceeds H100's 80 GB. The harness catches `torch.cuda.OutOfMemoryError` and reports `[OOM: ...]` for those configs; that's expected.
- **Softmax dominates.** One thread per row, three full passes over $$N$$ values from GMEM. For larger $$N$$ this is the slowest of the three kernels, even though it does the least arithmetic.

The profiling, once we have it, should make all three of these visible.

##### **Profiling**

<!-- TODO: NCU profile of K1. At minimum, capture: kernel time per stage (Score, Softmax, Output), achieved compute throughput, memory throughput, the dominant stall reason for each stage. The softmax kernel especially is worth inspecting; one thread per row with sequential GMEM reads should look catastrophic. -->

##### **What's next**

The first thing to fix is the materialized $$S$$ matrix. Everything else (Tensor Cores, async loads, warp specialization) is downstream of that decision; as long as we materialize $$S$$ we're paying $$\mathcal{O}(B N^2)$$ in memory and $$\mathcal{O}(B N^2)$$ in GMEM traffic for it. K2 fuses the three kernels into one and uses the tiled online softmax we derived in the math sections, so $$S$$ never leaves SMEM. This is also the first kernel where we'll see the basic FlashAttention structure (per-row state, max-tracking recurrence, output rescaling) in actual code.