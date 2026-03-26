---
layout: post
title: Fused SiLU+Mul+FP8 Block Quantization CUDA Kernel - vLLM Notes
date: 2026-03-25 00:00:00-0400
featured: false
description: Detailed walkthrough of a fused SiLU+Mul+FP8 block quantization CUDA kernel for vLLM, covering memory access patterns, quantization math, and dispatch mechanics
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
---

These are my personal notes on the fused SiLU+Mul+FP8 block quantization CUDA kernel I wrote for the [vLLM](https://github.com/vllm-project/vllm) project. The goal is to document the kernel's internals — what each block of code does, why it's designed that way, and how the memory access patterns work — so I can reference this later without having to re-derive everything from scratch.

**A few things before we proceed:**

- These notes were generated with the help of Claude, walking through the kernel block by block. If you spot any reasoning errors, please let me know in the comments.
- **References:**
  - Quantization fundamentals: [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
  - PR and source: [vllm-project/vllm#32996](https://github.com/vllm-project/vllm/pull/32996) — file: `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`
- **A thought on kernel work:** In production inference, the kernel itself is about 50% of the effort. The other 50% is making the kernel actually work with the existing system — pattern matching it into the compilation pipeline, handling all the dispatch variants, and making it performant end-to-end. That second half is covered in a separate post.
- This is a **WIP** — I'll add small notes where I feel more context is needed. These will be marked as **NOTES;**

---

#### **Block 1: Kernel Template Declaration & Function Signature**
```cpp
template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,     // [num_tokens, hidden_size] in FP8/INT8
    float* __restrict__ scales,         // [num_tokens, hidden_size / group_size]
                                        // or [hidden_size / group_size, num_tokens]
    scalar_t const* __restrict__ input, // [num_tokens, hidden_size * 2]
    float const* scale_ub,              // Optional scale upper bound
    int32_t const hidden_size
) {
```



This kernel fuses three operations that would otherwise be separate passes over global memory: SiLU activation, element-wise multiply, and per-block FP8/INT8 quantization. The "why fuse" is straightforward — each separate kernel would read/write the full tensor from/to HBM. Fusing means you read the input once, do all three ops in registers/shared memory, and write the quantized output once. For a memory-bound workload like activation + quantization on large hidden dimensions, this is a significant bandwidth saving.

##### **Template Parameters**

- **`scalar_t`** — input dtype (BF16 or FP16). Compile-time so the compiler generates specialized load instructions for each type.
- **`scalar_out_t`** — output dtype (FP8 e4m3 or INT8). Also compile-time because the quantization math (clamping range, rounding) differs per type.
- **`is_scale_transposed`** — controls scale tensor layout. DeepSeek-V3/R1 expects scales as `[hidden_size/group_size, num_tokens]` (transposed), while the default is `[num_tokens, hidden_size/group_size]`. Making this a template bool means the compiler eliminates the branch entirely — zero runtime cost for the layout decision.
- **`group_size`** — 64 or 128. Compile-time because it determines the block dimension, shared memory size, and the number of reduction steps. The compiler can fully unroll the reduction loop when this is known at compile time.

##### **`__restrict__` Qualifier**

Applied to `out`, `scales`, `input` — tells the compiler these pointers don't alias, enabling more aggressive load/store reordering and register optimization.

##### **Memory Layout of `input`**

The input is `[num_tokens, hidden_size * 2]` because the gate and up projections are concatenated along the last dimension as `[gate | up]`. This is the standard layout coming out of the fused gate-up projection in MoE/non-MoE models.

---

#### **Block 2: Static Assert & Grid/Thread Index Setup**
```cpp
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  // Grid: (num_tokens, num_groups)
  int const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;  // tid in [0, group_size)
  int const num_tokens = gridDim.x;
```



`static_assert` — This is a compile-time guard ensuring `group_size` is a power of 2. The reduction loop later uses `stride >>= 1` halving, which only produces a correct result when the array length is a power of 2. If someone tried to instantiate with e.g. `group_size=96`, this would fail at compile time rather than silently producing wrong results at runtime. The check `(n & (n-1)) == 0` is the standard bit-trick for power-of-2 detection.

##### **Grid Mapping — The Core Design Decision**

The kernel maps the 2D problem — `(num_tokens, num_groups)` — directly onto the 2D grid. Each thread block is responsible for exactly one quantization group within one token. This means:

- `blockIdx.x` → which token (row of the input matrix)
- `blockIdx.y` → which group within that token's hidden dimension
- `threadIdx.x` → which element within the group (range `[0, group_size)`)

So the block dimension equals `group_size` (64 or 128 threads), and every thread handles exactly one element. There's a 1:1 mapping between threads and elements in a group — no loops needed inside the kernel. This keeps the logic simple and means every thread's register holds exactly one `result` value.

`num_tokens = gridDim.x` — extracted from the grid dimension rather than passed as a parameter. This is used later for the transposed scale indexing. It's a minor optimization avoiding an extra kernel argument, but more importantly it keeps the parameter list clean since the grid already encodes this information.

##### **Memory Access Implication**

Since each block processes one group of one token, and groups are contiguous in memory (elements `[group_idx * group_size ... (group_idx+1) * group_size - 1]`), threads within a block access consecutive memory addresses. Thread 0 reads element 0 of the group, thread 1 reads element 1, etc. This gives you perfectly coalesced global memory reads — a single 128B or 256B transaction serves the whole warp.

---

#### **Block 3: Pointer Arithmetic & Memory Layout**
```cpp
  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = hidden_size * 2;
  int const group_start = group_idx * group_size;

  // Pointers to this token's data
  scalar_t const* token_input_gate =
      input + token_idx * input_stride + group_start;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size + group_start;

  // Scale pointer for this group
  int const num_groups = gridDim.y;
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;
```



This block computes all the pointers each thread block needs. Let's trace the memory layout carefully.

##### **Input Pointer Arithmetic**

The input tensor is `[num_tokens, hidden_size * 2]`, laid out row-major. For a given token, the row looks like:
```
[ gate_0, gate_1, ..., gate_{H-1}, up_0, up_1, ..., up_{H-1} ]
|<-------- hidden_size --------->|<-------- hidden_size ------->|
```

- `input_stride = hidden_size * 2` — the number of elements to skip to get to the next token's row.
- `group_start = group_idx * group_size` — the offset within the hidden dimension where this group begins.
- `token_input_gate` points to `gate[group_start]` for this token — the start of this group in the gate half.
- `token_input_up = token_input_gate + hidden_size` — since up is concatenated right after gate, adding `hidden_size` jumps from the same position in the gate half to the corresponding position in the up half. This means `gate[i]` and `up[i]` are `hidden_size` elements apart in memory, not adjacent. Each thread reads two non-adjacent locations, but within each half, the group's reads are contiguous and coalesced.

##### **Output Pointer**

The output tensor is `[num_tokens, hidden_size]` (half the input width, since SiLU(gate)*up collapses the two halves into one). `token_output` points to the start of this group in the output row.

##### **Scale Pointer — The Transposed Layout**

Each thread block produces one scale value for its group. The scale tensor has shape either:

- **Default:** `[num_tokens, num_groups]` — row-major by token. `scales + token_idx * num_groups + group_idx`.
- **Transposed:** `[num_groups, num_tokens]` — row-major by group. `scales + group_idx * num_tokens + token_idx`.

The transposed layout exists because downstream GEMM kernels (particularly for DeepSeek-V3/R1) expect scales in `[num_groups, num_tokens]` order. Rather than doing a separate transpose kernel afterward, the kernel writes scales directly in the layout the consumer needs. Since `is_scale_transposed` is a template bool, the compiler eliminates the dead branch entirely.

##### **Memory Access Pattern Summary**

- `token_input_gate[tid]` — threads 0..group_size-1 read contiguous BF16/FP16 elements → coalesced.
- `token_input_up[tid]` — same pattern, different base address (offset by `hidden_size`) → also coalesced.
- `token_output[tid]` — contiguous FP8/INT8 writes → coalesced.
- `*group_scale_ptr` — single float write by thread 0 only (later). No coalescing concern since it's one element per block.

---

#### **Block 4: Shared Memory Declaration, Loads & SiLU Computation**
```cpp
  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[group_size];

  // Step 1: Each thread loads one element, computes SiLU, stores in register
  float gate = static_cast<float>(token_input_gate[tid]);
  float up = static_cast<float>(token_input_up[tid]);

  // Compute SiLU(gate) * up
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  float silu_gate = gate * sigmoid_gate;
  float result = silu_gate * up;  // Keep in register
```



##### **Shared Memory Declaration**

`__shared__ float shared_max[group_size]` — allocates a block-local array sized exactly to the group. Since `group_size` is a template parameter (compile-time constant), this is statically sized — the compiler knows the exact shared memory footprint at compile time. This is preferable to dynamic shared memory (`extern __shared__`) because the compiler can reason about bank conflicts and optimize access patterns better. For `group_size=128`, this is `128 * 4 bytes = 512 bytes` — trivially small relative to the 48KB+ shared memory available per SM. This array will be used in the reduction step (block 5) to find the group max.

##### **Loads and Type Promotion**

Each thread loads exactly one gate element and one up element, then immediately casts to `float`. The cast serves two purposes:

1. **Precision** — SiLU involves `exp`, division, and multiple multiplies. Doing this in FP16/BF16 would accumulate significant rounding error, especially in the sigmoid where values near 0 or 1 lose precision. FP32 is the standard compute type for activation functions.
2. **Instruction efficiency** — CUDA math intrinsics like `expf` operate on `float`. Without the cast, the compiler would insert implicit conversions anyway, and potentially in a less optimal place.

##### **SiLU Computation — The Math**

SiLU (Sigmoid Linear Unit) is defined as `SiLU(x) = x * σ(x)` where `σ(x) = 1 / (1 + e^(-x))`.

The kernel computes this in two steps:
1. `sigmoid_gate = 1.0f / (1.0f + expf(-gate))` — the sigmoid. `expf` is the single-precision CUDA fast math intrinsic.
2. `silu_gate = gate * sigmoid_gate` — multiplying input by its own sigmoid.

Then `result = silu_gate * up` — the gated activation pattern used in SwiGLU/SiLU-gated FFNs. In the transformer FFN, the gate and up projections are computed as `gate = xW_gate` and `up = xW_up`, and the activation output is `SiLU(gate) * up`. This is what gives models like LLaMA, Qwen, DeepSeek their FFN expressiveness over plain ReLU.

##### **Register Residency**

`result` stays in a register — it's not written to shared or global memory yet. This is deliberate. The kernel still needs to find the group-wide max of `|result|` (for quantization scale computation) before it can quantize. Keeping `result` in a register means zero extra memory traffic — the value is computed once, used for the reduction, and then used again for the final quantized write. One load from global memory, all intermediate work in registers + shared memory.

##### **Memory Access Pattern**

- `token_input_gate[tid]` — one 2-byte load (BF16/FP16) per thread, contiguous across threads → one coalesced 64B or 128B transaction per warp depending on group_size.
- `token_input_up[tid]` — same, but base address is `hidden_size` elements away from gate. Still coalesced within this access, but this is a second global memory transaction to a different cache line region.
- No writes to global or shared memory in this step (shared_max is written in the next block).

---

#### **Block 5: Parallel Tree Reduction for Group Max**
```cpp
  // Step 2: Reduce to find group max
  shared_max[tid] = fabsf(result);
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }
```



The goal here is to find the maximum absolute value across all `group_size` elements in this group. This max is needed to compute the quantization scale — it determines how to map the FP32 range into the FP8/INT8 representable range.

##### **CUDA Math Intrinsics**

- **`fabsf(x)`** — single-precision floating-point absolute value. Returns `|x|` as a `float`. This compiles down to a single GPU instruction (`FABS`), no branching. We need absolute values because quantization cares about magnitude — negative and positive extremes both determine the scale.
- **`fmaxf(a, b)`** — single-precision floating-point maximum. Returns the larger of `a` and `b` as a `float`. Also compiles to a single instruction (`FMNMX`). Using this instead of a ternary `(a > b) ? a : b` avoids branch divergence and handles NaN consistently (NaN-propagation semantics per IEEE 754).

##### **Step 1 — Populate Shared Memory**

Each thread writes `fabsf(result)` into `shared_max[tid]`. This is a coalesced write to shared memory — consecutive threads write to consecutive 4-byte floats. Since shared memory is banked (32 banks, 4 bytes each), and threads in a warp access consecutive indices, there are zero bank conflicts. The `__syncthreads()` ensures all threads have written before any thread starts reading in the reduction.

##### **Step 2 — Parallel Tree Reduction**

This is the classic power-of-2 shared memory reduction pattern. For `group_size=128`, it works like this:
```
Iteration 1: stride=64  → threads 0-63 compare [0..63] with [64..127]
Iteration 2: stride=32  → threads 0-31 compare [0..31] with [32..63]
Iteration 3: stride=16  → threads 0-15 compare [0..15] with [16..31]
Iteration 4: stride=8   → threads 0-7  compare [0..7]  with [8..15]
Iteration 5: stride=4   → threads 0-3  compare [0..3]  with [4..7]
Iteration 6: stride=2   → threads 0-1  compare [0..1]  with [2..3]
Iteration 7: stride=1   → thread  0    compares [0]     with [1]
```

After `log2(group_size)` iterations (7 for 128, 6 for 64), the maximum is in `shared_max[0]`.

**`#pragma unroll`** — since `group_size` is a compile-time constant, the compiler knows exactly how many iterations this loop has. The pragma tells the compiler to fully unroll it, eliminating loop overhead (branch, counter increment, compare). Each iteration becomes straight-line code.

**`__syncthreads()` inside the loop** — required at every iteration because thread `tid` might read `shared_max[tid + stride]` which was written by a different thread in the same iteration. Without the barrier, you'd get a read-before-write race. Note that this is a block-wide barrier, so even threads that don't participate in the `if (tid < stride)` branch still hit the barrier — this is correct and required. Having threads diverge on the barrier (some calling it, some not) would be undefined behavior.

##### **Why Not Warp-Level Primitives (`__shfl_down_sync`)?**

For the last 5 iterations (stride ≤ 16, i.e. within a single warp), you could replace shared memory ops with warp shuffles, avoiding the `__syncthreads()` overhead. This is a common optimization but adds complexity. The current approach is simpler and still fast — shared memory access is ~5ns, and for a memory-bound kernel like this, the reduction is not the bottleneck. The bottleneck is the global memory loads/stores.

##### **Memory Access Pattern & Bank Conflict Analysis**

All accesses in this block are to `__shared__` memory — no global memory traffic.

Shared memory on NVIDIA GPUs is organized into **32 banks**, each 4 bytes wide. Consecutive 4-byte words map to consecutive banks: address 0 → bank 0, address 4 → bank 1, ..., address 124 → bank 31, address 128 → bank 0 again (wraps around). A bank conflict occurs when two or more threads in the same warp access **different** addresses that map to the **same** bank in the same instruction — the accesses get serialized.

In this reduction, at each iteration, active thread `tid` reads `shared_max[tid]` and `shared_max[tid + stride]`:

- **stride ≥ 32 (e.g. stride=64):** Thread `tid` accesses indices `tid` and `tid + 64`. Since banks wrap every 32, index `tid` maps to bank `tid % 32` and index `tid + 64` maps to bank `(tid + 64) % 32 = tid % 32`. These are two accesses to the **same bank, same thread** — this is a broadcast, not a conflict. Across different threads in a warp, thread 0 hits bank 0, thread 1 hits bank 1, etc. — each thread hits a unique bank. No conflict.
- **stride < 32 (e.g. stride=16):** Only threads 0..15 are active. Thread `tid` reads `shared_max[tid]` (bank `tid`) and `shared_max[tid + 16]` (bank `tid + 16`). Since `tid` ranges 0..15 and `tid+16` ranges 16..31, all 32 accesses hit distinct banks. No conflict.
- **stride=1:** Only thread 0 is active. Reads `shared_max[0]` (bank 0) and `shared_max[1]` (bank 1). Trivially conflict-free.

So across all iterations: **zero bank conflicts**.

---

#### **Block 5: Parallel Tree Reduction for Group Max**
```cpp
  // Step 2: Reduce to find group max
  shared_max[tid] = fabsf(result);
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }
```



The goal here is to find the maximum absolute value across all `group_size` elements in this group. This max is needed to compute the quantization scale — it determines how to map the FP32 range into the FP8/INT8 representable range.

##### **CUDA Math Intrinsics**

- **`fabsf(x)`** — single-precision floating-point absolute value. Returns `|x|` as a `float`. This compiles down to a single GPU instruction (`FABS`), no branching. We need absolute values because quantization cares about magnitude — negative and positive extremes both determine the scale.
- **`fmaxf(a, b)`** — single-precision floating-point maximum. Returns the larger of `a` and `b` as a `float`. Also compiles to a single instruction (`FMNMX`). Using this instead of a ternary `(a > b) ? a : b` avoids branch divergence and handles NaN consistently (NaN-propagation semantics per IEEE 754).

##### **Step 1 — Populate Shared Memory**

Each thread writes `fabsf(result)` into `shared_max[tid]`. This is a coalesced write to shared memory — consecutive threads write to consecutive 4-byte floats. Since shared memory is banked (32 banks, 4 bytes each), and threads in a warp access consecutive indices, there are zero bank conflicts. The `__syncthreads()` ensures all threads have written before any thread starts reading in the reduction.

##### **Step 2 — Parallel Tree Reduction**

This is the classic power-of-2 shared memory reduction pattern. For `group_size=128`, it works like this:
```
Iteration 1: stride=64  → threads 0-63 compare [0..63] with [64..127]
Iteration 2: stride=32  → threads 0-31 compare [0..31] with [32..63]
Iteration 3: stride=16  → threads 0-15 compare [0..15] with [16..31]
Iteration 4: stride=8   → threads 0-7  compare [0..7]  with [8..15]
Iteration 5: stride=4   → threads 0-3  compare [0..3]  with [4..7]
Iteration 6: stride=2   → threads 0-1  compare [0..1]  with [2..3]
Iteration 7: stride=1   → thread  0    compares [0]     with [1]
```

After `log2(group_size)` iterations (7 for 128, 6 for 64), the maximum is in `shared_max[0]`.

**`#pragma unroll`** — since `group_size` is a compile-time constant, the compiler knows exactly how many iterations this loop has. The pragma tells the compiler to fully unroll it, eliminating loop overhead (branch, counter increment, compare). Each iteration becomes straight-line code.

**`__syncthreads()` inside the loop** — required at every iteration because thread `tid` might read `shared_max[tid + stride]` which was written by a different thread in the same iteration. Without the barrier, you'd get a read-before-write race. Note that this is a block-wide barrier, so even threads that don't participate in the `if (tid < stride)` branch still hit the barrier — this is correct and required. Having threads diverge on the barrier (some calling it, some not) would be undefined behavior.

##### **Why Not Warp-Level Primitives (`__shfl_down_sync`)?**

For the last 5 iterations (stride ≤ 16, i.e. within a single warp), you could replace shared memory ops with warp shuffles, avoiding the `__syncthreads()` overhead. This is a common optimization but adds complexity. The current approach is simpler and still fast — shared memory access is ~5ns, and for a memory-bound kernel like this, the reduction is not the bottleneck. The bottleneck is the global memory loads/stores.

##### **Memory Access Pattern & Bank Conflict Analysis**

All accesses in this block are to `__shared__` memory — no global memory traffic.

Shared memory on NVIDIA GPUs is organized into **32 banks**, each 4 bytes wide. Consecutive 4-byte words map to consecutive banks: address 0 → bank 0, address 4 → bank 1, ..., address 124 → bank 31, address 128 → bank 0 again (wraps around). A bank conflict occurs when two or more threads in the same warp access **different** addresses that map to the **same** bank in the same instruction — the accesses get serialized.

In this reduction, at each iteration, active thread `tid` reads `shared_max[tid]` and `shared_max[tid + stride]`:

- **stride ≥ 32 (e.g. stride=64):** Thread `tid` accesses indices `tid` and `tid + 64`. Since banks wrap every 32, index `tid` maps to bank `tid % 32` and index `tid + 64` maps to bank `(tid + 64) % 32 = tid % 32`. These are two accesses to the **same bank, same thread** — this is a broadcast, not a conflict. Across different threads in a warp, thread 0 hits bank 0, thread 1 hits bank 1, etc. — each thread hits a unique bank. No conflict.
- **stride < 32 (e.g. stride=16):** Only threads 0..15 are active. Thread `tid` reads `shared_max[tid]` (bank `tid`) and `shared_max[tid + 16]` (bank `tid + 16`). Since `tid` ranges 0..15 and `tid+16` ranges 16..31, all 32 accesses hit distinct banks. No conflict.
- **stride=1:** Only thread 0 is active. Reads `shared_max[0]` (bank 0) and `shared_max[1]` (bank 1). Trivially conflict-free.

So across all iterations: **zero bank conflicts**.

---

#### **Block 6: Scale Computation & Broadcast**
```cpp
  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float group_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float group_scale = group_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      group_scale = fminf(group_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());

    // Store scale to global memory
    *group_scale_ptr = group_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = group_scale;
  }
  __syncthreads();

  float group_scale = shared_max[0];
```



After the reduction, `shared_max[0]` holds the maximum absolute value across the group. Only thread 0 executes this block — it computes the quantization scale and then broadcasts it to all other threads.

##### **Scale Computation — The Quantization Math**

The fundamental equation for symmetric quantization is: `quantized = value / scale`, where `scale = max_abs / quant_range`. This maps the largest magnitude value to the edge of the quantized type's representable range.

- `quant_type_max_v<scalar_out_t>` — a compile-time constant giving the max representable value of the output type. For FP8 e4m3fn this is `448.0f`, for INT8 this is `127.0f`. The template specialization ensures the right constant is used without runtime branching.
- `group_scale = group_max / quant_range` — if the largest absolute value in the group is, say, `3.5` and quant_range is `448.0`, then `scale = 3.5 / 448.0 ≈ 0.0078`. Later, each value gets divided by this scale, so `3.5 / 0.0078 ≈ 448.0` — fitting exactly at the edge of FP8 range. Smaller values map proportionally within the range.

##### **Scale Clamping — Two Guards**

1. **Upper bound (`scale_ub`):** Optional. When provided, the scale is clamped to not exceed this value. This is used in static quantization scenarios or when a calibration pass has determined a maximum expected scale. `fminf` — single-precision float minimum, same family as `fmaxf`, compiles to one instruction.

2. **Minimum safe scale (`min_scaling_factor`):** Prevents division by zero or near-zero. If all values in a group are zero (or very close), `group_max` would be 0 or tiny, making `group_scale` ≈ 0. Dividing by near-zero in the quantization step would produce `inf` or overflow the FP8 range. The minimum scaling factor is a small positive value that ensures numerical safety. This is the more critical guard of the two.

##### **Broadcast Pattern**

Thread 0 writes the final `group_scale` to both:

- `*group_scale_ptr` — global memory, so the downstream GEMM kernel can use it for dequantization.
- `shared_max[0]` — reusing the shared memory array as a broadcast channel. No need for a separate shared variable.

The `__syncthreads()` after the `if (tid == 0)` block ensures all threads see the updated `shared_max[0]` before reading it into their local `group_scale` variable. After this barrier, every thread has the same `group_scale` in a register, ready for the quantization step.

##### **Why Only Thread 0?**

Scale computation is a serial operation — it depends on the single reduced max value. Having multiple threads compute the same thing would be wasted work. The slight thread divergence (thread 0 does extra work, others idle at the barrier) is negligible because the work is just a few arithmetic ops and one global store.

##### **Memory Access Pattern**

- `shared_max[0]` — read by thread 0 (from reduction), written back by thread 0 (broadcast). Then read by all threads after barrier. Single shared memory location, no bank conflicts.
- `*group_scale_ptr` — one 4-byte float write to global memory by thread 0 only. This is a single non-coalesced write, but it's one per block, so the cost is trivial.
- `*scale_ub` — conditional 4-byte global read by thread 0 only. This pointer is the same across all blocks, so after the first block reads it, it'll be in L2/L1 cache for subsequent blocks. Effectively free.

---

#### **Block 7: Quantize & Write Output**
```cpp
  // Step 4: Quantize and write output
  token_output[tid] =
      vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
}
```



This is the final step — every thread quantizes its `result` value (still sitting in a register from block 4) using the group scale (broadcast from block 6), and writes the quantized output to global memory.

##### **The Quantization Operation**

`ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale)` is a vLLM utility that performs: `output = clamp(round(result / group_scale), -quant_max, +quant_max)`, then casts to `scalar_out_t` (FP8 or INT8).

Breaking down what happens inside:

1. **`result / group_scale`** — scales the FP32 value into the quantized range. Since `group_scale = group_max / quant_range`, this effectively maps the group's value distribution into `[-quant_range, +quant_range]`.
2. **`round(...)`** — rounds to nearest representable value. For INT8 this is standard rounding to integer. For FP8, the hardware/intrinsic handles rounding to the nearest representable FP8 value.
3. **`clamp(..., -quant_max, +quant_max)`** — safety clamp to ensure the value fits in the output type's range. In theory, the scale was computed to make the max value land exactly at `quant_max`, so clamping shouldn't alter anything. But floating-point rounding edge cases could push a value slightly beyond, so the clamp is a correctness guard.
4. **Cast to `scalar_out_t`** — the final type conversion from FP32 to FP8/INT8.

##### **Template Parameter `false`**

The second template parameter to `ScaledQuant` controls whether the quantization uses an element-wise scale (per-tensor or per-channel) or a pre-divided scale. `false` here means we pass the scale directly and the function does the division internally. This matches our use case — one scale per group, applied uniformly to all elements in that group.

##### **Register → Global Memory**

`result` has been in a register since block 4. `group_scale` has been in a register since the shared memory broadcast at the end of block 6. The quantization is pure arithmetic on registers — no shared or global memory reads. The only memory operation is the final store to `token_output[tid]`.

##### **Memory Access Pattern**

- `token_output[tid]` — each thread writes one FP8 (1 byte) or INT8 (1 byte) value to consecutive addresses. Threads 0..31 in a warp write to bytes 0..31 — this is a coalesced write of 32 bytes per warp. For `group_size=128` (4 warps), that's 4 × 32B = 128 bytes written in 4 coalesced transactions. Compare to the input which was 2 × 128 × 2 bytes (BF16) = 512 bytes read. The output is 4x smaller than the input — this is the compression effect of quantization plus the 2→1 collapse from gate+up fusion.

##### **Closing the Kernel**

After this line, the kernel is done. Every thread block has independently processed one quantization group of one token: loaded gate+up from global memory, computed SiLU*up in registers, reduced to find group max in shared memory, computed and stored the scale, and written the quantized output. No inter-block communication whatsoever — the kernel is embarrassingly parallel across the `(num_tokens × num_groups)` grid.

---

#### **Block 8: Templated Dispatch Function**
```cpp
template <typename scalar_in_t>
void silu_and_mul_per_block_quant_dispatch(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scales,
    int32_t group_size, std::optional<at::Tensor> const& scale_ub,
    bool is_scale_transposed) {
  int32_t hidden_size = out.size(-1);
  auto num_tokens = input.size(0);
  int32_t num_groups = hidden_size / group_size;

  TORCH_CHECK(input.size(-1) == hidden_size * 2,
              "input last dim must be 2x output hidden_size");
  TORCH_CHECK(hidden_size % group_size == 0,
              "hidden_size must be divisible by group_size");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Block size = group_size (64 or 128)
  dim3 grid(num_tokens, num_groups);
  dim3 block(group_size);

  VLLM_DISPATCH_QUANT_TYPES(
      out.scalar_type(), "silu_and_mul_per_block_quant_kernel", [&] {
        using scalar_out_t = scalar_t;

        VLLM_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
          VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
            vllm::silu_and_mul_per_block_quant_kernel<scalar_in_t, scalar_out_t,
                                                      transpose_scale, gs>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_out_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(),
                    scale_ub.has_value() ? scale_ub->data_ptr<float>()
                                         : nullptr,
                    hidden_size);
          });
        });
      });
}
```



This function bridges the PyTorch C++ world and the CUDA kernel. It's templated on `scalar_in_t` (input dtype, already resolved by the caller), and its job is to extract tensor metadata, validate shapes, set up the launch config, and resolve the remaining compile-time template parameters via dispatch macros.

##### **Shape Extraction & Validation**

- `hidden_size = out.size(-1)` — output's last dimension is the true hidden size.
- `num_tokens = input.size(0)` — batch/token dimension.
- `num_groups = hidden_size / group_size` — number of quantization groups per token.

The two `TORCH_CHECK`s are runtime safety guards:
1. Input's last dim must be `2 * hidden_size` — because the input is `[gate | up]` concatenated. If someone passes a tensor with wrong shape, this catches it immediately with a clear message rather than silently producing garbage.
2. `hidden_size` must be divisible by `group_size` — otherwise the last group would be partial, and the kernel assumes every group is exactly `group_size` elements. No partial group handling exists.

##### **CUDA Device & Stream Setup**

- `OptionalCUDAGuard device_guard(device_of(input))` — sets the active CUDA device to match the input tensor's device. This is critical in multi-GPU setups. Without this, the kernel might launch on the wrong GPU. The guard uses **RAII (Resource Acquisition Is Initialization)** — a C++ pattern where a resource (here, the active CUDA device setting) is acquired in the constructor and automatically released in the destructor. When `device_guard` goes out of scope (function exits, normally or via exception), its destructor restores the previously active device. This is safer than manual set/restore pairs because you can't forget to restore, and it's exception-safe — even if a `TORCH_CHECK` throws, the device gets restored.
- `cudaStream_t stream = at::cuda::getCurrentCUDAStream()` — gets PyTorch's current CUDA stream. The kernel launches on this stream to maintain proper ordering with other PyTorch ops. If PyTorch is using stream A for this computation graph, the kernel goes on stream A too.

##### **Launch Configuration**

- `dim3 grid(num_tokens, num_groups)` — 2D grid, exactly matching the kernel's `blockIdx.x = token, blockIdx.y = group` mapping we saw in **Block 2 (Static Assert & Grid/Thread Index Setup)**. Total blocks = `num_tokens × num_groups`.
- `dim3 block(group_size)` — 1D block with `group_size` threads (64 or 128). One thread per element in the group.
- `<<<grid, block, 0, stream>>>` — the `0` is dynamic shared memory size. We don't need any because `shared_max` is statically sized (from the template parameter).

##### **The Dispatch Macro Nesting**

This is where the remaining template parameters get resolved from runtime values to compile-time constants. The nesting works inside-out:

1. **`VLLM_DISPATCH_QUANT_TYPES(out.scalar_type(), ...)`** — checks the output dtype at runtime and sets `scalar_t` (aliased to `scalar_out_t`) to the corresponding C++ type (e.g., `c10::Float8_e4m3fn` or `int8_t`). This resolves the `scalar_out_t` template parameter.

2. **`VLLM_DISPATCH_GROUP_SIZE(group_size, gs, ...)`** — maps the runtime `int32_t group_size` (64 or 128) to a compile-time constant `gs`. This resolves the `group_size` template parameter.

3. **`VLLM_DISPATCH_BOOL(is_scale_transposed, transpose_scale, ...)`** — maps the runtime `bool` to a compile-time `bool` constant. This resolves the `is_scale_transposed` template parameter.

##### **Compilation Flow & Binary Generation**

The dispatch macros are essentially glorified `if/else` or `switch` blocks that instantiate the kernel template with every valid combination of compile-time parameters. At **compile time**, `nvcc` sees all the nested dispatch expansions and generates a separate, fully-specialized PTX/SASS kernel for each combination:
```
Combination 1:  <BF16, FP8_e4m3, false, 64>
Combination 2:  <BF16, FP8_e4m3, false, 128>
Combination 3:  <BF16, FP8_e4m3, true,  64>
Combination 4:  <BF16, FP8_e4m3, true,  128>
Combination 5:  <BF16, INT8,     false, 64>
...
Combination 16: <FP16, INT8,     true,  128>
```

That's `{BF16, FP16} × {FP8, INT8} × {64, 128} × {true, false}` = **16 kernel variants** compiled into the final `.so` binary. Each variant has its reduction fully unrolled, its dtype conversions baked in, and its scale layout branch eliminated. The compiler optimizes each independently — different register allocations, different instruction schedules.

The **trade-off** is:
- **Compile time** — 16 kernel instantiations means 16× the `nvcc` compilation work. This is why vLLM builds can be slow, especially with `MAX_JOBS` limited on shared clusters.
- **Binary size** — each instantiation adds a few KB of GPU machine code. 16 variants might add ~50-100KB total — negligible for a modern `.so`.
- **Runtime** — at kernel launch, the dispatch macros execute a few `if/else` branches (nanoseconds on CPU) to select the right pre-compiled variant. The GPU then runs fully specialized code with zero runtime branching on dtype, group size, or scale layout. This is the payoff: CPU-side nanoseconds of dispatch cost buys you a kernel that runs at maximum theoretical efficiency on the GPU.

##### **`scale_ub` Handling**

`scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr` — converts the `std::optional<at::Tensor>` to a raw pointer. If no upper bound was provided, the kernel gets `nullptr` and the `if (scale_ub != nullptr)` check in **Block 6 (Scale Computation & Broadcast)** skips that branch.

---

#### **Block 9: Top-Level Entry Point**
```cpp
void silu_and_mul_per_block_quant(torch::Tensor& out,
                                  torch::Tensor const& input,
                                  torch::Tensor& scales, int64_t group_size,
                                  std::optional<torch::Tensor> scale_ub,
                                  bool is_scale_transposed) {
  static c10::ScalarType kFp8Type = is_fp8_ocp()
                                        ? c10::ScalarType::Float8_e4m3fn
                                        : c10::ScalarType::Float8_e4m3fnuz;

  TORCH_CHECK(out.dtype() == kFp8Type || out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  TORCH_CHECK(
      input.dtype() == torch::kFloat16 || input.dtype() == torch::kBFloat16,
      "Input must be FP16 or BF16");
  TORCH_CHECK(scales.dtype() == torch::kFloat32, "Scales must be FP32");
  TORCH_CHECK(group_size == 128 || group_size == 64,
              "Unsupported group size: ", group_size);

  if (scale_ub.has_value()) {
    TORCH_CHECK(out.dtype() == kFp8Type);
  }

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant_dispatch", [&] {
        silu_and_mul_per_block_quant_dispatch<scalar_t>(
            out, input, scales, group_size, scale_ub, is_scale_transposed);
      });
}
```



This is the top-level entry point — the function that Python/PyTorch calls via pybind11. It's the outermost layer of the dispatch chain. Its job is to validate all preconditions, determine the FP8 variant for the platform, and dispatch on input dtype before handing off to the templated function in **Block 8 (Templated Dispatch Function)**.

##### **FP8 Type Selection**
```cpp
static c10::ScalarType kFp8Type = is_fp8_ocp()
                                      ? c10::ScalarType::Float8_e4m3fn
                                      : c10::ScalarType::Float8_e4m3fnuz;
```

There are two FP8 e4m3 variants in the wild:

- **`Float8_e4m3fn`** — the OCP (Open Compute Project) standard, used by NVIDIA H100/H200 and most mainstream hardware. The `fn` means "finite, no NaN" — it trades NaN representation for an extra finite value, giving max value `448.0`.
- **`Float8_e4m3fnuz`** — the AMD/alternative variant (`fnuz` = "finite, no NaN, unsigned zero"). Slightly different encoding with different range/precision tradeoffs.

`is_fp8_ocp()` is a vLLM utility that queries the hardware at runtime. The result is cached in a `static` variable — computed once on the first call, reused for all subsequent calls. This ensures the kernel uses the correct FP8 encoding for the hardware it's running on.

##### **Precondition Checks**

Five `TORCH_CHECK`s form the validation gate. Every invalid input gets caught here with a clear error message before anything touches the GPU:

1. **Output dtype** — must be either the platform-appropriate FP8 type or INT8. No other output types are supported.
2. **Contiguity** — both `out` and `input` must be contiguous in memory. The kernel's pointer arithmetic (strides, offsets) assumes row-major contiguous layout. A non-contiguous tensor (e.g., a transposed view or a slice) would have gaps in memory, and the kernel would read/write wrong addresses. PyTorch's `is_contiguous()` checks for C-contiguous (row-major) layout.
3. **Input dtype** — FP16 or BF16 only. These are the two standard half-precision types used in transformer inference. FP32 input isn't supported because it would be wasteful (the kernel promotes to FP32 internally anyway, but FP32 input doubles memory bandwidth for no precision gain in the final FP8 output).
4. **Scales dtype** — must be FP32. The quantization scale needs full precision to avoid compounding quantization error during dequantization in downstream GEMMs.
5. **Group size** — must be exactly 64 or 128. These are the two block quantization group sizes used in practice (128 for `kFp8Dynamic128Sym`, 64 for `kFp8Dynamic64Sym`). The dispatch macros in **Block 8** only have specializations for these two values.

##### **What `kFp8Dynamic128Sym` / `kFp8Dynamic64Sym` Means**

These are vLLM's quantization scheme identifiers. The name encodes three properties:

- **`Fp8`** — the quantized dtype is FP8 (e4m3).
- **`Dynamic`** — the quantization scale is computed dynamically at runtime from the actual tensor values (as this kernel does — finding the group max on the fly). This is in contrast to *static* quantization, where the scale is pre-calibrated from a calibration dataset and fixed at inference time.
- **`128` / `64`** — the group size, i.e., how many elements share a single scale factor. 128 means every 128 contiguous elements get one scale; 64 means every 64 elements. Smaller group size = finer granularity = better quantization accuracy (each group's scale fits its local range more tightly), but more scale overhead (more scale values to store and use in downstream GEMMs).
- **`Sym`** — symmetric quantization. The quantized range is `[-quant_max, +quant_max]` centered at zero, with a single scale factor. This is opposed to *asymmetric* quantization which uses a scale + zero-point pair to represent an offset range. Symmetric is simpler (no zero-point math) and is the standard for FP8 block quantization in practice.

So `kFp8Dynamic128Sym` means: "FP8 quantization with dynamically computed scales, 128-element groups, symmetric range." This is the scheme DeepSeek-R1 and Qwen models use for their activation quantization.

##### **`scale_ub` Guard**

If a scale upper bound is provided, the output must be FP8 (not INT8). This is because the upper bound feature is specifically designed for FP8 dynamic quantization workflows — it doesn't make sense for INT8 quantization which uses a different calibration approach.

##### **Input Dtype Dispatch**
```cpp
VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "silu_and_mul_per_block_quant_dispatch", [&] {
      silu_and_mul_per_block_quant_dispatch<scalar_t>(
          out, input, scales, group_size, scale_ub, is_scale_transposed);
    });
```

This is the first level of the dispatch chain. `VLLM_DISPATCH_FLOATING_TYPES` checks `input.scalar_type()` at runtime and resolves `scalar_t` to either `c10::BFloat16` or `c10::Half` (FP16). It then calls `silu_and_mul_per_block_quant_dispatch<scalar_t>(...)` which is **Block 8**, where the remaining three template parameters get dispatched.

##### **The Full Dispatch Chain**

Putting it all together, the call flow from Python to GPU is:
```
Python: ops.silu_and_mul_per_block_quant(out, input, scales, 128, None, True)
  │
  ▼
silu_and_mul_per_block_quant()          ← Block 9 (this function)
  │  validates everything
  │  dispatches on input dtype (BF16/FP16)
  ▼
silu_and_mul_per_block_quant_dispatch() ← Block 8
  │  extracts shapes, sets up grid/block
  │  dispatches on output dtype × group_size × is_scale_transposed
  ▼
silu_and_mul_per_block_quant_kernel()   ← Blocks 1-7 (the CUDA kernel)
     fully specialized: <BF16, FP8_e4m3, true, 128>
     runs on GPU
```

Each level strips away one layer of runtime ambiguity and converts it to a compile-time constant, until the kernel that actually runs on the GPU has zero runtime type checks or branches on configuration parameters.

---

#### **Summary / Conclusion**

This kernel solves a specific problem in the transformer FFN inference pipeline: after the fused gate-up projection produces a `[num_tokens, hidden_size * 2]` tensor, you need to apply `SiLU(gate) * up` and then quantize the result to FP8/INT8 with per-block scales — all before feeding into the next GEMM. Doing these as three separate kernels would mean three round-trips to HBM. This fused kernel does it in one.

##### **The Logical Pipeline Within Each Thread Block**
```
Global Memory (HBM)
    │
    │  ① Load gate[tid] and up[tid]  (2 coalesced reads, BF16/FP16)
    ▼
Registers
    │
    │  ② Promote to FP32
    │  ③ Compute sigmoid(gate) → SiLU(gate) = gate * sigmoid(gate)
    │  ④ result = SiLU(gate) * up
    │
    │  (result stays in register for the rest of the kernel)
    ▼
Shared Memory
    │
    │  ⑤ Write |result| into shared_max[tid]
    │  ⑥ Tree reduction: log2(group_size) steps to find max
    │
    ▼
Thread 0 only
    │
    │  ⑦ scale = max / quant_range
    │  ⑧ Clamp scale (upper bound + minimum safe value)
    │  ⑨ Write scale to global memory (for downstream GEMM)
    │  ⑩ Broadcast scale via shared_max[0]
    │
    ▼
All Threads
    │
    │  ⑪ Read scale from shared memory into register
    │  ⑫ quantized = ScaledQuant(result, scale)
    │  ⑬ Write quantized output  (1 coalesced write, FP8/INT8)
    │
    ▼
Global Memory (HBM)
```

##### **Memory Traffic Per Group (group_size=128, BF16 input, FP8 output)**

- **Read:** 128 gate elements × 2 bytes + 128 up elements × 2 bytes = **512 bytes**
- **Write:** 128 quantized elements × 1 byte + 1 scale × 4 bytes = **132 bytes**
- **Total:** 644 bytes per group, compared to ~1.5KB+ if you had three separate kernels (each reading and writing the full intermediate tensor)

##### **The Key Design Decisions and Why They Work**

1. **One thread = one element.** No loops, no vectorization. Keeps the kernel simple and the occupancy straightforward. For group_size 64 or 128, every thread in the block is doing useful work. The simplicity also means the compiler has an easy time optimizing register allocation.

2. **group_size = block dimension.** This is the central architectural choice. It means the reduction for computing the quantization scale happens entirely within one block using shared memory — no inter-block communication, no atomics, no multi-pass reductions. Each block is self-contained.

3. **All configuration via template parameters.** Dtype, group size, scale layout — all resolved at compile time. The GPU code has zero branches on these. The cost (16 kernel variants, longer compile times) is paid once at build time; the benefit (fully optimized per-variant code) is paid back on every inference call.

4. **Register residency of `result`.** The SiLU*up value is computed once in Block 4, lives in a register through the reduction (Blocks 5-6), and is consumed in the quantization (Block 7). It never touches shared or global memory as an intermediate. This is the core benefit of fusion — the intermediate activation tensor that would normally be materialized in HBM simply never exists.

5. **Shared memory reuse.** `shared_max` serves double duty — first as the reduction workspace, then as the broadcast channel for the scale. No extra shared memory needed.

6. **Scale layout flexibility.** The transposed scale write (controlled by `is_scale_transposed`) means downstream GEMMs don't need a separate transpose kernel. The layout the consumer needs is written directly, for free, since it's just a pointer arithmetic change.

##### **Where This Kernel Sits in the Model Forward Pass**
```
Input x: [num_tokens, hidden_size]
    │
    ▼
Fused Gate-Up Projection (GEMM)
    │  x @ [W_gate | W_up]
    ▼
Intermediate: [num_tokens, hidden_size * 2]  ← gate and up concatenated
    │
    ▼
┌─────────────────────────────────────────┐
│  THIS KERNEL                            │
│  SiLU(gate) * up + FP8 block quantize   │
└─────────────────────────────────────────┘
    │
    ▼
Output: [num_tokens, hidden_size] in FP8    +  Scales: [num_tokens, num_groups]
    │
    ▼
Down Projection (FP8 GEMM, uses scales for dequant)
    │
    ▼
Output: [num_tokens, hidden_size]
```

---

As mentioned at the start, the kernel is just 50% of the work. The other 50% is making it actually work within vLLM's `torch.compile` pipeline — pattern matching the unfused ops in the FX graph, handling all the dispatch variants, and wiring everything up so the fused kernel fires at the right place during inference. That second half is covered in the next post: [SiLU+Mul+FP8 Block Quant Pattern Matching Pipeline - vLLM Notes]({{ site.baseurl }}{% post_url 2026-03-25-silu-mul-fp8-block-quant-compile-vLLM %}).