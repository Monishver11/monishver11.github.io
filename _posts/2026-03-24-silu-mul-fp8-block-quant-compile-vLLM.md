---
layout: post
title: SiLU+Mul+FP8 Block Quant Pattern Matching Pipeline - vLLM Notes
date: 2026-03-25 00:05:00-0400
featured: false
description: Detailed walkthrough of vLLM's torch.compile pattern matching pipeline that fuses SiLU+Mul and FP8 block quantization into a single kernel launch, covering FX graphs, matchers, and the dispatch machinery
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
---

These are my personal notes on the pattern matching pipeline that integrates the fused SiLU+Mul+FP8 block quantization CUDA kernel into vLLM's `torch.compile` compilation flow. While the [kernel notes]({{ site.baseurl }}{% post_url 2026-03-24-silu-mul-fp8-block-quant-kernel-vLLM %}) cover what the fused kernel does at the hardware level, this post covers the other half of the story — how we teach `torch.compile` to find the unfused op sequence in the FX graph and replace it with the fused kernel.

**A few things before we proceed:**

- These notes were generated with the help of Claude, walking through the code block by block. If you spot any reasoning errors, please let me know in the comments.
- **References:**
  - Kernel walkthrough: [Fused SiLU+Mul+FP8 Block Quantization CUDA Kernel - vLLM Notes]({{ site.baseurl }}{% post_url 2026-03-24-silu-mul-fp8-block-quant-kernel-vLLM %})
  - PR and source: [vllm-project/vllm#32996](https://github.com/vllm-project/vllm/pull/32996) — files: `vllm/compilation/passes/fusion/act_quant_fusion.py`, `vllm/compilation/passes/fusion/matcher_utils.py`
  - Quantization fundamentals: [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
- **A thought on kernel work:** In production inference, the kernel itself is about 50% of the effort. The other 50% is making the kernel actually work with the existing system — pattern matching it into the compilation pipeline, handling all the dispatch variants, and making it performant end-to-end. This post covers that second half.
- This is a **WIP** — I'll add small notes where I feel more context is needed. These will be marked as **NOTES;**

---

#### **Block 1: `ActivationQuantPattern` Base Class**
```python
class ActivationQuantPattern(ABC):
    """
    The base class for Activation+Quant fusions.
    Should not be used directly.
    """

    def __init__(
        self,
        quant_key: QuantKey,
    ) -> None:
        self.quant_key = quant_key
        self.quant_dtype = quant_key.dtype

        assert self.quant_key in QUANT_OPS, (
            f"unsupported quantization scheme {self.quant_key}"
        )
        self.QUANT_OP = QUANT_OPS[self.quant_key]

        assert self.quant_key in FUSED_OPS, (
            f"unsupported fusion scheme {self.quant_key}"
        )
        self.FUSED_OP = FUSED_OPS[self.quant_key]

        self.silu_and_mul_matcher = MatcherSiluAndMul()
```

This is the abstract base class that all activation+quantization fusion patterns inherit from. Its job is to store the configuration that every fusion pattern needs: which quantization scheme we're targeting, what the unfused and fused ops are, and a matcher for the SiLU+mul part.

##### **`ABC` (Abstract Base Class)**

Python's `ABC` from the `abc` module. It means you can't instantiate `ActivationQuantPattern` directly — you must subclass it and implement the abstract methods (in this case, `register`). This enforces a contract: every fusion pattern must define how to register itself with the pattern matcher.

##### **`QuantKey`**

This is vLLM's identifier for a quantization scheme — it encodes the dtype, scale type (static vs dynamic), and group shape (per-tensor, per-token, per-group with a specific size). For our kernel, `kFp8Dynamic128Sym` is a `QuantKey` that says: FP8 dtype, dynamic scales, group shape `(1, 128)`, symmetric. The base class stores this and extracts `quant_dtype` from it.

##### **The Two Lookups — `QUANT_OPS` and `FUSED_OPS`**

These are dictionaries mapping `QuantKey` → `OpOverload` (a torch custom op reference). They serve different purposes:

- `QUANT_OPS[quant_key]` → the **unfused** quantization op. This is what the FX graph contains *before* fusion — the standalone quantization kernel. The pattern matcher needs to find this op in the graph to know where to apply the fusion. For block quantization, this is something like `torch.ops._C.cutlass_scaled_fp8_quant.default`.

- `FUSED_OPS[quant_key]` → the **fused** op. This is what replaces the pattern after fusion. For our case, `FUSED_OPS[kFp8Dynamic128Sym]` = `torch.ops._C.silu_and_mul_per_block_quant.default` — the kernel we analyzed in the first set of notes.

The two `assert` checks ensure that both the unfused and fused ops exist for this quant scheme before we try to register anything. If someone adds a new quantization scheme but forgets to implement the fused kernel, this fails loudly at initialization time.

##### **`self.silu_and_mul_matcher = MatcherSiluAndMul()`**

This creates a matcher object that knows how to match the SiLU+mul pattern in the FX graph. Every activation+quant fusion has a SiLU+mul as the first half of the pattern (the activation part), so the base class creates it once. We'll cover what `MatcherSiluAndMul` does internally when we get to the matcher utils.

##### **The Abstract Method — `register`**
```python
@abstractmethod
def register(self, pm_pass: PatternMatcherPass) -> None:
    raise NotImplementedError
```

Each subclass must implement `register`, which defines the `pattern` function (what to look for in the FX graph) and the `replacement` function (what to replace it with), then calls `register_replacement` to hook them into PyTorch's pattern matcher. The specifics differ per quantization scheme — static quant has different inputs/outputs than dynamic block quant — so this is left abstract.

##### **`register_replacement` — How PyTorch Pattern Matching Works**

`register_replacement` is a PyTorch Inductor API (`torch._inductor.pattern_matcher.register_replacement`). It takes four key arguments:

1. **`pattern`** — a Python function that, when traced, produces an FX subgraph representing what you want to find. You write it using the same ops that would appear in the compiled graph (e.g., `auto_functionalized(silu_and_mul_op, ...)` followed by `auto_functionalized(quant_op, ...)`). PyTorch traces this function to build a template subgraph.

2. **`replacement`** — a Python function that, when traced, produces the FX subgraph you want to substitute in. For our case, this calls the fused op (`silu_and_mul_per_block_quant`) instead of the two separate ops.

3. **`example_inputs`** — concrete dummy tensors with the right shapes/dtypes. PyTorch needs these to trace both `pattern` and `replacement` (since tracing requires actual tensor metadata to infer shapes, dtypes, and op dispatch).

4. **`fwd_only`** — a flag indicating this replacement only applies to the forward pass (not backward). Since vLLM inference doesn't need autograd, this is always `fwd_only`.

When `register_replacement` is called, PyTorch traces both functions, converts them to FX subgraphs, and stores them as a pattern-replacement pair in the `PatternMatcherPass`. Later, when the pass runs on the actual model's FX graph, it searches for subgraphs that structurally match the `pattern` subgraph and swaps them with the `replacement` subgraph. This is essentially a "find and replace" on the computation graph — but at the graph IR level, not source code level.

##### **FX Graph — Forward Reference**

> **Note:** We'll discuss what an FX graph is, why `torch.compile` produces one, and why we need to modify it in detail when we reach `SiluMulBlockQuantPattern.register()`. That's where we'll trace through a concrete example of what the graph looks like before and after fusion.

##### **Helper Method — `empty_quant`**
```python
def empty_quant(self, *args, **kwargs):
    kwargs = {"dtype": self.quant_dtype, "device": "cuda", **kwargs}
    return torch.empty(*args, **kwargs)
```

A convenience for creating dummy tensors with the right quantized dtype (FP8 or INT8) on CUDA. These dummy tensors are used when constructing example inputs for pattern registration — PyTorch's pattern matcher needs concrete example tensors to trace the pattern and replacement functions. The tensors' values don't matter; only their shapes and dtypes matter.

---

#### **Block 2: `MatcherSiluAndMul` — SiLU+Mul Pattern Matcher**
```python
class MatcherSiluAndMul(MatcherCustomOp):
    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = SiluAndMul.enabled()
        super().__init__(enabled)

    def inputs(self) -> list[torch.Tensor]:
        input = self.empty(5, 4)
        return [input]

    def forward_custom(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        result = auto_functionalized(SILU_MUL_OP, result=out, input=x)
        return result[1]

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return SiluAndMul.forward_native(x)
```

This matcher represents the SiLU+mul activation op in the FX graph. It's a "matcher" because its primary role is to produce the FX subgraph fragment that the pattern matcher should look for. When the base class `ActivationQuantPattern` calls `self.silu_and_mul_matcher(input)` inside a `pattern` function, it's this class's `forward_custom` (or `forward_native`) that gets traced.

##### **`MatcherCustomOp` Base Class**

`MatcherSiluAndMul` inherits from `MatcherCustomOp`, which is vLLM's base class for matchers. `MatcherCustomOp` handles the dispatch between `forward_custom` and `forward_native` based on the `enabled` flag. When `enabled=True`, calling the matcher as `self.silu_and_mul_matcher(input)` invokes `forward_custom` — which uses the custom C++ op. When `enabled=False`, it falls back to `forward_native` — a pure PyTorch implementation. The pattern matcher traces whichever path is active to build the subgraph template.

##### **`enabled` Flag**
```python
if enabled is None:
    enabled = SiluAndMul.enabled()
```

`SiluAndMul.enabled()` checks whether vLLM's custom SiLU+mul op is available and should be used in the current environment. If the custom op isn't compiled (e.g., CPU-only build), this returns `False` and the matcher falls back to the native path. In normal GPU inference, this is `True`.

##### **`inputs()`**
```python
def inputs(self) -> list[torch.Tensor]:
    input = self.empty(5, 4)
    return [input]
```

Returns dummy example tensors for tracing. `self.empty(5, 4)` creates a `torch.empty(5, 4)` tensor with the default dtype (BF16) on CUDA. The shape `(5, 4)` is arbitrary — it just needs to be valid for the op (i.e., last dim must be even since SiLU+mul splits it in half). These are used as `example_inputs` when registering patterns. The actual model tensors will have different shapes, but PyTorch's pattern matcher matches on graph structure, not tensor shapes.

##### **`forward_custom` — The Key Method**
```python
def forward_custom(self, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    result = auto_functionalized(SILU_MUL_OP, result=out, input=x)
    return result[1]
```

This is what gets traced when building the pattern subgraph. Breaking it down:

1. `d = x.shape[-1] // 2` — output hidden size is half the input (gate+up → fused output).
2. `out = torch.empty(...)` — allocates the output tensor. This is required by `auto_functionalized`.
3. `auto_functionalized(SILU_MUL_OP, result=out, input=x)` — this is the critical call.

##### **`auto_functionalized` — What It Is and Why It's Needed**

`torch.compile` operates on a *functional* graph — every operation takes inputs and returns outputs, with no in-place mutation. But vLLM's custom ops are *mutating* — they write results into pre-allocated output tensors (like `result=out`). These two worlds are incompatible.

`auto_functionalized` (from `torch._higher_order_ops.auto_functionalize`) bridges this gap. It wraps a mutating custom op and presents it as a functional op to `torch.compile`. Concretely:

- **Input:** the mutating op (`SILU_MUL_OP`) plus its arguments including the output tensor (`result=out`).
- **Output:** a tuple where each element corresponds to a mutated tensor argument. `result[0]` is typically the updated self/first arg, `result[1]` is the `result` tensor after the op has written to it.

So `result[1]` gives us the output of the SiLU+mul operation, but in a way that `torch.compile`'s FX graph can track as a pure function with no side effects.

**`SILU_MUL_OP`** is `torch.ops._C.silu_and_mul.default` — vLLM's unfused SiLU+mul custom op. This is what appears in the FX graph when the model runs SiLU+mul without fusion.

##### **`forward_native`**
```python
def forward_native(self, x: torch.Tensor) -> torch.Tensor:
    return SiluAndMul.forward_native(x)
```

Fallback path using pure PyTorch ops (no custom C++ kernel). This would produce a different FX subgraph — one using standard `torch.sigmoid`, `torch.mul`, etc. The pattern matcher would then look for this native pattern instead.

---

#### **Block 3: `MatcherQuantFP8` — FP8 Quantization Pattern Matcher**
```python
class MatcherQuantFP8(MatcherCustomOp):
    def __init__(
        self,
        quant_key: QuantKey,
        enabled: bool | None = None,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        match_rocm_aiter: bool = False,
        is_tma_aligned: bool = False,
    ) -> None:
        if enabled is None:
            enabled = QuantFP8.enabled()

        super().__init__(enabled)
        self.quant_key = quant_key
        self.has_col_major_scales = has_col_major_scales
        self.is_e8m0 = is_e8m0
        self.match_rocm_aiter = match_rocm_aiter
        self.is_tma_aligned = is_tma_aligned

        if match_rocm_aiter:
            # ... ROCm-specific path (not relevant for our case)
        else:
            assert quant_key in QUANT_OPS, (
                f"unsupported quantization scheme {quant_key}"
            )
            self.QUANT_OP = QUANT_OPS[quant_key]

            assert quant_key.dtype == current_platform.fp8_dtype(), (
                "Only QuantFP8 supported by"
            )
            assert quant_key.scale2 is None

        self.quant_fp8 = QuantFP8(
            quant_key.scale.static,
            quant_key.scale.group_shape,
            column_major_scales=has_col_major_scales,
            use_ue8m0=is_e8m0,
            tma_aligned_scales=self.is_tma_aligned,
            compile_native=False,
        )
```

This matcher represents the FP8 quantization op in the FX graph — the second half of the pattern we're trying to fuse. While `MatcherSiluAndMul` handles "find the SiLU+mul node," this handles "find the quantization node that consumes SiLU+mul's output."

##### **Constructor — Configuration**

The constructor takes several parameters that control what specific quantization pattern to match:

- **`quant_key`** — same `QuantKey` from the base class. For our case, `kFp8Dynamic128Sym` or `kFp8Dynamic64Sym`.
- **`has_col_major_scales`** — whether scales are stored in transposed (column-major) layout. Maps directly to `is_scale_transposed` in the CUDA kernel from our first set of notes.
- **`is_e8m0`** — whether to use E8M0 scale encoding (an alternative scale representation used by some hardware). For standard FP8 block quant, this is `False`.
- **`is_tma_aligned`** — whether scales need TMA (Tensor Memory Accelerator) alignment for H100's async copy hardware. Another variant that produces a different graph pattern.
- **`match_rocm_aiter`** — ROCm/AMD-specific path. Not relevant for NVIDIA GPUs, we'll skip this.

The key line for our case is:
```python
self.QUANT_OP = QUANT_OPS[quant_key]
```

This stores the **unfused** quantization op — the standalone block quantization kernel that appears in the FX graph before fusion. The pattern matcher needs to find this exact op node in the graph.

##### **`self.quant_fp8 = QuantFP8(...)` — The Native Fallback Module**
```python
self.quant_fp8 = QuantFP8(
    quant_key.scale.static,
    quant_key.scale.group_shape,
    column_major_scales=has_col_major_scales,
    use_ue8m0=is_e8m0,
    tma_aligned_scales=self.is_tma_aligned,
    compile_native=False,
)
```

`QuantFP8` is vLLM's quantization module — it's the high-level PyTorch `nn.Module` that models actually use in their forward pass to do FP8 quantization. Normally, when a model calls `QuantFP8.forward(input)`, it internally calls the custom C++ op (the one stored in `self.QUANT_OP`). But `QuantFP8` also has a "native" path — a pure-PyTorch implementation using standard ops like `torch.clamp`, `torch.round`, `torch.to`, etc., with no custom C++ kernels.

This `self.quant_fp8` instance is only used by `forward_native` — the fallback path when the custom op isn't available. When `forward_native` is called during tracing, `QuantFP8` executes using standard PyTorch ops, and the resulting FX graph contains those standard ops instead of the custom op.

**`compile_native=False`** — this is important. `QuantFP8` normally has an option to `torch.compile` its own forward method for better performance. But here we're already inside a `torch.compile` pass — we're building patterns for the Inductor compiler. If `QuantFP8` tried to `torch.compile` itself during this process, it would trigger a nested compilation: the outer `torch.compile` (which runs the pattern matching pass) would encounter an inner `torch.compile` (from `QuantFP8`), leading to undefined behavior or infinite recursion. Setting `compile_native=False` disables this, ensuring `QuantFP8` runs as plain eager-mode PyTorch when used as a fallback.

---
```python
    def forward_custom(
        self,
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.match_rocm_aiter:
            return self.forward_rocm_aiter(input, scale)

        result = torch.empty(
            input.shape, device=input.device, dtype=self.quant_key.dtype
        )

        if self.quant_key.scale.group_shape.is_per_group():
            if not self.is_tma_aligned:
                assert scale is None
                scale = self.make_scale(input, transposed=self.has_col_major_scales)

            finfo = torch.finfo(self.quant_key.dtype)
            fp8_min = finfo.min
            fp8_max = finfo.max

            _, result, scale = auto_functionalized(
                self.QUANT_OP,
                input=input,
                output_q=result,
                output_s=scale,
                group_size=self.quant_key.scale.group_shape[1],
                eps=1e-10,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=self.is_e8m0,
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )
            return result, scale

        if self.quant_key.scale.static:
            # Static per-tensor path (brief)
            ...

        else:
            # Dynamic per-token path (brief)
            ...
```

##### **The Per-Group Path (Our Focus)**

This is the path taken when `quant_key.scale.group_shape.is_per_group()` returns `True` — which it does for `kFp8Dynamic128Sym` (group shape `(1, 128)`) and `kFp8Dynamic64Sym` (group shape `(1, 64)`).

**Step 1 — Allocate output tensor:**
```python
result = torch.empty(input.shape, device=input.device, dtype=self.quant_key.dtype)
```
Creates an empty FP8 tensor with the same shape as input. This is the quantized output buffer that the unfused quant op will write into. Required by `auto_functionalized`.

**Step 2 — Create scale tensor:**
```python
if not self.is_tma_aligned:
    assert scale is None
    scale = self.make_scale(input, transposed=self.has_col_major_scales)
```

For non-TMA-aligned cases (which is our standard path), the scale tensor is created here via `make_scale`.

##### **`make_scale` and the Transposed View in the FX Graph**
```python
def make_scale(self, input: torch.Tensor, transposed: bool = False) -> torch.Tensor:
    normalized_group_shape = _normalize_quant_group_shape(
        input, self.quant_key.scale.group_shape
    )
    scale_shape = (
        input.shape[0] // normalized_group_shape[0],
        input.shape[1] // normalized_group_shape[1],
    )
    if transposed:
        scale_shape = tuple(reversed(scale_shape))
        return torch.empty(
            scale_shape, device=input.device, dtype=torch.float32
        ).permute(-1, -2)

    return torch.empty(scale_shape, device=input.device, dtype=torch.float32)
```

This computes the correct scale tensor shape from the input shape and group shape. For `group_size=128` with input `(num_tokens, hidden_size)`: scale shape is `(num_tokens, hidden_size/128)`.

When `transposed=True`, something subtle happens that directly affects pattern matching. Let's trace through a concrete example with `input=(4, 512)` and `group_size=128`:

1. `scale_shape` is computed as `(4, 4)` — 4 tokens, 4 groups each.
2. `reversed(scale_shape)` → `(4, 4)` (happens to be same here; for a non-square case like `input=(8, 512)` it would be `(4, 8)` reversed to `(8, 4)`).
3. `torch.empty((4, 4), ...)` — creates a tensor with physical layout `(4, 4)`, strides `(4, 1)`.
4. `.permute(-1, -2)` — returns a **view** with logical shape `(4, 4)` but strides `(1, 4)`.

The `.permute` call doesn't copy data — it just changes how the tensor's dimensions map to memory. But critically, **when `torch.compile` traces this function, the `.permute` becomes a node in the FX graph**. This means the FX graph for a transposed-scale pattern looks different from a non-transposed one:
```
Non-transposed FX graph:
  ... → empty(shape=(4, 4)) → quant_op(output_s=scale, ...) → ...

Transposed FX graph:
  ... → empty(shape=(4, 4)) → permute(-1, -2) → quant_op(output_s=scale, ...) → ...
```

The pattern matcher sees these as structurally different subgraphs. This is why we need separate pattern registrations for `is_scale_transposed=True` and `is_scale_transposed=False` — the FX graphs genuinely look different, and a single pattern can't match both.

**Step 3 — The `auto_functionalized` Call**
```python
_, result, scale = auto_functionalized(
    self.QUANT_OP,
    input=input,
    output_q=result,
    output_s=scale,
    group_size=self.quant_key.scale.group_shape[1],
    eps=1e-10,
    fp8_min=fp8_min,
    fp8_max=fp8_max,
    scale_ue8m0=self.is_e8m0,
    dummy_is_scale_transposed=self.has_col_major_scales,
    dummy_is_tma_aligned=self.is_tma_aligned,
)
```

This wraps the unfused block quantization op. The arguments worth noting:

- `input` — the tensor to quantize (in our fusion, this is the output of SiLU+mul).
- `output_q` — pre-allocated quantized output buffer.
- `output_s` — pre-allocated scale output buffer.
- `group_size` — 128 or 64, extracted from the `QuantKey`.
- `eps=1e-10` — minimum epsilon for numerical safety (analogous to `min_scaling_factor` in the CUDA kernel).
- `fp8_min`, `fp8_max` — the representable range of the FP8 type.

##### **The `dummy_` Argument Trick — Why It Exists**
```python
dummy_is_scale_transposed=self.has_col_major_scales,
dummy_is_tma_aligned=self.is_tma_aligned,
```

These arguments are prefixed `dummy_` because they **don't affect the unfused op's computation at all** — the unfused quant kernel ignores them. They exist purely to make the FX graph distinguishable for different configurations.

Here's the problem they solve: consider two `MatcherQuantFP8` instances, one with `is_scale_transposed=True` and one with `is_scale_transposed=False`. Both call the same `self.QUANT_OP` with the same computational arguments (`input`, `output_q`, `output_s`, `group_size`, etc.). If those were the only arguments, the FX graph would contain identical `auto_functionalized` nodes in both cases — same op, same argument structure. The pattern matcher matches on graph structure, so it would be unable to distinguish the two.

By adding `dummy_is_scale_transposed=True` vs `dummy_is_scale_transposed=False` as explicit keyword arguments, the FX graph nodes now carry different constant values:
```
Transposed pattern FX node:
  auto_functionalized(quant_op, ..., dummy_is_scale_transposed=True, dummy_is_tma_aligned=False)

Non-transposed pattern FX node:
  auto_functionalized(quant_op, ..., dummy_is_scale_transposed=False, dummy_is_tma_aligned=False)
```

These are structurally distinct in the FX graph, so the pattern matcher can register a separate replacement for each. The transposed pattern gets a replacement that writes scales in `[num_groups, num_tokens]` layout; the non-transposed pattern gets a replacement that writes scales in `[num_tokens, num_groups]` layout.

This is combined with the `.permute` difference from `make_scale` (described above) — together, the dummy args and the permute node give the pattern matcher two independent signals to distinguish the variants.

The same trick applies to `dummy_is_tma_aligned` — it creates a third axis of pattern variation without changing the unfused op's behavior.

The return is `_, result, scale` — unpacking the `auto_functionalized` tuple. `_` is the first (unused) return, `result` is the quantized tensor, `scale` is the computed per-group scales.

##### **Other Paths (Brief)**

**Static per-tensor (`self.quant_key.scale.static`):**
```python
_, result = auto_functionalized(
    self.QUANT_OP, result=result, input=input, scale=scale
)
return result, scale
```
Simpler — scale is pre-computed and passed in (not dynamically computed). Only `result` is mutated, scale is unchanged. Used for `kFp8StaticTensorSym`.

**Dynamic per-token (else branch):**
```python
scale = self.make_scale(input)
_, result, scale = auto_functionalized(
    self.QUANT_OP, result=result, input=input, scale=scale, scale_ub=None
)
return result, scale
```
Similar to per-group but with per-token scale shape and no group_size parameter. Scale shape is `(num_tokens, 1)`. Used for dynamic per-token quantization schemes.

##### **`inputs()` Method**
```python
def inputs(self) -> list[torch.Tensor]:
    input = self.empty(5, 16)
    if self.quant_key.scale.static:
        return [input, self.empty_f32(1, 1)]
    return [input]
```

Returns dummy inputs for tracing. For static quant, the scale is an input (pre-calibrated). For dynamic quant (our case), only the input tensor is needed — the scale is computed inside `forward_custom`.

---

#### **Block 4: `SiluMulBlockQuantPattern` Constructor**
```python
class SiluMulBlockQuantPattern(ActivationQuantPattern):
    """
    Fusion for SiluMul+BlockQuant (FP8 dynamic per-group) Pattern.
    Supports group_size 128 and 64 via QuantKey.
    Parameterized on is_scale_transposed for different scale layouts.
    """

    def __init__(
        self,
        quant_key: QuantKey,
        is_scale_transposed: bool = False,
        is_e8m0: bool = False,
        is_tma_aligned: bool = False,
    ) -> None:
        super().__init__(quant_key)
        self.quant_matcher = MatcherQuantFP8(
            quant_key,
            has_col_major_scales=is_scale_transposed,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )
        self.group_size = quant_key.scale.group_shape[1]
        self.is_scale_transposed = is_scale_transposed
        self.is_e8m0 = is_e8m0
        self.is_tma_aligned = is_tma_aligned
```

This is the concrete fusion pattern class for SiLU+mul followed by FP8 dynamic block quantization. It brings together the two matchers — `MatcherSiluAndMul` (created by the base class in **Block 1**) and `MatcherQuantFP8` (created here) — to define a complete "find this two-op sequence and replace with one fused op" pattern.

##### **`super().__init__(quant_key)`**

Calls `ActivationQuantPattern.__init__` (**Block 1**), which does:
- Stores `quant_key` and `quant_dtype`
- Looks up `self.QUANT_OP` (unfused quant op) from `QUANT_OPS`
- Looks up `self.FUSED_OP` (fused kernel) from `FUSED_OPS` — for `kFp8Dynamic128Sym`, this is `torch.ops._C.silu_and_mul_per_block_quant.default`, the CUDA kernel from our first set of notes
- Creates `self.silu_and_mul_matcher = MatcherSiluAndMul()`

##### **`self.quant_matcher = MatcherQuantFP8(...)`**

Creates the quantization matcher configured for this specific pattern variant. The arguments map the pattern's configuration to the matcher:

- `quant_key` — passed through (e.g., `kFp8Dynamic128Sym`)
- `has_col_major_scales=is_scale_transposed` — tells the matcher whether to produce the transposed `.permute` node in the FX graph pattern, and whether to include `dummy_is_scale_transposed=True` in the `auto_functionalized` call (as we covered in **Block 3**)
- `is_e8m0` and `is_tma_aligned` — passed through for the dummy arg differentiation

##### **Stored Configuration**
```python
self.group_size = quant_key.scale.group_shape[1]
self.is_scale_transposed = is_scale_transposed
self.is_e8m0 = is_e8m0
self.is_tma_aligned = is_tma_aligned
```

- `self.group_size` — extracted from the `QuantKey`'s group shape. For `kFp8Dynamic128Sym`, `group_shape` is `(1, 128)`, so `group_shape[1]` = `128`. This will be passed to the fused kernel at replacement time.
- The remaining three booleans are stored for use in the `register` method's `replacement` function.

##### **What This Class Now Has After `__init__`**

At this point, one instance of `SiluMulBlockQuantPattern` holds everything needed to register a fusion:
```
From base class (ActivationQuantPattern):
  self.quant_key            = kFp8Dynamic128Sym
  self.quant_dtype          = Float8_e4m3fn
  self.QUANT_OP             = unfused block quant op
  self.FUSED_OP             = silu_and_mul_per_block_quant (our CUDA kernel)
  self.silu_and_mul_matcher = MatcherSiluAndMul instance

From this class:
  self.quant_matcher        = MatcherQuantFP8 instance (configured for this variant)
  self.group_size           = 128
  self.is_scale_transposed  = False (or True)
  self.is_e8m0              = False (or True)
  self.is_tma_aligned       = False (or True)
```

Each unique combination of `(quant_key, is_scale_transposed, is_e8m0, is_tma_aligned)` produces a separate instance with different matchers that generate structurally different FX graph patterns. This is why the `ActivationQuantFusionPass` later creates multiple instances in a nested loop.

---

#### **Block 5: `SiluMulBlockQuantPattern.get_inputs`**
```python
    def get_inputs(self) -> list[torch.Tensor]:
        scale = self.quant_matcher.empty_f32(1, 1)
        return self.silu_and_mul_matcher.inputs() + [scale]
```

This method constructs the list of dummy example tensors needed for `register_replacement` to trace the `pattern` and `replacement` functions. As we discussed in **Block 1**, PyTorch's pattern matcher needs concrete tensors to trace through both functions and build FX subgraphs.

##### **Breaking Down the Inputs**

Two sources contribute to the input list:

1. **`self.silu_and_mul_matcher.inputs()`** — from **Block 2**, this returns `[torch.empty(5, 4, dtype=bf16, device='cuda')]`. A single BF16 tensor representing the concatenated `[gate | up]` input. Shape `(5, 4)` means 5 tokens, hidden_size*2 = 4 (so hidden_size = 2). The values are arbitrary — only shape and dtype matter for tracing.

2. **`self.quant_matcher.empty_f32(1, 1)`** — creates a `torch.empty(1, 1, dtype=float32, device='cuda')`. This is the scale tensor placeholder.

The final list is: `[input_bf16(5,4), scale_f32(1,1)]`.

##### **Why Is Scale an Explicit Input?**

This might seem surprising — in the per-group dynamic quantization path, the scale is *computed* inside `MatcherQuantFP8.forward_custom` via `make_scale`, not passed in from outside. So why is it listed as an input here?

The reason is how `register_replacement` works. The `pattern` and `replacement` functions must have the **same signature** — same number of positional arguments, same types. The pattern function needs a `scale` argument because it appears in the function signature (even though the per-group path internally creates its own scale via `make_scale`). The `replacement` function also needs it to maintain signature compatibility. The actual value of this dummy scale tensor doesn't matter — it's a structural placeholder that makes the tracing machinery happy.

##### **How These Flow Into `register`**

When `register` is called (next block), it does:
```python
inps = self.get_inputs()
register_replacement(pattern, replacement, inps, fwd_only, pm_pass)
```

PyTorch then calls `pattern(*inps)` and `replacement(*inps)` to trace both functions. The traced FX subgraphs become the pattern template and replacement template stored in the `PatternMatcherPass`.

---

#### **Block 6: `SiluMulBlockQuantPattern.register` — The Pattern-Replacement Core**
```python
    def register(self, pm_pass: PatternMatcherPass) -> None:
        is_scale_transposed = self.is_scale_transposed

        def pattern(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            silu_out = self.silu_and_mul_matcher(input)
            result = torch.empty(
                silu_out.shape,
                device=silu_out.device,
                dtype=self.quant_dtype,
            )
            assert scale is not None
            finfo = torch.finfo(self.quant_dtype)
            _, result, scale = auto_functionalized(
                self.quant_matcher.QUANT_OP,
                input=silu_out,
                output_q=result,
                output_s=scale,
                group_size=self.group_size,
                eps=1e-10,
                fp8_min=finfo.min,
                fp8_max=finfo.max,
                scale_ue8m0=self.is_e8m0,
                dummy_is_scale_transposed=is_scale_transposed,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )
            return result, scale

        def replacement(
            input: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            d = input.shape[-1] // 2
            output_shape = input.shape[:-1] + (d,)
            result = torch.empty(
                output_shape, device=input.device, dtype=self.quant_dtype
            )
            if is_scale_transposed:
                scale = torch.empty(
                    (d // self.group_size, input.shape[0]),
                    device=input.device,
                    dtype=torch.float32,
                ).permute(-1, -2)
            else:
                scale = torch.empty(
                    (input.shape[0], d // self.group_size),
                    device=input.device,
                    dtype=torch.float32,
                )
            at = auto_functionalized(
                self.FUSED_OP,
                out=result,
                input=input,
                scales=scale,
                group_size=self.group_size,
                scale_ub=None,
                is_scale_transposed=is_scale_transposed,
            )
            return at[1], at[2]

        inps = self.get_inputs()
        register_replacement(pattern, replacement, inps, fwd_only, pm_pass)
```

##### **The FX Graph — What It Is and Why We Modify It**

Before diving into the code, let's establish why this machinery exists.

When vLLM uses `torch.compile`, PyTorch traces the model's forward pass and converts it into an **FX graph** — an intermediate representation (IR) of the computation. An FX graph is a directed acyclic graph (DAG) where:

- **Nodes** represent operations (op calls, tensor allocations, function calls)
- **Edges** represent data flow (one node's output feeds into another node's input)

For example, without fusion, a model's FFN layer produces an FX graph fragment like:
```
[input tensor]
      │
      ▼
 auto_functionalized(silu_and_mul_op, input=..., result=...)
      │
      ▼
 [intermediate BF16 tensor — SiLU+mul output]
      │
      ▼
 auto_functionalized(block_quant_op, input=..., output_q=..., output_s=..., group_size=128, ...)
      │         │
      ▼         ▼
 [FP8 tensor] [scale tensor]
```

This graph represents two separate kernel launches — the intermediate BF16 tensor gets written to HBM by the first kernel and read back by the second. The fusion pass transforms this into:
```
[input tensor]
      │
      ▼
 auto_functionalized(silu_and_mul_per_block_quant_op, input=..., out=..., scales=..., group_size=128, ...)
      │         │
      ▼         ▼
 [FP8 tensor] [scale tensor]
```

One node instead of two. One kernel launch instead of two. The intermediate tensor never materializes in HBM. This is the same optimization as the CUDA kernel fusion — but applied at the graph IR level rather than the C++ level.

**Why can we modify the FX graph?** `torch.compile` has a pipeline: trace → optimize → lower to GPU code. The pattern matching pass runs during the "optimize" stage, after tracing but before code generation. This is the window where we can perform graph transformations like fusion.

**Why do we need to?** Because `torch.compile` doesn't automatically know that SiLU+mul followed by block quant can be replaced by a single fused kernel. We have to teach it by registering the pattern-replacement pair.

##### **`is_scale_transposed = self.is_scale_transposed`**
```python
is_scale_transposed = self.is_scale_transposed
```

This captures the instance variable into a local variable before defining the `pattern` and `replacement` closures. This is a Python closure scoping detail — the nested functions close over local variables. Using the local rather than `self.is_scale_transposed` inside the closures avoids capturing `self` entirely, which could cause issues with serialization or garbage collection of the pattern matcher state.

##### **The `pattern` Function — What to Find**
```python
def pattern(
    input: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    silu_out = self.silu_and_mul_matcher(input)
    result = torch.empty(
        silu_out.shape,
        device=silu_out.device,
        dtype=self.quant_dtype,
    )
    assert scale is not None
    finfo = torch.finfo(self.quant_dtype)
    _, result, scale = auto_functionalized(
        self.quant_matcher.QUANT_OP,
        input=silu_out,
        output_q=result,
        output_s=scale,
        group_size=self.group_size,
        eps=1e-10,
        fp8_min=finfo.min,
        fp8_max=finfo.max,
        scale_ue8m0=self.is_e8m0,
        dummy_is_scale_transposed=is_scale_transposed,
        dummy_is_tma_aligned=self.is_tma_aligned,
    )
    return result, scale
```

This function, when traced by `register_replacement`, produces the FX subgraph that the pattern matcher will search for in the model's compiled graph. It describes the unfused computation:

**Step 1 — SiLU+mul:**
```python
silu_out = self.silu_and_mul_matcher(input)
```
Calls `MatcherSiluAndMul.forward_custom(input)` (from **Block 2**), which traces into an `auto_functionalized(SILU_MUL_OP, ...)` node. This is the first node in the pattern subgraph.

**Step 2 — Allocate quant output:**
```python
result = torch.empty(silu_out.shape, device=silu_out.device, dtype=self.quant_dtype)
```
Traces into a `torch.empty` node. This allocation is part of the unfused graph because the standalone quant kernel needs a pre-allocated output buffer.

**Step 3 — Block quantization:**
```python
_, result, scale = auto_functionalized(
    self.quant_matcher.QUANT_OP,
    input=silu_out,          # ← consumes SiLU+mul output
    output_q=result,
    output_s=scale,
    group_size=self.group_size,
    eps=1e-10,
    fp8_min=finfo.min,
    fp8_max=finfo.max,
    scale_ue8m0=self.is_e8m0,
    dummy_is_scale_transposed=is_scale_transposed,
    dummy_is_tma_aligned=self.is_tma_aligned,
)
```

This traces into the second `auto_functionalized` node — the block quant op. Notice `input=silu_out`: the FX graph captures that the quant op's input is the SiLU+mul op's output. This data flow edge is what makes the pattern matcher recognize these two ops as a fusible sequence, not just two independent ops.

The key arguments that make this pattern variant-specific: `group_size` (128 vs 64), `dummy_is_scale_transposed` (True vs False), `dummy_is_tma_aligned` (True vs False). Different values → different FX nodes → different patterns.

**Important:** This `pattern` function doesn't replicate the quant matcher's `forward_custom` exactly — it directly calls `auto_functionalized` with `self.quant_matcher.QUANT_OP` rather than going through `self.quant_matcher(silu_out, scale)`. This is deliberate. The pattern function needs to precisely control what the traced FX graph looks like, matching exactly what the model's compiled graph will contain. Using the matcher's `forward_custom` would add extra nodes (like the `make_scale` call) that may or may not appear in the actual model graph depending on how the model was written.

##### **The `replacement` Function — What to Replace With**
```python
def replacement(
    input: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = input.shape[-1] // 2
    output_shape = input.shape[:-1] + (d,)
    result = torch.empty(
        output_shape, device=input.device, dtype=self.quant_dtype
    )
    if is_scale_transposed:
        scale = torch.empty(
            (d // self.group_size, input.shape[0]),
            device=input.device,
            dtype=torch.float32,
        ).permute(-1, -2)
    else:
        scale = torch.empty(
            (input.shape[0], d // self.group_size),
            device=input.device,
            dtype=torch.float32,
        )
    at = auto_functionalized(
        self.FUSED_OP,
        out=result,
        input=input,
        scales=scale,
        group_size=self.group_size,
        scale_ub=None,
        is_scale_transposed=is_scale_transposed,
    )
    return at[1], at[2]
```

This function, when traced, produces the FX subgraph that **replaces** the pattern. Instead of two ops (SiLU+mul → quant), it produces one fused op.

**Step 1 — Compute output shape:**
```python
d = input.shape[-1] // 2
output_shape = input.shape[:-1] + (d,)
```
The fused op takes the original `[num_tokens, hidden_size * 2]` input and produces `[num_tokens, hidden_size]` output. `d = hidden_size`.

**Step 2 — Allocate output tensors:**
```python
result = torch.empty(output_shape, device=input.device, dtype=self.quant_dtype)
```
The quantized output buffer — FP8, shape `[num_tokens, hidden_size]`.

For scales, the allocation depends on `is_scale_transposed`:
```python
if is_scale_transposed:
    scale = torch.empty(
        (d // self.group_size, input.shape[0]),  # [num_groups, num_tokens]
        device=input.device,
        dtype=torch.float32,
    ).permute(-1, -2)  # logical shape [num_tokens, num_groups], transposed strides
else:
    scale = torch.empty(
        (input.shape[0], d // self.group_size),  # [num_tokens, num_groups]
        device=input.device,
        dtype=torch.float32,
    )
```

This is the same transposed scale logic from `make_scale` in **Block 3** — for the transposed case, the physical memory is `[num_groups, num_tokens]` but the tensor view is permuted to `[num_tokens, num_groups]`. The CUDA kernel (from our first set of notes) knows to write in the transposed physical layout when `is_scale_transposed=True`.

**Step 3 — The fused op:**
```python
at = auto_functionalized(
    self.FUSED_OP,                          # silu_and_mul_per_block_quant
    out=result,
    input=input,                            # original [gate | up] tensor
    scales=scale,
    group_size=self.group_size,
    scale_ub=None,
    is_scale_transposed=is_scale_transposed,
)
return at[1], at[2]                         # (quantized output, scales)
```

`self.FUSED_OP` is `torch.ops._C.silu_and_mul_per_block_quant.default` — the CUDA kernel we analyzed in the first set of notes. The arguments map directly to the kernel's entry point in **Block 9 (Top-Level Entry Point)** of the kernel notes:

- `out` → pre-allocated FP8 output
- `input` → the original `[gate | up]` concatenated tensor (not the SiLU+mul intermediate — that's the whole point of fusion)
- `scales` → pre-allocated scale buffer
- `group_size` → 128 or 64
- `scale_ub` → `None` (no upper bound)
- `is_scale_transposed` → controls scale write layout in the kernel

`at[1]` is the quantized output, `at[2]` is the scales — matching the `pattern` function's return signature `(result, scale)`.

##### **`register_replacement` — Tying It Together**
```python
inps = self.get_inputs()
register_replacement(pattern, replacement, inps, fwd_only, pm_pass)
```

This is where everything comes together. `register_replacement` (from **Block 1**):

1. Calls `pattern(*inps)` — traces the pattern function with the dummy tensors from `get_inputs()` (**Block 5**), producing an FX subgraph of the unfused two-op sequence.
2. Calls `replacement(*inps)` — traces the replacement function with the same dummies, producing an FX subgraph of the fused one-op sequence.
3. Stores both subgraphs as a pattern-replacement pair in `pm_pass`.

Later, when `pm_pass.apply(graph)` runs on the actual model's FX graph, it:
1. Scans the graph for subgraphs structurally matching the `pattern` subgraph.
2. For each match, substitutes the matched subgraph with the `replacement` subgraph.
3. Updates all edges (data flow connections) so downstream nodes consume the replacement's outputs instead of the pattern's outputs.

##### **The Complete Transformation Visualized**

Before fusion (pattern):
```
input [num_tokens, hidden_size * 2]
  │
  ▼
auto_functionalized(silu_and_mul_op, result=empty(...), input=input)
  │
  ▼
silu_out [num_tokens, hidden_size] in BF16  ← THIS TENSOR HITS HBM
  │
  ▼
auto_functionalized(block_quant_op, input=silu_out, output_q=empty(...),
                    output_s=scale, group_size=128,
                    dummy_is_scale_transposed=False, ...)
  │         │
  ▼         ▼
result    scale
[FP8]     [FP32]
```

After fusion (replacement):
```
input [num_tokens, hidden_size * 2]
  │
  ▼
auto_functionalized(silu_and_mul_per_block_quant,
                    out=empty(...), input=input, scales=empty(...),
                    group_size=128, scale_ub=None,
                    is_scale_transposed=False)
  │         │
  ▼         ▼
result    scale
[FP8]     [FP32]
```

The intermediate BF16 tensor is gone. One kernel launch instead of two. This is the graph-level equivalent of the HBM bandwidth saving we analyzed in the CUDA kernel summary.

---

#### **Block 7: `ActivationQuantFusionPass` — The Top-Level Pass**
```python
class ActivationQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses a pre-defined set of custom ops into fused ops.
    It uses the torch pattern matcher to find the patterns and replace them.

    Because patterns can only be registered once, the pass is a singleton.
    This will be addressed in a future version of PyTorch:
    https://github.com/pytorch/pytorch/pull/139321#issuecomment-2452354980
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="activation_quant_fusion_pass"
        )

        pattern_silu_mul_fp8 = SiluMulFp8StaticQuantPattern()
        pattern_silu_mul_fp8.register(self.patterns)

        if silu_and_mul_nvfp4_quant_supported:
            pattern_silu_mul_nvfp4 = SiluMulNvfp4QuantPattern()
            pattern_silu_mul_nvfp4.register(self.patterns)

        if current_platform.is_cuda():
            for quant_key in [kFp8Dynamic128Sym, kFp8Dynamic64Sym]:
                for is_scale_transposed in [False, True]:
                    for is_e8m0 in [True, False]:
                        for is_tma_aligned in [False, True]:
                            SiluMulBlockQuantPattern(
                                quant_key,
                                is_scale_transposed=is_scale_transposed,
                                is_e8m0=is_e8m0,
                                is_tma_aligned=is_tma_aligned,
                            ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: torch.fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return VllmInductorPass.hash_source(
            self,
            ActivationQuantPattern,
            SiluMulFp8StaticQuantPattern,
            SiluMulNvfp4QuantPattern,
            SiluMulBlockQuantPattern,
        )
```

This is the top-level pass class — the entry point that `torch.compile`'s Inductor pipeline calls to perform all activation+quantization fusions. It creates every pattern variant during `__init__`, registers them all, and then applies them to the model's FX graph when `__call__` is invoked.

##### **`VllmPatternMatcherPass` Base Class**

`ActivationQuantFusionPass` inherits from `VllmPatternMatcherPass`, which is vLLM's base class for Inductor passes. It provides integration with `torch.compile`'s pass infrastructure — things like timing, logging, and pass ordering. The important contract is that the pass is callable: `torch.compile` invokes `pass_instance(graph)` to run it.

##### **`@enable_fake_mode`**

This decorator is critical. It wraps the `__init__` method in PyTorch's "fake mode" — a tracing context where `torch.empty(...)`, `torch.zeros(...)`, etc. create **fake tensors** instead of real GPU tensors. Fake tensors have shapes, dtypes, and device metadata but don't allocate actual GPU memory and don't hold real data.

Why is this needed? During `__init__`, every pattern registration calls `pattern(*inps)` and `replacement(*inps)`, which trace through `torch.empty(...)`, `auto_functionalized(...)`, etc. If these created real CUDA tensors, you'd be allocating GPU memory just to register patterns — wasteful and potentially crashing if the GPU is low on memory. Fake mode makes all of this purely symbolic: shapes and dtypes are tracked for graph tracing, but no memory is allocated.

##### **`PatternMatcherPass` — The Pattern Container**
```python
self.patterns: PatternMatcherPass = PatternMatcherPass(
    pass_name="activation_quant_fusion_pass"
)
```

This is PyTorch Inductor's container for pattern-replacement pairs. Every `register_replacement` call (from **Block 6**) adds a pair to this container. When `self.patterns.apply(graph)` is called later, it iterates through all registered patterns and applies matching replacements to the graph. The `pass_name` is used for logging and debugging.

##### **Pattern Registration — The Three Groups**

**1. Static FP8 quantization:**
```python
pattern_silu_mul_fp8 = SiluMulFp8StaticQuantPattern()
pattern_silu_mul_fp8.register(self.patterns)
```
One pattern for `kFp8StaticTensorSym` — simpler because static quant has a fixed scale, so there's no transposed/e8m0/tma variation. Always registered.

**2. NVFP4 quantization:**
```python
if silu_and_mul_nvfp4_quant_supported:
    pattern_silu_mul_nvfp4 = SiluMulNvfp4QuantPattern()
    pattern_silu_mul_nvfp4.register(self.patterns)
```
One pattern for NVIDIA FP4 quantization. Only registered if the platform supports it (CUDA + the op exists in the build).

**3. Dynamic block quantization (our focus):**
```python
if current_platform.is_cuda():
    for quant_key in [kFp8Dynamic128Sym, kFp8Dynamic64Sym]:
        for is_scale_transposed in [False, True]:
            for is_e8m0 in [True, False]:
                for is_tma_aligned in [False, True]:
                    SiluMulBlockQuantPattern(
                        quant_key,
                        is_scale_transposed=is_scale_transposed,
                        is_e8m0=is_e8m0,
                        is_tma_aligned=is_tma_aligned,
                    ).register(self.patterns)
```

This is the nested loop that registers all variants of `SiluMulBlockQuantPattern`. The combinations are:

- `quant_key`: 2 values (`kFp8Dynamic128Sym`, `kFp8Dynamic64Sym`)
- `is_scale_transposed`: 2 values (`False`, `True`)
- `is_e8m0`: 2 values (`True`, `False`)
- `is_tma_aligned`: 2 values (`False`, `True`)

Total: `2 × 2 × 2 × 2 = 16` pattern-replacement pairs registered.

Each combination produces a structurally different FX graph pattern (due to different `group_size` values, `dummy_` arguments, and `.permute` nodes). The pattern matcher needs all 16 because it does exact structural matching — it can't match a `group_size=128` pattern against a `group_size=64` graph node, or a `dummy_is_scale_transposed=True` pattern against a `False` node.

**Why only on CUDA?** The `if current_platform.is_cuda()` guard exists because the fused kernel `silu_and_mul_per_block_quant` is a CUDA kernel — it doesn't exist on ROCm or CPU. ROCm has its own fusion paths via `match_rocm_aiter`.

**The singleton note in the docstring:**
```
Because patterns can only be registered once, the pass is a singleton.
```
PyTorch's `register_replacement` has a global side effect — it modifies internal state in the pattern matcher. Registering the same pattern twice would cause conflicts. vLLM ensures this pass is instantiated only once. The docstring notes a PyTorch PR that aims to fix this limitation.

##### `__call__` — Applying the Pass
```python
@VllmInductorPass.time_and_log
def __call__(self, graph: torch.fx.Graph) -> None:
    self.matched_count = self.patterns.apply(graph)
    logger.debug("Replaced %s patterns", self.matched_count)
```

This is what `torch.compile` invokes during the optimization stage. It receives the model's FX graph and applies all registered patterns:

- `self.patterns.apply(graph)` — scans the graph for all registered pattern subgraphs, replaces matches with their corresponding replacement subgraphs, and returns the count of successful replacements.
- `self.matched_count` — stored for inspection/testing. For a model like Qwen 2.5 with multiple FFN layers, you'd expect `matched_count` to equal the number of FFN layers (each has one SiLU+mul → block quant sequence).
- `@VllmInductorPass.time_and_log` — decorator that logs how long the pass took. Useful for profiling compilation time.

##### **`uuid` — Pass Identity**
```python
def uuid(self) -> str:
    return VllmInductorPass.hash_source(
        self,
        ActivationQuantPattern,
        SiluMulFp8StaticQuantPattern,
        SiluMulNvfp4QuantPattern,
        SiluMulBlockQuantPattern,
    )
```

Returns a unique hash based on the source code of the pass and all pattern classes. `torch.compile` uses this to detect when a pass has changed and needs to invalidate its compilation cache. If you modify any of the listed classes, the UUID changes, and `torch.compile` knows it needs to recompile rather than using a cached compiled graph.

##### **The Complete Lifecycle**

Putting the entire flow together from initialization to execution:
```
1. vLLM startup → torch.compile begins
   │
   ▼
2. ActivationQuantFusionPass.__init__()
   │  @enable_fake_mode — all tensors are fake
   │  Creates PatternMatcherPass container
   │  Registers 1 static FP8 pattern
   │  Registers 1 NVFP4 pattern (if supported)
   │  Registers 16 block quant patterns (2 quant_keys × 2 transposed × 2 e8m0 × 2 tma)
   │  Each registration traces pattern() and replacement() with fake tensors
   │  Total: 18 pattern-replacement pairs stored
   │
   ▼
3. torch.compile traces the model → produces FX graph
   │
   ▼
4. ActivationQuantFusionPass.__call__(graph)
   │  self.patterns.apply(graph)
   │  For each FFN layer in the graph:
   │    - Finds SiLU+mul → block_quant subgraph matching one of the 16 patterns
   │    - Replaces with silu_and_mul_per_block_quant fused subgraph
   │  Returns matched_count
   │
   ▼
5. Optimized FX graph → Inductor lowers to GPU code
   │  Fused op nodes become calls to our CUDA kernel
   │
   ▼
6. Model inference runs with fused kernels
```
---

#### **Summary / Conclusion**

This pattern matching pipeline solves a specific problem in vLLM's `torch.compile` integration: the model's FX graph contains two separate ops — SiLU+mul and block quantization — that we know can be replaced by a single fused CUDA kernel. But `torch.compile` doesn't know this automatically. We have to explicitly teach it what to find and what to replace it with.

##### **The Architecture — Three Layers**

The codebase is structured in three layers, each with a clear responsibility:
```
Layer 1: Matchers (MatcherSiluAndMul, MatcherQuantFP8)
   │  Know how to produce FX subgraph fragments for individual ops
   │  Handle custom op vs native fallback paths
   │  Provide dummy inputs for tracing
   │
   ▼
Layer 2: Pattern Classes (SiluMulBlockQuantPattern)
   │  Compose matchers into complete pattern-replacement pairs
   │  Define the pattern function (what to find)
   │  Define the replacement function (what to substitute)
   │  Handle variant-specific logic (transposed scales, group sizes)
   │
   ▼
Layer 3: Fusion Pass (ActivationQuantFusionPass)
      Registers all pattern variants
      Integrates with torch.compile's Inductor pipeline
      Applies patterns to the model's FX graph at compile time
```

##### **The Key Concepts That Make This Work**

1. **FX Graph as the manipulation target.** `torch.compile` traces the model into a functional DAG of operations. This IR is where we perform graph surgery — replacing subgraphs with semantically equivalent but more efficient alternatives. The FX graph sits at the right abstraction level: high enough to see op-level patterns (SiLU+mul followed by quant), low enough that the replacement maps directly to a CUDA kernel launch.

2. **`auto_functionalized` as the bridge.** vLLM's custom ops are mutating (write into pre-allocated outputs), but `torch.compile` needs functional ops (no side effects). `auto_functionalized` wraps mutating ops to present them as functional — this is what makes custom ops visible and matchable in the FX graph.

3. **Structural pattern matching.** PyTorch's `register_replacement` matches on graph structure — the shape of the DAG, the op types at each node, and the constant values at each argument. It doesn't match on tensor shapes or values. This means one pattern registration works for all tensor sizes, but we need separate registrations for each combination of constant arguments (group_size, dummy flags).

4. **The `dummy_` argument trick.** When two pattern variants would produce identical FX graphs (same ops, same structure), adding dummy constant arguments that differ between variants makes them structurally distinguishable. The unfused kernel ignores these arguments; they exist purely for pattern differentiation.

5. **Fake mode for zero-cost registration.** Pattern registration requires tracing, which requires tensor operations. `@enable_fake_mode` ensures these operations create symbolic tensors with no GPU memory allocation — making it safe to register 18+ patterns during initialization without touching the GPU.

##### **The Variant Explosion — Why 16 Patterns**

For `SiluMulBlockQuantPattern` alone, we register 16 variants:
```
                    quant_key (2)
                   ┌─────┴─────┐
              128Sym           64Sym
             ┌──┴──┐         ┌──┴──┐
         transposed       transposed
         F      T         F      T
        ┌┴┐   ┌┴┐       ┌┴┐   ┌┴┐
       e8m0   e8m0      e8m0   e8m0
       F  T   F  T      F  T   F  T
      ┌┤  ┌┤  ┌┤  ┌┤   ┌┤  ┌┤  ┌┤  ┌┤
     tma tma tma tma   tma tma tma tma
     F T F T F T F T   F T F T F T F T
```

Each leaf is a unique pattern-replacement pair. This combinatorial explosion is the cost of exact structural matching. However:

- **Compile-time only** — the 16 registrations happen once during `torch.compile` initialization, using fake tensors. Zero GPU cost.
- **Only one matches** — at runtime (`__call__`), the model's actual FX graph will match exactly one of the 16 patterns per FFN layer. The pattern matcher checks all 16 but only one fires.
- **The alternative is worse** — without exact matching, you'd need runtime logic inside the replacement to figure out which variant to use, which complicates the graph transformation and risks correctness bugs.

##### **Connection to the CUDA Kernel**

The pattern matching pipeline and the CUDA kernel solve the same problem at different levels:
```
Pattern Matching (Graph Level)              CUDA Kernel (Hardware Level)
─────────────────────────────               ──────────────────────────────
Finds: SiLU+mul node → quant node          Receives: [gate | up] input tensor
Removes: intermediate BF16 tensor node      Avoids: intermediate HBM write/read
Replaces with: single fused op node         Executes: SiLU+mul+quant in one launch
Result: optimized FX graph                  Result: FP8 output + scales

The graph transformation ENABLES the kernel.
Without it, torch.compile would still launch two separate kernels.
The kernel IMPLEMENTS the fusion.
Without it, there's no fused op to replace the pattern with.
```

##### **Where This Fits in vLLM's Compilation Pipeline**
```
Model definition (Python)
    │
    ▼
torch.compile traces forward pass → FX Graph
    │
    ▼
Inductor optimization passes (in order):
    │  ...
    │  ActivationQuantFusionPass  ← THIS PIPELINE
    │    └─ SiLU+mul + block_quant → silu_and_mul_per_block_quant
    │  RMSNorm+Quant fusion pass (similar pattern, different ops)
    │  ...
    │
    ▼
Inductor code generation → Triton/CUDA kernels
    │  Fused op nodes → calls to our pre-compiled CUDA kernel
    │
    ▼
Runtime inference
    │  Each FFN layer executes one fused kernel instead of two
    │  ~2x reduction in HBM traffic for the activation+quant stage
```

---

This concludes the full picture — the fused CUDA kernel ([kernel notes]({{ site.baseurl }}{% post_url 2026-03-24-silu-mul-fp8-block-quant-kernel-vLLM %})) and how it integrates into vLLM's `torch.compile` pipeline. The [PR](https://github.com/vllm-project/vllm/pull/32996) is in final review — I'll update this post once it's merged, along with any related changes I make along the way.

There are still a few more important pieces beyond the kernel and pattern matching: extensive tests, benchmarks, and end-to-end evals, plus getting CI checks to pass. For those, you can take help of AI-assisted coding tools — they're well-suited for generating test cases and boilerplate. But make sure to be thorough and reason through the results yourself. A passing test doesn't mean correctness if the test isn't checking the right thing, and a good benchmark number doesn't mean much if the baseline is wrong.

I did fall into this trap, but was able to reason my way out. So, be careful. Also, this work took significant time, as the code repo is extensive and new to me. I'm hoping with this base, I'll be able to move faster with the next kernels — let's see. Happy coding!

