---
layout: post
title: Investigating Flaky `test_eagle_dp` — Batch Invariance Failure on L4 GPUs
date: 2026-04-13 22:13:00-0400
featured: false
description: Investigative notes and fixes for test_eagle_dp CI tests in vLLM
tags: GPU
categories: GPU-NYU
giscus_comments: true
related_posts: false
---

These are my personal notes from my investigation and fix to one of the CI environment checks in vLLM. The goal is to document this work, so I can reference this later without having to re-derive everything from scratch.

**A few things before we proceed:**

- These notes were generated with the help of Claude, given the all the context and my understandings. If you spot any reasoning errors, please let me know in the comments.
  
**Issue**: [vllm-project/vllm#31913](https://github.com/vllm-project/vllm/issues/31913)  
**Related**: [pytorch/pytorch#170563](https://github.com/pytorch/pytorch/issues/170563), [vllm PR #38566](https://github.com/vllm-project/vllm/pull/38566), [vllm PR #31915](https://github.com/vllm-project/vllm/pull/31915)  
**Fix PR**: [vllm-project/vllm#38938](https://github.com/vllm-project/vllm/pull/38938)  
**Follow-up Issue**: [vllm-project/vllm#39096](https://github.com/vllm-project/vllm/issues/39096) — Batch invariance breaks with torch.compile and/or CUDA graphs on SM<90

---

#### **1. Problem Statement**

The test `tests/v1/distributed/test_eagle_dp.py::test_run_eagle_dp[FLASH_ATTN]` is flaky on CI. It runs two engines sequentially on the same prompt with `temperature=0` (greedy sampling):

- **Engine A (EAGLE)**: Uses speculative decoding with a draft model. The target model verifies draft tokens in batches of `1 + num_draft_tokens = 4`.
- **Engine B (no EAGLE)**: Normal autoregressive decoding. The target model processes `1` token per decode step.

The test asserts `output_A == output_B`. With `VLLM_BATCH_INVARIANT=1` set, this equality should hold — speculative decoding must not change the final output. The test fails intermittently on CI (L4 GPUs) but passes on other hardware.

---

#### **2. Initial Hypotheses**

From the Slack discussion and issue tracker, three hypotheses existed:

1. **Async scheduling bug**: Nicolò (vLLM maintainer) observed he "couldn't reproduce when async correction is disabled." This pointed at the `AsyncScheduler` code path.
2. **Batch invariance failure**: zou3519 (PyTorch) hypothesized "something is either wrong with batch invariance or something else wrong with how our spec decoding is implemented."
3. **Zero-bubble spec decode interaction**: Matt Bonanni noted that timing changes from zero-bubble spec decode could surface an existing issue.

We investigated all three through static code analysis and targeted instrumentation.

---

#### **3. Static Code Analysis**

##### **3.1 Async Scheduling Path**

We traced the full lifecycle of EAGLE + async scheduling across these files:

| File | Role |
|------|------|
| `vllm/v1/engine/core.py` | Engine core loop: `step()` → `post_step()` |
| `vllm/v1/core/sched/scheduler.py` | Base scheduler: `schedule()`, `update_from_output()` |
| `vllm/v1/core/sched/async_scheduler.py` | Async scheduler: placeholder-based draft token tracking |
| `vllm/v1/worker/gpu/model_runner.py` | GPU model runner: `execute_model()`, `sample_tokens()` |
| `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` | EAGLE draft model: `propose()` |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | Rejection sampling: accept/reject draft tokens |
| `vllm/v1/worker/gpu/input_batch.py` | Input construction: `combine_sampled_and_draft_tokens()` |
| `vllm/v1/worker/gpu/async_utils.py` | Async D2H copy for output |
| `vllm/v1/request.py` | Request state: `num_output_placeholders`, `spec_token_ids` |
| `vllm/config/scheduler.py` | Scheduler config: `async_scheduling` resolution |

**Key finding**: When `async_scheduling=None` (default), it resolves to `True` for EAGLE models. Both the EAGLE and non-EAGLE engines run with async scheduling. The `AsyncScheduler` uses placeholder draft tokens `[-1, -1, -1]` and relies on the worker to fill in real draft tokens from GPU state.

We traced the full state transitions for prefill → first decode → subsequent steps and found the logic traced correctly on paper. No obvious bugs in the async scheduler state tracking, D2H copy synchronization, or draft token handling.

##### **3.2 Batch Invariance Implementation**

We examined `vllm/model_executor/layers/batch_invariant.py` and all 50+ locations where `VLLM_BATCH_INVARIANT` is checked.

**What batch invariance does**:

When enabled, it replaces standard GPU operations with deterministic alternatives:

- **Matrix multiply** (`torch.mm`, `torch.bmm`, `torch.matmul`, `torch.linear`): Replaced with custom Triton persistent kernels that use a fixed K-reduction order per output tile, independent of batch size.
- **FlashAttention**: Forces `num_splits=1`, ensuring each query's attention is computed by exactly one worker in one fixed order.
- **Softmax, log_softmax, mean**: Replaced with Triton kernels that process one row per thread block.
- **RMS norm**: Replaced with a Triton kernel with fixed reduction order.
- **cuBLAS settings**: Disables reduced precision, forces `cublaslt` backend.
- **NCCL settings**: Forces deterministic all-reduce.

**How these overrides are registered** (in `enable_batch_invariant_mode()`):

```python
_batch_invariant_LIB = torch.library.Library("aten", "IMPL")
_batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")
# etc.
```

`torch.library.Library("aten", "IMPL")` creates a library that overrides the implementation of existing PyTorch aten operators. When you call `.impl("aten::mean.dim", mean_batch_invariant, "CUDA")`, it means: whenever `aten::mean.dim` is dispatched on CUDA, use the custom `mean_batch_invariant` function instead of PyTorch's default. This works in eager mode because every op goes through the dispatch table.

**Hardware-specific code path** in `enable_batch_invariant_mode()`:

```python
if (
    current_platform.is_device_capability_family(100)  # Blackwell
    or current_platform.is_device_capability_family(80) # A100, L4/RTX 4090 (SM80 family)
):
    # Full Triton persistent matmul replacements
    _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "CUDA")
    _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "CUDA")
else:
    # H100 (SM90): only cuBLAS workspace workaround
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
```

Both A100 (SM80) and L4 (SM89) take the same code path — full Triton persistent matmul replacements. FlashAttention `num_splits=1` is enforced on both.

##### **3.3 FlashAttention `num_splits` Enforcement**

Three locations in `vllm/v1/attention/backends/flash_attn.py` set `num_splits`:

| Line | Context | Setting |
|------|---------|---------|
| 1019 | Encoder attention (`causal=False`) | `num_splits=1 if batch_invariant else 0` |
| 1165 | Cascade attention (prefix) | `num_splits=1 if BATCH_INVARIANT else max_num_splits` |
| 1190 | Cascade attention (suffix) | `num_splits=1 if BATCH_INVARIANT else max_num_splits` |

For this test (Llama decoder-only, `VLLM_BATCH_INVARIANT=1` disables cascade attention), only the main decode path is used, which correctly enforces `num_splits=1` via `attn_metadata.max_num_splits` set at line 426-427.

---

#### **4. Instrumentation Approach**

Since static analysis couldn't pinpoint the root cause, we added targeted logging at 6 points across 3 source files, controlled by a single tag `[spec_decode_debug]` for easy filtering.

##### **4.1 Scheduler Instrumentation (Files 2 & 3)**

Added to `async_scheduler.py` and `scheduler.py`:

| Point | Location | What it captures |
|-------|----------|-----------------|
| `async_sched.after_schedule` | End of `_update_after_schedule()` | `num_computed_tokens`, `num_output_placeholders`, `spec_token_ids`, `is_prefill_chunk` |
| `async_sched.update_output` | End of `_update_request_with_output()` | New token IDs, placeholder count, cache position |
| `sched.update_from_output.spec` | After rejection handling in `update_from_output()` | Generated tokens, accepted/rejected counts |
| `sched.update_from_output.tokens` | After `_update_request_with_output()` | Final new tokens, output token history |

##### **4.2 Model Runner Instrumentation (File 1)**

Added to `vllm/v1/worker/gpu/model_runner.py`:

| Point | Location | What it captures |
|-------|----------|-----------------|
| `logits` | After `self.model.compute_logits()` in `sample()` | Top-5 logit values and IDs, argmax |
| `prepare_inputs` | After `combine_sampled_and_draft_tokens()` | Input IDs, positions, draft token count |
| `sample_result` | After `self.sample()` in `sample_tokens()` | Sampled token IDs, num_sampled, num_rejected |
| `post_propose` | After `speculator.propose()` | Last sampled tokens, new draft tokens |

**Note**: The model runner instrumentation ran in the worker subprocess, whose stdout/stderr is not captured in CI logs. Only the scheduler instrumentation (running in the EngineCore process) was captured.

##### **4.3 Test Modification (File 4)**

Modified `test_eagle_dp.py` to print detailed divergence info on failure: exact index, surrounding tokens from both engines, and full output sequences.

---

#### **5. Experimental Results**

##### **5.1 A100 Runs (2× A100 80GB, vast.ai)**

**20 consecutive runs, all passed.**

Key observations:
- **100% deterministic**: Identical token sequences, identical acceptance patterns, identical draft token behavior across all 20 runs.
- **Acceptance pattern**: Exactly 53 spec decode steps, 29 full rejections, 24 with some acceptance — identical every run.
- **Token output**: EAGLE and non-EAGLE engines produced identical output every time.
- **The EAGLE engine's output** (`[323, 10344, 13, 578, 1296, 374, ...]`) **matches the "wrong" output seen in CI failures**, meaning the EAGLE engine is consistent across hardware — it's the non-EAGLE engine that differs.

##### **5.2 L4 CI Runs**

###### **Failure 1 — Divergence at token 81:**

```
Context: ...4188, 271, 2
EAGLE  (batch=4): argmax → 20400
No-EAGLE (batch=1): argmax → 4324
```

```
EAGLE:  generated=[271, 2, 20400] accepted=2 rejected=1
No-EAGLE: final_new_tokens=[271], [2], [4324]  (one at a time)
```

###### **Failure 2 — Divergence at token 33:**

```
Context: ...16572, 389, 220
EAGLE  (batch=4): argmax → 1490
No-EAGLE (batch=1): argmax → 4728
```

```
EAGLE:  generated=[1490] accepted=0 rejected=3
No-EAGLE: final_new_tokens=[4728]
```

###### **Failure 3 (after lm_head fix) — Divergence at token 32:**

```
Context: ...16572, 389, 220
EAGLE  (batch=4): argmax → 4728
No-EAGLE (batch=1): argmax → 1490
```

Note: the tokens **swapped** compared to Failure 2 — the lm_head fix changed which path produced which token, but the upstream divergence in the transformer layers remained.

In all cases: identical model, identical weights, identical input context, different batch size → different greedy output. This directly violates batch invariance.

---

#### **6. Root Cause Analysis**

##### **6.1 Phase 1: Finding the lm_head Gap**

We traced the Llama forward pass operation by operation, cross-referencing each against the batch invariance overrides.

**The Llama forward pass chain:**

```
input_ids
  → embed_tokens (embedding lookup, not a matmul)
  → 32× LlamaDecoderLayer:
      → input_layernorm (RMSNorm)
      → self_attn:
          → qkv_proj (linear) → RoPE → FlashAttention → o_proj (linear)
      → post_attention_layernorm (RMSNorm)
      → mlp:
          → gate_up_proj (linear) → SiluAndMul → down_proj (linear)
  → norm (final RMSNorm)
  → compute_logits → lm_head.quant_method.apply → matmul → argmax
```

**Key finding**: Two different `apply` methods for unquantized layers:

`UnquantizedLinearMethod.apply` (all model linear layers — qkv, o_proj, gate_up, down):
```python
def apply(self, layer, x, bias=None):
    if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
        return linear_batch_invariant(x, layer.weight, bias)  # ✅ COVERED
    return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

`UnquantizedEmbeddingMethod.apply` (lm_head — the final logits projection):
```python
def apply(self, layer, x, bias=None):
    return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)  # ❌ NOT COVERED
```

The lm_head's `apply` method had no `VLLM_BATCH_INVARIANT` check. It always used `dispatch_unquantized_gemm()` (which calls `F.linear` → cuBLAS), never the batch-invariant Triton persistent kernel. This is the operation that computes `hidden_states @ lm_head_weight.T → logits`, whose output is directly argmax'd to pick the next token.

**Fix 1**: Added the batch invariant check to `UnquantizedEmbeddingMethod.apply` in `vllm/model_executor/layers/vocab_parallel_embedding.py`.

##### **6.2 Phase 2: Finding the RMSNorm Gap**

After Fix 1, CI still failed (Failure 3 above), proving there was at least one more uncovered operation upstream. The token swap between Failures 2 and 3 confirmed the lm_head fix changed behavior but didn't resolve the root cause.

We traced the RMSNorm dispatch path and discovered the interaction with `torch.compile`:

**The `CustomOp` dispatch system:**

vLLM's `CustomOp` base class has two forward paths:
- `forward_cuda`: Used in eager mode. Contains explicit batch invariant checks (calls `rms_norm_batch_invariant()` directly).
- `forward_native`: Used under torch.compile. Contains pure PyTorch ops (`x.pow(2).mean(dim=-1)`) for Inductor to compile.

Which path is used is determined by `CustomOp.dispatch_forward()`:
```python
def dispatch_forward(self, compile_native):
    enabled = self._enforce_enable or self.enabled()
    if not enabled:
        return self.maybe_compile(self.forward_native, enable=compile_native)
    # ... platform checks ...
    return self.forward_cuda
```

`self.enabled()` checks `custom_ops` in the compilation config:
```python
@classmethod
def enabled(cls):
    compilation_config = get_cached_compilation_config()
    custom_ops = compilation_config.custom_ops
    enabled = f"+{cls.name}" in custom_ops
    disabled = f"-{cls.name}" in custom_ops
    return (CustomOp.default_on() or enabled) and not disabled
```

`default_on()` returns `True` if `"all"` is in `custom_ops`, `False` if `"none"` is in `custom_ops`.

**The critical chain for this test:**

1. Test sets `enforce_eager=False` → torch.compile is active
2. In `vllm/config/vllm.py`, when Inductor is the backend with compilation enabled:
   ```python
   if all(s not in self.compilation_config.custom_ops for s in ("all", "none")):
       if self.compilation_config.backend == "inductor" and self.compilation_config.mode != CompilationMode.NONE:
           self.compilation_config.custom_ops.append("none")  # ← THIS
       else:
           self.compilation_config.custom_ops.append("all")
   ```
3. `custom_ops = ["none"]` → `default_on()` returns `False`
4. `RMSNorm.enabled()` returns `False` (no `+rms_norm` in `custom_ops`)
5. `dispatch_forward()` → `forward_native` (not `forward_cuda`)
6. `forward_native` → `ir.ops.rms_norm` → `x.pow(2).mean(dim=-1, keepdim=True)`
7. Inductor lowers these ops to its own IR and generates its own Triton reduction kernels
8. **These Inductor-generated kernels bypass the `aten::mean.dim` batch invariant override** — Inductor has its own lowerings that don't go through the aten dispatch table
9. The Inductor-generated reduction kernel produces batch-size-dependent results on L4
10. This propagates through 32 transformer layers (64 RMSNorm calls), accumulates, and flips the argmax

**Why `override_envs_for_invariance` didn't help**: This function (called by `init_batch_invariance`) sets NCCL, cuBLAS, and AOT compile env vars, but **never touches `custom_ops`**. There is no code anywhere in vLLM that forces custom ops on when `VLLM_BATCH_INVARIANT=1`.

**Why `init_batch_invariance` can't fix this**: It runs in the worker subprocess during `gpu_worker.py`, but `CustomOp.dispatch_forward()` is called during `RMSNorm.__init__()` when the model is constructed. By the time the worker calls `init_batch_invariance`, the RMSNorm instances have already bound `_forward_method = forward_native`.

**Fix 2**: Added `compilation_config={"custom_ops": ["none", "+rms_norm"]}` to the test's `AsyncEngineArgs`, forcing RMSNorm to dispatch to `forward_cuda`.

##### **6.3 Why the Existing Batch Invariant Tests Didn't Catch This**

The e2e batch invariant tests (`tests/v1/determinism/test_batch_invariance.py`) set:

```python
enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90,
```

On L4 (SM89), `IS_DEVICE_CAPABILITY_BELOW_90` evaluates to `True`, so they run in eager mode → `custom_ops=["all"]` → `forward_cuda` is used → batch invariant RMSNorm works correctly. These tests never hit the Inductor `forward_native` path on L4.

The EAGLE DP test is the only test that uses `enforce_eager=False` unconditionally on L4, which is why it's the only test that exposed this gap.

##### **6.4 Why L4 But Not A100**

Both GPUs take the same batch invariance code path (Triton persistent matmul, `num_splits=1`, etc.). The uncovered operations (RMSNorm via Inductor, and previously lm_head) produce slightly different results depending on batch size due to GPU thread scheduling, memory access patterns, or internal reduction order differences.

On A100, either:
- The uncovered operations happen to be more numerically stable, or
- The logit gaps between top candidates are large enough that the tiny differences don't flip any argmax.

On L4, at specific positions in the sequence, the top-2 logit candidates are so close (within ~1e-6) that the uncovered operation noise flips the argmax.

##### **6.5 Why It's Flaky**

The test doesn't always fail because:
- Whether the sequence reaches a "tie-breaker" position (where top-2 logits are within the noise margin) depends on the specific token sequence.
- With 100 tokens, there are 100 chances to hit such a position.
- On some runs, the sequence never encounters a sufficiently close tie.
- Increasing the token count makes failures more frequent (more chances for a tie).
- The divergence position varies between runs (token 32, 33, 80 observed).

This matches zou3519's observation: "For pytorch 2.9 I had to adjust to 600; for PyTorch 2.10 adjusting to 50 will cause a failure."

---

#### **7. Fixes Applied**

##### **Fix 1: `UnquantizedEmbeddingMethod.apply`**

**File**: `vllm/model_executor/layers/vocab_parallel_embedding.py`

Added the missing batch invariant check to the lm_head's apply method:

```python
def apply(self, layer, x, bias=None):
    if envs.VLLM_BATCH_INVARIANT and current_platform.is_cuda_alike():
        return linear_batch_invariant(x, layer.weight, bias)
    return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

##### **Fix 2: RMSNorm Custom Op Under torch.compile**

**File**: `tests/v1/distributed/test_eagle_dp.py`

Added `compilation_config` to force RMSNorm's `forward_cuda` path:

```python
engine_args = AsyncEngineArgs(
    model=target_model,
    enforce_eager=False,
    compilation_config={"custom_ops": ["none", "+rms_norm"]},
    ...
)
```

This is scoped to the test as a workaround. The broader issue of batch invariant overrides being bypassed under torch.compile is tracked in [#39096](https://github.com/vllm-project/vllm/issues/39096).

##### **Long-term Fix (Proposed)**

When `VLLM_BATCH_INVARIANT=1`, either:
1. Automatically add `+rms_norm` (and any other custom ops with batch invariant logic in `forward_cuda`) to `custom_ops` in `vllm/config/vllm.py`
2. Ensure the `aten::IMPL` overrides are respected by Inductor during lowering, so `forward_native` also produces batch-invariant results

---

#### **8. Slack Thread Discussion Summary**

Key exchanges from the vLLM Slack thread on this issue:

**Q**: "Are we running this test with `VLLM_BATCH_INVARIANT=1`? And you're saying that torch.compile'd reductions are not guaranteed to be batch invariant?"

**A**: Yes, the test sets `VLLM_BATCH_INVARIANT=1` (via `monkeypatch.setenv`). The issue is that vLLM's batch invariance overrides PyTorch's aten operators via `torch.library.Library("aten", "IMPL")`. When RMSNorm uses `forward_native` under torch.compile, Inductor lowers the ops to its own IR and generates its own Triton reduction kernels — these bypass the aten dispatch table entirely, so the `aten::mean.dim` override never gets called.

**Q**: "Can you check what our e2e batch invariant tests do with RMS norm? Do they use `--enforce-eager`?"

**A**: The e2e tests set `enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90`, so on L4 (SM89) they run in eager mode → `custom_ops=["all"]` → `forward_cuda` is used. The EAGLE DP test uses `enforce_eager=False` unconditionally, which is the only test that exposes this gap on L4.

**Decision**: Scope the RMSNorm fix to the test via `compilation_config`, keep the `UnquantizedEmbeddingMethod.apply` fix as a code-level fix, and file a separate issue ([#39096](https://github.com/vllm-project/vllm/issues/39096)) for the broader torch.compile + batch invariance interaction.

---

#### **9. Evidence Summary**

| Observation | Implication |
|-------------|-------------|
| 20/20 passes on A100 with identical output | Batch invariance works on SM80 in eager mode |
| Divergence at varying token positions on L4 (32, 33, 80) | Not a systematic logic bug; position-dependent |
| Identical context through divergence point, different argmax | Same input → different output depending on batch size |
| EAGLE engine output is consistent across A100 and L4 | The EAGLE path itself is correct |
| Token swap after lm_head fix (Failure 2 vs 3) | lm_head was one gap, but upstream divergence remained |
| `custom_ops=["none"]` under torch.compile | RMSNorm's `forward_cuda` (batch invariant) is bypassed |
| E2e batch invariant tests use `enforce_eager=True` on L4 | Tests never exercise the Inductor path on SM89 |
| CI passes consistently after both fixes | Both gaps were the root cause |

---

#### **10. Files Modified**

| File | Changes | Purpose |
|------|---------|---------|
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | Added `VLLM_BATCH_INVARIANT` check to `UnquantizedEmbeddingMethod.apply` | Fix 1: lm_head batch invariance |
| `tests/v1/distributed/test_eagle_dp.py` | Added `compilation_config={"custom_ops": ["none", "+rms_norm"]}`, divergence diagnostics | Fix 2: force RMSNorm `forward_cuda` path |
| `vllm/v1/core/sched/async_scheduler.py` | Added `logger.info` at 2 points | Debugging instrumentation (to be removed) |
| `vllm/v1/core/sched/scheduler.py` | Added `logger.info` at 2 points | Debugging instrumentation (to be removed) |
| `vllm/v1/worker/gpu/model_runner.py` | Added `print` at 4 points | Debugging instrumentation (to be removed) |

---

#### **11. Appendix: Investigation Command Reference**

This section documents the `grep` and `sed` patterns used throughout this investigation. These are general-purpose techniques for tracing execution paths in large codebases.

##### **11.1 Finding Where Things Are Defined**

**Find class or function definitions:**
```bash
# Find class definitions across the codebase
grep -rn "class LlamaForCausalLM\|class LlamaModel\|class LlamaMLP" vllm/model_executor/models/llama.py

# Find all files that define a specific class
grep -rn "class UnquantizedLinearMethod" vllm/model_executor/layers/ --include="*.py"

# Find function definitions
grep -rn "def compute_logits" vllm/model_executor/layers/logits_processor.py
```

**Flags explained:**
- `-r`: Recursive search through directories
- `-n`: Show line numbers (critical for follow-up with `sed`)
- `-l`: Show only filenames (useful for broad searches)
- `--include="*.py"`: Restrict to Python files only
- `\|`: OR operator in grep (escape the pipe)

##### **11.2 Tracing a Feature Across the Codebase**

**Find every file that references a feature flag:**
```bash
# Find all files referencing BATCH_INVARIANT
grep -rn "BATCH_INVARIANT" vllm/model_executor/layers/ --include="*.py" -l

# Find specific usage patterns
grep -rn "BATCH_INVARIANT" vllm/model_executor/layers/ --include="*.py"
```

**Find callers of a specific function:**
```bash
# Who calls dispatch_unquantized_gemm?
grep -rn "dispatch_unquantized_gemm" vllm/ --include="*.py"

# Who calls init_batch_invariance?
grep -rn "init_batch_invariance" vllm/ --include="*.py"
```

##### **11.3 Reading Specific Code Sections**

**Use `sed` to view exact line ranges** (after `grep -n` gives you line numbers):
```bash
# View lines 81-170 of a file
sed -n '81,170p' vllm/model_executor/models/llama.py

# View the UnquantizedLinearMethod class (after grep told us it starts at line 182)
sed -n '182,230p' vllm/model_executor/layers/linear.py

# View RMSNorm forward_cuda method
sed -n '262,323p' vllm/model_executor/layers/layernorm.py
```

**The grep → sed workflow:**
```bash
# Step 1: Find where something is defined
grep -n "def forward_cuda" vllm/model_executor/layers/layernorm.py
# Output: 262:    def forward_cuda(

# Step 2: Read the full method
sed -n '262,323p' vllm/model_executor/layers/layernorm.py
```

##### **11.4 Filtering Search Results**

**Combine grep with grep to filter:**
```bash
# Find custom_ops mentions that relate to defaults
grep -rn '"none"\|"all"' vllm/config/compilation.py | grep -i "custom_ops\|append\|default"

# Find enforce_eager OR BATCH_INVARIANT across test files
grep -rn "enforce_eager\|BATCH_INVARIANT" tests/ --include="*.py" | grep -i "batch_invariant\|enforce_eager"
```

##### **11.5 Finding Files in Directory Trees**

**Use `find` for complex searches:**
```bash
# Find all Python files that define a specific class
find vllm/ -name "*.py" -exec grep -ln "class CompilationConfig" {} \;

# Find all files with "forward" defined in a directory tree
find vllm/model_executor/layers/rotary_embedding/ -name "*.py" -exec grep -ln "def forward" {} \;
```

##### **11.6 Practical Patterns Used in This Investigation**

**Pattern 1: Trace a forward pass chain**
```bash
# Start from the top-level model
grep -rn "class LlamaForCausalLM" vllm/model_executor/models/llama.py
# Find compute_logits
grep -rn "def compute_logits" vllm/model_executor/models/llama.py
# Read the method
sed -n '580,610p' vllm/model_executor/models/llama.py
# It calls self.logits_processor → find that
grep -rn "class LogitsProcessor" vllm/model_executor/layers/logits_processor.py
# Read its forward
sed -n '54,100p' vllm/model_executor/layers/logits_processor.py
# It calls lm_head.quant_method.apply → find the apply method
grep -rn "class UnquantizedEmbeddingMethod" vllm/model_executor/layers/vocab_parallel_embedding.py
sed -n '31,75p' vllm/model_executor/layers/vocab_parallel_embedding.py
```

**Pattern 2: Compare two similar code paths for differences**
```bash
# Side-by-side comparison: does LinearMethod have a check that EmbeddingMethod lacks?
grep -A5 "def apply" vllm/model_executor/layers/linear.py | head -10
grep -A5 "def apply" vllm/model_executor/layers/vocab_parallel_embedding.py | head -10
```

**Pattern 3: Trace a config value from CLI to runtime**
```bash
# Where does custom_ops get populated?
grep -rn 'custom_ops.*append' vllm/config/ --include="*.py"
# Where does "none" get added?
grep -rn 'custom_ops.*none' vllm/config/vllm.py
# Where is it consumed?
grep -rn "custom_ops" vllm/model_executor/custom_op.py
```

**Pattern 4: Verify a fix hasn't been applied elsewhere**
```bash
# Are there other callers that might have the same bug?
grep -rn "dispatch_unquantized_gemm" vllm/ --include="*.py"
# Check each caller for the missing BATCH_INVARIANT check
```

##### **11.7 Tips**

- **Always start with `grep -rn` to get line numbers**, then use `sed -n` to read context. This is faster than opening files in an editor.
- **Use `-l` (files only) first for broad searches**, then narrow down with specific patterns.
- **Pipe grep into grep** to filter: first grep finds all matches, second grep filters to relevant ones.
- **Use `| head -N`** to avoid being overwhelmed by large result sets. Start with `head -10`, increase if needed.
- **`--include="*.py"`** is essential when searching in repos with compiled objects, node_modules, or binary files.
- **The `\|` OR operator** in grep is powerful for finding related concepts together (e.g., `"class Foo\|def bar\|SOME_FLAG"`).
- **When tracing a bug**: start from the symptoms (the divergent output), trace backward to the operation that produces it, then compare that operation's code path between the two cases (batch_size=1 vs batch_size=4).

---

#### **12. Phase 3: Empirical Verification (Luka's Request)**

After Phase 2 concluded with the RMSNorm hypothesis, Luka Govedič (RH) pushed back on Slack to repro where we show that Inductor generates a kernel that is not batch invariant.

He also pointed out that PR [#27660](https://github.com/vllm-project/vllm/pull/27660) ("Batch invariant torch.compile") had already enabled and tested batch invariance under torch.compile on DeepSeek, so RMSNorm under torch.compile *should* work. This raised the possibility that our hypothesis was wrong.

##### **12.1 Discovery of the IR Op Priority System**

While searching for proof, we discovered a second dispatch system we had missed: `IrOpPriorityConfig`.

The engine config in CI logs (line 116229) revealed:
```
'custom_ops': ['none']
ir_op_priority=IrOpPriorityConfig(rms_norm=['native'])
```

This is **separate from** the `CustomOp.dispatch_forward()` system we analyzed in Phase 2. It controls which IR implementation (`native`, `vllm_c`, `oink`, `aiter`) is used inside `vllm.ir.ops.rms_norm`.

**Where it's set** — `vllm/platforms/cuda.py:get_default_ir_op_priority()`:
```python
default = ["native"] if using_inductor else ["vllm_c", "native"]
```

When Inductor is the backend, the IR priority is `["native"]` — meaning use the pure PyTorch reference implementation (`x.pow(2).mean(dim=-1)`) so Inductor can compile it. When not using Inductor, it falls back to the hand-written `vllm_c` CUDA kernel.

Both systems happen to push toward the same `forward_native` path under torch.compile, but they're independent mechanisms. The fix in PR #38938 (`+rms_norm`) works by overriding `CustomOp` dispatch to bind `forward_cuda`, which has an explicit `rms_norm_batch_invariant()` call that bypasses **both** the IR op priority system and the aten dispatch table.

##### **12.2 The Empirical Test in CI**

To prove the hypothesis (or disprove it), we added an inline check at the top of `test_run_eagle_dp` that runs on the same L4 hardware before any vLLM state is set up:

```python
def rms_norm_native(x, weight, eps=1e-5):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype)
    return x * weight

compiled = torch.compile(rms_norm_native, dynamic=False)

# Same row of data, presented as batch=1 and as the first row of batch=4
shared_row = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
x_b1 = shared_row.unsqueeze(0).clone()
x_b4 = torch.randn(4, hidden_size, dtype=torch.bfloat16, device="cuda")
x_b4[0] = shared_row

out_b1 = compiled(x_b1, weight)
out_b4 = compiled(x_b4, weight)

print(f"bitwise_equal: {torch.equal(out_b1[0], out_b4[0])}")
print(f"max_abs_diff: {(out_b1[0].to(torch.float32) - out_b4[0].to(torch.float32)).abs().max().item()}")
```

##### **12.3 Surprising Result**

CI output on L4 (NVIDIA L4, capability 8.9):
```
[inductor_bi_check] device: NVIDIA L4
[inductor_bi_check] capability: (8, 9)
[inductor_bi_check] bitwise_equal: True
[inductor_bi_check] max_abs_diff: 0.0
[inductor_bi_check] n_differing_elements: 0 / 4096
```

A second variant (`v2`) explicitly called `enable_batch_invariant_mode()` first to register the aten overrides. Same result:
```
[inductor_bi_check_v2] WITH enable_batch_invariant_mode()
[inductor_bi_check_v2] bitwise_equal: True
[inductor_bi_check_v2] max_abs_diff: 0.0
[inductor_bi_check_v2] n_differing_elements: 0 / 4096
```

**Inductor's compiled RMSNorm IS batch invariant on L4 — even for the simple isolated case, with or without `enable_batch_invariant_mode()` registered.** This contradicts our Phase 2 hypothesis.

##### **12.4 What This Means**

Our Phase 2 explanation was wrong, or at least incomplete. The simple case shows that:

- Inductor's `mean.dim` codegen is batch invariant on L4 (at least for this shape and dtype)
- The aten override registration doesn't affect this — Inductor produces the same invariant kernel either way
- So the divergence in the actual EAGLE DP test is **not** caused by Inductor's basic RMSNorm codegen being non-deterministic

##### **12.5 Open Questions**

The empirical result raises several possibilities for what's actually causing the EAGLE DP test failures:

1. **The PR #38938 fix may have worked for a different reason** than we documented. The token-level divergence we observed in CI may have been resolved by the fix, but our explanation of *why* may be incorrect.

2. **The issue may be in the fused-add RMSNorm path** (`fused_add_rms_norm`), which is what's actually used in Llama for all layers except the input layernorm. This is a different code path than the standalone RMSNorm we tested.

3. **The issue may be in op fusion** — Inductor fuses RMSNorm with surrounding ops (residual add, next linear's input prep). The fused kernel may behave differently than standalone RMSNorm.

4. **The issue may be in a different op entirely** — RoPE, attention, or activation. RMSNorm being fine doesn't rule out something else being broken.

5. **There may be an interaction with batch invariance machinery** that only manifests in the full graph context (with `enable_batch_invariant_mode()` having modified cuBLAS, NCCL, and reduction precision settings).

##### **12.6 v3 (Failed): vLLM RMSNorm Module Under Compile**

We tried to test the actual `vllm.model_executor.layers.layernorm.RMSNorm` module under `torch.compile`:

```python
from vllm.model_executor.layers.layernorm import RMSNorm
real_norm = RMSNorm(hidden_size, eps=1e-5).cuda().to(torch.bfloat16)
compiled_norm = torch.compile(real_norm, dynamic=False)
```

This crashed with:
```
AssertionError: Current vLLM config is not set. This typically means
get_current_vllm_config() was called outside of a set_current_vllm_config()
context, or a CustomOp was instantiated at module import time or model
forward time when config is not set.
```

`RMSNorm.__init__()` (a `CustomOp` subclass) requires an active vLLM config context to instantiate. The fix is to use the `default_vllm_config` pytest fixture, but this means the test failure result we got was from this assertion, not from a token divergence — so we don't yet know whether the rest of the test would have passed or failed at the actual greedy comparison.

##### **12.7 Next Steps**

1. **Test the fused-add RMSNorm variant** — write a standalone `fused_add_rms_norm_native(x, residual, weight)` function and run the same batch=1 vs batch=4 invariance check.

2. **Test in graph context** — instead of compiling RMSNorm in isolation, compile a small chunk that includes RMSNorm + a residual add + a linear layer, to let Inductor fuse them.

3. **Use the `default_vllm_config` fixture** — to enable testing the actual `vllm.model_executor.layers.layernorm.RMSNorm` module under compile.

4. **Bisect by removing fixes** — temporarily revert the lm_head fix and the RMSNorm fix one at a time to identify which one is actually doing the work, and compare CI outputs to see which divergences appear with which fix removed.

5. **Test other ops in the forward path** — RoPE, SiluAndMul, attention output projection — using the same shape-invariance pattern.

##### **12.8 Updated Status**

The fix in PR #38938 still passes CI consistently, so it's still the right short-term fix even if our explanation needs refinement. The follow-up issue [#39096](https://github.com/vllm-project/vllm/issues/39096) should be updated to reflect that the simple RMSNorm-under-Inductor case is actually batch invariant on L4, and the real source of the divergence is still under investigation.

---

#### **13. Resolution**

##### **13.1 Pivot: Move the Test to H100**

After confirming that `enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90` made the test pass consistently on L4, Benjamin Chislett pushed back on Slack that he would really not like to lose coverage of cuda graphs with this feature and that we could we migrate this test to 2xH100 to get over this issue.


The concern: the `enforce_eager` workaround disables torch.compile AND CUDA graphs on L4. If we kept the test only on L4, we'd lose coverage of the compiled + graph-captured path entirely for this test. Since the SM<90 limitation is well-documented and unlikely to be fixed soon, the better approach is to run the test on hardware where batch invariance + torch.compile + CUDA graphs actually works — H100 (SM90+).

Luka agreed and asked to move the test.

##### **13.2 Additional Experiment: Isolating torch.compile vs CUDA Graphs**

Before moving, Luka also asked whether we could isolate whether the SM<90 issue is specifically `torch.compile` or specifically CUDA graphs — since `enforce_eager=True` disables both. The goal was to determine if we could use `--cudagraph_mode=NONE` alone instead (keeping torch.compile active).

We tested `enforce_eager=False, compilation_config={"cudagraph_mode": "NONE"}` on L4. The test still failed with divergence at token 80 (the same position as the original Phase 1 failure, `20400 != 4324`).

**Result: torch.compile alone is sufficient to break batch invariance on SM89**, even without CUDA graphs. The comment in `tests/v1/determinism/utils.py` ("For devices with SM < 90, batch invariance does not support CUDA Graphs") is incomplete — both `torch.compile` and CUDA graphs contribute to the issue on SM<90, and disabling just one isn't enough. Also, disabling torch.compile while keeping CUDA graphs active isn't a valid configuration in vLLM (graph capture happens during the compiled forward pass), so we can't test the reverse direction.

##### **13.3 Final Fix**

**Two code changes:**

1. **`vllm/model_executor/layers/vocab_parallel_embedding.py`** — Added `VLLM_BATCH_INVARIANT` check to `UnquantizedEmbeddingMethod.apply`. This is a genuine bug in the lm_head projection path and applies to all GPUs, independent of the SM<90 issue. This was Phase 1's finding and it's a real fix.

2. **`.buildkite/test_areas/distributed.yaml`** — Moved `test_eagle_dp` out of the L4 jobs (`Distributed DP Tests (2 GPUs)` and `Distributed DP Tests (4 GPUs)`) and added two new dedicated H100 jobs (`Distributed EAGLE DP Tests (2 GPUs)(H100)` and `Distributed EAGLE DP Tests (4 GPUs)(H100)`). The H100 jobs have focused `source_file_dependencies` covering the Llama model, batch invariance machinery, spec decode, and the test file itself.

**Kept as defensive guard:**

The `enforce_eager=IS_DEVICE_CAPABILITY_BELOW_90` change in `tests/v1/distributed/test_eagle_dp.py` stays. Even though the test now runs on H100 in CI, the workaround is kept so the test still works correctly if anyone runs it locally on an L4 machine, or if it ends up in another L4-based CI job (e.g., `.buildkite/test_areas/model_runner_v2.yaml` which also exercises this test on L4).

##### **13.4 PR Bookkeeping**

The PR had accumulated ~35 commits during the investigation (many were scratch debugging commits + merges from main). Before merging:
- Squashed all commits into a single clean commit using `git reset --soft origin/main` followed by one `git commit`
- Force-pushed to overwrite the noisy history
- CI was retriggered with `git commit --allow-empty -m "Retrigger CI"` once due to a transient safetensors download race on a fresh H100 node (not related to the fix)

##### **13.5 Summary of the Full Journey**

| Phase | Hypothesis | Outcome |
|-------|-----------|---------|
| 1 | Async scheduling bug | Ruled out — scheduler logic traced correctly, A100 runs were 100% deterministic |
| 1 | lm_head missing batch invariance | Confirmed — `UnquantizedEmbeddingMethod.apply` was missing the check. Fixed. |
| 2 | RMSNorm `forward_native` bypasses aten override under Inductor | Disproven empirically — inline CI repro showed `torch.compile(rms_norm)` IS batch invariant on L4 |
| 3 | CUDA graphs on SM<90 are not batch invariant (per `utils.py:14`) | Confirmed as contributing factor, but incomplete — torch.compile alone is also implicated |
| 3 | Move test to H100 where batch invariance + compile + graphs works | Accepted by maintainers and implemented |

##### **13.6 What Generalizes**

A few lessons that generalize beyond this specific bug:

1. **Empirical verification beats plausible explanation.** The Phase 2 RMSNorm hypothesis was internally consistent and looked correct on paper. It took a direct measurement (the inline CI repro) to disprove it. Always test hypotheses with the cheapest possible experiment before committing to an explanation.

2. **Git history encodes institutional knowledge.** The single most valuable step in Phase 3 was finding PR #30018 via `git blame` on `IS_DEVICE_CAPABILITY_BELOW_90`. The comment in `utils.py:14` had the answer we were looking for — we just hadn't looked there. When investigating a bug that feels like it should already be known, check whether someone else has encoded a workaround in similar tests.

3. **Don't conflate similar-sounding dispatch systems.** `CustomOp.dispatch_forward()` and `IrOpPriorityConfig` look similar (both decide which implementation to use) but are separate mechanisms. Phase 2's investigation got tangled up in both because we found one and assumed it was the whole picture. Always map out all dispatch layers before drawing conclusions.

4. **CI logs are the primary evidence, not hypotheses.** Every conclusion in this investigation ultimately came from a log line or a direct measurement. When the maintainer (Luka) pushed back and asked for empirical proof, that pressure forced a real repro, which disproved the hypothesis we'd committed to. Maintainers asking for proof is a feature, not an obstacle.

5. **Sometimes the right fix is moving the test, not fixing the code.** The SM<90 + batch invariance + torch.compile interaction is a known limitation that isn't worth fixing given the available alternatives (run on newer hardware). Accepting that workaround over a deeper fix was the right tradeoff.

##### **13.7 Related Links**

- **PR #38938** — The fix (merged)
- **Issue #39096** — Open follow-up: batch invariance + torch.compile on SM<90
- **Issue #31913** — Original flaky test issue (closed by #38938)
- **PR #30018** — Where `IS_DEVICE_CAPABILITY_BELOW_90` was introduced
- **PR #27660** — Earlier batch invariant torch.compile work (DeepSeek)

