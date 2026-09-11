---
layout: post
title: Serving DeepSeek-V4.1-Flash on vLLM - Architecture, Config, and Where the Time Goes
date: 2026-09-11 14:00:00-0400
featured: false
description: A briefing on DeepSeek-V4.1-Flash for anyone about to tune it on vLLM, covering the new architecture piece by piece (CED, CSA2, the hierarchical indexer, FP4 KV, mHC, Engram, DSpark), what every flag in the recipe command does, what vLLM has and has not landed yet, and where to look first on an 8x B300 node
tags: GPU ML
categories:
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This post is a working briefing on [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), written for one purpose: to be able to tune it on vLLM without guessing. The model landed on 2026-09-10 with a stack of components that did not exist a year ago, a vLLM image that shipped the same day, and a recipe command with eight flags and two environment variables. Before profiling anything, I wanted three things in one place: what each new component is and what it costs to serve, what the launch command actually starts on the node, and what vLLM's implementation has and has not landed as of today.

The target is one 8x B300 node with the recipe's "Data + Expert Parallel" command:

```bash
docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  vllm/vllm-openai:deepseekv41-flash-0909 deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-mode deepseek_v41 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data
```

The plan:

- The numbers that drive everything: the B300 node, the model's sizes, and the arithmetic that follows
- The architecture, one briefing per component: CED, CSA2 and its three modes, the hierarchical indexer, the three caches, the MoE, single-pass mHC, Engram, DSpark, and the ViT
- The launch command, line by line, and what DP8 plus EP actually starts on the node
- The Engram placement problem, and why DP8 is not obviously the right layout
- vLLM day-0 status: what is merged, what is open, and what the image is
- Where to look for performance, as labeled hypotheses with a profiling plan
- Other vendors, briefly
- Takeaways, and a question bank

I'm assuming the [LLM inference systems post](/blog/2026/llm-inference-systems/) as background: the roofline, the decode floor, paged KV, DP attention with EP for MoE, speculative decoding, and CUDA graphs are all used here without re-derivation. Everything quoted from vLLM was checked against `main` at commit `2d75e586` as of 2026-09-11, and the recipe against the [vllm-project/recipes](https://github.com/vllm-project/recipes) repo at commit `6e016941` the same day. Both move daily right now, so treat file paths and defaults as pointers with a date on them. Model facts come from the [tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) and [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json); where I derive a number from those, the computation is shown.

Let's get started.

---

#### **The Numbers That Drive Everything**

##### **The machine**

The recipe's verified hardware is H200, GB200 NVL4, GB300, and MI350X. B300 is offered by the command builder because it fits the model's stated minimum, but it is not on the verified list ([recipe YAML](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml), `hardware:` block, as of 2026-09-11). So the numbers below are the datasheet, not a vendor's measured run.

| | B300 (Blackwell Ultra) | H200 SXM |
|---|---|---|
| HBM per GPU | 288 GB HBM3e | 141 GB HBM3e |
| Bandwidth per GPU | 8 TB/s | 4.8 TB/s |
| FP8 dense, per GPU | 4.5 PFLOPS | 1,979 TFLOPS |
| FP4 dense, per GPU | 13.5 PFLOPS | n/a |
| NVLink per GPU | 1.8 TB/s | 900 GB/s |
| 8-GPU node HBM | 2,304 GB (NVIDIA rounds to 2.1 TB) | 1,128 GB |

Sources: the [Blackwell Ultra architecture post](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) for the 288 GB and 8 TB/s per GPU; the [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) for the 8-GPU HGX B300 figures (FP4 108 PFLOPS dense, FP8 72 PFLOPS with sparsity, "Dense is half sparse", 2.1 TB total), which I divide by 8 for the per-GPU dense rates; the [H200 page](https://www.nvidia.com/en-us/data-center/h200/) for H200. One trap to flag: the architecture post quotes 15 PFLOPS dense FP4 and 5 PFLOPS dense FP8 per chip, about 10% above the HGX product page. I use the HGX numbers because they describe an 8-GPU x86 node, which is what a B300 box is.

The node-level figures that recur: $$8 \times 288 = 2{,}304$$ GB of HBM and $$8 \times 8 = 64$$ TB/s of aggregate bandwidth. The FP8 ridge point per GPU is $$4.5 \times 10^{15} / 8 \times 10^{12} \approx 562$$ FLOP per byte, and roughly triple that at the FP4 rate. Decode on this machine is memory-bound until a step carries several hundred tokens, same story as Hopper with a higher ceiling.

##### **The model**

| Quantity | Value | Source |
|---|---|---|
| Backbone parameters | 552B | [tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf), section 2.1 |
| Engram parameters (separate from the backbone) | 196B | same |
| Active per token, prefill | 8B | same |
| Active per token, decode | 16B | same |
| Layers | 40, as a 20-layer causal encoder plus a 20-layer decoder | same, and `num_hidden_layers` in config.json |
| Hidden size | 5120 | config.json |
| Experts per MoE layer | 384 routed plus 1 shared, top-6 | config.json |
| Expert intermediate size | 2304 | config.json |
| Context | 1,048,576 tokens, YaRN factor 16 over a 65,536 training window | config.json `rope_scaling` |
| Vocabulary | 129,280 | config.json |
| Checkpoint on disk | 510.3 GB (475.2 GiB) across 48 safetensors shards | [HF file tree](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main) |
| Weight formats | FP8 dense with 32x32 blocks and UE8M0 scales; FP4 routed experts | config.json `quantization_config` |

Two of those rows need a note. The vLLM recipe says "522B total parameters" everywhere; the report, the model card, and the report's Table 1 all say 552B, and the recipe's own memory table sums to more than either. Use 552B. And "8B in prefill, 16B in decode" is not a typo: the encoder-decoder design in the next section means most prompt tokens run only half the network.

The expert arithmetic is worth doing once because it explains the disk footprint. A SwiGLU expert is three matrices of $$5120 \times 2304$$:

$$
3 \times 5120 \times 2304 \approx 35.4\text{M parameters per expert}
$$

Across 384 routed experts and 40 layers that is $$384 \times 40 \times 35.4\text{M} \approx 544\text{B}$$, plus the shared experts and the DSpark draft blocks' own experts, which is consistent with the recipe's memory table putting routed plus DSpark experts at 557.2B parameters and 259.5 GiB in MXFP4 ([recipe guide](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)). The same table puts the Engram tables at 188.8 GiB in MXFP8, attention plus norms plus routers at 5.6 GiB, embedding plus LM head at 5.8 GiB, and quantization scales at 21.9 GiB. Those rows sum to about 481.6 GiB, slightly above the 476 GiB the guide states as the total, so treat the breakdown as approximate.

Per decode token, 6 routed experts times 40 layers is $$6 \times 40 \times 35.4\text{M} \approx 8.5\text{B}$$ routed-expert parameters, and the shared expert adds another 1.4B; with the 6.0B of attention and dense weights that reaches the report's 16B.

##### **The caches**

The report's title is "Pushing the Limits of KV Cache Compression", and its headline number is that the **global KV cache, the part that must stay in HBM, is 890 bytes per token**, about a quarter of DeepSeek-V4-Flash, with the persistent prefix cache on SSD and host memory at about an eighth (tech report, abstract and section 3.2.1). Global KV means the compressed main KV plus the indexer keys; the sliding-window cache is bounded by the window and does not grow with context.

How those 890 bytes arise, from config.json: only four layers write a main KV cache (`kv_source_layer_ids` is `[2, 8, 14, 20]`), three of them at compression ratio 2 (one entry per two tokens) and one at ratio 1, so a token adds

$$
3 \times \tfrac{1}{2} + 1 = 2.5 \text{ main KV entries}
$$

across the whole 40-layer network. Each entry is a 512-channel latent (`head_dim` is 512, `num_key_value_heads` is 1). In DeepSeek's FP4 format that is $$512 \times 0.5$$ bytes of E2M1 plus 32 E4M3 scales, one per 16 channels, or 288 bytes per entry, which lands at $$2.5 \times 288 = 720$$ bytes of main KV per token. The report does not publish this split, so the remaining 170 bytes being indexer keys is my reconstruction, not their statement.

vLLM does not store FP4 KV yet. Its layout for this model is `fp8_ds_mla`: 448 bytes of non-RoPE latent, 128 bytes of RoPE part, and an 8-byte scale, **584 bytes per entry** (`vllm/models/deepseek_v4_1/attention.py`, as of 2026-09-11), so the same 2.5 entries cost $$2.5 \times 584 = 1{,}460$$ bytes of main KV per token, roughly double DeepSeek's number. The FP4 layout arrives with the open [attention megakernel PR #56344](https://github.com/vllm-project/vllm/pull/56344), whose description gives 288 bytes per compressed entry and 528 bytes per sliding-window entry. Until it lands, size your KV pool off the vLLM figure, not the paper's.

##### **What the arithmetic says**

Everything below is derived from the numbers above, with assumptions stated.

**KV per sequence.** At vLLM's 1,460 bytes per token of main KV, a 128K-token sequence holds about 191 MB and a 1M-token sequence about 1.53 GB, before the indexer and sliding-window caches. At DeepSeek's 890 bytes it is 117 MB and 933 MB. Either way this is a small model to cache: for comparison, Llama 3 70B at bf16 carries 320 KiB per token, over 200 times more.

**What fits.** Under DP8 with expert parallelism, each GPU holds its slice of the experts (259.5 GiB over 8 is about 32.4 GiB), a full copy of the attention and dense weights (5.6 GiB plus 5.8 GiB), and its share of the scales: roughly 47 GiB per GPU before Engram. Against 268 GiB per B300 and vLLM's default 0.92 utilization (`vllm/config/cache.py`), that leaves on the order of 190 GiB per GPU for KV, activations, and graphs, over a hundred 1M-token sequences per GPU at the vLLM byte count. KV capacity is not the constraint on this node, unless the Engram tables land in HBM; the launch command section returns to that.

**The decode floor.** A single decode token touches the 6 routed experts (8.5B parameters at MXFP4, about 0.53 bytes per parameter with the 1-byte-per-32 scale), the shared expert, and the 6.0B of dense weights at about 1.03 bytes per parameter in MXFP8:

$$
8.5\text{B} \times 0.53 + 7.4\text{B} \times 1.03 \approx 4.5 + 7.6 \approx 12 \text{ GB per token}
$$

Under DP8, attention is replicated, so every rank streams the 7.6 GB of dense weights for its own batch every step; at 8 TB/s that is about 0.95 ms. The experts are split, so at large batch a rank reads only its 32.4 GiB slice, about 4.3 ms when every expert is touched. So the per-step floor on this layout sits between roughly 1 ms at tiny batch and 4 to 5 ms at saturating batch. Hold onto the 1 ms: the only public vLLM measurement of this model at concurrency 1 (on MI355X, discussed in the performance section) reports an inter-token latency around 25 ms, twenty-plus times the byte floor. That gap is not bandwidth.

**Batch at which experts saturate.** With top-6 of 384 and independent routing, the expected fraction of experts touched by a batch of $$B$$ tokens is $$1 - (1 - 6/384)^B$$: about 63% at 64 tokens and 98% at 256. Past a few hundred tokens per step, every step reads the whole expert set, and DSpark's extra draft compute stops paying for itself.

**References**
- [NVIDIA Blackwell Ultra architecture post](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/), [HGX page](https://www.nvidia.com/en-us/data-center/hgx/), [H200 page](https://www.nvidia.com/en-us/data-center/h200/)
- [DeepSeek-V4.1-Flash tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json), [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- [vLLM recipe for DeepSeek-V4.1-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash), [recipe YAML](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)
- vLLM source, as of 2026-09-11: `vllm/models/deepseek_v4_1/attention.py`, `vllm/config/cache.py`; [PR #56344](https://github.com/vllm-project/vllm/pull/56344)

---

#### **The Architecture, One Briefing at a Time**

The report frames every design choice around one goal: make long-context serving cheap by shrinking what has to be cached and what has to be recomputed. Each component below is one attack on that goal, and each is described the same way: what it is, why it exists, and what it costs at serving time.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/deepseek-v41-flash-vllm/layer-map.svg" title="The 40-layer map of DeepSeek-V4.1-Flash" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    All 40 backbone layers and what each one owns, from config.json. Layers 0 and 1 are sliding-window only. The encoder's CSA2 layers run at compression ratio 2 in three groups of six; the decoder's run at ratio 1 in five groups of four. Only the four Full-mode layers write a main KV cache, and only the eight Full and Reindex layers run an indexer; every other layer reuses. Engram modules sit at layers 1 and 14, and the three DSpark draft blocks hang off the end, reading the attention inputs of layers 37 to 39. Editable source: <a href="/assets/img/deepseek-v41-flash-vllm/layer-map.excalidraw">layer-map.excalidraw</a>.
</div>

##### **Causal encoder-decoder**

**What.** The 40 layers are split in half. The bottom 20 are the causal encoder; the top 20 are the decoder. For global attention, the decoder layers do not compute keys and values from their own hidden states. They project them from the encoder's final hidden state $$H_{L/2}$$ with layer-specific weights (tech report, section 2.2, equation 1). Sliding-window attention is the exception: every layer computes its own local keys and values from its own hidden state.

**Why.** Agentic workloads are prefill-heavy: every tool call re-sends a growing context. If the decoder's global KV can be produced from the encoder output, then a prompt token only needs to pass through the encoder to populate the cache that decoding will read. The report puts prefill cost at $$O(NL/2 + n_{win} L/2)$$ for a prompt of $$N$$ tokens, nearly halving it. That is where "8B active in prefill" comes from: half the layers, so half the experts.

**Serving cost.** The catch is the decoder's sliding-window cache, which decoding needs and which the encoder-only prefill did not produce. Exactly rebuilding it means running the decoder over the last $$n_{win} \times L/2$$ prompt tokens. The report's answer is Decoder SWA Bounded Replay: replay only the last 128 tokens through the decoder, accept approximate window states, and it reports negligible quality impact (section 3.2.2). SGLang exposes this as an opt-in flag; in vLLM it is an open PR, so vLLM today runs the full decoder over the prompt and does not yet realize the prefill halving. The vLLM day-0 section has the pointers.

##### **CSA2 and its three modes**

**What.** Compressed Sparse Attention 2 is the global-attention branch. Every CSA2 layer has a main KV cache of 512-channel latents, possibly compressed along the sequence (ratio 2 means one entry per two tokens), and a small **indexer** that scores those entries against an indexer query and keeps the top 512 per query (`index_topk`). Attention then runs over the selected entries plus the layer's own 128-token sliding window. Compared with V4's CSA, the compressor no longer overlaps source windows or adds absolute positions, and the indexer keys are projected from the main KV rather than from hidden states (section 2.3).

The novelty is cross-layer reuse. Each layer is statically one of three modes (section 2.3.1):

| Mode | Main KV | Indexer K | Top-K indices | What it computes itself |
|---|---|---|---|---|
| Full | computes its own | projects from its own KV | runs the indexer | everything |
| Reindex | reuses the most recent Full layer's | reuses | rescores with its own indexer Q, fresh top-K | query, SWA KV, indexer Q, scoring |
| Reuse | reuses | reuses | reuses the latest Full or Reindex layer's | query and SWA KV only |

All three compute their own query and sliding-window KV. The layer assignment, from the report's section 4.2.1 and config.json:

| Layers | Ratio | Modes |
|---|---|---|
| 0, 1 | none | sliding window only |
| 2 to 7, 8 to 13, 14 to 19 | 2 | first of each group Full, next five Reuse |
| 20 to 23 | 1 | first Full, next three Reuse |
| 24 to 27, 28 to 31, 32 to 35, 36 to 39 | 1 | first Reindex, next three Reuse |

So four Full layers own a main KV cache (`kv_source_layer_ids`), eight Full or Reindex layers own an indexer (`index_source_layer_ids`), and the other 26 CSA2 layers own neither.

**Why.** Three multiplicative levers on cache size: entry size (MLA's shared latent), sequence compression (ratio 2), and now the layer dimension. Sharing the main KV across a group of layers divides the cache by the group size; reusing top-K indices divides the indexer work. Decoupling the two is what Reindex mode is for: the selection can change across layers even while the cache does not.

**Serving cost.** For a Reuse-mode layer, decode is a sparse attention over 512 selected entries plus a 128-token window, with no indexer at all. The report says such a layer executes with 15 kernels in prefill and 11 in decode in DeepSeek's stack (section 3.2). In vLLM the modes are not named; they are implied by which layers appear in the two source lists, and consumers "reuse the most recently published source below them" (`vllm/models/deepseek_v4_1/attention.py`, as of 2026-09-11). Compression is why the decoder runs at ratio 1: it reads the encoder's final state, and the report keeps that cache uncompressed.

##### **The hierarchical sparse indexer**

**What.** Decoder-only. The first decoder Full layer (layer 20) scores every visible main-KV position and picks its own top 512. It also scores 8-position blocks by their maximum entry score, keeps the top 2,048 blocks, and publishes those 16,384 positions as a shared **candidate pool** (`candidate_source_layer_id`, `candidate_topk_blocks`, `candidate_block_size` in config.json). The four Reindex layers above it score only the pool (section 2.3.2).

**Why.** Index reuse cuts how many layers run an indexer; it does not cut what each indexer scans. Bounding later indexers to a fixed pool makes their per-query cost constant in context length. The first scan is still full-range, so the per-token indexer cost at 1M context is one full scan plus four bounded ones instead of five full scans. It was introduced in post-training so the model is trained under the same restriction it serves with.

**Serving cost.** In vLLM the layer-20 indexer writes a `candidate_block_buffer` that layers 24 through 36 read (same file). The full-range scan at layer 20 is the piece that still scales with context, and it is where DeepGEMM's sparse MQA-logits kernels and DeepSelect's top-k are aimed; both are open PRs today.

##### **Three caches, two precisions, and bounded replay**

Per token the model keeps three kinds of cache, and they are not stored alike (section 2.4.4):

| Cache | Grows with context? | DeepSeek's format | vLLM's format today |
|---|---|---|---|
| Main KV (4 source layers, 2.5 entries per token) | yes | FP4 E2M1, one E4M3 scale per 16 channels, quantized after RoPE | `fp8_ds_mla`, 584 bytes per entry |
| Indexer K (projected from main KV) | yes | MXFP4 | FP8 by default; MXFP4 on Blackwell via `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` |
| Sliding-window KV (every layer, 128 tokens) | no, bounded by the window | FP8, kept at FP8 "due to its sensitivity to quantization" | FP8, 32-token blocks |

**Why FP4 for the main KV, and why not NVFP4 exactly.** The format is NVFP4 minus its second-level global scale. The report's argument is dynamic range: E4M3 scales times E2M1 values reach magnitudes of $$448 \times 6 = 2688$$, while a 512-channel latent after RMSNorm has an L2 norm of at most about $$\sqrt{512} \approx 22.6$$ and observed maxima near 10, so the global scale buys nothing. FP4 here is a storage format, not a matmul format: values are dequantized before attention, which keeps the cache portable across hardware. The indexer stays on OCP MXFP4 for the same portability reason, and it was already FP4 in V4. Quantization-aware training for the main KV was added in post-training.

**Bounded replay.** The sliding-window caches are the problem child of a persistent prefix cache: they are cheap to hold in HBM but expensive to persist, and reconstructing them exactly means replaying $$L \times n_{win}$$ tokens. SWA Bounded Replay replays only the last 128 tokens and truncates each layer's window to the replayed segment (section 3.2.2). Two uses: **encoder replay** lets DeepSeek drop SWA KV from the persistent cache entirely, keeping it in a host-memory pool with a minutes-long TTL, which is where the "1/8 persistent cache" comes from; **decoder replay** is the CED companion from above. Both are approximate by construction, and both are absent from vLLM main today.

##### **The MoE**

Standard DeepSeekMoE shapes with bigger numbers: 384 routed experts, 1 shared, 6 active, expert intermediate 2304, in every layer. Routing uses `sqrtsoftplus` scoring with the auxiliary-loss-free `noaux_tc` bias, `routed_scaling_factor` 1.5, and SwiGLU clamped at 10 (config.json). The multimodal twist is a second set of routing biases for image tokens, so vision and text balance separately (section 2.1.1); vLLM implements this by passing raw token ids through to the router.

Serving-wise this is the familiar MoE picture from the inference post: all 259.5 GiB of experts must be resident somewhere on the node, routing turns into all-to-all traffic, and skew across experts is the throughput risk EPLB exists for. The vLLM-specific detail is that the MegaMoE kernel backends (DeepGEMM's `fp8_fp4_mega_moe` and the FlashInfer variants) require expert parallelism and SM100 (`vllm/models/deepseek_v4/nvidia/model.py`, as of 2026-09-11), which is one reason the recipe pairs EP with every single-node strategy.

##### **Hyper-connections: single-pass mHC and Mega-mHC**

**What.** mHC, introduced with V4, carries the residual stream as $$n = 4$$ parallel copies (`hc_mult`). Between blocks, three token-wise coefficient sets mix them: $$A_l$$ picks the block input, $$B_l$$ updates the streams, $$C_l$$ injects the block output. $$B_l$$ is made doubly stochastic by 20 Sinkhorn iterations (`hc_sinkhorn_iters`). V4 implemented the update as three dependent kernels, because $$A_l$$ depends on a reduction over all of $$X_l$$ before the input mix can start, costing $$(4n + 4)d$$ of activation traffic against a lower bound of $$(2n + 2)d$$ (section 2.4.1).

**Why single-pass.** Shift the input-mixing coefficient by one block: block $$l$$ mixes its input with $$A_{l-1}$$ instead of $$A_l$$. The dependency disappears, each tile of $$X_l$$ can be used for input mixing and coefficient prediction at once, and the report says the shift costs negligible quality. Pre-training keeps the multi-kernel form; deployment fuses residual update, input mixing, coefficient prediction, pre-norm, and FP8 conversion into one **Mega-mHC** kernel that reads and writes the residual exactly once, halving activation traffic.

**Serving cost.** This is the component to watch. It sits at every sublayer seam, so it runs 80-plus times per token, on a tiny $$4 \times 4$$ matrix per token. In vLLM the mHC pre and post ops are TileLang kernels with the 20 Sinkhorn iterations, warmed up at startup over token sizes up to 16,384 (`vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`, as of 2026-09-11). DeepGEMM's Mega-mHC is on the kernel tracker, not integrated. On the AMD path, the one public trace of this model attributes 85% of per-layer kernel launches to the mHC block (more in the performance section).

##### **Engram**

**What.** Engram is a conditional memory: a very large embedding table addressed by hashes of the last few tokens, added to the residual stream through a learned gate. It is DeepSeek's way to "decouple memorization from computation": store facts in a lookup instead of in matmul weights. V4.1-Flash carries two modules, at layers 1 and 14, with 196B parameters between them. Each module hashes n-grams of orders 2, 3, and 4 over a compressed vocabulary of 99,092 ids, through 8 hash heads per order, into tables of about 16M rows per head with distinct prime sizes; each row is 256 channels in FP8 (section 2.4.2; `engram_*` fields in config.json). The two tables have 384,006,168 and 384,016,682 rows respectively, which at $$384\text{M} \times 256$$ is about 98B parameters each.

**Why.** Capacity without FLOPs: a lookup costs a hash and a gather, not a GEMM, so the model holds far more parameters than it activates.

**Serving cost.** Per token per module, 3 orders times 8 heads is 24 gathers of 256 FP8 values, about 6 KB; call it 12 KB per token for both modules. The bandwidth is trivial; the problem is latency and placement. The tables are 188.8 GiB, larger than everything except the experts, and the addresses are random. DeepSeek's stack keeps them in host memory and prefetches over RDMA, overlapping the first module's lookup with the first Transformer block, which works because the addresses depend only on the token sequence and are known before the forward pass starts (section 2.4.2). vLLM's default is the same idea in simpler form: pinned host memory read over unified virtual addressing, with the table sharded by hash head across tensor-parallel ranks. What that means under DP8 is the subject of its own section below.

##### **DSpark**

**What.** DSpark is the speculative decoder, replacing the MTP head of V3 and V4. The drafter is three Transformer blocks with a 128-token sliding window, each an MoE of 128 routed experts with 3 active, reading the attention inputs of backbone layers 37 to 39 (config.json). One drafter pass produces base logits for **five draft positions at once**; a low-rank Markov head (rank 256) models dependencies among them; a confidence head predicts per-position acceptance, from which the engine computes prefix survival probabilities. A scheduler then combines survival probabilities with profiled throughput curves to pick, per request and per step, how many draft tokens to verify (section 2.4.3). The drafter was trained after backbone pre-training with the backbone frozen, and kept in sync through post-training without gradients flowing back.

**Why.** Semi-autoregressive drafting means one draft pass per five positions rather than five; confidence-scheduled verification means the engine can trim drafts at high load, when rejected tokens are compute wasted, and extend them at low load, when compute is idle anyway. vLLM's [adaptive verification doc](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/) describes exactly this trade and notes that a fixed number of speculative tokens is never right across concurrencies.

**Serving cost.** The draft weights ship in the checkpoint, so there is no second model. The recipe's toggle is `--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'`. Adaptive verification requires full CUDA graphs, since step costs are profiled from captured graphs. Expect the gain to be workload-dependent and to vanish at large batch: SGLang's cookbook turns DSpark off in its high-throughput cell because "at large batch the fixed step cost stops paying for itself" ([SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)). Acceptance and speedup numbers for V4.1 specifically are not published; the [DSpark paper](https://arxiv.org/abs/2607.05147) reports a 51% throughput improvement for V4-Flash at an 80 tokens per second per user target, on DeepSeek's own stack.

##### **DeepSeek-ViT**

A 32-layer ViT at hidden size 1024, patch 14, trained from scratch with 2D-RoPE, RMSNorm, and SwiGLU. A 3x3 pixel-unshuffle cuts the token count nine-fold before an MLP projects into the backbone; images are capped at 1024 tokens, which at patch 14 puts the upper resolution near $$1344 \times 1344$$ (section 2.1.1; config.json `vision_config`). At serving time it is a small dense model that runs once per image; the recipe's note is that at this size, tensor-parallel communication costs more than it saves, hence the encoder-parallel flag below.

##### **What vLLM has today**

| Component | In vLLM main, 2026-09-11 | Pointer |
|---|---|---|
| CED, CSA2 modes, source-layer reuse | yes | `vllm/models/deepseek_v4_1/attention.py`, `sparse_mla.py` |
| Hierarchical indexer, candidate pool | yes | same |
| FP4 main KV | no, fp8 layout only | open [PR #56344](https://github.com/vllm-project/vllm/pull/56344) |
| SWA bounded replay (encoder or decoder) | no | open [PR #56227](https://github.com/vllm-project/vllm/pull/56227) |
| MXFP4 indexer cache | yes, opt-in, Blackwell datacenter only | `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` |
| MoE with MegaMoE backends | yes, SM100 with EP | `--kernel-config '{"moe_backend": ...}'` |
| Single-pass mHC | yes, TileLang kernels | Mega-mHC not integrated, [tracker #56217](https://github.com/vllm-project/vllm/issues/56217) |
| Engram | yes, host-offloaded by default | async prefetch and DP sharding in open [PR #56512](https://github.com/vllm-project/vllm/pull/56512) |
| DSpark with adaptive verification | yes, opt-in | `--speculative-config`, method `dspark` |
| ViT with encoder data parallel | yes | `--mm-encoder-tp-mode data` |
| DeepGEMM sparse indexer logits, DeepSelect top-k | no | open [PR #56254](https://github.com/vllm-project/vllm/pull/56254), [PR #56464](https://github.com/vllm-project/vllm/pull/56464) |
| torch.compile | no | model is not decorated; breakable CUDA graphs instead |

**References**
- [DeepSeek-V4.1-Flash tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf), sections 2.1 to 2.4 and 3.2; [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)
- [DSpark paper](https://arxiv.org/abs/2607.05147); [vLLM adaptive verification doc](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/)
- [SGLang DeepSeek-V4.1 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)
- vLLM source, as of 2026-09-11: `vllm/models/deepseek_v4_1/`, `vllm/models/deepseek_v4/nvidia/model.py`, `vllm/model_executor/warmup/deepseek_v4_mhc_warmup.py`

---

#### **The Launch Command, Line by Line**

##### **The docker half**

| Piece | What it does | Why the recipe emits it |
|---|---|---|
| `--gpus all` | exposes all eight GPUs | required |
| `--privileged` | full device access inside the container | hardcoded by the recipes command builder for every non-NPU platform (`src/lib/command-synthesis.js`, recipes repo, as of 2026-09-11). vLLM's own docs only call for it when InfiniBand or RDMA is in play ([parallelism scaling doc](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)). On one NVLink node it is not functionally needed; the PD variant with NIXL is where it matters. |
| `--ipc=host` | share the host's IPC namespace, so `/dev/shm` is not the container's tiny default | vLLM's worker and engine processes exchange tensors over shared memory; the [docker deployment doc](https://docs.vllm.ai/en/latest/deployment/docker/) uses it in its canonical example |
| `vllm/vllm-openai:deepseekv41-flash-0909` | the only build that carries this architecture | no pip wheel does; see the day-0 section for what is known about this image |

##### **The environment variables**

| Variable | Default | What the recipe sets and why |
|---|---|---|
| `VLLM_ENGINE_READY_TIMEOUT_S` | 600 (`vllm/envs.py`) | 3600. Loading 510 GB plus kernel JIT and warmup can exceed ten minutes, and the front end would otherwise give up on the engine cores. |
| `VLLM_USE_RUST_FRONTEND` | 0 | 1. Replaces the Python API server processes with the Rust `vllm-rs` binary, which serves HTTP, renders the chat template, and parses reasoning and tool calls, talking to the same Python engine cores over ZMQ ([rust/README.md](https://github.com/vllm-project/vllm/blob/main/rust/README.md)). The README calls it experimental and not feature-complete. It is multithreaded, so `--api-server-count` is ignored with it. The V4.1 chat renderer and DSML parser exist in both frontends; the recipe defaults to Rust since [recipes PR #957](https://github.com/vllm-project/recipes/pull/957) and tells you to switch to Python on any compatibility issue. |

Two Rust-side gaps specific to this model as of 2026-09-11: DSML tool calls are buffered until the whole invoke block is complete rather than streamed incrementally ([PR #56334](https://github.com/vllm-project/vllm/pull/56334)), and multimodal placeholder ordering has a pending fix ([PR #56366](https://github.com/vllm-project/vllm/pull/56366)). Neither affects a text benchmark; both affect a client that streams tool calls.

##### **The vLLM half**

| Flag | What it does |
|---|---|
| `--tokenizer-mode deepseek_v41` | Selects `DeepseekV41Tokenizer`, which wraps the standard Rust BPE from `tokenizer.json` and replaces Jinja chat templating with a Python port of DeepSeek's reference encoder (`vllm/tokenizers/deepseek_v41_encoding.py`). Needed because the checkpoint ships no Jinja template; the model card points to `encoding/encoding.py` instead. vLLM would pick this mode automatically for the architecture (`vllm/config/model.py`), so the flag is explicit rather than required. |
| `--enable-expert-parallel` | Shards the 384 experts of every layer across the EP group, with $$\text{EP} = \text{TP} \times \text{DP} = 8$$, so each GPU holds 48 routed experts per layer. Routing becomes all-to-all; the default backend is `allgather_reducescatter` ([EP deployment doc](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)). The MegaMoE kernel backends refuse to run without it. |
| `--data-parallel-size 8` | Eight engine-core processes, one per GPU, each with a full copy of the attention and dense weights over its own batch. This is "internal load balancing": one endpoint, and the front end routes each request to the least-loaded engine by queue length ([DP deployment doc](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)). The expert layers do collectives, so all eight forward passes stay in lockstep, and a DP coordinator makes idle ranks run dummy passes so their peers' all-to-alls do not hang. `--max-num-seqs` applies per rank. |
| `--tool-call-parser deepseek_v41` plus `--enable-auto-tool-choice` | Registers the V4.1 DSML parser, which recognizes the spaced tags this model emits (`<｜DSML｜ calls>`, `<｜DSML｜ invoke name=...>`), and lets the model decide when to call a tool ([tool calling doc](https://docs.vllm.ai/en/latest/features/tool_calling/)). The doc page has no V4.1 section yet; the registration is in `vllm/tool_parsers/__init__.py`. |
| `--reasoning-parser deepseek_v41` | Same parser engine, splitting `<think>` blocks into `reasoning_content`. |
| `--mm-encoder-tp-mode data` | Runs the ViT with full weights on every TP rank and splits the image batch across ranks, instead of tensor-parallelizing a 32-layer, 1024-wide encoder ([multimodal config](https://github.com/vllm-project/vllm/blob/main/vllm/config/multimodal.py)). **In this command TP is 1, so the flag does nothing**: each DP rank already runs the whole ViT on its own requests (`vl_model.py` only takes the sharded path when the TP world size exceeds 1). It matters for the TP8 and TEP8 renderings of the same recipe. |

##### **What resolves silently**

Flags you did not pass still get values, and several of them are specific to this model or this GPU class (all from vLLM source as of 2026-09-11):

| Setting | Resolved value | Where |
|---|---|---|
| `--max-num-seqs` | 1024 per DP rank | `vllm/engine/arg_utils.py`: GPUs with at least 160 GiB get the largest defaults |
| `--max-num-batched-tokens` | 16384 | same |
| `--max-model-len` | 1,048,576 | from `max_position_embeddings` |
| `--gpu-memory-utilization` | 0.92 | `vllm/config/cache.py` |
| `--kv-cache-dtype` | `fp8_ds_mla` | `auto` resolves to it; bf16 is rejected for this layout |
| CUDA graphs | breakable, full and piecewise | `DeepseekV41ForCausalLM` is in `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` (`vllm/config/vllm.py`), which sets `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and disables torch.compile |
| Engram placement | pinned host memory, one full copy per DP rank | `vllm/config/engram.py` and `common/engram.py`; next section |
| DSpark | off | opt-in via `--speculative-config` |
| DeepGEMM warmup | `relax` | `VLLM_DEEP_GEMM_WARMUP` |
| Attention backend | FlashMLA sparse (`FLASHMLA_SPARSE_DSV41`) on SM90 and SM100 | `nvidia/model.py` |

Two to internalize: this model never goes through torch.compile, and instead gets CUDA graphs with runtime stream-capture breaks around the attention ops (`vllm/compilation/breakable_cudagraph.py`); and 1024 sequences per rank is a very large admission ceiling for a model whose per-step cost stops improving once every expert is touched.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/deepseek-v41-flash-vllm/dp8-topology.svg" title="What the DP8 plus EP command starts on one 8x B300 node" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    One node, one command. The Rust front end owns the port and routes each request to the least-loaded engine core; the DP coordinator keeps the eight cores' forward passes in lockstep. Each core drives one GPU holding a full copy of the attention and dense weights (about 11 GiB) and a 48-expert slice of every layer (about 32 GiB), with the expert all-to-all crossing NVLink every layer. Under the default configuration, each core also holds its own full 189 GiB Engram table in pinned host RAM and reads it over UVA. Editable source: <a href="/assets/img/deepseek-v41-flash-vllm/dp8-topology.excalidraw">dp8-topology.excalidraw</a>.
</div>

##### **The Engram placement problem**

This is the part of the config that surprised me, and it is the first thing I would check on the node.

The resolution chain in vLLM, as of 2026-09-11: `VllmConfig._resolve_and_verify_engram_config` leaves `engram_config` as `None` unless you pass `--engram-config` or set the legacy `VLLM_PLE_CPU_OFFLOAD` (`vllm/config/vllm.py`). The Engram module then builds its table with `cpu_offload=engram_config.cpu_offload if engram_config else True` (`vllm/models/deepseek_v4_1/common/engram.py`). So **with no flag, the tables go to pinned host memory**, allocated with `device="cpu", pin_memory=True`, and are read from the GPU over unified virtual addressing. The table is sharded by hash column across tensor-parallel ranks (`ParallelEngramEmbedding`), and with `embedding_across_dp` at its default of `False`, "each DP rank has a separate TP-sharded embedding replica" (`vllm/config/engram.py`).

Put the two together for this command. TP is 1, so the shard is the whole table. DP is 8, so there are eight replicas. That is eight copies of 188.8 GiB, about 1.5 TiB of pinned host memory, before anything else on the host. Whether that boots depends on the node's RAM, which NVIDIA's DGX B300 page does not state in the spec table I could find, so check it before launching:

```bash
free -g
```

The alternatives, and what each costs per GPU (weights from the recipe's memory table; Engram at 188.8 GiB):

| Layout | Attention and dense per GPU | Experts per GPU | Engram, default (host) | Engram if resident on GPU |
|---|---|---|---|---|
| DP8, EP8 (this command) | 11.4 GiB, replicated | 32.4 GiB | 8 host copies of 188.8 GiB | 188.8 GiB per GPU, leaving about 14 GiB for KV at 0.92 utilization |
| DP8, EP8 with `--engram-config '{"embedding_across_dp": true}'` | 11.4 GiB | 32.4 GiB | one host copy, sharded 8 ways | 23.6 GiB per GPU |
| TP8, EP8 (the recipe's TEP strategy) | about 1.4 GiB, sharded | 32.4 GiB | one host copy, sharded 8 ways | 23.6 GiB per GPU |
| 2x TP4, EP4 (SGLang's verified 8x B300 layout, run as two replicas) | 2.9 GiB | 64.9 GiB | 2 host copies, each sharded 4 ways | 47.2 GiB per GPU |

The knob to keep the tables in HBM is `--engram-config '{"cpu_offload": false}'`. Note the trap in the resolution chain: because `EngramConfig.cpu_offload` defaults to `False` while an absent config means `True`, passing any `--engram-config` at all without that key also moves the tables onto the GPU. Under DP8 that is 188.8 GiB of each B300 gone to a lookup table, which is why the DP-sharding flag exists.

What this means for the tuning plan: the recipe's DP8 command is the throughput layout for MoE models in general, but for this model it multiplies the one component that does not shard for free. The first experiment is not a kernel; it is the layout matrix above, with Engram on host versus GPU, measured at fixed concurrency. Open [PR #56512](https://github.com/vllm-project/vllm/pull/56512) adds asynchronous prefetch for offloaded lookups, shares a single host table across DP replicas through `/dev/shm`, and adds Engram DP sharding; when it lands, the DP8 row changes.

**References**
- [vLLM data parallel deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/), [expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), [docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/), [parallelism scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [vLLM Rust frontend README](https://github.com/vllm-project/vllm/blob/main/rust/README.md); [recipes PR #957](https://github.com/vllm-project/recipes/pull/957)
- vLLM source, as of 2026-09-11: `vllm/envs.py`, `vllm/engine/arg_utils.py`, `vllm/config/{vllm,cache,engram,multimodal,model}.py`, `vllm/tokenizers/deepseek_v41_encoding.py`, `vllm/tool_parsers/__init__.py`, `vllm/models/deepseek_v4_1/common/engram.py`, `vllm/models/deepseek_v4_1/nvidia/{model,vl_model}.py`, `vllm/compilation/breakable_cudagraph.py`
- [PR #56512](https://github.com/vllm-project/vllm/pull/56512), [PR #56334](https://github.com/vllm-project/vllm/pull/56334), [PR #56366](https://github.com/vllm-project/vllm/pull/56366)

---

#### **vLLM Day-0 Status**

##### **What landed, what is open**

Support arrived in three PRs over two days: frontends in [#56208](https://github.com/vllm-project/vllm/pull/56208) (2026-09-10), model definitions in [#56228](https://github.com/vllm-project/vllm/pull/56228) (2026-09-10), and the keystone with registry, config, kernels, and tests in [#56214](https://github.com/vllm-project/vllm/pull/56214) (merged 2026-09-11). Everything else is tracked in [issue #56400](https://github.com/vllm-project/vllm/issues/56400) and the [kernel tracker #56217](https://github.com/vllm-project/vllm/issues/56217). Neither the [supported models page](https://docs.vllm.ai/en/latest/models/supported_models/) nor the vLLM blog mentions V4.1 yet.

| Open item | PR or issue | Why it matters here |
|---|---|---|
| Attention megakernel plus FP4 main KV | [#56344](https://github.com/vllm-project/vllm/pull/56344) | fuses norm, RoPE, sparse attention, inverse RoPE, and FP8 cast into one FlashMLA kernel and adds the `nvfp4_ds_mla` cache; its own GB200 numbers show a fused decode kernel at 22.7 µs versus 43.1 µs for the split chain |
| SWA bounded replay | [#56227](https://github.com/vllm-project/vllm/pull/56227) | the prefill halving from CED |
| DeepGEMM sparse MQA logits | [#56254](https://github.com/vllm-project/vllm/pull/56254) | the layer-20 full-range indexer scan; the PR's GB300 microbenchmark shows 6.56x to 12.6x over dense for decode at 512K context and roughly break-even at 32K |
| DeepSelect top-k | [#56464](https://github.com/vllm-project/vllm/pull/56464) | 119 µs versus 613 µs at batch 256 and 1M KV on GB200, per the PR |
| Engram prefetch, shared host table, DP sharding | [#56512](https://github.com/vllm-project/vllm/pull/56512) | the DP8 host-memory multiplier |
| EPLB with DSpark | [#56387](https://github.com/vllm-project/vllm/pull/56387) | EPLB currently fails when the DSpark drafter is on |
| Mega-mHC, Mega-Gate | [#56217](https://github.com/vllm-project/vllm/issues/56217) | the fused kernels the report describes |

##### **Bugs worth knowing on an NVIDIA node**

- [#56389](https://github.com/vllm-project/vllm/issues/56389): the `dsv4_topk` Triton router hits an illegal memory access above 256 sequences under high concurrency on H20 with TP8, EP8, and DSpark. With the 1024-per-rank default above, this is the first thing to bound if you see a crash under load.
- [#56443](https://github.com/vllm-project/vllm/issues/56443): DSpark warmup asserts on H200 with the Marlin MXFP4 MoE backend, which is what SM90 falls back to. Not a B300 issue, but it says the SM90 path is less traveled.
- [#56461](https://github.com/vllm-project/vllm/issues/56461): cannot serve on SM120 at all, from a block-size mismatch. Irrelevant to B300 but indicative of how fresh the geometry code is.

##### **The image**

The tag in the command, `deepseekv41-flash-0909`, was pushed to Docker Hub on 2026-09-10; the bare `deepseekv41-flash` tag the recipe prose mentions does not exist. The image labels its build commit as unknown, users report an in-image version string pointing at a commit that no longer resolves on GitHub (a staging branch deleted once V4.1 merged, per [#56512](https://github.com/vllm-project/vllm/pull/56512)), and it was created about a day before the keystone PR merged. Practically: the container is not `main`, and a fix you read on GitHub may not be in it. The recipe's `min_vllm_version` of 0.30.0 is unreleased; v0.29.0 (2026-09-09) does not mention V4.1.

**References**
- [Tracking issue #56400](https://github.com/vllm-project/vllm/issues/56400), [kernel tracker #56217](https://github.com/vllm-project/vllm/issues/56217), PRs and issues as linked
- [Docker Hub tags for vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags?name=deepseekv41), [vLLM v0.29.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)

---

#### **Where to Look for Performance**

No one has published a throughput number for this model on B300 with vLLM. The only public vLLM measurement of any kind is an AMD [RFC #56506](https://github.com/vllm-project/vllm/issues/56506) on 8x MI355X at TP4 with DSpark: 35.9 output tokens per second at concurrency 1 with a 25.6 ms inter-token latency, rising to 469 tokens per second at concurrency 32 with ITL only 1.9x worse. The author's reading is the right one: a device whose ITL barely moves while throughput climbs 13x is not saturated; it is paying a large fixed cost per step. Their trace counts about 14,020 kernel launches per decode step, 85% of each layer's 350 launches in the mHC block, on a 4x4 matrix. That is ROCm with an eager-mode trace, so the wall-clock figures do not transfer to B300; the launch count and its composition do, because a CUDA graph replays the sequence it captured rather than merging kernels.

Everything below is a hypothesis, ordered by how much I expect it to matter, each tied to a place in the code to confirm or refute it.

**H1: decode is launch-bound, not byte-bound, and mHC is the launch count.** The byte floor computed earlier is about 1 ms per step at small batch. If a single-stream decode step on B300 lands anywhere near 20 ms, the gap is fixed cost. First check: count kernels per step with `nsys` on one engine core, and see what fraction sits in the TileLang mHC pre and post kernels and the 20-iteration Sinkhorn loop. The fix path is the Mega-mHC integration on the kernel tracker; the measurement tells you how much it is worth before it lands.

**H2: the Engram lookup is a random-access host read in the critical path.** By default every token gathers 24 rows of 256 FP8 bytes from a pinned host table over UVA, twice per token, at layers 1 and 14. The bandwidth is nothing; the latency is a PCIe round trip per gather with no prefetch on main. Check: run the layout matrix with `--engram-config '{"cpu_offload": false}'` on a layout where the table fits, and compare ITL at fixed concurrency. If the gap is large, the prefetch PR is the fix; if it is small, leave the tables on the host and spend the HBM on batch.

**H3: DP8 is the wrong layout for this model until Engram shards across DP.** Beyond the host-memory multiplier, DP8 replicates the 7.6 GB dense-weight read on every rank each step, while TP8 divides it, and that read is most of the byte floor. Check: TEP8 versus DP8 versus two TP4 replicas at the same concurrency per node. SGLang's verified 8x B300 layout is two TP4 replicas, a data point rather than a verdict.

**H4: the MoE backend that `auto` picks matters on SM100.** The MegaMoE backends (`deep_gemm_mega_moe`, `flashinfer_moe_ep_mega_deep_gemm`, `flashinfer_moe_ep_mega_cutedsl`) exist for exactly this checkpoint shape and require EP plus SM100. Which one `auto` resolves to is not documented. Check the startup log for the selected backend, then sweep `--kernel-config '{"moe_backend": ...}'` across the three at a saturating batch, where the expert GEMMs dominate.

**H5: DSpark helps at low concurrency and hurts at high, and adaptive verification is what makes one config survive both.** The engine's own doc says so, and SGLang turns DSpark off in its throughput cell. Check: ITL at concurrency 1, 8, 32, 128 with DSpark off, on with fixed verification, and on with adaptive verification. Watch the per-position acceptance counters, since acceptance is workload-dependent and V4.1's numbers are unpublished.

**H6: the layer-20 full-range indexer scan is the context-scaling cost.** Everything else in decode is bounded by the 512-entry selection, the candidate pool, or the window. Check: ITL at 8K, 128K, and 512K context at fixed batch. The DeepGEMM logits and DeepSelect PRs target exactly this, and the DeepGEMM PR's own numbers put the crossover near 32K context, with the sparse path pure overhead below it.

**H7: the admission defaults are too generous for the step cost curve.** With 1024 sequences per rank and 16,384 batched tokens, the scheduler will happily build steps far past the point where every expert is touched, trading ITL for nothing. Check: sweep `--max-num-seqs` from 32 to 1024 per rank and plot goodput against an ITL bound, as in the metrics section of the inference post. The recipe's PD variant caps at 32 for a reason, and the known router crash above 256 is a second reason.

**What to leave alone for now.** KV capacity: it is not the constraint on this node, so the FP4 cache PR is a memory win that mostly matters at 1M context with hundreds of sequences per GPU. Chunked prefill and prefix caching: both are on by default and neither is model-specific. torch.compile: not available for this model.

The order that follows: `nsys` on one engine core at concurrency 1 (H1); the layout matrix (H2, H3); the MoE backend sweep at saturating batch (H4); the DSpark and admission sweeps against an ITL bound (H5, H7); the context sweep last (H6). vLLM's Prometheus histograms give ITL and TTFT per phase without instrumentation.

**References**
- [RFC #56506, DeepSeek-V4.1-Flash performance on ROCm](https://github.com/vllm-project/vllm/issues/56506)
- [vLLM adaptive verification doc](https://docs.vllm.ai/en/latest/features/speculative_decoding/adaptive_verification/), [EP deployment doc](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [PR #56254](https://github.com/vllm-project/vllm/pull/56254), [PR #56464](https://github.com/vllm-project/vllm/pull/56464), [PR #56512](https://github.com/vllm-project/vllm/pull/56512), [tracker #56217](https://github.com/vllm-project/vllm/issues/56217)
- [LLM inference systems post, metrics section](/blog/2026/llm-inference-systems/)

---

#### **Other Vendors, Briefly**

vLLM is the baseline here, but it is worth knowing what everyone else shipped in the same 48 hours, as of 2026-09-11.

| Stack | Status | Notes |
|---|---|---|
| [SGLang](https://www.lmsys.org/blog/2026-09-10-deepseek-v41) | day-0 via preview image `lmsysorg/sglang:dev-dsv41`; upstream PR still open | verified cells for GB300, H200, B200, B300, MI350X; 8x B300 runs as two TP4, EP4 replicas; DSpark in the low-latency cell only; opt-in decoder bounded replay (reported 1.56x prefill on 8x H200); Engram host offload via an env var, reported plus 36% KV capacity on 4x GB300 ([cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)) |
| NVIDIA Dynamo | frontend and parsers merged ([#14690](https://github.com/ai-dynamo/dynamo/pull/14690)); SGLang-backed recipes on 8x GB200 in an open PR ([#14682](https://github.com/ai-dynamo/dynamo/pull/14682)) | the PR states neither target is benchmarked |
| NVIDIA TensorRT-LLM | nothing for V4.1 found in the repo | V4 support exists |
| AMD | through the frameworks | vLLM ROCm recipe verified on MI350X with `VLLM_USE_BREAKABLE_CUDAGRAPH=1` and the AITER MXFP4 MoE backend; the RFC above is the only public performance data |
| llama.cpp, Ollama | conversion PR open upstream ([#28696](https://github.com/ggml-org/llama.cpp/pull/28696)); Ollama offers a cloud tag only | community forks report a few tokens per second with expert streaming |
| DeepSeek's own [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe) | Rust and Python prompt encoder and output parser | no inference engine; vLLM's Rust tool-call rendering was aligned to it in [#56260](https://github.com/vllm-project/vllm/pull/56260) |

One cross-vendor discrepancy to carry in your head: DeepSeek's reference encoder and the tech report map the named reasoning efforts to low 50, high 75, max 100; vLLM's port maps low 25, high 50, xhigh 75, max 100, defaulting to 50 with thinking on (`vllm/tokenizers/deepseek_v41_encoding.py`, as of 2026-09-11). A benchmark that sends `reasoning_effort: "high"` is asking for different things on the two stacks. Send an integer.

**References**
- [SGLang and Miles day-0 post](https://www.lmsys.org/blog/2026-09-10-deepseek-v41), [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1)
- Dynamo [#14690](https://github.com/ai-dynamo/dynamo/pull/14690), [#14682](https://github.com/ai-dynamo/dynamo/pull/14682); llama.cpp [#28696](https://github.com/ggml-org/llama.cpp/pull/28696); [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)
- [DeepSeek reference encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/encoding.py)

---

#### **Takeaways**

- **The model is built to cache less.** Four layers write main KV, at 2.5 entries per token, in FP4; the decoder's KV is projected from the encoder so prefill runs half the network; the sliding-window caches are replayed rather than persisted. The paper's 890 bytes per token is the HBM cost on DeepSeek's stack.
- **vLLM today stores twice that.** The fp8 layout is 584 bytes per entry, FP4 is an open PR, and bounded replay is an open PR. Size off vLLM's numbers, and expect both to move.
- **The byte floor on 8x B300 is about 1 ms per decode step; the only public measurement is 25x that, on AMD.** The gap is fixed per-step cost, and the trace points at mHC's kernel count. Measure launches before bytes.
- **DP8 multiplies Engram.** Default placement is one full 189 GiB copy per DP rank in pinned host memory, eight copies for this command. TP8 or DP sharding turns that into one. This is the first experiment.
- **Several flags are inert or redundant in this exact command.** The encoder-parallel flag needs TP above 1; the tokenizer mode is auto-selected; the privileged flag is there for RDMA.
- **The admission defaults are large.** 1024 sequences per rank and 16,384 tokens per step, on a model whose step cost stops improving once every expert is touched, with a known router crash above 256 sequences.
- **The image is not main.** Built from a deleted staging branch, a day before the keystone merge, with its build commit unlabeled. Read PRs with that in mind.

---

#### **Test Yourself**

Try each from memory before reading the answer.

**1. Why does the model activate 8B parameters per token in prefill but 16B in decode?**
The causal encoder-decoder projects the decoder's global KV from the encoder's final hidden state, so a prompt token only needs to run the 20 encoder layers to populate the cache. Decode runs all 40. Half the layers means half the active experts.

**2. What do the three CSA2 modes share and what do they compute?**
Full computes main KV, indexer K, and top-K itself. Reindex reuses main KV and indexer K from the last Full layer but computes its own indexer Q and a fresh top-K. Reuse reuses main KV and the latest top-K and runs no indexer. All three compute their own query and sliding-window KV.

**3. How many main KV entries does one token add across the network, and where does the number come from?**
2.5: the three encoder source layers (2, 8, 14) at compression ratio 2 contribute half an entry each, and the decoder source layer (20) at ratio 1 contributes one. From `kv_source_layer_ids` and `compress_ratios` in config.json.

**4. What does the hierarchical sparse indexer make constant in context length, and what does it leave linear?**
The per-query cost of the four decoder Reindex layers, which score only the 16,384-position candidate pool published by layer 20. Layer 20's own full-range scan stays linear in context.

**5. What is single-pass mHC, in one sentence, and why does it matter for kernels?**
Block $$l$$ mixes its input with the previous block's coefficient $$A_{l-1}$$ instead of its own $$A_l$$, which removes the reduction dependency and lets residual update, input mixing, and coefficient prediction fuse into one pass over the residual, halving activation traffic.

**6. What does an Engram lookup cost per token in bytes, and why is it still a performance concern?**
About 12 KB: 24 gathers of 256 FP8 values per module, two modules. The concern is placement and latency, not bandwidth: the tables are 189 GiB, the addresses are random, and vLLM's default reads them from pinned host memory with no prefetch on main.

**7. Under the recipe's DP8 command, how many copies of the Engram table exist and where?**
Eight, one per DP rank, each the full 188.8 GiB, in pinned host memory, because TP is 1 (no sharding) and `embedding_across_dp` defaults to false.

**8. Which flag in the command does nothing, and why?**
`--mm-encoder-tp-mode data`, because the ViT's data-parallel path is only taken when the tensor-parallel world size exceeds 1, and this command has TP 1.

**9. Why is a 1 ms byte floor consistent with a measured 25 ms ITL, and what would you measure first?**
Because the floor counts bytes and the measurement includes fixed per-step cost. The AMD trace counted about 14,000 kernel launches per step, 85% of a layer's launches in the mHC block. Measure the launch count per step on one engine core before touching anything else.

**10. Why does SGLang turn DSpark off in its high-throughput recipe?**
A DSpark step has a fixed cost that does not shrink with accept length; at large batch the GPU is no longer memory-bound, so rejected draft tokens are compute wasted and the fixed step cost stops paying for itself. vLLM's adaptive verification exists to make one setting survive across that crossover.

---

#### **Wrapping up**

The thing to keep from this post is a shape, not a number. DeepSeek-V4.1-Flash is an answer to the question "what is the least state a long-context model can carry per token," and every component in it is one term of that answer: share the cache across layers, compress it along the sequence, store it in four bits, project the top half from the bottom half, and replay the window instead of saving it. The recipe command is a generic MoE throughput layout dropped onto that model, and the two do not quite agree. It replicates the one large component that does not shard on its own, it leaves the model's headline cache format on the table because the kernel has not landed, and it admits far more work per step than the step cost curve rewards.

None of that is a criticism of a day-old recipe. It is the map of where the time probably goes, and each item comes with a way to check it. The floor is a millisecond; the first job is to find out how far above it the node actually runs, and why.

If you find a mistake anywhere in here, please let me know and I'll fix it.
