---
layout: post
title: Expert Parallel MoE in Decode - From a Padded Grid to a Persistent Grid
date: 2026-10-15 12:00:00-0400
featured: false
description: How an expert-parallel MoE layer runs in decode on one GPU with DeepSeek-V4-Flash shapes. It covers DeepEP's padded buffers and masked_m, the T x topk bound, the flat row list and its prefix sum, and why a persistent grid launches about 1,184 CTAs. A padded grid launches 24,576.
tags: GPU ML
categories: GPU-NYU
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This post explains one MoE layer in decode, when expert parallelism spreads the experts over several GPUs. It follows one step of the layer: the SiLU, multiply and FP8 quantize step between the two expert GEMMs. The main question is: how many CTAs should this step launch? A CTA (cooperative thread array) is one thread block of a CUDA kernel. The post answers four questions:

- What does one MoE layer do on one GPU in decode, and what does the all-to-all exchange give to the kernels?
- Why does the receive buffer reserve a fixed number of rows for each expert, and what is `masked_m`?
- How can a CTA find its row, if only the GPU knows the row counts?
- Why does a grid with one CTA for each possible row waste work, and what is the alternative?

The [MoE in vLLM post](/blog/2026/moe-vllm/) explains (token, expert) pairs and grouped GEMMs on one GPU. This post uses the same ideas, but with expert parallelism over several GPUs.

The running example is [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) on 4 B200 GPUs. The model shapes come from its `config.json`.

The post compares two ways to size the grid of the activation step. From now on, these are their names:

- The **padded grid** launches one CTA for each row that could exist. On the running example, this is 24,576 CTAs.
- The **persistent grid** launches one CTA for each CTA slot of the GPU, and each CTA loops over the real rows. A **CTA slot** is room on one SM for one resident CTA. On the running example, this is about 1,184 CTAs, if each SM holds 8 CTAs.

The plan:

- One MoE layer on one GPU
- The padded layout and masked_m
- At most T x topk real rows
- One number for each real row: the flat list and its prefix sum
- The padded grid: one CTA for each possible row
- The persistent grid: one CTA for each slot, and a loop
- Inside one CTA: one row

Let's get started.

---

#### **One MoE layer on one GPU** {#layer}

##### **The shapes**

DeepSeek-V4-Flash has 256 routed experts in each MoE layer. Each token uses 6 of them. With expert parallelism over 4 GPUs, each GPU holds 64 of the 256 experts. These are its **local experts**. Each GPU also has its own batch of decode tokens.

| Symbol | Meaning | DeepSeek-V4-Flash, 4 x B200 |
|---|---|---|
| $$H$$ | hidden size, `hidden_size` | 4096 |
| $$D$$ | expert intermediate size, `moe_intermediate_size` | 2048 |
| $$E_{\text{total}}$$ | routed experts, `n_routed_experts` | 256 |
| topk | experts per token, `num_experts_per_tok` | 6 |
| GPUs | expert-parallel ranks | 4 |
| $$E_{\text{local}}$$ | experts on one GPU | 256 / 4 = 64 |
| $$b$$ | decode tokens on one GPU in one step | at most 1024 |
| $$T$$ | rows reserved for each local expert | 1024 x 4 = 4096 |

The cap of 1024 tokens on each GPU is a deployment setting. This post uses 1024 as the example. The [next section](#padded) explains $$T$$.

##### **The flow of one layer**

One MoE layer of one decode step does six steps on each GPU:

1. The router gives `topk_ids` and `topk_weights`, each with shape $$[b, \text{topk}]$$, from the input $$x$$, with shape $$[b, H]$$.
2. Dispatch sends each token to the GPUs that own its experts. Each GPU receives `recv_x`, $$[E_{\text{local}}, T, H]$$ in fp8, and `masked_m`, $$[E_{\text{local}}]$$ in int32.
3. GEMM 1 is a masked grouped GEMM. Expert $$e$$ multiplies its real rows by its gate-up weights $$W_{13,e}$$, with shape $$[H, 2D]$$. The output is `gateup`, $$[E_{\text{local}}, T, 2D]$$ in bf16. Each row holds the gate half and then the up half.
4. The activation step computes the SiLU and multiply and quantizes the result to FP8. The output is `q`, $$[E_{\text{local}}, T, D]$$ in fp8, plus the scales.
5. GEMM 2 is the masked down projection. The output is `down`, $$[E_{\text{local}}, T, H]$$ in bf16.
6. Combine sends each row back to the GPU of its token. That GPU adds the topk outputs of each token, each multiplied by its weight. The result is `out`, $$[b, H]$$.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/moe-persistent-grid/moe-layer.svg" title="One MoE layer of one decode step on one GPU" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    One MoE layer of one decode step on one GPU. The purple tensors use the padded layout of the next section. The red box is the activation step that this post is about. Editable source: <a href="/assets/img/moe-persistent-grid/moe-layer.excalidraw">moe-layer.excalidraw</a>.
</div>

The whole decode step runs as one CUDA graph. A CUDA graph fixes every shape at capture time, including $$T$$. In each step, only the values in `masked_m` change, and these values stay on the GPU.

##### **What the activation step computes**

The activation step reads one row of `gateup` for each real (token, expert) pair. Then it computes this value in fp32 for each of the $$D$$ outputs:

$$v = \text{SiLU}(\min(\text{gate}, 10)) \times \text{clamp}(\text{up}, -10, 10)$$

The limit 10 is `swiglu_limit` in the config. Then the step quantizes $$v$$ to FP8 e4m3, with one scale for each group of 128 outputs. GEMM 2 reads these FP8 values and scales. The scale layout depends on the GPU:

| GPU | Scale tensor | Scale format |
|---|---|---|
| Blackwell | int32 $$[E_{\text{local}}, D/512, T]$$ | 4 UE8M0 exponent bytes in each int32 |
| Hopper | fp32 $$[E_{\text{local}}, T, D/128]$$ | one fp32 value for each group |

---

#### **The padded layout and masked_m** {#padded}

##### **The receiver does not know the counts**

[DeepEP](https://github.com/deepseek-ai/DeepEP) is DeepSeek's expert-parallel communication library. It does the all-to-all exchange over NVLink and RDMA. In the layer flow, dispatch and combine are DeepEP.

Before dispatch, a GPU does not know how many rows each of its local experts will receive. The router choices on all the GPUs decide these counts. A receiver can use one of two options:

| Option | What it does | Cost |
|---|---|---|
| (a) Exact size | Send the counts to the host, then allocate exactly | One device-to-host sync in each layer. A CUDA graph cannot capture it. |
| (b) Worst case | Reserve the worst case for each expert, and keep the counts on the GPU | Most reserved rows are empty. Each consumer must skip the empty rows. |

DeepEP's low-latency mode, which decode uses, takes option (b). The receive buffer `recv_x` has $$T$$ rows for each local expert. For expert $$e$$, the rows $$[0, \text{masked\_m}[e])$$ are real. Dispatch does not write the other rows, and no kernel reads them. Every later buffer (`gateup`, `q`, `down`) uses the same real rows.

I did not read the DeepEP source. Two facts in this section come from DeepEP's API and its callers: the real rows are at the front of each expert, and the formula for $$T$$.

The figure uses a small example with 2 GPUs, 4 experts and topk = 2. The running example has 4 GPUs, 256 experts and topk = 6.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/moe-persistent-grid/padded-dispatch.svg" title="What GPU 0 receives from low-latency dispatch" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    The small example. GPU 0 owns experts e0 and e1 and receives 8 rows for each. Expert e0 gets 4 real rows and expert e1 gets 2, so masked_m = [4, 2]. Editable source: <a href="/assets/img/moe-persistent-grid/padded-dispatch.excalidraw">padded-dispatch.excalidraw</a>.
</div>

The cost of option (b) moves to the consumers. The masked grouped GEMMs read `masked_m` and skip the padded rows. The activation step must also find the real rows. The rest of this post is about how it does that.

##### **How T is computed**

$$T$$ is the largest number of rows that one expert can receive in one step:

$$T = (\text{max tokens that one GPU sends in one step}) \times (\text{GPUs})$$

The topk experts of one token are all different. So one expert receives each token at most once. In the small example, each GPU sends at most 4 tokens, so at most 8 tokens exist. In the worst case, all 8 tokens choose e0, so $$T = 4 \times 2 = 8$$. For DeepSeek-V4-Flash, $$T = 1024 \times 4 = 4096$$.

$$T$$ comes from the configured cap, not from the current batch. The serving engine allocates the buffer at startup, and the CUDA graph captures its shape.

At $$b = 64$$ tokens per GPU, almost all of the reserved rows are empty:

| | Small example, GPU 0 | DeepSeek-V4-Flash, $$b = 64$$ per GPU |
|---|---|---|
| Rows reserved, $$E_{\text{local}} \times T$$ | 2 x 8 = 16 | 64 x 4096 = 262,144 |
| Real rows, sum of `masked_m` | 6 | about 384 |
| Real share | 38% | 0.15% |

The value 384 assumes uniform routing:

$$64 \text{ tokens} \times 4 \text{ GPUs} \times 6 = 1536 \text{ pairs}, \quad 1536 / 4 \text{ GPUs} = 384 \text{ rows on each GPU}$$

##### **Prefill uses option (a)**

DeepEP's normal mode, which prefill uses, takes option (a) and returns the counts to the host. Prefill uses a contiguous layout, not the padded layout. Two reasons explain this choice:

- The sync cost is about the same in each layer, an estimated 20 to 50 us. The MoE compute of prefill takes several ms in each layer. So the sync adds an estimated 1% or less. Also, prefill mostly runs without full CUDA graphs.
- The padded layout is too large at prefill sizes. At 8192 tokens per GPU, $$T = 32768$$. Then `recv_x` in fp8, $$[64, 32768, 4096]$$, needs 8.6 GB, and `gateup` in bf16 needs 17.2 GB. The real rows of `recv_x`, about 49,000, need only about 0.2 GB.

I did not verify one detail. DeepEP's normal mode can possibly get the counts by CPU polling of pinned host memory, not by a full stream sync.

---

#### **At most T x topk real rows** {#bound}

Each row in the padded buffers is one (token, local expert) pair. One token adds at most topk rows on one GPU, because it has only topk experts. Over all the GPUs, at most $$T$$ tokens exist in one step. So the total over all the local experts of one GPU is at most $$T \times \text{topk}$$.

Three different bounds now exist. It is important to keep them apart:

| Bound | What it limits | Small example | DeepSeek-V4-Flash |
|---|---|---|---|
| $$T$$ | the rows of one expert | 8 | 4096 |
| $$T \times \text{topk}$$ | the rows of all local experts together | 16 | 24,576 |
| $$E_{\text{local}} \times T$$ | the reserved space | 16 | 262,144 |

In the small example, $$E_{\text{local}} = \text{topk} = 2$$, so the last two bounds are equal. In DeepSeek-V4-Flash, $$E_{\text{local}} = 64$$ is much larger than topk = 6.

In the small example, the tokens add these rows on GPU 0:

| Token | t0 | t1 | t2 | t3 | t4 | Total |
|---|---|---|---|---|---|---|
| Rows on GPU 0 | 1 | 2 | 0 | 1 | 2 | 6 = 4 + 2 |

The total 6 is the sum of `masked_m` = [4, 2].

---

#### **One number for each real row: the flat list and its prefix sum** {#flat}

##### **The flat list**

A CTA knows only its own number, `blockIdx.x`. The activation step needs a rule that gives each CTA a real row from this number.

Line up the real rows in one list with the gaps removed: the rows of e0 first, then the rows of e1. This post calls it the **flat list**, and the position in it the **flat row number** $$r$$. In the small example, the buffer slots and the flat list are:

| | Contents |
|---|---|
| Buffer slots of e0 (8) | t0, t1, t3, t4, then 4 unused slots |
| Buffer slots of e1 (8) | t1, t4, then 6 unused slots |
| Flat list (6) | t0, t1, t3, t4, t1, t4 at flat rows 0 to 5 |

A rule that uses the buffer slot number directly would put CTAs on empty slots. Also, the real slots change in each step. The flat list has no gaps, so any split of the flat rows over the CTAs gives each CTA only real rows. The kernel never builds the flat list in memory. A prefix sum gives each CTA the expert and the row of a flat row.

##### **The prefix sum s_row_end**

To read its row, a CTA needs the expert $$e$$ and the row $$j$$ inside that expert. With these two values it computes the address:

$$\text{address} = \text{base} + (e \times T + j) \times \text{row\_bytes}$$

The work comes as one flat row number $$r$$. The array `s_row_end` converts $$r$$ to $$(e, j)$$. It is the inclusive prefix sum of `masked_m`. For `masked_m` = [4, 2], `s_row_end` = [4, 6]. The rule has two parts:

1. Flat row $$r$$ belongs to the first expert $$e$$ with $$\text{s\_row\_end}[e] > r$$.
2. The row inside $$e$$ is $$j = r - \text{s\_row\_end}[e - 1]$$, and $$j = r$$ if $$e = 0$$.

Example with $$r = 5$$: `s_row_end[0]` = 4 is not larger than 5, and `s_row_end[1]` = 6 is larger than 5. So $$e = 1$$ and $$j = 5 - 4 = 1$$. This is token t4, in buffer slot $$1 \times 8 + 1 = 9$$.

An empty expert does not break the rule. For `masked_m` = [3, 0, 2, 1], `s_row_end` = [3, 3, 5, 6]. For $$r = 3$$, `s_row_end[1]` = 3 is not larger than 3, so the rule skips the empty expert e1. The row is in e2, at $$j = 3 - 3 = 0$$.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/moe-persistent-grid/flat-rows.svg" title="One number for each real row on GPU 0" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    The flat row number r counts the real rows of e0 and then of e1. The array s_row_end is the inclusive prefix sum of masked_m. Its last entry, 6, is the number of real rows. Editable source: <a href="/assets/img/moe-persistent-grid/flat-rows.excalidraw">flat-rows.excalidraw</a>.
</div>

The padded grid and the persistent grid use the same flat row numbers. They differ in two things: how many CTAs compute the prefix sum, and which flat rows each CTA takes.

##### **Why a prefix sum**

Two other ways to give out the work exist. Of the three ways, only the prefix sum needs few CTAs and keeps the work even:

| Way to give out the work | CTAs on DeepSeek-V4-Flash | Problem |
|---|---|---|
| One CTA for each (expert, slot) | 64 x 4096 = 262,144 | No scan, but most CTAs find an unused slot |
| One CTA for each expert | 64 | A hot expert lands on one CTA. 64 CTAs use only 64 of the 148 SMs. |
| Flat list plus prefix sum | any number | Any CTA count can split the rows evenly |

---

#### **The padded grid: one CTA for each possible row** {#padded-grid}

##### **The grid**

The host cannot read `masked_m` without a sync, so it does not know the number of real rows. The padded grid uses the worst case, $$T \times \text{topk}$$ CTAs, and gives CTA $$c$$ the flat row $$c$$. In this post, $$c$$ is the CTA number, `blockIdx.x`. Each CTA has $$D / 8$$ threads, so 256 threads for DeepSeek-V4-Flash.

##### **The scan in each CTA**

Each CTA first computes the prefix sum of `masked_m` with a parallel scan:

1. Thread $$e$$ loads `masked_m[e]`. Threads past $$E_{\text{local}}$$ use 0.
2. Each warp computes a prefix sum with shuffles. Warp 0 covers experts 0 to 31, and warp 1 covers experts 32 to 63.
3. Lane 31 of each warp writes the warp total to shared memory.
4. The CTA waits at `__syncthreads`.
5. Each thread adds the totals of the earlier warps.

Then the thread whose expert range holds flat row $$c$$ writes $$(e, j)$$ to shared memory. The CTA waits at a second `__syncthreads`. If no expert range holds $$c$$, then the CTA exits.

One scan costs one global load, about 10 shuffles and 2 `__syncthreads`. The load hits L2, because all the CTAs read the same 256 bytes.

##### **The cost: empty CTAs**

The main cost of the padded grid is not the scan. It is the number of CTAs. The GPU must schedule each of the 24,576 CTAs, and each CTA runs the scan and exits. Only about 384 of them find a row.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/moe-persistent-grid/padded-grid.svg" title="The padded grid: one CTA for each possible flat row" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    The padded grid in the small example: 16 CTAs for 6 real rows. The timeline uses made-up units (scan 1, row 3) on a toy GPU with 4 CTA slots. Empty CTAs still take a slot, a launch and a scan. The figures write the CTA number as b. Editable source: <a href="/assets/img/moe-persistent-grid/padded-grid.excalidraw">padded-grid.excalidraw</a>.
</div>

A B200 has 148 SMs. This post assumes 8 CTAs of 256 threads on each SM, so 148 x 8 = 1,184 CTA slots. I did not confirm this occupancy on hardware.

In the small example, most of the 10 empty CTAs run in the slots that the real rows do not use. Only CTAs 14 and 15 run after the real rows, so the timeline ends at 9 units. On DeepSeek-V4-Flash, the empty CTAs are about 20 times the slots:

| DeepSeek-V4-Flash, $$b = 64$$ per GPU | Value |
|---|---|
| CTAs launched, $$T \times \text{topk}$$ | 24,576 |
| CTA slots, 148 SMs x 8 CTAs | 1,184 |
| CTAs that find a row | about 384, all in the first 1,184 |
| Empty CTAs | about 24,192 |
| Empty CTAs for each slot | 24,192 / 1,184 = about 20 |

So every slot runs about 20 empty CTAs back to back. This post does not measure the time of one empty CTA.

---

#### **The persistent grid: one CTA for each slot, and a loop** {#persistent}

##### **The grid size**

The host still cannot read `masked_m`. So the persistent grid cannot launch exactly one CTA for each real row. It launches one CTA for each CTA slot, and caps this number at the padded grid size:

$$\text{grid} = \min(\text{SMs} \times \text{CTAs per SM}, \; T \times \text{topk})$$

On DeepSeek-V4-Flash with 4 B200 GPUs, this is $$\min(1184, 24576) = 1184$$. Each CTA scans `masked_m` once. If a CTA has no row, then it exits after the scan. If the real rows are more than the CTAs, then each CTA loops over several rows. The CTAs stay resident and loop over the work, so the grid size does not depend on the amount of work.

##### **The loop**

In pseudocode, each CTA does this:

```text
num_rows = min(scan(masked_m -> s_row_end), T x topk)
if (blockIdx.x >= num_rows) return
for (r = blockIdx.x; r < num_rows; r += gridDim.x) {
  e = find_expert(s_row_end, r)        // binary search
  j = r - (e == 0 ? 0 : s_row_end[e - 1])
  silu_mul_quant_row(e, j)             // same math in both grids
}
```

The parts work like this:

- **The scan** is the same as in the padded grid. It writes the inclusive prefix sum to `s_row_end` in shared memory and returns the number of real rows. Each CTA has its own copy of `s_row_end`, from its own scan.
- **The cap** $$T \times \text{topk}$$ never clips a real row, because the sum of `masked_m` is at most $$T \times \text{topk}$$.
- **The loop** gives CTA $$c$$ the flat rows $$c$$, $$c + \text{grid}$$, $$c + 2 \times \text{grid}$$, and so on, up to the number of real rows. This type of loop is a grid-stride loop.
- **`find_expert`** is a binary search over `s_row_end`. It takes $$\log_2(E_{\text{local}})$$ steps, so 6 steps for 64 experts. In each step, all the threads of the CTA read the same shared-memory word.
- **`silu_mul_quant_row`** is the per-row math. It is the same in both grids.

For the scan, each CTA uses 128 B for the warp totals plus 1 KB for `s_row_end`. At 8 CTAs on each SM, this is about 9 KB of the 228 KB on one SM. So it does not limit occupancy.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/moe-persistent-grid/persistent-grid.svg" title="The persistent grid: one CTA for each slot, and a loop in each CTA" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    The persistent grid in the small example. The grid is min(4, 16) = 4 CTAs, and each CTA takes every fourth flat row. On the same toy GPU and in the same units as the padded grid, the timeline ends at 7, not at 9. Editable source: <a href="/assets/img/moe-persistent-grid/persistent-grid.excalidraw">persistent-grid.excalidraw</a>.
</div>

##### **The two grids side by side**

| DeepSeek-V4-Flash, 4 x B200 | Padded grid | Persistent grid |
|---|---|---|
| CTAs launched | $$T \times \text{topk}$$ = 24,576 | min(1,184, 24,576) = 1,184 |
| Scans of `masked_m` | 24,576 | 1,184 |
| Rows for each CTA, $$b = 64$$ per GPU, about 384 rows | 0 or 1 | 0 or 1. About 800 CTAs exit after the scan. |
| Rows for each CTA, $$b = 1024$$ per GPU, about 6,144 rows | 0 or 1 | about 5 |
| Rows for each CTA, all 24,576 rows real | 1 | about 21 |

At $$b = 64$$ per GPU, the 384 real rows are fewer than the 1,184 CTAs. So each CTA takes at most one row, and the loop runs at most once. The difference is the 23,392 CTAs that the persistent grid does not launch. At $$b = 1024$$ per GPU, the loop runs about 5 times in each CTA.

##### **How vLLM does it**

vLLM uses a persistent grid for its masked SiLU-mul-quant kernel. [vllm-project/vllm#25074](https://github.com/vllm-project/vllm/pull/25074) ("Silu v2"), merged on 2025-10-10, added `persistent_masked_m_silu_mul_quant`. It launches a fixed grid of 132 x 32 blocks. Each block prefix-scans the per-expert counts into shared memory once. Then it iterates over the real tokens.

---

#### **Inside one CTA: one row** {#row}

The per-row math is the same in both grids. The grids differ only in which CTA runs which row.

One row of DeepSeek-V4-Flash has $$D = 2048$$ outputs, and the CTA has $$D / 8 = 256$$ threads. Each thread handles 8 outputs:

1. Thread $$i$$ loads the 8 gate values and the 8 up values at positions $$8i$$ to $$8i + 7$$, with two 16 B loads.
2. In bf16, the thread computes $$g = \min(\text{gate}, 10)$$ and $$u = \text{clamp}(\text{up}, -10, 10)$$.
3. In fp32, it computes $$v = g / (1 + e^{-g}) \times u$$.
4. It computes $$m$$, the largest $$\lvert v \rvert$$ of its 8 values.
5. One group of 128 outputs is 16 threads, half a warp. A shuffle-max over the 16 lanes gives the group absmax.
6. The thread computes the scale of its group, as the next table shows.
7. It computes $$q = \text{fp8\_e4m3}(\text{clip}(v / \text{scale}, -448, 448))$$ and writes its 8 values, 8 B.

| GPU | Scale |
|---|---|
| Blackwell, UE8M0 | $$2^{\lceil \log_2(\text{absmax} / 448) \rceil}$$ |
| Hopper, fp32 | $$\text{absmax} / 448$$ |

For UE8M0, the cast adds 1 to the biased exponent if the mantissa is not zero. This gives the ceiling in the formula.

<div class="row justify-content-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/moe-persistent-grid/one-row.svg" title="Inside one CTA: one row" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    One row of DeepSeek-V4-Flash in one CTA of 256 threads. DeepSeek-V4-Pro has D = 3072, so its CTA has 384 threads. Editable source: <a href="/assets/img/moe-persistent-grid/one-row.excalidraw">one-row.excalidraw</a>.
</div>

Each row of the activation step reads 8 KB and writes 2 KB plus 16 scales:

| DeepSeek-V4-Flash, $$b = 64$$, about 384 rows | Bytes |
|---|---|
| Read gate and up, 384 x 2 x 2048 x 2 B | about 3.1 MB |
| Write q and scales, 384 x (2 KB + 16 B) | about 0.8 MB |
| Time at 6 to 8 TB/s (estimate) | about 0.5 to 0.7 us |

So at $$b = 64$$ per GPU, the real data movement takes an estimated 0.5 to 0.7 us. The padded grid adds 24,192 empty CTAs to this work. The persistent grid adds about 800.

##### **References**

- DeepSeek-V4-Flash, [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)
- [DeepEP](https://github.com/deepseek-ai/DeepEP), DeepSeek's expert-parallel communication library
- vLLM, [vllm-project/vllm#25074](https://github.com/vllm-project/vllm/pull/25074), the persistent masked SiLU-mul-quant kernel

---

#### **Wrapping up**

In decode, DeepEP's low-latency mode reserves $$T$$ rows for each local expert and keeps the real row counts, `masked_m`, on the GPU. The host cannot read the counts without a sync, and a sync breaks the CUDA graph. The real rows of one GPU are at most $$T \times \text{topk}$$. A prefix sum of `masked_m` gives each real row one flat row number, and maps the number back to an expert and a row.

The padded grid launches $$T \times \text{topk}$$ CTAs, one for each flat row that could exist. On DeepSeek-V4-Flash with 4 B200 GPUs, this is 24,576 CTAs for about 384 real rows at $$b = 64$$. The persistent grid launches one CTA for each CTA slot, about 1,184 CTAs. Each CTA scans `masked_m` once and loops over its flat rows. vLLM uses this persistent design for its masked SiLU-mul-quant kernel.

If you find a mistake anywhere in here, please let me know and I'll fix it.
