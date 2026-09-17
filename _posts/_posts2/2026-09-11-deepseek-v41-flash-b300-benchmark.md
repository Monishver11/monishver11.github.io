---
layout: post
title: Benchmarking DeepSeek-V4.1-Flash on 8x B300 - Where the Step Time Goes
date: 2026-09-11 18:30:00-0400
featured: false
description: Measuring vLLM's throughput recipe for DeepSeek-V4.1-Flash on one 8x B300 node from first principles, covering what the harness metrics physically mean, why the published command does not boot, a concurrency sweep from 16 to 2048, a fixed-cost decomposition of the decode step, the elimination argument that locates the bottleneck off the GPU's busy timeline, and what each configuration change did and did not do
tags: GPU ML
categories:
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Hello all. This is the measurement half of a two-part look at serving [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) on vLLM. The [first part](/blog/2026/deepseek-v41-flash-vllm/) was a briefing: what the model's new components are, what every flag in the recipe command does, and a ranked list of hypotheses about where the time would go. This part puts the recipe on a real 8x B300 node, runs the recipe's own benchmark command, sweeps the one knob the recipe pins, and works out from the numbers what the decode step is actually made of.

The short version: the recipe's benchmark measures the machine at about 3% of what it can deliver, the decode step costs about 17 ms whether it holds 2 sequences or 16, and that fixed cost is not compute, not power, not expert imbalance, and not CPU-side kernel launching. Getting to that conclusion honestly takes some care, and the care is the point of this post. I want to be able to read a `vllm bench serve` table and know, from the definitions, what each column is measuring, and I want the bottleneck argument to rest on arithmetic I can redo rather than on intuition.

Everything here comes from four sessions on 2026-09-11 (run ids `20260911-222321`, `20260911-224931`, `20260911-230548`, `20260911-235937`), and every table names the run it came from. vLLM source is quoted from `main` as of 2026-09-11, and the container image is the recipe's `vllm/vllm-openai:deepseekv41-flash-0909`, which reports itself as `v0.1.dev20904+g179dd0fa9` and is not `main`; where the two differ I say so.

The plan:

- The numbers that drive everything: the node as measured, the model's byte floor per decode step, the workload's memory needs, and what an hour costs
- Metrics from first principles: how the harness computes TTFT, TPOT and throughput, why TPOT is the engine's step time, Little's law, and what nvidia-smi utilization and power do and do not mean
- The baseline: the published command, the two ceilings it hits at boot and why, the boot fix, and the recipe's own number
- The concurrency sweep from 16 to 2048, and the 33x gap it exposes
- The fixed-cost decomposition of the decode step
- Where the bottleneck is, by elimination, including a correction to how the CUDA graph experiment should be read
- The three configuration changes tested, each with its mechanism
- What is solid, what is not, and what to run next
- Takeaways, a test-yourself section, and wrapping up

Let's get started.

---

#### **The Numbers That Drive Everything**

##### **The node, as measured**

The briefing quoted NVIDIA's datasheet for B300. This section is what the node actually reported. The container is a Modal 8-GPU allocation; the values below are read from `nvidia-smi` inside it, sampled during the runs.

| Quantity | Measured | Where it comes from |
|---|---|---|
| GPUs | 8x NVIDIA B300 SXM6, one container | `nvidia-smi`, session `20260911-222321` |
| Memory per GPU | 275,040 MiB (268.6 GiB), of which vLLM sees 267.69 GiB | `nvidia-smi` `memory.total`; vLLM's `Free memory on device (264.8/267.69 GiB)` log line |
| Driver, CUDA | 580.95.05, 13.0 | `nvidia-smi` |
| Power limit | 1,100 W per GPU | `nvidia-smi` `power.limit`, every sample in every GPU trace |
| SM clock under load | about 2,030 MHz, flat across the run | GPU trace, run `20260911-235937`, concurrency 2048: per-GPU means of 2,027 to 2,032 MHz |
| NVLink | 18 links per GPU at 53.125 GB/s per direction, about 956 GB/s per direction | `nvidia-smi nvlink -s`; NVIDIA's [HGX page](https://www.nvidia.com/en-us/data-center/hgx/) quotes 1.8 TB/s GPU-to-GPU for HGX B300, which is the bidirectional figure |
| Peer-to-peer copy bandwidth | about 760 GB/s from GPU 0 to every peer, uniform | a `torch` copy probe in session 1; uniform bandwidth to all peers means an NVSwitch fabric, not a ring |
| Host | 32 CPU cores requested, 768 GiB memory request with a 1 TiB limit, 320 GB `/dev/shm` | the session driver's container spec and `free -g` inside it |

Two of those rows matter later. The 1,100 W cap is well below the 1,400 W figure widely reported for Blackwell Ultra in the liquid-cooled rack configuration, and NVIDIA's own [DGX B300 page](https://www.nvidia.com/en-us/data-center/dgx-b300/) states about 14 kW for the whole 8-GPU system including CPUs, so a sub-1,400 W per-GPU budget is consistent with the product; I could not find a vendor page that states the per-GPU figure for the air-cooled HGX part, so treat 1,100 W as what this node enforces rather than as a datasheet number. And 32 CPU cores for eight data-parallel ranks is a choice, not a default: Modal allocates 8 cores to an 8-GPU container unless told otherwise, and the sessions pinned 32 so that CPU starvation would not be an accidental confound. It is the one variable the bottleneck section ends up pointing at.

##### **The model, in the numbers that matter here**

The briefing has the full architecture; here is the subset the arithmetic below uses, all from the [tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) and [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json) unless noted.

| Quantity | Value |
|---|---|
| Backbone parameters | 552B, plus 196B of Engram tables (the recipe page says 522B; the report says 552B, and I use the report) |
| Active per token | 8B in prefill, 16B in decode |
| Layers, hidden size | 40, 5120 |
| Routed experts | 384 per layer, top-6, plus 1 shared; about 35.4M parameters per expert |
| Expert weight format | MXFP4, about 0.53 bytes per parameter with scales; dense weights MXFP8, about 1.03 bytes per parameter (the briefing's derivation) |
| Hyper-connections | 4 residual streams, 20 Sinkhorn iterations per sublayer (`hc_mult`, `hc_sinkhorn_iters`) |
| Engram | two modules, at layers 1 and 14; 384M rows of 256 FP8 channels each; 188.8 GiB on disk per the [recipe's memory table](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) |
| Main KV cache | four source layers, 2.5 entries per token, 584 bytes per entry in vLLM's `fp8_ds_mla` layout |
| Context | 1,048,576 tokens |
| torch.compile | not supported for this model; CUDA graphs only |

Under the recipe's layout, data-parallel 8 with expert parallelism, each GPU holds one full copy of the attention and dense weights and a 48-expert slice of every layer. The measured per-GPU weight footprint is **45.11 GiB** (`Model loading took 45.11 GiB memory`, every worker, every session), against a 475.25 GiB checkpoint. That is expert parallelism working as designed.

##### **The byte floor for one decode step**

The [inference systems post](/blog/2026/llm-inference-systems/) built the decode floor as weight bytes over bandwidth. For this model under this layout it has two parts, because the dense weights are replicated per rank and the experts are sharded.

Every rank reads its full copy of the dense and shared-expert weights every step, whatever the batch: about 7.4B parameters at 1.03 bytes each, call it 7.6 GB. Routed experts are different. A rank only reads the experts that some token in the step routed to, and only its own 48-per-layer slice of them. With 6 of 384 experts per token and $$B$$ tokens routed independently, the expected fraction of experts touched is

$$
1 - \left(1 - \tfrac{6}{384}\right)^{B}
$$

which is 22% at 16 tokens node-wide (2 sequences per rank), 87% at 128 tokens (16 per rank), and effectively 100% past a few hundred. A rank's full expert slice is $$259.5 / 8 \approx 32.4$$ GiB, about 34.8 GB. So the bytes a rank must stream per decode step run from

$$
7.6 + 0.22 \times 34.8 \approx 15.3 \text{ GB at 2 seqs/rank}
$$

to

$$
7.6 + 34.8 \approx 42.4 \text{ GB at full expert coverage}
$$

and at the B300's 8 TB/s that is **about 1.9 ms per step at 2 sequences per rank, rising to about 5.3 ms** once every expert is touched. Hold both numbers. The measured step is 17 ms at the low end and 36 ms at the high end, so the node runs at roughly nine times its byte floor at small batch and seven times at large batch. The whole post is about the gap.

##### **The KV cache this workload needs**

The recipe's benchmark uses 1024 input tokens and 1024 output tokens per request, so a request holds at most 2048 tokens of cache. At client concurrency 512, spread over 8 ranks, that is $$64 \times 2048 = 131{,}072$$ tokens per rank. The boot-fixed server reports `GPU KV cache size: 51,558,777 tokens` per rank, about **390 times** more than the workload can use. Every KV-related knob is therefore a free variable on this workload, which is what makes the boot fix in the baseline section cost nothing.

That log line also gives a number the briefing could only estimate: $$129.87 \text{ GiB} / 51{,}558{,}777 \approx 2{,}705$$ bytes of cache per token as vLLM actually allocates it, against the 1,460 bytes of main KV the briefing derived. The rest is the indexer key cache, the per-layer 128-token sliding-window pools, and block-granular allocation; I did not decompose it further.

##### **What an hour costs**

From [Modal's pricing page](https://modal.com/pricing) as of 2026-09-11: a B300 is $0.001972 per second, about $7.10 per GPU-hour; a physical CPU core is about $0.047 per core-hour; memory is about $0.008 per GiB-hour. The container this post used therefore lists at

$$
8 \times 7.10 + 32 \times 0.047 + 768 \times 0.008 \approx 56.8 + 1.5 + 6.1 \approx \$64 \text{ per hour}
$$

Memory is not a rounding error here: the 768 GiB request that the baseline needs (see the host-memory ceiling below) adds about 10% to the GPU bill. The four sessions billed $173.99 in total on the workspace dashboard ($154.07 GPU, $14.89 memory, $5.04 CPU), of which the two boot failures cost $37.36, about a fifth. At $64 per hour, the recipe's baseline of 1,451.5 output tokens per second works out to about $12.30 per million output tokens; the peak of 48,159 tokens per second works out to about $0.37 per million. Same node, same model, same hour.

**References**
- [NVIDIA DGX B300 page](https://www.nvidia.com/en-us/data-center/dgx-b300/), [NVIDIA HGX page](https://www.nvidia.com/en-us/data-center/hgx/)
- [Modal pricing](https://modal.com/pricing)
- [DeepSeek-V4.1-Flash tech report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf), [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json), [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- [Serving DeepSeek-V4.1-Flash on vLLM, the briefing](/blog/2026/deepseek-v41-flash-vllm/), [LLM inference systems post](/blog/2026/llm-inference-systems/)
- Server logs, runs `20260911-222321` and `20260911-230548`

---

#### **Metrics From First Principles**

Every number in this post comes out of `vllm bench serve`, so before reading any of them it is worth knowing exactly what the harness computes and what each quantity physically corresponds to inside the engine.

##### **What the harness does**

`vllm bench serve` is a closed-loop client. With `--max-concurrency C` it keeps at most $$C$$ requests outstanding at once, sends the next prompt the moment one completes, and runs until `--num-prompts` have finished ([benchmark CLI doc](https://docs.vllm.ai/en/latest/benchmarking/cli/)). The `random` dataset draws prompts of exactly `--random-input-len` tokens and asks for exactly `--random-output-len` tokens; because the prompts are random, no two share a prefix, so prefix caching (on by default) has nothing to hit. The default endpoint is `/v1/completions` (`vllm/benchmarks/serve.py`, the argument default, as of 2026-09-11), and a probe of that endpoint on the live server returned a three-token completion with `finish_reason: stop`, so the chat template, and with it the model's thinking-on-by-default behaviour, never enters the benchmark path.

Two facts checked rather than assumed: every run in this post reports total generated tokens equal to prompts times 1024 (or 128 for warmups), so no request stopped early and `--ignore-eos` was not needed; and the harness's own summary reports the same numbers as the JSON it writes, which is what the tables below are built from.

##### **The four numbers and how they are computed**

From `calculate_metrics` in `vllm/benchmarks/serve.py` (as of 2026-09-11), with the request timing captured per streamed chunk in `vllm/benchmarks/lib/endpoint_request_func.py`:

| Metric | Computation | What it corresponds to in the engine |
|---|---|---|
| TTFT | time from sending the request to the first streamed chunk | queueing behind other requests, plus the request's own prefill (1024 tokens here), plus HTTP and tokenization |
| TPOT | $$(\text{E2EL} - \text{TTFT}) / (\text{output tokens} - 1)$$, one value per request, then averaged | the mean gap between this request's consecutive tokens, which is the mean engine step time while it decoded |
| ITL | every gap between consecutive chunks, pooled across requests | one sample per step per request; its p99 is where stalls show up |
| Output throughput | total output tokens divided by the benchmark's wall-clock duration | tokens per second across the whole run, ramp-up and drain included |

The TPOT row deserves a sentence of justification, because the entire bottleneck argument rests on it. Under continuous batching a decoding sequence receives exactly one token per engine step, so the gap between its consecutive tokens **is** the duration of the step that produced the second one. TPOT averages those gaps over the request's 1,023 gaps; averaged again over requests, it is the mean step time of the engine while those requests were decoding. The distribution backs this up: at the recipe's operating point the mean TPOT is 17.23 ms and the p99 TPOT is 17.41 ms (run `20260911-230548`, `baseline_recipe_c32`), so the per-request means barely spread. **When TPOT does not change with batch size, the step is not doing batch-proportional work.**

##### **Little's law, and why the decode efficiency column is what it is**

A closed-loop benchmark is a textbook queueing system, and [Little's law](https://en.wikipedia.org/wiki/Little%27s_law) applies to it directly: the long-run average number of requests in the system equals the arrival rate times the average time each spends inside, $$L = \lambda W$$, regardless of the arrival or service distributions. Here $$L = C$$ by construction, since the client always keeps $$C$$ requests in flight, and $$W$$ is the mean end-to-end latency. So the request rate is $$C / \text{E2EL}$$, and with exactly 1024 output tokens per request,

$$
\text{output tokens/s} = \frac{1024 \, C}{\text{E2EL}}
$$

Checking that identity against the measured throughput is a cheap way to confirm the run was actually at steady state:

| Concurrency | $$1024 C / \text{E2EL}$$ | Measured tok/s | Ratio | TTFT as share of E2EL | Measured over $$C/\text{TPOT}$$ |
|---|---|---|---|---|---|
| 16 | 919.4 | 918.90 | 0.9995 | 0.8% | 0.992 |
| 32 (128 prompts) | 1,850.6 | 1,849.48 | 0.9994 | 1.4% | 0.986 |
| 64 | 3,552.8 | 3,550.93 | 0.9995 | 5.5% | 0.946 |
| 128 | 7,184.3 | 7,177.73 | 0.9991 | 3.7% | 0.963 |
| 256 | 12,939.8 | 12,924.70 | 0.9988 | 5.1% | 0.949 |
| 512 | 23,368.0 | 23,313.68 | 0.9977 | 7.5% | 0.924 |
| 1024 | 33,848.1 | 33,716.06 | 0.9961 | 11.8% | 0.880 |
| 2048 | 48,476.2 | 48,159.05 | 0.9935 | 14.2% | 0.853 |

Little's law holds to within 0.7% on every clean run. Now substitute the TPOT definition, $$\text{E2EL} = \text{TTFT} + 1023 \times \text{TPOT}$$, and the identity rearranges to

$$
\text{output tokens/s} \approx \frac{C}{\text{TPOT}} \times \left(1 - \frac{\text{TTFT}}{\text{E2EL}}\right)
$$

The first factor is the throughput the machine would deliver if every one of the $$C$$ sequences produced a token every step. The second factor is the fraction of a request's life it spends decoding rather than waiting for its first token. Compare the last two columns of the table: they are the same number. What looks like a "decode efficiency" that decays from 0.99 to 0.85 as concurrency rises is nothing more mysterious than TTFT growing from 0.8% to 14% of the request's lifetime, because a request arriving at concurrency 2048 queues behind more prefill work before it gets its first token. The engine is not becoming less efficient; each request is spending longer in the queue.

The same identity explains the recipe's own number. At concurrency 32 with 100 prompts, Little's law predicts $$1024 \times 32 / 17.878 \approx 1{,}833$$ tokens per second, but the measurement is 1,451.5. The closed-loop assumption fails: 100 prompts at concurrency 32 is 3.125 waves, so the run spends its first wave ramping up and its last wave with only 4 requests in flight, and $$L$$ is well below 32 for a large fraction of the wall clock. The wall clock is what the throughput divides by: 4 waves of about 17.9 s is about 71.5 s, the run took 70.5 s, and $$102{,}400 / 70.5 \approx 1{,}452$$. Running the same server at the same concurrency with 128 prompts, exactly 4 full waves, gives 1,849.48, a 27% higher number for the same machine in the same state. Both numbers are honest. The first is the recipe's command and is therefore the baseline; the second is the steady-state figure at the same operating point, and every other row in this post uses whole waves.

##### **What nvidia-smi utilization and power mean**

The elimination argument later leans on two telemetry signals, so their definitions matter. NVIDIA's [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html) defines GPU utilization as the percent of time over the past sample period during which one or more kernels was executing, with a sample period between one sixth of a second and one second depending on the product. It is a **busy fraction**, not an efficiency: a GPU running one tiny kernel at a time, back to back, reads 100%, and a GPU that is idle half of every millisecond reads 50% no matter how fast its kernels are. Power draw is the board's measured draw in watts, and the power limit is the enforced ceiling.

The session driver sampled every GPU roughly every two seconds during each benchmark (64 samples over a 151 s window around the 87 s concurrency-2048 run, a mean interval of 2.4 s) and reports the mean utilization over samples and GPUs, the spread between the highest and lowest per-GPU mean, and the mean power. Read together, the two signals split the world into four cases:

| Utilization | Power | Reading |
|---|---|---|
| high | high | the GPU is busy and working hard: compute or bandwidth bound |
| high | low | the GPU is busy with kernels that do little per unit time: many tiny kernels, or memory-latency-bound ones |
| low | high | short bursts of heavy work with gaps between them |
| low | low | the GPU is waiting for something else, most of the time |

Two caveats on the busy fraction. Gaps between kernels inside a CUDA graph count as idle, because no kernel is executing during them. And communication collectives run as kernels, so time spent in NCCL or the expert all-to-all counts as busy, not idle. Both matter when the elimination section reads a 50% number.

**References**
- [vLLM benchmark CLI doc](https://docs.vllm.ai/en/latest/benchmarking/cli/); vLLM source, as of 2026-09-11: [`vllm/benchmarks/serve.py`](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py), [`vllm/benchmarks/lib/endpoint_request_func.py`](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/lib/endpoint_request_func.py)
- [Little's law](https://en.wikipedia.org/wiki/Little%27s_law)
- [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
- [LLM inference systems post, metrics section](/blog/2026/llm-inference-systems/)
- Harness JSON for runs `20260911-230548` and `20260911-235937`

---

#### **The Baseline**

##### **The published command**

The recipe's single-node "Data + Expert Parallel" command for B300, minus the docker wrapper the briefing already walked through:

```bash
vllm serve deepseek-ai/DeepSeek-V4.1-Flash \
  --tokenizer-mode deepseek_v41 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data
```

with `VLLM_ENGINE_READY_TIMEOUT_S=3600` and `VLLM_USE_RUST_FRONTEND=1`. And its benchmark:

```bash
vllm bench serve \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --host localhost --port 8000 \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 100 --max-concurrency 32
```

The client ran inside the same container as the server, so no network round trip sits inside any latency number.

What the startup log resolves silently, as read from run `20260911-230548`:

| Setting | Resolved | Log line |
|---|---|---|
| Max model length | 1,048,576 | `Using max model len 1048576` |
| Chunked prefill budget | 16,384 tokens per step per rank | `Chunked prefill is enabled with max_num_batched_tokens=16384` (the recipe's Advanced panel suggests 8192, which would halve it) |
| Async scheduling | on | `Disabling NCCL for DP synchronization when using async scheduling` |
| CUDA graphs | `FULL_AND_PIECEWISE`, 83 capture sizes up to 1024, breakable graphs auto-enabled | the `compilation_config` dump; `Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| torch.compile | off | `mode: CompilationMode.NONE`, `splitting_ops: []` |
| MoE kernels | `FLASHINFER_TRTLLM_MXFP4_MXFP8` | `Using 'FLASHINFER_TRTLLM_MXFP4_MXFP8' Mxfp4 MoE backend` |
| All-to-all | allgather plus reduce-scatter | `Using AgRsAll2AllManager all2all manager` |
| Attention | `FLASHMLA_SPARSE_DSV41`, KV block size 128, `fp8_ds_mla` | `Setting kv cache block size to 128 for FLASHMLA_SPARSE_DSV41 backend` |
| Engram | sharded over 8 ranks, in pinned host memory | `Engram tables are sharded over 8 ranks (TP=1 x DP=8)`; twice per rank, `Engram table offloaded to pinned host memory: 48,00x,xxx rows x 256, 11.80 GiB per rank` |
| Rust frontend | ignores two of the flags | `argument 'enable_auto_tool_choice' currently has no effect in Rust frontend, ignoring`; likewise `structured_outputs_config`; and `placeholder tokens did not resolve; disabling this modality` |

So the FP4 expert kernels run natively rather than dequantizing to something wider, async scheduling is already on and is not a lever, and two of the correctness flags in the command are inert under the frontend the same command selects.

The Engram row corrects something the briefing got wrong. The briefing read `embedding_across_dp` (default `False`, "each DP rank has a separate TP-sharded embedding replica", `vllm/config/engram.py` as of 2026-09-11) and concluded that DP8 would pin eight full copies of the 188.8 GiB table. The image does not do that: it logs one table sharded across all eight ranks, 11.80 GiB per rank per module, 23.6 GiB per rank in total, 188.8 GiB per node, which is exactly the checkpoint's table size. Whether `main` behaves the same way with no flag depends on how the Engram parallel group is sized (`get_parallel_size` in the same file, and the `_ETP` group in `vllm/distributed/parallel_state.py`), and I have not run `main` on the node, so the safe statement is: check the `sharded over` log line on whatever build you run. The placement problem, pinned host memory read over UVA on every token, is real either way and is tested below.

##### **Ceiling one: the GPU runs out of memory during kernel warmup**

Run `20260911-222321` died 21 minutes into startup. The per-GPU sequence, from the log:

| Phase | Result | Log line |
|---|---|---|
| Load weights | 45.11 GiB, 355 s (475.25 GiB read over the volume mount) | `Model loading took 45.11 GiB memory and 354.95 seconds` |
| Memory profiling, including a throwaway graph capture | 168 s, 10.18 GiB captured, then freed | `Capturing CUDA graphs (PIECEWISE) 83/83`, `(FULL) 2/2`, `Graph capturing finished in 168 secs, took 10.18 GiB` |
| KV cache sizing | 167.35 GiB, 66,437,840 tokens per rank | `Available KV cache memory: 167.35 GiB`; `Maximum concurrency for 1,048,576 tokens per request: 63.36x` |
| JIT kernel warmup | 105 compile keys in 7 kernel groups; dies | `JIT kernel warmup starting`, then `CUDA out of memory. Tried to allocate 1.85 GiB. GPU 3 has a total capacity of 267.69 GiB of which 1.13 GiB is free` inside `flashinfer::trtllm_fp4_block_scale_moe` |

The mechanism is the order of those phases, and it is worth understanding from the source because it is not specific to this model. In `vllm/v1/worker/gpu_worker.py` (as of 2026-09-11): `determine_available_memory` runs a dummy forward to measure peak activation memory and, when CUDA graphs are enabled, calls `profile_cudagraph_memory`, which captures a small subset of graphs into a temporary pool to estimate their footprint and discards them (`vllm/v1/worker/gpu_model_runner.py`, the profiling loop captures the first two descriptors per mode). Then `initialize_from_config` allocates the KV cache. Only then does `compile_or_warm_up_model` run `kernel_warmup` and, after it, the real `capture_model`. The KV cache is sized from a snapshot taken before the warmup has allocated anything.

The arithmetic in the log bears this out. The requested budget is $$0.92 \times 267.69 = 246.27$$ GiB, and the image sized the cache as requested minus consumed (weights plus non-torch, 51.22 GiB) minus peak activation (27.7 GiB), which is 167.35 GiB to the hundredth. The graph-memory estimate was not subtracted; on `main` it is applied only under an opt-in environment variable, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, and the log's arithmetic says the image behaves the same way. After the KV allocation the GPU had $$267.69 - 51.22 - 167.35 \approx 49$$ GiB free. FlashInfer's FP4 MoE autotune, run inside the warmup, holds many tactic workspaces live at once; by the time it asked for one more 1.85 GiB buffer, 266.47 GiB were in use. And the real graph capture, another 9.6 GiB, had not even started.

Two consequences. The effective default is `--gpu-memory-utilization 0.92`, not the 0.9 that older docs quote; it is `Field(default=0.92)` in `vllm/config/cache.py` and the log says `--gpu-memory-utilization=0.9200` explicitly. And the recipe's Advanced panel suggests raising it to 0.95, which hands the KV cache another 8 GiB and makes this failure strictly worse.

##### **The fix, and why it is not an optimization**

Adding `--gpu-memory-utilization 0.78` lowers the request by $$0.14 \times 267.69 = 37.5$$ GiB per GPU. The KV cache shrinks to 129.87 GiB and 51,558,777 tokens per rank (`Maximum concurrency for 1,048,576 tokens per request: 49.17x`), the warmup and the 9.61 GiB real capture fit, and the server comes up. The final memory line from the boot-fixed run:

> Desired GPU memory utilization is (0.78, 208.8 GiB). Actual usage is 51.22 GiB for consumed memory (weights + non-torch), 27.7 GiB for peak activation, and 9.61 GiB for CUDAGraph memory. Current kv cache memory in use is 129.87 GiB.

This costs nothing measurable because, as computed above, the workload needs about 131k tokens of cache per rank and has 51.6 million. It is recorded as the minimum change required to start, not as a win, and every "baseline" number below includes it.

##### **Ceiling two: host memory**

Run `20260911-224931`, with the GPU fix, lost one worker during weight loading with no Python traceback, preceded by `all workers exited gracefully` and followed by `WorkerProc initialization failed due to an exception in a background process`. A worker that dies silently, with no exception, has been killed, and the cgroup out-of-memory killer is the usual sender.

The arithmetic supports it. Eight ranks pin 23.6 GiB each of Engram table in host memory, 188.8 GiB that cannot be reclaimed, before the page cache of a 475 GiB checkpoint being read through the same kernel is counted, all against a 400 GiB container request. Raised to a 768 GiB request with a 1 TiB limit, the next launch recorded a host memory peak of **792.1 GiB** during startup (793.8 GiB the session after), which is a number rather than an inference. Two boot ceilings, then, and both are memory-accounting gaps between phases: the GPU one between profiling and warmup, the host one between the pinned tables and the page cache.

##### **Startup cost**

Every configuration pays the whole sequence, which is why the sessions batched several configurations per booking. From the boot-fixed baseline of run `20260911-230548`:

| Phase | Time | Memory |
|---|---|---|
| Load weights | 534 s | 45.11 GiB |
| Profiling forward plus throwaway capture | 117 s | 10.18 GiB, freed |
| JIT kernel warmup (`BuildPrefillChunkMetadataKernel`, `CombineTopkSwaIndicesKernel`, `ComputePrefillMetadataKernel`, `ComputeSWAIndicesAndLensKernel`, `MHCPreNormKernel`, `PrepareUniformDecodeKernel`, and the MoE autotune) | 51 s for the JIT stage; about 6 min more for the router GEMM, sparse MLA and FlashInfer autotune warmups | the workspaces that killed the first run |
| Real graph capture | 45 s | 9.61 GiB: 83 piecewise graphs and 83 full graphs |
| Ready | 1,338 s total | 200.9 GiB in use per GPU |

Later launches in the same container are much faster (738 to 872 s), but that is confounded: the checkpoint is warm in the page cache and the JIT warmup finishes in about a second because the autotune results are cached. Do not attribute the difference to any flag.

##### **The baseline number**

The recipe's bench command, verbatim, against the recipe's server plus the boot fix (run `20260911-230548`, `baseline_recipe_c32`):

> **1,451.50 output tokens per second.** 100 requests of 1024 in and 1024 out at client concurrency 32; 70.55 s wall; mean TTFT 250.35 ms; mean TPOT 17.23 ms; mean end-to-end latency 17.88 s.

The metrics section already showed that this number sits 21% below the steady-state figure of 1,849.48 purely because of the 3.125-wave shape of the recipe's own command. Both go in the record. The bar to beat is 1,451.50, because that is what the recipe measures; the number to reason from is 1,849.48, because that is what the machine does at this operating point.

**References**
- [vLLM recipe for DeepSeek-V4.1-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- vLLM source, as of 2026-09-11: [`vllm/v1/worker/gpu_worker.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_worker.py), [`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py), [`vllm/config/cache.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py), [`vllm/config/engram.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/engram.py), [`vllm/distributed/parallel_state.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py)
- Server logs, runs `20260911-222321`, `20260911-224931`, `20260911-230548`

---

#### **The Concurrency Sweep**

The recipe pins `--max-concurrency 32`, and that is a client-side cap: no server-side change can exceed 32 sequences in flight while it stays there. So before touching any server flag, the first experiment is to sweep it on the unchanged server. Sessions `20260911-230548` (16 through 512) and `20260911-235937` (1024 and 2048); prompts scale with concurrency so that every row except the recipe's is a whole number of waves.

| Client concurrency | Seqs per rank | Prompts | Waves | Output tok/s | Gain per doubling | Mean TPOT | Mean TTFT | p99 TTFT |
|---|---|---|---|---|---|---|---|---|
| 8 (warmup, 128 out) | 1 | 32 | 4 | 441.39 | | 17.14 ms | 140.5 ms | 205 ms |
| 16 | 2 | 64 | 4 | 918.90 | | 17.27 ms | 150.0 ms | 223 ms |
| **32 (recipe)** | 4 | 100 | 3.1 | **1,451.50** | | 17.23 ms | 250.4 ms | 489 ms |
| 32 (clean) | 4 | 128 | 4 | 1,849.48 | 2.01x | 17.06 ms | 249.6 ms | 372 ms |
| 64 | 8 | 256 | 4 | 3,550.93 | 1.92x | 17.05 ms | 1,008.1 ms | 2,896 ms |
| 128 | 16 | 512 | 4 | 7,177.73 | 2.02x | 17.17 ms | 679.1 ms | 997 ms |
| 256 | 32 | 1,024 | 4 | 12,924.70 | 1.80x | 18.80 ms | 1,030.8 ms | 1,545 ms |
| 512 | 64 | 1,024 | 2 | 23,313.68 | 1.80x | 20.29 ms | 1,675.5 ms | 2,610 ms |
| 1024 | 128 | 3,072 | 3 | 33,716.06 | 1.45x | 26.71 ms | 3,651.1 ms | 8,266 ms |
| 2048 | 256 | 4,096 | 2 | **48,159.05** | 1.43x | 36.26 ms | 6,164.2 ms | 10,878 ms |

<div class="row justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/deepseek-v41-flash-b300-benchmark/throughput-vs-concurrency.svg" title="Output throughput against client concurrency" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Output tokens per second against client concurrency on the unchanged baseline server, both axes log base 2, with the gain per doubling written at each point and the dashed line showing what perfectly linear scaling from concurrency 16 would give. The red square is the recipe's own 100-prompt command at concurrency 32. Runs 20260911-230548 (16 to 512) and 20260911-235937 (1024, 2048).
</div>

Three things to read off it. Throughput doubles per doubling of concurrency up to 128, which is the signature of a step whose duration does not depend on batch: twice the sequences, same step time, twice the tokens. It then bends, 1.8x per doubling through 512 and 1.45x through 2048, as the step starts to lengthen. And it has not flattened: the curve is still climbing at 1.43x per doubling at the largest batch the client was asked for, so 48,159 tokens per second is a soft knee, not a wall.

**The peak is 33.2 times the recipe's own measurement point.** The recipe's bench command, at the recipe's concurrency, reads the machine at about 3% of what it delivers at concurrency 2048, on the same server with no flag changed. That is not a criticism of the recipe; a 100-prompt run at concurrency 32 is a smoke test, and a good one. But any optimization claim that compares a tuned configuration at high concurrency against this baseline at 32 is measuring the client, not the server, and the point of this section is to make sure no such comparison sneaks in later.

One row in that table is off-pattern: mean TTFT at concurrency 64 (1,008 ms) is higher than at 128 (679 ms), and its distribution is bimodal, median 549 ms, p90 2,888 ms. Something stalled admission for part of that run. It is worth remembering when the Engram experiment's TTFT delta at concurrency 64 comes up.

**References**
- Harness JSON and bench logs, runs `20260911-230548` and `20260911-235937`
- [vLLM benchmark CLI doc](https://docs.vllm.ai/en/latest/benchmarking/cli/)

---

#### **The Fixed-Cost Decomposition**

##### **The observation**

Look at the TPOT column of the sweep from concurrency 16 to 128. Batch size per rank goes from 2 to 16, an eight-fold change, and TPOT moves between 17.05 and 17.27 ms, a change of 1.3% in the wrong direction. A decode step that takes the same time with 2 sequences as with 16 is not doing work that scales with the batch. It is paying a toll per step.

##### **Quantifying it**

Model the step as a fixed term plus a per-sequence term,

$$
\text{TPOT} = t_{\text{fixed}} + t_{\text{marginal}} \times (\text{sequences per rank})
$$

The flat region gives $$t_{\text{fixed}} \approx 17.0$$ ms directly, and a least-squares fit over the bending region, 32 to 256 sequences per rank, gives $$15.9 + 0.080 \times (\text{seqs per rank})$$ ms. The two estimates of the fixed term disagree by about a millisecond, which is a fair statement of the uncertainty. Using the flat-region value:

| Seqs per rank | Client concurrency | Mean TPOT | Fixed share at 17.0 ms | Fixed share at the fitted 15.9 ms |
|---|---|---|---|---|
| 2 | 16 | 17.27 ms | 98.4% | 92.1% |
| 4 | 32 | 17.06 ms | 99.6% | 93.3% |
| 8 | 64 | 17.05 ms | 99.7% | 93.4% |
| 16 | 128 | 17.17 ms | 99.0% | 92.7% |
| 32 | 256 | 18.80 ms | 90.4% | 84.7% |
| 64 | 512 | 20.29 ms | 83.8% | 78.4% |
| 128 | 1024 | 26.71 ms | 63.6% | 59.6% |
| 256 | 2048 | 36.26 ms | 46.9% | 43.9% |

<div class="row justify-content-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/deepseek-v41-flash-b300-benchmark/tpot-vs-seqs-per-rank.svg" title="Mean TPOT against sequences per DP rank" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Mean TPOT, which is the decode step time, against sequences per DP rank on a log-2 x axis. The dotted line is the flat region's 17.0 ms; the dashed line is the least-squares fit over 32 to 256 sequences per rank, 15.9 ms plus 0.080 ms per sequence. Same runs as the throughput chart.
</div>

**At the recipe's operating point, the step is essentially all toll.** Even at 256 sequences per rank, the largest batch tested, the toll is still close to half the step. This single table is why the node delivers 33 times more throughput at concurrency 2048 than at 32: nothing got faster, a constant was amortized over more sequences.

##### **What the toll is not, by size**

The byte floor from the numbers section is the first thing to compare against, and the comparison is uncomfortable in an informative way. Between 2 and 16 sequences per rank, the expected expert coverage rises from 22% to 87%, so the bytes a rank streams per step rise from about 15 GB to about 38 GB and the byte floor from about 1.9 ms to about 4.7 ms. TPOT did not move. Either the weight streaming overlaps with whatever the toll is, which is exactly what async scheduling is designed to make happen when the step is bound by something other than the GPU, or the byte estimate is wrong by a factor that does not matter here. Either way, the 17 ms is not bandwidth.

The marginal term, 0.080 ms per sequence per rank, is also worth a sanity check. At 256 sequences per rank it adds about 20 ms, of which the expert slice being fully streamed accounts for about 4 ms; the remainder is per-token compute that genuinely scales with the batch: the expert GEMMs, the sparse attention over 512 selected entries plus a 128-token window per sequence, sampling, and the all-to-all volume.

##### **Why the shape was predictable**

The briefing's first hypothesis was that decode would be launch-bound rather than byte-bound, on the strength of one public trace: AMD's [RFC #56506](https://github.com/vllm-project/vllm/issues/56506) counted about 14,020 kernel launches per decode step on ROCm, 85% of each layer's launches in the hyper-connection block. The NVIDIA path in this image fuses the hyper-connection work into TileLang kernels (`mhc_pre_big_fuse_with_norm_tilelang`, `mhc_post_tilelang`, per the compile lines in the log), so the count here is lower, but the architecture still executes a great many small operations per layer, 40 layers deep, and that count is the same whether the step holds 2 sequences or 256. A large batch-independent term was the prediction. The measurement found one worth 17 ms. What the prediction got wrong is the mechanism, which is the next section's subject.

**References**
- [RFC #56506, DeepSeek-V4.1-Flash performance on ROCm](https://github.com/vllm-project/vllm/issues/56506)
- [Serving DeepSeek-V4.1-Flash on vLLM, the briefing](/blog/2026/deepseek-v41-flash-vllm/), the performance hypotheses section
- Harness JSON, runs `20260911-230548` and `20260911-235937`

---

#### **Where the Bottleneck Is, by Elimination**

A 17 ms toll exists. The interesting question is what it is made of. Five candidates were testable with what the sessions recorded, and four are now ruled out or nearly so.

##### **Not compute**

GPU utilization, the busy fraction defined in the metrics section, was sampled for every run in session `20260911-235937`:

| Run | Concurrency | Mean utilization | Spread across 8 GPUs | Mean power | Busy time per step | Idle time per step |
|---|---|---|---|---|---|---|
| baseline warmup | 8 | 15.0% | 4.8 pp | 265 W | 2.5 ms | 14.4 ms |
| `--max-model-len 4096` | 128 | 48.8% | 1.7 pp | 419 W | 8.0 ms | 8.4 ms |
| `FULL_DECODE_ONLY` | 128 | 49.3% | 1.1 pp | 414 W | 8.3 ms | 8.5 ms |
| `--max-model-len 4096` | 512 | 51.1% | 1.0 pp | 493 W | 10.5 ms | 10.1 ms |
| `FULL_DECODE_ONLY` | 512 | 52.7% | 1.4 pp | 502 W | 10.9 ms | 9.8 ms |
| baseline | 1024 | 55.9% | 3.3 pp | 578 W | 14.9 ms | 11.8 ms |
| `--max-model-len 4096` | 1024 | 52.8% | 2.0 pp | 587 W | 13.3 ms | 11.9 ms |
| `FULL_DECODE_ONLY` | 1024 | 53.6% | 1.6 pp | 593 W | 13.7 ms | 11.9 ms |
| baseline | 2048 | 51.2% | 0.8 pp | 563 W | 18.6 ms | 17.7 ms |

The last two columns are utilization times TPOT and its complement, so they say how much of each step the GPU spent with any kernel resident. From concurrency 128 upward the busy fraction sits between 49% and 56% for every configuration, and it **falls** from 55.9% at concurrency 1024 to 51.2% at 2048 while throughput rises 43%. The GPUs have nothing running about half the time at the largest batch the node was given, and giving them more work does not reduce the idle fraction. A compute-bound machine does not look like this.

##### **Not power**

Power sits between 414 and 593 W against an 1,100 W cap in every run, and the SM clock holds at about 2,030 MHz throughout the concurrency-2048 trace. Nothing is thermally or electrically limited; roughly half the power budget goes unused at every operating point.

##### **Not expert load imbalance**

If routing skew made one rank's expert slice the pacing item, the slowest GPU would be busy while the others waited, and the per-GPU utilization means would spread. They do the opposite: the spread is 4.8 percentage points at concurrency 8, 3.3 at 1024, and **0.8 at 2048**, tightening as load rises. With 6 of 384 experts per token and thousands of tokens per step node-wide, the law of large numbers evens the load out. `--enable-eplb` and redundant experts are not worth a booking on this workload.

##### **Not CPU-side kernel launching, and a correction on how to read the graph experiment**

The launch hypothesis had a clean test: `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`, which captures a full graph for every uniform decode batch size and no piecewise graphs at all ([vLLM CUDA graphs doc](https://docs.vllm.ai/en/latest/design/cuda_graphs/)). If the toll were the CPU cost of launching thousands of kernels per step, replaying one graph per step would remove it.

The result was a TPOT change of about 2% at concurrency 128 and within noise elsewhere (the numbers are in the changes section). The first reading of that was "the baseline captured only 2 full graphs, so the change bought the gap between piecewise and full capture." That reading is wrong, and the log shows why. Every configuration's startup has **two** capture passes, as the source in the baseline section predicts: a throwaway profiling pass, which in this image logged `PIECEWISE 83/83` and `FULL 2/2` and took 117 to 168 s, and then, after the kernel warmup, the real `capture_model`, which for the baseline logged `PIECEWISE 83/83`, then `FULL 83/83`, `Graph capturing finished in 45 secs, took 9.61 GiB`. The baseline already had a full graph for every one of the 83 decode batch sizes. `FULL_DECODE_ONLY` captured the same 83 full graphs (35 s, 9.76 GiB) and dropped the piecewise ones that mixed prefill-plus-decode steps would otherwise use.

So the two configurations ran the pure-decode step identically, as one full-graph replay, and the experiment could not have moved the decode toll. What it does establish is stronger than the original reading: **a decode step that is a single CUDA graph replay still costs 17 ms, with the GPU idle about half the time.** CPU launch overhead is what a graph removes; NVIDIA's own [CUDA graphs post](https://developer.nvidia.com/blog/cuda-graphs/) measures a launch-bound sequence going from 9.6 µs per kernel eager to 3.4 µs per kernel under a graph, against 2.9 µs of actual kernel time, and is explicit that the remaining cost is the device-side work of running each node. Since the toll survived full-graph replay, it is not the CPU launching kernels.

There is one hedge on this image specifically. The model runs under breakable CUDA graphs (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `vllm/compilation/breakable_cudagraph.py`), which insert stream-capture breaks around the attention ops so that a "full" graph is really a chain of graph segments with eager attention between them. Each break is a CPU-side handoff per layer per step. The count of breaks is fixed per step, which fits the shape of the toll, and only a timeline can say what they cost.

##### **What is left**

Three candidates survive, and the telemetry constrains their shape more than the original framing allowed.

The busy-and-idle table says the idle time per step is not a constant either: it grows from about 8.5 ms at 16 sequences per rank to about 17.7 ms at 256, tracking the busy time at roughly one to one. A single 17 ms pause at the step boundary, say a scheduler that takes that long, would give a constant idle term and a rising busy fraction; that is not what was measured. The idle is distributed through the step and scales with it. That points at gaps **between** kernels rather than one gap between steps:

1. **Device-side gaps between many short kernels.** A step made of thousands of kernels that each run for a few microseconds leaves the SMs idle between them even under graph replay, and the count of those gaps is batch-independent while each kernel's own duration grows with batch. This fits a toll that graphs do not remove, a busy fraction near 50% that does not change with batch, and an idle time that scales with the busy time. Communication collectives count as busy, so the idle is not the all-to-all itself.
2. **Host-side handoffs inside the step.** The breakable-graph attention breaks, and the Engram host-memory gathers on layers 1 and 14 when the tables are offloaded, are per-layer CPU-GPU synchronization points. Eight engine-core processes and eight worker processes, plus the Rust frontend and the DP coordinator, share 32 cores.
3. **Inter-rank synchronization.** Expert parallelism forces every rank into lockstep every step. Before each forward pass every DP rank all-reduces a small tensor carrying its token count, its padded token count, its micro-batching decision and its cudagraph mode, so that all eight agree on the step's shape (`_run_ar` and `coordinate_batch_across_dp` in `vllm/v1/worker/dp_utils.py`, as of 2026-09-11); with async scheduling that all-reduce goes over the CPU group rather than NCCL, per the `Disabling NCCL for DP synchronization` log line. Then the expert all-to-all inside the graph waits for the slowest rank every layer. Any per-step jitter on one rank is paid by all eight.

The CPU allocation is the one variable named in the briefing's hypotheses that has never been tested, and it is the cheapest discriminator between these three: rerun the sweep at 8, 32 and 64 cores. If TPOT does not move, the toll is on the device timeline and candidate 1 leads; if it does, candidates 2 and 3 do. After that, an Nsight Systems timeline of one engine core at concurrency 8 turns "17 ms of something" into named kernels and named gaps. That is the ordered plan in the last section.

**References**
- [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [NVIDIA CUDA graphs post](https://developer.nvidia.com/blog/cuda-graphs/)
- [vLLM CUDA graphs design doc](https://docs.vllm.ai/en/latest/design/cuda_graphs/), [vLLM data parallel deployment doc](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/), [vLLM expert parallel deployment doc](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- vLLM source, as of 2026-09-11: [`vllm/v1/worker/dp_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/dp_utils.py), [`vllm/v1/worker/gpu_model_runner.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py), [`vllm/compilation/breakable_cudagraph.py`](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/breakable_cudagraph.py)
- GPU traces and server logs, run `20260911-235937`

---

#### **The Changes Tested**

Three server-side changes were measured, each on its own server launch, each against the baseline at matched concurrency. One caution applies to all of them and is spelled out in the next section: the same configuration has never been run twice at the same concurrency, so every effect below sits inside a band that run-to-run variance could plausibly explain.

##### **Engram in HBM: keep**

`--engram-config '{"cpu_offload": false}'`. Confirmed in effect by `Resolved Engram configuration: EngramConfig(cpu_offload=False)` and by the disappearance of the `offloaded to pinned host memory` lines; the model load grew from 45.11 to **68.72 GiB per GPU**, a difference of 23.61 GiB, which is the two 11.80 GiB shards moving onto the device.

**What the default does.** With no `--engram-config`, the layer constructs its table with `cpu_offload=True` (`vllm/models/deepseek_v4_1/common/engram.py`, as of 2026-09-11), so each rank's 23.6 GiB shard lives in pinned host memory and every token's gathers, 24 rows of 256 FP8 bytes per module, go across PCIe through unified virtual addressing. The bandwidth is trivial. The latency is a host round trip on the critical path of layers 1 and 14, twice per token.

**Effect**, same server session, run `20260911-230548`:

| Concurrency | Baseline tok/s | Engram in HBM | Delta | Baseline TPOT | HBM TPOT | Baseline TTFT (median) | HBM TTFT (median) |
|---|---|---|---|---|---|---|---|
| 8 (128 out) | 441.39 | 450.66 | +2.1% | 17.14 ms | 16.86 ms | 141 (129) ms | 129 (126) ms |
| 64 | 3,550.93 | 3,710.46 | +4.5% | 17.05 ms | 16.81 ms | 1,008 (549) ms | 444 (403) ms |
| 128 | 7,177.73 | 7,350.75 | +2.4% | 17.17 ms | 16.71 ms | 679 (669) ms | 730 (800) ms |
| 256 | 12,924.70 | 13,050.37 | +1.0% | 18.80 ms | 18.54 ms | 1,031 (1,129) ms | 1,096 (1,240) ms |

TPOT is consistently 1.4 to 2.7% lower, which is the expected shape: decode gathers one token's rows per sequence and barely notices the placement. The TTFT column is the one to be careful with. Prefill gathers rows for all 1024 prompt tokens at once and does pay real PCIe traffic, so a TTFT improvement is mechanically plausible, but the concurrency-64 baseline TTFT is the anomalous bimodal run flagged in the sweep section, and at 128 and 256 the ordering flips. Trust the TPOT column; treat the headline "TTFT fell from 1,008 to 444 ms" as unconfirmed.

**Cost and side effects.** 23.6 GiB of HBM per rank, which this workload has no other use for; the KV cache absorbed it (106.33 GiB instead of 129.87 GiB, still 42 million tokens per rank). The host-memory peak during startup fell from 792.1 to **527.2 GiB**, which removes the second boot ceiling entirely. Small, consistent, free at this workload, and it deletes a failure mode.

##### **`--max-model-len 4096`: promising, unconfirmed**

**Rationale.** The engine is configured for a 1,048,576-token context while no request exceeds 2048. With a KV block size of 128, a sequence's block table is sized for 8,192 entries instead of 32, and the sparse-attention indexer's candidate machinery is sized for the same worst case. Per-step costs that depend on the configured maximum rather than on the tokens actually present are exactly the shape of the toll.

**What changed at startup.** Peak activation memory fell from 27.7 to 15.78 GiB and the real graph capture from 9.61 to 5.13 GiB, both from the memory-budget log line, which says the per-step working set really did shrink. The KV log line changed to `GPU KV cache size: 5,066,461 tokens, Maximum concurrency for 4,096 tokens per request: 1236.93x`, a token count an order of magnitude below the baseline's from a larger pool (145.71 GiB); the count is derived from the concurrency figure and the hybrid allocator's per-request accounting changes with the context limit, and I did not chase the arithmetic. Capacity is not the constraint either way.

**Effect** (run `20260911-235937`; the 128 and 512 baselines are from the previous session, marked with an asterisk):

| Concurrency | Baseline tok/s | maxlen tok/s | Delta | Baseline TPOT | maxlen TPOT | TPOT delta |
|---|---|---|---|---|---|---|
| 128 | 7,177.73\* | 7,475.17 | +4.1% | 17.17 ms\* | 16.45 ms | -4.2% |
| 512 | 23,313.68\* | 22,638.09 | -2.9% | 20.29 ms\* | 20.59 ms | +1.5% |
| 1024 | 33,716.06 | 35,842.59 | +6.3% | 26.71 ms | 25.17 ms | -5.8% |

The matched-session pair at concurrency 1024 is the interesting one: 52.8% utilization against the baseline's 55.9%, 587 W against 578 W, and 6.3% more throughput. Less wall-clock time with kernels resident, more power while they are, more tokens out: that is per-step work being removed rather than the scheduler packing more per step. Concurrency 512 disagrees, and the next section says why none of the three rows can be called a result yet.

##### **`FULL_DECODE_ONLY`: weak, and informative for the reason given above**

`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`. Capture was cheaper (35 s and 9.76 GiB for 83 full graphs against 45 s and 9.61 GiB for 83 full plus 83 piecewise), and the effect:

| Concurrency | Baseline tok/s | FDO tok/s | Delta | Baseline TPOT | FDO TPOT |
|---|---|---|---|---|---|
| 128 | 7,177.73\* | 7,264.69 | +1.2% | 17.17 ms\* | 16.81 ms |
| 512 | 23,313.68\* | 22,544.34 | -3.3% | 20.29 ms\* | 20.76 ms |
| 1024 | 33,716.06 | 35,462.62 | +5.2% | 26.71 ms | 25.62 ms |

As argued in the elimination section, the pure-decode path was already a full-graph replay in the baseline, so this change touches only the mixed steps that carry a prefill chunk, which it makes eager instead of piecewise. Its value is diagnostic, not as an optimization.

##### **The hypothesis board**

| # | Hypothesis from the briefing | Status | Evidence |
|---|---|---|---|
| H1 | Decode is launch-bound, not byte-bound; the hyper-connection kernel count is the reason | Shape confirmed, mechanism narrowed | 99% of step time is fixed at the recipe's operating point and the GPU is idle about half the time at peak; but a full-graph decode step still costs 17 ms, so the CPU launch cost is ruled out. Device-side inter-kernel gaps, per-layer host handoffs, and DP lockstep remain |
| H2 | Engram host reads sit on the critical path | Confirmed, small | Moving the tables to HBM lowers TPOT 1.4 to 2.7% and removes the host-memory boot ceiling; the TTFT effect is unconfirmed |
| H3 | DP8 multiplies the Engram tables | **Refuted** | The image shards one table over the eight ranks, 23.6 GiB each; the real cost is placement, not capacity |
| H4 | The MoE backend `auto` picks matters | Not tested | The log names `FLASHINFER_TRTLLM_MXFP4_MXFP8`; no alternative backend was run |
| H5 | DSpark helps at low concurrency and hurts at high | Not tested | DSpark stayed off in every run |
| H6 | The layer-20 indexer scan is the context-scaling cost | Not tested at length | Every request here holds at most 2048 tokens; the `--max-model-len` result hints that context-sized metadata matters even at short context |
| H7 | The admission defaults are too generous | Untested as such | The machine was never starved by `--max-num-seqs`; at 256 sequences per rank it is still climbing |
| new | Expert load imbalance | **Dead** | 0.8 percentage points of utilization spread across 8 ranks at concurrency 2048 |
| new | CUDA graph mode matters unusually much | **Refuted** | The baseline already replays 83 full decode graphs; `FULL_DECODE_ONLY` changes nothing on the decode path |
| new | CPU allocation gates the step | **Open, leading** | The one variable that separates device-side gaps from host-side handoffs, and the cheapest experiment left |

**References**
- vLLM source, as of 2026-09-11: [`vllm/models/deepseek_v4_1/common/engram.py`](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4_1/common/engram.py), [`vllm/config/engram.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/engram.py)
- [vLLM CUDA graphs design doc](https://docs.vllm.ai/en/latest/design/cuda_graphs/); [PR #56512](https://github.com/vllm-project/vllm/pull/56512), which adds asynchronous Engram prefetch
- Server logs, harness JSON and GPU traces, runs `20260911-230548` and `20260911-235937`

---

#### **What Is Solid and What Is Not**

##### **Solid**

- **The baseline number, 1,451.50 tokens per second.** The recipe's command, the recipe's harness, one documented deviation to make it boot, every request generating exactly 1024 tokens.
- **The saturation curve and the 33.2x gap.** Little's law checks to within 0.7% on every clean row, and the effect is thirty-fold, far outside any noise band.
- **The fixed-cost decomposition.** It rests on TPOT being flat across an eight-fold change in batch, with p99 TPOT within 1% of the mean.
- **The elimination of compute, power, expert imbalance, and CPU launch overhead.** All from utilization and power telemetry and from the two-pass capture in the logs, not from inference.

##### **Not solid: four specific holes**

1. **No repeatability estimate.** No configuration has been run twice at the same concurrency. Every configuration effect measured, +1.0% to +6.3% and -2.9% to -3.3%, sits inside a band that run-to-run variance could plausibly explain. The driver now accepts a `512x3` syntax so the next booking measures variance directly.
2. **A warmup that did not converge.** Session 5's baseline warmup returned **293.92 tokens per second** where every other warmup in the project returned 410 to 451, with a mean TTFT of 1,338 ms against the usual 130 to 140 ms and a p90 of 5,001 ms: a few requests stalled for five seconds. The concurrency-1024 baseline ran immediately after it. If that baseline still carried lazy-initialization cost, it would inflate both configurations' gains at 1024 by roughly the +5 to +6% each showed. There is a pattern consistent with exactly this: two mechanically unrelated changes landed within one percentage point of each other at 1024 (+6.3% and +5.2%) and within half a point at 512 (-2.9% and -3.3%). "The comparison point is wrong" fits that better than "both changes are right."
3. **Cross-session comparisons.** The concurrency-128 and -512 configuration rows compare against baselines from a different container on a possibly different host. Each such comparison is worth about three percentage points of doubt.
4. **The concurrency-64 baseline TTFT is anomalous**, bimodal with a p90 of 2.9 s, so the largest TTFT effect claimed for Engram in HBM rests on a bad comparison point.

None of this touches the fixed-cost decomposition or the elimination argument. Those rest on effects of thirty-fold and on telemetry, not on 5% deltas.

##### **What to run next, in order**

1. **Variance and warmup, one booking.** Baseline at concurrency 512 and 1024, three runs each, warming until two consecutive warmups agree before measuring. This produces the error bar every configuration claim currently lacks and re-measures the suspect concurrency-1024 baseline. Nothing else should be reported until it exists.
2. **CPU allocation.** 8 cores (Modal's default for an 8-GPU container), 32 (this post), and 64 or more, at fixed concurrency. Section by section this is the untested variable, and it discriminates between device-side gaps and host-side handoffs.
3. **A timeline.** Nsight Systems on one engine core at concurrency 8, which is where the step is almost pure toll. Count kernels per step, measure the gaps, and see what fraction of them sit at the attention breaks, the Engram gathers, and the DP all-reduce. This turns 17 ms of something into named functions.
4. **Stacking, if 1 and 2 justify it.** Engram in HBM plus `--max-model-len 4096` together; their mechanisms are independent.
5. **Then the remaining knobs**: `--all2all-backend deepep_low_latency` for the decode all-to-all, `--language-model-only`, `--max-num-batched-tokens`, and the MoE backend sweep the briefing proposed.

**References**
- Harness JSON, runs `20260911-230548` and `20260911-235937`

---

#### **Takeaways**

- **Read TPOT as step time, and read a flat TPOT as a fixed toll.** Under continuous batching each decoding sequence gets one token per step, so the per-token gap is the step. Eight times the batch, the same 17 ms: the step is not doing batch-proportional work.
- **Check Little's law before believing a throughput number.** Tokens per second equals $$1024 C / \text{E2EL}$$ for a closed loop at steady state. It held to 0.7% on every clean run and failed by 21% on the recipe's 100-prompt command, which is exactly the partial-wave effect. The "decode efficiency" decline is just TTFT's share of a request's life growing.
- **The recipe measures the client, not the server.** `--max-concurrency 32` is a client cap. The same server delivers 33 times more at concurrency 2048, and the curve has not flattened.
- **The published command does not boot on this node**, twice: the KV cache is sized before the kernel warmup allocates its workspaces, and the pinned Engram tables plus the checkpoint's page cache exceed a 400 GiB host. `--gpu-memory-utilization 0.78` and a 768 GiB host request are the minimum changes, and neither is a win.
- **The toll is not compute, not power, not expert imbalance, and not CPU launch overhead.** The GPUs are idle about half the time at peak with power at half the cap and a 0.8 point spread across ranks, and a decode step that is already a single full-graph replay still costs 17 ms. What is left is on the device timeline between kernels, at per-layer host handoffs, or in the eight-rank lockstep; the CPU allocation experiment separates them.
- **Engram in HBM is a small, free, real win** on this workload: 1.4 to 2.7% lower TPOT and a host-memory failure mode deleted. Everything else measured is inside the noise until the variance runs exist.
- **Two capture passes, not one.** vLLM captures a throwaway subset of graphs during memory profiling and the real set after the kernel warmup. Reading the first pass as the graph inventory is how the CUDA graph experiment got misread.

---

#### **Test Yourself**

Try each from memory before reading the answer.

**1. Why is mean TPOT a measurement of the engine's decode step time?**
Under continuous batching every decoding sequence receives exactly one token per step, so the gap between its consecutive tokens is the duration of the step that produced the later one. TPOT averages those gaps over the request's 1,023 output gaps.

**2. Write the identity that links output tokens per second, client concurrency, and end-to-end latency for this harness, and say what assumption it needs.**
$$\text{tok/s} = 1024 C / \text{E2EL}$$, from Little's law with $$L = C$$; it needs the closed loop to hold $$C$$ requests in flight for the whole run, which a whole number of waves gives and 3.125 waves does not.

**3. The "decode efficiency" ratio of measured throughput to $$C/\text{TPOT}$$ falls from 0.99 to 0.85 as concurrency rises. What is it, exactly?**
$$1 - \text{TTFT}/\text{E2EL}$$, the fraction of a request's life spent decoding rather than waiting for its first token. It falls because TTFT grows with queueing, not because the engine gets less efficient.

**4. Why does the recipe's own bench command read 21% below the same server at the same concurrency?**
100 prompts at concurrency 32 is 3.125 waves. The first wave ramps up and the last has only 4 requests in flight, so the system is below concurrency 32 for a large share of the wall clock the throughput divides by. 128 prompts is 4 full waves and reads 1,849 instead of 1,452.

**5. In what order does vLLM size the KV cache, capture graphs, and warm up kernels, and why did the published command OOM?**
Profile a dummy forward and a throwaway graph subset, size the KV cache from that snapshot, allocate it, then run the kernel warmup, then the real capture. The FlashInfer FP4 MoE autotune inside the warmup allocates tens of GiB of workspaces that the snapshot never saw, and the KV cache had already taken the space.

**6. The baseline at concurrency 64 uses a step of about 17 ms, and the byte floor for that batch is about 4 ms. What does that gap say, and what does it not say?**
It says the step is not bound by weight bandwidth. It does not say what the other 13 ms is; that takes the elimination argument.

**7. GPU utilization reads 51% at concurrency 2048 while power is half the cap. Which of the four utilization-and-power cases is this, and what does it rule out?**
Low busy fraction, low power: the GPU is waiting on something else about half the time. It rules out compute and power limits. Because collectives run as kernels and count as busy, it also says the idle is not the all-to-all itself.

**8. Why did `FULL_DECODE_ONLY` change nothing on the decode path, and what does that establish?**
The baseline's real capture pass already produced 83 full decode graphs; the "2 full graphs" in the log came from the throwaway profiling pass. Both configurations replay one full graph per decode step, and that step still costs 17 ms, so the toll is not CPU launch overhead.

**9. Name the three surviving candidates for the fixed toll and the one experiment that separates them.**
Device-side gaps between many short kernels; per-layer host handoffs (breakable-graph attention breaks, host Engram gathers); and the eight-rank lockstep (the per-step DP all-reduce and the all-to-all waiting on the slowest rank). Sweeping the container's CPU count separates device-side from host-side.

**10. Why does moving the Engram tables into HBM lower TPOT by only about 2% but make a plausible difference to TTFT?**
Decode gathers 24 rows of 256 bytes per module per token per sequence, and a host round trip for that is small next to a 17 ms step. Prefill gathers rows for 1024 prompt tokens at once and pays real PCIe traffic. The measured TTFT effect is unconfirmed because its comparison point is an anomalous run.

---

#### **Wrapping up**

The thing to keep from this post is a method rather than a number. Every conclusion came from one of three moves: define the metric from the harness source so its physical meaning is unambiguous; check an identity, Little's law or the fixed-plus-marginal model, against the data before interpreting it; and eliminate candidates with telemetry that measures the candidate directly, busy fraction for compute, power for power, per-rank spread for imbalance, the capture log for graphs. Where the argument got something wrong, the CUDA graph inventory, it was because I read a log line without asking which phase of startup emitted it, and the fix was to go back to the source and find the two passes.

The result is a clear shape and an unfinished attribution. The node runs at 3% of its capacity where the recipe measures it, the decode step is a 17 ms toll that batch size does not touch, and the toll is not where the first hypothesis put it. The next booking is a variance estimate, a CPU sweep, and a timeline, in that order, and I would rather report that than a 6% improvement I cannot distinguish from noise.

If you find a mistake anywhere in here, please let me know and I'll fix it.
