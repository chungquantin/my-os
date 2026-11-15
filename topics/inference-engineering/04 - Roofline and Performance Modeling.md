---
title: Roofline and Performance Modeling
tags: [inference-engineering, roofline, performance-modeling, mfu, mbu, back-of-envelope, foundations]
---

# Roofline and Performance Modeling

The most valuable skill in inference engineering is predicting how fast something *should* run before you run it. If you can say "this decode step should take 4.2 ms" and the profiler says 11 ms, you know there are 7 ms of waste to find — and roughly where to look. If the profiler says 4.6 ms, you know further kernel work is pointless and the only remaining levers are algorithmic.

This note teaches that skill: the **roofline model**, **arithmetic intensity** of the operations in a transformer, the **batch-size crossover** that governs decode, **latency lower bounds**, **MFU and MBU**, and a complete back-of-envelope worksheet with worked examples.

## The roofline model

Any kernel does two things: arithmetic (FLOPs) and data movement (bytes between HBM and the chip). Each has a hardware ceiling:

```
time_compute = FLOPs / peak_FLOPs_per_s
time_memory  = bytes / peak_bytes_per_s
time ≥ max(time_compute, time_memory)         (perfect overlap)
time ≈ time_compute + time_memory              (no overlap — pessimistic)
```

Well-written kernels overlap the two, so the max is the right first approximation.

Define **arithmetic intensity** `I = FLOPs / bytes`. Then attainable performance is:

```
attainable FLOP/s = min( peak_FLOPs,  I × peak_bandwidth )
```

Plotting that on log-log axes gives the "roofline": a sloped line (memory-bound region) meeting a flat line (compute-bound region) at the **ridge point**:

```
I_ridge = peak_FLOPs / peak_bandwidth          [FLOP / byte]
```

```
 attainable
 TFLOP/s   (log)
   ^
989|                        ______________________  compute roof (BF16 H100)
   |                      /
   |                    /     prefill GEMM (I ~ 1000s) lives here →
   |                  /
   |                /
   |              /  ← memory roof: slope = 3.35 TB/s
   |            /
   |   decode /  (I ≈ batch size)
   |   GEMV  /
   |        /
   +-------+--------+--------------------------->  arithmetic intensity (log)
           1       295 = ridge                        FLOP/byte
```

| Chip (dense) | Ridge BF16 | Ridge FP8 |
| --- | --- | --- |
| A100 SXM | 312 / 2.0 ≈ 156 | — |
| H100 SXM | 989 / 3.35 ≈ 295 | 1,979 / 3.35 ≈ 591 |
| H200 | 989 / 4.8 ≈ 206 | ≈ 412 |
| B200 | 2,250 / 8 ≈ 281 | 4,500 / 8 ≈ 562 |
| MI300X | 1,307 / 5.3 ≈ 247 | ≈ 493 |
| TPU v6e | 918 / 1.64 ≈ 560 | — |

Interpretation: **an operation whose intensity is below the ridge cannot use the chip's compute, no matter how good the kernel is.** It is memory-bound and its time is `bytes / bandwidth`. Above the ridge, it is compute-bound and its time is `FLOPs / peak`.

The roofline is a model, not a law. Real kernels sit below both roofs because of imperfect overlap, launch latency, tail effects, and caches. But it tells you *which* roof you are under — the most important question.

## Arithmetic intensity of transformer operations

### GEMM

Multiplying `A (M × K)` by `B (K × N)` into `C (M × N)`, with `b` bytes per element:

```
FLOPs = 2·M·K·N
bytes ≈ b·(M·K + K·N + M·N)            (each matrix read/written once)
I     = 2·M·K·N / (b·(M·K + K·N + M·N))
```

For a linear layer, `B` is the weight (`K = d_in`, `N = d_out`) and `M` is the number of tokens in the batch. When `M ≪ K, N` — the decode case — the weight term dominates bytes:

```
I ≈ 2·M·K·N / (b·K·N) = 2M / b
```

- BF16 weights (`b = 2`): `I ≈ M` = number of tokens in the batch.
- FP8 weights (`b = 1`): `I ≈ 2M`.
- INT4/FP4 weights (`b = 0.5`): `I ≈ 4M`.

For prefill, `M = B × prompt_len` is thousands, so `I` is in the thousands — solidly compute-bound.

### GEMV (batch 1 decode)

`M = 1`: `I ≈ 1` FLOP/byte in BF16. Two orders of magnitude below any modern ridge. A batch-1 decode step on an H100 uses well under 1% of the tensor cores.

### Decode attention

For one new query token attending to `T` cached tokens in one layer, with `H` query heads, `H_kv` KV heads, head dim `d_h`:

```
FLOPs = 2 (q·Kᵀ) · H·d_h·T  +  2 (p·V) · H·d_h·T  = 4·H·d_h·T
bytes ≈ 2 (K and V) · H_kv·d_h·T · b
I     = 4·H·d_h·T / (2·H_kv·d_h·T·b) = 2·(H / H_kv) / b
```

Llama-3-70B (`H/H_kv = 8`), BF16 KV: `I = 2·8/2 = 8` FLOP/byte. FP8 KV: 16. **Decode attention is always memory-bound and batching does not help it**: every sequence has its own KV cache, so adding sequences adds bytes in proportion to FLOPs. Only GQA/MQA/MLA (more query heads per KV byte), KV quantization, and multi-token approaches (speculative decoding verifying several tokens against the same KV) raise it.

MLA (DeepSeek-V2/V3) is notable here: queries attend against a shared compressed latent, so the effective query-heads-per-KV-byte ratio is very high, pushing decode attention intensity into the hundreds — a key reason MLA kernels like FlashMLA can be compute-bound. See [[06 - Attention Kernels]].

### Prefill attention

With `S = T = P` (prompt length) and FlashAttention (no materialized score matrix), FLOPs scale as `P²` while bytes scale as `P` — intensity grows with `P` and becomes compute-bound for any real prompt.

### Elementwise ops (norms, activations, residuals, RoPE)

A few FLOPs per element, `2b` bytes per element read+written: `I ≈ 0.5–2`. Pure bandwidth. Unfused, each one is a full round trip through HBM — which is why **kernel fusion** matters: fusing RMSNorm into the following GEMM or the residual add into the previous one deletes those round trips. See [[13 - Kernels, Compilers, and CUDA Graphs]].

### Summary table

| Operation | Intensity (approx.) | Regime on H100 (ridge ≈ 295 BF16) |
| --- | --- | --- |
| Prefill linear layers, P = 2k | ~thousands | Compute-bound |
| Prefill attention, P = 2k | ~hundreds–thousands | Compute-bound |
| Decode linear layers, batch B, BF16 weights | ≈ B | Memory-bound for B < ~295 |
| Decode linear layers, FP8 weights + FP8 compute | ≈ 2B | Memory-bound for B < ~295 (ridge FP8 ≈ 591) |
| Decode attention (GQA 8, BF16 KV) | ≈ 8 | Always memory-bound |
| Norms, activations, RoPE | ≈ 1 | Always memory-bound |
| All-reduce, all-to-all | N/A — bound by link bandwidth/latency | Network-bound |

Note a neat coincidence in the FP8 row: on H100, FP8 doubles both the intensity (half the bytes) *and* the ridge (twice the FLOPs), so the crossover batch stays ~295. FP8 still halves decode time at small batch — because time is `bytes / bandwidth` and bytes halved.

## The decode batch-size crossover

A decode step for batch `B` at average context `T` reads the weights once and each sequence's KV cache once:

```
bytes(B) = W + B · T · k_tok           W = weight bytes, k_tok = KV bytes per token (all layers)
FLOPs(B) = 2·N_active·B + 4·L·d·T·B
time(B)  ≈ max( bytes(B) / BW,  FLOPs(B) / F )
```

Ignoring KV for a moment (short context), decode becomes compute-bound when `2N·B / F = W / BW`. With `W = N·b`:

```
B* = (F / BW) · (b / 2) = I_ridge · b / 2
```

H100, BF16: `B* ≈ 295 × 1 ≈ 295`. H100, FP8: `B* ≈ 591 × 0.5 ≈ 295`.

**Below `B*`, adding sequences to the batch is almost free in step time** — you are paying for the weight read anyway. This is the economic engine of LLM serving: per-token cost falls nearly linearly with batch size until you approach `B*` or run out of KV memory.

Now add the KV term. Solving `(W + B·T·k_tok)/BW = 2N·B/F`:

```
B* = (W / BW) / ( 2N/F − T·k_tok/BW )
```

The denominator compares, per sequence, the compute time for its weight FLOPs (`2N/F`) to the time to read its KV cache (`T·k_tok/BW`). **If reading one sequence's KV takes longer than computing its weight FLOPs, the denominator is negative and decode never becomes compute-bound** — each added sequence adds more memory time than compute time. This is the regime of long-context decode, and it is why KV size reduction (GQA, MLA, FP8 KV, sliding window) is so valuable. See [[05 - KV Cache]] and [[16 - Long Context Inference]].

### Worked: Llama-3-70B FP8, 8×H100, TP=8

Aggregate: `F = 8 × 1,979 ≈ 15.8 PFLOP/s` (FP8), `BW = 8 × 3.35 = 26.8 TB/s`, `W = 70.6 GB`, `k_tok = 320 KiB` (BF16 KV).

```
W / BW            = 70.6e9 / 26.8e12          = 2.63 ms
2N / F            = 141.1e9 / 15.8e15          = 8.9 µs per sequence
KV read, T = 512  = 512 · 327,680 / 26.8e12    = 6.3 µs per sequence
KV read, T = 2048 = 2048 · 327,680 / 26.8e12   = 25.0 µs per sequence

No KV:    B* = 2.63 ms / 8.9 µs         ≈ 295
T = 512:  B* = 2.63 ms / (8.9 − 6.3) µs ≈ 1,000   (and KV capacity runs out first)
T = 2048: denominator < 0  → never compute-bound
```

With FP8 KV (`k_tok = 160 KiB`), the `T = 2048` case gives `8.9 − 12.5 < 0` — still never compute-bound. Decode on dense models with realistic contexts is a bandwidth problem all the way up.

## Latency lower bounds

The minimum time for one decode step (batch 1, short context) is the weight read:

```
TPOT_min ≈ W / (BW_per_GPU × n_GPUs)
```

| Model | Precision | Hardware | W | TPOT floor | Max tok/s per user |
| --- | --- | --- | --- | --- | --- |
| Llama-3-8B | BF16 | 1× H100 | 16 GB | 4.8 ms | ~210 |
| Llama-3-8B | FP8 | 1× H100 | 8 GB | 2.4 ms | ~420 |
| Llama-3-70B | FP8 | 8× H100 TP8 | 70.6 GB | 2.6 ms | ~380 |
| Llama-3-70B | FP8 | 1× B200 | 70.6 GB | 8.8 ms | ~113 |
| Llama-3-70B | BF16 | 8× H200 TP8 | 141 GB | 3.7 ms | ~270 |

Real systems reach 50–85% of these floors after accounting for achieved bandwidth (typically 80–90% of peak for good streaming kernels), communication, attention, and overheads. Beating the floor requires changing the problem: fewer bytes (quantization, MoE, smaller/distilled model), more tokens per byte (speculative decoding — see [[09 - Speculative Decoding]]), or more bandwidth.

For **MoE**, `W` per step is the bytes of *activated* experts across the batch, not the total. At batch 1, DeepSeek-V3 (37B active of 671B) reads ~37 GB in FP8 per step; at large batch, nearly every expert is touched by some token and the step reads nearly all 671 GB (spread over many GPUs with expert parallelism). See [[11 - Mixture of Experts Inference]].

## MFU and MBU

Two utilization metrics tell you how close you are to the roofs.

```
MFU (model FLOPs utilization)    = achieved model FLOP/s / peak FLOP/s
MBU (model bandwidth utilization) = achieved model bytes/s / peak bandwidth
```

"Model" FLOPs/bytes means the *necessary* work by the formulas above (2N per token plus attention; weights plus KV per step) — not whatever the hardware counters report, which include recomputation and waste. MFU was popularized by the PaLM paper for training; MBU was popularized for inference by Databricks/MosaicML.

- **Prefill** should be judged by **MFU**. Good: 40–70%.
- **Decode** should be judged by **MBU**. Good: 60–85% at moderate batch.

### Worked: computing MBU from a measurement

Llama-3-70B FP8, 8×H100, batch 64, average context 2,048, measured TPOT = 7.5 ms.

```
bytes per step = W + B·T·k_tok = 70.6 GB + 64·2048·327,680 B = 70.6 + 42.9 = 113.5 GB
achieved BW    = 113.5 GB / 7.5 ms = 15.1 TB/s
MBU            = 15.1 / 26.8 = 56%
```

56% says there is real headroom. Where does it go? Next section.

### Worked: computing MFU for prefill

Same system, one 2,048-token prompt measured at TTFT (excluding queue) = 40 ms.

```
FLOPs  ≈ 2·70.6e9·2048 + 2·80·8192·2048² ≈ 289.2 + 5.5 = 294.7 TFLOP
peak   = 15.8 PFLOP/s
MFU    = 294.7 / (15.8e15 × 0.040) = 294.7 / 632 = 47%
```

## Overheads and Amdahl's law

The roofline accounts for one kernel. A decode step is hundreds of kernels plus communication plus host work. Amdahl's law applies: if 30% of the step is non-roofline overhead, a perfect GEMM speedup can at best reduce step time by 70%.

Typical sources of the gap between roofline and reality:

| Overhead | Typical size | Fix |
| --- | --- | --- |
| **Kernel launch latency** | ~3–5 µs per launch; an 80-layer model with ~10–15 kernels per layer → 800–1,200 launches → **3–6 ms** per step | CUDA graphs; fusion; persistent kernels |
| **TP all-reduce latency** | 2 per layer; for small decode messages, latency (~5–20 µs each) dominates bandwidth → 160 × ~10 µs ≈ **1.6 ms** | Fewer TP ranks; fused all-reduce + norm kernels; one-shot/two-shot all-reduce; overlap with compute |
| **Unfused elementwise ops** | Each a full HBM round trip of the activations | Fusion (norm + quant, residual + norm, SwiGLU into GEMM epilogue) |
| **Tile / wave quantization** | GEMMs whose tile count is not a multiple of SM count leave SMs idle in the last wave | Split-K, stream-K, better tile shapes |
| **Host scheduling and sampling** | Python scheduler, `.item()` syncs, detokenization | Async scheduling overlapping CPU with GPU; GPU-side sampling |
| **Padding** | Static batching pads to the longest sequence | Continuous batching, ragged/varlen kernels |
| **Achieved vs peak bandwidth** | Even ideal streaming reads hit ~85–92% of peak | Nothing — budget for it |

### Communication roofline

Ring all-reduce of `m` bytes across `p` GPUs, per-GPU link bandwidth `β` (one direction), per-hop latency `α`:

```
t_allreduce ≈ 2(p−1)·α + 2·(p−1)/p · m / β
```

For decode, `m = B · d · b_act` is tiny: batch 64, `d = 8192`, BF16 → 1 MiB. On NVLink4 (`β ≈ 450 GB/s`): bandwidth term = 1.75 × 1 MiB / 450 GB/s ≈ 4 µs. The latency term (14α with α of a few µs in a naive ring) dominates — which is why NCCL's tree/NVLS algorithms and custom one-shot all-reduce kernels exist for inference. See [[10 - Parallelism]].

## The back-of-envelope worksheet

Use this template for any model + hardware + workload. Fill every line before benchmarking.

```
INPUTS
  Model: N_total, N_active, L, d, H, H_kv, d_h, V
  Precision: weight bytes b_w, KV bytes b_kv, compute format
  Hardware: n_GPUs, F (dense, chosen precision), BW (HBM), HBM capacity, link β
  Workload: prompt length P, output length O, target batch B, SLOs

DERIVED
  1. W         = N_total · b_w                                (weights in memory)
  2. k_tok     = 2·L·H_kv·d_h·b_kv                            (KV per token)
  3. KV budget = n_GPUs·capacity·0.9 − W − activations        (bytes available for KV)
  4. Max concurrent seqs = KV budget / ((P + O)·k_tok)
  5. Prefill time  ≈ (2·N_active·P + 2·L·d·P²) / (F_total · MFU_prefill)
  6. Decode step   ≈ (W_active(B) + B·T_avg·k_tok) / (BW_total · MBU) + comm + overhead
  7. TPOT          = decode step time   (per user tok/s = 1/TPOT)
  8. Throughput    = B / decode step time   (output tokens/s per replica)
  9. E2E           ≈ queue + prefill + O · TPOT
 10. $/M output tokens = ($/hour per replica) / (throughput · 3600) · 1e6
CHECK
  - Is B within KV capacity (line 4)?  Is TPOT within SLO?  Is TTFT within SLO?
  - Is decode memory-bound (B < B*)?  (it almost always is)
```

### Full worked example: Llama-3-70B FP8 chat on 8×H100

Inputs: FP8 weights and FP8 compute, BF16 KV. 8×H100 SXM, TP=8. Workload: P = 1,500, O = 500, so average decode context `T_avg ≈ 1,500 + 250 = 1,750`. SLO: p99 TPOT ≤ 40 ms, p99 TTFT ≤ 1 s. Price $2.50/GPU-hour.

```
1. W         = 70.6 GB
2. k_tok     = 2·80·8·128·2 = 327,680 B (320 KiB)
3. KV budget = 8·80 GB·0.9 − 70.6 − ~16 (activations, CUDA graphs, workspace) ≈ 489 GB
4. Max seqs  = 489e9 / (2000 · 327,680) ≈ 746
5. Prefill   ≈ (2·70.6e9·1500 + 2·80·8192·1500²) / (15.8e15 · 0.5)
             = (211.8 + 2.9) TFLOP / 7.9 PFLOP/s ≈ 27 ms
6. Decode, try B = 256:
     bytes   = 70.6 GB + 256·1750·327,680 B = 70.6 + 146.8 = 217.4 GB
     memory  = 217.4 GB / (26.8 TB/s · 0.75 MBU) ≈ 10.8 ms
     compute = (2·70.6e9·256 + 4·80·8192·1750·256) / 15.8e15 ≈ (36.1 + 1.2) TFLOP / 15.8 PFLOP/s ≈ 2.4 ms
     → memory-bound; add ~1.5 ms comm + ~0.5 ms overhead (with CUDA graphs) ≈ 12.8 ms
7. TPOT      ≈ 12.8 ms  → 78 tok/s per user  (well within 40 ms SLO)
8. Throughput = 256 / 12.8 ms = 20,000 output tok/s per node = 2,500 per GPU
9. E2E       ≈ queue + 27 ms + 500 · 12.8 ms ≈ 6.4 s + queue
10. $/M out  = (8 · 2.50) / (20,000 · 3600) · 1e6 ≈ $0.28
```

But continuous batching mixes prefills into decode steps: each new request's 1,500-token prefill (27 ms of compute) interrupts or extends decode steps. At 20,000 output tok/s with O = 500, the node completes 40 requests/s, which means 40 × 27 ms ≈ 1.08 s of prefill compute per second of wall time. **Prefill alone would consume more than the whole node.** The decode-only estimate was too optimistic; we must budget both phases:

```
per second: prefill 40 req × 27 ms = 1.08 s     ← infeasible at B = 256
reduce rate: let r = requests/s, then
   r · 27 ms (prefill) + (r · 500 / B) · 12.8 ms (decode steps) ≤ 1 s
   with B = 256: r · (0.027 + 0.025) ≤ 1  →  r ≈ 19 req/s ≈ 9,600 output tok/s per node
```

So the realistic throughput is ~1,200 output tok/s per GPU (~$0.58/M output tokens) — about half the decode-only number. This is precisely the tension that chunked prefill ([[07 - Batching and Scheduling]]), prefix caching ([[05 - KV Cache]]), and disaggregation ([[12 - Disaggregated Serving]]) address. Doing this arithmetic *before* deploying would have caught the 2× error.

### Second example: is a single B200 enough for Llama-3-8B at 50 ms TPOT?

```
W = 16 GB (BF16), k_tok = 128 KiB, T_avg = 4,000, B = 128
bytes = 16 GB + 128·4000·131,072 = 16 + 67.1 = 83.1 GB
time  = 83.1 GB / (8 TB/s · 0.75) ≈ 13.9 ms  → TPOT ≈ 15 ms with overhead
```

Comfortably within 50 ms. Notice KV reads are 80% of the bytes — at this context, KV quantization to FP8 would save more time than weight quantization.

## How practitioners use this

1. **Before a deployment**: fill the worksheet; sanity-check the configuration space; reject options that cannot meet SLOs even at the roofline.
2. **During profiling**: compute MBU/MFU per kernel class. Kernels far from their roof are optimization targets; kernels at their roof are not ([[18 - Benchmarking and Profiling]]).
3. **When evaluating claims**: a paper or vendor claims 3× decode speedup — check whether the baseline was already near its bandwidth roof. If it was at 80% MBU, a 3× speedup must come from reading fewer bytes or verifying more tokens per read, not from "better kernels."
4. **For hardware selection**: decode-heavy → bandwidth/$; prefill-heavy → FLOPs/$; long context → capacity and bandwidth ([[03 - Accelerator Hardware]], [[19 - Cost and Capacity Planning]]).

## Common mistakes

- **Using sparse peak FLOPs.** Your ridge point will be off by 2×.
- **Forgetting KV bytes in decode.** At realistic contexts KV reads can exceed weight reads.
- **Treating decode attention like a GEMM.** Batching does not raise its intensity.
- **Estimating throughput from decode alone.** Prefill steals capacity; budget both phases.
- **Ignoring the host and launch overhead.** Small models and short steps are dominated by them.
- **Assuming 100% of peak.** Use 50–70% MFU for prefill and 70–85% MBU for decode as planning numbers, then measure.
- **Over-trusting the model.** The roofline predicts lower bounds. When reality is 3× slower, measure; do not argue.

## Check yourself

1. What is the arithmetic intensity of a decode linear layer with batch 32 and FP8 weights? Is it memory- or compute-bound on an H100?
2. Why does batching not improve the arithmetic intensity of decode attention?
3. Derive the decode crossover batch `B*` ignoring KV, in terms of the ridge point and bytes per weight.
4. A Llama-3-8B BF16 deployment on one H100 measures 9 ms TPOT at batch 1. What is its MBU, and what would you look for?
5. Why is MFU the wrong metric for decode?
6. For a ring all-reduce of 1 MiB across 8 GPUs on NVLink, why does latency dominate?
7. In the 70B worked example, why was decode-only throughput off by ~2×?

<details>
<summary>Answers</summary>

1. `I ≈ 2·32 = 64` FLOP/byte; ridge FP8 ≈ 591 → memory-bound.
2. Every sequence has its own KV cache: adding a sequence adds proportional FLOPs and bytes, so the ratio stays at `2(H/H_kv)/b`.
3. `2N·B/F = N·b/BW` → `B* = (F/BW)·(b/2) = I_ridge · b/2`.
4. Floor = 16 GB / 3.35 TB/s = 4.8 ms → MBU ≈ 53%. Look for launch overhead (CUDA graphs on?), unfused elementwise kernels, host syncs, sampling cost, LM head efficiency.
5. Decode is bandwidth-bound by construction; MFU will be low (a few percent at small batch) even for a perfect implementation. MBU measures closeness to the binding roof.
6. The message is tiny, so the bandwidth term is ~4 µs, while the latency term is `2(p−1)α` = 14 hops × a few µs.
7. Prefill compute for the arriving requests (27 ms each at 40 req/s) exceeded the node's capacity; real throughput must split time between prefill and decode.

</details>

## Further reading

- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM 52(4), 2009 — the original roofline paper.
- Pope et al., *Efficiently Scaling Transformer Inference* (2022) — https://arxiv.org/abs/2211.05102 — latency/cost Pareto analysis, partitioning, MFU for inference.
- Chowdhery et al., *PaLM* (2022) — https://arxiv.org/abs/2204.02311 — defines MFU.
- Databricks, *LLM Inference Performance Engineering: Best Practices* — https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices — defines MBU, decode bandwidth analysis.
- kipply, *Transformer Inference Arithmetic* — https://kipp.ly/transformer-inference-arithmetic/
- JAX team, *How to Scale Your Model*: "Rooflines" — https://jax-ml.github.io/scaling-book/roofline/ and "Inference" — https://jax-ml.github.io/scaling-book/inference/
- Horace He, *Making Deep Learning Go Brrrr From First Principles* — https://horace.io/brrr_intro.html — compute vs memory vs overhead.
- NVIDIA, *GPU Performance Background User's Guide* — https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html — tile and wave quantization.

## Related

- [[02 - Transformer Inference Anatomy]]
- [[03 - Accelerator Hardware]]
- [[05 - KV Cache]]
- [[07 - Batching and Scheduling]]
- [[10 - Parallelism]]
- [[18 - Benchmarking and Profiling]]
- [[19 - Cost and Capacity Planning]]
- [[98 - Formula Cheatsheet]]
