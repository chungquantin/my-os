---
title: Cost and Capacity Planning
tags: [inference-engineering, cost, capacity-planning, tco, pricing, gpu-economics]
---

# Cost and Capacity Planning

Inference is a manufacturing business: GPUs are the factory, tokens are the product, and the unit economics are **dollars per million tokens at a given quality and latency**. Training is a one-time capital project; inference is the recurring bill, and for any successful product it ends up larger. This note turns benchmark numbers ([[18 - Benchmarking and Profiling]]) into fleet sizes and prices.

## 1. The core identity

```
$ / 1M tokens = (GPU $/hour × number of GPUs) / (tokens per hour) × 10⁶
              = (GPU $/hour) / (tokens/s per GPU × 3600) × 10⁶
```

Everything else is figuring out the honest value of **tokens/s per GPU** — at your SLO, at your traffic mix, at your real utilization.

**Worked example (decode-dominated chat).**

- GPU: H100 at an assumed $2.50/hour (rental prices vary widely by provider, contract, and year; plug in your own).
- Measured: a 70B FP8 model on 4×H100 (TP4) delivers 3,200 output tok/s total at your SLO (≥ 40 tok/s/user), i.e. 800 tok/s per GPU.

```
$ per 1M output tokens = 2.50 / (800 × 3600) × 10⁶ = 2.50 / 2.88 = $0.87
```

At 100% utilization. That last clause is where most estimates go wrong.

## 2. Utilization: the hidden multiplier

Real traffic is diurnal and bursty. If your fleet is sized for peak, average utilization might be 30–50%.

```
effective $/1M tokens = peak-efficiency $/1M tokens / average utilization
```

At 40% average utilization, the $0.87 above becomes **$2.17**. Levers to raise utilization:

- **Fill valleys with batch work** (offline evals, embeddings, synthetic data, batch API at a discount). Priority scheduling lets interactive traffic preempt it.
- **Autoscale** with warm pools ([[17 - Serving Systems, Routing, and Autoscaling]]). Effective only if cold start is fast relative to traffic swings.
- **Pool many models/tenants** on shared capacity (multi-LoRA, multi-model).
- **Global load shifting** across regions and time zones.
- **Off-peak pricing**: lower prices at night to shift demand. DeepSeek publishes off-peak discounts; their deployment Q&A quotes revenue of US$38 per GPU-hour *at the off-peak price*, with one cluster serving 155 trillion tokens per day (`projects/DeepInfer/doc/inference_deployment_qa.md`).

## 3. Input vs output: why prices are asymmetric

API prices typically charge 3–5× more for output than input tokens. The physics:

- **Prefill (input)** is compute-bound: a big batched matmul, tensor cores near peak. One H100 can prefill tens of thousands of tokens per second for a mid-size model.
- **Decode (output)** is memory-bandwidth-bound: every step streams the weights and the KV cache for only one new token per sequence ([[02 - Transformer Inference Anatomy]], [[04 - Roofline and Performance Modeling]]). Throughput per GPU is 10–50× lower than prefill.

Rough per-token cost from first principles for a dense model with `N` parameters:

```
prefill FLOPs per token ≈ 2N   (+ attention, grows with context)
prefill time per token  ≈ 2N / (peak FLOP/s × MFU)

decode time per step    ≈ (weight bytes + KV bytes read) / (HBM BW × efficiency)
decode tokens per step  = batch size B
```

**Worked example: 8B model, BF16, H100 SXM (≈ 989 dense BF16 TFLOP/s, 3.35 TB/s).**

- Prefill at 50% MFU: `2 × 8e9 / (989e12 × 0.5) ≈ 32 µs/token` → ~31K tok/s per GPU.
- Decode at batch 64, 2K average context, ~16 GB weights, KV ≈ 128 KB/token (Llama-3-8B-like GQA, BF16): KV read per step = 64 × 2048 × 128 KB ≈ 16.8 GB. Step time ≈ (16 + 16.8) GB / (3.35 TB/s × 0.8) ≈ 12 ms → 64 / 0.012 ≈ 5.3K tok/s per GPU.

Output costs ~6× input here, before considering that decode also has worse latency constraints. Longer contexts make decode even more expensive (KV reads grow), which is why long-context pricing tiers exist ([[16 - Long Context Inference]]).

## 4. Cached input pricing

With prefix caching ([[05 - KV Cache]]), a cache-hit input token costs almost nothing to compute — only the storage and transfer of its KV. Providers therefore discount cached input tokens heavily (commonly 50–90% off the input price), sometimes with a separate charge for writing to the cache.

Economics of a cache:

```
savings per hit token  = prefill cost per token − (KV storage cost + KV load cost)
KV storage cost        ∝ KV bytes per token × retention time × $/byte-hour of tier
```

Small KV per token (MLA, compressed/sparse KV, FP8/FP4 KV) makes long retention on cheap tiers worthwhile. DeepSeek keeps block KV cache on ~100 PiB of SSD in a 3FS cluster, about 14.4 TB of logical storage per online GPU, with a TTL of 3 days for agent traffic and 7 days for chatbot traffic; they report that a ~3-day TTL would cost chatbot traffic only 0.5–1% of hit rate (same source). For agentic workloads where each call re-sends a 50K-token history, hit rates of 80–95% are normal, and caching is the dominant cost lever.

**Worked example.** Agent with 40K-token context per call, 300 new input tokens, 400 output tokens; input $0.50/M, cached input $0.05/M, output $2.00/M.

| | No cache | With cache (40K hit) |
| --- | --- | --- |
| Input | 40,300 × 0.50/M = $0.0202 | 300 × 0.50/M + 40,000 × 0.05/M = $0.0022 |
| Output | 400 × 2.00/M = $0.0008 | $0.0008 |
| Total | $0.0210 | $0.0030 |

7× cheaper. For agents, input caching matters more than output price.

## 5. Capacity math: sizing for QPS and SLO

### Step 1: characterize traffic

- Peak request rate `λ_peak` (req/s), plus burst factor.
- Input length distribution (mean `L_in`, p99), output length distribution (mean `L_out`).
- Prefix cache hit rate `h`.
- SLOs: TTFT p90 ≤ T₁, TPOT p90 ≤ T₂.

### Step 2: demand in tokens

```
prefill demand  = λ_peak × L_in × (1 − h)        tokens/s (non-cached)
decode demand   = λ_peak × L_out                 tokens/s
concurrency     = λ_peak × E2E ≈ λ_peak × (TTFT + L_out × TPOT)    (Little's law)
```

### Step 3: supply per replica at the SLO

From benchmarks: prefill tokens/s per replica at the TTFT SLO; decode tokens/s per replica at the TPOT SLO (the Pareto point, not the peak); max concurrent sequences per replica limited by KV memory:

```
max_concurrency_kv = KV memory / (KV bytes per token × (L_in + L_out))
```

### Step 4: replicas and headroom

```
replicas = max(prefill_demand / prefill_supply,
               decode_demand  / decode_supply,
               concurrency    / max_concurrency_kv) / target_utilization
```

Target utilization 0.6–0.8 at peak, so bursts land below the knee.

### Full worked example

Chat product, 70B dense model FP8, replicas of 4×H100 (TP4).

- `λ_peak` = 40 req/s, `L_in` = 2,000, `L_out` = 500, `h` = 0.5.
- SLO: TTFT p90 ≤ 1 s, TPOT p90 ≤ 30 ms.
- Benchmarked per replica (colocated, chunked prefill): at TPOT ≤ 30 ms, decode 3,500 tok/s with ~110 concurrent sequences; prefill capacity 15,000 tok/s while keeping decode in SLO.

Demand:

```
prefill = 40 × 2000 × 0.5 = 40,000 tok/s
decode  = 40 × 500        = 20,000 tok/s
concurrency ≈ 40 × (0.5 + 500 × 0.025) ≈ 40 × 13 = 520 sequences
```

KV check: 70B Llama-like GQA ≈ 320 KB/token in BF16 → 160 KB in FP8. Per replica, 4×80 GB = 320 GB HBM; weights ~70 GB; overhead ~30 GB → ~220 GB for KV → 220e9 / (160e3 × 2,500) ≈ 550 sequences. Not the binding constraint.

Replicas: in a colocated deployment prefill and decode share each replica's time, so add the fractions of capacity:

```
replicas_needed = 40,000/15,000 + 20,000/3,500 = 2.67 + 5.71 = 8.4 replicas of full-time work
with 70% target utilization → 8.4 / 0.7 = 12 replicas = 48 GPUs
```

Cost at $2.50/GPU-hour: 48 × 2.50 × 24 × 30 = **$86,400/month** at peak-sized static capacity. If autoscaling tracks a diurnal curve averaging 55% of peak, roughly $50K/month.

Check per-token: monthly tokens at average 55% of peak ≈ 0.55 × 40 × 2,500 × 2.6M s ≈ 143B tokens → ~$0.35–0.60 per 1M blended tokens depending on scaling. Compare to API prices before building anything.

### Disaggregated variant

With P/D disaggregation ([[12 - Disaggregated Serving]]) you size two pools separately: prefill pool from non-cached tokens/s at the TTFT SLO, decode pool from concurrency and TPOT. The P:D GPU ratio follows from the traffic mix; DeepSeek reports roughly 1:1 in practice for their workload, with a smallest efficient unit of 64 prefill + 64 decode GPUs because decode wants EP ≥ 64 (`projects/DeepInfer/doc/inference_deployment_qa.md`).

## 6. Build vs buy

| Factor | Use a hosted API | Self-host open weights |
| --- | --- | --- |
| Volume | Low to medium, spiky | High, steady |
| Model | Need frontier closed models | Open model is good enough |
| Team | No inference engineers | Have (or will hire) them |
| Data control | Acceptable under provider terms | Strict residency / privacy |
| Customization | Prompting, some fine-tuning | Full: LoRA, custom decoding, constrained output, special kernels |
| Latency | Provider's SLOs | You control (and own) them |
| Cost structure | Pure opex per token | Capex/reserved commitments; utilization risk is yours |

Break-even rule of thumb: self-hosting wins when you can keep GPUs **well utilized** and your volume is large enough to amortize the engineering team. A two-person team running 8 GPUs at 25% utilization almost always loses to an API. Third-party open-model inference providers sit in between: open weights, someone else's utilization problem.

Hidden self-host costs: engineers, on-call, observability, idle headroom, failed GPUs, upgrades, evals, security reviews, egress, storage for weights and KV.

## 7. Buying compute: on-demand, reserved, spot

| Option | Price | Risk | Fit |
| --- | --- | --- | --- |
| **On-demand** | Highest | Capacity may be unavailable when you need it | Burst, experiments |
| **Reserved / committed (1–3 yr)** | Much lower per hour | Pay even when idle; hardware ages (a 3-year H100 commitment competes with B200/B300-era prices) | Baseline load |
| **Spot / preemptible** | Lowest | Can be reclaimed with short notice | Batch inference, evals, stateless prefill workers with retry |
| **Own hardware** | Lowest at high utilization | Capex, power, cooling, staff | Very large steady load |

Pattern: reserve for the trough, on-demand or autoscaled for the daily peak, spot for batch. Spot is a poor fit for decode workers holding long-lived KV state unless you can migrate sequences.

## 8. Cost levers ranked by typical impact

1. **Prefix caching + cache-aware routing** for agentic and multi-turn traffic (often 2–7× on input cost).
2. **Utilization** (batch backfill, autoscaling).
3. **Quantization** (FP8 weights/KV: ~1.5–2× decode throughput; FP4 on Blackwell more) — [[08 - Quantization]].
4. **Right-sizing the model** (a distilled 8B fine-tune can replace a 70B for narrow tasks at 5–10× lower cost).
5. **Speculative decoding** (1.5–3× faster decode at low/medium batch) — [[09 - Speculative Decoding]].
6. **Better parallelism** (wide EP for MoE decode, avoid unnecessary TP) — [[10 - Parallelism]], [[11 - Mixture of Experts Inference]].
7. **Hardware generation** (newer GPUs often give more tokens per dollar even at a higher hourly price).
8. **Kernel work** — important at scale, but usually after the above.

## 9. Energy

At scale, power is the binding constraint more than GPU count. Useful metrics: tokens per joule, and $/1M tokens including power at your PUE. An H100 SXM has a 700 W TDP; a B200 is around 1,000 W. Rough: a node at ~10 kW producing 20K tok/s → 2,000 tokens per kJ, or 0.5 J/token.

## 10. A cost calculator you can run

Keep the model in code so assumptions are explicit and versioned:

```python
from dataclasses import dataclass

@dataclass
class Replica:
    gpus: int
    gpu_hourly_usd: float
    prefill_tok_s: float      # non-cached prompt tokens/s at the TTFT SLO
    decode_tok_s: float       # output tokens/s at the TPOT SLO (Pareto point, not peak)
    max_seqs_kv: int          # concurrency allowed by KV memory

@dataclass
class Traffic:
    peak_rps: float
    avg_over_peak: float      # diurnal average / peak, e.g. 0.55
    in_tokens: float
    out_tokens: float
    cache_hit: float          # fraction of input tokens served from prefix cache
    ttft_s: float
    tpot_s: float

def plan(r: Replica, t: Traffic, target_util: float = 0.7) -> dict:
    prefill = t.peak_rps * t.in_tokens * (1 - t.cache_hit)
    decode = t.peak_rps * t.out_tokens
    concurrency = t.peak_rps * (t.ttft_s + t.out_tokens * t.tpot_s)   # Little's law
    work = prefill / r.prefill_tok_s + decode / r.decode_tok_s         # colocated: time shares add
    replicas = max(work, concurrency / r.max_seqs_kv) / target_util
    replicas = int(-(-replicas // 1))                                  # ceil
    gpu_hours_month = replicas * r.gpus * 24 * 30
    static_cost = gpu_hours_month * r.gpu_hourly_usd
    tokens_month = t.peak_rps * t.avg_over_peak * (t.in_tokens + t.out_tokens) * 30 * 86400
    return {
        "replicas": replicas,
        "gpus": replicas * r.gpus,
        "static_monthly_usd": round(static_cost),
        "usd_per_1M_blended_static": round(static_cost / tokens_month * 1e6, 3),
    }

print(plan(Replica(4, 2.50, 15_000, 3_500, 550),
           Traffic(40, 0.55, 2_000, 500, 0.5, 0.5, 0.025)))
# {'replicas': 12, 'gpus': 48, 'static_monthly_usd': 86400, 'usd_per_1M_blended_static': 0.606}
```

The output reproduces the worked example in section 5.

### Sensitivity: what moves the answer

Holding the section-5 example fixed and changing one input at a time:

| Change | Replicas (was 12) | Why |
| --- | --- | --- |
| Cache hit 0.5 → 0.8 | 10 | Prefill demand falls from 40K to 16K tok/s |
| Output length 500 → 1,500 | 29 | Decode demand triples; decode dominates |
| TPOT SLO 25 → 50 ms (decode_tok_s 3,500 → 5,000) | 10 | Larger batches allowed per replica |
| FP8 → FP4 weights (decode_tok_s +40%) | 10 | Decode is bandwidth-bound; fewer bytes per step |
| Target utilization 0.7 → 0.85 | 10 | Less headroom; more SLO risk in bursts |

Output length and SLO strictness usually matter more than anything you can do in a kernel. That is why reasoning models (thousands of output tokens) changed inference economics so sharply.

## 11. Pricing your own API

If you sell tokens, price from effective cost upward:

```
price_per_1M = effective_cost_per_1M × (1 + margin) + risk_buffer
```

Practical points:

- Price **input, cached input, and output separately**, because their costs differ by an order of magnitude.
- Long-context tiers: attention and KV costs grow with context; charge more above a threshold or cap context per tier.
- **Batch API** at a discount (often ~50%) for work that tolerates hours of latency; it fills your valleys and raises utilization.
- Rate limits in tokens per minute are capacity reservations; price committed throughput (provisioned capacity) separately from pay-as-you-go.
- Revisit prices when hardware, models, or cache hit rates change — the cost basis moves every few months.

## What practitioners actually do

- Maintain a **cost model spreadsheet** driven by benchmarks: tokens/s per GPU at SLO for each (model, hardware, config), traffic forecast, utilization assumption.
- Measure **real utilization** and **cache hit rate** weekly; both drift.
- Price products with a margin over **effective** (not peak) cost.
- Re-run the build-vs-buy math every hardware generation and every major open-model release.

## Common mistakes

- Using peak throughput instead of SLO-constrained throughput.
- Ignoring utilization (dividing by 100% when reality is 40%).
- Using "total tokens/s" that includes cheap prefill to price expensive decode.
- Forgetting KV memory as a capacity constraint for long contexts.
- Committing to 3-year reservations on a GPU generation that will be superseded in 18 months without modeling it.
- Ignoring engineering and on-call cost in build-vs-buy.

## Check yourself

1. **A GPU costs $3/hour and delivers 1,200 output tok/s at your SLO. Cost per 1M output tokens at 50% utilization?** `3 / (1200 × 3600) × 1e6 = $0.69` at full use; ÷ 0.5 → **$1.39**.
2. **Why are output tokens priced higher than input?** Decode is memory-bandwidth-bound with one token per sequence per step; prefill processes many tokens per weight read and runs near compute peak, so it yields far more tokens per GPU-second.
3. **Use Little's law: 30 req/s, average E2E 12 s. Concurrency?** 360 concurrent sequences.
4. **When is KV memory the binding capacity constraint?** Long contexts and/or long outputs with large per-token KV, so `KV memory / (bytes per token × tokens per sequence)` caps concurrency below what compute could serve.
5. **Why does input caching matter more for agents than for chat?** Agent calls re-send long, growing histories with tiny new deltas; most input tokens are cache hits, so input cost dominates and caching collapses it.
6. **Why is spot capacity a poor fit for decode workers?** Preemption destroys in-flight KV state; without migration the requests fail or must be recomputed.
7. **Your fleet is sized at 70% target peak utilization and averages 45%. Name two ways to raise it.** Backfill with batch/offline jobs at low priority; autoscale with warm pools; off-peak pricing to shift demand.

## Further reading

- DistServe (goodput-driven sizing, P/D ratios): https://arxiv.org/abs/2401.09670
- Splitwise (phase-split hardware economics): https://arxiv.org/abs/2311.18677
- Mooncake (KV-cache-centric architecture, Kimi production): https://arxiv.org/abs/2407.00079
- DeepSeek V4.1 Flash deployment Q&A (local): `projects/DeepInfer/doc/inference_deployment_qa.md`

## Related

- [[04 - Roofline and Performance Modeling]] — where the per-token physics comes from
- [[05 - KV Cache]] — cache economics
- [[12 - Disaggregated Serving]] — sizing P and D pools
- [[17 - Serving Systems, Routing, and Autoscaling]] — utilization and autoscaling
- [[18 - Benchmarking and Profiling]] — getting honest tokens/s at SLO
- [[98 - Formula Cheatsheet]] — all formulas in one place
