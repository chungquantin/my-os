---
title: Formula Cheatsheet
tags: [inference-engineering, cheatsheet, formulas, back-of-envelope]
---

# Formula Cheatsheet

Every back-of-envelope formula from the vault in one place. Symbols follow the notation table in [[00 - Inference Engineering Hub]]. All formulas are **first-order**: they ignore overheads on purpose. Real systems typically land at 50-80% of these ceilings; if you are below ~40%, something specific is wrong and worth finding.

Reference hardware used in examples (dense, no sparsity): **H100 SXM** — 989 TFLOP/s BF16, 1,979 TFLOP/s FP8, 80 GB HBM3 at 3.35 TB/s, NVLink 900 GB/s bidirectional per GPU. Other accelerators: see [[03 - Accelerator Hardware]].

## 1. Model size and memory

| Quantity | Formula | Notes |
| --- | --- | --- |
| Dense params (classic MHA, 4× FFN) | `N ≈ 12·L·d²` | Embeddings excluded |
| Params per layer, GQA + SwiGLU | `d·(2·h·d_h + 2·h_kv·d_h) + 3·d·d_ff` | Q,O + K,V + gate/up/down |
| Weight memory | `N · b` | `b` = bytes per param |
| Group-quantized weight memory | `N · (bits + scale_bits/group) / 8` | INT4, group 128, FP16 scale → 4.125 bits/weight |
| MoE active params | `N_act = N_shared + L_moe·k·N_expert + N_attn + N_embed` | Compute scales with `N_act`, memory with `N` |

**Example.** Llama-3-70B in BF16: `70.6e9 × 2 ≈ 141 GB`. FP8: ~71 GB. INT4 g128: ~36 GB.

## 2. KV cache

| Quantity | Formula |
| --- | --- |
| KV bytes per token (MHA/GQA/MQA) | `2 · L · h_kv · d_h · b_kv` |
| KV bytes per token (MLA) | `L · (d_c + d_rope) · b_kv` |
| KV bytes per sequence | `per_token × s` |
| Tokens that fit | `(HBM_total − weights − activations − reserve) / per_token` |
| Max concurrent sequences | `tokens_that_fit / avg_context_length` |
| Paged-KV internal waste per sequence | `< block_size` tokens (avg ≈ `block_size/2`) |

**Examples (BF16 KV).**

| Model | L | h_kv × d_h | KV per token |
| --- | --- | --- | --- |
| Llama-3-8B | 32 | 8 × 128 | `2·32·8·128·2` = 131,072 B = **128 KiB** |
| Llama-3-70B | 80 | 8 × 128 | `2·80·8·128·2` = 327,680 B = **320 KiB** |
| Same 70B with full MHA (64 heads) | 80 | 64 × 128 | **2.5 MiB** — why GQA exists |
| DeepSeek-V3 (MLA, `d_c=512`, `d_rope=64`) | 61 | — | `61·576·2` = 70,272 B ≈ **69 KiB** |

A single 128K-token Llama-3-70B sequence holds `128K × 320 KiB ≈ 40 GiB` of KV.

## 3. FLOPs

| Quantity | Formula | Notes |
| --- | --- | --- |
| Forward FLOPs per token (linear layers) | `≈ 2 · N` (dense), `≈ 2 · N_act` (MoE) | One multiply + one add per weight |
| Attention FLOPs per token at context `s` | `≈ 4 · L · s · h · d_h` | `QKᵀ` + `AV` |
| Prefill FLOPs for `S` tokens | `≈ 2·N·S + 2·L·S²·h·d_h` | Causal mask halves the attention term |
| Decode FLOPs per step | `≈ B · (2N + 4·L·s̄·h·d_h)` | `s̄` = mean context |

Attention starts to matter versus linear layers when `2·s·h·d_h ≳ N/L`, i.e. roughly `s ≳ 6d` for classic shapes — e.g. ~50K tokens for `d = 8192`.

## 4. Roofline

| Quantity | Formula |
| --- | --- |
| Arithmetic intensity | `I = FLOPs / bytes moved` |
| Attainable throughput | `min(F, I · BW)` |
| Ridge point | `I* = F / BW` (H100 BF16: 989e12 / 3.35e12 ≈ **295 FLOP/B**) |
| GEMM `(M×K)·(K×N)`, intensity | `2MKN / (b·(MK + KN + MN))` |
| Decode linear-layer intensity at batch `B` | `≈ 2B / b` (weights dominate traffic) |
| Decode compute/memory crossover batch | `B* ≈ F · b / (2 · BW)` → H100: **~295** in BF16 and in FP8 |
| Decode attention intensity | `≈ 2 · (h/h_kv) / b_kv` FLOP/B — independent of `B` (each sequence reads its own KV); batching never makes decode attention compute-bound. GQA group size and MLA are what raise it |

Takeaway: below `B*`, adding sequences to the batch is nearly free for the linear layers. Attention cost scales with total KV read, so at long context attention-memory traffic, not weights, sets the step time.

## 5. Latency

| Quantity | Formula |
| --- | --- |
| Decode step, memory bound | `t_step ≥ (weight_bytes + Σ KV_bytes_read) / (BW · n_gpus)` |
| Decode step, compute bound | `t_step ≥ B · 2N / (F · n_gpus)` |
| Decode step (combined estimate) | `t_step ≈ max(mem, compute) + t_comm + t_overhead` |
| Prefill time | `t_prefill ≈ prefill_FLOPs / (F · MFU · n_gpus)` |
| TTFT | `t_queue + t_prefill + t_first_decode (+ network)` |
| TPOT | `t_step` (for the step the sequence is in) |
| End-to-end latency | `E2E = TTFT + (T − 1) · TPOT` |
| Per-user tok/s | `1 / TPOT` |

**Example.** Llama-3-70B, 8×H100, TP8, batch 1, short context:
- BF16: `141 GB / (8 × 3.35 TB/s) = 5.3 ms` → ceiling ~190 tok/s per user.
- FP8: `71 GB / 26.8 TB/s = 2.6 ms` → ceiling ~380 tok/s per user.
- Real systems add all-reduce + launch overhead; 60-75% of ceiling is good.

**Prefill example.** 70B, 4K-token prompt, 8×H100 BF16 at 50% MFU: `2 · 70.6e9 · 4096 ≈ 5.8e14` FLOPs (+ ~4% attention) → `5.8e14 / (8 · 989e12 · 0.5) ≈ 0.15 s`.

## 6. Throughput and efficiency

| Quantity | Formula |
| --- | --- |
| Decode throughput (tokens/s, replica) | `B / t_step` |
| Goodput | `requests/s that meet all SLOs` (not raw tokens/s) |
| MFU | `achieved FLOP/s / peak FLOP/s` — prefill metric |
| MBU | `achieved bytes/s / peak BW` = `(weight + KV bytes per step) / (t_step · BW)` — decode metric |
| Little's law | `concurrency = arrival_rate × time_in_system` |
| M/M/1 mean wait | `W_q = ρ / (1 − ρ) · service_time` — latency explodes as `ρ → 1` |

## 7. Parallelism and communication

| Quantity | Formula |
| --- | --- |
| Ring all-reduce time | `t ≈ 2 · (p − 1)/p · M / BW_link + 2(p−1) · latency` |
| TP all-reduces per layer | 2 (after attention out-proj, after FFN down-proj) |
| TP decode message per all-reduce | `M = B · d · b` (tiny at small `B` → latency-bound) |
| TP comm per token, whole model | `2 · L · all-reduce(B·d·b)` |
| PP bubble fraction (`m` micro-batches, `p` stages) | `(p − 1) / (m + p − 1)` |
| EP all-to-all bytes per token per MoE layer | `≈ 2 · k · d · b` (dispatch + combine), × fraction leaving the node |
| Weights per GPU under TP=t, PP=p | `N · b / (t · p)` |

## 8. Mixture of experts

| Quantity | Formula |
| --- | --- |
| Expected distinct experts hit per layer by `T` tokens, top-`k` of `E` | `E · (1 − (1 − k/E)^T)` |
| Tokens per expert (balanced) | `T · k / E` |
| Expert GEMM intensity | `≈ 2 · (T·k/E) / b` — needs large `T` to escape memory bound |
| Load imbalance factor | `max_expert_load / mean_expert_load` — step time scales with the max |

**Example.** DeepSeek-V3: 256 routed experts, top-8. A batch of 32 tokens hits `256·(1 − (1 − 8/256)^32) ≈ 160` experts per layer — most expert weights get read for very few tokens. That is why MoE decode wants huge batches and wide EP.

## 9. Speculative decoding

| Quantity | Formula |
| --- | --- |
| Expected tokens per verification step (i.i.d. acceptance `α`, `γ` drafts) | `E[τ] = (1 − α^(γ+1)) / (1 − α)` |
| Speedup (walltime) | `E[τ] / (1 + γ · c)` where `c = t_draft / t_target` per token |
| Optimal `γ` | maximize the above; typically 3-6 for `α ≈ 0.7-0.85` |
| Acceptance rule | accept `x` with prob `min(1, p(x)/q(x))`; on reject, sample from `norm(max(0, p − q))` |

**Example.** `α = 0.8, γ = 4`: `E[τ] = (1 − 0.8⁵)/0.2 = 3.36`. With `c = 0.05`: speedup `3.36 / 1.2 = 2.8×` — at small batch. At large batch, verifying `γ+1` tokens per sequence eats compute you no longer have spare; gains shrink toward 1×.

## 10. Quantization

| Quantity | Formula |
| --- | --- |
| Symmetric scale | `s = max(|x|) / q_max`, `q = round(x / s)`, `x̂ = s · q` |
| Asymmetric | `s = (max − min)/(2ⁿ − 1)`, `z = round(−min/s)`, `q = round(x/s) + z` |
| Uniform quant noise variance | `s² / 12` |
| FP8 E4M3 max | 448; E5M2 max 57,344 |
| INT4 symmetric range | −8…7 (often −7…7) |
| SmoothQuant migration | `Ŷ = (X·diag(s)⁻¹)·(diag(s)·W)`, `s_j = max|X_j|^α / max|W_j|^(1−α)` |
| Decode speedup ceiling from weight quant | `≈ b_old / b_new` (memory-bound regime only) |

## 11. Cost and capacity

| Quantity | Formula |
| --- | --- |
| Cost per million output tokens | `$/Mtok = $/GPU-hr / (tok/s/GPU × 3600) × 1e6` |
| Cost per request | `(S · c_in + T · c_out)` with `c_in ≪ c_out` (prefill is batched compute; decode is bandwidth) |
| GPUs needed | `peak_required_tok/s / tok/s_per_GPU_at_SLO × (1 + headroom)` |
| Required decode tok/s | `QPS × avg_output_tokens` |
| Required prefill tok/s | `QPS × avg_input_tokens × (1 − prefix_hit_rate)` |

**Example.** A GPU at $2.50/hr serving 2,000 output tok/s at SLO: `2.50 / 7.2e6 × 1e6 ≈ $0.35/Mtok`. At 30% average utilization the effective cost is ~`$1.16/Mtok` — utilization is the biggest lever on the bill.

## 12. Edge

| Quantity | Formula |
| --- | --- |
| Single-stream decode ceiling | `tok/s ≤ BW / model_bytes` |
| Fits in memory | `model_bytes + KV + OS/app reserve < RAM` |

**Example.** 8B model at ~4.5 bits/weight ≈ 4.7 GB on a 400 GB/s laptop SoC: ceiling ~85 tok/s; expect ~50-70% of that. Same model on a ~60 GB/s phone: ceiling ~13 tok/s.

## 13. Sanity checks to run on any claimed number

- **Per-user tok/s above `BW_total / weight_bytes`** at batch 1 without speculation → impossible; check for speculation, caching, or misreported units.
- **MFU above ~70% on prefill** or **MBU above ~90% on decode** → suspicious; check the peak used (sparse vs dense FLOPs is a common 2× error).
- **Throughput quoted without a latency SLO** → meaningless for interactive serving; ask for goodput.
- **Prefix-cache hit rate not reported** → TTFT may reflect cache, not compute.

Related: [[04 - Roofline and Performance Modeling]], [[99 - Inference Engineering Glossary]], [[00 - Inference Engineering Hub]]
