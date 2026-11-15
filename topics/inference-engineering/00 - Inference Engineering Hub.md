---
title: Inference Engineering Hub
tags: [inference-engineering, hub, reading-map, llm, gpu]
---

# Inference Engineering Hub

Training a model is a one-time capital expense. Inference is the operating expense that runs every second the product exists. For any successful model, lifetime inference compute dwarfs training compute, and every user-visible property of the product — how fast the first word appears, how smoothly text streams, whether the service falls over at peak, what it costs per conversation — is decided by inference engineering.

An inference engineer owns the path from **weights on disk** to **tokens on a user's screen**: the numerics, the kernels, the engine's scheduler, the memory manager, the parallelism layout, the fleet's routing and autoscaling, and the measurement discipline that says whether any of it worked.

## The one idea to hold onto

Almost every note is a variation on one loop:

1. **Find the bound.** For the phase you care about (prefill or decode, at a given batch size and context length), is it limited by compute (FLOP/s), memory bandwidth (bytes/s), memory capacity (bytes), communication (interconnect bytes/s), or overhead (CPU, kernel launches, Python)? See [[04 - Roofline and Performance Modeling]].
2. **Compute the ideal.** Turn that bound into a number: the minimum milliseconds per token, the maximum tokens per second, the largest batch that fits. If you cannot write this number down, you cannot tell whether your system is good.
3. **Measure the gap.** Benchmark the real system under a realistic load ([[18 - Benchmarking and Profiling]]) and compare against the ideal. The gap is your work queue.
4. **Pull the cheapest lever that moves the bound without breaking quality.** Quantize to cut bytes, batch to raise arithmetic intensity, speculate to turn memory-bound steps into compute-bound ones, shard to add bandwidth, disaggregate to stop phases interfering — then go back to step 1, because the bound has moved.

Most beginner mistakes are skipping step 2 (optimizing without knowing the ceiling) or forgetting quality in step 4 (a 2× speedup that silently loses 3 points on the eval).

## Reading paths

**Foundations path** (you know what a transformer is, not how it runs):

1. [[01 - What Inference Engineering Is]]
2. [[02 - Transformer Inference Anatomy]]
3. [[03 - Accelerator Hardware]]
4. [[04 - Roofline and Performance Modeling]]
5. [[05 - KV Cache]]
6. [[07 - Batching and Scheduling]]

Keep [[98 - Formula Cheatsheet]] open alongside. After this you can explain why a 70B model streams at roughly the speed it does and why batching is nearly free for decode.

**Engine-builder path** (you want to write or hack on vLLM/SGLang-class code):

1. [[05 - KV Cache]]
2. [[07 - Batching and Scheduling]]
3. [[06 - Attention Kernels]]
4. [[13 - Kernels, Compilers, and CUDA Graphs]]
5. [[15 - Sampling, Determinism, and Structured Output]]
6. [[09 - Speculative Decoding]]
7. [[21 - Case Study - DeepInfer]]
8. [[22 - Projects and Learning Roadmap]] — build the nano engine

**Large-model / distributed path** (you deploy frontier-size and MoE models):

1. [[10 - Parallelism]]
2. [[11 - Mixture of Experts Inference]]
3. [[12 - Disaggregated Serving]]
4. [[16 - Long Context Inference]]
5. [[08 - Quantization]]

**Platform / SRE path** (you run the fleet and own the bill):

1. [[01 - What Inference Engineering Is]]
2. [[14 - Inference Engines Landscape]]
3. [[18 - Benchmarking and Profiling]]
4. [[17 - Serving Systems, Routing, and Autoscaling]]
5. [[19 - Cost and Capacity Planning]]
6. [[20 - Edge and On-Device Inference]]

## Note index

| Note | Covers |
| --- | --- |
| [[01 - What Inference Engineering Is]] | The trade space, TTFT/TPOT/goodput, SLOs, workload shapes, the stack map |
| [[02 - Transformer Inference Anatomy]] | Forward pass with shapes, prefill vs decode, FLOPs and bytes per token |
| [[03 - Accelerator Hardware]] | GPU architecture, HBM, tensor cores, NVLink/IB, spec tables across vendors |
| [[04 - Roofline and Performance Modeling]] | Arithmetic intensity, ridge point, batch crossover, MFU/MBU, envelope worksheet |
| [[05 - KV Cache]] | Size math, MQA/GQA/MLA, PagedAttention, prefix caching, offload tiers |
| [[06 - Attention Kernels]] | Online softmax, FlashAttention, FlashDecoding, FlashInfer, FlashMLA |
| [[07 - Batching and Scheduling]] | Continuous batching, chunked prefill, preemption, fairness, async scheduling |
| [[08 - Quantization]] | FP8/INT8/INT4/FP4 formats, scaling granularity, GPTQ/AWQ/SmoothQuant, eval |
| [[09 - Speculative Decoding]] | Draft-verify, acceptance math, Medusa, EAGLE, MTP, when it stops helping |
| [[10 - Parallelism]] | TP, PP, DP, CP, EP — communication cost and how to choose |
| [[11 - Mixture of Experts Inference]] | Routing, all-to-all, DeepEP, grouped GEMM, EPLB, wide-EP |
| [[12 - Disaggregated Serving]] | Prefill/decode split, KV transfer, DistServe, Splitwise, Mooncake, Dynamo |
| [[13 - Kernels, Compilers, and CUDA Graphs]] | Launch overhead, CUDA graphs, fusion, GEMM tiling, Triton, CUTLASS, TileLang |
| [[14 - Inference Engines Landscape]] | vLLM, SGLang, TensorRT-LLM, llama.cpp and others; how to read an engine |
| [[15 - Sampling, Determinism, and Structured Output]] | Logit processors, nondeterminism, batch invariance, grammar decoding |
| [[16 - Long Context Inference]] | RoPE scaling, sparse/sliding attention, context parallel, KV compression |
| [[17 - Serving Systems, Routing, and Autoscaling]] | API layer, prefix-aware routing, autoscaling signals, multi-LoRA, cold start |
| [[18 - Benchmarking and Profiling]] | Open-loop load, latency-throughput curves, pitfalls, nsys/ncu, accuracy evals |
| [[19 - Cost and Capacity Planning]] | $/Mtok, capacity for a QPS + SLO target, build vs buy |
| [[20 - Edge and On-Device Inference]] | llama.cpp/GGUF, MLX, NPUs, WebGPU, laptop/phone tok/s math |
| [[21 - Case Study - DeepInfer]] | Guided code tour of a production engine in `projects/DeepInfer` |
| [[22 - Projects and Learning Roadmap]] | Staged plan, build projects, skills checklist, interview questions |
| [[97 - Reading List]] | Papers, blogs, code, and courses in recommended order |
| [[98 - Formula Cheatsheet]] | All the envelope formulas |
| [[99 - Inference Engineering Glossary]] | Definitions |

## Notation used throughout

| Symbol | Meaning |
| --- | --- |
| `N` | Parameter count (total). `N_act` for active parameters in MoE |
| `L` | Number of transformer layers |
| `d` | Model (hidden) dimension, `d_model` |
| `h`, `h_kv` | Number of query heads, number of KV heads |
| `d_h` | Head dimension (`d / h` usually) |
| `B` | Batch size (concurrent sequences in a step) |
| `S`, `T` | Prompt (input) length, generated (output) length; `s` = current context length |
| `b` | Bytes per element (2 for BF16, 1 for FP8, 0.5 for INT4/FP4) |
| `F` | Peak accelerator FLOP/s at the chosen precision |
| `BW` | Memory bandwidth (HBM bytes/s) |
| `TTFT` | Time to first token |
| `TPOT` / `ITL` | Time per output token / inter-token latency |
| `α`, `γ` | Speculative acceptance rate, number of draft tokens |

Units: GB means 10⁹ bytes unless written GiB. Throughput "tok/s" is per GPU unless stated as per node or per replica.

## How each note is built

Every note follows the same skeleton so you can skim or go deep:

1. **Intuition** — the picture in words.
2. **Mechanics and math** — formulas with every symbol defined.
3. **Worked example** — real model, real hardware, real numbers.
4. **What practitioners do** — defaults, knobs, and how production systems choose.
5. **Common mistakes.**
6. **Check yourself** — questions with answers.
7. **Further reading** and **Related** links.

## What "expert" means here

You are an expert inference engineer when you can, without looking anything up:

- Estimate within ~30% the TTFT, TPOT, and max throughput of any dense or MoE model on any common accelerator, from the spec sheet and the config file.
- Explain which bound each kernel in a profile trace is hitting, and why.
- Pick a parallelism, quantization, batching, and speculation configuration for a given SLO and justify it with numbers.
- Design a benchmark that would convince a skeptic, including its accuracy side.
- Read any major engine's scheduler and memory manager and predict how it behaves under a burst.

[[22 - Projects and Learning Roadmap]] turns that list into a plan.
