---
title: Reading List
tags: [inference-engineering, reading-list, papers, resources]
---

# Reading List

Ordered so that each item mostly depends only on items above it. ★ marks the ones to read first if time is short. Where an arXiv ID is given, the paper is at `https://arxiv.org/abs/<id>`.

## Stage 0 — Mental models (read before anything else)

- ★ **Transformer Inference Arithmetic** — kipply (2022). The single best envelope-math primer: FLOPs, bytes, KV cache, latency bounds. https://kipp.ly/transformer-inference-arithmetic/
- ★ **How to Scale Your Model** — Google DeepMind JAX team (2025). Roofline-first book on TPUs/GPUs; the inference chapter is essential. https://jax-ml.github.io/scaling-book/
- ★ **Making Deep Learning Go Brrrr From First Principles** — Horace He. Compute vs memory vs overhead bound in ten minutes. https://horace.io/brrr_intro.html
- ★ **Inference Engineering**, by Philip Kiely (Baseten, 2026). A practitioner's book on the full stack: runtime, infrastructure, tooling. Free e-book: https://baseten.com/inference-engineering. Excerpt with framing: "What is inference engineering?", *The Pragmatic Engineer*: https://newsletter.pragmaticengineer.com/p/what-is-inference-engineering
- **Large Transformer Model Inference Optimization** — Lilian Weng (2023). Broad survey map. https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
- **Roofline: An Insightful Visual Performance Model** — Williams, Waterman, Patterson, CACM 2009. The original roofline paper. https://doi.org/10.1145/1498765.1498785

Pairs with [[01 - What Inference Engineering Is]], [[02 - Transformer Inference Anatomy]], [[04 - Roofline and Performance Modeling]].

## Stage 1 — Architecture facts that drive inference cost

- **Attention Is All You Need** — Vaswani et al. 2017 — 1706.03762
- **Fast Transformer Decoding: One Write-Head is All You Need** (MQA) — Shazeer 2019 — 1911.02150
- **GQA: Training Generalized Multi-Query Transformer Models** — Ainslie et al. 2023 — 2305.13245
- **DeepSeek-V2** (introduces MLA) — 2405.04434
- ★ **DeepSeek-V3 Technical Report** (MLA, MoE, MTP, FP8, deployment numbers) — 2412.19437
- **RoFormer** (RoPE) — 2104.09864

Pairs with [[02 - Transformer Inference Anatomy]], [[05 - KV Cache]], [[11 - Mixture of Experts Inference]].

## Stage 2 — The core of a modern engine

- ★ **Orca: A Distributed Serving System for Transformer-Based Generative Models** — Yu et al., OSDI 2022 (iteration-level / continuous batching). https://www.usenix.org/conference/osdi22/presentation/yu
- ★ **Efficient Memory Management for LLM Serving with PagedAttention** (vLLM) — Kwon et al., SOSP 2023 — 2309.06180
- ★ **SGLang: Efficient Execution of Structured Language Model Programs** (RadixAttention) — Zheng et al. 2023 — 2312.07104
- **SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills** — 2308.16369
- ★ **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** — OSDI 2024 — 2403.02310
- **Fairness in Serving Large Language Models** (Virtual Token Counter) — 2401.00588
- **Llumnix: Dynamic Scheduling for LLM Serving** — 2406.03243

Pairs with [[05 - KV Cache]], [[07 - Batching and Scheduling]].

## Stage 3 — Attention kernels

- ★ **FlashAttention** — Dao et al. 2022 — 2205.14135
- ★ **FlashAttention-2** — Dao 2023 — 2307.08691
- **FlashAttention-3** (Hopper: async, warp specialization, FP8) — 2407.08608
- **Flash-Decoding for long-context inference** — Dao, Haziza, Massa, Sizov (2023 blog). https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving** — 2501.01005
- **FlashMLA** (code) — https://github.com/deepseek-ai/FlashMLA
- **Online normalizer calculation for softmax** — Milakov & Gimelshein 2018 — 1805.02867

Pairs with [[06 - Attention Kernels]].

## Stage 4 — Quantization

- **LLM.int8()** — Dettmers et al. 2022 — 2208.07339
- **FP8 Formats for Deep Learning** — Micikevicius et al. 2022 — 2209.05433
- ★ **GPTQ** — Frantar et al. 2022 — 2210.17323
- ★ **SmoothQuant** — Xiao et al. 2022 — 2211.10438
- ★ **AWQ: Activation-aware Weight Quantization** — Lin et al. 2023 — 2306.00978
- **QServe: W4A8KV4 Quantization and System Co-design** — 2405.04532
- **MARLIN: Mixed-Precision Auto-Regressive Parallel Inference** — 2408.11743
- **KIVI: Tuning-Free Asymmetric 2bit Quantization for KV Cache** — 2402.02750
- **OCP Microscaling (MX) Formats Specification v1.0** — https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Pairs with [[08 - Quantization]].

## Stage 5 — Speculative decoding

- ★ **Fast Inference from Transformers via Speculative Decoding** — Leviathan, Kalman, Matias 2022 — 2211.17192
- **Accelerating LLM Decoding with Speculative Sampling** — Chen et al. (DeepMind) 2023 — 2302.01318
- **SpecInfer** (token-tree verification) — 2305.09781
- **Medusa** — 2401.10774
- ★ **EAGLE** — 2401.15077; **EAGLE-2** — 2406.16858; **EAGLE-3** — 2503.01840
- **Break the Sequential Dependency of LLM Inference Using Lookahead Decoding** — 2402.02057

Pairs with [[09 - Speculative Decoding]].

## Stage 6 — Parallelism, MoE, disaggregation

- **Megatron-LM** (tensor parallelism) — 1909.08053
- ★ **Efficiently Scaling Transformer Inference** — Pope et al. (Google) 2022 — 2211.05102. The partitioning-for-inference paper.
- **DeepSpeed-Inference** — 2207.00032
- **Switch Transformers** — 2101.03961; **Mixtral of Experts** — 2401.04088
- ★ **DeepEP** (expert-parallel communication library) — https://github.com/deepseek-ai/DeepEP
- **DeepSeek Open Infra Index** (inference system overview, EPLB, profiling data) — https://github.com/deepseek-ai/open-infra-index
- ★ **DistServe: Disaggregating Prefill and Decoding** — OSDI 2024 — 2401.09670
- **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** — ISCA 2024 — 2311.18677
- ★ **Mooncake: A KVCache-centric Disaggregated Architecture** — 2407.00079
- **Ring Attention with Blockwise Transformers** — 2310.01889

Pairs with [[10 - Parallelism]], [[11 - Mixture of Experts Inference]], [[12 - Disaggregated Serving]].

## Stage 7 — Long context, sampling, structure

- **YaRN: Efficient Context Window Extension** — 2309.00071
- **Efficient Streaming Language Models with Attention Sinks** (StreamingLLM) — 2309.17453
- **H2O: Heavy-Hitter Oracle** — 2306.14048
- **SnapKV** — 2404.14469
- **Native Sparse Attention** (DeepSeek) — 2502.11089
- **XGrammar: Flexible and Efficient Structured Generation Engine** — 2411.15100
- **Efficient Guided Generation for LLMs** (Outlines, FSM approach) — 2307.09702
- ★ **Defeating Nondeterminism in LLM Inference** — Thinking Machines Lab (2025). https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

Pairs with [[15 - Sampling, Determinism, and Structured Output]], [[16 - Long Context Inference]].

## Stage 8 — Serving at fleet scale

- **S-LoRA: Serving Thousands of Concurrent LoRA Adapters** — 2311.03285
- **Punica: Multi-Tenant LoRA Serving** — 2310.18547
- **FlexGen** (offloading, throughput-oriented) — 2303.06865
- **MLPerf Inference Benchmark** — 1911.02549
- **NVIDIA Dynamo** — https://github.com/ai-dynamo/dynamo
- **llm-d** — https://github.com/llm-d/llm-d

Pairs with [[17 - Serving Systems, Routing, and Autoscaling]], [[18 - Benchmarking and Profiling]], [[19 - Cost and Capacity Planning]].

## Codebases to read (in this order)

1. **nano-vLLM** or **mini-sglang**-style small engines — small enough to read in an afternoon; search GitHub for the current maintained one.
2. **vLLM** — https://github.com/vllm-project/vllm — start at `vllm/v1/engine/` and `vllm/v1/core/sched/`.
3. **SGLang** — https://github.com/sgl-project/sglang — start at `python/sglang/srt/managers/scheduler.py` and the radix cache.
4. **FlashInfer** — https://github.com/flashinfer-ai/flashinfer
5. **TensorRT-LLM** — https://github.com/NVIDIA/TensorRT-LLM
6. **llama.cpp** — https://github.com/ggml-org/llama.cpp — the best study of CPU/Metal/edge inference.
7. **DeepInfer** — local at `projects/DeepInfer`; see [[21 - Case Study - DeepInfer]].

Paths inside fast-moving repos drift; if a path is gone, search for the scheduler class.

## Courses, books, and talks

- ★ **GPU MODE lectures** — https://github.com/gpu-mode/lectures — CUDA, Triton, kernel case studies, by practitioners.
- **Programming Massively Parallel Processors** (Hwu, Kirk, El Hajj, 4th ed., 2022) — the CUDA textbook.
- **NVIDIA CUDA C++ Programming Guide** — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Triton tutorials** — https://triton-lang.org/main/getting-started/tutorials/
- **CMU 15-442/642 Machine Learning Systems** and **Stanford CS336 Language Modeling from Scratch** — public lecture materials cover systems for training and inference.
- **AI Engineering** — Chip Huyen (O'Reilly, 2025) — chapter on inference optimization for the product angle.

## How to read a systems paper fast

1. Read the abstract and find the **one bottleneck** the paper claims to remove.
2. Jump to the evaluation: what baseline, what hardware, what workload, what metric? Is it goodput under an SLO or only peak throughput?
3. Only then read the design section, asking: what does this cost (memory, complexity, quality), and does that cost survive at your batch size?
4. Write one line in [[98 - Formula Cheatsheet]] style capturing the paper's core arithmetic.

Related: [[00 - Inference Engineering Hub]], [[22 - Projects and Learning Roadmap]]
