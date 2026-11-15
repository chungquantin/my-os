---
title: Inference Engineering Glossary
tags: [inference-engineering, glossary, reference]
---

# Inference Engineering Glossary

Alphabetical. Each entry points to the note where the idea is developed.

**Acceptance rate (α)** — In speculative decoding, the probability the target model accepts a drafted token. Drives expected tokens per step. [[09 - Speculative Decoding]]

**Activation memory** — Transient tensors during a forward pass. Small for decode, large for long prefills; chunked prefill bounds it. [[02 - Transformer Inference Anatomy]]

**All-reduce / all-gather / reduce-scatter** — Collective operations across GPUs. Tensor parallelism uses all-reduce twice per layer. [[10 - Parallelism]]

**All-to-all** — Collective where every rank sends a distinct slice to every other rank; the communication pattern of expert parallelism (dispatch and combine). [[11 - Mixture of Experts Inference]]

**Arithmetic intensity** — FLOPs performed per byte moved from memory. Compared against the ridge point to decide compute- vs memory-bound. [[04 - Roofline and Performance Modeling]]

**Async scheduling** — Overlapping CPU-side scheduling and input preparation for step `t+1` with GPU execution of step `t`, hiding host overhead. [[07 - Batching and Scheduling]]

**Attention DP** — Running attention data-parallel (each rank owns different sequences and their KV) while MoE/FFN layers use expert or tensor parallelism; avoids duplicating KV under TP for MLA-style models. [[10 - Parallelism]]

**AWQ** — Activation-aware Weight Quantization: protects salient weight channels (chosen by activation magnitude) via per-channel scaling before INT4 quantization. [[08 - Quantization]]

**Batch invariance** — Property that a request's output is bitwise identical regardless of which other requests share its batch; requires kernels whose reduction order does not depend on batch shape. [[15 - Sampling, Determinism, and Structured Output]]

**Block table** — Per-sequence mapping from logical KV blocks to physical blocks in paged KV cache. [[05 - KV Cache]]

**Chunked prefill** — Splitting a long prompt into fixed-size chunks and mixing them with decode tokens in one batch under a token budget, to protect TPOT. [[07 - Batching and Scheduling]]

**Context parallelism (CP)** — Splitting a single sequence's tokens across GPUs (often with ring attention) to parallelize long prefills. [[16 - Long Context Inference]]

**Continuous batching** — Iteration-level scheduling: sequences join and leave the running batch at every decode step instead of waiting for the whole batch to finish. Introduced by Orca. [[07 - Batching and Scheduling]]

**CUDA graph** — A recorded sequence of GPU kernel launches replayed with one launch, eliminating per-kernel CPU overhead; requires static shapes, hence batch-size buckets and padding. [[13 - Kernels, Compilers, and CUDA Graphs]]

**Decode** — The autoregressive phase generating one token per sequence per step; memory-bandwidth-bound at typical batch sizes. [[02 - Transformer Inference Anatomy]]

**DeepEP** — DeepSeek's expert-parallel communication library with high-throughput (prefill) and low-latency (decode) all-to-all kernels. [[11 - Mixture of Experts Inference]]

**Disaggregated serving (PD disaggregation)** — Running prefill and decode on separate GPU pools and transferring KV between them, so each phase is scheduled and parallelized for its own bound. [[12 - Disaggregated Serving]]

**Draft model** — Small, cheap model (or head) proposing tokens for the target model to verify. [[09 - Speculative Decoding]]

**E4M3 / E5M2** — The two FP8 formats: 4 exponent/3 mantissa bits (more precision, max 448) and 5/2 (more range). E4M3 is used for inference weights and activations. [[08 - Quantization]]

**EAGLE** — Speculative decoding family drafting at the feature (hidden-state) level with a light head on the target model; EAGLE-2 adds dynamic draft trees, EAGLE-3 trains with multi-layer features. [[09 - Speculative Decoding]]

**EPLB** — Expert-Parallel Load Balancer: replicates hot experts and re-places experts across GPUs to equalize load. [[11 - Mixture of Experts Inference]]

**Expert parallelism (EP)** — Placing different MoE experts on different GPUs; tokens are routed to the GPU owning their experts. [[10 - Parallelism]]

**FlashAttention** — IO-aware exact attention that tiles Q/K/V through on-chip SRAM with online softmax, never materializing the `S×S` score matrix in HBM. [[06 - Attention Kernels]]

**FlashDecoding (split-KV)** — Parallelizing decode attention across the KV length (not just batch and heads), then reducing partial softmaxes; essential for long context at small batch. [[06 - Attention Kernels]]

**Goodput** — Throughput counting only requests that meet their latency SLOs. The right objective for interactive serving. [[01 - What Inference Engineering Is]]

**GPTQ** — Post-training weight quantization using approximate second-order (Hessian) information to compensate rounding error column by column. [[08 - Quantization]]

**GQA / MQA / MHA** — Grouped-, multi-, and multi-head attention: number of KV heads equals a fraction, one, or all of the query heads. Fewer KV heads → smaller KV cache. [[05 - KV Cache]]

**Grammar-constrained decoding** — Masking logits each step so only tokens consistent with a regex, JSON schema, or CFG remain. [[15 - Sampling, Determinism, and Structured Output]]

**Grouped GEMM** — One kernel launch performing many independent GEMMs of different sizes — e.g. all local experts in an MoE layer. [[11 - Mixture of Experts Inference]]

**HBM** — High Bandwidth Memory: stacked DRAM on the accelerator package; its bandwidth is the decode bottleneck. [[03 - Accelerator Hardware]]

**ITL / TPOT** — Inter-token latency / time per output token: gap between successive streamed tokens. Sometimes TPOT is defined as `(E2E − TTFT)/(T − 1)`. [[01 - What Inference Engineering Is]]

**KV cache** — Stored key and value vectors for all previous tokens in every layer, so decode does not recompute them. The central memory object of an engine. [[05 - KV Cache]]

**Logits processor** — Function transforming the logit vector before sampling (temperature, penalties, grammar masks, bias). [[15 - Sampling, Determinism, and Structured Output]]

**LoRA serving (multi-LoRA)** — Serving many low-rank adapters over one base model in the same batch using specialized gather-GEMM kernels (Punica, S-LoRA). [[17 - Serving Systems, Routing, and Autoscaling]]

**MBU** — Model Bandwidth Utilization: achieved memory bandwidth ÷ peak; the efficiency metric for decode. [[04 - Roofline and Performance Modeling]]

**Medusa** — Speculative method adding extra decoding heads that predict tokens `t+2, t+3, …`, verified with tree attention. [[09 - Speculative Decoding]]

**MFU** — Model FLOPs Utilization: achieved useful FLOP/s ÷ peak; the efficiency metric for prefill. [[04 - Roofline and Performance Modeling]]

**MLA** — Multi-head Latent Attention (DeepSeek-V2/V3): caches a compressed latent vector per token instead of full K/V, shrinking KV by an order of magnitude; up-projections can be absorbed into Q and O at decode. [[05 - KV Cache]]

**MoE** — Mixture of Experts: FFN replaced by many experts, of which a router activates top-k per token; compute scales with active params, memory with total. [[11 - Mixture of Experts Inference]]

**MTP** — Multi-Token Prediction: extra modules trained to predict further-ahead tokens; reused as a built-in draft for speculative decoding. [[09 - Speculative Decoding]]

**MXFP4 / NVFP4** — 4-bit floating-point (E2M1) block formats with shared scales: MX uses 32-element blocks with an E8M0 power-of-two scale; NVFP4 uses 16-element blocks with an FP8 scale plus a tensor scale. [[08 - Quantization]]

**NCCL** — NVIDIA Collective Communications Library implementing all-reduce and friends over NVLink/IB. [[10 - Parallelism]]

**NVLink / NVSwitch** — NVIDIA's GPU-to-GPU interconnect and switch fabric; defines the scale-up domain (8 GPUs in HGX, 72 in GB200 NVL72). [[03 - Accelerator Hardware]]

**Online softmax** — Computing softmax in one streaming pass by maintaining a running max and running sum, rescaling as new blocks arrive; enables FlashAttention. [[06 - Attention Kernels]]

**PagedAttention** — Storing KV in fixed-size non-contiguous blocks with a block table, like OS virtual memory; near-zero fragmentation and cheap sharing. [[05 - KV Cache]]

**Pipeline parallelism (PP)** — Splitting layers into stages across GPUs; adds latency per token at small batch but needs little bandwidth. [[10 - Parallelism]]

**Prefill** — Processing the full prompt in one forward pass to build the KV cache and produce the first token; compute-bound. [[02 - Transformer Inference Anatomy]]

**Prefix caching** — Reusing KV blocks for shared prompt prefixes (system prompts, few-shot, multi-turn history) across requests. [[05 - KV Cache]]

**Preemption** — Evicting a running sequence when KV memory runs out, either by swapping its KV to CPU or discarding it and recomputing later. [[07 - Batching and Scheduling]]

**RadixAttention** — SGLang's prefix cache organized as a radix tree over token sequences with LRU eviction. [[05 - KV Cache]]

**Ridge point** — Arithmetic intensity `F/BW` where a kernel transitions from memory- to compute-bound. [[04 - Roofline and Performance Modeling]]

**RoPE** — Rotary position embedding; context extension methods (PI, NTK-aware, YaRN) rescale its frequencies. [[16 - Long Context Inference]]

**Roofline** — Plot of attainable FLOP/s vs arithmetic intensity, bounded by peak compute and peak bandwidth. [[04 - Roofline and Performance Modeling]]

**SLO** — Service Level Objective, e.g. "P99 TTFT < 1 s and P99 TPOT < 50 ms". [[01 - What Inference Engineering Is]]

**SmoothQuant** — Migrates activation outliers into weights via per-channel scaling so both can be INT8-quantized. [[08 - Quantization]]

**Speculative decoding** — Draft several tokens cheaply, verify them in one target forward pass, keep the accepted prefix; lossless when rejection sampling is used. [[09 - Speculative Decoding]]

**Sliding window attention (SWA)** — Each token attends only to the last `w` tokens, bounding KV per layer; often interleaved with full-attention layers. [[16 - Long Context Inference]]

**Tensor core** — Matrix-multiply unit inside each SM; delivers the bulk of peak FLOPs at low precision. [[03 - Accelerator Hardware]]

**Tensor parallelism (TP)** — Splitting each weight matrix across GPUs (column-then-row), with all-reduces per layer; adds aggregate bandwidth and cuts per-token latency. [[10 - Parallelism]]

**Token budget** — Max tokens (prefill + decode) the scheduler packs into one step; the knob trading TTFT against TPOT. [[07 - Batching and Scheduling]]

**Tree attention** — Attention mask letting one forward pass verify a tree of candidate continuations. [[09 - Speculative Decoding]]

**TTFT** — Time to first token: queueing + prefill + first sampling + network. [[01 - What Inference Engineering Is]]

**Warp specialization** — Kernel design where some warps only move data (producers, e.g. via TMA) while others only compute (consumers), overlapping memory and math. [[06 - Attention Kernels]]

**xPyD** — Notation for a disaggregated deployment with `x` prefill and `y` decode instances. [[12 - Disaggregated Serving]]

Related: [[98 - Formula Cheatsheet]], [[00 - Inference Engineering Hub]]
