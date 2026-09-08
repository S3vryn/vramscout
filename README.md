# VRAMScout

**Can this modern LLM fit on my GPU(s) — where does the VRAM go, and how much context can I afford?**

VRAMScout is a small CLI for **deployment memory planning**. It reads Hugging Face model metadata without downloading checkpoint weights, detects the VRAM that is free right now, and estimates the per-GPU memory cost of:

- checkpoint weights;
- KV / MLA / hybrid recurrent cache state;
- sparse indexer state;
- runtime workspace;
- chunked-prefill scratch;
- safety reserve;
- the **maximum context length that fits**.

> **v0.4 alpha:** support now covers **Qwen3.8, DeepSeek-V4, GLM-5.2, Kimi-K2.5/K2.6 and MiniMax-M2.7**, plus ordinary decoder-only Transformers. Multi-GPU TP planning and Kimi MLA DCP cache sharding are included. Public accuracy checks live in [`validation/`](validation/README.md).

## Quick start

```bash
pip install -e .

# Current GPU
vramscout Qwen/Qwen3.8-27B

# 8 local GPUs, Kimi MLA cache sequence-sharded with DCP
vramscout moonshotai/Kimi-K2.6 --tp 8 --dcp 8 --kv-dtype fp8 --context 262144

# Plan a 4×96 GiB MiniMax deployment without owning the GPUs
vramscout MiniMaxAI/MiniMax-M2.7 --vram-gib 96 --tp 4 --context 196608

# DeepSeek / GLM architecture-specific caches
vramscout deepseek-ai/DeepSeek-V4-Flash --context 1048576
vramscout zai-org/GLM-5.2-FP8 --context 1048576
```

`--vram-gib` is a **per-GPU** budget. With `--tp 8 --vram-gib 96`, VRAMScout plans eight 96 GiB ranks, not one fictional 768 GiB device.

## Why modern models need architecture-aware memory math

There is no longer one universal `2 × layers × heads × head_dim × tokens` KV formula.

| Family | State VRAMScout models |
|---|---|
| Standard Transformer | MHA / GQA / MQA K+V |
| Qwen3.8 / `qwen3_5` | full-attention KV + constant GatedDeltaNet state |
| DeepSeek-V4 | SWA + C4 CSA + Lightning Indexer + C128 HCA |
| GLM-5.2 | compressed MLA latent + DSA IndexShare |
| Kimi-K2.5/K2.6 | compressed MLA latent; TP-replicated, DCP sequence-sharded |
| MiniMax-M2.7 | standard 62-layer GQA cache with TP KV-head sharding |

For example, **Kimi MLA does not get an 8× KV reduction just because the model uses TP=8**. The latent cache is shared across query heads and is replicated on pure TP ranks. vLLM Decode Context Parallelism instead shards that cache along sequence length; `--dcp` models this distinction.

## Example output

```text
VRAMScout  CAN RUN

GPU budget
Device / group          GPU 0 · 8× NVIDIA RTX PRO 6000 Blackwell
TP / DCP                                                    8 / 8
Per-GPU total VRAM                                      95.00 GiB
Limiting free                                           93.40 GiB

Model
Checkpoint                                  moonshotai/Kimi-K2.6
Architecture                                           kimi_k2
Cache architecture                 Kimi MLA · 576-dim latent
Weights                                                   int4
KV/cache                                                    fp8
Requested context                                      262,144

Per-GPU VRAM breakdown @ 262,144 tokens
Model weights / rank                                      ...
MLA latent KV (61 layers, fp8, DCP=8)                    ...
CUDA / runtime                                            ...
Prefill scratch                                           ...
Safety reserve                                            ...
-------------------------------------------------------------
Estimated peak / rank                                     ...
```

Exact numbers depend on live Hugging Face checkpoint metadata and your current GPU state.

## Multi-GPU semantics

```bash
# Use GPUs 0..7 on the current machine
vramscout moonshotai/Kimi-K2.6 --tp 8 --dcp 8

# Start from GPUs 4..7
vramscout MiniMaxAI/MiniMax-M2.7 --gpu-index 4 --tp 4

# Hypothetical/rented hardware (96 GiB per rank)
vramscout MiniMaxAI/MiniMax-M2.7 --vram-gib 96 --tp 4
```

VRAMScout uses the **least free VRAM among the selected local ranks** as the limiting budget.

For standard GQA/MQA caches, TP shards KV heads when the head layout is unambiguous. If a requested TP topology cannot be mapped safely, VRAMScout fails closed instead of guessing.

For MLA models such as Kimi, pure TP leaves the latent cache replicated. `--dcp N` divides the cache by N because DCP sequence-shards the latent cache. Current DCP support is deliberately limited to Kimi MLA, where public vLLM behavior is available for validation.

## Real checkpoint bytes, not fake uniform quantization

Modern FP8/NVFP4/INT4 checkpoints are mixed precision. Embeddings, attention, vision modules, MTP heads and MoE experts do not necessarily use the headline precision.

When Hugging Face exposes shard metadata, VRAMScout therefore prefers the **actual sum of `.safetensors` checkpoint bytes** over `parameter_count × guessed bits/parameter`.

Weights and cache precision are separate:

```text
weights: auto / fp32 / fp16 / bf16 / fp8 / int8 / int4 / nvfp4
cache:   auto / fp32 / fp16 / bf16 / fp8
indexer: auto / bf16 / fp16 / fp8 / fp4
```

A native INT4 Kimi checkpoint, for example, still uses BF16 MLA cache by default unless the serving engine is explicitly configured for FP8 KV.

## Public validation

Checked-in comparisons currently include:

| Model / target | VRAMScout | Public reference | Abs. error |
|---|---:|---:|---:|
| Qwen3.8-27B BF16 weights | 51.781 GiB | 51.70 GiB | **0.16%** |
| Qwen3.8-27B FP8 weights | 28.778 GiB | 28.56 GiB | **0.76%** |
| Qwen3.8-27B NVFP4 weights | 24.587 GiB | 24.60 GiB | **0.05%** |
| DeepSeek-V4 BF16 cache @ 1M | 9.625 GiB | 9.62 GiB | **0.05%** |
| GLM-5.2 vLLM effective cache | 55.219 KiB/token | 56.584 KiB/token | **2.41%** |
| **Kimi-K2.5 vLLM MLA, TP8/DCP1** | 68.625 KiB/token | 68.626 KiB/token | **0.0015%** |
| **MiniMax-M2.7 BF16 KV @ 204,800** | 48.4375 GiB | 48.44 GiB | **0.0052%** |

Run them locally:

```bash
python validation/run.py
```

The Kimi reference comes from a public vLLM startup log with 163.59 GiB cache capacity and 2,499,584 logical cache tokens at TP8/DCP1. The MiniMax reference is a vLLM startup requirement of 48.44 GiB for 204,800 BF16 KV tokens. See [`validation/README.md`](validation/README.md) for exact sources and methodology.

## GPU budget and max context

By default VRAMScout reads `nvidia-smi` and plans against **currently free** memory rather than marketing capacity.

```bash
vramscout Qwen/Qwen3.8-27B-FP8 --gpu-index 0
```

Or supply a hypothetical per-rank budget:

```bash
vramscout Qwen/Qwen3.8-27B-FP8 --vram-gib 48
```

The max-context solver evaluates the actual architecture-specific cache/state function. Long-prompt activation scratch is bounded by a serving-style prefill chunk (default 8192 active tokens):

```bash
vramscout moonshotai/Kimi-K2.6 --tp 8 --dcp 8 --prefill-chunk 16384
```

## Current coverage

**Supported:**

- standard decoder-only Transformer caches (Llama/Qwen/Mistral/Gemma-style families);
- GQA / MQA / MHA with TP KV-head placement;
- Qwen3.8 hybrid full-attention + GatedDeltaNet state;
- DeepSeek-V4 SWA + CSA/HCA compressed cache;
- GLM-5.2 MLA/DSA + IndexShare;
- Kimi-K2.5 and Kimi-K2.6 MLA, TP-replicated cache and DCP sharding;
- MiniMax-M2.7 GQA + multi-GPU TP planning;
- BF16 / FP8 / NVFP4 / INT4 native checkpoint-byte accounting where Hub metadata is available;
- current local NVIDIA GPU groups and hypothetical per-GPU budgets.

**Not yet fully modeled:**

- expert-parallel-specific per-rank weight imbalance / routing workspaces;
- full vLLM/SGLang process peak including CUDA graphs and backend workspaces;
- DCP for DeepSeek-V4 / GLM-5.2 compressed-cache paths;
- DeepSeek V2/V3 legacy MLA/DSA variants;
- Mamba/Jamba/recurrent architectures;
- exact multimodal activation memory beyond checkpoint weight residency.

Unsupported state semantics should fail closed rather than silently reuse a wrong formula.

## Development

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e . pytest
pytest -q
python validation/run.py
```

## Roadmap

- [x] Qwen3.8 hybrid cache/state accounting
- [x] native checkpoint-shard accounting for FP8/NVFP4/INT4
- [x] DeepSeek-V4 CSA/HCA + compressed indexer cache
- [x] GLM-5.2 MLA/DSA + IndexShare
- [x] Kimi-K2.5/K2.6 MLA + DCP semantics
- [x] MiniMax-M2.7 + TP GQA sharding
- [x] public architecture/cache validation corpus
- [ ] engine-aware vLLM/SGLang measured-peak profiles
- [ ] expert-parallel / pipeline-parallel weight placement
- [ ] DCP for additional MLA/compressed-cache architectures
- [ ] additional 2026 families as public checkpoints/serving logs land

## License

MIT
