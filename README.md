# VRAMScout

**Can this Hugging Face LLM fit on the GPU I have right now — and how much context can I afford?**

VRAMScout is a small CLI that detects the **current free VRAM** on an NVIDIA GPU, reads a model's Hugging Face metadata without downloading the checkpoint weights, and estimates:

- whether the model fits at a requested context length;
- model-weight memory;
- KV-cache memory;
- CUDA/runtime overhead;
- prefill scratch memory;
- safety reserve;
- the **maximum context length allowed by current free VRAM**.

It is designed for the common pre-deployment question: *"I have this GPU and this model. Will it actually fit?"*

> Status: **v0.1 / alpha.** Weight and standard Transformer KV arithmetic are near-exact once model metadata is known. Runtime and prefill memory are conservative static estimates and are clearly labeled as such.

## Quick start

```bash
pip install -e .

vramscout Qwen/Qwen3-8B
```

Example usage:

```bash
# Check the default 8K context (or the model limit if smaller)
vramscout Qwen/Qwen3-8B

# Ask whether 64K fits
vramscout Qwen/Qwen3-8B --context 65536

# Quantized weights; KV remains BF16 by default
vramscout Qwen/Qwen3-8B --dtype int4 --context 65536

# Explicit FP8 KV cache (only use if your serving engine supports it)
vramscout Qwen/Qwen3-32B --dtype int8 --kv-dtype fp8

# Select another installed NVIDIA GPU
vramscout meta-llama/Llama-3.1-8B-Instruct --gpu-index 1

# Plan against a hypothetical/rented 48 GiB GPU
vramscout Qwen/Qwen3-32B --vram-gib 48

# Script-friendly output
vramscout Qwen/Qwen3-8B --json
```

## What it reports

```text
VRAMScout  CAN RUN

GPU
Device                  GPU 0 · NVIDIA GeForce RTX 5090
Total VRAM                                   31.84 GiB
Currently used                                2.10 GiB
Currently free                               29.74 GiB

VRAM breakdown @ 32,768 tokens
Model weights                                15.26 GiB
KV cache                                      4.00 GiB
CUDA / runtime                                0.75 GiB
Prefill scratch                               1.50 GiB
Safety reserve                                1.59 GiB
-------------------------------------------------------
Estimated peak                               23.10 GiB
Free after estimate                           6.64 GiB

Context budget
VRAM-derived max                              70,xxx
Usable max                                    70,xxx
```

Numbers above are illustrative; VRAMScout calculates them from the selected model and GPU.

## Memory model

For a standard decoder-only Transformer, model weights are estimated as

```text
weights = parameter_count × bytes_per_weight
```

The KV cache is

```text
KV = 2 × layers × batch × context × kv_heads × head_dim × bytes_per_KV_element
```

The leading `2` is for **K + V**. This means GQA/MQA models can use much less KV memory than ordinary MHA models.

VRAMScout then reserves memory for CUDA/runtime state, prefill scratch buffers and a safety margin. The maximum context is obtained by solving the remaining linear memory budget for the token count, then clamping it to the model-declared context limit.

## Why current *free* VRAM matters

A 32 GiB GPU does not necessarily have 32 GiB available. Another process may already be using 12 GiB. VRAMScout reads `nvidia-smi` and plans against the memory that is free **right now**.

Use `--vram-gib` to ignore local GPU state and plan against a hypothetical GPU budget.

## Precision behavior

Weight dtype and KV-cache dtype are deliberately separate:

- BF16 / FP16 weights: ~2 bytes per parameter;
- FP32 weights: ~4 bytes per parameter;
- INT8 / INT4: includes a small packing/scale overhead estimate;
- quantized weights **do not automatically mean quantized KV cache**;
- `--kv-dtype auto` keeps KV in BF16 for FP16/BF16/INT8/INT4 weights.

## Model metadata

For Hugging Face model IDs, VRAMScout downloads only metadata/configuration, not checkpoint weights. Parameter count is taken from Hugging Face safetensors metadata when available; otherwise a dense decoder architecture estimate is used.

Local directories containing `config.json` are also supported.

## Scope and limitations

VRAMScout v0.1 intentionally keeps the scope narrow:

- NVIDIA auto-detection through `nvidia-smi`;
- single-GPU inference;
- standard decoder-only Transformer KV caches;
- conservative full-cache accounting for sliding-window models.

It **fails closed** on known non-standard cache/state architectures such as DeepSeek MLA and Mamba-style recurrent/state-space models instead of silently applying the wrong KV formula.

Real peak memory varies with serving engine (Transformers, vLLM, SGLang, llama.cpp), attention backend, CUDA graphs, allocator fragmentation, quantization format and prefill strategy. The current runtime/prefill terms are therefore estimates, not guarantees.

## Roadmap

- [ ] measured `vramscout probe` calibration on the local host;
- [ ] vLLM / SGLang engine-specific overhead profiles;
- [ ] multi-GPU tensor-parallel planning;
- [ ] native sliding-window/interleaved-local cache accounting;
- [ ] MLA / hybrid/recurrent inference-state accounting;
- [ ] AMD / Apple Silicon detection.

## Development

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e . pytest
pytest -q
```

## License

MIT
