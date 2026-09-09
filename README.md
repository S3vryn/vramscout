# VRAMScout

**English** | [中文](README_CN.md)

[![CI](https://github.com/S3vryn/vramscout/actions/workflows/ci.yml/badge.svg)](https://github.com/S3vryn/vramscout/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/S3vryn/vramscout)](https://github.com/S3vryn/vramscout/releases/latest)
[![Python](https://img.shields.io/badge/python-3.10--3.13-blue)](pyproject.toml)
[![License](https://img.shields.io/github/license/S3vryn/vramscout)](LICENSE)

> **Will this modern LLM fit on my GPU(s), where will the VRAM go, and how much context can I afford?**

VRAMScout is an architecture-aware **LLM deployment memory preflight** CLI. It reads Hugging Face metadata without downloading the full checkpoint, inspects the VRAM that is actually free, and estimates per-rank weights, cache/state, runtime workspace, prefill scratch, headroom, and the **maximum context length that fits**.

Unlike simple `params × bits + KV` calculators, VRAMScout models modern inference state such as **MLA, sliding/hybrid KV, sparse indexers, GatedDeltaNet/KDA, Mamba-2 state and compressed caches**, with TP/DCP placement and optional vLLM layout calibration.

## Why trust it?

VRAMScout keeps public vLLM/SGLang deployment receipts next to the estimator in [`validation/`](validation/).

Across the **six new hard cache receipts** added in v0.7:

```text
architecture-only MAPE        2.850%
audited public-profile fit     0.237%
```

The second number is an audit fit on known public receipts, **not a held-out guarantee**. Engine/version-specific calibration is shown explicitly and can be disabled.

A concrete example:

```text
Qwen3.8-Flash-Next · vLLM BF16
Raw architecture cache       12.750 KiB/logical-token
Public vLLM receipt          13.119 KiB/logical-token
Explicit calibrated estimate 13.120 KiB/logical-token
```

See the full auditable table: [`validation/PREDICTED_VS_ACTUAL.md`](validation/PREDICTED_VS_ACTUAL.md).

## Install

### GitHub release wheel

```bash
pip install https://github.com/S3vryn/vramscout/releases/download/v0.7.0/vramscout-0.7.0-py3-none-any.whl
```

### From source

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e .
```

PyPI trusted publishing is prepared; the package will switch to `pip install vramscout` once the PyPI publisher is authorized.

## 10-second demo

```bash
# Current GPU
vramscout Qwen/Qwen3.8-Flash-Next

# Match vLLM's memory-budget policy
vramscout Qwen/Qwen3.8-Flash-Next \
  --engine vllm \
  --gpu-memory-utilization 0.92

# Kimi K3 on 8 GPUs with DCP sequence sharding
vramscout moonshotai/Kimi-K3 --tp 8 --dcp 8 --context 262144

# Plan hardware you do not own yet: 8 × 192 GiB
vramscout nvidia/MiniMax-M3-NVFP4 \
  --vram-gib 192 --tp 8 --context 1048576
```

The output answers:

```text
GPU / group and limiting free VRAM
weight residency per rank
KV / MLA / recurrent / sparse-index state
runtime + prefill + reserve
estimated peak per rank
fit / no-fit
VRAM-derived maximum context
model-declared usable context limit
```

## Modern memory model

A checkpoint is interpreted as a graph of reusable inference-memory components:

```text
Hugging Face config
        ↓
architecture/profile parser
        ↓
Memory components
 ├─ KVCache
 ├─ LatentCache / MLA
 ├─ Sliding KV
 ├─ SparseIndexer
 ├─ GatedDeltaNet / KDA state
 ├─ Mamba-2 recurrent state
 └─ DeepSeek compressed K=V state
        ↓
TP / DCP placement
        ↓
optional engine-layout calibration
        ↓
per-rank VRAM + max context
```

A future checkpoint that reuses existing components can often work without a new model-specific formula. New code is mainly needed when a model introduces a genuinely new persistent inference-state primitive or placement rule.

## Current architecture coverage

| Model family | State modeled |
|---|---|
| Standard decoder | MHA / GQA / MQA K+V |
| Qwen3.8 / `qwen3_5` | full-attention KV + GatedDeltaNet |
| Qwen3.8-Flash-Next | QSA KV + compressed indexer + GatedDeltaNet |
| DeepSeek-V4 | SWA + compressed CSA/HCA / MLA-index state |
| GLM-5.2 | MLA latent + DSA IndexShare |
| GLM-5.3-Flash | sparse MLA/indexer + KDA recurrent state |
| Kimi-K2.5/K2.6 | MLA latent, TP-replicated / DCP-sharded |
| Kimi K3 | Gated MLA + KDA recurrent state |
| Tencent Hy4-preview | gated MLA/DSA + IndexShare |
| MiniMax-M2.7 | standard GQA + TP KV-head placement |
| MiniMax-M3 | paged KV + sparse-attention index state |
| Gemma 4 | global KV + bounded sliding-window KV |
| Nemotron-3.5 Lightning | attention KV + Mamba-2 recurrent state |
| Mistral Medium 3.5 | standard GQA |
| Mistral Small 4 | MLA latent cache |
| LongCat-2.0 | MLA latent + sparse index state |
| MiMo-V2.5 / Pro | full KV + bounded chunk/SWA KV, asymmetric K/V dims |

Run `vramscout --list-supported` for the CLI view. The dated checkpoint registry is [`validation/current_models_v06.json`](validation/current_models_v06.json).

## How max context is estimated

VRAMScout searches for the largest context `C` such that:

```text
weights
+ architecture cache/state(C)
+ runtime/CUDA estimate
+ prefill scratch(C)
+ reserve
<= per-rank GPU / engine budget
```

Then:

```text
usable max context = min(VRAM-derived max, model config limit)
```

Cache/token slope is already strongly validated for many supported architectures. The **exact OOM boundary** is less certain because CUDA graphs, allocator fragmentation, paged-cache rounding, backend workspaces, EP/PP buffers and multimodal activations can move the boundary. Treat max-context output as a strong deployment preflight, not a byte-perfect runtime guarantee.

Gemma 4 is intentionally marked version-sensitive rather than force-calibrated: one public vLLM receipt differs from simple tensor layout by about 36.7%, reflecting hybrid allocator behavior that changes across engine versions.

## vLLM / SGLang log ingestion

Feed real startup logs back into VRAMScout:

```bash
vramscout --parse-log vllm.log
vramscout --parse-log sglang.log --json
```

The parser extracts fields such as model-load memory, KV memory, logical cache tokens, maximum concurrency, CUDA-graph memory, peak activation memory and consumed memory when the engine prints them.

## Calibration policy

Public calibration is deliberately narrow. A factor only applies to an exact `(engine, memory family, cache dtype)` profile with public evidence, and appears as a separate VRAM row.

```bash
# Pure config/architecture math only
vramscout MODEL --calibration none
```

VRAMScout does not silently turn one engine/version measurement into a universal architecture constant.

## Multi-GPU semantics

`--vram-gib` is a **per-GPU** budget. `--vram-gib 96 --tp 8` means eight 96 GiB ranks, not one fictional 768 GiB device.

For local groups, the least-free selected GPU is the limiting rank. Standard KV/index heads follow explicit TP placement rules; MLA is not blindly divided by TP; DCP is accepted only where sequence-sharded behavior is supported, otherwise VRAMScout fails closed.

## Native checkpoint bytes

Modern FP8/NVFP4/INT4 checkpoints are mixed precision. VRAMScout therefore prefers the actual Hugging Face `.safetensors` shard sizes over `parameters × nominal bits`, and avoids double-counting repositories that publish both `model-*` and `consolidated-*` exports of the same checkpoint.

## Multimodal scope

For multimodal families, checkpoint residency and text-generation state are included. Request-dependent image/video/audio token expansion and encoder activation peaks are not yet exact, so text-only deployment preflight is currently more reliable than full multimodal process-peak prediction.

## Development

```bash
pip install -e . pytest
pytest -q
python validation/run.py
python validation/predicted_vs_actual.py
python validation/current_models_smoke.py
```

## Roadmap

- [x] composable cache/state component graph
- [x] current 2026 open-weight architecture coverage
- [x] TP/DCP placement semantics
- [x] vLLM memory-budget preflight
- [x] vLLM/SGLang startup-log parser
- [x] Predicted vs Actual public receipt corpus
- [x] transparent engine-layout calibration
- [ ] held-out calibration across multiple engine versions/hardware targets
- [ ] Gemma 4 hybrid allocator/version model
- [ ] EP/PP-aware MoE placement and routing workspace
- [ ] exact request-dependent multimodal activation peaks

## Release

Latest release: [VRAMScout v0.7.0](https://github.com/S3vryn/vramscout/releases/tag/v0.7.0)

## License

MIT
