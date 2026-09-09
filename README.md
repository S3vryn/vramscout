# VRAMScout

**English** | [中文](README_CN.md)

**Can this current open-weight model fit on my GPU(s) — where does the VRAM go, and how much context can I afford?**

VRAMScout is a CLI for **LLM deployment memory preflight**. It reads Hugging Face metadata without downloading checkpoint weights, inspects the GPU memory that is actually free, and estimates per-rank weights, cache/state, runtime workspace, prefill scratch, safety headroom, and the **maximum context length that fits**.

> **v0.7 alpha:** cache/state accounting is now built from **reusable memory components** instead of one monolithic formula per model family. A public vLLM/SGLang receipt corpus tracks `Predicted vs Actual`, and optional engine-layout calibration is explicit and auditable rather than hidden in the architecture math.

## Quick start

```bash
pip install -e .

# Current GPU
vramscout Qwen/Qwen3.8-Flash-Next

# vLLM startup policy + audited layout calibration when available
vramscout Qwen/Qwen3.8-Flash-Next \
  --engine vllm \
  --gpu-memory-utilization 0.92

# Kimi K3, eight GPUs with DCP sequence sharding
vramscout moonshotai/Kimi-K3 --tp 8 --dcp 8 --context 262144

# Hypothetical eight-GPU deployment
vramscout nvidia/MiniMax-M3-NVFP4 --vram-gib 192 --tp 8 --context 1048576

# See current architecture coverage
vramscout --list-supported
```

`--vram-gib` is **per GPU**. `--vram-gib 96 --tp 8` means eight 96 GiB ranks, not one fictional 768 GiB device.

## The v0.7 memory model

A modern model is interpreted as a graph of reusable inference-state components:

```text
Hugging Face config
        ↓
architecture/profile parser
        ↓
Memory components
 ├─ KVCache
 ├─ LatentCache / MLA
 ├─ SlidingKV
 ├─ SparseIndexer
 ├─ GatedDeltaNet / KDA state
 ├─ Mamba-2 recurrent state
 └─ DeepSeek compressed K=V state
        ↓
TP / DCP placement
        ↓
engine-layout calibration (optional, explicit)
        ↓
per-rank VRAM + max context
```

This matters because a future checkpoint that reuses existing components can often work without adding a new memory formula. Code changes are mainly needed when a model introduces a genuinely new persistent inference-state primitive or a new placement/layout rule.

The components also handle details that simple calculators commonly miss: asymmetric K/V dimensions, TP replication versus sharding, DCP being opt-in rather than a generic divide-by-N rule, sparse-index cache state, recurrent state, compressed-cache ratios, and quantized index scale metadata.

## Current architecture coverage

| Model family | State modeled |
|---|---|
| Standard decoder | MHA / GQA / MQA K+V |
| **Qwen3.8 / qwen3_5** | full-attention KV + GatedDeltaNet state |
| **Qwen3.8-Flash-Next / qwen4_exp** | QSA KV + compressed indexer + GatedDeltaNet |
| **DeepSeek-V4 Flash/Pro** | SWA + compressed CSA/HCA / MLA-index state |
| **GLM-5.2** | MLA latent + DSA IndexShare |
| **GLM-5.3-Flash** | sparse MLA/indexer + KDA recurrent state |
| **Kimi-K2.5/K2.6** | MLA latent; TP-replicated, DCP sequence-sharded |
| **Kimi K3** | Gated MLA + KDA recurrent state |
| **Tencent Hy4-preview** | gated MLA/DSA + IndexShare |
| **MiniMax-M2.7** | standard GQA with TP KV-head placement |
| **MiniMax-M3** | paged KV + sparse-attention index state |
| **Gemma 4** | global KV + bounded sliding-window KV |
| **Nemotron-3.5-Lightning** | attention KV + Mamba-2 recurrent state |
| **Mistral Medium 3.5** | standard GQA |
| **Mistral Small 4** | MLA latent cache |
| **LongCat-2.0** | MLA latent + sparse index state |
| **MiMo-V2.5 / Pro** | full KV + bounded chunk/SWA KV, asymmetric K/V dims |

The dated checkpoint registry is [`validation/current_models_v06.json`](validation/current_models_v06.json).

## Predicted vs Actual

v0.7 adds a public receipt audit from **vLLM and SGLang startup logs**. The full table is [`validation/PREDICTED_VS_ACTUAL.md`](validation/PREDICTED_VS_ACTUAL.md); source data and URLs are in [`validation/public_runs_v07.json`](validation/public_runs_v07.json).

Across the six new hard cache receipts in this release:

```text
architecture-only MAPE       2.850%
audited public-profile fit    0.237%
```

Selected rows:

| Target | Raw error | After explicit calibration |
|---|---:|---:|
| Qwen3.8-Flash-Next · vLLM BF16 | 2.813% | **0.005%** |
| Kimi K3 · vLLM BF16 | **0.493%** | 0.493% |
| GLM-5.3-Flash · vLLM BF16 | 6.500% | **0.889%** |
| Nemotron-3.5 Lightning · vLLM FP8 | 7.293% | **0.031%** |
| Mistral Small 4 · SGLang BF16 | **0.002%** | 0.002% |
| MiniMax M3 · vLLM BF16, TP8 | **0.001%** | 0.001% |

Calibration is not presented as universal truth. It only activates for an exact `(engine, memory family, cache dtype)` profile with public evidence, and the output shows the calibration as its own VRAM row. Use `--calibration none` to see pure architecture math.

Gemma 4 is deliberately kept as a **known unresolved/version-sensitive case**: a public vLLM receipt differs from the simple tensor layout by ~36.7%, and hybrid-cache page grouping has changed across vLLM versions. VRAMScout does not fit one release and pretend the factor is universal.

## Parse your own vLLM/SGLang logs

v0.7 can normalize startup telemetry into a machine-readable receipt:

```bash
vramscout --parse-log vllm.log
vramscout --parse-log sglang.log --json
```

It recognizes model-load memory, available/active KV memory, logical cache tokens, maximum concurrency, CUDA-graph memory, peak activation memory, and consumed memory where the engine prints them. This is the path toward continuously improving the public calibration corpus without hard-coding private benchmark numbers.

## vLLM memory-budget mode

```bash
vramscout MiniMaxAI/MiniMax-M3 \
  --engine vllm \
  --gpu-memory-utilization 0.92 \
  --tp 8
```

vLLM plans inside `total VRAM × gpu_memory_utilization` and refuses startup if current free memory is below that request. VRAMScout mirrors this preflight policy. In v0.7 the default inner safety reserve is **zero in vLLM mode** because `gpu_memory_utilization` already leaves memory outside the engine budget; adding another automatic 5% reserve was double-counting headroom. `--reserve-gib` remains available when you explicitly want extra headroom inside the vLLM budget.

## Multi-GPU semantics

```bash
# Local GPUs 0..7
vramscout moonshotai/Kimi-K3 --tp 8 --dcp 8

# Hypothetical 4×96 GiB deployment
vramscout mistralai/Mistral-Small-4-119B-2603-NVFP4 --vram-gib 96 --tp 4
```

For local groups, the least-free selected GPU is the limiting rank. Standard KV/index heads follow TP sharding/replication rules. MLA is **not** blindly divided by TP. DCP is currently accepted only for families whose sequence-sharded cache behavior has public evidence; unsupported DCP layouts fail closed.

## Native checkpoint bytes

For BF16/FP8/NVFP4/INT4 repositories, `parameters × nominal bits` is often wrong because embeddings, experts, vision modules, scales, and selected linears use mixed formats. VRAMScout prefers actual Hugging Face `.safetensors` shard sizes when available and avoids double-counting repositories that publish both `model-*` and `consolidated-*` exports of the same checkpoint.

## Multimodal scope

For multimodal families, checkpoint residency and text-generation state are included. Request-dependent image/video/audio token expansion and encoder activation peaks are not yet exact; the CLI reports that limitation rather than claiming a full multimodal process peak.

## Validation and development

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e . pytest

pytest -q
python validation/run.py
python validation/predicted_vs_actual.py
python validation/current_models_smoke.py
```

The validation policy is intentionally strict: a config-derived number is an **estimate**; a public vLLM/SGLang memory line is **evidence**; an engine/version-specific fit is a **calibration profile**, not a new architecture constant.

## Roadmap

- [x] composable cache/state component graph
- [x] current 2026 open-weight architecture coverage
- [x] TP/DCP placement semantics
- [x] vLLM memory-budget preflight
- [x] vLLM/SGLang startup-log parser
- [x] Predicted vs Actual public receipt corpus
- [x] transparent engine-layout calibration profiles
- [ ] held-out calibration splits across multiple engine versions/hardware targets
- [ ] Gemma4 hybrid allocator/version model
- [ ] EP/PP-aware MoE placement and routing workspace
- [ ] exact request-dependent multimodal activation peaks

## License

MIT
