# VRAMScout

**Can this current open-weight model fit on my GPU(s) — where does the VRAM go, and how much context can I afford?**

VRAMScout is a CLI for **LLM deployment memory preflight**. It reads Hugging Face metadata without downloading checkpoint weights, inspects the GPU memory that is actually free, and estimates per-rank:

- checkpoint weights;
- KV / MLA / hybrid recurrent state;
- sparse-attention/indexer state;
- CUDA/runtime workspace;
- chunked-prefill scratch;
- safety reserve;
- the **maximum context length that fits**.

> **v0.6 alpha:** the architecture registry now covers the main 2026 open-weight families: **Qwen3.8 / Qwen3.8-Flash-Next, DeepSeek-V4, GLM-5.2/5.3, Kimi-K2.x/K3, Hy4, MiniMax-M2.7/M3, Gemma 4, Nemotron 3.5 Lightning, Mistral Medium 3.5 / Small 4, LongCat-2.0, and MiMo-V2.5/Pro**. TP/DCP and vLLM memory-budget preflight are included.

## Quick start

```bash
pip install -e .

# What families are covered?
vramscout --list-supported

# Current GPU
vramscout Qwen/Qwen3.8-Flash-Next

# Current multimodal GLM family: checkpoint + text-serving state
vramscout zai-org/GLM-5.3-Flash --context 262144

# Huge MoE model on hypothetical 8×192 GiB GPUs
vramscout Qwen/Qwen3.8-2.4T-A95B --vram-gib 192 --tp 8 --context 131072

# Kimi K3
vramscout moonshotai/Kimi-K3 --vram-gib 192 --tp 8 --dcp 8 --context 262144

# Match vLLM's memory-budget policy
vramscout MiniMaxAI/MiniMax-M3 --engine vllm --gpu-memory-utilization 0.92 --tp 8
```

`--vram-gib` is a **per-GPU** budget. `--vram-gib 96 --tp 8` means eight 96 GiB ranks, not one fictional 768 GiB device.

## Current architecture coverage

Modern LLM inference state is no longer described by one universal KV formula. VRAMScout maps released configs into architecture-level memory families:

| Model family | State modeled |
|---|---|
| Standard decoder | MHA / GQA / MQA K+V |
| **Qwen3.8 / qwen3_5** | full-attention KV + GatedDeltaNet recurrent state |
| **Qwen3.8-Flash-Next / qwen4_exp** | QSA full-attention KV + compressed indexer + GatedDeltaNet state |
| **DeepSeek-V4 Flash/Pro** | compressed MLA/CSA/HCA or released MLA/indexer layout |
| **GLM-5.2** | MLA latent + DSA IndexShare |
| **GLM-5.3-Flash** | sparse MLA/indexer layers + linear/KDA recurrent layers |
| **Kimi-K2.5/K2.6** | MLA latent; TP-replicated and DCP sequence-sharded |
| **Kimi K3** | Gated MLA + KDA recurrent layers |
| **Tencent Hy4-preview** | gated MLA/DSA + IndexShare |
| **MiniMax-M2.7** | standard GQA with TP KV-head sharding |
| **MiniMax-M3** | standard KV + sparse-attention index state |
| **Gemma 4** | global-attention KV + bounded sliding-window KV |
| **Nemotron-3.5-Lightning** | attention KV + context-constant Mamba-2 recurrent state |
| **Mistral Medium 3.5** | standard GQA |
| **Mistral Small 4** | MLA latent cache |
| **LongCat-2.0** | MLA latent + released sparse index state |
| **MiMo-V2.5 / Pro** | full-attention KV + bounded chunk/SWA KV |

The implementation keys off architecture/config fields rather than a hard-coded model-name-only lookup. A new checkpoint using an already-supported memory family can often work without a new formula.

## Exact checkpoints covered by the v0.6 registry

The current smoke-test manifest includes:

```text
Qwen/Qwen3.8-27B
Qwen/Qwen3.8-2.4T-A95B
Qwen/Qwen3.8-Flash-Next

deepseek-ai/DeepSeek-V4-Pro-0813
deepseek-ai/DeepSeek-V4-Flash-Vision-Exp

zai-org/GLM-5.2
zai-org/GLM-5.3
zai-org/GLM-5.3-Flash

moonshotai/Kimi-K2.5
moonshotai/Kimi-K2.6
moonshotai/Kimi-K3

tencent/Hy4-preview
tencent/Hy4-preview-FP8

MiniMaxAI/MiniMax-M2.7
MiniMaxAI/MiniMax-M3

google/gemma-4-26B-A4B

nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

mistralai/Mistral-Medium-3.5-128B
mistralai/Mistral-Small-4-119B-2603
mistralai/Mistral-Small-4-119B-2603-NVFP4

meituan-longcat/LongCat-2.0

XiaomiMiMo/MiMo-V2.5
XiaomiMiMo/MiMo-V2.5-Pro
```

See [`validation/current_models_v06.json`](validation/current_models_v06.json) for the auditable registry and public sources.

## What the output means

```text
VRAMScout  CAN RUN

Deployment budget
GPU / group                     8× H200
TP / DCP                            8 / 8
Physical free / rank              ... GiB
Engine                                vllm
Planning budget / rank            ... GiB

Model
Checkpoint               moonshotai/Kimi-K3
Memory family             kimi_k3_hybrid
Parameters                       ... B
Requested context             262,144

Per-rank VRAM breakdown @ 262,144 tokens
Model weights / rank              ... GiB
Sparse/MLA latent cache           ... GiB
KDA recurrent state               ... GiB
CUDA / runtime                    ... GiB
Prefill scratch                   ... GiB
Safety reserve                    ... GiB
------------------------------------------------
Estimated peak / rank             ... GiB

Context budget
VRAM-derived max                  ...
Usable max                        ...
```

## Multi-GPU semantics

```bash
# Local GPUs 0..7
vramscout moonshotai/Kimi-K3 --tp 8 --dcp 8

# Hypothetical 4×96 GiB deployment
vramscout mistralai/Mistral-Small-4-119B-2603-NVFP4 --vram-gib 96 --tp 4
```

For local groups, the least-free selected GPU is the limiting rank. Standard GQA/MQA caches shard KV heads under TP only when the mapping is unambiguous. MLA caches are not blindly divided by TP; DCP is modeled separately because sequence sharding and tensor parallelism are different things.

## vLLM preflight

```bash
vramscout Qwen/Qwen3.8-Flash-Next \
  --engine vllm \
  --gpu-memory-utilization 0.92
```

vLLM requests a fraction of total GPU memory and performs its own on-device profile before creating the KV pool. VRAMScout mirrors the **startup memory-budget policy** and statically estimates the non-KV footprint. It is a preflight check, not a replacement for vLLM's runtime profiler.

## Native checkpoint bytes

For BF16/FP8/NVFP4/INT4 repositories, `parameters × nominal bits` is often wrong because embeddings, experts, vision modules, scales, and selected linears use mixed formats. VRAMScout therefore prefers actual Hugging Face `.safetensors` shard sizes when available.

v0.6 also avoids a common metadata trap: repositories such as Mistral may publish both `model-*.safetensors` and `consolidated-*.safetensors` representations of the **same checkpoint**. VRAMScout selects one logical shard set rather than double-counting both exports.

## Public validation

The existing numeric validation suite remains checked in:

| Target | VRAMScout | Public reference | Abs. error |
|---|---:|---:|---:|
| Qwen3.8-27B BF16 weights | 51.781 GiB | 51.70 GiB | **0.16%** |
| Qwen3.8-27B FP8 weights | 28.778 GiB | 28.56 GiB | **0.76%** |
| Qwen3.8-27B NVFP4 weights | 24.587 GiB | 24.60 GiB | **0.05%** |
| DeepSeek-V4 BF16 cache @ 1M | 9.625 GiB | 9.62 GiB | **0.05%** |
| GLM-5.2 vLLM effective cache | 55.219 KiB/token | 56.584 KiB/token | **2.41%** |
| Kimi-K2.5 vLLM MLA, TP8/DCP1 | 68.625 KiB/token | 68.626 KiB/token | **0.0015%** |
| MiniMax-M2.7 BF16 KV @ 204,800 | 48.4375 GiB | 48.44 GiB | **0.0052%** |
| MiniMax-M2.7 H200 vLLM KV pool | 11.38 GiB | 12.20 GiB | **6.69%** |

```bash
python validation/run.py
python validation/current_models_smoke.py
```

For newly released v0.6 families, architecture coverage and public-source provenance are added immediately; **numeric accuracy is only promoted into the table when a trustworthy public serving log or memory figure is available**. We do not turn config algebra into fake “ground truth.”

## Multimodal scope

For models such as GLM-5.3-Flash, Kimi K3, Qwen3.8-Flash-Next, MiniMax-M3, Gemma 4, Mistral Medium 3.5, and MiMo-V2.5:

- vision/audio **checkpoint residency is included** when it lives in the published checkpoint;
- text-generation cache/state is architecture-aware;
- request-dependent image/video/audio token expansion and encoder activation peaks are **not yet exact**.

The CLI prints this limitation instead of silently claiming full multimodal peak accuracy.

## Development

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e . pytest
pytest -q
python validation/run.py
```

## Roadmap

- [x] Qwen3.8 / Qwen3.8-Flash-Next hybrid state
- [x] DeepSeek-V4 compressed/MLA cache families
- [x] GLM-5.2 / GLM-5.3 sparse + linear attention
- [x] Kimi K2.x MLA / Kimi K3 KDA + Gated MLA
- [x] Hy4 gated MLA/DSA
- [x] MiniMax M2.7 / M3
- [x] Gemma 4 hybrid global/sliding cache
- [x] Nemotron 3.5 Mamba-2 hybrid state
- [x] Mistral Medium 3.5 / Small 4
- [x] LongCat-2.0
- [x] MiMo-V2.5 / Pro
- [x] TP/DCP and vLLM budget preflight
- [ ] measured vLLM/SGLang calibration import
- [ ] EP/PP-aware MoE placement and routing workspace
- [ ] exact request-dependent multimodal activation peaks

## License

MIT
