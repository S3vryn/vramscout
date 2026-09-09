# VRAMScout launch posts

These are ready-to-post launch drafts for communities where VRAMScout is relevant. Keep the tone technical and avoid cross-post spam; answer questions with concrete validation data.

## Reddit — r/LocalLLaMA

**Title**

I built VRAMScout: predict whether modern LLMs fit your GPUs, how much context you can afford, and validate it against real vLLM/SGLang logs

**Body**

I kept running into the same deployment question: *will this checkpoint actually fit on my GPU(s), and how much context can I afford?*

Most VRAM calculators still assume a standard KV-cache formula, which breaks down for newer architectures such as MLA, sliding/hybrid attention, GatedDeltaNet/KDA, sparse indexers and recurrent state.

So I built **VRAMScout**: https://github.com/S3vryn/vramscout

It reads Hugging Face metadata without downloading the full checkpoint, detects current free NVIDIA VRAM (or accepts hypothetical per-GPU budgets), and estimates per-rank weights, cache/state, runtime workspace, prefill scratch and max context. It understands TP/DCP and can mirror vLLM's `gpu_memory_utilization` budget.

The part I care most about is validation. I collected public vLLM/SGLang deployment receipts and keep a reproducible `Predicted vs Actual` table. On six newer hard cache receipts, raw architecture-only MAPE is 2.85%. Engine-layout calibration is shown explicitly rather than hidden in the architecture formula.

A few examples:

- Kimi K3 vLLM BF16: 0.493% raw error
- Mistral Small 4 SGLang BF16: 0.002%
- MiniMax M3 vLLM BF16: 0.001%
- Qwen3.8-Flash-Next: 2.813% raw, 0.005% with an explicit vLLM layout factor

Gemma 4 is currently marked unresolved/version-sensitive rather than force-fit to one vLLM allocator version.

You can also feed startup logs back into the tool:

```bash
vramscout --parse-log vllm.log
vramscout --parse-log sglang.log --json
```

I would especially appreciate real startup logs / OOM-boundary reports for configurations not already in the validation corpus.

## Hacker News — Show HN

**Title**

Show HN: VRAMScout – architecture-aware VRAM and max-context planning for modern LLMs

**Text**

VRAMScout is a Python CLI that estimates whether a modern Hugging Face model fits on one or more GPUs and how much context the deployment can afford.

https://github.com/S3vryn/vramscout

The motivation is that the old `2 × layers × heads × head_dim × tokens` KV-cache formula is no longer enough. Current models combine standard KV, MLA latent caches, sliding windows, sparse indexers, GatedDeltaNet/KDA recurrent state, Mamba state and compressed caches.

VRAMScout parses the released model config into reusable memory components, applies TP/DCP placement rules, and keeps architecture math separate from engine-specific allocator overhead.

I also built a public vLLM/SGLang `Predicted vs Actual` corpus. Six recent hard cache receipts have 2.85% raw architecture-only MAPE; the repository keeps every source and calibration factor auditable. It deliberately leaves version-sensitive cases such as Gemma 4 unresolved instead of fitting an opaque constant.

The CLI can also normalize your own vLLM/SGLang startup logs for future calibration.

MIT licensed, Python 3.10–3.13.

## Hugging Face community

**Title**

VRAMScout: architecture-aware GPU VRAM + max-context preflight for current open-weight models

**Body**

I released **VRAMScout**, a small CLI for answering a deployment question before downloading/renting hardware: *will this Hugging Face checkpoint fit on my GPU(s), where will the VRAM go, and how much context can I afford?*

Repository: https://github.com/S3vryn/vramscout

The tool reads model/config metadata and supports modern persistent-state layouts including standard MHA/GQA/MQA KV, MLA, sliding/hybrid KV, sparse indexers, GatedDeltaNet/KDA, Mamba-2 recurrent state and DeepSeek-style compressed cache. Current families include Qwen3.8/Flash-Next, DeepSeek-V4, GLM-5.2/5.3, Kimi K2/K3, MiniMax M2.7/M3, Gemma 4, Nemotron 3.5, Mistral Small 4, LongCat 2.0 and MiMo V2.5.

I am also maintaining public vLLM/SGLang deployment receipts under `validation/` so estimates are not just config algebra. Contributions of clean startup logs and max-context boundary measurements are welcome.

## vLLM / SGLang developer communities

**Title**

VRAMScout: preflight memory estimator + parser for public vLLM/SGLang memory receipts

**Body**

I built a small open-source tool around a problem that comes up repeatedly in serving issues: converting model architecture + parallelism into an expected per-rank memory/cache budget before launching the engine.

https://github.com/S3vryn/vramscout

VRAMScout keeps two layers separate:

1. config-derived persistent inference-state math (KV/MLA/sliding/indexer/recurrent/compressed state), and
2. engine/layout overhead derived from explicit public receipts.

It has a `--engine vllm` budget mode and can parse startup logs:

```bash
vramscout --parse-log vllm.log
vramscout --parse-log sglang.log --json
```

The validation corpus currently includes Qwen3.8-Flash-Next, Kimi K3, GLM-5.3-Flash, Nemotron-3.5 Lightning, Mistral Small 4, MiniMax M3 and earlier Qwen/DeepSeek/GLM/Kimi/MiniMax receipts.

I am looking for additional clean memory/token receipts and max-context boundary measurements across engine versions/hardware. The goal is not to replace the engine profiler, but to make preflight estimates auditable and useful before allocating hardware.
