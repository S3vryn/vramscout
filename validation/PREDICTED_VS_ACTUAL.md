# Predicted vs Actual

Snapshot: **2026-09-09**

This table separates **config-derived architecture math** (`Raw`) from optional, explicit **engine-layout calibration** (`Calibrated`). Calibration never changes checkpoint/config interpretation; it only represents measured page/layout overhead for an exact engine/family/dtype profile.

Across the **6 new hard cache receipts** added in v0.7, architecture-only MAPE is **2.850%** and the audited public-profile fit is **0.237%**. This is an audit fit on public receipts, not a held-out guarantee for every vLLM/SGLang version or hardware target.

| Target | Engine | Metric | Raw | Actual | Raw err | Calibrated | Cal err |
|---|---|---:|---:|---:|---:|---:|---:|
| [Qwen3.8-27B BF16 weights](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | vLLM/HF | GiB | 51.781 | 51.700 | 0.157% | 51.781 | **0.157%** |
| [Qwen3.8-27B FP8 weights](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | vLLM/HF | GiB | 28.778 | 28.560 | 0.763% | 28.778 | **0.763%** |
| [Qwen3.8-27B Inferact NVFP4 weights](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | vLLM/HF | GiB | 24.587 | 24.600 | 0.053% | 24.587 | **0.053%** |
| [DeepSeek-V4 61-layer BF16 cache @1M](https://vllm.ai/blog/2026-04-24-deepseek-v4) | vLLM | GiB | 9.625 | 9.620 | 0.048% | 9.625 | **0.048%** |
| [GLM-5.2 FP8 MLA + BF16 IndexShare](https://github.com/vllm-project/vllm/issues/47934) | vLLM | KiB/token | 55.219 | 56.584 | 2.413% | 55.219 | **2.413%** |
| [Kimi-K2.5 MLA TP8/DCP1](https://github.com/MoonshotAI/Kimi-K2.5/issues/34) | vLLM | KiB/token | 68.625 | 68.626 | 0.002% | 68.625 | **0.002%** |
| [MiniMax-M2.7 BF16 KV @204,800](https://github.com/vllm-project/vllm/issues/42017) | vLLM | GiB | 48.438 | 48.440 | 0.005% | 48.438 | **0.005%** |
| [Qwen/Qwen3.8-Flash-Next](https://github.com/vllm-project/vllm/issues/54559) | vLLM | KiB/logical-token | 12.750 | 13.119 | 2.813% | 13.120 | **0.005%** |
| [moonshotai/Kimi-K3](https://github.com/vllm-project/vllm/issues/50147) | vLLM | KiB/logical-token | 27.000 | 27.134 | 0.493% | 27.000 | **0.493%** |
| [zai-org/GLM-5.3-Flash](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark/blob/main/docs/DEPLOY-REPORT.md) | vLLM | KiB/logical-token | 11.688 | 12.500 | 6.500% | 12.389 | **0.889%** |
| [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://github.com/sojufx/Nemotron-3.5-Lightning-30B-A3B-NVFP4-DGX-Spark-Recipe/blob/main/results/1m-vllm0271-dspark-k3.md) | vLLM | KiB/logical-token | 3.000 | 3.236 | 7.293% | 3.237 | **0.031%** |
| [mistralai/Mistral-Small-4-119B-2603](https://github.com/sgl-project/sglang/issues/21611) | SGLang | KiB/logical-token | 22.500 | 22.500 | 0.002% | 22.500 | **0.002%** |
| [nvidia/MiniMax-M3-NVFP4](https://github.com/vllm-project/vllm/issues/51494) | vLLM | KiB/logical-token | 44.250 | 44.250 | 0.001% | 44.250 | **0.001%** |

## Known version-sensitive gap

The following public receipt is intentionally **not auto-calibrated** because the engine's hybrid allocator/page grouping has changed across releases:

| Target | Metric | Raw | Actual | Raw err | Status |
|---|---:|---:|---:|---:|---|
| [google/gemma-4-26B-A4B-it](https://github.com/eugr/spark-vllm-docker/issues/217) | GiB/per-131072-token-sequence | 1.348 | 2.131 | 36.749% | unresolved_version_sensitive_hybrid_allocator |

This large gap is useful: it identifies where a clean tensor formula is not enough. VRAMScout keeps Gemma4 uncalibrated instead of fitting one vLLM release and pretending the factor is universal.

## Deployment evidence without a clean comparable memory receipt

These models were audited too. A recipe or successful long-context deployment is useful evidence, but VRAMScout does **not** turn it into a fake numeric error when the public source lacks a comparable per-rank GiB/token receipt.

| Model | Engine(s) | Public evidence |
|---|---|---|
| [Qwen/Qwen3.8-2.4T-A95B](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3.8-2.4t-a95b/README.md) | vLLM, SGLang | Official Dynamo matrices validate FP8 weights/KV at TP16 on GB300/GB200, with 262,144-278,528 context depending on engine/hardware. |
| [deepseek-ai/DeepSeek-V4-Flash-Vision-Exp](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx) | vLLM, SGLang | Public/official recipes validate the multimodal serving path; community memory receipts use custom NVFP4 MLA KV and are not apples-to-apples with the standard cache dtypes in the CLI. |
| [tencent/Hy4-preview-FP8](https://github.com/Tencent-Hunyuan/Hy4-preview) | vLLM, SGLang | Tencent publishes official vLLM/SGLang images and commands; no clean hard memory/token receipt was found in the audited public deployment docs at snapshot time. |
| [mistralai/Mistral-Medium-3.5-128B](https://github.com/sgl-project/sglang/issues/25160) | vLLM, SGLang | Public SGLang evidence confirms TP2 can load on B200 while TP1 OOMs, but lacks a comparable KV GiB/token allocator receipt. |
| [meituan-longcat/LongCat-2.0-FP8](https://github.com/whn09/LongCat-2.0-SGLang-EFA/blob/main/LONGCAT2_1M_CONTEXT.md) | SGLang | Public TP32/EP32 H200 deployment completes a 1,048,576-token request; FP8 KV pool reaches about 1.23M tokens while BF16 reaches about 752K. |
| [XiaomiMiMo/MiMo-V2.5 / MiMo-V2.5-Pro](https://github.com/sgl-project/sglang/blob/main/docs/src/snippets/autoregressive/mimo-v25-deployment.jsx) | SGLang | Official SGLang deployment matrix validates model-specific TP/DP-attention topology across H200/H100/B200/GB300; public receipts found were not precise/comparable enough for an exact cache row. |

## Reproduce

```bash
python validation/predicted_vs_actual.py
python validation/run.py
```

To ingest your own serving receipt:

```bash
vramscout --parse-log vllm.log
vramscout --parse-log sglang.log --json
```

The source dataset is [`public_runs_v07.json`](public_runs_v07.json).
