# Public validation

VRAMScout keeps public reference data next to the estimator so accuracy claims are auditable. We deliberately separate three levels of evidence:

1. **checkpoint bytes** — weight residency proxy from actual safetensors shard sizes;
2. **architecture cache math** — KV / latent / recurrent / indexer state implied by the released config;
3. **engine-observed cache** — vLLM startup numbers, which also include block layout and allocator effects.

Run all checked-in comparisons:

```bash
python validation/run.py
```

## Current results

| Model / target | VRAMScout | Public reference | Absolute error |
|---|---:|---:|---:|
| Qwen3.8-27B BF16 weights | 51.781 GiB | 51.70 GiB | **0.16%** |
| Qwen3.8-27B FP8 weights | 28.778 GiB | 28.56 GiB | **0.76%** |
| Qwen3.8-27B NVFP4 weights | 24.587 GiB | 24.60 GiB | **0.05%** |
| DeepSeek-V4 61-layer BF16 cache @ 1M | 9.625 GiB | 9.62 GiB | **0.05%** |
| GLM-5.2 FP8 MLA + BF16 IndexShare cache | 55.219 KiB/token | 56.584 KiB/token | **2.41%** |

The GLM number is the most demanding comparison above: vLLM reported **126.41 GiB** available cache and **2,342,528 logical cache tokens**, so its reference includes real cache-block/layout overhead rather than being the same algebra repeated elsewhere.

## Qwen3.8-27B

The released config is hybrid: 16 full-attention layers grow ordinary KV state, while 48 GatedDeltaNet linear-attention layers keep constant recurrent/conv state. Treating all 64 layers as full attention would overestimate long-context KV by 4×.

Weight validation uses real safetensors shard byte totals for the native BF16 / FP8 / NVFP4 checkpoints rather than a uniform bits-per-parameter guess.

Sources are recorded in [`public/qwen3_8_27b.json`](public/qwen3_8_27b.json).

## DeepSeek-V4

VRAMScout models the short shared-K=V sliding window, CSA compressed cache at 1/4 sequence rate, CSA Lightning Indexer state, and HCA compressed cache at 1/128 sequence rate.

vLLM publishes a worked 61-layer BF16 example (30 C4 + 31 C128 layers) at 1,048,576 tokens: **9.62 GiB**. VRAMScout obtains **9.6246 GiB** from the released architecture math.

The released V4-Flash schedule (43 layers: 2 sliding-only + 21 C4 + 20 C128) is also represented, but its 1M numbers are labeled architecture estimates because an exact public engine byte total for that exact layout is not used as ground truth.

Sources are recorded in [`public/deepseek_v4.json`](public/deepseek_v4.json).

## GLM-5.2

The official config declares 78 layers, a 512-dim compressed KV latent plus 64 RoPE dimensions, and IndexShare: only **21** indexer caches are materialized while **57** layers share them.

For the public vLLM path with FP8 KV and `use_fp4_indexer_cache=False`, VRAMScout predicts 55.219 KiB/logical token. A public vLLM startup log reports 126.41 GiB for 2,342,528 logical tokens = 56.584 KiB/logical token, a **2.41%** difference.

Sources are recorded in [`public/glm_5_2.json`](public/glm_5_2.json).

## What is not claimed yet

VRAMScout does **not** yet claim that total process peak VRAM is within these error bands. Total serving memory also depends on tensor/expert parallel sharding, CUDA graphs, kernels, model-specific workspace, paged-cache allocation, prefix cache, concurrency and prefill settings.

The next validation layer is engine-aware: compare full vLLM/SGLang startup memory and max-context boundaries under pinned commands and hardware.
