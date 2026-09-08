# Public validation

VRAMScout keeps public reference data next to the estimator so accuracy claims are auditable. We separate three evidence levels:

1. **checkpoint bytes** — actual safetensors shard sizes for weight residency;
2. **architecture cache math** — KV / MLA / recurrent / sparse-indexer state implied by released configs;
3. **engine-observed cache** — vLLM startup memory and cache-token capacity.

Run the checked-in comparisons:

```bash
python validation/run.py
```

## Current results

| Model / target | VRAMScout | Public reference | Abs. error |
|---|---:|---:|---:|
| Qwen3.8-27B BF16 weights | 51.781 GiB | 51.70 GiB | **0.16%** |
| Qwen3.8-27B FP8 weights | 28.778 GiB | 28.56 GiB | **0.76%** |
| Qwen3.8-27B NVFP4 weights | 24.587 GiB | 24.60 GiB | **0.05%** |
| DeepSeek-V4 61-layer BF16 cache @ 1M | 9.625 GiB | 9.62 GiB | **0.05%** |
| GLM-5.2 FP8 MLA + BF16 IndexShare | 55.219 KiB/token | 56.584 KiB/token | **2.41%** |
| **Kimi-K2.5 BF16 MLA, TP8/DCP1** | 68.625 KiB/token | 68.626 KiB/token | **0.0015%** |
| **MiniMax-M2.7 BF16 KV @ 204,800** | 48.4375 GiB | 48.44 GiB | **0.0052%** |
| **MiniMax-M2.7 H200 vLLM KV pool** | 11.38 GiB | 12.20 GiB | **6.69%** |

The Kimi comparison is particularly useful for parallelism semantics. A public vLLM run reports **163.59 GiB** available KV memory and **2,499,584** cache tokens on each TP rank. That implies 70,273.06 bytes/logical-token, while the released MLA config predicts 61 × (512 + 64) × 2 = **70,272 bytes/token**. Pure TP therefore does not divide the MLA cache — exactly why DCP matters.

The MiniMax comparison comes from a vLLM startup error that states **48.44 GiB** of KV is required for a 204,800-token sequence. Standard GQA arithmetic from the released 62-layer / 8-KV-head / 128-head-dim config gives **48.4375 GiB**. A separate 2×H200 run (140.4 GiB/GPU, `gpu_memory_utilization=0.92`) reports 107.31 GiB model load and 12.2 GiB available KV. Conditioning on that measured weight residency, VRAMScout's static runtime/prefill/reserve model predicts 11.38 GiB KV, a **6.69% full-startup error**. This wider error is intentionally reported rather than hidden: engine workspaces are harder than raw cache geometry.

## Modern architecture notes

### Qwen3.8

16 full-attention layers grow ordinary KV state; 48 GatedDeltaNet layers keep constant recurrent/conv state. Treating all 64 layers as full attention overestimates long-context KV by roughly 4×.

### DeepSeek-V4

VRAMScout models the shared-K=V sliding window, C4 CSA compressed cache, Lightning Indexer, and C128 HCA compressed cache instead of forcing the model through a standard KV formula.

### GLM-5.2

The released config uses a 512-dim compressed latent plus 64 RoPE dimensions and IndexShare. Only 21 indexer caches are materialized while the remaining 57 layers share them.

### Kimi-K2.5 / Kimi-K2.6

Kimi uses MLA with a 512-dim latent plus 64 RoPE dimensions across 61 layers. The cache is effectively one shared latent rather than per-head K/V, so pure tensor parallelism replicates it. vLLM Decode Context Parallelism shards the cache along sequence length; VRAMScout exposes this as `--dcp`.

Sources:
- [`public/kimi_k2_5.json`](public/kimi_k2_5.json)
- MoonshotAI/Kimi-K2.5 issue #34 (vLLM TP8/DCP1 startup log)
- vLLM issue #40608 and the vLLM DCP write-up for Kimi-K2.6 DCP behavior

### MiniMax-M2.7

MiniMax-M2.7 uses an ordinary GQA cache (62 layers, 8 KV heads, 128 head dim), so the cache is cleanly tensor-parallel sharded when TP divides the KV-head count. The official deployment guide also reports roughly **220 GB weights**, **240 GB cache per 1M aggregate context tokens**, and a **196K single-sequence serving limit**.

Sources:
- [`public/minimax_m2_7.json`](public/minimax_m2_7.json)
- MiniMax official vLLM deployment guide
- vLLM issue #42017
- kaitakuai 2×H200 vLLM fit/cold-start experiment

## What is not claimed yet

The tiny cache/weight errors above do **not** mean total process peak is known to the same accuracy. Full serving peak also depends on CUDA graphs, allocator fragmentation, expert-parallel routing buffers, attention kernels, speculative decoding, multimodal preprocessor caches, paged-cache rounding and prefill settings.

VRAMScout labels these terms as estimates instead of hiding them inside a fake “exact” number. The MiniMax H200 row is the first full-startup heuristic check; it is intentionally looser than the raw cache arithmetic. The next validation layer is engine-aware calibration under more pinned vLLM/SGLang commands and hardware.
