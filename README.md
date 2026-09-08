# VRAMScout

**Can this modern LLM fit on my GPU — where does the VRAM go, and how much context can I afford?**

VRAMScout inspects the **VRAM that is free right now**, reads Hugging Face model metadata without downloading checkpoint weights, then estimates weights, cache/state, runtime workspace, prefill scratch and the maximum context that fits.

> **v0.3 alpha:** architecture-aware support now includes **Qwen3.8**, **DeepSeek-V4**, and **GLM-5.2 DSA/IndexShare**, alongside ordinary decoder-only Transformers. Public accuracy checks live in [`validation/`](validation/README.md).

## Quick start

```bash
pip install -e .

# Current GPU
vramscout Qwen/Qwen3.8-27B

# Ask whether a long context fits
vramscout Qwen/Qwen3.8-27B-FP8 --context 262144 --kv-dtype fp8

# Modern sparse / compressed-cache models
vramscout deepseek-ai/DeepSeek-V4-Flash --context 1048576
vramscout zai-org/GLM-5.2-FP8 --context 1048576

# Plan for a rented GPU without owning it
vramscout Inferact/Qwen3.8-27B-NVFP4 --vram-gib 32 --kv-dtype fp8
```

Classic decoder-only models still work:

```bash
vramscout Qwen/Qwen3-8B
vramscout meta-llama/Llama-3.1-8B-Instruct --context 65536
```

## Modern caches are not all “KV = layers × tokens”

VRAMScout parses cache semantics by architecture instead of forcing every model through one old Transformer formula.

| Family | Memory state modeled |
|---|---|
| Standard Transformer | MHA / GQA / MQA KV |
| Qwen3.8 / `qwen3_5` | full-attention KV + constant GatedDeltaNet recurrent state |
| DeepSeek-V4 | shared-K=V sliding window + C4 CSA + Lightning Indexer + C128 HCA |
| GLM-5.2 `glm_moe_dsa` | compressed MLA latent + DSA IndexShare cache |

For example, Qwen3.8-27B has 64 layers but only 16 full-attention layers. Treating all 64 as ordinary attention would overestimate its long-context KV by about 4×.

DeepSeek-V4 is even less compatible with the ordinary formula: cache growth is piecewise because one branch keeps a short uncompressed window while other layers persist entries at 1/4 or 1/128 sequence rate.

GLM-5.2 uses a 576-dim MLA state and IndexShare: the released 78-layer config materializes only 21 indexer caches while the other 57 layers share them.

## Example breakdown

Architecture-specific pieces show up separately:

```text
VRAM breakdown @ 1,048,576 tokens
Model weights                              ... GiB
MLA latent KV (78 layers, fp8)             ... GiB
DSA IndexShare indexer (21 groups, bf16)   ... GiB
Cache subtotal                             ... GiB
CUDA / runtime                             ... GiB
Prefill scratch                            ... GiB
Safety reserve                             ... GiB
---------------------------------------------------
Estimated peak                             ... GiB
```

For DeepSeek-V4 the rows become SWA, CSA compressed state, CSA indexer and HCA compressed state. For Qwen3.8 they become full-attention KV plus constant GatedDeltaNet state.

## Real checkpoint bytes, not fake uniform quantization

Modern FP8/NVFP4 checkpoints are mixed precision. Attention, embeddings, vision modules, MTP heads and MoE experts do not necessarily use the same format.

When Hugging Face exposes shard metadata, VRAMScout therefore prefers the **actual sum of `.safetensors` checkpoint bytes** over a simplistic `parameters × bits/parameter` estimate.

Weight and cache precision remain independent:

```text
weights: auto / fp32 / fp16 / bf16 / fp8 / int8 / int4 / nvfp4
cache:   auto / fp32 / fp16 / bf16 / fp8
indexer: auto / bf16 / fp16 / fp8 / fp4
```

## Public validation

Current checked-in comparisons:

| Model / target | VRAMScout | Public reference | Abs. error |
|---|---:|---:|---:|
| Qwen3.8-27B BF16 weights | 51.781 GiB | 51.70 GiB | **0.16%** |
| Qwen3.8-27B FP8 weights | 28.778 GiB | 28.56 GiB | **0.76%** |
| Qwen3.8-27B NVFP4 weights | 24.587 GiB | 24.60 GiB | **0.05%** |
| DeepSeek-V4 61-layer BF16 cache @ 1M | 9.625 GiB | 9.62 GiB | **0.05%** |
| GLM-5.2 FP8 MLA + BF16 IndexShare | 55.219 KiB/token | 56.584 KiB/token | **2.41%** |

The GLM comparison uses a real vLLM startup log (`126.41 GiB` cache / `2,342,528` logical tokens), so the reference includes cache-block/layout overhead rather than merely repeating the tensor algebra.

```bash
python validation/run.py
```

See [`validation/README.md`](validation/README.md) for sources and the exact scope of each claim.

## GPU budget and max context

By default VRAMScout reads `nvidia-smi` and plans against **currently free** memory, not the GPU's marketing capacity.

```bash
vramscout Qwen/Qwen3.8-27B-FP8 --gpu-index 0
```

Or supply a hypothetical budget:

```bash
vramscout Qwen/Qwen3.8-27B-FP8 --vram-gib 48
```

The max-context solver evaluates the actual architecture-specific memory function rather than assuming every cache grows with one linear slope.

Long-prompt activation scratch is bounded by a serving-style prefill chunk (default 8192 active tokens):

```bash
vramscout Qwen/Qwen3.8-27B-FP8 --prefill-chunk 16384
```

## Current coverage

**Supported:**

- standard decoder-only Transformer caches (Llama/Qwen/Mistral/Gemma-style families);
- GQA / MQA / MHA;
- Qwen3.8 hybrid full-attention + GatedDeltaNet state;
- DeepSeek-V4 Flash/Pro-style SWA + CSA/HCA compressed cache;
- GLM-5.2 DSA/MLA + IndexShare cache;
- BF16 / FP8 / NVFP4 native checkpoint byte accounting where Hub metadata is available;
- current-GPU and hypothetical single-GPU budgets.

**Not yet fully modeled:**

- multi-GPU TP / EP / PP per-GPU weight and cache placement;
- complete vLLM/SGLang process peak including CUDA graphs and backend workspaces;
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
- [x] native checkpoint-shard accounting for FP8/NVFP4
- [x] DeepSeek-V4 CSA/HCA + compressed indexer cache
- [x] GLM-5.2 MLA/DSA + IndexShare
- [x] public architecture/cache validation corpus
- [ ] tensor/expert-parallel per-GPU planning
- [ ] engine profiles for vLLM / SGLang
- [ ] full startup-memory + max-context validation matrix
- [ ] Kimi / MiniMax current MLA/hybrid families
- [ ] local `vramscout probe` calibration

## License

MIT
