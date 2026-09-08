# VRAMScout

**Can this modern Hugging Face model fit on the GPU I have right now — and how much context can I afford?**

VRAMScout detects current NVIDIA VRAM, reads Hugging Face model metadata **without downloading checkpoint weights**, and breaks inference memory into weights, KV cache, recurrent/hybrid state, runtime workspace and safety reserve.

> **v0.2 alpha:** Qwen3.8 hybrid attention is now supported, including the 27B BF16 / FP8 / NVFP4 checkpoints. Public validation lives in [`validation/`](validation/README.md).

## Quick start

```bash
pip install -e .

vramscout Qwen/Qwen3.8-27B
vramscout Qwen/Qwen3.8-27B --context 262144 --kv-dtype fp8
vramscout Qwen/Qwen3.8-27B-FP8 --kv-dtype fp8
vramscout Inferact/Qwen3.8-27B-NVFP4 --vram-gib 32 --kv-dtype fp8
```

Classic decoder-only models still work as before:

```bash
vramscout Qwen/Qwen3-8B
vramscout meta-llama/Llama-3.1-8B-Instruct --context 65536
```

## What it reports

For Qwen3.8 the output is architecture-aware rather than treating all 64 layers as ordinary KV attention:

```text
Model
Architecture              qwen3_5_text
Cache architecture         qwen3_5_hybrid · 16 KV + 48 recurrent layers
Vision encoder             included in checkpoint weights

VRAM breakdown @ 262,144 tokens
Model weights                         ... GiB
KV cache (16 full-attn layers)        ... GiB
GatedDeltaNet state (48 layers)       ... GiB
CUDA / runtime                        ... GiB
Prefill scratch                       ... GiB
Safety reserve                        ... GiB
```

## Why the Qwen3.8 case matters

`Qwen/Qwen3.8-27B` is a 64-layer hybrid model: **16 full-attention layers + 48 linear-attention layers**. The full-attention layers use a normal context-growing KV cache; the linear layers keep constant GatedDeltaNet state.

For FP8 KV:

```text
KV/token = 2 × 16 layers × 4 KV heads × 256 head_dim × 1 byte
         = 32 KiB/token
```

Treating all 64 layers as normal attention would overestimate its KV memory by 4×.

## Native checkpoint bytes, not fake uniform quantization

For Hugging Face repositories VRAMScout asks the Hub for safetensors metadata and file sizes only. When a checkpoint is natively quantized (for example FP8 or NVFP4), it prefers the **actual sum of safetensors shard bytes** over `parameter_count × guessed bytes/parameter`.

That matters because modern checkpoints often keep embeddings, attention, vision modules, MTP heads or selected linears at a different precision from the headline quantization format.

Current public Qwen3.8 validation:

| Variant | VRAMScout | Public vLLM reference | Error |
|---|---:|---:|---:|
| BF16 | 51.781 GiB | 51.70 GiB | 0.16% |
| FP8 | 28.778 GiB | 28.56 GiB | 0.76% |
| Inferact NVFP4 | 24.587 GiB | 24.60 GiB | 0.05% |

See [`validation/README.md`](validation/README.md) for sources and methodology.

## Memory model

For ordinary full-attention layers:

```text
KV = 2 × full_attention_layers × batch × context × kv_heads × head_dim × bytes
```

For Qwen3.8, VRAMScout additionally accounts for the fixed GatedDeltaNet recurrent + causal-convolution state from the public Transformers implementation.

Weights and KV/state arithmetic are separated from runtime terms on purpose. CUDA graphs, allocator fragmentation, hybrid-cache paging and serving-engine workspaces are not universal constants.

## GPU budget

By default VRAMScout reads the GPU that exists **right now** through `nvidia-smi`, including currently used/free memory.

```bash
vramscout Qwen/Qwen3.8-27B --gpu-index 0
```

Or plan against a hypothetical/rented GPU:

```bash
vramscout Qwen/Qwen3.8-27B-FP8 --vram-gib 48 --kv-dtype fp8
```

## Precision

Weight and cache precision are independent:

```text
weights: auto / fp32 / fp16 / bf16 / fp8 / int8 / int4 / nvfp4
KV:      auto / fp32 / fp16 / bf16 / fp8
```

Quantized weights do **not** imply quantized KV cache.

## Current architecture coverage

Supported now:

- standard decoder-only Transformer KV caches (Llama/Qwen/Mistral/Gemma-style families);
- GQA / MQA / MHA;
- Qwen3.5/Qwen3.8 `qwen3_5` hybrid stack with full attention + GatedDeltaNet linear attention;
- native BF16/FP8/NVFP4 checkpoint byte accounting when Hub shard metadata is available;
- conservative sliding-window fallback.

Still fail-closed / incomplete:

- DeepSeek V2/V3/V4 MLA/CSA/HCA;
- GLM-5.x DSA/MLA-style sparse attention;
- Mamba/Jamba/recurrent architectures;
- engine-specific vLLM/SGLang hybrid-cache paging;
- multi-GPU tensor/expert parallel planning;
- exact multimodal activation memory beyond checkpoint weights.

The policy is deliberate: **unsupported modern state semantics should error rather than silently reuse the wrong Transformer formula.**

## Validation

Public reference cases are checked into the repo:

```bash
python validation/run.py
```

The next validation layer will compare complete vLLM/SGLang peak memory and max-context boundaries, not only weights/raw cache math.

## Development

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e . pytest
pytest -q
```

## Roadmap

- [x] Qwen3.8 hybrid full-attention + linear-attention cache accounting
- [x] native checkpoint-shard byte accounting for FP8/NVFP4
- [x] public weight/cache validation dataset
- [ ] vLLM / SGLang engine profiles and measured peak-memory validation
- [ ] tensor/expert-parallel per-GPU planning
- [ ] DeepSeek-V4 CSA/HCA + MoE
- [ ] GLM-5.2 sparse/DSA + MoE
- [ ] Kimi / MiniMax modern MLA/hybrid families
- [ ] local `vramscout probe` calibration

## License

MIT
