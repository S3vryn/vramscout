# Public validation

VRAMScout keeps public reference numbers next to the estimator so accuracy claims can be audited instead of hand-waved.

## Qwen3.8-27B — 2026-09-08

Sources:

- Qwen config: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- BF16 checkpoint files: https://huggingface.co/Qwen/Qwen3.8-27B/tree/main
- FP8 checkpoint files: https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/main
- Inferact NVFP4 checkpoint files: https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/tree/main
- vLLM deployment recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-27B

### Weight footprint

VRAMScout v0.2 prefers the sum of actual `.safetensors` shard bytes for native quantized checkpoints rather than pretending every parameter has one uniform bit-width.

| Variant | VRAMScout from checkpoint bytes | Public vLLM reference | Abs. error |
|---|---:|---:|---:|
| BF16 | 51.781 GiB | 51.70 GiB | 0.16% |
| FP8 | 28.778 GiB | 28.56 GiB | 0.76% |
| Inferact NVFP4 | 24.587 GiB | 24.60 GiB | 0.05% |

The FP8 vLLM reference is `14.28 GiB/GPU` at TP=2, doubled here only to compare whole-checkpoint weight footprint. Engine-level per-GPU residency can differ because of tensor-parallel sharding and replicated modules.

### Hybrid cache math

The official config declares 64 decoder layers with `full_attention_interval=4`: 16 full-attention layers and 48 linear-attention layers. Only the 16 full-attention layers grow an ordinary KV cache. Therefore:

```text
FP8 KV  = 2 × 16 × 4 × 256 × 1 byte = 32 KiB/token
BF16 KV = 2 × 16 × 4 × 256 × 2 byte = 64 KiB/token
```

The linear-attention layers keep a constant GatedDeltaNet recurrent/conv state rather than an ordinary per-token KV cache in the Transformers cache model.

Run the checked-in report:

```bash
python validation/run.py
```

## What is not claimed yet

These numbers validate checkpoint-weight accounting and architecture/cache decomposition. A complete vLLM/SGLang peak-memory validator also needs engine-specific CUDA-graph, allocator, hybrid-cache paging, tensor-parallel and prefill-workspace models. Those are tracked separately rather than folded into an unjustified single fudge factor.
