from __future__ import annotations

from dataclasses import dataclass

from .types import GPUInfo, MemoryBreakdown, ModelSpec

GIB = 1024**3

WEIGHT_BYTES_PER_PARAM = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.05,
    "int4": 0.55,
    "nvfp4": 0.55,
}

KV_BYTES_PER_ELEMENT = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
}

INDEXER_BYTES_PER_ELEMENT = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp4": 0.5,
}

DEFAULT_PREFILL_CHUNK_TOKENS = 8192


@dataclass(frozen=True, slots=True)
class LinearMemoryModel:
    """Legacy helper for standard linearly growing Transformer caches."""

    fixed_gib: float
    per_token_gib: float


def resolve_weight_dtype(requested: str, model: ModelSpec) -> str:
    if requested != "auto":
        return requested
    quant = (model.quantization or "").lower()
    if "nvfp4" in quant:
        return "nvfp4"
    if "fp8" in quant:
        return "fp8"
    if "4" in quant:
        return "int4"
    if "8" in quant:
        return "int8"
    if model.torch_dtype in {"fp32", "fp16", "bf16", "fp8"}:
        return model.torch_dtype
    return "bf16"


def resolve_kv_dtype(requested: str, weight_dtype: str, model: ModelSpec | None = None) -> str:
    if requested != "auto":
        return requested
    if model is not None and model.cache_kind in {"deepseek_v4_hybrid", "glm_moe_dsa"}:
        return "fp8"
    if weight_dtype == "fp32":
        return "fp32"
    return "bf16"


def resolve_indexer_dtype(requested: str, model: ModelSpec, kv_dtype: str) -> str | None:
    if model.cache_kind not in {"deepseek_v4_hybrid", "glm_moe_dsa"}:
        return None
    if requested != "auto":
        return requested
    if model.cache_kind == "deepseek_v4_hybrid":
        return "fp4" if kv_dtype == "fp8" else "bf16"
    # Public GLM-5.2 vLLM measurements show use_fp4_indexer_cache=False.
    return "bf16"


def _native_checkpoint_matches_dtype(model: ModelSpec, dtype: str) -> bool:
    quant = (model.quantization or "").lower()
    if dtype == "nvfp4":
        return "nvfp4" in quant
    if dtype == "fp8":
        return "fp8" in quant
    if dtype in {"int4", "int8"}:
        return dtype[-1] in quant and "fp" not in quant
    return not quant and model.torch_dtype == dtype


def weight_memory_gib(model: ModelSpec, dtype: str) -> float:
    if model.checkpoint_size_bytes and _native_checkpoint_matches_dtype(model, dtype):
        return model.checkpoint_size_bytes / GIB
    try:
        bpp = WEIGHT_BYTES_PER_PARAM[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported weight dtype: {dtype}") from exc
    return model.num_params * bpp / GIB


def kv_cache_per_token_gib(model: ModelSpec, batch_size: int, kv_dtype: str) -> float:
    if model.cache_kind in {"deepseek_v4_hybrid", "glm_moe_dsa"}:
        raise ValueError(f"{model.cache_kind} uses architecture-specific cache state; use cache_memory_parts_gib().")
    try:
        bytes_per = KV_BYTES_PER_ELEMENT[kv_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported KV dtype: {kv_dtype}") from exc
    elements = 2 * model.effective_kv_layers * batch_size * model.num_kv_heads * model.head_dim
    return elements * bytes_per / GIB


def recurrent_state_gib(model: ModelSpec, batch_size: int) -> float:
    return model.recurrent_state_bytes_per_batch * batch_size / GIB


def _deepseek_v4_main_entry_bytes(model: ModelSpec, kv_dtype: str) -> float:
    md = model.cache_metadata
    head_dim = int(md["main_head_dim"])
    rope_dim = int(md["rope_head_dim"])
    if kv_dtype == "fp8":
        # vLLM fp8_ds_mla: NoPE E4M3 + RoPE BF16 + UE8M0 scales.
        return (head_dim - rope_dim) + rope_dim * 2 + int(md.get("fp8_scale_bytes_per_entry", 0))
    try:
        return head_dim * KV_BYTES_PER_ELEMENT[kv_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported DeepSeek-V4 cache dtype: {kv_dtype}") from exc


def _deepseek_v4_indexer_entry_bytes(model: ModelSpec, indexer_dtype: str) -> float:
    md = model.cache_metadata
    dim = int(md["index_head_dim"])
    try:
        bpe = INDEXER_BYTES_PER_ELEMENT[indexer_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported indexer dtype: {indexer_dtype}") from exc
    value_bytes = dim * bpe
    if indexer_dtype == "fp4":
        value_bytes += int(md.get("fp4_indexer_scale_bytes_per_entry", 0))
    elif indexer_dtype == "fp8":
        value_bytes += 4
    return value_bytes


def _glm_mla_entry_bytes(model: ModelSpec, kv_dtype: str) -> float:
    md = model.cache_metadata
    latent = int(md["kv_lora_rank"])
    rope = int(md["rope_head_dim"])
    if kv_dtype == "fp8":
        # GLM-5.2 vLLM layout: 512 FP8 latent + 64 BF16 RoPE + scales.
        return latent + rope * 2 + int(md.get("fp8_mla_scale_bytes_per_entry", 0))
    try:
        return int(md["mla_latent_dim"]) * KV_BYTES_PER_ELEMENT[kv_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported GLM MLA cache dtype: {kv_dtype}") from exc


def _glm_indexer_entry_bytes(model: ModelSpec, indexer_dtype: str) -> float:
    md = model.cache_metadata
    dim = int(md["index_head_dim"])
    try:
        bpe = INDEXER_BYTES_PER_ELEMENT[indexer_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported GLM indexer dtype: {indexer_dtype}") from exc
    value_bytes = dim * bpe
    if indexer_dtype == "fp4":
        value_bytes += int(md.get("fp4_indexer_scale_bytes_per_entry", 0))
    elif indexer_dtype == "fp8":
        value_bytes += int(md.get("fp8_indexer_scale_bytes_per_entry", 0))
    return value_bytes


def cache_memory_parts_gib(
    model: ModelSpec,
    context: int,
    batch_size: int,
    kv_dtype: str,
    indexer_dtype: str | None = None,
) -> dict[str, float]:
    if context < 0:
        raise ValueError("context cannot be negative")

    if model.cache_kind == "deepseek_v4_hybrid":
        if indexer_dtype is None:
            indexer_dtype = resolve_indexer_dtype("auto", model, kv_dtype) or "bf16"
        md = model.cache_metadata
        main_bytes = _deepseek_v4_main_entry_bytes(model, kv_dtype)
        indexer_bytes = _deepseek_v4_indexer_entry_bytes(model, indexer_dtype)
        swa = int(md["sliding_window"])
        csa_ratio = int(md["csa_compress_ratio"])
        hca_ratio = int(md["hca_compress_ratio"])
        csa_layers = int(md["csa_layers"])
        hca_layers = int(md["hca_layers"])
        total_layers = csa_layers + hca_layers + int(md["sliding_only_layers"])

        csa_entries = context // csa_ratio
        hca_entries = context // hca_ratio
        swa_entries = min(context, swa)
        scale = batch_size / GIB
        return {
            "Sliding-window shared K=V": total_layers * swa_entries * main_bytes * scale,
            f"CSA compressed K=V (1/{csa_ratio})": csa_layers * csa_entries * main_bytes * scale,
            f"CSA Lightning Indexer ({indexer_dtype})": csa_layers * csa_entries * indexer_bytes * scale,
            f"HCA compressed K=V (1/{hca_ratio})": hca_layers * hca_entries * main_bytes * scale,
        }

    if model.cache_kind == "glm_moe_dsa":
        if indexer_dtype is None:
            indexer_dtype = resolve_indexer_dtype("auto", model, kv_dtype) or "bf16"
        md = model.cache_metadata
        mla_bytes = _glm_mla_entry_bytes(model, kv_dtype)
        indexer_bytes = _glm_indexer_entry_bytes(model, indexer_dtype)
        layers = model.num_layers
        groups = int(md["indexer_full_layers"])
        scale = batch_size * context / GIB
        return {
            f"MLA latent KV ({layers} layers, {kv_dtype})": layers * mla_bytes * scale,
            f"DSA IndexShare indexer ({groups} groups, {indexer_dtype})": groups * indexer_bytes * scale,
        }

    per_token = kv_cache_per_token_gib(model, batch_size, kv_dtype)
    label = "KV cache"
    if model.cache_kind == "qwen3_5_hybrid":
        label = f"Full-attention KV ({model.effective_kv_layers} layers)"
    return {label: per_token * context}


def cache_memory_gib(
    model: ModelSpec,
    context: int,
    batch_size: int,
    kv_dtype: str,
    indexer_dtype: str | None = None,
) -> float:
    return sum(cache_memory_parts_gib(model, context, batch_size, kv_dtype, indexer_dtype).values())


def prefill_scratch_gib(
    model: ModelSpec,
    context: int,
    batch_size: int,
    weight_dtype: str,
    prefill_chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS,
) -> float:
    activation_bytes = 4.0 if weight_dtype == "fp32" else 2.0
    hidden_state_buffers = 5.0
    active_tokens = min(context, max(1, prefill_chunk_tokens))
    return batch_size * model.hidden_size * activation_bytes * hidden_state_buffers * active_tokens / GIB


def prefill_scratch_per_token_gib(model: ModelSpec, batch_size: int, weight_dtype: str) -> float:
    activation_bytes = 4.0 if weight_dtype == "fp32" else 2.0
    hidden_state_buffers = 5.0
    return batch_size * model.hidden_size * activation_bytes * hidden_state_buffers / GIB


def runtime_fixed_gib(model: ModelSpec, weights_gib: float, batch_size: int) -> float:
    cuda_framework = max(0.75, 0.03 * weights_gib)
    logits = (model.vocab_size or 0) * batch_size * 4.0 / GIB
    return cuda_framework + logits


def default_safety_reserve_gib(gpu: GPUInfo) -> float:
    return max(1.0, 0.05 * gpu.total_gib)


def breakdown_for_context(
    gpu: GPUInfo,
    model: ModelSpec,
    context: int,
    batch_size: int,
    weight_dtype: str,
    kv_dtype: str,
    safety_reserve_gib: float | None = None,
    indexer_dtype: str | None = None,
    prefill_chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS,
) -> MemoryBreakdown:
    weights = weight_memory_gib(model, weight_dtype)
    fixed_runtime = runtime_fixed_gib(model, weights, batch_size)
    recurrent = recurrent_state_gib(model, batch_size)
    reserve = default_safety_reserve_gib(gpu) if safety_reserve_gib is None else safety_reserve_gib
    if reserve < 0:
        raise ValueError("Safety reserve cannot be negative.")
    parts = cache_memory_parts_gib(model, context, batch_size, kv_dtype, indexer_dtype)
    cache = sum(parts.values())
    scratch = prefill_scratch_gib(model, context, batch_size, weight_dtype, prefill_chunk_tokens)
    return MemoryBreakdown(
        weights_gib=weights,
        kv_cache_gib=cache,
        recurrent_state_gib=recurrent,
        runtime_fixed_gib=fixed_runtime,
        prefill_scratch_gib=scratch,
        safety_reserve_gib=reserve,
        cache_parts_gib=parts,
    )


def build_linear_memory_model(
    gpu: GPUInfo,
    model: ModelSpec,
    batch_size: int,
    weight_dtype: str,
    kv_dtype: str,
    safety_reserve_gib: float | None = None,
) -> tuple[LinearMemoryModel, float, float, float, float]:
    """Compatibility helper for standard caches only."""
    if model.cache_kind in {"deepseek_v4_hybrid", "glm_moe_dsa"}:
        raise ValueError(f"{model.cache_kind} requires context-aware architecture-specific planning.")
    weights = weight_memory_gib(model, weight_dtype)
    fixed_runtime = runtime_fixed_gib(model, weights, batch_size)
    recurrent = recurrent_state_gib(model, batch_size)
    reserve = default_safety_reserve_gib(gpu) if safety_reserve_gib is None else safety_reserve_gib
    if reserve < 0:
        raise ValueError("Safety reserve cannot be negative.")
    per_token = kv_cache_per_token_gib(model, batch_size, kv_dtype)
    return LinearMemoryModel(weights + recurrent + fixed_runtime + reserve, per_token), weights, recurrent, fixed_runtime, reserve
