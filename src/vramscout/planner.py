from __future__ import annotations

from .memory import (
    DEFAULT_PREFILL_CHUNK_TOKENS,
    breakdown_for_context,
    resolve_indexer_dtype,
    resolve_kv_dtype,
    resolve_weight_dtype,
)
from .types import GPUInfo, ModelSpec, PlanResult


def _memory_fits(
    gpu: GPUInfo,
    model: ModelSpec,
    context: int,
    batch_size: int,
    weight_dtype: str,
    kv_dtype: str,
    indexer_dtype: str | None,
    safety_reserve_gib: float | None,
    prefill_chunk_tokens: int,
) -> bool:
    b = breakdown_for_context(
        gpu,
        model,
        context,
        batch_size,
        weight_dtype,
        kv_dtype,
        safety_reserve_gib,
        indexer_dtype=indexer_dtype,
        prefill_chunk_tokens=prefill_chunk_tokens,
    )
    return b.total_gib <= gpu.free_gib


def _max_context_by_vram(
    gpu: GPUInfo,
    model: ModelSpec,
    batch_size: int,
    weight_dtype: str,
    kv_dtype: str,
    indexer_dtype: str | None,
    safety_reserve_gib: float | None,
    prefill_chunk_tokens: int,
) -> int:
    if not _memory_fits(
        gpu, model, 1, batch_size, weight_dtype, kv_dtype, indexer_dtype,
        safety_reserve_gib, prefill_chunk_tokens,
    ):
        return 0

    if model.max_context is not None:
        hi = model.max_context
        if _memory_fits(
            gpu, model, hi, batch_size, weight_dtype, kv_dtype, indexer_dtype,
            safety_reserve_gib, prefill_chunk_tokens,
        ):
            return hi
    else:
        hi = 8192
        cap = 16_777_216
        while hi < cap and _memory_fits(
            gpu, model, hi, batch_size, weight_dtype, kv_dtype, indexer_dtype,
            safety_reserve_gib, prefill_chunk_tokens,
        ):
            hi *= 2
        hi = min(hi, cap)

    lo = 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _memory_fits(
            gpu, model, mid, batch_size, weight_dtype, kv_dtype, indexer_dtype,
            safety_reserve_gib, prefill_chunk_tokens,
        ):
            lo = mid
        else:
            hi = mid - 1
    return lo


def plan_inference(
    gpu: GPUInfo,
    model: ModelSpec,
    context: int | None = None,
    batch_size: int = 1,
    weight_dtype: str = "auto",
    kv_dtype: str = "auto",
    indexer_dtype: str = "auto",
    safety_reserve_gib: float | None = None,
    prefill_chunk_tokens: int = DEFAULT_PREFILL_CHUNK_TOKENS,
) -> PlanResult:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if prefill_chunk_tokens < 1:
        raise ValueError("prefill_chunk_tokens must be >= 1")

    resolved_weight_dtype = resolve_weight_dtype(weight_dtype, model)
    resolved_kv_dtype = resolve_kv_dtype(kv_dtype, resolved_weight_dtype, model)
    resolved_indexer_dtype = resolve_indexer_dtype(indexer_dtype, model, resolved_kv_dtype)

    if context is None:
        context = min(8192, model.max_context) if model.max_context else 8192
    if context < 1:
        raise ValueError("context must be >= 1")

    max_vram = _max_context_by_vram(
        gpu, model, batch_size, resolved_weight_dtype, resolved_kv_dtype,
        resolved_indexer_dtype, safety_reserve_gib, prefill_chunk_tokens,
    )
    max_usable = min(max_vram, model.max_context) if model.max_context else max_vram

    breakdown = breakdown_for_context(
        gpu,
        model,
        context,
        batch_size,
        resolved_weight_dtype,
        resolved_kv_dtype,
        safety_reserve_gib,
        indexer_dtype=resolved_indexer_dtype,
        prefill_chunk_tokens=prefill_chunk_tokens,
    )
    spare = gpu.free_gib - breakdown.total_gib
    fits = spare >= 0 and (model.max_context is None or context <= model.max_context)

    warnings = list(model.warnings)
    if model.max_context is not None and context > model.max_context:
        warnings.append(
            f"Requested context {context:,} exceeds the model-declared limit of {model.max_context:,}."
        )
    if resolved_weight_dtype in {"int4", "int8", "fp8", "nvfp4"}:
        warnings.append(
            "Quantized weight memory uses native checkpoint shard bytes when available; otherwise it falls back to a format-aware estimate. Exact loaded VRAM still depends on the serving engine."
        )
    if model.cache_kind == "deepseek_v4_hybrid":
        warnings.append(
            f"DeepSeek-V4 cache uses {resolved_kv_dtype} shared-KV storage and {resolved_indexer_dtype} CSA indexer storage; compressor residual buffers/paged allocator fragmentation remain engine overhead."
        )
    elif model.cache_kind == "glm_moe_dsa":
        warnings.append(
            f"GLM DSA cache uses {resolved_kv_dtype} MLA latent storage plus {resolved_indexer_dtype} IndexShare state; paged-cache block rounding and backend-specific layouts remain engine overhead."
        )
    warnings.append(
        f"Prefill scratch is bounded to {prefill_chunk_tokens:,} active tokens (serving-style chunked prefill), not the full prompt length."
    )
    warnings.append(
        "Runtime/prefill memory is a static estimate, not a measured peak. CUDA graphs, allocator state, tensor/expert parallelism and serving-engine kernels can change real usage."
    )

    return PlanResult(
        gpu=gpu,
        model=model,
        context=context,
        batch_size=batch_size,
        weight_dtype=resolved_weight_dtype,
        kv_dtype=resolved_kv_dtype,
        indexer_dtype=resolved_indexer_dtype,
        breakdown=breakdown,
        fits=fits,
        spare_gib=spare,
        max_context_vram=max_vram,
        max_context_usable=max_usable,
        warnings=warnings,
    )
