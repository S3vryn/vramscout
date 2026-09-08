from __future__ import annotations

import math

from .memory import (
    build_linear_memory_model,
    breakdown_for_context,
    resolve_kv_dtype,
    resolve_weight_dtype,
)
from .types import GPUInfo, ModelSpec, PlanResult


def plan_inference(
    gpu: GPUInfo,
    model: ModelSpec,
    context: int | None = None,
    batch_size: int = 1,
    weight_dtype: str = "auto",
    kv_dtype: str = "auto",
    safety_reserve_gib: float | None = None,
) -> PlanResult:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    resolved_weight_dtype = resolve_weight_dtype(weight_dtype, model)
    resolved_kv_dtype = resolve_kv_dtype(kv_dtype, resolved_weight_dtype)

    if context is None:
        context = min(8192, model.max_context) if model.max_context else 8192
    if context < 1:
        raise ValueError("context must be >= 1")

    linear, _, _, _ = build_linear_memory_model(
        gpu,
        model,
        batch_size,
        resolved_weight_dtype,
        resolved_kv_dtype,
        safety_reserve_gib,
    )
    remaining_for_tokens = gpu.free_gib - linear.fixed_gib
    if remaining_for_tokens <= 0:
        max_vram = 0
    elif linear.per_token_gib <= 0:
        max_vram = 0
    else:
        max_vram = max(0, math.floor(remaining_for_tokens / linear.per_token_gib))

    max_usable = min(max_vram, model.max_context) if model.max_context else max_vram
    breakdown = breakdown_for_context(
        gpu,
        model,
        context,
        batch_size,
        resolved_weight_dtype,
        resolved_kv_dtype,
        safety_reserve_gib,
    )
    spare = gpu.free_gib - breakdown.total_gib
    fits = spare >= 0 and (model.max_context is None or context <= model.max_context)

    warnings = list(model.warnings)
    if model.max_context is not None and context > model.max_context:
        warnings.append(
            f"Requested context {context:,} exceeds the model-declared limit of {model.max_context:,}."
        )
    if resolved_weight_dtype in {"int4", "int8"}:
        warnings.append(
            "Quantized weight memory includes a small packing/scale overhead estimate; exact usage depends on the quantization format and engine."
        )
    warnings.append(
        "Runtime/prefill memory is a conservative static estimate, not a measured peak. Kernel choice, CUDA graphs, allocator state and serving engine can change real usage."
    )

    return PlanResult(
        gpu=gpu,
        model=model,
        context=context,
        batch_size=batch_size,
        weight_dtype=resolved_weight_dtype,
        kv_dtype=resolved_kv_dtype,
        breakdown=breakdown,
        fits=fits,
        spare_gib=spare,
        max_context_vram=max_vram,
        max_context_usable=max_usable,
        warnings=warnings,
    )
