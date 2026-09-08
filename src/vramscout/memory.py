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


@dataclass(frozen=True, slots=True)
class LinearMemoryModel:
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


def resolve_kv_dtype(requested: str, weight_dtype: str) -> str:
    if requested != "auto":
        return requested
    if weight_dtype == "fp32":
        return "fp32"
    # Keep cache conservative unless the user explicitly asks for FP8 KV.
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
    # Native checkpoint byte size is a better predictor for mixed quantization
    # formats (FP8/NVFP4) than pretending every parameter uses one bit-width.
    if model.checkpoint_size_bytes and _native_checkpoint_matches_dtype(model, dtype):
        return model.checkpoint_size_bytes / GIB
    try:
        bpp = WEIGHT_BYTES_PER_PARAM[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported weight dtype: {dtype}") from exc
    return model.num_params * bpp / GIB


def kv_cache_per_token_gib(model: ModelSpec, batch_size: int, kv_dtype: str) -> float:
    try:
        bytes_per = KV_BYTES_PER_ELEMENT[kv_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported KV dtype: {kv_dtype}") from exc
    elements = 2 * model.effective_kv_layers * batch_size * model.num_kv_heads * model.head_dim
    return elements * bytes_per / GIB


def recurrent_state_gib(model: ModelSpec, batch_size: int) -> float:
    return model.recurrent_state_bytes_per_batch * batch_size / GIB


def prefill_scratch_per_token_gib(model: ModelSpec, batch_size: int, weight_dtype: str) -> float:
    activation_bytes = 4.0 if weight_dtype == "fp32" else 2.0
    # Fused/SDPA-style inference needs a handful of hidden-state-sized buffers.
    # 5 is calibrated to stay conservative without double-counting cache memory.
    hidden_state_buffers = 5.0
    return batch_size * model.hidden_size * activation_bytes * hidden_state_buffers / GIB


def runtime_fixed_gib(model: ModelSpec, weights_gib: float, batch_size: int) -> float:
    cuda_framework = max(0.75, 0.03 * weights_gib)
    logits = (model.vocab_size or 0) * batch_size * 4.0 / GIB
    return cuda_framework + logits


def default_safety_reserve_gib(gpu: GPUInfo) -> float:
    return max(1.0, 0.05 * gpu.total_gib)


def build_linear_memory_model(
    gpu: GPUInfo,
    model: ModelSpec,
    batch_size: int,
    weight_dtype: str,
    kv_dtype: str,
    safety_reserve_gib: float | None = None,
) -> tuple[LinearMemoryModel, float, float, float, float]:
    weights = weight_memory_gib(model, weight_dtype)
    fixed_runtime = runtime_fixed_gib(model, weights, batch_size)
    recurrent = recurrent_state_gib(model, batch_size)
    reserve = default_safety_reserve_gib(gpu) if safety_reserve_gib is None else safety_reserve_gib
    if reserve < 0:
        raise ValueError("Safety reserve cannot be negative.")
    per_token = kv_cache_per_token_gib(model, batch_size, kv_dtype) + prefill_scratch_per_token_gib(
        model, batch_size, weight_dtype
    )
    return LinearMemoryModel(weights + recurrent + fixed_runtime + reserve, per_token), weights, recurrent, fixed_runtime, reserve


def breakdown_for_context(
    gpu: GPUInfo,
    model: ModelSpec,
    context: int,
    batch_size: int,
    weight_dtype: str,
    kv_dtype: str,
    safety_reserve_gib: float | None = None,
) -> MemoryBreakdown:
    _, weights, recurrent, fixed_runtime, reserve = build_linear_memory_model(
        gpu,
        model,
        batch_size,
        weight_dtype,
        kv_dtype,
        safety_reserve_gib,
    )
    kv = kv_cache_per_token_gib(model, batch_size, kv_dtype) * context
    scratch = prefill_scratch_per_token_gib(model, batch_size, weight_dtype) * context
    return MemoryBreakdown(
        weights_gib=weights,
        kv_cache_gib=kv,
        recurrent_state_gib=recurrent,
        runtime_fixed_gib=fixed_runtime,
        prefill_scratch_gib=scratch,
        safety_reserve_gib=reserve,
    )
