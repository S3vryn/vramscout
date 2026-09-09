from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheCalibration:
    engine: str
    family: str
    kv_dtype: str
    factor: float
    evidence_count: int
    source: str
    note: str


# These factors correct *engine page/layout overhead*, not architecture tensor
# math. They are deliberately small, auditable, and only enabled for an exact
# engine/family/dtype match. Families with strongly version-sensitive layouts
# (notably Gemma4) are intentionally excluded from automatic calibration.
_PUBLIC_CACHE_CALIBRATIONS: tuple[CacheCalibration, ...] = (
    CacheCalibration(
        engine="vllm",
        family="qwen4_exp_hybrid",
        kv_dtype="bf16",
        factor=1.029,
        evidence_count=1,
        source="https://github.com/vllm-project/vllm/issues/54559",
        note="Qwen3.8-Flash-Next TP2 public vLLM log: 9.45 GiB / 755,316 logical cache tokens.",
    ),
    CacheCalibration(
        engine="vllm",
        family="glm53_hybrid",
        kv_dtype="bf16",
        factor=1.06,
        evidence_count=1,
        source="https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark",
        note="GLM-5.3-Flash public vLLM log reports roughly 6% hybrid-page padding.",
    ),
    CacheCalibration(
        engine="vllm",
        family="nemotron_mamba_hybrid",
        kv_dtype="fp8",
        factor=1.079,
        evidence_count=2,
        source="https://github.com/sojufx/Nemotron-3.5-Lightning-30B-A3B-NVFP4-DGX-Spark-Recipe",
        note="Nemotron-3.5 Lightning FP8 cache receipts are repeatable near a 1.079 layout factor.",
    ),
)


def public_cache_calibration(engine: str, family: str, kv_dtype: str) -> CacheCalibration | None:
    for item in _PUBLIC_CACHE_CALIBRATIONS:
        if item.engine == engine and item.family == family and item.kv_dtype == kv_dtype:
            return item
    return None


def known_calibrations() -> tuple[CacheCalibration, ...]:
    return _PUBLIC_CACHE_CALIBRATIONS
