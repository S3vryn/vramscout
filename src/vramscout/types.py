from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class GPUInfo:
    index: int
    name: str
    total_gib: float
    used_gib: float
    free_gib: float


@dataclass(slots=True)
class ModelSpec:
    model_id: str
    model_type: str
    num_params: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_context: int | None
    vocab_size: int | None = None
    torch_dtype: str | None = None
    quantization: str | None = None
    sliding_window: int | None = None
    parameter_source: str = "unknown"
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryBreakdown:
    weights_gib: float
    kv_cache_gib: float
    runtime_fixed_gib: float
    prefill_scratch_gib: float
    safety_reserve_gib: float

    @property
    def total_gib(self) -> float:
        return (
            self.weights_gib
            + self.kv_cache_gib
            + self.runtime_fixed_gib
            + self.prefill_scratch_gib
            + self.safety_reserve_gib
        )


@dataclass(slots=True)
class PlanResult:
    gpu: GPUInfo
    model: ModelSpec
    context: int
    batch_size: int
    weight_dtype: str
    kv_dtype: str
    breakdown: MemoryBreakdown
    fits: bool
    spare_gib: float
    max_context_vram: int
    max_context_usable: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["breakdown"]["total_gib"] = self.breakdown.total_gib
        return data
