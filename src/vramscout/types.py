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

    # Modern/hybrid architecture metadata. For an ordinary Transformer,
    # kv_layers == num_layers and recurrent_state_bytes_per_batch == 0.
    cache_kind: str = "standard"
    kv_layers: int | None = None
    recurrent_layers: int = 0
    recurrent_state_bytes_per_batch: int = 0
    recurrent_state_label: str | None = None
    has_vision_encoder: bool = False
    checkpoint_size_bytes: int | None = None
    checkpoint_size_source: str | None = None
    cache_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_kv_layers(self) -> int:
        return self.num_layers if self.kv_layers is None else self.kv_layers


@dataclass(slots=True)
class MemoryBreakdown:
    weights_gib: float
    kv_cache_gib: float
    recurrent_state_gib: float
    runtime_fixed_gib: float
    prefill_scratch_gib: float
    safety_reserve_gib: float
    cache_parts_gib: dict[str, float] = field(default_factory=dict)

    @property
    def total_gib(self) -> float:
        return (
            self.weights_gib
            + self.kv_cache_gib
            + self.recurrent_state_gib
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
    indexer_dtype: str | None
    breakdown: MemoryBreakdown
    fits: bool
    spare_gib: float
    max_context_vram: int
    max_context_usable: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["breakdown"]["total_gib"] = self.breakdown.total_gib
        data["model"]["effective_kv_layers"] = self.model.effective_kv_layers
        return data
