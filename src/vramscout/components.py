from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

GIB = 1024**3


class ProfileLike(Protocol):
    family: str
    layers: int
    kv_heads: int
    head_dim: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvalContext:
    context: int
    batch: int
    kv_bytes: float
    indexer_bytes: float
    tp: int
    dcp: int


def tp_kv_heads(kv_heads: int, tp: int) -> int:
    """Per-rank KV/index head count using vLLM-style replication rules."""
    if tp <= 1:
        return kv_heads
    if kv_heads % tp == 0:
        return kv_heads // tp
    if tp % kv_heads == 0:
        return 1
    raise ValueError(f"TP={tp} cannot be mapped safely to {kv_heads} KV heads.")


class MemoryComponent(Protocol):
    name: str

    def gib(self, env: EvalContext) -> float: ...


@dataclass(frozen=True, slots=True)
class KVCache:
    """Ordinary K/V state, optionally bounded by a local/sliding window.

    ``value_head_dim`` is separate because current models such as MiMo-V2.5
    store asymmetric K and V head widths (K=192, V=128).
    """

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    value_head_dim: int | None = None
    window: int | None = None
    tp_sharded: bool = True
    dcp_sharded: bool = False

    def gib(self, env: EvalContext) -> float:
        tokens = min(env.context, self.window) if self.window else env.context
        heads = tp_kv_heads(self.kv_heads, env.tp) if self.tp_sharded else self.kv_heads
        divisor = env.dcp if self.dcp_sharded else 1
        vdim = self.head_dim if self.value_head_dim is None else self.value_head_dim
        elements = self.layers * tokens * heads * (self.head_dim + vdim) * env.batch
        return elements * env.kv_bytes / GIB / divisor


@dataclass(frozen=True, slots=True)
class LatentCache:
    """MLA/compressed latent state.

    DCP is deliberately opt-in. Pure tensor parallelism does not automatically
    divide a shared latent cache, and not every engine/family supports DCP.
    """

    name: str
    layers: int
    latent_dim: int
    compression_ratio: int = 1
    dcp_sharded: bool = False
    scale_bytes_per_entry: float = 0.0

    def gib(self, env: EvalContext) -> float:
        ratio = max(1, self.compression_ratio)
        entries = env.context // ratio
        divisor = env.dcp if self.dcp_sharded else 1
        per_entry = self.latent_dim * env.kv_bytes + self.scale_bytes_per_entry
        return self.layers * entries * per_entry * env.batch / GIB / divisor


@dataclass(frozen=True, slots=True)
class SparseIndexer:
    """Persistent sparse-attention index key state."""

    name: str
    layers: int
    dim: int
    heads: int = 1
    compression_ratio: int = 1
    tp_sharded_heads: bool = False
    dcp_sharded: bool = False
    scale_bytes_per_entry: float = 0.0
    quant_scale_bytes: float = 0.0

    def gib(self, env: EvalContext) -> float:
        ratio = max(1, self.compression_ratio)
        entries = env.context // ratio
        divisor = env.dcp if self.dcp_sharded else 1
        heads = tp_kv_heads(self.heads, env.tp) if self.tp_sharded_heads else self.heads
        scale_bytes = self.scale_bytes_per_entry
        # DeepSeek-family FP8/FP4 index pages store small scale metadata next
        # to each packed vector; BF16/FP16 do not.
        if env.indexer_bytes <= 1.0:
            scale_bytes += self.quant_scale_bytes
        per_entry = self.dim * heads * env.indexer_bytes + scale_bytes
        return self.layers * entries * per_entry * env.batch / GIB / divisor


@dataclass(frozen=True, slots=True)
class ConstantState:
    name: str
    bytes_per_batch: int

    def gib(self, env: EvalContext) -> float:
        return self.bytes_per_batch * env.batch / GIB


@dataclass(frozen=True, slots=True)
class DeepSeekSharedKV:
    name: str
    layers: int
    dim: int
    rope_dim: int
    window: int
    fp8_scale_bytes: int = 8

    def _entry_bytes(self, env: EvalContext) -> float:
        if env.kv_bytes == 1.0:
            return (self.dim - self.rope_dim) + self.rope_dim * 2 + self.fp8_scale_bytes
        return self.dim * env.kv_bytes

    def gib(self, env: EvalContext) -> float:
        return self.layers * min(env.context, self.window) * self._entry_bytes(env) * env.batch / GIB


@dataclass(frozen=True, slots=True)
class DeepSeekCompressedKV:
    name: str
    layers: int
    dim: int
    rope_dim: int
    ratio: int
    fp8_scale_bytes: int = 8

    def _entry_bytes(self, env: EvalContext) -> float:
        if env.kv_bytes == 1.0:
            return (self.dim - self.rope_dim) + self.rope_dim * 2 + self.fp8_scale_bytes
        return self.dim * env.kv_bytes

    def gib(self, env: EvalContext) -> float:
        return self.layers * (env.context // self.ratio) * self._entry_bytes(env) * env.batch / GIB


@dataclass(frozen=True, slots=True)
class ComponentGraph:
    family: str
    components: tuple[MemoryComponent, ...]

    def parts_gib(self, env: EvalContext) -> dict[str, float]:
        out: dict[str, float] = {}
        for component in self.components:
            value = component.gib(env)
            if value > 0:
                out[component.name] = out.get(component.name, 0.0) + value
        return out
