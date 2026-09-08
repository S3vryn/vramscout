from __future__ import annotations

from dataclasses import dataclass

from .types import GPUInfo

VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.92


@dataclass(frozen=True, slots=True)
class EngineBudget:
    engine: str
    memory_budget_gib: float
    startup_ok: bool
    gpu_memory_utilization: float | None = None
    outside_engine_headroom_gib: float = 0.0


def resolve_engine_budget(
    gpu: GPUInfo,
    engine: str = "generic",
    gpu_memory_utilization: float | None = None,
) -> EngineBudget:
    """Resolve the per-rank memory ceiling used by the planner.

    Generic mode plans directly against memory that is free now. vLLM instead
    requests ``total_memory * gpu_memory_utilization`` and refuses to start when
    current free memory is below that request. This mirrors vLLM's V1 worker
    startup semantics before the engine profiles its non-KV footprint.
    """
    engine = engine.lower()
    if engine == "generic":
        if gpu_memory_utilization is not None:
            raise ValueError("--gpu-memory-utilization requires --engine vllm")
        return EngineBudget(
            engine="generic",
            memory_budget_gib=gpu.free_gib,
            startup_ok=True,
            gpu_memory_utilization=None,
            outside_engine_headroom_gib=max(0.0, gpu.total_gib - gpu.free_gib),
        )

    if engine != "vllm":
        raise ValueError(f"Unsupported engine: {engine}")

    util = (
        VLLM_DEFAULT_GPU_MEMORY_UTILIZATION
        if gpu_memory_utilization is None
        else float(gpu_memory_utilization)
    )
    if not 0.0 < util <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")

    requested = gpu.total_gib * util
    startup_ok = gpu.free_gib + 1e-9 >= requested
    return EngineBudget(
        engine="vllm",
        memory_budget_gib=requested,
        startup_ok=startup_ok,
        gpu_memory_utilization=util,
        outside_engine_headroom_gib=max(0.0, gpu.total_gib - requested),
    )
