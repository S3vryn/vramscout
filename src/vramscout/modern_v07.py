from __future__ import annotations

from .calibration import public_cache_calibration
from .component_registry import build_component_graph
from .components import EvalContext
from .engine import resolve_engine_budget
from .gpu import get_gpu_group
from .profile_fixups import apply_profile_fixups
from . import modern_core as base
from .modern_v06 import inspect_modern_model


def _runtime_gib(weight_rank_gib: float, family: str) -> float:
    """Conservative static runtime estimate kept separate from cache math."""
    family_floor = {
        "qwen4_exp_hybrid": 1.5,
        "deepseek_v4_compressed": 1.5,
        "glm53_hybrid": 1.5,
        "kimi_k3_hybrid": 1.5,
        "standard_sparse_index": 1.25,
        "nemotron_mamba_hybrid": 1.25,
    }.get(family, 0.75)
    return max(family_floor, 0.03 * weight_rank_gib)


def _prefill_gib(profile, context: int, batch: int, dtype: str, chunk: int) -> float:
    bytes_ = 4 if dtype == "fp32" else 2
    return min(context, chunk) * batch * profile.hidden_size * bytes_ * 5 / base.GIB


def _dcp_supported(profile) -> bool:
    if profile.family == "kimi_k3_hybrid":
        return True
    return profile.family == "mla" and (
        profile.model_type == "kimi_k2" or profile.root_model_type == "kimi_k2"
    )


def plan_modern(
    model_ref: str,
    *,
    revision: str | None = None,
    context: int | None = None,
    batch_size: int = 1,
    weight_dtype: str = "auto",
    kv_dtype: str = "auto",
    indexer_dtype: str = "bf16",
    tp: int = 1,
    dcp: int = 1,
    gpu_index: int = 0,
    vram_gib: float | None = None,
    reserve_gib: float | None = None,
    prefill_chunk: int = 8192,
    engine: str = "generic",
    gpu_memory_utilization: float | None = None,
    calibration: str = "public",
) -> base.ModernPlan:
    if tp < 1 or dcp < 1 or dcp > tp or tp % dcp != 0:
        raise ValueError("Require tp >= 1 and dcp to divide tp.")
    if batch_size < 1 or prefill_chunk < 1:
        raise ValueError("batch-size and prefill-chunk must be >= 1")
    if calibration not in {"none", "public"}:
        raise ValueError("calibration must be one of: none, public")

    profile = inspect_modern_model(model_ref, revision)
    profile = apply_profile_fixups(profile, model_ref, revision)
    if dcp > 1 and not _dcp_supported(profile):
        raise ValueError(
            f"DCP={dcp} is not publicly validated for {profile.family}; "
            "VRAMScout fails closed instead of silently dividing its cache."
        )
    graph = build_component_graph(profile)
    gpu = get_gpu_group(start_index=gpu_index, count=tp, vram_gib=vram_gib)
    budget = resolve_engine_budget(gpu, engine=engine, gpu_memory_utilization=gpu_memory_utilization)

    wd = base._resolve_weight_dtype(weight_dtype, profile)
    kd = base._resolve_cache_dtype(kv_dtype, profile)
    if wd not in base.WEIGHT_BPP and not (profile.checkpoint_size_bytes and wd == profile.quantization):
        raise ValueError(f"Unsupported weight dtype: {wd}")
    if kd not in base.CACHE_BPE or indexer_dtype not in base.CACHE_BPE:
        raise ValueError("Unsupported cache/indexer dtype")

    ctx = context or min(8192, profile.max_context or 8192)
    if ctx < 1:
        raise ValueError("context must be >= 1")

    # Generic planning uses an explicit safety reserve. vLLM already places
    # gpu_memory_utilization outside its own allocation budget; subtracting an
    # additional default 5% here would double-count headroom. Users can still
    # request an inner reserve explicitly with --reserve-gib.
    if reserve_gib is None:
        reserve = 0.0 if engine == "vllm" else max(1.0, 0.05 * gpu.total_gib)
    else:
        reserve = reserve_gib
    if reserve < 0:
        raise ValueError("reserve-gib cannot be negative")

    weights = base._weight_gib(profile, wd, tp)
    cache_cal = (
        public_cache_calibration(engine, profile.family, kd)
        if calibration == "public"
        else None
    )

    def breakdown(c: int) -> tuple[dict[str, float], float]:
        env = EvalContext(
            context=c,
            batch=batch_size,
            kv_bytes=base.CACHE_BPE[kd],
            indexer_bytes=base.CACHE_BPE[indexer_dtype],
            tp=tp,
            dcp=dcp,
        )
        state_parts = graph.parts_gib(env)
        parts = {"Model weights / rank": weights}
        parts.update(state_parts)
        if cache_cal is not None and cache_cal.factor != 1.0:
            raw_state = sum(state_parts.values())
            overhead = raw_state * (cache_cal.factor - 1.0)
            if overhead > 0:
                parts[f"{engine} cache-layout calibration"] = overhead
        parts["CUDA / runtime"] = _runtime_gib(weights, profile.family)
        parts["Prefill scratch"] = _prefill_gib(profile, c, batch_size, wd, prefill_chunk)
        parts["Safety reserve"] = reserve
        return parts, sum(parts.values())

    def fits_context(c: int) -> bool:
        return breakdown(c)[1] <= budget.memory_budget_gib

    upper = profile.max_context or 16_777_216
    if not budget.startup_ok or not fits_context(1):
        max_ctx = 0
    elif fits_context(upper):
        max_ctx = upper
    else:
        lo, hi = 1, upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits_context(mid):
                lo = mid
            else:
                hi = mid - 1
        max_ctx = lo

    parts, total = breakdown(ctx)
    fits = budget.startup_ok and total <= budget.memory_budget_gib and (
        profile.max_context is None or ctx <= profile.max_context
    )
    notes = list(profile.notes)
    notes.append(
        f"Cache/state is composed from {len(graph.components)} reusable memory component(s), not a model-name-specific monolithic formula."
    )
    if cache_cal is not None:
        notes.append(
            f"Applied public {engine} cache-layout calibration ×{cache_cal.factor:.3f} "
            f"({cache_cal.evidence_count} receipt(s)): {cache_cal.source}"
        )
    if engine == "vllm" and reserve_gib is None:
        notes.append(
            "Default inner safety reserve is 0 in vLLM mode because gpu_memory_utilization already leaves memory outside the engine budget."
        )
    if profile.has_vision:
        notes.append(
            "Vision checkpoint residency is included; request-dependent image-token and encoder activation memory is not yet modeled exactly."
        )
    if profile.has_audio:
        notes.append(
            "Audio checkpoint residency is included; request-dependent audio encoder activation memory is not yet modeled exactly."
        )
    notes.append(
        "Architecture state is config-derived; public calibration only represents observed engine page/layout overhead. CUDA graphs/workspaces remain a separate static estimate."
    )

    return base.ModernPlan(
        profile=profile,
        gpu_name=gpu.name,
        gpu_total_gib=gpu.total_gib,
        gpu_free_gib=gpu.free_gib,
        engine=engine,
        memory_budget_gib=budget.memory_budget_gib,
        startup_ok=budget.startup_ok,
        tp=tp,
        dcp=dcp,
        context=ctx,
        weight_dtype=wd,
        kv_dtype=kd,
        indexer_dtype=indexer_dtype,
        parts_gib=parts,
        total_gib=total,
        spare_gib=budget.memory_budget_gib - total,
        fits=fits,
        max_context_vram=max_ctx,
        max_context_usable=min(max_ctx, profile.max_context) if profile.max_context else max_ctx,
        notes=notes,
    )
