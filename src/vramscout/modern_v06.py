from __future__ import annotations

from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .engine import resolve_engine_budget
from .gpu import get_gpu_group
from . import modern_core as base


def _selected_checkpoint_size_bytes(model_ref: str, revision: str | None) -> int | None:
    """Return one logical safetensors checkpoint, avoiding duplicate export formats.

    Some current repos publish both ``model-*.safetensors`` and
    ``consolidated-*.safetensors`` for the same weights. Summing every
    safetensors file would double-count residency.
    """
    if Path(model_ref).expanduser().exists():
        return None
    try:
        info = HfApi().model_info(model_ref, revision=revision, files_metadata=True)
    except Exception:
        return None
    files: list[tuple[str, int]] = []
    for f in getattr(info, "siblings", None) or []:
        name, size = getattr(f, "rfilename", None), getattr(f, "size", None)
        if isinstance(name, str) and name.endswith(".safetensors") and isinstance(size, int) and size > 0:
            files.append((name, size))
    if not files:
        return None

    def pick(pred) -> int | None:
        chosen = [size for name, size in files if pred(name)]
        return sum(chosen) if chosen else None

    # Standard HF shards are preferred; consolidated exports are usually aliases.
    return (
        pick(lambda n: Path(n).name == "model.safetensors" or Path(n).name.startswith("model-"))
        or pick(lambda n: Path(n).name == "consolidated.safetensors" or Path(n).name.startswith("consolidated-"))
        or sum(size for _, size in files)
    )


def _better_quantization(model_ref: str, root: dict[str, Any], text: dict[str, Any]) -> str | None:
    q = root.get("quantization_config")
    if not isinstance(q, dict):
        q = text.get("quantization_config") if isinstance(text.get("quantization_config"), dict) else {}
    blob = (str(q) + " " + model_ref).lower()
    method = str(q.get("quant_method") or q.get("quantization_method") or "").lower()
    groups = q.get("config_groups") if isinstance(q.get("config_groups"), dict) else {}
    bits: set[int] = set()
    floatish = False
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        w = group.get("weights")
        if not isinstance(w, dict):
            continue
        if isinstance(w.get("num_bits"), int):
            bits.add(w["num_bits"])
        if "float" in str(w.get("type", "")).lower():
            floatish = True
    if "nvfp4" in blob or "mxfp4" in blob or (4 in bits and floatish and "modelopt" in method):
        return "nvfp4"
    if "fp8" in blob or "e4m3" in blob or "mxfp8" in blob or (8 in bits and floatish):
        return "fp8"
    if q.get("load_in_4bit") or q.get("bits") == 4 or 4 in bits:
        return "int4"
    if q.get("load_in_8bit") or q.get("bits") == 8 or 8 in bits:
        return "int8"
    return base._quantization(root) or base._quantization(text)


def inspect_modern_model(model_ref: str, revision: str | None = None) -> base.ModernProfile:
    p = base.inspect_modern_model(model_ref, revision)
    root = base._load_config(model_ref, revision)
    text = root.get("text_config") if isinstance(root.get("text_config"), dict) else root
    selected = _selected_checkpoint_size_bytes(model_ref, revision)
    if selected:
        p.checkpoint_size_bytes = selected
    p.quantization = _better_quantization(model_ref, root, text)

    # Kimi K3 exposes decomposed q/k dimensions; avoid deriving an invalid
    # head size from hidden_size / q_heads when that ratio is non-integral.
    if p.family == "kimi_k3_hybrid":
        nope = base._first_int(text, "qk_nope_head_dim")
        rope = base._first_int(text, "qk_rope_head_dim")
        if nope and rope:
            p.head_dim = nope + rope
    return p


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
) -> base.ModernPlan:
    if tp < 1 or dcp < 1 or dcp > tp or tp % dcp != 0:
        raise ValueError("Require tp >= 1 and dcp to divide tp.")
    if batch_size < 1 or prefill_chunk < 1:
        raise ValueError("batch-size and prefill-chunk must be >= 1")

    p = inspect_modern_model(model_ref, revision)
    gpu = get_gpu_group(start_index=gpu_index, count=tp, vram_gib=vram_gib)
    budget = resolve_engine_budget(gpu, engine=engine, gpu_memory_utilization=gpu_memory_utilization)
    wd = base._resolve_weight_dtype(weight_dtype, p)
    kd = base._resolve_cache_dtype(kv_dtype, p)
    if wd not in base.WEIGHT_BPP and not (p.checkpoint_size_bytes and wd == p.quantization):
        raise ValueError(f"Unsupported weight dtype: {wd}")
    if kd not in base.CACHE_BPE or indexer_dtype not in base.CACHE_BPE:
        raise ValueError("Unsupported cache/indexer dtype")
    ctx = context or min(8192, p.max_context or 8192)
    if ctx < 1:
        raise ValueError("context must be >= 1")
    reserve = reserve_gib if reserve_gib is not None else max(1.0, 0.05 * gpu.total_gib)
    if reserve < 0:
        raise ValueError("reserve-gib cannot be negative")
    weights = base._weight_gib(p, wd, tp)

    def breakdown(c: int) -> tuple[dict[str, float], float]:
        parts = {"Model weights / rank": weights}
        parts.update(base._cache_parts(p, c, batch_size, kd, indexer_dtype, tp, dcp))
        parts["CUDA / runtime"] = base._runtime_gib(weights)
        parts["Prefill scratch"] = base._prefill_gib(p, c, batch_size, wd, prefill_chunk)
        parts["Safety reserve"] = reserve
        return parts, sum(parts.values())

    def ok(c: int) -> bool:
        return breakdown(c)[1] <= budget.memory_budget_gib

    upper = p.max_context or 16_777_216
    if not budget.startup_ok or not ok(1):
        max_ctx = 0
    elif ok(upper):
        max_ctx = upper
    else:
        lo, hi = 1, upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        max_ctx = lo

    parts, total = breakdown(ctx)
    fits = budget.startup_ok and total <= budget.memory_budget_gib and (p.max_context is None or ctx <= p.max_context)
    notes = list(p.notes)
    if p.has_vision:
        notes.append("Vision checkpoint weights are included; request-dependent image token/activation memory is not modeled exactly yet.")
    if p.has_audio:
        notes.append("Audio checkpoint weights are included; request-dependent audio encoder activation memory is not modeled exactly yet.")
    if p.family == "nemotron_mamba_hybrid":
        notes.append("Mamba-2 recurrent-state math is architecture-derived; backend alignment and kernel workspaces remain runtime overhead.")
    notes.append("Native checkpoint shard bytes are preferred when available; CUDA graphs/workspaces remain a static preflight estimate.")

    return base.ModernPlan(
        profile=p,
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
        max_context_usable=min(max_ctx, p.max_context) if p.max_context else max_ctx,
        notes=notes,
    )
