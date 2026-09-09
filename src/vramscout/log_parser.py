from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class EngineReceipt:
    engine: str
    model_loading_gib: float | None = None
    available_kv_gib: float | None = None
    active_kv_gib: float | None = None
    kv_tokens: int | None = None
    context_tokens: int | None = None
    max_concurrency: float | None = None
    cuda_graph_gib: float | None = None
    peak_activation_gib: float | None = None
    consumed_memory_gib: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _float(pattern: str, text: str, flags: int = 0) -> float | None:
    matches = re.findall(pattern, text, flags)
    return float(matches[-1]) if matches else None


def _int(pattern: str, text: str, flags: int = 0) -> int | None:
    matches = re.findall(pattern, text, flags)
    return int(matches[-1].replace(",", "")) if matches else None


def parse_engine_log(text: str) -> EngineReceipt:
    lower = text.lower()
    engine = "sglang" if "kv cache is allocated" in lower else "vllm"

    model_loading = _float(r"Model loading took\s+([0-9.]+)\s+GiB", text, re.I)
    if model_loading is None and engine == "sglang":
        model_loading = _float(r"Load weight end\..*?mem usage=([0-9.]+)\s+GB", text, re.I)

    available_kv = _float(r"Available KV cache memory:\s*([0-9.]+)\s*GiB", text, re.I)
    active_kv = _float(r"Current kv cache memory in use is\s*([0-9.]+)\s*GiB", text, re.I)

    sgl_kv_size = _float(r"KV Cache is allocated\..*?KV size:\s*([0-9.]+)\s*GB", text, re.I)
    if available_kv is None and sgl_kv_size is not None:
        available_kv = sgl_kv_size
    if active_kv is None and sgl_kv_size is not None:
        active_kv = sgl_kv_size

    kv_tokens = _int(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text, re.I)
    if kv_tokens is None:
        kv_tokens = _int(r"KV Cache is allocated\.\s*#tokens:\s*([0-9,]+)", text, re.I)

    concurrency_matches = re.findall(
        r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x",
        text,
        re.I,
    )
    context_tokens = None
    max_concurrency = None
    if concurrency_matches:
        ctx, conc = concurrency_matches[-1]
        context_tokens = int(ctx.replace(",", ""))
        max_concurrency = float(conc)

    graph = _float(r"Graph capturing finished.*?took\s*([0-9.]+)\s*GiB", text, re.I)
    graph_pool = _float(r"CUDA graph pool memory:\s*([0-9.]+)\s*GiB", text, re.I)
    if graph_pool is not None:
        graph = graph_pool

    activation = _float(r"([0-9.]+)\s*GiB for peak activation", text, re.I)
    consumed = _float(r"Actual usage is\s*([0-9.]+)\s*GiB for consumed memory", text, re.I)

    return EngineReceipt(
        engine=engine,
        model_loading_gib=model_loading,
        available_kv_gib=available_kv,
        active_kv_gib=active_kv,
        kv_tokens=kv_tokens,
        context_tokens=context_tokens,
        max_concurrency=max_concurrency,
        cuda_graph_gib=graph,
        peak_activation_gib=activation,
        consumed_memory_gib=consumed,
    )


def parse_log_file(path: str | Path) -> EngineReceipt:
    return parse_engine_log(Path(path).expanduser().read_text(encoding="utf-8", errors="replace"))
