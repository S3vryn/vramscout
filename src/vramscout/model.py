from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from .types import ModelSpec


class ModelInspectionError(RuntimeError):
    pass


NONSTANDARD_CACHE_TYPES = {
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v4",
    "mamba",
    "mamba2",
    "jamba",
    "recurrentgemma",
}


def _first_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _normalize_dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower().replace("torch.", "")
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
    }
    return aliases.get(text, text)


def _infer_quantization(config: dict[str, Any]) -> str | None:
    q = config.get("quantization_config")
    if not isinstance(q, dict):
        return None
    method = str(q.get("quant_method") or q.get("quantization_method") or "").lower()
    bits = q.get("bits")
    load4 = bool(q.get("load_in_4bit"))
    load8 = bool(q.get("load_in_8bit"))
    if load4 or bits == 4:
        return f"{method or 'quantized'}-int4"
    if load8 or bits == 8:
        return f"{method or 'quantized'}-int8"
    return method or None


def _extract_hf_param_count(info: Any) -> int | None:
    st = getattr(info, "safetensors", None)
    if st is not None:
        total = getattr(st, "total", None)
        if isinstance(total, int) and total > 0:
            return total
        if isinstance(st, dict):
            total = st.get("total")
            if isinstance(total, int) and total > 0:
                return total

    card = getattr(info, "card_data", None)
    if isinstance(card, dict):
        total = card.get("safetensors", {}).get("total") if isinstance(card.get("safetensors"), dict) else None
        if isinstance(total, int) and total > 0:
            return total
    return None


def _estimate_dense_params(config: dict[str, Any]) -> int:
    hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
    layers = _first_int(config, "num_hidden_layers", "n_layer", "num_layers")
    vocab = _first_int(config, "vocab_size")
    q_heads = _first_int(config, "num_attention_heads", "n_head")
    kv_heads = _first_int(config, "num_key_value_heads", "num_kv_heads") or q_heads
    intermediate = _first_int(config, "intermediate_size", "ffn_dim", "n_inner")
    head_dim = _first_int(config, "head_dim")

    if not all([hidden, layers, vocab, q_heads, kv_heads]):
        raise ModelInspectionError(
            "The checkpoint does not expose enough architecture metadata to estimate parameter count."
        )
    if intermediate is None:
        intermediate = 4 * hidden
    if head_dim is None:
        head_dim = hidden // q_heads

    model_type = str(config.get("model_type") or "").lower()
    if model_type in NONSTANDARD_CACHE_TYPES or _first_int(config, "num_local_experts", "num_experts"):
        raise ModelInspectionError(
            "This architecture needs Hugging Face safetensors parameter metadata; "
            "the dense fallback would be misleading."
        )

    q_out = q_heads * head_dim
    kv_out = kv_heads * head_dim
    attention = hidden * q_out + hidden * kv_out * 2 + q_out * hidden

    # Modern decoder-only LLMs (Llama/Qwen/Mistral/Gemma families) use gated MLPs.
    mlp = 3 * hidden * intermediate
    norms = 2 * hidden
    per_layer = attention + mlp + norms

    tied = bool(config.get("tie_word_embeddings", False))
    embeddings = vocab * hidden
    lm_head = 0 if tied else vocab * hidden
    final_norm = hidden
    return int(embeddings + lm_head + layers * per_layer + final_norm)


def _load_config(model_ref: str, revision: str | None = None) -> tuple[dict[str, Any], str]:
    path = Path(model_ref).expanduser()
    if path.is_dir():
        config_path = path / "config.json"
        if not config_path.exists():
            raise ModelInspectionError(f"No config.json found in {path}")
        return json.loads(config_path.read_text(encoding="utf-8")), str(config_path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), str(path)

    try:
        config_path = hf_hub_download(repo_id=model_ref, filename="config.json", revision=revision)
    except Exception as exc:  # huggingface_hub exposes several transport-specific exceptions
        raise ModelInspectionError(f"Could not fetch config.json for {model_ref}: {exc}") from exc
    return json.loads(Path(config_path).read_text(encoding="utf-8")), config_path


def inspect_model(model_ref: str, revision: str | None = None) -> ModelSpec:
    config, _ = _load_config(model_ref, revision=revision)
    warnings: list[str] = []

    model_type = str(config.get("model_type") or "unknown").lower()
    if model_type in NONSTANDARD_CACHE_TYPES:
        raise ModelInspectionError(
            f"{model_type} uses non-standard inference state/cache semantics. "
            "VRAMScout v0.1 fails closed instead of applying the ordinary Transformer KV formula."
        )

    layers = _first_int(config, "num_hidden_layers", "n_layer", "num_layers")
    hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
    q_heads = _first_int(config, "num_attention_heads", "n_head")
    kv_heads = _first_int(config, "num_key_value_heads", "num_kv_heads") or q_heads
    head_dim = _first_int(config, "head_dim")
    if head_dim is None and hidden and q_heads:
        head_dim = hidden // q_heads

    missing = [
        name
        for name, value in {
            "num_hidden_layers": layers,
            "hidden_size": hidden,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
        }.items()
        if not value
    ]
    if missing:
        raise ModelInspectionError(
            "Unsupported/incomplete decoder config; missing: " + ", ".join(missing)
        )

    max_context = _first_int(
        config,
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    )
    sliding_window = _first_int(config, "sliding_window")
    if sliding_window:
        warnings.append(
            "This model declares sliding-window attention. VRAMScout currently reports a conservative full-cache upper bound; "
            "some serving engines can use less KV memory."
        )

    num_params: int | None = None
    parameter_source = "architecture estimate"
    path = Path(model_ref).expanduser()
    if not path.exists():
        try:
            info = HfApi().model_info(model_ref, revision=revision, files_metadata=True)
            num_params = _extract_hf_param_count(info)
            if num_params:
                parameter_source = "Hugging Face safetensors metadata"
        except Exception:
            warnings.append(
                "Hugging Face parameter metadata was unavailable; using a dense architecture estimate."
            )

    if not num_params:
        num_params = _estimate_dense_params(config)

    quantization = _infer_quantization(config)
    return ModelSpec(
        model_id=model_ref,
        model_type=model_type,
        num_params=num_params,
        num_layers=int(layers),
        hidden_size=int(hidden),
        num_attention_heads=int(q_heads),
        num_kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        max_context=max_context,
        vocab_size=_first_int(config, "vocab_size"),
        torch_dtype=_normalize_dtype(config.get("torch_dtype")),
        quantization=quantization,
        sliding_window=sliding_window,
        parameter_source=parameter_source,
        warnings=warnings,
    )
