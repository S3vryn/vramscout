from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from .engine import resolve_engine_budget
from .gpu import get_gpu_group

GIB = 1024**3
WEIGHT_BPP = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.05, "int4": 0.55, "nvfp4": 0.55}
CACHE_BPE = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "fp4": 0.5}


class ModernInspectionError(RuntimeError):
    pass


@dataclass(slots=True)
class ModernProfile:
    model_id: str
    root_model_type: str
    model_type: str
    family: str
    num_params: int
    checkpoint_size_bytes: int | None
    layers: int
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    max_context: int | None
    dtype: str | None
    quantization: str | None
    has_vision: bool = False
    has_audio: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModernPlan:
    profile: ModernProfile
    gpu_name: str
    gpu_total_gib: float
    gpu_free_gib: float
    engine: str
    memory_budget_gib: float
    startup_ok: bool
    tp: int
    dcp: int
    context: int
    weight_dtype: str
    kv_dtype: str
    indexer_dtype: str
    parts_gib: dict[str, float]
    total_gib: float
    spare_gib: float
    fits: bool
    max_context_vram: int
    max_context_usable: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_int(cfg: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _dtype(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).lower().replace("torch.", "")
    return {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32", "float8_e4m3fn": "fp8"}.get(text, text)


def _quantization(cfg: dict[str, Any]) -> str | None:
    q = cfg.get("quantization_config")
    if not isinstance(q, dict):
        return None
    method = str(q.get("quant_method") or q.get("quantization_method") or "").lower()
    algo = str(q.get("quant_algo") or q.get("fmt") or q.get("format") or "").lower()
    blob = method + " " + algo + " " + json.dumps(q.get("config_groups", {})).lower()
    if "nvfp4" in blob or "mxfp4" in blob:
        return "nvfp4"
    if "fp8" in blob or "e4m3" in blob or "mxfp8" in blob:
        return "fp8"
    if q.get("load_in_4bit") or q.get("bits") == 4:
        return "int4"
    if q.get("load_in_8bit") or q.get("bits") == 8:
        return "int8"
    return method or None


def _load_config(ref: str, revision: str | None) -> dict[str, Any]:
    p = Path(ref).expanduser()
    if p.is_dir():
        p = p / "config.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    try:
        path = hf_hub_download(repo_id=ref, filename="config.json", revision=revision)
    except Exception as exc:
        raise ModernInspectionError(f"Could not fetch config.json for {ref}: {exc}") from exc
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hub_metadata(ref: str, revision: str | None) -> tuple[int | None, int | None]:
    if Path(ref).expanduser().exists():
        return None, None
    try:
        info = HfApi().model_info(ref, revision=revision, files_metadata=True)
    except Exception:
        return None, None
    params = None
    st = getattr(info, "safetensors", None)
    if st is not None:
        params = getattr(st, "total", None)
        if params is None and isinstance(st, dict):
            params = st.get("total")
    total = 0
    found = False
    for f in getattr(info, "siblings", None) or []:
        name, size = getattr(f, "rfilename", None), getattr(f, "size", None)
        if isinstance(name, str) and name.endswith(".safetensors") and isinstance(size, int) and size > 0:
            total += size
            found = True
    return params if isinstance(params, int) else None, total if found else None


def _dense_param_fallback(cfg: dict[str, Any]) -> int:
    h = _first_int(cfg, "hidden_size", "d_model", "n_embd")
    l = _first_int(cfg, "num_hidden_layers", "num_layers", "n_layer")
    v = _first_int(cfg, "vocab_size")
    q = _first_int(cfg, "num_attention_heads", "n_head")
    k = _first_int(cfg, "num_key_value_heads", "num_kv_heads") or q
    hd = _first_int(cfg, "head_dim", "qk_head_dim") or (h // q if h and q else None)
    ff = _first_int(cfg, "intermediate_size", "ffn_hidden_size", "ffn_dim") or (4 * h if h else None)
    if not all([h, l, v, q, k, hd, ff]):
        raise ModernInspectionError("Hugging Face parameter metadata is required for this checkpoint.")
    if any(_first_int(cfg, x) for x in ("num_experts", "num_local_experts", "n_routed_experts")):
        raise ModernInspectionError("MoE checkpoint requires Hugging Face safetensors parameter metadata.")
    attn = h * (q * hd) + 2 * h * (k * hd) + (q * hd) * h
    return int(v * h * (1 if cfg.get("tie_word_embeddings") else 2) + l * (attn + 3 * h * ff + 2 * h) + h)


def _layer_counts(cfg: dict[str, Any], full_names: set[str]) -> tuple[int, int]:
    layer_types = cfg.get("layer_types")
    layers = _first_int(cfg, "num_hidden_layers", "num_layers") or 0
    if isinstance(layer_types, list) and layer_types:
        full = sum(str(x).lower() in full_names for x in layer_types[:layers])
        return full, layers - full
    interval = _first_int(cfg, "full_attention_interval") or 0
    if interval:
        full = layers // interval
        return full, layers - full
    return layers, 0


def _qwen_recurrent_bytes(cfg: dict[str, Any], linear_layers: int) -> int:
    nv = _first_int(cfg, "linear_num_value_heads")
    nk = _first_int(cfg, "linear_num_key_heads")
    kd = _first_int(cfg, "linear_key_head_dim")
    vd = _first_int(cfg, "linear_value_head_dim")
    kernel = _first_int(cfg, "linear_conv_kernel_dim")
    if not all([nv, nk, kd, vd, kernel]):
        return 0
    recurrent = nv * kd * vd * 4
    conv = (2 * nk * kd + nv * vd) * kernel * 2
    return linear_layers * (recurrent + conv)


def _kda_recurrent_bytes(cfg: dict[str, Any], linear_layers: int) -> int:
    lc = cfg.get("linear_attn_config") if isinstance(cfg.get("linear_attn_config"), dict) else {}
    heads = _first_int(lc, "num_heads") or _first_int(cfg, "linear_num_value_heads")
    hd = _first_int(lc, "head_dim") or _first_int(cfg, "linear_value_head_dim", "linear_key_head_dim")
    kernel = _first_int(lc, "short_conv_kernel_size") or _first_int(cfg, "linear_conv_kernel_dim") or 4
    if not heads or not hd:
        return 0
    # KDA/GatedDeltaNet matrix state is kept FP32; short-conv state follows activation dtype.
    recurrent = heads * hd * hd * 4
    conv = 3 * heads * hd * kernel * 2
    return linear_layers * (recurrent + conv)


def _family(root: dict[str, Any], text: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]]:
    rt = str(root.get("model_type") or "").lower()
    mt = str(text.get("model_type") or rt).lower()
    arch = " ".join(str(x).lower() for x in root.get("architectures", []) if isinstance(x, str))
    notes: list[str] = []
    md: dict[str, Any] = {}

    if rt == "qwen4_exp" or mt == "qwen4_exp_text":
        full, linear = _layer_counts(text, {"full_attention"})
        md.update(full_layers=full, linear_layers=linear, recurrent_bytes=_qwen_recurrent_bytes(text, linear),
                  indexer_dim=_first_int(text, "indexer_head_dim") or 0,
                  indexer_kv_heads=_first_int(text, "indexer_kv_heads") or 1,
                  indexer_ratio=_first_int(text, "indexer_compress_ratio") or 1)
        return "qwen4_exp_hybrid", md, notes

    if mt in {"qwen3_5_text", "qwen3_5_moe_text"} or "qwen3_5" in mt:
        full, linear = _layer_counts(text, {"full_attention"})
        md.update(full_layers=full, linear_layers=linear, recurrent_bytes=_qwen_recurrent_bytes(text, linear))
        return "qwen35_hybrid", md, notes

    if rt == "kimi_k3" or mt == "kimi_linear":
        lc = text.get("linear_attn_config") if isinstance(text.get("linear_attn_config"), dict) else {}
        full_ids = lc.get("full_attn_layers") if isinstance(lc.get("full_attn_layers"), list) else []
        full = len(full_ids)
        linear = (_first_int(text, "num_hidden_layers") or 0) - full
        md.update(full_layers=full, linear_layers=linear, recurrent_bytes=_kda_recurrent_bytes(text, linear),
                  mla_dim=(_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0))
        return "kimi_k3_hybrid", md, notes

    if mt == "kimi_k2" or rt == "kimi_k2":
        md["mla_dim"] = (_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0)
        return "mla", md, notes

    if rt == "deepseek_v4" or mt == "deepseek_v4":
        ratios = text.get("compress_ratios") if isinstance(text.get("compress_ratios"), list) else []
        if ratios:
            md.update(ratios=[int(x) for x in ratios], sliding_window=_first_int(text, "sliding_window") or 0,
                      main_dim=_first_int(text, "head_dim", "qk_head_dim") or 0,
                      rope_dim=_first_int(text, "qk_rope_head_dim") or 0,
                      index_dim=_first_int(text, "index_head_dim") or 0)
            return "deepseek_v4_compressed", md, notes
        md["mla_dim"] = (_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0)
        md["index_dim"] = _first_int(text, "index_head_dim") or 0
        return "mla_indexed", md, notes

    if mt == "glm_moe_dsa":
        idx = text.get("indexer_types") if isinstance(text.get("indexer_types"), list) else []
        md.update(mla_dim=(_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0),
                  index_dim=_first_int(text, "index_head_dim") or 0,
                  index_groups=sum(str(x).lower() == "full" for x in idx))
        return "mla_indexshare", md, notes

    if rt == "glm5_next" or mt == "glm5_next_text":
        types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
        sparse_ids = [i for i, x in enumerate(types) if str(x).lower() == "deepseek_sparse_attention"]
        linear = len(types) - len(sparse_ids)
        idx = text.get("indexer_types") if isinstance(text.get("indexer_types"), list) else []
        groups = sum(i < len(idx) and str(idx[i]).lower() == "full" for i in sparse_ids)
        md.update(full_layers=len(sparse_ids), linear_layers=linear, recurrent_bytes=_kda_recurrent_bytes(text, linear),
                  mla_dim=(_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0),
                  index_dim=_first_int(text, "index_head_dim") or 0, index_groups=groups)
        return "glm53_hybrid", md, notes

    if mt == "hy_v4" or rt == "hy_v4":
        idx = text.get("indexer_types") if isinstance(text.get("indexer_types"), list) else []
        md.update(mla_dim=(_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0),
                  index_dim=_first_int(text, "index_head_dim") or 0,
                  index_groups=sum(str(x).lower() == "full" for x in idx))
        return "mla_indexshare", md, notes

    if "longcat" in arch or str(text.get("attention_method", "")).upper() == "MLA":
        md.update(mla_dim=(_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0),
                  index_dim=_first_int(text, "index_head_dim") or 0,
                  index_groups=_first_int(text, "num_layers", "num_hidden_layers") or 0)
        return "mla_indexed", md, notes

    if rt == "minimax_m3_vl" or mt.startswith("minimax_m3"):
        sc = text.get("sparse_attention_config") if isinstance(text.get("sparse_attention_config"), dict) else {}
        freq = sc.get("sparse_attention_freq") if isinstance(sc.get("sparse_attention_freq"), list) else []
        md.update(index_dim=_first_int(sc, "sparse_index_dim") or 0,
                  index_heads=_first_int(sc, "sparse_num_index_heads") or 0,
                  sparse_layers=sum(bool(x) for x in freq))
        return "standard_sparse_index", md, notes

    if rt == "gemma4" or mt == "gemma4_text":
        types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
        md.update(full_layers=sum(str(x).lower() == "full_attention" for x in types),
                  sliding_layers=sum(str(x).lower() == "sliding_attention" for x in types),
                  sliding_window=_first_int(text, "sliding_window") or 0,
                  global_kv_heads=_first_int(text, "num_global_key_value_heads") or _first_int(text, "num_key_value_heads") or 1,
                  global_head_dim=_first_int(text, "global_head_dim") or _first_int(text, "head_dim") or 1)
        return "gemma4_hybrid", md, notes

    if mt == "mistral4":
        md["mla_dim"] = (_first_int(text, "kv_lora_rank") or 0) + (_first_int(text, "qk_rope_head_dim") or 0)
        return "mla", md, notes

    if mt == "mimo_v2" or rt == "mimo_v2":
        pattern = text.get("hybrid_layer_pattern") if isinstance(text.get("hybrid_layer_pattern"), list) else []
        md.update(full_layers=sum(int(x) == 0 for x in pattern), sliding_layers=sum(int(x) == 1 for x in pattern),
                  sliding_window=_first_int(text, "attention_chunk_size") or 128,
                  swa_kv_heads=_first_int(text, "swa_num_key_value_heads") or _first_int(text, "num_key_value_heads") or 1,
                  swa_head_dim=_first_int(text, "swa_v_head_dim", "swa_head_dim") or _first_int(text, "head_dim") or 1)
        return "mimo_hybrid", md, notes

    if mt == "nemotron_h" or rt == "nemotron_h":
        blocks = text.get("layers_block_type") if isinstance(text.get("layers_block_type"), list) else []
        md.update(attn_layers=sum(str(x).lower() == "attention" for x in blocks),
                  mamba_layers=sum(str(x).lower() == "mamba" for x in blocks),
                  mamba_heads=_first_int(text, "mamba_num_heads") or 0,
                  mamba_head_dim=_first_int(text, "mamba_head_dim") or 0,
                  ssm_state=_first_int(text, "ssm_state_size") or 0,
                  conv_kernel=_first_int(text, "conv_kernel") or 4,
                  expand=_first_int(text, "expand") or 2)
        return "nemotron_mamba_hybrid", md, notes

    return "standard", md, notes


def inspect_modern_model(ref: str, revision: str | None = None) -> ModernProfile:
    root = _load_config(ref, revision)
    text = root.get("text_config") if isinstance(root.get("text_config"), dict) else root
    rt, mt = str(root.get("model_type") or "unknown").lower(), str(text.get("model_type") or root.get("model_type") or "unknown").lower()
    layers = _first_int(text, "num_hidden_layers", "num_layers", "n_layer")
    hidden = _first_int(text, "hidden_size", "d_model", "n_embd")
    qh = _first_int(text, "num_attention_heads", "n_head")
    kvh = _first_int(text, "num_key_value_heads", "num_kv_heads") or qh
    hd = _first_int(text, "head_dim", "qk_head_dim")
    if hd is None and hidden and qh:
        hd = hidden // qh
    if not all([layers, hidden, qh, kvh, hd]):
        raise ModernInspectionError(f"Unsupported/incomplete decoder config for {ref}.")
    params, ckpt = _hub_metadata(ref, revision)
    if not params:
        params = _dense_param_fallback(text)
    family, md, notes = _family(root, text)
    max_ctx = _first_int(text, "max_position_embeddings", "max_sequence_length", "seq_length", "n_positions") or _first_int(root, "max_position_embeddings")
    return ModernProfile(
        model_id=ref, root_model_type=rt, model_type=mt, family=family, num_params=params,
        checkpoint_size_bytes=ckpt, layers=layers, hidden_size=hidden, q_heads=qh, kv_heads=kvh,
        head_dim=hd, max_context=max_ctx, dtype=_dtype(text.get("dtype", text.get("torch_dtype", root.get("dtype", root.get("torch_dtype"))))),
        quantization=_quantization(root) or _quantization(text),
        has_vision=isinstance(root.get("vision_config"), dict) or "vision" in ref.lower() or rt.endswith("_vl"),
        has_audio=isinstance(root.get("audio_config"), dict), metadata=md, notes=notes,
    )


def _resolve_weight_dtype(requested: str, p: ModernProfile) -> str:
    if requested != "auto":
        return requested
    return p.quantization or p.dtype or "bf16"


def _resolve_cache_dtype(requested: str, p: ModernProfile) -> str:
    if requested != "auto":
        return requested
    if p.family in {"deepseek_v4_compressed", "glm53_hybrid", "mla_indexshare"} and p.quantization == "fp8":
        return "fp8"
    return "bf16"


def _tp_kv_heads(kv_heads: int, tp: int) -> int:
    if tp <= 1:
        return kv_heads
    if kv_heads % tp == 0:
        return kv_heads // tp
    if tp % kv_heads == 0:
        return 1
    raise ValueError(f"TP={tp} cannot be mapped safely to {kv_heads} KV heads.")


def _weight_gib(p: ModernProfile, dtype: str, tp: int) -> float:
    if dtype == "auto":
        dtype = _resolve_weight_dtype(dtype, p)
    native = p.quantization or p.dtype
    if p.checkpoint_size_bytes and dtype == native:
        total = p.checkpoint_size_bytes / GIB
    else:
        if dtype not in WEIGHT_BPP:
            raise ValueError(f"Unsupported weight dtype: {dtype}")
        total = p.num_params * WEIGHT_BPP[dtype] / GIB
    return total / tp


def _cache_parts(p: ModernProfile, context: int, batch: int, kv_dtype: str, indexer_dtype: str, tp: int, dcp: int) -> dict[str, float]:
    bpe = CACHE_BPE[kv_dtype]
    ibpe = CACHE_BPE[indexer_dtype]
    md = p.metadata
    scale = batch / GIB
    fam = p.family

    if fam == "standard":
        h = _tp_kv_heads(p.kv_heads, tp)
        return {"KV cache": 2 * p.layers * context * h * p.head_dim * bpe * scale}

    if fam == "standard_sparse_index":
        h = _tp_kv_heads(p.kv_heads, tp)
        parts = {"KV cache": 2 * p.layers * context * h * p.head_dim * bpe * scale}
        idx = md.get("sparse_layers", 0) * context * md.get("index_dim", 0) * md.get("index_heads", 0) * ibpe * scale
        if idx:
            parts["Sparse indexer"] = idx
        return parts

    if fam in {"mla", "mla_indexed", "mla_indexshare"}:
        parts = {"MLA latent cache": p.layers * context * md.get("mla_dim", 0) * bpe * scale / dcp}
        groups = md.get("index_groups", 0)
        if groups and md.get("index_dim", 0):
            parts["Sparse/DSA indexer"] = groups * context * md["index_dim"] * ibpe * scale / dcp
        return parts

    if fam == "qwen35_hybrid":
        h = _tp_kv_heads(p.kv_heads, tp)
        parts = {f"Full-attention KV ({md['full_layers']} layers)": 2 * md["full_layers"] * context * h * p.head_dim * bpe * scale}
        if md.get("recurrent_bytes"):
            parts[f"GatedDeltaNet state ({md['linear_layers']} layers)"] = md["recurrent_bytes"] * batch / GIB
        return parts

    if fam == "qwen4_exp_hybrid":
        h = _tp_kv_heads(p.kv_heads, tp)
        parts = {f"QSA full-attention KV ({md['full_layers']} layers)": 2 * md["full_layers"] * context * h * p.head_dim * bpe * scale}
        ratio = max(1, md.get("indexer_ratio", 1))
        if md.get("indexer_dim"):
            parts["QSA compressed indexer"] = md["full_layers"] * math.ceil(context / ratio) * md["indexer_dim"] * md.get("indexer_kv_heads", 1) * ibpe * scale
        if md.get("recurrent_bytes"):
            parts[f"GatedDeltaNet state ({md['linear_layers']} layers)"] = md["recurrent_bytes"] * batch / GIB
        return parts

    if fam in {"kimi_k3_hybrid", "glm53_hybrid"}:
        parts = {"Sparse/MLA latent cache": md.get("full_layers", 0) * context * md.get("mla_dim", 0) * bpe * scale / dcp}
        if md.get("index_groups") and md.get("index_dim"):
            parts["Sparse indexer"] = md["index_groups"] * context * md["index_dim"] * ibpe * scale / dcp
        if md.get("recurrent_bytes"):
            parts[f"KDA recurrent state ({md.get('linear_layers', 0)} layers)"] = md["recurrent_bytes"] * batch / GIB
        return parts

    if fam == "deepseek_v4_compressed":
        ratios = md.get("ratios", [])[: p.layers]
        dim, rope = md.get("main_dim", p.head_dim), md.get("rope_dim", 0)
        entry = (dim - rope) + rope * 2 + 8 if kv_dtype == "fp8" else dim * bpe
        swa = md.get("sliding_window", 0)
        c4 = sum(r == 4 for r in ratios)
        c128 = sum(r == 128 for r in ratios)
        total = len(ratios)
        parts = {"Shared K=V sliding window": total * min(context, swa) * entry * scale}
        if c4:
            parts["CSA compressed K=V (1/4)"] = c4 * (context // 4) * entry * scale
            if md.get("index_dim"):
                parts["CSA indexer"] = c4 * (context // 4) * md["index_dim"] * ibpe * scale
        if c128:
            parts["HCA compressed K=V (1/128)"] = c128 * (context // 128) * entry * scale
        return parts

    if fam == "gemma4_hybrid":
        local_h = _tp_kv_heads(p.kv_heads, tp)
        global_h = _tp_kv_heads(md.get("global_kv_heads", p.kv_heads), tp)
        return {
            "Gemma4 sliding KV": 2 * md.get("sliding_layers", 0) * min(context, md.get("sliding_window", context)) * local_h * p.head_dim * bpe * scale,
            "Gemma4 global KV": 2 * md.get("full_layers", 0) * context * global_h * md.get("global_head_dim", p.head_dim) * bpe * scale,
        }

    if fam == "mimo_hybrid":
        global_h = _tp_kv_heads(p.kv_heads, tp)
        local_h = _tp_kv_heads(md.get("swa_kv_heads", p.kv_heads), tp)
        return {
            "MiMo full-attention KV": 2 * md.get("full_layers", 0) * context * global_h * p.head_dim * bpe * scale,
            "MiMo chunk/SWA KV": 2 * md.get("sliding_layers", 0) * min(context, md.get("sliding_window", 128)) * local_h * md.get("swa_head_dim", p.head_dim) * bpe * scale,
        }

    if fam == "nemotron_mamba_hybrid":
        h = _tp_kv_heads(p.kv_heads, tp)
        attn = 2 * md.get("attn_layers", 0) * context * h * p.head_dim * bpe * scale
        mh, mhd, ss, ker = md.get("mamba_heads", 0), md.get("mamba_head_dim", 0), md.get("ssm_state", 0), md.get("conv_kernel", 4)
        mamba = md.get("mamba_layers", 0) * (mh * mhd * ss * 4 + (mh * mhd + 2 * 8 * ss) * ker * 2) * batch / GIB
        return {"Attention KV": attn, "Mamba-2 recurrent state": mamba}

    raise ValueError(f"Unsupported cache family: {fam}")


def _runtime_gib(weight_rank_gib: float) -> float:
    return max(0.75, 0.03 * weight_rank_gib)


def _prefill_gib(p: ModernProfile, context: int, batch: int, dtype: str, chunk: int) -> float:
    bytes_ = 4 if dtype == "fp32" else 2
    return min(context, chunk) * batch * p.hidden_size * bytes_ * 5 / GIB


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
) -> ModernPlan:
    if tp < 1 or dcp < 1 or dcp > tp or tp % dcp != 0:
        raise ValueError("Require tp >= 1 and dcp to divide tp.")
    p = inspect_modern_model(model_ref, revision)
    gpu = get_gpu_group(start_index=gpu_index, count=tp, vram_gib=vram_gib)
    budget = resolve_engine_budget(gpu, engine=engine, gpu_memory_utilization=gpu_memory_utilization)
    wd = _resolve_weight_dtype(weight_dtype, p)
    kd = _resolve_cache_dtype(kv_dtype, p)
    if wd not in WEIGHT_BPP and not (p.checkpoint_size_bytes and wd == p.quantization):
        raise ValueError(f"Unsupported weight dtype: {wd}")
    if kd not in CACHE_BPE or indexer_dtype not in CACHE_BPE:
        raise ValueError("Unsupported cache/indexer dtype")
    ctx = context or min(8192, p.max_context or 8192)
    reserve = reserve_gib if reserve_gib is not None else max(1.0, 0.05 * gpu.total_gib)
    weights = _weight_gib(p, wd, tp)

    def breakdown(c: int) -> tuple[dict[str, float], float]:
        parts = {"Model weights / rank": weights}
        parts.update(_cache_parts(p, c, batch_size, kd, indexer_dtype, tp, dcp))
        parts["CUDA / runtime"] = _runtime_gib(weights)
        parts["Prefill scratch"] = _prefill_gib(p, c, batch_size, wd, prefill_chunk)
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
        notes.append("Vision checkpoint weights are included; request-dependent vision activation/token expansion is not yet modeled exactly.")
    if p.has_audio:
        notes.append("Audio checkpoint weights are included; request-dependent audio encoder activations are not yet modeled exactly.")
    if p.family == "nemotron_mamba_hybrid":
        notes.append("Nemotron Mamba-2 cache uses architecture-derived recurrent-state dimensions; backend alignment/stochastic-rounding buffers remain runtime overhead.")
    notes.append("Checkpoint shard bytes are preferred for native quantized weights; runtime/CUDA workspace remains a static preflight estimate.")
    return ModernPlan(
        profile=p, gpu_name=gpu.name, gpu_total_gib=gpu.total_gib, gpu_free_gib=gpu.free_gib,
        engine=engine, memory_budget_gib=budget.memory_budget_gib, startup_ok=budget.startup_ok,
        tp=tp, dcp=dcp, context=ctx, weight_dtype=wd, kv_dtype=kd, indexer_dtype=indexer_dtype,
        parts_gib=parts, total_gib=total, spare_gib=budget.memory_budget_gib - total, fits=fits,
        max_context_vram=max_ctx, max_context_usable=min(max_ctx, p.max_context) if p.max_context else max_ctx, notes=notes,
    )
