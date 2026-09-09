from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .components import (
    ComponentGraph,
    ConstantState,
    DeepSeekCompressedKV,
    DeepSeekSharedKV,
    KVCache,
    LatentCache,
    SparseIndexer,
)

Builder = Callable[[Any], ComponentGraph]
_BUILDERS: dict[str, Builder] = {}


def register(*families: str):
    def deco(fn: Builder) -> Builder:
        for family in families:
            _BUILDERS[family] = fn
        return fn
    return deco


def build_component_graph(profile: Any) -> ComponentGraph:
    try:
        builder = _BUILDERS[profile.family]
    except KeyError as exc:
        raise ValueError(f"No memory-component builder registered for family: {profile.family}") from exc
    return builder(profile)


@register("standard")
def _standard(p: Any) -> ComponentGraph:
    return ComponentGraph(p.family, (KVCache("KV cache", p.layers, p.kv_heads, p.head_dim),))


@register("standard_sparse_index")
def _standard_sparse(p: Any) -> ComponentGraph:
    md = p.metadata
    parts = [KVCache("KV cache", p.layers, p.kv_heads, p.head_dim)]
    if md.get("sparse_layers") and md.get("index_dim") and md.get("index_heads"):
        parts.append(
            SparseIndexer(
                "Sparse indexer",
                int(md["sparse_layers"]),
                int(md["index_dim"]),
                heads=int(md["index_heads"]),
            )
        )
    return ComponentGraph(p.family, tuple(parts))


@register("mla")
def _mla(p: Any) -> ComponentGraph:
    return ComponentGraph(
        p.family,
        (LatentCache("MLA latent cache", p.layers, int(p.metadata.get("mla_dim", 0))),),
    )


@register("mla_indexed", "mla_indexshare")
def _mla_indexed(p: Any) -> ComponentGraph:
    md = p.metadata
    parts = [LatentCache("MLA latent cache", p.layers, int(md.get("mla_dim", 0)))]
    groups = int(md.get("index_groups", 0))
    if groups and md.get("index_dim"):
        parts.append(SparseIndexer("Sparse/DSA indexer", groups, int(md["index_dim"])))
    return ComponentGraph(p.family, tuple(parts))


@register("qwen35_hybrid")
def _qwen35(p: Any) -> ComponentGraph:
    md = p.metadata
    parts = [
        KVCache(
            f"Full-attention KV ({md.get('full_layers', 0)} layers)",
            int(md.get("full_layers", 0)),
            p.kv_heads,
            p.head_dim,
        )
    ]
    if md.get("recurrent_bytes"):
        parts.append(
            ConstantState(
                f"GatedDeltaNet state ({md.get('linear_layers', 0)} layers)",
                int(md["recurrent_bytes"]),
            )
        )
    return ComponentGraph(p.family, tuple(parts))


@register("qwen4_exp_hybrid")
def _qwen4_exp(p: Any) -> ComponentGraph:
    md = p.metadata
    parts = [
        KVCache(
            f"QSA full-attention KV ({md.get('full_layers', 0)} layers)",
            int(md.get("full_layers", 0)),
            p.kv_heads,
            p.head_dim,
        )
    ]
    if md.get("indexer_dim"):
        parts.append(
            SparseIndexer(
                "QSA compressed indexer",
                int(md.get("full_layers", 0)),
                int(md["indexer_dim"]),
                heads=int(md.get("indexer_kv_heads", 1)),
                compression_ratio=max(1, int(md.get("indexer_ratio", 1))),
            )
        )
    if md.get("recurrent_bytes"):
        parts.append(
            ConstantState(
                f"GatedDeltaNet state ({md.get('linear_layers', 0)} layers)",
                int(md["recurrent_bytes"]),
            )
        )
    return ComponentGraph(p.family, tuple(parts))


@register("kimi_k3_hybrid", "glm53_hybrid")
def _kda_mla_hybrid(p: Any) -> ComponentGraph:
    md = p.metadata
    parts = [
        LatentCache(
            "Sparse/MLA latent cache",
            int(md.get("full_layers", 0)),
            int(md.get("mla_dim", 0)),
        )
    ]
    if md.get("index_groups") and md.get("index_dim"):
        parts.append(
            SparseIndexer(
                "Sparse indexer",
                int(md["index_groups"]),
                int(md["index_dim"]),
            )
        )
    if md.get("recurrent_bytes"):
        parts.append(
            ConstantState(
                f"KDA recurrent state ({md.get('linear_layers', 0)} layers)",
                int(md["recurrent_bytes"]),
            )
        )
    return ComponentGraph(p.family, tuple(parts))


@register("deepseek_v4_compressed")
def _deepseek_v4(p: Any) -> ComponentGraph:
    md = p.metadata
    ratios = list(md.get("ratios", []))[: p.layers]
    c4 = sum(int(r) == 4 for r in ratios)
    c128 = sum(int(r) == 128 for r in ratios)
    parts = [
        DeepSeekSharedKV(
            "Shared K=V sliding window",
            len(ratios),
            int(md.get("main_dim", p.head_dim)),
            int(md.get("rope_dim", 0)),
            int(md.get("sliding_window", 0)),
        )
    ]
    if c4:
        parts.append(
            DeepSeekCompressedKV(
                "CSA compressed K=V (1/4)",
                c4,
                int(md.get("main_dim", p.head_dim)),
                int(md.get("rope_dim", 0)),
                4,
            )
        )
        if md.get("index_dim"):
            parts.append(
                SparseIndexer(
                    "CSA indexer",
                    c4,
                    int(md["index_dim"]),
                    compression_ratio=4,
                )
            )
    if c128:
        parts.append(
            DeepSeekCompressedKV(
                "HCA compressed K=V (1/128)",
                c128,
                int(md.get("main_dim", p.head_dim)),
                int(md.get("rope_dim", 0)),
                128,
            )
        )
    return ComponentGraph(p.family, tuple(parts))


@register("gemma4_hybrid")
def _gemma4(p: Any) -> ComponentGraph:
    md = p.metadata
    return ComponentGraph(
        p.family,
        (
            KVCache(
                "Gemma4 sliding KV",
                int(md.get("sliding_layers", 0)),
                p.kv_heads,
                p.head_dim,
                window=int(md.get("sliding_window", 0)) or None,
            ),
            KVCache(
                "Gemma4 global KV",
                int(md.get("full_layers", 0)),
                int(md.get("global_kv_heads", p.kv_heads)),
                int(md.get("global_head_dim", p.head_dim)),
            ),
        ),
    )


@register("mimo_hybrid")
def _mimo(p: Any) -> ComponentGraph:
    md = p.metadata
    return ComponentGraph(
        p.family,
        (
            KVCache(
                "MiMo full-attention KV",
                int(md.get("full_layers", 0)),
                p.kv_heads,
                p.head_dim,
            ),
            KVCache(
                "MiMo chunk/SWA KV",
                int(md.get("sliding_layers", 0)),
                int(md.get("swa_kv_heads", p.kv_heads)),
                int(md.get("swa_head_dim", p.head_dim)),
                window=int(md.get("sliding_window", 128)),
            ),
        ),
    )


@register("nemotron_mamba_hybrid")
def _nemotron(p: Any) -> ComponentGraph:
    md = p.metadata
    mh = int(md.get("mamba_heads", 0))
    mhd = int(md.get("mamba_head_dim", 0))
    ss = int(md.get("ssm_state", 0))
    ker = int(md.get("conv_kernel", 4))
    recurrent = int(md.get("mamba_layers", 0)) * (
        mh * mhd * ss * 4 + (mh * mhd + 2 * 8 * ss) * ker * 2
    )
    return ComponentGraph(
        p.family,
        (
            KVCache(
                "Attention KV",
                int(md.get("attn_layers", 0)),
                p.kv_heads,
                p.head_dim,
            ),
            ConstantState("Mamba-2 recurrent state", recurrent),
        ),
    )
