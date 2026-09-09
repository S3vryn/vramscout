from types import SimpleNamespace

import pytest

from vramscout.component_registry import build_component_graph
from vramscout.components import EvalContext, KVCache, LatentCache, SparseIndexer, GIB


def env(*, context=100, kv_bytes=2.0, indexer_bytes=2.0, tp=1, dcp=1):
    return EvalContext(
        context=context,
        batch=1,
        kv_bytes=kv_bytes,
        indexer_bytes=indexer_bytes,
        tp=tp,
        dcp=dcp,
    )


def test_asymmetric_kv_cache_supports_mimo_style_k_v_dims():
    cache = KVCache("mimo", layers=2, kv_heads=4, head_dim=192, value_head_dim=128)
    expected_bytes = 2 * 100 * 2 * (192 + 128) * 2  # layers * tokens * local heads * (K+V) * bf16
    assert cache.gib(env(tp=2)) == pytest.approx(expected_bytes / GIB)


def test_sparse_index_heads_follow_tp_replication_rule():
    index = SparseIndexer(
        "msa-index",
        layers=57,
        dim=128,
        heads=4,
        tp_sharded_heads=True,
    )
    # TP8 with four index heads replicates one head per rank, matching vLLM's
    # get_num_kv_heads-style rule used by MiniMax-M3.
    expected_bytes = 57 * 100 * 128 * 2
    assert index.gib(env(tp=8)) == pytest.approx(expected_bytes / GIB)


def test_latent_cache_dcp_is_opt_in():
    replicated = LatentCache("mla", layers=10, latent_dim=576, dcp_sharded=False)
    sharded = LatentCache("mla", layers=10, latent_dim=576, dcp_sharded=True)
    base = replicated.gib(env(context=1024, dcp=4))
    assert sharded.gib(env(context=1024, dcp=4)) == pytest.approx(base / 4)


def test_minimax_m3_registry_shards_index_heads():
    profile = SimpleNamespace(
        family="standard_sparse_index",
        layers=60,
        kv_heads=4,
        head_dim=128,
        metadata={"sparse_layers": 57, "index_dim": 128, "index_heads": 4},
    )
    graph = build_component_graph(profile)
    parts = graph.parts_gib(env(context=1000, tp=8))
    raw_bytes = sum(parts.values()) * GIB
    expected = (60 * 1000 * 1 * (128 + 128) * 2) + (57 * 1000 * 1 * 128 * 2)
    assert raw_bytes == pytest.approx(expected)
