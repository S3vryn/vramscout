import math
import pytest

from vramscout.memory import kv_cache_per_token_gib
from vramscout.planner import plan_inference
from vramscout.types import GPUInfo, ModelSpec


def _gqa(kv_heads=8):
    return ModelSpec(
        model_id="gqa",
        model_type="llama",
        num_params=8_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=kv_heads,
        head_dim=128,
        max_context=131072,
    )


def test_tp_shards_gqa_cache_by_kv_heads():
    single = kv_cache_per_token_gib(_gqa(), 1, "bf16", tp_size=1)
    tp4 = kv_cache_per_token_gib(_gqa(), 1, "bf16", tp_size=4)
    assert math.isclose(single / 4, tp4, rel_tol=1e-12)


def test_tp_larger_than_kv_heads_replicates_one_head_per_rank():
    spec = _gqa(kv_heads=2)
    tp2 = kv_cache_per_token_gib(spec, 1, "bf16", tp_size=2)
    tp4 = kv_cache_per_token_gib(spec, 1, "bf16", tp_size=4)
    assert math.isclose(tp2, tp4, rel_tol=1e-12)


def test_nondivisible_tp_fails_closed():
    with pytest.raises(ValueError, match="KV-head placement"):
        kv_cache_per_token_gib(_gqa(kv_heads=8), 1, "bf16", tp_size=3)


def test_dcp_rejected_for_non_mla_model():
    gpu = GPUInfo(index=0, name="test", total_gib=80, used_gib=0, free_gib=80)
    with pytest.raises(ValueError, match="Decode Context Parallelism"):
        plan_inference(gpu, _gqa(), tp_size=2, dcp_size=2)
