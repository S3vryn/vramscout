import math

from vramscout.memory import cache_memory_parts_gib
from vramscout.model import _kimi_mla_cache_metadata
from vramscout.types import ModelSpec


KIMI_CFG = {
    "model_type": "kimi_k2",
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
    "max_position_embeddings": 262144,
}


def _spec() -> ModelSpec:
    return ModelSpec(
        model_id="moonshotai/Kimi-K2.6",
        model_type="kimi_k2",
        num_params=1,
        num_layers=61,
        hidden_size=7168,
        num_attention_heads=64,
        num_kv_heads=64,
        head_dim=192,
        max_context=262144,
        cache_kind="kimi_mla",
        kv_layers=0,
        cache_metadata=_kimi_mla_cache_metadata(KIMI_CFG),
    )


def test_kimi_mla_metadata():
    md = _kimi_mla_cache_metadata(KIMI_CFG)
    assert md["kv_lora_rank"] == 512
    assert md["rope_head_dim"] == 64
    assert md["mla_latent_dim"] == 576


def test_kimi_bf16_mla_cache_matches_public_vllm_log():
    spec = _spec()
    predicted_bytes = sum(
        cache_memory_parts_gib(spec, 1, 1, "bf16", tp_size=8, dcp_size=1).values()
    ) * 1024**3
    assert predicted_bytes == 61 * 576 * 2

    # MoonshotAI/Kimi-K2.5 issue #34: TP=8, DCP=1, 163.59 GiB available
    # KV cache and 2,499,584 cache tokens with FLASHINFER_MLA.
    observed_bytes = 163.59 * 1024**3 / 2_499_584
    error_pct = abs(predicted_bytes - observed_bytes) / observed_bytes * 100
    assert error_pct < 0.01


def test_kimi_dcp_sequence_shards_mla_cache():
    spec = _spec()
    dcp1 = sum(cache_memory_parts_gib(spec, 262144, 1, "bf16", tp_size=8, dcp_size=1).values())
    dcp8 = sum(cache_memory_parts_gib(spec, 262144, 1, "bf16", tp_size=8, dcp_size=8).values())
    assert math.isclose(dcp1 / 8, dcp8, rel_tol=1e-12)
