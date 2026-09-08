import math

from vramscout.memory import cache_memory_parts_gib, weight_memory_gib
from vramscout.planner import plan_inference
from vramscout.types import GPUInfo, ModelSpec


def _spec() -> ModelSpec:
    return ModelSpec(
        model_id="MiniMaxAI/MiniMax-M2.7",
        model_type="minimax_m2",
        num_params=230_000_000_000,
        num_layers=62,
        hidden_size=3072,
        num_attention_heads=48,
        num_kv_heads=8,
        head_dim=128,
        max_context=204800,
        vocab_size=200064,
        torch_dtype="bf16",
        quantization="fp8",
        cache_kind="minimax_m2",
        checkpoint_size_bytes=220 * 1024**3,
        cache_metadata={"official_single_sequence_limit": 196608, "mtp_modules": 3},
    )


def test_minimax_public_vllm_bf16_kv_requirement():
    spec = _spec()
    predicted = sum(cache_memory_parts_gib(spec, 204800, 1, "bf16").values())
    assert math.isclose(predicted, 48.4375, rel_tol=1e-12)
    # vLLM issue #42017 reports 48.44 GiB required for max_seq_len=204800.
    assert abs(predicted - 48.44) / 48.44 * 100 < 0.01


def test_minimax_tp4_shards_gqa_kv_and_weights():
    spec = _spec()
    single = sum(cache_memory_parts_gib(spec, 196608, 1, "bf16", tp_size=1).values())
    tp4 = sum(cache_memory_parts_gib(spec, 196608, 1, "bf16", tp_size=4).values())
    assert math.isclose(single / 4, tp4, rel_tol=1e-12)
    assert math.isclose(weight_memory_gib(spec, "fp8", tp_size=4), 55.0, rel_tol=1e-12)


def test_minimax_tp4_96g_plan_fits_196k():
    spec = _spec()
    gpu = GPUInfo(index=0, name="4x96G", total_gib=96, used_gib=0, free_gib=96)
    result = plan_inference(gpu, spec, context=196608, tp_size=4, weight_dtype="auto", kv_dtype="bf16")
    assert result.fits
    assert result.tp_size == 4
    assert result.breakdown.weights_gib == 55.0
