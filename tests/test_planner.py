from vramscout.planner import plan_inference
from vramscout.types import GPUInfo, ModelSpec


def test_plan_and_max_context():
    gpu = GPUInfo(index=0, name="Test GPU", total_gib=32.0, used_gib=0.0, free_gib=32.0)
    model = ModelSpec(
        model_id="toy-8b",
        model_type="llama",
        num_params=8_000_000_000,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_context=131072,
        vocab_size=128256,
        torch_dtype="bf16",
    )
    result = plan_inference(gpu, model, context=8192)
    assert result.fits
    assert result.max_context_vram > 8192
    assert result.max_context_usable <= 131072
    assert result.breakdown.weights_gib > 14
