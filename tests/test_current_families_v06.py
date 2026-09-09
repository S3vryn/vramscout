from vramscout.modern_core import ModernProfile, _cache_parts, _family


def test_family_detection_for_current_models():
    cases = [
        ({"model_type": "qwen3_5_moe_text", "num_hidden_layers": 8, "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2}, "qwen35_hybrid"),
        ({"model_type": "qwen4_exp_text", "num_hidden_layers": 8, "layer_types": ["linear_attention"] * 6 + ["full_attention"] * 2}, "qwen4_exp_hybrid"),
        ({"model_type": "kimi_linear", "num_hidden_layers": 8, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "linear_attn_config": {"full_attn_layers": [3, 7], "num_heads": 8, "head_dim": 64}}, "kimi_k3_hybrid"),
        ({"model_type": "deepseek_v4", "num_hidden_layers": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64}, "mla_indexed"),
        ({"model_type": "glm_moe_dsa", "num_hidden_layers": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "index_head_dim": 128, "indexer_types": ["full", "shared", "shared", "full"]}, "mla_indexshare"),
        ({"model_type": "glm5_next_text", "num_hidden_layers": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 0, "index_head_dim": 128, "layer_types": ["linear_attention", "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"], "indexer_types": ["full"] * 4, "linear_attn_config": {"num_heads": 8, "head_dim": 64}}, "glm53_hybrid"),
        ({"model_type": "hy_v4", "num_hidden_layers": 4, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "index_head_dim": 128, "indexer_types": ["full", "shared", "shared", "full"]}, "mla_indexshare"),
        ({"model_type": "minimax_m3_text", "num_hidden_layers": 4, "sparse_attention_config": {"sparse_index_dim": 128, "sparse_num_index_heads": 4, "sparse_attention_freq": [0, 1, 1, 1]}}, "standard_sparse_index"),
        ({"model_type": "gemma4_text", "num_hidden_layers": 4, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"], "sliding_window": 1024, "num_key_value_heads": 4, "num_global_key_value_heads": 2, "global_head_dim": 256}, "gemma4_hybrid"),
        ({"model_type": "mistral4", "num_hidden_layers": 4, "kv_lora_rank": 256, "qk_rope_head_dim": 64}, "mla"),
        ({"model_type": "mimo_v2", "num_hidden_layers": 4, "hybrid_layer_pattern": [0, 1, 1, 0], "attention_chunk_size": 128, "num_key_value_heads": 4}, "mimo_hybrid"),
        ({"model_type": "nemotron_h", "num_hidden_layers": 4, "layers_block_type": ["mamba", "moe", "attention", "mamba"], "mamba_num_heads": 8, "mamba_head_dim": 64, "ssm_state_size": 128}, "nemotron_mamba_hybrid"),
    ]
    for cfg, expected in cases:
        root = dict(cfg)
        family, _, _ = _family(root, cfg)
        assert family == expected


def _profile(family, metadata, *, layers=8, kv_heads=4, head_dim=128):
    return ModernProfile(
        model_id="synthetic", root_model_type="x", model_type="x", family=family,
        num_params=1_000_000, checkpoint_size_bytes=None, layers=layers, hidden_size=1024,
        q_heads=8, kv_heads=kv_heads, head_dim=head_dim, max_context=1_048_576,
        dtype="bf16", quantization=None, metadata=metadata,
    )


def test_gemma4_sliding_cache_saturates_but_global_grows():
    p = _profile("gemma4_hybrid", {"full_layers": 2, "sliding_layers": 6, "sliding_window": 1024, "global_kv_heads": 2, "global_head_dim": 256})
    a = _cache_parts(p, 2048, 1, "bf16", "bf16", 1, 1)
    b = _cache_parts(p, 4096, 1, "bf16", "bf16", 1, 1)
    assert a["Gemma4 sliding KV"] == b["Gemma4 sliding KV"]
    assert b["Gemma4 global KV"] == 2 * a["Gemma4 global KV"]


def test_nemotron_mamba_state_is_context_constant():
    p = _profile("nemotron_mamba_hybrid", {"attn_layers": 2, "mamba_layers": 4, "mamba_heads": 8, "mamba_head_dim": 64, "ssm_state": 128, "conv_kernel": 4})
    a = _cache_parts(p, 1024, 1, "bf16", "bf16", 1, 1)
    b = _cache_parts(p, 2048, 1, "bf16", "bf16", 1, 1)
    assert a["Mamba-2 recurrent state"] == b["Mamba-2 recurrent state"]
    assert b["Attention KV"] == 2 * a["Attention KV"]


def test_qwen4_and_glm53_expose_recurrent_and_sparse_state():
    q = _profile("qwen4_exp_hybrid", {"full_layers": 2, "linear_layers": 6, "recurrent_bytes": 1024, "indexer_dim": 128, "indexer_kv_heads": 1, "indexer_ratio": 4})
    qparts = _cache_parts(q, 4096, 1, "bf16", "bf16", 1, 1)
    assert "QSA compressed indexer" in qparts
    assert any("GatedDeltaNet" in k for k in qparts)

    g = _profile("glm53_hybrid", {"full_layers": 2, "linear_layers": 6, "recurrent_bytes": 1024, "mla_dim": 512, "index_dim": 128, "index_groups": 2})
    gparts = _cache_parts(g, 4096, 1, "fp8", "bf16", 1, 1)
    assert "Sparse/MLA latent cache" in gparts
    assert "Sparse indexer" in gparts
    assert any("KDA recurrent" in k for k in gparts)


def test_mla_dcp_shards_sequence_cache():
    p = _profile("mla", {"mla_dim": 576})
    a = _cache_parts(p, 8192, 1, "bf16", "bf16", 8, 1)["MLA latent cache"]
    b = _cache_parts(p, 8192, 1, "bf16", "bf16", 8, 8)["MLA latent cache"]
    assert a == 8 * b
