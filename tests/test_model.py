from vramscout.model import _estimate_dense_params


def test_dense_param_estimate_llama_like():
    cfg = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 128256,
        "tie_word_embeddings": False,
    }
    n = _estimate_dense_params(cfg)
    assert 7_000_000_000 < n < 9_500_000_000
