from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pct_error(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured * 100.0


def main() -> None:
    data = json.loads((ROOT / "public" / "qwen3_8_27b.json").read_text())
    print("Qwen3.8-27B public validation")
    print("variant                 predicted   public   abs error")
    print("-------------------------------------------------------")
    for case in data["weight_validation"]:
        pred = case["predicted_gib_from_checkpoint_bytes"]
        ref = case["public_vllm_weight_gib"]
        print(f"{case['variant']:<22} {pred:>8.3f}   {ref:>6.2f}   {pct_error(pred, ref):>7.3f}%")

    arch = data["architecture"]
    cache = data["cache_validation"]
    print()
    print(
        f"hybrid stack: {arch['full_attention_layers']} full-attn + "
        f"{arch['linear_attention_layers']} linear-attn layers"
    )
    print(f"FP8 KV:  {cache['fp8_kv_bytes_per_token'] / 1024:.1f} KiB/token")
    print(f"BF16 KV: {cache['bf16_kv_bytes_per_token'] / 1024:.1f} KiB/token")


if __name__ == "__main__":
    main()
