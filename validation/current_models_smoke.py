from __future__ import annotations

import json
from pathlib import Path

from vramscout.modern_v06 import inspect_modern_model

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "current_models_v06.json"


def main() -> int:
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    failed = 0
    print(f"VRAMScout current-model smoke test · snapshot {data['snapshot_date']}\n")
    print(f"{'model':58} {'family':28} {'params':>10} {'context':>10}")
    print("-" * 112)
    for item in data["models"]:
        ref = item["id"]
        try:
            p = inspect_modern_model(ref)
            params = f"{p.num_params / 1e9:.1f}B"
            context = f"{p.max_context:,}" if p.max_context else "?"
            print(f"{ref:58} {p.family:28} {params:>10} {context:>10}")
        except Exception as exc:
            failed += 1
            print(f"{ref:58} {'ERROR':28} {'-':>10} {'-':>10}")
            print(f"    {type(exc).__name__}: {exc}")
    print()
    if failed:
        print(f"FAILED: {failed} checkpoint(s) could not be inspected.")
        return 1
    print("OK: all registered checkpoints exposed a supported memory family.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
