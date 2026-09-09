# VRAMScout v0.7.0

VRAMScout is now a composable, validation-backed deployment-memory preflight tool for modern open-weight LLMs.

## Highlights

- Refactored cache/state accounting into reusable memory components instead of one monolithic formula per model family.
- Covers current families including Qwen3.8 / Flash-Next, DeepSeek-V4, GLM-5.2/5.3, Kimi-K2.x/K3, Hy4, MiniMax-M2.7/M3, Gemma 4, Nemotron 3.5 Lightning, Mistral Medium 3.5 / Small 4, LongCat-2.0 and MiMo-V2.5/Pro.
- Adds vLLM memory-budget semantics, TP/DCP-aware placement and explicit engine-layout calibration.
- Adds a public `Predicted vs Actual` corpus from vLLM/SGLang deployment receipts.
- Adds `vramscout --parse-log` for normalizing vLLM/SGLang startup telemetry.
- Adds Chinese documentation in `README_CN.md`.

## Validation snapshot

Across six new hard cache receipts added in v0.7:

- architecture-only MAPE: **2.850%**
- audited public-profile fit after explicit calibration: **0.237%**

These are cache/layout validation figures, not a guarantee that total process peak or max-context OOM boundaries have the same error.

See `validation/PREDICTED_VS_ACTUAL.md` for the full auditable table and sources.

## Install from source

```bash
git clone https://github.com/S3vryn/vramscout.git
cd vramscout
pip install -e .
```

PyPI publishing is prepared separately; until the trusted publisher is authorized on PyPI, source installation remains the supported path.
