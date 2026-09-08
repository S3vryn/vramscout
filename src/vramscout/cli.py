from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from .gpu import GPUDetectionError, get_gpu
from .model import ModelInspectionError, inspect_model
from .planner import plan_inference


def _positive_int(value: str) -> int:
    parsed = int(value.replace("_", ""))
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vramscout",
        description="Check whether a Hugging Face LLM fits the current GPU and estimate the maximum context length.",
    )
    p.add_argument("model", help="Hugging Face model id, local model directory, or local config.json")
    p.add_argument("--revision", default=None, help="Optional Hugging Face revision/commit")
    p.add_argument("--context", type=_positive_int, default=None, help="Context length to evaluate (default: min(8192, model limit))")
    p.add_argument("--batch-size", type=_positive_int, default=1, help="Active sequences / batch size (default: 1)")
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16", "fp8", "int8", "int4", "nvfp4"], default="auto", help="Weight precision")
    p.add_argument("--kv-dtype", choices=["auto", "fp32", "fp16", "bf16", "fp8"], default="auto", help="KV-cache precision")
    p.add_argument("--gpu-index", type=int, default=0, help="NVIDIA GPU index (default: 0)")
    p.add_argument("--vram-gib", type=float, default=None, help="Manual free/total VRAM budget; skips GPU auto-detection")
    p.add_argument("--reserve-gib", type=float, default=None, help="Override safety reserve (default: max(1 GiB, 5%% of total VRAM))")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return p


def _print_human(result) -> None:
    console = Console()
    status = "[bold green]CAN RUN[/bold green]" if result.fits else "[bold red]DOES NOT FIT[/bold red]"
    console.print(f"\n[bold]VRAMScout[/bold]  {status}\n")

    gpu = Table(title="GPU", show_header=False)
    gpu.add_column("Field", style="bold")
    gpu.add_column("Value", justify="right")
    gpu.add_row("Device", f"GPU {result.gpu.index} · {result.gpu.name}")
    gpu.add_row("Total VRAM", f"{result.gpu.total_gib:.2f} GiB")
    gpu.add_row("Currently used", f"{result.gpu.used_gib:.2f} GiB")
    gpu.add_row("Currently free", f"{result.gpu.free_gib:.2f} GiB")
    console.print(gpu)

    model = Table(title="Model", show_header=False)
    model.add_column("Field", style="bold")
    model.add_column("Value", justify="right")
    model.add_row("Checkpoint", result.model.model_id)
    model.add_row("Architecture", result.model.model_type)
    if result.model.cache_kind != "standard":
        model.add_row("Cache architecture", f"{result.model.cache_kind} · {result.model.effective_kv_layers} KV + {result.model.recurrent_layers} recurrent layers")
    if result.model.has_vision_encoder:
        model.add_row("Vision encoder", "included in checkpoint weights")
    model.add_row("Parameters", f"{result.model.num_params / 1e9:.3f} B")
    model.add_row("Param source", result.model.parameter_source)
    model.add_row("Weights", result.weight_dtype)
    model.add_row("KV cache", result.kv_dtype)
    model.add_row("Batch size", str(result.batch_size))
    model.add_row("Requested context", f"{result.context:,}")
    if result.model.max_context:
        model.add_row("Model-declared limit", f"{result.model.max_context:,}")
    console.print(model)

    b = result.breakdown
    mem = Table(title=f"VRAM breakdown @ {result.context:,} tokens")
    mem.add_column("Part")
    mem.add_column("GiB", justify="right")
    mem.add_row("Model weights", f"{b.weights_gib:.2f}")
    kv_label = "KV cache" if result.model.cache_kind == "standard" else f"KV cache ({result.model.effective_kv_layers} full-attn layers)"
    mem.add_row(kv_label, f"{b.kv_cache_gib:.2f}")
    if b.recurrent_state_gib > 0:
        mem.add_row(result.model.recurrent_state_label or "Recurrent state", f"{b.recurrent_state_gib:.2f}")
    mem.add_row("CUDA / runtime", f"{b.runtime_fixed_gib:.2f}")
    mem.add_row("Prefill scratch", f"{b.prefill_scratch_gib:.2f}")
    mem.add_row("Safety reserve", f"{b.safety_reserve_gib:.2f}")
    mem.add_section()
    mem.add_row("Estimated peak", f"[bold]{b.total_gib:.2f}[/bold]")
    mem.add_row("Free after estimate", f"{result.spare_gib:.2f}")
    console.print(mem)

    ctx = Table(title="Context budget", show_header=False)
    ctx.add_column("Field", style="bold")
    ctx.add_column("Tokens", justify="right")
    ctx.add_row("VRAM-derived max", f"{result.max_context_vram:,}")
    ctx.add_row("Usable max", f"[bold]{result.max_context_usable:,}[/bold]")
    console.print(ctx)

    if result.warnings:
        console.print("\n[bold yellow]Notes[/bold yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        gpu = get_gpu(index=args.gpu_index, vram_gib=args.vram_gib)
        model = inspect_model(args.model, revision=args.revision)
        result = plan_inference(
            gpu=gpu,
            model=model,
            context=args.context,
            batch_size=args.batch_size,
            weight_dtype=args.dtype,
            kv_dtype=args.kv_dtype,
            safety_reserve_gib=args.reserve_gib,
        )
    except (GPUDetectionError, ModelInspectionError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _print_human(result)
    return 0 if result.fits else 2


if __name__ == "__main__":
    sys.exit(main())
