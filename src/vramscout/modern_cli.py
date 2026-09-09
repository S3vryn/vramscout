from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from .gpu import GPUDetectionError
from .modern_core import ModernInspectionError
from .modern_v06 import plan_modern


CURRENT_FAMILIES = [
    "Qwen3.8-27B / Qwen3.8-2.4T-A95B",
    "Qwen3.8-Flash-Next",
    "DeepSeek-V4 Flash / Pro / Vision",
    "GLM-5.2 / GLM-5.3-Flash",
    "Kimi-K2.5 / K2.6 / K3",
    "Tencent Hy4-preview",
    "MiniMax-M2.7 / MiniMax-M3",
    "Gemma 4",
    "NVIDIA Nemotron-3.5-Lightning",
    "Mistral Medium 3.5 / Small 4",
    "LongCat-2.0",
    "MiMo-V2.5 / V2.5-Pro",
    "ordinary MHA/GQA/MQA decoder Transformers",
]


def _positive_int(value: str) -> int:
    out = int(value.replace("_", ""))
    if out <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vramscout",
        description="Preflight modern LLM deployment VRAM: weights, cache/state, runtime, and max context.",
    )
    p.add_argument("model", nargs="?", help="Hugging Face model id, local model directory, or config.json")
    p.add_argument("--revision", default=None)
    p.add_argument("--context", type=_positive_int, default=None)
    p.add_argument("--batch-size", type=_positive_int, default=1)
    p.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16", "fp8", "int8", "int4", "nvfp4"], default="auto")
    p.add_argument("--kv-dtype", choices=["auto", "fp32", "fp16", "bf16", "fp8"], default="auto")
    p.add_argument("--indexer-dtype", choices=["bf16", "fp16", "fp8", "fp4"], default="bf16")
    p.add_argument("--tp", type=_positive_int, default=1, help="tensor-parallel ranks")
    p.add_argument("--dcp", type=_positive_int, default=1, help="decode-context-parallel ranks; must divide TP")
    p.add_argument("--engine", choices=["generic", "vllm"], default="generic")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--prefill-chunk", type=_positive_int, default=8192)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--vram-gib", type=float, default=None, help="manual VRAM budget per GPU")
    p.add_argument("--reserve-gib", type=float, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--list-supported", action="store_true")
    return p


def _human(r) -> None:
    c = Console()
    status = "[bold green]CAN RUN[/bold green]" if r.fits else "[bold red]DOES NOT FIT[/bold red]"
    c.print(f"\n[bold]VRAMScout[/bold]  {status}\n")

    t = Table(title="Deployment budget", show_header=False)
    t.add_column("Field", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("GPU / group", r.gpu_name)
    t.add_row("TP / DCP", f"{r.tp} / {r.dcp}")
    t.add_row("Physical free / rank", f"{r.gpu_free_gib:.2f} GiB")
    t.add_row("Engine", r.engine)
    t.add_row("Planning budget / rank", f"{r.memory_budget_gib:.2f} GiB")
    t.add_row("Engine startup", "OK" if r.startup_ok else "BLOCKED")
    c.print(t)

    p = r.profile
    m = Table(title="Model", show_header=False)
    m.add_column("Field", style="bold")
    m.add_column("Value", justify="right")
    m.add_row("Checkpoint", p.model_id)
    m.add_row("Model type", p.model_type)
    m.add_row("Memory family", p.family)
    m.add_row("Parameters", f"{p.num_params / 1e9:.3f} B")
    m.add_row("Weight dtype", r.weight_dtype)
    m.add_row("Cache dtype", r.kv_dtype)
    m.add_row("Requested context", f"{r.context:,}")
    if p.max_context:
        m.add_row("Config context limit", f"{p.max_context:,}")
    if p.has_vision:
        m.add_row("Vision", "checkpoint weights included")
    if p.has_audio:
        m.add_row("Audio", "checkpoint weights included")
    c.print(m)

    b = Table(title=f"Per-rank VRAM breakdown @ {r.context:,} tokens")
    b.add_column("Part")
    b.add_column("GiB", justify="right")
    for name, gib in r.parts_gib.items():
        b.add_row(name, f"{gib:.3f}")
    b.add_section()
    b.add_row("Estimated peak / rank", f"[bold]{r.total_gib:.3f}[/bold]")
    b.add_row("Spare inside budget", f"{r.spare_gib:.3f}")
    c.print(b)

    ctx = Table(title="Context budget", show_header=False)
    ctx.add_column("Field", style="bold")
    ctx.add_column("Tokens", justify="right")
    ctx.add_row("VRAM-derived max", f"{r.max_context_vram:,}")
    ctx.add_row("Usable max", f"[bold]{r.max_context_usable:,}[/bold]")
    c.print(ctx)

    if r.notes:
        c.print("\n[bold yellow]Notes[/bold yellow]")
        for note in r.notes:
            c.print(f"  • {note}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_supported:
        Console().print("[bold]Current architecture coverage[/bold]")
        for x in CURRENT_FAMILIES:
            Console().print(f"  • {x}")
        return 0
    if not args.model:
        _parser().error("model is required unless --list-supported is used")

    try:
        result = plan_modern(
            args.model,
            revision=args.revision,
            context=args.context,
            batch_size=args.batch_size,
            weight_dtype=args.dtype,
            kv_dtype=args.kv_dtype,
            indexer_dtype=args.indexer_dtype,
            tp=args.tp,
            dcp=args.dcp,
            gpu_index=args.gpu_index,
            vram_gib=args.vram_gib,
            reserve_gib=args.reserve_gib,
            prefill_chunk=args.prefill_chunk,
            engine=args.engine,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    except (GPUDetectionError, ModernInspectionError, ValueError) as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            Console(stderr=True).print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        _human(result)
    return 0 if result.fits else 2


if __name__ == "__main__":
    sys.exit(main())
