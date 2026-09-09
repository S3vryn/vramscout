"""VRAMScout: architecture-aware LLM deployment memory planning."""

from .modern_core import inspect_modern_model, plan_modern
from .planner import plan_inference

__all__ = ["plan_modern", "inspect_modern_model", "plan_inference"]
__version__ = "0.6.0"
