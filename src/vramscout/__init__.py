"""VRAMScout: architecture- and engine-aware LLM deployment memory planning."""

from .modern_v06 import inspect_modern_model
from .modern_v07 import plan_modern
from .planner import plan_inference

__all__ = ["plan_modern", "inspect_modern_model", "plan_inference"]
__version__ = "0.7.0"
