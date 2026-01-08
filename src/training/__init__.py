"""
Training module for step-by-step distillation.

This module implements the "Distilling Step-by-Step" training approach:
- Label prediction loss (what decision to make)
- Rationale alignment loss (how to reason)
- Calibration loss (when to be uncertain)
"""

from .losses import DistillationLoss, LabelLoss, RationaleAlignmentLoss
from .trainer import DistillationTrainer

__all__ = [
    "DistillationLoss",
    "LabelLoss",
    "RationaleAlignmentLoss",
    "DistillationTrainer",
]
