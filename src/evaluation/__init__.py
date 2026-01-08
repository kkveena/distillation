"""
Evaluation module for email intent classification.

Implements comprehensive metrics including:
- Classification metrics (precision, recall, F1)
- Calibration metrics (ECE, reliability diagrams)
- Abstention-aware metrics (coverage, selective accuracy)
- Cost-sensitive error analysis
"""

from .metrics import (
    ClassificationMetrics,
    CalibrationMetrics,
    AbstentionMetrics,
    evaluate_model,
    generate_evaluation_report,
)

__all__ = [
    "ClassificationMetrics",
    "CalibrationMetrics",
    "AbstentionMetrics",
    "evaluate_model",
    "generate_evaluation_report",
]
