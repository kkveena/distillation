"""
Inference module for email intent classification.

Implements:
- Confidence-based decision gating
- Cascaded inference (student -> teacher fallback)
- Standardized decision payload generation
"""

from .predictor import IntentPredictor, DecisionPayload
from .cascaded_inference import CascadedClassifier

__all__ = [
    "IntentPredictor",
    "DecisionPayload",
    "CascadedClassifier",
]
