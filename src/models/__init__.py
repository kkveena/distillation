"""
Models module for email intent classification.

Contains:
- StudentModel: Low-latency Gemma-based classifier for production inference
- Calibration utilities: Temperature scaling for confidence calibration
"""

from .student_model import StudentModel, StudentModelForDistillation
from .calibration import TemperatureScaler, calibrate_model

__all__ = [
    "StudentModel",
    "StudentModelForDistillation",
    "TemperatureScaler",
    "calibrate_model",
]
