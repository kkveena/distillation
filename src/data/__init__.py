"""
Data module for email intent classification.

This module handles:
- Email preprocessing and normalization
- Dataset creation for training and evaluation
- Teacher-generated supervision (labels, rationales)
"""

from .preprocessing import EmailPreprocessor
from .dataset import EmailIntentDataset, DistillationDataset
from .teacher_generation import TeacherDataGenerator

__all__ = [
    "EmailPreprocessor",
    "EmailIntentDataset",
    "DistillationDataset",
    "TeacherDataGenerator",
]
