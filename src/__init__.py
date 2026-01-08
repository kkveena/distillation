"""
Email Intent Classification System with Teacher-Student Distillation

This package implements a production-grade email classification system
based on the "Distilling Step-by-Step" approach (Wang et al., 2023).

Architecture:
- Teacher Model (Gemini): Generates high-quality labels and rationales
- Student Model (Gemma): Low-latency inference with calibrated confidence
- Cascaded Inference: Fallback to teacher for low-confidence predictions
"""

__version__ = "0.1.0"
