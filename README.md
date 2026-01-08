# Email Intent Classification with Step-by-Step Distillation

A production-grade email classification system using teacher-student distillation based on the "Distilling Step-by-Step" approach (Wang et al., 2023).

## Overview

This system implements an email intent classifier that:
- Uses **Gemini** as a high-capacity teacher model for generating labels and rationales
- Trains a **Gemma** student model using multi-task distillation
- Provides **calibrated confidence scores** for safe automated decision-making
- Supports **cascaded inference** with teacher fallback for low-confidence predictions

### Key Features

- **Step-by-Step Distillation**: Student learns both WHAT to predict and HOW to reason
- **Confidence Calibration**: Temperature scaling ensures reliable confidence estimates
- **Abstention Support**: System can abstain when uncertain, avoiding risky predictions
- **Cascaded Inference**: Automatic fallback to teacher for difficult cases
- **Production Ready**: Standardized decision payloads for downstream automation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                              │
├─────────────────────────────────────────────────────────────────┤
│  Emails → Teacher (Gemini) → (label, rationale, confidence)     │
│                    ↓                                              │
│  Student (Gemma) ← Multi-task Loss:                              │
│                    • Label Loss (what to predict)                │
│                    • Rationale Loss (how to reason)              │
│                    • Calibration Loss (when to be uncertain)     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  Email → Student Model → Confidence Score                        │
│                    ↓                                              │
│              Decision Gate                                        │
│           ↙        ↓         ↘                                   │
│     Accept    Fallback    Abstain                                │
│   (conf≥τ₁)  (τ₂≤conf<τ₁)  (conf<τ₂)                            │
│        ↓          ↓           ↓                                  │
│     Output    Teacher      Human                                 │
│     Intent    Inference    Review                                │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/kkveena/distillation.git
cd distillation

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Training Data

Using the teacher model (requires Gemini API key):
```bash
export GOOGLE_API_KEY="your-api-key"
python scripts/generate_training_data.py \
    --input data/raw_emails.jsonl \
    --output data/distillation_train.jsonl
```

Or use mock teacher for testing:
```bash
python scripts/generate_training_data.py --use-mock --output data/distillation_train.jsonl
```

### 2. Train the Student Model

```bash
python scripts/train.py \
    --train-data data/distillation_train.jsonl \
    --output-dir outputs \
    --epochs 3 \
    --batch-size 8 \
    --use-lora \
    --fp16
```

### 3. Evaluate the Model

```bash
python scripts/evaluate.py \
    --checkpoint outputs/best \
    --test-data data/test.jsonl \
    --output-dir evaluation_results
```

### 4. Run Inference

Single email:
```bash
python scripts/inference.py \
    --checkpoint outputs/best \
    --email-subject "Question about my order" \
    --email-body "Hi, I placed an order last week but haven't received it yet."
```

Batch inference:
```bash
python scripts/inference.py \
    --checkpoint outputs/best \
    --input data/new_emails.jsonl \
    --output predictions.jsonl \
    --enable-fallback
```

Interactive mode:
```bash
python scripts/inference.py --checkpoint outputs/best
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Intent taxonomy
taxonomy:
  labels:
    - inquiry
    - complaint
    - request
    # ... add your intents

# Confidence thresholds
inference:
  acceptance_threshold: 0.8   # Accept prediction
  abstention_threshold: 0.5   # Abstain (too uncertain)
  fallback_threshold: 0.6     # Use teacher model

# Training parameters
training:
  label_loss_weight: 1.0      # Classification loss
  rationale_loss_weight: 0.5  # Reasoning alignment
  calibration_loss_weight: 0.1 # Confidence calibration
```

## Decision Payload

The system outputs standardized JSON payloads:

```json
{
  "intent_label": "inquiry",
  "confidence": 0.87,
  "status": "accepted",
  "abstained": false,
  "reason": null,
  "model_version": "student-v1",
  "taxonomy_version": "v1",
  "recommended_action": "route_to_support"
}
```

## Metrics

The system tracks comprehensive metrics:

### Classification Metrics
- Precision, Recall, F1 (micro and macro)
- Per-class performance

### Calibration Metrics
- Expected Calibration Error (ECE)
- Reliability diagrams
- Brier score

### Abstention Metrics
- Coverage (fraction of non-abstained predictions)
- Selective accuracy (accuracy on accepted predictions)
- Risk-coverage curves

## Project Structure

```
distillation/
├── src/
│   ├── config.py              # Configuration classes
│   ├── data/
│   │   ├── preprocessing.py   # Email preprocessing
│   │   ├── dataset.py         # Dataset classes
│   │   └── teacher_generation.py  # Teacher label generation
│   ├── models/
│   │   ├── student_model.py   # Gemma-based student
│   │   └── calibration.py     # Temperature scaling
│   ├── training/
│   │   ├── losses.py          # Multi-task losses
│   │   └── trainer.py         # Training loop
│   ├── inference/
│   │   ├── predictor.py       # Confidence-gated prediction
│   │   └── cascaded_inference.py  # Teacher fallback
│   └── evaluation/
│       └── metrics.py         # Comprehensive metrics
├── scripts/
│   ├── generate_training_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── config.yaml
└── requirements.txt
```

## References

1. Wang et al., "Distilling Step-by-Step: Instruction-Following Language Models Can Be Outperformed by Smaller Models via Step-by-Step Rationales" (2023). [arXiv:2305.02301](https://arxiv.org/abs/2305.02301)

2. Guo et al., "On Calibration of Modern Neural Networks" (2017). [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

## License

MIT License
