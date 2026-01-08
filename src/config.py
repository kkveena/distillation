"""
Configuration classes for the Email Intent Classification System.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class IntentTaxonomy:
    """Intent taxonomy configuration."""
    labels: list[str] = field(default_factory=lambda: [
        "inquiry",
        "complaint",
        "request",
        "confirmation",
        "cancellation",
        "feedback",
        "escalation",
        "follow_up",
        "out_of_scope",
    ])

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def label_to_id(self, label: str) -> int:
        return self.labels.index(label)

    def id_to_label(self, idx: int) -> str:
        return self.labels[idx]


@dataclass
class TeacherModelConfig:
    """Configuration for the teacher model (Gemini)."""
    model_name: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_output_tokens: int = 1024
    num_consistency_samples: int = 3  # For confidence estimation via consistency


@dataclass
class StudentModelConfig:
    """Configuration for the student model (Gemma)."""
    model_name: str = "google/gemma-2b"
    max_length: int = 512
    hidden_size: int = 2048
    num_attention_heads: int = 8
    num_hidden_layers: int = 18
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Paths
    output_dir: str = "./outputs"
    data_dir: str = "./data"

    # Training hyperparameters
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Loss weights for multi-task learning
    label_loss_weight: float = 1.0
    rationale_loss_weight: float = 0.5
    calibration_loss_weight: float = 0.1

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100

    # Hardware
    fp16: bool = True
    dataloader_num_workers: int = 4

    # Reproducibility
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Confidence thresholds
    acceptance_threshold: float = 0.8  # Accept prediction if confidence >= threshold
    abstention_threshold: float = 0.5  # Abstain if confidence < threshold

    # Cascaded inference
    enable_teacher_fallback: bool = True
    fallback_threshold: float = 0.6  # Use teacher if student confidence < threshold

    # Calibration
    temperature: float = 1.0  # Will be learned during calibration

    # Batch processing
    batch_size: int = 32
    max_length: int = 512


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    taxonomy: IntentTaxonomy = field(default_factory=IntentTaxonomy)
    teacher: TeacherModelConfig = field(default_factory=TeacherModelConfig)
    student: StudentModelConfig = field(default_factory=StudentModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            taxonomy=IntentTaxonomy(**data.get('taxonomy', {})),
            teacher=TeacherModelConfig(**data.get('teacher', {})),
            student=StudentModelConfig(**data.get('student', {})),
            training=TrainingConfig(**data.get('training', {})),
            inference=InferenceConfig(**data.get('inference', {})),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
