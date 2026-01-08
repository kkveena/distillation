"""
Student model implementation using Gemma for email intent classification.

The student model:
- Uses a pretrained Gemma model as the backbone
- Adds a classification head for intent prediction
- Supports LoRA for efficient fine-tuning
- Produces calibrated confidence scores
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)

from ..config import StudentModelConfig, IntentTaxonomy


@dataclass
class StudentModelOutput:
    """Output from the student model."""
    logits: torch.Tensor  # Classification logits [batch, num_labels]
    loss: Optional[torch.Tensor] = None  # Total loss if labels provided
    label_loss: Optional[torch.Tensor] = None  # Label prediction loss
    rationale_loss: Optional[torch.Tensor] = None  # Rationale alignment loss
    hidden_states: Optional[torch.Tensor] = None  # Last hidden states
    rationale_logits: Optional[torch.Tensor] = None  # Rationale generation logits


class IntentClassificationHead(nn.Module):
    """Classification head for intent prediction."""

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size]

        Returns:
            logits: [batch, num_labels]
        """
        # Use CLS token or mean pooling
        if hidden_states.dim() == 3:
            # Mean pooling over sequence length
            hidden_states = hidden_states.mean(dim=1)

        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class StudentModel(nn.Module):
    """
    Student model for email intent classification.

    Architecture:
    - Gemma backbone for contextual representations
    - Classification head for intent prediction
    - Optional LoRA adapters for efficient fine-tuning
    """

    def __init__(
        self,
        config: StudentModelConfig,
        taxonomy: IntentTaxonomy,
    ):
        super().__init__()
        self.config = config
        self.taxonomy = taxonomy
        self.num_labels = taxonomy.num_labels

        # Load backbone model
        self.backbone = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        # Get hidden size from backbone config
        backbone_config = self.backbone.config
        hidden_size = getattr(backbone_config, "hidden_size", config.hidden_size)

        # Classification head
        self.classifier = IntentClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
            dropout=config.lora_dropout if config.use_lora else 0.1,
        )

        # Apply LoRA if configured
        if config.use_lora:
            self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA adapters to the backbone model."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        except ImportError:
            raise ImportError("peft package required for LoRA. Install with: pip install peft")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> StudentModelOutput:
        """
        Forward pass for classification.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch] optional intent labels

        Returns:
            StudentModelOutput with logits and optional loss
        """
        # Get backbone hidden states
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state
        hidden_states = outputs.last_hidden_state

        # Apply attention mask for mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask

        # Get classification logits
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return StudentModelOutput(
            logits=logits,
            loss=loss,
            label_loss=loss,
            hidden_states=hidden_states,
        )

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with calibrated confidence.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            temperature: Temperature for calibration

        Returns:
            predictions: [batch] predicted intent indices
            confidences: [batch] calibrated confidence scores
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            logits = output.logits / temperature
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = probs.max(dim=-1)
        return predictions, confidences

    @classmethod
    def from_pretrained(cls, path: str, config: StudentModelConfig, taxonomy: IntentTaxonomy):
        """Load model from checkpoint."""
        model = cls(config, taxonomy)
        state_dict = torch.load(f"{path}/model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, path: str):
        """Save model to checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model.pt")


class StudentModelForDistillation(nn.Module):
    """
    Student model with rationale generation capability for distillation training.

    This extends StudentModel with:
    - A rationale generation head for step-by-step distillation
    - Multi-task forward pass supporting label + rationale objectives
    """

    def __init__(
        self,
        config: StudentModelConfig,
        taxonomy: IntentTaxonomy,
    ):
        super().__init__()
        self.config = config
        self.taxonomy = taxonomy
        self.num_labels = taxonomy.num_labels

        # Load causal LM backbone for generation
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

        # Get hidden size
        backbone_config = self.backbone.config
        hidden_size = getattr(backbone_config, "hidden_size", config.hidden_size)
        vocab_size = getattr(backbone_config, "vocab_size", 32000)

        # Classification head (separate from generation)
        self.classifier = IntentClassificationHead(
            hidden_size=hidden_size,
            num_labels=self.num_labels,
        )

        # Store vocab size for rationale loss
        self.vocab_size = vocab_size

        # Apply LoRA if configured
        if config.use_lora:
            self._apply_lora()

    def _apply_lora(self):
        """Apply LoRA adapters."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        except ImportError:
            raise ImportError("peft package required for LoRA. Install with: pip install peft")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        rationale_ids: Optional[torch.Tensor] = None,
        rationale_attention_mask: Optional[torch.Tensor] = None,
    ) -> StudentModelOutput:
        """
        Forward pass with multi-task objectives.

        Args:
            input_ids: [batch, seq_len] email input tokens
            attention_mask: [batch, seq_len]
            labels: [batch] intent labels
            rationale_ids: [batch, rationale_len] rationale tokens for alignment
            rationale_attention_mask: [batch, rationale_len]

        Returns:
            StudentModelOutput with all losses
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]

        # Mean pooling for classification
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask

        # Classification logits
        logits = self.classifier(pooled)

        # Initialize losses
        label_loss = None
        rationale_loss = None
        total_loss = None

        # Label prediction loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            label_loss = loss_fct(logits, labels)
            total_loss = label_loss

        # Rationale alignment loss (language modeling on rationale)
        if rationale_ids is not None:
            # Create combined input: email + rationale
            combined_ids = torch.cat([input_ids, rationale_ids], dim=1)
            combined_mask = torch.cat([attention_mask, rationale_attention_mask], dim=1)

            # Get LM outputs
            lm_outputs = self.backbone(
                input_ids=combined_ids,
                attention_mask=combined_mask,
            )

            # Compute rationale loss only on rationale portion
            lm_logits = lm_outputs.logits
            email_len = input_ids.size(1)

            # Shift for causal LM loss
            shift_logits = lm_logits[:, email_len:-1, :].contiguous()
            shift_labels = rationale_ids[:, 1:].contiguous()

            # Mask padding
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # Assume 0 is pad token
            rationale_loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

            if total_loss is not None:
                total_loss = total_loss + rationale_loss
            else:
                total_loss = rationale_loss

        return StudentModelOutput(
            logits=logits,
            loss=total_loss,
            label_loss=label_loss,
            rationale_loss=rationale_loss,
            hidden_states=hidden_states,
            rationale_logits=outputs.logits if rationale_ids is not None else None,
        )

    def get_classifier_model(self) -> StudentModel:
        """Extract a classification-only model for inference."""
        # This creates a StudentModel with shared weights
        classifier_model = StudentModel.__new__(StudentModel)
        classifier_model.config = self.config
        classifier_model.taxonomy = self.taxonomy
        classifier_model.num_labels = self.num_labels
        classifier_model.backbone = self.backbone.base_model if hasattr(self.backbone, 'base_model') else self.backbone
        classifier_model.classifier = self.classifier
        return classifier_model

    def save_pretrained(self, path: str):
        """Save model to checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model.pt")

    @classmethod
    def from_pretrained(cls, path: str, config: StudentModelConfig, taxonomy: IntentTaxonomy):
        """Load model from checkpoint."""
        model = cls(config, taxonomy)
        state_dict = torch.load(f"{path}/model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        return model
