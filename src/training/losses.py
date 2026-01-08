"""
Loss functions for step-by-step distillation training.

Implements multi-task learning objectives:
1. Label Loss: Cross-entropy for intent classification
2. Rationale Alignment Loss: Encourages student to match teacher's reasoning
3. Calibration Loss: Ensures confidence matches accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelLoss(nn.Module):
    """
    Standard cross-entropy loss for intent label prediction.

    This loss teaches the student WHAT answer to give.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes] classification logits
            labels: [batch] true intent labels

        Returns:
            Cross-entropy loss
        """
        return self.criterion(logits, labels)


class RationaleAlignmentLoss(nn.Module):
    """
    Loss for aligning student reasoning with teacher rationales.

    This loss teaches the student HOW to reason by:
    1. Training the student to generate similar rationales as the teacher
    2. Encouraging consistent internal representations

    This is a key component of step-by-step distillation that enables
    the student to learn the teacher's reasoning process, not just outputs.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        use_kl_divergence: bool = False,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.use_kl_divergence = use_kl_divergence
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        rationale_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute rationale alignment loss.

        Args:
            student_logits: [batch, seq_len, vocab_size] student's generation logits
            rationale_ids: [batch, seq_len] teacher rationale token ids
            attention_mask: [batch, seq_len] mask for valid positions
            teacher_logits: [batch, seq_len, vocab_size] optional teacher logits for KL loss

        Returns:
            Alignment loss value
        """
        if self.use_kl_divergence and teacher_logits is not None:
            # KL divergence between student and teacher distributions
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            # KL(teacher || student)
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum(dim=-1)

            # Apply mask
            if attention_mask is not None:
                kl_loss = kl_loss * attention_mask
                return kl_loss.sum() / attention_mask.sum()
            return kl_loss.mean()
        else:
            # Standard cross-entropy on rationale tokens
            # Shift for causal LM: predict next token
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = rationale_ids[:, 1:].contiguous()

            # Flatten
            vocab_size = shift_logits.size(-1)
            loss = self.ce_loss(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
            )
            return loss


class DistillationLoss(nn.Module):
    """
    Combined loss for step-by-step distillation training.

    Total Loss = α * Label_Loss + β * Rationale_Loss + γ * Calibration_Loss

    Where:
    - Label Loss teaches WHAT to predict
    - Rationale Loss teaches HOW to reason
    - Calibration Loss teaches WHEN to be confident
    """

    def __init__(
        self,
        label_weight: float = 1.0,
        rationale_weight: float = 0.5,
        calibration_weight: float = 0.1,
        label_smoothing: float = 0.0,
        calibration_bins: int = 15,
    ):
        super().__init__()
        self.label_weight = label_weight
        self.rationale_weight = rationale_weight
        self.calibration_weight = calibration_weight

        self.label_loss = LabelLoss(label_smoothing=label_smoothing)
        self.rationale_loss = RationaleAlignmentLoss()
        self.calibration_bins = calibration_bins

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        rationale_logits: Optional[torch.Tensor] = None,
        rationale_ids: Optional[torch.Tensor] = None,
        rationale_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.

        Args:
            logits: [batch, num_classes] classification logits
            labels: [batch] true intent labels
            rationale_logits: [batch, seq_len, vocab_size] generation logits
            rationale_ids: [batch, seq_len] teacher rationale tokens
            rationale_mask: [batch, seq_len] attention mask for rationale

        Returns:
            Dictionary with total loss and component losses
        """
        losses = {}

        # Label prediction loss
        label_loss = self.label_loss(logits, labels)
        losses["label_loss"] = label_loss

        total_loss = self.label_weight * label_loss

        # Rationale alignment loss
        if rationale_logits is not None and rationale_ids is not None:
            rat_loss = self.rationale_loss(
                rationale_logits,
                rationale_ids,
                rationale_mask,
            )
            losses["rationale_loss"] = rat_loss
            total_loss = total_loss + self.rationale_weight * rat_loss

        # Calibration loss (differentiable ECE approximation)
        if self.calibration_weight > 0:
            cal_loss = self._compute_calibration_loss(logits, labels)
            losses["calibration_loss"] = cal_loss
            total_loss = total_loss + self.calibration_weight * cal_loss

        losses["total_loss"] = total_loss
        return losses

    def _compute_calibration_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute differentiable calibration loss.

        This encourages the model to produce calibrated confidence scores
        throughout training, rather than relying only on post-hoc calibration.
        """
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        correct = (predictions == labels).float()

        # Soft binning for differentiability
        bin_boundaries = torch.linspace(0, 1, self.calibration_bins + 1, device=logits.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        calibration_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Soft membership using sigmoid
            in_bin = torch.sigmoid(20 * (confidences - bin_lower)) * \
                     torch.sigmoid(20 * (bin_upper - confidences))

            bin_size = in_bin.sum()
            if bin_size > 0:
                # Compute accuracy and average confidence in bin
                bin_accuracy = (in_bin * correct).sum() / (bin_size + 1e-8)
                bin_confidence = (in_bin * confidences).sum() / (bin_size + 1e-8)

                # Squared calibration error weighted by bin size
                calibration_loss = calibration_loss + \
                    (bin_accuracy - bin_confidence) ** 2 * (bin_size / logits.size(0))

        return calibration_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in intent classification.

    Focal loss down-weights well-classified examples and focuses training
    on hard, misclassified examples. This is useful for rare intents.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes]
            labels: [batch]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of correct class

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            focal_weight = focal_weight * alpha_t

        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
