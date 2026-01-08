"""
Temperature scaling calibration for student model confidence.

Temperature scaling adjusts the softmax temperature to ensure that
predicted confidence scores match empirical accuracy. This is critical
for reliable abstention and cascaded inference decisions.

Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np


class TemperatureScaler(nn.Module):
    """
    Temperature scaling module for post-hoc calibration.

    The temperature parameter T > 0 rescales logits before softmax:
        p(y|x) = softmax(logits / T)

    T > 1 reduces overconfidence (smooths distribution)
    T < 1 increases confidence (sharpens distribution)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        # Temperature as a learnable parameter
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Learn optimal temperature on validation data.

        Args:
            logits: [N, num_classes] uncalibrated logits
            labels: [N] true labels
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        # Use NLL loss for optimization
        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        return self.temperature.item()


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the discrepancy between predicted confidence and
    empirical accuracy across confidence bins.

    Args:
        confidences: [N] predicted confidence values
        predictions: [N] predicted class labels
        labels: [N] true class labels
        n_bins: Number of confidence bins

    Returns:
        ECE value (lower is better calibrated)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # Compute accuracy and confidence in bin
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            confidence_in_bin = confidences[in_bin].mean()

            # Weighted absolute difference
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin

    return ece


def compute_reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute data for reliability diagram.

    Args:
        confidences: [N] predicted confidence values
        predictions: [N] predicted class labels
        labels: [N] true class labels
        n_bins: Number of bins

    Returns:
        bin_centers: [n_bins] center of each bin
        bin_accuracies: [n_bins] accuracy in each bin
        bin_counts: [n_bins] number of samples in each bin
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_counts[i] = in_bin.sum()

        if bin_counts[i] > 0:
            bin_accuracies[i] = (predictions[in_bin] == labels[in_bin]).mean()

    return bin_centers, bin_accuracies, bin_counts


def calibrate_model(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
) -> TemperatureScaler:
    """
    Calibrate a model using temperature scaling on validation data.

    Args:
        model: The model to calibrate
        val_dataloader: Validation data loader
        device: Device to use

    Returns:
        Calibrated TemperatureScaler
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Optimize temperature
    scaler = TemperatureScaler()
    optimal_temp = scaler.calibrate(logits, labels)

    print(f"Optimal temperature: {optimal_temp:.4f}")

    # Compute ECE before and after calibration
    probs_before = F.softmax(logits, dim=-1)
    conf_before, pred_before = probs_before.max(dim=-1)
    ece_before = compute_ece(
        conf_before.numpy(),
        pred_before.numpy(),
        labels.numpy(),
    )

    probs_after = F.softmax(scaler(logits), dim=-1)
    conf_after, pred_after = probs_after.max(dim=-1)
    ece_after = compute_ece(
        conf_after.numpy(),
        pred_after.numpy(),
        labels.numpy(),
    )

    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration: {ece_after:.4f}")

    return scaler


class CalibrationLoss(nn.Module):
    """
    Auxiliary calibration loss to encourage calibrated predictions during training.

    This loss penalizes the model when its confidence doesn't match accuracy,
    encouraging better calibration throughout training rather than only post-hoc.
    """

    def __init__(self, n_bins: int = 15):
        super().__init__()
        self.n_bins = n_bins

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute calibration loss.

        Args:
            logits: [batch, num_classes]
            labels: [batch]

        Returns:
            Calibration loss (differentiable approximation of ECE)
        """
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)

        # Soft binning for differentiability
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        calibration_loss = torch.tensor(0.0, device=logits.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Soft membership (differentiable)
            in_bin = torch.sigmoid(20 * (confidences - bin_lower)) * \
                     torch.sigmoid(20 * (bin_upper - confidences))

            if in_bin.sum() > 0:
                # Weighted accuracy and confidence
                correct = (predictions == labels).float()
                bin_accuracy = (in_bin * correct).sum() / (in_bin.sum() + 1e-8)
                bin_confidence = (in_bin * confidences).sum() / (in_bin.sum() + 1e-8)

                # Add squared difference weighted by bin size
                calibration_loss += (bin_accuracy - bin_confidence) ** 2 * in_bin.sum()

        return calibration_loss / logits.size(0)
