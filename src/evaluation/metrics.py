"""
Comprehensive evaluation metrics for email intent classification.

Implements all metrics described in the system design:
1. Classification Metrics: Precision, Recall, F1 (micro/macro)
2. Abstention-Aware Metrics: Coverage, Selective Accuracy
3. Calibration Metrics: ECE, Reliability Diagrams
4. Cost-Sensitive Analysis: High-risk error categorization
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class ClassificationMetrics:
    """Standard classification metrics."""
    accuracy: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class_precision: dict
    per_class_recall: dict
    per_class_f1: dict
    confusion_matrix: np.ndarray

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["confusion_matrix"] = self.confusion_matrix.tolist()
        return result


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float
    reliability_diagram_data: dict  # For plotting

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AbstentionMetrics:
    """Metrics for abstention-aware evaluation."""
    coverage: float  # Fraction of non-abstained predictions
    selective_accuracy: float  # Accuracy on non-abstained predictions
    abstention_rate: float
    risk_coverage_auc: float  # Area under risk-coverage curve


def compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.

    Args:
        predictions: [N] predicted labels
        labels: [N] true labels
        label_names: List of label names

    Returns:
        ClassificationMetrics with all computed values
    """
    # Overall accuracy
    accuracy = (predictions == labels).mean()

    # Micro-averaged metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )

    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    # Per-class metrics
    precision_per, recall_per, f1_per, _ = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )

    per_class_precision = {
        label_names[i]: float(precision_per[i])
        for i in range(len(label_names))
        if i < len(precision_per)
    }
    per_class_recall = {
        label_names[i]: float(recall_per[i])
        for i in range(len(label_names))
        if i < len(recall_per)
    }
    per_class_f1 = {
        label_names[i]: float(f1_per[i])
        for i in range(len(label_names))
        if i < len(f1_per)
    }

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision_micro=float(precision_micro),
        recall_micro=float(recall_micro),
        f1_micro=float(f1_micro),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=cm,
    )


def compute_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """
    Compute calibration metrics including ECE and reliability diagram data.

    Args:
        confidences: [N] predicted confidence values
        predictions: [N] predicted labels
        labels: [N] true labels
        n_bins: Number of bins for ECE computation

    Returns:
        CalibrationMetrics with all computed values
    """
    # Bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        bin_counts.append(int(in_bin.sum()))

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            confidence_in_bin = confidences[in_bin].mean()

            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(confidence_in_bin))

            gap = abs(accuracy_in_bin - confidence_in_bin)
            ece += gap * prop_in_bin
            mce = max(mce, gap)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    # Brier score
    correct = (predictions == labels).astype(float)
    brier_score = float(((confidences - correct) ** 2).mean())

    # Reliability diagram data
    reliability_data = {
        "bin_edges": bin_boundaries.tolist(),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
    }

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        brier_score=brier_score,
        reliability_diagram_data=reliability_data,
    )


def compute_abstention_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> AbstentionMetrics:
    """
    Compute abstention-aware metrics.

    Args:
        confidences: [N] predicted confidence values
        predictions: [N] predicted labels
        labels: [N] true labels
        threshold: Confidence threshold for abstention

    Returns:
        AbstentionMetrics with coverage and selective accuracy
    """
    # Non-abstained predictions
    non_abstained = confidences >= threshold
    coverage = float(non_abstained.mean())
    abstention_rate = 1.0 - coverage

    # Selective accuracy
    if non_abstained.sum() > 0:
        selective_accuracy = float(
            (predictions[non_abstained] == labels[non_abstained]).mean()
        )
    else:
        selective_accuracy = 0.0

    # Risk-coverage curve AUC
    rc_auc = _compute_risk_coverage_auc(confidences, predictions, labels)

    return AbstentionMetrics(
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        abstention_rate=abstention_rate,
        risk_coverage_auc=rc_auc,
    )


def _compute_risk_coverage_auc(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_points: int = 100,
) -> float:
    """
    Compute area under the risk-coverage curve.

    The risk-coverage curve shows the trade-off between coverage
    (fraction of predictions made) and risk (error rate).
    """
    thresholds = np.linspace(0, 1, n_points)
    coverages = []
    risks = []

    for thresh in thresholds:
        non_abstained = confidences >= thresh
        coverage = non_abstained.mean()
        coverages.append(coverage)

        if non_abstained.sum() > 0:
            risk = 1.0 - (predictions[non_abstained] == labels[non_abstained]).mean()
        else:
            risk = 0.0
        risks.append(risk)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(risks, coverages)
    return float(auc)


def compute_coverage_accuracy_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_points: int = 100,
) -> dict:
    """
    Compute coverage-accuracy curve data.

    This helps identify optimal operating points for the confidence threshold.

    Returns:
        Dictionary with thresholds, coverages, and accuracies
    """
    thresholds = np.linspace(0, 1, n_points)
    coverages = []
    accuracies = []

    for thresh in thresholds:
        non_abstained = confidences >= thresh
        coverage = non_abstained.mean()
        coverages.append(coverage)

        if non_abstained.sum() > 0:
            accuracy = (predictions[non_abstained] == labels[non_abstained]).mean()
        else:
            accuracy = 0.0
        accuracies.append(accuracy)

    return {
        "thresholds": thresholds.tolist(),
        "coverages": coverages,
        "accuracies": accuracies,
    }


def analyze_errors(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    label_names: list[str],
    high_risk_labels: Optional[list[str]] = None,
) -> dict:
    """
    Perform cost-sensitive error analysis.

    Categorizes errors by severity and identifies high-risk misclassifications.

    Args:
        predictions: [N] predicted labels
        labels: [N] true labels
        confidences: [N] confidence values
        label_names: List of label names
        high_risk_labels: Labels with high operational risk

    Returns:
        Dictionary with error analysis results
    """
    errors = predictions != labels

    if high_risk_labels is None:
        high_risk_labels = []

    high_risk_indices = [label_names.index(l) for l in high_risk_labels if l in label_names]

    # High-cost false positives: predicted high-risk but actually not
    high_cost_fp = errors & np.isin(predictions, high_risk_indices) & ~np.isin(labels, high_risk_indices)

    # High-cost false negatives: actually high-risk but predicted as not
    high_cost_fn = errors & ~np.isin(predictions, high_risk_indices) & np.isin(labels, high_risk_indices)

    # Error breakdown
    error_analysis = {
        "total_errors": int(errors.sum()),
        "error_rate": float(errors.mean()),
        "high_cost_false_positives": int(high_cost_fp.sum()),
        "high_cost_false_negatives": int(high_cost_fn.sum()),
        "high_confidence_errors": int((errors & (confidences > 0.8)).sum()),
        "low_confidence_errors": int((errors & (confidences < 0.5)).sum()),
    }

    # Most common error pairs
    error_pairs = []
    for i in range(len(predictions)):
        if errors[i]:
            error_pairs.append((
                label_names[labels[i]] if labels[i] < len(label_names) else str(labels[i]),
                label_names[predictions[i]] if predictions[i] < len(label_names) else str(predictions[i]),
            ))

    from collections import Counter
    error_counts = Counter(error_pairs)
    error_analysis["most_common_errors"] = [
        {"true": pair[0], "predicted": pair[1], "count": count}
        for pair, count in error_counts.most_common(10)
    ]

    return error_analysis


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    label_names: list[str],
    device: torch.device,
    confidence_threshold: float = 0.5,
    temperature: float = 1.0,
) -> dict:
    """
    Comprehensive model evaluation.

    Args:
        model: The model to evaluate
        dataloader: Evaluation data loader
        label_names: List of label names
        device: Device to use
        confidence_threshold: Threshold for abstention
        temperature: Temperature for calibration

    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids, attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Apply temperature scaling
    logits = logits / temperature

    # Compute predictions and confidences
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    predictions = probs.argmax(axis=-1)
    confidences = probs.max(axis=-1)

    # Compute all metrics
    classification = compute_classification_metrics(predictions, labels, label_names)
    calibration = compute_calibration_metrics(confidences, predictions, labels)
    abstention = compute_abstention_metrics(confidences, predictions, labels, confidence_threshold)
    coverage_curve = compute_coverage_accuracy_curve(confidences, predictions, labels)
    error_analysis = analyze_errors(predictions, labels, confidences, label_names)

    return {
        "classification": classification.to_dict(),
        "calibration": calibration.to_dict(),
        "abstention": asdict(abstention),
        "coverage_curve": coverage_curve,
        "error_analysis": error_analysis,
    }


def generate_evaluation_report(
    metrics: dict,
    output_path: str,
    include_plots: bool = True,
) -> str:
    """
    Generate a comprehensive evaluation report.

    Args:
        metrics: Dictionary of evaluation metrics
        output_path: Path to save the report
        include_plots: Whether to generate visualization plots

    Returns:
        Path to the generated report
    """
    report_lines = [
        "=" * 60,
        "EMAIL INTENT CLASSIFICATION - EVALUATION REPORT",
        "=" * 60,
        "",
        "CLASSIFICATION METRICS",
        "-" * 40,
        f"Accuracy: {metrics['classification']['accuracy']:.4f}",
        "",
        "Micro-averaged:",
        f"  Precision: {metrics['classification']['precision_micro']:.4f}",
        f"  Recall: {metrics['classification']['recall_micro']:.4f}",
        f"  F1-score: {metrics['classification']['f1_micro']:.4f}",
        "",
        "Macro-averaged:",
        f"  Precision: {metrics['classification']['precision_macro']:.4f}",
        f"  Recall: {metrics['classification']['recall_macro']:.4f}",
        f"  F1-score: {metrics['classification']['f1_macro']:.4f}",
        "",
        "CALIBRATION METRICS",
        "-" * 40,
        f"Expected Calibration Error (ECE): {metrics['calibration']['ece']:.4f}",
        f"Maximum Calibration Error (MCE): {metrics['calibration']['mce']:.4f}",
        f"Brier Score: {metrics['calibration']['brier_score']:.4f}",
        "",
        "ABSTENTION METRICS",
        "-" * 40,
        f"Coverage: {metrics['abstention']['coverage']:.4f}",
        f"Selective Accuracy: {metrics['abstention']['selective_accuracy']:.4f}",
        f"Abstention Rate: {metrics['abstention']['abstention_rate']:.4f}",
        f"Risk-Coverage AUC: {metrics['abstention']['risk_coverage_auc']:.4f}",
        "",
        "ERROR ANALYSIS",
        "-" * 40,
        f"Total Errors: {metrics['error_analysis']['total_errors']}",
        f"Error Rate: {metrics['error_analysis']['error_rate']:.4f}",
        f"High-Confidence Errors (>0.8): {metrics['error_analysis']['high_confidence_errors']}",
        "",
        "=" * 60,
    ]

    report = "\n".join(report_lines)

    # Save report
    with open(output_path, "w") as f:
        f.write(report)

    # Save full metrics as JSON
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if include_plots:
        _generate_plots(metrics, output_path.replace(".txt", ""))

    return output_path


def _generate_plots(metrics: dict, output_prefix: str):
    """Generate visualization plots for the evaluation."""
    try:
        import matplotlib.pyplot as plt

        # Reliability diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        rel_data = metrics["calibration"]["reliability_diagram_data"]
        bin_centers = [(rel_data["bin_edges"][i] + rel_data["bin_edges"][i + 1]) / 2
                       for i in range(len(rel_data["bin_edges"]) - 1)]

        ax.bar(bin_centers, rel_data["bin_accuracies"], width=0.05, alpha=0.7, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram")
        ax.legend()
        plt.savefig(f"{output_prefix}_reliability.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Coverage-accuracy curve
        fig, ax = plt.subplots(figsize=(8, 6))
        curve_data = metrics["coverage_curve"]
        ax.plot(curve_data["coverages"], curve_data["accuracies"])
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective Accuracy")
        ax.set_title("Coverage-Accuracy Trade-off")
        plt.savefig(f"{output_prefix}_coverage.png", dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError:
        pass  # Skip plotting if matplotlib not available
