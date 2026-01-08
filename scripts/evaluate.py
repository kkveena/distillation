#!/usr/bin/env python3
"""
Evaluation script for the trained student model.

This script:
1. Loads a trained model checkpoint
2. Runs comprehensive evaluation on test data
3. Calibrates confidence using temperature scaling
4. Generates evaluation report with all metrics
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import DistillationDataset, collate_distillation_batch
from src.models.student_model import StudentModel, StudentModelForDistillation
from src.models.calibration import calibrate_model, compute_ece
from src.evaluation.metrics import evaluate_model, generate_evaluation_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained student model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test data JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-2b",
        help="Base model name for tokenizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--calibration-data",
        type=str,
        default=None,
        help="Path to calibration data (optional, uses test data if not provided)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for abstention metrics",
    )
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.student.model_name = args.model_name

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.student.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.student.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load test dataset
    logger.info(f"Loading test data from {args.test_data}")
    test_dataset = DistillationDataset.from_jsonl(
        args.test_data,
        tokenizer=tokenizer,
        taxonomy=config.taxonomy,
        max_length=config.student.max_length,
    )
    logger.info(f"Loaded {len(test_dataset)} test samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_distillation_batch,
    )

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = StudentModelForDistillation.from_pretrained(
        args.checkpoint,
        config.student,
        config.taxonomy,
    )
    model.to(device)
    model.eval()

    # Calibration
    if args.calibration_data:
        logger.info(f"Loading calibration data from {args.calibration_data}")
        cal_dataset = DistillationDataset.from_jsonl(
            args.calibration_data,
            tokenizer=tokenizer,
            taxonomy=config.taxonomy,
            max_length=config.student.max_length,
        )
        cal_loader = DataLoader(
            cal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_distillation_batch,
        )
    else:
        logger.info("Using test data for calibration")
        cal_loader = test_loader

    logger.info("Calibrating model with temperature scaling...")
    temperature_scaler = calibrate_model(model, cal_loader, device)
    temperature = temperature_scaler.temperature.item()
    logger.info(f"Optimal temperature: {temperature:.4f}")

    # Comprehensive evaluation
    logger.info("Running comprehensive evaluation...")
    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        label_names=config.taxonomy.labels,
        device=device,
        confidence_threshold=args.confidence_threshold,
        temperature=temperature,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    generate_evaluation_report(
        metrics,
        report_path,
        include_plots=True,
    )
    logger.info(f"Evaluation report saved to {report_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['classification']['accuracy']:.4f}")
    logger.info(f"F1 (macro): {metrics['classification']['f1_macro']:.4f}")
    logger.info(f"ECE: {metrics['calibration']['ece']:.4f}")
    logger.info(f"Coverage: {metrics['abstention']['coverage']:.4f}")
    logger.info(f"Selective Accuracy: {metrics['abstention']['selective_accuracy']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
