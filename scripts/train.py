#!/usr/bin/env python3
"""
Training script for step-by-step distillation.

This script:
1. Loads the distillation dataset
2. Initializes the student model
3. Trains with multi-task objectives (label + rationale + calibration)
4. Evaluates and saves checkpoints
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data.dataset import DistillationDataset, collate_distillation_batch
from src.models.student_model import StudentModelForDistillation
from src.training.trainer import DistillationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train student model with distillation")
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/distillation_train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data (optional, will split from train if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-2b",
        help="Base model name/path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient fine-tuning",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log to Weights & Biases",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio if no val-data provided",
    )
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load configuration
    config = Config()
    config.student.model_name = args.model_name
    config.student.use_lora = args.use_lora
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.training.output_dir = args.output_dir
    config.training.fp16 = args.fp16
    config.training.seed = args.seed

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.student.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.student.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    logger.info(f"Loading training data from {args.train_data}")
    full_dataset = DistillationDataset.from_jsonl(
        args.train_data,
        tokenizer=tokenizer,
        taxonomy=config.taxonomy,
        max_length=config.student.max_length,
    )
    logger.info(f"Loaded {len(full_dataset)} training samples")

    # Split into train/val if needed
    if args.val_data:
        train_dataset = full_dataset
        val_dataset = DistillationDataset.from_jsonl(
            args.val_data,
            tokenizer=tokenizer,
            taxonomy=config.taxonomy,
            max_length=config.student.max_length,
        )
    else:
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.dataloader_num_workers,
        collate_fn=collate_distillation_batch,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.dataloader_num_workers,
        collate_fn=collate_distillation_batch,
        pin_memory=True,
    )

    # Initialize model
    logger.info(f"Initializing student model: {config.student.model_name}")
    model = StudentModelForDistillation(config.student, config.taxonomy)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Initialize trainer
    trainer = DistillationTrainer(
        model=model,
        config=config.training,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        device=device,
        use_wandb=args.use_wandb,
    )

    # Train
    logger.info("Starting training...")
    results = trainer.train()

    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {results['best_accuracy']:.4f}")
    logger.info(f"Final checkpoint saved to: {config.training.output_dir}/final")


if __name__ == "__main__":
    main()
