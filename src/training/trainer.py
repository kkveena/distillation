"""
Training loop for step-by-step distillation.

Implements the complete training pipeline including:
- Multi-task loss optimization
- Learning rate scheduling
- Gradient accumulation
- Evaluation and checkpointing
- Logging to wandb/tensorboard
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from tqdm import tqdm

from .losses import DistillationLoss
from ..config import TrainingConfig
from ..models.calibration import compute_ece

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    epoch: int
    step: int
    total_loss: float
    label_loss: float
    rationale_loss: Optional[float] = None
    calibration_loss: Optional[float] = None
    learning_rate: float = 0.0
    accuracy: Optional[float] = None
    ece: Optional[float] = None


class DistillationTrainer:
    """
    Trainer for step-by-step distillation.

    Handles the complete training loop with:
    - Multi-task objective optimization
    - Gradient accumulation for effective batch size
    - Learning rate scheduling
    - Evaluation and early stopping
    - Model checkpointing
    - Metric logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Initialize loss function
        self.loss_fn = DistillationLoss(
            label_weight=config.label_loss_weight,
            rationale_weight=config.rationale_loss_weight,
            calibration_weight=config.calibration_loss_weight,
        )

        # Initialize optimizer
        self.optimizer = optimizer or self._create_optimizer()

        # Initialize scheduler
        total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = scheduler or self._create_scheduler(total_steps)

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.best_eval_accuracy = 0.0

        # Logging
        self.use_wandb = use_wandb
        if use_wandb:
            self._init_wandb()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters with and without weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)

    def _create_scheduler(self, total_steps: int):
        """Create learning rate scheduler."""
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        return OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy="cos",
        )

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            wandb.init(
                project="email-intent-distillation",
                config=vars(self.config),
            )
        except ImportError:
            logger.warning("wandb not installed, disabling logging")
            self.use_wandb = False

    def train(self) -> dict:
        """
        Run the complete training loop.

        Returns:
            Dictionary with final training metrics
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total training samples: {len(self.train_dataloader.dataset)}")

        # Enable mixed precision if configured
        scaler = torch.amp.GradScaler("cuda") if self.config.fp16 else None

        for epoch in range(self.config.num_epochs):
            epoch_metrics = self._train_epoch(epoch, scaler)

            # Evaluate if eval dataloader provided
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(
                    f"Epoch {epoch + 1} - Eval Loss: {eval_metrics['loss']:.4f}, "
                    f"Accuracy: {eval_metrics['accuracy']:.4f}, "
                    f"ECE: {eval_metrics['ece']:.4f}"
                )

                # Save best model
                if eval_metrics["accuracy"] > self.best_eval_accuracy:
                    self.best_eval_accuracy = eval_metrics["accuracy"]
                    self._save_checkpoint("best")

            # Save checkpoint
            if (epoch + 1) % 1 == 0:  # Save every epoch
                self._save_checkpoint(f"epoch_{epoch + 1}")

        # Save final model
        self._save_checkpoint("final")

        return {
            "best_accuracy": self.best_eval_accuracy,
            "final_step": self.global_step,
        }

    def _train_epoch(
        self,
        epoch: int,
        scaler: Optional[torch.amp.GradScaler],
    ) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_label_loss = 0.0
        total_rationale_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with optional mixed precision
            if self.config.fp16:
                with torch.amp.autocast("cuda"):
                    losses = self._forward_step(batch)
            else:
                losses = self._forward_step(batch)

            loss = losses["total_loss"] / self.config.gradient_accumulation_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                if scaler is not None:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_metrics(losses, epoch)

            # Track losses
            total_loss += losses["total_loss"].item()
            total_label_loss += losses["label_loss"].item()
            if "rationale_loss" in losses:
                total_rationale_loss += losses["rationale_loss"].item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                "loss": losses["total_loss"].item(),
                "lr": self.scheduler.get_last_lr()[0],
            })

        return {
            "loss": total_loss / num_batches,
            "label_loss": total_label_loss / num_batches,
            "rationale_loss": total_rationale_loss / num_batches if total_rationale_loss > 0 else 0,
        }

    def _forward_step(self, batch: dict) -> dict:
        """Execute forward pass and compute losses."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            rationale_ids=batch.get("rationale_ids"),
            rationale_attention_mask=batch.get("rationale_attention_mask"),
        )

        # Compute losses
        losses = self.loss_fn(
            logits=outputs.logits,
            labels=batch["labels"],
            rationale_logits=outputs.rationale_logits,
            rationale_ids=batch.get("rationale_ids"),
            rationale_mask=batch.get("rationale_attention_mask"),
        )

        return losses

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on validation set."""
        self.model.eval()

        all_logits = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            all_logits.append(outputs.logits.cpu())
            all_labels.append(batch["labels"].cpu())
            total_loss += outputs.loss.item()
            num_batches += 1

        # Compute metrics
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        probs = torch.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)

        accuracy = (predictions == labels).float().mean().item()

        ece = compute_ece(
            confidences.numpy(),
            predictions.numpy(),
            labels.numpy(),
        )

        self.model.train()

        return {
            "loss": total_loss / num_batches,
            "accuracy": accuracy,
            "ece": ece,
        }

    def _log_metrics(self, losses: dict, epoch: int):
        """Log metrics to console and wandb."""
        metrics = {
            "train/total_loss": losses["total_loss"].item(),
            "train/label_loss": losses["label_loss"].item(),
            "train/learning_rate": self.scheduler.get_last_lr()[0],
            "train/epoch": epoch,
            "train/step": self.global_step,
        }

        if "rationale_loss" in losses:
            metrics["train/rationale_loss"] = losses["rationale_loss"].item()
        if "calibration_loss" in losses:
            metrics["train/calibration_loss"] = losses["calibration_loss"].item()

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=self.global_step)

        logger.info(
            f"Step {self.global_step} - Loss: {losses['total_loss'].item():.4f}, "
            f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
        )

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)

        # Save optimizer and scheduler state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_eval_accuracy": self.best_eval_accuracy,
            },
            os.path.join(checkpoint_dir, "trainer_state.pt"),
        )

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load from checkpoint."""
        # Load model
        state_dict = torch.load(
            os.path.join(checkpoint_dir, "model.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)

        # Load trainer state
        trainer_state = torch.load(
            os.path.join(checkpoint_dir, "trainer_state.pt"),
            map_location=self.device,
        )
        self.optimizer.load_state_dict(trainer_state["optimizer"])
        self.scheduler.load_state_dict(trainer_state["scheduler"])
        self.global_step = trainer_state["global_step"]
        self.best_eval_accuracy = trainer_state["best_eval_accuracy"]

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")
