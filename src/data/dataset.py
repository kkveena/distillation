"""
Dataset classes for email intent classification with distillation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .preprocessing import Email, EmailPreprocessor
from ..config import IntentTaxonomy


@dataclass
class DistillationSample:
    """A single training sample with teacher supervision."""
    email: Email
    intent_label: str
    rationale: str
    teacher_confidence: Optional[float] = None


class EmailIntentDataset(Dataset):
    """Basic dataset for email intent classification."""

    def __init__(
        self,
        emails: list[Email],
        labels: list[str],
        tokenizer: PreTrainedTokenizer,
        taxonomy: IntentTaxonomy,
        max_length: int = 512,
        preprocessor: Optional[EmailPreprocessor] = None,
    ):
        self.emails = emails
        self.labels = labels
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.max_length = max_length
        self.preprocessor = preprocessor or EmailPreprocessor()

    def __len__(self) -> int:
        return len(self.emails)

    def __getitem__(self, idx: int) -> dict:
        email = self.preprocessor.preprocess(self.emails[idx])
        label = self.labels[idx]

        # Tokenize email text
        text = email.to_text()
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.taxonomy.label_to_id(label)),
        }


class DistillationDataset(Dataset):
    """
    Dataset for step-by-step distillation training.

    Each sample contains:
    - Email input (x)
    - Intent label (y) from teacher
    - Rationale (r) from teacher
    - Optional teacher confidence
    """

    def __init__(
        self,
        samples: list[DistillationSample],
        tokenizer: PreTrainedTokenizer,
        taxonomy: IntentTaxonomy,
        max_length: int = 512,
        max_rationale_length: int = 256,
        preprocessor: Optional[EmailPreprocessor] = None,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.taxonomy = taxonomy
        self.max_length = max_length
        self.max_rationale_length = max_rationale_length
        self.preprocessor = preprocessor or EmailPreprocessor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        email = self.preprocessor.preprocess(sample.email)

        # Tokenize email text
        email_text = email.to_text()
        email_encoding = self.tokenizer(
            email_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize rationale for rationale alignment loss
        rationale_encoding = self.tokenizer(
            sample.rationale,
            max_length=self.max_rationale_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": email_encoding["input_ids"].squeeze(0),
            "attention_mask": email_encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.taxonomy.label_to_id(sample.intent_label)),
            "rationale_ids": rationale_encoding["input_ids"].squeeze(0),
            "rationale_attention_mask": rationale_encoding["attention_mask"].squeeze(0),
        }

        # Add teacher confidence if available
        if sample.teacher_confidence is not None:
            result["teacher_confidence"] = torch.tensor(sample.teacher_confidence)

        return result

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        tokenizer: PreTrainedTokenizer,
        taxonomy: IntentTaxonomy,
        **kwargs,
    ) -> "DistillationDataset":
        """Load dataset from JSONL file."""
        samples = []
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                email = Email(
                    subject=data.get("subject", ""),
                    body=data.get("body", ""),
                    sender=data.get("sender"),
                    recipients=data.get("recipients", []),
                    attachments=data.get("attachments", []),
                    metadata=data.get("metadata", {}),
                )
                sample = DistillationSample(
                    email=email,
                    intent_label=data["intent_label"],
                    rationale=data["rationale"],
                    teacher_confidence=data.get("teacher_confidence"),
                )
                samples.append(sample)
        return cls(samples, tokenizer, taxonomy, **kwargs)

    def to_jsonl(self, path: str) -> None:
        """Save dataset to JSONL file."""
        with open(path, "w") as f:
            for sample in self.samples:
                data = {
                    "subject": sample.email.subject,
                    "body": sample.email.body,
                    "sender": sample.email.sender,
                    "recipients": sample.email.recipients,
                    "attachments": sample.email.attachments,
                    "metadata": sample.email.metadata,
                    "intent_label": sample.intent_label,
                    "rationale": sample.rationale,
                    "teacher_confidence": sample.teacher_confidence,
                }
                f.write(json.dumps(data) + "\n")


def collate_distillation_batch(batch: list[dict]) -> dict:
    """Custom collate function for distillation dataset."""
    result = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "rationale_ids": torch.stack([item["rationale_ids"] for item in batch]),
        "rationale_attention_mask": torch.stack(
            [item["rationale_attention_mask"] for item in batch]
        ),
    }

    # Add teacher confidence if present
    if "teacher_confidence" in batch[0]:
        result["teacher_confidence"] = torch.stack(
            [item["teacher_confidence"] for item in batch]
        )

    return result
