#!/usr/bin/env python3
"""
Script to generate training data using the teacher model.

This script:
1. Loads raw emails from a source file
2. Uses the teacher model (Gemini) to generate labels and rationales
3. Saves the distillation dataset for student training
"""

import argparse
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, TeacherModelConfig, IntentTaxonomy
from src.data.preprocessing import Email, EmailPreprocessor
from src.data.dataset import DistillationSample
from src.data.teacher_generation import TeacherDataGenerator, MockTeacherGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_emails_from_jsonl(path: str) -> list[Email]:
    """Load emails from a JSONL file."""
    emails = []
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
            emails.append(email)
    return emails


def save_distillation_data(samples: list[DistillationSample], path: str):
    """Save distillation samples to JSONL file."""
    with open(path, "w") as f:
        for sample in samples:
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


def create_sample_emails() -> list[Email]:
    """Create sample emails for demonstration."""
    return [
        Email(
            subject="Question about my order",
            body="Hi, I placed an order last week but haven't received any shipping confirmation. Could you please check the status? Thanks!",
            sender="customer@example.com",
        ),
        Email(
            subject="Problem with service",
            body="I've been experiencing issues with my account for the past three days. The system keeps logging me out and I can't access my dashboard. This is unacceptable!",
            sender="user@company.com",
        ),
        Email(
            subject="Request for refund",
            body="Please process a refund for order #12345. The product arrived damaged and I would like my money back.",
            sender="buyer@mail.com",
        ),
        Email(
            subject="RE: Meeting confirmation",
            body="Thank you for confirming the meeting for tomorrow at 2pm. I'll see you then.",
            sender="colleague@work.com",
        ),
        Email(
            subject="Cancel my subscription",
            body="I would like to cancel my monthly subscription effective immediately. Please confirm cancellation.",
            sender="subscriber@email.com",
        ),
        Email(
            subject="Feedback on new feature",
            body="I really like the new dashboard design! However, I think the navigation could be improved. Consider adding a search bar at the top.",
            sender="poweruser@domain.com",
        ),
        Email(
            subject="URGENT: Need supervisor",
            body="I've been waiting for over an hour and no one has helped me. I need to speak with a manager immediately about this issue.",
            sender="frustrated@customer.com",
        ),
        Email(
            subject="Following up on ticket #789",
            body="Hi, I'm following up on the support ticket I opened last week. What's the current status? Any updates?",
            sender="patient@user.com",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate training data using teacher model")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input JSONL file with emails",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/distillation_train.jsonl",
        help="Path to output distillation dataset",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock teacher for testing without API",
    )
    parser.add_argument(
        "--estimate-confidence",
        action="store_true",
        default=True,
        help="Estimate teacher confidence via consistency",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Set API key if provided
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        config.teacher.api_key = api_key

    # Load or create emails
    if args.input:
        logger.info(f"Loading emails from {args.input}")
        emails = load_emails_from_jsonl(args.input)
    else:
        logger.info("Using sample emails for demonstration")
        emails = create_sample_emails()

    if args.max_samples:
        emails = emails[:args.max_samples]

    logger.info(f"Processing {len(emails)} emails")

    # Initialize teacher generator
    if args.use_mock:
        logger.info("Using mock teacher generator")
        generator = MockTeacherGenerator(config.taxonomy)
    else:
        logger.info(f"Using teacher model: {config.teacher.model_name}")
        generator = TeacherDataGenerator(config.teacher, config.taxonomy)

    # Generate distillation data
    samples = []
    for i, email in enumerate(tqdm(emails, desc="Generating labels")):
        try:
            sample = generator.generate_supervision(
                email,
                estimate_confidence=args.estimate_confidence,
            )
            samples.append(sample)
            logger.debug(f"Sample {i + 1}: {sample.intent_label} (conf: {sample.teacher_confidence})")
        except Exception as e:
            logger.error(f"Failed to process email {i + 1}: {e}")
            continue

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Save results
    save_distillation_data(samples, args.output)
    logger.info(f"Saved {len(samples)} samples to {args.output}")

    # Print summary
    intent_counts = {}
    for sample in samples:
        intent_counts[sample.intent_label] = intent_counts.get(sample.intent_label, 0) + 1

    logger.info("Intent distribution:")
    for intent, count in sorted(intent_counts.items()):
        logger.info(f"  {intent}: {count}")


if __name__ == "__main__":
    main()
