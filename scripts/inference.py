#!/usr/bin/env python3
"""
Inference script for production email classification.

This script:
1. Loads a trained and calibrated model
2. Processes emails with confidence-based decision gating
3. Optionally uses cascaded inference with teacher fallback
4. Outputs standardized decision payloads
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, InferenceConfig
from src.data.preprocessing import Email, EmailPreprocessor
from src.models.student_model import StudentModelForDistillation
from src.models.calibration import TemperatureScaler
from src.inference.predictor import IntentPredictor
from src.inference.cascaded_inference import CascadedClassifier, BatchInferencePipeline
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


def main():
    parser = argparse.ArgumentParser(description="Run inference on emails")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input JSONL file with emails",
    )
    parser.add_argument(
        "--email-subject",
        type=str,
        help="Email subject for single inference",
    )
    parser.add_argument(
        "--email-body",
        type=str,
        help="Email body for single inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.jsonl",
        help="Path to output predictions file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-2b",
        help="Base model name for tokenizer",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for calibration",
    )
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for accepting predictions",
    )
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold below which to abstain",
    )
    parser.add_argument(
        "--enable-fallback",
        action="store_true",
        help="Enable teacher model fallback for low-confidence predictions",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for teacher fallback",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--use-mock-teacher",
        action="store_true",
        help="Use mock teacher for fallback (no API calls)",
    )
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.student.model_name = args.model_name

    # Update inference config
    inference_config = InferenceConfig(
        acceptance_threshold=args.acceptance_threshold,
        abstention_threshold=args.abstention_threshold,
        enable_teacher_fallback=args.enable_fallback,
        fallback_threshold=args.fallback_threshold,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

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

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = StudentModelForDistillation.from_pretrained(
        args.checkpoint,
        config.student,
        config.taxonomy,
    )
    model.to(device)
    model.eval()

    # Create temperature scaler
    temperature_scaler = TemperatureScaler(args.temperature)

    # Create predictor
    predictor = IntentPredictor(
        model=model,
        tokenizer=tokenizer,
        config=inference_config,
        taxonomy=config.taxonomy,
        temperature_scaler=temperature_scaler,
        device=device,
    )

    # Setup teacher fallback if enabled
    if args.enable_fallback:
        if args.use_mock_teacher:
            teacher = MockTeacherGenerator(config.taxonomy)
        else:
            teacher = TeacherDataGenerator(config.teacher, config.taxonomy)

        classifier = CascadedClassifier(
            student_predictor=predictor,
            teacher_generator=teacher,
            fallback_threshold=args.fallback_threshold,
            enable_teacher_fallback=True,
        )
    else:
        classifier = None

    # Process emails
    if args.email_subject or args.email_body:
        # Single email inference
        email = Email(
            subject=args.email_subject or "",
            body=args.email_body or "",
        )

        if classifier:
            result = classifier.predict(email)
        else:
            result = predictor.predict(email)

        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULT")
        print("=" * 60)
        print(result.to_json())
        print("=" * 60)

    elif args.input:
        # Batch inference from file
        logger.info(f"Loading emails from {args.input}")
        emails = load_emails_from_jsonl(args.input)
        logger.info(f"Processing {len(emails)} emails")

        if classifier:
            pipeline = BatchInferencePipeline(
                classifier=classifier,
                batch_size=args.batch_size,
            )
            results = pipeline.process(
                emails,
                progress_callback=lambda done, total: logger.info(f"Processed {done}/{total}"),
            )
            stats = classifier.get_statistics()
        else:
            results = predictor.predict_batch(emails)
            stats = predictor.get_statistics(results)

        # Save results
        with open(args.output, "w") as f:
            for result in results:
                f.write(result.to_json().replace("\n", " ") + "\n")

        logger.info(f"Saved predictions to {args.output}")

        # Print statistics
        print("\n" + "=" * 60)
        print("INFERENCE STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 60)

    else:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        print("Enter email subject and body separated by '|||'\n")

        while True:
            try:
                user_input = input("Email (subject|||body): ").strip()
                if user_input.lower() == "quit":
                    break

                if "|||" in user_input:
                    subject, body = user_input.split("|||", 1)
                else:
                    subject = ""
                    body = user_input

                email = Email(subject=subject.strip(), body=body.strip())

                if classifier:
                    result = classifier.predict(email)
                else:
                    result = predictor.predict(email)

                print("\nResult:")
                print(result.to_json())
                print()

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
