"""
Intent prediction with confidence-based decision gating.

The predictor:
1. Runs the student model on input emails
2. Applies temperature scaling for calibrated confidence
3. Applies decision gating based on confidence thresholds
4. Produces standardized decision payloads
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from ..data.preprocessing import Email, EmailPreprocessor
from ..config import InferenceConfig, IntentTaxonomy
from ..models.calibration import TemperatureScaler


class DecisionStatus(str, Enum):
    """Status of the classification decision."""
    ACCEPTED = "accepted"  # High confidence, prediction accepted
    ABSTAINED = "abstained"  # Low confidence, no prediction
    FALLBACK = "fallback"  # Routed to fallback mechanism


@dataclass
class DecisionPayload:
    """
    Standardized decision payload for downstream consumption.

    This payload is designed to be consumed deterministically by
    downstream automation and tool-calling systems.
    """
    # Core prediction
    intent_label: Optional[str]
    confidence: float

    # Decision metadata
    status: DecisionStatus
    abstained: bool
    reason: Optional[str] = None

    # Model metadata
    model_version: str = "student-v1"
    taxonomy_version: str = "v1"

    # Recommended action based on prediction
    recommended_action: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class IntentPredictor:
    """
    Production-ready intent predictor with confidence gating.

    Implements the model execution path from the system design:
    1. Input processing (preprocessing)
    2. Student model inference
    3. Confidence-based decision gating
    4. Decision payload emission
    """

    # Mapping of intents to recommended actions
    DEFAULT_ACTIONS = {
        "inquiry": "route_to_support",
        "complaint": "escalate_priority",
        "request": "create_ticket",
        "confirmation": "send_acknowledgment",
        "cancellation": "process_cancellation",
        "feedback": "log_feedback",
        "escalation": "route_to_supervisor",
        "follow_up": "check_previous_ticket",
        "out_of_scope": "route_to_general",
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: InferenceConfig,
        taxonomy: IntentTaxonomy,
        preprocessor: Optional[EmailPreprocessor] = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.taxonomy = taxonomy
        self.preprocessor = preprocessor or EmailPreprocessor()
        self.temperature_scaler = temperature_scaler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Get temperature from scaler or config
        self.temperature = (
            temperature_scaler.temperature.item()
            if temperature_scaler is not None
            else config.temperature
        )

    def predict(self, email: Email) -> DecisionPayload:
        """
        Make a single prediction with confidence gating.

        Args:
            email: The email to classify

        Returns:
            DecisionPayload with prediction and metadata
        """
        # Preprocess email
        processed = self.preprocessor.preprocess(email)
        text = processed.to_text()

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits

            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            confidence, prediction = probs.max(dim=-1)
            confidence = confidence.item()
            prediction = prediction.item()

        # Decision gating
        intent_label = self.taxonomy.id_to_label(prediction)

        if confidence >= self.config.acceptance_threshold:
            # High confidence - accept prediction
            status = DecisionStatus.ACCEPTED
            abstained = False
            reason = None
            recommended_action = self.DEFAULT_ACTIONS.get(intent_label)
        elif confidence < self.config.abstention_threshold:
            # Very low confidence - abstain
            status = DecisionStatus.ABSTAINED
            abstained = True
            intent_label = None
            reason = f"Confidence {confidence:.3f} below abstention threshold {self.config.abstention_threshold}"
            recommended_action = "route_to_human_review"
        else:
            # Medium confidence - might need fallback
            if self.config.enable_teacher_fallback and confidence < self.config.fallback_threshold:
                status = DecisionStatus.FALLBACK
                abstained = False
                reason = f"Confidence {confidence:.3f} below fallback threshold {self.config.fallback_threshold}"
                recommended_action = "invoke_teacher_model"
            else:
                status = DecisionStatus.ACCEPTED
                abstained = False
                reason = None
                recommended_action = self.DEFAULT_ACTIONS.get(intent_label)

        return DecisionPayload(
            intent_label=intent_label,
            confidence=confidence,
            status=status,
            abstained=abstained,
            reason=reason,
            recommended_action=recommended_action,
        )

    def predict_batch(self, emails: list[Email]) -> list[DecisionPayload]:
        """
        Make predictions for a batch of emails.

        Args:
            emails: List of emails to classify

        Returns:
            List of DecisionPayloads
        """
        # Preprocess all emails
        processed = [self.preprocessor.preprocess(email) for email in emails]
        texts = [p.to_text() for p in processed]

        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Batch inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits

            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            confidences, predictions = probs.max(dim=-1)
            confidences = confidences.cpu().tolist()
            predictions = predictions.cpu().tolist()

        # Create payloads with decision gating
        payloads = []
        for confidence, prediction in zip(confidences, predictions):
            intent_label = self.taxonomy.id_to_label(prediction)

            if confidence >= self.config.acceptance_threshold:
                status = DecisionStatus.ACCEPTED
                abstained = False
                reason = None
                recommended_action = self.DEFAULT_ACTIONS.get(intent_label)
            elif confidence < self.config.abstention_threshold:
                status = DecisionStatus.ABSTAINED
                abstained = True
                intent_label = None
                reason = f"Confidence {confidence:.3f} below threshold"
                recommended_action = "route_to_human_review"
            else:
                if self.config.enable_teacher_fallback and confidence < self.config.fallback_threshold:
                    status = DecisionStatus.FALLBACK
                    abstained = False
                    reason = f"Needs teacher fallback"
                    recommended_action = "invoke_teacher_model"
                else:
                    status = DecisionStatus.ACCEPTED
                    abstained = False
                    reason = None
                    recommended_action = self.DEFAULT_ACTIONS.get(intent_label)

            payloads.append(DecisionPayload(
                intent_label=intent_label,
                confidence=confidence,
                status=status,
                abstained=abstained,
                reason=reason,
                recommended_action=recommended_action,
            ))

        return payloads

    def get_statistics(self, payloads: list[DecisionPayload]) -> dict:
        """
        Compute statistics over a batch of predictions.

        Args:
            payloads: List of decision payloads

        Returns:
            Dictionary with prediction statistics
        """
        total = len(payloads)
        if total == 0:
            return {}

        accepted = sum(1 for p in payloads if p.status == DecisionStatus.ACCEPTED)
        abstained = sum(1 for p in payloads if p.status == DecisionStatus.ABSTAINED)
        fallback = sum(1 for p in payloads if p.status == DecisionStatus.FALLBACK)

        avg_confidence = sum(p.confidence for p in payloads) / total

        # Intent distribution
        intent_counts = {}
        for p in payloads:
            if p.intent_label:
                intent_counts[p.intent_label] = intent_counts.get(p.intent_label, 0) + 1

        return {
            "total_predictions": total,
            "accepted_rate": accepted / total,
            "abstention_rate": abstained / total,
            "fallback_rate": fallback / total,
            "average_confidence": avg_confidence,
            "intent_distribution": intent_counts,
        }
