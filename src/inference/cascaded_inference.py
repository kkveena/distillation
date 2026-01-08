"""
Cascaded inference strategy combining student and teacher models.

The cascaded approach:
1. Uses the fast student model for most predictions
2. Falls back to the teacher model when confidence is low
3. Balances accuracy, latency, throughput, and cost
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from ..data.preprocessing import Email, EmailPreprocessor
from ..data.teacher_generation import TeacherDataGenerator
from ..config import InferenceConfig, TeacherModelConfig, IntentTaxonomy
from .predictor import IntentPredictor, DecisionPayload, DecisionStatus

logger = logging.getLogger(__name__)


class CascadedClassifier:
    """
    Cascaded email classifier with student-teacher fallback.

    Implements the cascaded inference strategy described in the system design:
    1. Student model processes the email
    2. If confidence < threshold, invoke teacher model
    3. Return prediction with appropriate status

    Benefits:
    - Performance efficiency: Most emails handled by fast student
    - Accuracy preservation: Ambiguous cases use teacher
    - Cost control: Expensive teacher inference is selective
    - Operational safety: Explicit abstention prevents unsafe actions
    """

    def __init__(
        self,
        student_predictor: IntentPredictor,
        teacher_generator: Optional[TeacherDataGenerator] = None,
        fallback_threshold: float = 0.6,
        enable_teacher_fallback: bool = True,
    ):
        self.student = student_predictor
        self.teacher = teacher_generator
        self.fallback_threshold = fallback_threshold
        self.enable_teacher_fallback = enable_teacher_fallback and teacher_generator is not None

        # Statistics tracking
        self._student_calls = 0
        self._teacher_calls = 0
        self._abstentions = 0

    def predict(self, email: Email) -> DecisionPayload:
        """
        Make prediction with cascaded fallback.

        Args:
            email: The email to classify

        Returns:
            DecisionPayload from either student or teacher
        """
        # First, try student model
        student_result = self.student.predict(email)
        self._student_calls += 1

        # If student is confident enough, use its prediction
        if student_result.status == DecisionStatus.ACCEPTED:
            return student_result

        # If abstained (very low confidence), keep abstention
        if student_result.status == DecisionStatus.ABSTAINED:
            self._abstentions += 1
            return student_result

        # If needs fallback and teacher is available
        if student_result.status == DecisionStatus.FALLBACK and self.enable_teacher_fallback:
            return self._teacher_fallback(email, student_result)

        # No teacher available, use student prediction anyway
        return student_result

    def _teacher_fallback(
        self,
        email: Email,
        student_result: DecisionPayload,
    ) -> DecisionPayload:
        """
        Invoke teacher model for fallback prediction.

        Args:
            email: The email to classify
            student_result: Original student prediction

        Returns:
            DecisionPayload from teacher
        """
        try:
            self._teacher_calls += 1

            # Get teacher prediction
            teacher_sample = self.teacher.generate_supervision(
                email,
                estimate_confidence=True,
            )

            # Teacher confidence is heuristic-based
            teacher_confidence = teacher_sample.teacher_confidence or 0.8

            return DecisionPayload(
                intent_label=teacher_sample.intent_label,
                confidence=teacher_confidence,
                status=DecisionStatus.ACCEPTED,
                abstained=False,
                reason="Prediction from teacher model fallback",
                model_version="teacher-gemini",
                recommended_action=self.student.DEFAULT_ACTIONS.get(
                    teacher_sample.intent_label
                ),
            )

        except Exception as e:
            logger.error(f"Teacher fallback failed: {e}")
            # Fall back to student prediction
            return DecisionPayload(
                intent_label=student_result.intent_label,
                confidence=student_result.confidence,
                status=DecisionStatus.ACCEPTED,
                abstained=False,
                reason="Teacher fallback failed, using student prediction",
                recommended_action=student_result.recommended_action,
            )

    def predict_batch(
        self,
        emails: list[Email],
        parallel_teacher: bool = False,
    ) -> list[DecisionPayload]:
        """
        Make predictions for a batch of emails with cascaded fallback.

        Args:
            emails: List of emails to classify
            parallel_teacher: Whether to parallelize teacher calls (if supported)

        Returns:
            List of DecisionPayloads
        """
        # Get student predictions for all emails
        student_results = self.student.predict_batch(emails)
        self._student_calls += len(emails)

        results = []
        fallback_indices = []
        fallback_emails = []

        # Identify which need fallback
        for i, (email, result) in enumerate(zip(emails, student_results)):
            if result.status == DecisionStatus.FALLBACK and self.enable_teacher_fallback:
                fallback_indices.append(i)
                fallback_emails.append(email)
            elif result.status == DecisionStatus.ABSTAINED:
                self._abstentions += 1
            results.append(result)

        # Process fallbacks
        if fallback_emails:
            logger.info(f"Processing {len(fallback_emails)} teacher fallbacks")
            for i, email in zip(fallback_indices, fallback_emails):
                results[i] = self._teacher_fallback(email, results[i])

        return results

    def get_statistics(self) -> dict:
        """
        Get inference statistics.

        Returns:
            Dictionary with call counts and rates
        """
        total = self._student_calls
        if total == 0:
            return {}

        return {
            "total_predictions": total,
            "student_only": total - self._teacher_calls,
            "teacher_fallbacks": self._teacher_calls,
            "abstentions": self._abstentions,
            "fallback_rate": self._teacher_calls / total if total > 0 else 0,
            "abstention_rate": self._abstentions / total if total > 0 else 0,
        }

    def reset_statistics(self):
        """Reset inference statistics."""
        self._student_calls = 0
        self._teacher_calls = 0
        self._abstentions = 0


class BatchInferencePipeline:
    """
    High-throughput batch inference pipeline.

    Optimized for processing large volumes of emails with:
    - Efficient batching
    - Streaming output
    - Progress tracking
    """

    def __init__(
        self,
        classifier: CascadedClassifier,
        batch_size: int = 32,
    ):
        self.classifier = classifier
        self.batch_size = batch_size

    def process(
        self,
        emails: list[Email],
        progress_callback: Optional[callable] = None,
    ) -> list[DecisionPayload]:
        """
        Process a large list of emails in batches.

        Args:
            emails: List of emails to process
            progress_callback: Optional callback(processed, total)

        Returns:
            List of DecisionPayloads
        """
        results = []
        total = len(emails)

        for i in range(0, total, self.batch_size):
            batch = emails[i : i + self.batch_size]
            batch_results = self.classifier.predict_batch(batch)
            results.extend(batch_results)

            if progress_callback:
                progress_callback(len(results), total)

        return results

    def process_stream(
        self,
        emails: list[Email],
    ):
        """
        Process emails and yield results as they complete.

        Args:
            emails: List of emails to process

        Yields:
            DecisionPayload for each email
        """
        for i in range(0, len(emails), self.batch_size):
            batch = emails[i : i + self.batch_size]
            batch_results = self.classifier.predict_batch(batch)
            for result in batch_results:
                yield result
