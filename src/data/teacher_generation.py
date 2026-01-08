"""
Teacher model interface for generating training supervision.

Uses Gemini to generate:
- Intent labels aligned with taxonomy
- Step-by-step rationales
- Confidence estimates (via consistency)
"""

import json
import logging
from typing import Optional
from collections import Counter

from .preprocessing import Email
from .dataset import DistillationSample
from ..config import TeacherModelConfig, IntentTaxonomy

logger = logging.getLogger(__name__)


class TeacherDataGenerator:
    """
    Generates training supervision using a teacher model (Gemini).

    For each email, produces:
    - Intent label from the taxonomy
    - Step-by-step rationale explaining the classification
    - Confidence estimate based on prediction consistency
    """

    SYSTEM_PROMPT = """You are an expert email intent classifier. Your task is to:
1. Analyze the email content carefully
2. Classify the email into one of the provided intent categories
3. Provide a step-by-step rationale explaining your classification

Be precise, objective, and focus on the key signals in the email that indicate the intent."""

    CLASSIFICATION_TEMPLATE = """Classify the following email into one of these intent categories:
{intent_list}

Email:
{email_text}

Respond in JSON format with the following structure:
{{
    "intent": "<one of the intent categories above>",
    "rationale": "<step-by-step explanation of why this intent was chosen>",
    "key_signals": ["<signal 1>", "<signal 2>", ...]
}}

Important:
- The "intent" must be exactly one of the provided categories
- The "rationale" should explain your reasoning step by step
- List the key phrases or signals from the email that led to your classification"""

    def __init__(
        self,
        config: TeacherModelConfig,
        taxonomy: IntentTaxonomy,
    ):
        self.config = config
        self.taxonomy = taxonomy
        self._model = None

    def _init_model(self):
        """Initialize the Gemini model."""
        if self._model is not None:
            return

        try:
            import google.generativeai as genai

            if self.config.api_key:
                genai.configure(api_key=self.config.api_key)

            self._model = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                },
                system_instruction=self.SYSTEM_PROMPT,
            )
        except ImportError:
            raise ImportError(
                "google-generativeai package required. Install with: pip install google-generativeai"
            )

    def generate_supervision(
        self,
        email: Email,
        estimate_confidence: bool = True,
    ) -> DistillationSample:
        """
        Generate teacher supervision for a single email.

        Args:
            email: The email to classify
            estimate_confidence: Whether to estimate confidence via multiple samples

        Returns:
            DistillationSample with label, rationale, and optional confidence
        """
        self._init_model()

        intent_list = "\n".join(f"- {label}" for label in self.taxonomy.labels)
        prompt = self.CLASSIFICATION_TEMPLATE.format(
            intent_list=intent_list,
            email_text=email.to_text(),
        )

        if estimate_confidence:
            # Generate multiple samples to estimate confidence
            predictions = []
            rationales = []

            for _ in range(self.config.num_consistency_samples):
                result = self._generate_single(prompt)
                if result:
                    predictions.append(result["intent"])
                    rationales.append(result["rationale"])

            if not predictions:
                raise ValueError("Teacher model failed to generate valid predictions")

            # Use majority vote for final prediction
            intent_counts = Counter(predictions)
            final_intent = intent_counts.most_common(1)[0][0]

            # Confidence = fraction of consistent predictions
            confidence = intent_counts[final_intent] / len(predictions)

            # Use rationale from a consistent prediction
            final_rationale = rationales[predictions.index(final_intent)]
        else:
            # Single generation without confidence estimation
            result = self._generate_single(prompt)
            if not result:
                raise ValueError("Teacher model failed to generate valid prediction")

            final_intent = result["intent"]
            final_rationale = result["rationale"]
            confidence = None

        return DistillationSample(
            email=email,
            intent_label=final_intent,
            rationale=final_rationale,
            teacher_confidence=confidence,
        )

    def _generate_single(self, prompt: str) -> Optional[dict]:
        """Generate a single prediction from the teacher model."""
        try:
            response = self._model.generate_content(prompt)
            text = response.text.strip()

            # Extract JSON from response
            # Handle case where response might have markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text)

            # Validate intent is in taxonomy
            if result["intent"] not in self.taxonomy.labels:
                logger.warning(
                    f"Teacher predicted invalid intent: {result['intent']}"
                )
                # Try to find closest match
                result["intent"] = self._find_closest_intent(result["intent"])

            return result
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse teacher response: {e}")
            return None

    def _find_closest_intent(self, predicted: str) -> str:
        """Find the closest valid intent to the predicted one."""
        predicted_lower = predicted.lower().strip()
        for label in self.taxonomy.labels:
            if label.lower() in predicted_lower or predicted_lower in label.lower():
                return label
        # Default to first label if no match
        return self.taxonomy.labels[0]

    def generate_batch(
        self,
        emails: list[Email],
        estimate_confidence: bool = True,
        skip_errors: bool = True,
    ) -> list[DistillationSample]:
        """
        Generate teacher supervision for a batch of emails.

        Args:
            emails: List of emails to classify
            estimate_confidence: Whether to estimate confidence
            skip_errors: Whether to skip emails that fail processing

        Returns:
            List of DistillationSamples
        """
        samples = []
        for i, email in enumerate(emails):
            try:
                sample = self.generate_supervision(email, estimate_confidence)
                samples.append(sample)
                logger.info(f"Generated supervision for email {i + 1}/{len(emails)}")
            except Exception as e:
                logger.error(f"Failed to process email {i + 1}: {e}")
                if not skip_errors:
                    raise
        return samples


class MockTeacherGenerator(TeacherDataGenerator):
    """
    Mock teacher generator for testing without API calls.

    Generates deterministic labels and rationales based on simple rules.
    """

    def __init__(self, taxonomy: IntentTaxonomy):
        self.taxonomy = taxonomy
        self._model = None  # Not used in mock

    def _init_model(self):
        pass  # No model initialization needed

    def _generate_single(self, prompt: str) -> Optional[dict]:
        """Generate mock prediction based on simple keyword matching."""
        prompt_lower = prompt.lower()

        # Simple keyword-based classification for testing
        if any(word in prompt_lower for word in ["help", "question", "how", "what"]):
            intent = "inquiry"
            rationale = "The email contains question words indicating an inquiry."
        elif any(word in prompt_lower for word in ["problem", "issue", "broken", "not working"]):
            intent = "complaint"
            rationale = "The email mentions problems or issues, indicating a complaint."
        elif any(word in prompt_lower for word in ["please", "could you", "would you", "need"]):
            intent = "request"
            rationale = "The email contains request phrases asking for action."
        elif any(word in prompt_lower for word in ["confirm", "verification", "receipt"]):
            intent = "confirmation"
            rationale = "The email seeks or provides confirmation."
        elif any(word in prompt_lower for word in ["cancel", "stop", "unsubscribe"]):
            intent = "cancellation"
            rationale = "The email indicates intent to cancel or stop something."
        elif any(word in prompt_lower for word in ["feedback", "suggestion", "improve"]):
            intent = "feedback"
            rationale = "The email provides feedback or suggestions."
        elif any(word in prompt_lower for word in ["urgent", "escalate", "manager", "supervisor"]):
            intent = "escalation"
            rationale = "The email indicates urgency or desire for escalation."
        elif any(word in prompt_lower for word in ["following up", "status", "update on"]):
            intent = "follow_up"
            rationale = "The email is following up on a previous communication."
        else:
            intent = "out_of_scope"
            rationale = "The email does not match any specific intent category."

        return {
            "intent": intent,
            "rationale": f"Step 1: Analyzed the email content. Step 2: Identified key signals. Step 3: {rationale}",
            "key_signals": ["identified keywords", "context analysis"],
        }
