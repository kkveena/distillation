"""
Email preprocessing utilities for intent classification.

Handles:
- Thread normalization
- Signature stripping
- Metadata enrichment
- Text normalization
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Email:
    """Structured email representation."""
    subject: str
    body: str
    sender: Optional[str] = None
    recipients: list[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    thread_id: Optional[str] = None
    attachments: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_text(self, include_metadata: bool = True) -> str:
        """Convert email to text representation for model input."""
        parts = []
        if include_metadata and self.sender:
            parts.append(f"From: {self.sender}")
        parts.append(f"Subject: {self.subject}")
        parts.append(f"Body: {self.body}")
        if include_metadata and self.attachments:
            parts.append(f"Attachments: {', '.join(self.attachments)}")
        return "\n".join(parts)


class EmailPreprocessor:
    """Preprocessor for email content normalization."""

    # Common email signature patterns
    SIGNATURE_PATTERNS = [
        r"(?i)^[-_]{2,}\s*$",  # -- or __
        r"(?i)^best regards?,?\s*$",
        r"(?i)^regards?,?\s*$",
        r"(?i)^sincerely,?\s*$",
        r"(?i)^thanks?,?\s*$",
        r"(?i)^thank you,?\s*$",
        r"(?i)^cheers,?\s*$",
        r"(?i)^sent from my (iphone|ipad|android|mobile)",
    ]

    # Forwarded/replied message patterns
    THREAD_PATTERNS = [
        r"(?i)^-+\s*original message\s*-+\s*$",
        r"(?i)^-+\s*forwarded message\s*-+\s*$",
        r"(?i)^on .+ wrote:$",
        r"(?i)^from:.+sent:.+to:.+subject:",
    ]

    def __init__(
        self,
        strip_signatures: bool = True,
        normalize_whitespace: bool = True,
        extract_latest_message: bool = True,
        max_length: Optional[int] = None,
    ):
        self.strip_signatures = strip_signatures
        self.normalize_whitespace = normalize_whitespace
        self.extract_latest_message = extract_latest_message
        self.max_length = max_length

        # Compile patterns
        self.signature_re = [re.compile(p, re.MULTILINE) for p in self.SIGNATURE_PATTERNS]
        self.thread_re = [re.compile(p, re.MULTILINE) for p in self.THREAD_PATTERNS]

    def preprocess(self, email: Email) -> Email:
        """Apply all preprocessing steps to an email."""
        processed_body = email.body

        # Extract latest message from thread if configured
        if self.extract_latest_message:
            processed_body = self._extract_latest_message(processed_body)

        # Strip signatures if configured
        if self.strip_signatures:
            processed_body = self._strip_signature(processed_body)

        # Normalize whitespace
        if self.normalize_whitespace:
            processed_body = self._normalize_whitespace(processed_body)

        # Truncate if max_length specified
        if self.max_length:
            processed_body = processed_body[: self.max_length]

        # Clean subject line
        processed_subject = self._clean_subject(email.subject)

        return Email(
            subject=processed_subject,
            body=processed_body,
            sender=email.sender,
            recipients=email.recipients,
            timestamp=email.timestamp,
            thread_id=email.thread_id,
            attachments=email.attachments,
            metadata=email.metadata,
        )

    def _extract_latest_message(self, text: str) -> str:
        """Extract the latest message from an email thread."""
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Check if this line indicates start of quoted/forwarded content
            is_thread_marker = any(p.match(line) for p in self.thread_re)
            if is_thread_marker:
                break
            result_lines.append(line)

        return "\n".join(result_lines) if result_lines else text

    def _strip_signature(self, text: str) -> str:
        """Remove email signature from text."""
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # Check if this line starts a signature
            is_signature = any(p.match(line) for p in self.signature_re)
            if is_signature:
                break
            result_lines.append(line)

        return "\n".join(result_lines) if result_lines else text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _clean_subject(self, subject: str) -> str:
        """Clean email subject line."""
        # Remove Re:/Fwd: prefixes
        subject = re.sub(r"(?i)^(re|fwd|fw):\s*", "", subject)
        # Remove multiple Re:/Fwd: prefixes
        while re.match(r"(?i)^(re|fwd|fw):\s*", subject):
            subject = re.sub(r"(?i)^(re|fwd|fw):\s*", "", subject)
        return subject.strip()

    def batch_preprocess(self, emails: list[Email]) -> list[Email]:
        """Preprocess a batch of emails."""
        return [self.preprocess(email) for email in emails]
