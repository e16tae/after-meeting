"""LLM provider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used to structure meeting transcripts."""

    def complete(self, prompt: str) -> str:
        """Send a prompt and return the completion text."""
        ...
