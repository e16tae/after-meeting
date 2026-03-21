"""Renderer protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from after_meeting.models import StructuredMeeting


@runtime_checkable
class Renderer(Protocol):
    """Protocol for document renderers."""

    def render(self, meeting: StructuredMeeting, output_path: Path) -> Path:
        """Render structured meeting data to a document file."""
        ...
