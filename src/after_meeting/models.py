"""Core Pydantic data models for the after-meeting pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Utterance(BaseModel):
    """A single speaker utterance with timestamps."""

    speaker: str = Field(description="Speaker identifier (e.g. 'Speaker 1')")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text content")


class Transcript(BaseModel):
    """Full transcript with speaker diarization."""

    language: str = Field(description="Detected or specified language code (e.g. 'ko')")
    speakers: list[str] = Field(description="List of unique speaker identifiers")
    utterances: list[Utterance] = Field(default_factory=list)
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (duration, audio_file, created_at, etc.)",
    )


class ActionItem(BaseModel):
    """An action item extracted from the meeting."""

    assignee: str | None = Field(
        default=None, description="Speaker ID (e.g. 'Speaker 1')"
    )
    description: str
    deadline: str | None = Field(
        default=None, description="Deadline if mentioned"
    )


class AgendaDiscussion(BaseModel):
    """Discussion summary for a single agenda topic."""

    topic: str
    summary: str
    speaker_contributions: list[dict] = Field(
        default_factory=list,
        description="Per-speaker contribution summaries",
    )


class StructuredMeeting(BaseModel):
    """Structured meeting data ready for rendering."""

    title: str = Field(description="Meeting title (inferred by LLM)")
    date: str = Field(description="Meeting date (auto or override)")
    doc_type: str = Field(description="'minutes' or 'report'")
    agenda_discussions: list[AgendaDiscussion] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    full_transcript: Transcript | None = Field(
        default=None, description="Full transcript for appendix (optional)"
    )
    executive_summary: str | None = Field(
        default=None, description="Executive summary (report only)"
    )


class SpeakerMapping(BaseModel):
    """Speaker ID mapping from one chunk to the reference (chunk 0)."""

    chunk_index: int = Field(description="Index of the chunk this mapping applies to")
    mapping: dict[str, str] = Field(
        description="Map from this chunk's speaker ID to unified speaker ID"
    )
    confidence: float | None = Field(
        default=None, description="Confidence score 0.0-1.0"
    )

    def is_reliable(self, threshold: float = 0.6) -> bool:
        """Whether this mapping meets the confidence threshold."""
        if self.confidence is None:
            return False
        return self.confidence >= threshold


class ChunkInfo(BaseModel):
    """Metadata for a single audio chunk."""

    index: int
    start_time: float = Field(description="Start time in seconds relative to original audio")
    end_time: float = Field(description="End time in seconds relative to original audio")
    audio_path: str = Field(description="Path to the chunk audio file")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
