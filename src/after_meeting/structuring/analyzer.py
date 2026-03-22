"""Structuring analyser -- turns a Transcript into a StructuredMeeting via LLM."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from after_meeting.errors import LLMError
from after_meeting.llm import get_provider
from after_meeting.models import StructuredMeeting, Transcript
from after_meeting.structuring.merger import merge_structured_meetings
from after_meeting.structuring.prompts import (
    build_chunked_minutes_prompt,
    build_chunked_report_prompt,
    build_minutes_prompt,
    build_report_prompt,
)

if TYPE_CHECKING:
    from after_meeting.config import Settings

_PROMPT_BUILDERS = {
    "minutes": build_minutes_prompt,
    "report": build_report_prompt,
}

_CHUNKED_PROMPT_BUILDERS = {
    "minutes": build_chunked_minutes_prompt,
    "report": build_chunked_report_prompt,
}

_CONTEXT_WINDOW_UTTERANCES = 15

# JSON Schema for Responses API structured output (strict mode)
_STRUCTURED_MEETING_SCHEMA = {
    "name": "structured_meeting",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "date": {"type": "string"},
            "doc_type": {"type": "string"},
            "agenda_discussions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "summary": {"type": "string"},
                        "speaker_contributions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string"},
                                    "contribution": {"type": "string"},
                                },
                                "required": ["speaker", "contribution"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["topic", "summary", "speaker_contributions"],
                    "additionalProperties": False,
                },
            },
            "decisions": {"type": "array", "items": {"type": "string"}},
            "action_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "assignee": {"type": ["string", "null"]},
                        "description": {"type": "string"},
                        "deadline": {"type": ["string", "null"]},
                    },
                    "required": ["assignee", "description", "deadline"],
                    "additionalProperties": False,
                },
            },
            "executive_summary": {"type": ["string", "null"]},
        },
        "required": [
            "title", "date", "doc_type", "agenda_discussions",
            "decisions", "action_items", "executive_summary",
        ],
        "additionalProperties": False,
    },
}


def _chunk_transcript(transcript: Transcript, max_utterances: int) -> list[Transcript]:
    """Split transcript into chunks of roughly max_utterances each."""
    chunks = []
    for i in range(0, len(transcript.utterances), max_utterances):
        chunk_utts = transcript.utterances[i : i + max_utterances]
        chunk_speakers = sorted({u.speaker for u in chunk_utts})
        chunks.append(Transcript(
            language=transcript.language,
            speakers=chunk_speakers,
            utterances=chunk_utts,
            metadata=transcript.metadata,
        ))
    return chunks


def _get_context_prefix(chunks: list[Transcript], chunk_index: int) -> Transcript | None:
    """Extract the last N utterances from the previous chunk as context prefix."""
    if chunk_index == 0:
        return None
    prev_utts = chunks[chunk_index - 1].utterances
    context_utts = prev_utts[-_CONTEXT_WINDOW_UTTERANCES:]
    return Transcript(
        language=chunks[chunk_index].language,
        speakers=sorted({u.speaker for u in context_utts}),
        utterances=context_utts,
        metadata={},
    )


def _analyze_chunked(
    transcript: Transcript,
    *,
    doc_type: str,
    title: str | None,
    date: str,
    provider,
    max_utterances: int,
) -> StructuredMeeting:
    """Analyze a long transcript using chunked map-reduce."""
    chunks = _chunk_transcript(transcript, max_utterances)
    build_prompt = _CHUNKED_PROMPT_BUILDERS[doc_type]

    partial_results: list[StructuredMeeting] = []
    for i, chunk in enumerate(chunks):
        context = _get_context_prefix(chunks, i)
        prompt = build_prompt(
            chunk,
            title=title,
            date=date,
            chunk_index=i,
            total_chunks=len(chunks),
            context_prefix=context,
        )
        raw_response = provider.complete(
            prompt, json_schema=_STRUCTURED_MEETING_SCHEMA,
        )
        json_text = _try_extract_json(raw_response)
        try:
            partial = StructuredMeeting.model_validate_json(json_text)
        except Exception as exc:
            raise LLMError(
                f"Failed to validate LLM JSON for chunk {i}: {exc}",
                code="LLM_PARSE",
                recoverable=True,
            ) from exc
        partial_results.append(partial)

    return merge_structured_meetings(partial_results)


def _try_extract_json(text: str) -> str:
    """Extract JSON from LLM response.

    With structured output enabled, the response is already pure JSON.
    Falls back to stripping markdown code fences for non-schema responses.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped

    # Fallback: strip markdown code fences
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, stripped, re.DOTALL)
    if match:
        return match.group(1).strip()

    raise LLMError(
        "LLM response does not contain valid JSON.",
        code="LLM_PARSE",
        recoverable=True,
    )


def analyze(
    transcript: Transcript,
    *,
    doc_type: str,
    title: str | None = None,
    date: str | None = None,
    llm_provider: str = "codex",
    settings: Settings | None = None,
    max_utterances: int = 200,
) -> StructuredMeeting:
    """Analyse a transcript and return a structured meeting document.

    Parameters
    ----------
    transcript:
        The transcribed meeting.
    doc_type:
        Either ``"minutes"`` or ``"report"``.
    title:
        Optional meeting title; if *None* the LLM infers one.
    date:
        Meeting date string (``YYYY-MM-DD``).  Defaults to today.
    llm_provider:
        Name of the registered LLM provider to use.
    settings:
        Application settings forwarded to the LLM provider.
    max_utterances:
        When the transcript has more utterances than this threshold,
        a chunked map-reduce analysis path is used.  Defaults to 200.
    """
    if doc_type not in _PROMPT_BUILDERS:
        raise LLMError(
            f"Unsupported doc_type: {doc_type!r}. Choose from {list(_PROMPT_BUILDERS)}.",
            code="LLM_CONFIG",
            recoverable=False,
        )

    if date is None:
        date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Get LLM provider
    provider_kwargs: dict = {}
    if settings is not None:
        provider_kwargs["settings"] = settings
    provider = get_provider(llm_provider, **provider_kwargs)

    if len(transcript.utterances) > max_utterances:
        structured = _analyze_chunked(
            transcript,
            doc_type=doc_type,
            title=title,
            date=date,
            provider=provider,
            max_utterances=max_utterances,
        )
    else:
        build_prompt = _PROMPT_BUILDERS[doc_type]
        prompt = build_prompt(transcript, title, date)
        raw_response = provider.complete(
            prompt, json_schema=_STRUCTURED_MEETING_SCHEMA,
        )
        json_text = _try_extract_json(raw_response)
        try:
            structured = StructuredMeeting.model_validate_json(json_text)
        except Exception as exc:
            raise LLMError(
                f"Failed to validate LLM JSON against StructuredMeeting schema: {exc}",
                code="LLM_PARSE",
                recoverable=True,
            ) from exc

    # Attach the full transcript for optional appendix rendering
    structured.full_transcript = transcript

    return structured
