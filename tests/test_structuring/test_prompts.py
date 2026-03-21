"""Tests for prompt building including context prefix."""
from __future__ import annotations

from after_meeting.models import Transcript, Utterance
from after_meeting.structuring.prompts import (
    build_chunked_minutes_prompt,
    build_chunked_report_prompt,
)


def _make_transcript(n_utterances: int = 5, start_offset: float = 0.0) -> Transcript:
    utts = []
    t = start_offset
    for i in range(n_utterances):
        utts.append(Utterance(
            speaker=f"Speaker {(i % 2) + 1}",
            start_time=t,
            end_time=t + 10.0,
            text=f"발언 {i + 1}",
        ))
        t += 12.0
    return Transcript(language="ko", speakers=["Speaker 1", "Speaker 2"], utterances=utts)


class TestBuildChunkedMinutesPrompt:
    def test_first_chunk_has_no_context(self) -> None:
        transcript = _make_transcript(5)
        prompt = build_chunked_minutes_prompt(
            transcript, title=None, date="2026-03-18",
            chunk_index=0, total_chunks=4, context_prefix=None,
        )
        assert "참고 컨텍스트" not in prompt
        assert "발언 1" in prompt

    def test_subsequent_chunk_has_context_prefix(self) -> None:
        context = _make_transcript(3)
        main = _make_transcript(5, start_offset=180.0)
        prompt = build_chunked_minutes_prompt(
            main, title=None, date="2026-03-18",
            chunk_index=1, total_chunks=4, context_prefix=context,
        )
        assert "참고 컨텍스트" in prompt
        assert "분석 대상" in prompt

    def test_chunk_metadata_in_prompt(self) -> None:
        transcript = _make_transcript(3)
        prompt = build_chunked_minutes_prompt(
            transcript, title="회의", date="2026-03-18",
            chunk_index=2, total_chunks=4, context_prefix=None,
        )
        assert "3/4" in prompt or "3 / 4" in prompt


class TestBuildChunkedReportPrompt:
    def test_first_chunk_has_no_context(self) -> None:
        transcript = _make_transcript(5)
        prompt = build_chunked_report_prompt(
            transcript, title=None, date="2026-03-18",
            chunk_index=0, total_chunks=4, context_prefix=None,
        )
        assert "참고 컨텍스트" not in prompt
        assert "report" in prompt.lower() or "보고서" in prompt

    def test_subsequent_chunk_has_context_prefix(self) -> None:
        context = _make_transcript(3)
        main = _make_transcript(5, start_offset=180.0)
        prompt = build_chunked_report_prompt(
            main, title=None, date="2026-03-18",
            chunk_index=1, total_chunks=4, context_prefix=context,
        )
        assert "참고 컨텍스트" in prompt

    def test_report_requests_executive_summary(self) -> None:
        transcript = _make_transcript(3)
        prompt = build_chunked_report_prompt(
            transcript, title="보고서", date="2026-03-18",
            chunk_index=0, total_chunks=2, context_prefix=None,
        )
        assert "executive_summary" in prompt
