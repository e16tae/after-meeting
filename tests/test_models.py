"""Tests for Pydantic data models — serialization/deserialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

from after_meeting.models import (
    ActionItem,
    AgendaDiscussion,
    ChunkInfo,
    SpeakerMapping,
    StructuredMeeting,
    Transcript,
    Utterance,
)


class TestUtterance:
    def test_roundtrip(self) -> None:
        u = Utterance(speaker="Speaker 1", start_time=0.0, end_time=5.5, text="안녕하세요")
        data = json.loads(u.model_dump_json())
        restored = Utterance(**data)
        assert restored == u

    def test_required_fields(self) -> None:
        with pytest.raises(Exception):
            Utterance(speaker="Speaker 1")  # type: ignore[call-arg]


class TestTranscript:
    def test_roundtrip(self) -> None:
        t = Transcript(
            language="ko",
            speakers=["Speaker 1", "Speaker 2"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0.0, end_time=3.0, text="첫 번째 발언"),
                Utterance(speaker="Speaker 2", start_time=3.5, end_time=7.0, text="두 번째 발언"),
            ],
            metadata={"duration": 7.0, "audio_file": "meeting.mp3"},
        )
        json_str = t.model_dump_json()
        restored = Transcript.model_validate_json(json_str)
        assert restored == t
        assert len(restored.utterances) == 2

    def test_default_metadata(self) -> None:
        t = Transcript(language="en", speakers=[])
        assert t.metadata == {}
        assert t.utterances == []

    def test_from_fixture(self) -> None:
        fixture_path = FIXTURES_DIR / "sample_transcript.json"
        with open(fixture_path, encoding="utf-8") as f:
            data = json.load(f)
        t = Transcript(**data)
        assert t.language == "ko"
        assert len(t.speakers) >= 1


class TestActionItem:
    def test_optional_fields(self) -> None:
        a = ActionItem(description="리뷰 작성")
        assert a.assignee is None
        assert a.deadline is None

    def test_full(self) -> None:
        a = ActionItem(assignee="Speaker 1", description="리뷰 작성", deadline="2026-03-20")
        data = json.loads(a.model_dump_json())
        assert data["assignee"] == "Speaker 1"


class TestStructuredMeeting:
    def test_roundtrip_minutes(self) -> None:
        m = StructuredMeeting(
            title="팀 미팅",
            date="2026-03-17",
            doc_type="minutes",
            agenda_discussions=[
                AgendaDiscussion(topic="일정 검토", summary="일정 확인", speaker_contributions=[]),
            ],
            decisions=["일정 확정"],
            action_items=[ActionItem(description="문서 작성")],
        )
        json_str = m.model_dump_json()
        restored = StructuredMeeting.model_validate_json(json_str)
        assert restored.title == "팀 미팅"
        assert restored.doc_type == "minutes"
        assert restored.executive_summary is None
        assert restored.full_transcript is None

    def test_roundtrip_report(self) -> None:
        m = StructuredMeeting(
            title="분기 보고",
            date="2026-03-17",
            doc_type="report",
            executive_summary="핵심 요약입니다.",
            agenda_discussions=[],
            decisions=[],
            action_items=[],
        )
        json_str = m.model_dump_json()
        restored = StructuredMeeting.model_validate_json(json_str)
        assert restored.executive_summary == "핵심 요약입니다."

    def test_from_fixture(self) -> None:
        with open(FIXTURES_DIR / "sample_structured.json", encoding="utf-8") as f:
            data = json.load(f)
        m = StructuredMeeting(**data)
        assert m.doc_type == "minutes"
        assert len(m.agenda_discussions) >= 1
        assert len(m.action_items) >= 1


class TestSpeakerMapping:
    def test_roundtrip(self) -> None:
        m = SpeakerMapping(
            chunk_index=1,
            mapping={"Speaker 1": "Speaker 2", "Speaker 2": "Speaker 1"},
            confidence=0.92,
        )
        data = json.loads(m.model_dump_json())
        restored = SpeakerMapping(**data)
        assert restored == m
        assert restored.confidence == 0.92

    def test_default_confidence(self) -> None:
        m = SpeakerMapping(chunk_index=0, mapping={})
        assert m.confidence is None

    def test_is_reliable_above_threshold(self) -> None:
        m = SpeakerMapping(chunk_index=1, mapping={"S1": "S2"}, confidence=0.85)
        assert m.is_reliable()

    def test_is_not_reliable_below_threshold(self) -> None:
        m = SpeakerMapping(chunk_index=1, mapping={"S1": "S2"}, confidence=0.4)
        assert not m.is_reliable()


class TestChunkInfo:
    def test_roundtrip(self) -> None:
        c = ChunkInfo(
            index=0,
            start_time=0.0,
            end_time=3360.0,
            audio_path="/tmp/chunk_0.mp3",
        )
        data = json.loads(c.model_dump_json())
        restored = ChunkInfo(**data)
        assert restored.index == 0
        assert restored.end_time == 3360.0

    def test_duration(self) -> None:
        c = ChunkInfo(index=0, start_time=0.0, end_time=3360.0, audio_path="/tmp/c.mp3")
        assert c.duration == 3360.0
