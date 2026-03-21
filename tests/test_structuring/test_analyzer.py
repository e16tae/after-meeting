"""Tests for the structuring analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from after_meeting.config import Settings
from after_meeting.models import StructuredMeeting, Transcript, Utterance
from after_meeting.structuring.analyzer import analyze

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def sample_transcript() -> Transcript:
    """Load the sample transcript fixture."""
    raw = (FIXTURES_DIR / "sample_transcript.json").read_text(encoding="utf-8")
    return Transcript.model_validate_json(raw)


@pytest.fixture()
def minutes_json_response() -> str:
    """A valid StructuredMeeting JSON string for doc_type='minutes'."""
    return json.dumps(
        {
            "title": "신규 프로젝트 킥오프 회의",
            "date": "2026-03-17",
            "doc_type": "minutes",
            "agenda_discussions": [
                {
                    "topic": "신규 프로젝트 일정 수립",
                    "summary": "4월 1일 착수, 5월 말 1차 릴리스로 일정 확정",
                    "speaker_contributions": [
                        {
                            "speaker": "Speaker 1",
                            "contribution": "일정 제안 및 주간 회의 일정 확정",
                        },
                        {
                            "speaker": "Speaker 2",
                            "contribution": "일정 동의 및 기술 스택 검토 문서 작성 약속",
                        },
                    ],
                },
                {
                    "topic": "역할 분담",
                    "summary": "백엔드와 프론트엔드 역할 분담 논의",
                    "speaker_contributions": [
                        {
                            "speaker": "Speaker 2",
                            "contribution": "백엔드 담당 자원, 프론트엔드는 김 대리 추천",
                        },
                        {
                            "speaker": "Speaker 1",
                            "contribution": "역할 분담 승인",
                        },
                    ],
                },
            ],
            "decisions": [
                "프로젝트 시작일: 4월 1일",
                "1차 릴리스: 5월 말",
                "매주 수요일 진행 상황 공유 회의",
            ],
            "action_items": [
                {
                    "assignee": "Speaker 2",
                    "description": "기술 스택 검토 문서 작성 및 공유",
                    "deadline": "이번 주 금요일",
                }
            ],
            "executive_summary": None,
        },
        ensure_ascii=False,
    )


@pytest.fixture()
def report_json_response() -> str:
    """A valid StructuredMeeting JSON string for doc_type='report'."""
    return json.dumps(
        {
            "title": "신규 프로젝트 킥오프 보고서",
            "date": "2026-03-17",
            "doc_type": "report",
            "agenda_discussions": [
                {
                    "topic": "프로젝트 계획 및 역할 분담",
                    "summary": "신규 프로젝트의 일정과 담당 역할을 확정하였으며, 4월 초 착수 후 5월 말 1차 릴리스를 목표로 한다.",
                    "speaker_contributions": [
                        {
                            "speaker": "Speaker 1",
                            "contribution": "전체 일정 수립 및 주간 회의 체계 제안",
                        },
                        {
                            "speaker": "Speaker 2",
                            "contribution": "백엔드 개발 담당 및 기술 스택 검토 주도",
                        },
                    ],
                }
            ],
            "decisions": [
                "4월 1일 프로젝트 착수",
                "5월 말 1차 릴리스 목표",
                "매주 수요일 정기 회의 실시",
            ],
            "action_items": [
                {
                    "assignee": "Speaker 2",
                    "description": "기술 스택 검토 문서 작성 및 공유",
                    "deadline": "이번 주 금요일",
                }
            ],
            "executive_summary": "신규 프로젝트 킥오프 회의에서 일정 및 역할 분담이 확정되었다. 4월 1일 착수하여 5월 말 1차 릴리스를 목표로 하며, 주간 진행 상황 공유 체계를 마련하였다.",
        },
        ensure_ascii=False,
    )


def _make_mock_provider(response: str) -> MagicMock:
    """Create a mock LLM provider that returns the given response."""
    provider = MagicMock()
    provider.complete.return_value = response
    return provider


class TestAnalyzeMinutes:
    """Tests for analyze() with doc_type='minutes'."""

    def test_returns_structured_meeting(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(minutes_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date="2026-03-17",
            )

        assert isinstance(result, StructuredMeeting)
        assert result.doc_type == "minutes"
        assert result.title == "신규 프로젝트 킥오프 회의"
        assert result.date == "2026-03-17"

    def test_agenda_discussions(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(minutes_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date="2026-03-17",
            )

        assert len(result.agenda_discussions) == 2
        assert result.agenda_discussions[0].topic == "신규 프로젝트 일정 수립"

    def test_decisions_and_action_items(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(minutes_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date="2026-03-17",
            )

        assert len(result.decisions) == 3
        assert len(result.action_items) == 1
        assert result.action_items[0].assignee == "Speaker 2"

    def test_executive_summary_is_none(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(minutes_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date="2026-03-17",
            )

        assert result.executive_summary is None

    def test_full_transcript_attached(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(minutes_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date="2026-03-17",
            )

        assert result.full_transcript is not None
        assert result.full_transcript.language == "ko"
        assert len(result.full_transcript.utterances) == 4

    def test_handles_markdown_code_block(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        """LLM responses wrapped in ```json ... ``` should be parsed correctly."""
        wrapped = f"```json\n{minutes_json_response}\n```"
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(wrapped),
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date="2026-03-17",
            )

        assert isinstance(result, StructuredMeeting)
        assert result.doc_type == "minutes"


class TestAnalyzeReport:
    """Tests for analyze() with doc_type='report'."""

    def test_returns_structured_meeting(
        self, sample_transcript: Transcript, report_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(report_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="report",
                date="2026-03-17",
            )

        assert isinstance(result, StructuredMeeting)
        assert result.doc_type == "report"
        assert result.title == "신규 프로젝트 킥오프 보고서"

    def test_executive_summary_present(
        self, sample_transcript: Transcript, report_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(report_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="report",
                date="2026-03-17",
            )

        assert result.executive_summary is not None
        assert "킥오프" in result.executive_summary

    def test_action_items(
        self, sample_transcript: Transcript, report_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(report_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="report",
                date="2026-03-17",
            )

        assert len(result.action_items) == 1
        assert result.action_items[0].description == "기술 스택 검토 문서 작성 및 공유"

    def test_full_transcript_attached(
        self, sample_transcript: Transcript, report_json_response: str
    ) -> None:
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(report_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="report",
                date="2026-03-17",
            )

        assert result.full_transcript is not None
        assert len(result.full_transcript.speakers) == 2


class TestAnalyzeEdgeCases:
    """Tests for error handling and edge cases."""

    def test_invalid_doc_type(self, sample_transcript: Transcript) -> None:
        from after_meeting.errors import LLMError

        with pytest.raises(LLMError, match="Unsupported doc_type"):
            analyze(sample_transcript, doc_type="invalid")

    def test_default_date_used_when_none(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        """When date=None, analyze() should use today's date and not raise."""
        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider(minutes_json_response),
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date=None,
            )

        assert isinstance(result, StructuredMeeting)

    def test_invalid_json_raises_llm_error(
        self, sample_transcript: Transcript
    ) -> None:
        from after_meeting.errors import LLMError

        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=_make_mock_provider("this is not json at all"),
        ):
            with pytest.raises(LLMError, match="does not contain valid JSON"):
                analyze(
                    sample_transcript,
                    doc_type="minutes",
                    date="2026-03-17",
                )


class TestAnalyzeChunked:
    """Tests for analyze() with long transcripts that require chunking."""

    def _make_long_transcript(self, n_utterances: int = 200) -> Transcript:
        utts = []
        t = 0.0
        for i in range(n_utterances):
            utts.append(Utterance(
                speaker=f"Speaker {(i % 3) + 1}",
                start_time=t,
                end_time=t + 30.0,
                text=f"발언 내용 {i + 1}",
            ))
            t += 35.0
        return Transcript(
            language="ko",
            speakers=["Speaker 1", "Speaker 2", "Speaker 3"],
            utterances=utts,
            metadata={"duration": t},
        )

    def test_chunked_path_triggers_for_long_transcript(
        self, minutes_json_response: str
    ) -> None:
        transcript = self._make_long_transcript(200)
        mock_provider = _make_mock_provider(minutes_json_response)

        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=mock_provider,
        ):
            result = analyze(
                transcript,
                doc_type="minutes",
                date="2026-03-18",
                max_utterances=50,
            )

        assert isinstance(result, StructuredMeeting)
        assert mock_provider.complete.call_count > 1

    def test_short_transcript_uses_single_path(
        self, sample_transcript: Transcript, minutes_json_response: str
    ) -> None:
        mock_provider = _make_mock_provider(minutes_json_response)

        with patch(
            "after_meeting.structuring.analyzer.get_provider",
            return_value=mock_provider,
        ):
            result = analyze(
                sample_transcript,
                doc_type="minutes",
                date="2026-03-18",
                max_utterances=50,
            )

        assert isinstance(result, StructuredMeeting)
        assert mock_provider.complete.call_count == 1
