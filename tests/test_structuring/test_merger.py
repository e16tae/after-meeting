"""Tests for merging partial StructuredMeeting results."""
from __future__ import annotations

import pytest

from after_meeting.models import ActionItem, AgendaDiscussion, StructuredMeeting
from after_meeting.structuring.merger import merge_structured_meetings


class TestMergeStructuredMeetings:
    def _make_partial(self, *, topics: list[str], decisions: list[str], doc_type: str = "minutes") -> StructuredMeeting:
        return StructuredMeeting(
            title="회의",
            date="2026-03-18",
            doc_type=doc_type,
            agenda_discussions=[
                AgendaDiscussion(topic=t, summary=f"{t} 요약", speaker_contributions=[])
                for t in topics
            ],
            decisions=decisions,
            action_items=[],
        )

    def test_merges_agenda_discussions(self) -> None:
        parts = [
            self._make_partial(topics=["일정"], decisions=[]),
            self._make_partial(topics=["예산"], decisions=[]),
        ]
        result = merge_structured_meetings(parts)
        assert len(result.agenda_discussions) == 2

    def test_merges_decisions_without_duplicates(self) -> None:
        parts = [
            self._make_partial(topics=[], decisions=["결정A", "결정B"]),
            self._make_partial(topics=[], decisions=["결정B", "결정C"]),
        ]
        result = merge_structured_meetings(parts)
        assert len(result.decisions) == 3

    def test_merges_action_items(self) -> None:
        p1 = self._make_partial(topics=[], decisions=[])
        p1.action_items = [ActionItem(assignee="Speaker 1", description="작업1")]
        p2 = self._make_partial(topics=[], decisions=[])
        p2.action_items = [ActionItem(assignee="Speaker 2", description="작업2")]
        result = merge_structured_meetings([p1, p2])
        assert len(result.action_items) == 2

    def test_uses_first_title_and_date(self) -> None:
        parts = [
            self._make_partial(topics=[], decisions=[]),
            self._make_partial(topics=[], decisions=[]),
        ]
        parts[0].title = "첫 번째 제목"
        parts[1].title = "두 번째 제목"
        result = merge_structured_meetings(parts)
        assert result.title == "첫 번째 제목"

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            merge_structured_meetings([])
