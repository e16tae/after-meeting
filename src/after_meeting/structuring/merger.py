"""Merge partial StructuredMeeting results from chunked analysis."""
from __future__ import annotations

from after_meeting.models import StructuredMeeting


def merge_structured_meetings(parts: list[StructuredMeeting]) -> StructuredMeeting:
    """Merge multiple partial StructuredMeeting results into one."""
    if not parts:
        raise ValueError("merge_structured_meetings requires at least one part")

    if len(parts) == 1:
        return parts[0]

    all_agendas = []
    all_decisions: list[str] = []
    seen_decisions: set[str] = set()
    all_actions = []
    summaries: list[str] = []

    for part in parts:
        all_agendas.extend(part.agenda_discussions)

        for d in part.decisions:
            if d not in seen_decisions:
                all_decisions.append(d)
                seen_decisions.add(d)

        all_actions.extend(part.action_items)

        if part.executive_summary:
            summaries.append(part.executive_summary)

    return StructuredMeeting(
        title=parts[0].title,
        date=parts[0].date,
        doc_type=parts[0].doc_type,
        agenda_discussions=all_agendas,
        decisions=all_decisions,
        action_items=all_actions,
        executive_summary="\n\n".join(summaries) if summaries else None,
    )
