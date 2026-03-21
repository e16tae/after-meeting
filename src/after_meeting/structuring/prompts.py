"""Prompt templates for meeting structuring."""

from __future__ import annotations

from after_meeting.models import Transcript


def _format_transcript(transcript: Transcript) -> str:
    """Format transcript utterances into a readable text block."""
    lines: list[str] = []
    for utt in transcript.utterances:
        start = _fmt_time(utt.start_time)
        end = _fmt_time(utt.end_time)
        lines.append(f"[{start}-{end}] {utt.speaker}: {utt.text}")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """Format seconds into MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _json_schema_description() -> str:
    """Return a human-readable JSON schema description for StructuredMeeting."""
    return """\
{
  "title": "string (회의 제목)",
  "date": "string (회의 날짜, YYYY-MM-DD 형식)",
  "doc_type": "string ('minutes' 또는 'report')",
  "agenda_discussions": [
    {
      "topic": "string (안건 주제)",
      "summary": "string (논의 요약)",
      "speaker_contributions": [
        {
          "speaker": "string (발언자 ID, 예: 'Speaker 1')",
          "contribution": "string (해당 발언자의 기여 내용 요약)"
        }
      ]
    }
  ],
  "decisions": ["string (결정 사항)"],
  "action_items": [
    {
      "assignee": "string | null (담당자 Speaker ID)",
      "description": "string (실행 항목 설명)",
      "deadline": "string | null (기한)"
    }
  ],
  "executive_summary": "string | null (보고서 형식에서만 사용)"
}"""


def build_minutes_prompt(
    transcript: Transcript,
    title: str | None,
    date: str,
) -> str:
    """Build a prompt that instructs the LLM to produce meeting minutes JSON."""
    formatted = _format_transcript(transcript)
    title_instruction = (
        f'회의 제목은 "{title}"입니다.'
        if title
        else "회의 제목은 대화 내용을 바탕으로 적절하게 추론하여 작성하세요."
    )

    return f"""\
당신은 회의록 작성 전문가입니다. 아래의 회의 녹취록을 분석하여 구조화된 회의록을 JSON 형식으로 작성하세요.

## 지침
- {title_instruction}
- 회의 날짜: {date}
- 문서 유형: "minutes" (회의록)
- 각 발언자별 기여 내용을 구체적으로 정리하세요.
- 논의된 안건(agenda)을 주제별로 분류하고, 각 안건에 대한 요약과 발언자별 기여를 작성하세요.
- 회의에서 내려진 결정 사항(decisions)을 명확히 나열하세요.
- 실행 항목(action items)이 있다면 담당자, 설명, 기한을 포함하여 정리하세요.
- "executive_summary" 필드는 null로 설정하세요 (회의록 형식에서는 사용하지 않습니다).

## 출력 형식
반드시 아래 JSON 스키마에 맞는 **유효한 JSON만** 출력하세요. 다른 텍스트는 포함하지 마세요.

```json
{_json_schema_description()}
```

## 회의 녹취록
{formatted}

## JSON 출력:"""


def build_report_prompt(
    transcript: Transcript,
    title: str | None,
    date: str,
) -> str:
    """Build a prompt that instructs the LLM to produce a meeting report JSON."""
    formatted = _format_transcript(transcript)
    title_instruction = (
        f'회의 제목은 "{title}"입니다.'
        if title
        else "회의 제목은 대화 내용을 바탕으로 적절하게 추론하여 작성하세요."
    )

    return f"""\
당신은 회의 보고서 작성 전문가입니다. 아래의 회의 녹취록을 분석하여 구조화된 회의 보고서를 JSON 형식으로 작성하세요.

## 지침
- {title_instruction}
- 회의 날짜: {date}
- 문서 유형: "report" (보고서)
- 경영진을 위한 간결한 요약(executive_summary)을 반드시 작성하세요.
- 핵심 논의 사항을 주제별로 정리하고, 각 주제에 대한 심층 분석을 포함하세요.
- 후속 조치 계획(action items)을 구체적으로 작성하세요.
- 주요 결정 사항(decisions)을 명확히 나열하세요.
- 발언자별 기여는 보고서 맥락에 맞게 요약하세요.

## 출력 형식
반드시 아래 JSON 스키마에 맞는 **유효한 JSON만** 출력하세요. 다른 텍스트는 포함하지 마세요.

```json
{_json_schema_description()}
```

## 회의 녹취록
{formatted}

## JSON 출력:"""


def build_chunked_minutes_prompt(
    transcript: Transcript,
    title: str | None,
    date: str,
    chunk_index: int,
    total_chunks: int,
    context_prefix: Transcript | None,
) -> str:
    """Build a chunked minutes prompt with optional context prefix from prior chunk."""
    formatted = _format_transcript(transcript)
    title_instruction = (
        f'회의 제목은 "{title}"입니다.'
        if title
        else "회의 제목은 대화 내용을 바탕으로 적절하게 추론하여 작성하세요."
    )

    sections: list[str] = []

    sections.append(
        "당신은 회의록 작성 전문가입니다. "
        "아래의 회의 녹취록을 분석하여 구조화된 회의록을 JSON 형식으로 작성하세요."
    )

    # Context prefix section (only for non-first chunks)
    if context_prefix is not None:
        context_formatted = _format_transcript(context_prefix)
        sections.append(
            f"## 참고 컨텍스트 (이전 구간)\n"
            f"맥락 파악을 위한 참고 자료이며, "
            f"이 내용에서 안건/결정사항/실행항목을 추출하지 마세요.\n\n"
            f"{context_formatted}"
        )

    # Instructions
    sections.append(
        f"## 지침\n"
        f"- {title_instruction}\n"
        f"- 회의 날짜: {date}\n"
        f"- 문서 유형: \"minutes\" (회의록)\n"
        f"- 이 녹취록은 전체 회의의 일부입니다 (구간 {chunk_index + 1}/{total_chunks}).\n"
        f"- 각 발언자별 기여 내용을 구체적으로 정리하세요.\n"
        f"- 논의된 안건(agenda)을 주제별로 분류하고, 각 안건에 대한 요약과 발언자별 기여를 작성하세요.\n"
        f"- 회의에서 내려진 결정 사항(decisions)을 명확히 나열하세요.\n"
        f"- 실행 항목(action items)이 있다면 담당자, 설명, 기한을 포함하여 정리하세요.\n"
        f'- "executive_summary" 필드는 null로 설정하세요 (회의록 형식에서는 사용하지 않습니다).'
    )

    # Output format
    sections.append(
        f"## 출력 형식\n"
        f"반드시 아래 JSON 스키마에 맞는 **유효한 JSON만** 출력하세요. 다른 텍스트는 포함하지 마세요.\n\n"
        f"```json\n{_json_schema_description()}\n```"
    )

    # Main transcript
    sections.append(
        f"## 분석 대상 녹취록 (구간 {chunk_index + 1}/{total_chunks})\n"
        f"{formatted}"
    )

    sections.append("## JSON 출력:")

    return "\n\n".join(sections)


def build_chunked_report_prompt(
    transcript: Transcript,
    title: str | None,
    date: str,
    chunk_index: int,
    total_chunks: int,
    context_prefix: Transcript | None,
) -> str:
    """Build a chunked report prompt with optional context prefix from prior chunk."""
    formatted = _format_transcript(transcript)
    title_instruction = (
        f'회의 제목은 "{title}"입니다.'
        if title
        else "회의 제목은 대화 내용을 바탕으로 적절하게 추론하여 작성하세요."
    )

    sections: list[str] = []

    sections.append(
        "당신은 회의 보고서 작성 전문가입니다. "
        "아래의 회의 녹취록을 분석하여 구조화된 회의 보고서를 JSON 형식으로 작성하세요."
    )

    # Context prefix section (only for non-first chunks)
    if context_prefix is not None:
        context_formatted = _format_transcript(context_prefix)
        sections.append(
            f"## 참고 컨텍스트 (이전 구간)\n"
            f"맥락 파악을 위한 참고 자료이며, "
            f"이 내용에서 안건/결정사항/실행항목을 추출하지 마세요.\n\n"
            f"{context_formatted}"
        )

    # Instructions
    sections.append(
        f"## 지침\n"
        f"- {title_instruction}\n"
        f"- 회의 날짜: {date}\n"
        f"- 문서 유형: \"report\" (보고서)\n"
        f"- 이 녹취록은 전체 회의의 일부입니다 (구간 {chunk_index + 1}/{total_chunks}).\n"
        f"- 경영진을 위한 간결한 요약(executive_summary)을 반드시 작성하세요.\n"
        f"- 핵심 논의 사항을 주제별로 정리하고, 각 주제에 대한 심층 분석을 포함하세요.\n"
        f"- 후속 조치 계획(action items)을 구체적으로 작성하세요.\n"
        f"- 주요 결정 사항(decisions)을 명확히 나열하세요.\n"
        f"- 발언자별 기여는 보고서 맥락에 맞게 요약하세요."
    )

    # Output format
    sections.append(
        f"## 출력 형식\n"
        f"반드시 아래 JSON 스키마에 맞는 **유효한 JSON만** 출력하세요. 다른 텍스트는 포함하지 마세요.\n\n"
        f"```json\n{_json_schema_description()}\n```"
    )

    # Main transcript
    sections.append(
        f"## 분석 대상 녹취록 (구간 {chunk_index + 1}/{total_chunks})\n"
        f"{formatted}"
    )

    sections.append("## JSON 출력:")

    return "\n\n".join(sections)
