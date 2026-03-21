"""Pipeline orchestrator for the `process` convenience command."""

from __future__ import annotations

import logging
from pathlib import Path

from after_meeting.config import get_settings

logger = logging.getLogger(__name__)


def run_pipeline(
    *,
    audio_path: Path,
    doc_type: str,
    fmt: str = "docx",
    output_dir: Path | None = None,
    title: str | None = None,
    date: str | None = None,
    language: str | None = None,
    with_appendix: bool = False,
    stt_provider: str | None = None,
    llm_provider: str | None = None,
    chunk_minutes: int | None = None,
) -> dict:
    """Run the full transcribe → diarize → structure → render pipeline.

    Returns a JSON-serializable result dict.
    """
    from after_meeting.stt import get_provider as get_stt
    from after_meeting.structuring.analyzer import analyze
    from after_meeting.rendering import get_renderer

    settings = get_settings()
    out = Path(output_dir) if output_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Transcribe
    stt = get_stt(stt_provider or settings.stt_provider, settings=settings)
    transcript = stt.transcribe(audio_path, language=language)

    transcript_path = out / f"{audio_path.stem}_transcript.json"
    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    # Step 2: Diarize if speakers not already assigned
    if len(transcript.speakers) <= 1:
        try:
            from after_meeting.speaker.diarizer import diarize_transcript
            transcript = diarize_transcript(transcript, audio_path)
            transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
        except ImportError:
            logger.warning("Speaker diarization not available (speaker extras not installed)")
        except Exception as exc:
            logger.warning("Speaker diarization failed: %s, continuing without", exc)

    # Step 3: Structure
    structured = analyze(
        transcript,
        doc_type=doc_type,
        title=title,
        date=date,
        llm_provider=llm_provider or settings.llm_provider,
        settings=settings,
        max_utterances=settings.max_utterances,
    )

    structured_path = out / f"{audio_path.stem}_structured.json"
    structured_path.write_text(structured.model_dump_json(indent=2), encoding="utf-8")

    # Step 4: Render
    if not with_appendix:
        structured.full_transcript = None

    doc_path = out / f"{audio_path.stem}.{fmt}"
    renderer = get_renderer(fmt)
    result_path = renderer.render(structured, doc_path)

    return {
        "status": "success",
        "step": "process",
        "output_file": str(result_path.resolve()),
        "intermediate_files": {
            "transcript": str(transcript_path.resolve()),
            "structured": str(structured_path.resolve()),
        },
        "metadata": {
            "doc_type": doc_type,
            "format": fmt,
            "title": structured.title,
            "speakers_detected": len(transcript.speakers),
            "language": transcript.language,
        },
    }
