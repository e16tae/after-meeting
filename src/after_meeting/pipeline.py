"""Pipeline orchestrator for the `process` convenience command."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from after_meeting.config import get_settings
from after_meeting.models import Transcript, Utterance

logger = logging.getLogger(__name__)

_CHUNK_THRESHOLD_MINUTES = 30  # split audio longer than this


def _merge_chunk_transcripts(
    chunks_info: list,
    transcripts: list[Transcript],
) -> Transcript:
    """Merge per-chunk transcripts into one, adjusting timestamps and deduplicating overlap."""
    all_utterances: list[Utterance] = []

    for chunk, transcript in zip(chunks_info, transcripts):
        offset = chunk.start_time

        for utt in transcript.utterances:
            adjusted = Utterance(
                speaker=utt.speaker,
                start_time=utt.start_time + offset,
                end_time=utt.end_time + offset,
                text=utt.text,
            )

            # Skip duplicates in overlap region
            if all_utterances:
                last = all_utterances[-1]
                if (
                    adjusted.start_time <= last.end_time
                    and adjusted.text == last.text
                ):
                    continue

            all_utterances.append(adjusted)

    # Sort by time to ensure correct order
    all_utterances.sort(key=lambda u: u.start_time)

    lang = transcripts[0].language if transcripts else "auto"
    speakers = sorted({u.speaker for u in all_utterances})
    duration = all_utterances[-1].end_time if all_utterances else 0.0

    return Transcript(
        language=lang,
        speakers=speakers,
        utterances=all_utterances,
        metadata={
            "audio_file": str(chunks_info[0].audio_path) if chunks_info else "",
            "duration": duration,
            "chunks": len(transcripts),
        },
    )


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
    context: str | None = None,
    chunk_minutes: int | None = None,
) -> dict:
    """Run the full transcribe → diarize → structure → render pipeline.

    Long audio is automatically split into chunks for STT to avoid GPU OOM.
    Speaker diarization (pyannote) runs on the full audio for global consistency.
    """
    from after_meeting.audio.splitter import get_duration, split_audio
    from after_meeting.stt import get_provider as get_stt
    from after_meeting.structuring.analyzer import analyze
    from after_meeting.rendering import get_renderer

    settings = get_settings()
    if context is not None:
        settings = settings.model_copy(update={"stt_context": context})
    out = Path(output_dir) if output_dir else Path(".")
    out.mkdir(parents=True, exist_ok=True)

    chunk_min = chunk_minutes or settings.chunk_minutes
    stt = get_stt(stt_provider or settings.stt_provider, settings=settings)

    # Step 1: Transcribe (with auto-split for long audio)
    duration = get_duration(audio_path)
    if duration > _CHUNK_THRESHOLD_MINUTES * 60:
        transcript = _transcribe_chunked(
            audio_path, stt, language, chunk_min, settings.overlap_seconds,
        )
    else:
        transcript = stt.transcribe(audio_path, language=language)

    transcript_path = out / f"{audio_path.stem}_transcript.json"
    transcript_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    # Step 2: Diarize on full audio (pyannote, ~1.6GB VRAM)
    if len(transcript.speakers) <= 1:
        try:
            from after_meeting.speaker.diarizer import diarize_transcript
            transcript = diarize_transcript(
                transcript, audio_path, settings=settings,
            )
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
            "audio_duration_seconds": duration,
        },
    }


def _transcribe_chunked(
    audio_path: Path,
    stt,
    language: str | None,
    chunk_minutes: int,
    overlap_seconds: int,
) -> Transcript:
    """Split long audio into chunks, transcribe each, and merge."""
    from after_meeting.audio.splitter import split_audio

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        chunks = split_audio(
            audio_path,
            output_dir=tmp,
            chunk_minutes=chunk_minutes,
            overlap_seconds=overlap_seconds,
        )
        logger.info(
            "Audio split into %d chunks (%d min each, %ds overlap)",
            len(chunks), chunk_minutes, overlap_seconds,
        )

        transcripts: list[Transcript] = []
        for i, chunk in enumerate(chunks):
            logger.info(
                "Transcribing chunk %d/%d [%.0f-%.0fs]",
                i + 1, len(chunks), chunk.start_time, chunk.end_time,
            )
            t = stt.transcribe(Path(chunk.audio_path), language=language)
            transcripts.append(t)

    return _merge_chunk_transcripts(chunks, transcripts)
