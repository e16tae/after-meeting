"""CLI interface using Click — subcommands: transcribe, structure, render, process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from after_meeting.errors import AfterMeetingError


def _output_json(data: dict) -> None:
    """Write JSON to stdout."""
    click.echo(json.dumps(data, ensure_ascii=False, indent=2))


def _output_error(step: str, err: AfterMeetingError) -> None:
    """Write error JSON to stderr and exit."""
    click.echo(
        json.dumps({"status": "error", "step": step, "error": err.to_dict()}, ensure_ascii=False, indent=2),
        err=True,
    )
    sys.exit(err.exit_code)


@click.group()
@click.version_option(package_name="after-meeting")
def cli() -> None:
    """After Meeting — AI-powered meeting audio to minutes/report generator."""


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--language", "-l", default=None, help="Language code (e.g. 'ko'). Auto-detected if omitted.")
@click.option("--stt-provider", default=None, help="STT provider name.")
@click.option("--context", "-c", default=None, help="Contextual hints for ASR (topic, names, jargon).")
@click.option("--json", "as_json", is_flag=True, help="Output JSON for agent consumption.")
def transcribe(
    audio_path: Path,
    output: Path | None,
    language: str | None,
    stt_provider: str | None,
    context: str | None,
    as_json: bool,
) -> None:
    """Transcribe audio file to a transcript JSON."""
    from after_meeting.config import get_settings
    from after_meeting.stt import get_provider
    from after_meeting.errors import STTError

    settings = get_settings()
    if context is not None:
        settings = settings.model_copy(update={"stt_context": context})
    provider_name = stt_provider or settings.stt_provider

    try:
        provider = get_provider(provider_name, settings=settings)
        transcript = provider.transcribe(audio_path, language=language)
    except AfterMeetingError as e:
        if as_json:
            _output_error("transcribe", e)
        raise click.ClickException(str(e)) from e
    except Exception as e:
        err = STTError(str(e), code="STT_PROVIDER_FAILED", recoverable=True)
        if as_json:
            _output_error("transcribe", err)
        raise click.ClickException(str(e)) from e

    output = output or Path(audio_path.stem + "_transcript.json")
    output.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")

    if as_json:
        _output_json({
            "status": "success",
            "step": "transcribe",
            "output_file": str(output.resolve()),
            "metadata": {
                "speakers_detected": len(transcript.speakers),
                "duration_seconds": transcript.metadata.get("duration"),
                "language": transcript.language,
            },
        })
    else:
        click.echo(f"Transcript saved to {output}")


@cli.command()
@click.argument("transcript_path", type=click.Path(exists=True, path_type=Path))
@click.option("--doc-type", type=click.Choice(["minutes", "report"]), required=True)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--title", default=None, help="Override meeting title.")
@click.option("--date", default=None, help="Override meeting date.")
@click.option("--llm-provider", default=None, help="LLM provider name.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON for agent consumption.")
def structure(
    transcript_path: Path,
    doc_type: str,
    output: Path | None,
    title: str | None,
    date: str | None,
    llm_provider: str | None,
    as_json: bool,
) -> None:
    """Structure a transcript into meeting minutes or report."""
    from after_meeting.config import get_settings
    from after_meeting.models import Transcript
    from after_meeting.structuring.analyzer import analyze
    from after_meeting.errors import LLMError

    settings = get_settings()

    try:
        raw = transcript_path.read_text(encoding="utf-8")
        transcript = Transcript.model_validate_json(raw)
    except Exception as e:
        from after_meeting.errors import InputError
        err = InputError(f"Invalid transcript file: {e}", code="INVALID_TRANSCRIPT")
        if as_json:
            _output_error("structure", err)
        raise click.ClickException(str(err)) from e

    try:
        structured = analyze(
            transcript,
            doc_type=doc_type,
            title=title,
            date=date,
            llm_provider=llm_provider or settings.llm_provider,
            settings=settings,
        )
    except AfterMeetingError as e:
        if as_json:
            _output_error("structure", e)
        raise click.ClickException(str(e)) from e
    except Exception as e:
        err = LLMError(str(e), code="LLM_PROVIDER_FAILED", recoverable=True)
        if as_json:
            _output_error("structure", err)
        raise click.ClickException(str(e)) from e

    output = output or Path(transcript_path.stem.replace("_transcript", "") + "_structured.json")
    output.write_text(structured.model_dump_json(indent=2), encoding="utf-8")

    if as_json:
        _output_json({
            "status": "success",
            "step": "structure",
            "output_file": str(output.resolve()),
            "metadata": {
                "doc_type": doc_type,
                "title": structured.title,
                "agenda_count": len(structured.agenda_discussions),
                "action_items_count": len(structured.action_items),
            },
        })
    else:
        click.echo(f"Structured output saved to {output}")


@cli.command()
@click.argument("structured_path", type=click.Path(exists=True, path_type=Path))
@click.option("--format", "fmt", type=click.Choice(["docx", "pdf"]), default="docx")
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--with-appendix", is_flag=True, help="Include full transcript appendix.")
@click.option("--json", "as_json", is_flag=True, help="Output JSON for agent consumption.")
def render(
    structured_path: Path,
    fmt: str,
    output: Path | None,
    with_appendix: bool,
    as_json: bool,
) -> None:
    """Render structured meeting data to DOCX or PDF."""
    from after_meeting.models import StructuredMeeting
    from after_meeting.rendering import get_renderer
    from after_meeting.errors import RenderError

    try:
        raw = structured_path.read_text(encoding="utf-8")
        meeting = StructuredMeeting.model_validate_json(raw)
    except Exception as e:
        from after_meeting.errors import InputError
        err = InputError(f"Invalid structured file: {e}", code="INVALID_STRUCTURED")
        if as_json:
            _output_error("render", err)
        raise click.ClickException(str(err)) from e

    if not with_appendix:
        meeting.full_transcript = None

    stem = structured_path.stem.replace("_structured", "")
    output = output or Path(f"{stem}.{fmt}")

    try:
        renderer = get_renderer(fmt)
        result_path = renderer.render(meeting, output)
    except AfterMeetingError as e:
        if as_json:
            _output_error("render", e)
        raise click.ClickException(str(e)) from e
    except Exception as e:
        err = RenderError(str(e), code="RENDER_FAILED", recoverable=False)
        if as_json:
            _output_error("render", err)
        raise click.ClickException(str(e)) from e

    if as_json:
        _output_json({
            "status": "success",
            "step": "render",
            "output_file": str(result_path.resolve()),
            "metadata": {"format": fmt},
        })
    else:
        click.echo(f"Document saved to {result_path}")


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True, path_type=Path))
@click.option("--doc-type", type=click.Choice(["minutes", "report"]), required=True)
@click.option("--format", "fmt", type=click.Choice(["docx", "pdf"]), default="docx")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--title", default=None, help="Override meeting title.")
@click.option("--date", default=None, help="Override meeting date.")
@click.option("--language", "-l", default=None, help="Language code.")
@click.option("--with-appendix", is_flag=True, help="Include full transcript appendix.")
@click.option("--stt-provider", default=None)
@click.option("--llm-provider", default=None)
@click.option("--context", "-c", default=None, help="Contextual hints for ASR (topic, names, jargon).")
@click.option("--chunk-minutes", type=int, default=None, help="Audio chunk size in minutes (default: 55).")
@click.option("--json", "as_json", is_flag=True, help="Output JSON for agent consumption.")
def process(
    audio_path: Path,
    doc_type: str,
    fmt: str,
    output_dir: Path | None,
    title: str | None,
    date: str | None,
    language: str | None,
    with_appendix: bool,
    stt_provider: str | None,
    llm_provider: str | None,
    context: str | None,
    chunk_minutes: int | None,
    as_json: bool,
) -> None:
    """Full pipeline: audio → transcript → structured → document."""
    from after_meeting.pipeline import run_pipeline

    try:
        result = run_pipeline(
            audio_path=audio_path,
            doc_type=doc_type,
            fmt=fmt,
            output_dir=output_dir,
            title=title,
            date=date,
            language=language,
            with_appendix=with_appendix,
            stt_provider=stt_provider,
            llm_provider=llm_provider,
            context=context,
            chunk_minutes=chunk_minutes,
        )
    except AfterMeetingError as e:
        if as_json:
            _output_error("process", e)
        raise click.ClickException(str(e)) from e
    except Exception as e:
        err = AfterMeetingError(str(e), code="PIPELINE_FAILED", recoverable=False)
        if as_json:
            _output_error("process", err)
        raise click.ClickException(str(e)) from e

    if as_json:
        _output_json(result)
    else:
        click.echo(f"Pipeline complete. Output: {result.get('output_file', 'unknown')}")
