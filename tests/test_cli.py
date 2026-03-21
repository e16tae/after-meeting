"""CLI tests using click.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from after_meeting.cli import cli
from after_meeting.models import (
    ActionItem,
    AgendaDiscussion,
    StructuredMeeting,
    Transcript,
    Utterance,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def sample_transcript() -> Transcript:
    return Transcript(
        language="ko",
        speakers=["Speaker 1", "Speaker 2"],
        utterances=[
            Utterance(speaker="Speaker 1", start_time=0.0, end_time=5.0, text="안녕하세요"),
            Utterance(speaker="Speaker 2", start_time=5.5, end_time=10.0, text="네, 반갑습니다"),
        ],
        metadata={"duration": 10.0, "audio_file": "meeting.mp3"},
    )


@pytest.fixture()
def sample_structured() -> StructuredMeeting:
    return StructuredMeeting(
        title="테스트 회의",
        date="2026-03-17",
        doc_type="minutes",
        agenda_discussions=[
            AgendaDiscussion(
                topic="안건 1", summary="요약", speaker_contributions=[]
            ),
        ],
        decisions=["결정 1"],
        action_items=[ActionItem(description="할 일 1")],
    )


# ------------------------------------------------------------------
# Version / Help
# ------------------------------------------------------------------
class TestCLIBasics:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "transcribe" in result.output
        assert "structure" in result.output
        assert "render" in result.output
        assert "process" in result.output


# ------------------------------------------------------------------
# transcribe
# ------------------------------------------------------------------
class TestTranscribeCommand:
    def test_json_output_success(
        self, runner: CliRunner, sample_transcript: Transcript, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_provider = MagicMock()
        mock_provider.transcribe.return_value = sample_transcript

        with patch("after_meeting.stt.get_provider", return_value=mock_provider):
            result = runner.invoke(
                cli,
                ["transcribe", str(audio_file), "--json", "--output", str(tmp_path / "out.json")],
            )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "success"
        assert data["step"] == "transcribe"
        assert "output_file" in data
        assert data["metadata"]["speakers_detected"] == 2
        assert data["metadata"]["language"] == "ko"

    def test_missing_audio_file(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["transcribe", "/nonexistent.mp3", "--json"])
        assert result.exit_code != 0


# ------------------------------------------------------------------
# structure
# ------------------------------------------------------------------
class TestStructureCommand:
    def test_json_output_success(
        self,
        runner: CliRunner,
        sample_structured: StructuredMeeting,
        tmp_path: Path,
    ) -> None:
        # Write a valid transcript file
        transcript = Transcript(
            language="ko",
            speakers=["Speaker 1"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0, end_time=5, text="테스트"),
            ],
        )
        transcript_path = tmp_path / "transcript.json"
        transcript_path.write_text(transcript.model_dump_json(), encoding="utf-8")

        with patch("after_meeting.structuring.analyzer.analyze", return_value=sample_structured):
            result = runner.invoke(
                cli,
                [
                    "structure",
                    str(transcript_path),
                    "--doc-type", "minutes",
                    "--json",
                    "--output", str(tmp_path / "structured.json"),
                ],
            )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "success"
        assert data["step"] == "structure"
        assert data["metadata"]["doc_type"] == "minutes"
        assert data["metadata"]["title"] == "테스트 회의"

    def test_invalid_transcript_file(self, runner: CliRunner, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{", encoding="utf-8")

        result = runner.invoke(
            cli, ["structure", str(bad_file), "--doc-type", "minutes", "--json"]
        )
        assert result.exit_code != 0


# ------------------------------------------------------------------
# render
# ------------------------------------------------------------------
class TestRenderCommand:
    def test_json_output_success(
        self, runner: CliRunner, sample_structured: StructuredMeeting, tmp_path: Path
    ) -> None:
        structured_path = tmp_path / "structured.json"
        structured_path.write_text(
            sample_structured.model_dump_json(), encoding="utf-8"
        )

        out_path = tmp_path / "output.docx"
        result = runner.invoke(
            cli,
            [
                "render",
                str(structured_path),
                "--format", "docx",
                "--json",
                "--output", str(out_path),
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "success"
        assert data["step"] == "render"
        assert data["metadata"]["format"] == "docx"
        assert out_path.exists()

    def test_invalid_structured_file(self, runner: CliRunner, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{}", encoding="utf-8")

        result = runner.invoke(
            cli, ["render", str(bad_file), "--format", "docx", "--json"]
        )
        assert result.exit_code != 0

    def test_without_json_flag(
        self, runner: CliRunner, sample_structured: StructuredMeeting, tmp_path: Path
    ) -> None:
        structured_path = tmp_path / "structured.json"
        structured_path.write_text(
            sample_structured.model_dump_json(), encoding="utf-8"
        )

        result = runner.invoke(
            cli,
            ["render", str(structured_path), "--format", "docx", "--output", str(tmp_path / "out.docx")],
        )
        assert result.exit_code == 0
        assert "Document saved to" in result.output


# ------------------------------------------------------------------
# process
# ------------------------------------------------------------------
class TestProcessCommand:
    def test_json_output_success(self, runner: CliRunner, tmp_path: Path) -> None:
        audio_file = tmp_path / "meeting.mp3"
        audio_file.write_bytes(b"fake audio")

        mock_result = {
            "status": "success",
            "step": "process",
            "output_file": str(tmp_path / "meeting.docx"),
            "metadata": {"doc_type": "minutes", "format": "docx"},
        }

        with patch("after_meeting.pipeline.run_pipeline", return_value=mock_result):
            result = runner.invoke(
                cli,
                ["process", str(audio_file), "--doc-type", "minutes", "--format", "docx", "--json"],
            )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "success"
        assert data["step"] == "process"

    def test_missing_audio(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["process", "/nonexistent.mp3", "--doc-type", "minutes", "--json"]
        )
        assert result.exit_code != 0
