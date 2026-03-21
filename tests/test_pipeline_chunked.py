"""Integration test for the full pipeline with mocked providers."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from after_meeting.models import Transcript, Utterance


def _make_transcript(num_speakers: int = 2) -> Transcript:
    utts = []
    for i in range(10):
        speaker_idx = (i % num_speakers) + 1
        utts.append(Utterance(
            speaker=f"Speaker {speaker_idx}" if num_speakers > 1 else "Speaker 0",
            start_time=i * 25.0,
            end_time=i * 25.0 + 20.0,
            text=f"발언 {i + 1}",
        ))
    speakers = [f"Speaker {i + 1}" for i in range(num_speakers)] if num_speakers > 1 else ["Speaker 0"]
    return Transcript(language="ko", speakers=speakers, utterances=utts)


def _make_structured_json() -> str:
    return json.dumps({
        "title": "테스트 회의",
        "date": "2026-03-18",
        "doc_type": "minutes",
        "agenda_discussions": [{"topic": "안건", "summary": "요약", "speaker_contributions": []}],
        "decisions": ["결정"],
        "action_items": [],
        "executive_summary": None,
    }, ensure_ascii=False)


class TestSimplifiedPipeline:
    def test_pipeline_produces_output(self, tmp_path: Path) -> None:
        """Pipeline should transcribe, structure, and render a document."""
        mp3 = tmp_path / "meeting.mp3"
        mp3.write_bytes(b"fake audio")

        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = _make_transcript(num_speakers=2)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_structured_json()

        with patch("after_meeting.stt.get_provider", return_value=mock_stt), \
             patch("after_meeting.structuring.analyzer.get_provider", return_value=mock_llm):

            from after_meeting.pipeline import run_pipeline
            result = run_pipeline(
                audio_path=mp3,
                doc_type="minutes",
                output_dir=tmp_path,
            )

        assert result["status"] == "success"
        assert Path(result["output_file"]).exists()
        assert result["metadata"]["speakers_detected"] == 2

    def test_pipeline_triggers_diarization_for_single_speaker(self, tmp_path: Path) -> None:
        """When STT returns only 1 speaker, diarization should be attempted."""
        mp3 = tmp_path / "meeting.mp3"
        mp3.write_bytes(b"fake audio")

        single_speaker_transcript = _make_transcript(num_speakers=1)
        diarized_transcript = _make_transcript(num_speakers=2)

        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = single_speaker_transcript

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_structured_json()

        with patch("after_meeting.stt.get_provider", return_value=mock_stt), \
             patch("after_meeting.structuring.analyzer.get_provider", return_value=mock_llm), \
             patch("after_meeting.pipeline.logger") as mock_logger, \
             patch("after_meeting.speaker.diarizer.diarize_transcript", return_value=diarized_transcript) as mock_diarize:

            from after_meeting.pipeline import run_pipeline
            result = run_pipeline(
                audio_path=mp3,
                doc_type="minutes",
                output_dir=tmp_path,
            )

        assert result["status"] == "success"
        mock_diarize.assert_called_once()
        assert result["metadata"]["speakers_detected"] == 2

    def test_pipeline_skips_diarization_for_multi_speaker(self, tmp_path: Path) -> None:
        """When STT returns multiple speakers, diarization should be skipped."""
        mp3 = tmp_path / "meeting.mp3"
        mp3.write_bytes(b"fake audio")

        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = _make_transcript(num_speakers=2)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_structured_json()

        with patch("after_meeting.stt.get_provider", return_value=mock_stt), \
             patch("after_meeting.structuring.analyzer.get_provider", return_value=mock_llm), \
             patch("after_meeting.speaker.diarizer.diarize_transcript") as mock_diarize:

            from after_meeting.pipeline import run_pipeline
            result = run_pipeline(
                audio_path=mp3,
                doc_type="minutes",
                output_dir=tmp_path,
            )

        assert result["status"] == "success"
        mock_diarize.assert_not_called()
        assert result["metadata"]["speakers_detected"] == 2

    def test_pipeline_continues_if_diarization_fails(self, tmp_path: Path) -> None:
        """Pipeline should continue gracefully if diarization raises an error."""
        mp3 = tmp_path / "meeting.mp3"
        mp3.write_bytes(b"fake audio")

        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = _make_transcript(num_speakers=1)

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _make_structured_json()

        with patch("after_meeting.stt.get_provider", return_value=mock_stt), \
             patch("after_meeting.structuring.analyzer.get_provider", return_value=mock_llm), \
             patch("after_meeting.speaker.diarizer.diarize_transcript", side_effect=RuntimeError("test error")):

            from after_meeting.pipeline import run_pipeline
            result = run_pipeline(
                audio_path=mp3,
                doc_type="minutes",
                output_dir=tmp_path,
            )

        assert result["status"] == "success"
        assert result["metadata"]["speakers_detected"] == 1
