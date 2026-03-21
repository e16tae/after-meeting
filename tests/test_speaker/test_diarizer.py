"""Tests for per-utterance speaker diarization."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from after_meeting.models import Transcript, Utterance
from after_meeting.speaker.diarizer import diarize_transcript


def _make_transcript(n_utterances: int = 6) -> Transcript:
    utts = []
    t = 0.0
    for i in range(n_utterances):
        utts.append(Utterance(
            speaker="Speaker 0",
            start_time=t,
            end_time=t + 10.0,
            text=f"발언 {i + 1}",
        ))
        t += 12.0
    return Transcript(language="ko", speakers=["Speaker 0"], utterances=utts)


class TestDiarizeTranscript:
    def test_assigns_multiple_speakers(self) -> None:
        transcript = _make_transcript(6)

        mock_embedder = MagicMock()
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.98, 0.02, 0.0]),
            np.array([0.02, 0.97, 0.0]),
            np.array([0.95, 0.05, 0.0]),
            np.array([0.05, 0.95, 0.0]),
        ]
        mock_embedder.embed.side_effect = embeddings

        with patch("after_meeting.speaker.diarizer.SpeakerEmbedder", return_value=mock_embedder):
            result = diarize_transcript(transcript, Path("/tmp/audio.wav"), num_speakers=2)

        assert len(result.speakers) == 2
        assert result.utterances[0].speaker == result.utterances[2].speaker
        assert result.utterances[1].speaker == result.utterances[3].speaker
        assert result.utterances[0].speaker != result.utterances[1].speaker

    def test_single_speaker(self) -> None:
        transcript = _make_transcript(3)

        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = [
            np.array([1.0, 0.0]),
            np.array([0.98, 0.02]),
            np.array([0.97, 0.03]),
        ]

        with patch("after_meeting.speaker.diarizer.SpeakerEmbedder", return_value=mock_embedder):
            result = diarize_transcript(transcript, Path("/tmp/audio.wav"), num_speakers=1)

        assert len(result.speakers) == 1
        assert all(u.speaker == "Speaker 1" for u in result.utterances)

    def test_auto_detect_speakers(self) -> None:
        transcript = _make_transcript(4)

        mock_embedder = MagicMock()
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.99, 0.01, 0.0]),
            np.array([0.01, 0.99, 0.0]),
        ]
        mock_embedder.embed.side_effect = embeddings

        with patch("after_meeting.speaker.diarizer.SpeakerEmbedder", return_value=mock_embedder):
            result = diarize_transcript(transcript, Path("/tmp/audio.wav"), num_speakers=None)

        assert len(result.speakers) >= 2

    def test_empty_transcript(self) -> None:
        transcript = Transcript(language="ko", speakers=[], utterances=[])
        result = diarize_transcript(transcript, Path("/tmp/audio.wav"))
        assert len(result.utterances) == 0

    def test_embedding_failure_graceful(self) -> None:
        transcript = _make_transcript(3)

        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = [
            np.array([1.0, 0.0]),
            Exception("embedding failed"),
            np.array([0.98, 0.02]),
        ]

        with patch("after_meeting.speaker.diarizer.SpeakerEmbedder", return_value=mock_embedder):
            result = diarize_transcript(transcript, Path("/tmp/audio.wav"), num_speakers=1)

        assert len(result.utterances) == 3
