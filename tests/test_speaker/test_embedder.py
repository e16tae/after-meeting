"""Tests for speaker embedding extraction using wespeaker ONNX."""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from after_meeting.speaker.embedder import (
    SpeakerEmbedder,
    extract_speaker_segments,
    _load_audio_segment,
    _MODEL_PATH,
)
from after_meeting.models import Transcript, Utterance

_model_available = _MODEL_PATH.exists()
requires_model = pytest.mark.skipif(
    not _model_available,
    reason="wespeaker CAM++ model not downloaded",
)


def _create_wav_with_tone(path: Path, duration: float = 5.0, freq: float = 440.0) -> None:
    """Create a WAV with a sine tone."""
    sample_rate = 16000
    n_frames = int(sample_rate * duration)
    samples = [int(32767 * 0.5 * np.sin(2 * np.pi * freq * t / sample_rate)) for t in range(n_frames)]
    with wave.open(str(path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(struct.pack(f"<{n_frames}h", *samples))


class TestLoadAudioSegment:
    def test_loads_correct_duration(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _create_wav_with_tone(wav, duration=10.0)
        audio = _load_audio_segment(wav, start=2.0, end=5.0, sample_rate=16000)
        assert abs(len(audio) - 48000) < 1600

    def test_clamps_to_file_bounds(self, tmp_path: Path) -> None:
        wav = tmp_path / "test.wav"
        _create_wav_with_tone(wav, duration=5.0)
        audio = _load_audio_segment(wav, start=3.0, end=999.0, sample_rate=16000)
        assert len(audio) > 0
        assert len(audio) <= 16000 * 3


class TestExtractSpeakerSegments:
    def test_groups_utterances_by_speaker(self) -> None:
        transcript = Transcript(
            language="ko",
            speakers=["Speaker 1", "Speaker 2"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0.0, end_time=10.0, text="발언1"),
                Utterance(speaker="Speaker 2", start_time=10.0, end_time=25.0, text="발언2"),
                Utterance(speaker="Speaker 1", start_time=25.0, end_time=50.0, text="발언3"),
            ],
        )
        segments = extract_speaker_segments(transcript, top_k=2)
        assert "Speaker 1" in segments
        assert "Speaker 2" in segments
        assert len(segments["Speaker 1"]) == 2
        assert len(segments["Speaker 2"]) == 1

    def test_sorts_by_duration_descending(self) -> None:
        transcript = Transcript(
            language="ko",
            speakers=["Speaker 1"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0.0, end_time=3.0, text="짧은"),
                Utterance(speaker="Speaker 1", start_time=5.0, end_time=20.0, text="긴 발언"),
                Utterance(speaker="Speaker 1", start_time=25.0, end_time=35.0, text="중간"),
            ],
        )
        segments = extract_speaker_segments(transcript, top_k=3)
        durations = [end - start for start, end in segments["Speaker 1"]]
        assert durations == sorted(durations, reverse=True)


@requires_model
class TestSpeakerEmbedder:
    """These tests require the wespeaker CAM++ ONNX model (~28MB, auto-downloaded)."""

    def test_embed_returns_normalized_vector(self, tmp_path: Path) -> None:
        wav = tmp_path / "tone.wav"
        _create_wav_with_tone(wav, duration=5.0)
        embedder = SpeakerEmbedder()
        embedding = embedder.embed(wav, start=0.0, end=5.0)
        assert embedding.ndim == 1
        assert embedding.shape[0] == 512  # CAM++ outputs 512-dim
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01

    def test_same_audio_high_similarity(self, tmp_path: Path) -> None:
        wav = tmp_path / "tone.wav"
        _create_wav_with_tone(wav, duration=10.0)
        embedder = SpeakerEmbedder()
        emb1 = embedder.embed(wav, start=0.0, end=5.0)
        emb2 = embedder.embed(wav, start=2.0, end=7.0)
        similarity = float(np.dot(emb1, emb2))
        assert similarity > 0.8

    def test_embedding_dimension(self, tmp_path: Path) -> None:
        """Different length audio should produce same-dimension 512-d embeddings."""
        wav1 = tmp_path / "short.wav"
        wav2 = tmp_path / "long.wav"
        _create_wav_with_tone(wav1, duration=3.0, freq=200.0)
        _create_wav_with_tone(wav2, duration=8.0, freq=440.0)
        embedder = SpeakerEmbedder()
        emb1 = embedder.embed(wav1, start=0.0, end=3.0)
        emb2 = embedder.embed(wav2, start=0.0, end=8.0)
        assert emb1.shape == emb2.shape
        assert emb1.shape[0] == 512
