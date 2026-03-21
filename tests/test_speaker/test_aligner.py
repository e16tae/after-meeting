"""Tests for cross-chunk speaker alignment."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from after_meeting.speaker.aligner import (
    build_similarity_matrix,
    resolve_mapping,
    apply_speaker_mappings,
    align_speakers,
)
from after_meeting.models import SpeakerMapping, Transcript, Utterance


class TestBuildSimilarityMatrix:
    def test_identical_embeddings_give_ones(self) -> None:
        ref = {"S1": np.array([1.0, 0.0]), "S2": np.array([0.0, 1.0])}
        target = {"S1": np.array([1.0, 0.0]), "S2": np.array([0.0, 1.0])}
        matrix, ref_keys, tgt_keys = build_similarity_matrix(ref, target)
        assert matrix.shape == (2, 2)
        for i in range(2):
            assert matrix[i, i] == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_embeddings_give_zeros(self) -> None:
        ref = {"S1": np.array([1.0, 0.0])}
        target = {"S1": np.array([0.0, 1.0])}
        matrix, _, _ = build_similarity_matrix(ref, target)
        assert matrix[0, 0] == pytest.approx(0.0, abs=0.01)


class TestResolveMapping:
    def test_clear_mapping(self) -> None:
        sim_matrix = np.array([
            [0.2, 0.95],
            [0.93, 0.1],
        ])
        ref_keys = ["S1", "S2"]
        tgt_keys = ["S1", "S2"]
        mapping, confidence = resolve_mapping(sim_matrix, ref_keys, tgt_keys)
        assert mapping["S1"] == "S2"
        assert mapping["S2"] == "S1"
        assert confidence > 0.8

    def test_new_speaker_detection(self) -> None:
        sim_matrix = np.array([
            [0.95, 0.1, 0.05],
            [0.1, 0.92, 0.08],
        ])
        ref_keys = ["S1", "S2"]
        tgt_keys = ["S1", "S2", "S3"]
        mapping, confidence = resolve_mapping(sim_matrix, ref_keys, tgt_keys)
        assert mapping["S1"] == "S1"
        assert mapping["S2"] == "S2"
        assert "S3" not in mapping

    def test_low_similarity_yields_low_confidence(self) -> None:
        sim_matrix = np.array([[0.3, 0.35], [0.32, 0.28]])
        ref_keys = ["S1", "S2"]
        tgt_keys = ["S1", "S2"]
        _, confidence = resolve_mapping(sim_matrix, ref_keys, tgt_keys)
        assert confidence < 0.6


class TestApplySpeakerMappings:
    def test_remaps_speaker_ids(self) -> None:
        transcript = Transcript(
            language="ko",
            speakers=["Speaker 1", "Speaker 2"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0.0, end_time=5.0, text="A"),
                Utterance(speaker="Speaker 2", start_time=5.0, end_time=10.0, text="B"),
            ],
        )
        mapping = SpeakerMapping(
            chunk_index=1,
            mapping={"Speaker 1": "Speaker 2", "Speaker 2": "Speaker 1"},
            confidence=0.9,
        )
        result = apply_speaker_mappings(transcript, mapping)
        assert result.utterances[0].speaker == "Speaker 2"
        assert result.utterances[1].speaker == "Speaker 1"
        assert set(result.speakers) == {"Speaker 1", "Speaker 2"}

    def test_unmapped_speakers_get_new_id(self) -> None:
        transcript = Transcript(
            language="ko",
            speakers=["Speaker 1", "Speaker 2"],
            utterances=[
                Utterance(speaker="Speaker 1", start_time=0.0, end_time=5.0, text="A"),
                Utterance(speaker="Speaker 2", start_time=5.0, end_time=10.0, text="B"),
            ],
        )
        mapping = SpeakerMapping(
            chunk_index=1,
            mapping={"Speaker 1": "Speaker 1"},
            confidence=0.9,
        )
        result = apply_speaker_mappings(transcript, mapping, next_speaker_id=3)
        assert result.utterances[0].speaker == "Speaker 1"
        assert result.utterances[1].speaker == "Speaker 3"


class TestAlignSpeakers:
    def test_single_transcript_returns_unchanged(self) -> None:
        t = Transcript(language="ko", speakers=["S1"], utterances=[
            Utterance(speaker="S1", start_time=0, end_time=5, text="test"),
        ])
        aligned, mappings = align_speakers([t], [Path("/tmp/a.wav")])
        assert len(aligned) == 1
        assert mappings == []

    def test_two_chunks_with_mocked_embedder(self) -> None:
        t1 = Transcript(language="ko", speakers=["Speaker 1", "Speaker 2"], utterances=[
            Utterance(speaker="Speaker 1", start_time=0, end_time=10, text="발언1"),
            Utterance(speaker="Speaker 2", start_time=10, end_time=20, text="발언2"),
        ])
        t2 = Transcript(language="ko", speakers=["Speaker 1", "Speaker 2"], utterances=[
            Utterance(speaker="Speaker 1", start_time=0, end_time=10, text="발언3"),
            Utterance(speaker="Speaker 2", start_time=10, end_time=20, text="발언4"),
        ])

        mock_embedder = MagicMock()
        mock_embedder.embed_speaker.side_effect = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.05, 0.95]),
            np.array([0.97, 0.03]),
        ]

        with patch("after_meeting.speaker.aligner.SpeakerEmbedder", return_value=mock_embedder):
            aligned, mappings = align_speakers(
                [t1, t2],
                [Path("/tmp/a.wav"), Path("/tmp/b.wav")],
            )

        assert len(aligned) == 2
        assert len(mappings) == 1
        assert aligned[1].utterances[0].speaker == "Speaker 2"
        assert aligned[1].utterances[1].speaker == "Speaker 1"
