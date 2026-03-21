"""Cross-chunk speaker alignment using cosine similarity + Hungarian algorithm."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from after_meeting.models import SpeakerMapping, Transcript, Utterance
from after_meeting.speaker.embedder import SpeakerEmbedder, extract_speaker_segments

logger = logging.getLogger(__name__)


def build_similarity_matrix(
    ref_embeddings: dict[str, np.ndarray],
    target_embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a cosine similarity matrix between reference and target speaker embeddings."""
    ref_keys = sorted(ref_embeddings.keys())
    tgt_keys = sorted(target_embeddings.keys())

    matrix = np.zeros((len(ref_keys), len(tgt_keys)), dtype=np.float64)
    for i, rk in enumerate(ref_keys):
        for j, tk in enumerate(tgt_keys):
            matrix[i, j] = float(np.dot(ref_embeddings[rk], target_embeddings[tk]))

    return matrix, ref_keys, tgt_keys


def resolve_mapping(
    sim_matrix: np.ndarray,
    ref_keys: list[str],
    tgt_keys: list[str],
    threshold: float = 0.5,
) -> tuple[dict[str, str], float]:
    """Resolve optimal speaker mapping using the Hungarian algorithm."""
    n_ref = len(ref_keys)
    n_tgt = len(tgt_keys)

    size = max(n_ref, n_tgt)
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:n_ref, :n_tgt] = -sim_matrix

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[str, str] = {}
    similarities: list[float] = []

    for r, c in zip(row_ind, col_ind):
        if r >= n_ref or c >= n_tgt:
            continue
        sim = sim_matrix[r, c]
        if sim >= threshold:
            mapping[tgt_keys[c]] = ref_keys[r]
            similarities.append(sim)

    confidence = float(np.mean(similarities)) if similarities else 0.0
    return mapping, confidence


def apply_speaker_mappings(
    transcript: Transcript,
    mapping: SpeakerMapping,
    next_speaker_id: int | None = None,
) -> Transcript:
    """Apply a speaker mapping to a transcript, renaming speaker IDs."""
    id_map = dict(mapping.mapping)

    if next_speaker_id is not None:
        counter = next_speaker_id
        for utt in transcript.utterances:
            if utt.speaker not in id_map:
                id_map[utt.speaker] = f"Speaker {counter}"
                counter += 1

    new_utterances = [
        Utterance(
            speaker=id_map.get(u.speaker, u.speaker),
            start_time=u.start_time,
            end_time=u.end_time,
            text=u.text,
        )
        for u in transcript.utterances
    ]

    new_speakers = sorted({u.speaker for u in new_utterances})

    return Transcript(
        language=transcript.language,
        speakers=new_speakers,
        utterances=new_utterances,
        metadata=transcript.metadata,
    )


def align_speakers(
    transcripts: list[Transcript],
    audio_paths: list[Path],
) -> tuple[list[Transcript], list[SpeakerMapping]]:
    """Align speaker IDs across multiple chunk transcripts using voice embeddings."""
    if len(transcripts) <= 1:
        return transcripts, []

    embedder = SpeakerEmbedder()

    ref_segments = extract_speaker_segments(transcripts[0])
    ref_embeddings: dict[str, np.ndarray] = {}
    for speaker, segs in ref_segments.items():
        ref_embeddings[speaker] = embedder.embed_speaker(audio_paths[0], segs)

    aligned = [transcripts[0]]
    mappings: list[SpeakerMapping] = []
    known_speakers = set(transcripts[0].speakers)
    next_id = len(known_speakers) + 1

    for idx in range(1, len(transcripts)):
        tgt_segments = extract_speaker_segments(transcripts[idx])
        tgt_embeddings: dict[str, np.ndarray] = {}
        for speaker, segs in tgt_segments.items():
            tgt_embeddings[speaker] = embedder.embed_speaker(audio_paths[idx], segs)

        sim_matrix, ref_keys, tgt_keys = build_similarity_matrix(ref_embeddings, tgt_embeddings)
        mapping_dict, confidence = resolve_mapping(sim_matrix, ref_keys, tgt_keys)

        sp_mapping = SpeakerMapping(
            chunk_index=idx,
            mapping=mapping_dict,
            confidence=confidence,
        )
        mappings.append(sp_mapping)

        aligned_transcript = apply_speaker_mappings(
            transcripts[idx], sp_mapping, next_speaker_id=next_id
        )
        aligned.append(aligned_transcript)

        for spk in aligned_transcript.speakers:
            if spk not in known_speakers:
                known_speakers.add(spk)
                next_id += 1
                new_segs = extract_speaker_segments(
                    Transcript(
                        language=aligned_transcript.language,
                        speakers=[spk],
                        utterances=[u for u in aligned_transcript.utterances if u.speaker == spk],
                    )
                )
                if spk in new_segs:
                    ref_embeddings[spk] = embedder.embed_speaker(audio_paths[idx], new_segs[spk])

    return aligned, mappings
