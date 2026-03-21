"""Per-utterance speaker diarization using CAM++ embeddings.

Assigns speaker labels to utterances that have no diarization
(e.g., from Qwen3 ASR which outputs all utterances as "Speaker 0").
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from after_meeting.models import Transcript, Utterance
from after_meeting.speaker.embedder import SpeakerEmbedder

logger = logging.getLogger(__name__)


def _cluster_embeddings(
    embeddings: list[np.ndarray],
    num_speakers: int | None = None,
    similarity_threshold: float = 0.75,
) -> list[int]:
    """Cluster embeddings into speaker groups via agglomerative clustering.

    If num_speakers is None, auto-detect by merging until similarity < threshold.
    Returns list of cluster labels (0-indexed).
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]

    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            s = float(np.dot(embeddings[i], embeddings[j]))
            sim[i, j] = s
            sim[j, i] = s

    if num_speakers is not None and num_speakers == 1:
        return [0] * n

    labels = list(range(n))

    def cluster_sim(c1: list[int], c2: list[int]) -> float:
        total = sum(sim[i, j] for i in c1 for j in c2)
        count = len(c1) * len(c2)
        return total / count if count > 0 else 0.0

    while True:
        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels)

        if num_speakers is not None and n_clusters <= num_speakers:
            break
        if n_clusters <= 1:
            break

        best_sim = -1.0
        best_pair = (0, 0)
        clusters = {l: [i for i, lab in enumerate(labels) if lab == l] for l in unique_labels}

        for idx_a, la in enumerate(unique_labels):
            for lb in unique_labels[idx_a + 1:]:
                s = cluster_sim(clusters[la], clusters[lb])
                if s > best_sim:
                    best_sim = s
                    best_pair = (la, lb)

        if num_speakers is None and best_sim < similarity_threshold:
            break

        merge_from, merge_to = best_pair[1], best_pair[0]
        labels = [merge_to if l == merge_from else l for l in labels]

    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return [remap[l] for l in labels]


def diarize_transcript(
    transcript: Transcript,
    audio_path: Path,
    num_speakers: int | None = None,
) -> Transcript:
    """Assign speaker labels to utterances using CAM++ voice embeddings."""
    if not transcript.utterances:
        return transcript

    embedder = SpeakerEmbedder()
    embeddings: list[np.ndarray | None] = []

    for utt in transcript.utterances:
        try:
            emb = embedder.embed(audio_path, start=utt.start_time, end=utt.end_time)
            embeddings.append(emb)
        except Exception:
            logger.warning("Failed to embed utterance %.1f-%.1f", utt.start_time, utt.end_time)
            embeddings.append(None)

    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    valid_embeddings = [embeddings[i] for i in valid_indices]

    if not valid_embeddings:
        logger.warning("No embeddings extracted, returning transcript unchanged")
        return transcript

    cluster_labels = _cluster_embeddings(valid_embeddings, num_speakers=num_speakers)

    full_labels = [0] * len(embeddings)
    for idx, valid_idx in enumerate(valid_indices):
        full_labels[valid_idx] = cluster_labels[idx]

    for i, emb in enumerate(embeddings):
        if emb is None and valid_indices:
            nearest = min(valid_indices, key=lambda vi: abs(
                transcript.utterances[vi].start_time - transcript.utterances[i].start_time
            ))
            full_labels[i] = full_labels[nearest]

    new_utterances = [
        Utterance(
            speaker=f"Speaker {label + 1}",
            start_time=utt.start_time,
            end_time=utt.end_time,
            text=utt.text,
        )
        for utt, label in zip(transcript.utterances, full_labels)
    ]

    new_speakers = sorted({u.speaker for u in new_utterances})

    return Transcript(
        language=transcript.language,
        speakers=new_speakers,
        utterances=new_utterances,
        metadata=transcript.metadata,
    )
