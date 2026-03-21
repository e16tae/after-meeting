"""Speaker identification and alignment across audio chunks."""


def align_speakers(*args, **kwargs):
    """Align speaker IDs across multiple chunk transcripts using voice embeddings.

    Lazy-imported from after_meeting.speaker.aligner.
    """
    from after_meeting.speaker.aligner import align_speakers as _impl
    return _impl(*args, **kwargs)


__all__ = ["align_speakers"]
