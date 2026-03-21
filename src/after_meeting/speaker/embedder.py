"""Speaker embedding extraction using wespeaker CAM++ ONNX model."""
from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from after_meeting.errors import SpeakerError
from after_meeting.models import Transcript

logger = logging.getLogger(__name__)

_MODEL_URL = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-campplus/resolve/main/voxceleb_CAM%2B%2B.onnx"
_MODEL_DIR = Path.home() / ".cache" / "after-meeting" / "wespeaker"
_MODEL_PATH = _MODEL_DIR / "cam_pp.onnx"
_SAMPLE_RATE = 16000


def _load_audio_segment(
    audio_path: Path,
    start: float,
    end: float,
    sample_rate: int = _SAMPLE_RATE,
) -> np.ndarray:
    """Load a segment of an audio file as a float32 numpy array.

    Uses ffmpeg for any format (m4a, mp3, wav, flac, etc.).
    Falls back to the wave module for WAV files if ffmpeg is unavailable.
    """
    duration = end - start

    # Try ffmpeg first (supports all formats)
    try:
        import subprocess
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(duration),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and result.stdout:
            audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
            audio = audio / 32767.0
            return audio
    except Exception:
        pass

    # Fallback: wave module (WAV only)
    with wave.open(str(audio_path), "r") as f:
        sr = f.getframerate()
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        total_frames = f.getnframes()

        start_frame = max(0, int(start * sr))
        end_frame = min(total_frames, int(end * sr))

        f.setpos(start_frame)
        raw = f.readframes(end_frame - start_frame)

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        dtype = np.int16

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    if n_channels > 1:
        audio = audio[::n_channels]

    max_val = float(np.iinfo(dtype).max)
    audio = audio / max_val

    if sr != sample_rate:
        ratio = sample_rate / sr
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio)

    return audio


def _compute_fbank(
    audio: np.ndarray,
    sample_rate: int = _SAMPLE_RATE,
    num_mel_bins: int = 80,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
) -> np.ndarray:
    """Compute log Mel filterbank features from raw audio.

    Returns shape (num_frames, num_mel_bins) matching wespeaker CAM++ input spec.
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)

    # Round frame_length to next power of 2 for FFT
    nfft = 1
    while nfft < frame_length:
        nfft *= 2

    # Hamming window
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / (frame_length - 1))

    # Frame the signal
    num_frames = max(1, 1 + (len(audio) - frame_length) // frame_shift)
    frames = np.zeros((num_frames, frame_length), dtype=np.float32)
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_length
        if end <= len(audio):
            frames[i] = audio[start:end]
        else:
            frames[i, : len(audio) - start] = audio[start:]

    # Apply window and FFT
    frames *= window
    spec = np.fft.rfft(frames, n=nfft)
    power_spec = np.abs(spec) ** 2

    # Mel filterbank
    low_freq = 0.0
    high_freq = sample_rate / 2.0
    low_mel = 2595.0 * np.log10(1.0 + low_freq / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + high_freq / 700.0)
    mel_points = np.linspace(low_mel, high_mel, num_mel_bins + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)

    fbank_matrix = np.zeros((num_mel_bins, nfft // 2 + 1), dtype=np.float32)
    for m in range(num_mel_bins):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        for k in range(left, center):
            if center > left:
                fbank_matrix[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank_matrix[m, k] = (right - k) / (right - center)

    fbank_features = np.dot(power_spec, fbank_matrix.T)
    fbank_features = np.where(fbank_features > 0, fbank_features, np.finfo(np.float32).eps)
    log_fbank = np.log(fbank_features)

    # CMN (Cepstral Mean Normalization)
    log_fbank -= np.mean(log_fbank, axis=0, keepdims=True)

    return log_fbank.astype(np.float32)


def extract_speaker_segments(
    transcript: Transcript,
    top_k: int = 3,
) -> dict[str, list[tuple[float, float]]]:
    """For each speaker, return the top_k longest utterance time ranges."""
    speaker_utts: dict[str, list[tuple[float, float, float]]] = {}
    for utt in transcript.utterances:
        duration = utt.end_time - utt.start_time
        speaker_utts.setdefault(utt.speaker, []).append((utt.start_time, utt.end_time, duration))

    result: dict[str, list[tuple[float, float]]] = {}
    for speaker, segments in speaker_utts.items():
        segments.sort(key=lambda x: x[2], reverse=True)
        result[speaker] = [(s, e) for s, e, _ in segments[:top_k]]

    return result


def _ensure_model() -> Path:
    """Download wespeaker CAM++ ONNX model if not cached."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import httpx
        logger.info("Downloading wespeaker CAM++ model...")
        with httpx.stream("GET", _MODEL_URL, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(_MODEL_PATH, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)
        logger.info("Model saved to %s", _MODEL_PATH)
    except Exception as exc:
        _MODEL_PATH.unlink(missing_ok=True)
        raise SpeakerError(
            f"Failed to download wespeaker model: {exc}",
            code="SPEAKER_MODEL_DOWNLOAD",
            recoverable=True,
        ) from exc

    return _MODEL_PATH


class SpeakerEmbedder:
    """Extract speaker embeddings using wespeaker CAM++ via ONNX Runtime."""

    def __init__(self) -> None:
        self._session = None

    def _load(self):
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            raise SpeakerError(
                "onnxruntime is not installed. "
                "Install it with: uv pip install after-meeting[speaker]",
                code="SPEAKER_DEPENDENCY_MISSING",
                recoverable=False,
            )

        model_path = _ensure_model()
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(model_path), providers=providers)

    def embed(
        self,
        audio_path: Path,
        start: float = 0.0,
        end: float | None = None,
    ) -> np.ndarray:
        """Extract a speaker embedding from an audio segment. Returns L2-normalized 1-D array."""
        self._load()

        if end is None:
            with wave.open(str(audio_path), "r") as f:
                end = f.getnframes() / f.getframerate()

        audio = _load_audio_segment(audio_path, start, end)
        fbank = _compute_fbank(audio)
        # CAM++ expects (batch, num_frames, num_mel_bins)
        fbank_input = fbank.reshape(1, fbank.shape[0], fbank.shape[1])

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: fbank_input})
        embedding = outputs[0].flatten()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def embed_speaker(
        self,
        audio_path: Path,
        segments: list[tuple[float, float]],
    ) -> np.ndarray:
        """Compute a representative embedding by averaging embeddings from top utterances."""
        embeddings = []
        for start, end in segments:
            try:
                emb = self.embed(audio_path, start=start, end=end)
                embeddings.append(emb)
            except Exception:
                logger.warning("Failed to embed segment %.1f-%.1f, skipping", start, end)
                continue

        if not embeddings:
            raise SpeakerError(
                "Could not extract any speaker embeddings.",
                code="SPEAKER_EMBED",
                recoverable=False,
            )

        avg = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        return avg
