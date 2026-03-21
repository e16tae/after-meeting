"""Error types and exit codes for the after-meeting CLI."""

from __future__ import annotations

# Exit codes
EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_STT_ERROR = 2
EXIT_LLM_ERROR = 3
EXIT_RENDER_ERROR = 4
EXIT_CONFIG_ERROR = 5


class AfterMeetingError(Exception):
    """Base exception with an associated exit code."""

    exit_code: int = 1

    def __init__(self, message: str, *, code: str = "UNKNOWN", recoverable: bool = False):
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "recoverable": self.recoverable,
        }


class InputError(AfterMeetingError):
    exit_code = EXIT_INPUT_ERROR


class STTError(AfterMeetingError):
    exit_code = EXIT_STT_ERROR


class LLMError(AfterMeetingError):
    exit_code = EXIT_LLM_ERROR


class RenderError(AfterMeetingError):
    exit_code = EXIT_RENDER_ERROR


class ConfigError(AfterMeetingError):
    exit_code = EXIT_CONFIG_ERROR


EXIT_AUDIO_ERROR = 6
EXIT_SPEAKER_ERROR = 7


class AudioError(AfterMeetingError):
    exit_code = EXIT_AUDIO_ERROR


class SpeakerError(AfterMeetingError):
    exit_code = EXIT_SPEAKER_ERROR
