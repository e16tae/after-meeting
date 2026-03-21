"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings loaded from environment variables and .env files."""

    model_config = {"env_prefix": "AFTER_MEETING_", "env_file": ".env"}

    # STT
    stt_provider: str = "qwen3"
    stt_model: str = "Qwen/Qwen3-ASR-1.7B"
    stt_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B"

    # LLM
    llm_provider: str = "codex"
    # Rendering
    default_format: str = "docx"
    default_doc_type: str = "minutes"

    # Chunking
    chunk_minutes: int = 55
    overlap_seconds: int = 60
    max_utterances: int = 200

    # Output
    output_dir: str = "."


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
