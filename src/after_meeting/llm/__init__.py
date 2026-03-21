"""LLM provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from after_meeting.llm.base import LLMProvider

_PROVIDERS: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    """Register an LLM provider class by name."""
    _PROVIDERS[name] = cls


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Instantiate a registered LLM provider by name."""
    _ensure_builtins()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: {name!r}. Available: {list(_PROVIDERS)}"
        )
    return _PROVIDERS[name](**kwargs)


def _ensure_builtins() -> None:
    """Import built-in providers so they self-register."""
    if _PROVIDERS:
        return
    try:
        import after_meeting.llm.codex  # noqa: F401
    except ImportError:
        pass
