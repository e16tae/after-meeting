"""Rendering provider registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from after_meeting.rendering.base import Renderer

_RENDERERS: dict[str, type] = {}


def register(name: str, cls: type) -> None:
    """Register a renderer class by format name."""
    _RENDERERS[name] = cls


def get_renderer(name: str, **kwargs) -> Renderer:
    """Instantiate a registered renderer by format name."""
    _ensure_builtins()
    if name not in _RENDERERS:
        raise ValueError(
            f"Unknown renderer: {name!r}. Available: {list(_RENDERERS)}"
        )
    return _RENDERERS[name](**kwargs)


def _ensure_builtins() -> None:
    """Import built-in renderers so they self-register."""
    if _RENDERERS:
        return
    import after_meeting.rendering.docx_renderer  # noqa: F401
    import after_meeting.rendering.pdf_renderer  # noqa: F401
