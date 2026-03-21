"""Tests for Codex direct HTTP LLM provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from after_meeting.errors import LLMError
from after_meeting.llm.base import LLMProvider
from after_meeting.llm.codex import CodexProvider


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _fake_auth() -> dict:
    return {
        "tokens": {
            "access_token": "fake_access_token",
            "account_id": "fake_account_id",
            "refresh_token": "fake_refresh_token",
        }
    }


def _fake_sse_lines(text: str) -> list[str]:
    """Build SSE lines that the provider would parse."""
    return [
        f'data: {{"type":"response.output_text.delta","delta":"{text}"}}',
        'data: {"type":"response.completed","response":{"model":"gpt-5.4"}}',
        "data: [DONE]",
    ]


class FakeStreamResponse:
    """Fake httpx streaming response."""

    def __init__(self, status_code: int = 200, text: str = "Hello"):
        self.status_code = status_code
        self._lines = _fake_sse_lines(text) if status_code == 200 else []

    def iter_lines(self):
        yield from self._lines

    def read(self):
        return b'{"detail":"error"}'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ------------------------------------------------------------------
# Protocol
# ------------------------------------------------------------------
class TestProtocol:
    @patch.object(CodexProvider, "_load_auth", return_value=_fake_auth())
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_isinstance_check(self, *_) -> None:
        provider = CodexProvider()
        assert isinstance(provider, LLMProvider)


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------
class TestInit:
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_missing_auth_file(self, _) -> None:
        with patch("after_meeting.llm.codex._CODEX_AUTH_PATH", Path("/nonexistent")):
            with pytest.raises(LLMError, match="auth file not found"):
                CodexProvider()

    @patch.object(CodexProvider, "_load_auth", return_value=_fake_auth())
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_loads_auth(self, *_) -> None:
        provider = CodexProvider()
        assert provider._auth["tokens"]["access_token"] == "fake_access_token"


# ------------------------------------------------------------------
# Complete
# ------------------------------------------------------------------
class TestComplete:
    @patch.object(CodexProvider, "_load_auth", return_value=_fake_auth())
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_success(self, *_) -> None:
        provider = CodexProvider()

        with patch("httpx.stream", return_value=FakeStreamResponse(200, "42")):
            result = provider.complete("1+1?")

        assert result == "42"

    @patch.object(CodexProvider, "_load_auth", return_value=_fake_auth())
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_korean_response(self, *_) -> None:
        provider = CodexProvider()

        with patch("httpx.stream", return_value=FakeStreamResponse(200, "안녕하세요")):
            result = provider.complete("인사해줘")

        assert result == "안녕하세요"

    @patch.object(CodexProvider, "_load_auth", return_value=_fake_auth())
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_api_error(self, *_) -> None:
        provider = CodexProvider()

        with patch("httpx.stream", return_value=FakeStreamResponse(500)):
            with pytest.raises(LLMError, match="Codex API returned 500"):
                provider.complete("test")

    @patch.object(CodexProvider, "_load_auth", return_value=_fake_auth())
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_timeout(self, *_) -> None:
        provider = CodexProvider()

        import httpx as _httpx
        with patch("httpx.stream", side_effect=_httpx.TimeoutException("timeout")):
            with pytest.raises(LLMError, match="timed out"):
                provider.complete("test")

    @patch.object(CodexProvider, "_load_auth", return_value=_fake_auth())
    @patch.object(CodexProvider, "_load_model", return_value="gpt-5.4")
    def test_empty_response(self, *_) -> None:
        provider = CodexProvider()

        fake = FakeStreamResponse(200)
        fake._lines = ['data: {"type":"response.completed"}', "data: [DONE]"]

        with patch("httpx.stream", return_value=fake):
            with pytest.raises(LLMError, match="empty output"):
                provider.complete("test")


# ------------------------------------------------------------------
# SSE parsing
# ------------------------------------------------------------------
class TestCollectSSE:
    def test_multi_delta(self) -> None:
        lines = [
            'data: {"type":"response.output_text.delta","delta":"Hello"}',
            'data: {"type":"response.output_text.delta","delta":" World"}',
            "data: [DONE]",
        ]
        resp = MagicMock()
        resp.iter_lines.return_value = lines

        result = CodexProvider._collect_sse(resp)
        assert result == "Hello World"

    def test_ignores_non_data_lines(self) -> None:
        lines = [
            "event: ping",
            ": comment",
            'data: {"type":"response.output_text.delta","delta":"OK"}',
            "data: [DONE]",
        ]
        resp = MagicMock()
        resp.iter_lines.return_value = lines

        result = CodexProvider._collect_sse(resp)
        assert result == "OK"


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
class TestRegistration:
    def test_registered(self) -> None:
        from after_meeting.llm import _PROVIDERS
        assert "codex" in _PROVIDERS
        assert _PROVIDERS["codex"] is CodexProvider
