"""Codex LLM provider — direct HTTP call to ChatGPT Codex Responses API.

Uses the OAuth token from ~/.codex/auth.json (ChatGPT subscription).
No subprocess overhead — calls the streaming SSE endpoint directly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from after_meeting.errors import LLMError
from after_meeting.llm import register

logger = logging.getLogger(__name__)

_CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
_CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"
_CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
_REFRESH_ENDPOINT = "https://auth.openai.com/oauth/token"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_DEFAULT_MODEL = "gpt-5.4"


class CodexProvider:
    """LLM provider that calls the ChatGPT Codex Responses API directly.

    Reads OAuth tokens from ~/.codex/auth.json (managed by `codex login`).
    Uses SSE streaming (required by the endpoint).
    Billed through ChatGPT subscription — no per-token API cost.
    """

    def __init__(self, **kwargs) -> None:
        self._auth = self._load_auth()
        self._model = self._load_model()

    def complete(self, prompt: str, *, _retry: bool = True) -> str:
        access_token = self._auth["tokens"]["access_token"]
        account_id = self._auth["tokens"]["account_id"]

        try:
            with httpx.stream(
                "POST",
                _CODEX_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Account-Id": account_id,
                },
                json={
                    "model": self._model,
                    "instructions": "You are a helpful assistant that outputs valid JSON.",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}],
                        }
                    ],
                    "store": False,
                    "stream": True,
                },
                timeout=300,
            ) as resp:
                if resp.status_code == 401:
                    if not _retry:
                        raise LLMError(
                            "Codex API returned 401 after token refresh.",
                            code="LLM_AUTH",
                            recoverable=False,
                        )
                    self._refresh_token()
                    return self.complete(prompt, _retry=False)

                if resp.status_code != 200:
                    body = resp.read().decode()
                    raise LLMError(
                        f"Codex API returned {resp.status_code}: {body}",
                        code="LLM_API",
                        recoverable=resp.status_code in (429, 500, 502, 503),
                    )

                return self._collect_sse(resp)

        except LLMError:
            raise
        except httpx.TimeoutException as exc:
            raise LLMError(
                "Codex API request timed out after 300 seconds.",
                code="LLM_TIMEOUT",
                recoverable=True,
            ) from exc
        except Exception as exc:
            raise LLMError(
                f"Codex API call failed: {exc}",
                code="LLM_UNKNOWN",
                recoverable=False,
            ) from exc

    # ------------------------------------------------------------------
    # SSE response collection
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_sse(resp: httpx.Response) -> str:
        """Read SSE stream and collect output text deltas."""
        full_text = ""
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                full_text += event.get("delta", "")

        if not full_text.strip():
            raise LLMError(
                "Codex API returned empty output.",
                code="LLM_PARSE",
                recoverable=True,
            )
        return full_text.strip()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    @staticmethod
    def _load_auth() -> dict:
        """Read OAuth tokens from ~/.codex/auth.json."""
        if not _CODEX_AUTH_PATH.exists():
            raise LLMError(
                f"Codex auth file not found at {_CODEX_AUTH_PATH}. "
                "Run `codex login` first.",
                code="LLM_CONFIG",
                recoverable=False,
            )
        try:
            data = json.loads(_CODEX_AUTH_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LLMError(
                f"Failed to read {_CODEX_AUTH_PATH}: {exc}",
                code="LLM_CONFIG",
                recoverable=False,
            ) from exc

        tokens = data.get("tokens", {})
        if not tokens.get("access_token") or not tokens.get("account_id"):
            raise LLMError(
                "Codex auth.json missing access_token or account_id. "
                "Run `codex login` to re-authenticate.",
                code="LLM_CONFIG",
                recoverable=False,
            )
        return data

    def _refresh_token(self) -> None:
        """Refresh the access token using the refresh token."""
        refresh_token = self._auth.get("tokens", {}).get("refresh_token")
        if not refresh_token:
            raise LLMError(
                "No refresh token available. Run `codex login`.",
                code="LLM_CONFIG",
                recoverable=False,
            )

        try:
            resp = httpx.post(
                _REFRESH_ENDPOINT,
                json={
                    "grant_type": "refresh_token",
                    "client_id": _CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                raise LLMError(
                    f"Token refresh failed ({resp.status_code}): {resp.text}",
                    code="LLM_CONFIG",
                    recoverable=False,
                )

            new_tokens = resp.json()
            self._auth["tokens"]["access_token"] = new_tokens["access_token"]
            if "refresh_token" in new_tokens:
                self._auth["tokens"]["refresh_token"] = new_tokens["refresh_token"]

            # Persist updated tokens
            _CODEX_AUTH_PATH.write_text(
                json.dumps(self._auth, indent=2), encoding="utf-8"
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"Token refresh failed: {exc}. Run `codex login`.",
                code="LLM_CONFIG",
                recoverable=False,
            ) from exc

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model() -> str:
        """Read model from ~/.codex/config.toml if available."""
        if not _CODEX_CONFIG_PATH.exists():
            return _DEFAULT_MODEL
        try:
            import tomllib
            data = tomllib.loads(_CODEX_CONFIG_PATH.read_text(encoding="utf-8"))
            return data.get("model", _DEFAULT_MODEL)
        except Exception:
            return _DEFAULT_MODEL


register("codex", CodexProvider)
