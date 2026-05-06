# Copyright (c) 2026 Stable-Agent Contributors
# SPDX-License-Identifier: MIT
"""Minimal HTTP client for a local Ollama server.

This module wraps the Ollama REST API (https://github.com/ollama/ollama) for use
by the synthetic dataset generator, which drives ``llama3.1:8b`` for both
user-side and assistant-side turn generation.

Example
-------
    >>> from ollama_client import OllamaClient, OllamaConfig, ChatMessage
    >>> client = OllamaClient(OllamaConfig(model="llama3.1:8b", seed=42))
    >>> if client.health():
    ...     reply = client.chat([
    ...         ChatMessage(role="system", content="You are a concise assistant."),
    ...         ChatMessage(role="user", content="Say hi in one word."),
    ...     ])
    ...     # ``reply`` is a string; ``OllamaError`` is raised on failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError, RequestException, Timeout


class OllamaError(Exception):
    """Raised when an Ollama call fails.

    Covers connection errors, timeouts, non-200 HTTP responses, and malformed
    or unexpected response payloads.
    """


@dataclass
class ChatMessage:
    """A single chat message in the Ollama ``/api/chat`` schema."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class OllamaConfig:
    """Configuration for :class:`OllamaClient`."""

    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    timeout: float = 120.0
    seed: Optional[int] = None  # for deterministic generation


def _excerpt(text: str, limit: int = 500) -> str:
    """Return the first ``limit`` characters of ``text`` for error messages."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit]


class OllamaClient:
    """Thin synchronous client around the Ollama HTTP API."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config: OllamaConfig = config if config is not None else OllamaConfig()
        self._session: requests.Session = requests.Session()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _url(self, path: str) -> str:
        host = self.config.host.rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return f"{host}{path}"

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._url(path)
        try:
            response = self._session.post(
                url, json=payload, timeout=self.config.timeout
            )
        except Timeout as exc:
            raise OllamaError(
                f"Ollama request to {url} timed out after "
                f"{self.config.timeout}s: {exc}"
            ) from exc
        except RequestsConnectionError as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {url}: {exc}"
            ) from exc
        except RequestException as exc:
            raise OllamaError(
                f"HTTP error while contacting Ollama at {url}: {exc}"
            ) from exc

        return self._parse_response(response, url)

    def _get_json(self, path: str) -> dict[str, Any]:
        url = self._url(path)
        try:
            response = self._session.get(url, timeout=self.config.timeout)
        except Timeout as exc:
            raise OllamaError(
                f"Ollama request to {url} timed out after "
                f"{self.config.timeout}s: {exc}"
            ) from exc
        except RequestsConnectionError as exc:
            raise OllamaError(
                f"Could not connect to Ollama at {url}: {exc}"
            ) from exc
        except RequestException as exc:
            raise OllamaError(
                f"HTTP error while contacting Ollama at {url}: {exc}"
            ) from exc

        return self._parse_response(response, url)

    @staticmethod
    def _parse_response(
        response: requests.Response, url: str
    ) -> dict[str, Any]:
        body_text = response.text or ""
        if response.status_code != 200:
            raise OllamaError(
                f"Ollama call to {url} returned HTTP {response.status_code}: "
                f"{_excerpt(body_text)}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(
                f"Ollama response from {url} was not valid JSON "
                f"(HTTP {response.status_code}): {_excerpt(body_text)}"
            ) from exc
        if not isinstance(data, dict):
            raise OllamaError(
                f"Ollama response from {url} was not a JSON object: "
                f"{_excerpt(body_text)}"
            )
        return data

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> str:
        """Send ``messages`` to ``/api/chat`` and return the assistant content.

        Per-call ``temperature`` and ``seed`` override the values from
        :class:`OllamaConfig` when provided.

        Raises:
            OllamaError: If the HTTP call fails, the server returns a non-200
                response, the response body is not valid JSON, or the expected
                ``message.content`` field is missing.
        """
        effective_temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        effective_seed = seed if seed is not None else self.config.seed

        options: dict[str, Any] = {"temperature": float(effective_temperature)}
        if effective_seed is not None:
            options["seed"] = int(effective_seed)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "stream": False,
            "options": options,
        }

        data = self._post_json("/api/chat", payload)

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError(
                "Ollama chat response missing 'message' object: "
                f"{_excerpt(str(data))}"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError(
                "Ollama chat response missing 'message.content' string: "
                f"{_excerpt(str(data))}"
            )
        return content

    def list_models(self) -> List[str]:
        """Return installed model names from ``GET /api/tags``.

        Raises:
            OllamaError: On any HTTP/connection/parse failure or unexpected
                payload shape.
        """
        data = self._get_json("/api/tags")
        models = data.get("models")
        if not isinstance(models, list):
            raise OllamaError(
                "Ollama /api/tags response missing 'models' list: "
                f"{_excerpt(str(data))}"
            )
        names: List[str] = []
        for entry in models:
            if not isinstance(entry, dict):
                raise OllamaError(
                    "Ollama /api/tags entry was not an object: "
                    f"{_excerpt(str(entry))}"
                )
            name = entry.get("name")
            if not isinstance(name, str):
                raise OllamaError(
                    "Ollama /api/tags entry missing 'name' string: "
                    f"{_excerpt(str(entry))}"
                )
            names.append(name)
        return names

    def health(self) -> bool:
        """Return True if the server is reachable and the model is installed.

        Never raises: any exception is treated as an unhealthy server.

        Matching is lenient: if ``config.model`` has no ``:tag`` portion
        (e.g. ``"llama3.1"``), an installed ``llama3.1:latest`` (or any
        ``llama3.1:<tag>``) also counts as a match. Otherwise the full
        ``name:tag`` form must be present in the installed list.
        """
        try:
            installed = self.list_models()
        except Exception:
            return False

        target = self.config.model
        if target in installed:
            return True

        if ":" not in target:
            prefix = target + ":"
            for name in installed:
                if name == target or name.startswith(prefix):
                    return True
            return False

        base, _, _tag = target.partition(":")
        latest = f"{base}:latest"
        if latest in installed:
            return True

        return False
