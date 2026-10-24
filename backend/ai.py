"""Talks to any OpenAI-compatible AI service (OpenAI, Ollama, and others) over plain HTTP."""
from __future__ import annotations

import time
from typing import Any

import httpx

from .config import Settings


class AIError(Exception):
    """An AI call failed. `message` is safe to show to the user."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AIClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self._transport = transport
        self._ready_cache: tuple[float, bool] | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.ai_enabled

    # ---------- low-level ----------
    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        s = self.settings
        try:
            with httpx.Client(transport=self._transport, timeout=timeout) as http:
                response = http.post(
                    f"{s.base_url}{path}",
                    json=payload,
                    headers={"Authorization": f"Bearer {s.api_key}"},
                )
        except httpx.TimeoutException as exc:
            raise AIError(f"{s.provider_label} took too long to respond. Try again in a moment.", 504) from exc
        except httpx.HTTPError as exc:
            where = s.base_url if s.provider in {"ollama", "compatible"} else s.provider_label
            raise AIError(f"Could not reach {where}. {s.setup_hint()}", 503) from exc

        if response.status_code in (401, 403):
            raise AIError(f"{s.provider_label} rejected the API key. Update AI_API_KEY in .env.", 401)
        if response.status_code == 429:
            raise AIError(f"{s.provider_label} is rate-limiting requests or the account has no credit left.", 429)
        if response.status_code == 404:
            raise AIError(f"{s.provider_label} does not know one of the models. {s.setup_hint()}", 502)
        if response.status_code >= 400:
            raise AIError(f"{s.provider_label} returned an error ({response.status_code}).", 502)
        try:
            return response.json()
        except ValueError as exc:
            raise AIError(f"{s.provider_label} sent a response Knowledge Hub could not read.", 502) from exc

    # ---------- public ----------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            raise AIError("AI is not configured on this server.", 503)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            data = self._post(
                "/embeddings",
                {"model": self.settings.embedding_model, "input": batch},
                self.settings.embed_timeout,
            )
            items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
            if len(items) != len(batch):
                raise AIError("The embedding service returned the wrong number of vectors.", 502)
            vectors.extend([item["embedding"] for item in items])
        return vectors

    def chat(self, system: str, user: str, max_tokens: int = 700) -> str:
        if not self.enabled:
            raise AIError("AI is not configured on this server.", 503)
        data = self._post(
            "/chat/completions",
            {
                "model": self.settings.chat_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            self.settings.chat_timeout,
        )
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise AIError(f"{self.settings.provider_label} returned an empty answer.", 502) from exc

    def ready(self) -> bool:
        """Cheap readiness check for the status badge, cached for 30 seconds."""
        if not self.enabled:
            return False
        now = time.monotonic()
        if self._ready_cache and now - self._ready_cache[0] < 30:
            return self._ready_cache[1]
        ok = False
        try:
            with httpx.Client(transport=self._transport, timeout=4.0) as http:
                response = http.get(
                    f"{self.settings.base_url}/models",
                    headers={"Authorization": f"Bearer {self.settings.api_key}"},
                )
            if response.status_code == 200:
                ids = {item.get("id", "") for item in response.json().get("data", [])}
                if self.settings.provider == "ollama":
                    # Ollama lists models as "name:tag"; "nomic-embed-text" is listed as "nomic-embed-text:latest".
                    def has(model: str) -> bool:
                        return model in ids or f"{model}:latest" in ids

                    ok = has(self.settings.chat_model) and has(self.settings.embedding_model)
                else:
                    ok = True
        except (httpx.HTTPError, ValueError):
            ok = False
        self._ready_cache = (now, ok)
        return ok
