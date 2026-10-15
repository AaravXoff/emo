"""Local Ollama chat client; it does not use LM Studio's API."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen2.5:0.5b") -> None:
        self.host = (host or "http://127.0.0.1:11434").rstrip("/")
        self.model = model or "qwen2.5:0.5b"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:
                raise RuntimeError("Ollama's Python client is missing. Install requirements.txt.") from exc
            self._client = ollama.Client(host=self.host)
        return self._client

    @staticmethod
    def _get_field(value, key, default=None):
        if isinstance(value, dict):
            return value.get(key, default)
        return getattr(value, key, default)

    def check(self) -> tuple[bool, str]:
        try:
            response = self._get_client().list()
            models = self._get_field(response, "models", []) or []
            names = [
                str(self._get_field(item, "name", "") or self._get_field(item, "model", ""))
                for item in models
            ]
            selected = any(
                name == self.model or name.split(":")[0] == self.model.split(":")[0]
                for name in names
            )
            if not names:
                return False, "Ollama is reachable but has no model downloaded; run ollama pull " + self.model
            if not selected:
                return False, "Ollama is running but " + self.model + " is missing; run ollama pull " + self.model
            return True, "Ollama ready (" + self.model + ")"
        except Exception as exc:
            return False, f"Ollama unavailable at {self.host}: {type(exc).__name__}: {exc}"

    def complete(self, question: str, response_language: str = "en",
                 history: list[dict] | None = None) -> str:
        try:
            client = self._get_client()
            language = response_language or "en"
            system = (
                "You are EMO, a friendly compact desktop robot assistant. Answer accurately in 1-4 short sentences. "
                f"Reply in language code {language}, unless the user explicitly asks for another language. "
                "Never claim you performed real-world actions unless a tool actually did so. "
                "When unsure, clearly say so. Keep answers short and spoken-word friendly."
            )
            messages = [{"role": "system", "content": system}]
            if history:
                messages.extend(history[-4:])
            messages.append({"role": "user", "content": (question or "")[:4000]})
            response = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.45, "num_predict": 160},
                keep_alive=0,
            )
            message = self._get_field(response, "message", {})
            content = str(self._get_field(message, "content", "") or "").strip()
            if not content:
                raise RuntimeError("Ollama returned an empty response")
            return content[:5000]
        except Exception as exc:
            log.warning("Ollama request failed: %s", exc)
            raise RuntimeError(
                f"Ollama could not answer. Check that the service is running at {self.host} "
                f"and that the model ollama pull {self.model} has completed."
            ) from exc
