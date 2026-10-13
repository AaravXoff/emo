"""LM Studio client using its local OpenAI-compatible Chat Completions endpoint."""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from translation import LANGUAGE_NAMES


class LMStudioClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 35) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds

    def check(self) -> tuple[bool, str]:
        try:
            req = urllib.request.Request(self.base_url + "/models", headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=min(self.timeout, 4)) as response:
                data = json.loads(response.read().decode("utf-8"))
            models = [item.get("id", "") for item in data.get("data", [])]
            if not models:
                return True, "Server reachable; no model listed"
            selected = self.model in models
            detail = "Configured model ID listed" if selected else "Server reachable; check LM_STUDIO_MODEL against /v1/models"
            return True, detail
        except Exception as exc:
            return False, f"Unavailable: {type(exc).__name__}: {exc}"

    def complete(self, question: str, response_language: str = "en", history: list[dict] | None = None) -> str:
        language_name = LANGUAGE_NAMES.get(response_language, response_language or "the speaker's language")
        system = (
            "You are EMO, a friendly compact desktop robot assistant. Answer accurately in 1-4 short sentences. "
            f"The user speaks {language_name}. Reply in {language_name}, unless the user explicitly asks for another language. "
            "Do not claim you performed real-world actions unless a tool actually did so. If unsure, say so clearly. "
            "Avoid markdown tables and overly long answers."
        )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": question[:6000]})
        payload = {"model": self.model, "messages": messages, "temperature": 0.55, "max_tokens": 220, "stream": False}
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            result = data["choices"][0]["message"]["content"]
            if isinstance(result, list):
                result = " ".join(part.get("text", "") for part in result if isinstance(part, dict))
            result = str(result).strip()
            if not result:
                raise ValueError("LM Studio returned an empty answer")
            return result[:8000]
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "LM Studio is unreachable. Start its local server and set LM_STUDIO_BASE_URL to a Pi-accessible URL."
            ) from exc
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected LM Studio response: {exc}") from exc
