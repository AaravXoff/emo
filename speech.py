"""Local Whisper speech recognition with a language-aware English translation stage."""
from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
import threading

from translation import OfflineTranslator

log = logging.getLogger(__name__)


@dataclass
class SpeechResult:
    text: str
    language: str
    english_text: str
    translation_method: str


class WhisperRecognizer:
    def __init__(self, model_name: str, translation_model_name: str, download_root: Path,
                 translator: OfflineTranslator, status=None) -> None:
        self.model_name = model_name
        self.translation_model_name = translation_model_name
        self.download_root = Path(download_root).expanduser()
        self.translator = translator
        self.status = status
        self._model = None
        self._loaded_name = ""
        self._lock = threading.RLock()

    @staticmethod
    def dependency_check() -> tuple[bool, str]:
        try:
            import whisper  # noqa: F401
            return True, "OpenAI Whisper Python package is installed"
        except Exception as exc:
            return False, f"Whisper unavailable: {exc}"

    def preload(self) -> None:
        self._get_model(self.model_name)

    def _get_model(self, name: str):
        with self._lock:
            if self._model is not None and self._loaded_name == name:
                return self._model
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "Whisper is not installed. Review README.md for the Pi-compatible setup before installing openai-whisper."
                ) from exc
            self.download_root.mkdir(parents=True, exist_ok=True)
            if self.status is not None:
                self.status.set_stage("Loading Whisper", f"Loading the {name} speech model; first use may download model weights")
                self.status.set_boot_step("Whisper", "running", f"Loading {name}")
            self._model = None
            self._loaded_name = ""
            self._model = whisper.load_model(name, download_root=str(self.download_root), device="cpu")
            self._loaded_name = name
            if self.status is not None:
                self.status.set_boot_step("Whisper", "ok", f"{name} model loaded in CPU mode")
            return self._model

    def transcribe(self, audio_path: str | Path) -> SpeechResult:
        model = self._get_model(self.model_name)
        try:
            result = model.transcribe(str(audio_path), task="transcribe", fp16=False, verbose=False)
        except TypeError:
            # Compatibility with earlier Whisper builds that don't accept verbose here.
            result = model.transcribe(str(audio_path), task="transcribe", fp16=False)
        raw_text = str(result.get("text", "")).strip()
        language = str(result.get("language", "en") or "en").lower()
        if not raw_text:
            return SpeechResult("", language, "", "none")
        if language == "en":
            return SpeechResult(raw_text, language, raw_text, "already-english")

        # Argos is fully offline and is tried first. It also works with Whisper Turbo,
        # which is explicitly not trained for Whisper's built-in translation task.
        translated = self.translator.translate(raw_text, language, "en")
        if translated:
            return SpeechResult(raw_text, language, translated, "argos-offline")

        if self.model_name != "turbo":
            try:
                translated_result = model.transcribe(str(audio_path), task="translate", fp16=False, verbose=False)
            except TypeError:
                translated_result = model.transcribe(str(audio_path), task="translate", fp16=False)
            translated_text = str(translated_result.get("text", "")).strip()
            if translated_text:
                return SpeechResult(raw_text, language, translated_text, f"whisper-{self.model_name}")

        # A turbo model will return the source language even if task='translate' is passed.
        # Keep the source text rather than pretending it has been translated.
        log.warning("No offline %s -> English translation pack is installed; continuing with source text.", language)
        return SpeechResult(raw_text, language, raw_text, "translation-pack-missing")
