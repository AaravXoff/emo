"""Local Whisper speech recognition with a C++ backend for low-memory Raspberry Pi boards."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
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
    def __init__(
        self,
        model_name: str,
        translation_model_name: str,
        download_root: Path,
        translator: OfflineTranslator,
        status=None,
        backend: str = "auto",
        cpp_binary: Path | str | None = None,
        cpp_model: Path | str | None = None,
        cpp_threads: int = 2,
    ) -> None:
        self.model_name = model_name
        self.translation_model_name = translation_model_name
        self.download_root = Path(download_root).expanduser()
        self.translator = translator
        self.status = status
        self.backend = (backend or "auto").strip().lower()
        default_root = Path.home() / "whisper.cpp"
        self.cpp_binary = Path(cpp_binary or (default_root / "build" / "bin" / "whisper-cli")).expanduser()
        self.cpp_model = Path(cpp_model or (default_root / "models" / "ggml-tiny.bin")).expanduser()
        self.cpp_threads = max(1, int(cpp_threads))
        self._model = None
        self._loaded_name = ""
        self._lock = threading.RLock()
        self._resolved_backend: str | None = None

    def _cpp_available(self) -> bool:
        return (
            self.cpp_binary.is_file()
            and os.access(self.cpp_binary, os.X_OK)
            and self.cpp_model.is_file()
        )

    @staticmethod
    def _python_whisper_available() -> bool:
        try:
            return importlib.util.find_spec("whisper") is not None
        except (ImportError, ValueError):
            return False

    def _select_backend(self) -> str:
        if self._resolved_backend:
            return self._resolved_backend
        if self.backend in {"cpp", "whisper.cpp", "whisper-cpp"}:
            self._resolved_backend = "cpp"
        elif self.backend in {"python", "openai-whisper"}:
            self._resolved_backend = "python"
        elif self.backend == "auto":
            if self._cpp_available():
                self._resolved_backend = "cpp"
            elif self._python_whisper_available():
                self._resolved_backend = "python"
            else:
                self._resolved_backend = "none"
        else:
            self._resolved_backend = "none"
        return self._resolved_backend

    def dependency_check(self) -> tuple[bool, str]:
        selected = self._select_backend()
        if selected == "cpp":
            if not self.cpp_binary.is_file() or not os.access(self.cpp_binary, os.X_OK):
                return False, f"Whisper.cpp executable not found or not executable: {self.cpp_binary}"
            if not self.cpp_model.is_file():
                return False, f"Whisper.cpp model missing: {self.cpp_model}"
            return True, f"Whisper.cpp ready ({self.cpp_model.name}, {self.cpp_threads} CPU threads)"
        if selected == "python":
            return True, "OpenAI Whisper Python package is installed; model loads on first use"
        if self.backend == "auto":
            return False, (
                "No speech backend found. Recommended on Pi 3B+: run bash setup_whisper_cpp.sh. "
                "See README.md for the optional Python Whisper backend."
            )
        return False, f"Unsupported WHISPER_BACKEND={self.backend!r}; use auto, cpp, or python"

    def preload(self) -> None:
        selected = self._select_backend()
        if selected == "cpp":
            ok, detail = self.dependency_check()
            if not ok:
                raise RuntimeError(detail)
            return
        if selected == "python":
            self._get_model(self.model_name)
            return
        raise RuntimeError("No speech recognition backend is installed; see README.md.")

    def _get_model(self, name: str):
        with self._lock:
            if self._model is not None and self._loaded_name == name:
                return self._model
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "Python Whisper is not installed. On a Pi 3B+ use setup_whisper_cpp.sh instead."
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

    def _run_cpp(self, audio_path: str | Path, translate_to_english: bool = False,
                 language: str = "auto") -> tuple[str, str]:
        if not self.cpp_binary.is_file() or not os.access(self.cpp_binary, os.X_OK):
            raise RuntimeError(f"Whisper.cpp executable not found: {self.cpp_binary}")
        if not self.cpp_model.is_file():
            raise RuntimeError(f"Whisper.cpp model not found: {self.cpp_model}")

        with tempfile.TemporaryDirectory(prefix="emo-whisper-") as temp_dir:
            output_prefix = Path(temp_dir) / "result"
            command = [
                str(self.cpp_binary),
                "--model", str(self.cpp_model),
                "--file", str(Path(audio_path)),
                "--language", language or "auto",
                "--threads", str(self.cpp_threads),
                "--output-json-full",
                "--output-file", str(output_prefix),
                "--no-prints",
            ]
            if translate_to_english:
                command.append("--translate")
            try:
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True, timeout=180
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Whisper.cpp exceeded its 180-second processing timeout.") from exc
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or exc.stdout or str(exc)).strip()[-1200:]
                raise RuntimeError(f"Whisper.cpp failed: {detail}") from exc

            json_path = output_prefix.with_suffix(".json")
            if not json_path.is_file():
                details = (result.stderr or result.stdout or "no JSON output created").strip()[-1200:]
                raise RuntimeError(f"Whisper.cpp did not create its JSON output: {details}")
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Could not read Whisper.cpp JSON output: {exc}") from exc

            result_info = payload.get("result") or {}
            params = payload.get("params") or {}
            detected_language = str(
                result_info.get("language") or params.get("language") or language or "en"
            ).lower()
            if detected_language in {"auto", ""}:
                detected_language = "en"
            segments = payload.get("transcription") or []
            transcript = "".join(
                str(segment.get("text", "")) for segment in segments if isinstance(segment, dict)
            ).strip()
            return transcript, detected_language

    def transcribe(self, audio_path: str | Path) -> SpeechResult:
        if self.status is not None:
            self.status.set_stage("Recognizing speech", "Running local speech recognition")
        selected = self._select_backend()
        if selected == "cpp":
            raw_text, language = self._run_cpp(audio_path, translate_to_english=False, language="auto")
        elif selected == "python":
            model = self._get_model(self.model_name)
            try:
                result = model.transcribe(str(audio_path), task="transcribe", fp16=False, verbose=False)
            except TypeError:
                result = model.transcribe(str(audio_path), task="transcribe", fp16=False)
            raw_text = str(result.get("text", "")).strip()
            language = str(result.get("language", "en") or "en").lower()
        else:
            ok, detail = self.dependency_check()
            raise RuntimeError(detail if not ok else "No speech backend available")

        if not raw_text:
            return SpeechResult("", language, "", "none")
        if language == "en":
            return SpeechResult(raw_text, language, raw_text, "already-english")

        translated = self.translator.translate(raw_text, language, "en")
        if translated:
            return SpeechResult(raw_text, language, translated, "argos-offline")

        if selected == "cpp":
            try:
                translated_text, _translated_language = self._run_cpp(
                    audio_path, translate_to_english=True, language=language
                )
            except Exception as exc:
                log.warning("Whisper.cpp translation failed: %s", exc)
                translated_text = ""
            if translated_text:
                return SpeechResult(raw_text, language, translated_text, "whisper.cpp-translate")
        elif selected == "python" and self.model_name != "turbo":
            model = self._get_model(self.model_name)
            try:
                translated_result = model.transcribe(str(audio_path), task="translate", fp16=False, verbose=False)
            except TypeError:
                translated_result = model.transcribe(str(audio_path), task="translate", fp16=False)
            translated_text = str(translated_result.get("text", "")).strip()
            if translated_text:
                return SpeechResult(raw_text, language, translated_text, f"whisper-{self.model_name}")

        log.warning("No offline %s -> English translation was available; using source text.", language)
        return SpeechResult(raw_text, language, raw_text, "translation-pack-missing")
