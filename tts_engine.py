"""Speech output with optional Chatterbox and a lightweight espeak-ng fallback."""
from __future__ import annotations
import logging
import os
import shutil
import subprocess
import tempfile
import wave

log = logging.getLogger(__name__)

ESPEAK_VOICES = {
    "en": "en-us", "hi": "hi", "pa": "pa", "es": "es", "fr": "fr",
    "de": "de", "it": "it", "pt": "pt", "ru": "ru", "zh": "zh",
    "ja": "ja", "ko": "ko", "ar": "ar", "ur": "ur",
}


class SpeechEngine:
    def __init__(self, engine: str = "auto", status=None) -> None:
        self.engine = engine
        self.status = status
        self._english_model = None
        self._multilingual_model = None
        self.last_engine = "none"
        self._chatterbox_failed = False
        self._player = shutil.which("aplay")
        self._memory_mb = self._total_memory_mb()

    @staticmethod
    def _total_memory_mb() -> int:
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                first = handle.readline()
            return int(first.split()[1]) // 1024
        except Exception:
            return 0

    def check(self) -> tuple[bool, str]:
        if self.engine == "espeak":
            available = bool(shutil.which("espeak-ng") or shutil.which("espeak"))
            return available, "espeak-ng is available" if available else "Install espeak-ng"
        if self.engine in {"auto", "chatterbox"}:
            if self._memory_mb and self._memory_mb < 1800:
                if shutil.which("espeak-ng") or shutil.which("espeak"):
                    return True, f"Low-memory device ({self._memory_mb} MB); using espeak fallback to avoid loading Chatterbox"
            try:
                import chatterbox  # noqa: F401
                return True, "Chatterbox package found; model loads on first response"
            except Exception:
                available = bool(shutil.which("espeak-ng") or shutil.which("espeak"))
                return available, "Chatterbox unavailable; espeak fallback available" if available else "No usable TTS engine installed"
        return bool(shutil.which("espeak-ng") or shutil.which("espeak")), "Using espeak fallback"

    def speak(self, text: str, language: str = "en") -> bool:
        text = " ".join((text or "").split())
        if not text:
            return False
        # On a Pi 3B+ with roughly 1 GB RAM, don't attempt to load a neural TTS model.
        can_try_chatterbox = self.engine in {"auto", "chatterbox"} and not self._chatterbox_failed
        if self._memory_mb and self._memory_mb < 1800:
            can_try_chatterbox = self.engine == "chatterbox"  # explicit opt-in only on low-RAM hardware
        if self.engine == "espeak":
            can_try_chatterbox = False
        if can_try_chatterbox and self._speak_chatterbox(text, language):
            self.last_engine = "chatterbox"
            return True
        if self._speak_espeak(text, language):
            self.last_engine = "espeak-ng"
            return True
        self.last_engine = "console"
        print(f"EMO: {text}")
        return False

    def _speak_chatterbox(self, text: str, language: str) -> bool:
        try:
            import numpy as np
            import torch
            if language == "en":
                if self._english_model is None:
                    from chatterbox.tts import ChatterboxTTS
                    self._english_model = ChatterboxTTS.from_pretrained(device="cpu")
                model = self._english_model
                audio = model.generate(text)
            else:
                if self._multilingual_model is None:
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    self._multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
                model = self._multilingual_model
                supported = model.get_supported_languages()
                if language not in supported:
                    log.info("Chatterbox has no configured language ID %s; using fallback voice.", language)
                    return False
                audio = model.generate(text, language_id=language)
            data = audio.detach().cpu().float().squeeze().numpy()
            data = np.nan_to_num(data).clip(-1.0, 1.0)
            pcm = (data * 32767).astype("<i2", copy=False).tobytes()
            fd, path = tempfile.mkstemp(prefix="emo-voice-", suffix=".wav")
            os.close(fd)
            try:
                with wave.open(path, "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(int(model.sr))
                    output.writeframes(pcm)
                if not self._player:
                    log.warning("Chatterbox audio generated, but aplay is unavailable.")
                    return False
                subprocess.run([self._player, "-q", path], check=True, timeout=90)
                return True
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        except Exception as exc:
            self._chatterbox_failed = True
            log.warning("Chatterbox TTS failed; switching to lightweight voice: %s", exc)
            return False

    def _speak_espeak(self, text: str, language: str) -> bool:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if not executable:
            return False
        voice = ESPEAK_VOICES.get(language, "en-us")
        candidates = [voice] if voice == "en-us" else [voice, "en-us"]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [executable, "-v", candidate, "-s", "165", text],
                    check=False, timeout=45, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if result.returncode == 0:
                    if candidate != voice:
                        log.info("Voice %s is unavailable; using en-us fallback.", voice)
                    return True
                log.debug("espeak voice %s exited with status %s", candidate, result.returncode)
            except Exception as exc:
                log.warning("espeak playback failed for voice %s: %s", candidate, exc)
        return False
