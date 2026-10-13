#!/usr/bin/env python3
"""EMO local assistant entry point for Raspberry Pi OS."""
from __future__ import annotations
import logging
import os
import re
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from assistant import EMOAssistant
from config import Config
from hardware import HardwareController, ThermalMonitor
from html_face import HtmlFace
from llm_client import LMStudioClient
from speech import WhisperRecognizer
from state import RobotStatus
from translation import OfflineTranslator
from tts_engine import SpeechEngine

log = logging.getLogger("emo")


def strip_wake_word(text: str, phrases: tuple[str, ...]) -> tuple[bool, str]:
    clean = text.strip()
    folded = clean.casefold()
    for phrase in sorted(phrases, key=len, reverse=True):
        match = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", folded)
        if match:
            remaining = (clean[:match.start()] + " " + clean[match.end():]).strip()
            return True, remaining.lstrip(" ,;:").strip()
    # Handle common speech-recognition variants such as "Hey emu".
    words = clean.split()
    if (
        len(words) >= 2
        and words[0].casefold() in {"hey", "hi", "hello"}
        and words[1].strip(",.!?:;").casefold() in {"emu", "emo"}
    ):
        return True, " ".join(words[2:]).lstrip(" ,;:").strip()
    return False, clean

class EMOApplication:
    def __init__(self) -> None:
        self.config = Config.from_env()
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        self.status = RobotStatus()
        self.face = HtmlFace(status=self.status)
        self.translator = OfflineTranslator()
        self.recorder = None
        self.recognizer = WhisperRecognizer(
            self.config.whisper_model, self.config.whisper_translation_model,
            self.config.whisper_download_root, self.translator, status=self.status,
            backend=self.config.whisper_backend,
            cpp_binary=self.config.whisper_cpp_bin,
            cpp_model=self.config.whisper_cpp_model,
            cpp_threads=self.config.whisper_cpp_threads,
        )
        self.llm = LMStudioClient(
            self.config.lm_studio_base_url, self.config.lm_studio_model,
            self.config.lm_studio_timeout_seconds,
        )
        self.speech = SpeechEngine(self.config.tts_engine, status=self.status)
        self.hardware = HardwareController(self.config.touch_gpio, self.config.servo_gpio, self.status)
        self.thermal = ThermalMonitor(
            self.status, self.config.thermal_warning_c, self.config.thermal_lockout_c,
            self.config.thermal_resume_c,
        )
        self.assistant = None
        self.stop_event = threading.Event()
        self._awake_until = 0.0

    def boot(self) -> None:
        self.face.start_server()
        self.status.set_boot_step("Configuration", "ok", "Environment config loaded")
        self.status.set_stage("Boot sequence", "Initializing local cache, microphone, AI and voice services")

        try:
            # Create cache before the assistant is constructed so the database is ready at boot.
            from response_cache import ResponseCache
            boot_cache = ResponseCache(self.config.cache_db, enabled=self.config.cache_enabled)
            boot_cache.close()
            self.status.set_boot_step("Local cache", "ok", str(self.config.cache_db))
        except Exception as exc:
            self.status.set_boot_step("Local cache", "error", str(exc))

        from audio_io import AudioRecorder
        self.recorder = AudioRecorder(
            self.config.sample_rate, self.config.audio_rms_threshold,
            self.config.max_utterance_seconds, self.config.trailing_silence_seconds,
            status=self.status, input_device=self.config.audio_input_device,
        )
        mic_ok, mic_detail = self.recorder.check_device()
        self.status.set_boot_step("Microphone", "ok" if mic_ok else "warning", mic_detail)

        whisper_ok, whisper_detail = self.recognizer.dependency_check()
        if whisper_ok:
            if self.config.preload_whisper:
                try:
                    self.recognizer.preload()
                    whisper_detail = f"{self.config.whisper_model} model loaded"
                    whisper_status = "ok"
                except Exception as exc:
                    whisper_detail = f"Lazy load will retry: {exc}"
                    whisper_status = "warning"
            else:
                whisper_detail = f"Package present; {self.config.whisper_model} model lazy-loads at first speech"
                whisper_status = "ok"
        else:
            whisper_status = "warning"
        self.status.set_boot_step("Whisper", whisper_status, whisper_detail)

        llm_ok, llm_detail = self.llm.check()
        self.status.set_boot_step("LM Studio", "ok" if llm_ok else "warning", llm_detail)

        tts_ok, tts_detail = self.speech.check()
        self.status.set_boot_step("Speech output", "ok" if tts_ok else "warning", tts_detail)

        hw_ok, hw_detail = self.hardware.start()
        self.status.set_boot_step("Hardware", "ok" if hw_ok else "warning", hw_detail)

        self.thermal.start()
        self.assistant = EMOAssistant(
            self.config, self.status, self.face, self.hardware,
            self.recognizer, self.translator, self.llm, self.speech,
        )
        with self.status.lock:
            self.status.boot_complete = True
        self.face.set_state("idle")
        self.status.set_stage("Listening", "Say a wake phrase followed by a request")
        log.info("EMO boot complete. HTML face is at http://127.0.0.1:8765/")

    def _handle_temp_audio(self, path: str) -> None:
        try:
            if self.status.snapshot()["thermal_lockout"]:
                self.status.set_stage("Cooling down", "Heavy AI inference paused until the Pi cools")
                return
            self.status.set_stage("Recognizing speech", f"Running local Whisper model ({self.config.whisper_model})")
            result = self.recognizer.transcribe(path)
            if not result.text:
                return

            touched_recently = time.monotonic() - self.status.snapshot()["last_touch"] < 20
            if self.config.wake_word_required and time.monotonic() > self._awake_until and not touched_recently:
                heard, remainder = strip_wake_word(result.text, self.config.wake_words)
                english_remainder = result.english_text
                if not heard and result.english_text and result.english_text != result.text:
                    heard, english_remainder = strip_wake_word(result.english_text, self.config.wake_words)
                    if heard:
                        remainder = english_remainder
                if not heard:
                    log.debug("Ignored speech without wake phrase")
                    return
                self._awake_until = time.monotonic() + self.config.wake_timeout_seconds
                if result.english_text and result.english_text != result.text:
                    detected_english_wake, english_remainder = strip_wake_word(result.english_text, self.config.wake_words)
                    if not detected_english_wake:
                        english_remainder = result.english_text
                else:
                    english_remainder = remainder
                if not remainder:
                    self.status.set_stage("Awake", "EMO is awake; say your request")
                    self.face.set_state("listening", duration=1.2)
                    self.speech.speak("I'm listening.", result.language)
                    return
                result = type(result)(remainder, result.language, english_remainder or remainder, result.translation_method)

            if time.monotonic() > self._awake_until:
                self._awake_until = time.monotonic() + self.config.wake_timeout_seconds
            self.status.set_stage("Processing request", "Matching local instructions and checking response cache")
            self.assistant.process(
                result.text, result.language, result.english_text, result.translation_method
            )
            # Require a fresh wake phrase after each answer unless touch was used to wake it.
            if self.config.wake_word_required and not touched_recently:
                self._awake_until = 0.0
        except Exception as exc:
            log.exception("Turn failed")
            with self.status.lock:
                self.status.last_error = str(exc)[:240]
            self.status.set_stage("Recovering", str(exc)[:150])
            self.face.set_state("error", duration=2.0)

    def run(self) -> None:
        self.boot()
        if self.recorder is None:
            raise RuntimeError("Audio recorder did not initialize")
        try:
            while not self.stop_event.is_set():
                if self.status.snapshot()["thermal_lockout"]:
                    self.face.set_state("overheat")
                    self.status.set_stage("Cooling down", "AI inference paused because the Pi is too hot")
                    time.sleep(2.0)
                    continue
                try:
                    self.face.set_state("listening")
                    self.hardware.set_expression("listening")
                    self.status.set_stage("Listening", "Say a wake phrase followed by a request")
                    path = self.recorder.record_utterance()
                    if path:
                        try:
                            self._handle_temp_audio(path)
                        finally:
                            try:
                                os.unlink(path)
                            except OSError:
                                pass
                    else:
                        time.sleep(0.12)
                except KeyboardInterrupt:
                    break
                except Exception as exc:
                    # Avoid flooding logs if a microphone is unplugged or ALSA is not configured.
                    log.warning("Audio loop recoverable error: %s", exc)
                    with self.status.lock:
                        self.status.last_error = str(exc)[:240]
                    self.status.set_boot_step("Microphone", "warning", str(exc)[:200])
                    self.status.set_stage("Microphone unavailable", "Check AUDIO_INPUT_DEVICE and AUDIO_SAMPLE_RATE")
                    time.sleep(3.0)
        finally:
            self.close()

    def close(self) -> None:
        self.stop_event.set()
        self.thermal.stop()
        self.hardware.close()
        if self.assistant is not None:
            self.assistant.close()
        self.face.stop_server()


def main() -> int:
    app = EMOApplication()
    def stop_handler(_signum, _frame):
        app.stop_event.set()
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        app.run()
        return 0
    except KeyboardInterrupt:
        app.close()
        return 0
    except Exception as exc:
        logging.exception("Could not start EMO: %s", exc)
        app.close()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
