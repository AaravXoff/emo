"""Lightweight always-on wake-word listener; Whisper only runs after it fires."""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(self, model_path: Path, wake_words=("emo", "hey emo", "hi emo"),
                 sample_rate: int = 16000, input_device=None) -> None:
        self.model_path = Path(model_path).expanduser()
        self.wake_words = tuple(dict.fromkeys(w.strip().casefold() for w in wake_words if w.strip()))
        self.sample_rate = int(sample_rate)
        self.input_device = input_device
        self._model = None
        self._vosk = None

    def check(self) -> tuple[bool, str]:
        if not self.model_path.is_dir():
            return False, f"Wake model is missing: {self.model_path}; run bash setup_emo_robot.sh"
        try:
            import vosk  # noqa: F401
        except ImportError:
            return False, "Vosk is not installed; run bash setup_emo_robot.sh"
        return True, f"Vosk wake-word model ready ({self.model_path.name})"

    def _load(self):
        if self._model is not None:
            return
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("Vosk is missing; run bash setup_emo_robot.sh") from exc
        if not self.model_path.is_dir():
            raise RuntimeError(f"Wake model not found at {self.model_path}; run bash setup_emo_robot.sh")
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self._model = vosk.Model(str(self.model_path))

    def wait_for_wake(self, stop_event, touch_event=None) -> bool:
        """Block on low-cost Vosk audio; return on 'emo' or touch, never run Whisper here."""
        if touch_event is not None and touch_event.is_set():
            touch_event.clear()
            return True
        if stop_event.is_set():
            return False
        try:
            import sounddevice as sd
            self._load()
            grammar = list(self.wake_words) + ["hey emu", "hello emo", "[unk]"]
            recognizer = self._vosk.KaldiRecognizer(
                self._model, self.sample_rate, json.dumps(grammar)
            )
            blocksize = max(800, self.sample_rate // 4)
            with sd.RawInputStream(
                device=self.input_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=blocksize,
            ) as stream:
                while not stop_event.is_set():
                    if touch_event is not None and touch_event.is_set():
                        touch_event.clear()
                        return True
                    data, _overflow = stream.read(blocksize)
                    if recognizer.AcceptWaveform(bytes(data)):
                        payload = json.loads(recognizer.Result() or "{}")
                    else:
                        payload = json.loads(recognizer.PartialResult() or "{}")
                    heard = str(payload.get("text", payload.get("partial", ""))).casefold().strip()
                    words = heard.replace("-", " ").split()
                    if "emo" in words or "emu" in words:
                        log.info("Wake word detected")
                        return True
            return False
        except Exception as exc:
            if not stop_event.is_set():
                log.warning("Wake listener unavailable: %s", exc)
                stop_event.wait(2.0)
            return False
