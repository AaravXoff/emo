"""Local microphone capture and small synthesized animal/song effects."""
from __future__ import annotations
import math
import os
import shutil
import struct
import subprocess
import tempfile
import time
import wave
import logging

log = logging.getLogger(__name__)


class AudioRecorder:
    def __init__(self, sample_rate: int, threshold: float, max_seconds: float,
                 trailing_silence: float, status=None) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.max_seconds = max_seconds
        self.trailing_silence = trailing_silence
        self.status = status

    def check_device(self) -> tuple[bool, str]:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            inputs = [item for item in devices if item.get("max_input_channels", 0) > 0]
            return (bool(inputs), f"Found {len(inputs)} input device(s)" if inputs else "No microphone input device found")
        except Exception as exc:
            return False, f"Microphone check failed: {exc}"

    def record_utterance(self) -> str | None:
        """Wait for speech energy, record until silence, and return a temporary WAV path."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install numpy and sounddevice, and verify PortAudio is installed.") from exc

        block_frames = max(400, int(self.sample_rate * 0.10))
        chunks = []
        began = False
        voiced_seconds = 0.0
        silent_seconds = 0.0
        deadline = time.monotonic() + self.max_seconds + 12.0
        noise = 0.0
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=block_frames) as stream:
            while time.monotonic() < deadline:
                block, _overflowed = stream.read(block_frames)
                if block is None or len(block) == 0:
                    continue
                mono = block[:, 0].copy()
                rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
                if not began:
                    noise = 0.96 * noise + 0.04 * rms
                if self.status is not None:
                    with self.status.lock:
                        self.status.noise_floor = noise
                if rms >= self.threshold:
                    began = True
                    chunks.append(mono)
                    voiced_seconds += len(mono) / self.sample_rate
                    silent_seconds = 0.0
                elif began:
                    chunks.append(mono)
                    silent_seconds += len(mono) / self.sample_rate
                if began and (silent_seconds >= self.trailing_silence or voiced_seconds >= self.max_seconds):
                    break

        if not began or voiced_seconds < 0.18:
            return None
        audio = np.concatenate(chunks).clip(-1.0, 1.0)
        pcm = (audio * 32767).astype("<i2", copy=False).tobytes()
        fd, path = tempfile.mkstemp(prefix="emo-in-", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(pcm)
        return path


class SoundEffects:
    """Generates toy-like sound approximations locally; these are not animal recordings."""
    SAMPLE_RATE = 16000

    def __init__(self) -> None:
        self.player = shutil.which("aplay")

    def play_animal(self, animal: str) -> bool:
        specs = {"cat": (520, 860, .85, 1.1), "dog": (180, 100, .65, .7),
                 "wolf": (230, 470, 1.6, 1.0), "fox": (720, 1050, .35, .8)}
        start, end, duration, gain = specs.get(animal, specs["dog"])
        segments = 3 if animal == "dog" else 2 if animal in {"fox", "cat"} else 1
        samples = []
        total = int(self.SAMPLE_RATE * duration)
        for index in range(total):
            t = index / self.SAMPLE_RATE
            fraction = index / max(1, total - 1)
            freq = start + (end - start) * fraction
            if segments > 1:
                phase = (t * segments / duration) % 1.0
                envelope = min(1.0, phase * 9.0) * min(1.0, (1.0 - phase) * 8.0)
            else:
                envelope = min(1.0, t / .12) * min(1.0, (duration - t) / .16)
            value = (math.sin(2 * math.pi * freq * t) + .28 * math.sin(2 * math.pi * freq * 2.1 * t)) * envelope * gain
            samples.append(int(max(-1.0, min(1.0, value)) * 24000))
        return self._play_samples(samples)

    def play_original_tune(self) -> bool:
        notes = [(523.25, .18), (587.33, .18), (659.25, .22), (783.99, .28),
                 (659.25, .18), (783.99, .32), (880.00, .38)]
        samples = []
        sr = self.SAMPLE_RATE
        for frequency, duration in notes:
            count = int(sr * duration)
            for i in range(count):
                edge = min(1.0, i / (sr * .02), (count - i) / (sr * .04))
                samples.append(int(math.sin(2 * math.pi * frequency * i / sr) * max(0.0, edge) * .20 * 32767))
            samples.extend([0] * int(sr * .025))
        return self._play_samples(samples)

    def _play_samples(self, samples: list[int]) -> bool:
        path = None
        try:
            fd, path = tempfile.mkstemp(prefix="emo-fx-", suffix=".wav")
            os.close(fd)
            with wave.open(path, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(self.SAMPLE_RATE)
                output.writeframes(struct.pack("<" + "h" * len(samples), *samples))
            if self.player:
                subprocess.run([self.player, "-q", path], check=False, timeout=6,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            return False
        except Exception as exc:
            log.debug("Effect playback failed: %s", exc)
            return False
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
