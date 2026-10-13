from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from intent_router import IntentRouter
from response_cache import ResponseCache, normalize
from emo import strip_wake_word
from html_face import HtmlFace
from state import RobotStatus
from speech import WhisperRecognizer
from unittest.mock import patch
import json


class IntentTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()

    def test_direct_intents(self):
        self.assertEqual(self.router.match("What time is it?").name, "time")
        self.assertEqual(self.router.match("What's the weather?").name, "weather")
        self.assertEqual(self.router.match("What does the cat say?").entity, "cat")
        self.assertEqual(self.router.match("Can you sing a song?").name, "sing")
        self.assertEqual(self.router.match("Hello EMO").name, "greeting")

    def test_weather_city_extraction(self):
        intent = self.router.match("What's the weather in Delhi today?")
        self.assertEqual(intent.name, "weather")
        self.assertEqual(intent.entity, "delhi")

    def test_unknown_goes_to_llm(self):
        self.assertEqual(self.router.match("Explain black holes").name, "llm")

    def test_normalization(self):
        self.assertEqual(normalize(" Hello, EMO!!! "), "hello emo")

    def test_wake_word_strip(self):
        detected, remaining = strip_wake_word("Hey EMO, what's the weather?", ("hey emo", "hi emo"))
        self.assertTrue(detected)
        self.assertEqual(remaining, "what's the weather?")

    def test_wake_word_without_command_and_transcription_variant(self):
        self.assertEqual(strip_wake_word("Hey EMO", ("hey emo",))[1], "")
        detected, remaining = strip_wake_word("Hey emu, sing a song", ("hey emo",))
        self.assertTrue(detected)
        self.assertEqual(remaining, "sing a song")


class CacheTests(unittest.TestCase):
    def test_cache_and_turn_log(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = ResponseCache(Path(temp) / "cache.sqlite3")
            cache.put("Hello", "en", "greeting", "Hey there", 60)
            self.assertEqual(cache.get("Hello", "en")["response_text"], "Hey there")
            self.assertIsNone(cache.get("hello there", "en"))
            cache.log_turn("Hello", "en", "Hello", "greeting", "Hey there", process={"cache": "hit"})
            row = cache.connection.execute("SELECT original_text, route FROM turns").fetchone()
            self.assertEqual((row["original_text"], row["route"]), ("Hello", "greeting"))
            cache.close()


class FaceApiTests(unittest.TestCase):
    def test_state_endpoint_exposes_boot_status(self):
        status = RobotStatus()
        face = HtmlFace(status=status)
        response = face.app.test_client().get("/api/state")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["boot_complete"])
        self.assertIn("Whisper", payload["boot_steps"])


class WhisperCppTests(unittest.TestCase):
    def test_cpp_transcription_and_english_translation(self):
        class NoTranslation:
            @staticmethod
            def translate(text, source, target):
                return None

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary = root / "whisper-cli"
            model = root / "ggml-tiny.bin"
            wav = root / "input.wav"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(0o755)
            model.touch()
            wav.touch()
            recognizer = WhisperRecognizer(
                model_name="tiny", translation_model_name="tiny",
                download_root=root / "cache", translator=NoTranslation(),
                backend="cpp", cpp_binary=binary, cpp_model=model, cpp_threads=2,
            )

            def fake_run(command, **kwargs):
                output_prefix = Path(command[command.index("--output-file") + 1])
                translating = "--translate" in command
                payload = {
                    "params": {"language": "hi" if translating else "auto"},
                    "result": {"language": "hi"},
                    "transcription": [{"text": "Hello from EMO"}] if translating
                                    else [{"text": " नमस्ते EMO"}],
                }
                output_prefix.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
                return type("Completed", (), {"stdout": "", "stderr": "", "returncode": 0})()

            with patch("speech.subprocess.run", side_effect=fake_run) as run:
                result = recognizer.transcribe(wav)

            self.assertEqual(result.text, "नमस्ते EMO")
            self.assertEqual(result.language, "hi")
            self.assertEqual(result.english_text, "Hello from EMO")
            self.assertEqual(result.translation_method, "whisper.cpp-translate")
            self.assertEqual(run.call_count, 2)
            self.assertIn("--translate", run.call_args_list[1].args[0])


if __name__ == "__main__":
    unittest.main()
