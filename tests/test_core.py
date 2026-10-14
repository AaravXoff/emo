from __future__ import annotations
import tempfile
import unittest
import threading
import os
from pathlib import Path

from intent_router import IntentRouter
from response_cache import ResponseCache, normalize
from emo import strip_wake_word
from html_face import HtmlFace
from state import RobotStatus
from speech import WhisperRecognizer
from feature_hub import FEATURES, LocalFeatureHub
from audio_io import AudioRecorder, SoundEffects
from assistant import EMOAssistant
from services import TimeService
from tts_engine import SpeechEngine
from config import Config, BASE_DIR
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
import json


class ConfigValidationTests(unittest.TestCase):
    def test_invalid_values_use_safe_defaults_and_relative_paths_anchor_to_project(self):
        overrides = {
            "AUDIO_SAMPLE_RATE": "0",
            "AUDIO_RMS_THRESHOLD": "nan",
            "MAX_UTTERANCE_SECONDS": "-1",
            "TRAILING_SILENCE_SECONDS": "99",
            "WHISPER_CPP_THREADS": "999",
            "LM_STUDIO_TIMEOUT_SECONDS": "0",
            "WAKE_TIMEOUT_SECONDS": "9999",
            "TOUCH_GPIO": "999",
            "SERVO_GPIO": "28",
            "THERMAL_WARNING_C": "90",
            "THERMAL_LOCKOUT_C": "80",
            "THERMAL_RESUME_C": "85",
            "WAKE_WORDS": "",
            "EMO_DATA_DIR": "./relative-emo-data",
            "EMO_CACHE_DB": "./relative-cache.sqlite3",
        }
        with patch.dict(os.environ, overrides, clear=False):
            config = Config.from_env()

        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.audio_rms_threshold, 0.012)
        self.assertEqual(config.max_utterance_seconds, 8.0)
        self.assertEqual(config.trailing_silence_seconds, 0.75)
        self.assertEqual(config.whisper_cpp_threads, 2)
        self.assertEqual(config.lm_studio_timeout_seconds, 35)
        self.assertEqual(config.wake_timeout_seconds, 20)
        self.assertEqual(config.touch_gpio, 0)
        self.assertEqual(config.servo_gpio, 0)
        self.assertEqual(
            (config.thermal_warning_c, config.thermal_lockout_c, config.thermal_resume_c),
            (75.0, 82.0, 72.0),
        )
        self.assertTrue(config.wake_words)
        self.assertEqual(config.data_dir, (BASE_DIR / "relative-emo-data").resolve())
        self.assertEqual(config.cache_db, (BASE_DIR / "relative-cache.sqlite3").resolve())


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

    def test_greeting_substrings_do_not_match_inside_other_words(self):
        self.assertEqual(self.router.match("Which country has the largest rainforest?").name, "llm")
        self.assertEqual(self.router.match("This is a history question").name, "llm")
        self.assertEqual(self.router.match("Hi EMO").name, "greeting")

    def test_topic_mentions_do_not_become_time_weather_or_animal_commands(self):
        self.assertEqual(self.router.match("I have no time today").name, "llm")
        self.assertEqual(self.router.match("My clock is broken").name, "llm")
        self.assertEqual(self.router.match("What is the best time to plant roses?").name, "llm")
        self.assertEqual(self.router.match("Explain how temperature affects pressure").name, "llm")
        self.assertEqual(self.router.match("What is the temperature of a star?").name, "llm")
        self.assertEqual(self.router.match("Explain current weather patterns").name, "llm")
        self.assertEqual(self.router.match("What does the cat eat?").name, "llm")
        self.assertEqual(self.router.match("What's the temperature in Delhi today?").name, "weather")
        self.assertEqual(self.router.match("Is it raining in Delhi?").name, "weather")
        self.assertEqual(self.router.match("What time is it now?").name, "time")

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


class AudioDeviceSelectionTests(unittest.TestCase):
    def test_falls_back_to_first_input_when_os_default_is_minus_one(self):
        class FakeSD:
            default = type("Default", (), {"device": (-1, -1)})()
            @staticmethod
            def query_devices(device=None):
                devices = [
                    {"name": "USB Microphone", "max_input_channels": 1},
                    {"name": "HDMI Output", "max_input_channels": 0},
                ]
                return devices if device is None else devices[device]
        recorder = AudioRecorder(16000, 0.012, 8, 0.75)
        device, name, error = recorder._resolve_input_device(FakeSD)
        self.assertEqual(device, 0)
        self.assertEqual(name, "USB Microphone")
        self.assertEqual(error, "")

    def test_configured_device_name_selects_a_unique_input(self):
        class FakeSD:
            default = type("Default", (), {"device": (-1, -1)})()
            @staticmethod
            def query_devices(device=None):
                devices = [
                    {"name": "USB Headphones", "max_input_channels": 0},
                    {"name": "USB Microphone Mono", "max_input_channels": 1},
                ]
                return devices if device is None else devices[device]
        recorder = AudioRecorder(16000, 0.012, 8, 0.75, input_device="microphone mono")
        device, name, error = recorder._resolve_input_device(FakeSD)
        self.assertEqual(device, 1)
        self.assertEqual(name, "USB Microphone Mono")
        self.assertEqual(error, "")

    def test_ambiguous_configured_device_name_is_reported(self):
        class FakeSD:
            default = type("Default", (), {"device": (-1, -1)})()
            @staticmethod
            def query_devices(device=None):
                devices = [
                    {"name": "USB Microphone Left", "max_input_channels": 1},
                    {"name": "USB Microphone Right", "max_input_channels": 1},
                ]
                return devices if device is None else devices[device]
        recorder = AudioRecorder(16000, 0.012, 8, 0.75, input_device="USB Microphone")
        device, _name, error = recorder._resolve_input_device(FakeSD)
        self.assertIsNone(device)
        self.assertIn("ambiguous", error)

    def test_reports_missing_input_without_using_invalid_default(self):
        class FakeSD:
            default = type("Default", (), {"device": (-1, -1)})()
            @staticmethod
            def query_devices(device=None):
                return [{"name": "Speaker", "max_input_channels": 0}]
        recorder = AudioRecorder(16000, 0.012, 8, 0.75)
        device, _name, error = recorder._resolve_input_device(FakeSD)
        self.assertIsNone(device)
        self.assertIn("No microphone input device", error)

    def test_uses_device_native_rate_when_16khz_is_not_supported(self):
        class FakeSD:
            @staticmethod
            def check_input_settings(device, channels, dtype, samplerate):
                if samplerate != 48000:
                    raise ValueError("unsupported sample rate")
            @staticmethod
            def query_devices(device=None):
                return {"name": "USB mic", "default_samplerate": 48000}
        recorder = AudioRecorder(16000, 0.012, 8, 0.75)
        self.assertEqual(recorder._resolve_sample_rate(FakeSD, 0), 48000)

    def test_recording_stops_immediately_when_stop_event_is_set(self):
        stop = threading.Event()
        stop.set()
        recorder = AudioRecorder(16000, 0.012, 8, 0.75, stop_event=stop)
        self.assertIsNone(recorder.record_utterance())


class AssistantSafetyTests(unittest.TestCase):
    class FakeHardware:
        def set_expression(self, expression):
            pass

    class FakeSpeech:
        def __init__(self):
            self.last_text = ""
            self.face = None
            self.expression_during_speech = ""

        def speak(self, text, language="en"):
            self.last_text = text
            if self.face is not None:
                self.expression_during_speech = self.face.get_state()
            return True

    class FakeTranslator:
        @staticmethod
        def translate(text, source, target):
            return None

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            return "fallback answer"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = type("Config", (), {
            "data_dir": self.root / "data",
            "cache_db": self.root / "cache.sqlite3",
            "cache_enabled": True,
            "time_zone": "Asia/Kolkata",
            "cache_ttl_seconds": 3600,
            "weather_cache_ttl_seconds": 300,
        })()
        self.status = RobotStatus()
        self.face = HtmlFace(status=self.status)
        self.speech = self.FakeSpeech()
        self.speech.face = self.face
        self.llm = self.FakeLLM()
        self.assistant = EMOAssistant(
            self.config, self.status, self.face, self.FakeHardware(), None,
            self.FakeTranslator(), self.llm, self.speech,
        )

    def tearDown(self):
        self.assistant.close()
        self.temp.cleanup()

    def test_old_llm_cache_cannot_shadow_local_calculator(self):
        self.assistant.cache.put(
            "calculate 12 * (3 + 4)", "en", "llm", "incorrect cached answer", 3600
        )
        response = self.assistant.process(
            "calculate 12 * (3 + 4)", "en", "calculate 12 * (3 + 4)"
        )
        self.assertEqual(response, "The answer is 84.")

    def test_old_llm_cache_cannot_shadow_new_direct_greeting(self):
        self.assistant.cache.put(
            "Hello EMO", "en", "llm", "stale unrelated answer", 3600
        )
        response = self.assistant.process("Hello EMO", "en", "Hello EMO")
        self.assertEqual(response, "Hey! I'm EMO. What can I help you with?")

    def test_generated_password_is_not_saved_in_history_or_turn_log(self):
        response = self.assistant.process("generate password", "en", "generate password")
        self.assertIn("Generated password:", response)
        secret = response.split("Generated password:", 1)[1].strip()
        self.assertTrue(secret)
        history_text = " ".join(item["content"] for item in self.assistant._history)
        logged = self.assistant.cache.connection.execute(
            "SELECT response_text FROM turns ORDER BY id DESC LIMIT 1"
        ).fetchone()["response_text"]
        self.assertNotIn(secret, history_text)
        self.assertNotIn(secret, logged)
        self.assertEqual(logged, "[redacted: generated password]")

    def test_password_is_not_sent_to_translation_model(self):
        self.assistant.process("generate password", "hi", "generate password")
        self.assertEqual(self.llm.calls, 0)

    def test_malformed_date_does_not_crash_turn(self):
        response = self.assistant.process("days until 2026-02-30", "en", "days until 2026-02-30")
        self.assertIn("Please rephrase", response)

    def test_face_command_uses_requested_expression_while_speaking(self):
        self.assistant.process("be happy", "en", "be happy")
        self.assertEqual(self.speech.expression_during_speech, "happy")


class TimeZoneTests(unittest.TestCase):
    def test_time_service_offline_fallback_uses_configured_zone(self):
        zone = ZoneInfo("Pacific/Kiritimati")
        before = datetime.now(zone)
        with patch("services.urllib.request.urlopen", side_effect=OSError("offline")):
            response = TimeService.current_time("Pacific/Kiritimati", timeout=0.1)
        after = datetime.now(zone)
        acceptable_dates = {
            before.strftime("%A, %B") + f" {before.day}",
            after.strftime("%A, %B") + f" {after.day}",
        }
        self.assertTrue(any(value in response for value in acceptable_dates), response)

    def test_feature_hub_uses_configured_zone_for_calendar(self):
        with tempfile.TemporaryDirectory() as temp:
            hub = LocalFeatureHub(Path(temp), time_zone="Pacific/Kiritimati")
            try:
                self.assertEqual(hub.current_datetime().tzinfo.key, "Pacific/Kiritimati")
            finally:
                hub.close()


class SpeechOutputTests(unittest.TestCase):
    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def test_espeak_retries_english_when_requested_voice_fails(self):
        engine = SpeechEngine("espeak")
        with patch("tts_engine.shutil.which", return_value="/usr/bin/espeak-ng"), patch(
            "tts_engine.subprocess.run",
            side_effect=[self.Result(1), self.Result(0)],
        ) as run:
            self.assertTrue(engine._speak_espeak("hello", "pa"))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][2], "pa")
        self.assertEqual(run.call_args_list[1].args[0][2], "en-us")

    def test_espeak_reports_failure_when_all_voices_fail(self):
        engine = SpeechEngine("espeak")
        with patch("tts_engine.shutil.which", return_value="/usr/bin/espeak-ng"), patch(
            "tts_engine.subprocess.run",
            side_effect=[self.Result(1), self.Result(1)],
        ):
            self.assertFalse(engine._speak_espeak("hello", "pa"))

    def test_sound_effect_reports_aplay_failure(self):
        effect = SoundEffects()
        effect.player = "/usr/bin/aplay"
        with patch("audio_io.subprocess.run", return_value=self.Result(1)):
            self.assertFalse(effect._play_samples([0, 10]))


class LocalFeatureHubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.hub = LocalFeatureHub(Path(self.temp.name))

    def tearDown(self):
        self.hub.close()
        self.temp.cleanup()

    def test_feature_catalog_has_more_than_one_hundred_features(self):
        self.assertGreaterEqual(len(FEATURES), 100)
        self.assertEqual(self.hub.feature_count, len(FEATURES))

    def test_safe_calculator_and_conversion(self):
        self.assertEqual(self.hub.handle("calculate 12 * (3 + 4)")[1], "The answer is 84.")
        self.assertEqual(self.hub.handle("calculate sqrt(81)")[1], "The answer is 9.")
        self.assertEqual(self.hub.handle("calculate factorial(5)")[1], "The answer is 120.")
        self.assertIsNone(self.hub.handle("calculate factorial(100000000)"))
        route, text = self.hub.handle("convert 5 km to miles")
        self.assertEqual(route, "length_conversion")
        self.assertIn("3.106", text)

    def test_text_toolbox_and_encoders(self):
        self.assertEqual(self.hub.handle("count words in: hello there friend")[1], "There are 3 word(s).")
        self.assertEqual(self.hub.handle("uppercase: hello emo")[1], "HELLO EMO")
        encoded = self.hub.handle("base64 encode: hello")[1]
        self.assertEqual(encoded, "aGVsbG8=")
        self.assertEqual(self.hub.handle("base64 decode: aGVsbG8=")[1], "hello")

    def test_notes_and_todos_persist_in_local_database(self):
        route, saved = self.hub.handle("add note: test note")
        self.assertEqual(route, "note_add")
        self.assertIn("Saved note", saved)
        self.assertIn("test note", self.hub.handle("list notes")[1])
        route, task = self.hub.handle("add todo: feed cat")
        self.assertEqual(route, "todo_add")
        self.assertIn("feed cat", task)
        task_id = int(task.split("Added task ")[1].split(":")[0])
        self.assertEqual(self.hub.handle(f"complete todo {task_id}")[0], "todo_complete")
        self.assertIn("done", self.hub.handle("list todos")[1])

    def test_math_parser_rejects_python_execution(self):
        self.assertIsNone(self.hub.handle("__import__('os').system('echo unsafe')"))
        self.assertEqual(self.hub.handle("calculate 8 / 0")[1], "Division by zero is undefined.")

    def test_calendar_and_device_routes(self):
        self.assertEqual(self.hub.handle("what date is it")[0], "current_date")
        self.assertEqual(self.hub.handle("python version")[0], "python_version")
        self.assertEqual(self.hub.handle("next weekday monday")[0], "next_weekday")


if __name__ == "__main__":
    unittest.main()
