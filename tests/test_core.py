from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from intent_router import IntentRouter
from response_cache import ResponseCache, normalize
from emo import strip_wake_word


class IntentTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()

    def test_direct_intents(self):
        self.assertEqual(self.router.match("What time is it?").name, "time")
        self.assertEqual(self.router.match("What's the weather?").name, "weather")
        self.assertEqual(self.router.match("What does the cat say?").entity, "cat")
        self.assertEqual(self.router.match("Can you sing a song?").name, "sing")
        self.assertEqual(self.router.match("Hello EMO").name, "greeting")

    def test_unknown_goes_to_llm(self):
        self.assertEqual(self.router.match("Explain black holes").name, "llm")

    def test_normalization(self):
        self.assertEqual(normalize(" Hello, EMO!!! "), "hello emo")

    def test_wake_word_strip(self):
        detected, remaining = strip_wake_word("Hey EMO, what's the weather?", ("hey emo", "hi emo"))
        self.assertTrue(detected)
        self.assertEqual(remaining, "what's the weather?")


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


if __name__ == "__main__":
    unittest.main()
