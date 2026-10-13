"""Fast local phrase/intent matcher; no internet or LLM is required for direct commands."""
from __future__ import annotations
from dataclasses import dataclass
import re
from difflib import SequenceMatcher
from response_cache import normalize


@dataclass
class Intent:
    name: str
    entity: str = ""
    confidence: float = 1.0


class IntentRouter:
    def __init__(self) -> None:
        self.greetings = [
            "hello", "hi", "hey", "hello emo", "hi emo", "hey emo", "good morning",
            "good afternoon", "good evening", "how are you", "how are you emo",
            "what is up", "whats up", "thanks", "thank you",
        ]
        self.time_phrases = [
            "what time is it", "what is the time", "tell me the time",
            "current time", "time now", "what is today's time", "clock",
        ]
        self.weather_phrases = [
            "whats the weather", "what is the weather", "weather forecast",
            "current weather", "weather today", "will it rain", "is it raining",
            "temperature outside", "how is the weather",
        ]
        self.song_phrases = [
            "sing a song", "sing for me", "can you sing", "sing something",
            "play your song", "sing me a song",
        ]

    @staticmethod
    def _best_match(text: str, phrases: list[str], threshold: float = 0.76) -> float:
        if not text:
            return 0.0
        best = 0.0
        for phrase in phrases:
            if phrase in text or text in phrase:
                return 1.0
            best = max(best, SequenceMatcher(None, text, phrase).ratio())
        return best if best >= threshold else 0.0

    def match(self, text: str) -> Intent:
        original = text.strip().lower()
        t = normalize(original)
        if not t:
            return Intent("empty")

        animals = {
            "cat": r"\b(cat|kitten)\b", "dog": r"\b(dog|puppy)\b",
            "wolf": r"\b(wolf|wolves)\b", "fox": r"\b(fox|foxes)\b",
        }
        says = bool(re.search(r"\b(say|says|sound|sounds|noise|make the sound|what does)\b", t))
        if says:
            for animal, pattern in animals.items():
                if re.search(pattern, t):
                    return Intent("animal_sound", animal)
        for animal, pattern in animals.items():
            if re.search(pattern, t) and re.search(r"\b(sound|noise|voice|say|says)\b", t):
                return Intent("animal_sound", animal, .94)

        score = self._best_match(t, self.song_phrases)
        if score:
            return Intent("sing", confidence=score)

        if re.search(r"\b(weather|forecast|rain|raining|mausam|temperature)\b", t):
            score = self._best_match(t, self.weather_phrases, .55) or .78
            city = ""
            searchable = original.rstrip(" .?!")
            match = re.search(r"\b(?:in|for|at)\s+([a-z][a-z .'-]{1,45})$", searchable, flags=re.I)
            if match:
                city = re.sub(r"\\s+(today|now|tonight|tomorrow)$", "", match.group(1).strip(), flags=re.I).strip(" .?!")
            return Intent("weather", city, score)

        if re.search(r"\b(time|clock)\b", t) and re.search(r"\b(what|tell|current|now|time|clock)\b", t):
            return Intent("time", confidence=self._best_match(t, self.time_phrases, .54) or .78)

        if t in self.greetings or self._best_match(t, self.greetings, .90):
            return Intent("greeting", confidence=.94)

        if t in {"namaste", "namaskar", "sat sri akal", "satsriakal", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "नमस्ते"}:
            return Intent("greeting", confidence=.99)
        return Intent("llm", confidence=0.0)
