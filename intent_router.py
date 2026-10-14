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
            "what time is it", "what time is it now", "whats the time", "what is the time", "tell me the time",
            "current time", "time now", "time please", "what is today's time", "what is the current time",
            "do you know the time", "give me the time", "tell me what time it is",
            "can you tell me the time",
        ]
        self.weather_phrases = [
            "whats the weather", "what is the weather", "weather forecast",
            "current weather", "weather today", "will it rain", "is it raining",
            "temperature outside", "how is the weather", "weather in", "weather near",
            "forecast in", "forecast for", "rain in", "temperature in",
            "what is the temperature", "whats the temperature", "temperature today", "temperature now",
        ]
        self.song_phrases = [
            "sing a song", "sing for me", "can you sing", "sing something",
            "play your song", "sing me a song",
        ]

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        # Match whole words/phrases only. Substring checks make "which" match "hi".
        pattern = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)"
        return re.search(pattern, text, flags=re.UNICODE) is not None

    @classmethod
    def _best_match(cls, text: str, phrases: list[str], threshold: float = 0.76) -> float:
        text = normalize(text)
        if not text:
            return 0.0
        best = 0.0
        for item in phrases:
            phrase = normalize(item)
            if not phrase:
                continue
            if cls._contains_phrase(text, phrase):
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
        says = bool(re.search(r"\b(say|says|sound|sounds|noise|voice|bark|barks|meow|howl)\b", t))
        if says:
            for animal, pattern in animals.items():
                if re.search(pattern, t):
                    return Intent("animal_sound", animal)
        for animal, pattern in animals.items():
            if re.search(pattern, t) and re.search(r"\b(sound|noise|voice|say|says|bark|barks|meow|howl)\b", t):
                return Intent("animal_sound", animal, .94)

        score = self._best_match(t, self.song_phrases)
        if score:
            return Intent("sing", confidence=score)

        weather_cue = bool(re.search(r"\b(weather|forecast|rain|raining|mausam|temperature)\b", t))
        weather_score = self._best_match(t, self.weather_phrases, .68)
        # Mentioning weather/temperature in a general question is not itself a request for a forecast.
        if weather_cue and weather_score:
            city = ""
            searchable = original.rstrip(" .?!")
            match = re.search(r"\b(?:in|for|at|near)\s+([a-z][a-z .'-]{1,45})$", searchable, flags=re.I)
            if match:
                city_words = match.group(1).strip().split()
                if city_words and city_words[-1].casefold() in {"today", "now", "tonight", "tomorrow"}:
                    city_words.pop()
                city = " ".join(city_words).strip(" .?!")
            return Intent("weather", city, weather_score)

        time_score = self._best_match(t, self.time_phrases, .72)
        if time_score:
            return Intent("time", confidence=time_score)

        if t in self.greetings or self._best_match(t, self.greetings, .90):
            return Intent("greeting", confidence=.94)

        if t in {"namaste", "namaskar", "sat sri akal", "satsriakal", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "नमस्ते"}:
            return Intent("greeting", confidence=.99)
        return Intent("llm", confidence=0.0)
