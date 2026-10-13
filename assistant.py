"""Intent router, direct actions, local LLM fallback, TTS and context caching."""
from __future__ import annotations
import logging
import time

from audio_io import SoundEffects
from intent_router import IntentRouter
from response_cache import ResponseCache
from services import TimeService, WeatherService
from translation import OfflineTranslator, LANGUAGE_NAMES

log = logging.getLogger(__name__)
ORIGINAL_SONG = "I'm little EMO, bright and small. Here to help and learn with all."


class EMOAssistant:
    def __init__(self, config, status, face, hardware, recognizer, translator, llm, speech_engine) -> None:
        self.config = config
        self.status = status
        self.face = face
        self.hardware = hardware
        self.recognizer = recognizer
        self.translator: OfflineTranslator = translator
        self.llm = llm
        self.speech = speech_engine
        self.router = IntentRouter()
        self.cache = ResponseCache(config.cache_db, enabled=config.cache_enabled)
        self.weather = WeatherService()
        self.effects = SoundEffects()
        self._history: list[dict] = []

    def _localized(self, english_text: str, language: str) -> str:
        if language == "en":
            return english_text
        translated = self.translator.translate(english_text, "en", language)
        if translated:
            return translated
        # If a pack is missing, the local LLM can translate short direct-action responses.
        try:
            return self.llm.complete(
                "Translate the following response into " + LANGUAGE_NAMES.get(language, language) +
                ". Return only the translation and preserve all numbers, city names, and units:\n\n" + english_text,
                response_language=language,
            )
        except Exception:
            return english_text

    def _greeting(self, language: str, source_text: str) -> str:
        greetings = {
            "en": "Hey! I'm EMO. What can I help you with?",
            "hi": "नमस्ते! मैं EMO हूँ। मैं आपकी क्या मदद कर सकता हूँ?",
            "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ EMO ਹਾਂ। ਦੱਸੋ, ਮੈਂ ਕੀ ਮਦਦ ਕਰਾਂ?",
            "es": "¡Hola! Soy EMO. ¿En qué puedo ayudarte?",
            "fr": "Salut ! Je suis EMO. Comment puis-je t'aider ?",
            "de": "Hallo! Ich bin EMO. Wie kann ich dir helfen?",
        }
        if language in greetings:
            return greetings[language]
        return self._localized("Hey! I'm EMO. What can I help you with?", language)

    def _speak(self, text: str, language: str, expression: str = "talking") -> None:
        self.face.set_state(expression)
        self.hardware.set_expression(expression)
        self.face.set_talking(True)
        try:
            self.speech.speak(text, language)
        finally:
            self.face.set_talking(False)
            self.face.set_state("idle", duration=0.8)
            self.hardware.set_expression("idle")

    def process(self, original_text: str, language: str, english_text: str,
                translation_method: str = "unknown") -> str:
        original_text = original_text.strip()
        language = language or "en"
        english_text = (english_text or original_text).strip()
        if not original_text:
            return ""
        intent = self.router.match(english_text)
        started = time.monotonic()
        cache_hit = False
        cached = self.cache.get(original_text, language)
        response = ""
        route = intent.name
        process_data = {
            "detected_language": language,
            "translation_method": translation_method,
            "intent_confidence": intent.confidence,
        }

        # Do not serve a stale clock answer. Weather is cached only for the short configured TTL.
        if cached and cached["route"] not in {"time", "weather", "animal_sound", "sing"}:
            response = cached["response_text"]
            route = cached["route"]
            cache_hit = True
            process_data["cache"] = "hit"
        else:
            process_data["cache"] = "miss"
            try:
                if self.status.snapshot()["thermal_lockout"]:
                    response = self._localized("I'm getting too hot, so I'm pausing heavy AI work until I cool down.", language)
                    route = "thermal_guard"
                elif intent.name == "greeting":
                    response = self._greeting(language, original_text)
                elif intent.name == "time":
                    response = self._localized(TimeService.current_time(), language)
                elif intent.name == "weather":
                    city = intent.entity or self.config.weather_city
                    try:
                        weather_text = self.weather.current(city)
                        response = self._localized(weather_text, language)
                    except Exception as exc:
                        log.info("Weather unavailable: %s", exc)
                        response = self._localized(
                            "I couldn't retrieve current weather. Check the internet connection and set WEATHER_CITY in the .env file, or ask about a named city.",
                            language,
                        )
                elif intent.name == "animal_sound":
                    labels = {"cat": "meow meow", "dog": "woof woof", "wolf": "a long howl", "fox": "a quick yip"}
                    self.effects.play_animal(intent.entity)
                    response = self._localized(f"A {intent.entity} goes {labels.get(intent.entity, 'sound')}!", language)
                    self.face.set_state("excited", duration=1.5)
                elif intent.name == "sing":
                    self.effects.play_original_tune()
                    response = self._localized(ORIGINAL_SONG, language)
                    self.face.set_state("dance", duration=3.0)
                else:
                    self.status.set_stage("Thinking", "Sending the question to the local LM Studio model")
                    try:
                        response = self.llm.complete(english_text, response_language=language, history=self._history)
                    except Exception as exc:
                        log.warning("LLM fallback failed: %s", exc)
                        response = self._localized(
                            "I couldn't reach the local language model. Make sure LM Studio is running, the model is loaded, and LM_STUDIO_BASE_URL is correct.",
                            language,
                        )
            except Exception as exc:
                log.exception("Intent processing failed")
                response = self._localized("Sorry, something went wrong while I was handling that request.", language)
                route = "error"
                with self.status.lock:
                    self.status.last_error = str(exc)[:240]

        self._history.extend([
            {"role": "user", "content": original_text[:1200]},
            {"role": "assistant", "content": response[:1200]},
        ])
        self._history = self._history[-8:]
        if not cache_hit:
            if route in {"greeting", "llm"}:
                self.cache.put(original_text, language, route, response, self.config.cache_ttl_seconds)
            elif route == "weather":
                self.cache.put(original_text, language, route, response, self.config.weather_cache_ttl_seconds)

        with self.status.lock:
            self.status.last_action = route
        self.cache.log_turn(
            original_text, language, english_text, route, response, cache_hit=cache_hit,
            process={**process_data, "elapsed_seconds": round(time.monotonic() - started, 3)},
        )
        self._speak(response, language, expression="happy" if route == "greeting" else "talking")
        self.status.set_stage("Listening", "Say a wake phrase followed by a request")
        return response

    def close(self) -> None:
        self.cache.close()
