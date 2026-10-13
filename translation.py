"""Optional offline translations using installed Argos Translate language packs."""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "pa": "Punjabi", "ur": "Urdu",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean",
    "zh": "Chinese", "ar": "Arabic", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "ne": "Nepali",
    "id": "Indonesian", "nl": "Dutch", "tr": "Turkish", "vi": "Vietnamese",
    "pl": "Polish", "uk": "Ukrainian", "fa": "Persian", "sw": "Swahili",
}
ARGOS_CODE_ALIASES = {"iw": "he", "jw": "jv", "nb": "no"}


class OfflineTranslator:
    """Never sends text to a remote translation service."""
    def __init__(self) -> None:
        self._translate_module = None

    def _load(self) -> None:
        if self._translate_module is None:
            import argostranslate.translate as at
            self._translate_module = at

    def installed(self) -> bool:
        try:
            self._load()
            return bool(self._translate_module.get_installed_languages())
        except Exception:
            return False

    def translate(self, text: str, source: str, target: str) -> str | None:
        if not text or source == target:
            return text
        source = ARGOS_CODE_ALIASES.get(source, source)
        target = ARGOS_CODE_ALIASES.get(target, target)
        try:
            self._load()
            module = self._translate_module
            languages = module.get_installed_languages()
            source_language = next((lang for lang in languages if lang.code == source), None)
            target_language = next((lang for lang in languages if lang.code == target), None)
            if source_language is None or target_language is None:
                return None
            translation = source_language.get_translation(target_language)
            if translation is None:
                return None
            return translation.translate(text).strip() or None
        except ImportError:
            return None
        except Exception as exc:
            log.debug("Offline translation failed (%s -> %s): %s", source, target, exc)
            return None

    @staticmethod
    def language_name(code: str) -> str:
        return LANGUAGE_NAMES.get(code, code or "the speaker's language")
