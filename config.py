"""Environment configuration for EMO. No secrets are committed."""
from dataclasses import dataclass
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    base_dir: Path
    data_dir: Path
    cache_db: Path
    whisper_model: str
    whisper_translation_model: str
    whisper_download_root: Path
    preload_whisper: bool
    sample_rate: int
    audio_rms_threshold: float
    max_utterance_seconds: float
    trailing_silence_seconds: float
    wake_word_required: bool
    wake_words: tuple[str, ...]
    wake_timeout_seconds: int
    lm_studio_base_url: str
    lm_studio_model: str
    lm_studio_timeout_seconds: int
    weather_city: str
    cache_enabled: bool
    cache_ttl_seconds: int
    weather_cache_ttl_seconds: int
    tts_engine: str
    chatterbox_language_mode: str
    touch_gpio: int
    servo_gpio: int
    thermal_warning_c: float
    thermal_lockout_c: float
    thermal_resume_c: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        _load_dotenv(BASE_DIR / ".env")
        data_dir = Path(os.getenv("EMO_DATA_DIR", str(BASE_DIR / "data"))).expanduser()
        whisper_dir = Path(os.getenv("WHISPER_DOWNLOAD_ROOT", str(Path.home() / ".cache" / "whisper"))).expanduser()
        wake_words = tuple(v.strip().lower() for v in os.getenv(
            "WAKE_WORDS", "hey emo,hi emo,hello emo,okay emo,ok emo"
        ).split(",") if v.strip())
        return cls(
            base_dir=BASE_DIR,
            data_dir=data_dir,
            cache_db=Path(os.getenv("EMO_CACHE_DB", str(data_dir / "emo_cache.sqlite3"))).expanduser(),
            whisper_model=os.getenv("WHISPER_MODEL", "tiny").strip().lower(),
            whisper_translation_model=os.getenv("WHISPER_TRANSLATION_MODEL", "tiny").strip().lower(),
            whisper_download_root=whisper_dir,
            preload_whisper=_bool("PRELOAD_WHISPER", False),
            sample_rate=_int("AUDIO_SAMPLE_RATE", 16000),
            audio_rms_threshold=_float("AUDIO_RMS_THRESHOLD", 0.012),
            max_utterance_seconds=_float("MAX_UTTERANCE_SECONDS", 8.0),
            trailing_silence_seconds=_float("TRAILING_SILENCE_SECONDS", 0.75),
            wake_word_required=_bool("WAKE_WORD_REQUIRED", True),
            wake_words=wake_words,
            wake_timeout_seconds=_int("WAKE_TIMEOUT_SECONDS", 20),
            lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/"),
            lm_studio_model=os.getenv("LM_STUDIO_MODEL", "qwen2.5-0.5b-instruct"),
            lm_studio_timeout_seconds=_int("LM_STUDIO_TIMEOUT_SECONDS", 35),
            weather_city=os.getenv("WEATHER_CITY", "").strip(),
            cache_enabled=_bool("CACHE_ENABLED", True),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 86400),
            weather_cache_ttl_seconds=_int("WEATHER_CACHE_TTL_SECONDS", 300),
            tts_engine=os.getenv("TTS_ENGINE", "auto").strip().lower(),
            chatterbox_language_mode=os.getenv("CHATTERBOX_LANGUAGE_MODE", "multilingual").strip().lower(),
            touch_gpio=_int("TOUCH_GPIO", 0),
            servo_gpio=_int("SERVO_GPIO", 0),
            thermal_warning_c=_float("THERMAL_WARNING_C", 75.0),
            thermal_lockout_c=_float("THERMAL_LOCKOUT_C", 82.0),
            thermal_resume_c=_float("THERMAL_RESUME_C", 72.0),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
