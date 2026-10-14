"""Environment configuration for EMO. No secrets are committed."""
from dataclasses import dataclass
from pathlib import Path
import math
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


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = _int(name, default)
    return value if minimum <= value <= maximum else default


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = _float(name, default)
    return value if math.isfinite(value) and minimum <= value <= maximum else default


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _audio_device():
    """Parse AUDIO_INPUT_DEVICE: blank/default for auto; integer index or device-name string."""
    value = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
    if not value or value.casefold() in {"default", "auto", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        return value


@dataclass(frozen=True)
class Config:
    base_dir: Path
    data_dir: Path
    cache_db: Path
    whisper_model: str
    whisper_translation_model: str
    whisper_download_root: Path
    whisper_backend: str
    whisper_cpp_bin: Path
    whisper_cpp_model: Path
    whisper_cpp_threads: int
    preload_whisper: bool
    sample_rate: int
    audio_input_device: str | int | None
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
    time_zone: str
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
        data_dir = _project_path(os.getenv("EMO_DATA_DIR", str(BASE_DIR / "data")))
        whisper_dir = _project_path(os.getenv("WHISPER_DOWNLOAD_ROOT", str(Path.home() / ".cache" / "whisper")))
        wake_words = tuple(v.strip().lower() for v in os.getenv(
            "WAKE_WORDS", "hey emo,hi emo,hello emo,okay emo,ok emo"
        ).split(",") if v.strip())
        if not wake_words:
            wake_words = ("hey emo", "hi emo", "hello emo", "okay emo", "ok emo")

        thermal_warning = _bounded_float("THERMAL_WARNING_C", 75.0, 30.0, 120.0)
        thermal_lockout = _bounded_float("THERMAL_LOCKOUT_C", 82.0, 30.0, 120.0)
        thermal_resume = _bounded_float("THERMAL_RESUME_C", 72.0, 30.0, 120.0)
        if not thermal_resume < thermal_warning < thermal_lockout:
            thermal_warning, thermal_lockout, thermal_resume = 75.0, 82.0, 72.0

        cache_db = _project_path(os.getenv("EMO_CACHE_DB", str(data_dir / "emo_cache.sqlite3")))
        return cls(
            base_dir=BASE_DIR,
            data_dir=data_dir,
            cache_db=cache_db,
            whisper_model=os.getenv("WHISPER_MODEL", "tiny").strip().lower(),
            whisper_translation_model=os.getenv("WHISPER_TRANSLATION_MODEL", "tiny").strip().lower(),
            whisper_download_root=whisper_dir,
            whisper_backend=os.getenv("WHISPER_BACKEND", "auto").strip().lower(),
            whisper_cpp_bin=_project_path(os.getenv("WHISPER_CPP_BIN", str(Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"))),
            whisper_cpp_model=_project_path(os.getenv("WHISPER_CPP_MODEL", str(Path.home() / "whisper.cpp" / "models" / "ggml-tiny.bin"))),
            whisper_cpp_threads=_bounded_int("WHISPER_CPP_THREADS", 2, 1, 8),
            preload_whisper=_bool("PRELOAD_WHISPER", False),
            sample_rate=_bounded_int("AUDIO_SAMPLE_RATE", 16000, 8000, 48000),
            audio_input_device=_audio_device(),
            audio_rms_threshold=_bounded_float("AUDIO_RMS_THRESHOLD", 0.012, 0.0001, 0.5),
            max_utterance_seconds=_bounded_float("MAX_UTTERANCE_SECONDS", 8.0, 0.5, 30.0),
            trailing_silence_seconds=_bounded_float("TRAILING_SILENCE_SECONDS", 0.75, 0.1, 5.0),
            wake_word_required=_bool("WAKE_WORD_REQUIRED", True),
            wake_words=wake_words,
            wake_timeout_seconds=_bounded_int("WAKE_TIMEOUT_SECONDS", 20, 1, 300),
            lm_studio_base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/"),
            lm_studio_model=os.getenv("LM_STUDIO_MODEL", "qwen2.5-0.5b-instruct"),
            lm_studio_timeout_seconds=_bounded_int("LM_STUDIO_TIMEOUT_SECONDS", 35, 1, 120),
            weather_city=os.getenv("WEATHER_CITY", "").strip(),
            time_zone=os.getenv("TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata",
            cache_enabled=_bool("CACHE_ENABLED", True),
            cache_ttl_seconds=_bounded_int("CACHE_TTL_SECONDS", 86400, 0, 31_536_000),
            weather_cache_ttl_seconds=_bounded_int("WEATHER_CACHE_TTL_SECONDS", 300, 0, 86_400),
            tts_engine=os.getenv("TTS_ENGINE", "auto").strip().lower(),
            chatterbox_language_mode=os.getenv("CHATTERBOX_LANGUAGE_MODE", "multilingual").strip().lower(),
            touch_gpio=_bounded_int("TOUCH_GPIO", 0, 0, 27),
            servo_gpio=_bounded_int("SERVO_GPIO", 0, 0, 27),
            thermal_warning_c=thermal_warning,
            thermal_lockout_c=thermal_lockout,
            thermal_resume_c=thermal_resume,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
