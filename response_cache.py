"""SQLite transcript/context log and response cache. Audio itself is not retained."""
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path


def normalize(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


class ResponseCache:
    def __init__(self, db_path: Path, enabled: bool = True) -> None:
        self.db_path = Path(db_path)
        self.enabled = enabled
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10)
        self.connection.row_factory = sqlite3.Row
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA busy_timeout=5000")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY, normalized_text TEXT NOT NULL,
                    language TEXT NOT NULL, route TEXT NOT NULL, response_text TEXT NOT NULL,
                    created_at REAL NOT NULL, expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,
                    original_text TEXT NOT NULL, language TEXT NOT NULL, english_text TEXT NOT NULL,
                    route TEXT NOT NULL, response_text TEXT NOT NULL, cache_hit INTEGER NOT NULL DEFAULT 0,
                    process_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS turns_created_at ON turns(created_at);
            """)
            self.connection.commit()

    @staticmethod
    def make_key(text: str, language: str) -> tuple[str, str]:
        normalized = normalize(text)
        material = language.casefold() + chr(31) + normalized
        return hashlib.sha256(material.encode("utf-8")).hexdigest(), normalized

    def get(self, text: str, language: str) -> dict | None:
        if not self.enabled:
            return None
        key, _ = self.make_key(text, language)
        now = time.time()
        with self._lock:
            row = self.connection.execute(
                "SELECT route, response_text, created_at FROM response_cache WHERE cache_key=? AND expires_at>?",
                (key, now),
            ).fetchone()
            if row is None:
                self.connection.execute("DELETE FROM response_cache WHERE expires_at<=?", (now,))
                self.connection.commit()
                return None
            return dict(row)

    def put(self, text: str, language: str, route: str, response_text: str, ttl_seconds: int) -> None:
        if not self.enabled or ttl_seconds <= 0 or not text or not response_text:
            return
        key, normalized = self.make_key(text, language)
        now = time.time()
        with self._lock:
            self.connection.execute("""
                INSERT INTO response_cache(cache_key, normalized_text, language, route, response_text, created_at, expires_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                  normalized_text=excluded.normalized_text, language=excluded.language,
                  route=excluded.route, response_text=excluded.response_text,
                  created_at=excluded.created_at, expires_at=excluded.expires_at
            """, (key, normalized, language, route, response_text, now, now + ttl_seconds))
            self.connection.commit()

    def log_turn(self, original_text: str, language: str, english_text: str, route: str,
                 response_text: str, cache_hit: bool = False, process: dict | None = None) -> None:
        if not self.enabled:
            return
        safe_process = json.dumps(process or {}, ensure_ascii=False)[:8000]
        with self._lock:
            self.connection.execute("""
                INSERT INTO turns(created_at, original_text, language, english_text, route, response_text, cache_hit, process_json)
                VALUES(?,?,?,?,?,?,?,?)
            """, (time.time(), original_text[:4000], language[:24], english_text[:4000],
                  route[:80], response_text[:8000], int(cache_hit), safe_process))
            self.connection.commit()

    def close(self) -> None:
        with self._lock:
            self.connection.close()
