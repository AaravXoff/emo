#!/usr/bin/env python3
"""Pre-download the English-only Faster-Whisper model for offline first use."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tiny.en", choices=("tiny.en",))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "emo" / "whisper-en",
    )
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        parser.error(
            "faster-whisper is not installed; run setup_english_whisper.sh first"
        )
        raise exc

    print(f"Downloading/verifying {args.model} in {cache_dir}")
    # Instantiating the model downloads missing Hugging Face weights and validates
    # that they can be loaded by the CPU INT8 runtime used by EMO.
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        download_root=str(cache_dir),
    )
    del model
    print(f"Ready: English-only Faster-Whisper {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
