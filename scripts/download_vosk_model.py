#!/usr/bin/env python3
"""Download the small English Vosk wake-word model with safe ZIP extraction."""
from __future__ import annotations
import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path.home() / ".cache" / "emo")
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    target = destination / MODEL_NAME
    if (target / "conf").is_dir() and (target / "am").is_dir():
        print(f"Vosk wake model already available: {target}")
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="emo-vosk-") as temp:
        temp_path = Path(temp).resolve()
        archive = temp_path / "model.zip"
        request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "EMO-setup/1.0"})
        print(f"Downloading the Vosk model from {MODEL_URL}")
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                resolved = (temp_path / item.filename).resolve()
                if resolved != temp_path and temp_path not in resolved.parents:
                    raise RuntimeError(f"Unsafe archive path: {item.filename}")
            zipped.extractall(temp_path)
        extracted = temp_path / MODEL_NAME
        if not (extracted / "conf").is_dir() or not (extracted / "am").is_dir():
            raise RuntimeError("Vosk archive did not contain the expected model.")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
    print(f"Ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
