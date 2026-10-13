#!/usr/bin/env python3
"""Install requested Argos Translate language directions into the local machine."""
from __future__ import annotations
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", default="hi-en,en-hi,pa-en,en-pa",
        help="comma-separated source-target pairs, e.g. hi-en,en-hi,pa-en,en-pa",
    )
    args = parser.parse_args()
    try:
        import argostranslate.package
    except ImportError:
        print("Install the optional dependency first: pip install -r requirements-translation.txt", file=sys.stderr)
        return 2

    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    requested = []
    for raw in args.pairs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            source, target = raw.split("-", 1)
        except ValueError:
            print(f"Invalid language pair: {raw}", file=sys.stderr)
            return 2
        package = next((p for p in available if p.from_code == source and p.to_code == target), None)
        if package is None:
            print(f"No direct Argos package is available for {source} -> {target}; skipped.")
            continue
        requested.append((source, target, package))

    if not requested:
        print("No requested language packages were found.")
        return 1

    for source, target, package in requested:
        print(f"Downloading/installing offline translation {source} -> {target}...")
        path = package.download()
        argostranslate.package.install_from_path(path)
    print("Done. Translation packs are installed locally; runtime translation does not need the internet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
