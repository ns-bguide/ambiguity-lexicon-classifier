#!/usr/bin/env python3
"""List distinct language names and their ISO codes from wiktionary_full.jsonl.

Usage:
  .venv/bin/python scripts/list_wiktionary_langs.py --input data/raw/wiktionary/wiktionary_full.jsonl --top 50
"""

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from langcodes import Language  # type: ignore
except Exception:  # pragma: no cover
    Language = None  # type: ignore


def normalize_lang_name(name: str) -> str:
    return name.strip()


def to_code(name: str) -> str | None:
    if Language is None:
        return None
    try:
        # Prefer name lookup which maps language names to BCP-47 codes
        return Language.find(name).to_tag()
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="List languages present in Wiktionary JSONL with ISO codes")
    p.add_argument("--input", default="data/raw/wiktionary/wiktionary_full.jsonl", help="Path to wiktionary JSONL file")
    p.add_argument("--top", type=int, default=0, help="Show only top N languages by entry count (0 = all)")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        return 2

    counts = Counter()
    names = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            lang_name = obj.get("lang")
            if not isinstance(lang_name, str):
                continue
            lang_name = normalize_lang_name(lang_name)
            code = to_code(lang_name)
            if code is None:
                # Keep track of non-language headers separately
                names[lang_name] += 1
                continue
            counts[code] += 1

    items = counts.most_common()
    if args.top > 0:
        items = items[: args.top]

    for code, n in items:
        print(f"{code}\t{n}")

    if not items:
        print("No language-like entries found. The dump may not contain expected fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
