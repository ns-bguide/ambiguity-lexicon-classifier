#!/usr/bin/env python3
"""CLI to run regex stress tests across languages' lexicon tokens.

Examples:
  .venv/bin/python scripts/regex_stress_test.py --pattern '^[A-Z][a-z]+' --languages en,pt --mode search --ignore-case --limit-per-lang 200000 --max-matches-per-lang 1000 --output results.json
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure local src is importable when running as script
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from regex_tester.stress_tester import RegexStressTester  # type: ignore

LOGGER = logging.getLogger("regex_stress_test")


def write_output(matches_by_lang: dict, stats: list, output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        payload = {
            "stats": [s.__dict__ for s in stats],
            "matches": matches_by_lang,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "jsonl":
        with output.open("w", encoding="utf-8") as f:
            for s in stats:
                rec = {"type": "stats", **s.__dict__}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            for lang, toks in matches_by_lang.items():
                for t in toks:
                    if isinstance(t, dict):
                        rec = {"type": "match", "lang": lang, **t}
                    else:
                        rec = {"type": "match", "lang": lang, "token": t}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:  # csv
        # Determine metadata columns if present
        meta_cols: list[str] = []
        for toks in matches_by_lang.values():
            for t in toks:
                if isinstance(t, dict):
                    meta_cols = sorted([k for k in t.keys() if k != "token"])
                break
            if meta_cols:
                break
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lang", "token", *meta_cols])
            for lang, toks in matches_by_lang.items():
                for t in toks:
                    if isinstance(t, dict):
                        row = [lang, t.get("token")]
                        row.extend([t.get(c) for c in meta_cols])
                        writer.writerow(row)
                    else:
                        writer.writerow([lang, t, *([None] * len(meta_cols))])


def main() -> int:
    p = argparse.ArgumentParser(description="Regex stress testing across language lexicons")
    p.add_argument("--pattern", required=True, help="Regex pattern to test")
    p.add_argument("--languages", help="Comma-separated list of lang codes; default discovers all lexicons")
    p.add_argument("--mode", choices=["search", "fullmatch"], default="search", help="Use re.search or re.fullmatch")
    p.add_argument("--ignore-case", action="store_true", help="Use case-insensitive matching")
    p.add_argument("--limit-per-lang", type=int, help="Limit number of tokens scanned per language")
    p.add_argument("--max-matches-per-lang", type=int, help="Stop after N matches per language")
    p.add_argument("--timeout-sec-per-lang", type=float, help="Stop scanning a language after this many seconds")
    p.add_argument("--sample-k", type=int, default=20, help="Number of sample matches to report per language in summary")
    p.add_argument("--output", help="Output file path (.csv, .json, or .jsonl)")
    p.add_argument("--include-metadata", action="store_true", help="Include matched token metadata in outputs")
    p.add_argument(
        "--metadata-fields",
        help="Comma-separated fields to include when --include-metadata is set (default: freq_wordfreq,n_lexicons,len_token)",
    )
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    langs = args.languages.split(",") if args.languages else None
    tester = RegexStressTester()
    fields = None
    if args.metadata_fields:
        fields = [s for s in args.metadata_fields.split(",") if s]

    stats, matches = tester.run(
        pattern=args.pattern,
        languages=langs,
        mode=args.mode,
        ignore_case=args.ignore_case,
        limit_per_lang=args.limit_per_lang,
        max_matches_per_lang=args.max_matches_per_lang,
        timeout_sec_per_lang=args.timeout_sec_per_lang,
        sample_k=args.sample_k,
        include_metadata=bool(args.include_metadata),
        metadata_fields=fields,
    )

    # Print concise summary
    for s in stats:
        LOGGER.info(
            "lang=%s scanned=%d matched=%d elapsed=%.3fs samples=%s",
            s.lang,
            s.scanned,
            s.matched,
            s.elapsed_sec,
            ", ".join(s.samples[:min(5, len(s.samples))]),
        )

    if args.output:
        fmt = "csv"
        if args.output.endswith(".json"):
            fmt = "json"
        elif args.output.endswith(".jsonl"):
            fmt = "jsonl"
        write_output(matches, stats, Path(args.output), fmt)
        LOGGER.info("Wrote %d languages to %s", len(matches), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
