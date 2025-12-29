#!/usr/bin/env python3
"""CLI to score terms for ambiguity using heuristic model.

Usage:
  .venv/bin/python scripts/score_terms.py --lang-code en --input path/to/wordlist.txt --output results.csv
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Allow running as a script without installed package
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ambiguity.scorer import TermAmbiguityScorer  # type: ignore

LOGGER = logging.getLogger("score_terms")


def read_terms(path: Path) -> list[str]:
    terms: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            terms.append(t)
    return terms


def write_output(results: list[dict], output: Path, fmt: str, profile: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        # Support multiple output profiles: minimal | interpretable | full
        if profile == "minimal":
            fieldnames = [
                "term",
                "label",
                "score",
            ]
        elif profile == "full":
            fieldnames = [
                "term",
                "label",
                "score",
                # Winner summary
                "winner_signals",
                "winner_contrib",
                # Numeric context
                "freq_wordfreq",
                "freq_log",
                "n_lexicons",
                "len_token",
                "in_wordfreq",
                "in_hunspell",
                "in_wiktionary",
                "wiki_entries_count",
                "wiki_single_entry_page",
                "wiki_page_views_30d",
                "wiki_total_edits",
                # Signals (booleans)
                "sig_freq_common",
                "sig_nlex_common",
                "sig_in_wordfreq",
                "sig_wiki_entries",
                "sig_wiki_views",
                "sig_wiki_edits",
                "sig_freq_rare",
                "sig_nlex_zero",
                "sig_len_long",
                "sig_single_entry",
                # Per-class weighted scores
                "score_ambiguous",
                "score_review",
                "score_unambiguous",
            ]
        else:
            # interpretable (default): simplified schema with winner signals and minimal numeric context
            fieldnames = [
                "term",
                "label",
                "score",
                # Winner summary
                "winner_signals",
                "winner_contrib",
                # Minimal numeric context for tuning
                "freq_wordfreq",
                "n_lexicons",
                "len_token",
                "wiki_entries_count",
                "wiki_page_views_30d",
                "wiki_total_edits",
            ]
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                dbg = r.get("debug", {})
                scores = dbg.get("scores", {}) or {}
                signals = dbg.get("signals", {}) or {}

                # Winner summary: list signals that contributed to the chosen label
                label = r.get("label")
                winner_keys: list[str] = []
                if label == "likely ambiguous":
                    winner_keys = ["freq_common", "nlex_common", "in_wordfreq", "wiki_entries"]
                elif label == "need review":
                    winner_keys = ["wiki_views", "wiki_edits"]
                elif label == "likely unambiguous":
                    winner_keys = ["freq_rare", "nlex_zero", "len_long", "single_entry"]
                active = [k for k in winner_keys if signals.get(k)]
                winner_signals = "|".join(active) if active else ""
                winner_contrib = (
                    scores.get("review") if label == "need review" else scores.get("ambiguous") if label == "likely ambiguous" else scores.get("unambiguous")
                )
                if profile == "minimal":
                    d = {
                        "term": r.get("term"),
                        "label": label,
                        "score": r.get("score"),
                    }
                elif profile == "full":
                    d = {
                        "term": r.get("term"),
                        "label": label,
                        "score": r.get("score"),
                        "winner_signals": winner_signals,
                        "winner_contrib": winner_contrib,
                        # Numeric context
                        "freq_wordfreq": dbg.get("freq_wordfreq"),
                        "freq_log": dbg.get("freq_log"),
                        "n_lexicons": dbg.get("n_lexicons"),
                        "len_token": dbg.get("len_token"),
                        "in_wordfreq": dbg.get("in_wordfreq"),
                        "in_hunspell": dbg.get("in_hunspell"),
                        "in_wiktionary": dbg.get("in_wiktionary"),
                        "wiki_entries_count": dbg.get("wiki_entries_count"),
                        "wiki_single_entry_page": dbg.get("wiki_single_entry_page"),
                        "wiki_page_views_30d": dbg.get("wiki_page_views_30d"),
                        "wiki_total_edits": dbg.get("wiki_total_edits"),
                        # Signals
                        "sig_freq_common": signals.get("freq_common"),
                        "sig_nlex_common": signals.get("nlex_common"),
                        "sig_in_wordfreq": signals.get("in_wordfreq"),
                        "sig_wiki_entries": signals.get("wiki_entries"),
                        "sig_wiki_views": signals.get("wiki_views"),
                        "sig_wiki_edits": signals.get("wiki_edits"),
                        "sig_freq_rare": signals.get("freq_rare"),
                        "sig_nlex_zero": signals.get("nlex_zero"),
                        "sig_len_long": signals.get("len_long"),
                        "sig_single_entry": signals.get("single_entry"),
                        # Scores
                        "score_ambiguous": scores.get("ambiguous"),
                        "score_review": scores.get("review"),
                        "score_unambiguous": scores.get("unambiguous"),
                    }
                else:
                    # interpretable
                    d = {
                        "term": r.get("term"),
                        "label": label,
                        "score": r.get("score"),
                        "winner_signals": winner_signals,
                        "winner_contrib": winner_contrib,
                        # Minimal numeric context
                        "freq_wordfreq": dbg.get("freq_wordfreq"),
                        "n_lexicons": dbg.get("n_lexicons"),
                        "len_token": dbg.get("len_token"),
                        "wiki_entries_count": dbg.get("wiki_entries_count"),
                        "wiki_page_views_30d": dbg.get("wiki_page_views_30d"),
                        "wiki_total_edits": dbg.get("wiki_total_edits"),
                    }
                writer.writerow(d)
    else:
        with output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score terms for ambiguity")
    parser.add_argument("--lang-code", default="en", help="Language code, e.g., en")
    parser.add_argument("--input", required=True, help="Path to input wordlist (one term per line)")
    parser.add_argument("--output", required=True, help="Path to output file (csv or json)")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format")
    parser.add_argument(
        "--output-profile",
        choices=["minimal", "interpretable", "full"],
        default="interpretable",
        help="CSV schema profile: minimal | interpretable | full",
    )
    parser.add_argument("--lexicon", help="Optional path to a custom processed lexicon Parquet file")
    parser.add_argument("--config", help="Optional JSON config file with 'thresholds' and 'weights' sections")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # Threshold overrides
    parser.add_argument("--freq-log-common", type=float, help="Override freq_log threshold for common terms (default: -7.0)")
    parser.add_argument("--freq-log-rare", type=float, help="Override freq_log threshold for rare terms (default: -11.5)")
    parser.add_argument("--n-lexicons-common", type=int, help="Override minimum lexicon count to consider common (default: 2)")
    parser.add_argument("--long-token-len", type=int, help="Override token length threshold for long tokens (default: 14)")
    parser.add_argument("--wiki-entries-ambiguous-min", type=int, help="Minimum wiki entries to treat as likely ambiguous (default: 2)")
    parser.add_argument("--wiki-page-views-review-min", type=int, help="Minimum page views to flag need review (default: 1000)")
    parser.add_argument("--wiki-total-edits-review-min", type=int, help="Minimum total edits to flag need review (default: 100)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        LOGGER.error("Input file does not exist: %s", input_path)
        return 2

    thresholds = {}
    weights = {}
    # Load config JSON if provided
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            LOGGER.error("Config file does not exist: %s", cfg_path)
            return 2
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                if isinstance(cfg.get("thresholds"), dict):
                    thresholds.update(cfg["thresholds"])  # type: ignore[arg-type]
                if isinstance(cfg.get("weights"), dict):
                    weights.update(cfg["weights"])  # type: ignore[arg-type]
        except Exception as e:
            LOGGER.error("Failed to parse config: %s", e)
            return 2
    if args.freq_log_common is not None:
        thresholds["freq_log_common"] = args.freq_log_common
    if args.freq_log_rare is not None:
        thresholds["freq_log_rare"] = args.freq_log_rare
    if args.n_lexicons_common is not None:
        thresholds["n_lexicons_common"] = args.n_lexicons_common
    if args.long_token_len is not None:
        thresholds["long_token_len"] = args.long_token_len
    if args.wiki_entries_ambiguous_min is not None:
        thresholds["wiki_entries_ambiguous_min"] = args.wiki_entries_ambiguous_min
    if args.wiki_page_views_review_min is not None:
        thresholds["wiki_page_views_review_min"] = args.wiki_page_views_review_min
    if args.wiki_total_edits_review_min is not None:
        thresholds["wiki_total_edits_review_min"] = args.wiki_total_edits_review_min

    terms = read_terms(input_path)

    # Allow overriding the lexicon path when provided
    if args.lexicon:
        from language_resources import LanguageResources  # type: ignore
        from ambiguity.scorer import AmbiguityModelEN  # type: ignore
        resources = LanguageResources(args.lang_code, lexicon_path=args.lexicon)
        model = AmbiguityModelEN(resources, thresholds if thresholds else None, weights if weights else None)
        results = model.score_terms(terms)
    else:
        scorer = TermAmbiguityScorer(args.lang_code, thresholds=thresholds if thresholds else None, weights=weights if weights else None)
        results = scorer.score_terms(terms)
    write_output(results, output_path, args.format, args.output_profile)
    LOGGER.info("Wrote %d results to %s", len(results), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
