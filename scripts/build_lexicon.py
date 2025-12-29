"""Build consolidated lexicons for ambiguity scoring workflows.

This script merges vocabularies from Wordfreq, Hunspell, Wiktionary, and optional
custom text wordlists (one token per line) into a unified Parquet artifact
consumed by :class:`LanguageResources`.

Usage
-----
python scripts/build_lexicon.py --lang-code en --output data/processed/lexicon_en.parquet

The script expects the following inputs to be present:
* Wordfreq (downloaded automatically by the library on first use)
* Hunspell dictionary files placed under data/raw/<LANG_CODE>/hunspell/
* Wiktionary dump (defaults to data/raw/wiktionary/wiktionary_full.jsonl)
* Optional custom text files (*.txt) under data/raw/<LANG_CODE>/txtfiles/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

import pandas as pd
from wordfreq import top_n_list, word_frequency

try:  # Allow execution without installing the package in editable mode.
    from language_resources.hunspell_parser import load_hunspell_vocabulary
except ModuleNotFoundError:  # pragma: no cover - runtime convenience
    sys.path.append(str((Path(__file__).resolve().parents[1] / "src")))
    from language_resources.hunspell_parser import load_hunspell_vocabulary

LOGGER = logging.getLogger(__name__)

# Optional language name → code mapper (best-effort). If langcodes[data] is installed,
# we'll use that; otherwise fall back to simple heuristics + common-name mapping.
try:  # pragma: no cover - optional dependency at runtime
    from langcodes import Language  # type: ignore
except Exception:  # pragma: no cover - keep optional
    Language = None  # type: ignore


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s - %(message)s")


def _ensure_record(records: Dict[str, Dict[str, Any]], token: str) -> Dict[str, Any]:
    record = records.get(token)
    if record is None:
        record = {
            "token": token,
            "lemma": None,
            "len_token": len(token),
            "freq_wordfreq": 0.0,
            "in_hunspell": False,
            "in_wiktionary": False,
            "in_wordfreq": False,
            "in_extra_sources": False,
            "pos_set": set(),
            "wordfreq_rank": None,
        }
        records[token] = record
    return record


def load_wordfreq(lang_code: str, top_n: int) -> Dict[str, Dict[str, Any]]:
    LOGGER.info("Loading top %s wordfreq entries for %s", top_n, lang_code)
    wordfreq_records: Dict[str, Dict[str, Any]] = {}
    tokens = top_n_list(lang_code, n=top_n, wordlist="best")
    for rank, token in enumerate(tokens, start=1):
        freq = float(word_frequency(token, lang_code, minimum=0.0))
        record = _ensure_record(wordfreq_records, token)
        record["freq_wordfreq"] = max(record.get("freq_wordfreq", 0.0), freq)
        record["in_wordfreq"] = freq > 0.0
        record["wordfreq_rank"] = rank if record.get("wordfreq_rank") is None else min(record["wordfreq_rank"], rank)
    LOGGER.info("Loaded %s tokens from wordfreq", len(wordfreq_records))
    return wordfreq_records


def load_extra_txt_tokens(directory: Path) -> Set[str]:
    if not directory.exists():
        LOGGER.info("Extra txt directory %s not found; skipping", directory)
        return set()
    tokens: Set[str] = set()
    txt_paths = sorted(directory.glob("*.txt"))
    if not txt_paths:
        LOGGER.info("No .txt files found in %s; skipping", directory)
        return set()
    for path in txt_paths:
        LOGGER.info("Loading custom tokens from %s", path)
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                token = line.strip()
                if token:
                    tokens.add(token)
    LOGGER.info("Loaded %s unique tokens from custom text files", len(tokens))
    return tokens


def _normalize_lang_value_to_code(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # Prefer langcodes if available
    if Language is not None:
        try:
            tag = Language.find(s).to_tag()  # e.g., 'pt', 'pt-PT', 'cmn', 'mul'
            return tag.split("-", 1)[0].lower()
        except Exception:
            pass
    # Simple ISO-like pattern fallback (e.g., 'en', 'pt-BR')
    if re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]+)*", s):
        return s.split("-", 1)[0].lower()
    # Common language-name fallbacks
    mapping = {
        "english": "en",
        "portuguese": "pt",
        "spanish": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "russian": "ru",
        "chinese": "zh",
        "mandarin": "cmn",
        "japanese": "ja",
        "korean": "ko",
        "arabic": "ar",
        "greek": "el",
        "latin": "la",
    }
    return mapping.get(s.lower())


def _entry_matches_language(entry: Dict[str, Any], lang_code: str) -> bool:
    target = lang_code.split("-", 1)[0].lower()
    for key in ("lang_code", "lang", "language_code", "language"):
        code = _normalize_lang_value_to_code(entry.get(key))
        if code is not None and code == target:
            return True
    return False


def _extract_token(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("word", "token", "lemma"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_pos(entry: Dict[str, Any]) -> Iterable[str]:
    pos_candidates = []
    for key in ("pos", "part_of_speech", "pos_tags", "pos_list"):
        value = entry.get(key)
        if isinstance(value, str):
            pos_candidates.extend([item.strip() for item in value.split(",") if item.strip()])
        elif isinstance(value, (list, tuple, set)):
            pos_candidates.extend([str(item).strip() for item in value if str(item).strip()])
    return [candidate for candidate in pos_candidates if candidate]


def load_wiktionary_entries(path: Path, lang_code: str) -> Dict[str, Dict[str, Any]]:
    LOGGER.info("Loading Wiktionary entries from %s", path)
    entries: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "lemma": None,
        "pos": set(),
        # Enriched fields (optional)
        "wiki_page_views_30d": None,
        "wiki_total_edits": None,
        "wiki_entries_count": 0,
        "wiki_single_entry_page": False,
    })
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping malformed Wiktionary line %s: %s", line_no, exc)
                continue
            # Support enriched format: list of per-language entries
            token = _extract_token(entry)
            if not token:
                continue
            store = entries[token]

            # If enriched 'entries' list is present, count those matching target language
            enriched_items = entry.get("entries")
            matched_any = False
            if isinstance(enriched_items, list) and enriched_items:
                count_for_lang = 0
                for it in enriched_items:
                    if not isinstance(it, dict):
                        continue
                    lang_name = it.get("language")
                    if _normalize_lang_value_to_code(lang_name) == lang_code:
                        matched_any = True
                        count_for_lang += 1
                        # POS/tags accumulation from enriched entries
                        tags = it.get("tags")
                        if isinstance(tags, list):
                            store["pos"].update(str(t).strip() for t in tags if str(t).strip())
                # Update count
                store["wiki_entries_count"] = max(store["wiki_entries_count"] or 0, count_for_lang)

                # Enriched page-level metrics
                pv = entry.get("page_views_30d")
                te = entry.get("total_edits")
                if isinstance(pv, int):
                    store["wiki_page_views_30d"] = max(store["wiki_page_views_30d"] or 0, pv)
                if isinstance(te, int):
                    store["wiki_total_edits"] = max(store["wiki_total_edits"] or 0, te)
                # Single-entry page indicator from hidden_categories
                cats = entry.get("hidden_categories")
                if isinstance(cats, list):
                    store["wiki_single_entry_page"] = store["wiki_single_entry_page"] or (
                        "Category:Pages with 1 entry" in cats
                    )

            # Fallback to legacy format using top-level language fields
            if not matched_any and not _entry_matches_language(entry, lang_code):
                continue

            # Lemma fallback from legacy or enriched
            lemma = entry.get("lemma") or entry.get("canonical_form")
            if isinstance(lemma, str) and lemma.strip():
                store["lemma"] = lemma.strip()
            # POS accumulation from legacy keys
            store["pos"].update(_extract_pos(entry))
    LOGGER.info("Loaded %s tokens from Wiktionary", len(entries))
    return entries


def consolidate_records(
    lang_code: str,
    wordfreq_records: Dict[str, Dict[str, Any]],
    hunspell_tokens: Set[str],
    wiktionary_entries: Dict[str, Dict[str, Any]],
    extra_tokens: Set[str],
) -> pd.DataFrame:
    records: Dict[str, Dict[str, Any]] = {}

    for token, wf_record in wordfreq_records.items():
        record = _ensure_record(records, token)
        record.update({
            "freq_wordfreq": wf_record.get("freq_wordfreq", 0.0),
            "in_wordfreq": wf_record.get("in_wordfreq", False),
        })
        if wf_record.get("wordfreq_rank") is not None:
            record["wordfreq_rank"] = wf_record["wordfreq_rank"]

    for token in hunspell_tokens:
        record = _ensure_record(records, token)
        record["in_hunspell"] = True

    for token in extra_tokens:
        record = _ensure_record(records, token)
        record["in_extra_sources"] = True

    for token, wiki_data in wiktionary_entries.items():
        record = _ensure_record(records, token)
        record["in_wiktionary"] = True
        if wiki_data.get("lemma") and not record.get("lemma"):
            record["lemma"] = wiki_data["lemma"]
        record.setdefault("pos_set", set()).update(wiki_data.get("pos", set()))
        # Enriched metrics
        if "wiki_page_views_30d" in wiki_data and wiki_data["wiki_page_views_30d"] is not None:
            record["wiki_page_views_30d"] = int(wiki_data["wiki_page_views_30d"])  # type: ignore[arg-type]
        if "wiki_total_edits" in wiki_data and wiki_data["wiki_total_edits"] is not None:
            record["wiki_total_edits"] = int(wiki_data["wiki_total_edits"])  # type: ignore[arg-type]
        if "wiki_entries_count" in wiki_data and wiki_data["wiki_entries_count"] is not None:
            record["wiki_entries_count"] = int(wiki_data["wiki_entries_count"])  # type: ignore[arg-type]
        if "wiki_single_entry_page" in wiki_data and wiki_data["wiki_single_entry_page"] is not None:
            record["wiki_single_entry_page"] = bool(wiki_data["wiki_single_entry_page"])  # type: ignore[arg-type]

    rows = []
    for token, record in records.items():
        pos_set = record.get("pos_set", set()) or set()
        lemma = record.get("lemma") or None
        freq = float(record.get("freq_wordfreq", 0.0) or 0.0)
        in_hunspell = bool(record.get("in_hunspell", False))
        in_wiktionary = bool(record.get("in_wiktionary", False))
        in_wordfreq = bool(record.get("in_wordfreq", False)) or freq > 0.0
        in_extra = bool(record.get("in_extra_sources", False))
        n_lexicons = int(in_hunspell) + int(in_wiktionary) + int(in_wordfreq) + int(in_extra)
        rows.append(
            {
                "token": token,
                "lemma": lemma,
                "len_token": len(token),
                "freq_wordfreq": freq,
                "in_hunspell": in_hunspell,
                "in_wiktionary": in_wiktionary,
                "in_wordfreq": in_wordfreq,
                "in_extra_sources": in_extra,
                "n_lexicons": n_lexicons,
                "pos_set": json.dumps(sorted(pos_set)) if pos_set else "[]",
                "wordfreq_rank": record.get("wordfreq_rank"),
                # Optional enriched fields
                "wiki_page_views_30d": record.get("wiki_page_views_30d"),
                "wiki_total_edits": record.get("wiki_total_edits"),
                "wiki_entries_count": record.get("wiki_entries_count"),
                "wiki_single_entry_page": record.get("wiki_single_entry_page"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["freq_wordfreq", "token"], ascending=[False, True]).reset_index(drop=True)
    LOGGER.info("Consolidated %s unique tokens for language %s", len(df), lang_code)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated lexicon for a target language")
    parser.add_argument("--lang-code", required=True, help="Language code to process (e.g., en, pt)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output Parquet path (defaults to data/processed/lexicon_<lang>.parquet)",
    )
    parser.add_argument(
        "--wordfreq-top-n",
        type=int,
        default=1_500_000,
        help="Number of most frequent tokens to pull from wordfreq",
    )
    parser.add_argument(
        "--hunspell-dir",
        default=None,
        help="Directory containing Hunspell .dic/.aff files (defaults to data/raw/<lang>/hunspell)",
    )
    parser.add_argument(
        "--wiktionary",
        default=None,
        help="Path to Wiktionary JSONL dump (defaults to data/raw/wiktionary/wiktionary_full.jsonl)",
    )
    parser.add_argument(
        "--extra-txt-dir",
        default=None,
        help="Directory containing *.txt custom token lists (defaults to data/raw/<lang>/txtfiles)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    lang_code = args.lang_code.lower()

    output_path = Path(args.output) if args.output else Path(f"data/processed/lexicon_{lang_code}.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wordfreq_records = load_wordfreq(lang_code, args.wordfreq_top_n)

    base_lang_raw_dir = Path("data/raw") / lang_code

    if args.hunspell_dir:
        hunspell_dir = Path(args.hunspell_dir)
    else:
        hunspell_dir = base_lang_raw_dir / "hunspell"
        if not hunspell_dir.exists():
            legacy_hunspell_dir = Path("data/raw/hunspell")
            if legacy_hunspell_dir.exists():
                LOGGER.info(
                    "Default hunspell directory %s missing; falling back to legacy location %s",
                    hunspell_dir,
                    legacy_hunspell_dir,
                )
                hunspell_dir = legacy_hunspell_dir

    hunspell_tokens = load_hunspell_vocabulary(hunspell_dir)

    wiktionary_path_candidates = []
    if args.wiktionary:
        wiktionary_path_candidates.append(Path(args.wiktionary))
    else:
        wiktionary_path_candidates.extend(
            [
                base_lang_raw_dir / "wiktionary.jsonl",
                base_lang_raw_dir / "wiktionary_full.jsonl",
                Path("data/raw/wiktionary") / f"{lang_code}.jsonl",
                Path("data/raw/wiktionary") / "wiktionary_full.jsonl",
            ]
        )

    wiktionary_path: Optional[Path] = None
    for candidate in wiktionary_path_candidates:
        if candidate.exists():
            wiktionary_path = candidate
            break
    if wiktionary_path is None:
        raise FileNotFoundError(
            "Could not locate Wiktionary dump; specify --wiktionary or place a file in data/raw/<lang>/ or data/raw/wiktionary/."
        )

    wiktionary_entries = load_wiktionary_entries(wiktionary_path, lang_code)

    if args.extra_txt_dir:
        extra_txt_dir = Path(args.extra_txt_dir)
    else:
        extra_txt_dir = base_lang_raw_dir / "txtfiles"
        if not extra_txt_dir.exists():
            legacy_txt_dir = Path("data/raw/txtfiles")
            if legacy_txt_dir.exists():
                LOGGER.info(
                    "Default txt directory %s missing; falling back to legacy location %s",
                    extra_txt_dir,
                    legacy_txt_dir,
                )
                extra_txt_dir = legacy_txt_dir

    extra_tokens = load_extra_txt_tokens(extra_txt_dir)

    df = consolidate_records(
        lang_code,
        wordfreq_records,
        hunspell_tokens,
        wiktionary_entries,
        extra_tokens,
    )

    LOGGER.info("Writing consolidated lexicon to %s", output_path)
    df.to_parquet(output_path, index=False)
    LOGGER.info("Lexicon build complete")


if __name__ == "__main__":
    main()
