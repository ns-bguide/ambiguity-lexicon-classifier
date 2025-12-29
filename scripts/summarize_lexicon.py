"""Summarize and preview a consolidated lexicon Parquet file.

Usage examples:
  - By language code (uses default path):
      .venv/bin/python scripts/summarize_lexicon.py --lang-code en

  - By explicit path:
      .venv/bin/python scripts/summarize_lexicon.py --input data/processed/lexicon_en.parquet

Optional flags:
    --top-n 20         Number of top tokens by frequency to show (default: 20)
    --sample-n 20      Number of random samples to show (default: 20)
    --output-dir       Write CSV previews to a directory (schema, head, top, sample)
    --inputs           Repeatable: explicit Parquet paths (can appear multiple times)
    --lang-codes       Comma-separated list of language codes to summarize
    --verbose          Enable debug logging
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s - %(message)s")


def default_lexicon_path(lang_code: str) -> Path:
    return Path(f"data/processed/lexicon_{lang_code}.parquet")


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path)
    if "token" not in df:
        raise ValueError("Expected a 'token' column in the lexicon Parquet file")
    return df


def print_schema(df: pd.DataFrame) -> None:
    print("\n== Schema ==")
    dtypes = df.dtypes.astype(str)
    for col, dtype in dtypes.items():
        print(f"{col}: {dtype}")


def summarize_basic(df: pd.DataFrame) -> None:
    print("\n== Basic Stats ==")
    print(f"rows: {len(df):,}")
    print(f"unique tokens: {df['token'].nunique():,}")

    # Missing counts
    missing = df.isna().sum().sort_values(ascending=False)
    print("\nMissing values (top 10):")
    print(missing.head(10).to_string())

    # Numeric summaries
    numeric_cols = [c for c in [
        "freq_wordfreq", "len_token", "n_lexicons", "wordfreq_rank"
    ] if c in df]
    if numeric_cols:
        print("\n== Numeric Describe ==")
        print(df[numeric_cols].describe().to_string())


def summarize_flags(df: pd.DataFrame) -> None:
    print("\n== Source Flags ==")
    for col in ["in_wordfreq", "in_hunspell", "in_wiktionary", "in_extra_sources"]:
        if col in df:
            counts = df[col].value_counts(dropna=False)
            total = int(counts.sum()) if not counts.empty else 0
            true_count = int(counts.get(True, 0))
            false_count = int(counts.get(False, 0))
            print(f"{col}: True={true_count:,} False={false_count:,} Total={total:,}")

    if "n_lexicons" in df:
        vc = df["n_lexicons"].value_counts().sort_index()
        friendly = ", ".join(f"{int(k)}: {int(v):,}" for k, v in vc.items())
        print(f"n_lexicons distribution: {friendly}")


def top_tokens(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    cols = [c for c in [
        "token", "freq_wordfreq", "n_lexicons",
        "in_wordfreq", "in_hunspell", "in_wiktionary", "in_extra_sources"
    ] if c in df]
    top = df.sort_values(by=["freq_wordfreq", "token"], ascending=[False, True])
    return top[cols].head(top_n)


def pos_summary(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if "pos_set" not in df:
        return pd.DataFrame(columns=["pos", "count"])  # empty
    counts: Dict[str, int] = {}
    for val in df["pos_set"].dropna().values:
        if isinstance(val, (list, set, tuple)):
            items = [str(x) for x in val]
        else:
            s = str(val)
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items = [str(x) for x in parsed]
                else:
                    items = [s]
            except json.JSONDecodeError:
                items = [x.strip() for x in s.split(",") if x.strip()]
        for tag in items:
            if not tag:
                continue
            counts[tag] = counts.get(tag, 0) + 1
    ser = pd.Series(counts).sort_values(ascending=False)
    return ser.head(top_n).rename_axis("pos").reset_index(name="count")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize a lexicon Parquet file")
    p.add_argument("--input", type=str, default=None, help="Path to Parquet file")
    p.add_argument("--inputs", action="append", help="Repeatable explicit Parquet paths (overrides --input)")
    p.add_argument("--lang-code", type=str, default=None, help="Language code to infer default path")
    p.add_argument("--lang-codes", type=str, default=None, help="Comma-separated language codes to summarize")
    p.add_argument("--top-n", type=int, default=20, help="Top-N tokens to display by frequency")
    p.add_argument("--sample-n", type=int, default=20, help="Random sample size to display")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    p.add_argument("--output-dir", type=str, default=None, help="Optional directory to write CSV previews")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def _infer_lang_from_path(path: Path) -> Optional[str]:
    m = re.match(r"lexicon_([A-Za-z0-9\-]+)\.parquet$", path.name)
    return m.group(1) if m else None


def _plan_targets(args: argparse.Namespace) -> Sequence[Tuple[Optional[str], Path]]:
    targets: list[Tuple[Optional[str], Path]] = []
    if args.inputs:
        for s in args.inputs:
            p = Path(s)
            lang = _infer_lang_from_path(p)
            targets.append((lang, p))
        return targets
    if args.lang_codes:
        for code in [c.strip() for c in args.lang_codes.split(",") if c.strip()]:
            targets.append((code, default_lexicon_path(code)))
        return targets
    if args.input:
        p = Path(args.input)
        targets.append((_infer_lang_from_path(p), p))
        return targets
    if args.lang_code:
        p = default_lexicon_path(args.lang_code)
        targets.append((args.lang_code, p))
        return targets
    raise SystemExit("Provide --inputs, --lang-codes, or one of --input/--lang-code to locate lexicon files")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    targets = _plan_targets(args)
    for idx, (lang, parquet_path) in enumerate(targets, start=1):
        label = lang or parquet_path.name
        LOGGER.info("[%d/%d] Loading lexicon from %s", idx, len(targets), parquet_path)
        df = load_parquet(parquet_path)

        print(f"\n==================== {label} ====================")
        print_schema(df)
        summarize_basic(df)
        summarize_flags(df)

        # Display top tokens by frequency
        top = top_tokens(df, args.top_n)
        print(f"\n== Top {len(top)} tokens by freq_wordfreq ==")
        print(top.to_string(index=False))

        # POS summary
        pos_df = pos_summary(df, args.top_n)
        if not pos_df.empty:
            print(f"\n== Top {len(pos_df)} POS tags ==")
            print(pos_df.to_string(index=False))

        # Random sample
        if args.sample_n > 0:
            sample_cols = [c for c in [
                "token", "freq_wordfreq", "len_token", "n_lexicons",
                "in_wordfreq", "in_hunspell", "in_wiktionary", "in_extra_sources", "pos_set"
            ] if c in df]
            sample_df = df[sample_cols].sample(n=min(args.sample_n, len(df)), random_state=args.seed)
            print(f"\n== Random sample ({len(sample_df)}) ==")
            print(sample_df.to_string(index=False))

        # Optional CSV outputs
        if args.output_dir:
            outdir = Path(args.output_dir)
            subdir = outdir / (label if lang else parquet_path.stem)
            subdir.mkdir(parents=True, exist_ok=True)
            top.to_csv(subdir / "top_tokens.csv", index=False)
            if not pos_df.empty:
                pos_df.to_csv(subdir / "pos_summary.csv", index=False)
            df.head(50).to_csv(subdir / "head.csv", index=False)
            if args.sample_n > 0:
                sample_df.to_csv(subdir / "sample.csv", index=False)
            print(f"\nWrote CSV previews to {subdir}")


if __name__ == "__main__":
    main()
