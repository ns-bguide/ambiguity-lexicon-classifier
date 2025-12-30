# Ambiguity Toolkit

Multilingual toolkit for building consolidated lexical resources, scoring term ambiguity, and stress-testing regular expressions. The initial prototype targets English while keeping the architecture extensible to additional languages.

## Features
- Consolidates Wordfreq, Hunspell, Wiktionary, and optional custom wordlists into language-specific lexicons stored as Parquet.
- Provides a `LanguageResources` abstraction that exposes normalized lexical features for downstream models.
- `TermAmbiguityScorer` with an English heuristic model producing three labels: "likely ambiguous", "likely unambiguous", and "need review".
- `RegexStressTester` for evaluating regex patterns against consolidated lexicons across languages.
 - Scorer leverages enriched Wiktionary signals (page views, edits, entries count) and treats missing wordfreq as neutral, so strong common signals correctly classify common English words as "likely ambiguous".

## Requirements
- Python 3.11+
- Language resources organised under `data/raw/<lang_code>/`:
   - Hunspell dictionaries (`*.aff`, `*.dic`) in `data/raw/<lang_code>/hunspell/`
   - Optional custom wordlists (`*.txt`, one token per line) in `data/raw/<lang_code>/txtfiles/`
   - Wiktionary JSONL dump (defaults to `data/raw/wiktionary/wiktionary_full.jsonl` or drop-in per-language files)
 - Language normalization: `langcodes[data]` is used to robustly map language names to ISO codes. It is installed automatically via project dependencies.

Data download:
- Prebuilt raw data for this project can be downloaded from Google Drive: https://drive.google.com/drive/folders/1W0DZIFcstRf9-rNPpMofFlnJJ-oYg46P?usp=drive_link
- The Drive folder mirrors the expected `data/raw/` structure. You can place its contents under `data/raw/` to use directly with the scripts here.

### Wiktionary Enrichment

Wiktionary data used here is enriched via the Wikimedia API to include page-level metrics and per-language entry details. The builder consumes these signals and exposes them as `wiki_page_views_30d`, `wiki_total_edits`, `wiki_entries_count`, and a derived `wiki_single_entry_page` flag (from hidden categories like "Category:Pages with 1 entry"). A typical JSONL row looks like:

```json
{
   "word": "free",
   "languages": ["English", "Galician", "Low German", "English"],
   "tags": ["Adjective", "Adverb", "Noun", "Verb", "Verb", "Adjective", "Proper noun"],
   "entries": [
      {"language": "English", "tags": ["Adjective", "Adverb", "Noun", "Verb"]},
      {"language": "Galician", "tags": ["Verb"]},
      {"language": "Low German", "tags": ["Adjective"]},
      {"language": "English", "tags": ["Proper noun"]}
   ],
   "_pageid": 19,
   "_title": "free",
   "page_views_30d": 14637,
   "total_edits": 1472,
   "recent_authors_30d": 3,
   "hidden_categories": [
      "Category:Pages with 3 entries",
      "Category:Pages with entries"
      // ... many more per-page maintenance categories
   ]
}
```

The builder counts `entries` that match the target language to compute `wiki_entries_count` and maps page-level metrics to the consolidated lexicon for downstream scoring.

## Quickstart

1. **Create and populate a virtual environment**
   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install -e .[dev]
   ```
   Optional: install Hunspell bindings after native prerequisites are present via
   `.venv/bin/python -m pip install -e .[hunspell]`.

2. **Build the English lexicon**
   ```bash
   .venv/bin/python scripts/build_lexicon.py --lang-code en --output data/processed/lexicon_en.parquet
   ```
   Useful flags:
   - `--wordfreq-top-n`: limit the number of wordfreq tokens (default `1500000`).
   - `--extra-txt-dir`: override the directory containing `*.txt` custom lists.
   - `--verbose`: emit debug logging during ingestion.

   The generated Parquet file is the default input for `LanguageResources("en")`.

3. **Score a wordlist for ambiguity**
    - Prepare a file with one term per line (example):
       ```bash
       printf 'the\nbank\nzebra\nAmsterdam\nKubernetes\npneumonoultramicroscopicsilicovolcanoconiosis\n' > data/raw/en/sample_terms.txt
       ```
    - Run the scorer (CSV output):
       ```bash
       .venv/bin/python scripts/score_terms.py \
          --lang-code en \
          --input data/raw/en/sample_terms.txt \
          --output data/processed/amb_scores_en.csv \
          --format csv \
          --output-profile interpretable \
          --verbose
       ```
    - JSON output (alternative):
       ```bash
       .venv/bin/python scripts/score_terms.py \
          --lang-code en \
          --input data/raw/en/sample_terms.txt \
          --output data/processed/amb_scores_en.json \
          --format json
       ```

   Simplified CSV (interpretable profile) includes: `term, label, score, winner_signals, winner_contrib, freq_wordfreq, n_lexicons, len_token, wiki_entries_count, wiki_page_views_30d, wiki_total_edits`.
   - `winner_signals`: concise list of active signals that contributed to the chosen class (e.g., `nlex_common|in_wordfreq`, `freq_rare|len_long`).
   - `winner_contrib`: weighted contribution score for the winning class (useful for tuning weights).
   - Numeric context: minimal values to guide threshold/weight adjustments.

   Output profiles (`--output-profile` when `--format csv`):
   - `minimal`: `term, label, score` only.
   - `interpretable` (default): simplified schema with winner signals and minimal numeric context.
   - `full`: adds raw features, all signal booleans, and per-class weighted scores.

### Winner Signals Guide

- freq_common: higher `freq_wordfreq` (above `freq_log_common`) contributes to ambiguous.
- nlex_common: present in ≥ `n_lexicons_common` sources contributes to ambiguous.
- in_wordfreq: token appears in Wordfreq; contributes to ambiguous.
- wiki_entries: `wiki_entries_count` ≥ `wiki_entries_ambiguous_min`; contributes to ambiguous.
- wiki_views: `wiki_page_views_30d` ≥ `wiki_page_views_review_min`; contributes to need review.
- wiki_edits: `wiki_total_edits` ≥ `wiki_total_edits_review_min`; contributes to need review.
- freq_rare: very low frequency (below `freq_log_rare`); contributes to unambiguous.
- nlex_zero: absent from all sources; contributes to unambiguous.
- len_long: token length ≥ `long_token_len`; contributes to unambiguous.
- single_entry: Wiktionary page flagged single-entry; contributes to unambiguous.

Tuning tips:
- Adjust thresholds and weights via `--config configs/en_model_config.json`.
- Increasing a signal’s weight amplifies its influence on the chosen class.
- Use `winner_signals` and `winner_contrib` to validate and refine your settings.

   Use a custom lexicon (e.g., enriched Wiktionary fields) via:
    ```bash
    .venv/bin/python scripts/score_terms.py \
       --lang-code en \
       --lexicon data/processed/lexicon_en_enriched.parquet \
       --input data/raw/en/sample_terms.txt \
       --output data/processed/amb_scores_en_enriched.csv \
       --format csv \
       --output-profile interpretable \
       --config configs/en_model_config.yaml
    ```

4. **Summarize the built lexicon**
    ```bash
    .venv/bin/python scripts/summarize_lexicon.py --lang-code en --top-n 10 --sample-n 10 --verbose
    ```

5. **Run the test suite** (as components are implemented)
   ```bash
   .venv/bin/python -m pytest
   ```

## Regex Stress Tester
- Run a pattern across discovered languages (lexicon_*.parquet):
   ```bash
   .venv/bin/python scripts/regex_stress_test.py \
      --pattern '^[A-Z][a-z]+' \
      --languages en,pt \
      --mode search \
      --ignore-case \
      --limit-per-lang 200000 \
      --max-matches-per-lang 1000 \
      --output data/processed/regex_matches_en_pt.json \
      --verbose
   ```
- Modes and flags:
   - `--mode search|fullmatch`: choose `re.search` vs `re.fullmatch`.
   - `--ignore-case`: Unicode-aware case-insensitive matching.
   - `--limit-per-lang`: cap scanned tokens per language.
   - `--max-matches-per-lang`: stop after N matches per language.
   - `--timeout-sec-per-lang`: per-language time budget.
   - Output formats: `.csv`, `.json`, `.jsonl`.
 - Metadata:
    - `--include-metadata`: include per-token metadata in outputs.
    - `--metadata-fields freq_wordfreq,n_lexicons,len_token` (default): choose fields to include.
    - When using injected token sources (tests), only `len_token` can be computed reliably; other fields are `null`.

## Configuration
- Threshold overrides (via scorer CLI):
   - `--freq-log-common`: log-frequency threshold for common terms (default: -7.0)
   - `--freq-log-rare`: log-frequency threshold for rare terms (default: -11.5)
   - `--n-lexicons-common`: minimum number of lexicons to count as common (default: 2)
   - `--long-token-len`: token length threshold signaling likely unambiguous technical/proper terms (default: 14)
 - Config (YAML): use `--config` with a YAML file to set both `thresholds` and `weights`.
    - Thresholds: `freq_log_common`, `freq_log_rare`, `n_lexicons_common`, `long_token_len`, `wiki_entries_ambiguous_min`, `wiki_page_views_review_min`, `wiki_total_edits_review_min`, `wiki_entries_unambiguous_max`.
    - Weights: per-class signal weights for `ambiguous` (`freq_common`, `nlex_common`, `in_wordfreq`, `wiki_entries`), `review` (`wiki_views`, `wiki_edits`), and `unambiguous` (`freq_rare`, `nlex_zero`, `len_long`, `single_entry`).
    - Example config: YAML [configs/en_model_config.yaml](configs/en_model_config.yaml).

 YAML usage example:
 ```bash
 .venv/bin/python scripts/score_terms.py \
    --lang-code en \
    --input data/raw/en/sample_terms.txt \
    --output data/processed/amb_scores_en.csv \
    --format csv \
    --config configs/en_model_config.yaml
 ```
 
 Heuristic notes:
 - Strong common signals include: `n_lexicons >= 2`, membership in `wordfreq`, `wiki_page_views_30d >= 1000`, `wiki_total_edits >= 100`, or `wiki_entries_count >= 2`.
 - Rare signals (low frequency, single-entry page) are gated and do not override strong common signals. Very long tokens still lean "likely unambiguous".

Example with overrides:
```bash
.venv/bin/python scripts/score_terms.py \
   --lang-code en \
   --input data/raw/en/sample_terms.txt \
   --output data/processed/amb_scores_en.csv \
   --format csv \
   --freq-log-common -6.5 \
   --n-lexicons-common 3
```

## Development Workflow
- Package source code lives under `src/` with subpackages for language resources, ambiguity models, and regex stress testing.
- Dataset build scripts reside in `scripts/`. Additional languages should follow the same pattern (`build_lexicon_<lang>.py`).
- Processed artifacts live under `data/processed/`; raw inputs stay in `data/raw/`.
- Follow the project plan in `untitled:plan-pythonAmbiguityToolkit.prompt.md` for the current implementation roadmap.

## Troubleshooting
- If the lexicon build fails, ensure the Hunspell and Wiktionary files exist and are readable, and that network access is available for the initial `wordfreq` download.
- If language name → code normalization fails or logs that `langcodes` is missing, install it explicitly:
   ```bash
   .venv/bin/python -m pip install "langcodes[data]"
   ```
- Hunspell bindings are optional. If you opt in via `.[hunspell]`, install the OS packages (`libhunspell-dev`, dictionary locales) first.
- When running commands in CI or scripts, always call Python via the virtualenv path (`.venv/bin/python`) instead of activating the environment.

---
Questions or issues? Add notes to the plan file or open a ticket before modifying the pipeline.
