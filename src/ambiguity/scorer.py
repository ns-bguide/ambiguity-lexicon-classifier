"""Ambiguity scoring interfaces and English heuristic model.

Labels produced:
- "likely ambiguous": common words (not proper nouns), frequent or present in multiple lexicons.
- "likely unambiguous": strong proper nouns or rare/uncommon tokens.
- "need review": borderline terms (both common and proper signals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from language_resources import LanguageResources

LOGGER = logging.getLogger(__name__)


class BaseAmbiguityModel:
    """Abstract interface for language-specific ambiguity models."""

    def score_terms(self, terms: Sequence[str]) -> List[dict]:
        """Score a batch of terms.

        Concrete subclasses will return a list of dictionaries containing
        the term, its label, and associated metadata.
        """

        raise NotImplementedError


class AmbiguityModelEN(BaseAmbiguityModel):
    """English heuristic ambiguity model leveraging consolidated lexicon features.

    Heuristic rationale:
    - Frequency proxy: `freq_wordfreq` (log-transformed) as commonness signal.
    - Lexicon membership: `n_lexicons` and flags indicate common usage.
    - Token length: unusually long tokens tend to be proper/unambiguous.
    """

    def __init__(self, resources: LanguageResources, thresholds: Dict | None = None, weights: Dict | None = None) -> None:
        self.resources = resources
        # Tunable thresholds (documented, non-magic) with sane defaults.
        defaults = {
            "freq_log_common": -7.0,   # log(max(freq, 1e-12)); higher => likely ambiguous
            "freq_log_rare": -11.5,    # very low frequency => likely unambiguous
            "n_lexicons_common": 2,    # present in >=2 sources => common
            "long_token_len": 14,      # very long tokens lean unambiguous/technical
            "short_token_len_max": 3,  # very short tokens (<=3) lean ambiguous
            "alpha_token_len_max": 3,  # alphabetic-only length (<=3) lean ambiguous
            # Enriched Wiktionary heuristics (intuitions)
            "wiki_entries_ambiguous_min": 2,      # more entries => likely ambiguous
            "wiki_page_views_review_min": 1000,   # more views => probably needs review
            "wiki_total_edits_review_min": 100,   # more edits => probably needs review
            "wiki_entries_unambiguous_max": 1,    # single entry suggests lower ambiguity
            # Non-alphabetic ratio removed (no longer used)
        }
        cfg = {**defaults, **(thresholds or {})}
        self.freq_log_common = float(cfg["freq_log_common"]) 
        self.freq_log_rare = float(cfg["freq_log_rare"]) 
        self.n_lexicons_common = int(cfg["n_lexicons_common"]) 
        self.long_token_len = int(cfg["long_token_len"]) 
        self.short_token_len_max = int(cfg["short_token_len_max"]) 
        self.alpha_token_len_max = int(cfg["alpha_token_len_max"]) 
        self.wiki_entries_ambiguous_min = int(cfg["wiki_entries_ambiguous_min"]) 
        self.wiki_page_views_review_min = int(cfg["wiki_page_views_review_min"]) 
        self.wiki_total_edits_review_min = int(cfg["wiki_total_edits_review_min"]) 
        self.wiki_entries_unambiguous_max = int(cfg["wiki_entries_unambiguous_max"]) 
        # nonalpha_ratio threshold removed

        # Weights for signals; allow external override via config
        default_weights = {
            "ambiguous": {
                "freq_common": 1.0,
                "nlex_common": 1.0,
                "in_wordfreq": 0.5,
                "wiki_entries": 1.0,
                "len_short": 1.0,
                "len_alpha": 1.0,
            },
            "review": {
                "wiki_views": 1.0,
                "wiki_edits": 1.0,
            },
            "unambiguous": {
                "freq_rare": 1.0,
                "nlex_zero": 0.5,
                "len_long": 1.0,
                "single_entry": 0.5,
            },
        }
        self.weights = default_weights if weights is None else _merge_nested(default_weights, weights)

    def _score_one(self, term: str) -> Dict:
        entry = self.resources.get_entry(term)
        freq = self.resources.freq(term) if entry is None else entry.freq_wordfreq
        in_wordfreq = False if entry is None else bool(entry.in_wordfreq)
        n_lexicons = 0 if entry is None else int(entry.n_lexicons)
        len_token = len(term) if entry is None else int(entry.len_token)
        in_hunspell = False if entry is None else bool(entry.in_hunspell)
        in_wiktionary = False if entry is None else bool(entry.in_wiktionary)
        # Enriched signals from Wiktionary (optional)
        extra = {} if entry is None else dict(entry.extra)
        wiki_views = int(extra.get("wiki_page_views_30d", 0) or 0)
        wiki_edits = int(extra.get("wiki_total_edits", 0) or 0)
        wiki_entries_count = int(extra.get("wiki_entries_count", 0) or 0)
        wiki_single_entry_page = bool(extra.get("wiki_single_entry_page", False) or False)

        # Frequency handling: treat missing/zero wordfreq as neutral unless no other common evidence.
        freq_available = (freq > 0.0) or in_wordfreq
        freq_log = None
        if freq_available:
            freq_floor = max(freq, 1e-12)
            freq_log = float(__import__("math").log(freq_floor))

        # Signals per intuition (booleans)
        s_freq_common = bool(freq_available and freq_log is not None and freq_log >= self.freq_log_common)
        s_nlex_common = bool(n_lexicons >= self.n_lexicons_common)
        s_in_wordfreq = bool(in_wordfreq)
        s_wiki_entries = bool(wiki_entries_count >= self.wiki_entries_ambiguous_min)

        s_wiki_views = bool(wiki_views >= self.wiki_page_views_review_min)
        s_wiki_edits = bool(wiki_edits >= self.wiki_total_edits_review_min)

        s_freq_rare = bool(freq_available and freq_log is not None and freq_log <= self.freq_log_rare)
        s_nlex_zero = bool(n_lexicons == 0)
        s_len_long = bool(len_token >= self.long_token_len)
        s_len_short = bool(len_token <= self.short_token_len_max)
        s_single_entry = bool(wiki_single_entry_page and (wiki_entries_count <= self.wiki_entries_unambiguous_max))
        # Alphabetic-only length (count letters only)
        alpha_count_pre = sum(1 for ch in term if ch.isalpha())
        alpha_len = alpha_count_pre
        s_len_alpha = bool(alpha_len <= self.alpha_token_len_max)

        # Weighted scores for each label intent
        amb_w = self.weights.get("ambiguous", {})
        rev_w = self.weights.get("review", {})
        un_w = self.weights.get("unambiguous", {})

        ambiguous_score = (
            amb_w.get("freq_common", 0.0) * float(s_freq_common) +
            amb_w.get("nlex_common", 0.0) * float(s_nlex_common) +
            amb_w.get("in_wordfreq", 0.0) * float(s_in_wordfreq) +
            amb_w.get("wiki_entries", 0.0) * float(s_wiki_entries) +
            amb_w.get("len_short", 0.0) * float(s_len_short) +
            amb_w.get("len_alpha", 0.0) * float(s_len_alpha)
        )
        # Non-alphabetic ratio removed from signals

        review_score = (
            rev_w.get("wiki_views", 0.0) * float(s_wiki_views) +
            rev_w.get("wiki_edits", 0.0) * float(s_wiki_edits)
        )
        unambiguous_score = (
            un_w.get("freq_rare", 0.0) * float(s_freq_rare) +
            un_w.get("nlex_zero", 0.0) * float(s_nlex_zero) +
            un_w.get("len_long", 0.0) * float(s_len_long) +
            un_w.get("single_entry", 0.0) * float(s_single_entry)
        )

        # Decision logic
        # Decision by weighted max, with review priority when highest
        scores_tuple = (ambiguous_score, review_score, unambiguous_score)
        max_score = max(scores_tuple)
        if review_score == max_score and review_score > 0.0:
            label = "need review"
            confidence = 0.6
        elif ambiguous_score >= unambiguous_score and ambiguous_score > 0.0:
            label = "likely ambiguous"
            confidence = 0.75
        elif unambiguous_score > 0.0:
            label = "likely unambiguous"
            confidence = 0.75
        else:
            label = "need review"
            confidence = 0.5

        return {
            "term": term,
            "label": label,
            "score": confidence,
            "debug": {
                "freq_wordfreq": freq,
                "freq_log": freq_log,
                "n_lexicons": n_lexicons,
                "len_token": len_token,
                "alpha_len": alpha_len,
                "in_wordfreq": in_wordfreq,
                "in_hunspell": in_hunspell,
                "in_wiktionary": in_wiktionary,
                "wiki_page_views_30d": wiki_views,
                "wiki_total_edits": wiki_edits,
                "wiki_entries_count": wiki_entries_count,
                "wiki_single_entry_page": wiki_single_entry_page,
                "signals": {
                    "freq_common": s_freq_common,
                    "nlex_common": s_nlex_common,
                    "in_wordfreq": s_in_wordfreq,
                    "wiki_entries": s_wiki_entries,
                    "wiki_views": s_wiki_views,
                    "wiki_edits": s_wiki_edits,
                    "freq_rare": s_freq_rare,
                    "nlex_zero": s_nlex_zero,
                    "len_long": s_len_long,
                    "len_short": s_len_short,
                    "len_alpha": s_len_alpha,
                    "single_entry": s_single_entry,
                },
                "scores": {
                    "ambiguous": ambiguous_score,
                    "review": review_score,
                    "unambiguous": unambiguous_score,
                },
                "weights": self.weights,
            },
        }

    def score_terms(self, terms: Sequence[str]) -> List[dict]:
        return [self._score_one(t) for t in terms]


def _merge_nested(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_nested(out[k], v)
        else:
            out[k] = v
    return out


class TermAmbiguityScorer:
    """High-level façade for ambiguity scoring across languages."""

    def __init__(self, lang_code: str, thresholds: Dict | None = None, weights: Dict | None = None) -> None:
        self.lang_code = lang_code
        self._model = self._load_model(lang_code, thresholds, weights)

    def _load_model(self, lang_code: str, thresholds: Dict | None = None, weights: Dict | None = None) -> BaseAmbiguityModel:
        """Dispatch to the appropriate ambiguity model for the language."""
        resources = LanguageResources(lang_code)
        if lang_code.lower() == "en":
            LOGGER.info("Loading AmbiguityModelEN for language 'en'")
            return AmbiguityModelEN(resources, thresholds, weights)
        LOGGER.info("Loading heuristic fallback model for language '%s'", lang_code)
        return AmbiguityModelEN(resources, thresholds, weights)  # For now, reuse EN heuristic until localized models are added.

    def score_terms(self, terms: Iterable[str]) -> List[dict]:
        """Score a collection of terms using the configured model."""

        if not isinstance(terms, Iterable):
            raise TypeError("terms must be an iterable of strings")
        return self._model.score_terms(list(terms))
