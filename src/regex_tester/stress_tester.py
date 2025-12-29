"""Regex stress tester: match a regex against lexicon tokens across languages.

Scans tokens from processed lexicons (data/processed/lexicon_<lang>.parquet)
via LanguageResources, and matches a pattern with options. Useful to see what a
regex hits in real-world vocabularies across languages.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from language_resources import LanguageResources


@dataclass
class LanguageMatchResult:
    lang: str
    scanned: int
    matched: int
    elapsed_sec: float
    samples: List[str]


class RegexStressTester:
    def __init__(
        self,
        data_dir: str | Path = "data/processed",
        token_sources: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.token_sources = token_sources

    def discover_languages(self) -> List[str]:
        if self.token_sources is not None:
            return sorted(self.token_sources.keys())
        langs: List[str] = []
        for p in sorted(self.data_dir.glob("lexicon_*.parquet")):
            name = p.stem
            if name.startswith("lexicon_"):
                lang = name.split("_", 1)[1]
                if lang:
                    langs.append(lang)
        return langs

    def _iter_tokens(self, lang: str, limit: Optional[int] = None) -> Iterator[str]:
        if self.token_sources is not None and lang in self.token_sources:
            count = 0
            for t in self.token_sources[lang]:
                yield t
                count += 1
                if limit is not None and count >= limit:
                    return
            return
        lr = LanguageResources(lang)
        count = 0
        for e in lr.iter_entries():
            yield e.token
            count += 1
            if limit is not None and count >= limit:
                return

    def _compile(self, pattern: str, ignore_case: bool) -> re.Pattern[str]:
        flags = re.UNICODE
        if ignore_case:
            flags |= re.IGNORECASE
        return re.compile(pattern, flags)

    def run(
        self,
        pattern: str,
        languages: Optional[Sequence[str]] = None,
        *,
        mode: str = "search",
        ignore_case: bool = False,
        limit_per_lang: Optional[int] = None,
        max_matches_per_lang: Optional[int] = None,
        timeout_sec_per_lang: Optional[float] = None,
        sample_k: int = 20,
        include_metadata: bool = False,
        metadata_fields: Optional[Sequence[str]] = None,
    ) -> Tuple[List[LanguageMatchResult], Dict[str, List]]:
        """Run the regex against tokens across languages.

        Returns (stats_per_language, matches_by_language) where the second item
        contains the full list of matched tokens per language (may be large).
        """

        langs = list(languages) if languages else self.discover_languages()
        regex = self._compile(pattern, ignore_case)

        if mode not in {"search", "fullmatch"}:
            raise ValueError("mode must be 'search' or 'fullmatch'")

        def matcher(s: str) -> bool:
            return bool(regex.search(s)) if mode == "search" else bool(regex.fullmatch(s))

        stats: List[LanguageMatchResult] = []
        matches: Dict[str, List] = {}

        for lang in langs:
            resources = None
            if self.token_sources is None:
                resources = LanguageResources(lang)
            start = time.monotonic()
            matched_tokens: List = []
            scanned = 0
            for tok in self._iter_tokens(lang, limit=limit_per_lang):
                scanned += 1
                if matcher(tok):
                    if include_metadata:
                        fields = list(metadata_fields) if metadata_fields is not None else [
                            "freq_wordfreq",
                            "n_lexicons",
                            "len_token",
                        ]
                        rec = {"token": tok}
                        if resources is not None:
                            e = resources.get_entry(tok)
                            for f in fields:
                                if f == "len_token":
                                    rec[f] = (e.len_token if e is not None else len(tok))
                                else:
                                    rec[f] = getattr(e, f, None) if e is not None else None
                        else:
                            # token_sources case; can only compute len_token reliably
                            for f in fields:
                                if f == "len_token":
                                    rec[f] = len(tok)
                                else:
                                    rec[f] = None
                        matched_tokens.append(rec)
                    else:
                        matched_tokens.append(tok)
                    if max_matches_per_lang is not None and len(matched_tokens) >= max_matches_per_lang:
                        break
                if timeout_sec_per_lang is not None and (time.monotonic() - start) > timeout_sec_per_lang:
                    break
            elapsed = time.monotonic() - start
            matches[lang] = matched_tokens
            stats.append(
                LanguageMatchResult(
                    lang=lang,
                    scanned=scanned,
                    matched=len(matched_tokens),
                    elapsed_sec=elapsed,
                    samples=[(m["token"] if isinstance(m, dict) else m) for m in matched_tokens[:sample_k]],
                )
            )

        return stats, matches

