"""Language resource abstractions for multilingual lexicon management."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional

import pandas as pd

try:  # Optional dependency; used when available.
    import polars as pl
except ImportError:  # pragma: no cover - handled at runtime.
    pl = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

_DEFAULT_PROCESSED_DIR = Path("data/processed")
_DEFAULT_FILENAME_TEMPLATE = "lexicon_{lang_code}.parquet"

_CORE_COLUMNS: frozenset[str] = frozenset(
    {
        "token",
        "lemma",
        "len_token",
        "freq_wordfreq",
        "in_hunspell",
        "in_wiktionary",
        "in_wordfreq",
        "in_extra_sources",
        "n_lexicons",
        "pos_set",
    }
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _ensure_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0", ""}:
            return False
    return default


def _normalize_pos_set(value: Any) -> frozenset[str]:
    if _is_missing(value):
        return frozenset()
    if isinstance(value, (set, frozenset)):
        return frozenset(str(item) for item in value if str(item))
    if isinstance(value, (list, tuple)):
        return frozenset(str(item) for item in value if str(item))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return frozenset()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
            return frozenset(parts)
        if isinstance(parsed, list):
            return frozenset(str(item) for item in parsed if str(item))
        if isinstance(parsed, str):
            return frozenset({parsed})
        return frozenset()
    return frozenset({str(value)})


@dataclass(slots=True, frozen=True)
class LanguageLexiconEntry:
    """Normalized, immutable representation of a lexicon entry."""

    token: str
    lemma: Optional[str]
    len_token: int
    freq_wordfreq: float
    in_hunspell: bool
    in_wiktionary: bool
    in_wordfreq: bool
    in_extra_sources: bool
    n_lexicons: int
    pos_set: frozenset[str] = field(default_factory=frozenset)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - trivial setter
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


class LanguageResources:
    """High-level access to consolidated lexical resources for a language."""

    def __init__(
        self,
        lang_code: str,
        *,
        lexicon_path: str | Path | None = None,
        backend: str = "auto",
        lexicon_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self.lang_code = lang_code
        self._backend = "pandas"
        self._polars_table: Optional["pl.DataFrame"] = None

        if lexicon_df is not None:
            LOGGER.debug("Initializing LanguageResources from in-memory DataFrame for %s", lang_code)
            self._df = lexicon_df.copy(deep=True)
        else:
            resolved_backend = self._resolve_backend(backend)
            path = Path(lexicon_path) if lexicon_path else self.default_lexicon_path(lang_code)
            if not path.exists():
                raise FileNotFoundError(f"Lexicon file not found for language '{lang_code}': {path}")
            LOGGER.debug("Loading lexicon for %s from %s using backend=%s", lang_code, path, resolved_backend)
            if resolved_backend == "polars":
                if pl is None:  # pragma: no cover - runtime guard
                    raise ImportError("polars is not installed but backend 'polars' was requested")
                self._polars_table = pl.read_parquet(path)
                self._df = self._polars_table.to_pandas()
                self._backend = "polars"
            else:
                self._df = pd.read_parquet(path)
                self._backend = "pandas"

        self._prepare_dataframe()
        self._token_index = self._build_token_index()
        self._entry_cache: Dict[str, LanguageLexiconEntry] = {}

    @staticmethod
    def default_lexicon_path(lang_code: str) -> Path:
        """Return the default location for the processed lexicon file."""

        return _DEFAULT_PROCESSED_DIR / _DEFAULT_FILENAME_TEMPLATE.format(lang_code=lang_code)

    @property
    def backend(self) -> str:
        """Expose the backend used to load the lexicon."""

        return self._backend

    def _resolve_backend(self, backend: str) -> str:
        normalized = backend.lower()
        if normalized not in {"auto", "pandas", "polars"}:
            raise ValueError("backend must be 'auto', 'pandas', or 'polars'")
        if normalized == "auto" and pl is not None:
            return "polars"
        if normalized == "auto":
            return "pandas"
        if normalized == "polars" and pl is None:
            LOGGER.warning("polars requested but not installed; falling back to pandas")
            return "pandas"
        return normalized

    def _prepare_dataframe(self) -> None:
        if "token" not in self._df:
            raise ValueError("Lexicon DataFrame must contain a 'token' column")

        df = self._df
        df["token"] = df["token"].astype(str)
        if "lemma" not in df:
            df["lemma"] = None
        df["lemma"] = df["lemma"].where(~df["lemma"].isna(), None)

        if "len_token" not in df:
            df["len_token"] = df["token"].str.len()
        else:
            df["len_token"] = df["len_token"].fillna(df["token"].str.len()).astype(int)

        if "freq_wordfreq" in df:
            df["freq_wordfreq"] = df["freq_wordfreq"].fillna(0.0).astype(float)
        else:
            df["freq_wordfreq"] = 0.0

        bool_columns_defaults = {
            "in_hunspell": False,
            "in_wiktionary": False,
            "in_wordfreq": False,
            "in_extra_sources": False,
        }
        for column, default in bool_columns_defaults.items():
            if column in df:
                df[column] = df[column].map(lambda value: _ensure_bool(value, default)).astype(bool)
            else:
                df[column] = bool(default)

        available_bool_columns = [
            col
            for col in ("in_hunspell", "in_wiktionary", "in_wordfreq", "in_extra_sources")
            if col in df
        ]
        if available_bool_columns:
            counts = df[available_bool_columns].astype(int).sum(axis=1)
        else:
            counts = 0
        if "n_lexicons" not in df:
            df["n_lexicons"] = counts
        else:
            df["n_lexicons"] = df["n_lexicons"].fillna(counts).astype(int)

        if "pos_set" not in df:
            df["pos_set"] = None

        self._df = df.reset_index(drop=True)

    def _build_token_index(self) -> Dict[str, int]:
        index: Dict[str, int] = {}
        for idx, token in enumerate(self._df["token"].tolist()):
            index.setdefault(token, idx)
        return index

    def get_entry(self, token: str) -> Optional[LanguageLexiconEntry]:
        """Retrieve a lexicon entry for the supplied token."""

        idx = self._token_index.get(token)
        if idx is None:
            return None
        if token in self._entry_cache:
            return self._entry_cache[token]
        row = self._df.iloc[idx]
        entry = self._row_to_entry(row)
        self._entry_cache[token] = entry
        return entry

    def freq(self, token: str) -> float:
        """Return the frequency score associated with a token.

        Returns 0.0 when the token is absent from the lexicon.
        """

        idx = self._token_index.get(token)
        if idx is None:
            return 0.0
        return float(self._df.iloc[idx]["freq_wordfreq"])

    def is_in_lexicon(self, token: str) -> bool:
        """Return True when the token is present in the consolidated lexicon."""

        return token in self._token_index

    def iter_entries(self) -> Iterator[LanguageLexiconEntry]:
        """Iterate over all lexicon entries for the language."""

        for _, row in self._df.iterrows():
            token = row["token"]
            cached = self._entry_cache.get(token)
            if cached is None:
                cached = self._row_to_entry(row)
                self._entry_cache[token] = cached
            yield cached

    def to_dataframe(self, *, copy: bool = False) -> pd.DataFrame:
        """Expose the underlying pandas DataFrame."""

        return self._df.copy(deep=True) if copy else self._df

    def to_polars(self) -> "pl.DataFrame":
        """Return a polars representation of the lexicon.

        This converts from pandas when the original backend was not polars.
        """

        if pl is None:
            raise ImportError("polars is not installed")
        if self._polars_table is not None:
            return self._polars_table
        self._polars_table = pl.from_pandas(self._df)
        return self._polars_table

    def __contains__(self, token: object) -> bool:
        return isinstance(token, str) and token in self._token_index

    def __len__(self) -> int:
        return len(self._df)

    def _row_to_entry(self, row: Any) -> LanguageLexiconEntry:
        if hasattr(row, "to_dict"):
            data = row.to_dict()
        else:
            data = dict(row)
        token = str(data.get("token", ""))
        lemma = data.get("lemma")
        if _is_missing(lemma):
            lemma = None
        else:
            lemma = str(lemma)

        freq = float(data.get("freq_wordfreq", 0.0) or 0.0)
        in_hunspell = _ensure_bool(data.get("in_hunspell", False))
        in_wiktionary = _ensure_bool(data.get("in_wiktionary", False))
        in_wordfreq = _ensure_bool(data.get("in_wordfreq", freq > 0.0)) or freq > 0.0
        in_extra_sources = _ensure_bool(data.get("in_extra_sources", False))

        if "n_lexicons" in data and not _is_missing(data["n_lexicons"]):
            n_lexicons = int(data["n_lexicons"])
        else:
            n_lexicons = sum([in_hunspell, in_wiktionary, in_wordfreq, in_extra_sources])

        len_token = int(data.get("len_token", len(token)) or len(token))
        pos_set = _normalize_pos_set(data.get("pos_set"))

        extra_keys = set(data.keys()) - set(_CORE_COLUMNS)
        extra: Dict[str, Any] = {}
        for key in extra_keys:
            value = data[key]
            if not _is_missing(value):
                extra[key] = value

        return LanguageLexiconEntry(
            token=token,
            lemma=lemma,
            len_token=len_token,
            freq_wordfreq=freq,
            in_hunspell=in_hunspell,
            in_wiktionary=in_wiktionary,
            in_wordfreq=in_wordfreq,
            in_extra_sources=in_extra_sources,
            n_lexicons=n_lexicons,
            pos_set=pos_set,
            extra=extra,
        )
