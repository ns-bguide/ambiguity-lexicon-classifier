"""Lightweight Hunspell .aff/.dic parser for generating word forms.

This module parses Hunspell affix and dictionary files in pure Python to extract
root words and expand them using the affix rules. It is intentionally
feature-limited but covers the most common constructs (prefix/suffix rules,
flag formats ``short``, ``long`` and ``num``).

The main entry point is :func:`load_hunspell_vocabulary`, which returns the set
of base and derived forms found in all .dic/.aff pairs within a directory.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Pattern, Sequence, Set

LOGGER = logging.getLogger(__name__)
def is_valid_token(word: str) -> bool:
    """Return True if the token is at least 2 chars and alphabetic.

    Uses `str.isalpha()` to allow Unicode letters across languages.
    """

    return len(word) >= 2 and word.isalpha()


AffixKind = str  # limited to "PFX" or "SFX" in practice


@dataclass
class AffixEntry:
    """Single transformation rule from a Hunspell ``.aff`` file."""

    strip: str
    add: str
    condition: Pattern[str]
    raw_condition: str
    morph: Optional[str] = None

    def apply(self, word: str, kind: AffixKind) -> Optional[str]:
        """Apply the entry to ``word`` when possible and return the derived form."""

        if kind == "SFX":
            if self.strip and not word.endswith(self.strip):
                return None
            stem = word[: -len(self.strip)] if self.strip else word
            if not self.condition.match(stem):
                return None
            addition = self.add
            return stem + addition

        if kind == "PFX":
            if self.strip and not word.startswith(self.strip):
                return None
            stem = word[len(self.strip) :] if self.strip else word
            if not self.condition.match(stem):
                return None
            addition = self.add
            return addition + stem

        LOGGER.debug("Unsupported affix kind %s", kind)
        return None


@dataclass
class HunspellRule:
    """Collection of entries tied to a single flag for prefix/suffix rules."""

    kind: AffixKind
    flag: str
    cross: bool
    entries: List[AffixEntry] = field(default_factory=list)

    def apply(self, word: str) -> Set[str]:
        results: Set[str] = set()
        for entry in self.entries:
            derived = entry.apply(word, self.kind)
            if derived and derived != word:
                results.add(derived)
        return results


@dataclass
class HunspellAffixSet:
    """Parsed representation of Hunspell affixes."""

    flag_type: str = "short"
    prefixes: Dict[str, HunspellRule] = field(default_factory=dict)
    suffixes: Dict[str, HunspellRule] = field(default_factory=dict)


@dataclass
class HunspellEntry:
    """Root word and attached affix flags from the ``.dic`` file."""

    root: str
    flags: Set[str] = field(default_factory=set)


def parse_aff_file(path: Path) -> HunspellAffixSet:
    """Parse a Hunspell ``.aff`` file into a structured representation."""

    affixes = HunspellAffixSet()

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = list(handle)
    except FileNotFoundError:
        raise

    iter_lines = iter(lines)
    for raw_line in iter_lines:
        line = _strip_comment(raw_line)
        if not line:
            continue
        parts = line.split()
        keyword = parts[0]
        if keyword == "FLAG" and len(parts) >= 2:
            affixes.flag_type = parts[1].lower()
            continue
        if keyword not in {"PFX", "SFX"} or len(parts) < 4:
            continue

        kind: AffixKind = keyword
        flag = parts[1]
        cross = parts[2].upper() == "Y"
        try:
            expected_entries = int(parts[3])
        except ValueError:
            LOGGER.warning("Invalid entry count in %s: %s", path, line)
            continue

        entries: List[AffixEntry] = []
        while len(entries) < expected_entries:
            try:
                entry_line = next(iter_lines)
            except StopIteration:
                LOGGER.warning("Unexpected end of file while parsing %s", path)
                break
            entry_line = _strip_comment(entry_line)
            if not entry_line:
                continue
            entry_parts = entry_line.split()
            if len(entry_parts) < 5:
                LOGGER.debug("Skipping malformed affix line: %s", entry_line)
                continue
            _, entry_flag, strip, add, condition_raw, *rest = entry_parts
            if entry_flag != flag:
                LOGGER.debug(
                    "Flag mismatch in affix entry (expected %s, got %s)",
                    flag,
                    entry_flag,
                )
            strip = "" if strip == "0" else strip
            add = "" if add == "0" else add
            condition_regex = _compile_condition(condition_raw, kind)
            morph = " ".join(rest) if rest else None
            entries.append(
                AffixEntry(
                    strip=strip,
                    add=add,
                    condition=condition_regex,
                    raw_condition=condition_raw,
                    morph=morph,
                )
            )

        rule_map = affixes.prefixes if kind == "PFX" else affixes.suffixes
        if flag in rule_map:
            rule_map[flag].entries.extend(entries)
            rule_map[flag].cross = rule_map[flag].cross or cross
        else:
            rule_map[flag] = HunspellRule(kind=kind, flag=flag, cross=cross, entries=entries)

    return affixes


def parse_dic_file(path: Path, flag_type: str) -> List[HunspellEntry]:
    """Parse a Hunspell ``.dic`` file into root words and flag sets."""

    entries: List[HunspellEntry] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = True
            for raw in handle:
                line = _strip_comment(raw)
                if not line:
                    continue
                if first_line:
                    first_line = False
                    try:
                        int(line)
                        continue
                    except ValueError:
                        pass
                first_line = False

                token = line.split()[0]
                if "/" in token:
                    root, flag_segment = token.split("/", 1)
                else:
                    root, flag_segment = token, ""
                # Skip invalid root tokens early to avoid unnecessary expansion
                if not is_valid_token(root):
                    continue
                flags = _split_flags(flag_segment, flag_type)
                entries.append(HunspellEntry(root=root, flags=set(flags)))
    except FileNotFoundError:
        raise

    LOGGER.debug("Parsed %s roots from %s", len(entries), path)
    return entries


def generate_word_forms(entries: Sequence[HunspellEntry], affixes: HunspellAffixSet) -> Set[str]:
    """Expand all dictionary entries using the provided affix rules."""

    vocabulary: Set[str] = set()
    for entry in entries:
        base = entry.root
        if is_valid_token(base):
            vocabulary.add(base)

        prefix_rules = [affixes.prefixes[flag] for flag in entry.flags if flag in affixes.prefixes]
        suffix_rules = [affixes.suffixes[flag] for flag in entry.flags if flag in affixes.suffixes]

        prefix_forms: Set[str] = set()
        for rule in prefix_rules:
            prefix_forms.update(rule.apply(base))
        prefix_forms = {w for w in prefix_forms if is_valid_token(w)}

        suffix_forms: Set[str] = set()
        for rule in suffix_rules:
            suffix_forms.update(rule.apply(base))
        suffix_forms = {w for w in suffix_forms if is_valid_token(w)}

        combined_forms = set()
        if prefix_rules and suffix_rules:
            cross_prefix = [rule for rule in prefix_rules if rule.cross]
            cross_suffix = [rule for rule in suffix_rules if rule.cross]
            if cross_prefix and cross_suffix:
                prefix_intermediates: Set[str] = set()
                for rule in cross_prefix:
                    prefix_intermediates.update(rule.apply(base))
                prefix_intermediates = {w for w in prefix_intermediates if is_valid_token(w)}
                for intermediate in prefix_intermediates:
                    for rule in cross_suffix:
                        combined_forms.update(rule.apply(intermediate))
                combined_forms = {w for w in combined_forms if is_valid_token(w)}

                suffix_intermediates: Set[str] = set()
                for rule in cross_suffix:
                    suffix_intermediates.update(rule.apply(base))
                suffix_intermediates = {w for w in suffix_intermediates if is_valid_token(w)}
                for intermediate in suffix_intermediates:
                    for rule in cross_prefix:
                        combined_forms.update(rule.apply(intermediate))
                combined_forms = {w for w in combined_forms if is_valid_token(w)}

        vocabulary.update(prefix_forms)
        vocabulary.update(suffix_forms)
        vocabulary.update(combined_forms)

    return vocabulary


def load_hunspell_vocabulary(directory: Path) -> Set[str]:
    """Load and expand all Hunspell dictionaries in ``directory``."""

    if not directory.exists():
        LOGGER.info("Hunspell directory %s not found; skipping", directory)
        return set()

    vocab: Set[str] = set()
    aff_files = sorted(directory.glob("*.aff"))
    if not aff_files:
        LOGGER.info("No Hunspell .aff files found in %s; skipping", directory)
        return set()

    for aff_path in aff_files:
        dic_path = aff_path.with_suffix(".dic")
        if not dic_path.exists():
            LOGGER.warning("Missing .dic companion for %s", aff_path.name)
            continue
        LOGGER.info("Parsing Hunspell pair %s / %s", aff_path.name, dic_path.name)
        affixes = parse_aff_file(aff_path)
        entries = parse_dic_file(dic_path, affixes.flag_type)
        forms = generate_word_forms(entries, affixes)
        vocab.update(forms)
        LOGGER.info("Loaded %s word forms from %s", len(forms), aff_path.stem)

    LOGGER.info("Aggregated %s unique Hunspell word forms from %s", len(vocab), directory)
    return vocab


def _strip_comment(line: str) -> str:
    """Strip trailing Hunspell-style comments beginning with ``#``."""

    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


def _compile_condition(condition: str, kind: AffixKind) -> Pattern[str]:
    if condition in {"", "0"}:
        return re.compile(".*")
    if kind == "SFX":
        regex = f".*{condition}$"
    else:
        regex = f"^{condition}.*"
    return re.compile(regex)


def _split_flags(flag_segment: str, flag_type: str) -> Iterable[str]:
    if not flag_segment:
        return []
    if flag_type == "long":
        return [flag_segment[i : i + 2] for i in range(0, len(flag_segment), 2)]
    if flag_type == "num":
        return [flag.strip() for flag in flag_segment.split(",") if flag.strip()]
    return list(flag_segment)


__all__ = [
    "AffixEntry",
    "HunspellRule",
    "HunspellAffixSet",
    "HunspellEntry",
    "parse_aff_file",
    "parse_dic_file",
    "generate_word_forms",
    "load_hunspell_vocabulary",
]
