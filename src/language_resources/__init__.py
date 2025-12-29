"""Language-specific resource loaders for lexicon management."""

from .hunspell_parser import load_hunspell_vocabulary
from .resources import LanguageResources, LanguageLexiconEntry

__all__ = ["LanguageResources", "LanguageLexiconEntry", "load_hunspell_vocabulary"]
