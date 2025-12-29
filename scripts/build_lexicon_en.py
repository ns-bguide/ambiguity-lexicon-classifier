"""Backward-compatible wrapper for the generic lexicon builder."""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from build_lexicon import main as _generic_main

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> None:
    """Invoke :mod:`scripts.build_lexicon` defaulting to English resources."""

    LOGGER.warning(
        "scripts/build_lexicon_en.py is deprecated. Use build_lexicon.py with --lang-code en instead."
    )

    original_argv = list(sys.argv)
    cli_args = list(argv) if argv is not None else original_argv[1:]

    if not any(arg.startswith("--lang-code") for arg in cli_args):
        cli_args = ["--lang-code", "en", *cli_args]

    sys.argv = [original_argv[0], *cli_args]
    try:
        _generic_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
