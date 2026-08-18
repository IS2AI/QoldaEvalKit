"""Command-line interface: ``python -m translation ...``"""

from __future__ import annotations

import sys
from typing import List, Optional

from core.cli import run_cli

from .modality import TranslationModality

DESCRIPTION = "FLORES translation between Kazakh, Russian and English."


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(TranslationModality(), DESCRIPTION, with_images=False,
                   with_translation=True, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
