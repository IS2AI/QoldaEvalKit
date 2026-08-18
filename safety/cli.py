"""Command-line interface: ``python -m safety ...``"""

from __future__ import annotations

import sys
from typing import List, Optional

from core.cli import run_cli

from .modality import SafetyModality

DESCRIPTION = "Qorgau — Kazakh / Russian LLM safety evaluation."


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(SafetyModality(), DESCRIPTION, with_images=False, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
