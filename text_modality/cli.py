"""Command-line interface: ``python -m text_modality ...``"""

from __future__ import annotations

import sys
from typing import List, Optional

from core.cli import run_cli

from .modality import TextModality

DESCRIPTION = "Kazakh / Russian / English LLM text benchmarks."


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(TextModality(), DESCRIPTION, with_images=False, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
