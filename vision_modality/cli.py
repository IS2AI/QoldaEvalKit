"""Command-line interface: ``python -m vision_modality ...``"""

from __future__ import annotations

import sys
from typing import List, Optional

from core.cli import run_cli

from .modality import VisionModality

DESCRIPTION = "Kazakh / Russian / English VLM vision benchmarks."


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(VisionModality(), DESCRIPTION, with_images=True, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
