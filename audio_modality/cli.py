"""Command-line interface: ``python -m audio_modality ...``"""

from __future__ import annotations

import sys
from typing import List, Optional

from core.cli import run_cli

from .modality import AudioModality

DESCRIPTION = "Kazakh / Russian / English audio-LLM benchmarks."


def main(argv: Optional[List[str]] = None) -> int:
    return run_cli(AudioModality(), DESCRIPTION, with_images=False,
                   with_audio=True, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
