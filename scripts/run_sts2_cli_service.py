# -*- coding: utf-8 -*-
"""Launch the sts2-cli HTTP service through a local wrapper."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVICE_SCRIPT = (ROOT.parent / "sts2-cli" / "python" / "http_game_service.py").resolve()


def main() -> None:
    runpy.run_path(str(SERVICE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
