#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tools.mesh_conversion.asset_builders import convert_2d_sloan2013


if __name__ == "__main__":
    convert_2d_sloan2013(Path(__file__).resolve().parents[1])
