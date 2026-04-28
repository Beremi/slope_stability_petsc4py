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

from tools.mesh_conversion.asset_builders import convert_3d_hetero_slope


if __name__ == "__main__":
    convert_3d_hetero_slope(Path(__file__).resolve().parents[1])
