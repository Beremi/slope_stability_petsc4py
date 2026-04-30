#!/usr/bin/env python
"""Run an asset-backed slope-stability case from a TOML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from petsc4py import PETSc

from slope_stability.core.run_config import load_run_case_config
from slope_stability.execution.asset_case import run_case_config
from slope_stability.execution.asset_case.runner import _export_outputs


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PETSc slope-stability case from a TOML config.")
    parser.add_argument("config", type=Path, help="Path to the TOML config.")
    parser.add_argument("--out_dir", type=Path, default=None, help="Optional output directory override.")
    args = parser.parse_args()

    cfg = load_run_case_config(args.config)
    out_dir = args.out_dir
    if out_dir is None:
        safe_ts = np.datetime64("now").astype(str).replace(":", "-")
        out_dir = ROOT / "artifacts" / "config_runs" / cfg.problem.name / safe_ts

    result = run_case_config(cfg, Path(out_dir))
    if PETSc.COMM_WORLD.getRank() == 0:
        output_path = Path(result["output"]) if isinstance(result, dict) and "output" in result else Path(out_dir)
        _export_outputs(cfg, args.config.resolve(), output_path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
