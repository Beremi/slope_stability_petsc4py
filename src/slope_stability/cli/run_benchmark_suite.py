#!/usr/bin/env python3
"""Run the full canonical MATLAB-vs-PETSc benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from .run_benchmark_case import run_benchmark
from .report_benchmark_suite import generate_suite_readme


def _is_benchmark_case(config_path: Path) -> bool:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return bool(data.get("benchmark"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical MATLAB parity benchmark suite.")
    parser.add_argument(
        "--suite_dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "benchmarks",
    )
    args = parser.parse_args()
    suite_dir = args.suite_dir.resolve()
    configs = sorted(path for path in suite_dir.glob("*/case.toml") if _is_benchmark_case(path))
    for config in configs:
        run_benchmark(config)
    generate_suite_readme(suite_dir)


if __name__ == "__main__":
    main()
