"""Public runtime runner entry points."""

from __future__ import annotations

from petsc_ssr.runners.run_case_from_config import main as run_case_main


def run_case(argv: list[str] | None = None) -> int:
    return run_case_main(argv)


__all__ = ["run_case", "run_case_main"]
