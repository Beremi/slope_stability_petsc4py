"""Public runtime option resolver helpers.

This module owns Python-side assembly of the concrete PETSc option stream.
Cases and profiles still define the mathematical/profile model; this runtime
layer only combines resolved objects, profile option files, and artifact paths
into the string handed to the native engine.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Iterable

from petsc_ssr.options import DEFAULT_OPTIONS_FILE, LinearOptions, PmgOptions, SsrOptions, flatten_tokens

__all__ = [
    "DEFAULT_OPTIONS_FILE",
    "LinearOptions",
    "PmgOptions",
    "SsrOptions",
    "flatten_tokens",
    "quote_option_tokens",
    "read_options_file",
    "resolve_run_option_tokens",
    "resolve_run_options_string",
]


def read_options_file(path: str | Path | None) -> list[str]:
    """Read a PETSc options file into shell-tokenized option tokens."""

    if path is None:
        return []
    options_path = Path(path)
    if not options_path.exists():
        return []
    tokens: list[str] = []
    for raw in options_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def quote_option_tokens(tokens: Iterable[object]) -> str:
    """Return a shell-safe single options string for the native entry point."""

    return " ".join(shlex.quote(str(token)) for token in tokens)


def resolve_run_option_tokens(
    problem: Any,
    options: SsrOptions,
    output_dir: str | Path,
    *,
    write_solution_vtu: bool | None = None,
) -> list[str]:
    """Resolve profile, problem, solver, and artifact paths into PETSc tokens."""

    root = Path(output_dir)
    data_dir = root / "data"
    tokens = read_options_file(Path(options.pmg.options_file or DEFAULT_OPTIONS_FILE))
    tokens.extend(problem.option_tokens())
    tokens.extend(options.option_tokens())
    tokens.extend(
        [
            "-curve_csv",
            str(data_dir / "continuation_curve.csv"),
            "-summary_json",
            str(data_dir / "summary.json"),
        ]
    )
    if write_solution_vtu is None:
        metadata = getattr(problem, "metadata", {}) or {}
        write_solution_vtu = _metadata_bool(metadata.get("write_solution_vtu"), default=True)
    if write_solution_vtu:
        tokens.extend(
            [
                "-solution_binary",
                str(data_dir / "final_displacement.petscbin"),
                "-solution_points_csv",
                str(data_dir / "final_displacement_points.csv"),
                "-solution_vtk",
                str(root / "exports" / "final_solution.vtu"),
            ]
        )
    return tokens


def resolve_run_options_string(
    problem: Any,
    options: SsrOptions,
    output_dir: str | Path,
    *,
    write_solution_vtu: bool | None = None,
) -> str:
    return quote_option_tokens(
        resolve_run_option_tokens(
            problem,
            options,
            output_dir,
            write_solution_vtu=write_solution_vtu,
        )
    )


def _metadata_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)
