#!/usr/bin/env python
"""Execute benchmark visualisation notebooks against reusable artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

import notebook_support as nb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional benchmark directory names to execute. Defaults to all benchmarks.",
    )
    parser.add_argument(
        "--jupyter-backend",
        default="static",
        help="Notebook execution override for JUPYTER_BACKEND_OVERRIDE. Use 'static' for headless validation.",
    )
    parser.add_argument(
        "--surface-subdivision",
        type=int,
        default=None,
        help="Optional execution-time override for SURFACE_SUBDIVISION_OVERRIDE when present in the notebook.",
    )
    parser.add_argument(
        "--surface-decimate-reduction",
        type=float,
        default=None,
        help="Optional execution-time override for SURFACE_DECIMATE_REDUCTION_OVERRIDE when present in the notebook.",
    )
    parser.add_argument(
        "--executed-dir",
        type=Path,
        default=ROOT / "artifacts" / "notebook_execution",
        help="Directory where executed notebook copies will be written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Per-cell timeout in seconds.",
    )
    return parser.parse_args()


def _selected_case_tomls(case_names: list[str] | None) -> list[Path]:
    all_cases = {path.parent.name: path for path in nb.benchmark_case_tomls(BENCHMARKS_DIR)}
    if not case_names:
        return [all_cases[name] for name in sorted(all_cases)]
    missing = [name for name in case_names if name not in all_cases]
    if missing:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(missing)}")
    return [all_cases[name] for name in case_names]


def _python_literal(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return json.dumps(value)
    return repr(value)


def _rewrite_assignment(source: str, name: str, value: object) -> str:
    pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", flags=re.MULTILINE)
    replacement = f"{name} = {_python_literal(value)}"
    if not pattern.search(source):
        return source
    return pattern.sub(replacement, source, count=1)


def _prepare_notebook(notebook_path: Path, *, backend: str | None, subdivision: int | None, decimate: float | None):
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = str(cell.get("source", ""))
        source = _rewrite_assignment(source, "RUN_MODE", "reuse")
        source = _rewrite_assignment(source, "EXECUTION_PROFILE", "smoke")
        source = _rewrite_assignment(source, "MPI_RANKS", None)
        if backend is not None:
            source = _rewrite_assignment(source, "JUPYTER_BACKEND_OVERRIDE", backend)
        if subdivision is not None:
            source = _rewrite_assignment(source, "SURFACE_SUBDIVISION_OVERRIDE", subdivision)
        if decimate is not None:
            source = _rewrite_assignment(source, "SURFACE_DECIMATE_REDUCTION_OVERRIDE", decimate)
        cell["source"] = source
    return notebook


def _execute_notebook(
    notebook_path: Path,
    *,
    backend: str | None,
    subdivision: int | None,
    decimate: float | None,
    executed_dir: Path,
    timeout: int,
) -> dict[str, object]:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")

    notebook = _prepare_notebook(
        notebook_path,
        backend=backend,
        subdivision=subdivision,
        decimate=decimate,
    )
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
        record_timing=False,
    )
    started = time.perf_counter()
    executed = client.execute()
    runtime = time.perf_counter() - started
    out_path = executed_dir / notebook_path.parent.name / notebook_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, out_path)
    return {
        "benchmark": notebook_path.parent.name,
        "notebook": str(notebook_path),
        "executed_notebook": str(out_path),
        "runtime_seconds": runtime,
    }


def main() -> None:
    args = _parse_args()
    results = []
    for case_toml in _selected_case_tomls(args.cases):
        notebook_path = case_toml.parent / "visualisation.ipynb"
        result = _execute_notebook(
            notebook_path,
            backend=args.jupyter_backend,
            subdivision=args.surface_subdivision,
            decimate=args.surface_decimate_reduction,
            executed_dir=args.executed_dir,
            timeout=args.timeout,
        )
        results.append(result)
        print(
            f"{result['benchmark']:<40} {result['runtime_seconds']:>9.2f} s  {result['executed_notebook']}",
            flush=True,
        )
    summary_path = args.executed_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote execution summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
