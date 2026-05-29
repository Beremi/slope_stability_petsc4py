#!/usr/bin/env python3
"""Execute benchmark notebooks against local smoke artifacts.

The committed notebooks are authored for interactive use. This harness patches
temporary copies of their control cells so CI/local checks can run every case
without launching a full benchmark sweep.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = ROOT / "benchmarks" / "cases"


def _patch_controls(nb: nbformat.NotebookNode, *, run_mode: str, profile: str, mpi_ranks: int | None) -> None:
    replacements = {
        "RUN_MODE": f'RUN_MODE = "{run_mode}"',
        "EXECUTION_PROFILE": f'EXECUTION_PROFILE = "{profile}"',
        "MPI_RANKS": f"MPI_RANKS = {mpi_ranks if mpi_ranks is not None else 'None'}",
    }
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = str(cell.source)
        if not all(name in source for name in ("RUN_LABEL", "RUN_MODE", "EXECUTION_PROFILE", "MPI_RANKS")):
            continue
        lines: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            matched = False
            for name, replacement in replacements.items():
                if stripped.startswith(f"{name} ="):
                    indent = line[: len(line) - len(line.lstrip())]
                    lines.append(indent + replacement)
                    matched = True
                    break
            if not matched:
                lines.append(line)
        cell.source = "\n".join(lines) + "\n"
        return
    raise RuntimeError("Notebook has no standard benchmark control cell")


def _execute_notebook(
    source: Path,
    *,
    output_path: Path,
    run_mode: str,
    profile: str,
    mpi_ranks: int | None,
    kernel_name: str,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    nb = nbformat.read(source, as_version=4)
    _patch_controls(nb, run_mode=run_mode, profile=profile, mpi_ranks=mpi_ranks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output_path)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(nb, output_path)
    return {
        "source": str(source.relative_to(ROOT)),
        "executed": str(output_path.relative_to(ROOT)),
        "seconds": round(time.perf_counter() - started, 3),
        "run_mode": run_mode,
        "profile": profile,
        "mpi_ranks": mpi_ranks,
    }


def _artifact_summary(case_dir: Path, run_label: str) -> dict[str, Any]:
    from notebook_support import artifact_dir_complete, load_run_artifacts

    out_dir = case_dir / "artifacts" / run_label
    summary: dict[str, Any] = {
        "artifact_dir": str(out_dir.relative_to(ROOT)),
        "complete": artifact_dir_complete(out_dir),
    }
    if not summary["complete"]:
        return summary
    artifacts = load_run_artifacts(out_dir)
    run_info = dict(artifacts.run_info.get("run_info", {}))
    summary.update(
        {
            "step_count": int(run_info.get("step_count", 0)),
            "lambda_last": run_info.get("lambda_last"),
            "omega_last": run_info.get("omega_last"),
            "runtime_seconds": run_info.get("runtime_seconds"),
            "vtu": str(artifacts.vtu_path.relative_to(ROOT)),
        }
    )
    return summary


def _configure_environment() -> None:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.environ.setdefault("PETSC_SSR_ENGINE_PYTHON", str(venv_python))
        os.environ["PATH"] = os.pathsep.join([str(venv_python.parent), os.environ.get("PATH", "")])
    paths = [str(ROOT), str(ROOT / "src"), str(ROOT / "benchmarks" / "tools")]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("SLOPE_STABILITY_MPI_OVERSUBSCRIBE", "true")


def _case_dirs(selected: list[str]) -> list[Path]:
    all_cases = sorted(path.parent for path in CASES_ROOT.glob("*/case.toml"))
    if not selected:
        return all_cases
    wanted = set(selected)
    cases = [path for path in all_cases if path.name in wanted]
    missing = sorted(wanted.difference(path.name for path in cases))
    if missing:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(missing)}")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="Case directory name to execute; repeatable.")
    parser.add_argument("--output-dir", default=".local/notebook-execution", help="Directory for executed notebook copies and report.")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "benchmark"], help="Execution profile patched into temporary notebooks.")
    parser.add_argument("--mpi-ranks", type=int, default=1, help="MPI ranks patched into simulation notebooks; use 0 for notebook default.")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel name.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-notebook cell timeout in seconds.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after notebook failures and report all failures.")
    args = parser.parse_args(argv)

    _configure_environment()

    output_root = (ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mpi_ranks = None if args.mpi_ranks == 0 else args.mpi_ranks
    report: dict[str, Any] = {
        "profile": args.profile,
        "mpi_ranks": mpi_ranks,
        "kernel": args.kernel,
        "cases": [],
        "failures": [],
    }

    for case_dir in _case_dirs(args.case):
        case_report: dict[str, Any] = {"case": case_dir.name, "notebooks": []}
        print(f"== {case_dir.name} ==", flush=True)
        for notebook_name, run_mode in (("simulation.ipynb", "run"), ("visualisation.ipynb", "reuse")):
            source = case_dir / notebook_name
            executed = output_root / case_dir.name / notebook_name
            try:
                result = _execute_notebook(
                    source,
                    output_path=executed,
                    run_mode=run_mode,
                    profile=args.profile,
                    mpi_ranks=mpi_ranks,
                    kernel_name=args.kernel,
                    timeout=args.timeout,
                )
            except Exception as exc:
                failure = {
                    "case": case_dir.name,
                    "notebook": notebook_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["failures"].append(failure)
                case_report["notebooks"].append(failure)
                print(f"  {notebook_name}: FAIL ({failure['error']})", flush=True)
                if not args.keep_going:
                    report["cases"].append(case_report)
                    (output_root / "notebook_execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                    return 1
                continue
            case_report["notebooks"].append(result)
            print(f"  {notebook_name}: ok ({result['seconds']:.1f}s)", flush=True)
        case_report["artifacts"] = _artifact_summary(case_dir, "simulation")
        report["cases"].append(case_report)

    report_path = output_root / "notebook_execution_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path.relative_to(ROOT)}", flush=True)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
