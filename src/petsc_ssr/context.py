from __future__ import annotations

import csv
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from mpi4py import MPI
from petsc4py import PETSc

from .options import DEFAULT_OPTIONS_FILE, SsrOptions
from .problem import ProblemSpec


def _read_options_file(path: Path) -> list[str]:
    tokens: list[str] = []
    if not path.exists():
        return tokens
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _quote_tokens(tokens: list[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens)


@dataclass(slots=True)
class EngineRunResult:
    output_dir: Path
    curve_csv: Path
    summary_json: Path
    wall_time: float
    summary: dict[str, Any]


class SsrContext:
    """Python-facing owner of one PETSc SSR engine run.

    The maintained runner uses the full C continuation/Newton implementation.
    The Python-loop driver remains available for debugging through
    run_python_loop(), but it is not the default benchmark path.
    """

    def __init__(
        self,
        problem: ProblemSpec,
        options: SsrOptions | None = None,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.problem = problem
        self.options = options or SsrOptions.current_baseline()
        self.output_dir = Path(output_dir) if output_dir is not None else Path(".local") / "tmp" / problem.name
        self.last_result: EngineRunResult | None = None
        self._engine = None

    def __enter__(self) -> "SsrContext":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.destroy()

    @property
    def rank(self) -> int:
        return int(PETSc.COMM_WORLD.getRank())

    def destroy(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def option_tokens(self) -> list[str]:
        data_dir = self.output_dir / "data"
        curve_csv = data_dir / "continuation_curve.csv"
        summary_json = data_dir / "summary.json"
        solution_binary = data_dir / "final_displacement.petscbin"
        solution_points_csv = data_dir / "final_displacement_points.csv"
        solution_vtk = self.output_dir / "exports" / "final_solution.vtu"
        tokens = _read_options_file(Path(self.options.pmg.options_file or DEFAULT_OPTIONS_FILE))
        tokens.extend(self.problem.option_tokens())
        tokens.extend(self.options.option_tokens())
        tokens.extend([
            "-curve_csv", str(curve_csv),
            "-summary_json", str(summary_json),
            "-solution_binary", str(solution_binary),
            "-solution_points_csv", str(solution_points_csv),
            "-solution_vtk", str(solution_vtk),
        ])
        return tokens

    def options_string(self) -> str:
        return _quote_tokens(self.option_tokens())

    def run(self) -> EngineRunResult:
        if self.options.analysis.lower() == "ll":
            return self.run_python_loop()
        return self.run_monolithic()

    def run_python_loop(self) -> EngineRunResult:
        if self.options.analysis.lower() == "ll":
            from .limit_load import run_limit_load_continuation

            run_limit_load_continuation(self)
        else:
            from .continuation import run_indirect_ssr

            run_indirect_ssr(self)
        if self.last_result is None:
            raise RuntimeError("SSR run finished without a result")
        return self.last_result

    def create_engine(self):
        if self._engine is None:
            from .native import _core

            self.prepare_output_dir()
            self._engine = _core.Engine(self.options_string())
        return self._engine

    def prepare_output_dir(self) -> None:
        comm = PETSc.COMM_WORLD.tompi4py()
        data_dir = self.output_dir / "data"
        if self.rank == 0:
            data_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "exports").mkdir(parents=True, exist_ok=True)
            (data_dir / "problem.json").write_text(json.dumps(self.problem.to_dict(), indent=2), encoding="utf-8")
            (data_dir / "options.txt").write_text(self.options_string() + "\n", encoding="utf-8")
        comm.Barrier()

    def run_monolithic(self) -> EngineRunResult:
        from .native import _core

        comm = PETSc.COMM_WORLD.tompi4py()
        self.prepare_output_dir()
        data_dir = self.output_dir / "data"

        t0 = perf_counter()
        _core.run_options(self.options_string())
        wall = perf_counter() - t0
        comm.Barrier()

        summary_json = data_dir / "summary.json"
        curve_csv = data_dir / "continuation_curve.csv"
        summary: dict[str, Any] = {}
        if self.rank == 0 and summary_json.exists():
            summary = json.loads(summary_json.read_text(encoding="utf-8"))
        summary = comm.bcast(summary, root=0)
        result = EngineRunResult(self.output_dir, curve_csv, summary_json, wall, summary)
        self.last_result = result
        return result

    def phase_summary(self) -> dict[str, Any]:
        return dict(self.last_result.summary) if self.last_result else {}

    def assemble(self, lambda_value: float) -> None:
        raise NotImplementedError("fine-grained assembly callbacks are reserved for the next C-context extraction")

    def form_regularized_operator(self, r: float) -> None:
        raise NotImplementedError("fine-grained operator callbacks are reserved for the next C-context extraction")

    def solve_indirect_pair(self, rhs_mode: str = "current") -> None:
        raise NotImplementedError("fine-grained pair solves are reserved for the next C-context extraction")

    def evaluate_trial(self, alpha: float, omega_target: float) -> None:
        raise NotImplementedError("fine-grained trial callbacks are reserved for the next C-context extraction")

    def accept_trial(self, alpha: float) -> None:
        raise NotImplementedError("fine-grained trial callbacks are reserved for the next C-context extraction")

    def rescale_to_omega(self, omega_target: float) -> None:
        raise NotImplementedError("fine-grained state callbacks are reserved for the next C-context extraction")

    def snapshot_deflation(self) -> None:
        raise NotImplementedError("fine-grained deflation snapshots are reserved for the next C-context extraction")

    def restore_deflation(self) -> None:
        raise NotImplementedError("fine-grained deflation snapshots are reserved for the next C-context extraction")

    def append_deflation_from_update(self) -> None:
        raise NotImplementedError("fine-grained deflation updates are reserved for the next C-context extraction")


def read_curve_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
