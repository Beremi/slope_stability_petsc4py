from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from petsc4py import PETSc

from .config.manifest import build_environment_manifest, build_resolved_config, build_resolved_run_manifest, dumps_resolved_config_toml
from .options import SsrOptions
from .problem import ProblemSpec
from .runtime.options import quote_option_tokens, resolve_run_option_tokens


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
        return resolve_run_option_tokens(self.problem, self.options, self.output_dir)

    def options_string(self) -> str:
        return quote_option_tokens(self.option_tokens())

    def run(self) -> EngineRunResult:
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

    def debug_engine_ops(self):
        """Return the Python debug-loop compatibility wrapper for the native engine."""
        from .operations import EngineOps

        return EngineOps(self.create_engine())

    def prepare_output_dir(self) -> None:
        comm = PETSc.COMM_WORLD.tompi4py()
        data_dir = self.output_dir / "data"
        if self.rank == 0:
            mpi_size = int(PETSc.COMM_WORLD.getSize())
            data_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "logs").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "exports").mkdir(parents=True, exist_ok=True)
            (data_dir / "problem.json").write_text(json.dumps(self.problem.to_dict(), indent=2), encoding="utf-8")
            (data_dir / "options.txt").write_text(self.options_string() + "\n", encoding="utf-8")
            (data_dir / "resolved_options.txt").write_text(self.options_string() + "\n", encoding="utf-8")
            (data_dir / "environment.json").write_text(
                json.dumps(build_environment_manifest(mpi_size=mpi_size), indent=2) + "\n",
                encoding="utf-8",
            )
            (data_dir / "resolved_run_manifest.json").write_text(
                json.dumps(
                    build_resolved_run_manifest(
                        self.problem,
                        self.options,
                        output_dir=self.output_dir,
                        mpi_size=mpi_size,
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (data_dir / "resolved_config.toml").write_text(
                dumps_resolved_config_toml(
                    build_resolved_config(
                        self.problem,
                        self.options,
                        output_dir=self.output_dir,
                        mpi_size=mpi_size,
                    )
                ),
                encoding="utf-8",
            )
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


def read_curve_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
