from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "mpi_preconditioner_linear_check.py"
OVERLAP_SCRIPT = ROOT / "tests" / "mpi_bddc_overlap_check.py"


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec is not available")
@pytest.mark.parametrize("ranks", [2, 4])
def test_preconditioner_mpi_smoke(ranks: int) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        ["mpiexec", "-n", str(int(ranks)), sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    results = {entry["case"]: entry for entry in payload["results"]}

    assert set(results) == {"hypre_current", "hypre_lagged_current", "bddc"}
    assert results["hypre_current"]["pc_backend"] == "hypre"
    assert results["hypre_lagged_current"]["preconditioner_matrix_policy"] == "lagged"
    assert results["bddc"]["pc_backend"] == "bddc"
    for entry in results.values():
        assert entry["preconditioner_rebuild_count"] >= 1
        assert entry["solution_norm"] >= 0.0
        assert entry["residual_norm"] >= 0.0


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec is not available")
def test_bddc_overlap_matis_matches_owned_vector_layout() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        ["mpiexec", "-n", "2", sys.executable, str(OVERLAP_SCRIPT)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    results = payload["results"]
    assert len(results) == 2
    for entry in results:
        assert entry["pc_backend"] == "bddc"
        assert entry["preconditioner_rebuild_count"] == 1
        assert entry["residual_norm"] < 1.0e-10
