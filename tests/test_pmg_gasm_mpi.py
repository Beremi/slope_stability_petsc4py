from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "mpi_pmg_gasm_smoother_check.py"


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec is not available")
def test_pmg_shell_gasm_smoother_mpi_smoke() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        "OMPI_MCA_mpi_yield_when_idle": "1",
    }
    proc = subprocess.run(
        ["mpiexec", "-n", "4", sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "unused database options" not in proc.stderr.lower()
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    assert payload["pc_type"] == "gasm"
    assert payload["sub_ksp_type"] == "preonly"
    assert payload["total_subdomains"] == 2
    assert payload["ranks_per_subdomain"] == 2
    assert payload["residual_norm_max"] >= 0.0
    assert payload["solution_norm_local"] >= 0.0
