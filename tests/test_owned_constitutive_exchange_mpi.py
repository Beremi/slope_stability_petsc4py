from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "mpi_owned_constitutive_exchange_check.py"


@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="mpiexec is not available")
@pytest.mark.parametrize("ranks", [2, 4])
def test_owned_constitutive_modes_match_under_mpi(ranks: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OMPI_MCA_rmaps_base_oversubscribe", "1")

    subprocess.run(
        [
            "mpiexec",
            "-n",
            str(int(ranks)),
            sys.executable,
            str(SCRIPT),
            "--elem-type",
            "P2",
            "--elems-per-rank",
            "3",
            "--rtol",
            "1e-11",
            "--atol",
            "1e-11",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
