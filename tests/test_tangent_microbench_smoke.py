from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "3d_hetero_ssr_default" / "archive" / "bench_tangent_kernels.py"


def test_tangent_microbenchmark_smoke(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    out_dir = tmp_path / "microbench"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "small",
            "--elem-type",
            "P2",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--no-use-compiled",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    csv_path = out_dir / "summary.csv"
    assert csv_path.exists()
    assert payload["mode"] == "small"
    assert payload["elem_type"] == "P2"
    assert "legacy" in payload["results"]
    assert "rows" in payload["results"]
