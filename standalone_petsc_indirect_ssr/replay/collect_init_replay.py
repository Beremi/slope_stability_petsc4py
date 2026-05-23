#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


FIELD_RE = re.compile(r"(\w+)=([^\s]+)")


def fields(line: str) -> dict[str, str]:
    return dict(FIELD_RE.findall(line))


def fnum(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inum(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def sample_meta(sample: Path) -> dict[str, object]:
    path = sample / "meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_log(path: Path) -> dict[str, object]:
    row: dict[str, object] = {"log": str(path)}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("INIT_REPLAY_META "):
            data = fields(line)
            row.update(
                {
                    "lambda": fnum(data.get("lambda")),
                    "newton_iteration": inum(data.get("it")),
                    "basis_cols": inum(data.get("basis_cols")),
                }
            )
        elif line.startswith("INIT_REPLAY_VEC_DIFF "):
            data = fields(line)
            label = data.get("label", "")
            if label in {"u_before_exported_minus_C", "f_free_exported_minus_C", "F_free_exported_minus_C", "rhs_exported_minus_C", "du_exported_minus_C_solve", "u_after_exported_minus_C_damping_on_exported_du"}:
                row[f"{label}_rel"] = fnum(data.get("rel_to_export"))
                row[f"{label}_diff"] = fnum(data.get("diff"))
        elif line.startswith("REPLAY_VEC_DIFF "):
            data = fields(line)
            label = data.get("label", "")
            if label.startswith("probe_du_"):
                row[f"{label}_rel"] = fnum(data.get("rel_to_a"))
                row[f"{label}_diff"] = fnum(data.get("diff"))
        elif line.startswith("REPLAY_PC_PROBE "):
            data = fields(line)
            label = data.get("label", "")
            if label:
                row[f"probe_{label}_h00_C"] = fnum(data.get("h00"))
                row[f"probe_{label}_h10_C"] = fnum(data.get("h10"))
        elif line.startswith("INIT_REPLAY_MATRIX_DIFF "):
            data = fields(line)
            label = data.get("label", "")
            if label:
                row[f"{label}_rel"] = fnum(data.get("rel_to_export"))
        elif line.startswith("INIT_REPLAY_LINEAR_RESULT "):
            data = fields(line)
            row.update(
                {
                    "exported_matrix": data.get("exported_matrix"),
                    "exported_rhs": data.get("exported_rhs"),
                    "expected_iterations": inum(data.get("expected_iterations")),
                    "c_iterations": inum(data.get("C_iterations")),
                    "expected_reported_final": fnum(data.get("expected_reported_final")),
                    "c_state_rel_residual": fnum(data.get("rel_residual_C_state")),
                }
            )
        elif line.startswith("INIT_REPLAY_DAMPING_COMPARE "):
            data = fields(line)
            row.update(
                {
                    "expected_alpha": fnum(data.get("expected_alpha")),
                    "c_alpha": fnum(data.get("C_alpha")),
                    "alpha_diff": fnum(data.get("alpha_diff")),
                    "expected_ls": inum(data.get("expected_ls")),
                    "c_ls": inum(data.get("C_ls")),
                    "expected_initial_decrease": fnum(data.get("expected_initial_decrease")),
                    "c_initial_decrease": fnum(data.get("C_initial_decrease")),
                    "expected_rel_correction": fnum(data.get("expected_rel_correction")),
                    "c_rel_correction": fnum(data.get("C_rel_correction")),
                }
            )
    return row


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: collect_init_replay.py LOG_DIR [EXPORT_ROOT]", file=sys.stderr)
        return 2
    log_dir = Path(argv[1])
    rows = [parse_log(path) for path in sorted(log_dir.glob("*.log"))]
    if len(argv) >= 3:
        export_root = Path(argv[2])
        by_sample = {sample.name: sample_meta(sample) for sample in export_root.glob("init_lambda_*")}
        for row in rows:
            log_name = Path(str(row["log"])).name
            for sample_name, meta in by_sample.items():
                if log_name.startswith(sample_name):
                    row["sample"] = sample_name
                    row.setdefault("lambda", meta.get("lambda"))
                    row.setdefault("newton_iteration", meta.get("newton_iteration"))
                    row.setdefault("basis_cols", meta.get("basis_cols"))
                    expected = meta.get("expected", {}) if isinstance(meta, dict) else {}
                    if isinstance(expected, dict):
                        row.setdefault("expected_iterations", expected.get("iterations"))
                        row.setdefault("expected_alpha", expected.get("alpha"))
                    break
    columns = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(sys.stdout, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
