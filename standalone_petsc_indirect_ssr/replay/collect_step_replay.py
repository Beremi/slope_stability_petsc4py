#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")


def _kv(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^ \n]+)", line))


def _float(value: str | None, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int(value: str | None, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _max_rel_diff(text: str, labels: tuple[str, ...]) -> float:
    worst = 0.0
    found = False
    for line in text.splitlines():
        if not line.startswith("REPLAY_VEC_DIFF "):
            continue
        data = _kv(line)
        label = data.get("label", "")
        if label not in labels:
            continue
        found = True
        worst = max(worst, abs(_float(data.get("rel_to_a"))))
    return worst if found else math.nan


def _matrix_rel(text: str) -> float:
    vals = []
    for line in text.splitlines():
        if line.startswith("INIT_REPLAY_MATRIX_DIFF ") and "step_Areg_exported_minus_C" in line:
            vals.append(abs(_float(_kv(line).get("rel_to_export"))))
    return max(vals) if vals else math.nan


def _probe_rel(text: str) -> float:
    labels = (
        "probe_dW_pc_v0_exported_minus_C",
        "probe_dW_z0_exported_minus_C",
        "probe_dW_Az0_exported_minus_C",
        "probe_dW_arnoldi0_exported_minus_C",
        "probe_dV_pc_v0_exported_minus_C",
        "probe_dV_z0_exported_minus_C",
        "probe_dV_Az0_exported_minus_C",
        "probe_dV_arnoldi0_exported_minus_C",
    )
    return _max_rel_diff(text, labels)


def _first_mismatch(row: dict[str, object], assembly_rel: float, matrix_rel: float, probe_rel: float) -> str:
    # G=dF/dlambda is exported from a finite-difference perturbation, so its
    # replay agreement is naturally a little looser than direct residual/matrix
    # comparisons. Treat O(1e-8) as assembly noise and let the PMG probe expose
    # the first meaningful semantic mismatch.
    if math.isfinite(assembly_rel) and assembly_rel > 1.0e-7:
        return "assembly_or_rhs"
    if math.isfinite(matrix_rel) and matrix_rel > 1.0e-10:
        return "matrix_action"
    if math.isfinite(probe_rel) and probe_rel > 1.0e-8:
        return "pmg_pcapply"
    if int(row["expected_w"]) != int(row["c_w"]) or int(row["expected_v"]) != int(row["c_v"]):
        return "linear_iterations"
    if abs(float(row["expected_alpha"]) - float(row["c_alpha"])) > 1.0e-12:
        return "damping_alpha"
    if abs(float(row["expected_lambda"]) - float(row["c_lambda"])) > 1.0e-10 * max(1.0, abs(float(row["expected_lambda"]))):
        return "lambda_propagation"
    if int(row["expected_basis"]) != int(row["c_basis"]):
        return "deflation_basis"
    return "match"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--profile", default="")
    parser.add_argument("--csv-out", type=Path, required=True)
    args = parser.parse_args()

    text = _read(args.log)
    assembly_rel = _max_rel_diff(
        text,
        (
            "step_f_free_exported_minus_C",
            "step_F_free_exported_minus_C",
            "step_G_free_exported_minus_C",
            "step_rhsW_exported_minus_C",
            "step_rhsV_exported_minus_C",
        ),
    )
    mat_rel = _matrix_rel(text)
    pc_rel = _probe_rel(text)
    result = {}
    for line in text.splitlines():
        if line.startswith("STEP_REPLAY_RESULT "):
            result = _kv(line)

    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line.startswith("STEP_REPLAY_NEWTON_COMPARE "):
            continue
        data = _kv(line)
        row: dict[str, object] = {
            "profile": args.profile,
            "it": _int(data.get("it")),
            "expected_w": _int(data.get("expected_w")),
            "c_w": _int(data.get("c_w")),
            "expected_v": _int(data.get("expected_v")),
            "c_v": _int(data.get("c_v")),
            "expected_total": _int(data.get("expected_total")),
            "c_total": _int(data.get("c_total")),
            "expected_alpha": _float(data.get("expected_alpha")),
            "c_alpha": _float(data.get("c_alpha")),
            "expected_lambda": _float(data.get("expected_lambda")),
            "c_lambda": _float(data.get("c_lambda")),
            "expected_r": _float(data.get("expected_r")),
            "c_r": _float(data.get("c_r")),
            "expected_rel_res": _float(data.get("expected_rel_res")),
            "c_rel_res": _float(data.get("c_rel_res")),
            "expected_rel_corr": _float(data.get("expected_rel_corr")),
            "c_rel_corr": _float(data.get("c_rel_corr")),
            "expected_ls": _int(data.get("expected_ls")),
            "c_ls": _int(data.get("c_ls")),
            "expected_basis": _int(data.get("expected_basis")),
            "c_basis": _int(data.get("c_basis")),
            "assembly_rel_max": assembly_rel,
            "matrix_action_rel": mat_rel,
            "probe_rel_max": pc_rel,
            "sample_dir": str(args.sample_dir),
            "log": str(args.log),
        }
        row["first_mismatch_layer"] = _first_mismatch(row, assembly_rel, mat_rel, pc_rel)
        rows.append(row)

    if not rows and result:
        rows.append(
            {
                "profile": args.profile,
                "it": -1,
                "expected_w": -1,
                "c_w": -1,
                "expected_v": -1,
                "c_v": -1,
                "expected_total": -1,
                "c_total": _int(result.get("linear_its")),
                "assembly_rel_max": assembly_rel,
                "matrix_action_rel": mat_rel,
                "probe_rel_max": pc_rel,
                "first_mismatch_layer": "no_expected_rows",
                "sample_dir": str(args.sample_dir),
                "log": str(args.log),
            }
        )

    fields = [
        "profile",
        "it",
        "expected_w",
        "c_w",
        "expected_v",
        "c_v",
        "expected_total",
        "c_total",
        "expected_alpha",
        "c_alpha",
        "expected_lambda",
        "c_lambda",
        "expected_r",
        "c_r",
        "expected_rel_res",
        "c_rel_res",
        "expected_rel_corr",
        "c_rel_corr",
        "expected_ls",
        "c_ls",
        "expected_basis",
        "c_basis",
        "assembly_rel_max",
        "matrix_action_rel",
        "probe_rel_max",
        "first_mismatch_layer",
        "sample_dir",
        "log",
    ]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
