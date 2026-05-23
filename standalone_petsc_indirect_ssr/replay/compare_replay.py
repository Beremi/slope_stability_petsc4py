#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")


def _kv_pairs(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^ \n]+)", line))


def _hist_from_text(text: str) -> dict[str, list[float]]:
    pat = re.compile(r'DEFLATED_MATLAB_DFGMRES label="REPLAY (d[WV])" it=\d+ rel=([0-9.eE+-]+)')
    out = {"dW": [], "dV": []}
    for label, value in pat.findall(text):
        out[label].append(float(value))
    return out


def _samples_from_text(text: str) -> dict[str, dict[str, float]]:
    initial_pat = re.compile(
        r'DEFLATED_MATLAB_DFGMRES_INITIAL label="REPLAY (d[WV])" '
        r'basis_cols=\d+ rhs_norm=([0-9.eE+-]+) beta=([0-9.eE+-]+) rel=([0-9.eE+-]+)'
    )
    sample_pat = re.compile(
        r'DEFLATED_MATLAB_DFGMRES_SAMPLE label="REPLAY (d[WV])" it=1 h00=([0-9.eE+-]+) h10=([0-9.eE+-]+)'
    )
    out: dict[str, dict[str, float]] = {"dW": {}, "dV": {}}
    for label, rhs_norm, beta, rel in initial_pat.findall(text):
        out[label].update({"rhs_norm": float(rhs_norm), "beta": float(beta), "initial_rel": float(rel)})
    for label, h00, h10 in sample_pat.findall(text):
        out[label].update({"h00": float(h00), "h10": float(h10)})
    return out


def _result_line(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("REPLAY_RESULT "):
            return line
    return ""


def _diagnostic_lines(text: str) -> list[str]:
    prefixes = ("REPLAY_MAP_CHECK ", "REPLAY_VEC_ROUNDTRIP ", "REPLAY_VEC_DIFF ")
    return [line for line in text.splitlines() if line.startswith(prefixes)]


def _vec_diffs_from_text(text: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line.startswith("REPLAY_VEC_DIFF "):
            continue
        data = _kv_pairs(line)
        label = data.get("label")
        if not label:
            continue
        out[label] = {
            "diff": float(data.get("diff", "nan")),
            "rel": float(data.get("rel_to_a", "nan")),
            "norm_a": float(data.get("norm_a", "nan")),
            "norm_b": float(data.get("norm_b", "nan")),
        }
    return out


def _pc_probe_from_text(text: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line.startswith("REPLAY_PC_PROBE "):
            continue
        data = _kv_pairs(line)
        label = data.get("label")
        if not label:
            continue
        out[label] = {
            "h00": float(data.get("h00", "nan")),
            "h10": float(data.get("h10", "nan")),
        }
    return out


def _map_status_from_text(text: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for line in text.splitlines():
        if line.startswith("REPLAY_MAP_CHECK "):
            data = _kv_pairs(line)
            out["map_missing"] = data.get("missing", "")
            out["map_duplicate"] = data.get("duplicate", "")
            out["map_max_coord_error"] = data.get("max_coord_error", "")
        elif line.startswith("REPLAY_VEC_ROUNDTRIP "):
            data = _kv_pairs(line)
            if data.get("label") == "u":
                out["u_roundtrip_rel"] = data.get("rel", data.get("rel_to_a", ""))
    return out


def _matrix_status_from_text(text: str) -> dict[str, str]:
    loaded = any(line.startswith("REPLAY_MATRIX ") and "status=loaded_permuted" in line for line in text.splitlines())
    result = _kv_pairs(_result_line(text))
    return {
        "exported_rhs_used": result.get("exported_rhs", ""),
        "exported_matrix_loaded": "true" if loaded else "false",
    }


def _history_rel_diff(expected: list[float], actual: list[float]) -> float:
    if len(expected) == len(actual) + 1:
        expected = expected[1:]
    if len(expected) != len(actual):
        return math.inf
    worst = 0.0
    for e, a in zip(expected, actual):
        denom = max(abs(e), 1.0e-300)
        worst = max(worst, abs(a - e) / denom)
    return worst


def _first_mismatch_layer(row: dict[str, object], *, include_assembly: bool) -> str:
    if str(row.get("map_missing", "0")) not in {"", "0"} or str(row.get("map_duplicate", "0")) not in {"", "0"}:
        return "mapping"
    if str(row.get("exported_matrix_loaded", "")).lower() != "true":
        return "operator_not_exported"
    assembly_keys = ("u_roundtrip_rel", "f_free_rel", "F_free_rel", "G_free_rel", "rhs_rel")
    if not include_assembly:
        assembly_keys = ("u_roundtrip_rel",)
    for key in assembly_keys:
        try:
            if float(row.get(key, "nan")) > 1.0e-10:
                return "assembly_or_rhs"
        except Exception:
            pass
    try:
        if float(row.get("coarse_initial_rel_diff", "nan")) > 1.0e-8:
            return "coarse_initial_guess"
    except Exception:
        pass
    for key in ("pc_v0_rel", "mg_fine_pre_rel", "mg_fine_residual_rel", "mg_p2_rhs_rel", "mg_p2_pre_rel", "mg_p1_x_rel"):
        try:
            if float(row.get(key, "nan")) > 1.0e-8:
                return "pmg_pcapply"
        except Exception:
            pass
    for key in ("Az0_rel", "arnoldi0_rel", "h00_rel_diff", "h10_rel_diff"):
        try:
            if float(row.get(key, "nan")) > 1.0e-8:
                return "arnoldi"
        except Exception:
            pass
    if row.get("iteration_match") is False:
        return "later_krylov_history"
    return "match"


def _fmt_hist(values: list[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _probe_scalars(sample_dir: Path, expected: dict, label: str) -> dict[str, float]:
    probe = expected.get(f"{label}_probe")
    if not isinstance(probe, dict):
        return {}
    json_name = probe.get("json")
    if not json_name:
        return {}
    path = sample_dir / str(json_name)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    scalars = payload.get("scalars", {})
    h_col0 = scalars.get("h_col0", [])
    out: dict[str, float] = {}
    for key in ("b_norm", "beta", "initial_rel"):
        if key in scalars:
            out[key if key != "b_norm" else "rhs_norm"] = float(scalars[key])
    if isinstance(h_col0, list) and len(h_col0) >= 2:
        out["h00"] = float(h_col0[0])
        out["h10"] = float(h_col0[1])
    return out


def _make_csv_rows(sample_dir: Path, log: Path, profile: str) -> list[dict[str, object]]:
    meta = json.loads((sample_dir / "meta.json").read_text())
    expected = meta["expected"]
    text = _read_text(log)
    hist = _hist_from_text(text)
    samples = _samples_from_text(text)
    result = _kv_pairs(_result_line(text))
    diffs = _vec_diffs_from_text(text)
    probes = _pc_probe_from_text(text)
    map_status = _map_status_from_text(text)
    rows: list[dict[str, object]] = []
    shared_diffs = {
        "f_free_rel": diffs.get("f_free_exported_minus_C", {}).get("rel", math.nan),
        "F_free_rel": diffs.get("F_free_exported_minus_C", {}).get("rel", math.nan),
        "G_free_rel": diffs.get("G_free_exported_minus_C", {}).get("rel", math.nan),
        "rhsW_rel": diffs.get("rhsW_exported_minus_C", {}).get("rel", math.nan),
        "rhsV_rel": diffs.get("rhsV_exported_minus_C", {}).get("rel", math.nan),
    }
    for label in ("dW", "dV"):
        expected_hist = [float(v) for v in expected.get(f"{label}_reported_residual_history", [])]
        actual_hist = hist[label]
        expected_probe = _probe_scalars(sample_dir, expected, label)
        actual_sample = samples.get(label, {})
        actual_probe = probes.get(label, {})
        solution_key = "solutionW_exported_minus_C" if label == "dW" else "solutionV_exported_minus_C"
        rhs_rel = shared_diffs["rhsW_rel"] if label == "dW" else shared_diffs["rhsV_rel"]
        matrix_status = _matrix_status_from_text(text)
        row: dict[str, object] = {
            "profile": profile,
            "log": str(log),
            "sample_dir": str(sample_dir),
            "sample_id": meta.get("sample_id", ""),
            "omega": meta.get("omega", ""),
            "lambda": meta.get("lambda", ""),
            "basis_cols_exported": meta.get("basis_cols", ""),
            "basis_cols_c": result.get("basis_cols", ""),
            "solve": label,
            "expected_iterations": int(expected.get(f"{label}_iterations", -1)),
            "c_iterations": len(actual_hist),
            "iteration_match": len(actual_hist) == int(expected.get(f"{label}_iterations", -1)),
            "expected_final_rel": expected_hist[-1] if expected_hist else math.nan,
            "c_final_rel": actual_hist[-1] if actual_hist else math.nan,
            "history_rel_max": _history_rel_diff(expected_hist, actual_hist),
            "expected_rhs_norm": expected_probe.get("rhs_norm", math.nan),
            "c_rhs_norm": actual_sample.get("rhs_norm", math.nan),
            "expected_beta": expected_probe.get("beta", math.nan),
            "c_beta": actual_sample.get("beta", math.nan),
            "expected_initial_rel": expected_probe.get("initial_rel", math.nan),
            "c_initial_rel": actual_sample.get("initial_rel", math.nan),
            "expected_h00": expected_probe.get("h00", math.nan),
            "c_h00": actual_sample.get("h00", actual_probe.get("h00", math.nan)),
            "expected_h10": expected_probe.get("h10", math.nan),
            "c_h10": actual_sample.get("h10", actual_probe.get("h10", math.nan)),
            "solution_rel": diffs.get(solution_key, {}).get("rel", math.nan),
            "solution_diff": diffs.get(solution_key, {}).get("diff", math.nan),
            "rhs_rel": rhs_rel,
            **matrix_status,
            **shared_diffs,
            **map_status,
        }
        for key in ("rhs_norm", "beta", "initial_rel", "h00", "h10"):
            e = row.get(f"expected_{key}")
            c = row.get(f"c_{key}")
            try:
                row[f"{key}_rel_diff"] = abs(float(c) - float(e)) / max(abs(float(e)), 1.0e-300)
            except Exception:
                row[f"{key}_rel_diff"] = math.nan
        stage_labels = {
            "pc_v0": f"probe_{label}_pc_v0_exported_minus_C",
            "z0": f"probe_{label}_z0_exported_minus_C",
            "Az0": f"probe_{label}_Az0_exported_minus_C",
            "arnoldi0": f"probe_{label}_arnoldi0_exported_minus_C",
            "mg_fine_pre": f"probe_{label}_mg_fine_pre_exported_minus_C",
            "mg_fine_residual": f"probe_{label}_mg_fine_residual_exported_minus_C",
            "mg_p2_rhs": f"probe_{label}_mg_p2_rhs_local_exported_minus_C",
            "mg_p2_pre": f"probe_{label}_mg_p2_pre_local_exported_minus_C",
            "mg_p2_residual": f"probe_{label}_mg_p2_residual_local_exported_minus_C",
            "mg_p1_rhs": f"probe_{label}_mg_p1_rhs_local_exported_minus_C",
            "mg_p1_x": f"probe_{label}_mg_p1_x_local_exported_minus_C",
            "mg_p2_post": f"probe_{label}_mg_p2_post_local_exported_minus_C",
        }
        for col, diff_label in stage_labels.items():
            row[f"{col}_rel"] = diffs.get(diff_label, {}).get("rel", math.nan)
            row[f"{col}_diff"] = diffs.get(diff_label, {}).get("diff", math.nan)
        row["strict_first_mismatch_layer"] = _first_mismatch_layer(row, include_assembly=True)
        row["linear_first_mismatch_layer"] = _first_mismatch_layer(row, include_assembly=False)
        row["first_mismatch_layer"] = row["linear_first_mismatch_layer"]
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "profile", "sample_id", "solve", "omega", "lambda", "basis_cols_exported", "basis_cols_c",
        "expected_iterations", "c_iterations", "iteration_match", "expected_final_rel", "c_final_rel", "history_rel_max",
        "expected_rhs_norm", "c_rhs_norm", "rhs_norm_rel_diff", "expected_beta", "c_beta", "beta_rel_diff",
        "expected_initial_rel", "c_initial_rel", "initial_rel_rel_diff", "expected_h00", "c_h00", "h00_rel_diff",
        "expected_h10", "c_h10", "h10_rel_diff", "solution_rel", "rhs_rel", "f_free_rel", "F_free_rel", "G_free_rel",
        "exported_rhs_used", "exported_matrix_loaded",
        "pc_v0_rel", "z0_rel", "Az0_rel", "arnoldi0_rel", "mg_fine_pre_rel", "mg_fine_residual_rel",
        "mg_p2_rhs_rel", "mg_p2_pre_rel", "mg_p2_residual_rel", "mg_p1_rhs_rel", "mg_p1_x_rel", "mg_p2_post_rel",
        "strict_first_mismatch_layer", "linear_first_mismatch_layer", "first_mismatch_layer", "log", "sample_dir",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="")
    ap.add_argument("--csv-out", type=Path)
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("logs", nargs="+", type=Path)
    ns = ap.parse_args()

    meta = json.loads((ns.sample_dir / "meta.json").read_text())
    expected = meta["expected"]
    print(f"sample={ns.sample_dir}")
    print(f"omega={meta['omega']:.8e} lambda={meta['lambda']:.8e} basis_cols={meta['basis_cols']}")
    for key in ("dW", "dV"):
        print(f"{key} petsc4py iters={expected[f'{key}_iterations']} hist={_fmt_hist(expected[f'{key}_reported_residual_history'])}")
        probe = _probe_scalars(ns.sample_dir, expected, key)
        if probe:
            print(
                f"{key} petsc4py sample "
                f"rhs_norm={probe.get('rhs_norm', float('nan')):.6e} "
                f"beta={probe.get('beta', float('nan')):.6e} "
                f"rel={probe.get('initial_rel', float('nan')):.6e} "
                f"h00={probe.get('h00', float('nan')):.6e} "
                f"h10={probe.get('h10', float('nan')):.6e}"
            )
    for log in ns.logs:
        text = _read_text(log)
        hist = _hist_from_text(text)
        samples = _samples_from_text(text)
        print(f"\nlog={log}")
        for key in ("dW", "dV"):
            print(f"{key} C iters={len(hist[key])} hist={_fmt_hist(hist[key])}")
            sample = samples.get(key, {})
            if sample:
                print(
                    f"{key} C sample "
                    f"rhs_norm={sample.get('rhs_norm', float('nan')):.6e} "
                    f"beta={sample.get('beta', float('nan')):.6e} "
                    f"rel={sample.get('initial_rel', float('nan')):.6e} "
                    f"h00={sample.get('h00', float('nan')):.6e} "
                    f"h10={sample.get('h10', float('nan')):.6e}"
                )
        line = _result_line(text)
        if line:
            print(line)
        for line in _diagnostic_lines(text):
            print(line)
    if ns.csv_out:
        rows: list[dict[str, object]] = []
        for log in ns.logs:
            profile = ns.profile or log.stem
            rows.extend(_make_csv_rows(ns.sample_dir, log, profile))
        _write_csv(ns.csv_out, rows)
        print(f"\nCSV {ns.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
