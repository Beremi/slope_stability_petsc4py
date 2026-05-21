#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


EVENTS = (
    "PCApply",
    "KSPSolve",
    "MatMult",
    "VecScatterBegin",
    "VecScatterEnd",
    "VecMDot",
    "VecMAXPY",
    "VecNorm",
    "VecDot",
    "KSPGMRESOrthog",
    "MatPtAPNumeric",
    "MatPtAPSymbolic",
    "PCSetUp",
)

SUMMARY_FIELDS = (
    "job_id",
    "run_label",
    "engine",
    "profile",
    "nodes",
    "tasks_per_node",
    "ranks",
    "partition",
    "qos",
    "nodelist",
    "state",
    "exit_code",
    "elapsed",
    "maxrss",
    "averss",
    "maxrss_gib_per_rank",
    "averss_gib_per_rank",
    "approx_total_averss_gib",
    "status",
    "accepted_steps",
    "omega_last",
    "lambda_last",
    "total_newton_its",
    "total_linear_its",
    "total_line_search_its",
    "init_linear_iterations",
    "attempt_linear_iterations_total",
    "elastic_assembly_time",
    "continuation_wall_time",
    "wall_time",
    "time_v_wall_s",
    "final_rel",
    "final_rel_correction",
    "stop_reason",
    "deflation_basis_cols",
    "deflation_orthogonalization_time",
    "deflation_coarse_initial_time",
    "deflation_pc_apply_time",
    "deflation_projector_time",
    "pmg_apply_calls",
    "pmg_operator_updates",
    "pmg_fine_smooth",
    "pmg_p2_smooth",
    "pmg_restrict",
    "pmg_prolong",
    "pmg_coarse_solve",
    "pmg_residual",
    "pmg_operator_update",
    "py_constitutive_reduction",
    "py_constitutive_stress",
    "py_constitutive_stress_tangent",
    "py_constitutive_build_F",
    "py_constitutive_build_F_K_tangent",
    "py_constitutive_build_tangent_local",
    "py_constitutive_local_strain",
    "py_constitutive_local_constitutive",
    "py_constitutive_local_constitutive_comm",
    "py_constitutive_local_force_assembly",
    "py_constitutive_local_force_gather",
    "py_linear_solve_time",
    "py_linear_preconditioner_time",
    "py_linear_orthogonalization_time",
    "py_preconditioner_setup_time_total",
    "py_preconditioner_apply_time_total",
    "py_manualmg_setup_time_s",
    "py_manualmg_apply_count",
    "py_manualmg_fine_pre_smoother_time_total_s",
    "py_manualmg_fine_post_smoother_time_total_s",
    "py_manualmg_mid_pre_smoother_time_total_s",
    "py_manualmg_mid_post_smoother_time_total_s",
    "py_manualmg_restrict_fine_to_mid_time_total_s",
    "py_manualmg_restrict_mid_to_coarse_time_total_s",
    "py_manualmg_prolong_coarse_to_mid_time_total_s",
    "py_manualmg_prolong_mid_to_fine_time_total_s",
    "py_manualmg_fine_residual_time_total_s",
    "py_manualmg_mid_residual_time_total_s",
    "py_manualmg_vector_sum_time_total_s",
    "py_manualmg_coarse_hypre_time_total_s",
    "py_manualmg_coarse_ksp_type",
    "py_manualmg_coarse_pc_type",
    "py_manualmg_fine_iterations",
    "py_manualmg_mid_iterations",
    "py_manualmg_coarse_ksp_iterations_total",
    "log",
    "result_dir",
)

STEP_FIELDS = (
    "job_id",
    "run_label",
    "engine",
    "profile",
    "nodes",
    "tasks_per_node",
    "ranks",
    "step",
    "phase",
    "omega",
    "lambda",
    "d_omega",
    "d_lambda",
    "attempts",
    "newton_iterations",
    "linear_iterations",
    "line_search_iterations",
    "rel_residual",
    "rel_correction",
    "step_wall_time",
    "linear_solve_time",
    "linear_preconditioner_time",
    "linear_orthogonalization_time",
    "deflation_basis_dim",
    "stop_reason",
)

EVENT_FIELDS = (
    "job_id",
    "run_label",
    "engine",
    "profile",
    "nodes",
    "tasks_per_node",
    "ranks",
    "event",
    "count",
    "time",
    "flops",
    "messages",
    "message_length",
    "reductions",
)


def read_text(path: Path) -> str:
    try:
        return path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def read_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in read_text(path).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def kv_from_line(line: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^ ]+)", line or ""))


def coerce(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    text = str(value)
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def parse_time_v(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    wall = re.search(r"Elapsed \(wall clock\) time.*: (?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", text)
    if wall:
        hours = int(wall.group(1) or 0)
        minutes = int(wall.group(2))
        seconds = float(wall.group(3))
        out["time_v_wall_s"] = hours * 3600 + minutes * 60 + seconds
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if rss:
        out["maxrss_kb_time_v"] = int(rss.group(1))
    return out


def sacct_value(path: Path, column: str) -> str:
    text = read_text(path)
    if not text.strip():
        return ""
    rows = [line.split("|") for line in text.splitlines() if line.strip()]
    if not rows:
        return ""
    header = rows[0]
    try:
        idx = header.index(column)
    except ValueError:
        return ""
    fallback = ""
    batch = ""
    for row in rows[1:]:
        if idx >= len(row):
            continue
        jobid = row[0] if row else ""
        value = row[idx]
        if re.search(r"\.\d+$", jobid) and value:
            return value
        if jobid.endswith(".batch") and value:
            batch = value
        if not fallback and value:
            fallback = value
    return batch or fallback


def rss_to_gib(value: str) -> float | str:
    if not value:
        return ""
    match = re.match(r"([0-9.]+)([KMGT]?)$", value.strip())
    if not match:
        return ""
    val = float(match.group(1))
    unit = match.group(2)
    scale = {"": 1.0 / 1024.0, "K": 1.0, "M": 1024.0, "G": 1024.0 * 1024.0, "T": 1024.0 * 1024.0 * 1024.0}[unit]
    return val * scale / (1024.0 * 1024.0)


def flatten_json(prefix: str, mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {f"{prefix}{key}": mapping.get(key, "") for key in keys if key in mapping}


def parse_event_rows(log_text: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if parts[0] not in EVENTS or len(parts) < 4:
            continue
        row = {key: meta.get(key, "") for key in ("job_id", "run_label", "engine", "profile", "nodes", "tasks_per_node", "ranks")}
        row.update(
            {
                "event": parts[0],
                "count": parts[1] if len(parts) > 1 else "",
                "time": parts[3] if len(parts) > 3 else "",
                "flops": parts[5] if len(parts) > 5 else "",
                "messages": parts[7] if len(parts) > 7 else "",
                "message_length": parts[8] if len(parts) > 8 else "",
                "reductions": parts[9] if len(parts) > 9 else "",
            }
        )
        rows.append(row)
    return rows


def event_time_map(event_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {f"petsc_{row['event']}_time": row.get("time", "") for row in event_rows}


def base_row(result_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    sacct = result_dir / "sacct.txt"
    row: dict[str, Any] = {
        "job_id": env.get("JOB_ID", ""),
        "run_label": env.get("RUN_LABEL", result_dir.name),
        "engine": env.get("ENGINE", ""),
        "profile": env.get("PROFILE", ""),
        "nodes": env.get("NODES", ""),
        "tasks_per_node": env.get("TASKS_PER_NODE", ""),
        "ranks": env.get("RANKS", ""),
        "partition": env.get("PARTITION", ""),
        "qos": env.get("QOS", ""),
        "nodelist": env.get("NODELIST", ""),
        "exit_code": env.get("EXIT_CODE", ""),
        "result_dir": str(result_dir),
        "log": str(result_dir / "run.log"),
    }
    row["state"] = sacct_value(sacct, "State")
    row["elapsed"] = sacct_value(sacct, "Elapsed")
    row["maxrss"] = sacct_value(sacct, "MaxRSS")
    row["averss"] = sacct_value(sacct, "AveRSS")
    maxrss_gib = rss_to_gib(str(row["maxrss"]))
    averss_gib = rss_to_gib(str(row["averss"]))
    row["maxrss_gib_per_rank"] = maxrss_gib
    row["averss_gib_per_rank"] = averss_gib
    try:
        row["approx_total_averss_gib"] = float(averss_gib) * float(row["ranks"])
    except (TypeError, ValueError):
        row["approx_total_averss_gib"] = ""
    return row


def update_c_result(row: dict[str, Any], log_text: str, result_dir: Path) -> None:
    result_line = ""
    ssr_line = ""
    defl_line = ""
    pmg_line = ""
    for line in log_text.splitlines():
        if line.startswith("RESULT "):
            result_line = line
        elif line.startswith("SSR_RESULT "):
            ssr_line = line
        elif line.startswith("DEFLATION_TIMING "):
            defl_line = line
        elif line.startswith("PMG_SHELL_APPLY_SUMMARY "):
            pmg_line = line
    result = kv_from_line(result_line)
    ssr = kv_from_line(ssr_line)
    defl = kv_from_line(defl_line)
    pmg = kv_from_line(pmg_line)
    source = {**ssr, **result}
    row["status"] = "done" if result_line else ("failed" if log_text else "missing")
    for src_key, dst_key in (
        ("accepted_steps", "accepted_steps"),
        ("total_newton_its", "total_newton_its"),
        ("total_newton_iterations", "total_newton_its"),
        ("total_linear_its", "total_linear_its"),
        ("total_linear_iterations", "total_linear_its"),
        ("total_line_search_its", "total_line_search_its"),
        ("total_line_search_iterations", "total_line_search_its"),
        ("omega_last", "omega_last"),
        ("lambda_last", "lambda_last"),
        ("elastic_assembly_time", "elastic_assembly_time"),
        ("continuation_wall_time", "continuation_wall_time"),
        ("wall_time", "wall_time"),
        ("final_rel", "final_rel"),
        ("final_rel_correction", "final_rel_correction"),
        ("stop_reason", "stop_reason"),
        ("deflation_basis_cols", "deflation_basis_cols"),
    ):
        if src_key in source:
            row[dst_key] = source[src_key]
    for src_key, dst_key in (
        ("orthogonalization_time", "deflation_orthogonalization_time"),
        ("coarse_initial_time", "deflation_coarse_initial_time"),
        ("pc_apply_time", "deflation_pc_apply_time"),
        ("projector_time", "deflation_projector_time"),
    ):
        if src_key in defl:
            row[dst_key] = defl[src_key]
    for src_key, dst_key in (
        ("apply_calls", "pmg_apply_calls"),
        ("operator_updates", "pmg_operator_updates"),
        ("fine_smooth", "pmg_fine_smooth"),
        ("p2_smooth", "pmg_p2_smooth"),
        ("restrict", "pmg_restrict"),
        ("prolong", "pmg_prolong"),
        ("coarse_solve", "pmg_coarse_solve"),
        ("residual", "pmg_residual"),
        ("operator_update", "pmg_operator_update"),
    ):
        if src_key in pmg:
            row[dst_key] = pmg[src_key]
    row.update(parse_time_v(read_text(result_dir / "time.txt")))


def c_step_rows(result_dir: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    curve = result_dir / "continuation_curve.csv"
    if not curve.exists():
        return []
    rows: list[dict[str, Any]] = []
    with curve.open(newline="") as fh:
        for rec in csv.DictReader(fh):
            row = {key: meta.get(key, "") for key in ("job_id", "run_label", "engine", "profile", "nodes", "tasks_per_node", "ranks")}
            row.update(
                {
                    "step": rec.get("step", ""),
                    "phase": rec.get("phase", ""),
                    "omega": rec.get("omega", ""),
                    "lambda": rec.get("lambda", ""),
                    "d_omega": rec.get("d_omega", ""),
                    "d_lambda": rec.get("d_lambda", ""),
                    "attempts": rec.get("attempts", ""),
                    "newton_iterations": rec.get("newton_iterations", ""),
                    "linear_iterations": rec.get("linear_iterations", ""),
                    "line_search_iterations": rec.get("line_search_iterations", ""),
                    "rel_residual": rec.get("rel_residual", ""),
                    "rel_correction": rec.get("rel_correction", ""),
                    "step_wall_time": rec.get("step_wall_time", ""),
                    "stop_reason": rec.get("stop_reason", ""),
                }
            )
            rows.append(row)
    return rows


def update_py_result(row: dict[str, Any], result_dir: Path, log_text: str) -> None:
    info_path = result_dir / "out" / "data" / "run_info.json"
    row["status"] = "done" if info_path.exists() else ("failed" if log_text else "missing")
    if not info_path.exists():
        row.update(parse_time_v(read_text(result_dir / "time.txt")))
        return
    info = json.loads(info_path.read_text())
    run_info = info.get("run_info", {}) if isinstance(info.get("run_info"), dict) else {}
    timings = info.get("timings", {}) if isinstance(info.get("timings"), dict) else {}
    linear = timings.get("linear", {}) if isinstance(timings.get("linear"), dict) else {}
    constitutive = timings.get("constitutive", {}) if isinstance(timings.get("constitutive"), dict) else {}
    row["wall_time"] = info.get("runtime", run_info.get("runtime_seconds", ""))
    row["continuation_wall_time"] = timings.get("continuation_total_wall_time", "")
    row["accepted_steps"] = info.get("steps", run_info.get("step_count", ""))
    row["omega_last"] = info.get("omega_last", "")
    row["lambda_last"] = info.get("lambda_last", "")
    row["total_newton_its"] = info.get("newton_iterations_total", info.get("total_newton_iterations", ""))
    init_lin = linear.get("init_linear_iterations", "")
    attempt_lin = linear.get("attempt_linear_iterations_total", "")
    row["init_linear_iterations"] = init_lin
    row["attempt_linear_iterations_total"] = attempt_lin
    row["total_linear_its"] = info.get("linear_iterations_total", "")
    if row["total_linear_its"] == "" and init_lin != "" and attempt_lin != "":
        row["total_linear_its"] = int(init_lin) + int(attempt_lin)
    row["stop_reason"] = info.get("stop_reason", "")
    row.update(flatten_json("py_constitutive_", constitutive, (
        "reduction",
        "stress",
        "stress_tangent",
        "build_F",
        "build_F_K_tangent",
        "build_tangent_local",
        "local_strain",
        "local_constitutive",
        "local_constitutive_comm",
        "local_force_assembly",
        "local_force_gather",
    )))
    if linear:
        row["py_linear_solve_time"] = float(linear.get("init_linear_solve_time", 0.0)) + float(linear.get("attempt_linear_solve_time_total", 0.0))
        row["py_linear_preconditioner_time"] = float(linear.get("init_linear_preconditioner_time", 0.0)) + float(linear.get("attempt_linear_preconditioner_time_total", 0.0))
        row["py_linear_orthogonalization_time"] = float(linear.get("init_linear_orthogonalization_time", 0.0)) + float(linear.get("attempt_linear_orthogonalization_time_total", 0.0))
        for key in (
            "preconditioner_setup_time_total",
            "preconditioner_apply_time_total",
            "manualmg_setup_time_s",
            "manualmg_apply_count",
            "manualmg_fine_pre_smoother_time_total_s",
            "manualmg_fine_post_smoother_time_total_s",
            "manualmg_mid_pre_smoother_time_total_s",
            "manualmg_mid_post_smoother_time_total_s",
            "manualmg_restrict_fine_to_mid_time_total_s",
            "manualmg_restrict_mid_to_coarse_time_total_s",
            "manualmg_prolong_coarse_to_mid_time_total_s",
            "manualmg_prolong_mid_to_fine_time_total_s",
            "manualmg_fine_residual_time_total_s",
            "manualmg_mid_residual_time_total_s",
            "manualmg_vector_sum_time_total_s",
            "manualmg_coarse_hypre_time_total_s",
            "manualmg_coarse_ksp_type",
            "manualmg_coarse_pc_type",
            "manualmg_fine_iterations",
            "manualmg_mid_iterations",
            "manualmg_coarse_ksp_iterations_total",
        ):
            if key in linear:
                row[f"py_{key}"] = linear[key]
    progress = result_dir / "out" / "data" / "progress.jsonl"
    init_newton_total = 0
    cont_newton_total = 0
    cont_linear_total = 0
    line_search_total = 0
    basis_last = ""
    for line in read_text(progress).splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "init_complete":
            init_newton_total = sum(int(v) for v in rec.get("init_newton_iterations", []))
            if row.get("init_linear_iterations", "") == "":
                row["init_linear_iterations"] = rec.get("init_linear_iterations", "")
        elif rec.get("event") == "step_accepted":
            cont_newton_total += int(rec.get("step_newton_iterations") or 0)
            cont_linear_total += int(rec.get("step_linear_iterations") or 0)
            line_search_total += int(rec.get("line_search_iterations") or 0)
            basis_last = rec.get("deflation_basis_dim_end_last", basis_last)
            row["omega_last"] = rec.get("omega_value", row.get("omega_last", ""))
            row["lambda_last"] = rec.get("lambda_value", row.get("lambda_last", ""))
    if row.get("total_newton_its", "") in {"", -1} and init_newton_total + cont_newton_total:
        row["total_newton_its"] = init_newton_total + cont_newton_total
    if row.get("total_linear_its", "") in {"", -1} and row.get("init_linear_iterations", "") != "":
        row["total_linear_its"] = int(row["init_linear_iterations"]) + cont_linear_total
    if line_search_total:
        row["total_line_search_its"] = line_search_total
    if basis_last != "":
        row["deflation_basis_cols"] = basis_last
    done_match = re.search(r"\[done\].*?steps=(\d+).*?lambda=([0-9.eE+-]+).*?omega=([0-9.eE+-]+).*?reason=([A-Za-z0-9_+-]+)", log_text)
    if done_match:
        row["accepted_steps"] = done_match.group(1)
        row["lambda_last"] = done_match.group(2)
        row["omega_last"] = done_match.group(3)
        row["stop_reason"] = done_match.group(4)
    row.update(parse_time_v(read_text(result_dir / "time.txt")))


def py_step_rows(result_dir: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    progress = result_dir / "out" / "data" / "progress.jsonl"
    rows: list[dict[str, Any]] = []
    for line in read_text(progress).splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") not in {"init_complete", "step_accepted"}:
            continue
        row = {key: meta.get(key, "") for key in ("job_id", "run_label", "engine", "profile", "nodes", "tasks_per_node", "ranks")}
        if rec.get("event") == "init_complete":
            row.update(
                {
                    "step": rec.get("accepted_steps", 2),
                    "phase": "init",
                    "omega": ";".join(str(v) for v in rec.get("omega_hist", [])),
                    "lambda": ";".join(str(v) for v in rec.get("lambda_hist", [])),
                    "newton_iterations": ";".join(str(v) for v in rec.get("init_newton_iterations", [])),
                    "linear_iterations": rec.get("init_linear_iterations", ""),
                    "linear_solve_time": rec.get("init_linear_solve_time", ""),
                    "linear_preconditioner_time": rec.get("init_linear_preconditioner_time", ""),
                    "linear_orthogonalization_time": rec.get("init_linear_orthogonalization_time", ""),
                    "step_wall_time": rec.get("total_wall_time", ""),
                    "stop_reason": "init_complete",
                }
            )
        else:
            row.update(
                {
                    "step": rec.get("accepted_step", ""),
                    "phase": "continuation",
                    "omega": rec.get("omega_value", ""),
                    "lambda": rec.get("lambda_value", ""),
                    "d_omega": rec.get("d_omega", ""),
                    "d_lambda": rec.get("d_lambda", ""),
                    "attempts": rec.get("step_attempt_count", ""),
                    "newton_iterations": rec.get("step_newton_iterations", ""),
                    "linear_iterations": rec.get("step_linear_iterations", ""),
                    "line_search_iterations": rec.get("line_search_iterations", ""),
                    "rel_residual": rec.get("step_newton_relres_end", ""),
                    "rel_correction": rec.get("step_newton_relcorr_end", ""),
                    "step_wall_time": rec.get("step_wall_time", ""),
                    "linear_solve_time": rec.get("step_linear_solve_time", ""),
                    "linear_preconditioner_time": rec.get("step_linear_preconditioner_time", ""),
                    "linear_orthogonalization_time": rec.get("step_linear_orthogonalization_time", ""),
                    "deflation_basis_dim": rec.get("deflation_basis_dim_end_last", ""),
                    "stop_reason": "accepted",
                }
            )
        rows.append(row)
    return rows


def collect(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    results_dir = run_root / "results"
    summary_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for result_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        env = read_env(result_dir / "job.env")
        row = base_row(result_dir, env)
        log_text = read_text(result_dir / "run.log")
        if row.get("engine") == "c":
            update_c_result(row, log_text, result_dir)
            steps = c_step_rows(result_dir, row)
        elif row.get("engine") == "py":
            update_py_result(row, result_dir, log_text)
            steps = py_step_rows(result_dir, row)
        else:
            row["status"] = "unknown_engine"
            steps = []
        events = parse_event_rows(log_text, row)
        row.update(event_time_map(events))
        summary_rows.append(row)
        step_rows.extend(steps)
        event_rows.extend(events)
    return summary_rows, step_rows, event_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    extra_fields = sorted({key for row in rows for key in row if key not in fields})
    all_fields = list(fields) + extra_fields
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: coerce(row.get(key, "")) for key in all_fields})


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} RUN_ROOT", file=sys.stderr)
        return 2
    run_root = Path(sys.argv[1]).resolve()
    results_dir = run_root / "results"
    if not results_dir.is_dir():
        print(f"ERROR: missing results directory: {results_dir}", file=sys.stderr)
        return 2
    summary_rows, step_rows, event_rows = collect(run_root)
    write_csv(run_root / "ssr_omega7_results.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(run_root / "ssr_omega7_steps.csv", step_rows, STEP_FIELDS)
    write_csv(run_root / "ssr_omega7_petsc_events.csv", event_rows, EVENT_FIELDS)
    print(f"Wrote {run_root / 'ssr_omega7_results.csv'}")
    print(f"Wrote {run_root / 'ssr_omega7_steps.csv'}")
    print(f"Wrote {run_root / 'ssr_omega7_petsc_events.csv'}")
    if summary_rows:
        compact = ("engine", "profile", "nodes", "tasks_per_node", "ranks", "status", "accepted_steps", "total_newton_its", "total_linear_its", "wall_time", "continuation_wall_time", "time_v_wall_s", "stop_reason")
        print(",".join(compact))
        for row in summary_rows:
            print(",".join(str(coerce(row.get(field, ""))) for field in compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
