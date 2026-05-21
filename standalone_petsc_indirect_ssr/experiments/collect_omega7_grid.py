#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    except FileNotFoundError:
        return ""


def _parse_time_v(text: str) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    wall = re.search(r"Elapsed \(wall clock\) time.*: (?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", text)
    if wall:
        hours = int(wall.group(1) or 0)
        minutes = int(wall.group(2))
        seconds = float(wall.group(3))
        out["time_v_wall_s"] = hours * 3600 + minutes * 60 + seconds
    bash_wall = re.search(r"^real\s+((?:(\d+)h)?(?:(\d+)m)?(\d+(?:\.\d+)?)s)$", text, re.M)
    if bash_wall:
        hours = int(bash_wall.group(2) or 0)
        minutes = int(bash_wall.group(3) or 0)
        seconds = float(bash_wall.group(4))
        out["time_v_wall_s"] = hours * 3600 + minutes * 60 + seconds
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if rss:
        out["max_rss_kb"] = int(rss.group(1))
    return out


def _parse_c(case_dir: Path, engine: str, profile: str, ranks: int) -> dict[str, object]:
    log = _read_text(case_dir / "run.log")
    row: dict[str, object] = {"engine": engine, "profile": profile, "ranks": ranks, "status": "missing"}
    if log:
        row["status"] = "done" if "RESULT " in log else "failed"
    result = ""
    for line in log.splitlines():
        if line.startswith("RESULT "):
            result = line
    if result:
        for key, value in re.findall(r"(\w+)=([^ ]+)", result):
            if key in {"accepted_steps", "total_newton_its", "total_linear_its", "total_line_search_its", "global_dofs", "deflation_basis_cols"}:
                row[key] = int(float(value))
            elif key in {"elastic_assembly_time", "continuation_wall_time", "wall_time", "omega_last", "lambda_last", "final_rel", "final_rel_correction"}:
                row[key] = float(value)
            elif key in {"stop_reason", "deflation", "deflation_solver", "partitioner"}:
                row[key] = value
    row.update(_parse_time_v(log))
    curve = case_dir / "continuation_curve.csv"
    if curve.exists():
        with curve.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        row["curve_rows"] = len(rows)
        if rows:
            row["last_step"] = int(rows[-1]["step"])
            row["last_step_linear_iterations"] = int(rows[-1]["linear_iterations"])
            row["last_step_newton_iterations"] = int(rows[-1]["newton_iterations"])
    return row


def _parse_py(case_dir: Path, engine: str, profile: str, ranks: int) -> dict[str, object]:
    log = _read_text(case_dir / "run.log")
    row: dict[str, object] = {"engine": engine, "profile": profile, "ranks": ranks, "status": "missing"}
    if log:
        row["status"] = "done" if '"lambda_last"' in log or "runtime" in log else "failed"
    info_path = case_dir / "out" / "data" / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        run_info = info.get("run_info", {}) if isinstance(info.get("run_info"), dict) else {}
        timings = info.get("timings", {}) if isinstance(info.get("timings"), dict) else {}
        linear = timings.get("linear", {}) if isinstance(timings.get("linear"), dict) else {}
        params = info.get("params", {}) if isinstance(info.get("params"), dict) else {}
        row["status"] = "done"
        row["accepted_steps"] = int(info.get("steps", run_info.get("step_count", info.get("accepted_steps", -1))))
        row["omega_last"] = float(info.get("omega_last", params.get("omega_last", float("nan"))))
        row["lambda_last"] = float(info.get("lambda_last", params.get("lambda_last", float("nan"))))
        row["wall_time"] = float(
            info.get("runtime", run_info.get("runtime_seconds", timings.get("continuation_total_wall_time", float("nan"))))
        )
        row["total_newton_its"] = int(
            info.get("newton_iterations_total", info.get("total_newton_iterations", -1))
        )
        init_lin = int(linear.get("init_linear_iterations", info.get("init_linear_iterations", -1)))
        attempt_lin = int(
            linear.get("attempt_linear_iterations_total", info.get("attempt_linear_iterations_total", -1))
        )
        row["init_linear_iterations"] = init_lin
        row["attempt_linear_iterations_total"] = attempt_lin
        if "linear_iterations_total" in info:
            row["total_linear_its"] = int(info["linear_iterations_total"])
        elif init_lin >= 0 and attempt_lin >= 0:
            row["total_linear_its"] = init_lin + attempt_lin
        else:
            row["total_linear_its"] = -1
        row["stop_reason"] = str(info.get("stop_reason", "omega_max" if row.get("omega_last") else ""))
        diag = info.get("linear_solver_diagnostics", {})
        if not isinstance(diag, dict):
            diag = {}
        diag = {**linear, **diag}
        if isinstance(diag, dict):
            for key in (
                "manualmg_apply_count",
                "manualmg_coarse_ksp_type",
                "manualmg_coarse_pc_type",
                "manualmg_fine_iterations",
                "manualmg_mid_iterations",
            ):
                if key in diag:
                    row[key] = diag[key]
    init_match = re.search(r"\[init\].*?newton=\[([^\]]+)\].*?\blin=(\d+)", log)
    step_matches = list(
        re.finditer(
            r"\[step\s+(\d+)\s+ok\].*?\blambda=([0-9.eE+-]+).*?\bomega=([0-9.eE+-]+).*?\bnewton=(\d+).*?\blin=(\d+)",
            log,
        )
    )
    if init_match:
        init_newton = sum(int(part.strip()) for part in init_match.group(1).split(",") if part.strip())
        init_linear = int(init_match.group(2))
        cont_newton = sum(int(m.group(4)) for m in step_matches)
        cont_linear = sum(int(m.group(5)) for m in step_matches)
        row.setdefault("init_linear_iterations", init_linear)
        row.setdefault("attempt_linear_iterations_total", cont_linear)
        if row.get("total_newton_its", -1) == -1:
            row["total_newton_its"] = init_newton + cont_newton
        if row.get("total_linear_its", -1) == -1:
            row["total_linear_its"] = init_linear + cont_linear
    if step_matches:
        last = step_matches[-1]
        row.setdefault("accepted_steps", 2 + len(step_matches))
        row.setdefault("lambda_last", float(last.group(2)))
        row.setdefault("omega_last", float(last.group(3)))
    done_match = re.search(r"\[done\].*?steps=(\d+).*?lambda=([0-9.eE+-]+).*?omega=([0-9.eE+-]+).*?reason=([A-Za-z0-9_+-]+)", log)
    if done_match:
        row["status"] = "done"
        row["accepted_steps"] = int(done_match.group(1))
        row["lambda_last"] = float(done_match.group(2))
        row["omega_last"] = float(done_match.group(3))
        row["stop_reason"] = done_match.group(4)
    row.update(_parse_time_v(log))
    return row


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} OUT_ROOT", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    rows = []
    for case_dir in sorted(root.glob("*_*_r*")):
        match = re.match(r"^(c|py)_(baseline|petsc4py)_r(\d+)$", case_dir.name)
        if not match:
            continue
        engine, profile, ranks_s = match.groups()
        ranks = int(ranks_s)
        parser = _parse_c if engine == "c" else _parse_py
        rows.append(parser(case_dir, engine, profile, ranks))
    fields = [
        "engine",
        "profile",
        "ranks",
        "status",
        "accepted_steps",
        "omega_last",
        "lambda_last",
        "total_newton_its",
        "total_linear_its",
        "init_linear_iterations",
        "attempt_linear_iterations_total",
        "continuation_wall_time",
        "wall_time",
        "time_v_wall_s",
        "max_rss_kb",
        "stop_reason",
        "manualmg_coarse_ksp_type",
        "manualmg_coarse_pc_type",
        "manualmg_fine_iterations",
        "manualmg_mid_iterations",
    ]
    with (root / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(",".join(fields))
    for row in rows:
        print(",".join(str(row.get(field, "")) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
