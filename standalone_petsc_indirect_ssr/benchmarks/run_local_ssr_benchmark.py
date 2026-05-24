#!/usr/bin/env python3
"""Run local C/petsc4py indirect SSR benchmarks with memory sampling."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOLVER_DIR = REPO / "standalone_petsc_indirect_ssr"


def _proc_children(pid: int) -> list[int]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        text = children_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not text:
        return []
    return [int(part) for part in text.split() if part.isdigit()]


def _proc_tree(pid: int) -> list[int]:
    out: list[int] = []
    stack = [int(pid)]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if not Path(f"/proc/{current}").exists():
            continue
        out.append(current)
        stack.extend(_proc_children(current))
    return out


def _status_value(pid: int, key: str) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(key + ":"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1])
    except OSError:
        return None
    return None


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").strip()
    except OSError:
        return ""
    return raw.decode("utf-8", "replace")


def _pids_matching(markers: list[str], *, first_token_contains: str | None = None) -> list[int]:
    if not markers:
        return []
    out: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        cmd = _cmdline(pid)
        if not cmd or not all(marker in cmd for marker in markers):
            continue
        if first_token_contains:
            first = cmd.split(maxsplit=1)[0]
            if first_token_contains not in first:
                continue
        out.append(pid)
    return out


def _sample_memory(
    proc: subprocess.Popen,
    csv_path: Path,
    interval_s: float,
    *,
    match_markers: list[str] | None = None,
    rank_executable_marker: str | None = None,
) -> dict[str, float | int]:
    max_by_pid: dict[int, int] = {}
    rank_max_by_pid: dict[int, int] = {}
    peak_total = 0
    sample_count = 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["t_s", "pid", "rss_kib", "hwm_kib", "cmdline"])
        writer.writeheader()
        t0 = time.monotonic()
        while proc.poll() is None:
            matched_pids = set(_pids_matching(match_markers or [], first_token_contains=rank_executable_marker))
            pids = sorted(set(_proc_tree(proc.pid)) | matched_pids)
            total = 0
            for pid in pids:
                rss = _status_value(pid, "VmRSS")
                if rss is None:
                    continue
                hwm = _status_value(pid, "VmHWM")
                pid_peak = max(int(rss), int(hwm or rss), int(max_by_pid.get(pid, 0)))
                max_by_pid[pid] = pid_peak
                if pid in matched_pids:
                    rank_max_by_pid[pid] = max(pid_peak, int(rank_max_by_pid.get(pid, 0)))
                total += int(rss)
                writer.writerow(
                    {
                        "t_s": f"{time.monotonic() - t0:.3f}",
                        "pid": pid,
                        "rss_kib": rss,
                        "hwm_kib": "" if hwm is None else hwm,
                        "cmdline": _cmdline(pid),
                    }
                )
            peak_total = max(peak_total, total)
            sample_count += 1
            fh.flush()
            time.sleep(float(interval_s))
    rank_like = rank_max_by_pid or max_by_pid
    if rank_like:
        values = list(rank_like.values())
        max_rank = max(values)
        avg_rank = sum(values) / len(values)
    else:
        max_rank = 0
        avg_rank = 0.0
    return {
        "sample_count": int(sample_count),
        "process_count_seen": int(len(rank_like)),
        "all_process_count_seen": int(len(max_by_pid)),
        "max_rss_kib_per_process": int(max_rank),
        "avg_hwm_kib_per_process": float(avg_rank),
        "total_rss_kib_peak_sampled": int(peak_total),
    }


def _parse_keyvals(line: str) -> dict[str, str]:
    return {key: value for key, value in re.findall(r"(\w+)=([^ ]+)", line)}


def _last_line_with_prefix(text: str, prefix: str) -> str:
    out = ""
    for line in text.splitlines():
        if line.startswith(prefix):
            out = line
    return out


def _parse_c_log(log_text: str) -> dict[str, object]:
    row: dict[str, object] = {}
    result = _parse_keyvals(_last_line_with_prefix(log_text, "RESULT "))
    for key in (
        "wall_time",
        "continuation_wall_time",
        "lambda_last",
        "omega_last",
        "final_rel",
        "final_rel_correction",
    ):
        if key in result:
            row[key] = float(result[key])
    for key in ("accepted_steps", "total_newton_its", "total_linear_its", "total_line_search_its", "global_dofs"):
        if key in result:
            row[key] = int(float(result[key]))
    defl = _parse_keyvals(_last_line_with_prefix(log_text, "DEFLATION_TIMING "))
    for key, out_key in (
        ("orthogonalization", "deflation_orthogonalization_s"),
        ("coarse_initial", "deflation_coarse_initial_s"),
        ("pc_apply", "deflation_pc_apply_s"),
        ("projector", "deflation_projector_s"),
    ):
        if key in defl:
            row[out_key] = float(defl[key])
    pmg = _parse_keyvals(_last_line_with_prefix(log_text, "PMG_SHELL_APPLY_SUMMARY "))
    for key, out_key in (
        ("fine_smooth", "pmg_fine_smooth_s"),
        ("p2_smooth", "pmg_p2_smooth_s"),
        ("coarse_solve", "pmg_coarse_solve_s"),
        ("residual", "pmg_residual_s"),
        ("restrict", "pmg_restrict_s"),
        ("prolong", "pmg_prolong_s"),
    ):
        if key in pmg:
            row[out_key] = float(pmg[key])
    return row


def _parse_py_log(case_dir: Path, log_text: str) -> dict[str, object]:
    row: dict[str, object] = {}
    info_path = case_dir / "out" / "data" / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        run_info = info.get("run_info", {})
        timings = info.get("timings", {})
        linear = timings.get("linear", {})
        c_hotpath = info.get("c_hotpath_summary", {})
        row["wall_time"] = float(run_info.get("runtime_seconds", info.get("runtime", 0.0)))
        row["continuation_wall_time"] = float(timings.get("continuation_total_wall_time", row["wall_time"]))
        row["accepted_steps"] = int(run_info.get("step_count", info.get("steps", 0)))
        row["lambda_last"] = float(info.get("lambda_last", 0.0))
        row["omega_last"] = float(info.get("omega_last", 0.0))
        init = int(linear.get("init_linear_iterations", 0))
        attempt = int(linear.get("attempt_linear_iterations_total", 0))
        row["total_linear_its"] = init + attempt
        row["total_newton_its"] = int(info.get("newton_iterations_total", 0))
        if c_hotpath:
            row["total_line_search_its"] = int(c_hotpath.get("total_line_search_its", 0))
            row["final_rel"] = float(c_hotpath.get("final_rel", 0.0))
            row["deflation_orthogonalization_s"] = float(c_hotpath.get("deflation_orthogonalization_time", 0.0))
            row["deflation_pc_apply_s"] = float(c_hotpath.get("deflation_pc_apply_time", 0.0))
            row["deflation_projector_s"] = float(c_hotpath.get("deflation_projector_time", 0.0))
        row["manualmg_apply_count"] = int(linear.get("manualmg_apply_count", 0))
        row["manualmg_active_layout_status"] = str(linear.get("manualmg_active_layout_status", ""))
    done = re.search(r"\[done\].*?steps=(\d+).*?lambda=([0-9.eE+-]+).*?omega=([0-9.eE+-]+)", log_text)
    if done:
        row.setdefault("accepted_steps", int(done.group(1)))
        row.setdefault("lambda_last", float(done.group(2)))
        row.setdefault("omega_last", float(done.group(3)))
    return row


def _write_py_config(path: Path, *, ranks: int, omega_max: float, ksp_max_it: int, refine_levels: int = 0) -> None:
    petsc_opt = ""
    if int(refine_levels) != 0:
        petsc_opt = f'petsc_opt = ["-refine_levels={int(refine_levels)}"]\n'
    path.write_text(
        f"""[benchmark]
title = "petsc4py DMPlex C-hotpath unrefined L1 SSR"
comparison_kind = "continuation"
mpi_ranks = {int(ranks)}

[problem]
name = "petsc4py_dmplex_c_hotpath_l1"
asset = "3d_hetero_slope"
mesh_variant = "adaptive_family_a_l1.msh"
analysis = "ssr"
elem_type = "P4"
davis_type = "B"

[execution]
mechanics_backend = "dmplex_c_hotpath"
node_ordering = "block_metis"
mpi_distribute_by_nodes = true
constitutive_mode = "overlap"
tangent_kernel = "rows"
store_step_u = false

[continuation]
method = "indirect"
lambda_init = 1.0
d_lambda_init = 0.1
d_lambda_min = 1e-3
d_lambda_diff_scaled_min = 1e-3
omega_max = {float(omega_max):.17g}
init_newton_stopping_criterion = "relative_correction"
init_newton_stopping_tol = 1e-3
step_max = 100

[newton]
it_max = 200
it_damp_max = 10
tol = 1e-4
r_min = 1e-4
stopping_criterion = "absolute_delta_lambda"
stopping_tol = 1e-4

[linear_solver]
solver_type = "KSPFGMRES"
tolerance = 1e-1
max_iterations = {int(ksp_max_it)}
deflation_basis_tolerance = 1e-3
threads = 1
compiled_outer = false
recycle_preconditioner = true
pc_backend = "pmg_shell"
pmg_profile = "c_split_smoother"
pmg_shell_p2_active_ranks = 64
pmg_shell_p1_active_ranks = 32
pmg_shell_subcomm_type = "interlaced"
pmg_shell_fine_ksp_max_it = 5
pmg_shell_p2_ksp_max_it = 10
pmg_shell_p1_pc_type = "redundant"
pmg_shell_p1_redundant_number = 1
pmg_shell_p1_redundant_ksp_type = "fgmres"
pmg_shell_p1_redundant_ksp_rtol = 1e-3
pmg_shell_p1_redundant_ksp_max_it = 5
pmg_shell_p1_redundant_pc_type = "gamg"
{petsc_opt}

[export]
write_custom_debug_bundle = false
write_history_json = true
write_solution_vtu = false
""",
        encoding="utf-8",
    )


def _run_case(args, *, engine: str, ranks: int, out_root: Path) -> dict[str, object]:
    case_dir = out_root / f"{engine}_r{int(ranks)}"
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "run.log"
    mem_path = case_dir / "memory_samples.csv"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env.setdefault("PETSC_DIR", str(REPO / ".build/src/petsc-3.24.5"))
    env.setdefault("PETSC_ARCH", "linux-c-opt")
    if engine == "c":
        cmd = [
            "mpiexec",
            "-n",
            str(ranks),
        ]
        if args.oversubscribe:
            cmd.extend(["--map-by", ":OVERSUBSCRIBE"])
        cmd.extend(
            [
            str(SOLVER_DIR / "p4_indirect_ssr"),
            "-options_file",
            str(SOLVER_DIR / "options/pmg_shell_split_smoother.opts"),
            "-mesh",
            str(SOLVER_DIR / "data/adaptive_family_a_l1.msh"),
            "-refine_levels",
            str(args.refine_levels),
            "-omega_max",
            str(args.omega_max),
            "-linear_rtol",
            "1e-1",
            "-ksp_max_it",
            str(args.ksp_max_it),
            "-petscpartitioner_type",
            "parmetis",
            "-curve_csv",
            str(case_dir / "continuation_curve.csv"),
            "-pmg_shell_p1_pc_type",
            "redundant",
            "-pmg_shell_p1_pc_redundant_number",
            "1",
            "-pmg_shell_p1_redundant_ksp_type",
            "fgmres",
            "-pmg_shell_p1_redundant_ksp_rtol",
            "1e-3",
            "-pmg_shell_p1_redundant_ksp_max_it",
            "5",
            "-pmg_shell_p1_redundant_pc_type",
            "gamg",
            ]
        )
        match_markers = [str(SOLVER_DIR / "p4_indirect_ssr"), str(case_dir)]
        rank_executable_marker = "p4_indirect_ssr"
    elif engine == "py":
        cfg = case_dir / "case.toml"
        _write_py_config(cfg, ranks=ranks, omega_max=args.omega_max, ksp_max_it=args.ksp_max_it, refine_levels=args.refine_levels)
        python = args.python or str(REPO / ".venv/bin/python")
        cmd = [
            "mpiexec",
            "-n",
            str(ranks),
        ]
        if args.oversubscribe:
            cmd.extend(["--map-by", ":OVERSUBSCRIBE"])
        cmd.extend(
            [
            python,
            "-m",
            "slope_stability.cli.run_case_from_config",
            str(cfg),
            "--out_dir",
            str(case_dir / "out"),
            ]
        )
        match_markers = ["slope_stability.cli.run_case_from_config", str(cfg)]
        rank_executable_marker = Path(python).name
    else:
        raise ValueError(f"unknown engine {engine!r}")

    if args.dry_run:
        return {"engine": engine, "ranks": ranks, "status": "dry_run", "command": " ".join(shlex.quote(part) for part in cmd)}

    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            memory = _sample_memory(
                proc,
                mem_path,
                args.memory_interval,
                match_markers=match_markers,
                rank_executable_marker=rank_executable_marker,
            )
            ret = proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            raise
    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_c_log(text) if engine == "c" else _parse_py_log(case_dir, text)
    row: dict[str, object] = {
        "engine": engine,
        "ranks": int(ranks),
        "status": "ok" if ret == 0 else f"exit_{ret}",
        "log": str(log_path),
        "memory_samples": str(mem_path),
        **parsed,
        **memory,
    }
    linear = int(row.get("total_linear_its") or 0)
    if linear > 0:
        if "wall_time" in row:
            row["wall_per_linear_s"] = float(row["wall_time"]) / linear
        if "continuation_wall_time" in row:
            row["continuation_per_linear_s"] = float(row["continuation_wall_time"]) / linear
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", default=["c", "py"], choices=["c", "py"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--out-root", type=Path, default=REPO / ".local/tmp/ssr_local_benchmark")
    parser.add_argument("--omega-max", type=float, default=7.0e6)
    parser.add_argument("--ksp-max-it", type=int, default=200)
    parser.add_argument("--refine-levels", type=int, default=0)
    parser.add_argument("--memory-interval", type=float, default=0.5)
    parser.add_argument("--python", default=None)
    parser.add_argument("--oversubscribe", action="store_true", help="Pass Open MPI oversubscription mapping to mpiexec.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.dry_run and "c" in args.engines:
        subprocess.run(["make", "-C", str(SOLVER_DIR)], cwd=str(REPO), check=True)
    for ranks in args.ranks:
        for engine in args.engines:
            row = _run_case(args, engine=engine, ranks=int(ranks), out_root=args.out_root)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    fields = sorted({key for row in rows for key in row})
    with (args.out_root / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (args.out_root / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OUT_ROOT={args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
