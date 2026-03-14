#!/usr/bin/env python
"""Run the PETSc SSR capture under MPI with RSS monitoring and guardrails."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcInfo:
    pid: int
    ppid: int
    rss_kb: int
    args: str
    rank: int | None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_rank(pid: int) -> int | None:
    env_path = Path("/proc") / str(pid) / "environ"
    try:
        data = env_path.read_bytes().split(b"\x00")
    except OSError:
        return None
    for entry in data:
        if entry.startswith(b"OMPI_COMM_WORLD_RANK="):
            try:
                return int(entry.split(b"=", 1)[1])
            except ValueError:
                return None
        if entry.startswith(b"PMI_RANK="):
            try:
                return int(entry.split(b"=", 1)[1])
            except ValueError:
                return None
    return None


def _snapshot_descendants(root_pid: int) -> list[ProcInfo]:
    try:
        raw = subprocess.check_output(
            ["ps", "-e", "-o", "pid=", "-o", "ppid=", "-o", "rss=", "-o", "args="],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []

    parent_map: dict[int, list[tuple[int, int, str]]] = {}
    for line in raw.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss = int(parts[2])
        except ValueError:
            continue
        args = parts[3]
        parent_map.setdefault(ppid, []).append((pid, rss, args))

    descendants: list[ProcInfo] = []
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for pid, rss, args in parent_map.get(current, []):
            rank = _read_rank(pid)
            info = ProcInfo(pid=pid, ppid=current, rss_kb=rss, args=args, rank=rank)
            descendants.append(info)
            stack.append(pid)
    return descendants


def _terminate_process_group(proc: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return


def _format_gb_from_kb(kb: int) -> float:
    return kb / (1024.0 * 1024.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PETSc capture under MPI with RSS monitoring.")
    parser.add_argument("--nproc", type=int, required=True)
    parser.add_argument("--capture_out_dir", type=Path, required=True)
    parser.add_argument("--monitor_out_dir", type=Path, required=True)
    parser.add_argument("--rss_limit_gb", type=float, default=6.0, help="Kill the run if any monitored rank exceeds this RSS.")
    parser.add_argument("--sample_sec", type=float, default=1.0)
    parser.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    parser.add_argument("--capture_script", type=Path, default=Path("src/slope_stability/cli/run_3D_hetero_SSR_capture.py"))
    parser.add_argument("--use_hwthread_cpus", action="store_true", help="Pass --use-hwthread-cpus to mpiexec.")
    parser.add_argument("--oversubscribe", action="store_true", help="Pass --map-by :OVERSUBSCRIBE to mpiexec.")
    args, capture_args = parser.parse_known_args()
    if capture_args and capture_args[0] == "--":
        capture_args = capture_args[1:]

    monitor_out_dir = _ensure_dir(args.monitor_out_dir)
    capture_out_dir = args.capture_out_dir
    capture_out_dir.parent.mkdir(parents=True, exist_ok=True)

    log_path = monitor_out_dir / "run.log"
    samples_path = monitor_out_dir / "memory_samples.jsonl"
    summary_path = monitor_out_dir / "memory_summary.json"

    env = os.environ.copy()
    src_path = str((Path.cwd() / "src").resolve())
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"

    cmd = [
        "mpiexec",
        *(["--use-hwthread-cpus"] if args.use_hwthread_cpus else []),
        *(["--map-by", ":OVERSUBSCRIBE"] if args.oversubscribe else []),
        "-n",
        str(args.nproc),
        str(args.python),
        str(args.capture_script),
        "--out_dir",
        str(capture_out_dir),
        *capture_args,
    ]

    killed_for_rss = False
    rss_limit_kb = int(args.rss_limit_gb * 1024.0 * 1024.0)
    max_total_rss_kb = 0
    max_rank_rss_kb: dict[str, int] = {}
    max_pid_rss_kb: dict[str, int] = {}
    sample_count = 0
    started = time.time()

    with log_path.open("w", encoding="utf-8") as log_fp, samples_path.open("w", encoding="utf-8") as sample_fp:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )

        while True:
            rc = proc.poll()
            descendants = _snapshot_descendants(proc.pid)
            total_rss = 0
            ranks_this_sample: dict[str, int] = {}
            pids_this_sample: list[dict[str, int | str | None]] = []
            python_desc = [p for p in descendants if "python" in p.args]
            for info in python_desc:
                total_rss += info.rss_kb
                pids_this_sample.append(
                    {
                        "pid": info.pid,
                        "ppid": info.ppid,
                        "rss_kb": info.rss_kb,
                        "rank": info.rank,
                        "args": info.args,
                    }
                )
                max_pid_rss_kb[str(info.pid)] = max(max_pid_rss_kb.get(str(info.pid), 0), info.rss_kb)
                if info.rank is not None:
                    key = str(info.rank)
                    ranks_this_sample[key] = max(ranks_this_sample.get(key, 0), info.rss_kb)
                    max_rank_rss_kb[key] = max(max_rank_rss_kb.get(key, 0), info.rss_kb)

            max_total_rss_kb = max(max_total_rss_kb, total_rss)
            sample_fp.write(
                json.dumps(
                    {
                        "t_sec": time.time() - started,
                        "total_rss_kb": total_rss,
                        "rank_rss_kb": ranks_this_sample,
                        "python_processes": pids_this_sample,
                    }
                )
                + "\n"
            )
            sample_fp.flush()
            sample_count += 1

            if any(rss > rss_limit_kb for rss in ranks_this_sample.values()):
                killed_for_rss = True
                _terminate_process_group(proc, signal.SIGTERM)
                time.sleep(2.0)
                if proc.poll() is None:
                    _terminate_process_group(proc, signal.SIGKILL)

            if rc is not None:
                break
            time.sleep(args.sample_sec)

        return_code = proc.wait()

    summary = {
        "command": cmd,
        "return_code": return_code,
        "killed_for_rss": killed_for_rss,
        "rss_limit_gb": args.rss_limit_gb,
        "sample_sec": args.sample_sec,
        "duration_sec": time.time() - started,
        "sample_count": sample_count,
        "max_total_rss_kb": max_total_rss_kb,
        "max_total_rss_gb": _format_gb_from_kb(max_total_rss_kb),
        "max_rank_rss_kb": max_rank_rss_kb,
        "max_rank_rss_gb": {key: _format_gb_from_kb(val) for key, val in max_rank_rss_kb.items()},
        "max_pid_rss_kb": max_pid_rss_kb,
        "capture_out_dir": str(capture_out_dir),
        "log_path": str(log_path),
        "samples_path": str(samples_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if return_code != 0 or killed_for_rss:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
