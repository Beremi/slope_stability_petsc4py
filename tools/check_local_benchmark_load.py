#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Refuse local benchmark runs when the workstation is already busy.")
    parser.add_argument("--max-load", type=float, default=float(os.environ.get("MAX_PRE_BENCH_LOAD", "4.0")))
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    with open("/proc/loadavg", encoding="utf-8") as fh:
        load1 = float(fh.read().split()[0])

    if load1 <= args.max_load:
        return 0

    print(
        f"LOCAL_BENCHMARK_LOAD_REFUSED load1={load1:.2f} max_load={args.max_load:.2f} "
        "set SKIP_LOAD_CHECK=1 to run anyway",
        file=sys.stderr,
    )
    try:
        ps = subprocess.check_output(
            ["ps", "-eo", "user:16,pid,ppid,stat,pcpu,pmem,rss,comm,args", "--sort=-pcpu"],
            text=True,
        )
        lines = ps.splitlines()
        header, rows = lines[0], lines[1:]
        self_pid = os.getpid()
        filtered = [
            row
            for row in rows
            if f"{self_pid}" not in row.split()[1:3]
            and "check_local_benchmark_load.py" not in row
            and "ps -eo" not in row
        ]
        print("\n".join([header, *filtered[: args.top]]), file=sys.stderr)
    except Exception as exc:
        print(f"Failed to collect process list: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
