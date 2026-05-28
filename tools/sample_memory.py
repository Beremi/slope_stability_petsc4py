#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample RSS for matching process names.")
    parser.add_argument("--pattern", default="petsc_ssr")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["time", "pid", "rss_kib", "command"])
        writer.writeheader()
        while not args.stop_file.exists():
            now = time.time()
            try:
                out = subprocess.check_output(["pgrep", "-af", args.pattern], text=True)
            except subprocess.CalledProcessError:
                out = ""
            for line in out.splitlines():
                parts = line.split(maxsplit=1)
                if not parts:
                    continue
                pid = parts[0]
                command = parts[1] if len(parts) > 1 else ""
                try:
                    rss = subprocess.check_output(["ps", "-o", "rss=", "-p", pid], text=True).strip()
                except subprocess.CalledProcessError:
                    continue
                if rss:
                    writer.writerow({"time": f"{now:.6f}", "pid": pid, "rss_kib": rss, "command": command})
            fh.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
