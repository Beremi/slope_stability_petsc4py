#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT\s+(.*)$")


def parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in text.split():
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def collect(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in sorted(root.glob("nodes_*")):
        row: dict[str, str] = {"run_dir": str(run_dir), "nodes": run_dir.name.split("_", 1)[-1]}
        summary = run_dir / "output" / "data" / "summary.json"
        if summary.exists():
            payload = json.loads(summary.read_text(encoding="utf-8"))
            for key in (
                "wall_time",
                "continuation_wall_time",
                "total_newton_its",
                "total_linear_its",
                "omega_last",
                "lambda_last",
                "final_rel",
                "deflation_pc_apply_time",
                "deflation_orthogonalization_time",
            ):
                if key in payload:
                    row[key] = str(payload[key])
        log = run_dir / "run.log"
        if log.exists():
            for line in log.read_text(errors="replace").splitlines():
                match = RESULT_RE.match(line)
                if match:
                    row.update({f"result_{k}": v for k, v in parse_kv(match.group(1)).items()})
        rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} RUN_ROOT", file=sys.stderr)
        return 2
    root = Path(argv[1])
    rows = collect(root)
    if not rows:
        return 1
    keys = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(sys.stdout, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
