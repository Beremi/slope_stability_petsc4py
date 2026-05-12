#!/usr/bin/env python3
"""Prepare Karolina Qexp NUMA-coalesced PMG socket-scaling configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "artifacts/experiments/pmg_numa_coalesced_karolina_socket_scaling_p4_l1_omega7"

RANKS_PER_NUMA = 16
CASES = [
    {
        "name": "numa_1x16",
        "nodes": 1,
        "numa_domains_per_node": 1,
        "total_numa_domains": 1,
        "ntasks_per_node": 16,
    },
    {
        "name": "numa_2x16",
        "nodes": 1,
        "numa_domains_per_node": 2,
        "total_numa_domains": 2,
        "ntasks_per_node": 32,
    },
    {
        "name": "numa_4x16",
        "nodes": 1,
        "numa_domains_per_node": 4,
        "total_numa_domains": 4,
        "ntasks_per_node": 64,
    },
    {
        "name": "numa_8x16",
        "nodes": 1,
        "numa_domains_per_node": 8,
        "total_numa_domains": 8,
        "ntasks_per_node": 128,
    },
    {
        "name": "numa_16x16",
        "nodes": 2,
        "numa_domains_per_node": 8,
        "total_numa_domains": 16,
        "ntasks_per_node": 128,
    },
]


def _format_toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _replace_assignment(text: str, key: str, value: object) -> str:
    prefix = f"{key} ="
    replacement = f"{key} = {_format_toml_value(value)}"
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}{replacement}"
            return "\n".join(lines) + "\n"
    raise KeyError(f"Could not find TOML field {key!r}")


def _case_by_name(name: str) -> dict[str, object]:
    for case in CASES:
        if case["name"] == name:
            return dict(case)
    raise KeyError(f"Unknown case {name!r}")


def _case_with_derived_fields(case: dict[str, object]) -> dict[str, object]:
    row = dict(case)
    row["ranks_per_numa"] = RANKS_PER_NUMA
    row["ranks"] = int(row["nodes"]) * int(row["ntasks_per_node"])
    expected_per_node = int(row["numa_domains_per_node"]) * RANKS_PER_NUMA
    if int(row["ntasks_per_node"]) != expected_per_node:
        raise ValueError(
            f"{row['name']} has ntasks_per_node={row['ntasks_per_node']}, "
            f"expected {expected_per_node} from "
            f"{row['numa_domains_per_node']} NUMA domains * {RANKS_PER_NUMA} ranks/domain."
        )
    return row


def _render_case(case: dict[str, object], omega_max: float, step_max: int) -> str:
    row = _case_with_derived_fields(case)
    text = (BENCH_DIR / "gasm_numa_coalesced.toml").read_text(encoding="utf-8")
    text = _replace_assignment(text, "title", f"NUMA-coalesced PMG {row['name']} P4(L1)")
    text = _replace_assignment(text, "mpi_ranks", int(row["ranks"]))
    text = _replace_assignment(text, "name", f"experiment_pmg_numa_{row['name']}_3D_hetero_SSR_P4_L1")
    text = _replace_assignment(text, "omega_max", float(omega_max))
    text = _replace_assignment(text, "step_max", int(step_max))
    text = _replace_assignment(text, "numa_domains_per_node", int(row["numa_domains_per_node"]))
    return text


def _write_manifest(out_root: Path, cases: list[dict[str, object]], config_dir: Path) -> None:
    manifest = []
    for case in cases:
        row = _case_with_derived_fields(case)
        row["config"] = str((config_dir / f"{row['name']}.toml").resolve())
        manifest.append(row)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    keys = [
        "name",
        "nodes",
        "ntasks_per_node",
        "ranks",
        "numa_domains_per_node",
        "total_numa_domains",
        "ranks_per_numa",
        "config",
    ]
    lines = ["\t".join(keys)]
    for row in manifest:
        lines.append("\t".join(str(row[key]) for key in keys))
    (out_root / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--only", action="append", help="Generate only this case name; may be repeated.")
    parser.add_argument("--omega-max", type=float, default=7.0e6)
    parser.add_argument("--step-max", type=int, default=100)
    args = parser.parse_args()

    selected = [_case_by_name(name) for name in args.only] if args.only else [dict(case) for case in CASES]
    out_root = args.out_root.resolve()
    config_dir = out_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for case in selected:
        row = _case_with_derived_fields(case)
        (config_dir / f"{row['name']}.toml").write_text(
            _render_case(case, args.omega_max, args.step_max),
            encoding="utf-8",
        )

    _write_manifest(out_root, selected, config_dir)
    print(f"Wrote {len(selected)} NUMA socket-scaling config(s) to {config_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
