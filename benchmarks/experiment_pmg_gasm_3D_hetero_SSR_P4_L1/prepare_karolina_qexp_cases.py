#!/usr/bin/env python3
"""Prepare Karolina one-node Qexp configs for the PMG/GASM P4(L1) grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "artifacts/experiments/pmg_gasm_karolina_qexp_one_node_p4_l1_omega7"

CASES = [
    {"name": "baseline_16", "kind": "baseline", "ranks": 16, "sockets": None, "ranks_per_socket": None},
    {"name": "baseline_32", "kind": "baseline", "ranks": 32, "sockets": None, "ranks_per_socket": None},
    {"name": "baseline_64", "kind": "baseline", "ranks": 64, "sockets": None, "ranks_per_socket": None},
    {"name": "baseline_128", "kind": "baseline", "ranks": 128, "sockets": None, "ranks_per_socket": None},
    {"name": "gasm_1x16", "kind": "gasm", "ranks": 16, "sockets": 1, "ranks_per_socket": 16},
    {"name": "gasm_2x16", "kind": "gasm", "ranks": 32, "sockets": 2, "ranks_per_socket": 16},
    {"name": "gasm_4x16", "kind": "gasm", "ranks": 64, "sockets": 4, "ranks_per_socket": 16},
    {"name": "gasm_8x16", "kind": "gasm", "ranks": 128, "sockets": 8, "ranks_per_socket": 16},
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
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}{replacement}"
            replaced = True
            break
    if not replaced:
        raise KeyError(f"Could not find TOML field {key!r}")
    return "\n".join(lines) + "\n"


def _case_by_name(name: str) -> dict[str, object]:
    for case in CASES:
        if case["name"] == name:
            return case
    raise KeyError(f"Unknown case {name!r}")


def _render_case(case: dict[str, object], omega_max: float, step_max: int) -> str:
    template = BENCH_DIR / ("gasm.toml" if case["kind"] == "gasm" else "baseline.toml")
    text = template.read_text()
    text = _replace_assignment(text, "mpi_ranks", case["ranks"])
    text = _replace_assignment(text, "omega_max", omega_max)
    text = _replace_assignment(text, "step_max", step_max)
    if case["kind"] == "gasm":
        text = _replace_assignment(text, "pmg_smoother_gasm_total_subdomains", case["sockets"])
    return text


def _write_manifest(out_root: Path, cases: list[dict[str, object]], config_dir: Path) -> None:
    manifest = []
    for index, case in enumerate(cases):
        row = dict(case)
        row["array_index"] = index
        row["config"] = str((config_dir / f"{case['name']}.toml").resolve())
        manifest.append(row)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tsv_lines = ["array_index\tname\tkind\tranks\tsockets\tranks_per_socket\tconfig"]
    for row in manifest:
        tsv_lines.append(
            "\t".join(
                "" if row.get(key) is None else str(row.get(key))
                for key in ("array_index", "name", "kind", "ranks", "sockets", "ranks_per_socket", "config")
            )
        )
    (out_root / "manifest.tsv").write_text("\n".join(tsv_lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--only", action="append", help="Generate only this case name; may be repeated.")
    parser.add_argument("--omega-max", type=float, default=7.0e6)
    parser.add_argument("--step-max", type=int, default=100)
    args = parser.parse_args()

    selected = [_case_by_name(name) for name in args.only] if args.only else CASES
    out_root = args.out_root.resolve()
    config_dir = out_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for case in selected:
        (config_dir / f"{case['name']}.toml").write_text(_render_case(case, args.omega_max, args.step_max))

    _write_manifest(out_root, selected, config_dir)
    print(f"Wrote {len(selected)} config(s) to {config_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
