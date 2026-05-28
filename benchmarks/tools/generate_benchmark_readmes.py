#!/usr/bin/env python3
"""Generate concise asset-first benchmark READMEs from case configs."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = ROOT / "benchmarks" / "cases"


def _load_case(case_toml: Path) -> dict[str, object]:
    raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    case = dict(raw.get("case", {}))
    mesh = dict(raw.get("mesh", {}))
    physics = dict(raw.get("physics", {}))
    mechanics = dict(physics.get("mechanics", {}))
    benchmark = dict(raw.get("benchmark", {}))
    problem = dict(raw.get("problem", {}))
    if case or mesh:
        linear = dict(raw.get("linear", {}))
        model = str(mechanics.get("model", ""))
        analysis = "ll" if "limit" in model else ("ssr" if mechanics else "seepage")
        title = str(case.get("title", case_toml.parent.name))
        return {
            "case_dir_name": case_toml.parent.name,
            "case_toml": case_toml,
            "title": title,
            "notes": "",
            "suite": False,
            "analysis": analysis,
            "asset": str(mesh.get("asset", "")).strip(),
            "mesh_variant": str(mesh.get("variant", "")).strip(),
            "profile": str(linear.get("profile", "baseline-pmg-deflated") or "baseline-pmg-deflated").strip(),
            "elem_type": str(mesh.get("element", "")).strip(),
        }
    return {
        "case_dir_name": case_toml.parent.name,
        "case_toml": case_toml,
        "title": str(benchmark.get("title", case_toml.parent.name)),
        "notes": str(benchmark.get("notes", "")).strip(),
        "suite": bool(benchmark.get("suite", False)),
        "analysis": str(problem.get("analysis", "")).strip(),
        "asset": str(problem.get("asset", "")).strip(),
        "mesh_variant": str(problem.get("mesh_variant", "")).strip(),
        "profile": str(problem.get("profile", "default") or "default").strip(),
        "elem_type": str(problem.get("elem_type", "")).strip(),
    }


def _analysis_label(value: object) -> str:
    text = str(value).strip().lower()
    if text == "ssr":
        return "shear strength reduction (SSR)"
    if text == "ll":
        return "limit-load (LL)"
    if text == "seepage":
        return "seepage"
    return text or "configured"


def _dimension_label(asset: object) -> str:
    text = str(asset).strip().lower()
    if text.startswith("2d_"):
        return "2D"
    if text.startswith("3d_"):
        return "3D"
    return "configured"


def _render_readme(case: dict[str, object]) -> str:
    title = str(case["title"])
    asset = str(case["asset"])
    mesh_variant = str(case["mesh_variant"])
    profile = str(case["profile"] or "default")
    analysis = str(case["analysis"]).lower()
    elem_type = str(case["elem_type"])
    dimension = _dimension_label(asset)
    suite_text = " It is part of the MATLAB-parity benchmark suite." if case["suite"] else ""
    lines = [
        f"# {title}",
        "",
        (
            f"This {dimension} case runs a config-driven {_analysis_label(analysis)} analysis using "
            f"asset `{asset}` and mesh variant `{mesh_variant}`.{suite_text}"
        ),
        "",
        "## Run",
        "",
        "```bash",
        "./run.sh",
        "```",
        "",
        "## Case Inputs",
        "",
        f"- Case config: [`case.toml`](case.toml)",
        f"- Asset: `{asset}`",
        f"- Mesh variant: `{mesh_variant}`",
        f"- Solver profile: `{profile}`",
        f"- Analysis: `{analysis}`",
        f"- Element order: `{elem_type}`",
        "",
        (
            f"Geometry, materials, hydraulic behavior, and boundary conditions are defined in "
            f"[`../../../meshes/{asset}/definition.py`](../../../meshes/{asset}/definition.py)."
        ),
    ]
    notes = str(case.get("notes", "")).strip()
    if notes:
        lines.extend(["", "## Notes", "", notes])
    lines.append("")
    return "\n".join(lines)


def generate_readme_for_case(case_toml: Path) -> None:
    case = _load_case(case_toml)
    (case_toml.parent / "README.md").write_text(_render_readme(case), encoding="utf-8")


def generate_readmes(benchmarks_dir: Path) -> None:
    for case_toml in sorted(benchmarks_dir.glob("*/case.toml")):
        generate_readme_for_case(case_toml)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate concise benchmark READMEs.")
    parser.add_argument("--benchmarks-dir", type=Path, default=BENCHMARKS_DIR)
    args = parser.parse_args()
    generate_readmes(args.benchmarks_dir.resolve())


if __name__ == "__main__":
    main()
