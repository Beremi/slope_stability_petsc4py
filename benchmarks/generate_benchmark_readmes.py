#!/usr/bin/env python3
"""Generate concise descriptive benchmark READMEs from case metadata and MATLAB scripts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from textwrap import fill
import tomllib


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
MATLAB_ROOT = ROOT / "slope_stability_matlab"

_RUN_INFO_SCRIPT_RE = re.compile(r"""run_info\.script\s*=\s*["']([^"']+\.m)["']""")


def _load_case(case_toml: Path) -> dict[str, object]:
    raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    benchmark = dict(raw.get("benchmark", {}))
    problem = dict(raw.get("problem", {}))
    return {
        "case_dir_name": case_toml.parent.name,
        "case_toml": case_toml,
        "title": str(benchmark.get("title", case_toml.parent.name)),
        "matlab_script": str(benchmark.get("matlab_script", "")),
        "notes": str(benchmark.get("notes", "")).strip(),
        "analysis": str(problem.get("analysis", "")).strip(),
        "dimension": problem.get("dimension"),
        "variant": str(problem.get("variant", "")).strip(),
        "elem_type": str(problem.get("elem_type", "")).strip(),
        "seepage": bool(problem.get("seepage", False)),
    }


def _resolve_matlab_script(script_name: str) -> Path | None:
    if not script_name:
        return None
    candidates = []
    raw = Path(script_name)
    if raw.suffix == ".m":
        candidates.extend([MATLAB_ROOT / raw.name, MATLAB_ROOT / "scripts" / raw.name])
    else:
        candidates.extend([MATLAB_ROOT / f"{script_name}.m", MATLAB_ROOT / "scripts" / f"{script_name}.m"])
    for path in candidates:
        if path.exists():
            return path
    return None


def _description_source(script_path: Path | None) -> Path | None:
    if script_path is None:
        return None
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    match = _RUN_INFO_SCRIPT_RE.search(text)
    if match:
        candidate = MATLAB_ROOT / match.group(1)
        if candidate.exists():
            return candidate
    return script_path


def _extract_comment_paragraph(script_path: Path | None) -> str:
    if script_path is None:
        return ""
    lines = script_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    comments: list[str] = []
    started = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("function "):
            continue
        if line.startswith("%%"):
            cleaned = line[2:].strip()
            if cleaned and not set(cleaned) <= {"=", "-", "*"}:
                if started and comments:
                    break
                started = True
                comments.append(cleaned)
            continue
        if line.startswith("%"):
            cleaned = line[1:].strip()
            if not cleaned:
                continue
            if set(cleaned) <= {"=", "-", "*"}:
                continue
            started = True
            comments.append(cleaned)
            continue
        if started:
            break
    if not comments:
        return ""
    text = " ".join(comments)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _description(case: dict[str, object], script_path: Path | None) -> str:
    title = str(case["title"]).strip()
    text = _extract_comment_paragraph(script_path)
    for needle in ("This program solves", "This benchmark defines"):
        idx = text.find(needle)
        if idx > 0:
            text = text[idx:]
            break
    if not text:
        dim = case.get("dimension")
        analysis = str(case.get("analysis", "")).upper()
        variant = str(case.get("variant", "")).lower()
        seepage = bool(case.get("seepage"))
        variant_label = {"homo": "homogeneous", "hetero": "heterogeneous"}.get(variant, variant or "configured")
        kind = "slope-stability case"
        if str(case.get("analysis", "")).lower() == "seepage":
            kind = "seepage case"
        parts = [f"This benchmark defines a {dim}D {variant_label} {kind}".strip()]
        if seepage:
            parts[-1] += " with seepage"
        text = parts[-1] + "."
    text = re.sub(rf"^{re.escape(title)}\s*", "", text, flags=re.IGNORECASE).strip()
    if "concave" in title.lower():
        text += " The current PETSc benchmark uses the concave seepage geometry carried by this repository configuration."
    if case["case_dir_name"].endswith("_default"):
        text += " This folder keeps the current default PETSc configuration for that benchmark family."
    return fill(text, width=92)


def _render_readme(case: dict[str, object], script_path: Path | None) -> str:
    title = str(case["title"])
    matlab_script = str(case["matlab_script"])
    lines = [f"# {title}", "", _description(case, script_path), "", "## Run", "", "```bash", "./run.sh", "```", ""]
    lines.extend(
        [
            "## Source",
            "",
            f"- MATLAB driver: `{matlab_script}`",
            "- PETSc config: [`case.toml`](case.toml)",
        ]
    )
    notes = str(case.get("notes", "")).strip()
    if notes:
        lines.extend(["", "## Notes", "", fill(notes, width=92)])
    lines.append("")
    return "\n".join(lines)


def generate_readmes(benchmarks_dir: Path) -> None:
    for case_toml in sorted(benchmarks_dir.glob("*/case.toml")):
        case = _load_case(case_toml)
        script_path = _description_source(_resolve_matlab_script(str(case["matlab_script"])))
        readme = _render_readme(case, script_path)
        (case_toml.parent / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate concise benchmark READMEs.")
    parser.add_argument("--benchmarks-dir", type=Path, default=BENCHMARKS_DIR)
    args = parser.parse_args()
    generate_readmes(args.benchmarks_dir.resolve())


if __name__ == "__main__":
    main()
