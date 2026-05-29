"""Case inspection and dry-run command helpers."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

from petsc_ssr.config.resolver import (
    explain_case_payload,
    validate_case_payload,
)


ENGINE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CASES_ROOT = ENGINE_ROOT / "benchmarks" / "cases"


def validate_all_cases_payload(cases_root: Path = DEFAULT_CASES_ROOT) -> dict[str, Any]:
    root = Path(cases_root)
    files: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    for case_toml in sorted(root.glob("*/case.toml")):
        try:
            payload = validate_case_payload(case_toml)
            files.append(
                {
                    "case": payload["case"],
                    "path": str(case_toml),
                    "analysis": payload["analysis"],
                    "asset": payload["asset"],
                    "mesh_variant": payload["mesh_variant"],
                    "element": payload["element"],
                    "linear_profile": payload["linear_profile"],
                    "continuation_profile": payload["continuation_profile"],
                    "newton_profile": payload["newton_profile"],
                    "seepage_profile": payload["seepage_profile"],
                    "resolved_world_size": payload["resolved_world_size"],
                }
            )
        except Exception as exc:
            issues.append({"path": str(case_toml), "error": str(exc)})
    return {
        "root": str(root),
        "count": len(files) + len(issues),
        "valid": len(files),
        "errors": len(issues),
        "cases": files,
        "issues": issues,
    }


def case_override(path: Path, *, profile: str | None = None, output_preset: str | None = None) -> Path:
    if not profile and not output_preset:
        return path

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    labels: list[str] = []
    if profile:
        if "linear" in data:
            data.setdefault("linear", {})["profile"] = profile
        elif "linear_solver" in data:
            data.setdefault("linear", {"profile": profile})
        else:
            data["linear"] = {"profile": profile}
        labels.append(profile)
    if output_preset:
        data.setdefault("output", {})["preset"] = output_preset
        labels.append(f"output-{output_preset}")
    out = ENGINE_ROOT / ".local" / "tmp" / "case_overrides" / path.parent.name / ("__".join(labels) + ".toml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dumps_simple_toml(data), encoding="utf-8")
    return out


def dry_run_case(args: argparse.Namespace) -> int:
    from petsc_ssr.runners import run_case_from_config

    case_toml = case_override(args.case_toml, profile=args.profile, output_preset=args.output_preset)
    argv = [str(case_toml), "--dry-run"]
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    if args.output_preset is not None:
        argv.extend(["--output-preset", str(args.output_preset)])
    if args.write_coordinate_bc_table:
        argv.append("--write-coordinate-bc-table")
    return run_case_from_config.main(argv)


def _dumps_simple_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    for section, payload in data.items():
        if not isinstance(payload, dict):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in payload.items():
            if isinstance(value, dict):
                lines.append("")
                lines.append(f"[{section}.{key}]")
                for inner_key, inner_value in value.items():
                    lines.append(f"{inner_key} = {_toml_value(inner_value)}")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines).strip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return json.dumps(str(value))
