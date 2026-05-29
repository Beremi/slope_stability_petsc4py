from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from petsc_ssr.benchmarks.targets import validate_target_payload


def validate_targets_payload(target: Path) -> dict[str, Any]:
    root = Path(target)
    paths = _target_json_paths(root)
    files: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []
    validated = 0
    legacy_parse_only = 0

    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("target JSON must be an object.")
            if "case" in payload:
                validate_target_payload(payload, source=path)
                validated += 1
                files.append({"path": str(path), "kind": "first_class", "case": payload.get("case")})
            else:
                legacy_parse_only += 1
                files.append({"path": str(path), "kind": "legacy_parse_only", "case": None})
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issues.append({"path": str(path), "error": str(exc)})

    return {
        "root": str(root),
        "count": len(paths),
        "validated": validated,
        "legacy_parse_only": legacy_parse_only,
        "errors": len(issues),
        "files": files,
        "issues": issues,
    }


def _target_json_paths(root: Path) -> list[Path]:
    if root.is_file():
        if root.suffix != ".json":
            raise ValueError(f"Target path {root} is not a JSON file.")
        return [root]
    if root.is_dir():
        return sorted(root.rglob("*.json"))
    raise ValueError(f"Target path {root} does not exist.")
