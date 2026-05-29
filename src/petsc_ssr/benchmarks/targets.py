"""First-class benchmark target schema helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TARGET_TOP_LEVEL_FIELDS = {"case", "profile", "suite", "status", "notes", "metrics", "rank_metrics", "groups"}
TARGET_RANK_GROUP_FIELDS = {"metrics", "source", "source_profile", "notes"}
TARGET_SWEEP_GROUP_FIELDS = {
    "profile",
    "ranks",
    "refine_levels",
    "linear_rtol",
    "continuation_step_max",
    "metrics",
    "source",
    "notes",
}
TARGET_METRIC_SPEC_FIELDS = {"max", "expected", "value", "abs_tol", "rel_tol"}


def load_target(path: str | Path, *, require_case: bool = True) -> dict[str, Any]:
    target_path = Path(path)
    payload = json.loads(target_path.read_text(encoding="utf-8"))
    validate_target_payload(payload, source=target_path, require_case=require_case)
    return payload


def validate_target_payload(
    payload: dict[str, Any],
    *,
    source: str | Path | None = None,
    require_case: bool = True,
) -> None:
    section = _source_label(source)
    if not isinstance(payload, dict):
        raise ValueError(f"{section} target must be a JSON object.")
    _reject_unknown_fields(section, payload, TARGET_TOP_LEVEL_FIELDS)
    if "case" in payload and not isinstance(payload["case"], str):
        raise ValueError(f"{section}.case must be a string.")
    case = str(payload.get("case", "") or "").strip()
    if require_case and not case:
        raise ValueError(f"{section}.case must be a non-empty string.")
    _validate_optional_string(section, payload, "profile")
    _validate_optional_string(section, payload, "suite")
    _validate_optional_string(section, payload, "status")
    _validate_optional_string(section, payload, "notes")
    _validate_metrics_table(f"{section}.metrics", payload.get("metrics", {}))
    _validate_rank_metrics(section, payload.get("rank_metrics", {}))
    _validate_groups(section, payload.get("groups", []))


def target_metric_names(payload: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    metrics = payload.get("metrics", {})
    if isinstance(metrics, dict):
        names.update(str(name) for name in metrics)
    rank_metrics = payload.get("rank_metrics", {})
    if isinstance(rank_metrics, dict):
        for value in rank_metrics.values():
            if isinstance(value, dict):
                names.update(str(name) for name in target_metric_payload(value))
    groups = payload.get("groups", [])
    if isinstance(groups, list):
        for group in groups:
            if isinstance(group, dict):
                names.update(str(name) for name in target_metric_payload(group))
    return sorted(names)


def target_metric_payload(data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics", data)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _validate_rank_metrics(section: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{section}.rank_metrics must be an object keyed by MPI rank.")
    for rank, payload in value.items():
        rank_section = f"{section}.rank_metrics[{rank!r}]"
        try:
            parsed_rank = int(str(rank))
        except ValueError as exc:
            raise ValueError(f"{rank_section} key must be a positive integer rank.") from exc
        if parsed_rank <= 0:
            raise ValueError(f"{rank_section} key must be a positive integer rank.")
        if not isinstance(payload, dict):
            raise ValueError(f"{rank_section} must be an object.")
        _reject_unknown_fields(rank_section, payload, TARGET_RANK_GROUP_FIELDS)
        _validate_optional_string(rank_section, payload, "source")
        _validate_optional_string(rank_section, payload, "source_profile")
        _validate_optional_string(rank_section, payload, "notes")
        _validate_metrics_table(f"{rank_section}.metrics", payload.get("metrics", {}))


def _validate_groups(section: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        raise ValueError(f"{section}.groups must be an array of metric-group objects.")
    for index, group in enumerate(value):
        group_section = f"{section}.groups[{index}]"
        if not isinstance(group, dict):
            raise ValueError(f"{group_section} must be an object.")
        _reject_unknown_fields(group_section, group, TARGET_SWEEP_GROUP_FIELDS)
        _validate_optional_string(group_section, group, "profile")
        _validate_optional_string(group_section, group, "source")
        _validate_optional_string(group_section, group, "notes")
        for numeric_key in ("ranks", "refine_levels", "linear_rtol", "continuation_step_max"):
            if numeric_key in group and not _is_number(group[numeric_key]):
                raise ValueError(f"{group_section}.{numeric_key} must be numeric.")
        _validate_metrics_table(f"{group_section}.metrics", group.get("metrics", {}))


def _validate_metrics_table(section: str, value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be an object.")
    for name, spec in value.items():
        metric = str(name).strip()
        if not metric:
            raise ValueError(f"{section} metric names must be non-empty strings.")
        _validate_metric_spec(f"{section}.{metric}", spec)


def _validate_metric_spec(section: str, spec: object) -> None:
    if _is_number(spec):
        return
    if not isinstance(spec, dict):
        raise ValueError(f"{section} must be numeric or an object with max/expected/value.")
    _reject_unknown_fields(section, spec, TARGET_METRIC_SPEC_FIELDS)
    has_max = "max" in spec
    has_expected = "expected" in spec or "value" in spec
    if has_max and has_expected:
        raise ValueError(f"{section} must use either max or expected/value, not both.")
    if not has_max and not has_expected:
        raise ValueError(f"{section} must define max, expected, or value.")
    if "expected" in spec and "value" in spec:
        raise ValueError(f"{section} must use either expected or value, not both.")
    for key in ("max", "expected", "value", "abs_tol", "rel_tol"):
        if key in spec and not _is_number(spec[key]):
            raise ValueError(f"{section}.{key} must be numeric.")
    for key in ("abs_tol", "rel_tol"):
        if key in spec and float(spec[key]) < 0.0:
            raise ValueError(f"{section}.{key} must be non-negative.")


def _validate_optional_string(section: str, payload: dict[str, Any], key: str) -> None:
    if key in payload and payload[key] is not None and not isinstance(payload[key], str):
        raise ValueError(f"{section}.{key} must be a string.")


def _reject_unknown_fields(section: str, payload: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{section} fields {unknown} are not supported in benchmark target JSON.")


def _source_label(source: str | Path | None) -> str:
    return "target" if source is None else f"target {Path(source)}"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
