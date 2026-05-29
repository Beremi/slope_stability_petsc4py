"""Benchmark suite parsing, expansion, manifests, and compatibility wrappers."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from petsc_ssr.benchmarks.compare import compare_targets as _compare_targets
from petsc_ssr.benchmarks.logs import materialize_suite_log_artifacts
from petsc_ssr.benchmarks.report import write_report as _write_report
from petsc_ssr.config import (
    load_run_case_config,
    load_solver_profile,
    native_linear_algorithm_selector,
    pc_variant_from_backend,
)
from petsc_ssr.config.validators import normalize_output_preset as _normalize_config_output_preset
from petsc_ssr.runtime.results import run_artifact_manifest


ENGINE_ROOT = Path(__file__).resolve().parents[3]
CASE_ROOT = ENGINE_ROOT / "benchmarks" / "cases"


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    id: str
    title: str
    cases: tuple[str, ...]
    profiles: tuple[str, ...]
    ranks: tuple[int, ...]
    refine_levels: tuple[int | None, ...] = (None,)
    linear_rtols: tuple[float | None, ...] = (None,)
    continuation_step_max: tuple[int | None, ...] = (None,)
    repeats: int = 1
    timeout: str | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    collect: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None


@dataclass(frozen=True, slots=True)
class SuiteRun:
    suite_id: str
    run_id: str
    case: str
    profile: str
    ranks: int
    refine_levels: int | None
    linear_rtol: float | None
    continuation_step_max: int | None
    output_preset: str | None
    repeat: int
    resource: str | None
    launcher: tuple[str, ...]
    case_toml: Path
    output_dir: Path
    command: tuple[str, ...]
    sweep: dict[str, Any]
    resolved_profile: dict[str, Any]
    artifacts: dict[str, str]


def load_suite(path: str | Path) -> SuiteSpec:
    suite_path = Path(path).resolve()
    payload = tomllib.loads(suite_path.read_text(encoding="utf-8"))
    _reject_unknown_fields("Top-level", payload, {"suite", "sweeps", "resources", "environment", "overrides", "collect"})
    return _load_modern_suite(suite_path, payload)


def expand_suite(spec: SuiteSpec, *, run_root: str | Path | None = None) -> list[SuiteRun]:
    root = Path(run_root) if run_root is not None else ENGINE_ROOT / ".local" / "runs" / spec.id
    runs: list[SuiteRun] = []
    for case, profile, ranks, refine_levels, linear_rtol, continuation_step_max, repeat in product(
        spec.cases,
        spec.profiles,
        spec.ranks,
        spec.refine_levels,
        spec.linear_rtols,
        spec.continuation_step_max,
        range(1, spec.repeats + 1),
    ):
        case_toml = CASE_ROOT / case / "case.toml"
        if not case_toml.exists():
            raise ValueError(f"Suite {spec.id!r} references unknown case {case!r}: {case_toml}")
        axis_labels = _axis_labels(
            refine_levels=refine_levels,
            linear_rtol=linear_rtol,
            continuation_step_max=continuation_step_max,
        )
        axis_suffix = "".join(f"__{label}" for label in axis_labels)
        run_id = f"{case}__{profile}__r{ranks}{axis_suffix}__rep{repeat}"
        output_dir = root / case / profile / f"r{ranks}"
        for label in axis_labels:
            output_dir = output_dir / label
        output_dir = output_dir / f"repeat-{repeat}"
        sweep = {
            "refine_levels": refine_levels,
            "linear_rtol": linear_rtol,
            "continuation_step_max": continuation_step_max,
        }
        output_preset = _output_preset_override(spec.overrides)
        resource_name, resource = _select_resource(spec.resources, int(ranks))
        launcher = _launcher_tokens(resource)
        command = tuple(
            [
                *launcher,
                "-n",
                str(ranks),
                "petsc-ssr",
                "run",
                str(case_toml),
                "--profile",
                profile,
                "--output",
                str(output_dir),
                *_override_args(
                    spec.overrides,
                    refine_levels=refine_levels,
                    linear_rtol=linear_rtol,
                    continuation_step_max=continuation_step_max,
                ),
                *_collect_args(spec.collect, output_dir),
            ]
        )
        runs.append(
            SuiteRun(
                suite_id=spec.id,
                run_id=run_id,
                case=case,
                profile=profile,
                ranks=int(ranks),
                refine_levels=refine_levels,
                linear_rtol=linear_rtol,
                continuation_step_max=continuation_step_max,
                output_preset=output_preset,
                repeat=int(repeat),
                resource=resource_name,
                launcher=launcher,
                case_toml=case_toml,
                output_dir=output_dir,
                command=command,
                sweep=sweep,
                resolved_profile=_resolved_profile_manifest(profile, int(ranks), case_toml),
                artifacts=run_artifact_manifest(output_dir),
            )
        )
    return runs


def manifest_payload(spec: SuiteSpec, runs: list[SuiteRun]) -> dict[str, Any]:
    return {
        "suite": {
            "id": spec.id,
            "title": spec.title,
            "source": None if spec.source is None else str(spec.source),
            "cases": list(spec.cases),
            "profiles": list(spec.profiles),
            "ranks": list(spec.ranks),
            "sweeps": {
                "refine_levels": [value for value in spec.refine_levels if value is not None],
                "linear_rtol": [value for value in spec.linear_rtols if value is not None],
                "continuation_step_max": [value for value in spec.continuation_step_max if value is not None],
            },
            "repeats": int(spec.repeats),
            "timeout": spec.timeout,
            "resources": spec.resources,
            "environment": spec.environment,
            "overrides": spec.overrides,
            "collect": spec.collect,
        },
        "runs": [
            {
                **asdict(run),
                "case_toml": str(run.case_toml),
                "output_dir": str(run.output_dir),
                "command": list(run.command),
            }
            for run in runs
        ],
    }


def write_manifest(spec: SuiteSpec, runs: list[SuiteRun], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest_payload(spec, runs), indent=2) + "\n", encoding="utf-8")
    return out


def run_suite(spec: SuiteSpec, *, run_root: str | Path | None = None, dry_run: bool = False, max_runs: int | None = None) -> Path:
    root = Path(run_root) if run_root is not None else ENGINE_ROOT / ".local" / "runs" / spec.id
    runs = expand_suite(spec, run_root=root)
    manifest = write_manifest(spec, runs, root / "manifest.json")
    if dry_run:
        return manifest
    selected = runs if max_runs is None else runs[: max(0, int(max_runs))]
    run_env = os.environ.copy()
    run_env.update(spec.environment)
    for run in selected:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = run.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (run.output_dir / "command.json").write_text(
            json.dumps(_suite_command_payload(run, environment=spec.environment), indent=2) + "\n",
            encoding="utf-8",
        )
        with (logs_dir / "stdout.txt").open("w", encoding="utf-8") as stdout:
            completed = subprocess.run(
                list(run.command),
                cwd=ENGINE_ROOT,
                check=False,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                env=run_env,
            )
        materialize_suite_log_artifacts(run.output_dir, command=run.command)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, list(run.command))
    return manifest


def write_report(run_root: str | Path, *, output: str | Path | None = None) -> Path:
    return _write_report(run_root, output=output)


def compare_targets(run_root: str | Path, target_root: str | Path, *, output: str | Path | None = None) -> Path:
    return _compare_targets(run_root, target_root, output=output)


def _load_modern_suite(suite_path: Path, payload: dict[str, Any]) -> SuiteSpec:
    suite = dict(payload.get("suite", {}))
    _reject_unknown_fields("[suite]", suite, {"id", "title", "cases", "profiles", "ranks", "repeats", "timeout"})
    sweeps = _load_sweeps(payload)
    resources = _load_resources(payload)
    environment = _load_environment(payload)
    overrides = dict(payload.get("overrides", {}))
    _validate_overrides(overrides)
    collect = _load_collect(payload)
    suite_id = str(suite.get("id", "")).strip()
    title = str(suite.get("title", suite_id))
    cases = tuple(str(item) for item in suite.get("cases", []))
    profiles = tuple(str(item) for item in suite.get("profiles", []))
    ranks = tuple(int(item) for item in suite.get("ranks", []))
    repeats = int(suite.get("repeats", 1))
    if not suite_id:
        raise ValueError("[suite].id must be a non-empty string.")
    if not cases:
        raise ValueError("[suite].cases must list at least one benchmark case.")
    if not profiles:
        raise ValueError("[suite].profiles must list at least one solver profile.")
    if not ranks:
        raise ValueError("[suite].ranks must list at least one MPI size.")
    if any(rank <= 0 for rank in ranks):
        raise ValueError("[suite].ranks entries must be positive integers.")
    if repeats <= 0:
        raise ValueError("[suite].repeats must be a positive integer.")
    _validate_resource_limits(resources, ranks)
    _validate_performance_collection(suite_id=suite_id, title=title, overrides=overrides, collect=collect)
    return SuiteSpec(
        id=suite_id,
        title=title,
        cases=cases,
        profiles=profiles,
        ranks=ranks,
        refine_levels=sweeps["refine_levels"],
        linear_rtols=sweeps["linear_rtols"],
        continuation_step_max=sweeps["continuation_step_max"],
        repeats=repeats,
        timeout=None if suite.get("timeout") is None else str(suite.get("timeout")),
        resources=resources,
        environment=environment,
        overrides=overrides,
        collect=collect,
        source=suite_path,
    )


def _suite_command_payload(run: SuiteRun, *, environment: dict[str, str]) -> dict[str, Any]:
    return {
        "kind": "petsc_ssr_suite_command",
        "schema_version": 1,
        "suite": run.suite_id,
        "run_id": run.run_id,
        "case": run.case,
        "profile": run.profile,
        "ranks": int(run.ranks),
        "repeat": int(run.repeat),
        "resource": run.resource,
        "launcher": list(run.launcher),
        "command": list(run.command),
        "environment": dict(environment),
        "output_dir": str(run.output_dir),
        "sweep": dict(run.sweep),
        "resolved_profile": run.resolved_profile,
        "artifacts": run.artifacts,
    }


def _load_sweeps(payload: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    sweeps = dict(payload.get("sweeps", {}))
    _reject_unknown_fields("[sweeps]", sweeps, {"refine_levels", "linear_rtol", "continuation_step_max"})
    return {
        "refine_levels": _optional_int_axis(sweeps.get("refine_levels")),
        "linear_rtols": _optional_float_axis(sweeps.get("linear_rtol")),
        "continuation_step_max": _optional_int_axis(sweeps.get("continuation_step_max")),
    }


def _optional_int_axis(value: object) -> tuple[int | None, ...]:
    if value is None:
        return (None,)
    values = value if isinstance(value, list) else [value]
    parsed = tuple(int(item) for item in values)
    return parsed or (None,)


def _optional_float_axis(value: object) -> tuple[float | None, ...]:
    if value is None:
        return (None,)
    values = value if isinstance(value, list) else [value]
    parsed = tuple(float(item) for item in values)
    return parsed or (None,)


def _load_resources(payload: dict[str, Any]) -> dict[str, Any]:
    resources = dict(payload.get("resources", {}))
    out: dict[str, Any] = {}
    for name, value in resources.items():
        section = f"[resources.{name}]"
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be a table in suite TOML.")
        item = dict(value)
        _reject_unknown_fields(
            section,
            item,
            {"machine", "cores", "nodes", "ranks_per_node", "max_ranks", "launcher", "partition", "time_limit"},
        )
        normalized = _jsonable(item)
        for key in ("cores", "nodes", "ranks_per_node", "max_ranks"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = int(normalized[key])
                if normalized[key] <= 0:
                    raise ValueError(f"{section}.{key} must be a positive integer.")
        if "max_ranks" in normalized and "cores" in normalized and normalized["max_ranks"] > normalized["cores"]:
            raise ValueError(f"{section}.max_ranks must not exceed cores when both are set.")
        if (
            "max_ranks" in normalized
            and "nodes" in normalized
            and "ranks_per_node" in normalized
            and normalized["max_ranks"] > normalized["nodes"] * normalized["ranks_per_node"]
        ):
            raise ValueError(f"{section}.max_ranks must not exceed nodes * ranks_per_node when all are set.")
        if "launcher" in normalized:
            launcher = str(normalized["launcher"]).strip()
            if not launcher:
                raise ValueError(f"{section}.launcher must be a non-empty command name.")
            normalized["launcher"] = launcher
        out[str(name)] = normalized
    return out


def _select_resource(resources: dict[str, Any], ranks: int) -> tuple[str | None, dict[str, Any]]:
    if not resources:
        return None, {}
    for name, resource in resources.items():
        if not isinstance(resource, dict):
            continue
        max_ranks = resource.get("max_ranks")
        if max_ranks is None or int(ranks) <= int(max_ranks):
            return str(name), dict(resource)
    cap_text = ", ".join(
        f"{name}={resource.get('max_ranks')}"
        for name, resource in resources.items()
        if isinstance(resource, dict) and resource.get("max_ranks") is not None
    )
    raise ValueError(f"No suite resource can launch {int(ranks)} ranks; declared max_ranks: {cap_text or 'none'}.")


def _launcher_tokens(resource: dict[str, Any]) -> tuple[str, ...]:
    launcher = str(resource.get("launcher") or "mpiexec").strip()
    tokens = tuple(shlex.split(launcher))
    if not tokens:
        raise ValueError("Resolved suite launcher must not be empty.")
    return tokens


def _validate_resource_limits(resources: dict[str, Any], ranks: tuple[int, ...]) -> None:
    if not resources:
        return
    caps = {
        name: int(item["max_ranks"])
        for name, item in resources.items()
        if isinstance(item, dict) and item.get("max_ranks") is not None
    }
    if not caps:
        return
    unsupported = [rank for rank in ranks if all(rank > cap for cap in caps.values())]
    if unsupported:
        cap_text = ", ".join(f"{name}={cap}" for name, cap in sorted(caps.items()))
        raise ValueError(
            f"[suite].ranks entries {unsupported} exceed declared [resources.*].max_ranks limits ({cap_text})."
        )


def _load_environment(payload: dict[str, Any]) -> dict[str, str]:
    environment = dict(payload.get("environment", {}))
    out: dict[str, str] = {}
    for name, value in environment.items():
        key = str(name).strip()
        if not key:
            raise ValueError("[environment] keys must be non-empty.")
        if isinstance(value, (dict, list)):
            raise ValueError(f"[environment].{key} must be a scalar value.")
        out[key] = str(value)
    return out


def _validate_overrides(overrides: dict[str, Any]) -> None:
    _reject_unknown_fields("[overrides]", overrides, {"continuation", "linear", "mesh", "output"})
    _reject_unknown_fields("[overrides.continuation]", _table(overrides, "continuation"), {"omega_max", "step_max"})
    _reject_unknown_fields("[overrides.linear]", _table(overrides, "linear"), {"rtol", "max_it"})
    _reject_unknown_fields("[overrides.mesh]", _table(overrides, "mesh"), {"refine_levels"})
    output = _table(overrides, "output")
    _reject_unknown_fields("[overrides.output]", output, {"preset"})
    if "preset" in output:
        _normalize_output_preset(output["preset"])


def _load_collect(payload: dict[str, Any]) -> dict[str, bool]:
    collect = dict(payload.get("collect", {}))
    _validate_collect(collect)
    out: dict[str, bool] = {}
    for name, value in collect.items():
        key = str(name).strip()
        if not isinstance(value, bool):
            raise ValueError(f"[collect].{key} must be a TOML boolean.")
        out[key] = value
    return out


def _validate_collect(collect: dict[str, Any]) -> None:
    _reject_unknown_fields(
        "[collect]",
        collect,
        {
            "petsc_log_view",
            "log_view",
            "options_view",
            "petsc_options_view",
            "options_left",
            "environment",
            "resolved_manifest",
        },
    )
    if "petsc_log_view" in collect and "log_view" in collect:
        raise ValueError("[collect] must use either petsc_log_view or log_view, not both.")
    if "options_view" in collect and "petsc_options_view" in collect:
        raise ValueError("[collect] must use either options_view or petsc_options_view, not both.")


def _validate_performance_collection(
    *,
    suite_id: str,
    title: str,
    overrides: dict[str, Any],
    collect: dict[str, bool],
) -> None:
    if not _is_performance_suite(suite_id=suite_id, title=title, overrides=overrides):
        return
    missing: list[str] = []
    if not _collect_enabled(collect, "petsc_log_view", "log_view"):
        missing.append("petsc_log_view")
    if not _collect_enabled(collect, "options_left"):
        missing.append("options_left")
    if missing:
        missing_text = ", ".join(f"[collect].{name} = true" for name in missing)
        raise ValueError(
            f"Performance/scaling suite {suite_id!r} must enable {missing_text} "
            "so reports have PETSc timing data and options-left cleanliness."
        )


def _is_performance_suite(*, suite_id: str, title: str, overrides: dict[str, Any]) -> bool:
    if _output_preset_override(overrides) == "performance":
        return True
    label = f"{suite_id} {title}".casefold()
    return "scaling" in label or "performance" in label


def _collect_enabled(collect: dict[str, bool], *names: str) -> bool:
    return any(bool(collect.get(name, False)) for name in names)


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"[overrides.{key}] must be a table in suite TOML.")
    return dict(value)


def _override_args(
    overrides: dict[str, Any],
    *,
    refine_levels: int | None = None,
    linear_rtol: float | None = None,
    continuation_step_max: int | None = None,
) -> list[str]:
    args: list[str] = []
    continuation = dict(overrides.get("continuation", {}))
    linear = dict(overrides.get("linear", {}))
    mesh = dict(overrides.get("mesh", {}))
    output = dict(overrides.get("output", {}))
    if refine_levels is not None:
        mesh["refine_levels"] = refine_levels
    if linear_rtol is not None:
        linear["rtol"] = linear_rtol
    if continuation_step_max is not None:
        continuation["step_max"] = continuation_step_max
    if "omega_max" in continuation:
        args.extend(["--omega-max", str(continuation["omega_max"])])
    if "step_max" in continuation:
        args.extend(["--continuation-step-max", str(continuation["step_max"])])
    if "rtol" in linear:
        args.extend(["--linear-rtol", str(linear["rtol"])])
    if "max_it" in linear:
        args.extend(["--ksp-max-it", str(linear["max_it"])])
    if "refine_levels" in mesh:
        args.extend(["--refine-levels", str(mesh["refine_levels"])])
    if "preset" in output:
        args.extend(["--output-preset", _normalize_output_preset(output["preset"])])
    return args


def _output_preset_override(overrides: dict[str, Any]) -> str | None:
    output = dict(overrides.get("output", {}))
    if "preset" not in output:
        return None
    return _normalize_output_preset(output["preset"])


def _normalize_output_preset(value: object) -> str:
    return _normalize_config_output_preset(value, section_name="[overrides.output]")


def _axis_labels(*, refine_levels: int | None, linear_rtol: float | None, continuation_step_max: int | None) -> list[str]:
    labels: list[str] = []
    if refine_levels is not None:
        labels.append(f"refine-l{refine_levels}")
    if linear_rtol is not None:
        labels.append(f"linear-rtol-{_slug_number(linear_rtol)}")
    if continuation_step_max is not None:
        labels.append(f"steps-{continuation_step_max}")
    return labels


def _slug_number(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collect_args(collect: dict[str, Any], output_dir: Path) -> list[str]:
    args: list[str] = []
    logs_dir = output_dir / "logs"
    if bool(collect.get("petsc_log_view", collect.get("log_view", False))):
        args.extend([f"--petsc-opt=-log_view", f"--petsc-opt=:{logs_dir / 'petsc_log.txt'}"])
    if bool(collect.get("options_view", collect.get("petsc_options_view", False))):
        args.extend([f"--petsc-opt=-options_view", f"--petsc-opt=:{logs_dir / 'options_view.txt'}"])
    if bool(collect.get("options_left", False)):
        args.append("--petsc-opt=-options_left")
    return args


def _resolved_profile_manifest(profile: str, ranks: int, case_toml: Path) -> dict[str, Any]:
    resolved = load_solver_profile(profile, world_size=ranks)
    data = resolved.data
    cfg = load_run_case_config(case_toml).validate()
    element_degree = int(str(cfg.problem.elem_type).strip().upper()[1:])
    supported = (
        ("gamg", "pmg", "none")
        if str(cfg.problem.analysis).strip().lower() == "seepage"
        else ("gamg", "bddc", "fetidp", "pmg", "none")
    )
    pc_policy = pc_variant_from_backend(
        data.get("pc_backend"),
        element_degree=element_degree,
        supported=supported,
    )
    native_algorithm = native_linear_algorithm_selector(
        data.get("algorithm"),
        pc_variant=pc_policy.variant,
        deflation=bool(data.get("deflation", False)),
    )
    return {
        "name": resolved.name,
        "world_size": int(ranks),
        "linear": {
            "algorithm": data.get("algorithm"),
            "native_algorithm": native_algorithm,
            "ksp_type": data.get("ksp_type"),
            "norm_type": data.get("norm_type"),
            "rtol": data.get("tolerance"),
            "max_it": data.get("max_iterations"),
            "deflation": data.get("deflation"),
        },
        "pc": {
            "backend": data.get("pc_backend"),
            "variant": pc_policy.variant,
            "requested_variant": pc_policy.requested_variant,
            "fallback_reason": pc_policy.fallback_reason,
        },
        "pmg": _resolved_pmg_profile_payload(data),
    }


def _resolved_pmg_profile_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank_policy": data.get("pmg_rank_policy"),
        "apply_backend": data.get("pmg_apply_backend"),
        "p2_active_ranks": data.get("pmg_shell_p2_active_ranks"),
        "p1_active_ranks": data.get("pmg_shell_p1_active_ranks"),
        "p2_policy": data.get("pmg_shell_p2_rank_policy"),
        "p1_policy": data.get("pmg_shell_p1_rank_policy"),
        "subcomm_type": data.get("pmg_shell_subcomm_type"),
        "fine_ksp_max_it": data.get("pmg_shell_fine_ksp_max_it"),
        "p2_ksp_max_it": data.get("pmg_shell_p2_ksp_max_it"),
        "smoother_ksp_type": data.get("pmg_smoother_ksp_type"),
        "smoother_pc_type": data.get("pmg_smoother_pc_type"),
        "smoother_max_it": data.get("pmg_smoother_max_it"),
        "coarse_pc_type": data.get("pmg_coarse_pc_type"),
        "coarse_lu_max_dofs": data.get("pmg_coarse_lu_max_dofs"),
        "coarse_redundant_group_size": data.get("pmg_coarse_redundant_group_size"),
        "coarse_gamg_aggressive_square_graph": data.get("pmg_coarse_gamg_aggressive_square_graph"),
        "coarse_telescope_active_ranks": data.get("pmg_coarse_telescope_active_ranks"),
        "coarse_telescope_subcomm_type": data.get("pmg_coarse_telescope_subcomm_type"),
        "coarse_telescope_ksp_type": data.get("pmg_coarse_telescope_ksp_type"),
        "coarse_telescope_ksp_rtol": data.get("pmg_coarse_telescope_ksp_rtol"),
        "coarse_telescope_ksp_max_it": data.get("pmg_coarse_telescope_ksp_max_it"),
        "coarse_telescope_pc_type": data.get("pmg_coarse_telescope_pc_type"),
        "p2_telescope_active_ranks": data.get("pmg_p2_telescope_active_ranks"),
        "p2_telescope_subcomm_type": data.get("pmg_p2_telescope_subcomm_type"),
        "p2_telescope_ksp_type": data.get("pmg_p2_telescope_ksp_type"),
        "p2_telescope_ksp_rtol": data.get("pmg_p2_telescope_ksp_rtol"),
        "p2_telescope_ksp_max_it": data.get("pmg_p2_telescope_ksp_max_it"),
        "p2_telescope_pc_type": data.get("pmg_p2_telescope_pc_type"),
    }


def _reject_unknown_fields(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{section} fields {unknown} are not supported in suite TOML.")
