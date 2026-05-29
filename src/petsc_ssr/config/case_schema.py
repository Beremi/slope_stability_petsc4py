"""General TOML configuration loading for config-driven case execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import tomllib

from ..core.elements import validate_supported_elem_type
from ..problem_assets import load_material_rows_for_asset
from ..assets import load_problem_asset
from .profiles import load_continuation_profile, load_newton_profile, load_seepage_profile, load_solver_profile
from .validators import (
    normalize_output_preset,
    reject_profile_default_repeats,
    reject_unknown_fields,
    validate_case_metadata,
)


TomlValue = str | int | float | bool | list[Any] | dict[str, Any]
ENGINE_ROOT = Path(__file__).resolve().parents[3]
MODERN_MECHANICS_BACKEND = "petsc_ssr_full_c"
MODERN_NODE_ORDERING = "native_dmplex"
LEGACY_MECHANICS_BACKEND = "legacy_array"
LEGACY_NODE_ORDERING = "block_metis"


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    case: str = ""
    analysis: str = "ssr"
    dimension: int = 3
    elem_type: str = "P2"
    davis_type: str = "B"
    asset: str = ""
    mesh_variant: str | None = None
    profile: str | None = None
    refine_levels: int = 0
    partitioner: str = "parmetis"
    seepage_coupled: bool = False


@dataclass(frozen=True)
class ExecutionConfig:
    mechanics_backend: str = MODERN_MECHANICS_BACKEND
    node_ordering: str = MODERN_NODE_ORDERING
    mpi_distribute_by_nodes: bool = True
    constitutive_mode: str = "overlap"
    tangent_kernel: str = "rows"
    store_step_u: bool = True


@dataclass(frozen=True)
class NewtonConfig:
    profile: str = ""
    profile_description: str = ""
    algorithm: str = ""
    it_max: int = 200
    it_damp_max: int = 10
    tol: float = 1e-4
    r_min: float = 1e-4
    stopping_criterion: str = "relative_residual"
    stopping_tol: float | None = None
    line_search: str = "alg5"
    armijo_alpha0: float = 1.0
    armijo_c1: float = 1.0e-4
    armijo_shrink: float = 0.5
    armijo_max_ls: int | None = None
    armijo_rescale_trial_to_omega: bool = True
    armijo_fallback_to_alg5: bool = True


@dataclass(frozen=True)
class ContinuationConfig:
    profile: str = ""
    profile_description: str = ""
    algorithm: str = ""
    method: str = "indirect"
    predictor: str = "secant"
    omega_step_controller: str = "classic"
    secant_correction_mode: str = "none"
    first_newton_warm_start_mode: str = "none"
    lambda_init: float = 1.0
    d_lambda_init: float = 0.1
    d_lambda_min: float = 1e-3
    d_lambda_diff_scaled_min: float = 1e-3
    lambda_ell: float = 1.0
    omega_max: float = 1.2e7
    step_max: int = 100
    d_omega_ini_scale: float = 0.2
    d_t_min: float = 1e-3
    omega_no_increase_newton_threshold: int | None = None
    omega_half_newton_threshold: int | None = None
    omega_target_newton_iterations: float | None = None
    omega_adapt_min_scale: float | None = None
    omega_adapt_max_scale: float | None = None
    omega_hard_newton_threshold: int | None = None
    omega_hard_linear_threshold: int | None = None
    omega_efficiency_floor: float | None = None
    omega_efficiency_drop_ratio: float | None = None
    omega_efficiency_window: int = 3
    omega_hard_shrink_scale: float | None = None
    step_length_cap_mode: str = "none"
    step_length_cap_factor: float = 1.0
    init_newton_stopping_criterion: str | None = None
    init_newton_stopping_tol: float | None = None
    fine_newton_stopping_criterion: str | None = None
    fine_newton_stopping_tol: float | None = None
    fine_switch_mode: str = "none"
    fine_switch_distance_factor: float = 2.0


@dataclass(frozen=True)
class LinearSolverConfig:
    profile: str = ""
    profile_description: str = ""
    resolved_world_size: int = 1
    algorithm: str = ""
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
    ksp_type: str = "fgmres"
    norm_type: str = "unpreconditioned"
    deflation: bool = True
    deflation_solver: str | None = None
    tolerance: float = 1e-1
    max_iterations: int = 100
    deflation_basis_tolerance: float = 1e-3
    verbose: bool = False
    threads: int = 16
    print_level: int = 0
    use_as_preconditioner: bool = True
    factor_solver_type: str | None = None
    pc_backend: str | None = "hypre"
    pmg_profile: str | None = None
    pmg_coarse_mesh_variant: str | None = None
    pmg_fine_hierarchy_mode: str = "default"
    pmg_shell_p2_active_ranks: int | None = None
    pmg_shell_p1_active_ranks: int | None = None
    pmg_shell_p2_rank_policy: str | None = None
    pmg_shell_p1_rank_policy: str | None = None
    pmg_rank_policy: str | None = None
    pmg_apply_backend: str | None = None
    pmg_coarse_pc_type: str | None = None
    pmg_coarse_lu_max_dofs: int | None = None
    pmg_coarse_redundant_group_size: int | None = None
    pmg_coarse_gamg_aggressive_square_graph: bool | None = None
    pmg_coarse_telescope_active_ranks: int | None = None
    pmg_coarse_telescope_subcomm_type: str | None = None
    pmg_coarse_telescope_ksp_type: str | None = None
    pmg_coarse_telescope_ksp_rtol: float | None = None
    pmg_coarse_telescope_ksp_max_it: int | None = None
    pmg_coarse_telescope_pc_type: str | None = None
    pmg_p2_telescope_active_ranks: int | None = None
    pmg_p2_telescope_subcomm_type: str | None = None
    pmg_p2_telescope_ksp_type: str | None = None
    pmg_p2_telescope_ksp_rtol: float | None = None
    pmg_p2_telescope_ksp_max_it: int | None = None
    pmg_p2_telescope_pc_type: str | None = None
    pmg_smoother_ksp_type: str | None = None
    pmg_smoother_pc_type: str | None = None
    pmg_smoother_max_it: int | None = None
    pmg_shell_subcomm_type: str | None = None
    pmg_shell_fine_ksp_max_it: int | None = None
    pmg_shell_p2_ksp_max_it: int | None = None
    pmg_shell_p1_pc_type: str | None = None
    pmg_shell_p1_redundant_number: int | None = None
    pmg_shell_p1_redundant_ksp_type: str | None = None
    pmg_shell_p1_redundant_ksp_rtol: float | None = None
    pmg_shell_p1_redundant_ksp_max_it: int | None = None
    pmg_shell_p1_redundant_pc_type: str | None = None
    max_deflation_basis_vectors: int = 48
    preconditioner_matrix_source: str = "tangent"
    preconditioner_matrix_policy: str = "current"
    preconditioner_rebuild_policy: str = "every_newton"
    preconditioner_rebuild_interval: int = 1
    pc_gamg_process_eq_limit: int | None = None
    pc_gamg_threshold: float | None = None
    pc_gamg_aggressive_coarsening: int | None = None
    pc_gamg_aggressive_square_graph: bool | None = None
    pc_gamg_aggressive_mis_k: int | None = None
    pc_hypre_coarsen_type: str | None = "HMIS"
    pc_hypre_interp_type: str | None = "ext+i"
    pc_hypre_strong_threshold: float | None = None
    pc_hypre_boomeramg_max_iter: int | None = 1
    pc_hypre_P_max: int | None = None
    pc_hypre_agg_nl: int | None = None
    pc_hypre_nongalerkin_tol: float | None = None
    pc_bddc_symmetric: bool | None = None
    pc_bddc_dirichlet_ksp_type: str | None = None
    pc_bddc_dirichlet_pc_type: str | None = None
    pc_bddc_neumann_ksp_type: str | None = None
    pc_bddc_neumann_pc_type: str | None = None
    pc_bddc_coarse_ksp_type: str | None = None
    pc_bddc_coarse_pc_type: str | None = None
    pc_bddc_dirichlet_approximate: bool | None = None
    pc_bddc_neumann_approximate: bool | None = None
    pc_bddc_monolithic: bool | None = None
    pc_bddc_coarse_redundant_pc_type: str | None = None
    pc_bddc_switch_static: bool | None = None
    pc_bddc_use_deluxe_scaling: bool | None = None
    pc_bddc_use_vertices: bool | None = None
    pc_bddc_use_edges: bool | None = None
    pc_bddc_use_faces: bool | None = None
    pc_bddc_use_change_of_basis: bool | None = None
    pc_bddc_use_change_on_faces: bool | None = None
    pc_bddc_check_level: int | None = None
    compiled_outer: bool = False
    recycle_preconditioner: bool = True
    pmg_options_file: Path | None = None
    petsc_opt: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeepageConfig:
    profile: str = ""
    profile_description: str = ""
    linear_tolerance: float = 1e-10
    linear_max_iter: int = 500
    nonlinear_max_iter: int = 50
    extra: dict[str, TomlValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportConfig:
    preset: str = "standard"
    write_custom_debug_bundle: bool = True
    write_history_json: bool = True
    write_solution_vtu: bool = True
    custom_debug_name: str = "run_debug.h5"
    history_name: str = "continuation_history.json"
    solution_name: str = "final_solution.vtu"


@dataclass(frozen=True)
class RunCaseConfig:
    problem: ProblemConfig
    execution: ExecutionConfig = ExecutionConfig()
    continuation: ContinuationConfig = ContinuationConfig()
    newton: NewtonConfig = NewtonConfig()
    linear_solver: LinearSolverConfig = LinearSolverConfig()
    seepage: SeepageConfig = SeepageConfig()
    export: ExportConfig = ExportConfig()
    geometry: dict[str, TomlValue] = field(default_factory=dict)

    def validate(self) -> "RunCaseConfig":
        valid_stopping_criteria = {
            "residual",
            "rel_residual",
            "relative_residual",
            "correction",
            "rel_correction",
            "relative_correction",
            "relative_newton_correction",
            "delta_lambda",
            "abs_delta_lambda",
            "absolute_delta_lambda",
        }
        valid_line_search_modes = {
            "alg5",
            "armijo_residual",
        }
        if not self.problem.asset:
            raise ValueError("[problem].asset must be set.")
        if self.problem.analysis.lower() not in {"ssr", "ll", "seepage"}:
            raise ValueError(f"Unsupported analysis {self.problem.analysis!r}.")
        if str(self.newton.stopping_criterion).strip().lower() not in valid_stopping_criteria:
            raise ValueError(
                "The newton stopping_criterion must be relative_residual, relative_correction, or absolute_delta_lambda."
            )
        if str(self.newton.line_search).strip().lower() not in valid_line_search_modes:
            raise ValueError("The newton line_search must be alg5 or armijo_residual.")
        analysis = self.problem.analysis.lower()
        continuation_method = str(self.continuation.method).strip().lower().replace("_", "-")
        continuation_algorithm = str(self.continuation.algorithm or continuation_method).strip().lower().replace("_", "-")
        newton_algorithm = str(self.newton.algorithm).strip().lower().replace("_", "-")
        if analysis != "seepage":
            if continuation_method not in {"indirect", "direct"}:
                raise ValueError("The continuation method must be indirect or direct.")
            if continuation_algorithm not in {"indirect", "direct"}:
                raise ValueError("The continuation algorithm must be indirect or direct.")
            if continuation_algorithm != continuation_method:
                raise ValueError("The continuation algorithm must match the continuation method for native runs.")
            expected_newton = "indirect-ssr" if continuation_method == "indirect" else "fixed-load"
            newton_algorithm = newton_algorithm or expected_newton
            if newton_algorithm != expected_newton:
                raise ValueError(
                    f"The newton algorithm must be {expected_newton} for continuation method {continuation_method}."
                )
            if analysis == "ll" and continuation_method != "direct":
                raise ValueError("Limit-load cases must use a direct continuation profile.")
        for field_name in ("init_newton_stopping_criterion", "fine_newton_stopping_criterion"):
            value = getattr(self.continuation, field_name)
            if value is not None and str(value).strip().lower() not in valid_stopping_criteria:
                raise ValueError(
                    f"The continuation {field_name} must be relative_residual, relative_correction, or absolute_delta_lambda."
                )
        validate_supported_elem_type(self.problem.dimension, self.problem.elem_type)
        if self.problem.analysis.lower() != "seepage" and not self.material_rows():
            raise ValueError("At least one asset material row is required for non-seepage cases.")
        return self

    def material_rows(self) -> list[list[float]]:
        rows = load_material_rows_for_asset(self.problem.asset)
        return [] if rows is None else rows


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _resolve_section_paths(config_path: Path, data: dict[str, Any]) -> dict[str, TomlValue]:
    resolved: dict[str, TomlValue] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_section_paths(config_path, value)
        elif isinstance(value, list):
            resolved[key] = value
        elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_dir")):
            resolved[key] = _resolve_path(config_path, value)
        else:
            resolved[key] = value
    return resolved


def _normalize_modern_case_schema(data: dict[str, Any]) -> dict[str, Any]:
    """Convert the standalone engine case schema to the internal config model.

    The public schema intentionally describes cases, meshes, physics, and
    profile choices.  The internal model still carries several migration-era field
    names; this adapter keeps that debt out of active benchmark TOMLs.
    """

    modern_keys = {"case", "mesh", "physics", "linear", "output"}
    if not (modern_keys & set(data)):
        return data

    allowed_top_level = {
        "case",
        "mesh",
        "physics",
        "continuation",
        "newton",
        "linear",
        "seepage",
        "output",
        "geometry",
    }
    reject_unknown_fields(
        "Top-level",
        data,
        allowed_top_level,
        "active case files use case, mesh, physics, continuation, newton, linear, output, seepage, and geometry. Notebook metadata belongs in benchmark sidecars.",
    )

    case = dict(data.get("case", {}))
    mesh = dict(data.get("mesh", {}))
    physics = dict(data.get("physics", {}))
    mechanics = dict(physics.get("mechanics", {}))
    seepage_physics = dict(physics.get("seepage", {}))
    continuation_raw = dict(data.get("continuation", {}))
    newton_raw = dict(data.get("newton", {}))
    linear_raw = dict(data.get("linear", {}))
    seepage_raw = dict(data.get("seepage", {}))
    output_raw = dict(data.get("output", {}))

    reject_unknown_fields("[case]", case, {"id", "name", "title", "tags"}, "case metadata must be compact and user-facing.")
    reject_unknown_fields("[mesh]", mesh, {"asset", "variant", "element", "profile"}, "mesh setup selects an asset variant and element order; refinement and partitioning belong to suites or launch overrides.")
    reject_unknown_fields("[physics]", physics, {"mechanics", "seepage"}, "physics blocks must be named by physics family.")
    reject_unknown_fields("[physics.mechanics]", mechanics, {"model", "davis", "profile"}, "mechanics physics uses model, davis, and optional profile.")
    reject_unknown_fields("[physics.seepage]", seepage_physics, {"model"}, "seepage physics uses a model name only; asset files own coefficients and BCs.")
    reject_unknown_fields(
        "[continuation]",
        continuation_raw,
        {
            "profile",
            "omega_max",
            "lambda_init",
            "d_lambda_init",
            "d_lambda_min",
            "d_lambda_diff_scaled_min",
            "lambda_ell",
            "d_t_min",
            "d_omega_ini_scale",
            "step_max",
        },
        "continuation algorithm selectors and controller policy belong in continuation profiles; cases may only set mathematical load/cap controls.",
    )
    reject_unknown_fields(
        "[newton]",
        newton_raw,
        {"profile"},
        "Newton iteration, tolerance, algorithm, damping, and line-search policy belong in Newton profiles.",
    )
    reject_unknown_fields(
        "[linear]",
        linear_raw,
        {"profile"},
        "linear solver details belong in solver profiles; use CLI/debug overrides for one-off experiments.",
    )
    reject_unknown_fields(
        "[seepage]",
        seepage_raw,
        {"profile"},
        "seepage runtime policy belongs in named seepage profiles; seepage physics belongs in meshes/<asset>/definition.py.",
    )
    reject_unknown_fields("[output]", output_raw, {"preset"}, "output uses named presets; generated artifact paths and raw artifact arrays are not case inputs.")
    validate_case_metadata(case, mesh=mesh, physics=physics)

    tags = {str(tag).strip().lower() for tag in case.get("tags", [])}
    mechanics_model = str(mechanics.get("model", "")).strip().lower()
    if mechanics_model:
        analysis = "ll" if "limit" in mechanics_model or mechanics_model.endswith("_ll") or "ll" in tags else "ssr"
    elif seepage_physics:
        analysis = "seepage"
    else:
        analysis = "ssr"

    default_continuation_profile = (
        "direct-limit-load" if analysis == "ll" else ("indirect-classic" if analysis == "ssr" else "")
    )
    continuation_profile = str(continuation_raw.pop("profile", default_continuation_profile)).strip()
    if continuation_profile:
        continuation_data = load_continuation_profile(continuation_profile).data
        reject_profile_default_repeats("[continuation]", continuation_raw, continuation_data)
        continuation_data.update(continuation_raw)
    else:
        if continuation_raw:
            raise ValueError("[continuation] controls require an explicit profile.")
        continuation_data = {}

    default_newton_profile = (
        "limit-load-regularized" if analysis == "ll" else ("indirect-regularized" if analysis == "ssr" else "")
    )
    newton_profile = str(newton_raw.pop("profile", default_newton_profile)).strip()
    if newton_profile:
        newton_data = load_newton_profile(newton_profile).data
        reject_profile_default_repeats("[newton]", newton_raw, newton_data)
        newton_data.update(newton_raw)
    else:
        if newton_raw:
            raise ValueError("[newton] controls require an explicit profile.")
        newton_data = {}

    linear_profile = str(linear_raw.pop("profile", "pmg-deflated-baseline")).strip()
    linear_data = load_solver_profile(linear_profile).data
    linear_data.update(linear_raw)

    output_preset = normalize_output_preset(output_raw.get("preset", "standard"))
    write_outputs = output_preset != "none"

    normalized: dict[str, Any] = {
        "benchmark": {
            "title": str(case.get("title", case.get("id", case.get("name", "case")))),
            "comparison_kind": "continuation" if analysis in {"ssr", "ll"} else "seepage",
            "suite": False,
        },
        "problem": {
            "name": str(case.get("id", case.get("name", "case"))),
            "analysis": analysis,
            "asset": str(mesh.get("asset", "")),
            "mesh_variant": None if mesh.get("variant") is None else str(mesh.get("variant")),
            "profile": None if mesh.get("profile") is None else str(mesh.get("profile")),
            "elem_type": str(mesh.get("element", "P2")),
            "davis_type": str(mechanics.get("davis", "B")),
            "seepage_coupled": bool(mechanics and seepage_physics),
        },
        "continuation": continuation_data,
        "newton": newton_data,
        "linear_solver": linear_data,
        "export": {
            "preset": output_preset,
            "write_solution_vtu": write_outputs,
            "write_history_json": write_outputs,
            "write_custom_debug_bundle": False,
        },
    }
    if data.get("seepage") is not None:
        normalized["seepage"] = seepage_raw
    if data.get("geometry") is not None:
        normalized["geometry"] = data["geometry"]
    return normalized


def _uses_modern_case_schema(data: dict[str, Any]) -> bool:
    return bool({"case", "mesh", "physics", "linear", "output"} & set(data))


def _legacy_case_schema_enabled() -> bool:
    return os.environ.get("PETSC_SSR_ALLOW_LEGACY_CASE_SCHEMA", "").strip().lower() in {"1", "true", "yes", "on"}


def _execution_defaults(*, modern_schema: bool) -> ExecutionConfig:
    if modern_schema:
        return ExecutionConfig()
    return ExecutionConfig(
        mechanics_backend=LEGACY_MECHANICS_BACKEND,
        node_ordering=LEGACY_NODE_ORDERING,
    )


def load_run_case_config(path: str | Path) -> RunCaseConfig:
    config_path = Path(path).resolve()
    raw_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    modern_schema = _uses_modern_case_schema(raw_data)
    if not modern_schema and not _legacy_case_schema_enabled():
        raise ValueError(
            "Legacy/internal case schema is not part of the public benchmark model; "
            "use [case], [mesh], [physics], [continuation], [newton], [linear], [seepage], and [output]. "
            "Set PETSC_SSR_ALLOW_LEGACY_CASE_SCHEMA=1 only for explicit migration/debug compatibility."
        )
    data = _normalize_modern_case_schema(raw_data)

    allowed_top_level = {
        "benchmark",
        "notebook",
        "problem",
        "execution",
        "continuation",
        "newton",
        "linear_solver",
        "seepage",
        "export",
        "geometry",
        "materials",
        "case_data",
    }
    reject_unknown_fields(
        "Top-level",
        data,
        allowed_top_level,
        "committed case configs must use the documented benchmark, problem, numerical, export, and geometry sections.",
    )

    problem_data = dict(data.get("problem", {}))
    execution_data = dict(data.get("execution", {}))
    continuation_data = dict(data.get("continuation", {}))
    newton_data = dict(data.get("newton", {}))
    linear_data = dict(data.get("linear_solver", {}))
    seepage_data = dict(data.get("seepage", {}))
    export_data = dict(data.get("export", {}))
    geometry_raw = dict(data.get("geometry", {}))
    execution_defaults = _execution_defaults(modern_schema=modern_schema)
    reject_unknown_fields(
        "[geometry]",
        geometry_raw,
        {"quadrature_rule"},
        "only generic numerical geometry controls are allowed here; problem geometry belongs in meshes/<asset>/definition.py.",
    )
    geometry_data = _resolve_section_paths(config_path, geometry_raw)
    if data.get("materials") is not None:
        raise ValueError("Committed case configs must not define [[materials]]; use meshes/<asset>/definition.py.")
    if data.get("case_data") is not None:
        raise ValueError("Committed case configs must not define [case_data]; use [problem].asset and mesh_variant.")
    forbidden_problem_fields = {"dimension", "mesh_path", "mesh_" + "boundary_type", "seepage", "variant"}
    forbidden_present = sorted(forbidden_problem_fields & set(problem_data))
    if forbidden_present:
        raise ValueError(f"[problem] fields {forbidden_present} are not supported; use asset mesh variants.")
    forbidden_seepage_fields = {"water_unit_weight", "conductivity"}
    forbidden_seepage = sorted(forbidden_seepage_fields & set(seepage_data))
    if forbidden_seepage:
        raise ValueError(f"[seepage] fields {forbidden_seepage} are asset-owned; use meshes/<asset>/definition.py.")
    reject_unknown_fields(
        "[seepage]",
        seepage_data,
        {"profile"},
        "seepage runtime policy belongs in named seepage profiles; seepage physics belongs in meshes/<asset>/definition.py.",
    )

    asset_name = str(problem_data.get("asset", "")).strip()
    if not asset_name:
        raise ValueError("[problem].asset must be set.")
    asset = load_problem_asset(asset_name)
    analysis = str(problem_data.get("analysis", "ssr")).strip().lower()
    seepage_enabled = analysis == "seepage" or bool(problem_data.get("seepage_coupled", False)) or bool(seepage_data)
    problem = ProblemConfig(
        name=str(problem_data.get("name", "case")),
        case=str(problem_data.get("case", "")),
        analysis=analysis,
        dimension=int(asset.dimension),
        elem_type=str(problem_data.get("elem_type", "P2")),
        davis_type=str(problem_data.get("davis_type", "B")),
        asset=asset_name,
        mesh_variant=None if problem_data.get("mesh_variant") is None else str(problem_data.get("mesh_variant")),
        profile=None if problem_data.get("profile") is None else str(problem_data.get("profile")),
        refine_levels=int(problem_data.get("refine_levels", 0)),
        partitioner=str(problem_data.get("partitioner", "parmetis")),
        seepage_coupled=bool(problem_data.get("seepage_coupled", False)),
    )
    execution = ExecutionConfig(
        mechanics_backend=str(execution_data.get("mechanics_backend", execution_defaults.mechanics_backend)),
        node_ordering=str(execution_data.get("node_ordering", execution_defaults.node_ordering)),
        mpi_distribute_by_nodes=bool(execution_data.get("mpi_distribute_by_nodes", execution_defaults.mpi_distribute_by_nodes)),
        constitutive_mode=str(execution_data.get("constitutive_mode", execution_defaults.constitutive_mode)),
        tangent_kernel=str(execution_data.get("tangent_kernel", execution_defaults.tangent_kernel)),
        store_step_u=bool(execution_data.get("store_step_u", execution_defaults.store_step_u)),
    )
    continuation = ContinuationConfig(
        profile=str(continuation_data.get("profile", "")),
        profile_description=str(continuation_data.get("profile_description", "")),
        algorithm=str(continuation_data.get("algorithm", continuation_data.get("method", ""))),
        method=str(continuation_data.get("method", "indirect")),
        predictor=str(continuation_data.get("predictor", "secant")),
        omega_step_controller=str(continuation_data.get("omega_step_controller", "classic")),
        secant_correction_mode=str(continuation_data.get("secant_correction_mode", "none")),
        first_newton_warm_start_mode=str(continuation_data.get("first_newton_warm_start_mode", "none")),
        lambda_init=float(continuation_data.get("lambda_init", 1.0)),
        d_lambda_init=float(continuation_data.get("d_lambda_init", 0.1)),
        d_lambda_min=float(continuation_data.get("d_lambda_min", 1e-3)),
        d_lambda_diff_scaled_min=float(continuation_data.get("d_lambda_diff_scaled_min", 1e-3)),
        lambda_ell=float(continuation_data.get("lambda_ell", 1.0)),
        omega_max=float(continuation_data.get("omega_max", 1.2e7)),
        step_max=int(continuation_data.get("step_max", 100)),
        d_omega_ini_scale=float(continuation_data.get("d_omega_ini_scale", 0.2)),
        d_t_min=float(continuation_data.get("d_t_min", 1e-3)),
        omega_no_increase_newton_threshold=(
            None
            if continuation_data.get("omega_no_increase_newton_threshold") is None
            else int(continuation_data.get("omega_no_increase_newton_threshold"))
        ),
        omega_half_newton_threshold=(
            None
            if continuation_data.get("omega_half_newton_threshold") is None
            else int(continuation_data.get("omega_half_newton_threshold"))
        ),
        omega_target_newton_iterations=(
            None
            if continuation_data.get("omega_target_newton_iterations") is None
            else float(continuation_data.get("omega_target_newton_iterations"))
        ),
        omega_adapt_min_scale=(
            None if continuation_data.get("omega_adapt_min_scale") is None else float(continuation_data.get("omega_adapt_min_scale"))
        ),
        omega_adapt_max_scale=(
            None if continuation_data.get("omega_adapt_max_scale") is None else float(continuation_data.get("omega_adapt_max_scale"))
        ),
        omega_hard_newton_threshold=(
            None
            if continuation_data.get("omega_hard_newton_threshold") is None
            else int(continuation_data.get("omega_hard_newton_threshold"))
        ),
        omega_hard_linear_threshold=(
            None
            if continuation_data.get("omega_hard_linear_threshold") is None
            else int(continuation_data.get("omega_hard_linear_threshold"))
        ),
        omega_efficiency_floor=(
            None if continuation_data.get("omega_efficiency_floor") is None else float(continuation_data.get("omega_efficiency_floor"))
        ),
        omega_efficiency_drop_ratio=(
            None
            if continuation_data.get("omega_efficiency_drop_ratio") is None
            else float(continuation_data.get("omega_efficiency_drop_ratio"))
        ),
        omega_efficiency_window=int(continuation_data.get("omega_efficiency_window", 3)),
        omega_hard_shrink_scale=(
            None
            if continuation_data.get("omega_hard_shrink_scale") is None
            else float(continuation_data.get("omega_hard_shrink_scale"))
        ),
        step_length_cap_mode=str(continuation_data.get("step_length_cap_mode", "none")),
        step_length_cap_factor=float(continuation_data.get("step_length_cap_factor", 1.0)),
        init_newton_stopping_criterion=(
            None
            if continuation_data.get("init_newton_stopping_criterion") is None
            else str(continuation_data.get("init_newton_stopping_criterion"))
        ),
        init_newton_stopping_tol=(
            None
            if continuation_data.get("init_newton_stopping_tol") is None
            else float(continuation_data.get("init_newton_stopping_tol"))
        ),
        fine_newton_stopping_criterion=(
            None
            if continuation_data.get("fine_newton_stopping_criterion") is None
            else str(continuation_data.get("fine_newton_stopping_criterion"))
        ),
        fine_newton_stopping_tol=(
            None
            if continuation_data.get("fine_newton_stopping_tol") is None
            else float(continuation_data.get("fine_newton_stopping_tol"))
        ),
        fine_switch_mode=str(continuation_data.get("fine_switch_mode", "none")),
        fine_switch_distance_factor=float(continuation_data.get("fine_switch_distance_factor", 2.0)),
    )
    newton = NewtonConfig(
        profile=str(newton_data.get("profile", "")),
        profile_description=str(newton_data.get("profile_description", "")),
        algorithm=str(newton_data.get("algorithm", "")),
        it_max=int(newton_data.get("it_max", 200)),
        it_damp_max=int(newton_data.get("it_damp_max", 10)),
        tol=float(newton_data.get("tol", 1e-4)),
        r_min=float(newton_data.get("r_min", 1e-4)),
        stopping_criterion=str(newton_data.get("stopping_criterion", "relative_residual")),
        stopping_tol=(
            None if newton_data.get("stopping_tol") is None else float(newton_data.get("stopping_tol"))
        ),
        line_search=str(newton_data.get("line_search", "alg5")),
        armijo_alpha0=float(newton_data.get("armijo_alpha0", 1.0)),
        armijo_c1=float(newton_data.get("armijo_c1", 1.0e-4)),
        armijo_shrink=float(newton_data.get("armijo_shrink", 0.5)),
        armijo_max_ls=(
            None if newton_data.get("armijo_max_ls") is None else int(newton_data.get("armijo_max_ls"))
        ),
        armijo_rescale_trial_to_omega=bool(newton_data.get("armijo_rescale_trial_to_omega", True)),
        armijo_fallback_to_alg5=bool(newton_data.get("armijo_fallback_to_alg5", True)),
    )
    linear_solver = LinearSolverConfig(
        profile=str(linear_data.get("profile", "")),
        profile_description=str(linear_data.get("profile_description", "")),
        resolved_world_size=int(linear_data.get("resolved_world_size", 1)),
        algorithm=str(linear_data.get("algorithm", "")),
        solver_type=str(linear_data.get("solver_type", "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")),
        ksp_type=str(linear_data.get("ksp_type", "fgmres")),
        norm_type=str(linear_data.get("norm_type", "unpreconditioned")),
        deflation=bool(linear_data.get("deflation", True)),
        deflation_solver=(
            None if linear_data.get("deflation_solver") is None else str(linear_data.get("deflation_solver"))
        ),
        tolerance=float(linear_data.get("tolerance", 1e-1)),
        max_iterations=int(linear_data.get("max_iterations", 100)),
        deflation_basis_tolerance=float(linear_data.get("deflation_basis_tolerance", 1e-3)),
        verbose=bool(linear_data.get("verbose", False)),
        threads=int(linear_data.get("threads", 16)),
        print_level=int(linear_data.get("print_level", 0)),
        use_as_preconditioner=bool(linear_data.get("use_as_preconditioner", True)),
        factor_solver_type=(
            None if linear_data.get("factor_solver_type") is None else str(linear_data.get("factor_solver_type"))
        ),
        pc_backend=str(linear_data.get("pc_backend", "hypre")),
        pmg_profile=(
            None if linear_data.get("pmg_profile") is None else str(linear_data.get("pmg_profile"))
        ),
        pmg_coarse_mesh_variant=(
            None if linear_data.get("pmg_coarse_mesh_variant") is None else str(linear_data.get("pmg_coarse_mesh_variant"))
        ),
        pmg_fine_hierarchy_mode=str(linear_data.get("pmg_fine_hierarchy_mode", "default")),
        pmg_shell_p2_active_ranks=(
            None if linear_data.get("pmg_shell_p2_active_ranks") is None else int(linear_data.get("pmg_shell_p2_active_ranks"))
        ),
        pmg_shell_p1_active_ranks=(
            None if linear_data.get("pmg_shell_p1_active_ranks") is None else int(linear_data.get("pmg_shell_p1_active_ranks"))
        ),
        pmg_shell_p2_rank_policy=(
            None if linear_data.get("pmg_shell_p2_rank_policy") is None else str(linear_data.get("pmg_shell_p2_rank_policy"))
        ),
        pmg_shell_p1_rank_policy=(
            None if linear_data.get("pmg_shell_p1_rank_policy") is None else str(linear_data.get("pmg_shell_p1_rank_policy"))
        ),
        pmg_rank_policy=(
            None if linear_data.get("pmg_rank_policy") is None else str(linear_data.get("pmg_rank_policy"))
        ),
        pmg_apply_backend=(
            None if linear_data.get("pmg_apply_backend") is None else str(linear_data.get("pmg_apply_backend"))
        ),
        pmg_coarse_pc_type=(
            None if linear_data.get("pmg_coarse_pc_type") is None else str(linear_data.get("pmg_coarse_pc_type"))
        ),
        pmg_coarse_lu_max_dofs=(
            None if linear_data.get("pmg_coarse_lu_max_dofs") is None else int(linear_data.get("pmg_coarse_lu_max_dofs"))
        ),
        pmg_coarse_redundant_group_size=(
            None
            if linear_data.get("pmg_coarse_redundant_group_size") is None
            else int(linear_data.get("pmg_coarse_redundant_group_size"))
        ),
        pmg_coarse_gamg_aggressive_square_graph=(
            None
            if linear_data.get("pmg_coarse_gamg_aggressive_square_graph") is None
            else bool(linear_data.get("pmg_coarse_gamg_aggressive_square_graph"))
        ),
        pmg_coarse_telescope_active_ranks=(
            None
            if linear_data.get("pmg_coarse_telescope_active_ranks") is None
            else int(linear_data.get("pmg_coarse_telescope_active_ranks"))
        ),
        pmg_coarse_telescope_subcomm_type=(
            None
            if linear_data.get("pmg_coarse_telescope_subcomm_type") is None
            else str(linear_data.get("pmg_coarse_telescope_subcomm_type"))
        ),
        pmg_coarse_telescope_ksp_type=(
            None
            if linear_data.get("pmg_coarse_telescope_ksp_type") is None
            else str(linear_data.get("pmg_coarse_telescope_ksp_type"))
        ),
        pmg_coarse_telescope_ksp_rtol=(
            None
            if linear_data.get("pmg_coarse_telescope_ksp_rtol") is None
            else float(linear_data.get("pmg_coarse_telescope_ksp_rtol"))
        ),
        pmg_coarse_telescope_ksp_max_it=(
            None
            if linear_data.get("pmg_coarse_telescope_ksp_max_it") is None
            else int(linear_data.get("pmg_coarse_telescope_ksp_max_it"))
        ),
        pmg_coarse_telescope_pc_type=(
            None
            if linear_data.get("pmg_coarse_telescope_pc_type") is None
            else str(linear_data.get("pmg_coarse_telescope_pc_type"))
        ),
        pmg_p2_telescope_active_ranks=(
            None
            if linear_data.get("pmg_p2_telescope_active_ranks") is None
            else int(linear_data.get("pmg_p2_telescope_active_ranks"))
        ),
        pmg_p2_telescope_subcomm_type=(
            None
            if linear_data.get("pmg_p2_telescope_subcomm_type") is None
            else str(linear_data.get("pmg_p2_telescope_subcomm_type"))
        ),
        pmg_p2_telescope_ksp_type=(
            None
            if linear_data.get("pmg_p2_telescope_ksp_type") is None
            else str(linear_data.get("pmg_p2_telescope_ksp_type"))
        ),
        pmg_p2_telescope_ksp_rtol=(
            None
            if linear_data.get("pmg_p2_telescope_ksp_rtol") is None
            else float(linear_data.get("pmg_p2_telescope_ksp_rtol"))
        ),
        pmg_p2_telescope_ksp_max_it=(
            None
            if linear_data.get("pmg_p2_telescope_ksp_max_it") is None
            else int(linear_data.get("pmg_p2_telescope_ksp_max_it"))
        ),
        pmg_p2_telescope_pc_type=(
            None
            if linear_data.get("pmg_p2_telescope_pc_type") is None
            else str(linear_data.get("pmg_p2_telescope_pc_type"))
        ),
        pmg_smoother_ksp_type=(
            None if linear_data.get("pmg_smoother_ksp_type") is None else str(linear_data.get("pmg_smoother_ksp_type"))
        ),
        pmg_smoother_pc_type=(
            None if linear_data.get("pmg_smoother_pc_type") is None else str(linear_data.get("pmg_smoother_pc_type"))
        ),
        pmg_smoother_max_it=(
            None if linear_data.get("pmg_smoother_max_it") is None else int(linear_data.get("pmg_smoother_max_it"))
        ),
        pmg_shell_subcomm_type=(
            None if linear_data.get("pmg_shell_subcomm_type") is None else str(linear_data.get("pmg_shell_subcomm_type"))
        ),
        pmg_shell_fine_ksp_max_it=(
            None if linear_data.get("pmg_shell_fine_ksp_max_it") is None else int(linear_data.get("pmg_shell_fine_ksp_max_it"))
        ),
        pmg_shell_p2_ksp_max_it=(
            None if linear_data.get("pmg_shell_p2_ksp_max_it") is None else int(linear_data.get("pmg_shell_p2_ksp_max_it"))
        ),
        pmg_shell_p1_pc_type=(
            None if linear_data.get("pmg_shell_p1_pc_type") is None else str(linear_data.get("pmg_shell_p1_pc_type"))
        ),
        pmg_shell_p1_redundant_number=(
            None if linear_data.get("pmg_shell_p1_redundant_number") is None else int(linear_data.get("pmg_shell_p1_redundant_number"))
        ),
        pmg_shell_p1_redundant_ksp_type=(
            None if linear_data.get("pmg_shell_p1_redundant_ksp_type") is None else str(linear_data.get("pmg_shell_p1_redundant_ksp_type"))
        ),
        pmg_shell_p1_redundant_ksp_rtol=(
            None if linear_data.get("pmg_shell_p1_redundant_ksp_rtol") is None else float(linear_data.get("pmg_shell_p1_redundant_ksp_rtol"))
        ),
        pmg_shell_p1_redundant_ksp_max_it=(
            None if linear_data.get("pmg_shell_p1_redundant_ksp_max_it") is None else int(linear_data.get("pmg_shell_p1_redundant_ksp_max_it"))
        ),
        pmg_shell_p1_redundant_pc_type=(
            None if linear_data.get("pmg_shell_p1_redundant_pc_type") is None else str(linear_data.get("pmg_shell_p1_redundant_pc_type"))
        ),
        max_deflation_basis_vectors=int(linear_data.get("max_deflation_basis_vectors", 48)),
        preconditioner_matrix_source=str(linear_data.get("preconditioner_matrix_source", "tangent")),
        preconditioner_matrix_policy=str(linear_data.get("preconditioner_matrix_policy", "current")),
        preconditioner_rebuild_policy=str(linear_data.get("preconditioner_rebuild_policy", "every_newton")),
        preconditioner_rebuild_interval=int(linear_data.get("preconditioner_rebuild_interval", 1)),
        pc_gamg_process_eq_limit=(
            None if linear_data.get("pc_gamg_process_eq_limit") is None else int(linear_data.get("pc_gamg_process_eq_limit"))
        ),
        pc_gamg_threshold=(
            None if linear_data.get("pc_gamg_threshold") is None else float(linear_data.get("pc_gamg_threshold"))
        ),
        pc_gamg_aggressive_coarsening=(
            None
            if linear_data.get("pc_gamg_aggressive_coarsening") is None
            else int(linear_data.get("pc_gamg_aggressive_coarsening"))
        ),
        pc_gamg_aggressive_square_graph=(
            None
            if linear_data.get("pc_gamg_aggressive_square_graph") is None
            else bool(linear_data.get("pc_gamg_aggressive_square_graph"))
        ),
        pc_gamg_aggressive_mis_k=(
            None
            if linear_data.get("pc_gamg_aggressive_mis_k") is None
            else int(linear_data.get("pc_gamg_aggressive_mis_k"))
        ),
        pc_hypre_coarsen_type=(
            None if linear_data.get("pc_hypre_coarsen_type") is None else str(linear_data.get("pc_hypre_coarsen_type"))
        ),
        pc_hypre_interp_type=(
            None if linear_data.get("pc_hypre_interp_type") is None else str(linear_data.get("pc_hypre_interp_type"))
        ),
        pc_hypre_strong_threshold=(
            None if linear_data.get("pc_hypre_strong_threshold") is None else float(linear_data.get("pc_hypre_strong_threshold"))
        ),
        pc_hypre_boomeramg_max_iter=int(linear_data.get("pc_hypre_boomeramg_max_iter", 1)),
        pc_hypre_P_max=(
            None if linear_data.get("pc_hypre_P_max") is None else int(linear_data.get("pc_hypre_P_max"))
        ),
        pc_hypre_agg_nl=(
            None if linear_data.get("pc_hypre_agg_nl") is None else int(linear_data.get("pc_hypre_agg_nl"))
        ),
        pc_hypre_nongalerkin_tol=(
            None
            if linear_data.get("pc_hypre_nongalerkin_tol") is None
            else float(linear_data.get("pc_hypre_nongalerkin_tol"))
        ),
        pc_bddc_symmetric=(
            None
            if linear_data.get("pc_bddc_symmetric") is None
            else bool(linear_data.get("pc_bddc_symmetric"))
        ),
        pc_bddc_dirichlet_ksp_type=(
            None
            if linear_data.get("pc_bddc_dirichlet_ksp_type") is None
            else str(linear_data.get("pc_bddc_dirichlet_ksp_type"))
        ),
        pc_bddc_dirichlet_pc_type=(
            None
            if linear_data.get("pc_bddc_dirichlet_pc_type") is None
            else str(linear_data.get("pc_bddc_dirichlet_pc_type"))
        ),
        pc_bddc_neumann_ksp_type=(
            None
            if linear_data.get("pc_bddc_neumann_ksp_type") is None
            else str(linear_data.get("pc_bddc_neumann_ksp_type"))
        ),
        pc_bddc_neumann_pc_type=(
            None
            if linear_data.get("pc_bddc_neumann_pc_type") is None
            else str(linear_data.get("pc_bddc_neumann_pc_type"))
        ),
        pc_bddc_coarse_ksp_type=(
            None
            if linear_data.get("pc_bddc_coarse_ksp_type") is None
            else str(linear_data.get("pc_bddc_coarse_ksp_type"))
        ),
        pc_bddc_coarse_pc_type=(
            None
            if linear_data.get("pc_bddc_coarse_pc_type") is None
            else str(linear_data.get("pc_bddc_coarse_pc_type"))
        ),
        pc_bddc_dirichlet_approximate=(
            None
            if linear_data.get("pc_bddc_dirichlet_approximate") is None
            else bool(linear_data.get("pc_bddc_dirichlet_approximate"))
        ),
        pc_bddc_neumann_approximate=(
            None
            if linear_data.get("pc_bddc_neumann_approximate") is None
            else bool(linear_data.get("pc_bddc_neumann_approximate"))
        ),
        pc_bddc_monolithic=(
            None
            if linear_data.get("pc_bddc_monolithic") is None
            else bool(linear_data.get("pc_bddc_monolithic"))
        ),
        pc_bddc_coarse_redundant_pc_type=(
            None
            if linear_data.get("pc_bddc_coarse_redundant_pc_type") is None
            else str(linear_data.get("pc_bddc_coarse_redundant_pc_type"))
        ),
        pc_bddc_switch_static=(
            None
            if linear_data.get("pc_bddc_switch_static") is None
            else bool(linear_data.get("pc_bddc_switch_static"))
        ),
        pc_bddc_use_deluxe_scaling=(
            None
            if linear_data.get("pc_bddc_use_deluxe_scaling") is None
            else bool(linear_data.get("pc_bddc_use_deluxe_scaling"))
        ),
        pc_bddc_use_vertices=(
            None
            if linear_data.get("pc_bddc_use_vertices") is None
            else bool(linear_data.get("pc_bddc_use_vertices"))
        ),
        pc_bddc_use_edges=(
            None
            if linear_data.get("pc_bddc_use_edges") is None
            else bool(linear_data.get("pc_bddc_use_edges"))
        ),
        pc_bddc_use_faces=(
            None
            if linear_data.get("pc_bddc_use_faces") is None
            else bool(linear_data.get("pc_bddc_use_faces"))
        ),
        pc_bddc_use_change_of_basis=(
            None
            if linear_data.get("pc_bddc_use_change_of_basis") is None
            else bool(linear_data.get("pc_bddc_use_change_of_basis"))
        ),
        pc_bddc_use_change_on_faces=(
            None
            if linear_data.get("pc_bddc_use_change_on_faces") is None
            else bool(linear_data.get("pc_bddc_use_change_on_faces"))
        ),
        pc_bddc_check_level=(
            None
            if linear_data.get("pc_bddc_check_level") is None
            else int(linear_data.get("pc_bddc_check_level"))
        ),
        compiled_outer=bool(linear_data.get("compiled_outer", False)),
        recycle_preconditioner=bool(linear_data.get("recycle_preconditioner", True)),
        pmg_options_file=(
            None if linear_data.get("pmg_options_file") is None else Path(linear_data.get("pmg_options_file"))
        ),
        petsc_opt=[str(value) for value in list(linear_data.get("petsc_opt", []))],
    )
    seepage_profile = str(seepage_data.pop("profile", "darcy-tight" if seepage_enabled else "")).strip()
    if seepage_profile:
        seepage_profile_data = load_seepage_profile(seepage_profile).data
        reject_profile_default_repeats("[seepage]", seepage_data, seepage_profile_data)
        seepage_profile_data.update(seepage_data)
    else:
        seepage_profile_data = dict(seepage_data)
    seepage = SeepageConfig(
        profile=str(seepage_profile_data.get("profile", seepage_profile)),
        profile_description=str(seepage_profile_data.get("profile_description", "")),
        linear_tolerance=float(seepage_profile_data.get("linear_tolerance", 1e-10)),
        linear_max_iter=int(seepage_profile_data.get("linear_max_iter", 500)),
        nonlinear_max_iter=int(seepage_profile_data.get("nonlinear_max_iter", 50)),
        extra=_resolve_section_paths(
            config_path,
            {
                k: v
                for k, v in seepage_profile_data.items()
                if k not in {"profile", "profile_description", "linear_tolerance", "linear_max_iter", "nonlinear_max_iter"}
            },
        ),
    )
    export = ExportConfig(
        preset=normalize_output_preset(export_data.get("preset", "standard")),
        write_custom_debug_bundle=bool(export_data.get("write_custom_debug_bundle", True)),
        write_history_json=bool(export_data.get("write_history_json", True)),
        write_solution_vtu=bool(export_data.get("write_solution_vtu", True)),
        custom_debug_name=str(export_data.get("custom_debug_name", "run_debug.h5")),
        history_name=str(export_data.get("history_name", "continuation_history.json")),
        solution_name=str(export_data.get("solution_name", "final_solution.vtu")),
    )

    return RunCaseConfig(
        problem=problem,
        execution=execution,
        continuation=continuation,
        newton=newton,
        linear_solver=linear_solver,
        seepage=seepage,
        export=export,
        geometry=geometry_data,
    ).validate()
