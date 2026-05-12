"""General TOML configuration loading for config-driven case execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from .elements import validate_supported_elem_type
from ..problem_assets import load_material_rows_for_asset
from ..assets import load_problem_asset


TomlValue = str | int | float | bool | list[Any] | dict[str, Any]


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


@dataclass(frozen=True)
class ExecutionConfig:
    node_ordering: str = "block_metis"
    mpi_distribute_by_nodes: bool = True
    constitutive_mode: str = "overlap"
    tangent_kernel: str = "rows"
    store_step_u: bool = True


@dataclass(frozen=True)
class NewtonConfig:
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
    method: str = "indirect"
    predictor: str = "secant"
    omega_step_controller: str = "legacy"
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
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
    tolerance: float = 1e-1
    max_iterations: int = 100
    deflation_basis_tolerance: float = 1e-3
    verbose: bool = False
    threads: int = 16
    print_level: int = 0
    use_as_preconditioner: bool = True
    factor_solver_type: str | None = None
    pc_backend: str | None = "hypre"
    pmg_coarse_mesh_variant: str | None = None
    pmg_fine_hierarchy_mode: str = "default"
    numa_domains_per_node: int = 8
    pmg_numa_partition_mode: str = "rank_metis"
    pmg_smoother_pc_type: str | None = None
    pmg_smoother_gasm_total_subdomains: int | None = None
    pmg_smoother_gasm_grouping: str = "contiguous"
    pmg_smoother_gasm_overlap: int = 1
    pmg_smoother_gasm_type: str = "restrict"
    pmg_smoother_gasm_sub_ksp_type: str = "preonly"
    pmg_smoother_gasm_sub_ksp_max_it: int = 1
    pmg_smoother_gasm_sub_pc_type: str = "jacobi"
    pmg_smoother_gasm_view_subdomains: bool = False
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
    petsc_opt: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeepageConfig:
    linear_tolerance: float = 1e-10
    linear_max_iter: int = 500
    nonlinear_max_iter: int = 50
    extra: dict[str, TomlValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportConfig:
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
        for field_name in ("init_newton_stopping_criterion", "fine_newton_stopping_criterion"):
            value = getattr(self.continuation, field_name)
            if value is not None and str(value).strip().lower() not in valid_stopping_criteria:
                raise ValueError(
                    f"The continuation {field_name} must be relative_residual, relative_correction, or absolute_delta_lambda."
                )
        smoother_pc_type = self.linear_solver.pmg_smoother_pc_type
        gasm_grouping = str(self.linear_solver.pmg_smoother_gasm_grouping).strip().lower()
        if gasm_grouping not in {"contiguous", "numa_coalesced"}:
            raise ValueError(
                "The linear_solver pmg_smoother_gasm_grouping must be 'contiguous' or 'numa_coalesced'."
            )
        if int(self.linear_solver.numa_domains_per_node) <= 0:
            raise ValueError("The linear_solver numa_domains_per_node must be positive.")
        if str(self.linear_solver.pmg_numa_partition_mode).strip().lower() not in {
            "rank_metis",
            "domain_metis_split",
        }:
            raise ValueError(
                "The linear_solver pmg_numa_partition_mode must be 'rank_metis' or 'domain_metis_split'."
            )
        if smoother_pc_type is not None:
            if str(smoother_pc_type).strip().lower() != "gasm":
                raise ValueError("The linear_solver pmg_smoother_pc_type must be 'gasm' when set.")
            if gasm_grouping == "contiguous" and self.linear_solver.pmg_smoother_gasm_total_subdomains is None:
                raise ValueError(
                    "The linear_solver pmg_smoother_gasm_total_subdomains must be set when pmg_smoother_pc_type='gasm'."
                )
            if (
                self.linear_solver.pmg_smoother_gasm_total_subdomains is not None
                and int(self.linear_solver.pmg_smoother_gasm_total_subdomains) <= 0
            ):
                raise ValueError("The linear_solver pmg_smoother_gasm_total_subdomains must be positive.")
            if int(self.linear_solver.pmg_smoother_gasm_overlap) < 0:
                raise ValueError("The linear_solver pmg_smoother_gasm_overlap must be nonnegative.")
            if str(self.linear_solver.pmg_smoother_gasm_type).strip().lower() not in {
                "basic",
                "restrict",
                "interpolate",
                "none",
            }:
                raise ValueError(
                    "The linear_solver pmg_smoother_gasm_type must be basic, restrict, interpolate, or none."
                )
            if int(self.linear_solver.pmg_smoother_gasm_sub_ksp_max_it) <= 0:
                raise ValueError("The linear_solver pmg_smoother_gasm_sub_ksp_max_it must be positive.")
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


def _reject_unknown_fields(section_name: str, data: dict[str, Any], allowed: set[str], message: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{section_name} fields {unknown} are not supported; {message}")


def load_run_case_config(path: str | Path) -> RunCaseConfig:
    config_path = Path(path).resolve()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

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
    _reject_unknown_fields(
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
    _reject_unknown_fields(
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
    _reject_unknown_fields(
        "[seepage]",
        seepage_data,
        {"linear_tolerance", "linear_max_iter", "nonlinear_max_iter"},
        "only numerical seepage controls are allowed here; seepage physics belongs in meshes/<asset>/definition.py.",
    )

    asset_name = str(problem_data.get("asset", "")).strip()
    if not asset_name:
        raise ValueError("[problem].asset must be set.")
    asset = load_problem_asset(asset_name)
    problem = ProblemConfig(
        name=str(problem_data.get("name", "case")),
        case=str(problem_data.get("case", "")),
        analysis=str(problem_data.get("analysis", "ssr")),
        dimension=int(asset.dimension),
        elem_type=str(problem_data.get("elem_type", "P2")),
        davis_type=str(problem_data.get("davis_type", "B")),
        asset=asset_name,
        mesh_variant=None if problem_data.get("mesh_variant") is None else str(problem_data.get("mesh_variant")),
        profile=None if problem_data.get("profile") is None else str(problem_data.get("profile")),
    )
    execution = ExecutionConfig(
        node_ordering=str(execution_data.get("node_ordering", "block_metis")),
        mpi_distribute_by_nodes=bool(execution_data.get("mpi_distribute_by_nodes", True)),
        constitutive_mode=str(execution_data.get("constitutive_mode", "overlap")),
        tangent_kernel=str(execution_data.get("tangent_kernel", "rows")),
        store_step_u=bool(execution_data.get("store_step_u", True)),
    )
    continuation = ContinuationConfig(
        method=str(continuation_data.get("method", "indirect")),
        predictor=str(continuation_data.get("predictor", "secant")),
        omega_step_controller=str(continuation_data.get("omega_step_controller", "legacy")),
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
        solver_type=str(linear_data.get("solver_type", "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE")),
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
        pmg_coarse_mesh_variant=(
            None if linear_data.get("pmg_coarse_mesh_variant") is None else str(linear_data.get("pmg_coarse_mesh_variant"))
        ),
        pmg_fine_hierarchy_mode=str(linear_data.get("pmg_fine_hierarchy_mode", "default")),
        numa_domains_per_node=int(linear_data.get("numa_domains_per_node", 8)),
        pmg_numa_partition_mode=str(linear_data.get("pmg_numa_partition_mode", "rank_metis")),
        pmg_smoother_pc_type=(
            None if linear_data.get("pmg_smoother_pc_type") is None else str(linear_data.get("pmg_smoother_pc_type"))
        ),
        pmg_smoother_gasm_total_subdomains=(
            None
            if linear_data.get("pmg_smoother_gasm_total_subdomains") is None
            else int(linear_data.get("pmg_smoother_gasm_total_subdomains"))
        ),
        pmg_smoother_gasm_grouping=str(linear_data.get("pmg_smoother_gasm_grouping", "contiguous")),
        pmg_smoother_gasm_overlap=int(linear_data.get("pmg_smoother_gasm_overlap", 1)),
        pmg_smoother_gasm_type=str(linear_data.get("pmg_smoother_gasm_type", "restrict")),
        pmg_smoother_gasm_sub_ksp_type=str(
            linear_data.get("pmg_smoother_gasm_sub_ksp_type", "preonly")
        ),
        pmg_smoother_gasm_sub_ksp_max_it=int(linear_data.get("pmg_smoother_gasm_sub_ksp_max_it", 1)),
        pmg_smoother_gasm_sub_pc_type=str(linear_data.get("pmg_smoother_gasm_sub_pc_type", "jacobi")),
        pmg_smoother_gasm_view_subdomains=bool(
            linear_data.get("pmg_smoother_gasm_view_subdomains", False)
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
        petsc_opt=[str(value) for value in list(linear_data.get("petsc_opt", []))],
    )
    seepage = SeepageConfig(
        linear_tolerance=float(seepage_data.get("linear_tolerance", 1e-10)),
        linear_max_iter=int(seepage_data.get("linear_max_iter", 500)),
        nonlinear_max_iter=int(seepage_data.get("nonlinear_max_iter", 50)),
        extra=_resolve_section_paths(
            config_path,
            {
                k: v
                for k, v in seepage_data.items()
                if k not in {"linear_tolerance", "linear_max_iter", "nonlinear_max_iter"}
            },
        ),
    )
    export = ExportConfig(
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
