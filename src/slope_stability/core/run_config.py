"""General TOML configuration loading for config-driven case execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import tomllib

from .elements import validate_supported_elem_type
from ..problem_assets import load_material_rows_for_path


TomlValue = str | int | float | bool | list[Any] | dict[str, Any]


@dataclass(frozen=True)
class MaterialConfig:
    name: str
    c0: float
    phi: float
    psi: float
    young: float
    poisson: float
    gamma_sat: float
    gamma_unsat: float
    hydraulic_conductivity: float | None = None

    def as_row(self) -> list[float]:
        return [
            float(self.c0),
            float(self.phi),
            float(self.psi),
            float(self.young),
            float(self.poisson),
            float(self.gamma_sat),
            float(self.gamma_unsat),
        ]


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    case: str
    analysis: str = "ssr"
    dimension: int = 3
    variant: str = "hetero"
    elem_type: str = "P2"
    davis_type: str = "B"
    seepage: bool = False
    mesh_path: Path | None = None
    mesh_boundary_type: int = 0


@dataclass(frozen=True)
class ExecutionConfig:
    node_ordering: str = "block_metis"
    mpi_distribute_by_nodes: bool = True
    constitutive_mode: str = "overlap"
    tangent_kernel: str = "rows"


@dataclass(frozen=True)
class NewtonConfig:
    it_max: int = 200
    it_damp_max: int = 10
    tol: float = 1e-4
    r_min: float = 1e-4
    stopping_criterion: str = "relative_residual"
    stopping_tol: float | None = None


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


@dataclass(frozen=True)
class SeepageConfig:
    linear_tolerance: float = 1e-10
    linear_max_iter: int = 500
    nonlinear_max_iter: int = 50
    water_unit_weight: float = 9.81
    conductivity: tuple[float, ...] = ()
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
    materials: tuple[MaterialConfig, ...] = ()
    geometry: dict[str, TomlValue] = field(default_factory=dict)
    case_data: dict[str, TomlValue] = field(default_factory=dict)

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
        if not self.problem.case:
            raise ValueError("[problem].case must be set.")
        if self.problem.analysis.lower() not in {"ssr", "ll", "seepage"}:
            raise ValueError(f"Unsupported analysis {self.problem.analysis!r}.")
        if str(self.newton.stopping_criterion).strip().lower() not in valid_stopping_criteria:
            raise ValueError(
                "The newton stopping_criterion must be relative_residual, relative_correction, or absolute_delta_lambda."
            )
        for field_name in ("init_newton_stopping_criterion", "fine_newton_stopping_criterion"):
            value = getattr(self.continuation, field_name)
            if value is not None and str(value).strip().lower() not in valid_stopping_criteria:
                raise ValueError(
                    f"The continuation {field_name} must be relative_residual, relative_correction, or absolute_delta_lambda."
                )
        validate_supported_elem_type(self.problem.dimension, self.problem.elem_type)
        if self.problem.analysis.lower() != "seepage" and not self.material_rows():
            raise ValueError("At least one [[materials]] entry is required for non-seepage cases.")
        return self

    def material_rows(self) -> list[list[float]]:
        if self.materials:
            return [m.as_row() for m in self.materials]
        if self.problem.mesh_path is None:
            return []
        rows = load_material_rows_for_path(self.problem.mesh_path)
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


def _load_materials(data: dict[str, Any]) -> tuple[MaterialConfig, ...]:
    raw = data.get("materials", [])
    materials = []
    for item in raw:
        materials.append(
            MaterialConfig(
                name=str(item["name"]),
                c0=float(item["c0"]),
                phi=float(item["phi"]),
                psi=float(item["psi"]),
                young=float(item["young"]),
                poisson=float(item["poisson"]),
                gamma_sat=float(item["gamma_sat"]),
                gamma_unsat=float(item["gamma_unsat"]),
                hydraulic_conductivity=(
                    None if item.get("hydraulic_conductivity") is None else float(item["hydraulic_conductivity"])
                ),
            )
        )
    return tuple(materials)


def load_run_case_config(path: str | Path) -> RunCaseConfig:
    config_path = Path(path).resolve()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    problem_data = dict(data.get("problem", {}))
    execution_data = dict(data.get("execution", {}))
    continuation_data = dict(data.get("continuation", {}))
    newton_data = dict(data.get("newton", {}))
    linear_data = dict(data.get("linear_solver", {}))
    seepage_data = dict(data.get("seepage", {}))
    export_data = dict(data.get("export", {}))
    geometry_data = _resolve_section_paths(config_path, dict(data.get("geometry", {})))
    case_data = _resolve_section_paths(config_path, dict(data.get("case_data", {})))

    mesh_path = problem_data.get("mesh_path")
    problem = ProblemConfig(
        name=str(problem_data.get("name", "case")),
        case=str(problem_data.get("case", "")),
        analysis=str(problem_data.get("analysis", "ssr")),
        dimension=int(problem_data.get("dimension", 3)),
        variant=str(problem_data.get("variant", "hetero")),
        elem_type=str(problem_data.get("elem_type", "P2")),
        davis_type=str(problem_data.get("davis_type", "B")),
        seepage=bool(problem_data.get("seepage", False)),
        mesh_path=None if mesh_path is None else _resolve_path(config_path, str(mesh_path)),
        mesh_boundary_type=int(problem_data.get("mesh_boundary_type", 0)),
    )
    execution = ExecutionConfig(
        node_ordering=str(execution_data.get("node_ordering", "block_metis")),
        mpi_distribute_by_nodes=bool(execution_data.get("mpi_distribute_by_nodes", True)),
        constitutive_mode=str(execution_data.get("constitutive_mode", "overlap")),
        tangent_kernel=str(execution_data.get("tangent_kernel", "rows")),
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
    )
    conductivity = seepage_data.get("conductivity", ())
    if isinstance(conductivity, (int, float)):
        conductivity = (float(conductivity),)
    elif isinstance(conductivity, list):
        conductivity = tuple(float(v) for v in conductivity)
    elif isinstance(conductivity, tuple):
        conductivity = tuple(float(v) for v in conductivity)
    else:
        conductivity = ()
    seepage = SeepageConfig(
        linear_tolerance=float(seepage_data.get("linear_tolerance", 1e-10)),
        linear_max_iter=int(seepage_data.get("linear_max_iter", 500)),
        nonlinear_max_iter=int(seepage_data.get("nonlinear_max_iter", 50)),
        water_unit_weight=float(seepage_data.get("water_unit_weight", 9.81)),
        conductivity=tuple(conductivity),
        extra=_resolve_section_paths(
            config_path,
            {
                k: v
                for k, v in seepage_data.items()
                if k not in {"linear_tolerance", "linear_max_iter", "nonlinear_max_iter", "water_unit_weight", "conductivity"}
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
        materials=_load_materials(data),
        geometry=geometry_data,
        case_data=case_data,
    ).validate()
