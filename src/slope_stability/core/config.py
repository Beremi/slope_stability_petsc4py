"""Configuration dataclasses and TOML loading for PETSc slope-stability runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


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
class Problem3DConfig:
    name: str = "3d_hetero_ssr"
    variant: str = "hetero"
    elem_type: str = "P2"
    davis_type: str = "B"
    seepage: bool = False
    mesh_path: Path = Path("meshes/3d_hetero_ssr/SSR_hetero_ada_L1.h5")
    materials: tuple[MaterialConfig, ...] = ()


@dataclass(frozen=True)
class ExecutionConfig:
    node_ordering: str = "block_metis"
    mpi_distribute_by_nodes: bool = True
    constitutive_mode: str = "overlap"


@dataclass(frozen=True)
class NewtonConfig:
    it_max: int = 200
    it_damp_max: int = 10
    tol: float = 1e-4
    r_min: float = 1e-4


@dataclass(frozen=True)
class ContinuationConfig:
    method: str = "indirect"
    lambda_init: float = 1.0
    d_lambda_init: float = 0.1
    d_lambda_min: float = 1e-3
    d_lambda_diff_scaled_min: float = 1e-3
    omega_max: float = 1.2e7
    step_max: int = 100
    d_omega_ini_scale: float = 0.2
    d_t_min: float = 1e-3


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
    pc_gamg_process_eq_limit: int | None = None
    pc_gamg_threshold: float | None = None
    pc_hypre_coarsen_type: str | None = "HMIS"
    pc_hypre_interp_type: str | None = "ext+i"
    pc_hypre_strong_threshold: float | None = None
    compiled_outer: bool = False
    recycle_preconditioner: bool = True


@dataclass(frozen=True)
class Run3DSSRConfig:
    problem: Problem3DConfig
    execution: ExecutionConfig = ExecutionConfig()
    continuation: ContinuationConfig = ContinuationConfig()
    newton: NewtonConfig = NewtonConfig()
    linear_solver: LinearSolverConfig = LinearSolverConfig()

    def validate(self) -> "Run3DSSRConfig":
        if self.problem.seepage:
            raise NotImplementedError(
                "Seepage configs are intentionally not wired yet; use this scheme for 3D non-seepage SSR first."
            )
        if self.problem.elem_type.upper() != "P2":
            raise NotImplementedError("Only 3D P2 runs are wired into the current config runner.")
        if self.continuation.method.lower() != "indirect":
            raise NotImplementedError("The config runner currently targets indirect 3D SSR continuation only.")
        if not self.problem.materials:
            raise ValueError("At least one material must be provided in [[materials]].")
        return self

    def to_run_capture_kwargs(self) -> dict:
        self.validate()
        return {
            "mesh_path": self.problem.mesh_path,
            "node_ordering": self.execution.node_ordering,
            "lambda_init": self.continuation.lambda_init,
            "d_lambda_init": self.continuation.d_lambda_init,
            "d_lambda_min": self.continuation.d_lambda_min,
            "d_lambda_diff_scaled_min": self.continuation.d_lambda_diff_scaled_min,
            "omega_max_stop": self.continuation.omega_max,
            "step_max": self.continuation.step_max,
            "it_newt_max": self.newton.it_max,
            "it_damp_max": self.newton.it_damp_max,
            "tol": self.newton.tol,
            "r_min": self.newton.r_min,
            "linear_tolerance": self.linear_solver.tolerance,
            "linear_max_iter": self.linear_solver.max_iterations,
            "solver_type": self.linear_solver.solver_type,
            "factor_solver_type": self.linear_solver.factor_solver_type,
            "mpi_distribute_by_nodes": self.execution.mpi_distribute_by_nodes,
            "pc_gamg_process_eq_limit": self.linear_solver.pc_gamg_process_eq_limit,
            "pc_gamg_threshold": self.linear_solver.pc_gamg_threshold,
            "pc_hypre_coarsen_type": self.linear_solver.pc_hypre_coarsen_type,
            "pc_hypre_interp_type": self.linear_solver.pc_hypre_interp_type,
            "pc_hypre_strong_threshold": self.linear_solver.pc_hypre_strong_threshold,
            "compiled_outer": self.linear_solver.compiled_outer,
            "recycle_preconditioner": self.linear_solver.recycle_preconditioner,
            "constitutive_mode": self.execution.constitutive_mode,
            "elem_type": self.problem.elem_type,
            "davis_type": self.problem.davis_type,
            "material_rows": [m.as_row() for m in self.problem.materials],
        }


def _resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _load_materials(data: dict) -> tuple[MaterialConfig, ...]:
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


def load_run_3d_ssr_config(path: str | Path) -> Run3DSSRConfig:
    config_path = Path(path).resolve()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))

    problem_data = data.get("problem", {})
    execution_data = data.get("execution", {})
    continuation_data = data.get("continuation", {})
    newton_data = data.get("newton", {})
    linear_data = data.get("linear_solver", {})

    problem = Problem3DConfig(
        name=str(problem_data.get("name", "3d_hetero_ssr")),
        variant=str(problem_data.get("variant", "hetero")),
        elem_type=str(problem_data.get("elem_type", "P2")),
        davis_type=str(problem_data.get("davis_type", "B")),
        seepage=bool(problem_data.get("seepage", False)),
        mesh_path=_resolve_path(config_path, str(problem_data.get("mesh_path", "meshes/3d_hetero_ssr/SSR_hetero_ada_L1.h5"))),
        materials=_load_materials(data),
    )

    execution = ExecutionConfig(
        node_ordering=str(execution_data.get("node_ordering", "block_metis")),
        mpi_distribute_by_nodes=bool(execution_data.get("mpi_distribute_by_nodes", True)),
        constitutive_mode=str(execution_data.get("constitutive_mode", "overlap")),
    )

    continuation = ContinuationConfig(
        method=str(continuation_data.get("method", "indirect")),
        lambda_init=float(continuation_data.get("lambda_init", 1.0)),
        d_lambda_init=float(continuation_data.get("d_lambda_init", 0.1)),
        d_lambda_min=float(continuation_data.get("d_lambda_min", 1e-3)),
        d_lambda_diff_scaled_min=float(continuation_data.get("d_lambda_diff_scaled_min", 1e-3)),
        omega_max=float(continuation_data.get("omega_max", 1.2e7)),
        step_max=int(continuation_data.get("step_max", 100)),
        d_omega_ini_scale=float(continuation_data.get("d_omega_ini_scale", 0.2)),
        d_t_min=float(continuation_data.get("d_t_min", 1e-3)),
    )

    newton = NewtonConfig(
        it_max=int(newton_data.get("it_max", 200)),
        it_damp_max=int(newton_data.get("it_damp_max", 10)),
        tol=float(newton_data.get("tol", 1e-4)),
        r_min=float(newton_data.get("r_min", 1e-4)),
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
        pc_gamg_process_eq_limit=(
            None if linear_data.get("pc_gamg_process_eq_limit") is None else int(linear_data.get("pc_gamg_process_eq_limit"))
        ),
        pc_gamg_threshold=(
            None if linear_data.get("pc_gamg_threshold") is None else float(linear_data.get("pc_gamg_threshold"))
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
        compiled_outer=bool(linear_data.get("compiled_outer", False)),
        recycle_preconditioner=bool(linear_data.get("recycle_preconditioner", True)),
    )

    return Run3DSSRConfig(
        problem=problem,
        execution=execution,
        continuation=continuation,
        newton=newton,
        linear_solver=linear_solver,
    ).validate()
