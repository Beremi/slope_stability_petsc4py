from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import os


ENGINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPTIONS_FILE = ENGINE_ROOT / "configs" / "petsc" / "pmg_shell_baseline.opts"


def _bool(value: bool) -> str:
    return "true" if value else "false"


@dataclass(slots=True)
class PmgOptions:
    """Scalable shell PMG profile used by the maintained C baseline."""

    options_file: Path = DEFAULT_OPTIONS_FILE
    apply_backend: str = "shell_vcycle"
    coarse_pc_type: str = "gamg"
    coarse_lu_max_dofs: int = 50000
    coarse_redundant_group_size: int = 0
    coarse_gamg_aggressive_square_graph: bool = False
    coarse_telescope_active_ranks: int = 0
    coarse_telescope_subcomm_type: str = "interlaced"
    coarse_telescope_ksp_type: str = "fgmres"
    coarse_telescope_ksp_rtol: float = 1.0e-3
    coarse_telescope_ksp_max_it: int = 5
    coarse_telescope_pc_type: str = "gamg"
    p2_telescope_active_ranks: int = 0
    p2_telescope_subcomm_type: str = "interlaced"
    p2_telescope_ksp_type: str = "fgmres"
    p2_telescope_ksp_rtol: float = 1.0e-3
    p2_telescope_ksp_max_it: int = 50
    p2_telescope_pc_type: str = "jacobi"
    smoother_ksp_type: str = "chebyshev"
    smoother_pc_type: str = "jacobi"
    smoother_max_it: int = 2
    p2_active_ranks: int = 0
    p1_active_ranks: int = 0
    subcomm_type: str = "interlaced"
    fine_ksp_max_it: int = 5
    p2_ksp_max_it: int = 10
    p1_pc_type: str | None = None
    p1_redundant_number: int | None = None
    p1_redundant_ksp_type: str | None = None
    p1_redundant_ksp_rtol: float | None = None
    p1_redundant_ksp_max_it: int | None = None
    p1_redundant_pc_type: str | None = None

    @classmethod
    def current_baseline(cls) -> "PmgOptions":
        world = _world_size()
        return cls(p2_active_ranks=min(world, 64), p1_active_ranks=max(1, min(world, 64, world // 2 if world > 1 else 1)))

    def option_tokens(self) -> list[str]:
        tokens: list[str] = []
        for key, value in (
            ("-pmg_apply_backend", self.apply_backend),
            ("-pmg_coarse_pc_type", self.coarse_pc_type),
            ("-pmg_coarse_lu_max_dofs", self.coarse_lu_max_dofs),
            ("-pmg_coarse_redundant_group_size", self.coarse_redundant_group_size),
            ("-pmg_coarse_gamg_aggressive_square_graph", self.coarse_gamg_aggressive_square_graph),
            ("-pmg_coarse_telescope_active_ranks", self.coarse_telescope_active_ranks),
            ("-pmg_coarse_telescope_subcomm_type", self.coarse_telescope_subcomm_type),
            ("-pmg_coarse_telescope_ksp_type", self.coarse_telescope_ksp_type),
            ("-pmg_coarse_telescope_ksp_rtol", self.coarse_telescope_ksp_rtol),
            ("-pmg_coarse_telescope_ksp_max_it", self.coarse_telescope_ksp_max_it),
            ("-pmg_coarse_telescope_pc_type", self.coarse_telescope_pc_type),
            ("-pmg_p2_telescope_active_ranks", self.p2_telescope_active_ranks),
            ("-pmg_p2_telescope_subcomm_type", self.p2_telescope_subcomm_type),
            ("-pmg_p2_telescope_ksp_type", self.p2_telescope_ksp_type),
            ("-pmg_p2_telescope_ksp_rtol", self.p2_telescope_ksp_rtol),
            ("-pmg_p2_telescope_ksp_max_it", self.p2_telescope_ksp_max_it),
            ("-pmg_p2_telescope_pc_type", self.p2_telescope_pc_type),
            ("-pmg_smoother_ksp_type", self.smoother_ksp_type),
            ("-pmg_smoother_pc_type", self.smoother_pc_type),
            ("-pmg_smoother_max_it", self.smoother_max_it),
            ("-pmg_shell_p2_active_ranks", self.p2_active_ranks),
            ("-pmg_shell_p1_active_ranks", self.p1_active_ranks),
            ("-pmg_shell_subcomm_type", self.subcomm_type),
            ("-pmg_shell_fine_ksp_max_it", self.fine_ksp_max_it),
            ("-pmg_shell_p2_ksp_max_it", self.p2_ksp_max_it),
            ("-pmg_shell_p1_pc_type", self.p1_pc_type),
            ("-pmg_shell_p1_pc_redundant_number", self.p1_redundant_number),
            ("-pmg_shell_p1_redundant_ksp_type", self.p1_redundant_ksp_type),
            ("-pmg_shell_p1_redundant_ksp_rtol", self.p1_redundant_ksp_rtol),
            ("-pmg_shell_p1_redundant_ksp_max_it", self.p1_redundant_ksp_max_it),
            ("-pmg_shell_p1_redundant_pc_type", self.p1_redundant_pc_type),
        ):
            if value is not None:
                token_value = _bool(value) if isinstance(value, bool) else str(value)
                tokens.extend([key, token_value])
        return tokens


@dataclass(slots=True)
class LinearOptions:
    rtol: float = 1.0e-1
    max_it: int = 200
    ksp_type: str = "fgmres"
    norm_type: str = "unpreconditioned"
    deflation: bool = True
    deflation_solver: str = "fgmres"

    def option_tokens(self) -> list[str]:
        return [
            "-linear_rtol",
            str(self.rtol),
            "-ksp_max_it",
            str(self.max_it),
            "-ksp_type",
            self.ksp_type,
            "-ksp_norm_type",
            self.norm_type,
            "-deflation",
            _bool(self.deflation),
            "-deflation_solver",
            self.deflation_solver,
        ]


@dataclass(slots=True)
class SsrOptions:
    analysis: str = "ssr"
    continuation_algorithm: str = "indirect"
    continuation_method: str = "indirect"
    newton_algorithm: str = "indirect-ssr"
    linear_algorithm: str = "pmg-deflated"
    omega_max: float = 7.0e6
    lambda_init: float = 1.0
    d_lambda_init: float = 0.1
    d_lambda_min: float = 1.0e-3
    d_lambda_diff_scaled_min: float = 1.0e-3
    lambda_ell: float = 1.0
    d_t_min: float = 1.0e-3
    d_omega_ini_scale: float = 0.2
    continuation_step_max: int = 100
    newton_max_it: int = 200
    newton_rtol: float = 1.0e-4
    newton_stopping_criterion: str = "absolute_delta_lambda"
    newton_stopping_tol: float = 1.0e-4
    init_newton_stopping_criterion: str = "relative_correction"
    init_newton_stopping_tol: float = 1.0e-3
    it_damp_max: int = 10
    r_min: float = 1.0e-4
    damping_min: float = 1.0e-3
    line_search: bool = True
    continuation_predictor: str = "secant"
    omega_step_controller: str = "classic"
    pc_variant: str = "pmg"
    partitioner: str = "parmetis"
    linear: LinearOptions = field(default_factory=LinearOptions)
    pmg: PmgOptions = field(default_factory=PmgOptions.current_baseline)
    petsc_options: list[str] = field(default_factory=list)
    profile_name: str = ""

    @classmethod
    def current_baseline(cls, *, omega_max: float = 7.0e6) -> "SsrOptions":
        return cls(omega_max=omega_max)

    def option_tokens(self) -> list[str]:
        tokens = [
            "-analysis",
            self.analysis,
            "-continuation_algorithm",
            self.continuation_algorithm,
            "-continuation_method",
            self.continuation_method,
            "-newton_algorithm",
            self.newton_algorithm,
            "-linear_algorithm",
            self.linear_algorithm,
            "-omega_max",
            str(self.omega_max),
            "-lambda_init",
            str(self.lambda_init),
            "-d_lambda_init",
            str(self.d_lambda_init),
            "-d_lambda_min",
            str(self.d_lambda_min),
            "-d_lambda_diff_scaled_min",
            str(self.d_lambda_diff_scaled_min),
            "-lambda_ell",
            str(self.lambda_ell),
            "-d_t_min",
            str(self.d_t_min),
            "-d_omega_ini_scale",
            str(self.d_omega_ini_scale),
            "-continuation_step_max",
            str(self.continuation_step_max),
            "-newton_max_it",
            str(self.newton_max_it),
            "-newton_rtol",
            str(self.newton_rtol),
            "-newton_stopping_criterion",
            self.newton_stopping_criterion,
            "-newton_stopping_tol",
            str(self.newton_stopping_tol),
            "-init_newton_stopping_criterion",
            self.init_newton_stopping_criterion,
            "-init_newton_stopping_tol",
            str(self.init_newton_stopping_tol),
            "-it_damp_max",
            str(self.it_damp_max),
            "-r_min",
            str(self.r_min),
            "-damping_min",
            str(self.damping_min),
            "-line_search",
            _bool(self.line_search),
            "-continuation_predictor",
            self.continuation_predictor,
            "-omega_step_controller",
            self.omega_step_controller,
            "-pc_variant",
            self.pc_variant,
            "-petscpartitioner_type",
            self.partitioner,
        ]
        tokens.extend(self.linear.option_tokens())
        if self.pc_variant.lower() == "pmg":
            tokens.extend(self.pmg.option_tokens())
        tokens.extend(self.petsc_options)
        return tokens


def flatten_tokens(chunks: Iterable[Iterable[str]]) -> list[str]:
    tokens: list[str] = []
    for chunk in chunks:
        tokens.extend(str(part) for part in chunk)
    return tokens


def _world_size() -> int:
    for env_name in ("PETSC_SSR_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
        raw = os.environ.get(env_name)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    try:
        from mpi4py import MPI

        return max(1, int(MPI.COMM_WORLD.Get_size()))
    except Exception:
        return 1
