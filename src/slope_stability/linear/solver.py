"""Solver facade objects used by nonlinear and continuation modules."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from mpi4py import MPI

from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve

try:
    from petsc4py import PETSc
except Exception:  # pragma: no cover - optional when PETSc is unavailable
    PETSc = None

from ..core.config import LinearSolverConfig
from ..fem.distributed_elastic import assemble_owned_elastic_rows_for_comm
from ..utils import (
    bddc_pc_coordinates_from_metadata,
    get_petsc_matrix_metadata,
    global_array_to_petsc_vec,
    local_csr_to_petsc_aij_matrix,
    owned_coo_to_petsc_aij_matrix,
    owned_block_range,
    petsc_vec_to_global_array,
    q_to_free_indices,
    release_petsc_aij_matrix,
    to_petsc_aij_matrix,
    to_scipy_csr_from_petsc,
)
from .collector import IterationCollector
from .deflated_fgmres import dfgmres, dfgmres_matlab_exact, dfgmres_matlab_exact_distributed, dfgmres_matlab_exact_distributed_compiled
from .elasticity import impose_zero_dirichlet_full_system
from .orthogonalize import a_orthogonalize, a_orthogonalize_with_info, a_orthogonalize_with_local_metadata
from .preconditioners import attach_near_nullspace, build_preconditioner, make_near_nullspace_elasticity


PreconditionerFactory = Callable[[object], Callable[[np.ndarray], np.ndarray]]


@dataclass
class PreconditionerDiagnostics:
    pc_backend: str
    preconditioner_matrix_source: str
    preconditioner_matrix_policy: str
    preconditioner_rebuild_policy: str
    preconditioner_rebuild_interval: int
    preconditioner_rebuild_count: int = 0
    preconditioner_reuse_count: int = 0
    preconditioner_age_max: int = 0
    preconditioner_setup_time_last: float = 0.0
    preconditioner_setup_time_total: float = 0.0
    preconditioner_apply_time_last: float = 0.0
    preconditioner_apply_time_total: float = 0.0
    preconditioner_last_rebuild_reason: str = "initial"

    def as_dict(self) -> dict[str, object]:
        return {
            "pc_backend": str(self.pc_backend),
            "preconditioner_matrix_source": str(self.preconditioner_matrix_source),
            "preconditioner_matrix_policy": str(self.preconditioner_matrix_policy),
            "preconditioner_rebuild_policy": str(self.preconditioner_rebuild_policy),
            "preconditioner_rebuild_interval": int(self.preconditioner_rebuild_interval),
            "preconditioner_rebuild_count": int(self.preconditioner_rebuild_count),
            "preconditioner_reuse_count": int(self.preconditioner_reuse_count),
            "preconditioner_age_max": int(self.preconditioner_age_max),
            "preconditioner_setup_time_last": float(self.preconditioner_setup_time_last),
            "preconditioner_setup_time_total": float(self.preconditioner_setup_time_total),
            "preconditioner_apply_time_last": float(self.preconditioner_apply_time_last),
            "preconditioner_apply_time_total": float(self.preconditioner_apply_time_total),
            "preconditioner_last_rebuild_reason": str(self.preconditioner_last_rebuild_reason),
        }


@dataclass
class _PMGPetscHierarchyState:
    hierarchy: object
    prolongations: tuple
    restrictions: tuple
    level_orders: tuple[int, ...]
    level_global_sizes: tuple[int, ...]
    level_owned_ranges: tuple[tuple[int, int], ...]

    def destroy(self) -> None:
        for mat in (*self.prolongations, *self.restrictions):
            if mat is None:
                continue
            release_petsc_aij_matrix(mat)
            mat.destroy()


class _ManualPMGShellPC:
    """Manual three-level V-cycle with Hypre coarse solve."""

    _UNSAFE_HYPRE_VECTOR_KEYS = {
        "pc_hypre_boomeramg_numfunctions",
        "pc_hypre_boomeramg_nodal_coarsen",
        "pc_hypre_boomeramg_nodal_coarsen_diag",
        "pc_hypre_boomeramg_vec_interp_variant",
        "pc_hypre_boomeramg_vec_interp_qmax",
        "pc_hypre_boomeramg_vec_interp_smooth",
    }

    def _vector_hypre_options_requested(self) -> bool:
        if bool(self.solver.preconditioner_options.get("coarse_hypre_full_system", False)):
            return True
        if bool(self.solver.preconditioner_options.get("allow_unsafe_hypre_vector_options", False)):
            return True
        return any(key in self.solver.preconditioner_options for key in self._UNSAFE_HYPRE_VECTOR_KEYS)

    def _use_full_system_hypre_coarse(self) -> bool:
        coarse_pc_type = str(
            self.solver.preconditioner_options.get("mg_coarse_pc_type", PETSc.PC.Type.HYPRE)
        ).strip()
        return coarse_pc_type.lower() == str(PETSc.PC.Type.HYPRE).lower() and self._vector_hypre_options_requested()

    def _allow_unsafe_hypre_vector_options(self) -> bool:
        return self._use_full_system_hypre_coarse()

    def _use_direct_elastic_coarse_operator(self, hierarchy) -> bool:
        source = str(
            self.solver.preconditioner_options.get("manualmg_coarse_operator_source", "auto")
        ).strip().lower()
        if source in {"galerkin", "galerkin_free", "galerkin_full_lift"}:
            return False
        if source in {"elastic", "direct", "direct_elastic", "direct_elastic_full_system"}:
            return True
        if source != "auto":
            return False
        levels = getattr(hierarchy, "levels", ())
        if len(levels) < 2:
            return False
        if not tuple(getattr(hierarchy, "materials", ())):
            return False
        coarse_level = levels[0]
        next_level = levels[1]
        return (
            self._coarse_use_full_system
            and int(getattr(coarse_level, "order", -1)) == 1
            and int(getattr(next_level, "order", -1)) == 1
        )

    @staticmethod
    def _is_mixed_p1_tail_to_p2_hierarchy(hierarchy) -> bool:
        levels = tuple(getattr(hierarchy, "levels", ()))
        if len(levels) < 3:
            return False
        orders = tuple(int(getattr(level, "order", -1)) for level in levels)
        return orders[-1] == 2 and all(order == 1 for order in orders[:-1])

    def _default_smoother_options(self, hierarchy, *, comm_size: int) -> tuple[str, str, int]:
        if comm_size > 1 and self._is_mixed_p1_tail_to_p2_hierarchy(hierarchy):
            return (str(PETSc.KSP.Type.CHEBYSHEV), str(PETSc.PC.Type.JACOBI), 3)
        return (
            str(PETSc.KSP.Type.RICHARDSON),
            str(PETSc.PC.Type.SOR),
            3,
        )

    def __init__(self, solver) -> None:
        self.solver = solver
        self.state: _PMGPetscHierarchyState | None = None
        self.hierarchy = None
        self.A_levels_free: list[object] = []
        self.A_fine = None
        self.A_mid = None
        self.A_coarse = None
        self.A_coarse_free = None
        self.smoothers: list[object | None] = []
        self.smoother_fine = None
        self.smoother_mid = None
        self.coarse_ksp = None
        self._coarse_nsp = None
        self._coarse_nsp_vecs: list[object] = []
        self._coarse_use_full_system = False
        self._coarse_operator_source = "galerkin_free"
        self._level_work: list[dict[str, object]] = []
        self._fine_work: dict[str, object] = {}
        self._mid_work: dict[str, object] = {}
        self._coarse_work: dict[str, object] = {}
        self._stats_total: dict[str, float | int | str] = {
            "manualmg_apply_count": 0,
            "manualmg_fine_pre_smoother_time_total_s": 0.0,
            "manualmg_fine_post_smoother_time_total_s": 0.0,
            "manualmg_mid_pre_smoother_time_total_s": 0.0,
            "manualmg_mid_post_smoother_time_total_s": 0.0,
            "manualmg_restrict_fine_to_mid_time_total_s": 0.0,
            "manualmg_restrict_mid_to_coarse_time_total_s": 0.0,
            "manualmg_prolong_coarse_to_mid_time_total_s": 0.0,
            "manualmg_prolong_mid_to_fine_time_total_s": 0.0,
            "manualmg_fine_residual_time_total_s": 0.0,
            "manualmg_mid_residual_time_total_s": 0.0,
            "manualmg_vector_sum_time_total_s": 0.0,
            "manualmg_coarse_hypre_time_total_s": 0.0,
            "manualmg_fine_smoother_iterations_total": 0,
            "manualmg_mid_smoother_iterations_total": 0,
            "manualmg_coarse_ksp_iterations_total": 0,
            "manualmg_coarse_solve_count": 0,
        }
        self._stats_last: dict[str, float | int | str] = {}

    @staticmethod
    def _copy_vec(dst, src) -> None:
        dst_arr = dst.getArray(readonly=False)
        dst_arr[...] = np.asarray(src.getArray(readonly=True), dtype=np.float64)

    @staticmethod
    def _difference_vec(out, lhs, rhs) -> None:
        out_arr = out.getArray(readonly=False)
        lhs_arr = np.asarray(lhs.getArray(readonly=True), dtype=np.float64)
        rhs_arr = np.asarray(rhs.getArray(readonly=True), dtype=np.float64)
        out_arr[...] = lhs_arr - rhs_arr

    @staticmethod
    def _sum_vec(out, left, right) -> None:
        out_arr = out.getArray(readonly=False)
        left_arr = np.asarray(left.getArray(readonly=True), dtype=np.float64)
        right_arr = np.asarray(right.getArray(readonly=True), dtype=np.float64)
        out_arr[...] = left_arr + right_arr

    def _destroy_dynamic(self) -> None:
        for ksp in (*self.smoothers, self.coarse_ksp):
            if ksp is not None:
                ksp.destroy()
        self.smoothers = []
        self.smoother_fine = None
        self.smoother_mid = None
        self.coarse_ksp = None
        seen_handles: set[int] = set()
        owned_level_mats = tuple(self.A_levels_free[:-1]) if self.A_levels_free else ()
        for mat in (*owned_level_mats, self.A_coarse):
            if mat is not None:
                handle = int(mat.handle)
                if handle in seen_handles:
                    continue
                seen_handles.add(handle)
                release_petsc_aij_matrix(mat)
                mat.destroy()
        self.A_levels_free = []
        self.A_fine = None
        self.A_mid = None
        self.A_coarse = None
        self.A_coarse_free = None
        self._coarse_nsp = None
        self._coarse_nsp_vecs = []
        self._coarse_use_full_system = False
        self._coarse_operator_source = "galerkin_free"
        self._level_work = []
        self._fine_work = {}
        self._mid_work = {}
        self._coarse_work = {}

    def destroy(self, pc=None) -> None:
        self._destroy_dynamic()
        self.state = None
        self.hierarchy = None
        self.A_fine = None
        self._stats_total = {
            "manualmg_apply_count": 0,
            "manualmg_fine_pre_smoother_time_total_s": 0.0,
            "manualmg_fine_post_smoother_time_total_s": 0.0,
            "manualmg_mid_pre_smoother_time_total_s": 0.0,
            "manualmg_mid_post_smoother_time_total_s": 0.0,
            "manualmg_restrict_fine_to_mid_time_total_s": 0.0,
            "manualmg_restrict_mid_to_coarse_time_total_s": 0.0,
            "manualmg_prolong_coarse_to_mid_time_total_s": 0.0,
            "manualmg_prolong_mid_to_fine_time_total_s": 0.0,
            "manualmg_fine_residual_time_total_s": 0.0,
            "manualmg_mid_residual_time_total_s": 0.0,
            "manualmg_vector_sum_time_total_s": 0.0,
            "manualmg_coarse_hypre_time_total_s": 0.0,
            "manualmg_fine_smoother_iterations_total": 0,
            "manualmg_mid_smoother_iterations_total": 0,
            "manualmg_coarse_ksp_iterations_total": 0,
            "manualmg_coarse_solve_count": 0,
        }
        self._stats_last = {}

    def _alloc_work_vectors(self) -> None:
        self._level_work = []
        self._fine_work = {}
        self._mid_work = {}
        self._coarse_work = {}
        for level_idx, A_level in enumerate(self.A_levels_free):
            work = {
                "rhs": A_level.createVecRight(),
                "e": A_level.createVecRight(),
            }
            if level_idx > 0:
                work.update(
                    {
                        "residual": A_level.createVecRight(),
                        "Ae": A_level.createVecRight(),
                        "corr": A_level.createVecRight(),
                    }
                )
            self._level_work.append(work)
        if self.A_levels_free:
            self._coarse_work = {
                "rhs_coarse_free": self._level_work[0]["rhs"],
                "ecoarse_free": self._level_work[0]["e"],
            }
            self.A_coarse_free = self.A_levels_free[0]
        if self.A_fine is not None:
            self._fine_work = self._level_work[-1]
        if len(self._level_work) >= 3:
            self._mid_work = self._level_work[-2]
        if self.A_coarse is not None:
            self._coarse_work.update(
                {
                    "rhs_coarse": self.A_coarse.createVecRight(),
                    "ecoarse": self.A_coarse.createVecRight(),
                }
            )

    def _build_full_system_coarse_matrix(self, A_free, level):
        import scipy.sparse as sp

        lo, hi = tuple(int(v) for v in A_free.getOwnershipRange())
        if (lo, hi) != tuple(int(v) for v in level.owned_free_range):
            raise ValueError(
                f"manualmg coarse free ownership mismatch: matrix {(lo, hi)} vs level {level.owned_free_range}"
            )
        rows, cols, vals = A_free.getValuesCSR()
        local_free = sp.csr_matrix((vals, cols, rows), shape=(hi - lo, A_free.getSize()[1]))
        local_coo = local_free.tocoo()

        total0, total1 = tuple(int(v) for v in level.owned_total_range)
        total_size = int(level.total_size)
        local_n_rows = int(total1 - total0)
        free_to_total = np.asarray(level.freedofs, dtype=np.int64)

        mapped_local_rows = free_to_total[lo:hi][np.asarray(local_coo.row, dtype=np.int64)] - total0
        mapped_global_cols = free_to_total[np.asarray(local_coo.col, dtype=np.int64)]
        local_full = sp.coo_matrix(
            (
                np.asarray(local_coo.data, dtype=np.float64),
                (np.asarray(mapped_local_rows, dtype=np.int64), np.asarray(mapped_global_cols, dtype=np.int64)),
            ),
            shape=(local_n_rows, total_size),
        ).tocsr()

        free_mask = np.asarray(level.q_mask, dtype=bool).reshape(-1, order="F")
        constrained_local = np.flatnonzero(~free_mask[total0:total1]).astype(np.int64)
        if constrained_local.size:
            identity_rows = sp.coo_matrix(
                (
                    np.ones(constrained_local.size, dtype=np.float64),
                    (constrained_local, constrained_local + total0),
                ),
                shape=(local_n_rows, total_size),
            ).tocsr()
            local_full = local_full + identity_rows

        A_full = local_csr_to_petsc_aij_matrix(
            local_full,
            global_shape=(total_size, total_size),
            comm=A_free.getComm(),
            block_size=int(level.dim),
            local_col_size=int(total1 - total0),
        )
        return A_full

    def _build_direct_elastic_coarse_matrix(self, *, comm, hierarchy):
        coarse_level = hierarchy.coarse_level if hasattr(hierarchy, "coarse_level") else hierarchy.level_p1
        materials = tuple(getattr(hierarchy, "materials", ()))
        if not materials:
            raise ValueError("manualmg direct elastic coarse operator requires hierarchy materials.")
        owned_rows = assemble_owned_elastic_rows_for_comm(
            coarse_level.coord,
            coarse_level.elem,
            coarse_level.q_mask,
            coarse_level.material_identifier,
            list(materials),
            comm,
            elem_type=str(coarse_level.elem_type),
        )
        if tuple(int(v) for v in owned_rows.owned_row_range) != tuple(int(v) for v in coarse_level.owned_total_range):
            raise ValueError(
                "manualmg direct elastic coarse ownership mismatch: "
                f"{tuple(int(v) for v in owned_rows.owned_row_range)} vs {tuple(int(v) for v in coarse_level.owned_total_range)}"
            )
        return local_csr_to_petsc_aij_matrix(
            owned_rows.local_matrix,
            global_shape=(int(coarse_level.total_size), int(coarse_level.total_size)),
            comm=comm,
            block_size=int(coarse_level.dim),
            local_col_size=int(coarse_level.owned_total_range[1] - coarse_level.owned_total_range[0]),
        )

    def _copy_coarse_free_to_full(self, dst_full, src_free, level) -> None:
        total0, _ = tuple(int(v) for v in level.owned_total_range)
        lo, hi = tuple(int(v) for v in level.owned_free_range)
        dst_arr = dst_full.getArray(readonly=False)
        dst_arr[...] = 0.0
        if hi <= lo:
            return
        src_arr = np.asarray(src_free.getArray(readonly=True), dtype=np.float64)
        owned_total = np.asarray(level.freedofs[lo:hi], dtype=np.int64) - total0
        dst_arr[owned_total] = src_arr

    def _copy_coarse_full_to_free(self, dst_free, src_full, level) -> None:
        total0, _ = tuple(int(v) for v in level.owned_total_range)
        lo, hi = tuple(int(v) for v in level.owned_free_range)
        dst_arr = dst_free.getArray(readonly=False)
        if hi <= lo:
            dst_arr[...] = 0.0
            return
        src_arr = np.asarray(src_full.getArray(readonly=True), dtype=np.float64)
        owned_total = np.asarray(level.freedofs[lo:hi], dtype=np.int64) - total0
        dst_arr[...] = src_arr[owned_total]

    def _build_smoother(self, A, *, prefix: str, hierarchy):
        default_ksp_type, default_pc_type, default_max_it = self._default_smoother_options(
            hierarchy,
            comm_size=int(A.getComm().getSize()),
        )
        ksp = PETSc.KSP().create(comm=A.getComm())
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(A)
        ksp.setType(str(self.solver.preconditioner_options.get("mg_levels_ksp_type", default_ksp_type)))
        ksp.setInitialGuessNonzero(True)
        ksp.setTolerances(
            rtol=0.0,
            atol=0.0,
            max_it=int(self.solver.preconditioner_options.get("mg_levels_ksp_max_it", default_max_it)),
        )
        pc = ksp.getPC()
        pc.setType(str(self.solver.preconditioner_options.get("mg_levels_pc_type", default_pc_type)))
        ksp.setFromOptions()
        ksp.setUp()
        return ksp

    def _configure_coarse_hypre_options(self, *, prefix: str) -> None:
        opts = PETSc.Options()
        allow_unsafe = self._allow_unsafe_hypre_vector_options()
        defaults = {
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_tol": 0.0,
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_P_max": 4,
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_grid_sweeps_all": 1,
            "pc_hypre_boomeramg_cycle_type": "V",
            "pc_hypre_boomeramg_agg_nl": 0,
        }
        for key, value in self.solver._default_hypre_options().items():
            if key in self._UNSAFE_HYPRE_VECTOR_KEYS and not allow_unsafe:
                continue
            defaults.setdefault(key, value)
        for key, value in defaults.items():
            if key not in self.solver.preconditioner_options:
                self.solver._set_petsc_option(opts, f"{prefix}{key}", value)
        for key, value in self.solver.preconditioner_options.items():
            if not key.startswith("pc_hypre_"):
                continue
            if key in self._UNSAFE_HYPRE_VECTOR_KEYS and not allow_unsafe:
                continue
            self.solver._set_petsc_option(opts, f"{prefix}{key}", value)

    def _build_coarse_ksp(self, A, *, prefix: str):
        ksp = PETSc.KSP().create(comm=A.getComm())
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(A)
        coarse_ksp_type = str(self.solver.preconditioner_options.get("mg_coarse_ksp_type", PETSc.KSP.Type.PREONLY))
        coarse_pc_type = str(
            self.solver.preconditioner_options.get("mg_coarse_pc_type", PETSc.PC.Type.HYPRE)
        ).strip()
        ksp.setType(coarse_ksp_type)
        ksp.setInitialGuessNonzero(True)
        coarse_rtol = float(self.solver.preconditioner_options.get("mg_coarse_rtol", 1.0e-10))
        coarse_atol = float(self.solver.preconditioner_options.get("mg_coarse_atol", 0.0))
        coarse_max_it = int(self.solver.preconditioner_options.get("mg_coarse_max_it", 200))
        if (
            coarse_ksp_type.strip().lower() == str(PETSc.KSP.Type.RICHARDSON).lower()
            and coarse_pc_type.lower() == str(PETSc.PC.Type.HYPRE).lower()
        ):
            # PETSc's Richardson path for BoomerAMG reports the actual inner Hypre cycle count.
            coarse_rtol = float(
                self.solver.preconditioner_options.get(
                    "pc_hypre_boomeramg_tol",
                    self.solver.preconditioner_options.get("mg_coarse_rtol", 0.0),
                )
            )
            coarse_max_it = int(self.solver.preconditioner_options.get("mg_coarse_max_it", 1))
        ksp.setTolerances(rtol=coarse_rtol, atol=coarse_atol, max_it=coarse_max_it)
        pc = ksp.getPC()
        pc.setType(coarse_pc_type)
        if coarse_pc_type.lower() == str(PETSc.PC.Type.HYPRE).lower():
            pc.setHYPREType(str(self.solver.preconditioner_options.get("mg_coarse_pc_hypre_type", "boomeramg")))
            self._configure_coarse_hypre_options(prefix=prefix)
        ksp.setFromOptions()
        ksp.setUp()
        return ksp

    def configure(self, *, matrix_ref, state: _PMGPetscHierarchyState, hierarchy) -> None:
        self._destroy_dynamic()
        self.state = state
        self.hierarchy = hierarchy
        levels = tuple(getattr(hierarchy, "levels", ()))
        if len(levels) < 2:
            raise ValueError("manualmg backend requires at least two levels.")
        self.A_levels_free = [None] * len(levels)
        self.A_levels_free[-1] = matrix_ref
        for level_idx in range(len(levels) - 2, -1, -1):
            galerkin = self.A_levels_free[level_idx + 1].PtAP(state.prolongations[level_idx])
            galerkin.assemble()
            self.A_levels_free[level_idx] = galerkin
        self.A_fine = self.A_levels_free[-1]
        self.A_mid = self.A_levels_free[-2] if len(self.A_levels_free) >= 2 else None
        self.A_coarse_free = self.A_levels_free[0]
        coarse_level = hierarchy.coarse_level if hasattr(hierarchy, "coarse_level") else levels[0]
        self._coarse_use_full_system = self._use_full_system_hypre_coarse()
        if self._coarse_use_full_system and self._use_direct_elastic_coarse_operator(hierarchy):
            self.A_coarse = self._build_direct_elastic_coarse_matrix(comm=matrix_ref.getComm(), hierarchy=hierarchy)
            self._coarse_operator_source = "direct_elastic_full_system"
        elif self._coarse_use_full_system:
            self.A_coarse = self._build_full_system_coarse_matrix(self.A_coarse_free, coarse_level)
            self._coarse_operator_source = "galerkin_full_lift"
        else:
            self.A_coarse = self.A_coarse_free
            self._coarse_operator_source = "galerkin_free"
        coarse_basis = make_near_nullspace_elasticity(
            coarse_level.coord,
            q_mask=coarse_level.q_mask,
            center_coordinates=True,
            return_full=bool(self._coarse_use_full_system),
        )
        self.A_coarse, self._coarse_nsp, self._coarse_nsp_vecs = attach_near_nullspace(self.A_coarse, coarse_basis)
        self.smoothers = [None] * len(levels)
        for level_idx in range(1, len(levels)):
            if level_idx == len(levels) - 1:
                prefix = f"{self.solver._options_prefix}manualmg_fine_"
            elif len(levels) == 3 and level_idx == 1:
                prefix = f"{self.solver._options_prefix}manualmg_mid_"
            else:
                prefix = f"{self.solver._options_prefix}manualmg_level{level_idx}_"
            self.smoothers[level_idx] = self._build_smoother(
                self.A_levels_free[level_idx],
                prefix=prefix,
                hierarchy=hierarchy,
            )
        self.smoother_fine = self.smoothers[-1]
        self.smoother_mid = self.smoothers[-2] if len(self.smoothers) >= 3 else None
        self.coarse_ksp = self._build_coarse_ksp(self.A_coarse, prefix=f"{self.solver._options_prefix}manualmg_coarse_")
        self._alloc_work_vectors()

    def diagnostics(self, *, phase: str | None = None, apply_elapsed_s: float | None = None) -> dict[str, object]:
        if self.state is None or self.coarse_ksp is None:
            return {}
        level_count = len(self.state.level_orders)
        payload: dict[str, object] = {
            "manualmg_levels": int(level_count),
            "manualmg_level_orders": [int(v) for v in self.state.level_orders],
            "manualmg_level_global_sizes": [int(v) for v in self.state.level_global_sizes],
            "manualmg_level_owned_ranges": [[int(lo), int(hi)] for lo, hi in self.state.level_owned_ranges],
            "manualmg_transfer_shapes": [
                [int(v) for v in mat.getSize()] for mat in self.state.prolongations
            ],
            "manualmg_restriction_shapes": [
                [int(v) for v in mat.getSize()] for mat in self.state.restrictions
            ],
            "manualmg_coarse_ksp_type": str(self.coarse_ksp.getType()),
            "manualmg_coarse_pc_type": str(self.coarse_ksp.getPC().getType()),
            "manualmg_coarse_iterations": int(self.coarse_ksp.getIterationNumber()),
            "manualmg_coarse_converged_reason": int(self.coarse_ksp.getConvergedReason()),
            "manualmg_coarse_ksp_rtol": float(self.coarse_ksp.getTolerances()[0]),
            "manualmg_coarse_ksp_max_it": int(self.coarse_ksp.getTolerances()[3]),
            "manualmg_coarse_full_system": bool(self._coarse_use_full_system),
            "manualmg_coarse_operator_source": str(self._coarse_operator_source),
            "manualmg_coarse_solve_global_size": int(self.A_coarse.getSize()[0]),
            "manualmg_coarse_block_size": int(self.A_coarse.getBlockSize() or 1),
        }
        if self.A_coarse_free is not None and self.A_coarse_free is not self.A_coarse:
            payload["manualmg_coarse_free_global_size"] = int(self.A_coarse_free.getSize()[0])
        payload.update({str(k): v for k, v in self._stats_total.items()})
        if self._stats_last:
            payload.update({str(k): v for k, v in self._stats_last.items()})
        if str(self.coarse_ksp.getPC().getType()).lower() == str(PETSc.PC.Type.HYPRE).lower():
            payload["manualmg_coarse_hypre_type"] = str(self.coarse_ksp.getPC().getHYPREType())
            coarse_prefix = f"{self.solver._options_prefix}manualmg_coarse_"
            opts = PETSc.Options()
            for key in (
                "pc_hypre_boomeramg_max_iter",
                "pc_hypre_boomeramg_tol",
                "pc_hypre_boomeramg_numfunctions",
                "pc_hypre_boomeramg_nodal_coarsen",
                "pc_hypre_boomeramg_nodal_coarsen_diag",
                "pc_hypre_boomeramg_vec_interp_variant",
                "pc_hypre_boomeramg_vec_interp_qmax",
                "pc_hypre_boomeramg_vec_interp_smooth",
                "pc_hypre_boomeramg_coarsen_type",
                "pc_hypre_boomeramg_interp_type",
                "pc_hypre_boomeramg_P_max",
                "pc_hypre_boomeramg_strong_threshold",
                "pc_hypre_boomeramg_grid_sweeps_all",
                "pc_hypre_boomeramg_cycle_type",
                "pc_hypre_boomeramg_agg_nl",
                "pc_hypre_boomeramg_relax_type_all",
            ):
                try:
                    value = opts.getString(f"{coarse_prefix}{key}")
                except KeyError:
                    value = None
                if value is not None:
                    payload[f"manualmg_coarse_{key}"] = value
            if str(self.coarse_ksp.getType()).strip().lower() == str(PETSc.KSP.Type.RICHARDSON).lower():
                payload["manualmg_coarse_iteration_count_mode"] = "hypre_inner_v_cycles_via_pcapplyrichardson"
            else:
                payload["manualmg_coarse_iteration_count_mode"] = "coarse_ksp_iterations_only"
        for level_idx in range(1, level_count):
            smoother = self.smoothers[level_idx]
            if smoother is None:
                continue
            payload[f"manualmg_level_{level_idx}_ksp_type"] = str(smoother.getType())
            payload[f"manualmg_level_{level_idx}_pc_type"] = str(smoother.getPC().getType())
            payload[f"manualmg_level_{level_idx}_iterations"] = int(smoother.getIterationNumber())
            payload[f"manualmg_level_{level_idx}_converged_reason"] = int(smoother.getConvergedReason())
        if self.smoother_mid is not None:
            payload["manualmg_mid_ksp_type"] = str(self.smoother_mid.getType())
            payload["manualmg_mid_pc_type"] = str(self.smoother_mid.getPC().getType())
            payload["manualmg_mid_iterations"] = int(self.smoother_mid.getIterationNumber())
        if self.smoother_fine is not None:
            payload["manualmg_fine_ksp_type"] = str(self.smoother_fine.getType())
            payload["manualmg_fine_pc_type"] = str(self.smoother_fine.getPC().getType())
            payload["manualmg_fine_iterations"] = int(self.smoother_fine.getIterationNumber())
        if phase is not None:
            payload["manualmg_last_phase"] = str(phase)
        if apply_elapsed_s is not None:
            payload["manualmg_last_pc_apply_time_s"] = float(apply_elapsed_s)
        return payload

    def apply(self, pc, x, y) -> None:
        if self.state is None or self.smoother_fine is None or self.coarse_ksp is None:
            raise RuntimeError("manualmg backend has not been configured")
        level_count = len(self.A_levels_free)
        fine_idx = level_count - 1
        restrictions = self.state.restrictions
        prolongations = self.state.prolongations
        if level_count < 2:
            raise RuntimeError("manualmg backend requires at least two levels")

        fine_pre_s = 0.0
        fine_post_s = 0.0
        mid_pre_s = 0.0
        mid_post_s = 0.0
        fine_resid_s = 0.0
        mid_resid_s = 0.0
        restrict_f2m_s = 0.0
        restrict_m2c_s = 0.0
        prolong_c2m_s = 0.0
        prolong_m2f_s = 0.0
        vec_sum_s = 0.0

        for work in self._level_work:
            work["e"].set(0.0)
        self._copy_vec(self._level_work[fine_idx]["rhs"], x)

        for level_idx in range(fine_idx, 0, -1):
            work = self._level_work[level_idx]
            rhs = work["rhs"]
            e = work["e"]
            residual = work["residual"]
            A_e = work["Ae"]
            smoother = self.smoothers[level_idx]
            t = perf_counter()
            smoother.solve(rhs, e)
            pre_s = perf_counter() - t
            if level_idx == fine_idx:
                fine_pre_s += pre_s
            else:
                mid_pre_s += pre_s

            t = perf_counter()
            self.A_levels_free[level_idx].mult(e, A_e)
            self._difference_vec(residual, rhs, A_e)
            resid_s = perf_counter() - t
            if level_idx == fine_idx:
                fine_resid_s += resid_s
            else:
                mid_resid_s += resid_s

            t = perf_counter()
            restrictions[level_idx - 1].mult(residual, self._level_work[level_idx - 1]["rhs"])
            restrict_s = perf_counter() - t
            if level_idx == fine_idx:
                restrict_f2m_s += restrict_s
            else:
                restrict_m2c_s += restrict_s

        rhs_coarse_free = self._coarse_work["rhs_coarse_free"]
        ecoarse_free = self._coarse_work["ecoarse_free"]
        rhs_coarse = self._coarse_work.get("rhs_coarse", rhs_coarse_free)
        ecoarse = self._coarse_work.get("ecoarse", ecoarse_free)
        if self._coarse_use_full_system:
            self._copy_coarse_free_to_full(rhs_coarse, rhs_coarse_free, self.hierarchy.coarse_level)
        else:
            self._copy_vec(rhs_coarse, rhs_coarse_free)
        ecoarse.set(0.0)
        t = perf_counter()
        self.coarse_ksp.solve(rhs_coarse, ecoarse)
        coarse_s = perf_counter() - t
        if self._coarse_use_full_system:
            self._copy_coarse_full_to_free(ecoarse_free, ecoarse, self.hierarchy.coarse_level)
        else:
            self._copy_vec(ecoarse_free, ecoarse)

        for level_idx in range(1, level_count):
            work = self._level_work[level_idx]
            corr = work["corr"]
            t = perf_counter()
            prolongations[level_idx - 1].mult(self._level_work[level_idx - 1]["e"], corr)
            prolong_s = perf_counter() - t
            if level_idx == fine_idx:
                prolong_m2f_s += prolong_s
            else:
                prolong_c2m_s += prolong_s
            work["e"].axpy(1.0, corr)
            t = perf_counter()
            self.smoothers[level_idx].solve(work["rhs"], work["e"])
            post_s = perf_counter() - t
            if level_idx == fine_idx:
                fine_post_s += post_s
            else:
                mid_post_s += post_s

        t = perf_counter()
        self._copy_vec(y, self._level_work[fine_idx]["e"])
        vec_sum_s = perf_counter() - t

        self._stats_total["manualmg_apply_count"] = int(self._stats_total["manualmg_apply_count"]) + 1
        self._stats_total["manualmg_fine_pre_smoother_time_total_s"] = float(self._stats_total["manualmg_fine_pre_smoother_time_total_s"]) + float(fine_pre_s)
        self._stats_total["manualmg_fine_post_smoother_time_total_s"] = float(self._stats_total["manualmg_fine_post_smoother_time_total_s"]) + float(fine_post_s)
        self._stats_total["manualmg_mid_pre_smoother_time_total_s"] = float(self._stats_total["manualmg_mid_pre_smoother_time_total_s"]) + float(mid_pre_s)
        self._stats_total["manualmg_mid_post_smoother_time_total_s"] = float(self._stats_total["manualmg_mid_post_smoother_time_total_s"]) + float(mid_post_s)
        self._stats_total["manualmg_restrict_fine_to_mid_time_total_s"] = float(self._stats_total["manualmg_restrict_fine_to_mid_time_total_s"]) + float(restrict_f2m_s)
        self._stats_total["manualmg_restrict_mid_to_coarse_time_total_s"] = float(self._stats_total["manualmg_restrict_mid_to_coarse_time_total_s"]) + float(restrict_m2c_s)
        self._stats_total["manualmg_prolong_coarse_to_mid_time_total_s"] = float(self._stats_total["manualmg_prolong_coarse_to_mid_time_total_s"]) + float(prolong_c2m_s)
        self._stats_total["manualmg_prolong_mid_to_fine_time_total_s"] = float(self._stats_total["manualmg_prolong_mid_to_fine_time_total_s"]) + float(prolong_m2f_s)
        self._stats_total["manualmg_fine_residual_time_total_s"] = float(self._stats_total["manualmg_fine_residual_time_total_s"]) + float(fine_resid_s)
        self._stats_total["manualmg_mid_residual_time_total_s"] = float(self._stats_total["manualmg_mid_residual_time_total_s"]) + float(mid_resid_s)
        self._stats_total["manualmg_vector_sum_time_total_s"] = float(self._stats_total["manualmg_vector_sum_time_total_s"]) + float(vec_sum_s)
        self._stats_total["manualmg_coarse_hypre_time_total_s"] = float(self._stats_total["manualmg_coarse_hypre_time_total_s"]) + float(coarse_s)
        self._stats_total["manualmg_fine_smoother_iterations_total"] = int(
            self._stats_total["manualmg_fine_smoother_iterations_total"]
        ) + 2 * int(self.smoothers[fine_idx].getIterationNumber())
        intermediate_iterations = 0
        for level_idx in range(1, fine_idx):
            intermediate_iterations += 2 * int(self.smoothers[level_idx].getIterationNumber())
        self._stats_total["manualmg_mid_smoother_iterations_total"] = int(
            self._stats_total["manualmg_mid_smoother_iterations_total"]
        ) + intermediate_iterations
        coarse_iters = int(self.coarse_ksp.getIterationNumber())
        self._stats_total["manualmg_coarse_ksp_iterations_total"] = int(self._stats_total["manualmg_coarse_ksp_iterations_total"]) + coarse_iters
        if str(self.coarse_ksp.getType()).strip().lower() == str(PETSc.KSP.Type.RICHARDSON).lower():
            self._stats_total["manualmg_coarse_hypre_inner_iterations_total"] = int(
                self._stats_total.get("manualmg_coarse_hypre_inner_iterations_total", 0)
            ) + coarse_iters
        self._stats_total["manualmg_coarse_solve_count"] = int(self._stats_total["manualmg_coarse_solve_count"]) + 1
        self._stats_last = {
            "manualmg_last_fine_pre_smoother_time_s": float(fine_pre_s),
            "manualmg_last_fine_post_smoother_time_s": float(fine_post_s),
            "manualmg_last_mid_pre_smoother_time_s": float(mid_pre_s),
            "manualmg_last_mid_post_smoother_time_s": float(mid_post_s),
            "manualmg_last_restrict_fine_to_mid_time_s": float(restrict_f2m_s),
            "manualmg_last_restrict_mid_to_coarse_time_s": float(restrict_m2c_s),
            "manualmg_last_prolong_coarse_to_mid_time_s": float(prolong_c2m_s),
            "manualmg_last_prolong_mid_to_fine_time_s": float(prolong_m2f_s),
            "manualmg_last_fine_residual_time_s": float(fine_resid_s),
            "manualmg_last_mid_residual_time_s": float(mid_resid_s),
            "manualmg_last_vector_sum_time_s": float(vec_sum_s),
            "manualmg_last_coarse_hypre_time_s": float(coarse_s),
            "manualmg_last_coarse_iterations": coarse_iters,
        }


def _mat_scale_add(alpha: float, A, beta: float, B):
    if alpha == 0 and beta == 1:
        return B
    if beta == 0 and alpha == 1:
        return A

    if PETSc is not None and isinstance(A, PETSc.Mat) and isinstance(B, PETSc.Mat):
        C = A.copy()
        C.scale(alpha)
        C.axpy(beta, B)
        C.assemble()
        return C

    if isinstance(A, np.ndarray) or issparse(A):
        return alpha * np.asarray(A) + beta * np.asarray(B)

    if hasattr(A, "__array__"):
        return alpha * np.asarray(A) + beta * np.asarray(B)

    raise TypeError("Unsupported matrix type for linear combination")


def _to_numpy_array(v) -> np.ndarray:
    if hasattr(v, "getArray"):
        return np.asarray(v.getArray(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def _matvec(A, x: np.ndarray) -> np.ndarray:
    if issparse(A):
        return A @ x
    if isinstance(A, np.ndarray):
        return A @ x
    if PETSc is not None and isinstance(A, PETSc.Mat):
        xv = global_array_to_petsc_vec(
            x,
            comm=A.getComm(),
            ownership_range=A.getOwnershipRange() if int(A.getComm().getSize()) > 1 else None,
            bsize=A.getBlockSize() or None,
        )
        y = A.createVecRight()
        A.mult(xv, y)
        return petsc_vec_to_global_array(y)
    if callable(A):
        return np.asarray(A(x), dtype=np.float64)
    if hasattr(A, "dot"):
        return np.asarray(A.dot(x), dtype=np.float64)
    raise TypeError("Unsupported matrix type")


def _coarse_deflation_correction(A, b: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if basis.size == 0:
        return np.zeros_like(b), np.asarray(b, dtype=np.float64)
    x0 = basis @ (basis.T @ np.asarray(b, dtype=np.float64))
    rhs = np.asarray(b, dtype=np.float64) - _matvec(A, x0)
    return x0, rhs


def _basis_diagnostics(basis: np.ndarray, A=None, *, basis_A: np.ndarray | None = None) -> dict[str, float | int]:
    raw = np.asarray(basis, dtype=np.float64)
    if raw.size == 0:
        rows = int(raw.shape[0]) if raw.ndim >= 1 else 0
        return {
            "basis_rows": rows,
            "basis_cols": 0,
            "diag_min": np.nan,
            "diag_max": np.nan,
            "negative_diag_count": 0,
            "offdiag_max_abs": 0.0,
            "offdiag_fro": 0.0,
            "signed_identity_fro": 0.0,
        }
    if raw.ndim == 1:
        raw = raw[:, None]
    if basis_A is None:
        basis_A = np.column_stack([_matvec(A, raw[:, j]) for j in range(raw.shape[1])])
    gram = raw.T @ np.asarray(basis_A, dtype=np.float64)
    diag = np.diag(gram).astype(np.float64, copy=False)
    signed = np.sign(diag)
    signed[signed == 0.0] = 1.0
    offdiag = gram - np.diag(diag)
    return {
        "basis_rows": int(raw.shape[0]),
        "basis_cols": int(raw.shape[1]),
        "diag_min": float(diag.min()) if diag.size else np.nan,
        "diag_max": float(diag.max()) if diag.size else np.nan,
        "negative_diag_count": int(np.count_nonzero(diag < 0.0)),
        "offdiag_max_abs": float(np.max(np.abs(offdiag))) if offdiag.size else 0.0,
        "offdiag_fro": float(np.linalg.norm(offdiag)),
        "signed_identity_fro": float(np.linalg.norm(gram - np.diag(signed))),
    }


class DirectSolver:
    """Direct solver wrapper using PETSc LU (or dense NumPy fallback)."""

    def __init__(self, *, factor_solver_type: str | None = None) -> None:
        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()
        self._ksp = None
        self._A_petsc = None
        self._owns_A_petsc = False
        self.factor_solver_type = factor_solver_type

    def setup_preconditioner(self, A, *, preconditioning_matrix=None, **_kwargs):
        t0 = perf_counter()

        if PETSc is None:
            self._ksp = None
            self._A_petsc = None
            self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)
            return

        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._A_petsc is not None:
            if self._owns_A_petsc:
                release_petsc_aij_matrix(self._A_petsc)
                self._A_petsc.destroy()
            self._A_petsc = None
            self._owns_A_petsc = False
        self._owns_A_petsc = not (PETSc is not None and isinstance(A, PETSc.Mat))
        self._A_petsc = to_petsc_aij_matrix(A, comm=A.getComm() if hasattr(A, "getComm") else PETSc.COMM_SELF)
        self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._ksp.setOperators(self._A_petsc)
        self._ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = self._ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        if self.factor_solver_type:
            pc.setFactorSolverType(self.factor_solver_type)
        self._ksp.setUp()
        self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)

    def A_orthogonalize(self, A):
        # Not applicable for direct solver.
        return

    def solve(self, A, b):
        b = np.asarray(b, dtype=np.float64)
        t0 = perf_counter()

        if PETSc is not None and (isinstance(A, PETSc.Mat) or issparse(A)):
            if self._ksp is None or self._A_petsc is None:
                self.setup_preconditioner(A)
            rhs = PETSc.Vec().createWithArray(b, comm=self._A_petsc.getComm())
            x = self._A_petsc.createVecRight()
            x.set(0.0)
            self._ksp.solve(rhs, x)
            solution = np.asarray(x.getArray(readonly=False)).copy()
        elif issparse(A) or isinstance(A, np.ndarray):
            solution = spsolve(to_scipy_csr_from_petsc(A), b) if issparse(A) else np.linalg.solve(A, b)
        else:
            solution = np.linalg.solve(np.asarray(A), b)

        elapsed = perf_counter() - t0
        nit = int(self._ksp.getIterationNumber()) if self._ksp is not None else 1
        self.iteration_collector.store_iteration(self.instance_id, max(1, nit), elapsed)
        return solution

    def expand_deflation_basis(self, additional_vectors):
        # Direct solver does not use deflation.
        return

    def copy(self):
        clone = self.__class__(factor_solver_type=self.factor_solver_type)
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        return clone

    def prefers_full_system_operator(self) -> bool:
        return False

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        return

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        return {
            "pc_backend": "lu",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "preconditioner_rebuild_count": 0,
            "preconditioner_reuse_count": 0,
            "preconditioner_age_max": 0,
            "preconditioner_setup_time_total": 0.0,
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_last_rebuild_reason": "direct",
        }

    def get_deflation_basis_snapshot(self):
        return None

    def restore_deflation_basis(self, _snapshot) -> None:
        return

    def release_iteration_resources(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._A_petsc is not None:
            if self._owns_A_petsc:
                release_petsc_aij_matrix(self._A_petsc)
                self._A_petsc.destroy()
            self._A_petsc = None
            self._owns_A_petsc = False


class ScipyDirectSolver:
    """Legacy direct solver wrapper using SciPy/NumPy factorization."""

    def __init__(self) -> None:
        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()

    def setup_preconditioner(self, A, *, preconditioning_matrix=None, **_kwargs):
        self.iteration_collector.store_preconditioner_time(self.instance_id, 0.0)

    def A_orthogonalize(self, A):
        return

    def solve(self, A, b):
        b = np.asarray(b, dtype=np.float64)
        t0 = perf_counter()
        if issparse(A):
            solution = spsolve(to_scipy_csr_from_petsc(A), b)
        elif PETSc is not None and isinstance(A, PETSc.Mat):
            solution = spsolve(to_scipy_csr_from_petsc(A), b)
        elif isinstance(A, np.ndarray):
            solution = np.linalg.solve(A, b)
        else:
            solution = np.linalg.solve(np.asarray(A), b)
        self.iteration_collector.store_iteration(self.instance_id, 1, perf_counter() - t0)
        return solution

    def expand_deflation_basis(self, additional_vectors):
        return

    def copy(self):
        clone = self.__class__()
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        return clone

    def prefers_full_system_operator(self) -> bool:
        return False

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        return

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        return {
            "pc_backend": "scipy_direct",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "preconditioner_rebuild_count": 0,
            "preconditioner_reuse_count": 0,
            "preconditioner_age_max": 0,
            "preconditioner_setup_time_total": 0.0,
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_last_rebuild_reason": "direct",
        }

    def get_deflation_basis_snapshot(self):
        return None

    def restore_deflation_basis(self, _snapshot) -> None:
        return

    def release_iteration_resources(self) -> None:
        return


class DeflatedFGMRESSolver:
    """Deflated Flexible GMRES solver.

    API is intentionally compatible with the legacy MATLAB ``DFGMRES`` object.
    """

    def __init__(
        self,
        preconditioner_builder: PreconditionerFactory,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        self.deflation_basis: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.preconditioner_builder = preconditioner_builder
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.tolerance_deflation_basis = float(tolerance_deflation_basis)
        self.verbose = bool(verbose)
        self.preconditioner = None

        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()
        self._diagnostics_enabled = False
        self._last_solve_info: dict[str, object] = {}
        self._last_orthogonalization_info: dict[str, object] = {}

    def setup_preconditioner(self, A, *, preconditioning_matrix=None, **_kwargs):
        t0 = perf_counter()
        self.preconditioner = self.setup_preconditioner_core(A)
        self.iteration_collector.store_preconditioner_time(self.instance_id, perf_counter() - t0)

    def setup_preconditioner_core(self, A):
        return self.preconditioner_builder(A)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        before_cols = int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else int(bool(self.deflation_basis.size))
        self.deflation_basis, basis_norms = a_orthogonalize_with_info(self.deflation_basis, A, self.tolerance_deflation_basis)
        elapsed = perf_counter() - t0
        self.iteration_collector.store_orthogonalization_time(self.instance_id, elapsed)
        if self._diagnostics_enabled:
            self._last_orthogonalization_info = {
                "time_s": float(elapsed),
                "basis_cols_before": int(before_cols),
                "basis_cols_after": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "basis_norm_signs": np.sign(basis_norms).astype(np.float64).tolist(),
                **_basis_diagnostics(self.deflation_basis, A),
            }

    def _as_prec(self):
        if self.preconditioner is None:
            return lambda x: np.asarray(x, dtype=np.float64)
        return self.preconditioner

    def solve(self, A, b):
        t0 = perf_counter()
        rhs = np.asarray(b, dtype=np.float64)
        x, nit, res_hist = self.solve_core(A, rhs)
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, int(nit), elapsed)
        if self._diagnostics_enabled:
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "rhs_norm": float(np.linalg.norm(rhs)),
                "true_residual_history": np.asarray(res_hist, dtype=np.float64).tolist(),
                "reported_residual_final": float(np.asarray(res_hist, dtype=np.float64).reshape(-1)[-1]) if np.asarray(res_hist).size else None,
                "hit_max_iterations": bool(int(nit) >= int(self.max_iterations)),
                "converged": bool(np.asarray(res_hist, dtype=np.float64).size and float(np.asarray(res_hist, dtype=np.float64).reshape(-1)[-1]) <= float(self.tolerance)),
                "converged_reason": None,
            }
        if self.verbose:
            print(f"{nit}|", end="")
        return x

    def solve_core(self, A, b):
        preconditioner = self._as_prec()
        if self.deflation_basis.size == 0:
            basis = np.empty((np.asarray(b).size, 0), dtype=np.float64)
        else:
            basis = self.deflation_basis
        x, nit, res_hist = dfgmres(
            A,
            np.asarray(b, dtype=np.float64),
            preconditioner,
            basis,
            self.max_iterations,
            self.tolerance,
            np.zeros_like(np.asarray(b, dtype=np.float64)),
        )
        return x, nit, res_hist

    def expand_deflation_basis(self, additional_vectors):
        if additional_vectors is None:
            return
        v = np.asarray(additional_vectors, dtype=np.float64)
        if v.size == 0:
            return
        if v.ndim == 1:
            v = v[:, None]
        if self.deflation_basis.size == 0:
            self.deflation_basis = np.asarray(v, dtype=np.float64)
        else:
            self.deflation_basis = np.hstack((self.deflation_basis, v))

    def copy(self):
        clone = self.__class__(
            self.preconditioner_builder,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.preconditioner = self.preconditioner
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        return clone

    def enable_diagnostics(self, enabled: bool = True) -> None:
        self._diagnostics_enabled = bool(enabled)

    def get_last_solve_info(self) -> dict[str, object]:
        return dict(self._last_solve_info)

    def get_last_orthogonalization_info(self) -> dict[str, object]:
        return dict(self._last_orthogonalization_info)

    def prefers_full_system_operator(self) -> bool:
        return False

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        return

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        return {
            "pc_backend": "python",
            "preconditioner_matrix_source": "tangent",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
            "preconditioner_rebuild_interval": 1,
            "preconditioner_rebuild_count": 0,
            "preconditioner_reuse_count": 0,
            "preconditioner_age_max": 0,
            "preconditioner_setup_time_total": 0.0,
            "preconditioner_apply_time_total": 0.0,
            "preconditioner_last_rebuild_reason": "python",
        }

    def get_deflation_basis_snapshot(self):
        return np.array(self.deflation_basis, dtype=np.float64, copy=True)

    def restore_deflation_basis(self, snapshot) -> None:
        if snapshot is None:
            self.deflation_basis = np.empty((0, 0), dtype=np.float64)
        else:
            self.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)

    def release_iteration_resources(self) -> None:
        return


class PetscKSPFGMRESSolver:
    """PETSc-native FGMRES solve with configurable PETSc preconditioner."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        self.pc_type = str(pc_type).upper()
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.tolerance_deflation_basis = float(tolerance_deflation_basis)
        self.verbose = bool(verbose)
        self.q_mask = np.array([], dtype=bool) if q_mask is None else np.asarray(q_mask, dtype=bool)
        self.coord = None if coord is None else np.asarray(coord, dtype=np.float64)
        self.preconditioner_options = dict(preconditioner_options or {})
        self.deflation_basis: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self._A_petsc = None
        self._P_petsc = None
        self._ksp = None
        self._near_nullspace = None
        self._near_nullspace_vecs = []
        self._owns_A_petsc = False
        self._owns_P_petsc = False
        self._using_full_system = False
        self._active_free_indices = np.array([], dtype=np.int64)
        self._ownership_range = None
        self._pmg_state: _PMGPetscHierarchyState | None = None
        self._default_free_indices = (
            np.array([], dtype=np.int64) if self.q_mask.size == 0 else q_to_free_indices(self.q_mask)
        )
        self._options_prefix = f"petsc_linear_{id(self)}_"
        self._diagnostics_enabled = False
        self._last_solve_info: dict[str, object] = {}
        self._last_orthogonalization_info: dict[str, object] = {}
        self._pmg_last_setup_info: dict[str, object] = {}
        self._pmg_last_apply_info: dict[str, object] = {}
        self._pmg_microbenchmark_info: dict[str, object] = {}
        self._manualmg_context: _ManualPMGShellPC | None = None
        self._manualmg_last_setup_info: dict[str, object] = {}
        self._manualmg_last_apply_info: dict[str, object] = {}
        self._pc_backend = self._normalize_pc_backend()
        if self._pc_backend in {"pmg", "pmg_shell"}:
            self.preconditioner_options.setdefault("full_system_preconditioner", False)
        if self._pc_backend == "pmg":
            self.preconditioner_options.setdefault("pc_mg_galerkin", "both")
        self._full_system_preconditioner = bool(
            self.preconditioner_options.get("full_system_preconditioner", self._pc_backend not in {"pmg", "pmg_shell"})
        )
        self._preconditioner_matrix_source = self._normalize_preconditioner_matrix_source()
        self._preconditioner_matrix_policy = self._normalize_preconditioner_matrix_policy()
        self._preconditioner_rebuild_policy = self._normalize_preconditioner_rebuild_policy()
        self._preconditioner_rebuild_interval = self._normalize_preconditioner_rebuild_interval()
        self._validate_pmg_configuration()
        self._preconditioner_rebuild_requested = True
        self._preconditioner_age = 0
        self._preconditioner_newton_calls_since_rebuild = 0
        self._preconditioner_diagnostics = PreconditionerDiagnostics(
            pc_backend=self._pc_backend,
            preconditioner_matrix_source=self._preconditioner_matrix_source,
            preconditioner_matrix_policy=self._preconditioner_matrix_policy,
            preconditioner_rebuild_policy=self._preconditioner_rebuild_policy,
            preconditioner_rebuild_interval=self._preconditioner_rebuild_interval,
        )

        self.iteration_collector = IterationCollector()
        self.instance_id = self.iteration_collector.register_instance()

    def _normalize_pc_backend(self) -> str:
        raw = self.preconditioner_options.get("pc_backend")
        if raw is not None:
            backend = str(raw).strip().lower()
            if backend in {"hypre", "gamg", "bddc", "pmg", "pmg_shell", "jacobi", "none"}:
                return backend
        if self.pc_type == "HYPRE":
            return "hypre"
        if self.pc_type == "GAMG":
            return "gamg"
        if self.pc_type == "JACOBI":
            return "jacobi"
        return "none"

    def _normalize_preconditioner_matrix_policy(self) -> str:
        policy = str(self.preconditioner_options.get("preconditioner_matrix_policy", "current")).strip().lower()
        if policy not in {"current", "lagged"}:
            raise ValueError(f"Unsupported preconditioner_matrix_policy {policy!r}")
        if self._normalize_pc_backend() == "bddc" and policy == "lagged":
            return policy
        return policy

    def _normalize_preconditioner_matrix_source(self) -> str:
        source = str(self.preconditioner_options.get("preconditioner_matrix_source", "tangent")).strip().lower()
        if source not in {"tangent", "regularized", "elastic"}:
            raise ValueError(f"Unsupported preconditioner_matrix_source {source!r}")
        return source

    def _normalize_preconditioner_rebuild_policy(self) -> str:
        policy = str(self.preconditioner_options.get("preconditioner_rebuild_policy", "every_newton")).strip().lower()
        if policy not in {"every_newton", "every_n_newton", "accepted_step", "accepted_or_rejected_step"}:
            raise ValueError(f"Unsupported preconditioner_rebuild_policy {policy!r}")
        return policy

    def _normalize_preconditioner_rebuild_interval(self) -> int:
        try:
            interval = int(self.preconditioner_options.get("preconditioner_rebuild_interval", 1))
        except Exception:
            interval = 1
        return max(1, interval)

    def _validate_pmg_configuration(self) -> None:
        if self._pc_backend not in {"pmg", "pmg_shell"}:
            return
        if self.q_mask.size and int(self.q_mask.shape[0]) != 3:
            raise ValueError(f"{self._pc_backend} backend currently supports only 3D problems.")
        if self._preconditioner_matrix_source != "tangent":
            raise ValueError(f"{self._pc_backend} backend currently supports only preconditioner_matrix_source='tangent'.")
        if self._preconditioner_matrix_policy != "current":
            raise ValueError(f"{self._pc_backend} backend currently supports only preconditioner_matrix_policy='current'.")
        if self._preconditioner_rebuild_policy != "every_newton":
            raise ValueError(f"{self._pc_backend} backend currently supports only preconditioner_rebuild_policy='every_newton'.")
        if self._full_system_preconditioner:
            raise ValueError(f"{self._pc_backend} backend requires full_system_preconditioner=false.")
        levels = self._pmg_levels()
        orders = [int(getattr(level, "order", -1)) for level in levels]
        if any(int(getattr(level, "dim", 0)) != 3 for level in levels):
            raise ValueError("pmg hierarchy levels must be 3D.")
        if any(order < 1 for order in orders):
            raise ValueError(f"pmg hierarchy contains an invalid order sequence {orders!r}.")
        if any(orders[idx] > orders[idx + 1] for idx in range(len(orders) - 1)):
            raise ValueError(f"pmg hierarchy orders must be nondecreasing, got {orders!r}.")
        if orders[0] != 1 or orders[-1] not in {2, 4}:
            raise ValueError(
                f"{self._pc_backend} backend currently supports only 3D hierarchies with coarse P1 and fine P2/P4, got {orders!r}."
            )

    def _record_preconditioner_setup_time(self, elapsed: float) -> None:
        self.iteration_collector.store_preconditioner_time(self.instance_id, elapsed)
        self._preconditioner_diagnostics.preconditioner_setup_time_last = float(elapsed)
        self._preconditioner_diagnostics.preconditioner_setup_time_total += float(elapsed)
        if self._pc_backend == "pmg" and self._pmg_state is not None:
            self._pmg_last_setup_info = {
                "pmg_setup_time_s": float(elapsed),
                **self._pmg_collect_pc_diagnostics(phase="setup"),
            }
        if self._pc_backend == "pmg_shell" and self._manualmg_context is not None:
            self._manualmg_last_setup_info = {
                "manualmg_setup_time_s": float(elapsed),
                **self._manualmg_context.diagnostics(phase="setup"),
            }

    def _record_preconditioner_apply_time(self, elapsed: float) -> None:
        self._preconditioner_diagnostics.preconditioner_apply_time_last = float(elapsed)
        self._preconditioner_diagnostics.preconditioner_apply_time_total += float(elapsed)
        if self._pc_backend == "pmg" and self._pmg_state is not None:
            self._pmg_last_apply_info = self._pmg_collect_pc_diagnostics(
                apply_elapsed_s=float(elapsed),
                phase="solve",
            )
        if self._pc_backend == "pmg_shell" and self._manualmg_context is not None:
            self._manualmg_last_apply_info = self._manualmg_context.diagnostics(
                apply_elapsed_s=float(elapsed),
                phase="solve",
            )

    def _destroy_owned_petsc_matrix(self, A, owns: bool) -> None:
        if A is not None and owns:
            release_petsc_aij_matrix(A)
            A.destroy()

    def _materialize_petsc_matrix(self, matrix, *, comm, block_size=None, ownership_range=None):
        if PETSc is not None and isinstance(matrix, PETSc.Mat):
            if self._pc_backend in {"pmg", "pmg_shell"}:
                mat = matrix.copy()
                return mat, True
            return matrix, False
        mat = to_petsc_aij_matrix(
            matrix,
            comm=matrix.getComm() if hasattr(matrix, "getComm") else comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        return mat, True

    def _matrix_signature(self, A) -> tuple[tuple[int, int], int, int, str] | None:
        if PETSc is None or A is None:
            return None
        if not isinstance(A, PETSc.Mat):
            return None
        return (
            tuple(int(v) for v in A.getSize()),
            int(A.getComm().getSize()),
            int(A.getBlockSize() or 1),
            str(A.getType()),
        )

    def _matrix_compatible(self, A, B) -> bool:
        sig_a = self._matrix_signature(A)
        sig_b = self._matrix_signature(B)
        return sig_a is not None and sig_a == sig_b

    def _mark_preconditioner_rebuilt(self, *, reason: str) -> None:
        self._preconditioner_diagnostics.preconditioner_rebuild_count += 1
        self._preconditioner_diagnostics.preconditioner_last_rebuild_reason = str(reason)
        self._preconditioner_diagnostics.preconditioner_age_max = max(
            int(self._preconditioner_diagnostics.preconditioner_age_max),
            int(self._preconditioner_age),
        )
        self._preconditioner_age = 0
        self._preconditioner_newton_calls_since_rebuild = 0
        self._preconditioner_rebuild_requested = False

    def _mark_preconditioner_reused(self) -> None:
        self._preconditioner_diagnostics.preconditioner_reuse_count += 1
        self._preconditioner_age += 1
        self._preconditioner_newton_calls_since_rebuild += 1
        self._preconditioner_diagnostics.preconditioner_age_max = max(
            int(self._preconditioner_diagnostics.preconditioner_age_max),
            int(self._preconditioner_age),
        )

    def _preconditioner_source_is_static(self) -> bool:
        return bool(self.preconditioner_requires_explicit_matrix() and self._preconditioner_matrix_source == "elastic")

    def _should_rebuild_preconditioner(self) -> tuple[bool, str]:
        if self._preconditioner_source_is_static():
            if self._P_petsc is None:
                return True, "initial"
            policy = self._preconditioner_rebuild_policy
            if policy == "every_n_newton":
                if self._preconditioner_newton_calls_since_rebuild + 1 >= self._preconditioner_rebuild_interval:
                    return True, "every_n_newton"
                return False, "elastic_static"
            if self._preconditioner_rebuild_requested:
                return True, "attempt_trigger"
            return False, "elastic_static"
        if self._preconditioner_matrix_policy == "current":
            return True, "current_policy"
        if self._P_petsc is None:
            return True, "initial"
        policy = self._preconditioner_rebuild_policy
        if policy == "every_newton":
            return True, "every_newton"
        if policy == "every_n_newton":
            if self._preconditioner_newton_calls_since_rebuild + 1 >= self._preconditioner_rebuild_interval:
                return True, "every_n_newton"
            return False, "lagged_reuse"
        if self._preconditioner_rebuild_requested:
            return True, "attempt_trigger"
        return False, "lagged_reuse"

    def preconditioner_requires_explicit_matrix(self) -> bool:
        return self._pc_backend in {"bddc"}

    def needs_preconditioning_matrix_refresh(self) -> bool:
        return False

    def notify_continuation_attempt(self, *, success: bool) -> None:
        if self._preconditioner_rebuild_policy == "accepted_step":
            if success:
                self._preconditioner_rebuild_requested = True
            return
        if self._preconditioner_rebuild_policy == "accepted_or_rejected_step":
            self._preconditioner_rebuild_requested = True

    def supports_dynamic_deflation_basis(self) -> bool:
        return self._pc_backend != "pmg" and not self._deflation_basis_disabled()

    def supports_a_orthogonalization(self) -> bool:
        return self._pc_backend != "pmg" and not self._deflation_basis_disabled()

    def get_preconditioner_diagnostics(self) -> dict[str, object]:
        diagnostics = self._preconditioner_diagnostics.as_dict()
        diagnostics["preconditioner_age_current"] = int(self._preconditioner_age)
        if self._pc_backend == "pmg":
            diagnostics.update(self._pmg_diagnostics_snapshot())
        if self._pc_backend == "pmg_shell" and self._manualmg_context is not None:
            diagnostics.update(self._manualmg_context.diagnostics())
            diagnostics.update({str(k): v for k, v in self._manualmg_last_setup_info.items()})
            diagnostics.update({str(k): v for k, v in self._manualmg_last_apply_info.items()})
        return diagnostics

    def get_preconditioner_matrix_source(self) -> str:
        return str(self._preconditioner_matrix_source)

    def get_deflation_basis_snapshot(self):
        return np.array(self.deflation_basis, dtype=np.float64, copy=True)

    def restore_deflation_basis(self, snapshot) -> None:
        if snapshot is None:
            self.deflation_basis = np.empty((0, 0), dtype=np.float64)
        else:
            self.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)

    def _reset_petsc_objects(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        if self._manualmg_context is not None:
            self._manualmg_context.destroy()
            self._manualmg_context = None
        if self._pmg_state is not None:
            self._pmg_state.destroy()
            self._pmg_state = None
        self._destroy_owned_petsc_matrix(self._P_petsc, self._owns_P_petsc and self._P_petsc is not self._A_petsc)
        self._destroy_owned_petsc_matrix(self._A_petsc, self._owns_A_petsc)
        self._A_petsc = None
        self._P_petsc = None
        self._owns_A_petsc = False
        self._owns_P_petsc = False
        self._near_nullspace = None
        self._near_nullspace_vecs = []
        self._ownership_range = None
        self._last_solve_info = {}
        self._last_orthogonalization_info = {}
        self._pmg_last_setup_info = {}
        self._pmg_last_apply_info = {}
        self._pmg_microbenchmark_info = {}
        self._manualmg_last_setup_info = {}
        self._manualmg_last_apply_info = {}

    def _reset_runtime_petsc_objects_keep_pmg_state(self) -> None:
        if self._ksp is not None:
            self._ksp.destroy()
            self._ksp = None
        inner_ksp = getattr(self, "_inner_ksp", None)
        if inner_ksp is not None:
            inner_ksp.destroy()
            self._inner_ksp = None
        if self._manualmg_context is not None:
            self._manualmg_context.destroy()
            self._manualmg_context = None
        self._destroy_owned_petsc_matrix(self._P_petsc, self._owns_P_petsc and self._P_petsc is not self._A_petsc)
        self._destroy_owned_petsc_matrix(self._A_petsc, self._owns_A_petsc)
        self._A_petsc = None
        self._P_petsc = None
        self._owns_A_petsc = False
        self._owns_P_petsc = False
        self._near_nullspace = None
        self._near_nullspace_vecs = []
        self._ownership_range = None
        self._last_solve_info = {}
        self._last_orthogonalization_info = {}
        self._pmg_last_setup_info = {}
        self._pmg_last_apply_info = {}
        self._pmg_microbenchmark_info = {}
        self._manualmg_last_setup_info = {}
        self._manualmg_last_apply_info = {}

    def _active_free(self, free_indices: np.ndarray | None = None) -> np.ndarray:
        if free_indices is not None:
            return np.asarray(free_indices, dtype=np.int64)
        if self._active_free_indices.size:
            return self._active_free_indices
        return self._default_free_indices

    def _should_use_full_system(self, full_matrix) -> bool:
        return bool(self._full_system_preconditioner and full_matrix is not None and self.q_mask.size)

    def _prepare_operator(self, A, *, full_matrix=None, free_indices: np.ndarray | None = None):
        if self._should_use_full_system(full_matrix):
            self._using_full_system = True
            self._active_free_indices = self._active_free(free_indices)
            if PETSc is not None and isinstance(full_matrix, PETSc.Mat):
                return full_matrix
            zero_rhs = np.zeros(self.q_mask.shape, dtype=np.float64)
            A_operator, _, _ = impose_zero_dirichlet_full_system(full_matrix, zero_rhs, self.q_mask)
            return A_operator

        self._using_full_system = False
        self._active_free_indices = self._active_free(free_indices)
        return A

    def _default_hypre_options(self) -> dict[str, object]:
        if self.q_mask.size == 0:
            return {}
        dim = int(self.q_mask.shape[0])
        return {
            "pc_hypre_boomeramg_max_iter": 1,
            "pc_hypre_boomeramg_tol": 0.0,
            "pc_hypre_boomeramg_numfunctions": dim,
            "pc_hypre_boomeramg_nodal_coarsen": 4,
            "pc_hypre_boomeramg_nodal_coarsen_diag": 1,
            "pc_hypre_boomeramg_vec_interp_variant": 2,
            "pc_hypre_boomeramg_vec_interp_qmax": 4,
            "pc_hypre_boomeramg_vec_interp_smooth": True,
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_interp_type": "ext+i",
            "pc_hypre_boomeramg_P_max": 4,
            "pc_hypre_boomeramg_strong_threshold": 0.5,
            "pc_hypre_boomeramg_grid_sweeps_all": 1,
            "pc_hypre_boomeramg_cycle_type": "V",
            "pc_hypre_boomeramg_agg_nl": 0,
        }

    def _distribution_enabled(self) -> bool:
        return bool(self.preconditioner_options.get("mpi_distribute_by_nodes", False))

    def _matrix_comm(self) -> PETSc.Comm:
        if self._distribution_enabled():
            return PETSc.COMM_WORLD
        return PETSc.COMM_SELF

    @staticmethod
    def _set_petsc_option(options, key: str, value) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            options[key] = "true" if value else "false"
        else:
            options[key] = value

    def _configure_prefixed_options(self, prefix: str) -> None:
        opts = PETSc.Options()
        defaults: dict[str, object] = {}
        if self.pc_type == "HYPRE" and self._using_full_system:
            defaults.update(self._default_hypre_options())
        skip_keys = {
            "threads",
            "print_level",
            "use_as_preconditioner",
            "factor_solver_type",
            "allow_unsafe_hypre_vector_options",
            "coarse_hypre_full_system",
            "full_system_preconditioner",
            "null_space",
            "use_coordinates",
            "pc_backend",
            "preconditioner_matrix_policy",
            "preconditioner_rebuild_policy",
            "preconditioner_rebuild_interval",
            "recycle_preconditioner",
            "compiled_outer",
            "max_deflation_basis_vectors",
            "pmg_hierarchy",
        }

        for key, value in defaults.items():
            if key not in self.preconditioner_options:
                self._set_petsc_option(opts, f"{prefix}{key}", value)

        for key, value in self.preconditioner_options.items():
            if key in skip_keys:
                continue
            if key.startswith(("pc_", "mg_", "ksp_", "mat_")):
                self._set_petsc_option(opts, f"{prefix}{key}", value)

    def _pmg_hierarchy_spec(self):
        hierarchy = self.preconditioner_options.get("pmg_hierarchy")
        if hierarchy is None:
            raise ValueError(f"{self._pc_backend} backend requires a 'pmg_hierarchy' entry in preconditioner_options")
        return hierarchy

    def _pmg_prolongations(self):
        hierarchy = self._pmg_hierarchy_spec()
        prolongations = getattr(hierarchy, "prolongations", None)
        if prolongations is None:
            prolongations = (
                getattr(hierarchy, "prolongation_p21", None),
                getattr(hierarchy, "prolongation_p42", None),
            )
        prolongations = tuple(prolongations)
        if len(prolongations) < 1 or any(transfer is None for transfer in prolongations):
            raise ValueError("pmg hierarchy must provide at least one adjacent-level prolongation.")
        return tuple(prolongations)

    def _pmg_pc(self):
        if self._pc_backend != "pmg":
            return None
        if getattr(self, "_inner_ksp", None) is not None:
            return self._inner_ksp.getPC()
        if self._ksp is not None:
            return self._ksp.getPC()
        return None

    def _pmg_redundant_number(self, *, comm_size: int) -> int:
        raw = self.preconditioner_options.get("mg_coarse_pc_redundant_number")
        if raw is None:
            return int(comm_size)
        try:
            value = int(raw)
        except Exception:
            value = int(comm_size)
        return max(1, min(int(comm_size), value))

    def _pmg_coarse_subcomm_size(self, coarse_ksp) -> int:
        comm_size = int(coarse_ksp.getComm().getSize())
        coarse_pc_type = str(coarse_ksp.getPC().getType()).lower()
        if coarse_pc_type == str(PETSc.PC.Type.REDUNDANT).lower():
            nsub = self._pmg_redundant_number(comm_size=comm_size)
            return max(1, comm_size // max(1, nsub))
        if coarse_pc_type == str(PETSc.PC.Type.TELESCOPE).lower():
            raw = self.preconditioner_options.get("mg_coarse_pc_telescope_reduction_factor")
            try:
                factor = int(raw)
            except Exception:
                factor = int(comm_size)
            factor = max(1, min(int(comm_size), factor))
            return max(1, comm_size // factor)
        return int(comm_size)

    def _pmg_levels(self):
        hierarchy = self._pmg_hierarchy_spec()
        levels = getattr(hierarchy, "levels", None)
        if levels is None:
            levels = (
                getattr(hierarchy, "level_p1", None),
                getattr(hierarchy, "level_p2", None),
                getattr(hierarchy, "level_p4", None),
            )
        levels = tuple(levels)
        if len(levels) < 2 or any(level is None for level in levels):
            raise ValueError("pmg hierarchy must provide at least two levels.")
        return tuple(levels)

    def _validate_pmg_layout(self, *, matrix_ref, state: _PMGPetscHierarchyState) -> None:
        levels = self._pmg_levels()
        fine_level = levels[-1]
        if tuple(int(v) for v in matrix_ref.getSize()) != (int(fine_level.free_size), int(fine_level.free_size)):
            raise ValueError(
                "pmg fine operator size does not match the fine free-space hierarchy: "
                f"{tuple(int(v) for v in matrix_ref.getSize())} vs {(int(fine_level.free_size), int(fine_level.free_size))}"
            )
        if tuple(int(v) for v in matrix_ref.getOwnershipRange()) != tuple(int(v) for v in fine_level.owned_row_range):
            raise ValueError(
                "pmg fine operator ownership does not match the fine free-space hierarchy: "
                f"{tuple(int(v) for v in matrix_ref.getOwnershipRange())} vs {tuple(int(v) for v in fine_level.owned_row_range)}"
            )

        for transfer_idx, mat in enumerate(state.prolongations):
            source_level = levels[transfer_idx]
            target_level = levels[transfer_idx + 1]
            name = f"P{int(target_level.order)}{int(source_level.order)}"
            if tuple(int(v) for v in mat.getSize()) != (int(target_level.free_size), int(source_level.free_size)):
                raise ValueError(
                    f"pmg {name} size mismatch: "
                    f"{tuple(int(v) for v in mat.getSize())} vs {(int(target_level.free_size), int(source_level.free_size))}"
                )
            if tuple(int(v) for v in mat.getOwnershipRange()) != tuple(int(v) for v in target_level.owned_row_range):
                raise ValueError(
                    f"pmg {name} row ownership mismatch: "
                    f"{tuple(int(v) for v in mat.getOwnershipRange())} vs {tuple(int(v) for v in target_level.owned_row_range)}"
                )
            local_rows, local_cols = (int(v) for v in mat.getLocalSize())
            expected_rows = int(target_level.hi - target_level.lo)
            expected_cols = int(source_level.hi - source_level.lo)
            if local_rows != expected_rows:
                raise ValueError(
                    f"pmg {name} local row count mismatch: {local_rows} vs expected {expected_rows}"
                )
                if local_cols != expected_cols:
                    raise ValueError(
                        f"pmg {name} local column count mismatch: {local_cols} vs expected {expected_cols}"
                    )
        for transfer_idx, mat in enumerate(state.restrictions):
            source_level = levels[transfer_idx + 1]
            target_level = levels[transfer_idx]
            name = f"R{int(target_level.order)}{int(source_level.order)}"
            if tuple(int(v) for v in mat.getSize()) != (int(target_level.free_size), int(source_level.free_size)):
                raise ValueError(
                    f"pmg {name} size mismatch: "
                    f"{tuple(int(v) for v in mat.getSize())} vs {(int(target_level.free_size), int(source_level.free_size))}"
                )
            if tuple(int(v) for v in mat.getOwnershipRange()) != tuple(int(v) for v in target_level.owned_row_range):
                raise ValueError(
                    f"pmg {name} row ownership mismatch: "
                    f"{tuple(int(v) for v in mat.getOwnershipRange())} vs {tuple(int(v) for v in target_level.owned_row_range)}"
                )
            local_rows, local_cols = (int(v) for v in mat.getLocalSize())
            expected_rows = int(target_level.hi - target_level.lo)
            expected_cols = int(source_level.hi - source_level.lo)
            if local_rows != expected_rows:
                raise ValueError(
                    f"pmg {name} local row count mismatch: {local_rows} vs expected {expected_rows}"
                )
            if local_cols != expected_cols:
                raise ValueError(
                    f"pmg {name} local column count mismatch: {local_cols} vs expected {expected_cols}"
                )

    def _pmg_collect_pc_diagnostics(self, *, apply_elapsed_s: float | None = None, phase: str | None = None) -> dict[str, object]:
        if self._pc_backend != "pmg" or self._pmg_state is None:
            return {}
        pc = self._pmg_pc()
        if pc is None:
            return {}
        coarse = pc.getMGCoarseSolve()
        payload: dict[str, object] = {
            "pmg_levels": int(pc.getMGLevels()),
            "pmg_level_orders": [int(v) for v in self._pmg_state.level_orders],
            "pmg_level_global_sizes": [int(v) for v in self._pmg_state.level_global_sizes],
            "pmg_level_owned_ranges": [[int(lo), int(hi)] for lo, hi in self._pmg_state.level_owned_ranges],
            "pmg_transfer_shapes": [[int(v) for v in mat.getSize()] for mat in self._pmg_state.prolongations],
            "pmg_transfer_owned_ranges": [[int(v) for v in mat.getOwnershipRange()] for mat in self._pmg_state.prolongations],
            "pmg_restriction_shapes": [[int(v) for v in mat.getSize()] for mat in self._pmg_state.restrictions],
            "pmg_restriction_owned_ranges": [[int(v) for v in mat.getOwnershipRange()] for mat in self._pmg_state.restrictions],
            "pmg_coarse_ksp_type": str(coarse.getType()),
            "pmg_coarse_pc_type": str(coarse.getPC().getType()),
            "pmg_coarse_iterations": int(coarse.getIterationNumber()),
            "pmg_coarse_converged_reason": int(coarse.getConvergedReason()),
            "pmg_coarse_subcomm_size": int(self._pmg_coarse_subcomm_size(coarse)),
        }
        for level_idx in range(1, len(self._pmg_state.level_orders)):
            smoother = pc.getMGSmoother(level_idx)
            payload[f"pmg_level_{level_idx}_ksp_type"] = str(smoother.getType())
            payload[f"pmg_level_{level_idx}_pc_type"] = str(smoother.getPC().getType())
            payload[f"pmg_level_{level_idx}_iterations"] = int(smoother.getIterationNumber())
            payload[f"pmg_level_{level_idx}_converged_reason"] = int(smoother.getConvergedReason())
            payload[f"pmg_level_{level_idx}_owned_range"] = [int(v) for v in self._pmg_state.level_owned_ranges[level_idx]]
        if apply_elapsed_s is not None:
            payload["pmg_last_pc_apply_time_s"] = float(apply_elapsed_s)
        if phase is not None:
            payload["pmg_last_phase"] = str(phase)
        return payload

    def _pmg_diagnostics_snapshot(self) -> dict[str, object]:
        diagnostics: dict[str, object] = {}
        if self._pmg_state is not None:
            diagnostics.update(self._pmg_collect_pc_diagnostics())
        if self._pmg_last_setup_info:
            diagnostics.update({str(k): v for k, v in self._pmg_last_setup_info.items()})
        if self._pmg_last_apply_info:
            diagnostics.update({str(k): v for k, v in self._pmg_last_apply_info.items()})
        if self._pmg_microbenchmark_info:
            diagnostics["pmg_microbenchmark"] = dict(self._pmg_microbenchmark_info)
        return diagnostics

    def _configure_manualmg_pc(self, pc, *, matrix_ref) -> None:
        if matrix_ref is None:
            raise ValueError("pmg_shell backend requires a fine-level preconditioning matrix")
        state = self._ensure_pmg_state()
        self._validate_pmg_layout(matrix_ref=matrix_ref, state=state)
        if self._manualmg_context is None:
            self._manualmg_context = _ManualPMGShellPC(self)
        self._manualmg_context.configure(matrix_ref=matrix_ref, state=state, hierarchy=self._pmg_hierarchy_spec())
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(self._manualmg_context)
        self._manualmg_last_setup_info = {
            "manualmg_setup_time_s": float(self._preconditioner_diagnostics.preconditioner_setup_time_last),
            **self._manualmg_context.diagnostics(phase="setup"),
        }

    def run_pmg_microbenchmark(self, rhs: np.ndarray | None = None) -> dict[str, object]:
        if self._pc_backend != "pmg":
            return {}
        if self._A_petsc is None:
            raise RuntimeError("pmg microbenchmark requires setup_preconditioner() first.")
        pc = self._pmg_pc()
        if pc is None:
            raise RuntimeError("pmg microbenchmark requires a configured PETSc MG preconditioner.")

        x = self._A_petsc.createVecRight()
        y = self._A_petsc.createVecRight()
        if rhs is None:
            x.set(1.0)
        else:
            rhs_arr = np.asarray(rhs, dtype=np.float64).reshape(-1)
            x_arr = x.getArray(readonly=False)
            if rhs_arr.size == x_arr.size:
                x_arr[...] = rhs_arr
            elif self._ownership_range is not None and rhs_arr.size >= int(self._ownership_range[1]):
                r0, r1 = self._ownership_range
                x_arr[...] = rhs_arr[r0:r1]
            else:
                raise ValueError(
                    f"pmg microbenchmark rhs has size {rhs_arr.size}, expected local {x_arr.size} or global >= {self._ownership_range[1] if self._ownership_range is not None else x_arr.size}"
                )
        y.set(0.0)
        t0 = perf_counter()
        pc.apply(x, y)
        elapsed = perf_counter() - t0
        y_local = np.asarray(y.getArray(readonly=True), dtype=np.float64)
        result = {
            "status": "completed",
            "phase": "microbenchmark",
            "rhs_norm_local": float(np.linalg.norm(np.asarray(x.getArray(readonly=True), dtype=np.float64))),
            "solution_norm_local": float(np.linalg.norm(y_local)),
            "pc_apply_elapsed_s": float(elapsed),
            **self._pmg_collect_pc_diagnostics(apply_elapsed_s=elapsed, phase="microbenchmark"),
        }
        self._pmg_microbenchmark_info = dict(result)
        return result

    def _can_reuse_pmg_ksp(self, A_petsc) -> bool:
        if self._pc_backend != "pmg":
            return False
        if self._ksp is None:
            return False
        if self._ksp.getType() != PETSc.KSP.Type.FGMRES:
            return False
        return self._matrix_compatible(A_petsc, self._A_petsc)

    def _ensure_pmg_state(self) -> _PMGPetscHierarchyState:
        if self._pmg_state is not None:
            return self._pmg_state

        hierarchy = self._pmg_hierarchy_spec()
        levels = self._pmg_levels()
        transfers = self._pmg_prolongations()
        comm = self._matrix_comm()
        petsc_prolongations = []
        for transfer_idx, transfer in enumerate(transfers):
            source_level = levels[transfer_idx]
            petsc_prolongations.append(
                owned_coo_to_petsc_aij_matrix(
                    transfer.coo_rows,
                    transfer.coo_cols,
                    transfer.coo_data,
                    global_shape=transfer.global_shape,
                    owned_row_range=transfer.owned_row_range,
                    comm=comm,
                    local_col_size=int(source_level.owned_row_range[1] - source_level.owned_row_range[0]),
                )
            )
        restrictions = []
        for prolongation in petsc_prolongations:
            restriction = prolongation.copy()
            restriction = restriction.transpose()
            restrictions.append(restriction)
        self._pmg_state = _PMGPetscHierarchyState(
            hierarchy=hierarchy,
            prolongations=tuple(petsc_prolongations),
            restrictions=tuple(restrictions),
            level_orders=tuple(int(level.order) for level in levels),
            level_global_sizes=tuple(int(level.free_size) for level in levels),
            level_owned_ranges=tuple(tuple(int(v) for v in level.owned_row_range) for level in levels),
        )
        return self._pmg_state

    def _configure_pmg_pc(self, pc, *, matrix_ref) -> None:
        if matrix_ref is None:
            raise ValueError("pmg backend requires a fine-level preconditioning matrix")
        state = self._ensure_pmg_state()
        self._validate_pmg_layout(matrix_ref=matrix_ref, state=state)
        pc.setType(PETSc.PC.Type.MG)
        pc.setMGLevels(len(state.level_orders))
        pc.setMGType(PETSc.PC.MGType.MULTIPLICATIVE)
        pc.setMGCycleType(PETSc.PC.MGCycleType.V)
        for level_idx in range(1, len(state.level_orders)):
            pc.setMGInterpolation(level_idx, state.prolongations[level_idx - 1])
            pc.setMGRestriction(level_idx, state.restrictions[level_idx - 1])

        coarse = pc.getMGCoarseSolve()
        coarse_ksp_type = str(
            self.preconditioner_options.get("mg_coarse_ksp_type", PETSc.KSP.Type.PREONLY)
        ).strip().lower()
        coarse_pc_default = "lu" if int(matrix_ref.getComm().getSize()) == 1 else "redundant"
        coarse_pc_type = str(
            self.preconditioner_options.get("mg_coarse_pc_type", coarse_pc_default)
        ).strip().lower()
        coarse.setType(coarse_ksp_type)
        coarse_pc = coarse.getPC()
        coarse_pc.setType(coarse_pc_type)
        if coarse_pc_type == str(PETSc.PC.Type.HYPRE).lower():
            coarse_pc.setHYPREType(
                str(self.preconditioner_options.get("mg_coarse_pc_hypre_type", "boomeramg"))
            )
        coarse.setTolerances(rtol=1.0e-10, atol=0.0, max_it=200)

        for level_idx in range(1, len(state.level_orders)):
            smoother = pc.getMGSmoother(level_idx)
            smoother.setType(PETSc.KSP.Type.RICHARDSON)
            smoother.setTolerances(rtol=0.0, atol=0.0, max_it=3)
            smoother.getPC().setType(PETSc.PC.Type.SOR)
        self._pmg_last_setup_info = {
            "pmg_setup_time_s": float(self._preconditioner_diagnostics.preconditioner_setup_time_last),
            **self._pmg_collect_pc_diagnostics(phase="setup"),
        }

    def _max_deflation_basis_vectors(self) -> int | None:
        raw = self.preconditioner_options.get("max_deflation_basis_vectors")
        if raw is None:
            return None
        try:
            value = int(raw)
        except Exception:
            return None
        return value if value >= 0 else None

    def _deflation_basis_disabled(self) -> bool:
        max_cols = self._max_deflation_basis_vectors()
        return max_cols == 0

    def _expand_to_full_space(self, vectors: np.ndarray) -> np.ndarray:
        vec = np.asarray(vectors, dtype=np.float64)
        if vec.ndim == 1:
            vec = vec[:, None]
        if not (self._full_system_preconditioner and self.q_mask.size):
            return vec
        if vec.shape[0] == self.q_mask.size:
            return vec

        free_idx = self._active_free()
        if free_idx.size and vec.shape[0] == free_idx.size:
            full = np.zeros((self.q_mask.size, vec.shape[1]), dtype=np.float64)
            full[free_idx, :] = vec
            return full
        return vec

    def _prepare_rhs(self, b, *, full_rhs=None) -> np.ndarray:
        rhs_free = np.asarray(b, dtype=np.float64).reshape(-1)
        if not self._using_full_system:
            return rhs_free

        if full_rhs is None:
            rhs = np.zeros(self.q_mask.size, dtype=np.float64)
            rhs[self._active_free()] = rhs_free
        else:
            rhs_arr = np.asarray(full_rhs, dtype=np.float64)
            rhs = rhs_arr.reshape(-1, order="F").copy() if rhs_arr.ndim > 1 else rhs_arr.reshape(-1).copy()

        rhs[~self.q_mask.reshape(-1, order="F")] = 0.0
        return rhs

    def _restrict_solution(self, x: np.ndarray) -> np.ndarray:
        if not self._using_full_system:
            return x
        return np.asarray(x, dtype=np.float64)[self._active_free()]

    def enable_diagnostics(self, enabled: bool = True) -> None:
        self._diagnostics_enabled = bool(enabled)

    def get_last_solve_info(self) -> dict[str, object]:
        return dict(self._last_solve_info)

    def get_last_orthogonalization_info(self) -> dict[str, object]:
        return dict(self._last_orthogonalization_info)

    def _install_true_residual_monitor(
        self,
        rhs_total: np.ndarray,
        *,
        solution_offset: np.ndarray | None = None,
    ) -> tuple[list[float], list[float]]:
        reported_history: list[float] = []
        true_history: list[float] = []
        rhs_ref = np.asarray(rhs_total, dtype=np.float64).reshape(-1)
        rhs_norm = float(np.linalg.norm(rhs_ref))
        if rhs_norm == 0.0:
            rhs_norm = 1.0
        offset = None if solution_offset is None else np.asarray(solution_offset, dtype=np.float64).reshape(-1)

        self._ksp.cancelMonitor()

        def _monitor(ksp, it, rnorm):
            x_iter = petsc_vec_to_global_array(ksp.buildSolution())
            if offset is not None:
                x_iter = x_iter + offset
            resid = rhs_ref - _matvec(self._A_petsc, x_iter)
            reported_history.append(float(rnorm))
            true_history.append(float(np.linalg.norm(resid) / rhs_norm))

        self._ksp.setMonitor(_monitor)
        return reported_history, true_history

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for KSPFGMRES solver types.")

        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (PETSc is not None and isinstance(operator_matrix, PETSc.Mat)):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        new_A_petsc, new_A_owned = self._materialize_petsc_matrix(
            operator_matrix,
            comm=comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        new_ownership_range = new_A_petsc.getOwnershipRange()

        if self._pc_backend not in {"pmg", "pmg_shell"}:
            self._reset_petsc_objects()
            self._A_petsc = new_A_petsc
            self._owns_A_petsc = new_A_owned
            self._P_petsc = None
            self._owns_P_petsc = False
            self._ownership_range = new_ownership_range
        else:
            reuse_ksp = self._can_reuse_pmg_ksp(new_A_petsc)
            old_A_petsc = self._A_petsc
            old_A_owned = self._owns_A_petsc
            old_P_petsc = self._P_petsc
            old_P_owned = self._owns_P_petsc
            if not reuse_ksp and self._ksp is not None:
                self._ksp.destroy()
                self._ksp = None
            self._A_petsc = new_A_petsc
            self._owns_A_petsc = new_A_owned
            self._P_petsc = self._A_petsc
            self._owns_P_petsc = False
            self._ownership_range = new_ownership_range
            self._near_nullspace = None
            self._near_nullspace_vecs = []
            self._last_solve_info = {}
            self._last_orthogonalization_info = {}
            self._pmg_last_apply_info = {}
            self._manualmg_last_apply_info = {}
            if old_A_petsc is not None and old_A_petsc is not self._A_petsc:
                self._destroy_owned_petsc_matrix(old_A_petsc, old_A_owned)
            if old_P_petsc is not None and old_P_petsc is not self._P_petsc and old_P_petsc is not self._A_petsc:
                self._destroy_owned_petsc_matrix(old_P_petsc, old_P_owned)

        if self._using_full_system and self.q_mask.size:
            self._A_petsc.setBlockSize(int(self.q_mask.shape[0]))

        null_space = self.preconditioner_options.get("null_space")
        if null_space is None and self.pc_type in {"GAMG", "HYPRE"} and self.coord is not None and self.q_mask.size:
            null_space = make_near_nullspace_elasticity(
                self.coord,
                q_mask=self.q_mask,
                center_coordinates=True,
                return_full=self._using_full_system,
            )

        if self.pc_type in {"GAMG", "HYPRE"}:
            self._A_petsc, self._near_nullspace, self._near_nullspace_vecs = attach_near_nullspace(
                self._A_petsc,
                null_space,
            )
        else:
            self._near_nullspace = None
            self._near_nullspace_vecs = []

        if self._ksp is None:
            self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
            self._ksp.setOptionsPrefix(self._options_prefix)
            self._ksp.setType(PETSc.KSP.Type.FGMRES)
            self._ksp.setInitialGuessNonzero(False)
            self._ksp.setTolerances(rtol=self.tolerance, atol=1e-30, max_it=self.max_iterations)
        self._ksp.setOperators(self._A_petsc)
        self._configure_prefixed_options(self._options_prefix)
        pc = self._ksp.getPC()
        matrix_ref = self._P_petsc if self._P_petsc is not None else self._A_petsc
        if self._pc_backend == "pmg":
            self._configure_pmg_pc(pc, matrix_ref=matrix_ref)
        elif self._pc_backend == "pmg_shell":
            self._configure_manualmg_pc(pc, matrix_ref=matrix_ref)
        elif self.pc_type == "GAMG":
            pc.setType(PETSc.PC.Type.GAMG)
            if self._using_full_system and self.coord is not None and self.preconditioner_options.get("use_coordinates", True):
                if int(self._A_petsc.getComm().getSize()) > 1:
                    dim = int(self.q_mask.shape[0])
                    r0, r1 = self._ownership_range
                    node0, node1 = r0 // dim, r1 // dim
                    pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                else:
                    pc.setCoordinates(self.coord.T.copy())
        elif self.pc_type == "HYPRE":
            pc.setType(PETSc.PC.Type.HYPRE)
            pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
        elif self.pc_type == "JACOBI":
            pc.setType(PETSc.PC.Type.JACOBI)
        else:
            pc.setType(PETSc.PC.Type.NONE)
        self._ksp.setFromOptions()
        self._ksp.setUp()
        self._record_preconditioner_setup_time(perf_counter() - t0)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        A_ref = self._A_petsc if self._A_petsc is not None else A
        before_cols = int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else int(bool(self.deflation_basis.size))
        self.deflation_basis, basis_norms = a_orthogonalize_with_info(
            self.deflation_basis,
            A_ref,
            self.tolerance_deflation_basis,
        )
        elapsed = perf_counter() - t0
        self.iteration_collector.store_orthogonalization_time(self.instance_id, elapsed)
        if self._diagnostics_enabled:
            self._last_orthogonalization_info = {
                "time_s": float(elapsed),
                "basis_cols_before": int(before_cols),
                "basis_cols_after": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "basis_norm_signs": np.sign(basis_norms).astype(np.float64).tolist(),
                **_basis_diagnostics(self.deflation_basis, A_ref),
            }

    def solve(self, A, b, *, full_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for KSPFGMRES solver types.")
        if self._ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)

        rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
        basis = self.deflation_basis if self.deflation_basis.size else np.empty((rhs_arr.size, 0), dtype=np.float64)

        t0 = perf_counter()
        x0, rhs_eff = _coarse_deflation_correction(self._A_petsc, rhs_arr, basis)
        reported_history: list[float] = []
        true_history: list[float] = []
        if self._diagnostics_enabled:
            reported_history, true_history = self._install_true_residual_monitor(rhs_arr, solution_offset=x0)
        rhs = global_array_to_petsc_vec(
            rhs_eff,
            comm=self._A_petsc.getComm(),
            ownership_range=self._ownership_range if int(self._A_petsc.getComm().getSize()) > 1 else None,
            bsize=self._A_petsc.getBlockSize() or None,
        )
        delta = self._A_petsc.createVecRight()
        delta.set(0.0)
        self._ksp.solve(rhs, delta)

        x = x0 + petsc_vec_to_global_array(delta)
        nit = int(self._ksp.getIterationNumber())
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, nit, elapsed)
        if self._diagnostics_enabled:
            resid = rhs_arr - _matvec(self._A_petsc, x)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            if not true_history:
                true_history = [float(np.linalg.norm(resid) / rhs_norm)]
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": float(np.linalg.norm(x0)),
                "reported_residual_history": list(reported_history),
                "true_residual_history": list(true_history),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
                "reported_residual_final": float(reported_history[-1]) if reported_history else None,
                "hit_max_iterations": bool(int(nit) >= int(self.max_iterations)),
                "converged": bool(int(self._ksp.getConvergedReason()) > 0),
                "converged_reason": int(self._ksp.getConvergedReason()),
            }
            self._ksp.cancelMonitor()
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(x)

    def expand_deflation_basis(self, additional_vectors):
        if additional_vectors is None:
            return
        v = np.asarray(additional_vectors, dtype=np.float64)
        if v.size == 0:
            return
        v = self._expand_to_full_space(v)
        if self._deflation_basis_disabled():
            self.deflation_basis = np.empty((v.shape[0], 0), dtype=np.float64)
            return
        if self.deflation_basis.size == 0:
            self.deflation_basis = np.asarray(v, dtype=np.float64)
        else:
            self.deflation_basis = np.hstack((self.deflation_basis, v))
        max_cols = self._max_deflation_basis_vectors()
        if max_cols is not None and self.deflation_basis.ndim == 2 and self.deflation_basis.shape[1] > max_cols:
            self.deflation_basis = np.asarray(self.deflation_basis[:, -max_cols:], dtype=np.float64)

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        clone._preconditioner_diagnostics = self._preconditioner_diagnostics
        return clone

    def prefers_full_system_operator(self) -> bool:
        return bool(self._full_system_preconditioner and self.q_mask.size)

    def release_iteration_resources(self) -> None:
        if self._pc_backend in {"pmg", "pmg_shell"}:
            self._last_solve_info = {}
            self._last_orthogonalization_info = {}
            return
        self._reset_petsc_objects()

    def close(self) -> None:
        self._reset_petsc_objects()


class PetscKSPGMRESDeflationSolver(PetscKSPFGMRESSolver):
    """PETSc-native GMRES with PCDEFLATION and a user-supplied recycle space."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )
        self._using_native_deflation = False
        self._deflation_space_mat = None
        self._deflation_space_buffers = None

    def _reset_petsc_objects(self) -> None:
        if self._deflation_space_mat is not None:
            release_petsc_aij_matrix(self._deflation_space_mat)
            self._deflation_space_mat.destroy()
            self._deflation_space_mat = None
        self._deflation_space_buffers = None
        self._using_native_deflation = False
        super()._reset_petsc_objects()

    def _basis_qr_tolerance(self) -> float:
        return max(1.0e-12, self.tolerance_deflation_basis * 1.0e-3)

    def _compress_basis(self, basis: np.ndarray) -> np.ndarray:
        raw = np.asarray(basis, dtype=np.float64)
        if raw.size == 0:
            rows = int(raw.shape[0]) if raw.ndim >= 1 else 0
            return np.empty((rows, 0), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[:, None]

        col_norm = np.linalg.norm(raw, axis=0)
        keep = col_norm > self._basis_qr_tolerance()
        if not np.any(keep):
            return np.empty((raw.shape[0], 0), dtype=np.float64)

        q, r = np.linalg.qr(raw[:, keep], mode="reduced")
        diag = np.abs(np.diag(r))
        if diag.size == 0:
            return np.empty((raw.shape[0], 0), dtype=np.float64)
        keep_q = diag > (self._basis_qr_tolerance() * float(diag.max()))
        if not np.any(keep_q):
            return np.empty((raw.shape[0], 0), dtype=np.float64)
        return np.asfortranarray(q[:, keep_q], dtype=np.float64)

    def _build_deflation_space(self):
        basis = self._compress_basis(self.deflation_basis)
        self.deflation_basis = basis
        if basis.size == 0:
            return None, False

        comm = self._A_petsc.getComm()
        n_rows, n_basis = basis.shape
        import scipy.sparse as sp

        if int(comm.getSize()) == 1:
            W = to_petsc_aij_matrix(sp.csr_matrix(basis), comm=comm)
            self._deflation_space_buffers = None
        else:
            r0, r1 = self._ownership_range
            csr = sp.csr_matrix(basis[r0:r1, :])
            indptr = np.array(csr.indptr, dtype=PETSc.IntType, copy=True)
            indices = np.array(csr.indices, dtype=PETSc.IntType, copy=True)
            data = np.array(csr.data, dtype=np.float64, copy=True)
            W = PETSc.Mat().createAIJ(
                size=((r1 - r0, n_rows), (PETSc.DECIDE, n_basis)),
                csr=(indptr, indices, data),
                comm=comm,
            )
            W.assemble()
            self._deflation_space_buffers = (indptr, indices, data)
        self._deflation_space_mat = W
        return W, False

    def _configure_native_deflation_options(self, prefix: str) -> None:
        opts = PETSc.Options()
        defaults: dict[str, object] = {
            "pc_deflation_correction_factor": self.preconditioner_options.get("pc_deflation_correction_factor", 1.0),
            "pc_deflation_init_only": False,
        }

        if self.pc_type == "HYPRE" and self._using_full_system:
            for key, value in self._default_hypre_options().items():
                if key not in self.preconditioner_options:
                    self._set_petsc_option(opts, f"{prefix}deflation_{key}", value)

        for key, value in defaults.items():
            if key not in self.preconditioner_options:
                self._set_petsc_option(opts, f"{prefix}{key}", value)

        for key, value in self.preconditioner_options.items():
            if key in {
                "threads",
                "print_level",
                "use_as_preconditioner",
                "factor_solver_type",
                "allow_unsafe_hypre_vector_options",
                "coarse_hypre_full_system",
                "full_system_preconditioner",
                "null_space",
                "use_coordinates",
                "pc_backend",
                "preconditioner_matrix_policy",
                "preconditioner_rebuild_policy",
                "preconditioner_rebuild_interval",
                "recycle_preconditioner",
                "compiled_outer",
                "max_deflation_basis_vectors",
            }:
                continue
            if key.startswith("pc_deflation_"):
                self._set_petsc_option(opts, f"{prefix}{key}", value)
            elif key.startswith("ksp_"):
                self._set_petsc_option(opts, f"{prefix}{key}", value)
            elif key.startswith(("pc_", "mg_", "mat_")):
                self._set_petsc_option(opts, f"{prefix}deflation_{key}", value)

    def _configure_inner_deflation_pc(self) -> None:
        pc = self._ksp.getPC()
        inner_pc = pc.getDeflationPC()
        if self.pc_type == "GAMG":
            inner_pc.setType(PETSc.PC.Type.GAMG)
            if self._using_full_system and self.coord is not None and self.preconditioner_options.get("use_coordinates", True):
                if int(self._A_petsc.getComm().getSize()) > 1:
                    dim = int(self.q_mask.shape[0])
                    r0, r1 = self._ownership_range
                    node0, node1 = r0 // dim, r1 // dim
                    inner_pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                else:
                    inner_pc.setCoordinates(self.coord.T.copy())
        elif self.pc_type == "HYPRE":
            inner_pc.setType(PETSc.PC.Type.HYPRE)
            inner_pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
        elif self.pc_type == "JACOBI":
            inner_pc.setType(PETSc.PC.Type.JACOBI)
        else:
            inner_pc.setType(PETSc.PC.Type.NONE)
        # Leave PC setup to the owning KSP so prefixed `mg_*` nested options
        # are applied before PETSc constructs MG smoothers/coarse solvers.
        inner_pc.setFromOptions()

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for native deflation KSP solver types.")

        self._reset_petsc_objects()

        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (PETSc is not None and isinstance(operator_matrix, PETSc.Mat)):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        self._A_petsc = to_petsc_aij_matrix(
            operator_matrix,
            comm=operator_matrix.getComm() if hasattr(operator_matrix, "getComm") else comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        self._ownership_range = self._A_petsc.getOwnershipRange()

        if self._using_full_system and self.q_mask.size:
            self._A_petsc.setBlockSize(int(self.q_mask.shape[0]))

        null_space = self.preconditioner_options.get("null_space")
        if null_space is None and self.pc_type in {"GAMG", "HYPRE"} and self.coord is not None and self.q_mask.size:
            null_space = make_near_nullspace_elasticity(
                self.coord,
                q_mask=self.q_mask,
                center_coordinates=True,
                return_full=self._using_full_system,
            )

        if self.pc_type in {"GAMG", "HYPRE"}:
            self._A_petsc, self._near_nullspace, self._near_nullspace_vecs = attach_near_nullspace(
                self._A_petsc,
                null_space,
            )
        else:
            self._near_nullspace = None
            self._near_nullspace_vecs = []

        self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._ksp.setOptionsPrefix(self._options_prefix)
        self._ksp.setOperators(self._A_petsc)
        self._ksp.setType(PETSc.KSP.Type.GMRES)
        self._ksp.setInitialGuessNonzero(False)
        self._ksp.setTolerances(rtol=self.tolerance, atol=1e-30, max_it=self.max_iterations)

        basis_mat, basis_is_transpose = self._build_deflation_space()
        self._using_native_deflation = basis_mat is not None
        pc = self._ksp.getPC()
        if self._using_native_deflation:
            pc.setType(PETSc.PC.Type.DEFLATION)
            pc.setDeflationSpace(basis_mat, basis_is_transpose)
            pc.setDeflationInitOnly(False)
            if "pc_deflation_correction_factor" in self.preconditioner_options:
                pc.setDeflationCorrectionFactor(float(self.preconditioner_options["pc_deflation_correction_factor"]))
            else:
                pc.setDeflationCorrectionFactor(1.0)
            self._configure_native_deflation_options(self._options_prefix)
            self._ksp.setFromOptions()
            self._ksp.setUp()
            self._configure_inner_deflation_pc()
        else:
            self._configure_prefixed_options(self._options_prefix)
            if self.pc_type == "GAMG":
                pc.setType(PETSc.PC.Type.GAMG)
                if self._using_full_system and self.coord is not None and self.preconditioner_options.get("use_coordinates", True):
                    if int(self._A_petsc.getComm().getSize()) > 1:
                        dim = int(self.q_mask.shape[0])
                        r0, r1 = self._ownership_range
                        node0, node1 = r0 // dim, r1 // dim
                        pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                    else:
                        pc.setCoordinates(self.coord.T.copy())
            elif self.pc_type == "HYPRE":
                pc.setType(PETSc.PC.Type.HYPRE)
                pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
            elif self.pc_type == "JACOBI":
                pc.setType(PETSc.PC.Type.JACOBI)
            else:
                pc.setType(PETSc.PC.Type.NONE)
            self._ksp.setFromOptions()
            self._ksp.setUp()

        self._record_preconditioner_setup_time(perf_counter() - t0)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        if self.deflation_basis.size:
            self.deflation_basis = self._compress_basis(self.deflation_basis)
        self.iteration_collector.store_orthogonalization_time(self.instance_id, perf_counter() - t0)

    def solve(self, A, b, *, full_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for native deflation KSP solver types.")
        if self._ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)

        rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
        t0 = perf_counter()
        reported_history: list[float] = []
        true_history: list[float] = []
        if self._diagnostics_enabled:
            reported_history, true_history = self._install_true_residual_monitor(rhs_arr)
        rhs = global_array_to_petsc_vec(
            rhs_arr,
            comm=self._A_petsc.getComm(),
            ownership_range=self._ownership_range if int(self._A_petsc.getComm().getSize()) > 1 else None,
            bsize=self._A_petsc.getBlockSize() or None,
        )
        x = self._A_petsc.createVecRight()
        x.set(0.0)
        self._ksp.solve(rhs, x)

        nit = int(self._ksp.getIterationNumber())
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, nit, elapsed)
        if self._diagnostics_enabled:
            x_total = petsc_vec_to_global_array(x)
            resid = rhs_arr - _matvec(self._A_petsc, x_total)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            if not true_history:
                true_history = [float(np.linalg.norm(resid) / rhs_norm)]
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": 0.0,
                "reported_residual_history": list(reported_history),
                "true_residual_history": list(true_history),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
                "reported_residual_final": float(reported_history[-1]) if reported_history else None,
                "hit_max_iterations": bool(int(nit) >= int(self.max_iterations)),
                "converged": bool(int(self._ksp.getConvergedReason()) > 0),
                "converged_reason": int(self._ksp.getConvergedReason()),
            }
            self._ksp.cancelMonitor()
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(petsc_vec_to_global_array(x))

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        return clone


class _ProjectedMatlabDeflationPC:
    """Python PC context applying MATLAB-style projected preconditioning."""

    def __init__(self, solver: "PetscKSPMatlabDeflatedFGMRESSolver") -> None:
        self.solver = solver
        self._tmp = None

    def apply(self, pc, x, y) -> None:
        if self._tmp is None:
            self._tmp = self.solver._A_petsc.createVecRight()
        self._tmp.set(0.0)
        t0 = perf_counter()
        self.solver._inner_ksp.solve(x, self._tmp)
        self.solver._record_preconditioner_apply_time(perf_counter() - t0)
        z_local = np.asarray(self._tmp.getArray(readonly=True), dtype=np.float64)
        y_local = self.solver._project_local_vector(z_local)
        y_arr = y.getArray(readonly=False)
        y_arr[...] = y_local


class PetscKSPMatlabDeflatedFGMRESSolver(PetscKSPFGMRESSolver):
    """PETSc FGMRES using a MATLAB-style projected preconditioner."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )
        self._inner_ksp = None
        self._projected_pc_context = None
        self._basis_local = np.empty((0, 0), dtype=np.float64)
        self._basis_A_local = np.empty((0, 0), dtype=np.float64)
        self._basis_global = np.empty((0, 0), dtype=np.float64)
        self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        self._mpi_comm = None
        self._last_basis_reorth_passes = 0
        self._projector_dirty = True

    def _reset_petsc_objects(self) -> None:
        if self._inner_ksp is not None:
            self._inner_ksp.destroy()
            self._inner_ksp = None
        self._projected_pc_context = None
        self._basis_local = np.empty((0, 0), dtype=np.float64)
        self._basis_A_local = np.empty((0, 0), dtype=np.float64)
        self._basis_global = np.empty((0, 0), dtype=np.float64)
        self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        self._mpi_comm = None
        self._last_basis_reorth_passes = 0
        self._projector_dirty = True
        super()._reset_petsc_objects()

    def _deflation_reorth_passes(self) -> int:
        try:
            passes = int(self.preconditioner_options.get("deflation_reorth_passes", 1))
        except Exception:
            passes = 1
        return max(1, passes)

    def _refresh_projector_data(self, A_ref) -> None:
        basis = self.deflation_basis
        if basis.size == 0:
            self._basis_global = np.empty((0, 0), dtype=np.float64)
            self._basis_A_global = np.empty((0, 0), dtype=np.float64)
            self._basis_local = np.empty((0, 0), dtype=np.float64)
            self._basis_A_local = np.empty((0, 0), dtype=np.float64)
            self._last_basis_reorth_passes = 0
            self._projector_dirty = False
            return

        used_passes = 0
        distributed_local = PETSc is not None and isinstance(A_ref, PETSc.Mat) and int(A_ref.getComm().getSize()) > 1

        if distributed_local:
            basis_local = None
            kept_source_idx = None
            for _ in range(self._deflation_reorth_passes()):
                basis_local, _basis_norms, kept_source_idx = a_orthogonalize_with_local_metadata(
                    basis,
                    A_ref,
                    self.tolerance_deflation_basis,
                )
                used_passes += 1
                if basis_local.size == 0 or kept_source_idx.size == 0:
                    basis = np.empty((basis.shape[0], 0), dtype=np.float64)
                    break
                basis = np.asarray(basis[:, kept_source_idx], dtype=np.float64)
            self.deflation_basis = basis
        else:
            for _ in range(self._deflation_reorth_passes()):
                basis, _basis_norms = a_orthogonalize_with_info(basis, A_ref, self.tolerance_deflation_basis)
                used_passes += 1
                if basis.size == 0:
                    break
            self.deflation_basis = basis
            basis_local = None
            kept_source_idx = None

        if basis.size == 0:
            self._basis_global = np.empty((0, 0), dtype=np.float64)
            self._basis_A_global = np.empty((0, 0), dtype=np.float64)
            self._basis_local = np.empty((0, 0), dtype=np.float64)
            self._basis_A_local = np.empty((0, 0), dtype=np.float64)
            self._last_basis_reorth_passes = used_passes
            self._projector_dirty = False
            return

        r0, r1 = self._ownership_range if self._ownership_range is not None else (0, basis.shape[0])
        if distributed_local:
            x_vec = A_ref.createVecRight()
            y_vec = A_ref.createVecRight()
            x_arr = x_vec.getArray(readonly=False)
            self._basis_local = np.asarray(basis_local, dtype=np.float64)
            basis_A_local = np.empty((r1 - r0, self._basis_local.shape[1]), dtype=np.float64)
            for j in range(self._basis_local.shape[1]):
                x_arr[...] = self._basis_local[:, j]
                A_ref.mult(x_vec, y_vec)
                col_local = np.asarray(y_vec.getArray(readonly=True), dtype=np.float64).copy()
                basis_A_local[:, j] = col_local
            self._basis_A_local = basis_A_local
            if self._diagnostics_enabled:
                gathered_basis = self._mpi_comm.allgather(self._basis_local)
                gathered_basis_A = self._mpi_comm.allgather(self._basis_A_local)
                self._basis_global = np.vstack(gathered_basis) if gathered_basis else np.empty((0, 0), dtype=np.float64)
                self._basis_A_global = np.vstack(gathered_basis_A) if gathered_basis_A else np.empty((0, 0), dtype=np.float64)
            else:
                self._basis_global = np.empty((0, 0), dtype=np.float64)
                self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        else:
            self._basis_local = basis[r0:r1, :]
            basis_A = np.column_stack([_matvec(A_ref, basis[:, j]) for j in range(basis.shape[1])])
            self._basis_A_local = basis_A[r0:r1, :]
            if self._diagnostics_enabled:
                self._basis_global = basis
                self._basis_A_global = basis_A
            else:
                self._basis_global = np.empty((0, 0), dtype=np.float64)
                self._basis_A_global = np.empty((0, 0), dtype=np.float64)
        self._last_basis_reorth_passes = used_passes
        self._projector_dirty = False

    def _ensure_projector_ready(self) -> None:
        if self._projector_dirty:
            A_ref = self._A_petsc if self._A_petsc is not None else None
            if A_ref is None:
                return
            self._refresh_projector_data(A_ref)

    def _project_local_vector(self, z_local: np.ndarray) -> np.ndarray:
        self._ensure_projector_ready()
        if self._basis_local.size == 0:
            return np.asarray(z_local, dtype=np.float64)
        coeff_local = self._basis_A_local.T @ np.asarray(z_local, dtype=np.float64)
        coeff = self._mpi_comm.allreduce(coeff_local, op=MPI.SUM)
        return np.asarray(z_local, dtype=np.float64) - self._basis_local @ coeff

    def _coarse_initial_guess_local(self, b_local: np.ndarray) -> np.ndarray:
        self._ensure_projector_ready()
        if self._basis_local.size == 0:
            return np.zeros_like(np.asarray(b_local, dtype=np.float64))
        coeff_local = self._basis_local.T @ np.asarray(b_local, dtype=np.float64)
        coeff = self._mpi_comm.allreduce(coeff_local, op=MPI.SUM)
        return self._basis_local @ coeff

    def expand_deflation_basis(self, additional_vectors):
        super().expand_deflation_basis(additional_vectors)
        self._projector_dirty = True

    def _configure_inner_pc(self) -> None:
        inner_pc = self._inner_ksp.getPC()
        if hasattr(inner_pc, "setOptionsPrefix"):
            inner_pc.setOptionsPrefix(f"{self._options_prefix}inner_")
        matrix_ref = self._P_petsc if self._P_petsc is not None else self._A_petsc
        if self._pc_backend == "gamg":
            inner_pc.setType(PETSc.PC.Type.GAMG)
            if (
                self._using_full_system
                and self.coord is not None
                and self.preconditioner_options.get("use_coordinates", True)
                and matrix_ref is not None
            ):
                if int(matrix_ref.getComm().getSize()) > 1:
                    dim = int(self.q_mask.shape[0])
                    r0, r1 = self._ownership_range
                    node0, node1 = r0 // dim, r1 // dim
                    inner_pc.setCoordinates(self.coord[:, node0:node1].T.copy())
                else:
                    inner_pc.setCoordinates(self.coord.T.copy())
        elif self._pc_backend == "pmg":
            self._configure_pmg_pc(inner_pc, matrix_ref=matrix_ref)
        elif self._pc_backend == "pmg_shell":
            self._configure_manualmg_pc(inner_pc, matrix_ref=matrix_ref)
        elif self._pc_backend == "hypre":
            inner_pc.setType(PETSc.PC.Type.HYPRE)
            inner_pc.setHYPREType(str(self.preconditioner_options.get("pc_hypre_type", "boomeramg")))
        elif self._pc_backend == "bddc":
            inner_pc.setType(PETSc.PC.Type.BDDC)
            metadata = get_petsc_matrix_metadata(matrix_ref)
            coordinates = bddc_pc_coordinates_from_metadata(matrix_ref)
            if coordinates is not None and self.preconditioner_options.get("use_coordinates", True):
                inner_pc.setCoordinates(np.asarray(coordinates, dtype=np.float64))
            field_is = metadata.get("bddc_field_is_local")
            if field_is:
                if not all(isinstance(v, PETSc.IS) for v in field_is):
                    field_is = tuple(
                        PETSc.IS().createGeneral(np.asarray(v, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
                        for v in field_is
                    )
                inner_pc.setBDDCDofsSplittingLocal(field_is)
            dirichlet = metadata.get("bddc_dirichlet_local")
            if dirichlet is not None:
                if not isinstance(dirichlet, PETSc.IS):
                    dirichlet = PETSc.IS().createGeneral(np.asarray(dirichlet, dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
                inner_pc.setBDDCDirichletBoundariesLocal(dirichlet)
            adjacency = metadata.get("bddc_local_adjacency")
            if adjacency is not None:
                inner_pc.setBDDCLocalAdjacency(adjacency)
            primal_vertices = metadata.get("bddc_primal_vertices_local")
            if primal_vertices is not None:
                if not isinstance(primal_vertices, PETSc.IS):
                    primal_vertices = PETSc.IS().createGeneral(
                        np.asarray(primal_vertices, dtype=PETSc.IntType),
                        comm=PETSc.COMM_SELF,
                    )
                inner_pc.setBDDCPrimalVerticesLocalIS(primal_vertices)
        elif self._pc_backend == "jacobi":
            inner_pc.setType(PETSc.PC.Type.JACOBI)
        else:
            inner_pc.setType(PETSc.PC.Type.NONE)
        # Leave PC setup to the owning KSP so prefixed `mg_*` nested options
        # are applied before PETSc constructs MG smoothers/coarse solvers.
        inner_pc.setFromOptions()

    def _configure_outer_ksp_only_options(self) -> None:
        opts = PETSc.Options()
        for key, value in self.preconditioner_options.items():
            if key.startswith("ksp_"):
                self._set_petsc_option(opts, f"{self._options_prefix}{key}", value)

    def _configure_inner_pc_only_options(self) -> None:
        opts = PETSc.Options()
        prefix = f"{self._options_prefix}inner_"
        for key, value in self.preconditioner_options.items():
            if key in {
                "threads",
                "print_level",
                "use_as_preconditioner",
                "factor_solver_type",
                "allow_unsafe_hypre_vector_options",
                "coarse_hypre_full_system",
                "full_system_preconditioner",
                "null_space",
                "use_coordinates",
                "deflation_reorth_passes",
                "pc_backend",
                "preconditioner_matrix_policy",
                "preconditioner_rebuild_policy",
                "preconditioner_rebuild_interval",
                "recycle_preconditioner",
                "compiled_outer",
                "max_deflation_basis_vectors",
                "pmg_hierarchy",
            }:
                continue
            if self._pc_backend == "pmg_shell" and key.startswith(("pc_hypre_", "mg_")):
                continue
            if key.startswith(("pc_", "mg_", "mat_")):
                self._set_petsc_option(opts, f"{prefix}{key}", value)

    def _outer_presolve(self, ksp, b, x) -> None:
        x_arr = x.getArray(readonly=False)
        if self._basis_local.size == 0:
            x_arr[...] = 0.0
            return
        b_local = np.asarray(b.getArray(readonly=True), dtype=np.float64)
        x_arr[...] = self._coarse_initial_guess_local(b_local)

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for MATLAB-like KSPFGMRES solver types.")

        self._reset_petsc_objects()

        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (PETSc is not None and isinstance(operator_matrix, PETSc.Mat)):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        self._A_petsc = to_petsc_aij_matrix(
            operator_matrix,
            comm=operator_matrix.getComm() if hasattr(operator_matrix, "getComm") else comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        self._ownership_range = self._A_petsc.getOwnershipRange()
        self._mpi_comm = self._A_petsc.getComm().tompi4py()

        if self._using_full_system and self.q_mask.size:
            self._A_petsc.setBlockSize(int(self.q_mask.shape[0]))

        null_space = self.preconditioner_options.get("null_space")
        if null_space is None and self.pc_type in {"GAMG", "HYPRE"} and self.coord is not None and self.q_mask.size:
            null_space = make_near_nullspace_elasticity(
                self.coord,
                q_mask=self.q_mask,
                center_coordinates=True,
                return_full=self._using_full_system,
            )

        if self.pc_type in {"GAMG", "HYPRE"}:
            self._A_petsc, self._near_nullspace, self._near_nullspace_vecs = attach_near_nullspace(
                self._A_petsc,
                null_space,
            )
        else:
            self._near_nullspace = None
            self._near_nullspace_vecs = []

        self._inner_ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._inner_ksp.setOptionsPrefix(f"{self._options_prefix}inner_")
        self._inner_ksp.setOperators(self._A_petsc)
        self._inner_ksp.setType(PETSc.KSP.Type.PREONLY)
        self._configure_inner_pc_only_options()
        self._configure_inner_pc()
        self._inner_ksp.setFromOptions()
        self._inner_ksp.setUp()

        self._ksp = PETSc.KSP().create(comm=self._A_petsc.getComm())
        self._ksp.setOptionsPrefix(self._options_prefix)
        self._ksp.setOperators(self._A_petsc)
        self._ksp.setType(PETSc.KSP.Type.FGMRES)
        self._ksp.setInitialGuessNonzero(True)
        self._ksp.setTolerances(rtol=self.tolerance, atol=1e-30, max_it=self.max_iterations)
        self._configure_outer_ksp_only_options()
        self._projected_pc_context = _ProjectedMatlabDeflationPC(self)
        pc = self._ksp.getPC()
        pc.setType(PETSc.PC.Type.PYTHON)
        pc.setPythonContext(self._projected_pc_context)
        self._ksp.setPreSolve(self._outer_presolve)
        self._ksp.setFromOptions()
        self._ksp.setUp()
        self._refresh_projector_data(self._A_petsc)
        self._record_preconditioner_setup_time(perf_counter() - t0)

    def A_orthogonalize(self, A):
        t0 = perf_counter()
        A_ref = self._A_petsc if self._A_petsc is not None else A
        before_cols = int(self.deflation_basis.shape[1]) if self.deflation_basis.ndim == 2 else int(bool(self.deflation_basis.size))
        self._refresh_projector_data(A_ref)
        elapsed = perf_counter() - t0
        self.iteration_collector.store_orthogonalization_time(self.instance_id, elapsed)
        if self._diagnostics_enabled:
            self._last_orthogonalization_info = {
                "time_s": float(elapsed),
                "basis_cols_before": int(before_cols),
                "basis_cols_after": int(self._basis_global.shape[1]) if self._basis_global.ndim == 2 else 0,
                "basis_reorth_passes": int(self._last_basis_reorth_passes),
                **_basis_diagnostics(self._basis_global, basis_A=self._basis_A_global),
            }

    def solve(self, A, b, *, full_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for MATLAB-like KSPFGMRES solver types.")
        if self._ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)

        rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
        t0 = perf_counter()
        reported_history: list[float] = []
        true_history: list[float] = []
        if self._diagnostics_enabled:
            reported_history, true_history = self._install_true_residual_monitor(rhs_arr)
        rhs = global_array_to_petsc_vec(
            rhs_arr,
            comm=self._A_petsc.getComm(),
            ownership_range=self._ownership_range if int(self._A_petsc.getComm().getSize()) > 1 else None,
            bsize=self._A_petsc.getBlockSize() or None,
        )
        x = self._A_petsc.createVecRight()
        x.set(0.0)
        self._ksp.solve(rhs, x)
        nit = int(self._ksp.getIterationNumber())
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, nit, elapsed)
        if self._diagnostics_enabled:
            x_total = petsc_vec_to_global_array(x)
            resid = rhs_arr - _matvec(self._A_petsc, x_total)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            if not true_history:
                true_history = [float(np.linalg.norm(resid) / rhs_norm)]
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(self._basis_global.shape[1]) if self._basis_global.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": float(np.linalg.norm(self._coarse_initial_guess_local(np.asarray(rhs.getArray(readonly=True), dtype=np.float64)))),
                "reported_residual_history": list(reported_history),
                "true_residual_history": list(true_history),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
                "reported_residual_final": float(reported_history[-1]) if reported_history else None,
                "hit_max_iterations": bool(int(nit) >= int(self.max_iterations)),
                "converged": bool(int(self._ksp.getConvergedReason()) > 0),
                "converged_reason": int(self._ksp.getConvergedReason()),
            }
            self._ksp.cancelMonitor()
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(petsc_vec_to_global_array(x))

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        return clone


class PetscKSPMatlabDeflatedFGMRESReorthSolver(PetscKSPMatlabDeflatedFGMRESSolver):
    """MATLAB-like PETSc FGMRES with stronger Krylov orthogonality and repeated A-reorthogonalization."""

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        opts = dict(preconditioner_options or {})
        opts.setdefault("ksp_gmres_classicalgramschmidt", True)
        opts.setdefault("ksp_gmres_cgs_refinement_type", "refine_always")
        opts.setdefault("deflation_reorth_passes", 3)
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=opts,
        )


class PetscMatlabExactDFGMRESSolver(PetscKSPMatlabDeflatedFGMRESSolver):
    """MATLAB-style DFGMRES outer iteration driven directly in Python.

    PETSc is used for:
    - the assembled operator storage
    - the inner PREONLY preconditioner apply
    - GAMG/HYPRE setup

    The outer Arnoldi/projection loop intentionally mirrors MATLAB's
    ``dfgmres_solver.m`` instead of PETSc's built-in KSP implementation.
    """

    def __init__(
        self,
        pc_type: str = "GAMG",
        *,
        tolerance: float = 1e-6,
        max_iterations: int = 100,
        tolerance_deflation_basis: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ) -> None:
        super().__init__(
            pc_type,
            tolerance=tolerance,
            max_iterations=max_iterations,
            tolerance_deflation_basis=tolerance_deflation_basis,
            verbose=verbose,
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )
        self._matvec_in = None
        self._matvec_out = None
        self._prec_in = None
        self._prec_out = None

    def _reuse_preconditioner_enabled(self) -> bool:
        return bool(self.preconditioner_options.get("recycle_preconditioner", False))

    def _preserve_preconditioner_state_between_solves(self) -> bool:
        return bool(
            self._reuse_preconditioner_enabled()
            or self._preconditioner_matrix_policy == "lagged"
            or self._preconditioner_source_is_static()
        )

    def _clear_transient_vectors(self) -> None:
        self._matvec_in = None
        self._matvec_out = None
        self._prec_in = None
        self._prec_out = None

    def _reset_petsc_objects(self) -> None:
        self._clear_transient_vectors()
        super()._reset_petsc_objects()

    def release_iteration_resources(self) -> None:
        if self._preserve_preconditioner_state_between_solves():
            self._clear_transient_vectors()
            self._last_solve_info = {}
            self._last_orthogonalization_info = {}
            return
        if self._pc_backend in {"pmg", "pmg_shell"}:
            self._clear_transient_vectors()
            self._last_solve_info = {}
            self._last_orthogonalization_info = {}
            return
        self._reset_petsc_objects()

    def _default_null_space(self):
        null_space = self.preconditioner_options.get("null_space")
        if null_space is not None:
            return null_space
        if self._pc_backend not in {"gamg", "hypre"} or self.coord is None or not self.q_mask.size:
            return None
        return make_near_nullspace_elasticity(
            self.coord,
            q_mask=self.q_mask,
            center_coordinates=True,
            return_full=self._using_full_system,
        )

    def _configure_reuse_flags(self) -> None:
        if self._inner_ksp is None:
            return
        if hasattr(self._inner_ksp, "setReusePreconditioner"):
            self._inner_ksp.setReusePreconditioner(True)
        pc = self._inner_ksp.getPC()
        if hasattr(pc, "setReusePreconditioner"):
            pc.setReusePreconditioner(True)

    def _can_reuse_inner_ksp(self, A_petsc, P_petsc) -> bool:
        if self._inner_ksp is None:
            return False
        if self._inner_ksp.getType() != PETSc.KSP.Type.PREONLY:
            return False
        return self._matrix_compatible(A_petsc, self._A_petsc) and self._matrix_compatible(P_petsc, self._P_petsc)

    def _prepare_operator_matrix(self, A, *, full_matrix=None, free_indices: np.ndarray | None = None):
        operator_matrix = self._prepare_operator(A, full_matrix=full_matrix, free_indices=free_indices)
        comm = self._matrix_comm()
        block_size = None
        ownership_range = None
        if self._using_full_system and self.q_mask.size:
            block_size = int(self.q_mask.shape[0])
        elif self.preconditioner_options.get("block_size") is not None:
            block_size = int(self.preconditioner_options["block_size"])
        if int(comm.getSize()) > 1 and block_size is not None and not (
            PETSc is not None and isinstance(operator_matrix, PETSc.Mat)
        ):
            ownership_range = owned_block_range(operator_matrix.shape[0] // block_size, block_size, comm)
        A_petsc, owns = self._materialize_petsc_matrix(
            operator_matrix,
            comm=comm,
            block_size=block_size,
            ownership_range=ownership_range,
        )
        if self._using_full_system and self.q_mask.size:
            A_petsc.setBlockSize(int(self.q_mask.shape[0]))
        null_space = self._default_null_space()
        if self._pc_backend in {"gamg", "hypre"}:
            A_petsc, near_nullspace, near_nullspace_vecs = attach_near_nullspace(A_petsc, null_space)
        else:
            near_nullspace = None
            near_nullspace_vecs = []
        return A_petsc, owns, A_petsc.getOwnershipRange(), A_petsc.getComm().tompi4py(), near_nullspace, near_nullspace_vecs

    def _prepare_preconditioning_matrix(self, source_matrix):
        if source_matrix is None:
            raise ValueError("Explicit preconditioning matrix is required for this backend")
        if PETSc is not None and isinstance(source_matrix, PETSc.Mat):
            return source_matrix, False
        comm = self._matrix_comm()
        block_size = int(self.q_mask.shape[0]) if self._using_full_system and self.q_mask.size else None
        P_petsc = to_petsc_aij_matrix(source_matrix, comm=comm, block_size=block_size)
        return P_petsc, True

    def needs_preconditioning_matrix_refresh(self) -> bool:
        if not self.preconditioner_requires_explicit_matrix():
            return False
        rebuild, _reason = self._should_rebuild_preconditioner()
        return rebuild

    def setup_preconditioner(
        self,
        A,
        *,
        full_matrix=None,
        free_indices: np.ndarray | None = None,
        preconditioning_matrix=None,
    ):
        t0 = perf_counter()
        if PETSc is None:
            raise RuntimeError("PETSc is required for explicit MATLAB-style DFGMRES solver types.")
        new_A_petsc, new_A_owned, ownership_range, mpi_comm, near_nullspace, near_nullspace_vecs = self._prepare_operator_matrix(
            A,
            full_matrix=full_matrix,
            free_indices=free_indices,
        )

        rebuild_preconditioner, rebuild_reason = self._should_rebuild_preconditioner()
        new_P_petsc = self._P_petsc
        new_P_owned = self._owns_P_petsc
        if self.preconditioner_requires_explicit_matrix():
            if rebuild_preconditioner:
                new_P_petsc, new_P_owned = self._prepare_preconditioning_matrix(preconditioning_matrix)
            elif new_P_petsc is None:
                raise ValueError("BDDC backend requested lagged reuse without an initialized preconditioning matrix")
        else:
            if rebuild_preconditioner:
                if preconditioning_matrix is not None:
                    new_P_petsc, new_P_owned = self._prepare_preconditioning_matrix(preconditioning_matrix)
                elif self._preconditioner_matrix_policy == "current":
                    new_P_petsc = new_A_petsc
                    new_P_owned = False
                else:
                    new_P_petsc = new_A_petsc.copy()
                    new_P_owned = True
            elif new_P_petsc is None:
                new_P_petsc = new_A_petsc
                new_P_owned = False
                rebuild_preconditioner = True
                rebuild_reason = "initial"

        if self._using_full_system and self.q_mask.size and new_P_petsc is not None:
            try:
                new_P_petsc.setBlockSize(int(self.q_mask.shape[0]))
            except Exception:
                pass

        reuse_inner = self._can_reuse_inner_ksp(new_A_petsc, new_P_petsc)
        old_A_petsc = self._A_petsc
        old_A_owned = self._owns_A_petsc
        old_P_petsc = self._P_petsc
        old_P_owned = self._owns_P_petsc

        if not reuse_inner:
            self._clear_transient_vectors()
            if self._inner_ksp is not None:
                self._inner_ksp.destroy()
                self._inner_ksp = None
            if self._ksp is not None:
                self._ksp.destroy()
                self._ksp = None
            self._inner_ksp = PETSc.KSP().create(comm=new_A_petsc.getComm())
            self._inner_ksp.setOptionsPrefix(f"{self._options_prefix}inner_")
            self._inner_ksp.setType(PETSc.KSP.Type.PREONLY)

        self._clear_transient_vectors()
        self._A_petsc = new_A_petsc
        self._owns_A_petsc = new_A_owned
        self._P_petsc = new_P_petsc
        self._owns_P_petsc = new_P_owned
        self._ownership_range = ownership_range
        self._mpi_comm = mpi_comm
        self._near_nullspace = near_nullspace
        self._near_nullspace_vecs = near_nullspace_vecs
        self._configure_inner_pc_only_options()
        self._inner_ksp.setOperators(self._A_petsc, self._P_petsc)
        self._configure_inner_pc()
        self._inner_ksp.setFromOptions()
        self._configure_reuse_flags()
        self._inner_ksp.setUp()

        if old_A_petsc is not None and old_A_petsc is not self._A_petsc:
            self._destroy_owned_petsc_matrix(old_A_petsc, old_A_owned)
        if old_P_petsc is not None and old_P_petsc is not self._P_petsc and old_P_petsc is not self._A_petsc:
            self._destroy_owned_petsc_matrix(old_P_petsc, old_P_owned)

        self._projector_dirty = True
        if rebuild_preconditioner:
            self._mark_preconditioner_rebuilt(reason=rebuild_reason)
        else:
            self._mark_preconditioner_reused()
        self._record_preconditioner_setup_time(perf_counter() - t0)

    def _apply_inner_preconditioner(self, x: np.ndarray) -> np.ndarray:
        if self._prec_in is None or self._prec_out is None:
            self._prec_in = self._A_petsc.createVecRight()
            self._prec_out = self._A_petsc.createVecRight()
        rhs_arr = self._prec_in.getArray(readonly=False)
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.size != rhs_arr.size:
            r0, r1 = self._ownership_range if self._ownership_range is not None else (0, rhs_arr.size)
            x_arr = x_arr[r0:r1]
        rhs_arr[...] = x_arr
        self._prec_out.set(0.0)
        t0 = perf_counter()
        self._inner_ksp.getPC().apply(self._prec_in, self._prec_out)
        self._record_preconditioner_apply_time(perf_counter() - t0)
        return petsc_vec_to_global_array(self._prec_out)

    def _apply_inner_preconditioner_local(self, x_local: np.ndarray) -> np.ndarray:
        if self._prec_in is None or self._prec_out is None:
            self._prec_in = self._A_petsc.createVecRight()
            self._prec_out = self._A_petsc.createVecRight()
        rhs_arr = self._prec_in.getArray(readonly=False)
        rhs_arr[...] = np.asarray(x_local, dtype=np.float64)
        self._prec_out.set(0.0)
        t0 = perf_counter()
        self._inner_ksp.getPC().apply(self._prec_in, self._prec_out)
        self._record_preconditioner_apply_time(perf_counter() - t0)
        return np.asarray(self._prec_out.getArray(readonly=True), dtype=np.float64).copy()

    def _petsc_matvec(self, x: np.ndarray) -> np.ndarray:
        if self._matvec_in is None or self._matvec_out is None:
            self._matvec_in = self._A_petsc.createVecRight()
            self._matvec_out = self._A_petsc.createVecLeft()
        x_arr = self._matvec_in.getArray(readonly=False)
        x_global = np.asarray(x, dtype=np.float64)
        if x_global.size != x_arr.size:
            r0, r1 = self._ownership_range if self._ownership_range is not None else (0, x_arr.size)
            x_global = x_global[r0:r1]
        x_arr[...] = x_global
        self._A_petsc.mult(self._matvec_in, self._matvec_out)
        return petsc_vec_to_global_array(self._matvec_out)

    def _petsc_matvec_local(self, x_local: np.ndarray) -> np.ndarray:
        if self._matvec_in is None or self._matvec_out is None:
            self._matvec_in = self._A_petsc.createVecRight()
            self._matvec_out = self._A_petsc.createVecLeft()
        x_arr = self._matvec_in.getArray(readonly=False)
        x_arr[...] = np.asarray(x_local, dtype=np.float64)
        self._A_petsc.mult(self._matvec_in, self._matvec_out)
        return np.asarray(self._matvec_out.getArray(readonly=True), dtype=np.float64).copy()

    def solve(self, A, b, *, full_rhs=None, local_rhs=None, free_indices: np.ndarray | None = None):
        if PETSc is None:
            raise RuntimeError("PETSc is required for explicit MATLAB-style DFGMRES solver types.")
        if self._inner_ksp is None or self._A_petsc is None:
            self.setup_preconditioner(A, free_indices=free_indices)
        self._ensure_projector_ready()

        t0 = perf_counter()
        timing_stats: dict[str, float] = {}

        use_distributed_local = bool(self._mpi_comm is not None and int(self._A_petsc.getComm().getSize()) > 1)
        rhs_arr = None

        def _timed_matvec(v: np.ndarray) -> np.ndarray:
            t_mat = perf_counter()
            out = self._petsc_matvec(v)
            timing_stats["matvec_s"] = timing_stats.get("matvec_s", 0.0) + (perf_counter() - t_mat)
            return out

        def _timed_prec(v: np.ndarray) -> np.ndarray:
            t_prec = perf_counter()
            out = self._apply_inner_preconditioner(v)
            timing_stats["preconditioner_apply_s"] = timing_stats.get("preconditioner_apply_s", 0.0) + (perf_counter() - t_prec)
            return out

        if use_distributed_local:
            r0, r1 = self._ownership_range
            if local_rhs is not None:
                rhs_local = np.asarray(local_rhs, dtype=np.float64).reshape(-1)
            else:
                rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
                rhs_local = np.asarray(rhs_arr[r0:r1], dtype=np.float64)
            basis_local = self._basis_local if self._basis_local.size else np.empty((rhs_local.size, 0), dtype=np.float64)

            def _timed_matvec_local(v_local: np.ndarray) -> np.ndarray:
                t_mat = perf_counter()
                out = self._petsc_matvec_local(v_local)
                timing_stats["matvec_s"] = timing_stats.get("matvec_s", 0.0) + (perf_counter() - t_mat)
                return out

            def _timed_prec_local(v_local: np.ndarray) -> np.ndarray:
                t_prec = perf_counter()
                out = self._apply_inner_preconditioner_local(v_local)
                timing_stats["preconditioner_apply_s"] = timing_stats.get("preconditioner_apply_s", 0.0) + (perf_counter() - t_prec)
                return out

            compiled_outer = bool(self.preconditioner_options.get("compiled_outer", False))
            if compiled_outer:
                x_local, nit, res_hist = dfgmres_matlab_exact_distributed_compiled(
                    _timed_matvec_local,
                    rhs_local,
                    _timed_prec_local,
                    basis_local,
                    self.max_iterations,
                    self.tolerance,
                    self._mpi_comm,
                    None,
                    stats=timing_stats,
                )
            else:
                x_local, nit, res_hist = dfgmres_matlab_exact_distributed(
                    _timed_matvec_local,
                    rhs_local,
                    _timed_prec_local,
                    basis_local,
                    self.max_iterations,
                    self.tolerance,
                    self._mpi_comm,
                    None,
                    stats=timing_stats,
                )
            x_total = np.concatenate(self._mpi_comm.allgather(np.asarray(x_local, dtype=np.float64)))
            if self._diagnostics_enabled:
                rhs_arr = np.concatenate(self._mpi_comm.allgather(rhs_local))
        else:
            rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
            basis = self.deflation_basis if self.deflation_basis.size else np.empty((rhs_arr.size, 0), dtype=np.float64)
            x_total, nit, res_hist = dfgmres_matlab_exact(
                _timed_matvec,
                rhs_arr,
                _timed_prec,
                basis,
                self.max_iterations,
                self.tolerance,
                None,
                stats=timing_stats,
            )
        elapsed = perf_counter() - t0
        self.iteration_collector.store_iteration(self.instance_id, int(nit), elapsed)
        if self._diagnostics_enabled:
            if rhs_arr is None:
                rhs_arr = self._prepare_rhs(b, full_rhs=full_rhs)
            basis = self.deflation_basis if self.deflation_basis.size else np.empty((rhs_arr.size, 0), dtype=np.float64)
            rhs_norm = float(np.linalg.norm(rhs_arr))
            if rhs_norm == 0.0:
                rhs_norm = 1.0
            resid = rhs_arr - self._petsc_matvec(x_total)
            coarse_guess = basis @ (basis.T @ rhs_arr) if basis.size else np.zeros_like(rhs_arr)
            self._last_solve_info = {
                "iterations": int(nit),
                "time_s": float(elapsed),
                "basis_cols": int(basis.shape[1]) if basis.ndim == 2 else 0,
                "rhs_norm": rhs_norm,
                "coarse_initial_guess_norm": float(np.linalg.norm(coarse_guess)),
                "reported_residual_history": np.asarray(res_hist, dtype=np.float64).tolist(),
                "true_residual_history": np.asarray(res_hist, dtype=np.float64).tolist(),
                "true_residual_final": float(np.linalg.norm(resid) / rhs_norm),
                "reported_residual_final": float(np.asarray(res_hist, dtype=np.float64).reshape(-1)[-1]) if np.asarray(res_hist).size else None,
                "hit_max_iterations": bool(int(nit) >= int(self.max_iterations)),
                "converged": bool(float(np.linalg.norm(resid) / rhs_norm) <= float(self.tolerance)),
                "converged_reason": None,
                "timings": {k: float(v) for k, v in timing_stats.items()},
            }
        if self.verbose:
            print(f"{nit}|", end="")
        return self._restrict_solution(x_total)

    def copy(self):
        clone = self.__class__(
            self.pc_type,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
            tolerance_deflation_basis=self.tolerance_deflation_basis,
            verbose=self.verbose,
            q_mask=self.q_mask,
            coord=self.coord,
            preconditioner_options=self.preconditioner_options,
        )
        # Share the current basis until the clone appends/reorthogonalizes and rebinds it.
        clone.deflation_basis = self.deflation_basis
        clone.iteration_collector = self.iteration_collector
        clone.instance_id = self.iteration_collector.register_instance()
        clone._diagnostics_enabled = self._diagnostics_enabled
        clone._preconditioner_diagnostics = self._preconditioner_diagnostics
        return clone


class SolverFactory:
    """Factory for creating solver backends from a compact configuration."""

    @staticmethod
    def create(
        solver_type: str = "DFGMRES_GAMG",
        *,
        tolerance: float = 1e-1,
        max_iterations: int = 200,
        deflation_basis_tolerance: float = 1e-3,
        verbose: bool = False,
        q_mask: np.ndarray | None = None,
        coord: np.ndarray | None = None,
        preconditioner_options: dict | None = None,
    ):
        if preconditioner_options is None:
            preconditioner_options = {}

        solver_type = str(solver_type).upper()
        q_mask = np.array([], dtype=bool) if q_mask is None else np.asarray(q_mask, dtype=bool)

        def _pc_builder(kind: str):
            return lambda A: build_preconditioner(
                kind,
                A,
                q_mask=q_mask,
                coord=coord,
                **preconditioner_options,
            ).apply

        if solver_type == "DIRECT":
            return ScipyDirectSolver()

        if solver_type in {"PETSC_DIRECT", "KSPDIRECT", "KSPDIRECT_LU", "KSPPREONLY_LU"}:
            return DirectSolver(
                factor_solver_type=preconditioner_options.get("factor_solver_type"),
            )

        if solver_type == "FGMRES":
            return FGMRESSolver(
                _pc_builder("JACOBI"),
                tolerance,
                max_iterations,
                deflation_basis_tolerance,
                verbose,
            )

        if solver_type.startswith("PETSC_MATLAB_DFGMRES"):
            pc_type = "GAMG"
            if "HYPRE" in solver_type:
                pc_type = "HYPRE"
            if "JACOBI" in solver_type:
                pc_type = "JACOBI"
            return PetscMatlabExactDFGMRESSolver(
                pc_type,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=preconditioner_options,
            )

        if solver_type.startswith("KSPFGMRES"):
            if "MATLAB" in solver_type or "DFGMRES" in solver_type:
                pc_type = "GAMG"
                if "HYPRE" in solver_type:
                    pc_type = "HYPRE"
                if "JACOBI" in solver_type:
                    pc_type = "JACOBI"
                solver_cls = PetscKSPMatlabDeflatedFGMRESSolver
                if "REORTH" in solver_type:
                    solver_cls = PetscKSPMatlabDeflatedFGMRESReorthSolver
                return solver_cls(
                    pc_type,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    tolerance_deflation_basis=deflation_basis_tolerance,
                    verbose=verbose,
                    q_mask=q_mask,
                    coord=coord,
                    preconditioner_options=preconditioner_options,
                )
            pc_type = "GAMG"
            if "HYPRE" in solver_type:
                pc_type = "HYPRE"
            if "JACOBI" in solver_type:
                pc_type = "JACOBI"
            return PetscKSPFGMRESSolver(
                pc_type,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=preconditioner_options,
            )

        if "DEFLATION" in solver_type and solver_type.startswith("KSPGMRES"):
            pc_type = "GAMG"
            if "HYPRE" in solver_type:
                pc_type = "HYPRE"
            if "JACOBI" in solver_type:
                pc_type = "JACOBI"
            return PetscKSPGMRESDeflationSolver(
                pc_type,
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
                q_mask=q_mask,
                coord=coord,
                preconditioner_options=preconditioner_options,
            )

        # default to GAMG-preconditioned deflated FGMRES
        if "GAMG" in solver_type:
            return DeflatedFGMRESSolver(
                _pc_builder("GAMG"),
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
            )

        if "JACOBI" in solver_type:
            return DeflatedFGMRESSolver(
                _pc_builder("JACOBI"),
                tolerance=tolerance,
                max_iterations=max_iterations,
                tolerance_deflation_basis=deflation_basis_tolerance,
                verbose=verbose,
            )

        if "DIRECT" in solver_type:
            return DirectSolver()

        raise ValueError(f"Unsupported solver_type {solver_type!r}")

    @staticmethod
    def from_config(config: LinearSolverConfig, *, q_mask: np.ndarray | None = None, coord: np.ndarray | None = None):
        preconditioner_options = {
            "threads": config.threads,
            "print_level": config.print_level,
            "use_as_preconditioner": config.use_as_preconditioner,
            "pc_backend": config.pc_backend,
            "preconditioner_matrix_source": config.preconditioner_matrix_source,
            "preconditioner_matrix_policy": config.preconditioner_matrix_policy,
            "preconditioner_rebuild_policy": config.preconditioner_rebuild_policy,
            "preconditioner_rebuild_interval": config.preconditioner_rebuild_interval,
        }
        if config.factor_solver_type is not None:
            preconditioner_options["factor_solver_type"] = config.factor_solver_type
        if config.pc_gamg_process_eq_limit is not None:
            preconditioner_options["pc_gamg_process_eq_limit"] = config.pc_gamg_process_eq_limit
        if config.pc_gamg_threshold is not None:
            preconditioner_options["pc_gamg_threshold"] = config.pc_gamg_threshold
        if config.pc_gamg_aggressive_coarsening is not None:
            preconditioner_options["pc_gamg_aggressive_coarsening"] = config.pc_gamg_aggressive_coarsening
        if config.pc_gamg_aggressive_square_graph is not None:
            preconditioner_options["pc_gamg_aggressive_square_graph"] = config.pc_gamg_aggressive_square_graph
        if config.pc_gamg_aggressive_mis_k is not None:
            preconditioner_options["pc_gamg_aggressive_mis_k"] = config.pc_gamg_aggressive_mis_k
        if config.pc_hypre_coarsen_type is not None:
            preconditioner_options["pc_hypre_boomeramg_coarsen_type"] = config.pc_hypre_coarsen_type
        if config.pc_hypre_interp_type is not None:
            preconditioner_options["pc_hypre_boomeramg_interp_type"] = config.pc_hypre_interp_type
        if config.pc_hypre_strong_threshold is not None:
            preconditioner_options["pc_hypre_boomeramg_strong_threshold"] = config.pc_hypre_strong_threshold
        if getattr(config, "pc_hypre_boomeramg_max_iter", None) is not None:
            preconditioner_options["pc_hypre_boomeramg_max_iter"] = int(config.pc_hypre_boomeramg_max_iter)
        if config.pc_hypre_P_max is not None:
            preconditioner_options["pc_hypre_boomeramg_P_max"] = config.pc_hypre_P_max
        if config.pc_hypre_agg_nl is not None:
            preconditioner_options["pc_hypre_boomeramg_agg_nl"] = config.pc_hypre_agg_nl
        if config.pc_hypre_nongalerkin_tol is not None:
            preconditioner_options["pc_hypre_boomeramg_nongalerkin_tol"] = config.pc_hypre_nongalerkin_tol
        if config.pc_bddc_symmetric is not None:
            preconditioner_options["pc_bddc_symmetric"] = bool(config.pc_bddc_symmetric)
        if config.pc_bddc_dirichlet_ksp_type is not None:
            preconditioner_options["pc_bddc_dirichlet_ksp_type"] = config.pc_bddc_dirichlet_ksp_type
        if config.pc_bddc_dirichlet_pc_type is not None:
            preconditioner_options["pc_bddc_dirichlet_pc_type"] = config.pc_bddc_dirichlet_pc_type
        if config.pc_bddc_neumann_ksp_type is not None:
            preconditioner_options["pc_bddc_neumann_ksp_type"] = config.pc_bddc_neumann_ksp_type
        if config.pc_bddc_neumann_pc_type is not None:
            preconditioner_options["pc_bddc_neumann_pc_type"] = config.pc_bddc_neumann_pc_type
        if config.pc_bddc_coarse_ksp_type is not None:
            preconditioner_options["pc_bddc_coarse_ksp_type"] = config.pc_bddc_coarse_ksp_type
        if config.pc_bddc_coarse_pc_type is not None:
            preconditioner_options["pc_bddc_coarse_pc_type"] = config.pc_bddc_coarse_pc_type
        if config.pc_bddc_dirichlet_approximate is not None:
            preconditioner_options["pc_bddc_dirichlet_approximate"] = config.pc_bddc_dirichlet_approximate
        if config.pc_bddc_neumann_approximate is not None:
            preconditioner_options["pc_bddc_neumann_approximate"] = config.pc_bddc_neumann_approximate
        if getattr(config, "pc_bddc_monolithic", None) is not None:
            preconditioner_options["pc_bddc_monolithic"] = bool(config.pc_bddc_monolithic)
        if getattr(config, "pc_bddc_coarse_redundant_pc_type", None) is not None:
            preconditioner_options["pc_bddc_coarse_redundant_pc_type"] = config.pc_bddc_coarse_redundant_pc_type
        if config.pc_bddc_switch_static is not None:
            preconditioner_options["pc_bddc_switch_static"] = config.pc_bddc_switch_static
        if config.pc_bddc_use_deluxe_scaling is not None:
            preconditioner_options["pc_bddc_use_deluxe_scaling"] = config.pc_bddc_use_deluxe_scaling
        if config.pc_bddc_use_vertices is not None:
            preconditioner_options["pc_bddc_use_vertices"] = config.pc_bddc_use_vertices
        if config.pc_bddc_use_edges is not None:
            preconditioner_options["pc_bddc_use_edges"] = config.pc_bddc_use_edges
        if config.pc_bddc_use_faces is not None:
            preconditioner_options["pc_bddc_use_faces"] = config.pc_bddc_use_faces
        if config.pc_bddc_use_change_of_basis is not None:
            preconditioner_options["pc_bddc_use_change_of_basis"] = config.pc_bddc_use_change_of_basis
        if config.pc_bddc_use_change_on_faces is not None:
            preconditioner_options["pc_bddc_use_change_on_faces"] = config.pc_bddc_use_change_on_faces
        if config.pc_bddc_check_level is not None:
            preconditioner_options["pc_bddc_check_level"] = config.pc_bddc_check_level
        if config.compiled_outer:
            preconditioner_options["compiled_outer"] = True
        if config.recycle_preconditioner:
            preconditioner_options["recycle_preconditioner"] = True
        return SolverFactory.create(
            config.solver_type,
            tolerance=config.tolerance,
            max_iterations=config.max_iterations,
            deflation_basis_tolerance=config.deflation_basis_tolerance,
            verbose=bool(config.verbose),
            q_mask=q_mask,
            coord=coord,
            preconditioner_options=preconditioner_options,
        )


class FGMRESSolver(DeflatedFGMRESSolver):
    """FGMRES variant with deflation disabled."""

    def expand_deflation_basis(self, _additional_vectors):
        return
