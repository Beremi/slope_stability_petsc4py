from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .telemetry import NewtonStats


class Slot(IntEnum):
    PREV = 0
    CUR = 1
    GUESS = 2
    TMP = 3


@dataclass(frozen=True, slots=True)
class MatrixHandle:
    name: str


@dataclass(frozen=True, slots=True)
class VectorHandle:
    name: str
    role: str


class DisplacementVector:
    def __init__(self, ops: "EngineOps", name: str, slot: Slot) -> None:
        self.ops = ops
        self.name = name
        self.slot = slot

    def zero(self) -> None:
        self.ops.zero(self.slot)

    def copy_from(self, other: "DisplacementVector") -> None:
        self.ops.copy(other.slot, self.slot)

    def omega(self) -> float:
        return self.ops.dot_omega(self.slot)

    def max_displacement(self) -> float:
        return self.ops.displacement_max(self.slot)

    def append_to_deflation_basis(self, label: str) -> None:
        self.ops.append_state_to_basis(self.slot, label)

    def secant_predict_from(self, U_old: "DisplacementVector", U: "DisplacementVector", *, alpha: float, work: "DisplacementVector") -> None:
        self.ops.secant_predict(previous=U_old.slot, current=U.slot, guess=self.slot, work=work.slot, alpha=alpha)


class DisplacementVectors:
    def __init__(self, ops: "EngineOps") -> None:
        self.U_old = DisplacementVector(ops, "U_old", Slot.PREV)
        self.U = DisplacementVector(ops, "U", Slot.CUR)
        self.U_ini = DisplacementVector(ops, "U_ini", Slot.GUESS)
        self.work = DisplacementVector(ops, "work", Slot.TMP)

    def at(self, slot: Slot, name: str = "U_it") -> DisplacementVector:
        return DisplacementVector(self.U_old.ops, name, slot)


class AlgebraVectors:
    F = VectorHandle("F", "residual")
    G = VectorHandle("G", "lambda_derivative")
    rhs = VectorHandle("-F", "fixed_rhs")
    rhs_W = VectorHandle("-G", "rhs_W")
    rhs_V = VectorHandle("f-F", "rhs_V")
    du = VectorHandle("dU_fixed", "du")
    W = VectorHandle("W", "W")
    V = VectorHandle("V", "V")
    d_U = VectorHandle("d_U", "d_U")


class Operators:
    K_elast = MatrixHandle("K_elast")
    K_tangent = MatrixHandle("K_tangent")
    K_r = MatrixHandle("K_r")

    def __init__(self, measured: "MeasuredNewtonOps") -> None:
        self.measured = measured

    def regularized_tangent(self, *, r: float, K_elast: MatrixHandle, K_tangent: MatrixHandle) -> MatrixHandle:
        del K_elast, K_tangent
        self.measured.build_regularized_operator(r)
        return self.K_r


class ConstitutiveMatrixBuilder:
    def __init__(self, measured: "MeasuredNewtonOps") -> None:
        self.measured = measured

    def build_F_K_tangent_all(self, lambda_it: float, U_it: DisplacementVector) -> dict[str, Any]:
        return self.measured.assemble_residual_jacobian(U_it.slot, lambda_it)

    def build_lambda_derivative(self, lambda_it: float, U_it: DisplacementVector, F: VectorHandle) -> VectorHandle:
        del F
        self.measured.compute_lambda_derivative(U_it.slot, lambda_it)
        return AlgebraVectors.G


class RhsBuilder:
    def __init__(self, measured: "MeasuredNewtonOps") -> None:
        self.measured = measured

    def fixed_lambda_rhs(self, F: VectorHandle) -> VectorHandle:
        del F
        self.measured.build_fixed_correction_rhs()
        return AlgebraVectors.rhs

    def indirect_rhs(self, G: VectorHandle, F: VectorHandle) -> tuple[VectorHandle, VectorHandle]:
        del G, F
        self.measured.build_indirect_rhs()
        return AlgebraVectors.rhs_W, AlgebraVectors.rhs_V


class LinearSolver:
    def __init__(self, measured: "MeasuredNewtonOps") -> None:
        self.measured = measured

    def setup_preconditioner(self, A: MatrixHandle, *, force_reuse: bool = False) -> None:
        del A
        self.measured.ksp_setup(force_reuse_preconditioner=force_reuse)

    def A_orthogonalize(self, A: MatrixHandle, *, label: str) -> None:
        del A
        self.measured.a_orthogonalize(label)

    def solve(self, A: MatrixHandle, rhs: VectorHandle, out: VectorHandle, *, label: str) -> dict[str, Any]:
        del A, rhs, label
        if out.role == "du":
            return self.measured.ksp_solve_fixed_correction()
        if out.role == "W":
            return self.measured.ksp_solve_indirect_w()
        if out.role == "V":
            return self.measured.ksp_solve_indirect_v()
        raise ValueError(f"Unsupported linear solve output vector {out.name!r}")


class NewtonAlgebra:
    def __init__(self, measured: "MeasuredNewtonOps") -> None:
        self.measured = measured

    def combine_indirect_directions(self, V: VectorHandle, W: VectorHandle) -> dict[str, Any]:
        del V, W
        return self.measured.form_indirect_update()


class Damping:
    def __init__(self, measured: "MeasuredNewtonOps") -> None:
        self.measured = measured

    def fixed_lambda_directional(self, U_it: DisplacementVector, lambda_it: float, du: VectorHandle, F: VectorHandle) -> dict[str, Any]:
        del du, F
        return self.measured.fixed_line_search(U_it.slot, lambda_it)

    def ALG5(self, U_it: DisplacementVector, lambda_it: float, d_U: VectorHandle, d_lambda: float, omega: float, criterion: float) -> dict[str, Any]:
        del d_U
        return self.measured.indirect_line_search(U_it.slot, lambda_it, omega, criterion, d_lambda)


class NewtonUpdate:
    def __init__(self, measured: "MeasuredNewtonOps") -> None:
        self.measured = measured

    def apply_fixed_lambda(self, U_it: DisplacementVector, alpha: float, r: float, *, update_basis: bool) -> dict[str, Any]:
        return self.measured.apply_fixed_correction(U_it.slot, alpha, r, update_basis=update_basis)

    def accept_indirect(self, U_it: DisplacementVector, lambda_it: float, omega: float, alpha: float, d_lambda: float, r: float, *, update_basis: bool) -> dict[str, Any]:
        return self.measured.accept_indirect_update(U_it.slot, lambda_it, omega, alpha, d_lambda, r, update_basis=update_basis)


class AlgorithmObjects:
    def __init__(self, ops: "EngineOps", stats: NewtonStats) -> None:
        measured = ops.collecting(stats)
        self.vectors = AlgebraVectors()
        self.operators = Operators(measured)
        self.constitutive_matrix_builder = ConstitutiveMatrixBuilder(measured)
        self.rhs_builder = RhsBuilder(measured)
        self.linear_solver = LinearSolver(measured)
        self.algebra = NewtonAlgebra(measured)
        self.damping = Damping(measured)
        self.update = NewtonUpdate(measured)


class EngineOps:
    """Small semantic wrapper over the C/Cython PETSc engine.

    The algorithm files use this vocabulary instead of raw vector slot and
    Cython method names. All vectors and matrices still live in PETSc/C.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def info(self) -> dict[str, Any]:
        return self._engine.info()

    def basis_cols(self) -> int:
        return int(self._engine.basis_cols())

    def truncate_basis(self, n_keep: int) -> None:
        self._engine.truncate_basis(n_keep)

    def append_state_to_basis(self, slot: Slot, label: str) -> None:
        self._engine.append_basis_from_slot(int(slot), label)

    def zero(self, slot: Slot) -> None:
        self._engine.vec_zero(int(slot))

    def copy(self, src: Slot, dst: Slot) -> None:
        self._engine.vec_copy(int(src), int(dst))

    def secant_predict(self, *, previous: Slot, current: Slot, guess: Slot, work: Slot, alpha: float) -> None:
        self.copy(current, guess)
        self._engine.vec_waxpy(int(work), -1.0, int(previous), int(current))
        self._engine.vec_axpy(int(guess), alpha, int(work))

    def dot_omega(self, slot: Slot) -> float:
        return float(self._engine.dot_omega(int(slot)))

    def displacement_max(self, slot: int | Slot) -> float:
        return float(self._engine.displacement_max(int(slot)))

    def write_solution(self, slot: int | Slot) -> None:
        self._engine.write_solution_from_slot(int(slot))

    def residual_rel(self, slot: Slot, lambda_value: float) -> float:
        return float(self._engine.residual_rel(int(slot), lambda_value))

    def solve_elastic_initial_guess(self, slot: Slot, scale: float) -> dict[str, Any]:
        return self._engine.solve_elastic_initial_guess(int(slot), scale)

    def collecting(self, stats: NewtonStats) -> "MeasuredNewtonOps":
        return MeasuredNewtonOps(self, stats)

    @property
    def displacement_vectors(self) -> DisplacementVectors:
        return DisplacementVectors(self)

    def algorithm_objects(self, stats: NewtonStats) -> AlgorithmObjects:
        return AlgorithmObjects(self, stats)


class MeasuredNewtonOps:
    """Newton-level operations with automatic timing and iteration collection."""

    def __init__(self, ops: EngineOps, stats: NewtonStats) -> None:
        self.ops = ops
        self.stats = stats

    def _record(self, phase: str, result: dict[str, Any], *, time_key: str = "wall_time") -> dict[str, Any]:
        self.stats.record_phase(phase, float(result.get(time_key, 0.0)))
        return result

    def assemble_residual_jacobian(self, slot: Slot, lambda_value: float) -> dict[str, Any]:
        result = self.ops._engine.assemble_residual_jacobian(int(slot), lambda_value)
        self.stats.assembly_time += float(result["assembly_time"])
        self.stats.final_rel = float(result["rel_residual"])
        self.stats.record_phase("assemble_residual_jacobian", float(result["assembly_time"]))
        return result

    def compute_lambda_derivative(self, slot: Slot, lambda_value: float) -> dict[str, Any]:
        result = self.ops._engine.compute_lambda_derivative(int(slot), lambda_value)
        self.stats.record_phase("compute_lambda_derivative", float(result["wall_time"]))
        return result

    def assemble_limit_load(self, slot: Slot, lambda_ell: float, load_t: float, r: float) -> dict[str, Any]:
        result = self.ops._engine.assemble_limit_load(int(slot), lambda_ell, load_t, r)
        self.stats.assembly_time += float(result["assembly_time"])
        self.stats.final_rel = float(result["rel_residual"])
        self.stats.record_phase("assemble_limit_load", float(result["assembly_time"]))
        return result

    def build_regularized_operator(self, r: float) -> dict[str, Any]:
        return self._record("build_regularized_operator", self.ops._engine.build_regularized_operator(r))

    def build_fixed_correction_rhs(self) -> dict[str, Any]:
        return self._record("build_fixed_correction_rhs", self.ops._engine.build_fixed_correction_rhs())

    def build_indirect_rhs(self) -> dict[str, Any]:
        return self._record("build_indirect_rhs", self.ops._engine.build_indirect_rhs())

    def build_limit_load_rhs(self, load_t: float) -> dict[str, Any]:
        return self._record("build_limit_load_rhs", self.ops._engine.build_limit_load_rhs(load_t))

    def ksp_setup(self, *, force_reuse_preconditioner: bool) -> dict[str, Any]:
        return self._record("ksp_setup", self.ops._engine.ksp_setup(force_reuse_preconditioner))

    def a_orthogonalize(self, label: str) -> dict[str, Any]:
        return self._record("a_orthogonalize", self.ops._engine.a_orthogonalize(label))

    def ksp_solve_fixed_correction(self) -> dict[str, Any]:
        result = self.ops._engine.ksp_solve_fixed_correction()
        self.stats.solve_time += float(result["solve_time"])
        self.stats.total_linear_its += int(result["linear_its"])
        self.stats.newton_its += int(result["newton_its"])
        self.stats.record_phase("ksp_solve_fixed_correction", float(result["solve_time"]))
        return result

    def fixed_line_search(self, slot: Slot, lambda_value: float) -> dict[str, Any]:
        result = self.ops._engine.fixed_line_search(int(slot), lambda_value)
        self.stats.line_search_its += int(result["line_search_its"])
        self.stats.final_rel_correction = float(result["rel_correction"])
        self.stats.record_phase("fixed_line_search", float(result["wall_time"]))
        return result

    def apply_fixed_correction(self, slot: Slot, alpha: float, r: float, *, update_basis: bool) -> dict[str, Any]:
        return self._record("apply_fixed_correction", self.ops._engine.apply_fixed_correction(int(slot), alpha, r, update_basis))

    def ksp_solve_indirect_w(self) -> dict[str, Any]:
        result = self.ops._engine.ksp_solve_indirect_w()
        self.stats.solve_time += float(result["solve_time"])
        self.stats.total_linear_its += int(result["linear_its"])
        self.stats.record_phase("ksp_solve_indirect_w", float(result["solve_time"]))
        return result

    def ksp_solve_indirect_v(self) -> dict[str, Any]:
        result = self.ops._engine.ksp_solve_indirect_v()
        self.stats.solve_time += float(result["solve_time"])
        self.stats.total_linear_its += int(result["linear_its"])
        self.stats.newton_its += 1
        self.stats.record_phase("ksp_solve_indirect_v", float(result["solve_time"]))
        return result

    def form_indirect_update(self) -> dict[str, Any]:
        return self._record("form_indirect_update", self.ops._engine.form_indirect_update())

    def form_limit_load_update(self) -> dict[str, Any]:
        return self._record("form_limit_load_update", self.ops._engine.form_limit_load_update())

    def indirect_line_search(self, slot: Slot, lambda_value: float, omega_target: float, current_rel: float, d_lambda: float) -> dict[str, Any]:
        result = self.ops._engine.indirect_line_search(int(slot), lambda_value, omega_target, current_rel, d_lambda)
        self.stats.line_search_its += int(result["line_search_its"])
        self.stats.final_rel_correction = float(result["rel_correction"])
        self.stats.record_phase("indirect_line_search", float(result["wall_time"]))
        return result

    def limit_load_line_search(self, slot: Slot, lambda_ell: float, load_t: float) -> dict[str, Any]:
        result = self.ops._engine.limit_load_line_search(int(slot), lambda_ell, load_t)
        self.stats.line_search_its += int(result["line_search_its"])
        self.stats.final_rel_correction = float(result["rel_correction"])
        self.stats.record_phase("limit_load_line_search", float(result["wall_time"]))
        return result

    def accept_indirect_update(
        self,
        slot: Slot,
        lambda_value: float,
        omega_target: float,
        alpha: float,
        d_lambda: float,
        r: float,
        *,
        update_basis: bool,
    ) -> dict[str, Any]:
        return self._record("accept_indirect_update", self.ops._engine.accept_indirect_update(int(slot), lambda_value, omega_target, alpha, d_lambda, r, update_basis))

    def accept_limit_load_update(
        self,
        slot: Slot,
        load_t: float,
        omega_target: float,
        alpha: float,
        d_t: float,
        r: float,
        *,
        update_basis: bool,
    ) -> dict[str, Any]:
        return self._record("accept_limit_load_update", self.ops._engine.accept_limit_load_update(int(slot), load_t, omega_target, alpha, d_t, r, update_basis))
