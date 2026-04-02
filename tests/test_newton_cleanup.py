from __future__ import annotations

import importlib

import numpy as np

newton_module = importlib.import_module("slope_stability.nonlinear.newton")


class _DummyBuilder:
    def __init__(self, tangent_token) -> None:
        self.tangent_token = tangent_token

    def build_F_K_tangent_reduced(self, U):
        return np.zeros_like(U, dtype=np.float64), self.tangent_token


class _DummySolver:
    def A_orthogonalize(self, _A) -> None:
        raise AssertionError("solver path should not be entered when Newton converges before the linear solve")


class _DummyCachedBuilder:
    def __init__(self, tangent_token, regularized_token) -> None:
        self.tangent_token = tangent_token
        self.regularized_token = regularized_token
        self.owned_tangent_pattern = object()
        self._owned_tangent_mat = tangent_token
        self._owned_regularized_mat = regularized_token

    def build_F_K_tangent_reduced_free(self, U):
        return np.zeros(U.size, dtype=np.float64), self.tangent_token

    def build_F_reduced_free(self, U):
        return np.zeros(U.size, dtype=np.float64)

    def build_K_regularized(self, r):
        return self.regularized_token


class _DummyLinearSolver:
    def supports_a_orthogonalization(self) -> bool:
        return False

    def supports_dynamic_deflation_basis(self) -> bool:
        return False


class _WarmStartCollector:
    def get_total_iterations(self) -> int:
        return 0

    def get_total_solve_time(self) -> float:
        return 0.0

    def get_total_preconditioner_time(self) -> float:
        return 0.0

    def get_total_orthogonalization_time(self) -> float:
        return 0.0


class _WarmStartSolver:
    def __init__(self) -> None:
        self.iteration_collector = _WarmStartCollector()
        self.deflation_basis = np.empty((0, 0), dtype=np.float64)

    def supports_a_orthogonalization(self) -> bool:
        return True

    def A_orthogonalize(self, _A) -> None:
        return None

    def supports_dynamic_deflation_basis(self) -> bool:
        return True

    def expand_deflation_basis(self, additional_vectors) -> None:
        v = np.asarray(additional_vectors, dtype=np.float64).reshape(-1, 1)
        if self.deflation_basis.size == 0:
            self.deflation_basis = v
        else:
            self.deflation_basis = np.hstack((self.deflation_basis, v))

    def get_deflation_basis_snapshot(self):
        return np.array(self.deflation_basis, dtype=np.float64, copy=True)

    def restore_deflation_basis(self, snapshot) -> None:
        if snapshot is None:
            self.deflation_basis = np.empty((0, 0), dtype=np.float64)
        else:
            self.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)


class _WarmStartBuilder:
    def build_F_K_tangent_all(self, lambda_value, U):
        U_arr = np.asarray(U, dtype=np.float64)
        return 0.5 * U_arr.copy(), 0.5 * np.eye(U_arr.size, dtype=np.float64)

    def build_F_all(self, lambda_value, U):
        return 0.5 * np.asarray(U, dtype=np.float64).copy()


def test_newton_early_convergence_cleans_pre_solve_tangent(monkeypatch) -> None:
    destroyed: list[object] = []

    def _record_destroy(obj) -> None:
        if obj is not None:
            destroyed.append(obj)

    monkeypatch.setattr(newton_module, "_destroy_petsc_mat", _record_destroy)

    tangent_token = object()
    U_ini = np.zeros((1, 1), dtype=np.float64)
    q_mask = np.ones((1, 1), dtype=bool)
    f = np.zeros((1, 1), dtype=np.float64)

    U_out, flag_N, iterations = newton_module.newton(
        U_ini,
        tol=1.0e-8,
        it_newt_max=3,
        it_damp_max=1,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=q_mask,
        f=f,
        constitutive_matrix_builder=_DummyBuilder(tangent_token),
        linear_system_solver=_DummySolver(),
    )

    assert flag_N == 0
    assert iterations == 1
    assert np.array_equal(U_out, U_ini)
    assert tangent_token in destroyed


def test_newton_postsolve_cleanup_preserves_builder_cached_mats(monkeypatch) -> None:
    destroyed: list[object] = []

    def _record_destroy(obj) -> None:
        if obj is not None:
            destroyed.append(obj)

    monkeypatch.setattr(newton_module, "_destroy_petsc_mat", _record_destroy)
    monkeypatch.setattr(newton_module, "_setup_linear_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(newton_module, "_solve_linear_system", lambda *args, **kwargs: np.zeros(1, dtype=np.float64))
    monkeypatch.setattr(newton_module, "_release_iteration_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(newton_module, "damping", lambda *args, **kwargs: 1.0)

    tangent_token = object()
    regularized_token = object()
    submatrix_token = object()
    monkeypatch.setattr(newton_module, "_combine_matrices", lambda *args, **kwargs: regularized_token)
    monkeypatch.setattr(newton_module, "extract_submatrix_free", lambda *args, **kwargs: submatrix_token)
    U_ini = np.zeros((1, 1), dtype=np.float64)
    q_mask = np.ones((1, 1), dtype=bool)
    f = np.ones((1, 1), dtype=np.float64)

    _U_out, flag_N, iterations = newton_module.newton(
        U_ini,
        tol=1.0e-8,
        it_newt_max=1,
        it_damp_max=1,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=q_mask,
        f=f,
        constitutive_matrix_builder=_DummyCachedBuilder(tangent_token, regularized_token),
        linear_system_solver=_DummyLinearSolver(),
    )

    assert flag_N == 1
    assert iterations == 1
    assert tangent_token not in destroyed
    assert regularized_token not in destroyed


def test_newton_ind_ssr_applies_first_iteration_warm_start_then_restores_basis(monkeypatch) -> None:
    observed_basis: list[np.ndarray] = []

    def _record_solve(linear_system_solver, A_free, b_free, *, b_full=None, free_idx=None):
        observed_basis.append(np.array(linear_system_solver.deflation_basis, dtype=np.float64, copy=True))
        return np.asarray(b_free, dtype=np.float64).reshape(-1)

    monkeypatch.setattr(newton_module, "_setup_linear_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(newton_module, "_solve_linear_system", _record_solve)
    monkeypatch.setattr(newton_module, "_release_iteration_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(newton_module, "damping_alg5", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(newton_module, "extract_submatrix_free", lambda A, free_idx: np.asarray(A, dtype=np.float64))

    solver = _WarmStartSolver()
    builder = _WarmStartBuilder()
    U_out, lambda_out, flag_N, iterations, history = newton_module.newton_ind_ssr(
        np.zeros((1, 1), dtype=np.float64),
        1.0,
        1.0,
        it_newt_max=2,
        it_damp_max=1,
        tol=1.0e-8,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=builder,
        linear_system_solver=solver,
        first_iteration_extra_basis_free=[np.array([7.0], dtype=np.float64)],
    )

    assert flag_N == 1
    assert iterations == 2
    assert lambda_out == 1.0
    np.testing.assert_allclose(U_out, np.array([[1.0]], dtype=np.float64))
    assert history["first_iteration_warm_start_active"] is True
    assert history["first_iteration_warm_start_basis_dim"] == 1
    assert history["first_accepted_correction_iteration"] == 1
    assert history["first_accepted_correction_norm"] == np.float64(1.0)
    assert observed_basis[0].shape == (1, 1)
    np.testing.assert_allclose(observed_basis[0][:, 0], np.array([7.0], dtype=np.float64))
    assert len(observed_basis) == 4
    assert all(not np.any(np.isclose(basis, 7.0)) for basis in observed_basis[2:])


def test_newton_ind_ssr_can_stop_on_absolute_delta_lambda(monkeypatch) -> None:
    monkeypatch.setattr(newton_module, "_setup_linear_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(newton_module, "_solve_linear_system", lambda linear_system_solver, A_free, b_free, **kwargs: np.asarray(b_free, dtype=np.float64).reshape(-1))
    monkeypatch.setattr(newton_module, "_release_iteration_resources", lambda *args, **kwargs: None)
    monkeypatch.setattr(newton_module, "damping_alg5", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(newton_module, "extract_submatrix_free", lambda A, free_idx: np.asarray(A, dtype=np.float64))

    solver = _WarmStartSolver()
    builder = _WarmStartBuilder()
    U_out, lambda_out, flag_N, iterations, history = newton_module.newton_ind_ssr(
        np.zeros((1, 1), dtype=np.float64),
        1.0,
        1.0,
        it_newt_max=4,
        it_damp_max=1,
        tol=1.0e-8,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=builder,
        linear_system_solver=solver,
        stopping_criterion="absolute_delta_lambda",
        stopping_tol=1.0e-12,
    )

    assert flag_N == 0
    assert iterations == 1
    assert history["stop_criterion"] == "absolute_delta_lambda"
    np.testing.assert_allclose(history["delta_lambda"], np.array([0.0], dtype=np.float64))
    np.testing.assert_allclose(U_out, np.array([[1.0]], dtype=np.float64))
    assert lambda_out == 1.0
