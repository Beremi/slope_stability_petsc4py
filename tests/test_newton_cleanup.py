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
