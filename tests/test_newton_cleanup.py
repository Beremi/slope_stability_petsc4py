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
