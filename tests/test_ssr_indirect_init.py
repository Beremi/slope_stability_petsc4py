from __future__ import annotations

import numpy as np
import pytest

from slope_stability.continuation import indirect as indirect_module


class _DummyCollector:
    def get_total_iterations(self) -> int:
        return 0

    def get_total_solve_time(self) -> float:
        return 0.0

    def get_total_preconditioner_time(self) -> float:
        return 0.0

    def get_total_orthogonalization_time(self) -> float:
        return 0.0


class _DummyContinuationSolver:
    def __init__(self) -> None:
        self.iteration_collector = _DummyCollector()
        self.basis_vectors: list[np.ndarray] = []

    def copy(self):
        return self

    def supports_dynamic_deflation_basis(self) -> bool:
        return True

    def expand_deflation_basis(self, values: np.ndarray) -> None:
        self.basis_vectors.append(np.asarray(values, dtype=np.float64))


class _DummyBuilder:
    def __init__(self) -> None:
        self.current_lambda = np.nan
        self.lambda_history: list[float] = []

    def reduction(self, lambda_value: float) -> None:
        self.current_lambda = float(lambda_value)
        self.lambda_history.append(float(lambda_value))


def test_init_phase_ssr_indirect_backs_off_initial_lambda(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _DummyBuilder()
    solver = _DummyContinuationSolver()
    attempted_lambdas: list[float] = []

    def _fake_newton(
        U_ini,
        tol,
        it_newt_max,
        it_damp_max,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver,
    ):
        lam = float(constitutive_matrix_builder.current_lambda)
        attempted_lambdas.append(lam)
        if lam > 0.35 and len(attempted_lambdas) == 1:
            return np.zeros_like(U_ini, dtype=np.float64), 1, 4
        return np.full_like(U_ini, lam, dtype=np.float64), 0, 2

    monkeypatch.setattr(indirect_module, "newton", _fake_newton)

    U1, U2, omega1, omega2, lambda1, lambda2, all_newton_its = indirect_module.init_phase_SSR_indirect_continuation(
        lambda_init=0.7,
        d_lambda_init=0.1,
        d_lambda_min=1.0e-5,
        it_newt_max=20,
        it_damp_max=5,
        tol=1.0e-4,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=builder,
        linear_system_solver=solver,
    )

    assert attempted_lambdas == pytest.approx([0.7, 0.35, 0.4])
    assert all_newton_its == [4, 2, 2]
    assert lambda1 == pytest.approx(0.35)
    assert lambda2 == pytest.approx(0.4)
    assert omega1 == pytest.approx(0.35)
    assert omega2 == pytest.approx(0.4)
    np.testing.assert_allclose(U1, np.array([[0.35]], dtype=np.float64))
    np.testing.assert_allclose(U2, np.array([[0.4]], dtype=np.float64))


def test_ssr_indirect_default_omega_controller_uses_legacy_branch_shape_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _DummyContinuationSolver()

    def _fake_init(*args, **kwargs):
        return (
            np.array([[1.0]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            1.0,
            2.0,
            1.0,
            0.9,
            [2, 2],
        )

    call_count = {"value": 0}

    def _fake_newton_ind_ssr(
        U_ini,
        omega_it,
        lambda_init,
        it_newt_max,
        it_damp_max,
        tol,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver,
        progress_callback=None,
    ):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return np.array([[3.0]], dtype=np.float64), 0.75, 0, 11, {"residual": np.array([1.0e-4])}
        return np.array([[4.0]], dtype=np.float64), 0.60, 0, 5, {"residual": np.array([1.0e-4])}

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)

    _, _, omega_hist, _, stats = indirect_module.SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.1,
        d_lambda_min=1.0e-3,
        d_lambda_diff_scaled_min=-1.0,
        step_max=4,
        omega_max_stop=10.0,
        it_newt_max=50,
        it_damp_max=5,
        tol=1.0e-4,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=object(),
        linear_system_solver=solver,
        store_step_u=False,
    )

    np.testing.assert_allclose(omega_hist[:4], np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float64))
    np.testing.assert_allclose(stats["step_d_omega_scale"], np.array([2.0, 2.0], dtype=np.float64))


def test_ssr_indirect_adaptive_omega_controller_uses_newton_target_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _DummyContinuationSolver()

    def _fake_init(*args, **kwargs):
        return (
            np.array([[1.0]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            1.0,
            2.0,
            1.0,
            0.9,
            [2, 2],
        )

    call_count = {"value": 0}

    def _fake_newton_ind_ssr(
        U_ini,
        omega_it,
        lambda_init,
        it_newt_max,
        it_damp_max,
        tol,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver,
        progress_callback=None,
    ):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return np.array([[3.0]], dtype=np.float64), 0.75, 0, 48, {"residual": np.array([1.0e-4])}
        return np.array([[3.5]], dtype=np.float64), 0.72, 0, 12, {"residual": np.array([1.0e-4])}

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)

    _, _, omega_hist, _, stats = indirect_module.SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.1,
        d_lambda_min=1.0e-3,
        d_lambda_diff_scaled_min=-1.0,
        step_max=4,
        omega_max_stop=10.0,
        it_newt_max=50,
        it_damp_max=5,
        tol=1.0e-4,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=object(),
        linear_system_solver=solver,
        store_step_u=False,
        omega_step_controller="adaptive",
        omega_target_newton_iterations=12.0,
        omega_adapt_min_scale=0.5,
        omega_adapt_max_scale=1.25,
    )

    np.testing.assert_allclose(omega_hist[:4], np.array([1.0, 2.0, 3.0, 3.5], dtype=np.float64))
    np.testing.assert_allclose(stats["step_d_omega_scale"], np.array([0.5, 1.0], dtype=np.float64))


@pytest.mark.parametrize(
    ("predictor_name", "window_size", "expected_kind"),
    [
        ("reduced_newton_all_prev", None, "reduced_newton_all_prev_projected"),
        ("reduced_newton_affine_all_prev", None, "reduced_newton_affine_all_prev_projected"),
        ("reduced_newton_window", 3, "reduced_newton_window_projected"),
        ("reduced_newton_increment_power", 2, "reduced_newton_increment_power_projected"),
    ],
)
def test_ssr_indirect_reduced_newton_predictor_dispatches_supported_modes(
    monkeypatch: pytest.MonkeyPatch,
    predictor_name: str,
    window_size: int | None,
    expected_kind: str,
) -> None:
    solver = _DummyContinuationSolver()

    def _fake_init(*args, **kwargs):
        return (
            np.array([[1.0]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            1.0,
            2.0,
            1.0,
            0.9,
            [2, 2],
        )

    def _fake_newton_ind_ssr(
        U_ini,
        omega_it,
        lambda_init,
        it_newt_max,
        it_damp_max,
        tol,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver,
        progress_callback=None,
    ):
        U_ini_arr = np.asarray(U_ini, dtype=np.float64)
        return U_ini_arr, float(lambda_init), 0, 3, {"residual": np.array([1.0e-4])}

    def _fake_projected_predictor(**kwargs):
        info = indirect_module._predictor_info_defaults()
        info["basis_dim"] = 2.0
        info["reduced_newton_iterations"] = 1.0
        return np.array([[3.0]], dtype=np.float64), float(kwargs["lambda_value"]) + 0.1, expected_kind, info

    def _fake_affine_predictor(**kwargs):
        info = indirect_module._predictor_info_defaults()
        info["basis_dim"] = 3.0
        info["state_coefficients"] = np.array([1.0, 0.0], dtype=np.float64)
        info["state_coefficients_ref"] = np.array([1.0, 0.0], dtype=np.float64)
        info["state_coefficient_sum"] = 1.0
        info["reduced_newton_iterations"] = 1.0
        return np.array([[3.0]], dtype=np.float64), float(kwargs["lambda_value"]) + 0.1, expected_kind, info

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)
    monkeypatch.setattr(indirect_module, "_projected_reduced_newton_predictor", _fake_projected_predictor)
    monkeypatch.setattr(indirect_module, "_affine_state_reduced_newton_predictor", _fake_affine_predictor)
    monkeypatch.setattr(indirect_module, "_increment_power_reduced_newton_predictor", _fake_projected_predictor)

    _, _, _, _, stats = indirect_module.SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.1,
        d_lambda_min=1.0e-3,
        d_lambda_diff_scaled_min=-1.0,
        step_max=4,
        omega_max_stop=10.0,
        it_newt_max=50,
        it_damp_max=5,
        tol=1.0e-4,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=object(),
        linear_system_solver=solver,
        store_step_u=False,
        continuation_predictor=predictor_name,
        continuation_predictor_window_size=window_size,
    )

    assert list(stats["step_predictor_kind"]) == [expected_kind, expected_kind]
    np.testing.assert_allclose(stats["step_predictor_reduced_newton_iterations"], np.array([1.0, 1.0], dtype=np.float64))


def test_ssr_indirect_reduced_newton_can_refine_lambda_from_current_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _DummyContinuationSolver()
    lambda_inputs: list[float] = []

    def _fake_init(*args, **kwargs):
        return (
            np.array([[1.0]], dtype=np.float64),
            np.array([[2.0]], dtype=np.float64),
            1.0,
            2.0,
            1.0,
            0.9,
            [2, 2],
        )

    def _fake_newton_ind_ssr(
        U_ini,
        omega_it,
        lambda_init,
        it_newt_max,
        it_damp_max,
        tol,
        r_min,
        K_elast,
        Q,
        f,
        constitutive_matrix_builder,
        linear_system_solver,
        progress_callback=None,
    ):
        lambda_inputs.append(float(lambda_init))
        U_ini_arr = np.asarray(U_ini, dtype=np.float64)
        return U_ini_arr, float(lambda_init), 0, 3, {"residual": np.array([1.0e-4])}

    def _fake_projected_predictor(**kwargs):
        info = indirect_module._predictor_info_defaults()
        info["basis_dim"] = 2.0
        info["reduced_newton_iterations"] = 2.0
        return np.array([[3.0]], dtype=np.float64), 1.8, "reduced_newton_all_prev_projected", info

    def _fake_refine_lambda(**kwargs):
        return 1.25, 0.5, 3

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)
    monkeypatch.setattr(indirect_module, "_projected_reduced_newton_predictor", _fake_projected_predictor)
    monkeypatch.setattr(indirect_module, "_refine_lambda_for_fixed_u_gauss_newton", _fake_refine_lambda)

    _, _, _, _, stats = indirect_module.SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.1,
        d_lambda_min=1.0e-3,
        d_lambda_diff_scaled_min=-1.0,
        step_max=4,
        omega_max_stop=10.0,
        it_newt_max=50,
        it_damp_max=5,
        tol=1.0e-4,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=object(),
        linear_system_solver=solver,
        store_step_u=False,
        continuation_predictor="reduced_newton_all_prev",
        continuation_predictor_use_projected_lambda=False,
        continuation_predictor_refine_lambda_for_fixed_u=True,
    )

    assert lambda_inputs == pytest.approx([1.25, 1.25])
    np.testing.assert_allclose(stats["step_predictor_energy_eval_count"], np.array([3.0, 3.0], dtype=np.float64))
