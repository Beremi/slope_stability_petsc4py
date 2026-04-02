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
        self.deflation_basis = np.empty((0, 0), dtype=np.float64)

    def copy(self):
        return self

    def supports_dynamic_deflation_basis(self) -> bool:
        return True

    def expand_deflation_basis(self, values: np.ndarray) -> None:
        vec = np.asarray(values, dtype=np.float64).reshape(-1, 1)
        self.basis_vectors.append(vec.reshape(-1).copy())
        if self.deflation_basis.size == 0:
            self.deflation_basis = vec
        else:
            self.deflation_basis = np.hstack((self.deflation_basis, vec))

    def get_deflation_basis_snapshot(self):
        return np.array(self.deflation_basis, dtype=np.float64, copy=True)

    def restore_deflation_basis(self, snapshot) -> None:
        if snapshot is None:
            self.deflation_basis = np.empty((0, 0), dtype=np.float64)
        else:
            self.deflation_basis = np.array(snapshot, dtype=np.float64, copy=True)


class _DummyBuilder:
    def __init__(self) -> None:
        self.current_lambda = np.nan
        self.lambda_history: list[float] = []

    def reduction(self, lambda_value: float) -> None:
        self.current_lambda = float(lambda_value)
        self.lambda_history.append(float(lambda_value))


class _LinearSecantCorrectionBuilder:
    def __init__(self) -> None:
        self.K = np.array([[1.0, 0.0], [-1.0, 1.0]], dtype=np.float64)

    def build_F_K_tangent_all(self, lambda_value: float, U: np.ndarray):
        U_arr = np.asarray(U, dtype=np.float64)
        flat = U_arr.reshape(-1, order="F")
        F_flat = self.K @ flat
        return F_flat.reshape(U_arr.shape, order="F"), self.K.copy()


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
        **kwargs,
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


def test_init_phase_absolute_delta_lambda_uses_residual_stop_for_plain_newton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _DummyBuilder()
    solver = _DummyContinuationSolver()
    stopping_calls: list[tuple[str | None, float | None]] = []

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
        **kwargs,
    ):
        stopping_calls.append((kwargs.get("stopping_criterion"), kwargs.get("stopping_tol")))
        lam = float(constitutive_matrix_builder.current_lambda)
        return np.full_like(U_ini, lam, dtype=np.float64), 0, 1

    monkeypatch.setattr(indirect_module, "newton", _fake_newton)

    indirect_module.init_phase_SSR_indirect_continuation(
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
        newton_stopping_criterion="absolute_delta_lambda",
        newton_stopping_tol=1.0e-2,
    )

    assert stopping_calls == [("relative_residual", None), ("relative_residual", None)]


def test_init_phase_uses_init_stop_overrides_and_half_lambda_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = _DummyBuilder()
    solver = _DummyContinuationSolver()
    stopping_calls: list[tuple[str | None, float | None, float]] = []

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
        **kwargs,
    ):
        lam = float(constitutive_matrix_builder.current_lambda)
        stopping_calls.append((kwargs.get("stopping_criterion"), kwargs.get("stopping_tol"), lam))
        return np.full_like(U_ini, lam, dtype=np.float64), 0, 1

    monkeypatch.setattr(indirect_module, "newton", _fake_newton)

    _U1, _U2, _omega1, _omega2, lambda1, lambda2, _all_newton_its = indirect_module.init_phase_SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.05,
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
        newton_stopping_criterion="absolute_delta_lambda",
        newton_stopping_tol=1.0e-2,
        init_newton_stopping_criterion="relative_correction",
        init_newton_stopping_tol=1.0e-2,
    )

    assert stopping_calls == [
        ("relative_correction", 1.0e-2, 1.0),
        ("relative_correction", 1.0e-2, 1.05),
    ]
    assert lambda1 == pytest.approx(1.0)
    assert lambda2 == pytest.approx(1.05)


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
        **kwargs,
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
        **kwargs,
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


def test_streaming_micro_domega_matches_reference_segment_scaling() -> None:
    domega, alpha_sec, ds = indirect_module._streaming_micro_domega(
        omega_prev=1.0,
        omega_curr=2.0,
        lambda_prev=1.0,
        lambda_curr=2.0,
        omega_scale=1.0,
        lambda_scale=1.0,
        s_micro=0.5,
        omega_remaining=10.0,
    )

    assert domega == pytest.approx(0.5 / np.sqrt(2.0))
    assert alpha_sec == pytest.approx(0.5 / np.sqrt(2.0))
    assert ds == pytest.approx(0.5)


def test_initial_segment_length_cap_matches_first_step_limit() -> None:
    domega, raw_length, length_limit = indirect_module._initial_segment_length_cap(
        domega_candidate=2.0,
        domega_initial=1.0,
        dlambda_initial=0.1,
        omega_anchor_prev=2.0,
        omega_anchor_curr=3.0,
        lambda_anchor_prev=1.1,
        lambda_anchor_curr=1.2,
        cap_factor=1.0,
    )

    assert domega == pytest.approx(1.0)
    assert raw_length == pytest.approx(2.0 * np.sqrt(2.0))
    assert length_limit == pytest.approx(np.sqrt(2.0))


def test_history_box_length_cap_matches_moving_box_definition() -> None:
    domega, raw_length, length_limit = indirect_module._history_box_step_length_cap(
        domega_candidate=3.0,
        omega_hist=np.array([10.0, 11.0, 12.0], dtype=np.float64),
        lambda_hist=np.array([1.0, 1.1, 1.2], dtype=np.float64),
        omega_anchor_prev=11.0,
        omega_anchor_curr=12.0,
        lambda_anchor_prev=1.1,
        lambda_anchor_curr=1.2,
        cap_factor=1.0,
    )

    assert domega == pytest.approx(1.0)
    assert raw_length == pytest.approx(3.0 / np.sqrt(2.0))
    assert length_limit == pytest.approx(1.0 / np.sqrt(2.0))


def test_history_box_path_and_projected_lengths_use_current_box_scaling() -> None:
    omega_hist = np.array([10.0, 11.0, 12.0], dtype=np.float64)
    lambda_hist = np.array([1.0, 1.1, 1.2], dtype=np.float64)

    path_length = indirect_module._history_box_path_length(
        omega_hist=omega_hist,
        lambda_hist=lambda_hist,
        start_idx=0,
        end_idx=2,
    )
    projected_length = indirect_module._history_box_projected_length(
        domega_candidate=1.0,
        omega_hist=omega_hist,
        lambda_hist=lambda_hist,
        omega_anchor_prev=11.0,
        omega_anchor_curr=12.0,
        lambda_anchor_prev=1.1,
        lambda_anchor_curr=1.2,
    )

    assert path_length == pytest.approx(np.sqrt(2.0))
    assert projected_length == pytest.approx(1.0 / np.sqrt(2.0))


def test_ssr_indirect_hybrid_fine_switch_uses_crossing_step_and_resets_after_fine(
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
            1.05,
            [2, 2],
        )

    call_log: list[tuple[str, float | None]] = []

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
        **kwargs,
    ):
        call_log.append((str(kwargs.get("stopping_criterion")), kwargs.get("stopping_tol")))
        history = {
            "residual": np.array([1.0e-4], dtype=np.float64),
            "accepted_relative_correction_norm": np.array([1.0e-3], dtype=np.float64),
            "first_iteration_linear_iterations": 1,
            "first_iteration_linear_solve_time": 0.0,
            "first_iteration_linear_preconditioner_time": 0.0,
            "first_iteration_linear_orthogonalization_time": 0.0,
            "first_accepted_correction_norm": 1.0,
        }
        idx = len(call_log)
        if idx == 1:
            return np.array([[float(omega_it)]], dtype=np.float64), float(lambda_init) + 0.03, 0, 2, history
        if idx == 2:
            return np.array([[float(omega_it)]], dtype=np.float64), float(lambda_init) + 0.02, 1, 1, history
        if idx == 3:
            return np.array([[float(omega_it)]], dtype=np.float64), float(lambda_init) + 0.02, 0, 2, history
        return np.array([[float(omega_it)]], dtype=np.float64), float(lambda_init) + 0.01, 0, 1, history

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)
    monkeypatch.setattr(indirect_module, "_history_box_initial_segment_length", lambda **kwargs: 1.0)
    monkeypatch.setattr(indirect_module, "_history_box_path_length", lambda **kwargs: 0.75 * (kwargs["end_idx"] - kwargs["start_idx"]))
    monkeypatch.setattr(indirect_module, "_history_box_projected_length", lambda **kwargs: 1.5)

    _U_last, _lambda_hist, _omega_hist, _Umax_hist, stats = indirect_module.SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.05,
        d_lambda_min=1.0e-3,
        d_lambda_diff_scaled_min=-1.0,
        step_max=5,
        omega_max_stop=20.0,
        it_newt_max=20,
        it_damp_max=5,
        tol=1.0e-4,
        r_min=1.0e-4,
        K_elast=np.eye(1, dtype=np.float64),
        Q=np.ones((1, 1), dtype=bool),
        f=np.ones((1, 1), dtype=np.float64),
        constitutive_matrix_builder=object(),
        linear_system_solver=solver,
        store_step_u=False,
        step_length_cap_mode="history_box",
        step_length_cap_factor=1.0,
        newton_stopping_criterion="absolute_delta_lambda",
        newton_stopping_tol=1.0e-2,
        fine_newton_stopping_criterion="absolute_delta_lambda",
        fine_newton_stopping_tol=1.0e-3,
        fine_switch_mode="history_box_cumulative_distance",
        fine_switch_distance_factor=2.0,
    )

    assert call_log == [
        ("absolute_delta_lambda", 1.0e-2),
        ("absolute_delta_lambda", 1.0e-3),
        ("absolute_delta_lambda", 1.0e-3),
        ("absolute_delta_lambda", 1.0e-2),
    ]
    assert list(stats["step_precision_mode"]) == ["rough", "fine", "rough"]
    assert list(stats["step_fine_switch_triggered"]) == [False, True, False]
    assert list(stats["step_fine_reference_step"]) == [2, 2, 4]
    assert list(stats["attempt_fine_reference_step"][:3]) == [2, 2, 2]


def test_ssr_indirect_zero_flat_stop_threshold_disables_branch_flatness_stop(
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
            1.1,
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
        **kwargs,
    ):
        call_count["value"] += 1
        history = {
            "residual": np.array([1.0e-4], dtype=np.float64),
            "accepted_relative_correction_norm": np.array([1.0e-3], dtype=np.float64),
            "first_iteration_linear_iterations": 1,
            "first_iteration_linear_solve_time": 0.0,
            "first_iteration_linear_preconditioner_time": 0.0,
            "first_iteration_linear_orthogonalization_time": 0.0,
            "first_accepted_correction_norm": 1.0,
        }
        if call_count["value"] == 1:
            return np.array([[float(omega_it)]], dtype=np.float64), 1.08, 0, 2, history
        return np.array([[float(omega_it)]], dtype=np.float64), 1.06, 0, 2, history

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)

    _U_last, _lambda_hist, omega_hist, _Umax_hist, stats = indirect_module.SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.1,
        d_lambda_min=1.0e-3,
        d_lambda_diff_scaled_min=0.0,
        step_max=4,
        omega_max_stop=10.0,
        it_newt_max=20,
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

    assert call_count["value"] == 2
    np.testing.assert_allclose(omega_hist[:4], np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float64))
    assert list(stats["step_index"]) == [3, 4]


def test_ssr_indirect_streaming_microstep_accepts_after_reaching_unit_arc_length(
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
            2.0,
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
        first_iteration_extra_basis_free=None,
        **kwargs,
    ):
        U_out = np.array([[float(omega_it)]], dtype=np.float64)
        history = {
            "residual": np.array([1.0e-6], dtype=np.float64),
            "alpha": np.array([1.0], dtype=np.float64),
            "first_iteration_linear_iterations": 1,
            "first_iteration_linear_solve_time": 0.0,
            "first_iteration_linear_preconditioner_time": 0.0,
            "first_iteration_linear_orthogonalization_time": 0.0,
            "first_accepted_correction_norm": 1.0,
        }
        return U_out, float(omega_it), 1, 1, history

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)

    _U_last, _lambda_hist, omega_hist, _Umax_hist, stats = indirect_module.SSR_indirect_continuation(
        lambda_init=1.0,
        d_lambda_init=0.1,
        d_lambda_min=1.0e-3,
        d_lambda_diff_scaled_min=-1.0,
        step_max=3,
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
        continuation_mode="streaming_microstep",
        streaming_micro_target_length=1.0,
        streaming_micro_min_length=1.0,
        streaming_micro_max_length=1.0,
        streaming_move_relres_threshold=1.0e-3,
        streaming_alpha_advance_threshold=0.5,
        streaming_micro_max_corrections=10,
    )

    expected_omega3 = 2.0 + 1.0 / np.sqrt(2.0)
    assert omega_hist[2] == pytest.approx(expected_omega3)
    assert int(stats["step_index"][-1]) == 3
    assert int(stats["step_micro_attempt_count"][-1]) == 2


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
        **kwargs,
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
        **kwargs,
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


def test_secant_orthogonal_increment_ls_predictor_rescales_target_omega_and_trust_clips() -> None:
    builder = _LinearSecantCorrectionBuilder()
    U_pred, lambda_ini, kind, info = indirect_module._secant_orthogonal_increment_ls_predictor(
        omega_old=0.0,
        omega=0.5,
        omega_target=1.0,
        U_old=np.array([[0.0, 0.0]], dtype=np.float64),
        U=np.array([[0.5, 0.0]], dtype=np.float64),
        lambda_value=1.2,
        Q=np.ones((1, 2), dtype=bool),
        f=np.array([[1.0, 0.0]], dtype=np.float64),
        K_elast=np.eye(2, dtype=np.float64),
        constitutive_matrix_builder=builder,
        continuation_increment_free_hist=[
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.1, 1.0], dtype=np.float64),
        ],
        r_min=1.0e-4,
    )

    assert kind == "secant_orthogonal_increment_ls"
    assert lambda_ini == pytest.approx(1.2)
    assert bool(info["secant_correction_active"]) is True
    assert bool(info["secant_correction_trust_region_clipped"]) is True
    assert float(info["secant_correction_basis_dim"]) == pytest.approx(1.0)
    assert float(info["secant_correction_predicted_residual_decrease"]) > 1.0e-3
    assert np.dot(
        np.array([[1.0, 0.0]], dtype=np.float64).reshape(-1, order="F"),
        np.asarray(U_pred, dtype=np.float64).reshape(-1, order="F"),
    ) == pytest.approx(1.0)
    assert U_pred[0, 0] == pytest.approx(1.0)
    assert U_pred[0, 1] == pytest.approx(0.075, rel=1.0e-6, abs=1.0e-6)


def test_secant_orthogonal_increment_ls_predictor_falls_back_when_basis_empty() -> None:
    builder = _LinearSecantCorrectionBuilder()
    U_pred, lambda_ini, kind, info = indirect_module._secant_orthogonal_increment_ls_predictor(
        omega_old=0.0,
        omega=0.5,
        omega_target=1.0,
        U_old=np.array([[0.0, 0.0]], dtype=np.float64),
        U=np.array([[0.5, 0.0]], dtype=np.float64),
        lambda_value=1.2,
        Q=np.ones((1, 2), dtype=bool),
        f=np.array([[1.0, 0.0]], dtype=np.float64),
        K_elast=np.eye(2, dtype=np.float64),
        constitutive_matrix_builder=builder,
        continuation_increment_free_hist=[np.array([1.0, 0.0], dtype=np.float64)],
        r_min=1.0e-4,
    )

    assert kind == "secant_orthogonal_increment_ls_fallback_secant"
    assert lambda_ini == pytest.approx(1.2)
    assert bool(info["fallback_used"]) is True
    assert "empty_secant_orthogonal_increment_basis" in str(info["fallback_error"])
    np.testing.assert_allclose(U_pred, np.array([[1.0, 0.0]], dtype=np.float64))


def test_ssr_indirect_history_deflation_warm_start_activates_after_first_accepted_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = _DummyContinuationSolver()
    warm_start_dims: list[int] = []

    def _fake_init(*args, **kwargs):
        return (
            np.array([[0.0]], dtype=np.float64),
            np.array([[1.0]], dtype=np.float64),
            0.0,
            1.0,
            1.0,
            1.0,
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
        first_iteration_extra_basis_free=None,
        **kwargs,
    ):
        warm_start_dims.append(0 if first_iteration_extra_basis_free is None else len(first_iteration_extra_basis_free))
        U_ini_arr = np.asarray(U_ini, dtype=np.float64)
        return (
            U_ini_arr + 1.0,
            float(lambda_init),
            0,
            2,
            {
                "residual": np.array([1.0e-4], dtype=np.float64),
                "first_iteration_warm_start_active": bool(first_iteration_extra_basis_free),
                "first_iteration_warm_start_basis_dim": 0
                if first_iteration_extra_basis_free is None
                else len(first_iteration_extra_basis_free),
                "first_iteration_linear_iterations": 7,
                "first_iteration_linear_solve_time": 0.25,
                "first_iteration_linear_preconditioner_time": 0.1,
                "first_iteration_linear_orthogonalization_time": 0.05,
                "first_accepted_correction_norm": 2.0,
                "first_accepted_correction_free": np.array([2.0], dtype=np.float64),
            },
        )

    monkeypatch.setattr(indirect_module, "init_phase_SSR_indirect_continuation", _fake_init)
    monkeypatch.setattr(indirect_module, "newton_ind_ssr", _fake_newton_ind_ssr)

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
        continuation_first_newton_warm_start_mode="history_deflation",
    )

    assert warm_start_dims == [0, 1]
    np.testing.assert_allclose(
        np.asarray(stats["step_first_newton_warm_start_active"], dtype=bool),
        np.array([False, True], dtype=bool),
    )
    np.testing.assert_allclose(
        np.asarray(stats["step_first_newton_warm_start_basis_dim"], dtype=np.float64),
        np.array([0.0, 1.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(stats["step_first_newton_linear_iterations"], dtype=np.float64),
        np.array([7.0, 7.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(stats["step_first_newton_correction_norm"], dtype=np.float64),
        np.array([2.0, 2.0], dtype=np.float64),
    )
