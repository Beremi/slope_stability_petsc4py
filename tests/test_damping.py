from __future__ import annotations

import numpy as np

from slope_stability.nonlinear.damping import damping_alg5


class _LambdaSensitiveBuilder:
    def __init__(self) -> None:
        self.tried_lambdas: list[float] = []

    def build_F_all_free_local(self, lambda_value: float, U_alpha) -> np.ndarray:
        lam = float(lambda_value)
        self.tried_lambdas.append(lam)
        if lam <= 0.0:
            raise ValueError("Reduction parameter lambda must be positive")
        return np.zeros(1, dtype=np.float64)


def test_damping_alg5_rejects_nonpositive_lambda_trials_and_keeps_halving() -> None:
    builder = _LambdaSensitiveBuilder()

    alpha = damping_alg5(
        it_damp_max=5,
        U_it=np.zeros((1, 1), dtype=np.float64),
        lambda_it=0.1,
        d_U=np.ones((1, 1), dtype=np.float64),
        d_l=-0.4,
        f=np.zeros((1, 1), dtype=np.float64),
        criterion=1.0,
        q_mask=np.ones((1, 1), dtype=bool),
        constitutive_matrix_builder=builder,
        f_local_free=np.zeros(1, dtype=np.float64),
        comm=None,
    )

    assert alpha == 0.125
    assert builder.tried_lambdas == [0.05]


def test_damping_alg5_returns_zero_when_damping_budget_never_reaches_positive_lambda() -> None:
    builder = _LambdaSensitiveBuilder()

    alpha = damping_alg5(
        it_damp_max=5,
        U_it=np.zeros((1, 1), dtype=np.float64),
        lambda_it=0.01,
        d_U=np.ones((1, 1), dtype=np.float64),
        d_l=-10.0,
        f=np.zeros((1, 1), dtype=np.float64),
        criterion=1.0,
        q_mask=np.ones((1, 1), dtype=bool),
        constitutive_matrix_builder=builder,
        f_local_free=np.zeros(1, dtype=np.float64),
        comm=None,
    )

    assert alpha == 0.0
    assert builder.tried_lambdas == []


class _RaisingBuilder:
    def build_F_all(self, lambda_value: float, U_alpha) -> np.ndarray:
        raise ValueError("some other constitutive failure")


def test_damping_alg5_reraises_unrelated_builder_errors() -> None:
    try:
        damping_alg5(
            it_damp_max=2,
            U_it=np.zeros((1, 1), dtype=np.float64),
            lambda_it=1.0,
            d_U=np.ones((1, 1), dtype=np.float64),
            d_l=-0.1,
            f=np.zeros((1, 1), dtype=np.float64),
            criterion=1.0,
            q_mask=np.ones((1, 1), dtype=bool),
            constitutive_matrix_builder=_RaisingBuilder(),
        )
    except ValueError as exc:
        assert "some other constitutive failure" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unrelated builder failure to be re-raised")
