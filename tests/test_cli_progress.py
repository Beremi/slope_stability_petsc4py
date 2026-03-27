from __future__ import annotations

import io
import json

from slope_stability.cli.progress import make_progress_logger


def test_progress_logger_renders_compact_continuation_output(tmp_path) -> None:
    console = io.StringIO()
    logger = make_progress_logger(tmp_path, console=console)

    logger(
        {
            "event": "init_complete",
            "continuation_kind": "ssr_indirect",
            "phase": "init",
            "accepted_steps": 2,
            "lambda_hist": [1.0, 1.1],
            "omega_hist": [10.0, 25.0],
            "init_newton_iterations": [3, 2],
            "init_linear_iterations": 40,
            "total_wall_time": 1.25,
        }
    )
    logger(
        {
            "event": "newton_iteration",
            "continuation_kind": "ssr_indirect",
            "phase": "continuation",
            "target_step": 3,
            "accepted_steps": 2,
            "attempt_in_step": 1,
            "omega_target": 40.0,
            "lambda_before": 1.1,
            "iteration": 1,
            "criterion": 12.0,
            "rel_residual": 1.2e-2,
            "alpha": 1.0,
            "r": 1.0e-4,
            "lambda_value": 1.1,
            "accepted_delta_lambda": -0.01,
            "linear_iterations": 84,
            "linear_solve_time": 0.24,
            "linear_preconditioner_time": 0.05,
            "linear_orthogonalization_time": 0.01,
            "iteration_wall_time": 0.37,
            "status": "iterate",
        }
    )
    logger(
        {
            "event": "newton_iteration",
            "continuation_kind": "ssr_indirect",
            "phase": "continuation",
            "target_step": 3,
            "accepted_steps": 2,
            "attempt_in_step": 1,
            "omega_target": 40.0,
            "lambda_before": 1.1,
            "iteration": 2,
            "criterion": 1.5,
            "rel_residual": 2.5e-4,
            "alpha": None,
            "r": 1.0e-4,
            "lambda_value": 1.09,
            "accepted_delta_lambda": None,
            "linear_iterations": 0,
            "linear_solve_time": 0.0,
            "linear_preconditioner_time": 0.0,
            "linear_orthogonalization_time": 0.0,
            "iteration_wall_time": 0.02,
            "status": "converged",
        }
    )
    logger(
        {
            "event": "attempt_complete",
            "continuation_kind": "ssr_indirect",
            "phase": "continuation",
            "target_step": 3,
            "accepted_steps": 2,
            "attempt_in_step": 1,
            "success": False,
            "newton_iterations": 2,
            "newton_relres_end": 2.5e-4,
            "linear_iterations": 84,
            "attempt_wall_time": 0.41,
        }
    )
    logger(
        {
            "event": "newton_iteration",
            "continuation_kind": "ssr_indirect",
            "phase": "continuation",
            "target_step": 3,
            "accepted_steps": 2,
            "attempt_in_step": 2,
            "omega_target": 32.0,
            "lambda_before": 1.1,
            "iteration": 1,
            "criterion": 8.0,
            "rel_residual": 8.0e-3,
            "alpha": 0.5,
            "r": 2.0e-4,
            "lambda_value": 1.1,
            "accepted_delta_lambda": -0.005,
            "linear_iterations": 52,
            "linear_solve_time": 0.18,
            "linear_preconditioner_time": 0.03,
            "linear_orthogonalization_time": 0.01,
            "iteration_wall_time": 0.29,
            "status": "iterate",
        }
    )
    logger(
        {
            "event": "step_accepted",
            "continuation_kind": "ssr_indirect",
            "phase": "continuation",
            "accepted_step": 3,
            "step_attempt_count": 2,
            "lambda_value": 1.095,
            "d_lambda": -0.005,
            "d_lambda_diff_scaled": 0.0125,
            "omega_value": 32.0,
            "d_omega": 7.0,
            "u_max": 3.2,
            "step_newton_iterations_total": 3,
            "step_newton_relres_end": 8.0e-3,
            "step_linear_iterations": 136,
            "step_wall_time": 0.70,
        }
    )
    logger(
        {
            "event": "finished",
            "continuation_kind": "ssr_indirect",
            "phase": "continuation",
            "accepted_steps": 3,
            "lambda_last": 1.095,
            "omega_last": 32.0,
            "stop_reason": "d_lambda_diff_scaled_min",
            "total_wall_time": 2.0,
        }
    )

    output = console.getvalue()
    assert "[init]" in output
    assert output.count("[step 03 try 1]") == 1
    assert "[step 03 try 2]" in output
    assert "N01" in output
    assert "reject" in output
    assert "scaled=" in output
    assert "dlam=" in output
    assert "[done]" in output
    assert "rank 0" not in output.lower()

    latest = json.loads((tmp_path / "progress_latest.json").read_text(encoding="utf-8"))
    assert latest["event"] == "finished"
    assert latest["stop_reason"] == "d_lambda_diff_scaled_min"

    jsonl_lines = (tmp_path / "progress.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl_lines) == 7
