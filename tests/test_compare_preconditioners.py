from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "3d_hetero_ssr_default" / "compare_preconditioners.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_preconditioners_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load compare_preconditioners module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _baseline_metrics() -> dict[str, float]:
    return {
        "runtime_seconds": 100.0,
        "final_accepted_states": 12,
        "accepted_continuation_advances": 6,
        "lambda_last": 1.0,
        "omega_last": 2.0,
        "umax_last": 3.0,
        "preconditioner_time_total": 40.0,
        "linear_solve_time_total": 20.0,
        "peak_rss_gib": 10.0,
        "init_linear_preconditioner_time": 5.0,
        "attempt_linear_preconditioner_time_total": 35.0,
        "init_linear_solve_time": 2.0,
        "attempt_linear_solve_time_total": 18.0,
    }


def test_final_report_handles_reused_baseline_and_gated_out_bddc() -> None:
    module = _load_module()
    summary_payload = {
        "screening": {"hypre_current": {}, "gamg_lagged_lowmem": {}},
        "promoted_aij_variants": [],
        "best_aij_name": "hypre_current",
        "bddc_gate": {"status": "failed", "eligible_for_full_trajectory": False},
        "full_compare": {
            "baseline": {
                "name": "hypre_current",
                "status": "reused_baseline",
                "metrics": _baseline_metrics(),
                "reused_baseline": True,
            },
            "best_aij": {
                "name": "hypre_current",
                "status": "reused_baseline",
                "metrics": _baseline_metrics(),
                "reused_baseline": True,
            },
            "bddc": {
                "name": "bddc",
                "status": "not_run_gate_failed",
                "reason": "gate failed",
            },
        },
    }

    lines = module._final_report_lines(
        mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
        summary_payload=summary_payload,
        baseline_summary_path=ROOT / "artifacts" / "p2_p4_compare_rank8_final_memfix" / "summary.json",
        baseline_report_path=ROOT / "benchmarks" / "3d_hetero_ssr_default" / "report_p2_vs_p4_rank8_final_memfix.md",
        recycle_failure_report=ROOT / "benchmarks" / "3d_hetero_ssr_default" / "report_p4_rank8_recycle_guard80_failed.md",
    )

    text = "\n".join(lines)
    assert "Best AIJ candidate: `hypre_current` (reused baseline)" in text
    assert "BDDC did not clear the step-2 gate" in text
    assert "| Metric | Reused baseline | Best AIJ candidate | BDDC |" in text


def test_step2_and_gate_reports_include_bddc_runtime_smokes() -> None:
    module = _load_module()
    smoke_metrics = {
        "runtime_seconds": 12.5,
        "peak_rss_gib": 4.0,
        "final_accepted_states": 2,
        "bddc_local_primal_vertices_count": 6,
        "bddc_local_total_bytes": 2048.0,
        "pc_backend": "bddc",
    }
    summary_payload = {
        "screen_ranks": [1, 8],
        "scale_ranks": [1, 2, 4, 8],
        "option_smokes": {"hypre_current": {"status": "completed", "metrics": {"pc_backend": "hypre"}}},
        "bddc_runtime_smokes": {
            "status": "completed",
            "runs": {
                "rank1_p2_step1": {"status": "completed", "metrics": smoke_metrics},
                "rank2_p2_step1": {"status": "completed", "metrics": smoke_metrics},
                "rank2_p4_step1": {"status": "completed", "metrics": smoke_metrics},
            },
        },
        "screening": {},
        "promoted_aij_variants": ["hypre_current"],
        "best_aij_name": "hypre_current",
        "scaling": {},
        "bddc_gate": {
            "status": "completed",
            "eligible_for_full_trajectory": True,
            "run": {"status": "completed", "metrics": smoke_metrics},
        },
    }

    step2_text = "\n".join(
        module._step2_report_lines(
            mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
            summary_payload=summary_payload,
        )
    )
    gate_text = "\n".join(
        module._bddc_gate_report_lines(
            mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
            summary_payload=summary_payload,
        )
    )

    assert "## BDDC Runtime Smokes" in step2_text
    assert "rank2_p4_step1" in step2_text
    assert "Primal vertices" in gate_text
    assert "Eligible for full trajectory: `yes`" in gate_text


def test_run_monitored_command_reports_startup_stall_without_progress_file() -> None:
    module = _load_module()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        stdout_path = tmp / "stdout.log"
        stderr_path = tmp / "stderr.log"
        progress_path = tmp / "data" / "progress.jsonl"
        cmd = [
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
        ]
        return_code, summary = module._run_monitored_command(
            cmd=cmd,
            cwd=ROOT,
            env=dict(os.environ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            memory_log_path=None,
            guard_limit_gib=None,
            wall_time_limit_s=None,
            sample_interval_s=0.05,
            progress_path=progress_path,
            startup_progress_timeout_s=0.2,
            startup_backtrace_path=None,
        )

        assert summary["event"] == "startup_stall"
        assert summary["progress_file_created"] is False
        assert summary["startup_stall_reason"] == "progress_timeout"


def test_bddc_short_report_marks_rejected_candidates() -> None:
    module = _load_module()
    metrics_ok = {
        "progress_file_created": True,
        "first_progress_elapsed_s": 12.0,
        "runtime_seconds": 20.0,
        "linear_total_rank_metric": 10.0,
        "final_accepted_states": 3,
        "peak_rss_gib": 4.0,
        "lambda_last": 1.0,
        "omega_last": 2.0,
        "umax_last": 3.0,
    }
    metrics_bad = dict(metrics_ok)
    metrics_bad["runtime_seconds"] = 100.0
    summary_payload = {
        "bddc_prototype": {
            "winner": "bddc_local_ilu",
            "winner_evaluation": {
                "bddc_local_ilu": {"accepted": True, "reasons": []},
                "bddc_exact_current": {"accepted": False, "reasons": ["runtime_limit"]},
            },
            "p2_short": {
                "hypre_current": {"status": "completed", "metrics": metrics_ok},
                "bddc_candidates": {
                    "bddc_local_ilu": {"status": "completed", "metrics": metrics_ok},
                    "bddc_exact_current": {"status": "completed", "metrics": metrics_bad},
                },
            },
            "p4_short": {
                "hypre_current": {"status": "completed", "metrics": metrics_ok},
                "bddc": {"status": "failed"},
            },
            "p4_gate": {"eligible": False, "reasons": ["runtime_limit"]},
        }
    }

    text = "\n".join(
        module._bddc_short_report_lines(
            mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
            summary_payload=summary_payload,
        )
    )

    assert "bddc_local_ilu (winner)" in text
    assert "bddc_exact_current | completed / rejected" in text
    assert "Gate rejection reasons: `runtime_limit`" in text


def test_run_case_reuse_existing_loads_sibling_memory_guard() -> None:
    module = _load_module()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        out_dir = tmp / "case"
        data_dir = out_dir / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "run_info.json").write_text("{}", encoding="utf-8")
        (out_dir.parent / f"{out_dir.name}.memory_guard.jsonl").write_text(
            '{"event":"sample","rss_gib":1.5,"mem_available_gib":10.0}\n',
            encoding="utf-8",
        )
        (out_dir / "startup_summary.json").write_text(
            '{"progress_file_created": true, "first_progress_elapsed_s": 1.0, "startup_stall_reason": null}\n',
            encoding="utf-8",
        )

        observed: dict[str, object] = {}

        def _fake_load_case_metrics(out_dir_arg, *, memory_summary=None, startup_summary=None):
            observed["out_dir"] = out_dir_arg
            observed["memory_summary"] = memory_summary
            observed["startup_summary"] = startup_summary
            return {"runtime_seconds": 1.0}

        original = module._load_case_metrics
        module._load_case_metrics = _fake_load_case_metrics
        try:
            result = module._run_case(
                variant=module.Variant(name="dummy", description="dummy", category="bddc", cli_args=()),
                ranks=1,
                mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
                step_max=1,
                out_dir=out_dir,
                guard_limit_gib=None,
                max_deflation_basis_vectors=1,
                elem_type="P2",
                reuse_existing=True,
                startup_progress_timeout_s=None,
            )
        finally:
            module._load_case_metrics = original

        assert result["status"] == "completed"
        assert observed["memory_summary"]["peak_rss_gib"] == 1.5
        assert observed["startup_summary"]["progress_file_created"] is True


def test_variant_registry_contains_elastic_first_bddc_candidates() -> None:
    module = _load_module()
    registry = module._variant_registry(include_nongalerkin=False)

    exact = registry["bddc_exact_elastic"]
    ilu = registry["bddc_ilu_elastic"]
    deluxe = registry["bddc_ilu_elastic_deluxe"]

    assert "--preconditioner_matrix_source" in exact.cli_args
    assert "elastic" in exact.cli_args
    assert "--outer_solver_family" in exact.cli_args
    assert "native_petsc" in exact.cli_args
    assert "--native_ksp_type" in exact.cli_args
    assert "cg" in exact.cli_args
    assert "--pc_bddc_use_edges" in exact.cli_args
    assert "--pc_bddc_use_faces" in exact.cli_args
    assert "--no-pc_bddc_use_change_of_basis" in exact.cli_args
    assert "--no-pc_bddc_use_change_on_faces" in exact.cli_args
    assert "--pc_bddc_dirichlet_approximate" in ilu.cli_args
    assert "--pc_bddc_neumann_approximate" in ilu.cli_args
    assert "--no-pc_bddc_use_deluxe_scaling" in ilu.cli_args
    assert "--pc_bddc_use_deluxe_scaling" in deluxe.cli_args
