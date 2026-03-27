from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "3d_hetero_ssr_default" / "archive" / "compare_preconditioners.py"


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
        baseline_report_path=ROOT / "benchmarks" / "3d_hetero_ssr_default" / "archive" / "report_p2_vs_p4_rank8_final_memfix.md",
        recycle_failure_report=ROOT / "benchmarks" / "3d_hetero_ssr_default" / "archive" / "report_p4_rank8_recycle_guard80_failed.md",
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
        "bddc_local_primal_vertices_count": 0,
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


def test_run_probe_case_reuse_existing_loads_failed_probe_without_rerun() -> None:
    module = _load_module()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        out_dir = tmp / "probe"
        out_dir.mkdir(parents=True)
        (out_dir / "startup_summary.json").write_text(
            '{"progress_file_created": true, "first_progress_elapsed_s": 1.0, "startup_stall_reason": null}\n',
            encoding="utf-8",
        )
        (out_dir.parent / f"{out_dir.name}.stderr.log").write_text(
            "PETSC ERROR: unused database option(s) foo\n",
            encoding="utf-8",
        )

        result = module._run_probe_case(
            variant=module.Variant(name="dummy", description="dummy", category="sweep_bddc", cli_args=()),
            ranks=1,
            mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
            out_dir=out_dir,
            elem_type="P4",
            mode="single_solve",
            linear_tolerance=1.0e-5,
            linear_max_iter=10,
            guard_limit_gib=None,
            wall_time_limit_s=None,
            reuse_existing=True,
            startup_progress_timeout_s=1.0,
        )

        assert result["status"] == "unsupported_option"
        assert "unused database option" in result["failure_excerpt"].lower()


def test_variant_registry_contains_elastic_first_bddc_candidates() -> None:
    module = _load_module()
    registry = module._variant_registry(include_nongalerkin=False)

    exact = registry["bddc_exact_elastic"]
    gamg = registry["bddc_gamg_elastic"]

    assert "--preconditioner_matrix_source" in exact.cli_args
    assert "elastic" in exact.cli_args
    assert "--outer_solver_family" in exact.cli_args
    assert "native_petsc" in exact.cli_args
    assert "--native_ksp_type" in exact.cli_args
    assert "cg" in exact.cli_args
    assert "--pc_bddc_symmetric" in exact.cli_args
    assert "--pc_bddc_use_edges" in exact.cli_args
    assert "--pc_bddc_use_faces" in exact.cli_args
    assert "--no-pc_bddc_use_change_of_basis" in exact.cli_args
    assert "--no-pc_bddc_use_change_on_faces" in exact.cli_args
    assert "--pc_bddc_dirichlet_approximate" in gamg.cli_args
    assert "--pc_bddc_neumann_approximate" in gamg.cli_args
    assert "--pc_bddc_switch_static" in gamg.cli_args
    assert "--pc_bddc_dirichlet_pc_type" in gamg.cli_args
    assert "gamg" in gamg.cli_args
    assert "--no-pc_bddc_use_deluxe_scaling" in gamg.cli_args


def test_bddc_sweep_registry_contains_doc_guided_candidates() -> None:
    module = _load_module()
    registry = module._bddc_sweep_registry(include_adaptive=True)

    base = registry["bddc_gamg_doc_base_v2"]
    smooth = registry["bddc_gamg_ex56_v2"]
    adaptive = registry["bddc_gamg_ex71_adaptive2_v2"]

    assert "--native_ksp_norm_type" in base.cli_args
    assert "unpreconditioned" in base.cli_args
    assert "--pc_bddc_symmetric" in base.cli_args
    assert "--pc_bddc_monolithic" in base.cli_args
    assert "--pc_bddc_coarse_redundant_pc_type" in base.cli_args
    assert "svd" in base.cli_args
    assert "--pc_bddc_use_faces" in base.cli_args
    assert "--pc_bddc_switch_static" in base.cli_args
    assert "--petsc-opt" in smooth.cli_args
    assert "pc_bddc_dirichlet_pc_gamg_threshold=0.05" in smooth.cli_args
    assert "pc_bddc_adaptive_threshold=2.0" in adaptive.cli_args


def test_bddc_sweep_report_lists_linear_screen_and_promotions() -> None:
    module = _load_module()
    summary_payload = {
        "bddc_sweep": {
            "linear_tolerance": 1.0e-5,
            "mumps_available": True,
            "phase0_note": "Superseded v1 because it lacked the corrected BDDC baseline.",
            "option_smokes": {
                "hypre_control_v2": {"status": "completed", "startup_summary": {"first_progress_elapsed_s": 1.0}},
            },
            "linear_screen": {
                "hypre_control_v2": {
                    "status": "completed",
                    "metrics": {
                        "iteration_count": 18,
                        "setup_elapsed_s": 10.0,
                        "solve_time": 20.0,
                        "runtime_seconds": 40.0,
                        "final_relative_residual": 1.0e-6,
                        "peak_rss_gib": 2.0,
                    },
                },
                "bddc_gamg_doc_base_v2": {
                    "status": "completed",
                    "metrics": {
                        "iteration_count": 120,
                        "setup_elapsed_s": 30.0,
                        "solve_time": 40.0,
                        "runtime_seconds": 80.0,
                        "final_relative_residual": 9.0e-6,
                        "peak_rss_gib": 3.0,
                    },
                },
            },
            "promoted_candidates": ["bddc_gamg_doc_base_v2"],
            "diagnostic_linear": {
                "hypre_control_v2": {
                    "status": "completed",
                    "metrics": {"runtime_seconds": 40.0, "petsc_log": "artifacts/hypre.log"},
                },
                "bddc_gamg_doc_base_v2": {
                    "status": "completed",
                    "metrics": {"runtime_seconds": 85.0, "petsc_log": "artifacts/bddc.log"},
                },
            },
            "nonlinear_short": {
                "hypre_control_v2": {"status": "completed", "metrics": {"runtime_seconds": 100.0, "final_accepted_states": 3, "linear_total_rank_metric": 50.0}},
                "bddc_gamg_doc_base_v2": {"status": "completed", "accepted": True, "metrics": {"runtime_seconds": 120.0, "final_accepted_states": 3, "linear_total_rank_metric": 70.0}},
            },
            "nonlinear_step10": {"winner": "bddc_gamg_doc_base_v2", "run": {"status": "completed"}},
            "rank8_linear": {"status": "not_run"},
            "plots": {"overlays": {"bddc_gamg_doc_base_v2": "artifacts/x.png"}, "aggregate": "artifacts/all.png"},
        }
    }

    text = "\n".join(
        module._bddc_sweep_report_lines(
            mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
            summary_payload=summary_payload,
        )
    )

    assert "Phase 1: Rank-2 Linear Elastic Screen" in text
    assert "bddc_gamg_doc_base_v2" in text
    assert "Promoted candidates: `bddc_gamg_doc_base_v2`" in text
    assert "Aggregate completed-candidate plot" in text


def test_parse_bddc_coarse_info_prefers_local_candidate_counts() -> None:
    module = _load_module()
    parsed = module._parse_bddc_coarse_info(
        """
        Subdomain 0000 got 00 local candidate vertices (1)
        Subdomain 0000 got 03 local candidate edges    (1)
        Subdomain 0000 got 12 local candidate faces    (1)
        Size of coarse problem is 9
        """
    )

    assert parsed == {
        "candidate_vertices": 0,
        "candidate_edges": 3,
        "candidate_faces": 12,
        "coarse_size": 9,
    }


def test_bddc_sweep_workflow_smoke_generates_plot_artifacts() -> None:
    module = _load_module()
    registry = module._bddc_sweep_registry(include_adaptive=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        out_root = tmp / "sweep"

        def _write_run_info(out_dir: Path, *, backend: str, iters: int, runtime: float, peak_rss: float) -> None:
            data_dir = out_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            history = [1.0, 1.0e-2, 1.0e-4, 9.0e-6]
            payload = {
                "pc_backend": backend,
                "reported_residual_history": history,
                "relative_reported_residual_history": history,
                "reported_residual_histories": [history],
                "relative_reported_residual_histories": [history],
                "converged_reason": 4,
                "iteration_count": iters,
                "iteration_counts": [iters],
                "setup_elapsed_s": runtime * 0.3,
                "solve_time": runtime * 0.5,
                "solve_times_s": [runtime * 0.5],
                "runtime_seconds": runtime,
                "final_relative_residual": history[-1],
                "peak_rss_gib": peak_rss,
            }
            (data_dir / "run_info.json").write_text(json.dumps(payload), encoding="utf-8")

        def _fake_run_probe_case(**kwargs):
            variant = kwargs["variant"]
            out_dir = Path(kwargs["out_dir"])
            ranks = int(kwargs["ranks"])
            result = {
                "status": "completed",
                "metrics": {
                    "converged_reason": 4,
                    "iteration_count": 18 if variant.name == "hypre_control_v2" else 24,
                    "setup_elapsed_s": 8.0 if variant.name == "hypre_control_v2" else 18.0,
                    "solve_time": 12.0 if variant.name == "hypre_control_v2" else 24.0,
                    "runtime_seconds": 30.0 if variant.name == "hypre_control_v2" else 56.0,
                    "final_relative_residual": 9.0e-6,
                    "peak_rss_gib": 2.0 if variant.name == "hypre_control_v2" else 2.8,
                    "bddc_candidate_faces": 4 if variant.name != "hypre_control_v2" else None,
                    "bddc_coarse_size": 24 if variant.name != "hypre_control_v2" else None,
                },
                "startup_summary": {
                    "first_progress_elapsed_s": 1.0,
                    "progress_file_created": True,
                    "startup_stall_reason": None,
                },
            }
            if variant.name == "bddc_exact_lu_ref_v2":
                return {
                    "status": "runtime_failure",
                    "reason": "synthetic failure",
                    "startup_summary": {
                        "first_progress_elapsed_s": 1.0,
                        "progress_file_created": True,
                        "startup_stall_reason": None,
                    },
                }
            _write_run_info(
                out_dir,
                backend="hypre" if variant.name == "hypre_control_v2" else "bddc",
                iters=18 if variant.name == "hypre_control_v2" else (24 if ranks == 1 else 24),
                runtime=30.0 if variant.name == "hypre_control_v2" else (45.0 if ranks == 1 else 56.0),
                peak_rss=2.0 if variant.name == "hypre_control_v2" else 2.8,
            )
            return result

        def _fake_run_case(**kwargs):
            variant = kwargs["variant"]
            runtime = 80.0 if "hypre" in variant.name else 120.0
            return {
                "status": "completed",
                "metrics": {
                    "runtime_seconds": runtime,
                    "linear_total_rank_metric": 50.0 if "hypre" in variant.name else 70.0,
                    "final_accepted_states": 1,
                    "lambda_last": 1.0,
                    "omega_last": 2.0,
                    "umax_last": 3.0,
                    "peak_rss_gib": 4.0 if "hypre" in variant.name else 5.0,
                    "progress_file_created": True,
                    "first_progress_elapsed_s": 2.0,
                },
                "startup_summary": {
                    "first_progress_elapsed_s": 2.0,
                    "progress_file_created": True,
                    "startup_stall_reason": None,
                },
            }

        original_probe = module._run_probe_case
        original_case = module._run_case
        original_has_pkg = module._petsc_has_external_package
        try:
            module._run_probe_case = _fake_run_probe_case
            module._run_case = _fake_run_case
            module._petsc_has_external_package = lambda name: False
            summary = module._run_bddc_sweep_workflow(
                registry=registry,
                mesh_path=ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh",
                out_root=out_root,
                linear_tolerance=1.0e-5,
                linear_max_iter=500,
                max_deflation_basis_vectors=16,
                reuse_existing=False,
                requested_variants=("bddc_gamg_doc_base_v2",),
            )
        finally:
            module._run_probe_case = original_probe
            module._run_case = original_case
            module._petsc_has_external_package = original_has_pkg

        overlay_rel = summary["plots"]["overlays"]["bddc_gamg_doc_base_v2"]
        aggregate_rel = summary["plots"]["aggregate"]
        assert summary["promoted_candidates"][0] == "bddc_gamg_doc_base_v2"
        assert "bddc_exact_lu_ref_v2" not in summary["linear_screen"]
        assert summary["nonlinear_step10"]["winner"] in summary["promoted_candidates"]
        assert summary["nonlinear_step10"]["run"]["status"] == "completed"
        assert (ROOT / overlay_rel).exists()
        assert aggregate_rel is not None
        assert (ROOT / aggregate_rel).exists()
