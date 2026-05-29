from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

import petsc_ssr.benchmarks.suites as suites_mod
from petsc_ssr.benchmarks.compare import collect_run_rows
from petsc_ssr.benchmarks.suites import compare_targets, expand_suite, load_suite, run_suite, write_manifest, write_report


ROOT = Path(__file__).resolve().parents[2]


def test_local_32_suite_expands_to_rank_sweep(tmp_path: Path) -> None:
    spec = load_suite(ROOT / "benchmarks" / "suites" / "local-32-smoke.toml")
    runs = expand_suite(spec, run_root=tmp_path)

    assert spec.id == "local-32-smoke"
    assert spec.resources["local"]["machine"] == "local-32-core"
    assert spec.resources["local"]["cores"] == 32
    assert spec.resources["local"]["max_ranks"] == 32
    assert spec.environment == {"OMP_NUM_THREADS": "1"}
    assert [run.ranks for run in runs] == [1, 2, 4, 8, 16, 32]
    assert all(run.profile == "pmg-deflated-baseline" for run in runs)
    assert all(run.resource == "local" for run in runs)
    assert all(run.launcher == ("mpiexec",) for run in runs)
    assert runs[0].command[:3] == ("mpiexec", "-n", "1")
    assert all("--continuation-step-max" in run.command for run in runs)
    assert any("--petsc-opt=-log_view" in run.command for run in runs)
    assert any("--petsc-opt=-options_left" in run.command for run in runs)
    assert runs[-1].resolved_profile["linear"]["algorithm"] == "ksp_deflated"
    assert runs[-1].resolved_profile["linear"]["native_algorithm"] == "pmg-deflated"
    assert runs[-1].resolved_profile["linear"]["ksp_type"] == "fgmres"
    assert runs[-1].resolved_profile["pc"] == {
        "backend": "pmg_shell",
        "variant": "pmg",
        "requested_variant": "pmg",
        "fallback_reason": None,
    }
    assert runs[-1].resolved_profile["pmg"]["p2_active_ranks"] == 32
    assert runs[-1].resolved_profile["pmg"]["p1_active_ranks"] == 16
    assert runs[-1].resolved_profile["pmg"]["apply_backend"] == "shell_vcycle"
    assert runs[-1].resolved_profile["pmg"]["coarse_pc_type"] == "gamg"
    assert runs[-1].resolved_profile["pmg"]["coarse_telescope_ksp_max_it"] == 5
    assert runs[-1].resolved_profile["pmg"]["smoother_max_it"] == 2


def test_suite_manifest_records_p1_pc_fallback(tmp_path: Path) -> None:
    suite_toml = tmp_path / "seepage-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "seepage-suite"
cases = ["2d-sloan2013-seepage"]
profiles = ["pmg-deflated-baseline"]
ranks = [4]
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    runs = expand_suite(spec, run_root=tmp_path)

    assert runs[0].resolved_profile["pc"] == {
        "backend": "pmg_shell",
        "variant": "gamg",
        "requested_variant": "pmg",
        "fallback_reason": "p1_has_no_p_hierarchy",
    }
    assert runs[0].resolved_profile["linear"]["algorithm"] == "ksp_deflated"
    assert runs[0].resolved_profile["linear"]["native_algorithm"] == "gamg"


def test_suite_manifest_and_report_scaffold(tmp_path: Path) -> None:
    spec = load_suite(ROOT / "benchmarks" / "suites" / "local-32-smoke.toml")
    runs = expand_suite(spec, run_root=tmp_path)
    manifest = write_manifest(spec, runs, tmp_path / "manifest.json")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["suite"]["ranks"] == [1, 2, 4, 8, 16, 32]
    assert payload["suite"]["resources"]["local"]["cores"] == 32
    assert payload["suite"]["resources"]["local"]["time_limit"] == "00:20:00"
    assert payload["suite"]["environment"] == {"OMP_NUM_THREADS": "1"}
    assert payload["runs"][0]["case"] == "3d-heterogeneous-ssr-p4"
    assert payload["runs"][0]["resource"] == "local"
    assert payload["runs"][0]["launcher"] == ["mpiexec"]
    assert payload["runs"][0]["resolved_profile"]["pc"]["variant"] == "pmg"
    assert payload["runs"][0]["resolved_profile"]["linear"]["algorithm"] == "ksp_deflated"
    assert payload["runs"][0]["resolved_profile"]["linear"]["native_algorithm"] == "pmg-deflated"
    assert payload["runs"][0]["artifacts"]["resolved_run_manifest_json"].endswith(
        "data/resolved_run_manifest.json"
    )
    assert payload["runs"][0]["artifacts"]["command_json"].endswith("command.json")
    assert payload["runs"][0]["artifacts"]["petsc_log_txt"].endswith("logs/petsc_log.txt")
    assert payload["runs"][0]["artifacts"]["options_left_txt"].endswith("logs/options_left.txt")
    assert payload["runs"][-1]["resolved_profile"]["pmg"]["p1_active_ranks"] == 16
    assert payload["runs"][-1]["resolved_profile"]["pmg"]["p2_telescope_active_ranks"] == 0
    assert payload["runs"][-1]["resolved_profile"]["pmg"]["smoother_pc_type"] == "jacobi"

    report = write_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "Local 32-core smoke sweep" in text
    assert "options-left" in text
    assert "native linear" in text
    assert "pmg-deflated" in text
    assert "## Artifact Paths" in text
    assert "command.json" in text
    assert "resolved_run_manifest.json" in text


def test_collect_run_rows_uses_completed_summary_provenance_fallbacks(tmp_path: Path) -> None:
    output = tmp_path / "run"
    data_dir = output / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "summary.json").write_text(
        json.dumps(
            {
                "wall_time": 1.0,
                "native_linear_algorithm": "pmg-deflated",
                "pc_variant": "pmg",
                "requested_pc_variant": "pmg",
                "pc_variant_fallback_reason": "",
                "pmg_p2_active_ranks": 4,
                "pmg_p1_active_ranks": 2,
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "runs": [
            {
                "case": "case-a",
                "profile": "profile-a",
                "ranks": 4,
                "repeat": 1,
                "output_dir": str(output),
                "artifacts": {},
            }
        ]
    }

    row = collect_run_rows(manifest)[0]

    assert row["native_linear_algorithm"] == "pmg-deflated"
    assert row["pc_variant"] == "pmg"
    assert row["requested_pc_variant"] == "pmg"
    assert row["pc_variant_fallback_reason"] == ""
    assert row["pmg_p2_active_ranks"] == 4
    assert row["pmg_p1_active_ranks"] == 2
    assert row["command_manifest"] is None


def test_suite_output_override_expands_command_and_manifest(tmp_path: Path) -> None:
    suite_toml = tmp_path / "output-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "output-suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[overrides.output]
preset = "smoke"
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    runs = expand_suite(spec, run_root=tmp_path)
    manifest = write_manifest(spec, runs, tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert runs[0].output_preset == "smoke"
    assert "--output-preset" in runs[0].command
    assert _flag_value(runs[0].command, "--output-preset") == "smoke"
    assert payload["suite"]["overrides"]["output"] == {"preset": "smoke"}
    assert payload["runs"][0]["output_preset"] == "smoke"


def test_committed_suites_use_modern_public_schema() -> None:
    for suite_toml in sorted((ROOT / "benchmarks" / "suites").glob("*.toml")):
        payload = tomllib.loads(suite_toml.read_text(encoding="utf-8"))
        assert "solver" not in payload, suite_toml
        assert "id" in payload.get("suite", {}), suite_toml
        assert "name" not in payload.get("suite", {}), suite_toml
        assert "description" not in payload.get("suite", {}), suite_toml

        spec = load_suite(suite_toml)
        assert spec.cases
        assert spec.profiles
        assert spec.ranks


def test_suite_run_materializes_options_left_artifact_and_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    suite_toml = tmp_path / "one-run-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "one-run-suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[resources.local]
max_ranks = 1
launcher = "mpiexec"

[environment]
OMP_NUM_THREADS = "1"
PETSC_SSR_WORLD_SIZE = "1"

[collect]
options_left = true
""",
        encoding="utf-8",
    )

    captured_env: dict[str, str] = {}

    def fake_run(command, *, cwd, check, stdout, stderr, text, env):
        captured_env.update({key: env[key] for key in ("OMP_NUM_THREADS", "PETSC_SSR_WORLD_SIZE")})
        stdout.write("CASE_RESULT output=/tmp/run\nThere are no unused options.\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(suites_mod.subprocess, "run", fake_run)

    spec = load_suite(suite_toml)
    run_suite(spec, run_root=tmp_path)
    run = expand_suite(spec, run_root=tmp_path)[0]

    options_left = run.output_dir / "logs" / "options_left.txt"
    assert options_left.exists()
    assert options_left.read_text(encoding="utf-8").startswith("status: clean\n")
    assert captured_env == {"OMP_NUM_THREADS": "1", "PETSC_SSR_WORLD_SIZE": "1"}
    command_payload = json.loads((run.output_dir / "command.json").read_text(encoding="utf-8"))
    assert command_payload["kind"] == "petsc_ssr_suite_command"
    assert command_payload["suite"] == "one-run-suite"
    assert command_payload["run_id"] == run.run_id
    assert command_payload["case"] == "3d-heterogeneous-ssr-p4"
    assert command_payload["profile"] == "pmg-deflated-baseline"
    assert command_payload["ranks"] == 1
    assert command_payload["resource"] == "local"
    assert command_payload["launcher"] == ["mpiexec"]
    assert command_payload["command"] == list(run.command)
    assert command_payload["environment"] == {"OMP_NUM_THREADS": "1", "PETSC_SSR_WORLD_SIZE": "1"}
    assert command_payload["sweep"] == {
        "refine_levels": None,
        "linear_rtol": None,
        "continuation_step_max": None,
    }
    assert command_payload["resolved_profile"]["linear"]["native_algorithm"] == "pmg-deflated"
    assert command_payload["artifacts"]["resolved_run_manifest_json"].endswith("data/resolved_run_manifest.json")

    report = write_report(tmp_path)
    assert "| 3d-heterogeneous-ssr-p4 | pmg-deflated-baseline | 1 |" in report.read_text(encoding="utf-8")


def test_target_compare_scaffold(tmp_path: Path) -> None:
    spec = load_suite(ROOT / "benchmarks" / "suites" / "local-32-smoke.toml")
    runs = expand_suite(spec, run_root=tmp_path)
    write_manifest(spec, runs, tmp_path / "manifest.json")

    out = compare_targets(tmp_path, ROOT / "benchmarks" / "targets" / "local-32")
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["rows"]
    assert payload["rows"][0]["status"] in {"missing_summary", "no_metric_targets"}
    assert payload["rows"][0]["native_linear_algorithm"] == "pmg-deflated"
    assert payload["rows"][0]["pc_variant"] == "pmg"
    assert payload["rows"][0]["pmg_p2_active_ranks"] == 1


def test_target_compare_top_level_root_prefers_matching_suite_target(tmp_path: Path) -> None:
    spec = load_suite(ROOT / "benchmarks" / "suites" / "local-32-strong-scaling.toml")
    runs = expand_suite(spec, run_root=tmp_path)
    write_manifest(spec, runs, tmp_path / "manifest.json")

    out = compare_targets(tmp_path, ROOT / "benchmarks" / "targets")
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]

    assert rows
    assert all("benchmarks/targets/local-32/3d-heterogeneous-ssr-p4.json" in row["target"] for row in rows)


def test_suite_sweep_axes_expand_commands_manifests_and_reports(tmp_path: Path) -> None:
    suite_toml = tmp_path / "axis-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "axis-suite"
title = "Axis suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [4]
repeats = 1

[sweeps]
refine_levels = [0, 1]
linear_rtol = [0.1, 0.05]
continuation_step_max = [3]

[overrides.continuation]
step_max = 9

[overrides.linear]
rtol = 0.2

[overrides.mesh]
refine_levels = 5
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    runs = expand_suite(spec, run_root=tmp_path)

    assert len(runs) == 4
    assert runs[0].run_id == "3d-heterogeneous-ssr-p4__pmg-deflated-baseline__r4__refine-l0__linear-rtol-0p1__steps-3__rep1"
    assert runs[-1].run_id == "3d-heterogeneous-ssr-p4__pmg-deflated-baseline__r4__refine-l1__linear-rtol-0p05__steps-3__rep1"
    assert runs[0].sweep == {"refine_levels": 0, "linear_rtol": 0.1, "continuation_step_max": 3}
    assert runs[-1].sweep == {"refine_levels": 1, "linear_rtol": 0.05, "continuation_step_max": 3}
    assert "refine-l0" in runs[0].output_dir.parts
    assert "linear-rtol-0p05" in runs[-1].output_dir.parts
    assert _flag_value(runs[0].command, "--refine-levels") == "0"
    assert _flag_value(runs[0].command, "--linear-rtol") == "0.1"
    assert _flag_value(runs[0].command, "--continuation-step-max") == "3"
    assert _flag_value(runs[-1].command, "--refine-levels") == "1"
    assert _flag_value(runs[-1].command, "--linear-rtol") == "0.05"

    manifest = write_manifest(spec, runs, tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["suite"]["sweeps"] == {
        "refine_levels": [0, 1],
        "linear_rtol": [0.1, 0.05],
        "continuation_step_max": [3],
    }
    assert payload["runs"][0]["sweep"] == {"refine_levels": 0, "linear_rtol": 0.1, "continuation_step_max": 3}

    report = write_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "| case | profile | ranks | resource | refine | linear rtol | step max | repeat |" in text
    assert "| 3d-heterogeneous-ssr-p4 | pmg-deflated-baseline | 4 |  | 1 | 0.05 | 3 | 1 | planned |" in text

    comparison = compare_targets(tmp_path, ROOT / "benchmarks" / "targets" / "local-32")
    rows = json.loads(comparison.read_text(encoding="utf-8"))["rows"]
    assert rows[-1]["refine_levels"] == 1
    assert rows[-1]["linear_rtol"] == 0.05
    assert rows[-1]["continuation_step_max"] == 3
    assert rows[-1]["native_linear_algorithm"] == "pmg-deflated"


def test_suite_report_computes_scaling_iteration_and_numerical_summaries(tmp_path: Path) -> None:
    suite_toml = tmp_path / "measured-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "measured-suite"
title = "Measured suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1, 2]
repeats = 2
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    runs = expand_suite(spec, run_root=tmp_path)
    write_manifest(spec, runs, tmp_path / "manifest.json")

    wall_by_run = {
        (1, 1): 10.0,
        (1, 2): 14.0,
        (2, 1): 6.0,
        (2, 2): 8.0,
    }
    for run in runs:
        data_dir = run.output_dir / "data"
        logs_dir = run.output_dir / "logs"
        data_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        wall = wall_by_run[(run.ranks, run.repeat)]
        (data_dir / "summary.json").write_text(
            json.dumps(
                {
                    "wall_time": wall,
                    "continuation_wall_time": wall - 1.0,
                    "lambda_last": 1.2 + 0.01 * run.ranks,
                    "omega_last": 5.0e6 + 100.0 * run.ranks,
                    "final_rel": 1.0e-5 * run.ranks,
                    "global_dofs": 1000,
                    "accepted_steps": 3,
                    "total_newton_its": 10 + run.ranks,
                    "total_linear_its": 100 + 10 * run.ranks,
                    "total_line_search_its": run.ranks,
                }
            ),
            encoding="utf-8",
        )
        (logs_dir / "stdout.txt").write_text("There are no unused options.\n", encoding="utf-8")
        if run.ranks == 2 and run.repeat == 1:
            (logs_dir / "petsc_log.txt").write_text(
                """
Event                Count      Time (sec)
KSPSolve                 2      4.0000e+00
SSR_PMGApply             4      7.5000e+00
MatMult                 20      1.2500e+00
""",
                encoding="utf-8",
            )

    report = write_report(tmp_path)
    text = report.read_text(encoding="utf-8")

    assert "## Scaling Summary" in text
    assert "| 3d-heterogeneous-ssr-p4 | pmg-deflated-baseline | 1 | 2 | 12 | 1 | 1 | clean |" in text
    assert "| 3d-heterogeneous-ssr-p4 | pmg-deflated-baseline | 2 | 2 | 7 | 1.71429 | 0.857143 | clean |" in text
    assert "## Iteration Summary" in text
    assert "| 3d-heterogeneous-ssr-p4 | 2 | 3 | 12 | 120 | 2 |" in text
    assert "## Numerical Summary" in text
    assert "| 3d-heterogeneous-ssr-p4 | 2 | 1.22 | 5.0002e+06 | 2e-05 | 1000 |" in text
    assert "## PETSc Log Events" in text
    assert "| 3d-heterogeneous-ssr-p4 | 2 | 1 | 1 | SSR_PMGApply | 4 | 7.5 |" in text
    assert "## Artifact Paths" in text
    assert "command.json" in text
    assert "resolved_options.txt" in text

    scaling_csv = report.with_suffix(".scaling.csv")
    assert scaling_csv.exists()
    assert "parallel_efficiency" in scaling_csv.read_text(encoding="utf-8")
    report_csv = report.with_suffix(".csv")
    assert "native_linear_algorithm" in report_csv.read_text(encoding="utf-8")
    assert "command_manifest" in report_csv.read_text(encoding="utf-8")
    assert "resolved_run_manifest" in report_csv.read_text(encoding="utf-8")
    events_csv = report.with_suffix(".petsc-events.csv")
    assert events_csv.exists()
    assert "SSR_PMGApply" in events_csv.read_text(encoding="utf-8")


def test_target_compare_uses_median_groups_and_rank_overrides(tmp_path: Path) -> None:
    suite_toml = tmp_path / "target-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "target-suite"
title = "Target suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1, 2]
repeats = 2
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    runs = expand_suite(spec, run_root=tmp_path)
    write_manifest(spec, runs, tmp_path / "manifest.json")

    wall_by_run = {
        (1, 1): 10.0,
        (1, 2): 14.0,
        (2, 1): 6.0,
        (2, 2): 8.0,
    }
    for run in runs:
        data_dir = run.output_dir / "data"
        logs_dir = run.output_dir / "logs"
        data_dir.mkdir(parents=True)
        logs_dir.mkdir(parents=True)
        (data_dir / "summary.json").write_text(
            json.dumps({"wall_time": wall_by_run[(run.ranks, run.repeat)], "total_newton_its": 10 + run.ranks}),
            encoding="utf-8",
        )
        (logs_dir / "stdout.txt").write_text("There are no unused options.\n", encoding="utf-8")

    target_root = tmp_path / "targets"
    target_root.mkdir()
    (target_root / "3d-heterogeneous-ssr-p4.json").write_text(
        json.dumps(
            {
                "case": "3d-heterogeneous-ssr-p4",
                "metrics": {
                    "wall_time": {"max": 12.0},
                    "total_newton_its": {"expected": 11.0, "abs_tol": 1.0},
                },
                "groups": [
                    {
                        "ranks": 2,
                        "metrics": {
                            "wall_time": {"max": 6.5},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_targets(tmp_path, target_root)
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    rows = {row["ranks"]: row for row in payload["rows"]}

    assert payload["comparison_level"] == "median"
    assert rows[1]["status"] == "pass"
    assert rows[1]["repeats"] == 2
    assert rows[1]["results"]["wall_time"]["actual"] == 12.0
    assert rows[1]["results"]["wall_time"]["actual_metric"] == "wall_time_median"
    assert rows[2]["status"] == "fail"
    assert rows[2]["results"]["wall_time"]["actual"] == 7.0
    assert rows[2]["results"]["wall_time"]["max"] == 6.5


def test_target_compare_fails_on_options_left_check_even_when_metrics_pass(tmp_path: Path) -> None:
    suite_toml = tmp_path / "options-left-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "options-left-suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    run = expand_suite(spec, run_root=tmp_path)[0]
    write_manifest(spec, [run], tmp_path / "manifest.json")

    data_dir = run.output_dir / "data"
    logs_dir = run.output_dir / "logs"
    data_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    (data_dir / "summary.json").write_text(json.dumps({"wall_time": 1.0}), encoding="utf-8")
    (logs_dir / "options_left.txt").write_text(
        "status: check\nsource: stdout.txt\nUnused option: -misspelled_option\n",
        encoding="utf-8",
    )
    target_root = tmp_path / "targets"
    target_root.mkdir()
    (target_root / "3d-heterogeneous-ssr-p4.json").write_text(
        json.dumps({"case": "3d-heterogeneous-ssr-p4", "metrics": {"wall_time": {"max": 2.0}}}),
        encoding="utf-8",
    )

    comparison = compare_targets(tmp_path, target_root)
    row = json.loads(comparison.read_text(encoding="utf-8"))["rows"][0]

    assert row["options_left"] == "check"
    assert row["status"] == "options_left_check"
    assert row["metrics"] == {"wall_time": {"max": 2.0}}
    assert row["results"] == {}


def test_target_compare_requires_options_left_evidence_for_completed_runs(tmp_path: Path) -> None:
    suite_toml = tmp_path / "missing-options-left-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "missing-options-left-suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    run = expand_suite(spec, run_root=tmp_path)[0]
    write_manifest(spec, [run], tmp_path / "manifest.json")

    data_dir = run.output_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "summary.json").write_text(json.dumps({"wall_time": 1.0}), encoding="utf-8")
    target_root = tmp_path / "targets"
    target_root.mkdir()
    (target_root / "3d-heterogeneous-ssr-p4.json").write_text(
        json.dumps({"case": "3d-heterogeneous-ssr-p4", "metrics": {"wall_time": {"max": 2.0}}}),
        encoding="utf-8",
    )

    missing = compare_targets(tmp_path, target_root)
    row = json.loads(missing.read_text(encoding="utf-8"))["rows"][0]

    assert row["options_left"] == "missing"
    assert row["status"] == "options_left_missing"
    assert row["results"] == {}

    logs_dir = run.output_dir / "logs"
    logs_dir.mkdir(parents=True)
    (logs_dir / "stdout.txt").write_text("solver finished without PETSc options-left footer\n", encoding="utf-8")
    unknown = compare_targets(tmp_path, target_root)
    row = json.loads(unknown.read_text(encoding="utf-8"))["rows"][0]

    assert row["options_left"] == "unknown"
    assert row["status"] == "options_left_unknown"
    assert row["results"] == {}


def test_suite_rejects_unknown_sweep_fields(tmp_path: Path) -> None:
    suite_toml = tmp_path / "bad-sweep.toml"
    suite_toml.write_text(
        """
[suite]
id = "bad-sweep"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[sweeps]
mesh_size = [1]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[sweeps\].*mesh_size"):
        load_suite(suite_toml)


def test_modern_suite_requires_public_axes(tmp_path: Path) -> None:
    suite_toml = tmp_path / "empty-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "empty-suite"
cases = []
profiles = ["pmg-deflated-baseline"]
ranks = [1]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[suite\]\.cases"):
        load_suite(suite_toml)


def test_modern_suite_without_id_is_rejected(tmp_path: Path) -> None:
    suite_toml = tmp_path / "missing-id.toml"
    suite_toml.write_text(
        """
[suite]
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[suite\]\.id"):
        load_suite(suite_toml)


def test_modern_suite_rejects_legacy_solver_section(tmp_path: Path) -> None:
    suite_toml = tmp_path / "bad-top-level.toml"
    suite_toml.write_text(
        """
[suite]
id = "bad-top-level"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[solver]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Top-level.*solver"):
        load_suite(suite_toml)


def test_suite_rejects_unsupported_override_sections(tmp_path: Path) -> None:
    suite_toml = tmp_path / "bad-override.toml"
    suite_toml.write_text(
        """
[suite]
id = "bad-override"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[overrides.execution]
wall_time = "00:10:00"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[overrides\].*execution"):
        load_suite(suite_toml)


def test_suite_rejects_unknown_resource_fields(tmp_path: Path) -> None:
    suite_toml = tmp_path / "bad-resource.toml"
    suite_toml.write_text(
        """
[suite]
id = "bad-resource"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[resources.local]
cores = 32
gpu_type = "none"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[resources.local\].*gpu_type"):
        load_suite(suite_toml)


def test_suite_resource_limits_validate_rank_sweep(tmp_path: Path) -> None:
    suite_toml = tmp_path / "resource-limits.toml"
    suite_toml.write_text(
        """
[suite]
id = "resource-limits"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1, 8]

[resources.local]
cores = 4
max_ranks = 4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[suite\]\.ranks.*8.*max_ranks.*local=4"):
        load_suite(suite_toml)


def test_suite_resource_limits_allow_rank_supported_by_any_resource(tmp_path: Path) -> None:
    suite_toml = tmp_path / "multi-resource.toml"
    suite_toml.write_text(
        """
[suite]
id = "multi-resource"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [32, 128]

[resources.local]
max_ranks = 32

[resources.hpc]
nodes = 2
ranks_per_node = 64
max_ranks = 128
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    assert spec.ranks == (32, 128)

    runs = expand_suite(spec, run_root=tmp_path)
    assert [run.resource for run in runs] == ["local", "hpc"]
    assert [run.launcher for run in runs] == [("mpiexec",), ("mpiexec",)]


def test_suite_resource_launcher_is_resolved_into_commands_and_manifest(tmp_path: Path) -> None:
    suite_toml = tmp_path / "launcher-suite.toml"
    suite_toml.write_text(
        """
[suite]
id = "launcher-suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [4]

[resources.hpc]
max_ranks = 8
launcher = "srun --mpi=pmix"
""",
        encoding="utf-8",
    )

    spec = load_suite(suite_toml)
    runs = expand_suite(spec, run_root=tmp_path)
    manifest = write_manifest(spec, runs, tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert runs[0].resource == "hpc"
    assert runs[0].launcher == ("srun", "--mpi=pmix")
    assert runs[0].command[:4] == ("srun", "--mpi=pmix", "-n", "4")
    assert payload["runs"][0]["resource"] == "hpc"
    assert payload["runs"][0]["launcher"] == ["srun", "--mpi=pmix"]


def test_suite_resource_integer_fields_must_be_positive_and_consistent(tmp_path: Path) -> None:
    nonpositive = tmp_path / "nonpositive-resource.toml"
    nonpositive.write_text(
        """
[suite]
id = "nonpositive-resource"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[resources.local]
cores = 0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[resources.local\]\.cores.*positive"):
        load_suite(nonpositive)

    inconsistent = tmp_path / "inconsistent-resource.toml"
    inconsistent.write_text(
        """
[suite]
id = "inconsistent-resource"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [4]

[resources.hpc]
nodes = 1
ranks_per_node = 2
max_ranks = 4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"max_ranks.*nodes \* ranks_per_node"):
        load_suite(inconsistent)


def test_suite_collect_policy_requires_booleans_and_single_alias(tmp_path: Path) -> None:
    string_collect = tmp_path / "string-collect.toml"
    string_collect.write_text(
        """
[suite]
id = "string-collect"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[collect]
options_left = "false"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[collect\]\.options_left.*boolean"):
        load_suite(string_collect)

    duplicate_alias = tmp_path / "duplicate-collect.toml"
    duplicate_alias.write_text(
        """
[suite]
id = "duplicate-collect"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[collect]
log_view = true
petsc_log_view = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"petsc_log_view.*log_view"):
        load_suite(duplicate_alias)


def test_performance_suites_require_log_view_and_options_left(tmp_path: Path) -> None:
    missing_log_view = tmp_path / "missing-log-view.toml"
    missing_log_view.write_text(
        """
[suite]
id = "bad-scaling"
title = "Bad scaling suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1, 2]

[collect]
options_left = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Performance/scaling suite 'bad-scaling'.*petsc_log_view"):
        load_suite(missing_log_view)

    missing_options_left = tmp_path / "missing-options-left.toml"
    missing_options_left.write_text(
        """
[suite]
id = "performance-suite"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[overrides.output]
preset = "performance"

[collect]
petsc_log_view = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Performance/scaling suite 'performance-suite'.*options_left"):
        load_suite(missing_options_left)


def test_committed_performance_suites_collect_petsc_evidence() -> None:
    for suite_toml in sorted((ROOT / "benchmarks" / "suites").glob("*.toml")):
        spec = load_suite(suite_toml)
        label = f"{spec.id} {spec.title}".casefold()
        performance = "scaling" in label or spec.overrides.get("output", {}).get("preset") == "performance"
        if not performance:
            continue
        assert spec.collect.get("petsc_log_view") or spec.collect.get("log_view"), suite_toml
        assert spec.collect.get("options_left"), suite_toml


def test_suite_rejects_structured_environment_values(tmp_path: Path) -> None:
    suite_toml = tmp_path / "bad-environment.toml"
    suite_toml.write_text(
        """
[suite]
id = "bad-environment"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[environment]
OMP_NUM_THREADS = ["1", "2"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[environment\]\.OMP_NUM_THREADS"):
        load_suite(suite_toml)


def test_suite_rejects_unknown_collect_fields(tmp_path: Path) -> None:
    suite_toml = tmp_path / "bad-collect.toml"
    suite_toml.write_text(
        """
[suite]
id = "bad-collect"
cases = ["3d-heterogeneous-ssr-p4"]
profiles = ["pmg-deflated-baseline"]
ranks = [1]

[collect]
archive_tarball = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"\[collect\].*archive_tarball"):
        load_suite(suite_toml)


def _flag_value(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]
