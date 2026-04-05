from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]

DEFAULT_FINE_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L2.msh"
DEFAULT_COARSE_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
DEFAULT_STATE_NPZ = ROOT / "artifacts" / "l2_p2_hypre_step1_for_mixed_pmg" / "data" / "petsc_run.npz"
DEFAULT_STATE_RUN_INFO = ROOT / "artifacts" / "l2_p2_hypre_step1_for_mixed_pmg" / "data" / "run_info.json"
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "l2_p2_mixed_pmg_shell_scaling"
DEFAULT_REPORT = SCRIPT_DIR / "report_p2_l2_mixed_pmg_shell_scaling.md"
DEFAULT_RANKS = (1, 2, 4, 8)

PETSC_OPTIONS = (
    "manualmg_coarse_operator_source=direct_elastic_full_system",
    "mg_levels_ksp_type=chebyshev",
    "mg_levels_ksp_max_it=3",
    "mg_levels_pc_type=jacobi",
    "mg_coarse_ksp_type=cg",
    "mg_coarse_max_it=4",
    "mg_coarse_rtol=0.0",
    "pc_hypre_boomeramg_numfunctions=3",
    "pc_hypre_boomeramg_nodal_coarsen=6",
    "pc_hypre_boomeramg_nodal_coarsen_diag=1",
    "pc_hypre_boomeramg_vec_interp_variant=3",
    "pc_hypre_boomeramg_vec_interp_qmax=4",
    "pc_hypre_boomeramg_vec_interp_smooth=true",
    "pc_hypre_boomeramg_coarsen_type=HMIS",
    "pc_hypre_boomeramg_interp_type=ext+i",
    "pc_hypre_boomeramg_P_max=4",
    "pc_hypre_boomeramg_strong_threshold=0.5",
    "pc_hypre_boomeramg_max_iter=4",
    "pc_hypre_boomeramg_tol=0.0",
    "pc_hypre_boomeramg_relax_type_all=symmetric-SOR/Jacobi",
)

BREAKDOWN_KEYS = (
    ("fine_pre", "manualmg_fine_pre_smoother_time_total_s"),
    ("fine_post", "manualmg_fine_post_smoother_time_total_s"),
    ("mid_pre", "manualmg_mid_pre_smoother_time_total_s"),
    ("mid_post", "manualmg_mid_post_smoother_time_total_s"),
    ("fine_residual", "manualmg_fine_residual_time_total_s"),
    ("mid_residual", "manualmg_mid_residual_time_total_s"),
    ("restrict_f2m", "manualmg_restrict_fine_to_mid_time_total_s"),
    ("restrict_m2c", "manualmg_restrict_mid_to_coarse_time_total_s"),
    ("prolong_c2m", "manualmg_prolong_coarse_to_mid_time_total_s"),
    ("prolong_m2f", "manualmg_prolong_mid_to_fine_time_total_s"),
    ("vector_sum", "manualmg_vector_sum_time_total_s"),
    ("coarse_hypre", "manualmg_coarse_hypre_time_total_s"),
)

RUNTIME_STAGE_KEYS = (
    ("problem_build", "problem_build_elapsed_s"),
    ("operator_build", "operator_build_elapsed_s"),
    ("linear_setup_solve", "solve_plus_setup_elapsed_s"),
    ("other", "runtime_other_elapsed_s"),
)


def _parse_ranks(text: str) -> tuple[int, ...]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one rank count.")
    return tuple(values)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_case(*, ranks: int, out_dir: Path, fine_mesh: Path, coarse_mesh: Path, state_npz: Path, state_run_info: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    cmd = [
        "mpirun",
        "-n",
        str(int(ranks)),
        sys.executable,
        str(SCRIPT_DIR / "probe_hypre_frozen.py"),
        "--out-dir",
        str(out_dir),
        "--state-npz",
        str(state_npz),
        "--state-run-info",
        str(state_run_info),
        "--state-selector",
        "final",
        "--mesh-path",
        str(fine_mesh),
        "--pmg-coarse-mesh-path",
        str(coarse_mesh),
        "--elem-type",
        "P2",
        "--node-ordering",
        "original",
        "--outer-solver-family",
        "repo",
        "--solver-type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc-backend",
        "pmg_shell",
        "--pmat-source",
        "tangent",
        "--linear-tolerance",
        "1e-3",
        "--linear-max-iter",
        "80",
    ]
    for opt in PETSC_OPTIONS:
        cmd.extend(["--petsc-opt", opt])
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, object], key: str, *, reduce: str | None = None, default: float = 0.0) -> float:
    if reduce is not None:
        reduced_key = f"{key}_{reduce}"
        if reduced_key in payload:
            return float(payload[reduced_key])
    if key in payload:
        return float(payload[key])
    return float(default)


def _load_case(run_info_path: Path) -> dict[str, object]:
    data = _read_json(run_info_path)
    apply_count = int(data.get("manualmg_apply_count", 0))
    wall_runtime = _metric(data, "runtime_seconds", reduce="max")
    wall_setup = _metric(data, "setup_elapsed_s", reduce="max")
    wall_solve = _metric(data, "solve_elapsed_s", reduce="max")
    wall_setup_plus_solve = _metric(data, "solve_plus_setup_elapsed_s", reduce="max")
    pc_setup_wall = _metric(data, "preconditioner_setup_time_total", reduce="max")
    pc_apply_wall = _metric(data, "preconditioner_apply_time_total", reduce="max")

    breakdown_max = {label: _metric(data, key, reduce="max") for label, key in BREAKDOWN_KEYS}
    summed_apply_parts_wall = float(sum(breakdown_max.values()))
    unattributed_apply_wall = max(pc_apply_wall - summed_apply_parts_wall, 0.0)
    breakdown_max["unattributed"] = unattributed_apply_wall

    avg_per_apply_wall = {
        label: (value / float(apply_count) if apply_count else 0.0)
        for label, value in breakdown_max.items()
    }
    runtime_stages_max = {label: _metric(data, key, reduce="max") for label, key in RUNTIME_STAGE_KEYS}

    return {
        "run_info": data,
        "iteration_count": int(data["iteration_count"]),
        "final_relative_residual": float(data["final_relative_residual"]),
        "runtime_seconds_max": wall_runtime,
        "setup_elapsed_s_max": wall_setup,
        "solve_elapsed_s_max": wall_solve,
        "solve_plus_setup_elapsed_s_max": wall_setup_plus_solve,
        "preconditioner_setup_time_total_max": pc_setup_wall,
        "preconditioner_apply_time_total_max": pc_apply_wall,
        "manualmg_apply_count": apply_count,
        "manualmg_coarse_ksp_iterations_total": int(data.get("manualmg_coarse_ksp_iterations_total", 0)),
        "manualmg_fine_smoother_iterations_total": int(data.get("manualmg_fine_smoother_iterations_total", 0)),
        "manualmg_mid_smoother_iterations_total": int(data.get("manualmg_mid_smoother_iterations_total", 0)),
        "manualmg_coarse_ksp_type": str(data.get("manualmg_coarse_ksp_type", "")),
        "manualmg_coarse_pc_type": str(data.get("manualmg_coarse_pc_type", "")),
        "manualmg_fine_ksp_type": str(data.get("manualmg_fine_ksp_type", "")),
        "manualmg_fine_pc_type": str(data.get("manualmg_fine_pc_type", "")),
        "manualmg_mid_ksp_type": str(data.get("manualmg_mid_ksp_type", "")),
        "manualmg_mid_pc_type": str(data.get("manualmg_mid_pc_type", "")),
        "breakdown_max": breakdown_max,
        "avg_per_apply_wall": avg_per_apply_wall,
        "runtime_stages_max": runtime_stages_max,
        "run_info_path": str(run_info_path),
    }


def _fmt(x: float, digits: int = 3) -> str:
    return f"{float(x):.{digits}f}"


def _report_relpath(report_path: Path, target: Path | str) -> str:
    return os.path.relpath(Path(target).resolve(), report_path.parent.resolve())


def _write_summary(out_root: Path, results: dict[int, dict[str, object]]) -> Path:
    rank1 = results[min(results)]
    summary: dict[str, object] = {"ranks": sorted(results), "cases": {}}
    for ranks in sorted(results):
        case = dict(results[ranks])
        speedup = float(rank1["runtime_seconds_max"]) / float(case["runtime_seconds_max"])
        efficiency = speedup / float(ranks)
        case["speedup_vs_rank1_runtime"] = speedup
        case["efficiency_vs_rank1_runtime"] = efficiency
        summary["cases"][str(ranks)] = case
    path = out_root / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def _plot_wall_time(out_root: Path, results: dict[int, dict[str, object]]) -> Path:
    ranks = np.asarray(sorted(results), dtype=np.int64)
    runtime = np.asarray([results[r]["runtime_seconds_max"] for r in ranks], dtype=np.float64)
    solve = np.asarray([results[r]["solve_plus_setup_elapsed_s_max"] for r in ranks], dtype=np.float64)
    problem = np.asarray([results[r]["runtime_stages_max"]["problem_build"] for r in ranks], dtype=np.float64)
    operator = np.asarray([results[r]["runtime_stages_max"]["operator_build"] for r in ranks], dtype=np.float64)
    setup = np.asarray([results[r]["setup_elapsed_s_max"] for r in ranks], dtype=np.float64)
    pc_apply = np.asarray([results[r]["preconditioner_apply_time_total_max"] for r in ranks], dtype=np.float64)
    speedup_runtime = runtime[0] / runtime
    speedup_problem = problem[0] / problem
    speedup_solve = solve[0] / solve
    speedup_setup = setup[0] / setup
    speedup_pc_apply = pc_apply[0] / pc_apply

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ranks, runtime, marker="o", label="Runtime")
    axes[0].plot(ranks, solve, marker="s", label="Setup+Solve")
    axes[0].plot(ranks, problem, marker="^", label="Problem build")
    axes[0].plot(ranks, operator, marker="d", label="Operator build")
    axes[0].set_xlabel("MPI Ranks")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Wall Time")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ranks, speedup_runtime, marker="o", label="Runtime")
    axes[1].plot(ranks, speedup_problem, marker="^", label="Problem build")
    axes[1].plot(ranks, speedup_solve, marker="s", label="Setup+Solve")
    axes[1].plot(ranks, speedup_setup, marker="d", label="Setup")
    axes[1].plot(ranks, speedup_pc_apply, marker="x", label="PC apply")
    axes[1].plot(ranks, ranks / ranks[0], linestyle="--", color="black", label="Ideal")
    axes[1].set_xlabel("MPI Ranks")
    axes[1].set_ylabel("Speedup vs 1 rank")
    axes[1].set_title("Stage Speedups")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    path = out_root / "wall_time_scaling.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_breakdown(out_root: Path, results: dict[int, dict[str, object]], *, per_apply: bool) -> Path:
    ranks = [int(r) for r in sorted(results)]
    labels = [label for label, _ in BREAKDOWN_KEYS] + ["unattributed"]
    data = np.asarray(
        [
            [
                float(results[r]["avg_per_apply_wall" if per_apply else "breakdown_max"].get(label, 0.0))
                for label in labels
            ]
            for r in ranks
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(ranks), dtype=np.float64)
    for idx, label in enumerate(labels):
        ax.bar(ranks, data[:, idx], bottom=bottom, label=label)
        bottom += data[:, idx]
    ax.set_xlabel("MPI Ranks")
    ax.set_ylabel("Wall seconds per apply (rank max)" if per_apply else "Wall seconds (rank max)")
    ax.set_title("PCApply Wall-Time Breakdown" + (" Per Apply" if per_apply else ""))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path = out_root / ("pc_breakdown_per_apply.png" if per_apply else "pc_breakdown_sum.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_iterations(out_root: Path, results: dict[int, dict[str, object]]) -> Path:
    ranks = np.asarray(sorted(results), dtype=np.int64)
    outer = np.asarray([results[r]["iteration_count"] for r in ranks], dtype=np.float64)
    coarse = np.asarray([results[r]["manualmg_coarse_ksp_iterations_total"] for r in ranks], dtype=np.float64)
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.plot(ranks, outer, marker="o", label="Outer iterations")
    ax1.set_xlabel("MPI Ranks")
    ax1.set_ylabel("Outer iterations")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log", base=2)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(ranks, coarse, marker="s", color="tab:red", label="Coarse CG iterations total")
    ax2.set_ylabel("Coarse CG iterations total")
    ax2.set_yscale("log", base=2)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    ax1.set_title("Iteration Counts")
    fig.tight_layout()
    path = out_root / "iteration_counts.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_report(report_path: Path, out_root: Path, results: dict[int, dict[str, object]], plots: dict[str, Path]) -> None:
    rank1 = results[min(results)]
    plot_wall = _report_relpath(report_path, plots["wall_time"])
    plot_sum = _report_relpath(report_path, plots["breakdown_sum"])
    plot_per_apply = _report_relpath(report_path, plots["breakdown_per_apply"])
    plot_iters = _report_relpath(report_path, plots["iterations"])
    summary_lines = [
        "# P2(L2) Mixed PMG-Shell Scaling",
        "",
        "## Configuration",
        "",
        "- Fine hierarchy: `P2(L2) -> P1(L2) -> P1(L1)`",
        "- Backend: `pc_backend=pmg_shell`",
        "- Fine/mid smoother: `chebyshev + jacobi`, `3` steps",
        "- Coarse solve: `cg + hypre(boomeramg)`",
        "- Coarse operator source: `direct_elastic_full_system`",
        "- Frozen state source: `artifacts/l2_p2_hypre_step1_for_mixed_pmg/data/petsc_run.npz`",
        "- All scaling tables use wall-time maxima across ranks, not rank-summed CPU time",
        "",
        "## Summary",
        "",
        "| Ranks | Outer Iters | Final Rel Residual | Runtime Max [s] | Setup+Solve Max [s] | Speedup | Efficiency |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ranks in sorted(results):
        case = results[ranks]
        speedup = float(rank1["runtime_seconds_max"]) / float(case["runtime_seconds_max"])
        efficiency = speedup / float(ranks)
        summary_lines.append(
            f"| {ranks} | {case['iteration_count']} | {_fmt(case['final_relative_residual'], 6)} | "
            f"{_fmt(case['runtime_seconds_max'])} | {_fmt(case['solve_plus_setup_elapsed_s_max'])} | "
            f"{_fmt(speedup)}x | {_fmt(efficiency)} |"
        )

    summary_lines.extend(
        [
            "",
            "## Timing Table",
            "",
            "| Ranks | Problem Build Max [s] | Operator Build Max [s] | Setup Max [s] | Solve Max [s] | Other Max [s] | PC Setup Max [s] | PC Apply Max [s] | Applies | Coarse CG Total |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ranks in sorted(results):
        case = results[ranks]
        summary_lines.append(
            f"| {ranks} | {_fmt(case['runtime_stages_max']['problem_build'])} | {_fmt(case['runtime_stages_max']['operator_build'])} | "
            f"{_fmt(case['setup_elapsed_s_max'])} | {_fmt(case['solve_elapsed_s_max'])} | {_fmt(case['runtime_stages_max']['other'])} | "
            f"{_fmt(case['preconditioner_setup_time_total_max'])} | {_fmt(case['preconditioner_apply_time_total_max'])} | "
            f"{case['manualmg_apply_count']} | {case['manualmg_coarse_ksp_iterations_total']} |"
        )

    summary_lines.extend(
        [
            "",
            "## Speedup By Stage",
            "",
            "| Ranks | Runtime | Problem Build | Operator Build | Setup+Solve | Setup | Solve | PC Setup | PC Apply |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ranks in sorted(results):
        case = results[ranks]
        summary_lines.append(
            f"| {ranks} | "
            f"{_fmt(float(rank1['runtime_seconds_max']) / float(case['runtime_seconds_max']))}x | "
            f"{_fmt(float(rank1['runtime_stages_max']['problem_build']) / float(case['runtime_stages_max']['problem_build']))}x | "
            f"{_fmt(float(rank1['runtime_stages_max']['operator_build']) / float(case['runtime_stages_max']['operator_build']))}x | "
            f"{_fmt(float(rank1['solve_plus_setup_elapsed_s_max']) / float(case['solve_plus_setup_elapsed_s_max']))}x | "
            f"{_fmt(float(rank1['setup_elapsed_s_max']) / float(case['setup_elapsed_s_max']))}x | "
            f"{_fmt(float(rank1['solve_elapsed_s_max']) / float(case['solve_elapsed_s_max']))}x | "
            f"{_fmt(float(rank1['preconditioner_setup_time_total_max']) / float(case['preconditioner_setup_time_total_max']))}x | "
            f"{_fmt(float(rank1['preconditioner_apply_time_total_max']) / float(case['preconditioner_apply_time_total_max']))}x |"
        )

    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The gap between `Runtime Max` and `Setup+Solve Max` is mostly `Problem Build Max`, not hidden solver time.",
            "- The stage columns are independent rank-max wall times, so they are not additive; the slowest rank in one stage need not be the slowest rank in another.",
            "- `Other Max` is the leftover wall time after problem build, operator build, and linear setup+solve; it is tiny in these runs.",
            "- `Operator Build Max` is a rank-max wall time. It can be larger than the local value on rank 0 because the slowest rank sets the column.",
            "- The PC breakdown tables below are also rank-max wall times, not summed CPU seconds over all MPI ranks.",
        ]
    )

    breakdown_labels = [label for label, _ in BREAKDOWN_KEYS] + ["unattributed"]
    summary_lines.extend(
        [
            "",
            "## Rank-Max PCApply Wall-Time Breakdown",
            "",
            "| Ranks | " + " | ".join(label for label in breakdown_labels) + " |",
            "| ---: | " + " | ".join("---:" for _ in breakdown_labels) + " |",
        ]
    )
    for ranks in sorted(results):
        case = results[ranks]
        row = [f"{_fmt(case['breakdown_max'].get(label, 0.0))}" for label in breakdown_labels]
        summary_lines.append(f"| {ranks} | " + " | ".join(row) + " |")

    summary_lines.extend(
        [
            "",
            "## Rank-Max PCApply Wall-Time Breakdown Per Apply",
            "",
            "| Ranks | " + " | ".join(label for label in breakdown_labels) + " |",
            "| ---: | " + " | ".join("---:" for _ in breakdown_labels) + " |",
        ]
    )
    for ranks in sorted(results):
        case = results[ranks]
        row = [f"{_fmt(case['avg_per_apply_wall'].get(label, 0.0), 4)}" for label in breakdown_labels]
        summary_lines.append(f"| {ranks} | " + " | ".join(row) + " |")

    summary_lines.extend(
        [
            "",
            "## Plots",
            "",
            f"[wall_time_scaling.png]({plot_wall})",
            "",
            f"![Wall Time Scaling]({plot_wall})",
            "",
            f"[pc_breakdown_sum.png]({plot_sum})",
            "",
            f"![PCApply Breakdown Sum]({plot_sum})",
            "",
            f"[pc_breakdown_per_apply.png]({plot_per_apply})",
            "",
            f"![PCApply Breakdown Per Apply]({plot_per_apply})",
            "",
            f"[iteration_counts.png]({plot_iters})",
            "",
            f"![Iteration Counts]({plot_iters})",
            "",
            "## Raw Artifacts",
            "",
        ]
    )
    for ranks in sorted(results):
        case = results[ranks]
        rel_run_info = _report_relpath(report_path, str(case["run_info_path"]))
        summary_lines.append(f"- rank {ranks}: [{Path(case['run_info_path']).name}]({rel_run_info})")

    report_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rank scaling for the mixed P2(L2) PMG-shell frozen probe.")
    parser.add_argument("--ranks", default="1,2,4,8")
    parser.add_argument("--fine-mesh", default=str(DEFAULT_FINE_MESH))
    parser.add_argument("--coarse-mesh", default=str(DEFAULT_COARSE_MESH))
    parser.add_argument("--state-npz", default=str(DEFAULT_STATE_NPZ))
    parser.add_argument("--state-run-info", default=str(DEFAULT_STATE_RUN_INFO))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    ranks = _parse_ranks(args.ranks)
    fine_mesh = Path(args.fine_mesh).resolve()
    coarse_mesh = Path(args.coarse_mesh).resolve()
    state_npz = Path(args.state_npz).resolve()
    state_run_info = Path(args.state_run_info).resolve()
    out_root = _mkdir(Path(args.out_root).resolve())

    for nproc in ranks:
        case_dir = _mkdir(out_root / f"rank{int(nproc)}")
        if not args.skip_run:
            _run_case(
                ranks=int(nproc),
                out_dir=case_dir,
                fine_mesh=fine_mesh,
                coarse_mesh=coarse_mesh,
                state_npz=state_npz,
                state_run_info=state_run_info,
            )

    results: dict[int, dict[str, object]] = {}
    for nproc in ranks:
        run_info_path = out_root / f"rank{int(nproc)}" / "data" / "run_info.json"
        results[int(nproc)] = _load_case(run_info_path)

    _write_summary(out_root, results)
    plots_dir = _mkdir(out_root / "plots")
    plots = {
        "wall_time": _plot_wall_time(plots_dir, results),
        "breakdown_sum": _plot_breakdown(plots_dir, results, per_apply=False),
        "breakdown_per_apply": _plot_breakdown(plots_dir, results, per_apply=True),
        "iterations": _plot_iterations(plots_dir, results),
    }
    _write_report(Path(args.report).resolve(), out_root, results, plots)


if __name__ == "__main__":
    main()
