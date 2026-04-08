#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from slope_stability.fem import quadrature_volume_3d


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent
REPO_ROOT = STUDY_DIR.parents[1]
DATA_DIR = STUDY_DIR / "data"
FIG_DIR = STUDY_DIR / "figures"
GEN_DIR = STUDY_DIR / "generated"
REFERENCE_RULES = [1, 4, 11, 24, 45]
REFERENCE_TETRA_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
REFERENCE_TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


ELEMENT_COLORS = {
    "P1": "#1f2937",
    "P2": "#0f766e",
    "P4": "#7c3aed",
}

QUADRATURE_COLORS = {
    1: "#111827",
    4: "#0f766e",
    5: "#2563eb",
    11: "#9333ea",
    15: "#d97706",
    24: "#dc2626",
    31: "#059669",
    45: "#7c3aed",
}

TEXT_WIDTH_CM = 18.0
CM_TO_INCH = 1.0 / 2.54
BASE_FONT_SIZE = 11.0
SMALL_FONT_SIZE = 9.0
TICK_FONT_SIZE = 8.75


def _quadrature_matrix(study_meta: dict) -> dict[str, list[int]]:
    matrix = study_meta.get("quadrature_matrix")
    if matrix:
        return {str(key): [int(value) for value in values] for key, values in dict(matrix).items()}
    shared = [int(value) for value in study_meta["quadrature_rules"]]
    return {str(elem_type): list(shared) for elem_type in study_meta["element_types"]}


def _allowed_pairs(study_meta: dict) -> set[tuple[str, int]]:
    return {
        (str(elem_type), int(rule))
        for elem_type, rules in _quadrature_matrix(study_meta).items()
        for rule in rules
    }


def _ordered_runs(study_meta: dict, runs: pd.DataFrame) -> pd.DataFrame:
    matrix = _quadrature_matrix(study_meta)
    order_map: dict[tuple[str, int], int] = {}
    counter = 0
    for elem_type in study_meta["element_types"]:
        for rule in matrix[str(elem_type)]:
            order_map[(str(elem_type), int(rule))] = counter
            counter += 1
    ordered = runs.copy()
    if ordered.empty:
        return ordered
    ordered["__order"] = ordered.apply(lambda row: order_map.get((str(row["elem_type"]), int(row["quadrature_rule"])), 10**9), axis=1)
    ordered = ordered.sort_values(["elem_type", "__order"]).drop(columns="__order")
    return ordered


def _load_inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    study_meta = json.loads((DATA_DIR / "study_meta.json").read_text())
    runs = pd.read_csv(DATA_DIR / "runs.csv")
    curves = pd.read_csv(DATA_DIR / "continuation_curves.csv")
    allowed_pairs = _allowed_pairs(study_meta)
    runs = runs[runs.apply(lambda row: (str(row["elem_type"]), int(row["quadrature_rule"])) in allowed_pairs, axis=1)].copy()
    curves = curves[curves.apply(lambda row: (str(row["elem_type"]), int(row["quadrature_rule"])) in allowed_pairs, axis=1)].copy()
    for col in ["timed_out", "omega_monotone", "lambda_monotone", "reached_omega_target"]:
        if col in runs.columns:
            runs[col] = runs[col].map(lambda value: str(value).strip().lower() == "true")
    return study_meta, runs, curves


def _artifact_root(study_meta: dict) -> Path:
    return Path(study_meta["artifact_dir"])


def _style() -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
            "font.family": "serif",
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "legend.fontsize": SMALL_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linewidth": 0.45,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _figure_size(height_cm: float) -> tuple[float, float]:
    return (TEXT_WIDTH_CM * CM_TO_INCH, float(height_cm) * CM_TO_INCH)


def _format_num(value: float, *, digits: int = 4) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{int(digits)}g}"


def _format_runtime_seconds(value: float) -> str:
    if pd.isna(value):
        return "--"
    return str(int(round(float(value))))


def _fmt_rule(rule: int) -> str:
    return f"q{int(rule)}"


def _tex_escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _load_progress_events(progress_jsonl_path: Path) -> list[dict]:
    events: list[dict] = []
    if not progress_jsonl_path.exists():
        return events
    for raw_line in progress_jsonl_path.read_text().splitlines():
        text = raw_line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _has_curve(row: pd.Series) -> bool:
    return bool(int(row["accepted_steps"]) > 0)


def _is_stable_target(row: pd.Series) -> bool:
    return bool(row["status"] == "success" and row["reached_omega_target"] and row["omega_monotone"] and row["lambda_monotone"])


def _is_monotone_curve(row: pd.Series) -> bool:
    return bool(row["accepted_steps"] > 0 and row["omega_monotone"] and row["lambda_monotone"])


def _stop_reason_label(reason: object) -> str:
    mapping = {
        "finished": "finished",
        "d_lambda_diff_scaled_min": "scaled dlambda min",
        "newton_maxit_last_step": "last-step maxit",
        "capped_linear_zero_correction_stall": "capped linear zero-corr. stall",
        "zero_correction_stall": "zero-corr. stall",
        "timeout": "timeout",
    }
    key = str(reason).strip()
    return mapping.get(key, key.replace("_", " "))


def _reference_quadrature_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for rule in REFERENCE_RULES:
        xi, wf = quadrature_volume_3d("P4", rule)
        lambda_origin = 1.0 - np.sum(xi, axis=0)
        for idx in range(xi.shape[1]):
            weight = float(wf[idx])
            rows.append(
                {
                    "quadrature_rule": int(rule),
                    "point_id": int(idx + 1),
                    "x": float(xi[0, idx]),
                    "y": float(xi[1, idx]),
                    "z": float(xi[2, idx]),
                    "lambda_origin": float(lambda_origin[idx]),
                    "weight": weight,
                    "weight_abs": abs(weight),
                    "weight_sign": "negative" if weight < 0.0 else "positive",
                }
            )
    return pd.DataFrame(rows)


def _write_reference_quadrature_csvs(points_df: pd.DataFrame) -> None:
    summary_rows: list[dict] = []
    for rule in REFERENCE_RULES:
        subset = points_df[points_df["quadrature_rule"] == int(rule)].copy()
        summary_rows.append(
            {
                "quadrature_rule": int(rule),
                "n_points": int(subset.shape[0]),
                "negative_weight_count": int((subset["weight"] < 0.0).sum()),
                "min_weight": float(subset["weight"].min()),
                "max_weight": float(subset["weight"].max()),
                "sum_weight": float(subset["weight"].sum()),
            }
        )
    points_df.to_csv(DATA_DIR / "reference_tetra_quadrature_points.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(DATA_DIR / "reference_tetra_quadrature_summary.csv", index=False)


def _write_reference_quadrature_summary_table(points_df: pd.DataFrame, study_meta: dict) -> None:
    used_by_rules: dict[int, list[str]] = {}
    matrix = _quadrature_matrix(study_meta)
    for elem_type, rules in matrix.items():
        for rule in rules:
            used_by_rules.setdefault(int(rule), []).append(str(elem_type))

    lines = [
        r"\begin{tabularx}{\textwidth}{@{}lrrrrX@{}}",
        r"\toprule",
        r"Rule & Points & Negative weights & $w_{\min}$ & $w_{\max}$ & Used in study for \\",
        r"\midrule",
    ]
    for rule in REFERENCE_RULES:
        subset = points_df[points_df["quadrature_rule"] == int(rule)].copy()
        lines.append(
            f"{_fmt_rule(rule)} & "
            f"{int(subset.shape[0])} & "
            f"{int((subset['weight'] < 0.0).sum())} & "
            f"{_format_num(float(subset['weight'].min()), digits=6)} & "
            f"{_format_num(float(subset['weight'].max()), digits=6)} & "
            f"{'/'.join(used_by_rules.get(int(rule), [])) or '--'} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    (GEN_DIR / "reference_quadrature_summary_table.tex").write_text("\n".join(lines))


def _write_reference_quadrature_detail_tables(points_df: pd.DataFrame) -> None:
    blocks: list[str] = []
    for rule in REFERENCE_RULES:
        subset = points_df[points_df["quadrature_rule"] == int(rule)].copy()
        size_command = r"\scriptsize" if subset.shape[0] <= 24 else r"\tiny"
        blocks.extend(
            [
                rf"\subsection*{{Reference rule {_fmt_rule(rule)}}}",
                rf"{size_command}",
                r"\begin{table}[H]",
                r"\centering",
                rf"\caption{{Reference tetrahedron points and weights for {_fmt_rule(rule)}. Coordinates are reported on the unit tetrahedron with vertices $(0,0,0)$, $(1,0,0)$, $(0,1,0)$, and $(0,0,1)$.}}",
                r"\begin{tabular}{@{}rrrrrr@{}}",
                r"\toprule",
                r"Pt & $x$ & $y$ & $z$ & $1-x-y-z$ & $w$ \\",
                r"\midrule",
            ]
        )
        for _, row in subset.iterrows():
            blocks.append(
                f"{int(row['point_id'])} & "
                f"{_format_num(row['x'], digits=8)} & "
                f"{_format_num(row['y'], digits=8)} & "
                f"{_format_num(row['z'], digits=8)} & "
                f"{_format_num(row['lambda_origin'], digits=8)} & "
                f"{_format_num(row['weight'], digits=8)} \\\\"
            )
        blocks.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\normalsize", ""])
    (GEN_DIR / "reference_quadrature_detail_tables.tex").write_text("\n".join(blocks))


def _plot_reference_tetra_quadratures(points_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=_figure_size(11.4))
    axes = [fig.add_subplot(2, 3, idx + 1, projection="3d") for idx in range(6)]
    max_abs_weight = float(points_df["weight_abs"].max()) if not points_df.empty else 1.0

    for ax, rule in zip(axes[: len(REFERENCE_RULES)], REFERENCE_RULES, strict=True):
        subset = points_df[points_df["quadrature_rule"] == int(rule)].copy()
        for i0, i1 in REFERENCE_TETRA_EDGES:
            edge = REFERENCE_TETRA_VERTICES[[i0, i1], :]
            ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="#9ca3af", linewidth=0.8, zorder=1)
        positive = subset[subset["weight"] >= 0.0]
        negative = subset[subset["weight"] < 0.0]
        if not positive.empty:
            sizes = 30.0 + 260.0 * np.sqrt(positive["weight_abs"].to_numpy(dtype=float) / max_abs_weight)
            ax.scatter(
                positive["x"],
                positive["y"],
                positive["z"],
                s=sizes,
                c="#0f766e",
                edgecolors="#083344",
                linewidths=0.35,
                depthshade=False,
                zorder=3,
            )
        if not negative.empty:
            sizes = 30.0 + 260.0 * np.sqrt(negative["weight_abs"].to_numpy(dtype=float) / max_abs_weight)
            ax.scatter(
                negative["x"],
                negative["y"],
                negative["z"],
                s=sizes,
                c="#b91c1c",
                edgecolors="#7f1d1d",
                linewidths=0.35,
                depthshade=False,
                zorder=4,
            )
        ax.set_title(fr"{_fmt_rule(rule)} ({int(subset.shape[0])} pts)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=22.0, azim=38.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_zticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=7.2, pad=0.0)
        ax.set_xlabel(r"$x$", labelpad=-6.0)
        ax.set_ylabel(r"$y$", labelpad=-8.0)
        ax.set_zlabel(r"$z$", labelpad=-6.0)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(False)

    legend_ax = axes[-1]
    legend_ax.axis("off")
    positive_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#0f766e", markeredgecolor="#083344", markersize=7, linewidth=0)
    negative_handle = Line2D([0], [0], marker="o", color="w", markerfacecolor="#b91c1c", markeredgecolor="#7f1d1d", markersize=7, linewidth=0)
    small_handle = Line2D([0], [0], marker="o", color="#374151", markerfacecolor="#d1d5db", markersize=5, linewidth=0)
    large_handle = Line2D([0], [0], marker="o", color="#374151", markerfacecolor="#d1d5db", markersize=10, linewidth=0)
    legend_ax.legend(
        [positive_handle, negative_handle, small_handle, large_handle],
        [r"$w > 0$", r"$w < 0$", r"small $|w|$", r"large $|w|$"],
        loc="center",
        frameon=False,
    )
    legend_ax.text2D(
        0.5,
        0.22,
        "Reference tetrahedron:\n$(0,0,0)$, $(1,0,0)$, $(0,1,0)$, $(0,0,1)$",
        ha="center",
        va="center",
        fontsize=SMALL_FONT_SIZE,
        transform=legend_ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reference_tetra_quadratures.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_raw_capture_summary(artifact_dir: Path, *, omega_target: float) -> dict | None:
    run_info_path = artifact_dir / "data" / "run_info.json"
    npz_path = artifact_dir / "data" / "petsc_run.npz"
    progress_jsonl_path = artifact_dir / "data" / "progress.jsonl"
    if not run_info_path.exists() or not npz_path.exists():
        return None

    run_info = json.loads(run_info_path.read_text())
    progress_events: list[dict] = []
    if progress_jsonl_path.exists():
        for line in progress_jsonl_path.read_text().splitlines():
            text = line.strip()
            if not text:
                continue
            try:
                progress_events.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    stop_reason = ""
    attempt_count_total = 0
    for event in reversed(progress_events):
        reason = str(event.get("stop_reason", "")).strip()
        if reason:
            stop_reason = reason
            break
    for event in progress_events:
        if str(event.get("event", "")).strip() == "attempt_complete":
            attempt_count_total += 1

    with np.load(npz_path, allow_pickle=True) as npz:
        lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
        omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
        umax_hist = np.asarray(npz["Umax_hist"], dtype=np.float64)
        step_newton = np.asarray(npz["stats_step_newton_iterations"], dtype=np.float64) if "stats_step_newton_iterations" in npz.files else np.asarray([], dtype=np.float64)
        step_linear = np.asarray(npz["stats_step_linear_iterations"], dtype=np.float64) if "stats_step_linear_iterations" in npz.files else np.asarray([], dtype=np.float64)

    return {
        "artifact_dir": str(artifact_dir),
        "runtime_seconds": float(run_info["run_info"]["runtime_seconds"]),
        "step_count": int(lambda_hist.size),
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "umax_hist": umax_hist,
        "curve_df": pd.DataFrame(
            {
                "step": np.arange(1, lambda_hist.size + 1, dtype=int),
                "omega": omega_hist,
                "lambda": lambda_hist,
                "umax": umax_hist,
            }
        ),
        "lambda_last": float(lambda_hist[-1]) if lambda_hist.size else float("nan"),
        "omega_last": float(omega_hist[-1]) if omega_hist.size else float("nan"),
        "umax_last": float(umax_hist[-1]) if umax_hist.size else float("nan"),
        "omega_monotone": bool(np.all(np.diff(omega_hist) >= -1.0e-10)) if omega_hist.size else True,
        "lambda_monotone": bool(np.all(np.diff(lambda_hist) >= -1.0e-10)) if lambda_hist.size else True,
        "reached_target": bool(omega_hist.size and float(omega_hist[-1]) >= float(omega_target) - 1.0e-8),
        "stop_reason": stop_reason or "finished",
        "attempt_count_total": int(attempt_count_total),
        "step_newton_total": float(np.nansum(step_newton)) if step_newton.size else float("nan"),
        "step_linear_total": float(np.nansum(step_linear)) if step_linear.size else float("nan"),
        "max_step_newton": float(np.nanmax(step_newton)) if step_newton.size else float("nan"),
        "max_step_linear": float(np.nanmax(step_linear)) if step_linear.size else float("nan"),
    }


def _run_display_name(run_id: str) -> str:
    elem_type, quadrature_rule = _run_identity(run_id)
    return f"{elem_type}/{_fmt_rule(quadrature_rule)}"


def _run_identity(run_id: str) -> tuple[str, int]:
    prefix, qtoken = str(run_id).split("_q", maxsplit=1)
    return prefix.upper(), int(qtoken)


def _replacement_curve_frame(run_id: str, diagnostic_run: dict) -> pd.DataFrame:
    elem_type, quadrature_rule = _run_identity(run_id)
    replacement_curve = diagnostic_run["curve_df"].copy()
    replacement_curve["run_id"] = str(run_id)
    replacement_curve["elem_type"] = str(elem_type)
    replacement_curve["quadrature_rule"] = int(quadrature_rule)
    replacement_curve["newton_iterations"] = np.nan
    replacement_curve["linear_iterations"] = np.nan
    replacement_curve["newton_relres_end"] = np.nan
    replacement_curve["newton_relcorr_end"] = np.nan
    replacement_curve["attempt_count"] = np.nan
    replacement_curve["branch_efficiency"] = np.nan
    return replacement_curve[
        [
            "run_id",
            "elem_type",
            "quadrature_rule",
            "step",
            "omega",
            "lambda",
            "umax",
            "newton_iterations",
            "linear_iterations",
            "newton_relres_end",
            "newton_relcorr_end",
            "attempt_count",
            "branch_efficiency",
        ]
    ]


def _collect_no_dlambda_diagnostics(study_meta: dict, runs: pd.DataFrame) -> list[dict]:
    artifact_root = _artifact_root(study_meta)
    diagnostics: list[dict] = []
    for run_id in _ordered_runs(study_meta, runs)["run_id"].tolist():
        main_dir = artifact_root / str(run_id)
        diagnostic_dir = artifact_root / f"{run_id}_no_dlambda_scaled"
        if not diagnostic_dir.exists():
            continue
        main_run = _load_raw_capture_summary(main_dir, omega_target=float(study_meta["omega_final"]))
        diagnostic_run = _load_raw_capture_summary(diagnostic_dir, omega_target=float(study_meta["omega_final"]))
        if main_run is None or diagnostic_run is None:
            continue
        elem_type, quadrature_rule = _run_identity(str(run_id))
        diagnostics.append(
            {
                "run_id": str(run_id),
                "label": _run_display_name(str(run_id)),
                "elem_type": str(elem_type),
                "quadrature_rule": int(quadrature_rule),
                "main_run": main_run,
                "diagnostic_run": diagnostic_run,
            }
        )
    return diagnostics


def _promote_corrected_runs(
    runs: pd.DataFrame,
    curves: pd.DataFrame,
    diagnostics: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not diagnostics:
        return runs, curves

    updated_runs = runs.copy()
    updated_curves = curves.copy()
    for item in diagnostics:
        run_id = str(item["run_id"])
        diagnostic_run = item["diagnostic_run"]
        row_mask = updated_runs["run_id"] == run_id
        if not bool(row_mask.any()):
            continue
        target_reached = bool(diagnostic_run["reached_target"])
        updated_runs.loc[row_mask, "status"] = "success" if target_reached else "incomplete"
        updated_runs.loc[row_mask, "returncode"] = 0
        updated_runs.loc[row_mask, "timed_out"] = False
        updated_runs.loc[row_mask, "runtime_seconds_wall"] = float(diagnostic_run["runtime_seconds"])
        updated_runs.loc[row_mask, "runtime_seconds_capture"] = float(diagnostic_run["runtime_seconds"])
        updated_runs.loc[row_mask, "continuation_d_lambda_diff_scaled_min"] = -1.0
        updated_runs.loc[row_mask, "accepted_steps"] = int(diagnostic_run["step_count"])
        updated_runs.loc[row_mask, "omega_last"] = float(diagnostic_run["omega_last"])
        updated_runs.loc[row_mask, "lambda_last"] = float(diagnostic_run["lambda_last"])
        updated_runs.loc[row_mask, "umax_last"] = float(diagnostic_run["umax_last"])
        updated_runs.loc[row_mask, "omega_monotone"] = bool(diagnostic_run["omega_monotone"])
        updated_runs.loc[row_mask, "lambda_monotone"] = bool(diagnostic_run["lambda_monotone"])
        updated_runs.loc[row_mask, "reached_omega_target"] = bool(target_reached)
        updated_runs.loc[row_mask, "lambda_at_omega_target"] = (
            float(diagnostic_run["lambda_last"]) if target_reached else np.nan
        )
        updated_runs.loc[row_mask, "umax_at_omega_target"] = (
            float(diagnostic_run["umax_last"]) if target_reached else np.nan
        )
        updated_runs.loc[row_mask, "stop_reason"] = str(diagnostic_run["stop_reason"])
        updated_runs.loc[row_mask, "step_newton_iterations_total"] = float(diagnostic_run["step_newton_total"])
        updated_runs.loc[row_mask, "step_linear_iterations_total"] = float(diagnostic_run["step_linear_total"])
        updated_runs.loc[row_mask, "max_step_newton_iterations"] = float(diagnostic_run["max_step_newton"])
        updated_runs.loc[row_mask, "max_step_linear_iterations"] = float(diagnostic_run["max_step_linear"])
        updated_runs.loc[row_mask, "attempt_count_total"] = int(diagnostic_run["attempt_count_total"])
        updated_runs.loc[row_mask, "error"] = np.nan
        updated_runs.loc[row_mask, "traceback"] = np.nan

        updated_curves = updated_curves[updated_curves["run_id"] != run_id].copy()
        updated_curves = pd.concat(
            [updated_curves, _replacement_curve_frame(run_id, diagnostic_run)],
            ignore_index=True,
        )

    updated_curves = updated_curves.sort_values(["elem_type", "quadrature_rule", "step", "run_id"]).reset_index(drop=True)
    return updated_runs, updated_curves


def _write_diagnostic_data_files(diagnostics: list[dict]) -> None:
    for item in diagnostics:
        run_id = str(item["run_id"])
        diagnostic_run = item["diagnostic_run"]
        summary = {
            key: value
            for key, value in diagnostic_run.items()
            if key not in {"lambda_hist", "omega_hist", "umax_hist", "curve_df"}
        }
        (DATA_DIR / f"{run_id}_no_dlambda_scaled_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        diagnostic_run["curve_df"].to_csv(DATA_DIR / f"{run_id}_no_dlambda_scaled_curve.csv", index=False)


def _plot_dlambda_stop_diagnostic_overview(study_meta: dict, diagnostics: list[dict]) -> None:
    if not diagnostics:
        return
    n_panels = len(diagnostics)
    ncols = 1 if n_panels == 1 else 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=_figure_size(7.0 + 5.0 * (nrows - 1)), squeeze=False)
    flat_axes = axes.reshape(-1)

    for ax, item in zip(flat_axes, diagnostics, strict=False):
        main_run = item["main_run"]
        diagnostic_run = item["diagnostic_run"]
        rule_color = QUADRATURE_COLORS.get(int(item["quadrature_rule"]), "#374151")
        ax.plot(
            main_run["omega_hist"],
            main_run["lambda_hist"],
            marker="o",
            markersize=2.4,
            linewidth=1.35,
            linestyle=":",
            color=rule_color,
            label="original cutoff run",
        )
        ax.plot(
            diagnostic_run["omega_hist"],
            diagnostic_run["lambda_hist"],
            marker="o",
            markersize=2.4,
            linewidth=1.45,
            linestyle="-",
            color="#111827",
            label="rerun without scaled $\\Delta\\lambda$ stop",
        )
        ax.axvline(float(study_meta["omega_final"]), color="black", linestyle="--", linewidth=1.0)
        ax.set_title(item["label"])
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\lambda$")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend(loc="best", frameon=False)

    for ax in flat_axes[n_panels:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dlambda_stop_diagnostic_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_dlambda_stop_diagnostic_table(diagnostics: list[dict]) -> None:
    if not diagnostics:
        return
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}lrrXrrX@{}}",
        r"\toprule",
        r"Run & Main time [s] & Main $\omega_{\mathrm{last}}$ & Main stop & Rerun time [s] & Rerun $\omega_{\mathrm{last}}$ & Rerun stop \\",
        r"\midrule",
    ]
    for item in diagnostics:
        main_run = item["main_run"]
        diagnostic_run = item["diagnostic_run"]
        lines.append(
            f"{_tex_escape(item['label'])} & "
            f"{_format_runtime_seconds(main_run['runtime_seconds'])} & "
            f"{_format_num(main_run['omega_last'], digits=6)} & "
            f"{_tex_escape(_stop_reason_label(main_run['stop_reason']))} & "
            f"{_format_runtime_seconds(diagnostic_run['runtime_seconds'])} & "
            f"{_format_num(diagnostic_run['omega_last'], digits=6)} & "
            f"{_tex_escape(_stop_reason_label(diagnostic_run['stop_reason']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    (GEN_DIR / "dlambda_stop_diagnostic_table.tex").write_text("\n".join(lines))


def _write_nonfinished_analysis(
    study_meta: dict,
    runs: pd.DataFrame,
    diagnostics: list[dict],
) -> None:
    artifact_root = _artifact_root(study_meta)
    lines: list[str] = []
    partial_branches = _ordered_runs(
        study_meta,
        runs[
            runs.apply(_has_curve, axis=1)
            & runs.apply(_is_monotone_curve, axis=1)
            & ~runs["reached_omega_target"]
        ].copy(),
    )
    for _, row in partial_branches.iterrows():
        run_id = str(row["run_id"])
        label = _run_display_name(run_id)
        progress_jsonl_path = artifact_root / run_id / "data" / "progress.jsonl"
        progress_events = _load_progress_events(progress_jsonl_path)
        failed_attempt = next(
            (
                event
                for event in reversed(progress_events)
                if str(event.get("event", "")) == "attempt_complete"
                and int(event.get("target_step", -1)) == int(row["accepted_steps"]) + 1
                and not bool(event.get("success", False))
            ),
            None,
        )
        if str(row["stop_reason"]) == "newton_maxit_last_step" and failed_attempt is not None:
            lines.append(
                f"The \\texttt{{{_tex_escape(label)}}} run produced a monotone accepted branch through "
                f"step {int(row['accepted_steps'])}, reaching "
                f"\\(\\omega_{{\\mathrm{{last}}}} = \\num{{{_format_num(row['omega_last'], digits=6)}}}\\). "
                f"The next continuation solve targeting "
                f"\\(\\omega = \\num{{{_format_num(failed_attempt.get('omega_target', np.nan), digits=6)}}}\\) "
                f"exhausted the Newton cap of {int(failed_attempt.get('newton_iterations', 0))} iterations with terminal relative residual "
                f"\\num{{{_format_num(failed_attempt.get('newton_relres_end', np.nan), digits=6)}}}. "
                "The continuation code then started a reduced-step retry, but that retry was intentionally terminated and is not promoted into the published branch. "
                "The report therefore freezes the accepted branch at the previous point and classifies the missing final step as a last-step maxit failure."
            )
        else:
            lines.append(
                f"The \\texttt{{{_tex_escape(label)}}} run produced a monotone partial branch but did not reach the shared target. "
                f"It ended at \\(\\omega_{{\\mathrm{{last}}}} = \\num{{{_format_num(row['omega_last'], digits=6)}}}\\) "
                f"with recorded stop reason \\texttt{{{_tex_escape(_stop_reason_label(row['stop_reason']))}}}."
            )
        lines.append("")

    no_branch = _ordered_runs(study_meta, runs[runs["accepted_steps"] <= 0].copy())
    for _, row in no_branch.iterrows():
        run_id = str(row["run_id"])
        label = _run_display_name(run_id)
        progress_path = artifact_root / run_id / "data" / "progress_latest.json"
        if str(row["stop_reason"]) == "capped_linear_zero_correction_stall" and progress_path.exists():
            progress = json.loads(progress_path.read_text())
            lines.append(
                f"The \\texttt{{{_tex_escape(label)}}} run is a genuine initialization failure rather than a horizon-stop artifact. "
                "After the first seed Newton iteration, the Krylov solve already saturated the configured cap of 100 linear iterations. "
                "Subsequent init iterations repeated with \\(\\alpha = 0\\), accepted correction norm equal to zero, and relative residual effectively equal to one "
                f"(latest recorded value \\num{{{_format_num(progress.get('rel_residual', np.nan), digits=6)}}}). "
                "The damping floor \\(r\\) was increased repeatedly without producing a usable Newton correction, "
                "so the watchdog abort reflects real capped-linear stagnation in the init solve, not a continuation stop."
            )
        else:
            lines.append(
                f"The \\texttt{{{_tex_escape(label)}}} run ended without an accepted continuation branch. "
                f"The recorded stop reason is \\texttt{{{_tex_escape(_stop_reason_label(row['stop_reason']))}}}."
            )
        lines.append("")

    for item in diagnostics:
        label = item["label"]
        main_run = item["main_run"]
        diagnostic_run = item["diagnostic_run"]
        lines.append(
            f"The original study-configuration \\texttt{{{_tex_escape(label)}}} run stopped at "
            f"\\(\\omega_{{\\mathrm{{last}}}} = \\num{{{_format_num(main_run['omega_last'], digits=6)}}}\\) because "
            f"\\texttt{{{_tex_escape(_stop_reason_label(main_run['stop_reason']))}}} fired before the shared target \\(\\omega_{{\\mathrm{{final}}}}\\). "
            "That is a protocol stop, not by itself evidence of immediate solver breakdown."
        )
        lines.append("")
        if diagnostic_run["reached_target"]:
            lines.append(
                f"A rerun of \\texttt{{{_tex_escape(label)}}} with the scaled-\\(\\Delta\\lambda\\) continuation stop disabled reached "
                f"\\(\\omega_{{\\mathrm{{last}}}} = \\num{{{_format_num(diagnostic_run['omega_last'], digits=6)}}}\\) "
                f"in \\num{{{_format_runtime_seconds(diagnostic_run['runtime_seconds'])}}} s with stop reason "
                f"\\texttt{{{_tex_escape(_stop_reason_label(diagnostic_run['stop_reason']))}}}. "
                "The published dataset therefore promotes the no-cutoff rerun and keeps the original cutoff run only as a diagnostic comparison."
            )
        else:
            lines.append(
                f"A rerun of \\texttt{{{_tex_escape(label)}}} with the scaled-\\(\\Delta\\lambda\\) continuation stop disabled ended at "
                f"\\(\\omega_{{\\mathrm{{last}}}} = \\num{{{_format_num(diagnostic_run['omega_last'], digits=6)}}}\\) "
                f"after \\num{{{_format_runtime_seconds(diagnostic_run['runtime_seconds'])}}} s with stop reason "
                f"\\texttt{{{_tex_escape(_stop_reason_label(diagnostic_run['stop_reason']))}}}. "
                "The published dataset therefore reflects that relaxed-stop terminal behavior rather than the earlier protocol cutoff."
            )
        lines.append("")

    if not lines:
        lines.append("All published runs produced accepted continuation branches, and no no-cutoff diagnostic reruns were required.")
    (GEN_DIR / "nonfinished_run_analysis.tex").write_text("\n".join(lines).strip() + "\n")

def _write_meta_macros(study_meta: dict, runs: pd.DataFrame) -> None:
    stable_runs = runs[runs.apply(_is_stable_target, axis=1)]
    monotone_runs = runs[runs.apply(_is_monotone_curve, axis=1)]
    macros = [
        f"\\newcommand{{\\StudyOmegaFinal}}{{{study_meta['omega_final']}}}",
        f"\\newcommand{{\\StudyReferenceQuadrature}}{{{int(study_meta['reference_quadrature_rule'])}}}",
        f"\\newcommand{{\\StudyMpiRanks}}{{{int(study_meta['mpi_ranks'])}}}",
        f"\\newcommand{{\\StudyMonotoneTargetRunCount}}{{{int(stable_runs.shape[0])}}}",
        f"\\newcommand{{\\StudyMonotoneRunCount}}{{{int(monotone_runs.shape[0])}}}",
        f"\\newcommand{{\\StudyRunCount}}{{{int(runs.shape[0])}}}",
    ]
    (GEN_DIR / "study_meta.tex").write_text("\n".join(macros) + "\n")


def _plot_continuation_by_element(study_meta: dict, runs: pd.DataFrame, curves: pd.DataFrame) -> None:
    element_order = list(study_meta["element_types"])
    quadrature_matrix = _quadrature_matrix(study_meta)
    quadrature_order = list(study_meta["quadrature_rules"])
    curve_ids = set(runs.loc[runs.apply(_has_curve, axis=1), "run_id"].tolist())
    curves_ok = curves[curves["run_id"].isin(curve_ids)].copy()

    fig, axes = plt.subplots(1, len(element_order), figsize=_figure_size(7.4), sharex=False, sharey=True)
    if len(element_order) == 1:
        axes = [axes]

    for ax, elem_type in zip(axes, element_order, strict=True):
        subset_elem = curves_ok[curves_ok["elem_type"] == elem_type]
        subset_runs = runs[runs["elem_type"] == elem_type].copy()
        panel_rules = [int(value) for value in quadrature_matrix[elem_type]]
        target_reached_rules = set(
            int(value)
            for value in subset_runs.loc[subset_runs.apply(_is_stable_target, axis=1), "quadrature_rule"].tolist()
        )
        partial_rules = sorted(
            int(value)
            for value in subset_runs.loc[
                subset_runs.apply(_is_monotone_curve, axis=1) & ~subset_runs["reached_omega_target"],
                "quadrature_rule",
            ].tolist()
        )
        nonmonotone_rules = sorted(
            int(value)
            for value in subset_runs.loc[
                subset_runs.apply(_has_curve, axis=1) & ~subset_runs.apply(_is_monotone_curve, axis=1),
                "quadrature_rule",
            ].tolist()
        )
        failed_rules = sorted(
            int(value)
            for value in subset_runs.loc[~subset_runs.apply(_has_curve, axis=1), "quadrature_rule"].tolist()
        )
        for rule in panel_rules:
            subset = subset_elem[subset_elem["quadrature_rule"] == rule].sort_values("step")
            if subset.empty:
                continue
            if int(rule) in target_reached_rules:
                linestyle = "-"
            elif int(rule) in partial_rules:
                linestyle = "--"
            else:
                linestyle = ":"
            ax.plot(
                subset["omega"].to_numpy(dtype=float),
                subset["lambda"].to_numpy(dtype=float),
                marker="o",
                markersize=2.8,
                linewidth=1.35,
                linestyle=linestyle,
                color=QUADRATURE_COLORS.get(int(rule), "#374151"),
                label=_fmt_rule(int(rule)),
            )
        ax.axvline(float(study_meta["omega_final"]), color="black", linestyle="--", linewidth=1.0)
        ax.set_title(fr"{elem_type} element")
        ax.set_xlabel(r"$\omega$")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        notes: list[str] = []
        if partial_rules:
            notes.append("partial: " + ", ".join(_fmt_rule(rule) for rule in partial_rules))
        if nonmonotone_rules:
            notes.append("non-monotone: " + ", ".join(_fmt_rule(rule) for rule in nonmonotone_rules))
        if failed_rules:
            notes.append("no branch: " + ", ".join(_fmt_rule(rule) for rule in failed_rules))
        if notes:
            ax.text(
                0.02,
                0.04,
                "\n".join(notes),
                transform=ax.transAxes,
                fontsize=7.75,
                color="#7f1d1d",
                va="bottom",
            )
    axes[0].set_ylabel(r"$\lambda$")
    handles = [Line2D([0], [0], color=QUADRATURE_COLORS.get(rule, "#374151"), marker="o", linewidth=1.35, markersize=3) for rule in quadrature_order]
    labels = [_fmt_rule(rule) for rule in quadrature_order]
    handles.extend(
        [
            Line2D([0], [0], color="#111827", linewidth=1.35, linestyle="-"),
            Line2D([0], [0], color="#111827", linewidth=1.35, linestyle="--"),
            Line2D([0], [0], color="#111827", linewidth=1.35, linestyle=":"),
        ]
    )
    labels.extend(["monotone to target", "partial monotone", "non-monotone"])
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(FIG_DIR / "continuation_by_element.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_bars_by_element(
    study_meta: dict,
    runs: pd.DataFrame,
    *,
    value_column: str,
    ylabel: str,
    output_name: str,
    allow_unstable_values: bool,
) -> None:
    element_order = list(study_meta["element_types"])
    quadrature_matrix = _quadrature_matrix(study_meta)
    fig, axes = plt.subplots(1, len(element_order), figsize=_figure_size(6.6), sharey=True)
    if len(element_order) == 1:
        axes = [axes]

    for ax, elem_type in zip(axes, element_order, strict=True):
        subset = runs[runs["elem_type"] == elem_type].set_index("quadrature_rule")
        panel_rules = [int(value) for value in quadrature_matrix[elem_type]]
        x = np.arange(len(panel_rules), dtype=float)
        heights: list[float] = []
        stable_mask: list[bool] = []
        for rule in panel_rules:
            if int(rule) not in subset.index:
                heights.append(np.nan)
                stable_mask.append(False)
                continue
            row = subset.loc[int(rule)]
            stable = _is_stable_target(row)
            stable_mask.append(stable)
            if not allow_unstable_values and not stable:
                heights.append(np.nan)
            else:
                heights.append(float(row[value_column]))

        bars = ax.bar(
            x,
            heights,
            width=0.72,
            color=[QUADRATURE_COLORS.get(int(rule), "#4b5563") for rule in panel_rules],
            edgecolor="#111827",
            linewidth=0.35,
        )
        for idx, (bar, stable) in enumerate(zip(bars, stable_mask, strict=True)):
            if not stable:
                bar.set_hatch("//")
                bar.set_alpha(0.35)
                bar.set_edgecolor("#991b1b")
                if allow_unstable_values:
                    y_val = heights[idx]
                    if np.isfinite(y_val):
                        ax.plot(bar.get_x() + bar.get_width() / 2.0, y_val, marker="x", color="#991b1b", markersize=5, mew=1.0)
                    else:
                        ax.plot(x[idx], 0.0, marker="x", color="#991b1b", markersize=5, mew=1.0)
                else:
                    ax.plot(x[idx], 0.0, marker="x", color="#991b1b", markersize=5, mew=1.0)

        ax.set_title(fr"{elem_type} element")
        ax.set_xticks(x)
        ax.set_xticklabels([_fmt_rule(int(rule)) for rule in panel_rules])
        ax.set_xlabel("Quadrature rule")
    axes[0].set_ylabel(ylabel)
    if value_column == "runtime_seconds_wall":
        axes[0].set_yscale("log")
    stable_patch = mpatches.Patch(facecolor="#9ca3af", edgecolor="#111827", label="monotone to target")
    unstable_patch = mpatches.Patch(facecolor="#d1d5db", edgecolor="#991b1b", hatch="//", label="not to target / not monotone")
    fig.legend(handles=[stable_patch, unstable_patch], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(FIG_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def _write_results_table(study_meta: dict, runs: pd.DataFrame) -> None:
    reference_rule = int(study_meta["reference_quadrature_rule"])
    ref_lookup = (
        runs[runs["quadrature_rule"] == reference_rule][["elem_type", "lambda_at_omega_target"]]
        .rename(columns={"lambda_at_omega_target": "lambda_at_reference"})
        .set_index("elem_type")
    )
    ordered = _ordered_runs(study_meta, runs)
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}llrrrrrXc@{}}",
        r"\toprule",
        rf"Element & Rule & Time [s] & Steps & $\lambda(\omega_{{\mathrm{{final}}}})$ & $\Delta\lambda_{{\mathrm{{vs}}\ q{reference_rule}}}$ & $\omega_{{\mathrm{{last}}}}$ & Stop reason & Monotone to target \\",
        r"\midrule",
    ]
    for _, row in ordered.iterrows():
        elem_type = str(row["elem_type"])
        reference_lambda = float(ref_lookup.loc[elem_type, "lambda_at_reference"]) if elem_type in ref_lookup.index else np.nan
        current_lambda = float(row["lambda_at_omega_target"])
        delta_lambda = current_lambda - reference_lambda if np.isfinite(current_lambda) and np.isfinite(reference_lambda) else np.nan
        lines.append(
            f"{elem_type} & {_fmt_rule(int(row['quadrature_rule']))} & "
            f"{_format_runtime_seconds(row['runtime_seconds_wall'])} & {int(row['accepted_steps'])} & "
            f"{_format_num(current_lambda, digits=6)} & {_format_num(delta_lambda, digits=4)} & "
            f"{_format_num(row['omega_last'], digits=6)} & {_tex_escape(_stop_reason_label(row['stop_reason']))} & "
            f"{'yes' if _is_stable_target(row) else 'no'} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    (GEN_DIR / "main_results_table.tex").write_text("\n".join(lines))


def _write_summary_findings(study_meta: dict, runs: pd.DataFrame) -> None:
    reference_rule = int(study_meta["reference_quadrature_rule"])
    quadrature_matrix = _quadrature_matrix(study_meta)
    lines = []
    lines.append(
        "Out of \\StudyRunCount{} element/quadrature combinations, \\StudyMonotoneRunCount{} produced a monotone continuation branch and "
        "\\StudyMonotoneTargetRunCount{} reached "
        "$\\omega_{\\mathrm{final}} = \\num{\\StudyOmegaFinal}$ monotonically. "
        f"The report treats the highest quadrature rule, $q{reference_rule}$, as the within-element reference curve."
    )
    lines.append("")

    for elem_type in study_meta["element_types"]:
        panel_rules = [int(value) for value in quadrature_matrix[elem_type]]
        subset = runs[runs["elem_type"] == elem_type].copy()
        subset["__order"] = subset["quadrature_rule"].map({rule: idx for idx, rule in enumerate(panel_rules)})
        subset = subset.sort_values("__order").drop(columns="__order")
        stable_subset = subset[subset.apply(_is_stable_target, axis=1)].copy()
        curve_subset = subset[subset.apply(_has_curve, axis=1)].copy()
        monotone_subset = subset[subset.apply(_is_monotone_curve, axis=1)].copy()
        partial_rules = subset.loc[
            subset.apply(_is_monotone_curve, axis=1) & ~subset["reached_omega_target"],
            "quadrature_rule",
        ].tolist()
        nonmonotone_subset = subset.loc[
            subset.apply(_has_curve, axis=1) & ~subset.apply(_is_monotone_curve, axis=1)
        ].copy()
        no_branch_rules = subset.loc[~subset.apply(_has_curve, axis=1), "quadrature_rule"].tolist()
        if curve_subset.empty:
            lines.append(
                f"For {elem_type}, no quadrature rule produced an accepted continuation branch under the study settings."
            )
            lines.append("")
            continue

        fastest = monotone_subset.sort_values("runtime_seconds_wall").iloc[0]
        reference_rows = stable_subset[stable_subset["quadrature_rule"] == reference_rule]
        ref_lambda = float(reference_rows["lambda_at_omega_target"].iloc[0]) if not reference_rows.empty else np.nan
        summary = (
            f"For {elem_type}, the fastest monotone curve was { _fmt_rule(int(fastest['quadrature_rule'])) } "
            f"at \\num{{{_format_runtime_seconds(fastest['runtime_seconds_wall'])}}} s."
        )
        if np.isfinite(ref_lambda):
            compare = stable_subset.copy()
            compare["lambda_delta_abs"] = np.abs(compare["lambda_at_omega_target"] - ref_lambda)
            compare = compare[compare["quadrature_rule"] != reference_rule]
            if not compare.empty:
                largest = compare.sort_values(["lambda_delta_abs", "quadrature_rule"], ascending=[False, True]).iloc[0]
                summary += (
                    f" Relative to the { _fmt_rule(reference_rule) } reference, the largest absolute shift in "
                    f"$\\lambda(\\omega_{{\\mathrm{{final}}}})$ came from { _fmt_rule(int(largest['quadrature_rule'])) } "
                    f"with $|\\Delta\\lambda| = \\num{{{_format_num(largest['lambda_delta_abs'], digits=4)}}}$."
                )
        else:
            furthest = monotone_subset.sort_values("omega_last", ascending=False).iloc[0]
            summary += (
                f" The furthest monotone branch under the study settings was { _fmt_rule(int(furthest['quadrature_rule'])) } "
                f"at $\\omega_{{\\mathrm{{last}}}} = \\num{{{_format_num(furthest['omega_last'], digits=6)}}}$."
            )
        if partial_rules:
            summary += " Partial monotone branches: " + ", ".join(_fmt_rule(int(rule)) for rule in partial_rules) + "."
        if not nonmonotone_subset.empty:
            labels = []
            for _, nonmono in nonmonotone_subset.iterrows():
                labels.append(
                    f"{_fmt_rule(int(nonmono['quadrature_rule']))} ("
                    f"$\\omega_{{\\mathrm{{last}}}} = \\num{{{_format_num(nonmono['omega_last'], digits=6)}}}$, "
                    f"{_stop_reason_label(nonmono['stop_reason'])})"
                )
            summary += " Non-monotone accepted branches: " + ", ".join(labels) + "."
        if no_branch_rules:
            summary += " No accepted branch: " + ", ".join(_fmt_rule(int(rule)) for rule in no_branch_rules) + "."
        lines.append(summary)
        lines.append("")

    (GEN_DIR / "summary_findings.tex").write_text("\n".join(lines).strip() + "\n")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    study_meta, runs, curves = _load_inputs()
    diagnostics = _collect_no_dlambda_diagnostics(study_meta, runs)
    runs, curves = _promote_corrected_runs(runs, curves, diagnostics)
    runs.to_csv(DATA_DIR / "runs.csv", index=False)
    curves.to_csv(DATA_DIR / "continuation_curves.csv", index=False)
    reference_points = _reference_quadrature_rows()
    _write_reference_quadrature_csvs(reference_points)
    _write_meta_macros(study_meta, runs)
    _write_reference_quadrature_summary_table(reference_points, study_meta)
    _write_reference_quadrature_detail_tables(reference_points)
    _plot_reference_tetra_quadratures(reference_points)
    _plot_continuation_by_element(study_meta, runs, curves)
    _plot_bars_by_element(
        study_meta,
        runs,
        value_column="runtime_seconds_wall",
        ylabel="Wall time [s]",
        output_name="timings_by_element.pdf",
        allow_unstable_values=True,
    )
    _plot_bars_by_element(
        study_meta,
        runs,
        value_column="lambda_at_omega_target",
        ylabel=r"$\lambda(\omega_{\mathrm{final}})$",
        output_name="lambda_at_target_by_element.pdf",
        allow_unstable_values=True,
    )
    _write_results_table(study_meta, runs)
    _write_summary_findings(study_meta, runs)
    _write_diagnostic_data_files(diagnostics)
    _plot_dlambda_stop_diagnostic_overview(study_meta, diagnostics)
    _write_dlambda_stop_diagnostic_table(diagnostics)
    _write_nonfinished_analysis(study_meta, runs, diagnostics)


if __name__ == "__main__":
    main()
