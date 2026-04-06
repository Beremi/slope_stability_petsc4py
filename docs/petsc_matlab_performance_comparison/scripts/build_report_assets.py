from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from study_common import load_study


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value) -> float:
    if value in {None, ""}:
        return float("nan")
    return float(value)


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def humanize_stop_reason(value: str) -> str:
    mapping = {
        "omega_max_stop": "omega max",
        "d_lambda_diff_scaled_min": r"$\Delta\lambda$ min",
        "": "n/a",
    }
    return mapping.get(str(value).strip(), escape_latex(str(value)))


def placeholder_figure(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axis("off")
    ax.text(0.5, 0.55, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.40, "No completed runs available.", ha="center", va="center", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def sorted_levels(rows: list[dict]) -> list[tuple[str, str, int]]:
    by_key = {}
    for row in rows:
        key = row["level_id"]
        by_key[key] = (row["level_id"], row["level_label"], int(row["level_order"]))
    return sorted(by_key.values(), key=lambda item: item[2])


def continuation_series(curves: list[dict], *, case_id: str, level_id: str, engine: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row for row in curves
        if row["case_id"] == case_id and row["level_id"] == level_id and row["engine"] == engine and row["variant"] == variant
    ]
    rows.sort(key=lambda row: int(row["step"]))
    omega = np.asarray([to_float(row["omega"]) for row in rows], dtype=np.float64)
    lam = np.asarray([to_float(row["lambda"]) for row in rows], dtype=np.float64)
    return omega, lam


def create_case_continuation_figure(curves: list[dict], *, case_id: str, case_label: str, series_defs: list[tuple[str, str, str, str]], out_path: Path) -> None:
    case_rows = [row for row in curves if row["case_id"] == case_id]
    levels = sorted_levels(case_rows)
    if not levels:
        placeholder_figure(out_path, f"{case_label}: continuation")
        return

    ncols = 2
    nrows = math.ceil(len(levels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 2.8 * nrows), squeeze=False)
    flat_axes = axes.ravel()

    for ax, (level_id, level_label, _order) in zip(flat_axes, levels, strict=False):
        for label, engine, variant, color in series_defs:
            omega, lam = continuation_series(curves, case_id=case_id, level_id=level_id, engine=engine, variant=variant)
            if omega.size:
                ax.plot(omega, lam, marker="o", markersize=3.5, linewidth=1.2, label=label, color=color)
        ax.set_title(level_label)
        ax.set_xlabel(r"$\omega$")
        ax.set_ylabel(r"$\lambda$")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8, loc="best")

    for ax in flat_axes[len(levels):]:
        ax.axis("off")

    fig.suptitle(case_label, fontsize=13, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def find_run_row(rows: list[dict], *, case_id: str, level_id: str, engine: str, variant: str) -> dict | None:
    for row in rows:
        if (
            row["case_id"] == case_id
            and row["phase"] == "main"
            and row["level_id"] == level_id
            and row["engine"] == engine
            and row["variant"] == variant
        ):
            return row
    return None


def create_timing_figure(rows: list[dict], *, case_id: str, case_label: str, series_defs: list[tuple[str, str, str, str]], out_path: Path) -> None:
    case_rows = [row for row in rows if row["case_id"] == case_id and row["phase"] == "main"]
    levels = sorted_levels(case_rows)
    if not levels:
        placeholder_figure(out_path, f"{case_label}: timings")
        return

    x = np.arange(len(levels), dtype=np.float64)
    width = 0.75 / max(len(series_defs), 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    for idx, (label, engine, variant, color) in enumerate(series_defs):
        values = []
        for level_id, _level_label, _order in levels:
            row = find_run_row(rows, case_id=case_id, level_id=level_id, engine=engine, variant=variant)
            values.append(to_float(row["runtime_seconds"]) if row else float("nan"))
        offset = (idx - (len(series_defs) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width * 0.95, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _level_id, label, _order in levels])
    ax.set_ylabel("Runtime [s]")
    ax.set_title(case_label)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend()

    if len(series_defs) >= 2:
        for idx, (level_id, _level_label, _order) in enumerate(levels):
            matlab_row = find_run_row(rows, case_id=case_id, level_id=level_id, engine="matlab", variant="main")
            petsc_row = find_run_row(rows, case_id=case_id, level_id=level_id, engine="petsc", variant="main")
            if matlab_row and petsc_row:
                matlab_runtime = to_float(matlab_row["runtime_seconds"])
                petsc_runtime = to_float(petsc_row["runtime_seconds"])
                if petsc_runtime > 0:
                    ratio = matlab_runtime / petsc_runtime
                    ymax = max(matlab_runtime, petsc_runtime)
                    ax.text(idx, ymax * 1.03, f"M/P={ratio:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_case_table(rows: list[dict], *, case_id: str, case_label: str, out_path: Path) -> None:
    case_rows = [row for row in rows if row["case_id"] == case_id and row["phase"] == "main"]
    levels = sorted_levels(case_rows)
    if not levels:
        out_path.write_text("\\textit{No completed runs available.}\n", encoding="utf-8")
        return

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{escape_latex(case_label)} summary.}}",
        "\\begin{tabularx}{\\textwidth}{lrrrrrrX}",
        "\\toprule",
        "Level & Unknowns & PETSc [s] & MATLAB [s] & M/P & PETSc steps & MATLAB steps & Stop (P/M)\\\\",
        "\\midrule",
    ]
    for level_id, level_label, _order in levels:
        petsc = find_run_row(rows, case_id=case_id, level_id=level_id, engine="petsc", variant="main")
        matlab = find_run_row(rows, case_id=case_id, level_id=level_id, engine="matlab", variant="main")
        if petsc is None or matlab is None:
            continue
        petsc_runtime = to_float(petsc["runtime_seconds"])
        matlab_runtime = to_float(matlab["runtime_seconds"])
        ratio = matlab_runtime / petsc_runtime if petsc_runtime > 0 else float("nan")
        stop_reason = f"{humanize_stop_reason(petsc['stop_reason'])} / {humanize_stop_reason(matlab['stop_reason'])}"
        lines.append(
            f"{escape_latex(level_label)} & "
            f"{escape_latex(str(petsc['unknowns']))} & "
            f"{petsc_runtime:.3f} & {matlab_runtime:.3f} & {ratio:.2f} & "
            f"{escape_latex(str(petsc['accepted_steps']))} & {escape_latex(str(matlab['accepted_steps']))} & "
            f"{stop_reason}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_lambda_table(rows: list[dict], *, case_id: str, case_label: str, out_path: Path) -> None:
    case_rows = [row for row in rows if row["case_id"] == case_id and row["phase"] == "main"]
    levels = sorted_levels(case_rows)
    if not levels:
        out_path.write_text("\\textit{No completed runs available.}\n", encoding="utf-8")
        return

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{escape_latex(case_label)} appendix: PETSc residual-stop vs PETSc delta-lambda-stop.}}",
        "\\begin{tabularx}{\\textwidth}{lrrrrrr}",
        "\\toprule",
        "Level & PETSc res. [s] & PETSc $\\Delta\\lambda$ [s] & MATLAB [s] & Res. steps & $\\Delta\\lambda$ steps & MATLAB steps\\\\",
        "\\midrule",
    ]
    for level_id, level_label, _order in levels:
        petsc_main = find_run_row(rows, case_id=case_id, level_id=level_id, engine="petsc", variant="main")
        petsc_delta = find_run_row(rows, case_id=case_id, level_id=level_id, engine="petsc", variant="delta_lambda")
        matlab = find_run_row(rows, case_id=case_id, level_id=level_id, engine="matlab", variant="main")
        if petsc_main is None or petsc_delta is None or matlab is None:
            continue
        lines.append(
            f"{escape_latex(level_label)} & "
            f"{to_float(petsc_main['runtime_seconds']):.3f} & "
            f"{to_float(petsc_delta['runtime_seconds']):.3f} & "
            f"{to_float(matlab['runtime_seconds']):.3f} & "
            f"{escape_latex(str(petsc_main['accepted_steps']))} & "
            f"{escape_latex(str(petsc_delta['accepted_steps']))} & "
            f"{escape_latex(str(matlab['accepted_steps']))}\\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build figures and LaTeX table snippets from normalized CSV study data.")
    parser.add_argument("--study", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    study = load_study(args.study, manifest_path=args.manifest, allow_missing_manifest=True)
    runs = load_csv_rows(study["data_dir"] / "runs.csv")
    curves = load_csv_rows(study["data_dir"] / "continuation_curves.csv")

    series_main = [
        ("MATLAB", "matlab", "main", "#c55a11"),
        ("PETSc", "petsc", "main", "#1f77b4"),
    ]
    series_delta = [
        ("MATLAB", "matlab", "main", "#c55a11"),
        ("PETSc residual", "petsc", "main", "#1f77b4"),
        ("PETSc $\\Delta\\lambda$", "petsc", "delta_lambda", "#2a9d8f"),
    ]

    for case in study["cases"]:
        create_case_continuation_figure(
            curves,
            case_id=case["id"],
            case_label=case["label"],
            series_defs=series_main,
            out_path=study["figures_dir"] / f"{case['report_slug']}_continuation.pdf",
        )
        create_timing_figure(
            runs,
            case_id=case["id"],
            case_label=case["label"],
            series_defs=series_main,
            out_path=study["figures_dir"] / f"{case['report_slug']}_timings.pdf",
        )
        write_case_table(
            runs,
            case_id=case["id"],
            case_label=case["label"],
            out_path=study["generated_dir"] / f"{case['report_slug']}_table.tex",
        )

    create_case_continuation_figure(
        curves,
        case_id="hetero_3d",
        case_label="Heterogeneous 3D SSR Appendix",
        series_defs=series_delta,
        out_path=study["figures_dir"] / "hetero_3d_delta_lambda_continuation.pdf",
    )
    create_timing_figure(
        runs,
        case_id="hetero_3d",
        case_label="Heterogeneous 3D SSR Appendix",
        series_defs=series_delta,
        out_path=study["figures_dir"] / "hetero_3d_delta_lambda_timings.pdf",
    )
    write_delta_lambda_table(
        runs,
        case_id="hetero_3d",
        case_label="Heterogeneous 3D SSR",
        out_path=study["generated_dir"] / "hetero_3d_delta_lambda_table.tex",
    )


if __name__ == "__main__":
    main()
