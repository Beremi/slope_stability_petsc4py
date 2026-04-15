from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = Path(__file__).resolve().parent / "assets"
P2_CURVES_CSV = ROOT / "docs" / "petsc_matlab_performance_comparison" / "data" / "continuation_curves.csv"
P4_ROOT = ROOT / "artifacts" / "comparisons" / "p4_l1_vs_p2_presentation_20260410" / "runs"

LEVEL_COLORS = {
    "L1": "#1f77b4",
    "L2": "#ff7f0e",
    "L3": "#2ca02c",
    "L4": "#d62728",
    "concave_L2": "#1f77b4",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_p2_series(rows: list[dict[str, str]], *, case_id: str, variant: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    matched = [
        row
        for row in rows
        if row["case_id"] == case_id and row["engine"] == "petsc" and row["variant"] == variant
    ]
    by_level: dict[str, list[dict[str, str]]] = {}
    for row in matched:
        by_level.setdefault(row["level_id"], []).append(row)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for level_id, level_rows in by_level.items():
        level_rows.sort(key=lambda row: int(row["step"]))
        omega = np.asarray([float(row["omega"]) for row in level_rows], dtype=np.float64)
        lam = np.asarray([float(row["lambda"]) for row in level_rows], dtype=np.float64)
        series[level_id] = (omega, lam)
    return series


def load_p4_series(*, case_id: str, level_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    npz_path = P4_ROOT / case_id / level_id / "main_petsc_p4_delta_lambda" / "data" / "petsc_run.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path, allow_pickle=True) as npz:
        omega = np.asarray(npz["omega_hist"], dtype=np.float64)
        lam = np.asarray(npz["lambda_hist"], dtype=np.float64)
    return omega, lam


def style_axis(ax, *, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\lambda$")
    ax.grid(True, alpha=0.30)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))


def finalize_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def placeholder(out_path: Path, *, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    finalize_figure(fig, out_path)


def build_homo(rows: list[dict[str, str]]) -> None:
    out_path = ASSET_DIR / "homo_3d_p2_vs_p4_curves.pdf"
    p2_main = load_p2_series(rows, case_id="homo_3d", variant="main")
    p4 = load_p4_series(case_id="homo_3d", level_id="L1")
    if not p2_main or p4 is None:
        placeholder(out_path, title="Homogeneous 3D", message="Required P2 or P4 continuation data are missing.")
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    for level_id in ["L1", "L2", "L3", "L4"]:
        if level_id not in p2_main:
            continue
        omega, lam = p2_main[level_id]
        ax.plot(
            omega,
            lam,
            color=LEVEL_COLORS[level_id],
            linewidth=2.0,
            marker="o",
            markersize=3.4,
            label=f"P2 {level_id}",
        )
    ax.plot(
        p4[0],
        p4[1],
        color="#111111",
        linewidth=2.8,
        marker="s",
        markersize=4.0,
        label=r"P4 L1, $\Delta\lambda$, April 10 rerun",
    )
    style_axis(ax, title="Committed P2 main ladder and new P4(L1) rerun")
    ax.legend(ncol=3, fontsize=8.7, loc="upper left")
    finalize_figure(fig, out_path)


def build_hetero(rows: list[dict[str, str]]) -> None:
    out_path = ASSET_DIR / "hetero_3d_p2_vs_p4_curves.pdf"
    p2_main = load_p2_series(rows, case_id="hetero_3d", variant="main")
    p2_delta = load_p2_series(rows, case_id="hetero_3d", variant="delta_lambda")
    p4 = load_p4_series(case_id="hetero_3d", level_id="L1")
    if not p2_main or not p2_delta or p4 is None:
        placeholder(out_path, title="Heterogeneous 3D", message="Required P2 or P4 continuation data are missing.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), sharey=True)
    for ax, series, title in [
        (axes[0], p2_main, "Committed P2 main ladder"),
        (axes[1], p2_delta, r"Available P2 $\Delta\lambda$ appendix ladder"),
    ]:
        for level_id in ["L1", "L2", "L3", "L4"]:
            if level_id not in series:
                continue
            omega, lam = series[level_id]
            ax.plot(
                omega,
                lam,
                color=LEVEL_COLORS[level_id],
                linewidth=2.0,
                marker="o",
                markersize=3.1,
                label=f"P2 {level_id}",
            )
        ax.plot(
            p4[0],
            p4[1],
            color="#111111",
            linewidth=2.8,
            marker="s",
            markersize=3.8,
            label=r"P4 L1, $\Delta\lambda$, April 10 rerun",
        )
        style_axis(ax, title=title)
        ax.legend(fontsize=8.2, loc="upper left")
    axes[1].set_ylabel("")
    finalize_figure(fig, out_path)


def build_seepage(rows: list[dict[str, str]]) -> None:
    out_path = ASSET_DIR / "seepage_3d_p2_vs_p4_curves.pdf"
    p2_main = load_p2_series(rows, case_id="seepage_3d", variant="main")
    p4 = load_p4_series(case_id="seepage_3d", level_id="concave_L2")
    if not p2_main or p4 is None:
        placeholder(out_path, title="Seepage 3D", message="Required P2 or P4 continuation data are missing.")
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    for level_id, label in [("concave_L2", "P2 concave_L2")]:
        if level_id not in p2_main:
            continue
        omega, lam = p2_main[level_id]
        ax.plot(
            omega,
            lam,
            color=LEVEL_COLORS[level_id],
            linewidth=2.0,
            marker="o",
            markersize=3.4,
            label=label,
        )
    ax.plot(
        p4[0],
        p4[1],
        color="#111111",
        linewidth=2.8,
        marker="s",
        markersize=4.0,
        label=r"P4 concave\_L2, $\Delta\lambda$, April 11 rerun",
    )
    style_axis(ax, title="Committed P2 seepage run and new P4(concave_L2) rerun")
    ax.legend(fontsize=8.7, loc="upper left")
    finalize_figure(fig, out_path)


def main() -> None:
    rows = load_csv_rows(P2_CURVES_CSV)
    build_homo(rows)
    build_hetero(rows)
    build_seepage(rows)


if __name__ == "__main__":
    main()
