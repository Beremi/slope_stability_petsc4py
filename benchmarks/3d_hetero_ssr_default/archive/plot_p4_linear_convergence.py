from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_history(path: Path) -> tuple[list[float], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    histories = (
        data.get("relative_reported_residual_histories")
        or data.get("relative_residual_histories")
        or ([data["relative_reported_residual_history"]] if data.get("relative_reported_residual_history") else [])
    )
    if not histories:
        raise ValueError(f"No relative residual history found in {path}")
    label = str(data.get("pc_backend", path.parent.parent.name))
    return [float(v) for v in histories[0]], label


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    text = str(spec).strip()
    if "=" not in text:
        raise ValueError(f"Expected run spec in LABEL=PATH form, got {spec!r}")
    label, path = text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Expected non-empty label in {spec!r}")
    return label, Path(path.strip())


def plot_histories(
    runs: list[tuple[str, Path]],
    *,
    out: Path,
    title: str,
    target: float | None = None,
) -> None:
    if not runs:
        raise ValueError("Expected at least one run to plot")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, path in runs:
        hist, inferred_label = _load_history(path)
        effective_label = str(label or inferred_label)
        ax.semilogy(
            range(1, len(hist) + 1),
            hist,
            marker="o",
            ms=3,
            lw=1.8,
            label=f"{effective_label} ({len(hist)} it)",
        )
    if target is not None:
        ax.axhline(float(target), color="black", ls="--", lw=1.0, alpha=0.6, label=f"target={float(target):.1e}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Relative residual norm")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot P4 linear convergence histories for one or more probe runs.")
    parser.add_argument("--run", action="append", default=None, help="Run in LABEL=PATH form; may be repeated.")
    parser.add_argument("--hypre-run-info", type=Path, required=False)
    parser.add_argument("--bddc-run-info", type=Path, required=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", type=str, default="P4 Linear Elastic Convergence")
    parser.add_argument("--target", type=float, default=None)
    args = parser.parse_args()

    runs: list[tuple[str, Path]] = []
    if args.run:
        runs.extend(_parse_run_spec(spec) for spec in args.run)
    elif args.hypre_run_info is not None and args.bddc_run_info is not None:
        runs = [("hypre", args.hypre_run_info), ("bddc", args.bddc_run_info)]
    else:
        raise ValueError("Provide either repeated --run LABEL=PATH arguments or both --hypre-run-info and --bddc-run-info")
    plot_histories(runs, out=args.out, title=args.title, target=args.target)


if __name__ == "__main__":
    main()
