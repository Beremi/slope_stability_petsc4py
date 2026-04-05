from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/p4_l1_step13_three_param_compare/rank8_secant_step13/data"
NEW = ROOT / "artifacts/p4_l1_step13_three_param_compare/rank8_switch11_three_param_step13/data"
OUT = ROOT / "artifacts/p4_l1_step13_three_param_compare/report"


def load_case(data_dir: Path) -> dict[str, Any]:
    npz = np.load(data_dir / "petsc_run.npz", allow_pickle=True)
    run_info = json.loads((data_dir / "run_info.json").read_text())
    return {"npz": npz, "run_info": run_info}


def step_index_for_accepted_step(accepted_step: int) -> int:
    return accepted_step - 3


def scalar(arr: np.ndarray, accepted_step: int) -> float:
    return float(np.asarray(arr)[step_index_for_accepted_step(accepted_step)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plots = OUT / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    base = load_case(BASE)
    new = load_case(NEW)
    s = 13

    def extract(case: dict[str, Any]) -> dict[str, float | str]:
        npz = case["npz"]
        return {
            "predictor_kind": str(np.asarray(npz["stats_step_predictor_kind"])[step_index_for_accepted_step(s)]),
            "predictor_wall": scalar(npz["stats_step_predictor_wall_time"], s),
            "predictor_alpha": scalar(npz["stats_step_predictor_alpha"], s),
            "predictor_beta": scalar(npz["stats_step_predictor_beta"], s) if "stats_step_predictor_beta" in npz.files else np.nan,
            "predictor_gamma": scalar(npz["stats_step_predictor_gamma"], s) if "stats_step_predictor_gamma" in npz.files else np.nan,
            "predictor_evals": scalar(npz["stats_step_predictor_energy_eval_count"], s),
            "predictor_merit": scalar(npz["stats_step_predictor_energy_value"], s),
            "step_wall": scalar(npz["stats_step_wall_time"], s),
            "step_newton": scalar(npz["stats_step_newton_iterations"], s),
            "step_linear": scalar(npz["stats_step_linear_iterations"], s),
            "step_solve": scalar(npz["stats_step_linear_solve_time"], s),
            "step_pc": scalar(npz["stats_step_linear_preconditioner_time"], s),
            "step_orth": scalar(npz["stats_step_linear_orthogonalization_time"], s),
            "step_lambda": scalar(npz["lambda_hist"], s),
            "step_omega": scalar(npz["omega_hist"], s),
            "guess_u_diff": scalar(npz["stats_step_initial_guess_displacement_diff_volume_integral"], s),
            "guess_dev_diff": scalar(npz["stats_step_initial_guess_deviatoric_strain_diff_volume_integral"], s),
            "guess_lambda_err": scalar(npz["stats_step_lambda_initial_guess_abs_error"], s),
        }

    b = extract(base)
    n = extract(new)

    plt.figure(figsize=(7, 5))
    labels = ["predictor", "linear solve", "pc", "orth", "other step"]
    b_other = b["step_wall"] - b["step_solve"] - b["step_pc"] - b["step_orth"]
    n_other = n["step_wall"] - n["step_solve"] - n["step_pc"] - n["step_orth"]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, [b["predictor_wall"], b["step_solve"], b["step_pc"], b["step_orth"], b_other], width, label="secant")
    plt.bar(x + width / 2, [n["predictor_wall"], n["step_solve"], n["step_pc"], n["step_orth"], n_other], width, label="3-param")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Time [s]")
    plt.title("Accepted Step 13 Timing Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "step13_timing_split.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels2 = ["Newton", "Linear", "u_init diff", "dev diff"]
    x2 = np.arange(len(labels2))
    plt.bar(x2 - width / 2, [b["step_newton"], b["step_linear"], b["guess_u_diff"], b["guess_dev_diff"]], width, label="secant")
    plt.bar(x2 + width / 2, [n["step_newton"], n["step_linear"], n["guess_u_diff"], n["guess_dev_diff"]], width, label="3-param")
    plt.xticks(x2, labels2, rotation=20)
    plt.title("Accepted Step 13 Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "step13_outcome.png", dpi=180)
    plt.close()

    summary = {
        "step": s,
        "baseline": b,
        "three_param": n,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    report = f"""# P4(L1) Accepted Step 13: Secant vs 3-Parameter Residual Search

Compared runs:

- baseline secant: [run_info.json](../rank8_secant_step13/data/run_info.json)
- switched 3-parameter predictor on accepted step 13 only: [run_info.json](../rank8_switch11_three_param_step13/data/run_info.json)

The switched run used plain secant through accepted step `12`, then switched only step `13` to the new predictor

\\[
u_{{ini}} = \\alpha u_i + \\beta u_{{i-1}} + \\gamma u_{{i-2}}
\\]

with a derivative-free residual-plus-penalty search around the secant-centered coefficients.

## Step 13 Summary

| Metric | Secant | 3-Parameter |
| --- | ---: | ---: |
| Predictor kind | `{b["predictor_kind"]}` | `{n["predictor_kind"]}` |
| Predictor wall time [s] | `{b["predictor_wall"]:.3f}` | `{n["predictor_wall"]:.3f}` |
| Chosen alpha | `{b["predictor_alpha"]:.6f}` | `{n["predictor_alpha"]:.6f}` |
| Chosen beta | `-` | `{n["predictor_beta"]:.6f}` |
| Chosen gamma | `-` | `{n["predictor_gamma"]:.6f}` |
| Merit evaluations | `{b["predictor_evals"]:.0f}` | `{n["predictor_evals"]:.0f}` |
| Merit value | `{b["predictor_merit"]:.6e}` | `{n["predictor_merit"]:.6e}` |
| Step wall time [s] | `{b["step_wall"]:.3f}` | `{n["step_wall"]:.3f}` |
| Newton iterations | `{b["step_newton"]:.0f}` | `{n["step_newton"]:.0f}` |
| Linear iterations | `{b["step_linear"]:.0f}` | `{n["step_linear"]:.0f}` |
| Linear solve [s] | `{b["step_solve"]:.3f}` | `{n["step_solve"]:.3f}` |
| PC apply [s] | `{b["step_pc"]:.3f}` | `{n["step_pc"]:.3f}` |
| Orthogonalization [s] | `{b["step_orth"]:.3f}` | `{n["step_orth"]:.3f}` |
| `u_ini -> u_newton` displacement integral | `{b["guess_u_diff"]:.3f}` | `{n["guess_u_diff"]:.3f}` |
| `u_ini -> u_newton` deviatoric integral | `{b["guess_dev_diff"]:.3f}` | `{n["guess_dev_diff"]:.3f}` |
| `|lambda_ini - lambda_newton|` | `{b["guess_lambda_err"]:.6e}` | `{n["guess_lambda_err"]:.6e}` |

## Predictor Cost vs Newton Cost

| Metric | Secant | 3-Parameter |
| --- | ---: | ---: |
| Predictor / step wall | `{(b["predictor_wall"] / b["step_wall"] if b["step_wall"] else 0.0):.4f}` | `{(n["predictor_wall"] / n["step_wall"] if n["step_wall"] else 0.0):.4f}` |
| Predictor / linear solve | `{(b["predictor_wall"] / b["step_solve"] if b["step_solve"] else 0.0):.4f}` | `{(n["predictor_wall"] / n["step_solve"] if n["step_solve"] else 0.0):.4f}` |

## Plots

![Timing split](plots/step13_timing_split.png)

![Outcome](plots/step13_outcome.png)
"""
    (OUT / "README.md").write_text(report)


if __name__ == "__main__":
    main()
