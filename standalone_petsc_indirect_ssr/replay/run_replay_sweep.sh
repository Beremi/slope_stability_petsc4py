#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="${1:?usage: run_replay_sweep.sh SAMPLE_DIR}"
OUT_DIR="${OUT_DIR:-/tmp/ssr_replay_sweep_$(date +%Y%m%d_%H%M%S)}"
RANKS="${RANKS:-32}"
LINEAR_RTOL="${LINEAR_RTOL:-1e-1}"
KSP_MAX_IT="${KSP_MAX_IT:-200}"

mkdir -p "${OUT_DIR}"

run_one() {
  local name="$1"
  local profile="$2"
  local extra="${3:-}"
  echo "REPLAY_SWEEP_START name=${name} profile=${profile} ranks=${RANKS} extra=${extra}"
  RANKS="${RANKS}" \
  LINEAR_RTOL="${LINEAR_RTOL}" \
  KSP_MAX_IT="${KSP_MAX_IT}" \
  LOG="${OUT_DIR}/${name}.log" \
  SUMMARY_CSV="${OUT_DIR}/${name}.csv" \
  EXTRA_OPTS="${extra}" \
    "${SCRIPT_DIR}/run_c_replay.sh" "${SAMPLE_DIR}" "${profile}"
  echo "REPLAY_SWEEP_DONE name=${name}"
}

run_one petsc4py petsc4py
run_one baseline baseline

# One-setting-at-a-time baseline perturbations.
run_one baseline_smoother3 baseline "-pmg_smoother_max_it 3"
run_one baseline_coarse_hypre_preonly baseline "-pmg_coarse_telescope_ksp_type preonly -pmg_coarse_telescope_pc_type hypre -pmg_shell_p1_pc_hypre_boomeramg_max_iter 4 -pmg_shell_p1_pc_hypre_boomeramg_tol 0.0 -pmg_shell_p1_pc_hypre_boomeramg_coarsen_type HMIS -pmg_shell_p1_pc_hypre_boomeramg_interp_type ext+i"
run_one baseline_coarse_hypre_iter1 baseline "-pmg_coarse_telescope_ksp_type preonly -pmg_coarse_telescope_pc_type hypre -pmg_shell_p1_pc_hypre_boomeramg_max_iter 1 -pmg_shell_p1_pc_hypre_boomeramg_tol 0.0"
run_one baseline_coarse_hypre_iter2 baseline "-pmg_coarse_telescope_ksp_type preonly -pmg_coarse_telescope_pc_type hypre -pmg_shell_p1_pc_hypre_boomeramg_max_iter 2 -pmg_shell_p1_pc_hypre_boomeramg_tol 0.0"
run_one baseline_p2_64_p1_16 baseline "-pmg_shell_p2_active_ranks 64 -pmg_shell_p1_active_ranks 16"
run_one baseline_p2_32_p1_16 baseline "-pmg_shell_p2_active_ranks 32 -pmg_shell_p1_active_ranks 16"
run_one baseline_p2_32_p1_8 baseline "-pmg_shell_p2_active_ranks 32 -pmg_shell_p1_active_ranks 8"

python - <<'PY' "${OUT_DIR}"
from __future__ import annotations
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*.csv")):
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            row["case"] = path.stem
            rows.append(row)

fields = [
    "case", "profile", "solve", "expected_iterations", "c_iterations", "iteration_match",
    "history_rel_max", "first_mismatch_layer", "pc_v0_rel", "Az0_rel", "arnoldi0_rel",
    "mg_fine_pre_rel", "mg_p2_pre_rel", "mg_p1_x_rel",
]
with (root / "summary.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(",".join(fields))
for row in rows:
    print(",".join(str(row.get(field, "")) for field in fields))
PY

echo "OUT_DIR=${OUT_DIR}"
