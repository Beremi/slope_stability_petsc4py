#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "cases" / "experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1"


@dataclass(frozen=True)
class Variant:
    name: str
    config: str
    notes: str
    redundant_coarse_groups: str | None = None


VARIANTS: dict[str, Variant] = {
    "pmg_shell_hypre": Variant(
        "pmg_shell_hypre",
        "petsc_aij_pmg_shell_32.toml",
        "current PETSc-AIJ PMG-shell baseline with hypre coarse solve",
    ),
    "pmg_shell_redundant_nodes": Variant(
        "pmg_shell_redundant_nodes",
        "petsc_aij_pmg_shell_redundant_coarse_32.toml",
        "PMG-shell with PCREDUNDANT+MUMPS coarse solve, one redundant group per node",
        redundant_coarse_groups="nodes",
    ),
    "pmg_shell_redundant_one": Variant(
        "pmg_shell_redundant_one",
        "petsc_aij_pmg_shell_redundant_coarse_32.toml",
        "PMG-shell with one global PCREDUNDANT+MUMPS coarse solve",
        redundant_coarse_groups="1",
    ),
    "hypre_current": Variant(
        "hypre_current",
        "petsc_aij_hypre_32.toml",
        "direct AIJ hypre control",
    ),
    "hypre_lagged_pmis": Variant(
        "hypre_lagged_pmis",
        "petsc_aij_hypre_lagged_pmis_32.toml",
        "direct AIJ hypre with PMIS/ext+i-mm and accepted-step preconditioner rebuild",
    ),
    "gamg_current": Variant(
        "gamg_current",
        "petsc_aij_gamg_32.toml",
        "direct AIJ GAMG control",
    ),
    "gamg_lagged_lowcomm": Variant(
        "gamg_lagged_lowcomm",
        "petsc_aij_gamg_lagged_lowcomm_32.toml",
        "direct AIJ GAMG with process_eq_limit/repartition/reuse options",
    ),
    "bddc_ilu": Variant(
        "bddc_ilu",
        "petsc_aij_bddc_32.toml",
        "MATIS/BDDC with rank-local ILU subsolves",
    ),
    "bddc_gamg": Variant(
        "bddc_gamg",
        "petsc_aij_bddc_gamg_32.toml",
        "MATIS/BDDC with approximate local GAMG subsolves",
    ),
}


def _parse_layout(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError(f"layout must look like NODESxRANKS_PER_NODE, got {value!r}")
    nodes = int(match.group(1))
    ranks_per_node = int(match.group(2))
    if nodes <= 0 or ranks_per_node <= 0:
        raise argparse.ArgumentTypeError("layout nodes and ranks per node must be positive")
    return nodes, ranks_per_node


def _replace_line(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*).*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}", text, count=1)
    return text


def _make_case_config(
    *,
    source: Path,
    destination: Path,
    case_name: str,
    ranks: int,
    redundant_groups: int | None,
    step_max: int | None,
) -> None:
    text = source.read_text(encoding="utf-8")
    text = _replace_line(text, "name", f'"{case_name}"')
    text = _replace_line(text, "mpi_ranks", str(int(ranks)))
    if step_max is not None:
        text = _replace_line(text, "step_max", str(int(step_max)))
    if redundant_groups is not None:
        text = re.sub(
            r'"mg_coarse_pc_redundant_number=\d+"',
            f'"mg_coarse_pc_redundant_number={int(redundant_groups)}"',
            text,
        )
    destination.write_text(text, encoding="utf-8")


def _script_text(
    *,
    job_name: str,
    case_config: Path,
    run_out: Path,
    nodes: int,
    ranks_per_node: int,
    walltime: str,
    queue: str,
) -> str:
    ranks = nodes * ranks_per_node
    repo = ROOT
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={queue}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ranks_per_node}
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --output={run_out}/slurm-%j.out
#SBATCH --error={run_out}/slurm-%j.err

set -euo pipefail

REPO="{repo}"
CASE_CONFIG="{case_config}"
RUN_OUT="{run_out}"
RANKS={ranks}
RANKS_PER_NODE={ranks_per_node}

mkdir -p "$RUN_OUT"
cd "$REPO"

ml -f purge
ml GCC/14.3.0 OpenMPI/5.0.8-GCC-14.3.0 Python/3.13.5-GCCcore-14.3.0
source build_scripts/activate_local_petsc_env.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

{{
  echo "job_name={job_name}"
  echo "nodes={nodes}"
  echo "ranks_per_node=$RANKS_PER_NODE"
  echo "ranks=$RANKS"
  echo "case_config=$CASE_CONFIG"
  echo "run_out=$RUN_OUT"
  echo "slurm_job_id=${{SLURM_JOB_ID:-}}"
  echo "slurm_nodelist=${{SLURM_NODELIST:-}}"
  date --iso-8601=seconds
}} > "$RUN_OUT/job_metadata.txt"

srun --nodes={nodes} --ntasks-per-node={ranks_per_node} --cpus-per-task=1 \\
  --distribution=block:block --cpu-bind=cores bash -lc 'printf "%05d %s %s\\n" "$SLURM_PROCID" "$(hostname)" "$(taskset -pc $$)"' \\
  | sort -n > "$RUN_OUT/rank_placement.txt"

COMMAND=(
  mpiexec
  --mca pml ob1
  --mca btl self,sm,tcp
  --bind-to core
  --map-by ppr:${{RANKS_PER_NODE}}:node
  -n "$RANKS"
  ./.venv/bin/python
  -u
  -m slope_stability.cli.run_case_from_config
  "$CASE_CONFIG"
  --out_dir "$RUN_OUT"
)

printf "%q " "${{COMMAND[@]}}" > "$RUN_OUT/command.txt"
printf "\\n" >> "$RUN_OUT/command.txt"

/usr/bin/time -v "${{COMMAND[@]}}" > "$RUN_OUT/stdout.txt" 2> "$RUN_OUT/stderr.txt"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout",
        action="append",
        type=_parse_layout,
        default=[],
        help="Karolina layout as NODESxRANKS_PER_NODE, for example 1x128 or 2x64. May be repeated.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        default=[],
        help="Variant to generate. May be repeated. Defaults to the high-signal screen.",
    )
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--walltime", default="00:10:00")
    parser.add_argument("--queue", default="qexp")
    parser.add_argument("--step-max", type=int, default=None, help="Override continuation.step_max in generated configs.")
    parser.add_argument("--submit", action="store_true", help="Submit generated jobs with sbatch.")
    args = parser.parse_args()
    if args.step_max is not None and int(args.step_max) <= 0:
        raise SystemExit("--step-max must be positive")

    layouts = args.layout or [_parse_layout("1x128"), _parse_layout("2x64"), _parse_layout("2x128")]
    selected = args.variant or [
        "pmg_shell_hypre",
        "pmg_shell_redundant_nodes",
        "hypre_lagged_pmis",
        "gamg_lagged_lowcomm",
        "bddc_ilu",
        "bddc_gamg",
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (args.out_root or (DEFAULT_OUT_ROOT / f"karolina_pc_screen_{stamp}")).resolve()
    config_root = out_root / "configs"
    script_root = out_root / "slurm"
    run_root = out_root / "runs"
    config_root.mkdir(parents=True, exist_ok=True)
    script_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    submitted: list[str] = []
    manifest_lines = ["variant\tlayout\tnodes\tranks_per_node\tranks\tconfig\tscript\trun_out\tnotes"]
    for nodes, ranks_per_node in layouts:
        ranks = nodes * ranks_per_node
        layout_name = f"{nodes}x{ranks_per_node}"
        for variant_name in selected:
            variant = VARIANTS[variant_name]
            run_name = f"{variant.name}_{layout_name}"
            source = CASE_DIR / variant.config
            case_config = config_root / f"{run_name}.toml"
            if variant.redundant_coarse_groups == "nodes":
                redundant_groups = nodes
            elif variant.redundant_coarse_groups is None:
                redundant_groups = None
            else:
                redundant_groups = int(variant.redundant_coarse_groups)
            _make_case_config(
                source=source,
                destination=case_config,
                case_name=f"p4_l1_{run_name}",
                ranks=ranks,
                redundant_groups=redundant_groups,
                step_max=args.step_max,
            )
            run_out = run_root / run_name
            run_out.mkdir(parents=True, exist_ok=True)
            script = script_root / f"{run_name}.sbatch"
            job_name = f"p4pc-{variant.name[:10]}-{layout_name}"
            script.write_text(
                _script_text(
                    job_name=job_name,
                    case_config=case_config,
                    run_out=run_out,
                    nodes=nodes,
                    ranks_per_node=ranks_per_node,
                    walltime=str(args.walltime),
                    queue=str(args.queue),
                ),
                encoding="utf-8",
            )
            script.chmod(0o755)
            manifest_lines.append(
                "\t".join(
                    [
                        variant.name,
                        layout_name,
                        str(nodes),
                        str(ranks_per_node),
                        str(ranks),
                        str(case_config),
                        str(script),
                        str(run_out),
                        variant.notes,
                    ]
                )
            )
            if args.submit:
                result = subprocess.run(["sbatch", str(script)], check=True, text=True, capture_output=True)
                submitted.append(result.stdout.strip())

    manifest = out_root / "manifest.tsv"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"prepared {len(manifest_lines) - 1} jobs under {out_root}")
    print(f"manifest: {manifest}")
    if submitted:
        print("\n".join(submitted))


if __name__ == "__main__":
    main()
