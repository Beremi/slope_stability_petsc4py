from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from study_common import (
    bool_cli_flag,
    canonical_matlab_env,
    canonical_petsc_env,
    load_horizons,
    load_study,
    quote_for_matlab,
    run_complete,
    save_horizons,
    write_json,
)


def append_ledger(artifact_root: Path, event: dict) -> None:
    path = artifact_root / "run_ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def output_dir_for_run(artifact_root: Path, case_id: str, level_id: str, phase: str, engine: str, variant: str) -> Path:
    run_tag = f"{phase}_{engine}_{variant}"
    return artifact_root / "runs" / case_id / level_id / run_tag


def write_run_metadata(output_dir: Path, metadata: dict) -> None:
    payload = dict(metadata)
    payload["output_dir"] = str(output_dir)
    write_json(output_dir / "study_run.json", payload)


def read_petsc_metrics(output_dir: Path) -> dict:
    run_info = json.loads((output_dir / "data" / "run_info.json").read_text(encoding="utf-8"))
    with np.load(output_dir / "data" / "petsc_run.npz", allow_pickle=True) as npz:
        omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
        lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    return {
        "runtime_seconds": float(run_info["run_info"]["runtime_seconds"]),
        "final_omega": float(omega_hist[-1]),
        "final_lambda": float(lambda_hist[-1]),
        "accepted_steps": int(lambda_hist.size),
    }


def read_matlab_metrics(output_dir: Path) -> dict:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    cont = summary.get("continuation", {})
    return {
        "runtime_seconds": float(summary["run_info"].get("runtime_seconds", 0.0)),
        "final_omega": float(cont.get("final_omega", float("nan"))),
        "final_lambda": float(cont.get("final_lambda", float("nan"))),
        "accepted_steps": int(cont.get("accepted_steps", 0)),
    }


def run_petsc_command(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("DRY-RUN PETSc:", " ".join(str(part) for part in cmd))
        return
    with log_path.open("w", encoding="utf-8") as log_handle:
        subprocess.run(cmd, cwd=cwd, env={**os.environ, **env}, check=True, text=True, stdout=log_handle, stderr=subprocess.STDOUT)


def run_matlab_capture(
    *,
    study: dict,
    script_name: str,
    config_json: Path,
    output_dir: Path,
    out_mat: Path,
    summary_json: Path,
    summary_h5: Path,
    threads: int,
    dry_run: bool,
) -> None:
    matlab_root = study["root"] / "slope_stability_matlab"
    matlab_scripts = matlab_root / "scripts"
    batch = (
        f"cd('{quote_for_matlab(str(study['root']))}'); "
        f"addpath(genpath('{quote_for_matlab(str(matlab_root.resolve()))}')); "
        f"addpath('{quote_for_matlab(str(matlab_scripts.resolve()))}'); "
        f"{script_name}('{quote_for_matlab(str(config_json))}', "
        f"'{quote_for_matlab(str(out_mat))}', "
        f"'{quote_for_matlab(str(output_dir))}'); "
        f"export_benchmark_summary('{quote_for_matlab(str(out_mat))}', "
        f"'{quote_for_matlab(str(summary_json))}', "
        f"'{quote_for_matlab(str(summary_h5))}');"
    )
    cmd = [str(study["matlab_bin"]), "-batch", batch]
    if dry_run:
        print("DRY-RUN MATLAB:", " ".join(cmd[:2]), "<batch omitted>")
        return

    env = {**os.environ, **canonical_matlab_env(threads)}
    proc = subprocess.Popen(cmd, cwd=study["root"], env=env, text=True)
    success_files = (out_mat, summary_json, summary_h5)
    success_seen_at: float | None = None
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc != 0 and not all(path.exists() for path in success_files):
                raise subprocess.CalledProcessError(rc, proc.args)
            break
        if all(path.exists() for path in success_files):
            if success_seen_at is None:
                success_seen_at = time.monotonic()
            elif time.monotonic() - success_seen_at >= 5.0:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)
                break
        time.sleep(1.0)


def build_common_petsc_args(study: dict, case: dict, level: dict, *, omega_max_stop: float, d_lambda_diff_scaled_min: float, newton_stopping_criterion: str, newton_stopping_tol: float | None) -> list[str]:
    defaults = study["defaults"]
    args = [
        "--out_dir", "{OUT_DIR}",
        "--mesh_path", str(level["petsc_mesh"]),
        "--elem_type", str(defaults["elem_type"]),
        "--node_ordering", str(defaults["node_ordering"]),
        "--lambda_init", str(case["lambda_init"]),
        "--d_lambda_init", str(case["d_lambda_init"]),
        "--d_lambda_min", str(case["d_lambda_min"]),
        "--d_lambda_diff_scaled_min", str(d_lambda_diff_scaled_min),
        "--omega_max_stop", str(omega_max_stop),
        "--continuation_predictor", str(defaults["continuation_predictor"]),
        "--omega_step_controller", str(defaults["omega_step_controller"]),
        "--step_max", str(defaults["step_max"]),
        "--it_newt_max", str(case["it_newt_max"]),
        "--it_damp_max", str(case["it_damp_max"]),
        "--tol", str(case["tol"]),
        "--r_min", str(defaults["r_min"]),
        "--newton_stopping_criterion", str(newton_stopping_criterion),
        "--linear_tolerance", str(defaults["petsc_linear_tolerance"]),
        "--linear_max_iter", str(defaults["petsc_linear_max_iter"]),
        "--preconditioner_threads", str(defaults["petsc_preconditioner_threads"]),
        "--solver_type", str(defaults["petsc_solver_type"]),
        "--pc_backend", str(defaults["petsc_pc_backend"]),
        "--pmg_fine_hierarchy_mode", str(defaults["pmg"]["fine_hierarchy_mode"]),
        bool_cli_flag("mpi_distribute_by_nodes", bool(defaults["mpi_distribute_by_nodes"])),
        "--pc_hypre_coarsen_type", "HMIS",
        "--pc_hypre_interp_type", "ext+i",
        bool_cli_flag("recycle_preconditioner", True),
        "--constitutive_mode", str(defaults["constitutive_mode"]),
        "--tangent_kernel", str(defaults["tangent_kernel"]),
    ]
    if level["pmg_coarse_mesh"] is not None:
        args.extend(["--pmg_coarse_mesh_path", str(level["pmg_coarse_mesh"])])
    if newton_stopping_tol is not None:
        args.extend(["--newton_stopping_tol", str(newton_stopping_tol)])
    for opt in defaults["pmg"]["petsc_opt"]:
        args.extend(["--petsc-opt", str(opt)])
    if case["boundary_mode"] != "none":
        args.extend(["--boundary_mode", str(case["boundary_mode"])])
        args.extend(["--seepage_linear_tolerance", str(defaults["petsc_seepage_linear_tolerance"])])
        args.extend(["--seepage_linear_max_iter", str(defaults["petsc_seepage_linear_max_iter"])])
        args.extend(["--water_unit_weight", str(case["water_unit_weight"])])
        for value in case["conductivity"]:
            args.extend(["--conductivity", str(value)])
    return args


def build_matlab_config(study: dict, case: dict, level: dict, *, omega_max_stop: float, d_lambda_diff_scaled_min: float) -> dict:
    defaults = study["defaults"]
    payload = {
        "mesh_file": str(level["matlab_mesh"]),
        "lambda_init": float(case["lambda_init"]),
        "d_lambda_init": float(case["d_lambda_init"]),
        "d_lambda_min": float(case["d_lambda_min"]),
        "d_lambda_diff_scaled_min": float(d_lambda_diff_scaled_min),
        "omega_max_stop": float(omega_max_stop),
        "step_max": int(defaults["step_max"]),
        "it_newt_max": int(case["it_newt_max"]),
        "it_damp_max": int(case["it_damp_max"]),
        "tol": float(case["tol"]),
        "r_min": float(defaults["r_min"]),
        "solver_type": str(defaults["matlab_solver_type"]),
        "boomeramg_threads": int(defaults["matlab_boomeramg_threads"]),
        "linear_solver_tolerance": float(defaults["petsc_linear_tolerance"]),
        "linear_solver_maxit": int(defaults["petsc_linear_max_iter"]),
        "deflation_basis_tolerance": 1e-3,
    }
    if case["boundary_mode"] != "none":
        payload["water_unit_weight"] = float(case["water_unit_weight"])
        payload["conductivity"] = list(case["conductivity"])
    return payload


def run_one_petsc(study: dict, case: dict, level: dict, *, phase: str, variant: str, omega_max_stop: float, d_lambda_diff_scaled_min: float, newton_stopping_criterion: str, newton_stopping_tol: float | None, dry_run: bool, resume: bool) -> dict:
    artifact_root = study["artifact_root"]
    output_dir = output_dir_for_run(artifact_root, case["id"], level["id"], phase, "petsc", variant)
    meta = {
        "case_id": case["id"],
        "case_label": case["label"],
        "case_order": case["order"],
        "level_id": level["id"],
        "level_label": level["label"],
        "level_order": level["order"],
        "engine": "petsc",
        "phase": phase,
        "variant": variant,
        "petsc_module": case["petsc_module"],
        "mesh_path": str(level["petsc_mesh"]),
        "shared_omega_max_stop": float(omega_max_stop),
        "d_lambda_diff_scaled_min": float(d_lambda_diff_scaled_min),
        "newton_residual_tol": float(case["tol"]),
        "newton_stopping_criterion": str(newton_stopping_criterion),
        "newton_stopping_tol": None if newton_stopping_tol is None else float(newton_stopping_tol),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, meta)
    append_ledger(artifact_root, {"event": "queued", **meta, "output_dir": str(output_dir)})

    if resume and run_complete(output_dir, "petsc"):
        append_ledger(artifact_root, {"event": "skipped_complete", **meta, "output_dir": str(output_dir)})
        return {"output_dir": output_dir, **read_petsc_metrics(output_dir)}

    args = build_common_petsc_args(
        study,
        case,
        level,
        omega_max_stop=omega_max_stop,
        d_lambda_diff_scaled_min=d_lambda_diff_scaled_min,
        newton_stopping_criterion=newton_stopping_criterion,
        newton_stopping_tol=newton_stopping_tol,
    )
    args[1] = str(output_dir)
    cmd = ["mpirun", "-n", str(study["mpi_ranks"]), str(study["petsc_python"]), "-m", case["petsc_module"], *args]
    append_ledger(artifact_root, {"event": "started", **meta, "output_dir": str(output_dir)})
    run_petsc_command(
        cmd,
        env=canonical_petsc_env(),
        cwd=study["root"],
        log_path=output_dir / "run.log",
        dry_run=dry_run,
    )
    if dry_run:
        return {"output_dir": output_dir}
    metrics = read_petsc_metrics(output_dir)
    append_ledger(artifact_root, {"event": "completed", **meta, "output_dir": str(output_dir), **metrics})
    return {"output_dir": output_dir, **metrics}


def run_one_matlab(study: dict, case: dict, level: dict, *, omega_max_stop: float, d_lambda_diff_scaled_min: float, dry_run: bool, resume: bool) -> dict:
    artifact_root = study["artifact_root"]
    output_dir = output_dir_for_run(artifact_root, case["id"], level["id"], "main", "matlab", "main")
    meta = {
        "case_id": case["id"],
        "case_label": case["label"],
        "case_order": case["order"],
        "level_id": level["id"],
        "level_label": level["label"],
        "level_order": level["order"],
        "engine": "matlab",
        "phase": "main",
        "variant": "main",
        "matlab_script": case["matlab_script"],
        "mesh_path": str(level["matlab_mesh"]),
        "shared_omega_max_stop": float(omega_max_stop),
        "d_lambda_diff_scaled_min": float(d_lambda_diff_scaled_min),
        "newton_residual_tol": float(case["tol"]),
        "newton_stopping_criterion": str(study["defaults"]["newton"]["criterion"]),
        "newton_stopping_tol": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(output_dir, meta)
    append_ledger(artifact_root, {"event": "queued", **meta, "output_dir": str(output_dir)})

    if resume and run_complete(output_dir, "matlab"):
        append_ledger(artifact_root, {"event": "skipped_complete", **meta, "output_dir": str(output_dir)})
        return {"output_dir": output_dir, **read_matlab_metrics(output_dir)}

    config_json = output_dir / "matlab_config.json"
    out_mat = output_dir / "matlab_run.mat"
    summary_json = output_dir / "summary.json"
    summary_h5 = output_dir / "summary.h5"
    write_json(config_json, build_matlab_config(study, case, level, omega_max_stop=omega_max_stop, d_lambda_diff_scaled_min=d_lambda_diff_scaled_min))
    append_ledger(artifact_root, {"event": "started", **meta, "output_dir": str(output_dir)})
    run_matlab_capture(
        study=study,
        script_name=case["matlab_script"],
        config_json=config_json,
        output_dir=output_dir,
        out_mat=out_mat,
        summary_json=summary_json,
        summary_h5=summary_h5,
        threads=int(study["defaults"]["matlab_omp_threads"]),
        dry_run=dry_run,
    )
    if dry_run:
        return {"output_dir": output_dir}
    metrics = read_matlab_metrics(output_dir)
    append_ledger(artifact_root, {"event": "completed", **meta, "output_dir": str(output_dir), **metrics})
    return {"output_dir": output_dir, **metrics}


def selected_cases(study: dict, ids: list[str] | None) -> list[dict]:
    if not ids:
        return list(study["cases"])
    keep = set(ids)
    return [case for case in study["cases"] if case["id"] in keep]


def ensure_matlab_meshes(study: dict, cases: list[dict]) -> None:
    missing = []
    for case in cases:
        for level in case["levels"]:
            if level["matlab_mesh"] is None or not level["matlab_mesh"].exists():
                missing.append(f"{case['id']}:{level['id']} -> {level['matlab_mesh']}")
    if missing:
        raise FileNotFoundError("MATLAB H5 meshes are unresolved:\n- " + "\n- ".join(missing))


def run_smoke_phase(study: dict, cases: list[dict], *, dry_run: bool, resume: bool) -> dict[str, dict]:
    horizons_path = study["artifact_root"] / "horizons.json"
    horizons = load_horizons(horizons_path)
    defaults = study["defaults"]
    for case in cases:
        if resume and case["id"] in horizons:
            continue
        level = case["levels"][0]
        result = run_one_petsc(
            study,
            case,
            level,
            phase="smoke",
            variant="smoke",
            omega_max_stop=float(case["omega_smoke_seed"]),
            d_lambda_diff_scaled_min=float(defaults["d_lambda_diff_scaled_min_smoke"]),
            newton_stopping_criterion=str(defaults["newton"]["criterion"]),
            newton_stopping_tol=None,
            dry_run=dry_run,
            resume=resume,
        )
        if not dry_run:
            horizons[case["id"]] = {
                "case_id": case["id"],
                "case_label": case["label"],
                "level_id": level["id"],
                "level_label": level["label"],
                "omega_max_stop": float(result["final_omega"]),
                "runtime_seconds": float(result["runtime_seconds"]),
                "accepted_steps": int(result["accepted_steps"]),
                "output_dir": str(result["output_dir"]),
            }
            save_horizons(horizons_path, horizons)
    return horizons


def run_main_phase(study: dict, cases: list[dict], *, dry_run: bool, resume: bool) -> None:
    horizons_path = study["artifact_root"] / "horizons.json"
    horizons = load_horizons(horizons_path)
    defaults = study["defaults"]
    runtime_limit = float(study["petsc_runtime_limit_seconds"])
    if not horizons and not dry_run:
        raise RuntimeError("No smoke horizons found. Run `run_study.py --phase smoke` first.")

    for case in cases:
        shared_omega = float(horizons.get(case["id"], {}).get("omega_max_stop", case["omega_smoke_seed"]))
        stop_after_case = False
        for level in case["levels"]:
            petsc_main = run_one_petsc(
                study,
                case,
                level,
                phase="main",
                variant="main",
                omega_max_stop=shared_omega,
                d_lambda_diff_scaled_min=float(defaults["d_lambda_diff_scaled_min_main"]),
                newton_stopping_criterion=str(defaults["newton"]["criterion"]),
                newton_stopping_tol=None,
                dry_run=dry_run,
                resume=resume,
            )
            run_one_matlab(
                study,
                case,
                level,
                omega_max_stop=shared_omega,
                d_lambda_diff_scaled_min=float(defaults["d_lambda_diff_scaled_min_main"]),
                dry_run=dry_run,
                resume=resume,
            )
            if case["appendix_delta_lambda"]:
                run_one_petsc(
                    study,
                    case,
                    level,
                    phase="main",
                    variant="delta_lambda",
                    omega_max_stop=shared_omega,
                    d_lambda_diff_scaled_min=float(defaults["d_lambda_diff_scaled_min_main"]),
                    newton_stopping_criterion=str(defaults["newton"]["delta_lambda_criterion"]),
                    newton_stopping_tol=float(defaults["newton"]["delta_lambda_tol"]),
                    dry_run=dry_run,
                    resume=resume,
                )
            if not dry_run and float(petsc_main["runtime_seconds"]) > runtime_limit:
                stop_after_case = True
                break
        if stop_after_case:
            continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PETSc/MATLAB 3D SSR comparison study sequentially.")
    parser.add_argument("--study", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--phase", type=str, default="all", choices=["all", "smoke", "main"])
    parser.add_argument("--case", action="append", default=None, dest="cases")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    study = load_study(args.study, manifest_path=args.manifest, allow_missing_manifest=args.dry_run)
    cases = selected_cases(study, args.cases)
    study["artifact_root"].mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        ensure_matlab_meshes(study, cases)

    if args.phase in {"smoke", "all"}:
        run_smoke_phase(study, cases, dry_run=args.dry_run, resume=args.resume)
    if args.phase in {"main", "all"}:
        run_main_phase(study, cases, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
