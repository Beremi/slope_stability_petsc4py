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


def _toml_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _quote_toml(value: object) -> str:
    return json.dumps(str(value))


def _toml_string_list(values: list[object]) -> str:
    return "[" + ", ".join(_quote_toml(value) for value in values) + "]"


def write_petsc_case_config(
    path: Path,
    study: dict,
    case: dict,
    level: dict,
    *,
    phase: str,
    variant: str,
    omega_max_stop: float,
    d_lambda_diff_scaled_min: float,
    newton_stopping_criterion: str,
    newton_stopping_tol: float | None,
) -> Path:
    defaults = study["defaults"]
    run_name = f"{case['id']}_{level['id']}_{phase}_{variant}"
    newton_lines = [
        "[newton]",
        f"it_max = {int(case['it_newt_max'])}",
        f"it_damp_max = {int(case['it_damp_max'])}",
        f"tol = {float(case['tol']):.16g}",
        f"r_min = {float(defaults['r_min']):.16g}",
        f"stopping_criterion = {_quote_toml(newton_stopping_criterion)}",
    ]
    if newton_stopping_tol is not None:
        newton_lines.append(f"stopping_tol = {float(newton_stopping_tol):.16g}")
    linear_lines = [
        "[linear_solver]",
        f"solver_type = {_quote_toml(defaults['petsc_solver_type'])}",
        f"tolerance = {float(defaults['petsc_linear_tolerance']):.16g}",
        f"max_iterations = {int(defaults['petsc_linear_max_iter'])}",
        f"threads = {int(defaults['petsc_preconditioner_threads'])}",
        f"pc_backend = {_quote_toml(defaults['petsc_pc_backend'])}",
        f"pmg_fine_hierarchy_mode = {_quote_toml(defaults['pmg']['fine_hierarchy_mode'])}",
    ]
    if level["pmg_coarse_mesh_variant"] is not None:
        linear_lines.append(f"pmg_coarse_mesh_variant = {_quote_toml(level['pmg_coarse_mesh_variant'])}")
    linear_lines.extend(
        [
            'pc_hypre_coarsen_type = "HMIS"',
            'pc_hypre_interp_type = "ext+i"',
            "recycle_preconditioner = true",
            f"petsc_opt = {_toml_string_list(list(defaults['pmg']['petsc_opt']))}",
        ]
    )
    lines = [
        "[problem]",
        f"name = {_quote_toml(run_name)}",
        f"asset = {_quote_toml(level['asset'])}",
        f"mesh_variant = {_quote_toml(level['mesh_variant'])}",
        'analysis = "ssr"',
        f"elem_type = {_quote_toml(defaults['elem_type'])}",
        "",
        "[execution]",
        f"node_ordering = {_quote_toml(defaults['node_ordering'])}",
        f"mpi_distribute_by_nodes = {_toml_bool(defaults['mpi_distribute_by_nodes'])}",
        f"constitutive_mode = {_quote_toml(defaults['constitutive_mode'])}",
        f"tangent_kernel = {_quote_toml(defaults['tangent_kernel'])}",
        "",
        "[continuation]",
        f"method = {_quote_toml(defaults['continuation_method'])}",
        f"predictor = {_quote_toml(defaults['continuation_predictor'])}",
        f"omega_step_controller = {_quote_toml(defaults['omega_step_controller'])}",
        f"lambda_init = {float(case['lambda_init']):.16g}",
        f"d_lambda_init = {float(case['d_lambda_init']):.16g}",
        f"d_lambda_min = {float(case['d_lambda_min']):.16g}",
        f"d_lambda_diff_scaled_min = {float(d_lambda_diff_scaled_min):.16g}",
        f"omega_max = {float(omega_max_stop):.16g}",
        f"step_max = {int(defaults['step_max'])}",
        "",
        *newton_lines,
        "",
        *linear_lines,
        "",
        "[seepage]",
        f"linear_tolerance = {float(defaults['petsc_seepage_linear_tolerance']):.16g}",
        f"linear_max_iter = {int(defaults['petsc_seepage_linear_max_iter'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


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
        "petsc_entrypoint": "slope_stability.cli.run_case_from_config",
        "asset": str(level["asset"]),
        "mesh_variant": str(level["mesh_variant"]),
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

    case_config = write_petsc_case_config(
        output_dir / "case.toml",
        study,
        case,
        level,
        phase=phase,
        variant=variant,
        omega_max_stop=omega_max_stop,
        d_lambda_diff_scaled_min=d_lambda_diff_scaled_min,
        newton_stopping_criterion=newton_stopping_criterion,
        newton_stopping_tol=newton_stopping_tol,
    )
    cmd = [
        "mpirun",
        "-n",
        str(study["mpi_ranks"]),
        str(study["petsc_python"]),
        "-m",
        "slope_stability.cli.run_case_from_config",
        str(case_config),
        "--out_dir",
        str(output_dir),
    ]
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
