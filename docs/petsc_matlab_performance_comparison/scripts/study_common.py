from __future__ import annotations

import json
import os
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = STUDY_DIR / "data"
FIGURES_DIR = STUDY_DIR / "figures"
GENERATED_DIR = STUDY_DIR / "generated"
DEFAULT_STUDY_PATH = STUDY_DIR / "study.toml"
DEFAULT_MANIFEST_PATH = STUDY_DIR / "mesh_manifest.local.toml"
EXAMPLE_MANIFEST_PATH = STUDY_DIR / "mesh_manifest.example.toml"


def load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def resolve_path(base: Path, value: str | os.PathLike | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def absolutize_path(base: Path, value: str | os.PathLike | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.absolute()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def quote_for_matlab(text: str) -> str:
    return str(text).replace("'", "''")


def bool_cli_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def canonical_petsc_env() -> dict[str, str]:
    return {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _find_hypre_lib_dir() -> Path | None:
    search_roots = [
        ROOT / ".build",
        ROOT,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.glob("**/libHYPRE-3.0.0.so"):
            return candidate.parent
    return None


def canonical_matlab_env(threads: int) -> dict[str, str]:
    value = str(int(threads))
    env = {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }
    hypre_lib_dir = _find_hypre_lib_dir()
    if hypre_lib_dir is not None:
        existing = os.environ.get("LD_LIBRARY_PATH", "").strip()
        env["LD_LIBRARY_PATH"] = (
            f"{hypre_lib_dir}:{existing}" if existing else str(hypre_lib_dir)
        )
    return env


def load_mesh_manifest(
    manifest_path: Path | None = None,
    *,
    allow_missing: bool = False,
) -> dict:
    path = DEFAULT_MANIFEST_PATH if manifest_path is None else Path(manifest_path)
    if not path.exists():
        if allow_missing:
            return {"path": path, "exists": False, "mapping": {}}
        raise FileNotFoundError(
            f"Missing mesh manifest: {path}. Copy {EXAMPLE_MANIFEST_PATH.name} to {path.name} and fill in the local H5 paths."
        )

    raw = load_toml(path)
    mapping = {}
    for key, value in raw.get("matlab_h5", {}).items():
        mapping[str(key)] = resolve_path(path.parent, value)
    return {"path": path, "exists": True, "mapping": mapping}


def load_study(
    study_path: Path | None = None,
    *,
    manifest_path: Path | None = None,
    allow_missing_manifest: bool = False,
) -> dict:
    path = DEFAULT_STUDY_PATH if study_path is None else Path(study_path)
    raw = load_toml(path)
    manifest = load_mesh_manifest(manifest_path, allow_missing=allow_missing_manifest)

    study_cfg = dict(raw.get("study", {}))
    defaults = dict(raw.get("defaults", {}))
    defaults.setdefault("newton", {})
    defaults.setdefault("pmg", {})

    cases = []
    for case_order, case in enumerate(raw.get("cases", [])):
        case_item = dict(case)
        levels = []
        for level_order, level in enumerate(case_item.get("levels", [])):
            level_item = dict(level)
            matlab_key = str(level_item["matlab_mesh_key"])
            asset = str(level_item.get("asset", case_item.get("asset", "")))
            mesh_variant = str(level_item["mesh_variant"])
            pmg_coarse_mesh_variant = level_item.get("pmg_coarse_mesh_variant")
            if pmg_coarse_mesh_variant is not None:
                pmg_coarse_mesh_variant = str(pmg_coarse_mesh_variant)

            levels.append(
                {
                    "id": str(level_item["id"]),
                    "label": str(level_item["label"]),
                    "order": int(level_order),
                    "asset": asset,
                    "mesh_variant": mesh_variant,
                    "pmg_coarse_mesh_variant": pmg_coarse_mesh_variant,
                    "matlab_mesh_key": matlab_key,
                    "matlab_mesh": manifest["mapping"].get(matlab_key),
                }
            )
        cases.append(
            {
                "id": str(case_item["id"]),
                "label": str(case_item["label"]),
                "report_slug": str(case_item.get("report_slug", case_item["id"])),
                "order": int(case_order),
                "matlab_script": str(case_item["matlab_script"]),
                "boundary_mode": str(case_item.get("boundary_mode", "none")),
                "lambda_init": float(case_item["lambda_init"]),
                "d_lambda_init": float(case_item.get("d_lambda_init", defaults["d_lambda_init"])),
                "d_lambda_min": float(case_item["d_lambda_min"]),
                "omega_smoke_seed": float(case_item["omega_smoke_seed"]),
                "it_newt_max": int(case_item["it_newt_max"]),
                "it_damp_max": int(case_item["it_damp_max"]),
                "tol": float(case_item.get("tol", defaults["tol"])),
                "appendix_delta_lambda": bool(case_item.get("appendix_delta_lambda", False)),
                "water_unit_weight": float(case_item.get("water_unit_weight", 9.81)),
                "conductivity": list(case_item.get("conductivity", [])),
                "levels": levels,
            }
        )

    artifact_root = resolve_path(path.parent, study_cfg["artifact_root"])
    petsc_python = absolutize_path(path.parent, study_cfg["petsc_python"])
    matlab_bin = absolutize_path(path.parent, study_cfg["matlab_bin"])

    return {
        "study_path": path,
        "root": ROOT,
        "study_dir": STUDY_DIR,
        "data_dir": DATA_DIR,
        "figures_dir": FIGURES_DIR,
        "generated_dir": GENERATED_DIR,
        "artifact_root": artifact_root,
        "petsc_python": petsc_python,
        "matlab_bin": matlab_bin,
        "mpi_ranks": int(study_cfg.get("mpi_ranks", 8)),
        "petsc_runtime_limit_seconds": float(study_cfg.get("petsc_runtime_limit_seconds", 1000.0)),
        "title": str(study_cfg.get("title", "")),
        "defaults": defaults,
        "cases": cases,
        "manifest": manifest,
    }


def case_map(study: dict) -> dict[str, dict]:
    return {case["id"]: case for case in study["cases"]}


def completion_files(engine: str) -> tuple[str, ...]:
    if engine == "petsc":
        return ("data/run_info.json", "data/petsc_run.npz", "study_run.json")
    if engine == "matlab":
        return ("summary.json", "summary.h5", "matlab_run.mat", "study_run.json")
    raise ValueError(f"Unsupported engine {engine!r}")


def run_complete(output_dir: Path, engine: str) -> bool:
    return all((output_dir / rel).exists() for rel in completion_files(engine))


def load_horizons(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return read_json(path)


def save_horizons(path: Path, horizons: dict[str, dict]) -> None:
    write_json(path, horizons)
