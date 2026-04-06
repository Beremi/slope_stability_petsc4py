from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from study_common import DEFAULT_MANIFEST_PATH, load_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the PETSc/MATLAB performance comparison study.")
    parser.add_argument("--study", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--allow-missing-manifest", action="store_true", default=False)
    args = parser.parse_args()

    study = load_study(
        args.study,
        manifest_path=args.manifest,
        allow_missing_manifest=args.allow_missing_manifest,
    )

    issues: list[str] = []

    if not study["petsc_python"] or not study["petsc_python"].exists():
        issues.append(f"Missing PETSc python executable: {study['petsc_python']}")
    if not shutil.which("mpirun"):
        issues.append("`mpirun` is not available on PATH.")

    matlab_bin = study["matlab_bin"]
    if not matlab_bin or not matlab_bin.exists():
        issues.append(f"Missing MATLAB executable: {matlab_bin}")

    if not study["manifest"]["exists"] and not args.allow_missing_manifest:
        issues.append(
            f"Missing local manifest {DEFAULT_MANIFEST_PATH}. Copy mesh_manifest.example.toml and fill in the absolute H5 paths."
        )

    seen_case_ids: set[str] = set()
    for case in study["cases"]:
        if case["id"] in seen_case_ids:
            issues.append(f"Duplicate case id: {case['id']}")
        seen_case_ids.add(case["id"])

        level_ids: set[str] = set()
        for level in case["levels"]:
            if level["id"] in level_ids:
                issues.append(f"Duplicate level id in {case['id']}: {level['id']}")
            level_ids.add(level["id"])

            petsc_mesh = level["petsc_mesh"]
            if petsc_mesh is None or not petsc_mesh.exists():
                issues.append(f"Missing PETSc mesh for {case['id']}:{level['id']} -> {petsc_mesh}")

            coarse_mesh = level["pmg_coarse_mesh"]
            if coarse_mesh is not None and not coarse_mesh.exists():
                issues.append(f"Missing PMG coarse mesh for {case['id']}:{level['id']} -> {coarse_mesh}")

            matlab_mesh = level["matlab_mesh"]
            if matlab_mesh is None:
                if not args.allow_missing_manifest:
                    issues.append(
                        f"Manifest does not resolve MATLAB H5 key `{level['matlab_mesh_key']}` for {case['id']}:{level['id']}"
                    )
            elif not matlab_mesh.exists():
                if not args.allow_missing_manifest:
                    issues.append(f"Missing MATLAB H5 mesh for {case['id']}:{level['id']} -> {matlab_mesh}")

    if issues:
        print("Validation failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("Validation passed.")
    print(f"- Study: {study['study_path']}")
    print(f"- Manifest: {study['manifest']['path']}")
    print(f"- PETSc python: {study['petsc_python']}")
    print(f"- MATLAB: {study['matlab_bin']}")
    print(f"- MPI ranks: {study['mpi_ranks']}")
    print(f"- Cases: {len(study['cases'])}")


if __name__ == "__main__":
    main()
