"""Doctor command helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from petsc_ssr.assets import available_problem_assets
from petsc_ssr.config.profiles import CONTINUATION_PROFILE_DIR, NEWTON_PROFILE_DIR, SEEPAGE_PROFILE_DIR, SOLVER_PROFILE_DIR


def doctor_payload(engine_root: Path) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "modules": {
            name: importlib.util.find_spec(name) is not None
            for name in ("numpy", "mpi4py", "petsc4py", "meshio", "scipy", "h5py", "matplotlib", "nbformat")
        },
        "native_extension": importlib.util.find_spec("petsc_ssr.native._core") is not None,
        "assets": available_problem_assets(),
        "continuation_profiles": sorted(path.stem for path in CONTINUATION_PROFILE_DIR.glob("*.toml")),
        "newton_profiles": sorted(path.stem for path in NEWTON_PROFILE_DIR.glob("*.toml")),
        "seepage_profiles": sorted(path.stem for path in SEEPAGE_PROFILE_DIR.glob("*.toml")),
        "solver_profiles": sorted(path.stem for path in SOLVER_PROFILE_DIR.glob("*.toml")),
        "suites": sorted(path.name for path in (engine_root / "benchmarks" / "suites").glob("*.toml")),
    }
