from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_orchestration_modules_import_without_petsc4py() -> None:
    code = r'''
import importlib
import importlib.abc

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == "petsc4py":
            raise ImportError(f"blocked PETSc dependency {fullname}")
        return None

import sys
sys.meta_path.insert(0, Blocker())

for module in (
        "petsc_ssr.case_config",
        "petsc_ssr.config.manifest",
        "petsc_ssr.hydro_cases",
        "petsc_ssr.runners.run_case_from_config",
):
    importlib.import_module(module)
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def test_pyproject_keeps_runtime_dependency_footprint_minimal() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    runtime = set(pyproject["project"]["dependencies"])
    optional = pyproject["project"]["optional-dependencies"]

    assert runtime == {"numpy>=1.24", "mpi4py>=4.0", "petsc4py>=3.24,<3.25"}
    assert optional["mesh"] == ["meshio>=5.3"]
    assert optional["hdf5"] == ["h5py>=3.8"]
    assert "h5py>=3.8" in optional["reports"]
    assert "h5py>=3.8" not in optional["mesh"]
    assert "scipy>=1.10" in optional["seepage"]
    assert "meshio>=5.3" not in optional["reports"]
    assert "pyvista[jupyter]>=0.43" in optional["notebooks"]


def test_lightweight_cli_and_registry_import_without_optional_extras() -> None:
    code = r'''
import importlib
import importlib.abc
import json
import sys

blocked = {"meshio", "scipy", "h5py", "matplotlib", "nbformat", "nbclient", "pyvista", "ipywidgets"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ImportError(f"blocked optional dependency {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
for module in (
    "petsc_ssr",
    "petsc_ssr.cli.main",
    "petsc_ssr.config.case_schema",
    "petsc_ssr.config.manifest",
        "petsc_ssr.config.profiles",
        "petsc_ssr.export",
        "petsc_ssr.io",
        "petsc_ssr.mesh.loader",
        "petsc_ssr.runtime",
        "petsc_ssr.runtime.environment",
        "petsc_ssr.runtime.results",
        "petsc_ssr.benchmarks.registry",
    "petsc_ssr.benchmarks.suites",
    "petsc_ssr.runners.run_case_from_config",
):
    importlib.import_module(module)

from petsc_ssr.benchmarks.registry import discover_benchmark_registry
payload = discover_benchmark_registry()
assert payload["cases"] and payload["suites"] and payload["targets"]
print(json.dumps({"cases": len(payload["cases"]), "suites": len(payload["suites"]), "targets": len(payload["targets"])}))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, check=True)

    payload = json.loads(result.stdout)
    assert payload["suites"] >= 1


def test_hdf5_features_fail_at_feature_boundary_without_h5py() -> None:
    code = r'''
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] == "h5py":
            raise ImportError(f"blocked optional dependency {fullname}")
        return None

sys.meta_path.insert(0, Blocker())

from petsc_ssr.export import _load_h5py as load_export_h5py
from petsc_ssr.io import _load_h5py as load_mesh_h5py

for loader in (load_export_h5py, load_mesh_h5py):
    try:
        loader()
    except ImportError as exc:
        assert "h5py" in str(exc)
    else:
        raise AssertionError("h5py feature boundary did not reject missing optional dependency")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
