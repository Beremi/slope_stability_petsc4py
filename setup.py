import os
import shlex
import subprocess
from pathlib import Path
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import numpy as np


ROOT = Path(__file__).resolve().parent


def _petsc_pkg_config_flags() -> tuple[list[str], list[str]]:
    petsc_dir = os.environ.get("PETSC_DIR")
    petsc_arch = os.environ.get("PETSC_ARCH")
    if not petsc_dir:
        try:
            import petsc4py

            cfg = petsc4py.get_config()
            petsc_dir = cfg.get("PETSC_DIR")
            petsc_arch = cfg.get("PETSC_ARCH")
        except Exception:
            petsc_dir = None
    if not petsc_dir:
        candidate = ROOT / ".build" / "src" / "petsc-3.24.5"
        if candidate.exists():
            petsc_dir = str(candidate)
            petsc_arch = petsc_arch or "linux-c-opt"
    if not petsc_dir:
        return [], []

    env = os.environ.copy()
    pkg_dirs = []
    if petsc_arch:
        pkg_dirs.append(str(Path(petsc_dir) / petsc_arch / "lib" / "pkgconfig"))
    pkg_dirs.append(str(Path(petsc_dir) / "lib" / "pkgconfig"))
    existing = env.get("PKG_CONFIG_PATH")
    env["PKG_CONFIG_PATH"] = os.pathsep.join(pkg_dirs + ([existing] if existing else []))
    try:
        cflags = shlex.split(
            subprocess.check_output(["pkg-config", "--cflags", "petsc"], text=True, env=env).strip()
        )
        libs = shlex.split(
            subprocess.check_output(["pkg-config", "--libs", "petsc"], text=True, env=env).strip()
        )
    except Exception:
        include_dirs = [str(Path(petsc_dir) / "include")]
        link_flags = []
        if petsc_arch:
            include_dirs.insert(0, str(Path(petsc_dir) / petsc_arch / "include"))
            link_flags.extend(["-L" + str(Path(petsc_dir) / petsc_arch / "lib"), "-lpetsc"])
        return include_dirs, link_flags

    include_dirs = [flag[2:] for flag in cflags if flag.startswith("-I")]
    compile_flags = [flag for flag in cflags if not flag.startswith("-I")]
    return include_dirs, compile_flags + libs


def _optional_petsc_ssr_extension() -> Extension | None:
    petsc_include_dirs, petsc_flags = _petsc_pkg_config_flags()
    if not petsc_include_dirs and not any(flag == "-lpetsc" or flag.endswith("/libpetsc.so") for flag in petsc_flags):
        return None
    standalone = ROOT / "standalone_petsc_indirect_ssr"
    standalone_rel = Path("standalone_petsc_indirect_ssr")
    link_flags = [flag for flag in petsc_flags if flag.startswith("-l") or flag.startswith("-L") or flag.startswith("-Wl,")]
    rpath_flags = ["-Wl,-rpath," + flag[2:] for flag in link_flags if flag.startswith("-L")]
    return Extension(
        name="slope_stability._petsc_ssr",
        sources=[
            str(Path("src") / "slope_stability" / "cython" / "_petsc_ssr.pyx"),
            str(standalone_rel / "p4_indirect_ssr.c"),
            str(standalone_rel / "assembly.c"),
            str(standalone_rel / "material_mc.c"),
            str(standalone_rel / "p4_basis.c"),
        ],
        include_dirs=[
            np.get_include(),
            str(Path("src") / "slope_stability" / "cython"),
            str(standalone),
            *petsc_include_dirs,
        ],
        extra_compile_args=[
            "-O2",
            "-std=c11",
            "-DP4_INDIRECT_SSR_NO_MAIN",
            "-Wno-unused-function",
            *[flag for flag in petsc_flags if not flag.startswith("-l") and not flag.startswith("-L") and not flag.startswith("-Wl,")],
        ],
        extra_link_args=[*rpath_flags, *link_flags],
    )


extensions = [
    Extension(
        name="slope_stability._kernels",
        sources=[
            str(Path("src") / "slope_stability" / "cython" / "_kernels.pyx"),
            str(Path("src") / "slope_stability" / "cython" / "assemble_tangent_values_3d.c"),
            str(Path("src") / "slope_stability" / "cython" / "constitutive_3d_batch.c"),
        ],
        include_dirs=[np.get_include(), str(Path("src") / "slope_stability" / "cython")],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]
_petsc_ssr_ext = _optional_petsc_ssr_extension()
if _petsc_ssr_ext is not None:
    extensions.append(_petsc_ssr_ext)

setup(
    ext_modules=cythonize(extensions, annotate=False),
)
