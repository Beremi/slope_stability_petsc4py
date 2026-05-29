import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
PACKAGE_DIR = SRC_DIR / "petsc_ssr"
NATIVE_DIR = PACKAGE_DIR / "native"
NATIVE_INCLUDE_DIRS = [
    NATIVE_DIR / "include",
    NATIVE_DIR / "assembly",
    NATIVE_DIR / "materials",
    NATIVE_DIR / "mesh",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _petsc_flags() -> tuple[list[str], list[str], list[str]]:
    petsc_dir = os.environ.get("PETSC_DIR")
    petsc_arch = os.environ.get("PETSC_ARCH")
    if not petsc_dir:
        candidate = ROOT / ".build" / "src" / "petsc-3.24.5"
        if candidate.exists():
            petsc_dir = str(candidate)
            petsc_arch = petsc_arch or "linux-c-opt"
    if not petsc_dir:
        try:
            import petsc4py

            cfg = petsc4py.get_config()
            petsc_dir = cfg.get("PETSC_DIR")
            petsc_arch = petsc_arch or cfg.get("PETSC_ARCH")
        except Exception:
            petsc_dir = None
    if not petsc_dir:
        raise RuntimeError("Set PETSC_DIR/PETSC_ARCH or install petsc4py with PETSc metadata")

    env = os.environ.copy()
    pkg_dirs = []
    if petsc_arch:
        pkg_dirs.append(str(Path(petsc_dir) / petsc_arch / "lib" / "pkgconfig"))
    pkg_dirs.append(str(Path(petsc_dir) / "lib" / "pkgconfig"))
    old_pkg = env.get("PKG_CONFIG_PATH")
    env["PKG_CONFIG_PATH"] = os.pathsep.join(pkg_dirs + ([old_pkg] if old_pkg else []))
    try:
        cflags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "petsc"], text=True, env=env))
        libs = shlex.split(subprocess.check_output(["pkg-config", "--libs", "petsc"], text=True, env=env))
    except Exception:
        include_dirs = [str(Path(petsc_dir) / "include")]
        if petsc_arch:
            include_dirs.insert(0, str(Path(petsc_dir) / petsc_arch / "include"))
            libs = ["-L" + str(Path(petsc_dir) / petsc_arch / "lib"), "-lpetsc"]
        else:
            libs = ["-L" + str(Path(petsc_dir) / "lib"), "-lpetsc"]
        return include_dirs, [], libs

    include_dirs = [flag[2:] for flag in cflags if flag.startswith("-I")]
    compile_flags = [flag for flag in cflags if not flag.startswith("-I")]
    return include_dirs, compile_flags, libs


petsc_includes, petsc_cflags, petsc_libs = _petsc_flags()
link_flags = [flag for flag in petsc_libs if flag.startswith("-l") or flag.startswith("-L") or flag.startswith("-Wl,")]
rpath_flags = ["-Wl,-rpath," + flag[2:] for flag in link_flags if flag.startswith("-L")]

extension = Extension(
    name="petsc_ssr.native._core",
    sources=[
        _rel(NATIVE_DIR / "cython" / "_core.pyx"),
        _rel(NATIVE_DIR / "core" / "engine_main.c"),
        _rel(NATIVE_DIR / "mesh" / "hydro_seepage.c"),
        _rel(NATIVE_DIR / "assembly" / "assembly.c"),
        _rel(NATIVE_DIR / "assembly" / "neumann.c"),
        _rel(NATIVE_DIR / "materials" / "material_mc.c"),
        _rel(NATIVE_DIR / "assembly" / "p4_basis.c"),
    ],
    depends=[
        _rel(path)
        for path in sorted(NATIVE_DIR.glob("*/*.c.inc"))
    ]
    + [
        *(_rel(path) for path in sorted((NATIVE_DIR / "include").glob("*.h"))),
        _rel(NATIVE_DIR / "mesh" / "hydro_seepage.h"),
        _rel(NATIVE_DIR / "assembly" / "assembly.h"),
        _rel(NATIVE_DIR / "assembly" / "p4_basis.h"),
        _rel(NATIVE_DIR / "materials" / "material_mc.h"),
    ],
    include_dirs=[np.get_include(), *(str(path) for path in NATIVE_INCLUDE_DIRS), *petsc_includes],
    extra_compile_args=[
        "-O2",
        "-std=c11",
        "-DP4_INDIRECT_SSR_NO_MAIN",
        "-Wno-unused-function",
        *petsc_cflags,
    ],
    extra_link_args=[*rpath_flags, *link_flags],
)


setup(
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=cythonize([extension], annotate=False),
)
