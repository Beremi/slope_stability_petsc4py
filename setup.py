from pathlib import Path
from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import numpy as np


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

setup(
    ext_modules=cythonize(extensions, annotate=False),
)
