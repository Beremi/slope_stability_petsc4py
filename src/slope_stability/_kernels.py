"""Python fallback kernels used when the optional Cython extension is unavailable."""

from __future__ import annotations

import numpy as np


def dot(x: np.ndarray, y: np.ndarray) -> float:
    """Dot product of two 1D arrays."""

    return float(np.dot(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)))


def norm2(x: np.ndarray) -> float:
    """Euclidean norm of a 1D array."""

    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))
