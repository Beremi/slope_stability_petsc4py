"""Distributed load-vector helpers for owned-row nonlinear solves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _comm_allreduce_sum(comm, value: float) -> float:
    if comm is None:
        return float(value)
    try:
        if int(comm.Get_size()) == 1:
            return float(value)
    except Exception:
        pass
    if hasattr(comm, "allreduce"):
        return float(comm.allreduce(float(value)))
    return float(comm.tompi4py().allreduce(float(value)))


@dataclass
class OwnedLocalLoadVector:
    """Owned-row representation of a global FE load vector.

    The row block is in global total-DOF ordering.  It is sufficient for the
    owned-row Newton path, where residual norms and load/displacement dots can
    be reduced from non-overlapping local row ranges.
    """

    local_values: np.ndarray
    owned_row_range: tuple[int, int]
    owned_free_mask: np.ndarray
    global_shape: tuple[int, int]
    comm: object | None = None

    def __post_init__(self) -> None:
        self.local_values = np.asarray(self.local_values, dtype=np.float64).reshape(-1)
        self.owned_row_range = tuple(int(v) for v in self.owned_row_range)
        self.owned_free_mask = np.asarray(self.owned_free_mask, dtype=bool).reshape(-1)
        expected = int(self.owned_row_range[1] - self.owned_row_range[0])
        if self.local_values.size != expected:
            raise ValueError(
                f"local_values has size {self.local_values.size}, expected owned row count {expected}."
            )
        if self.owned_free_mask.size != expected:
            raise ValueError(
                f"owned_free_mask has size {self.owned_free_mask.size}, expected owned row count {expected}."
            )
        self.global_shape = tuple(int(v) for v in self.global_shape)
        self._local_free_values: np.ndarray | None = None

    @classmethod
    def from_pattern(cls, local_values: np.ndarray, pattern, *, global_shape: tuple[int, int], comm=None):
        return cls(
            local_values=np.asarray(local_values, dtype=np.float64),
            owned_row_range=tuple(int(v) for v in pattern.owned_row_range),
            owned_free_mask=np.asarray(pattern.owned_free_mask, dtype=bool),
            global_shape=tuple(int(v) for v in global_shape),
            comm=comm,
        )

    @property
    def size(self) -> int:
        return int(self.global_shape[0] * self.global_shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.global_shape)

    def owned_rows(self, pattern=None) -> np.ndarray:
        if pattern is not None and tuple(int(v) for v in pattern.owned_row_range) != self.owned_row_range:
            raise ValueError("OwnedLocalLoadVector row range does not match the owned tangent pattern")
        return self.local_values

    def owned_free_rows(self, pattern=None) -> np.ndarray:
        if pattern is not None and tuple(int(v) for v in pattern.owned_row_range) != self.owned_row_range:
            raise ValueError("OwnedLocalLoadVector row range does not match the owned tangent pattern")
        if self._local_free_values is None:
            self._local_free_values = np.asarray(self.local_values[self.owned_free_mask], dtype=np.float64)
        return self._local_free_values

    def free_norm(self, pattern=None) -> float:
        local = self.owned_free_rows(pattern)
        return float(np.sqrt(max(_comm_allreduce_sum(self.comm, float(np.dot(local, local))), 0.0)))

    def dot_field(self, field: np.ndarray, pattern=None) -> float:
        if pattern is not None and tuple(int(v) for v in pattern.owned_row_range) != self.owned_row_range:
            raise ValueError("OwnedLocalLoadVector row range does not match the owned tangent pattern")
        row0, row1 = self.owned_row_range
        field_flat = np.asarray(field, dtype=np.float64).reshape(-1, order="F")
        if field_flat.size < row1:
            raise ValueError(f"field has {field_flat.size} entries, but owned row range ends at {row1}")
        field_local_free = np.asarray(field_flat[row0:row1][self.owned_free_mask], dtype=np.float64)
        local = self.owned_free_rows(pattern)
        return _comm_allreduce_sum(self.comm, float(np.dot(local, field_local_free)))

    def materialize_full(self) -> np.ndarray:
        """Gather the full load vector as a compatibility fallback."""

        row0, row1 = self.owned_row_range
        if self.comm is None:
            full = np.zeros(self.size, dtype=np.float64)
            full[row0:row1] = self.local_values
            return full.reshape(self.global_shape, order="F")
        parts = self.comm.allgather((row0, row1, np.asarray(self.local_values, dtype=np.float64)))
        full = np.zeros(self.size, dtype=np.float64)
        for lo, hi, values in parts:
            full[int(lo) : int(hi)] = np.asarray(values, dtype=np.float64).reshape(-1)
        return full.reshape(self.global_shape, order="F")

    def free_vector(self, q_mask: np.ndarray) -> np.ndarray:
        full = self.materialize_full()
        q_mask = np.asarray(q_mask, dtype=bool)
        from ..utils import q_to_free_indices

        return full.reshape(-1, order="F")[q_to_free_indices(q_mask)]


def is_owned_local_load(value: object) -> bool:
    return isinstance(value, OwnedLocalLoadVector)
