"""Iteration bookkeeping for linear solvers.

The implementation follows the MATLAB ``IterationCollector`` structure but keeps
per-instance vectors of total solver calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class IterationCollector:
    """Collect counters and timing statistics for iterative solves."""

    iterations: List[List[int]] = field(default_factory=list)
    solve_times: List[List[float]] = field(default_factory=list)
    preconditioner_times: List[List[float]] = field(default_factory=list)
    orthogonalization_times: List[List[float]] = field(default_factory=list)

    def register_instance(self) -> int:
        """Register a new solver instance and return its index."""

        self.iterations.append([])
        self.solve_times.append([])
        self.preconditioner_times.append([])
        self.orthogonalization_times.append([])
        return len(self.iterations)

    def store_iteration(self, instance_id: int, nit: int, time_sec: float) -> None:
        self.iterations[instance_id - 1].append(int(nit))
        self.solve_times[instance_id - 1].append(float(time_sec))

    def store_preconditioner_time(self, instance_id: int, time_sec: float) -> None:
        self.preconditioner_times[instance_id - 1].append(float(time_sec))

    def store_orthogonalization_time(self, instance_id: int, time_sec: float) -> None:
        self.orthogonalization_times[instance_id - 1].append(float(time_sec))

    def _flatten(self, values: List[List[float | int]]) -> List[float]:
        return [float(x) for block in values for x in block]

    def get_total_iterations(self) -> int:
        return sum(sum(values) for values in self.iterations)

    def get_total_solve_time(self) -> float:
        return sum(self._flatten(self.solve_times))

    def get_total_preconditioner_time(self) -> float:
        return sum(self._flatten(self.preconditioner_times))

    def get_total_orthogonalization_time(self) -> float:
        return sum(self._flatten(self.orthogonalization_times))

    def get_total_time(self) -> float:
        return (
            self.get_total_solve_time()
            + self.get_total_preconditioner_time()
            + self.get_total_orthogonalization_time()
        )

    def get_iterations_vector(self) -> List[int]:
        return [int(v) for block in self.iterations for v in block]

    def get_solve_time_vector(self) -> List[float]:
        return self._flatten(self.solve_times)

    def get_preconditioner_time_vector(self) -> List[float]:
        return self._flatten(self.preconditioner_times)

    def get_orthogonalization_time_vector(self) -> List[float]:
        return self._flatten(self.orthogonalization_times)
