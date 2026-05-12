from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np
from mpi4py import MPI


@dataclass(frozen=True)
class NumaMPILayout:
    rank: int
    size: int

    node_rank: int
    node_size: int
    node_id: int
    node_count: int

    numa_domains_per_node: int
    hw_numa_id: int
    local_numa_id: int
    global_numa_id: int

    local_rank_in_numa: int
    ranks_per_numa: int
    total_numa_domains: int

    rank_global_numa: tuple[int, ...]
    rank_local_in_numa: tuple[int, ...]
    rank_node_id: tuple[int, ...]


def _current_bound_cpu() -> int:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != 1:
        raise RuntimeError(
            "Expected exactly one bound CPU per MPI rank. "
            f"Got affinity mask {cpus}. Launch with --cpus-per-task=1 --cpu-bind=cores."
        )
    return int(cpus[0])


def _numa_node_for_cpu(cpu: int) -> int:
    cpu_dir = Path(f"/sys/devices/system/cpu/cpu{int(cpu)}")
    matches = sorted(cpu_dir.glob("node[0-9]*"))
    if not matches:
        raise RuntimeError(f"Could not find NUMA node for CPU {cpu} under {cpu_dir}.")
    return int(matches[0].name.removeprefix("node"))


def discover_numa_mpi_layout(
    comm: MPI.Comm = MPI.COMM_WORLD,
    *,
    numa_domains_per_node: int = 8,
    require_contiguous_world_ranks: bool = True,
) -> NumaMPILayout:
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        node_rank = int(node_comm.Get_rank())
        node_size = int(node_comm.Get_size())

        leader_rank = int(node_comm.allreduce(rank if node_rank == 0 else size + rank, op=MPI.MIN))
        all_leaders = sorted(set(int(v) for v in comm.allgather(leader_rank)))
        node_id = int(all_leaders.index(leader_rank))
        node_count = int(len(all_leaders))

        cpu = _current_bound_cpu()
        hw_numa_id = _numa_node_for_cpu(cpu)

        local_hw_numa = [int(v) for v in node_comm.allgather(hw_numa_id)]
        local_numa_values = sorted(set(local_hw_numa))

        if len(local_numa_values) != int(numa_domains_per_node):
            raise RuntimeError(
                f"Expected {numa_domains_per_node} NUMA domains on this node, "
                f"but ranks see hardware NUMA ids {local_numa_values}."
            )

        local_numa_id = int(local_numa_values.index(hw_numa_id))

        ranks_in_this_numa = [i for i, v in enumerate(local_hw_numa) if int(v) == int(hw_numa_id)]
        local_rank_in_numa = int(ranks_in_this_numa.index(node_rank))
        ranks_per_numa = int(len(ranks_in_this_numa))

        local_counts = [sum(1 for v in local_hw_numa if int(v) == int(nid)) for nid in local_numa_values]
        if len(set(local_counts)) != 1:
            raise RuntimeError(f"Unequal MPI-rank counts per NUMA domain on node {node_id}: {local_counts}")

        global_numa_id = int(node_id * int(numa_domains_per_node) + local_numa_id)
        total_numa_domains = int(node_count * int(numa_domains_per_node))

        rank_global_numa = tuple(int(v) for v in comm.allgather(global_numa_id))
        rank_local_in_numa = tuple(int(v) for v in comm.allgather(local_rank_in_numa))
        rank_node_id = tuple(int(v) for v in comm.allgather(node_id))

        if require_contiguous_world_ranks:
            expected_group = int(ranks_per_numa)
            for domain_id in range(total_numa_domains):
                first = domain_id * expected_group
                last = first + expected_group
                got = rank_global_numa[first:last]
                if len(got) != expected_group or any(int(v) != domain_id for v in got):
                    raise RuntimeError(
                        "MPI ranks are not contiguous by NUMA domain. "
                        "PETSc PCGASM coalescing will not produce one block per NUMA. "
                        f"Domain {domain_id} expected ranks {first}:{last}, got global NUMA ids {got}. "
                        "Fix the Slurm/MPI rank binding before running production solves."
                    )

        return NumaMPILayout(
            rank=rank,
            size=size,
            node_rank=node_rank,
            node_size=node_size,
            node_id=node_id,
            node_count=node_count,
            numa_domains_per_node=int(numa_domains_per_node),
            hw_numa_id=int(hw_numa_id),
            local_numa_id=int(local_numa_id),
            global_numa_id=int(global_numa_id),
            local_rank_in_numa=int(local_rank_in_numa),
            ranks_per_numa=int(ranks_per_numa),
            total_numa_domains=int(total_numa_domains),
            rank_global_numa=rank_global_numa,
            rank_local_in_numa=rank_local_in_numa,
            rank_node_id=rank_node_id,
        )
    finally:
        node_comm.Free()


def split_domain_offsets_to_rank_offsets(
    *,
    n_blocks: int,
    domain_offsets: np.ndarray,
    layout: NumaMPILayout,
) -> np.ndarray:
    """Split each contiguous NUMA-domain block into per-MPI-rank blocks."""

    domain_offsets = np.asarray(domain_offsets, dtype=np.int64).reshape(-1)
    if domain_offsets.size != layout.total_numa_domains + 1:
        raise ValueError(
            "domain_offsets must have size total_numa_domains + 1 "
            f"({layout.total_numa_domains + 1}), got {domain_offsets.size}."
        )
    if int(domain_offsets[0]) != 0 or int(domain_offsets[-1]) != int(n_blocks):
        raise ValueError("domain_offsets must start at 0 and end at n_blocks.")
    if np.any(domain_offsets[:-1] > domain_offsets[1:]):
        raise ValueError("domain_offsets must be monotone.")

    rank_global_numa = np.asarray(layout.rank_global_numa, dtype=np.int64)
    rank_local_in_numa = np.asarray(layout.rank_local_in_numa, dtype=np.int64)
    ranks_per_domain = np.bincount(rank_global_numa, minlength=layout.total_numa_domains)
    if rank_global_numa.size != int(layout.size) or rank_local_in_numa.size != int(layout.size):
        raise ValueError("Layout rank maps must have one entry per MPI rank.")
    if np.any(ranks_per_domain[: layout.total_numa_domains] <= 0):
        raise ValueError("Every NUMA domain must contain at least one MPI rank.")

    starts = np.empty(layout.size, dtype=np.int64)
    stops = np.empty(layout.size, dtype=np.int64)

    for r in range(layout.size):
        d = int(rank_global_numa[r])
        k = int(rank_local_in_numa[r])
        if d < 0 or d >= layout.total_numa_domains:
            raise ValueError(f"Rank {r} has invalid NUMA domain id {d}.")
        nr = int(ranks_per_domain[d])
        if k < 0 or k >= nr:
            raise ValueError(f"Rank {r} has invalid local NUMA rank {k} for domain {d}.")

        d0 = int(domain_offsets[d])
        d1 = int(domain_offsets[d + 1])
        dn = d1 - d0

        starts[r] = d0 + (dn * k) // nr
        stops[r] = d0 + (dn * (k + 1)) // nr

    if int(starts[0]) != 0 or int(stops[-1]) != int(n_blocks):
        raise RuntimeError("Rank offsets do not cover the global block range.")

    if not np.all(stops[:-1] == starts[1:]):
        bad = int(np.flatnonzero(stops[:-1] != starts[1:])[0])
        raise RuntimeError(
            "Rank ownership ranges are not contiguous in MPI rank order. "
            "This means MPI ranks are not ordered by NUMA domains. "
            f"First mismatch: rank {bad} stops at {stops[bad]}, "
            f"rank {bad + 1} starts at {starts[bad + 1]}."
        )

    offsets = np.empty(layout.size + 1, dtype=np.int64)
    offsets[:-1] = starts
    offsets[-1] = stops[-1]
    return offsets
