from __future__ import annotations

import numpy as np
import pytest

from slope_stability.hpc.numa_layout import NumaMPILayout, split_domain_offsets_to_rank_offsets


def fake_layout(*, nodes: int = 2, numa_per_node: int = 8, ranks_per_numa: int = 16) -> NumaMPILayout:
    size = int(nodes) * int(numa_per_node) * int(ranks_per_numa)
    rank_global_numa: list[int] = []
    rank_local_in_numa: list[int] = []
    rank_node_id: list[int] = []
    for node in range(int(nodes)):
        for numa in range(int(numa_per_node)):
            gid = node * int(numa_per_node) + numa
            for k in range(int(ranks_per_numa)):
                rank_global_numa.append(gid)
                rank_local_in_numa.append(k)
                rank_node_id.append(node)

    return NumaMPILayout(
        rank=0,
        size=size,
        node_rank=0,
        node_size=int(numa_per_node) * int(ranks_per_numa),
        node_id=0,
        node_count=int(nodes),
        numa_domains_per_node=int(numa_per_node),
        hw_numa_id=0,
        local_numa_id=0,
        global_numa_id=0,
        local_rank_in_numa=0,
        ranks_per_numa=int(ranks_per_numa),
        total_numa_domains=int(nodes) * int(numa_per_node),
        rank_global_numa=tuple(rank_global_numa),
        rank_local_in_numa=tuple(rank_local_in_numa),
        rank_node_id=tuple(rank_node_id),
    )


def test_split_domain_offsets_to_rank_offsets() -> None:
    layout = fake_layout(nodes=2, numa_per_node=8, ranks_per_numa=16)
    n_blocks = 16000
    domain_offsets = np.linspace(0, n_blocks, layout.total_numa_domains + 1, dtype=np.int64)

    rank_offsets = split_domain_offsets_to_rank_offsets(
        n_blocks=n_blocks,
        domain_offsets=domain_offsets,
        layout=layout,
    )

    assert rank_offsets.size == layout.size + 1
    assert rank_offsets[0] == 0
    assert rank_offsets[-1] == n_blocks
    assert np.all(rank_offsets[:-1] <= rank_offsets[1:])

    for r in range(layout.size):
        d = layout.rank_global_numa[r]
        assert domain_offsets[d] <= rank_offsets[r]
        assert rank_offsets[r + 1] <= domain_offsets[d + 1]


def test_split_domain_offsets_rejects_wrong_size() -> None:
    layout = fake_layout(nodes=1, numa_per_node=2, ranks_per_numa=2)

    with pytest.raises(ValueError, match="total_numa_domains"):
        split_domain_offsets_to_rank_offsets(
            n_blocks=12,
            domain_offsets=np.asarray([0, 6, 9, 12], dtype=np.int64),
            layout=layout,
        )


def test_split_domain_offsets_rejects_nonmonotone_offsets() -> None:
    layout = fake_layout(nodes=1, numa_per_node=2, ranks_per_numa=2)

    with pytest.raises(ValueError, match="monotone"):
        split_domain_offsets_to_rank_offsets(
            n_blocks=12,
            domain_offsets=np.asarray([0, 13, 12], dtype=np.int64),
            layout=layout,
        )


def test_split_domain_offsets_rejects_noncontiguous_rank_domains() -> None:
    layout = fake_layout(nodes=1, numa_per_node=2, ranks_per_numa=2)
    bad_layout = NumaMPILayout(
        **{
            **layout.__dict__,
            "rank_global_numa": (0, 1, 0, 1),
            "rank_local_in_numa": (0, 0, 1, 1),
        }
    )

    with pytest.raises(RuntimeError, match="not contiguous"):
        split_domain_offsets_to_rank_offsets(
            n_blocks=12,
            domain_offsets=np.asarray([0, 6, 12], dtype=np.int64),
            layout=bad_layout,
        )
