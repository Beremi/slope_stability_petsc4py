from __future__ import annotations

import numpy as np
import pytest

from slope_stability.mesh import reorder as reorder_module
from slope_stability.mesh.reorder import reorder_mesh_nodes
from slope_stability.utils import owned_block_range


class _FakeComm:
    def __init__(self, size: int, rank: int) -> None:
        self._size = int(size)
        self._rank = int(rank)

    def getSize(self) -> int:
        return self._size

    def getRank(self) -> int:
        return self._rank


def test_owned_block_range_can_use_partition_offsets() -> None:
    offsets = np.asarray([0, 3, 8, 10], dtype=np.int64)

    assert owned_block_range(10, 3, _FakeComm(3, 0), partition_offsets=offsets) == (0, 9)
    assert owned_block_range(10, 3, _FakeComm(3, 1), partition_offsets=offsets) == (9, 24)
    assert owned_block_range(10, 3, _FakeComm(3, 2), partition_offsets=offsets) == (24, 30)


def test_owned_block_range_rejects_bad_partition_offsets() -> None:
    with pytest.raises(ValueError, match="size comm_size"):
        owned_block_range(10, 3, _FakeComm(3, 0), partition_offsets=np.asarray([0, 5, 10]))
    with pytest.raises(ValueError, match="start at 0"):
        owned_block_range(10, 3, _FakeComm(3, 0), partition_offsets=np.asarray([1, 5, 8, 10]))
    with pytest.raises(ValueError, match="monotone"):
        owned_block_range(10, 3, _FakeComm(3, 0), partition_offsets=np.asarray([0, 5, 4, 10]))


def test_block_metis_reorder_preserves_partition_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePyMetis:
        @staticmethod
        def CSRAdjacency(indptr, indices):
            return (indptr, indices)

        @staticmethod
        def part_graph(*args, **kwargs):
            return 0, [1, 1, 0, 0]

    monkeypatch.setattr(reorder_module, "pymetis", _FakePyMetis)

    coord = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    elem = np.asarray([[0], [1], [2], [3]], dtype=np.int64)
    surf = np.asarray([[0], [1], [2]], dtype=np.int64)
    q_mask = np.ones((3, 4), dtype=bool)

    reordered = reorder_mesh_nodes(coord, elem, surf, q_mask, strategy="block_metis", n_parts=2)

    assert reordered.partition_offsets is not None
    np.testing.assert_array_equal(reordered.partition_offsets, np.asarray([0, 2, 4], dtype=np.int64))
    np.testing.assert_array_equal(reordered.permutation, np.asarray([2, 3, 0, 1], dtype=np.int64))
