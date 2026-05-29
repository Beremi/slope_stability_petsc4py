from __future__ import annotations

import numpy as np

from petsc_ssr.mesh import canonical_node_ordering_strategy, node_ordering_requires_partitions, reorder_mesh_nodes


def test_native_dmplex_ordering_is_not_legacy_partitioned_reorder() -> None:
    assert canonical_node_ordering_strategy("native_dmplex") == "original"
    assert not node_ordering_requires_partitions("native_dmplex")
    assert node_ordering_requires_partitions("block_metis")


def test_native_dmplex_ordering_preserves_asset_node_order_for_python_exports() -> None:
    coord = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elem = np.asarray([[0], [1], [2]], dtype=np.int64)
    surf = np.asarray([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    q_mask = np.asarray([[True, False, True], [True, True, False]], dtype=bool)

    reordered = reorder_mesh_nodes(coord, elem, surf, q_mask, strategy="native_dmplex", n_parts=8)

    np.testing.assert_array_equal(reordered.permutation, np.arange(coord.shape[1], dtype=np.int64))
    np.testing.assert_array_equal(reordered.coord, coord)
    np.testing.assert_array_equal(reordered.elem, elem)
    np.testing.assert_array_equal(reordered.surf, surf)
    np.testing.assert_array_equal(reordered.q_mask, q_mask)
