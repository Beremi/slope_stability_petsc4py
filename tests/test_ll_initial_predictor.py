from __future__ import annotations

import numpy as np

from slope_stability.utils import full_field_from_free_values, q_to_free_indices


def test_full_field_from_free_values_preserves_column_major_dof_layout() -> None:
    q_mask = np.array(
        [
            [True, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    free_idx = q_to_free_indices(q_mask)
    free_values = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)

    field = full_field_from_free_values(free_values, free_idx, q_mask.shape)

    expected = np.array(
        [
            [10.0, 0.0, 30.0],
            [0.0, 20.0, 40.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(field, expected)
