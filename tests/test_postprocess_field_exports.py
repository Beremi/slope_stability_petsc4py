from __future__ import annotations

import numpy as np

from slope_stability.postprocess import build_field_exports


def test_build_field_exports_prefers_reordered_pore_pressure() -> None:
    arrays = {
        "U": np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]], dtype=np.float64),
        "seepage_pw": np.array([10.0, 20.0, 30.0], dtype=np.float64),
        "seepage_pw_reordered": np.array([20.0, 10.0, 30.0], dtype=np.float64),
        "seepage_grad_p": np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]], dtype=np.float64),
        "seepage_mater_sat": np.array([0.0, 1.0], dtype=np.float64),
    }

    point_data, cell_data = build_field_exports(arrays, n_cells=2)

    np.testing.assert_allclose(point_data["displacement"], np.array([[0.0, 0.0, 0.0], [1.0, 3.0, 0.0], [2.0, 4.0, 0.0]]))
    np.testing.assert_allclose(point_data["pore_pressure"], np.array([20.0, 10.0, 30.0]))
    np.testing.assert_allclose(cell_data["pressure_gradient"], np.array([[1.0, 3.0, 0.0], [2.0, 4.0, 0.0]]))
    np.testing.assert_allclose(cell_data["saturation"], np.array([0.0, 1.0]))
