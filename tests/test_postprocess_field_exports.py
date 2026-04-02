from __future__ import annotations

import numpy as np

from slope_stability.core.simplex_lagrange import tetra_reference_nodes
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


def test_build_field_exports_adds_p4_point_deviatoric_strain() -> None:
    coord = tetra_reference_nodes(4)
    elem = np.arange(35, dtype=np.int64).reshape(35, 1)
    arrays = {
        "U": np.zeros((3, 35), dtype=np.float64),
    }

    point_data, cell_data = build_field_exports(
        arrays,
        n_cells=1,
        coord=coord,
        elem=elem,
        elem_type="P4",
        dim=3,
    )

    assert "deviatoric_strain" in point_data
    assert point_data["deviatoric_strain"].shape == (35,)
    np.testing.assert_allclose(point_data["deviatoric_strain"], 0.0)
    assert "deviatoric_strain" not in cell_data


def test_build_field_exports_adds_cell_deviatoric_fallback_for_non_p4() -> None:
    coord = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elem = np.arange(4, dtype=np.int64).reshape(4, 1)
    arrays = {
        "U": np.zeros((3, 4), dtype=np.float64),
    }

    point_data, cell_data = build_field_exports(
        arrays,
        n_cells=1,
        coord=coord,
        elem=elem,
        elem_type="P1",
        dim=3,
    )

    assert "deviatoric_strain" not in point_data
    assert "deviatoric_strain" in cell_data
    np.testing.assert_allclose(cell_data["deviatoric_strain"], np.array([0.0]))
