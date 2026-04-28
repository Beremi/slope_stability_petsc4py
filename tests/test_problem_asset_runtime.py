from __future__ import annotations

from pathlib import Path

import numpy as np

from slope_stability.core.run_config import load_run_case_config
from slope_stability.mesh import (
    franz_dam_pressure_boundary,
    generate_homogeneous_slope_mesh_2d,
    generate_sloan2013_mesh_2d,
    load_mesh_franz_dam_2d,
    load_mesh_kozinec_2d,
    load_mesh_luzec_2d,
    luzec_pressure_boundary,
)
from slope_stability.postprocess.case_mesh import rebuild_case_mesh
from slope_stability.problem_asset_runtime import (
    build_mesh_for_resolved_asset,
    build_seepage_boundary_for_resolved_asset,
    resolve_problem_asset,
    resolve_problem_asset_from_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _sorted_node_order(coord: np.ndarray) -> np.ndarray:
    keys = tuple(np.asarray(coord[axis, :], dtype=np.float64) for axis in range(coord.shape[0] - 1, -1, -1))
    return np.asarray(np.lexsort(keys), dtype=np.int64)


def _node_permutation_by_coordinates(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    lookup: dict[tuple[float, ...], int] = {}
    for idx in range(target.shape[1]):
        key = tuple(np.round(target[:, idx], 8))
        if key in lookup:
            raise ValueError("Target coordinates are not unique enough for permutation recovery.")
        lookup[key] = idx
    perm = np.empty(source.shape[1], dtype=np.int64)
    for idx in range(source.shape[1]):
        key = tuple(np.round(source[:, idx], 8))
        perm[idx] = lookup[key]
    return perm


def _normalized_columns(connectivity: np.ndarray) -> list[tuple[int, ...]]:
    conn = np.asarray(connectivity, dtype=np.int64)
    return sorted(tuple(sorted(int(v) for v in conn[:, idx])) for idx in range(conn.shape[1]))


def _legacy_sloan_boundary(coord: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1 = 15.0
    x3 = 20.0
    y11 = 6.75
    y12 = 0.5
    y13 = 0.75
    y21 = 1.0
    y22 = 9.25
    y23 = 2.0
    y1 = y11 + y12 + y13
    y2 = y21 + y22 + y23
    beta = np.deg2rad(26.6)
    x2 = y2 / np.tan(beta)
    grho = 9.81

    q_w = np.ones(coord.shape[1], dtype=bool)
    q_w[coord[0, :] <= 0.001] = False
    q_w[coord[0, :] >= x1 + x2 + x3 - 0.001] = False
    q_w[coord[1, :] >= y1 + y2 - 0.001] = False
    q_w[(coord[1, :] >= y1 - 0.001) & (coord[0, :] >= x1 + x2 - 0.001)] = False
    q_w[(coord[1, :] >= y1 - 0.001) & (coord[1, :] >= -(y2 / x2) * coord[0, :] + y1 + y2 * (1.0 + x1 / x2) - 0.001)] = False

    x_bar = x1 + (1.0 - y21 / y2) * x2
    pw_d = np.zeros(coord.shape[1], dtype=np.float64)
    part1 = (coord[0, :] < x_bar) & (coord[1, :] <= -(y22 / x_bar) * coord[0, :] + y1 + y21 + y22)
    part2 = coord[0, :] >= x_bar
    pw_d[part1] = grho * ((y22 / x_bar) * (x_bar - coord[0, part1]) + y1 + y21 - coord[1, part1])
    pw_d[part2] = grho * (y1 + y21 - coord[1, part2])
    return q_w, pw_d


def test_configs_resolve_assets_across_current_benchmarks() -> None:
    expected = {
        "benchmarks/run_2D_homo_SSR_capture/case.toml": ("2d_homo_slope", "h1.0.msh", "default"),
        "benchmarks/run_2D_sloan2013_seepage_capture/case.toml": ("2d_sloan2013", "default.msh", "default"),
        "benchmarks/slope_stability_2D_Kozinec_SSR/case.toml": ("2d_kozinec", "default.msh", "default"),
        "benchmarks/slope_stability_2D_Luzec_SSR/case.toml": ("2d_luzec", "default.msh", "default"),
        "benchmarks/slope_stability_2D_Franz_dam_SSR/case.toml": ("2d_franz_dam", "default.msh", "default"),
        "benchmarks/run_3D_hetero_SSR_capture/case.toml": ("3d_hetero_slope", "adaptive_family_a_l1.msh", "default"),
        "benchmarks/run_3D_hetero_seepage_capture/case.toml": ("3d_hetero_seepage", "concave_family_b.msh", "default"),
        "benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml": ("3d_hetero_seepage_transition", "transition_default.msh", "fixed_base"),
        "benchmarks/SIOPT_SSR/case.toml": ("3d_siopt", "reference_l0.msh", "fixed_base"),
    }

    for rel_path, (asset_name, variant_name, profile) in expected.items():
        cfg = load_run_case_config(ROOT / rel_path)
        resolved = resolve_problem_asset_from_config(cfg)
        assert resolved.asset_name == asset_name
        assert resolved.variant_name == variant_name
        assert resolved.resolved_variant.profile == profile


def test_homogeneous_canonical_asset_matches_generated_reference() -> None:
    resolved = resolve_problem_asset(asset_name="2d_homo_slope", mesh_variant="h1.0.msh")
    built = build_mesh_for_resolved_asset(resolved, elem_type="P1")
    legacy = generate_homogeneous_slope_mesh_2d(
        elem_type="P1",
        h=1.0,
        x1=15.0,
        x2=10.0,
        x3=15.0,
        y1=10.0,
        y2=10.0,
    )
    perm = _node_permutation_by_coordinates(built.coord, legacy.coord)
    assert _normalized_columns(perm[built.elem]) == _normalized_columns(legacy.elem)
    assert _normalized_columns(perm[built.surf]) == _normalized_columns(legacy.surf)
    assert np.array_equal(built.q_mask, legacy.q_mask[:, perm])


def test_textmesh_assets_match_legacy_q_masks() -> None:
    cases = [
        ("2d_kozinec", load_mesh_kozinec_2d, ROOT / "meshes" / "2d_kozinec" / "legacy" / "source"),
        ("2d_luzec", load_mesh_luzec_2d, ROOT / "meshes" / "2d_luzec" / "legacy" / "source"),
        ("2d_franz_dam", load_mesh_franz_dam_2d, ROOT / "meshes" / "2d_franz_dam" / "legacy" / "source"),
    ]
    for asset_name, loader, mesh_dir in cases:
        resolved = resolve_problem_asset(asset_name=asset_name, mesh_variant="default.msh")
        built = build_mesh_for_resolved_asset(resolved, elem_type="P2")
        legacy = loader("P2", mesh_dir)
        perm = _node_permutation_by_coordinates(built.coord, legacy.coord)
        assert _normalized_columns(perm[built.elem]) == _normalized_columns(legacy.elem)
        assert _normalized_columns(perm[built.surf]) == _normalized_columns(legacy.surf)
        legacy_q = legacy.q_mask[:, perm]
        if asset_name == "2d_luzec":
            diff = np.asarray(built.q_mask != legacy_q, dtype=bool)
            assert int(np.count_nonzero(diff)) == 7
            assert int(np.count_nonzero(diff[1, :])) == 0
            assert int(np.count_nonzero(diff[0, :])) == 7
        else:
            assert np.array_equal(built.q_mask, legacy_q)


def test_sloan_generated_asset_matches_legacy_mesh_and_boundary() -> None:
    resolved = resolve_problem_asset(asset_name="2d_sloan2013", mesh_variant="default.msh")
    built = build_mesh_for_resolved_asset(resolved, elem_type="P1")
    legacy = generate_sloan2013_mesh_2d(elem_type="P1")
    perm = _node_permutation_by_coordinates(built.coord, legacy.coord)
    assert _normalized_columns(perm[built.elem]) == _normalized_columns(legacy.elem)
    assert _normalized_columns(perm[built.surf]) == _normalized_columns(legacy.surf)

    q_w, pw_d = build_seepage_boundary_for_resolved_asset(
        resolved,
        built.coord,
        built.surf,
        built.boundary_labels,
        grho=9.81,
    )
    q_w_ref, pw_d_ref = _legacy_sloan_boundary(built.coord)
    assert np.array_equal(q_w, q_w_ref)
    assert np.allclose(pw_d, pw_d_ref)


def test_textmesh_seepage_boundaries_match_legacy_helpers() -> None:
    luzec = resolve_problem_asset(asset_name="2d_luzec", mesh_variant="default.msh")
    luzec_mesh = build_mesh_for_resolved_asset(luzec, elem_type="P2")
    q_w_l, pw_d_l = build_seepage_boundary_for_resolved_asset(
        luzec,
        luzec_mesh.coord,
        luzec_mesh.surf,
        luzec_mesh.boundary_labels,
        grho=9.81,
    )
    q_w_l_ref, pw_d_l_ref = luzec_pressure_boundary(luzec_mesh.coord, luzec_mesh.surf, 9.81)
    assert np.array_equal(q_w_l, q_w_l_ref)
    assert np.allclose(pw_d_l, pw_d_l_ref)

    franz = resolve_problem_asset(asset_name="2d_franz_dam", mesh_variant="default.msh")
    franz_mesh = build_mesh_for_resolved_asset(franz, elem_type="P2")
    q_w_f, pw_d_f = build_seepage_boundary_for_resolved_asset(
        franz,
        franz_mesh.coord,
        franz_mesh.surf,
        franz_mesh.boundary_labels,
        grho=9.81,
    )
    q_w_f_ref, pw_d_f_ref = franz_dam_pressure_boundary(franz_mesh.coord, franz_mesh.surf, 9.81)
    assert np.array_equal(q_w_f, q_w_f_ref)
    assert np.allclose(pw_d_f, pw_d_f_ref)


def test_rebuild_case_mesh_uses_asset_runtime() -> None:
    for rel_path in (
        "benchmarks/run_2D_homo_SSR_capture/case.toml",
        "benchmarks/run_2D_sloan2013_seepage_capture/case.toml",
        "benchmarks/slope_stability_2D_Kozinec_SSR/case.toml",
        "benchmarks/slope_stability_2D_Luzec_SSR/case.toml",
        "benchmarks/run_3D_hetero_SSR_capture/case.toml",
    ):
        cfg = load_run_case_config(ROOT / rel_path)
        mesh = rebuild_case_mesh(cfg, mpi_size=1)
        assert mesh.coord.shape[1] > 0
        assert mesh.elem.shape[1] > 0
        assert mesh.material_id.shape[0] == mesh.elem.shape[1]
