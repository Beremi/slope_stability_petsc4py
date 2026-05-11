from __future__ import annotations

from types import SimpleNamespace

import pytest

from slope_stability.core.run_config import load_run_case_config
from slope_stability.linear.solver import _ManualPMGShellPC


def _minimal_case_text(linear_solver: str = "") -> str:
    return f"""
[problem]
name = "pmg_gasm_config_test"
asset = "3d_hetero_slope"
mesh_variant = "adaptive_family_a_l1.msh"
analysis = "ssr"
elem_type = "P4"

[linear_solver]
{linear_solver}
"""


def test_pmg_gasm_config_defaults_preserve_current_behavior(tmp_path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(_minimal_case_text(), encoding="utf-8")

    cfg = load_run_case_config(path)

    assert cfg.linear_solver.pmg_smoother_pc_type is None
    assert cfg.linear_solver.pmg_smoother_gasm_total_subdomains is None
    assert cfg.linear_solver.pmg_smoother_gasm_grouping == "contiguous"
    assert cfg.linear_solver.pmg_smoother_gasm_overlap == 1
    assert cfg.linear_solver.pmg_smoother_gasm_sub_ksp_type == "preonly"
    assert cfg.linear_solver.pmg_smoother_gasm_sub_ksp_max_it == 1


def test_pmg_gasm_config_parses_explicit_fields(tmp_path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        _minimal_case_text(
            """
pc_backend = "pmg_shell"
pmg_smoother_pc_type = "gasm"
pmg_smoother_gasm_total_subdomains = 2
pmg_smoother_gasm_grouping = "contiguous"
pmg_smoother_gasm_overlap = 1
pmg_smoother_gasm_type = "restrict"
pmg_smoother_gasm_sub_ksp_type = "richardson"
pmg_smoother_gasm_sub_ksp_max_it = 3
pmg_smoother_gasm_sub_pc_type = "jacobi"
pmg_smoother_gasm_view_subdomains = false
"""
        ),
        encoding="utf-8",
    )

    cfg = load_run_case_config(path)

    assert cfg.linear_solver.pmg_smoother_pc_type == "gasm"
    assert cfg.linear_solver.pmg_smoother_gasm_total_subdomains == 2
    assert cfg.linear_solver.pmg_smoother_gasm_sub_ksp_max_it == 3
    assert cfg.linear_solver.pmg_smoother_gasm_view_subdomains is False


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"pmg_smoother_gasm_total_subdomains": 0}, "must be positive"),
        ({"pmg_smoother_gasm_total_subdomains": 5}, "no larger than the MPI communicator size"),
        ({"pmg_smoother_gasm_total_subdomains": 3}, "must divide the MPI communicator size"),
        (
            {"pmg_smoother_gasm_total_subdomains": 2, "pmg_smoother_gasm_grouping": "socket"},
            "supports only 'contiguous'",
        ),
    ],
)
def test_pmg_gasm_smoother_validation_errors(options, message: str) -> None:
    preconditioner_options = {
        "pmg_smoother_pc_type": "gasm",
        "pmg_smoother_gasm_total_subdomains": 2,
        "pmg_smoother_gasm_grouping": "contiguous",
        "pmg_smoother_gasm_overlap": 1,
        "pmg_smoother_gasm_type": "restrict",
        "pmg_smoother_gasm_sub_ksp_type": "richardson",
        "pmg_smoother_gasm_sub_ksp_max_it": 3,
        "pmg_smoother_gasm_sub_pc_type": "jacobi",
        **options,
    }
    context = _ManualPMGShellPC(SimpleNamespace(preconditioner_options=preconditioner_options))

    with pytest.raises(ValueError, match=message):
        context._gasm_smoother_config(comm_size=4)
