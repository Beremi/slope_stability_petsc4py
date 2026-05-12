from __future__ import annotations

from types import SimpleNamespace

import pytest

from slope_stability.core.run_config import load_run_case_config
from slope_stability.hpc.numa_layout import NumaMPILayout
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
    assert cfg.linear_solver.numa_domains_per_node == 8
    assert cfg.linear_solver.pmg_numa_partition_mode == "rank_metis"
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


def test_pmg_gasm_config_parses_numa_coalesced(tmp_path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        _minimal_case_text(
            """
pc_backend = "pmg_shell"
numa_domains_per_node = 4
pmg_numa_partition_mode = "domain_metis_split"
pmg_smoother_pc_type = "gasm"
pmg_smoother_gasm_grouping = "numa_coalesced"
pmg_smoother_gasm_overlap = 0
"""
        ),
        encoding="utf-8",
    )

    cfg = load_run_case_config(path)

    assert cfg.linear_solver.numa_domains_per_node == 4
    assert cfg.linear_solver.pmg_numa_partition_mode == "domain_metis_split"
    assert cfg.linear_solver.pmg_smoother_gasm_grouping == "numa_coalesced"
    assert cfg.linear_solver.pmg_smoother_gasm_total_subdomains is None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"pmg_smoother_gasm_total_subdomains": 0}, "total_subdomains must be positive"),
        ({"pmg_smoother_gasm_total_subdomains": 5}, "no larger than the MPI communicator size"),
        ({"pmg_smoother_gasm_total_subdomains": 3}, "must divide the MPI communicator size"),
        (
            {"pmg_smoother_gasm_total_subdomains": 2, "pmg_smoother_gasm_grouping": "socket"},
            "must be 'contiguous' or 'numa_coalesced'",
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


def _fake_numa_layout() -> NumaMPILayout:
    return NumaMPILayout(
        rank=0,
        size=4,
        node_rank=0,
        node_size=4,
        node_id=0,
        node_count=1,
        numa_domains_per_node=2,
        hw_numa_id=0,
        local_numa_id=0,
        global_numa_id=0,
        local_rank_in_numa=0,
        ranks_per_numa=2,
        total_numa_domains=2,
        rank_global_numa=(0, 0, 1, 1),
        rank_local_in_numa=(0, 1, 0, 1),
        rank_node_id=(0, 0, 0, 0),
    )


def test_pmg_gasm_smoother_numa_coalesced_uses_layout() -> None:
    preconditioner_options = {
        "pmg_smoother_pc_type": "gasm",
        "pmg_smoother_gasm_grouping": "numa_coalesced",
        "pmg_smoother_gasm_overlap": 0,
        "numa_layout": _fake_numa_layout(),
    }
    context = _ManualPMGShellPC(SimpleNamespace(preconditioner_options=preconditioner_options))

    config = context._gasm_smoother_config(comm_size=4)

    assert config is not None
    assert config["total_subdomains"] == 2
    assert config["ranks_per_subdomain"] == 2
    assert config["sub_ksp_type"] == "preonly"
    assert config["sub_ksp_max_it"] == 1
    assert config["sub_pc_type"] == "jacobi"


def test_pmg_gasm_smoother_numa_coalesced_rejects_missing_layout() -> None:
    context = _ManualPMGShellPC(
        SimpleNamespace(
            preconditioner_options={
                "pmg_smoother_pc_type": "gasm",
                "pmg_smoother_gasm_grouping": "numa_coalesced",
                "pmg_smoother_gasm_overlap": 0,
            }
        )
    )

    with pytest.raises(ValueError, match="requires solver.numa_layout"):
        context._gasm_smoother_config(comm_size=4)


def test_pmg_gasm_smoother_numa_coalesced_rejects_overlap() -> None:
    context = _ManualPMGShellPC(
        SimpleNamespace(
            preconditioner_options={
                "pmg_smoother_pc_type": "gasm",
                "pmg_smoother_gasm_grouping": "numa_coalesced",
                "pmg_smoother_gasm_overlap": 1,
                "numa_layout": _fake_numa_layout(),
            }
        )
    )

    with pytest.raises(ValueError, match="overlap=0"):
        context._gasm_smoother_config(comm_size=4)


def test_pmg_gasm_smoother_numa_coalesced_defaults_to_legacy_jacobi_subsolve() -> None:
    recorded: dict[str, object] = {}

    def record_option(_opts, key: str, value) -> None:
        recorded[key] = value

    class FakePC:
        def __init__(self) -> None:
            self.pc_type = None

        def setType(self, pc_type) -> None:
            self.pc_type = pc_type

    solver = SimpleNamespace(
        preconditioner_options={
            "pmg_smoother_pc_type": "gasm",
            "pmg_smoother_gasm_grouping": "numa_coalesced",
            "pmg_smoother_gasm_overlap": 0,
            "numa_layout": _fake_numa_layout(),
        },
        _set_petsc_option=record_option,
    )
    context = _ManualPMGShellPC(solver)

    context._configure_gasm_smoother_options(FakePC(), prefix="manualmg_fine_", comm_size=4)

    assert recorded["manualmg_fine_pc_gasm_total_subdomains"] == 2
    assert recorded["manualmg_fine_pc_gasm_overlap"] == 0
    assert recorded["manualmg_fine_sub_ksp_type"] == "preonly"
    assert "manualmg_fine_sub_ksp_max_it" not in recorded
    assert "manualmg_fine_sub_ksp_rtol" not in recorded
    assert "manualmg_fine_sub_ksp_atol" not in recorded
    assert recorded["manualmg_fine_sub_pc_type"] == "jacobi"
    assert "manualmg_fine_sub_sub_pc_type" not in recorded


def test_pmg_gasm_smoother_still_accepts_explicit_sub_gmres_bjacobi_ilu_options() -> None:
    recorded: dict[str, object] = {}

    def record_option(_opts, key: str, value) -> None:
        recorded[key] = value

    class FakePC:
        def __init__(self) -> None:
            self.pc_type = None

        def setType(self, pc_type) -> None:
            self.pc_type = pc_type

    solver = SimpleNamespace(
        preconditioner_options={
            "pmg_smoother_pc_type": "gasm",
            "pmg_smoother_gasm_grouping": "numa_coalesced",
            "pmg_smoother_gasm_overlap": 0,
            "pmg_smoother_gasm_sub_ksp_type": "gmres",
            "pmg_smoother_gasm_sub_ksp_max_it": 4,
            "pmg_smoother_gasm_sub_pc_type": "bjacobi",
            "numa_layout": _fake_numa_layout(),
        },
        _set_petsc_option=record_option,
    )
    context = _ManualPMGShellPC(solver)

    context._configure_gasm_smoother_options(FakePC(), prefix="manualmg_fine_", comm_size=4)

    assert recorded["manualmg_fine_sub_ksp_type"] == "gmres"
    assert recorded["manualmg_fine_sub_ksp_max_it"] == 4
    assert recorded["manualmg_fine_sub_ksp_rtol"] == 0.0
    assert recorded["manualmg_fine_sub_ksp_atol"] == 0.0
    assert recorded["manualmg_fine_sub_pc_type"] == "bjacobi"
    assert recorded["manualmg_fine_sub_sub_pc_type"] == "ilu"
    assert recorded["manualmg_fine_sub_sub_pc_factor_levels"] == 1
