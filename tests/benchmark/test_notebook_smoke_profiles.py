from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmarks" / "tools"
CASE_ROOT = ROOT / "benchmarks" / "cases"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import notebook_support as nb  # noqa: E402


def test_modern_notebook_smoke_profile_keeps_small_cases_mathematical() -> None:
    case_toml = CASE_ROOT / "2d-homogeneous-ssr" / "case.toml"
    sections = nb.load_case_sections(case_toml)

    smoke = nb._profile_sections(case_toml, sections, "smoke")

    assert smoke["mesh"]["element"] == "P2"
    assert smoke["linear"]["profile"] == "pmg-deflated-baseline"
    assert smoke["continuation"]["step_max"] == 2
    assert smoke["output"]["preset"] == "smoke"


def test_modern_notebook_smoke_profile_downshifts_known_heavy_cases() -> None:
    case_toml = CASE_ROOT / "2d-kozinec-ll" / "case.toml"
    sections = nb.load_case_sections(case_toml)

    smoke = nb._profile_sections(case_toml, sections, "smoke")

    assert smoke["mesh"]["element"] == "P1"
    assert smoke["linear"]["profile"] == "gamg-p1-baseline"
    assert smoke["newton"]["profile"] == "limit-load-regularized-it100"
    assert smoke["continuation"]["step_max"] == 2
    assert smoke["output"]["preset"] == "smoke"


def test_coupled_seepage_smoke_profile_uses_explicit_coordinate_bc_debug_override() -> None:
    case_toml = CASE_ROOT / "3d-heterogeneous-seepage-ssr-comsol" / "case.toml"

    assert nb._profile_solver_args(case_toml, "smoke") == ["--write-coordinate-bc-table"]
    assert nb._profile_solver_args(case_toml, "full") == []
