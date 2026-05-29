from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from petsc_ssr.config import load_run_case_config
from petsc_ssr.config.profiles import (
    load_continuation_profile,
    load_newton_profile,
    load_seepage_profile,
    load_solver_profile,
    native_linear_algorithm_selector,
    pc_variant_from_backend,
)


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "benchmarks" / "cases"
CONTINUATION_PROFILE_ROOT = ROOT / "configs" / "continuation_profiles"
NEWTON_PROFILE_ROOT = ROOT / "configs" / "newton_profiles"
SOLVER_PROFILE_ROOT = ROOT / "configs" / "solver_profiles"


def test_profile_resolves_adaptive_pmg_ranks_for_local_32() -> None:
    profile = load_solver_profile("pmg-deflated-baseline", world_size=32)

    assert profile.name == "pmg-deflated-baseline"
    assert profile.data["algorithm"] == "ksp_deflated"
    assert profile.data["ksp_type"] == "fgmres"
    assert profile.data["norm_type"] == "unpreconditioned"
    assert profile.data["pmg_shell_p2_active_ranks"] == 32
    assert profile.data["pmg_shell_p1_active_ranks"] == 16
    assert profile.data["pmg_shell_p2_rank_policy"] == "cap"
    assert profile.data["pmg_shell_p1_rank_policy"] == "fraction"
    assert profile.data["pmg_apply_backend"] == "shell_vcycle"
    assert profile.data["pmg_coarse_pc_type"] == "gamg"
    assert profile.data["pmg_coarse_redundant_group_size"] == 0
    assert profile.data["pmg_coarse_telescope_ksp_max_it"] == 5
    assert profile.data["pmg_p2_telescope_active_ranks"] == 0
    assert profile.data["pmg_smoother_max_it"] == 2
    assert profile.data["deflation"] is True


def test_solver_profile_legacy_name_alias_resolves_to_canonical_profile() -> None:
    profile = load_solver_profile("baseline-pmg-deflated", world_size=8)

    assert profile.name == "pmg-deflated-baseline"
    assert profile.data["profile"] == "pmg-deflated-baseline"
    assert profile.data["profile_alias"] == "baseline-pmg-deflated"
    assert profile.data["pmg_shell_p2_active_ranks"] == 8
    assert profile.data["pmg_shell_p1_active_ranks"] == 4


def test_debug_and_gamg_solver_profiles_resolve_runtime_options() -> None:
    gamg = load_solver_profile("gamg-p1-baseline", world_size=4)
    direct = load_solver_profile("direct-debug", world_size=1)

    assert gamg.data["profile"] == "gamg-p1-baseline"
    assert gamg.data["algorithm"] == "fgmres"
    assert gamg.data["pc_backend"] == "gamg"
    assert gamg.data["ksp_type"] == "fgmres"
    assert gamg.data["deflation"] is False
    assert direct.data["profile"] == "direct-debug"
    assert direct.data["algorithm"] == "direct_debug"
    assert direct.data["pc_backend"] == "none"
    assert direct.data["ksp_type"] == "preonly"
    assert direct.data["deflation"] is False
    assert direct.data["petsc_opt"] == ["-pc_type", "lu"]


def test_solver_profile_rejects_legacy_linear_solver_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from petsc_ssr.config import profiles as profile_module

    (tmp_path / "legacy.toml").write_text(
        """
description = "legacy"

[linear_solver]
solver_type = "PETSC_DMPLEX_C_FGMRES"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(profile_module, "SOLVER_PROFILE_DIR", tmp_path)

    with pytest.raises(ValueError, match="linear_solver"):
        profile_module.load_solver_profile("legacy", world_size=4)


def test_profiles_reject_unsupported_algorithm_selectors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from petsc_ssr.config import profiles as profile_module

    solver_root = tmp_path / "solver"
    continuation_root = tmp_path / "continuation"
    newton_root = tmp_path / "newton"
    solver_root.mkdir()
    continuation_root.mkdir()
    newton_root.mkdir()
    (solver_root / "bad-solver.toml").write_text(
        """
description = "bad"

[linear]
algorithm = "python_array_solver"
""",
        encoding="utf-8",
    )
    (continuation_root / "bad-continuation.toml").write_text(
        """
description = "bad"

[continuation]
algorithm = "arc-length"
method = "indirect"
""",
        encoding="utf-8",
    )
    (newton_root / "bad-newton.toml").write_text(
        """
description = "bad"

[newton]
algorithm = "trust-region"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(profile_module, "SOLVER_PROFILE_DIR", solver_root)
    monkeypatch.setattr(profile_module, "CONTINUATION_PROFILE_DIR", continuation_root)
    monkeypatch.setattr(profile_module, "NEWTON_PROFILE_DIR", newton_root)

    with pytest.raises(ValueError, match=r"\[linear\]\.algorithm"):
        profile_module.load_solver_profile("bad-solver", world_size=4)
    with pytest.raises(ValueError, match="Continuation profile .* algorithm"):
        profile_module.load_continuation_profile("bad-continuation")
    with pytest.raises(ValueError, match="Newton profile .* algorithm"):
        profile_module.load_newton_profile("bad-newton")


def test_profile_pc_backend_resolves_concrete_pc_variant() -> None:
    p2 = pc_variant_from_backend("pmg_shell", element_degree=2)
    p1 = pc_variant_from_backend("pmg_shell", element_degree=1)

    assert p2.variant == "pmg"
    assert p2.requested_variant == "pmg"
    assert p2.fallback_reason is None
    assert p1.variant == "gamg"
    assert p1.requested_variant == "pmg"
    assert p1.fallback_reason == "p1_has_no_p_hierarchy"


def test_profile_algorithm_resolves_to_native_linear_selector() -> None:
    assert native_linear_algorithm_selector("ksp_deflated", pc_variant="pmg", deflation=True) == "pmg-deflated"
    assert native_linear_algorithm_selector("fgmres", pc_variant="pmg", deflation=False) == "pmg"
    assert native_linear_algorithm_selector("ksp_deflated", pc_variant="gamg", deflation=True) == "gamg"
    assert native_linear_algorithm_selector("direct_debug", pc_variant="none", deflation=False) == "debug-direct"


def test_continuation_and_newton_profiles_resolve_named_policy() -> None:
    continuation = load_continuation_profile("indirect-classic")
    direct_continuation = load_continuation_profile("direct-limit-load")
    newton = load_newton_profile("indirect-regularized")
    capped_newton = load_newton_profile("indirect-regularized-it50")
    dlambda_newton = load_newton_profile("indirect-regularized-dlambda-stop")
    limit_newton = load_newton_profile("limit-load-regularized")
    capped_limit_newton = load_newton_profile("limit-load-regularized-it100")
    seepage = load_seepage_profile("sloan2013-steady")

    assert continuation.data["profile"] == "indirect-classic"
    assert continuation.data["algorithm"] == "indirect"
    assert continuation.data["method"] == "indirect"
    assert continuation.data["predictor"] == "secant"
    assert continuation.data["omega_step_controller"] == "classic"
    assert continuation.data["init_newton_stopping_criterion"] == "relative_correction"
    assert newton.data["profile"] == "indirect-regularized"
    assert newton.data["algorithm"] == "indirect-ssr"
    assert newton.data["it_max"] == 200
    assert newton.data["line_search"] == "alg5"
    assert capped_newton.data["profile"] == "indirect-regularized-it50"
    assert capped_newton.data["algorithm"] == "indirect-ssr"
    assert capped_newton.data["it_max"] == 50
    assert dlambda_newton.data["profile"] == "indirect-regularized-dlambda-stop"
    assert dlambda_newton.data["stopping_criterion"] == "absolute_delta_lambda"
    assert dlambda_newton.data["stopping_tol"] == 0.0001
    assert direct_continuation.data["profile"] == "direct-limit-load"
    assert direct_continuation.data["algorithm"] == "direct"
    assert direct_continuation.data["method"] == "direct"
    assert limit_newton.data["profile"] == "limit-load-regularized"
    assert limit_newton.data["algorithm"] == "fixed-load"
    assert capped_limit_newton.data["profile"] == "limit-load-regularized-it100"
    assert capped_limit_newton.data["algorithm"] == "fixed-load"
    assert capped_limit_newton.data["it_max"] == 100
    assert seepage.data["profile"] == "sloan2013-steady"
    assert seepage.data["linear_tolerance"] == 1e-10
    assert seepage.data["linear_max_iter"] == 300
    assert seepage.data["nonlinear_max_iter"] == 100


def test_committed_continuation_profiles_do_not_expose_legacy_controller_name() -> None:
    offenders = {
        path.name: tomllib.loads(path.read_text(encoding="utf-8")).get("continuation", {}).get("omega_step_controller")
        for path in CONTINUATION_PROFILE_ROOT.glob("*.toml")
        if tomllib.loads(path.read_text(encoding="utf-8")).get("continuation", {}).get("omega_step_controller") == "legacy"
    }

    assert offenders == {}


def test_committed_profiles_declare_algorithm_selectors() -> None:
    for path in sorted(CONTINUATION_PROFILE_ROOT.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))["continuation"]
        assert data.get("algorithm") in {"indirect", "direct"}, path
    for path in sorted(NEWTON_PROFILE_ROOT.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))["newton"]
        assert data.get("algorithm") in {"indirect-ssr", "fixed-load"}, path
    for path in sorted(SOLVER_PROFILE_ROOT.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))["linear"]
        assert data.get("algorithm") in {"ksp_deflated", "fgmres", "direct_debug"}, path


def test_committed_p4_case_uses_profile_resolved_rank_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "32")

    cfg = load_run_case_config(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml")

    assert cfg.execution.mechanics_backend == "petsc_ssr_full_c"
    assert cfg.execution.node_ordering == "native_dmplex"
    assert cfg.linear_solver.profile == "pmg-deflated-baseline"
    assert cfg.linear_solver.algorithm == "ksp_deflated"
    assert cfg.continuation.profile == "indirect-classic"
    assert cfg.continuation.algorithm == "indirect"
    assert cfg.continuation.method == "indirect"
    assert cfg.newton.profile == "indirect-regularized-dlambda-stop"
    assert cfg.newton.algorithm == "indirect-ssr"
    assert cfg.linear_solver.pmg_shell_p2_active_ranks == 32
    assert cfg.linear_solver.pmg_shell_p1_active_ranks == 16
    assert cfg.linear_solver.pmg_apply_backend == "shell_vcycle"
    assert cfg.linear_solver.pmg_coarse_pc_type == "gamg"
    assert cfg.linear_solver.pmg_coarse_telescope_ksp_max_it == 5
    assert cfg.linear_solver.pmg_smoother_max_it == 2


def test_limit_load_defaults_resolve_direct_profiles(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "ll-defaults"
tags = ["regression"]

[mesh]
asset = "2d_homo_slope"
variant = "h0.5"
element = "P2"

[physics.mechanics]
model = "mohr_coulomb_limit_load"
davis = "B"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    cfg = load_run_case_config(path)

    assert cfg.problem.analysis == "ll"
    assert cfg.continuation.profile == "direct-limit-load"
    assert cfg.continuation.algorithm == "direct"
    assert cfg.continuation.method == "direct"
    assert cfg.newton.profile == "limit-load-regularized"
    assert cfg.newton.algorithm == "fixed-load"


def test_limit_load_rejects_indirect_continuation_profile(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad-ll"
tags = ["regression"]

[mesh]
asset = "2d_homo_slope"
variant = "h0.5"
element = "P2"

[physics.mechanics]
model = "mohr_coulomb_limit_load"
davis = "B"

[continuation]
profile = "indirect-classic"

[newton]
profile = "indirect-regularized"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Limit-load cases must use a direct continuation profile"):
        load_run_case_config(path)


def test_committed_limit_load_cases_use_direct_fixed_load_profiles() -> None:
    for case_toml in sorted(CASE_ROOT.glob("*-ll/case.toml")):
        cfg = load_run_case_config(case_toml)

        assert cfg.problem.analysis == "ll", case_toml
        assert cfg.continuation.profile == "direct-limit-load", case_toml
        assert cfg.continuation.algorithm == "direct", case_toml
        assert cfg.continuation.method == "direct", case_toml
        assert cfg.newton.profile in {"limit-load-regularized", "limit-load-regularized-it100"}, case_toml
        assert cfg.newton.algorithm == "fixed-load", case_toml


@pytest.mark.parametrize("case_toml", sorted(CASE_ROOT.glob("*/case.toml")))
def test_committed_cases_validate(case_toml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PETSC_SSR_WORLD_SIZE", "4")

    load_run_case_config(case_toml).validate()


@pytest.mark.parametrize("case_toml", sorted(CASE_ROOT.glob("*/case.toml")))
def test_committed_cases_use_canonical_case_id(case_toml: Path) -> None:
    raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    case = dict(raw.get("case", {}))

    assert case.get("id") == case_toml.parent.name
    assert "name" not in case


@pytest.mark.parametrize("case_toml", sorted(CASE_ROOT.glob("*/case.toml")))
def test_committed_cases_do_not_repeat_profile_defaults(case_toml: Path) -> None:
    raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    repeated: list[str] = []
    for section_name, loader in (
        ("continuation", load_continuation_profile),
        ("newton", load_newton_profile),
        ("seepage", load_seepage_profile),
    ):
        section = raw.get(section_name, {})
        profile = str(section.get("profile", "")).strip()
        if not profile:
            continue
        defaults = loader(profile).data
        repeated.extend(f"{section_name}.{key}" for key, value in section.items() if key != "profile" and defaults.get(key) == value)

    assert repeated == []


@pytest.mark.parametrize("case_toml", sorted(CASE_ROOT.glob("*/case.toml")))
def test_committed_seepage_cases_use_named_runtime_profiles(case_toml: Path) -> None:
    raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    seepage = dict(raw.get("seepage", {}))
    if not seepage:
        return

    assert str(seepage.get("profile", "")).strip()
    assert {"linear_tolerance", "linear_max_iter", "nonlinear_max_iter"}.isdisjoint(seepage)


def test_cli_run_only_uses_force_c_baseline_when_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from petsc_ssr.cli import main as cli_main
    from petsc_ssr.runners import run_case_from_config

    forwarded: list[list[str]] = []

    def fake_runner(argv: list[str]) -> int:
        forwarded.append(argv)
        return 0

    monkeypatch.setattr(run_case_from_config, "main", fake_runner)
    case_toml = tmp_path / "case.toml"

    assert cli_main.main(["run", str(case_toml)]) == 0
    assert "--force-c-baseline" not in forwarded[-1]
    assert "--write-coordinate-bc-table" not in forwarded[-1]

    assert cli_main.main(["run", str(case_toml), "--force-c-baseline"]) == 0
    assert "--force-c-baseline" in forwarded[-1]

    assert cli_main.main(["run", str(case_toml), "--write-coordinate-bc-table"]) == 0
    assert "--write-coordinate-bc-table" in forwarded[-1]


def test_case_explain_records_profile_pc_variant(capsys: pytest.CaptureFixture[str]) -> None:
    from petsc_ssr.cli import main as cli_main

    status = cli_main.main(["case", "explain", str(CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml")])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["linear"]["pc_backend"] == "pmg_shell"
    assert payload["continuation"]["algorithm"] == "indirect"
    assert payload["newton"]["algorithm"] == "indirect-ssr"
    assert payload["linear"]["algorithm"] == "ksp_deflated"
    assert payload["linear"]["native_algorithm"] == "pmg-deflated"
    assert payload["linear"]["pc_variant"] == "pmg"
    assert payload["linear"]["requested_pc_variant"] == "pmg"
    assert payload["linear"]["pc_variant_fallback_reason"] is None
    assert payload["resolved_pmg"]["apply_backend"] == "shell_vcycle"
    assert payload["resolved_pmg"]["coarse_pc_type"] == "gamg"
    assert payload["resolved_pmg"]["smoother_max_it"] == 2
    assert payload["seepage"] is None


def test_case_explain_records_seepage_profile(capsys: pytest.CaptureFixture[str]) -> None:
    from petsc_ssr.cli import main as cli_main

    status = cli_main.main(["case", "explain", str(CASE_ROOT / "2d-sloan2013-seepage" / "case.toml")])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["seepage"]["profile"] == "sloan2013-steady"
    assert payload["seepage"]["linear_tolerance"] == 1e-10
    assert payload["seepage"]["linear_max_iter"] == 300
    assert payload["seepage"]["nonlinear_max_iter"] == 100


def test_case_validate_records_p1_profile_pc_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    from petsc_ssr.cli import main as cli_main

    status = cli_main.main(["case", "validate", str(CASE_ROOT / "2d-sloan2013-seepage" / "case.toml")])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pc_backend"] == "pmg_shell"
    assert payload["linear_algorithm"] == "ksp_deflated"
    assert payload["native_linear_algorithm"] == "gamg"
    assert payload["pc_variant"] == "gamg"
    assert payload["requested_pc_variant"] == "pmg"
    assert payload["pc_variant_fallback_reason"] == "p1_has_no_p_hierarchy"
    assert payload["seepage_profile"] == "sloan2013-steady"


def test_doctor_lists_all_public_profile_families(capsys: pytest.CaptureFixture[str]) -> None:
    from petsc_ssr.cli import main as cli_main

    status = cli_main.main(["doctor"])

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert "indirect-classic" in payload["continuation_profiles"]
    assert "indirect-regularized" in payload["newton_profiles"]
    assert "darcy-tight" in payload["seepage_profiles"]
    assert payload["solver_profiles"] == ["direct-debug", "gamg-p1-baseline", "pmg-deflated-baseline"]


def test_case_rejects_launcher_fields(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
ranks = [1, 2]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ranks"):
        load_run_case_config(path)


def test_case_rejects_structured_tags(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
tags = ["3d", "p4", "ssr"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate structured state"):
        load_run_case_config(path)


def test_case_rejects_linear_solver_details(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
tags = ["regression"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[linear]
profile = "pmg-deflated-baseline"
tolerance = 0.1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="linear solver details belong"):
        load_run_case_config(path)

    path.write_text(
        path.read_text(encoding="utf-8").replace("tolerance = 0.1", 'deflation_solver = "cg"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="linear solver details belong"):
        load_run_case_config(path)


def test_case_rejects_unknown_continuation_and_newton_profiles(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
tags = ["regression"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[continuation]
profile = "missing-continuation"

[newton]
profile = "indirect-regularized"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown continuation profile"):
        load_run_case_config(path)

    path.write_text(
        path.read_text(encoding="utf-8").replace('profile = "missing-continuation"', 'profile = "indirect-classic"').replace(
            'profile = "indirect-regularized"', 'profile = "missing-newton"', 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown newton profile"):
        load_run_case_config(path)


def test_case_rejects_profile_default_repeats(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
tags = ["regression"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[continuation]
profile = "indirect-classic"
step_max = 100

[newton]
profile = "indirect-regularized"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate the selected profile defaults"):
        load_run_case_config(path)


def test_case_rejects_profile_owned_continuation_and_newton_policy(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad-policy"
tags = ["regression"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[continuation]
profile = "indirect-classic"
predictor = "secant"

[newton]
profile = "indirect-regularized"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continuation algorithm selectors"):
        load_run_case_config(path)

    path.write_text(
        path.read_text(encoding="utf-8").replace('predictor = "secant"', "").replace(
            '[newton]\nprofile = "indirect-regularized"',
            '[newton]\nprofile = "indirect-regularized"\nit_max = 50',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Newton iteration"):
        load_run_case_config(path)


def test_pure_seepage_case_can_omit_continuation_and_newton_profiles(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "seepage"
tags = ["regression"]

[mesh]
asset = "3d_hetero_seepage"
variant = "concave_family_b"
element = "P2"

[physics.seepage]
model = "darcy"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    cfg = load_run_case_config(path).validate()
    assert cfg.problem.analysis == "seepage"
    assert cfg.continuation.profile == ""
    assert cfg.newton.profile == ""
    assert cfg.seepage.profile == "darcy-tight"


def test_case_rejects_seepage_runtime_policy(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad-seepage-profile"
tags = ["regression"]

[mesh]
asset = "3d_hetero_seepage"
variant = "concave_family_b"
element = "P2"

[physics.seepage]
model = "darcy"

[linear]
profile = "pmg-deflated-baseline"

[seepage]
profile = "darcy-tight"
linear_tolerance = 1e-10
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seepage runtime policy belongs"):
        load_run_case_config(path)


def test_case_rejects_mesh_runtime_defaults(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
tags = ["regression"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"
refine_levels = 0

[physics.mechanics]
model = "mohr_coulomb_ssr"

[linear]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="refinement and partitioning belong"):
        load_run_case_config(path)


def test_case_rejects_notebook_section(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
tags = ["regression"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[linear]
profile = "pmg-deflated-baseline"

[notebook]
family = "3d_continuation"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Notebook metadata belongs"):
        load_run_case_config(path)


def test_case_rejects_raw_output_arrays(tmp_path: Path) -> None:
    path = tmp_path / "case.toml"
    path.write_text(
        """
[case]
name = "bad"
tags = ["regression"]

[mesh]
asset = "3d_hetero_slope"
variant = "adaptive_family_a_l1"
element = "P4"

[physics.mechanics]
model = "mohr_coulomb_ssr"

[linear]
profile = "pmg-deflated-baseline"

[output]
solution = ["vtu"]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output uses named presets"):
        load_run_case_config(path)


def test_legacy_internal_case_schema_requires_explicit_debug_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "legacy.toml"
    path.write_text(
        """
[problem]
name = "legacy"
asset = "3d_hetero_slope"
mesh_variant = "adaptive_family_a_l1"
elem_type = "P4"

[execution]
mechanics_backend = "legacy_array"

[linear_solver]
profile = "pmg-deflated-baseline"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy/internal case schema"):
        load_run_case_config(path)

    monkeypatch.setenv("PETSC_SSR_ALLOW_LEGACY_CASE_SCHEMA", "1")

    cfg = load_run_case_config(path)
    assert cfg.problem.asset == "3d_hetero_slope"
    assert cfg.execution.mechanics_backend == "legacy_array"
    assert cfg.execution.node_ordering == "block_metis"
