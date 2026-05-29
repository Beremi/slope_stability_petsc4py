from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from petsc_ssr.assets import load_problem_asset


ENGINE_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ENGINE_ROOT / "benchmarks" / "tools"
DEFAULT_CASES_ROOT = ENGINE_ROOT / "benchmarks" / "cases"


def create_case_skeleton(
    slug: str,
    *,
    asset: str,
    cases_root: Path = DEFAULT_CASES_ROOT,
    variant: str | None = None,
    element: str = "P2",
    analysis: str = "ssr",
    title: str | None = None,
    linear_profile: str = "pmg-deflated-baseline",
    overwrite: bool = False,
    generate_notebooks: bool = True,
) -> Path:
    """Create a compact benchmark case skeleton from an asset definition."""

    case_id = _validate_slug(slug)
    asset_def = load_problem_asset(asset)
    analysis_name = _normalize_analysis(analysis)
    _validate_asset_supports_analysis(asset_def, analysis_name)
    resolved_variant = asset_def.resolve_variant(variant or asset_def.default_variant)

    case_dir = cases_root / case_id
    case_toml = case_dir / "case.toml"
    if case_toml.exists() and not overwrite:
        raise FileExistsError(f"Case {case_id!r} already exists at {case_toml}; pass --overwrite to replace generated files.")
    case_dir.mkdir(parents=True, exist_ok=True)

    variant_name = _display_variant(resolved_variant.name)
    case_toml.write_text(
        _render_case_toml(
            case_id=case_id,
            title=title or _title_from_slug(case_id),
            asset=asset_def.asset_id,
            variant=variant_name,
            element=element,
            analysis=analysis_name,
            linear_profile=linear_profile,
        ),
        encoding="utf-8",
    )
    _write_notebook_sidecar(case_dir / "notebook.toml", dimension=int(asset_def.dimension), analysis=analysis_name)
    _write_run_sh(case_dir / "run.sh")
    generate_case_readme(case_toml)
    if generate_notebooks:
        generate_case_notebooks(case_toml)
    return case_toml


def generate_case_readme(case_toml: Path) -> None:
    _load_tool("generate_benchmark_readmes").generate_readme_for_case(case_toml)


def generate_case_notebooks(case_toml: Path) -> None:
    _load_tool("generate_benchmark_notebooks").generate_notebooks_for_case(case_toml)


def generate_all(cases_root: Path) -> None:
    readmes = _load_tool("generate_benchmark_readmes")
    notebooks = _load_tool("generate_benchmark_notebooks")
    readmes.generate_readmes(cases_root)
    notebooks.generate_notebooks(cases_root)


def check_generated_cases(cases_root: Path = DEFAULT_CASES_ROOT, *, check_notebooks: bool = True) -> list[str]:
    """Return reproducibility issues for generated benchmark scaffolding."""

    issues: list[str] = []
    for case_toml in sorted(cases_root.glob("*/case.toml")):
        issues.extend(check_case_artifacts(case_toml, check_notebooks=check_notebooks))
    return issues


def check_case_artifacts(case_toml: Path, *, check_notebooks: bool = True) -> list[str]:
    """Check that a case's generated files match the public benchmark model."""

    case_toml = Path(case_toml)
    issues: list[str] = []
    try:
        from petsc_ssr.config import load_run_case_config

        load_run_case_config(case_toml).validate()
    except Exception as exc:
        issues.append(f"{_rel(case_toml)}: case schema validation failed: {exc}")
        return issues

    try:
        raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(f"{_rel(case_toml)}: cannot read case TOML: {exc}")
        return issues
    if "notebook" in raw:
        issues.append(f"{_rel(case_toml)}: notebook metadata belongs in notebook.toml, not case.toml")

    readmes = _load_tool("generate_benchmark_readmes")
    expected_readme = readmes._render_readme(readmes._load_case(case_toml))
    readme_path = case_toml.parent / "README.md"
    if not readme_path.exists():
        issues.append(f"{_rel(readme_path)}: missing generated README")
    elif readme_path.read_text(encoding="utf-8") != expected_readme:
        issues.append(f"{_rel(readme_path)}: stale generated README")

    run_sh = case_toml.parent / "run.sh"
    expected_run_sh = _render_run_sh()
    if not run_sh.exists():
        issues.append(f"{_rel(run_sh)}: missing generated run wrapper")
    else:
        if run_sh.read_text(encoding="utf-8") != expected_run_sh:
            issues.append(f"{_rel(run_sh)}: stale generated run wrapper")
        if run_sh.stat().st_mode & 0o111 == 0:
            issues.append(f"{_rel(run_sh)}: run wrapper is not executable")

    notebook_sidecar = case_toml.parent / "notebook.toml"
    if not notebook_sidecar.exists():
        issues.append(f"{_rel(notebook_sidecar)}: missing notebook sidecar")
    else:
        try:
            sidecar = tomllib.loads(notebook_sidecar.read_text(encoding="utf-8"))
            family = str(dict(sidecar.get("notebook", {})).get("family", "")).strip()
            if not family:
                issues.append(f"{_rel(notebook_sidecar)}: [notebook].family must be set")
        except Exception as exc:
            issues.append(f"{_rel(notebook_sidecar)}: cannot read notebook sidecar: {exc}")

    if check_notebooks:
        issues.extend(_check_notebook_pair(case_toml))
    return issues


def _validate_slug(slug: str) -> str:
    text = str(slug).strip()
    if not text:
        raise ValueError("Case slug must not be empty.")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if any(ch not in allowed for ch in text) or "--" in text or text.startswith("-") or text.endswith("-"):
        raise ValueError(f"Case slug {slug!r} must be lower-kebab ASCII, e.g. '3d-heterogeneous-ssr-p4'.")
    return text


def _normalize_analysis(value: str) -> str:
    text = str(value).strip().lower().replace("_", "-")
    aliases = {
        "ssr": "ssr",
        "shear-strength-reduction": "ssr",
        "ll": "ll",
        "limit-load": "ll",
        "limitload": "ll",
        "seepage": "seepage",
    }
    if text not in aliases:
        raise ValueError("Analysis must be one of: ssr, ll, seepage.")
    return aliases[text]


def _validate_asset_supports_analysis(asset_def: Any, analysis: str) -> None:
    if analysis in {"ssr", "ll"} and asset_def.mechanics_spec() is None:
        raise ValueError(f"Asset {asset_def.asset_id!r} does not declare mechanics supports/materials.")
    if analysis == "seepage" and asset_def.seepage_spec() is None:
        raise ValueError(f"Asset {asset_def.asset_id!r} does not declare seepage supports.")


def _display_variant(value: str) -> str:
    text = str(value).strip()
    return text[:-4] if text.endswith(".msh") else text


def _title_from_slug(slug: str) -> str:
    return " ".join(part.upper() if part in {"2d", "3d", "ssr", "ll", "p1", "p2", "p4"} else part.capitalize() for part in slug.split("-"))


def _render_case_toml(
    *,
    case_id: str,
    title: str,
    asset: str,
    variant: str,
    element: str,
    analysis: str,
    linear_profile: str,
) -> str:
    physics = (
        "[physics.seepage]\nmodel = \"darcy\"\n"
        if analysis == "seepage"
        else f"[physics.mechanics]\nmodel = \"{'mohr_coulomb_limit_load' if analysis == 'll' else 'mohr_coulomb_ssr'}\"\ndavis = \"B\"\n"
    )
    continuation = (
        ""
        if analysis == "seepage"
        else (
            "\n[continuation]\n"
            f"profile = \"{'direct-limit-load' if analysis == 'll' else 'indirect-classic'}\"\n"
        )
    )
    newton = (
        ""
        if analysis == "seepage"
        else (
            "\n[newton]\n"
            f"profile = \"{'limit-load-regularized' if analysis == 'll' else 'indirect-regularized-dlambda-stop'}\"\n"
        )
    )
    return (
        "[case]\n"
        f"id = \"{case_id}\"\n"
        f"title = \"{title}\"\n"
        "tags = [\"experimental\"]\n"
        "\n"
        "[mesh]\n"
        f"asset = \"{asset}\"\n"
        f"variant = \"{variant}\"\n"
        f"element = \"{str(element).strip().upper()}\"\n"
        "\n"
        f"{physics}"
        f"{continuation}"
        f"{newton}"
        "\n[linear]\n"
        f"profile = \"{linear_profile}\"\n"
        "\n[output]\n"
        f"preset = \"{'standard-seepage' if analysis == 'seepage' else 'standard-continuation'}\"\n"
    )


def _write_notebook_sidecar(path: Path, *, dimension: int, analysis: str) -> None:
    family = "seepage" if analysis == "seepage" else f"{dimension}d_continuation"
    path.write_text(f"[notebook]\nfamily = \"{family}\"\n", encoding="utf-8")


def _write_run_sh(path: Path) -> None:
    path.write_text(_render_run_sh(), encoding="utf-8")
    path.chmod(0o755)


def _render_run_sh() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec "$CASE_DIR/../../tools/run_standalone_case.sh" "$CASE_DIR" "$@"\n'
    )


def _check_notebook_pair(case_toml: Path) -> list[str]:
    issues: list[str] = []
    simulation_path = case_toml.parent / "simulation.ipynb"
    visualisation_path = case_toml.parent / "visualisation.ipynb"
    for notebook_path in (simulation_path, visualisation_path):
        if not notebook_path.exists():
            issues.append(f"{_rel(notebook_path)}: missing generated notebook")
    if issues:
        return issues

    try:
        notebooks = _load_tool("generate_benchmark_notebooks")
    except Exception:
        return issues

    expected = {
        simulation_path: notebooks.build_simulation_notebook(case_toml),
        visualisation_path: notebooks.build_visualisation_notebook(case_toml),
    }
    for notebook_path, expected_notebook in expected.items():
        try:
            existing = notebooks.nbf.read(notebook_path, as_version=4)
        except Exception as exc:
            issues.append(f"{_rel(notebook_path)}: cannot read generated notebook: {exc}")
            continue
        if _normalized_notebook(existing) != _normalized_notebook(expected_notebook):
            issues.append(f"{_rel(notebook_path)}: stale generated notebook")
    return issues


def _normalized_notebook(notebook: Any) -> dict[str, Any]:
    payload = json.loads(json.dumps(notebook))
    for cell in payload.get("cells", []):
        if isinstance(cell, dict):
            cell.pop("id", None)
    return payload


def _load_tool(name: str):
    import importlib.util

    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load benchmark tool {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ENGINE_ROOT))
    except ValueError:
        return str(path)
