from __future__ import annotations

from pathlib import Path


ENGINE_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ENGINE_ROOT / "benchmarks" / "tools"


def generate_case_readme(case_toml: Path) -> None:
    _load_tool("generate_benchmark_readmes").generate_readme_for_case(case_toml)


def generate_case_notebooks(case_toml: Path) -> None:
    _load_tool("generate_benchmark_notebooks").generate_notebooks_for_case(case_toml)


def generate_all(cases_root: Path) -> None:
    readmes = _load_tool("generate_benchmark_readmes")
    notebooks = _load_tool("generate_benchmark_notebooks")
    readmes.generate_readmes(cases_root)
    notebooks.generate_notebooks(cases_root)


def _load_tool(name: str):
    import importlib.util

    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load benchmark tool {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
