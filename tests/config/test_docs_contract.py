from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DOCS = {
    "quickstart.md": ["petsc-ssr doctor", "case validate", "suite expand"],
    "create-a-benchmark.md": ["asset validate", "benchmark init", "case dry-run"],
    "assets.md": ["Mesh assets own", "asset validate"],
    "curved-boundaries.md": ["boundary_geometry", "DMPlex"],
    "neumann-bcs.md": ["mechanics_neumann_labels.csv", "affine native face quadrature"],
    "local-32-testing.md": ["local-32-smoke", "local-32-strong-scaling", "targets compare"],
    "hpc.md": ["minimal", "optional", "suite"],
    "architecture.md": ["PETSc-first", "Public Model Invariants"],
}


def test_public_workflow_docs_exist_and_cover_required_terms() -> None:
    docs_root = ROOT / "docs"

    for name, required_terms in REQUIRED_DOCS.items():
        path = docs_root / name
        assert path.exists(), name
        text = path.read_text(encoding="utf-8")
        for term in required_terms:
            assert term in text, f"{name} is missing {term!r}"


def test_layout_links_public_workflow_docs() -> None:
    layout = (ROOT / "docs" / "layout.md").read_text(encoding="utf-8")

    for name in REQUIRED_DOCS:
        assert f"`docs/{name}`" in layout
