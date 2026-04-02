#!/usr/bin/env python3
"""Generate shared benchmark notebooks from case metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"


def _markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def _code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def _load_metadata(case_toml: Path) -> dict[str, object]:
    import tomllib

    raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
    benchmark = dict(raw.get("benchmark", {}))
    notebook = dict(raw.get("notebook", {}))
    return {
        "case_dir_name": case_toml.parent.name,
        "title": str(benchmark.get("title", case_toml.parent.name)),
        "matlab_script": str(benchmark.get("matlab_script", "")),
        "comparison_kind": str(benchmark.get("comparison_kind", "")).lower(),
        "mpi_ranks": int(benchmark.get("mpi_ranks", 8)),
        "family": str(notebook.get("family", "")),
        "material_palette": notebook.get("material_palette"),
        "slice_planes_x": list(notebook.get("slice_planes_x", [])),
        "slice_planes_y": list(notebook.get("slice_planes_y", [])),
        "slice_planes_z": list(notebook.get("slice_planes_z", [])),
        "strain_clim_scale_max": notebook.get("strain_clim_scale_max"),
    }


def _import_cell(case_dir_name: str):
    return _code_cell(
        f"""
        from pathlib import Path
        import sys

        ROOT = next(
            path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / "benchmarks").is_dir() and (path / "src").is_dir()
        )
        if str(ROOT / "benchmarks") not in sys.path:
            sys.path.insert(0, str(ROOT / "benchmarks"))

        import notebook_support as nb

        CASE_TOML = ROOT / "benchmarks" / "{case_dir_name}" / "case.toml"
        CASE_DIR = CASE_TOML.parent
        """
    )


def _load_case_cell():
    return _code_cell(
        """
        metadata = nb.load_case_metadata(CASE_TOML)
        sections = nb.load_case_sections(CASE_TOML)
        materials = nb.load_case_materials(CASE_TOML)

        print(nb.summarize_sections(sections, materials))
        metadata
        """
    )


def _simulation_controls_cell():
    return _code_cell(
        """
        RUN_LABEL = "simulation"
        RUN_MODE = "run"  # change to "auto" or "reuse" only if you want artifact reuse
        EXECUTION_PROFILE = "benchmark"  # "smoke" or "benchmark"
        MPI_RANKS = None  # None uses the selected profile default
        PREVIEW_CONFIG = nb.write_generated_case_toml(
            case_toml=CASE_TOML,
            sections=sections,
            materials=materials,
            run_label=RUN_LABEL,
            root=CASE_DIR,
        )
        PREVIEW_CONFIG
        """
    )


def _execution_cell():
    return _code_cell(
        """
        execution = nb.ensure_notebook_artifacts(
            case_toml=CASE_TOML,
            sections=sections,
            materials=materials,
            run_label=RUN_LABEL,
            run_mode=RUN_MODE,
            execution_profile=EXECUTION_PROFILE,
            mpi_ranks=MPI_RANKS,
            root=CASE_DIR,
        )
        GENERATED_CONFIG = execution.generated_config
        ACTIVE_CONFIG = execution.active_config
        OUT_DIR = execution.out_dir
        execution
        """
    )


def _simulation_summary_cell():
    return _code_cell(
        """
        artifacts = nb.load_run_artifacts(OUT_DIR)
        nb.show_run_summary(artifacts)
        print("")
        print("Artifact source:", execution.source_label)
        print("Active config:", ACTIVE_CONFIG)
        print("")
        print("Saved files:")
        for path in nb.list_saved_files(OUT_DIR):
            print(" ", path.relative_to(ROOT))
        """
    )


def _visualisation_controls_cell():
    return _code_cell(
        """
        RUN_LABEL = "simulation"
        RUN_MODE = "reuse"  # change to "auto" only if you want this notebook to fall back to a smoke solve
        EXECUTION_PROFILE = "smoke"
        MPI_RANKS = None
        JUPYTER_BACKEND_OVERRIDE = None  # set to "client", "trame", "server", "html", or "static"; None uses case.toml
        SURFACE_SUBDIVISION_OVERRIDE = None  # set to 1, 2, 3, or 4 to trade quality for speed; None uses case.toml default
        SURFACE_DECIMATE_REDUCTION_OVERRIDE = None  # set e.g. 0.5, 0.75, or 0.9 for a lighter preview mesh
        BOUNDARY_EDGE_OVERLAY_OVERRIDE = None  # set to True to draw coarse tetra boundary edges over smooth surface colors
        """
    )


def _artifacts_cell():
    return _code_cell(
        """
        artifacts = nb.load_run_artifacts(OUT_DIR)
        nb.show_run_summary(artifacts)
        print("")
        print("Artifact source:", execution.source_label)
        print("Active config:", ACTIVE_CONFIG)
        print("")
        print("Saved files:")
        for path in nb.list_saved_files(OUT_DIR):
            print(" ", path.relative_to(ROOT))
        """
    )


def _continuation_cells():
    return [
        _markdown_cell(
            """
            ## Continuation Summary

            MATLAB-style continuation and timing summaries rebuilt from the PETSc artifacts.
            """
        ),
        _code_cell(
            """
            _ = nb.plot_convergence_dashboard(artifacts)
            _ = nb.plot_timing_breakdown(artifacts)
            """
        ),
    ]


def _family_cells(meta: dict[str, object]):
    family = str(meta["family"])
    palette = meta.get("material_palette")
    slice_planes_x = meta.get("slice_planes_x", [])
    slice_planes_y = meta.get("slice_planes_y", [])
    slice_planes_z = meta.get("slice_planes_z", [])
    strain_clim_scale_max = meta.get("strain_clim_scale_max")

    cells = []
    if family.startswith("2d_"):
        cells.append(_markdown_cell("## 2D Visualizations"))
        if family == "2d_seepage_continuation" and palette:
            cells.append(_code_cell(f"_ = nb.plot_2d_heterogeneity(CASE_TOML, palette_name={palette!r})"))
            cells.append(_code_cell("_ = nb.plot_2d_mesh(CASE_TOML)"))
        if family in {"2d_seepage", "2d_seepage_continuation"}:
            cells.append(_code_cell("_ = nb.plot_2d_pore_pressure(artifacts, ACTIVE_CONFIG)"))
            cells.append(_code_cell("_ = nb.plot_2d_saturation(artifacts, ACTIVE_CONFIG)"))
        if family in {"2d_continuation", "2d_seepage_continuation"}:
            cells.append(_code_cell("_ = nb.plot_2d_displacement(artifacts, ACTIVE_CONFIG)"))
            cells.append(_code_cell("_ = nb.plot_2d_deviatoric_strain(artifacts, ACTIVE_CONFIG)"))
        return cells

    cells.append(
        _markdown_cell(
            """
            ## 3D Visualizations

            These cells use PyVista when the optional `.[viz]` extras are available. In a non-viz environment they
            return a clear status message instead of failing import-time.
            """
        )
    )
    cells.append(_code_cell("nb.viz_support_status()"))
    if family == "3d_seepage_continuation":
        cells.append(
            _code_cell(
                "_ = nb.show_3d_mesh_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
            )
        )
        cells.append(
            _code_cell(
                "_ = nb.show_3d_pore_pressure_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
            )
        )
        cells.append(
            _code_cell(
                "_ = nb.show_3d_displacement_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
            )
        )
        cells.append(
            _code_cell(
                "_ = nb.show_3d_deviatoric_surface_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
            )
        )
        if slice_planes_x or slice_planes_y or slice_planes_z:
            cells.append(
                _code_cell(
                    f"""
                    _ = nb.show_3d_deviatoric_slices(
                        artifacts,
                        ACTIVE_CONFIG,
                        slice_planes_x={slice_planes_x!r},
                        slice_planes_y={slice_planes_y!r},
                        slice_planes_z={slice_planes_z!r},
                        clim_scale_max={strain_clim_scale_max!r},
                        jupyter_backend=JUPYTER_BACKEND_OVERRIDE,
                    )
                    """
                )
            )
        return cells
    if family == "3d_seepage":
        cells.append(
            _code_cell(
                "_ = nb.show_3d_pore_pressure_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
            )
        )
        cells.append(
            _code_cell(
                "_ = nb.show_3d_saturation_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
            )
        )
        return cells
    cells.append(
        _code_cell(
            "_ = nb.show_3d_displacement_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
        )
    )
    cells.append(
        _code_cell(
            "_ = nb.show_3d_deviatoric_surface_view(artifacts, ACTIVE_CONFIG, surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE, surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE, boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE, jupyter_backend=JUPYTER_BACKEND_OVERRIDE)"
        )
    )
    if slice_planes_x or slice_planes_y or slice_planes_z:
        cells.append(
            _code_cell(
                f"""
                _ = nb.show_3d_deviatoric_slices(
                    artifacts,
                    ACTIVE_CONFIG,
                    slice_planes_x={slice_planes_x!r},
                    slice_planes_y={slice_planes_y!r},
                    slice_planes_z={slice_planes_z!r},
                    clim_scale_max={strain_clim_scale_max!r},
                    jupyter_backend=JUPYTER_BACKEND_OVERRIDE,
                )
                """
            )
        )
    return cells


def _common_intro_cell(meta: dict[str, object], *, role: str):
    return _markdown_cell(
        f"""
        # {meta["title"]} ({role})

        This notebook is generated from the shared benchmark notebook framework.

        - Benchmark folder: `{meta["case_dir_name"]}`
        - Original MATLAB driver: `{meta["matlab_script"]}`
        - Comparison kind: `{meta["comparison_kind"]}`
        - Notebook family: `{meta["family"]}`
        """
    )


def build_simulation_notebook(case_toml: Path):
    meta = _load_metadata(case_toml)
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}

    cells = [
        _common_intro_cell(meta, role="simulation"),
        _import_cell(str(meta["case_dir_name"])),
        _load_case_cell(),
        _markdown_cell(
            """
            ## Editable Runtime Sections

            Modify values directly in `sections` or `materials`, then rerun the config-write and solver cells below.
            Generated notebook configs and notebook-local solver artifacts are written under
            `artifacts/<run_label>/` inside this benchmark folder.
            """
        ),
        _code_cell("sections"),
        _code_cell("materials"),
        _simulation_controls_cell(),
        _execution_cell(),
        _simulation_summary_cell(),
        _markdown_cell(
            """
            ## Next Step

            Open `visualisation.ipynb` to inspect the generated VTU fields, continuation history, and MATLAB-style plots.
            """
        ),
    ]
    notebook.cells = cells
    return notebook


def build_visualisation_notebook(case_toml: Path):
    meta = _load_metadata(case_toml)
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python"}

    cells = [
        _common_intro_cell(meta, role="visualisation"),
        _import_cell(str(meta["case_dir_name"])),
        _load_case_cell(),
        _markdown_cell(
            """
            ## Artifact Source

            This notebook is post-processing oriented. By default it reuses the artifacts from `simulation.ipynb`.
            """
        ),
        _visualisation_controls_cell(),
        _execution_cell(),
        _artifacts_cell(),
    ]
    if str(meta["comparison_kind"]).lower() == "continuation":
        cells.extend(_continuation_cells())
    cells.extend(_family_cells(meta))
    notebook.cells = cells
    return notebook


def generate_all(benchmarks_dir: Path = BENCHMARKS_DIR) -> list[Path]:
    generated: list[Path] = []
    for case_toml in sorted(benchmarks_dir.glob("*/case.toml")):
        simulation = build_simulation_notebook(case_toml)
        visualisation = build_visualisation_notebook(case_toml)
        simulation_path = case_toml.parent / "simulation.ipynb"
        visualisation_path = case_toml.parent / "visualisation.ipynb"
        legacy_path = case_toml.parent / "pyvista_workflow.ipynb"
        nbf.write(simulation, simulation_path)
        nbf.write(visualisation, visualisation_path)
        if legacy_path.exists():
            legacy_path.unlink()
        generated.extend((simulation_path, visualisation_path))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared benchmark notebooks.")
    parser.add_argument("--benchmarks_dir", type=Path, default=BENCHMARKS_DIR)
    args = parser.parse_args()
    generated = generate_all(args.benchmarks_dir.resolve())
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
