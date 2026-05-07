# Archive

This folder contains the historical analysis material that no longer belongs in
the clean benchmark root.

The benchmark root now keeps only `README.md`, `case.toml`, `run.sh`,
`simulation.ipynb`, and `visualisation.ipynb`. Legacy scaling scripts,
comparison helpers, and follow-up reports live here.

These files are provenance records, not current run instructions. Some archived
commands and data fields still refer to deleted route-specific CLI modules,
raw `mesh_path` arguments, or `pmg_coarse_mesh_path`. New config-driven runs use
`slope_stability.cli.run_case_from_config`, select an asset and mesh variant in
the case config, and use `pmg_coarse_mesh_variant` where a PMG coarse mesh is
needed.
