# Problem Definition Audit

This document replaces the pre-migration audit.

The repository now has one active problem-definition contract:

- `meshes/<asset>/definition.py`
- one or more canonical Gmsh `MSH 4.1` files in `meshes/<asset>/`
- optional legacy inputs and deterministic converter scripts in `meshes/<asset>/legacy/`

The active runtime no longer depends on generated geometry assets, textmesh bundles, COMSOL loaders, water-level loaders, coordinate selectors, duplicated LL/SSR asset wrappers, or integer `boundary_type` switches. Those inputs still exist only as migration references under `legacy/` or in historical archive material.

## Scope And Validation

- Scope: every current executable asset under `meshes/*/definition.py`
- Current canonical asset count: `10`
- Active benchmark asset count: `10`
- Health result: every active canonical asset is `sound`

Validation used for this post-migration audit:

- `PYTHONPATH=src ./.venv/bin/pytest -q tests/test_benchmark_notebooks.py tests/test_pmg_hierarchy.py tests/test_compare_preconditioners.py`
  - result: `93 passed`
- `PYTHONPATH=src ./.venv/bin/pytest -q tests/test_executable_asset_definitions.py tests/test_problem_asset_runtime.py tests/test_problem_assets_seepage_mesh_folders.py tests/test_3d_hetero_ssr_gmsh_loader.py tests/test_canonical_curved_boundary_geometry.py`
  - result: `26 passed`

## Canonical Contract

### Definition Surface

Every canonical asset now exposes `ASSET = ...` from `meshes/<asset>/definition.py` and uses only:

- `build_problem_asset_2d(...)`
- `build_problem_asset_3d(...)`
- `build_seepage_spec(...)`

Active canonical definitions contain only:

- `asset_id`
- dimension
- default variant and mesh variants
- named material models
- `region_assignment`
- mechanics BCs by logical boundary name
- seepage BCs by logical boundary or node-support name
- optional named mechanics profiles

Active canonical definitions do not contain:

- coordinate selectors
- geometry recipes
- custom source-kind loaders
- asset-specific runtime hooks
- integer `boundary_type` logic

### Mesh Contract

Canonical mesh naming inside `.msh`:

- volume regions: `region:<name>`
- support boundaries: `boundary:<name>`
- node supports: `nodeset:<name>`
- optional curved geometry patches: `boundary_geom:<name>`

Canonical mesh guarantees:

- mesh format is `MSH 4.1`
- 2D volume cells are linear `triangle`
- 3D volume cells are linear `tetra`
- support entities are linear `line` in 2D and linear `triangle` in 3D
- higher-order solver meshes are promoted on demand from the linear canonical mesh
- allowed extra cell types are only geometry/support helpers (`vertex`, `line3`, `triangle6`)

### Runtime Flow

Active runtime resolution now goes through:

- asset discovery: `src/slope_stability/assets/__init__.py`
- definition normalization and problem-spec construction: `src/slope_stability/assets/factories.py`
- canonical Gmsh loading and higher-order promotion: `src/slope_stability/assets/support/canonical_gmsh.py`
- config-to-asset resolution: `src/slope_stability/problem_asset_runtime.py`
- post-processing mesh reconstruction: `src/slope_stability/postprocess/case_mesh.py`

Active benchmark configs now use:

- `problem.asset`
- `problem.mesh_variant`
- optional `problem.profile`
- `problem.analysis`
- `problem.elem_type`

Active benchmark configs no longer rely on:

- `problem.dimension`
- `problem.variant`
- `problem.seepage`
- `problem.case`
- `problem.mesh_path`
- `problem.mesh_boundary_type`
- `[[materials]]`
- `[case_data]`
- `[seepage].water_unit_weight`
- `[seepage].conductivity`

`src/slope_stability/cli/run_case_from_config.py` now dispatches by generic problem class: dimension, analysis, and seepage capability. It does not branch on old asset/source-kind families.

## Cross-Cutting Findings

### 1. The Active Asset Layer Is Canonical And Mesh-Local

- Every current asset under `meshes/` is `.msh` + `definition.py`
- all non-`.msh` historical inputs were converted and moved under per-asset `legacy/source/`
- converter inventory lives in `meshes/CONVERTERS.md`

### 2. LL And SSR Wrapper Duplication Is Gone

- `2d_generated_homo` became `2d_homo_slope`
- `3d_homo_ll` and `3d_homo_ssr` were merged into `3d_homo_slope`
- `3d_hetero_ll` and `3d_hetero_ssr` were merged into `3d_hetero_slope`
- old source-family distinctions survive only as mesh variants, mainly `family_a` and `family_b`

### 3. Old Special-Case Geometry Paths Are No Longer In The Active Runtime

- old textmesh, generated-geometry, water-level, and COMSOL families are now historical converter inputs
- the active runtime uses one source kind: `gmsh_problem_asset`
- the production mesh build path is generic and asset-agnostic

### 4. Old Hidden Conventions Were Replaced With Explicit Mesh Names

- 2D boundary selectors were replaced by explicit `boundary:*` or `nodeset:*` entities in the canonical `.msh`
- homogeneous 3D assets no longer depend on single-row material broadcasting across mesh ids `0..3`
- seepage transition and SIOPT base-fix switches now use named `profile`s, not integer `boundary_type`

### 5. Higher-Order Promotion Is Now A Generic Mesh Capability

- canonical assets store only linear volume simplices
- solver meshes for `P2`, `P3`, and `P4` are generated on demand in `src/slope_stability/assets/support/canonical_gmsh.py`
- curved-boundary geometry patches are part of the generic API through `boundary_geom:<name>`
- curved geometry support is validated in `tests/test_canonical_curved_boundary_geometry.py`
- no production asset currently needs curved Neumann patches, but the API is already wired for them

### 6. Compatibility Residue Is Limited To Explicit Generic Aliases

- `src/slope_stability/assets/factories.py` still exports `build_asset(...)` as a compatibility alias
- `problem.case` remains a harmless optional reporting label
- archived studies, stored artifact snapshots, and historical docs may mention old asset names or raw mesh paths
- the active config loader rejects raw mesh paths, committed materials, and problem-owned hydraulic values

## Quick Matrix

| Asset | Benchmark-Active | Health | Primary Note |
| --- | --- | --- | --- |
| `2d_franz_dam` | yes | `sound` | Converted text-bundle dam mesh with explicit head-support boundary. |
| `2d_homo_slope` | yes | `sound` | Canonical homogeneous 2D slope, shared cleanly by SSR and LL. |
| `2d_kozinec` | yes | `sound` | Canonical 2D heterogeneity mesh; mechanical-only with explicit boundaries. |
| `2d_luzec` | yes | `sound` | Canonical 2D seepage-plus-mechanics mesh with by-material conductivity. |
| `2d_sloan2013` | yes | `sound` | Canonical seepage-only asset; misleading mechanics surface was removed. |
| `3d_hetero_seepage` | yes | `sound` | Generic 3D seepage/mechanics asset with canonical head node supports. |
| `3d_hetero_seepage_transition` | yes | `sound` | Former COMSOL transition asset now expressed as canonical `.msh` plus profiles. |
| `3d_hetero_slope` | yes | `sound` | Canonical merged 3D heterogeneous slope family. |
| `3d_homo_slope` | yes | `sound` | Canonical merged 3D homogeneous slope family. |
| `3d_siopt` | yes | `sound` | Canonical SIOPT family with named base-fix profiles. |

## Asset-By-Asset Audit

### `2d_franz_dam`

- Active benchmarks: `benchmarks/slope_stability_2D_Franz_dam_SSR/case.toml`
- Geometry: `meshes/2d_franz_dam/default.msh`
- Materials: ten named material models in `meshes/2d_franz_dam/definition.py`, assigned one-to-one through `region_assignment`
- Mechanics BCs: logical boundaries `left`, `right`, `base`
- Seepage BCs: `head_support` with `piecewise_linear_level`, `conductivity_mode="by_material"`, `scope="domain_below_head"`
- Health: `sound`
- Evidence:
  - `meshes/2d_franz_dam/definition.py`
  - `meshes/2d_franz_dam/default.msh`
  - `meshes/2d_franz_dam/legacy/convert_to_msh.py`

### `2d_homo_slope`

- Active benchmarks:
  - `benchmarks/run_2D_homo_SSR_capture/case.toml`
  - `benchmarks/slope_stability_2D_homo_LL/case.toml`
- Geometry: `meshes/2d_homo_slope/h1.0.msh` and `meshes/2d_homo_slope/h0.5.msh`
- Materials: one material model `homogeneous_slope`; region `slope_mass`
- Mechanics BCs: logical boundaries `left`, `right`, `base`
- Seepage BCs: none
- Health: `sound`
- Evidence:
  - `meshes/2d_homo_slope/definition.py`
  - `meshes/2d_homo_slope/h1.0.msh`
  - `meshes/2d_homo_slope/legacy/convert_to_msh.py`

### `2d_kozinec`

- Active benchmarks:
  - `benchmarks/slope_stability_2D_Kozinec_SSR/case.toml`
  - `benchmarks/slope_stability_2D_Kozinec_LL/case.toml`
- Geometry: `meshes/2d_kozinec/default.msh`
- Materials: seven named material models mapped from `subdomain_1` through `subdomain_7`
- Mechanics BCs: logical boundaries `left`, `right`, `base`
- Additional mechanics state: `mechanics.hydraulic_state` is a generic piecewise-linear water-level specification used by the mechanical drivers
- Seepage BCs: none
- Health: `sound`
- Evidence:
  - `meshes/2d_kozinec/definition.py`
  - `meshes/2d_kozinec/default.msh`
  - `meshes/2d_kozinec/legacy/convert_to_msh.py`

### `2d_luzec`

- Active benchmarks: `benchmarks/slope_stability_2D_Luzec_SSR/case.toml`
- Geometry: `meshes/2d_luzec/default.msh`
- Materials: eight named material models, each carrying mechanics and hydraulic conductivity
- Mechanics BCs: logical boundaries `left`, `right`, `base`
- Seepage BCs: `head_support` with `piecewise_linear_level`, `conductivity_mode="by_material"`, `scope="domain_below_head"`
- Health: `sound`
- Evidence:
  - `meshes/2d_luzec/definition.py`
  - `meshes/2d_luzec/default.msh`
  - `meshes/2d_luzec/legacy/convert_to_msh.py`

### `2d_sloan2013`

- Active benchmarks: `benchmarks/run_2D_sloan2013_seepage_capture/case.toml`
- Geometry: `meshes/2d_sloan2013/default.msh`
- Materials: two hydraulic materials, `slope_mass` and `weak_layer`
- Mechanics BCs: none in the active canonical contract
- Seepage BCs: `head_support` with `piecewise_linear_level`, `conductivity_mode="by_material"`, `scope="domain_below_head"`
- Health: `sound`
- Evidence:
  - `meshes/2d_sloan2013/definition.py`
  - `meshes/2d_sloan2013/default.msh`
  - `meshes/2d_sloan2013/legacy/convert_to_msh.py`

### `3d_hetero_seepage`

- Active benchmarks: `benchmarks/run_3D_hetero_seepage_capture/case.toml`
- Geometry: canonical variants `family_*.msh` and `concave_family_*.msh`
- Materials: four named material models with mechanics and hydraulic conductivity
- Mechanics BCs: logical boundaries `x_lock`, `y_lateral_lock`, `base`
- Seepage BCs: node-support-based heads `head_dry`, `head_porous`, `head_free`; conductivity by material
- Health: `sound`
- Evidence:
  - `meshes/3d_hetero_seepage/definition.py`
  - `meshes/3d_hetero_seepage/family_b.msh`
  - `meshes/3d_hetero_seepage/concave_family_b.msh`
  - `meshes/3d_hetero_seepage/legacy/retag_to_canonical.py`

### `3d_hetero_seepage_transition`

- Active benchmarks:
  - `benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml`
  - `benchmarks/slope_stability_3D_homo_seepage_SSR_concave/case.toml`
- Geometry: `meshes/3d_hetero_seepage_transition/transition_default.msh`
- Materials: four named mechanical material models assigned through logical regions
- Mechanics BCs: logical boundaries `x_lock`, `z_lock`, `base`; named profiles `roller_base` and `fixed_base`
- Seepage BCs: node-support-based heads `head_dry`, `head_porous`, `head_free`; uniform conductivity `[1.0]`
- Health: `sound`
- Evidence:
  - `meshes/3d_hetero_seepage_transition/definition.py`
  - `meshes/3d_hetero_seepage_transition/transition_default.msh`
  - `meshes/3d_hetero_seepage_transition/legacy/convert_to_msh.py`

### `3d_hetero_slope`

- Active benchmarks:
  - `benchmarks/run_3D_hetero_SSR_capture/case.toml`
  - `benchmarks/slope_stability_3D_hetero_LL/case.toml`
  - `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`
- Geometry: canonical variants `adaptive_family_a_*`, `uniform_family_a`, `adaptive_family_b_*`, `uniform_family_b`
- Materials: four named material models mapped from logical regions `cover_layer`, `general_foundation`, `weak_foundation`, `slope_mass`
- Mechanics BCs: logical boundaries `x_lock`, `base`, `z_lock`
- Seepage BCs: none
- Health: `sound`
- Evidence:
  - `meshes/3d_hetero_slope/definition.py`
  - `meshes/3d_hetero_slope/adaptive_family_a_l1.msh`
  - `meshes/3d_hetero_slope/adaptive_family_b_l1.msh`
  - `meshes/3d_hetero_slope/legacy/retag_to_canonical.py`

### `3d_homo_slope`

- Active benchmarks:
  - `benchmarks/slope_stability_3D_homo_LL/case.toml`
  - `benchmarks/slope_stability_3D_homo_SSR/case.toml`
  - `benchmarks/slope_stability_3D_homo_SSR_default/case.toml`
- Geometry: canonical variants `adaptive_family_a_*`, `uniform_family_a`, `adaptive_family_b_*`, `uniform_family_b`
- Materials: one material model `homogeneous_slope`; the canonical meshes collapse the physical volume to one logical region `slope_mass`
- Mechanics BCs: logical boundaries `x_lock`, `base`, `z_lock`
- Seepage BCs: none
- Health: `sound`
- Evidence:
  - `meshes/3d_homo_slope/definition.py`
  - `meshes/3d_homo_slope/adaptive_family_a_l1.msh`
  - `meshes/3d_homo_slope/adaptive_family_b_l1.msh`
  - `meshes/3d_homo_slope/legacy/retag_to_canonical.py`

### `3d_siopt`

- Active benchmarks:
  - `benchmarks/SIOPT_LL/case.toml`
  - `benchmarks/SIOPT_SSR/case.toml`
- Geometry: `reference_l0.msh`, `reference_l1.msh`, `reference_l5.msh`
- Materials: one named mechanical material model `siopt_reference`; canonical meshes expose logical region `reference_mass`
- Mechanics BCs: logical boundaries `x_lock`, `z_lock`, `base`; named profiles `roller_base` and `fixed_base`
- Seepage BCs: none
- Health: `sound`
- Evidence:
  - `meshes/3d_siopt/definition.py`
  - `meshes/3d_siopt/reference_l0.msh`
  - `meshes/3d_siopt/legacy/retag_to_canonical.py`

## Converter Inventory

The migration is reproducible from the repository itself.

- Index: `meshes/CONVERTERS.md`
- Per-asset rerun helpers: `meshes/<asset>/legacy/convert_to_msh.py` or `meshes/<asset>/legacy/retag_to_canonical.py`
- Per-asset source preservation: `meshes/<asset>/legacy/source/`

This means the canonical `.msh` files are runtime artifacts, but their provenance remains local to each asset folder.

## Generic Extension Points

The active runtime already exposes generic API slots for future loads and curved geometry:

- `NeumannBCSpec`
- `BoundaryGeometrySpec`
- `BoundaryGeometryPatch`
- scalar seepage flux BCs through the same generic shape as mechanics Neumann BCs

Current rules:

- if no geometry patch is referenced, loads integrate on the promoted simplex support faces
- if `boundary_geom:<name>` exists, the support boundary still determines the receiving DOFs
- the geometry patch determines the curved mapping and quadrature measure

Current production assets do not use Neumann loads yet, but the generic mesh/API path for them is already present in:

- `src/slope_stability/assets/api.py`
- `src/slope_stability/assets/factories.py`
- `src/slope_stability/assets/support/canonical_gmsh.py`

## Conclusion

The active repository is now aligned with the canonical mesh-local blueprint:

- all active problem definitions are `.msh` + `definition.py`
- all active benchmark assets are `sound`
- all active benchmark configs resolve through `problem.asset` and `mesh_variant`
- non-canonical source logic is preserved only for reproducibility under `legacy/`
- the runtime mesh/build path is generic and solver-facing, not problem-family-specific
