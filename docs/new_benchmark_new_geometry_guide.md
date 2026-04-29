# New Benchmark On A New Geometry

This is the complete entry point for adding a benchmark that uses a new mesh geometry.

The rule is simple: benchmark configs choose an asset and numerical settings; mesh assets
own problem physics. A new benchmark should not require `src/` edits unless it needs a new
generic value model or boundary-condition evaluator.

## Asset-First Data Flow

1. `benchmarks/<benchmark>/case.toml` selects:
   - `problem.asset`
   - `problem.mesh_variant`
   - optional `problem.profile`
   - analysis, element type, solver, export, and notebook settings
2. `meshes/<asset>/definition.py` exports `ASSET`.
3. Runtime loads `ASSET`, resolves the selected mesh variant/profile, reads the canonical
   linear Gmsh `.msh`, elevates it to `P1`, `P2`, or `P4`, maps regions to material ids,
   maps boundaries/nodesets to solver masks, and builds seepage arrays from declared head
   boundary conditions.
4. Solver runners receive resolved arrays/specs only. They do not contain problem-specific
   geometry, material, hydraulic, or boundary-condition defaults.

## Required Files

For a new geometry:

- `meshes/<asset>/definition.py`
- `meshes/<asset>/<variant>.msh`
- `benchmarks/<benchmark>/case.toml`
- `benchmarks/<benchmark>/run.sh`
- `benchmarks/<benchmark>/README.md`

Update `tests/test_executable_asset_definitions.py` only when the asset is intended to be a
canonical required asset. Optional generated/support files are:

- `benchmarks/<benchmark>/simulation.ipynb`
- `benchmarks/<benchmark>/visualisation.ipynb`
- `meshes/<asset>/legacy/*`

## Canonical Gmsh Contract

Mesh files are linear Gmsh `MSH 4.1` files. Higher-order solver meshes are generated at
runtime.

Allowed physical-name prefixes:

- `region:<name>`: volume material regions
- `boundary:<name>`: boundary support groups
- `nodeset:<name>`: explicit support nodes
- `boundary_geom:<name>`: optional curved boundary geometry patches

Cell requirements:

- 2D volume cells: `triangle`
- 3D volume cells: `tetra`
- 2D support boundaries: `line`
- 3D support boundaries: `triangle`

Region names in the mesh must appear in `region_assignment`. Boundary or nodeset names used
by mechanics/seepage BCs must appear in the mesh.

## `definition.py`

Use `build_problem_asset_2d` for 2D and `build_problem_asset_3d` for 3D.

```python
from pathlib import Path

from slope_stability.assets.factories import build_problem_asset_2d, build_seepage_spec


ASSET_DIR = Path(__file__).resolve().parent

ASSET = build_problem_asset_2d(
    asset_id="2d_new_geometry",
    asset_dir=ASSET_DIR,
    default_variant="default.msh",
    mesh_variants={
        "default.msh": {"source": {"path": "default.msh"}},
    },
    materials={
        "soil": {
            "c0": 15.0,
            "phi": 30.0,
            "psi": 0.0,
            "young": 10000.0,
            "poisson": 0.33,
            "gamma_sat": 19.0,
            "gamma_unsat": 18.0,
            "hydraulic_conductivity": 1.0,
        },
    },
    region_assignment={
        "slope_mass": "soil",
    },
    mechanics={
        "dirichlet": [
            {"target": "left", "components": ["x"]},
            {"target": "base", "components": ["x", "y"]},
        ],
        "hydraulic_state": {
            "kind": "constant_level",
            "level": 5.0,
        },
    },
    seepage=build_seepage_spec(
        water_unit_weight=9.81,
        conductivity_mode="by_material",
        head_bcs=[
            {"target": "upstream", "kind": "constant_level", "level": 8.0},
            {"target": "downstream", "kind": "dry"},
        ],
    ),
)
```

### Top-Level Asset Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `asset_id` | yes | Stable id used by `case.toml` as `problem.asset`. Use a lowercase descriptive name. |
| `asset_dir` | yes | Usually `Path(__file__).resolve().parent`. Relative mesh paths resolve from here. |
| `default_variant` | yes | Mesh variant name used when a config omits `mesh_variant`. |
| `mesh_variants` | yes | Mapping from variant name to source metadata. |
| `materials` | yes | Material model definitions. Mechanical and hydraulic fields live here. |
| `region_assignment` | yes | Maps Gmsh `region:<name>` groups to material names. |
| `mechanics` | optional | Mechanical boundary conditions, profiles, and hydraulic saturation state. |
| `seepage` | optional | Seepage physics and head boundary conditions. |
| `boundary_geometry` | optional | Curved boundary geometry patch metadata. |

The exported name must be `ASSET`.

### `mesh_variants`

Each key is a variant name. The standard value is:

```python
"default.msh": {"source": {"path": "default.msh"}}
```

`source.path` is relative to `asset_dir`. Variant names should usually match the file name.
Extra metadata keys may be present; runtime preserves them for diagnostics.

### `materials`

Materials may be a dictionary keyed by material name or a list of dictionaries with `name`.

Mechanical material fields:

| Field | Meaning |
| --- | --- |
| `c0` | cohesion |
| `phi` | friction angle in degrees |
| `psi` | dilation angle in degrees |
| `young` | Young's modulus |
| `poisson` | Poisson ratio |
| `gamma_sat` | saturated unit weight |
| `gamma_unsat` | unsaturated unit weight |

Hydraulic field:

| Field | Meaning |
| --- | --- |
| `hydraulic_conductivity` | Conductivity for `conductivity_mode="by_material"`. |

A mechanics-capable asset must define complete mechanical rows for all materials. A
seepage-only asset may define only hydraulic conductivity if it does not need mechanics.

### `region_assignment`

Maps mesh region names to material names:

```python
region_assignment = {
    "slope_mass": "soil",
    "weak_layer": "weak_soil",
}
```

The keys are logical names without the `region:` prefix. The referenced material names must
exist in `materials`.

### `mechanics`

Supported keys:

| Key | Meaning |
| --- | --- |
| `dirichlet` | Default displacement constraints. |
| `neumann` | Default external boundary loads. Parsed generically; only use when the solver path supports the `kind`. |
| `profiles` | Named overrides of `dirichlet`/`neumann`. |
| `default_profile` | Profile selected when `case.toml` omits `problem.profile`. Defaults to `default`. |
| `hydraulic_state` | Saturation model for mechanics-only or seepage-coupled mechanics cases. |

Dirichlet entries:

```python
{"target": "base", "components": ["x", "y"]}
```

Options:

- `target`: boundary or nodeset logical name, without `boundary:` or `nodeset:`
- `components`: 2D accepts `x`, `y`; 3D accepts `x`, `y`, `z`
- `values`: optional list matching `components`; non-zero values are not wired through the
  current mechanics solver stack

Profiles:

```python
mechanics={
    "default_profile": "fixed_base",
    "dirichlet": [{"target": "base", "components": ["y"]}],
    "profiles": {
        "fixed_base": {
            "dirichlet": [{"target": "base", "components": ["x", "y", "z"]}],
        },
        "roller_base": {
            "dirichlet": [{"target": "base", "components": ["y"]}],
        },
    },
}
```

Hydraulic state value models:

```python
{"kind": "constant_level", "level": 5.0}
```

```python
{
    "kind": "piecewise_linear_level",
    "axis": "x",
    "points": [[0.0, 8.0], [20.0, 4.0]],
    "left_mode": "constant",
    "right_mode": "constant",
}
```

The saturation test uses `y <= level`.

### `seepage`

Prefer `build_seepage_spec(...)`.

| Field | Required | Meaning |
| --- | --- | --- |
| `water_unit_weight` | yes | Unit weight used to convert head to pressure. |
| `conductivity_mode` | yes | `by_material`, `uniform`, or `by_region`. |
| `conductivity` | only for `uniform` | Scalar/list conductivity values. |
| `region_conductivity` | only for `by_region` | Mapping from region name to conductivity. |
| `head_bcs` | normally yes | Seepage head boundary conditions. |
| `flux_bcs` | optional | Parsed as generic Neumann specs; only use when supported by the solver path. |

Conductivity modes:

- `by_material`: each material must define `hydraulic_conductivity`
- `uniform`: use `conductivity=[value]`
- `by_region`: use `region_conductivity={"region_name": value}`

Head BC entries:

```python
{"target": "head_porous", "kind": "constant_level", "level": 55.0}
```

```python
{"target": "head_dry", "kind": "dry"}
```

```python
{
    "target": "head_support",
    "kind": "piecewise_linear_level",
    "axis": "x",
    "points": [[0.0, 10.0], [20.0, 5.0]],
    "scope": "domain_below_head",
    "left_mode": "constant",
    "right_mode": "constant",
}
```

Supported head kinds:

- `dry`: fixes target nodes to zero pore pressure
- `constant_level`: pressure is `water_unit_weight * max(level - y, 0)`
- `piecewise_linear_level`: level is interpolated along `axis`, then converted to pressure

Supported `scope` values:

- omitted or `support_only`: apply values to target nodes only
- `domain_below_head`: apply the maximum pressure envelope over the full domain

## `case.toml`

`case.toml` contains benchmark metadata and numerical controls. It must not contain mesh
paths, material rows, or hydraulic constants.

Minimal example:

```toml
[benchmark]
title = "2D new geometry SSR"
matlab_script = "none"
comparison_kind = "continuation"
mpi_ranks = 1
suite = false

[notebook]
family = "2d_continuation"

[problem]
name = "new_geometry_ssr"
asset = "2d_new_geometry"
mesh_variant = "default.msh"
analysis = "ssr"
elem_type = "P2"
davis_type = "B"

[execution]
node_ordering = "block_metis"
mpi_distribute_by_nodes = true
constitutive_mode = "overlap"

[continuation]
method = "indirect"
lambda_init = 1.0
d_lambda_init = 0.1
omega_max = 1.2e7
step_max = 100

[newton]
it_max = 50
tol = 1e-4

[linear_solver]
solver_type = "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"
tolerance = 1e-1
max_iterations = 100

[export]
write_custom_debug_bundle = true
write_history_json = true
write_solution_vtu = true
```

### `[benchmark]`

| Key | Meaning |
| --- | --- |
| `title` | Human-readable name. |
| `matlab_script` | MATLAB parity driver name, or an identifying string for non-parity cases. |
| `comparison_kind` | `continuation` or `seepage`. Used by benchmark reporting. |
| `mpi_ranks` | Default ranks used by benchmark runners. |
| `suite` | Include in the canonical parity suite when `true`. |
| `notes` | Optional text shown in benchmark docs/reports. |

### `[notebook]`

| Key | Meaning |
| --- | --- |
| `family` | Notebook template family. |
| `jupyter_backend` | Optional visualization backend hint. |
| `nonlinear_surface_subdivision` | Optional visualization subdivision for nonlinear surfaces. |
| `surface_decimate_reduction` | Optional 3D visualization decimation amount. |
| `boundary_edge_overlay` | Optional boolean for boundary edge overlays. |

Notebook families:

- `2d_continuation`
- `2d_seepage`
- `2d_seepage_continuation`
- `3d_continuation`
- `3d_seepage`
- `3d_seepage_continuation`

### `[problem]`

| Key | Meaning |
| --- | --- |
| `name` | Stable case name. |
| `case` | Optional legacy/reporting label. |
| `asset` | Required mesh asset id. |
| `mesh_variant` | Optional mesh variant; defaults to the asset default. |
| `profile` | Optional asset mechanics profile. |
| `analysis` | `ssr`, `ll`, or `seepage`. |
| `elem_type` | `P1`, `P2`, or `P4`. |
| `davis_type` | Davis approach for mechanics, usually `B`. |

The loader derives dimension from `asset`.

### `[execution]`

| Key | Default | Meaning |
| --- | --- | --- |
| `node_ordering` | `block_metis` | `original` or `block_metis`. |
| `mpi_distribute_by_nodes` | `true` | Partition nodal ownership for distributed mechanics. |
| `constitutive_mode` | `overlap` | Constitutive assembly ownership mode. |
| `tangent_kernel` | `rows` | Tangent assembly kernel selection. |

### `[continuation]`

Main controls:

- `method`: `indirect` or solver-supported direct mode
- `predictor`: predictor selection, commonly `secant`
- `lambda_init`, `d_lambda_init`, `d_lambda_min`, `d_lambda_diff_scaled_min`
- `lambda_ell`, `omega_max`, `step_max`, `d_omega_ini_scale`, `d_t_min`
- `omega_step_controller`, `secant_correction_mode`, `first_newton_warm_start_mode`
- `omega_no_increase_newton_threshold`, `omega_half_newton_threshold`
- `omega_target_newton_iterations`, `omega_adapt_min_scale`, `omega_adapt_max_scale`
- `omega_hard_newton_threshold`, `omega_hard_linear_threshold`
- `omega_efficiency_floor`, `omega_efficiency_drop_ratio`, `omega_efficiency_window`
- `omega_hard_shrink_scale`, `step_length_cap_mode`, `step_length_cap_factor`
- `init_newton_stopping_criterion`, `init_newton_stopping_tol`
- `fine_newton_stopping_criterion`, `fine_newton_stopping_tol`
- `fine_switch_mode`, `fine_switch_distance_factor`

### `[newton]`

Controls:

- `it_max`, `it_damp_max`, `tol`, `r_min`
- `stopping_criterion`: `relative_residual`, `relative_correction`, or `absolute_delta_lambda`
- `stopping_tol`
- `line_search`: `alg5` or `armijo_residual`
- `armijo_alpha0`, `armijo_c1`, `armijo_shrink`, `armijo_max_ls`
- `armijo_rescale_trial_to_omega`, `armijo_fallback_to_alg5`

### `[linear_solver]`

Common controls:

- `solver_type`, `tolerance`, `max_iterations`, `deflation_basis_tolerance`
- `verbose`, `threads`, `print_level`, `use_as_preconditioner`
- `factor_solver_type`, `pc_backend`
- `preconditioner_matrix_source`, `preconditioner_matrix_policy`
- `preconditioner_rebuild_policy`, `preconditioner_rebuild_interval`
- `compiled_outer`, `recycle_preconditioner`

HYPRE/GAMG controls:

- `pc_gamg_process_eq_limit`, `pc_gamg_threshold`
- `pc_gamg_aggressive_coarsening`, `pc_gamg_aggressive_square_graph`
- `pc_gamg_aggressive_mis_k`
- `pc_hypre_coarsen_type`, `pc_hypre_interp_type`
- `pc_hypre_strong_threshold`, `pc_hypre_boomeramg_max_iter`
- `pc_hypre_P_max`, `pc_hypre_agg_nl`, `pc_hypre_nongalerkin_tol`

BDDC controls:

- `pc_bddc_symmetric`
- `pc_bddc_dirichlet_ksp_type`, `pc_bddc_dirichlet_pc_type`
- `pc_bddc_neumann_ksp_type`, `pc_bddc_neumann_pc_type`
- `pc_bddc_coarse_ksp_type`, `pc_bddc_coarse_pc_type`
- `pc_bddc_dirichlet_approximate`, `pc_bddc_neumann_approximate`
- `pc_bddc_monolithic`, `pc_bddc_coarse_redundant_pc_type`
- `pc_bddc_switch_static`, `pc_bddc_use_deluxe_scaling`
- `pc_bddc_use_vertices`, `pc_bddc_use_edges`, `pc_bddc_use_faces`
- `pc_bddc_use_change_of_basis`, `pc_bddc_use_change_on_faces`, `pc_bddc_check_level`

### `[seepage]`

Only numerical controls are allowed:

- `linear_tolerance`
- `linear_max_iter`
- `nonlinear_max_iter`

Physical seepage values belong in `meshes/<asset>/definition.py`.

### `[export]`

| Key | Default | Meaning |
| --- | --- | --- |
| `write_custom_debug_bundle` | `true` | Write `exports/run_debug.h5`. |
| `write_history_json` | `true` | Write continuation/debug history JSON. |
| `write_solution_vtu` | `true` | Write `exports/final_solution.vtu`. |
| `custom_debug_name` | `run_debug.h5` | Debug bundle file name. |
| `history_name` | `continuation_history.json` | History file name. |
| `solution_name` | `final_solution.vtu` | VTU file name. |

### Forbidden Config Fields

Do not put these in committed `case.toml` files:

- `[problem].dimension`
- `[problem].variant`
- `[problem].seepage`
- `[problem].mesh_path`
- `[problem].mesh_boundary_type`
- `[case_data]`
- `[[materials]]`
- `[seepage].water_unit_weight`
- `[seepage].conductivity`

If a new benchmark seems to need one of these, the data belongs in `meshes/<asset>/definition.py`.

## Validation Commands

```bash
python -m compileall -q src/slope_stability benchmarks tests tests_local
```

```bash
.venv/bin/pytest -q tests/test_executable_asset_definitions.py \
  tests/test_problem_asset_runtime.py \
  tests/test_problem_assets_seepage_mesh_folders.py
```

```bash
.venv/bin/pytest -q
```

Useful static checks:

```bash
.venv/bin/pytest -q tests/test_no_problem_specific_src.py
```

```bash
rg -n "mesh_path|\\[\\[materials\\]\\]|water_unit_weight|conductivity" benchmarks -g 'case.toml'
```

The last search should be empty for committed benchmark configs.

## When `src/` Edits Are Allowed

Do not edit `src/` for ordinary benchmark data. Edit `src/slope_stability/assets/evaluators.py`
only when the new mesh needs a genuinely new generic value model, such as a head function
that can be reused by multiple assets. Add tests for the new generic behavior and keep the
problem-specific numbers in `definition.py`.
