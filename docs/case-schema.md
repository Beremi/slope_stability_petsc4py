# Case Schema

`benchmarks/cases/<slug>/case.toml` is the public mathematical case surface.
It must stay short enough that a benchmark review can see the physical problem,
asset, element order, and selected profiles without reading launcher policy.

Allowed active sections:

- `[case]`: canonical `id`, `title`, and orthogonal tags. The loader still
  accepts legacy `name` during migration, but committed benchmark cases should
  use `id`.
- `[mesh]`: `asset`, `variant`, `element`, and optional mesh asset `profile`.
- `[physics.mechanics]`: mechanics model, Davis choice, and optional physics profile.
- `[physics.seepage]`: seepage model name.
- `[continuation]`: continuation profile plus case-specific mathematical load
  and cap controls such as `omega_max`, `lambda_init`, and step-size limits.
- `[newton]`: Newton profile only. Iteration caps, tolerances, stopping,
  damping, algorithm, and line-search policy live in Newton profiles.
- `[linear]`: linear solver profile only.
- `[output]`: named output preset.
- `[seepage]`: seepage runtime profile only for supported seepage cases.

The old internal `[problem]`, `[execution]`, `[linear_solver]`, `[export]`,
`[[materials]]`, and `[case_data]` shape is not part of the public benchmark
model. It can be loaded only for explicit migration/debug compatibility with
`PETSC_SSR_ALLOW_LEGACY_CASE_SCHEMA=1`.

Modern cases resolve to the maintained `petsc_ssr_full_c` native backend and
`native_dmplex` mesh ordering internally. The older `legacy_array` backend name
and Python `block_metis` ordering are retained only for explicit legacy-schema
migration/debug runs.

The schema rejects launcher, machine, and artifact concerns in cases: MPI ranks,
nodes, wall time, machine names, generated output paths, raw output arrays,
notebook settings, mesh refinement defaults, partitioners, and low-level linear
solver tuning belong elsewhere. One-off solver experiments should use a profile,
suite override, CLI debug override, or explicit PETSc option append rather than
committed case fields.
Continuation selectors such as `method`, `predictor`, and
`omega_step_controller` also belong in continuation profiles, not cases.
Newton iteration, tolerance, stopping, damping, algorithm, and line-search
policy belong in Newton profiles, not cases.

Tags must not duplicate structured state already represented by sections, such
as dimension, element order, analysis type, or solver family. Use tags for
orthogonal classifications like `regression`, `scaling`, `validation`, `slow`,
`nightly`, and `experimental`.

Profile-owned defaults should not be repeated in committed cases. If a
continuation, Newton, or seepage runtime value equals the selected profile
default, remove it from the case TOML. If a policy really needs a new reusable
default, create or select a profile instead of copying that policy into every
case.
Limit-load mechanics cases resolve to direct continuation and fixed-load Newton
profiles; indirect SSR profiles are rejected for LL cases because normal runs
must follow the resolved native algorithm policy.

Validate cases with:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case validate \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case validate --all
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case explain \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
```

Dry-runs write the same resolved preflight bundle that benchmark manifests and
native startup consume:

```text
data/problem.json
data/resolved_config.toml
data/resolved_options.txt
data/resolved_run_manifest.json
data/environment.json
data/native_problem_manifest.json
data/mechanics_bc_labels.csv
exports/resolved_config.toml
```

The resolved linear section records both the requested profile backend and the
concrete native PC variant. This is especially important for P1 cases, where a
profile PMG request is recorded as requested PMG with a concrete GAMG fallback.
Resolved continuation, Newton, and linear sections also record profile-owned
algorithm names, so algorithm selection remains inspectable without adding
algorithm fields to case TOMLs.

Seepage-only dry-runs use the same contract and additionally write
`data/hydro_options.txt` plus `data/seepage_boundary_labels.csv` when asset
head/flux supports are declared.

`mechanics_bc_nodes.csv` is an optional coordinate compatibility table for debug
and migration runs. Normal `run` and `case dry-run` invocations do not write or
pass it; use `--write-coordinate-bc-table` only when comparing against the old
coordinate-matching path. That flag also emits the native
`-debug_coordinate_bc_table true` guard; the C/PETSc engine rejects
`-mechanics_bc_nodes_csv` without it. If the path is supplied through
`native_problem_manifest.json`, the manifest must carry
`native_inputs.debug_coordinate_bc_table = true`. Native mechanics startup
requires the label table for manifest-declared Dirichlet rules.

Coupled seepage mechanics still uses a hydro-prepass pressure CSV until native
field/section ingestion lands. That pressure table is not implicit: resolved
options and manifests must record `seepage_pressure_source =
"hydro_prepass_coordinate_bridge"` next to the pressure CSV path. Coupled
dry-runs record the planned `hydro_prepass/data/coupled_pressure_nodes.csv`
bridge path and source contract even though the hydro prepass is not executed.
