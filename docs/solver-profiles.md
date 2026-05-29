# Solver Profiles

Cases select profiles; profiles own reusable algorithm policy.

Profile directories:

- `configs/continuation_profiles`: continuation method, predictor, step control,
  initialization defaults, and continuation caps.
- `configs/newton_profiles`: nonlinear iteration, stopping, regularization, and
  line-search policy.
- `configs/seepage_profiles`: Darcy/seepage runtime tolerances and iteration
  caps for hydro prepasses.
- `configs/solver_profiles`: linear/KSP/PC/deflation/PMG policy and PETSc
  option defaults.
- `configs/petsc`: raw PETSc option files referenced by solver profiles.

Continuation and Newton profiles carry explicit `algorithm` selectors. The
current maintained continuation algorithms are `indirect` and `direct`; the
current Newton algorithms are `indirect-ssr` and `fixed-load`. Linear solver
profiles likewise carry `[linear].algorithm`, currently one of `ksp_deflated`,
`fgmres`, or `direct_debug`. These profile-level names are recorded in
`case explain`, suite manifests, `data/resolved_config.toml`, and
`data/resolved_run_manifest.json` so changing a profile, not a case or engine
patch, is the visible way to change algorithm policy. Those artifacts also
record the concrete native linear selector after PC/deflation resolution, for
example `native_algorithm = "pmg-deflated"` for P4 PMG-deflated runs or
`native_algorithm = "gamg"` for P1 PMG fallback runs. Mechanics run options
carry concrete native registry selectors such as `-continuation_algorithm`,
`-newton_algorithm`, and `-linear_algorithm`; the native engine validates those
selectors against the current method and PC variant before solving.
The maintained PMG raw options file is limited to PETSc-owned extras such as
DMPlex partition balancing and convergence-reason logging; solver policy lives
in the resolved profile fields.
Limit-load mechanics cases use the `direct-limit-load` continuation profile
and a `limit-load-regularized*` Newton profile so the normal path stays in the
native direct/fixed-load implementation; the Python loop is an explicit debug
compatibility path only. SSR cases likewise select `indirect-regularized*`
Newton profiles for reusable iteration caps, tolerances, and stopping criteria
instead of carrying those controls inline in case TOMLs.

Public linear solver profiles currently include:

- `pmg-deflated-baseline`: rank-adaptive PMG shell with cached deflation, used
  by normal SSR and limit-load benchmarks.
- `gamg-p1-baseline`: P1/GAMG validation profile for cases without a
  p-hierarchy.
- `direct-debug`: explicit debug profile for serial/direct PETSc LU checks.

The old `baseline-pmg-deflated` spelling is accepted as a compatibility alias
and resolves to `pmg-deflated-baseline`; new cases, suites, targets, and
manifests should use the canonical name.

The normal `petsc-ssr run` path uses the resolved profile choices. It does not
force the C baseline unless `--force-c-baseline` is passed explicitly as a debug
escape hatch.

Continuation step-control policy uses stable profile names. The maintained
classic controller is recorded as `omega_step_controller = "classic"` in
profiles and resolved artifacts; `legacy` is accepted only as a native/debug
compatibility alias and is normalized to `classic`.

PMG policy is profile-owned. The profile expresses rank policy plus concrete
apply backend, shell subcommunicator layout, P1 coarse solver policy, P1/P2
telescope policy, and smoother limits. Suite expansion or a concrete run
resolves the rank policy for the actual MPI size. Resolved choices are recorded
in suite `manifest.json`, per-run `data/resolved_config.toml`, and per-run
`data/resolved_run_manifest.json`.
If the native engine is launched without profile-generated concrete values, its
fallback PMG shell ranks are also derived from the MPI size rather than fixed
64/32 constants.
Completed native mechanics runs also copy the concrete linear selector, PC
variant, requested PC variant, fallback reason, PMG active-rank counts,
preconditioner-reuse flag, and deflation policy into `data/summary.json`. The
resolved manifests carry the fuller PMG policy, including coarse/telescope and
smoother settings, so completed and planned runs can be audited without opening
the raw PETSc options file.
Inside the native engine, the resolved linear profile is also mirrored into
typed PMG and deflation subcontexts before Newton dispatches through the linear
registry, so future PMG/KSP policy can move behind that boundary without
changing case TOMLs. Both PMG setup paths consume that resolved PMG view for
active layouts, telescope defaults, smoother policy, and coarse KSP defaults.

PC backend choices are profile-owned too. `case explain`, profile validation,
suite manifests, and resolved run artifacts expose both the requested and
concrete choice plus the concrete native linear selector.
`[pc].type = "pmg_shell"` resolves to
the native `pc_variant = "pmg"` for P2/P4 mechanics and seepage runs. P1 has no
p-hierarchy, so a PMG request resolves to the concrete `pc_variant = "gamg"`
with `requested_pc_variant = "pmg"` and
`pc_variant_fallback_reason = "p1_has_no_p_hierarchy"` in the resolved
artifacts.

Seepage cases select `[seepage].profile`; committed case TOMLs must not set
`linear_tolerance`, `linear_max_iter`, or `nonlinear_max_iter`. Put reusable
hydro solver policy in `configs/seepage_profiles/` and use suites or explicit
debug overrides for one-off experiments.
Pure seepage runs record the concrete seepage profile in the hydro resolved
artifacts; coupled SSR runs also record the selected seepage profile in the main
mechanics `data/resolved_config.toml` and `data/resolved_run_manifest.json`.
`case explain` and `case validate` also report the resolved seepage profile so
users can inspect hydro policy before launch.

Inspect available profiles with:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main doctor
```

Inspect one profile without a case launch:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main profile validate
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main profile explain \
  pmg-deflated-baseline --world-size 32 --element P4
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main profile explain \
  indirect-classic --kind continuation
```

`profile validate` loads every committed continuation, Newton, seepage, and
solver profile; solver profiles are checked across representative world sizes
and P1/P4 element orders so rank-adaptive PMG policy and PC fallbacks remain
reviewable. `benchmark init --check` runs the same profile-registry validation.
For solver profiles, `profile explain` resolves rank-adaptive PMG policy and
requested-vs-concrete PC variant for one supplied world size and element order.
This is the profile-only counterpart to `case explain`; it lets users review
algorithm policy before adding a profile to cases or suites.

Inspect one resolved case/profile combination with:

```bash
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case explain \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
```

When changing a profile, update or add schema tests for:

- unknown profile rejection;
- resolved PMG ranks for representative MPI sizes;
- resolved run manifest fields;
- PETSc option token consumers and options-left cleanliness.
