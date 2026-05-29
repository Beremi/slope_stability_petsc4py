# HPC

Production and HPC use should keep the Python footprint minimal. Runtime
dependencies are the PETSc/MPI stack plus the package runtime dependencies;
mesh conversion, plotting, notebooks, reports, and HDF5/debug tooling are
optional extras.

Use the minimal install for solver runs, then add extras only on machines that
need them:

```bash
pip install .
pip install .[mesh]
pip install .[hdf5]
pip install .[reports]
pip install .[notebooks]
```

Suites are the place for machine policy. Put ranks, repeats, resource labels,
time limits, launchers, environment pins, and collection policy in
`benchmarks/suites/*.toml`, not in case TOMLs. Case TOMLs should remain portable
mathematical descriptions that can be resolved on a login node before an
allocation is used.

Recommended preflight on a login node:

```bash
petsc-ssr doctor
petsc-ssr case validate benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr case explain benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
petsc-ssr suite expand benchmarks/suites/local-32-smoke.toml \
  --output .local/runs/local-32-smoke/manifest.json
```

For production launches, enable PETSc logging and options-left collection from
the suite, profile, or launcher so reports can detect unused PETSc options and
summarize PETSc events. Resolved manifests should be archived with the run; they
record concrete PMG active ranks, requested-vs-concrete PC variants,
environment, profile selections, the concrete suite resource/launcher selected
for each rank count, and artifact paths. Completed suite runs also write a
per-run `command.json` provenance file so copied run directories remain
self-describing.

The committed HPC-oriented suite scaffold is:

```bash
petsc-ssr suite expand benchmarks/suites/hpc-strong-scaling.toml \
  --output .local/runs/hpc-strong-scaling/manifest.json
```

Do not use case-level machine fields as a shortcut for scheduler metadata. Add
or update a suite instead.
