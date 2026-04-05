# 3D heterogeneous seepage

Capture seepage-only results for the 3D heterogeneous water-level problem.

## Run

```bash
./run.sh
```

## Source

- MATLAB driver: `run_3D_hetero_seepage_capture`
- PETSc config: [`case.toml`](case.toml)

## Notes

Current PETSc seepage implementation is scalar; under MPI the benchmark executes on rank 0
only.
