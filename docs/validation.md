# Validation

Recommended local validation:

```bash
python -m compileall -q src tools benchmarks/tools
bash -n tools/*.sh benchmarks/tools/*.sh cluster/karolina/*.sh cluster/karolina/*.sbatch
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make
PETSC_DIR=$PWD/.build/src/petsc-3.24.5 PETSC_ARCH=linux-c-opt make smoke
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case validate \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main case dry-run \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
PYTHONPATH=$PWD/src .venv/bin/python -m petsc_ssr.cli.main mesh-only \
  benchmarks/cases/3d-heterogeneous-ssr-p4/case.toml
```

Long performance checks compare 32-rank and 64-rank L1 runs against the targets
in `benchmarks/targets/`.
