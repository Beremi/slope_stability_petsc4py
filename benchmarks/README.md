# Benchmarks

Benchmark cases are stored under `benchmarks/cases/<slug>/`.

Each case contains a `case.toml`, generated `README.md`, `simulation.ipynb`, and
`visualisation.ipynb`.  Curated historical records live in `benchmarks/reports`;
machine-readable performance targets live in `benchmarks/targets`.

Run a case manually:

```bash
MPI_RANKS=4 benchmarks/tools/run_standalone_case.sh benchmarks/cases/2d-homogeneous-ssr \
  --continuation-step-max 3
```

Regenerate case documentation and notebooks:

```bash
PYTHONPATH=$PWD/src .venv/bin/python benchmarks/tools/generate_benchmark_readmes.py
PYTHONPATH=$PWD/src .venv/bin/python benchmarks/tools/generate_benchmark_notebooks.py
```
