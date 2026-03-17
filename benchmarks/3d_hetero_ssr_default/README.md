# 3d_hetero_ssr_default

Runnable case configuration.

Files:
- `case.toml`
- `run.sh`

Run with:

```bash
./run.sh
```

By default outputs go to `artifacts/cases/3d_hetero_ssr_default/latest`.

Notebook workflow:

- `pyvista_workflow.ipynb` walks the case from editable config sections through a parallel run and PyVista post-processing.
- `notebook_support.py` contains the helper functions used by the notebook.

Archived exploratory scripts and reports:

- Post-benchmark preconditioner investigations live under `archive/`.
- The main benchmark root keeps only the original case, scaling, and reporting files.
