# PMG GASM 3D Hetero SSR P4(L1) Experiment

This directory holds a short local comparison for the PMG socket-aggregate smoother prototype.

Run both cases with four MPI ranks and two fake socket aggregates:

```bash
./benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/run.sh
```

Run the fake `8 sockets x 4 ranks` shape locally with:

```bash
RANKS=32 FAKE_SOCKETS=8 ./benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/run.sh
```

Use `RUN_BASELINE=0` to run only the GASM case.

The baseline keeps the existing `pmg_shell` smoother. The GASM case adds `pmg_smoother_pc_type = "gasm"` with `pmg_smoother_gasm_total_subdomains = 2`, so ranks are tested as two contiguous fake socket groups. The experimental GASM subdomain solver is `preonly+jacobi`; the earlier `richardson(3)+jacobi` variant is much slower and should be used only as a stress comparison.

## Karolina one-node Qexp grid

The Karolina script prepares full P4(L1) SSR runs up to `omega_max = 7.0e6` and submits one Slurm array task per case, each with a 20 minute limit:

- baseline: `16`, `32`, `64`, `128` ranks;
- GASM: `1x16`, `2x16`, `4x16`, `8x16` fake socket aggregates.

Submit from the repository root on Karolina:

```bash
sbatch benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/karolina_one_node_qexp.sbatch
```

The script uses `qcpu_exp`, account `fta-26-40`, and QoS `3571_6328`, matching the existing Karolina templates in this checkout. Override those at submit time if your allocation changes:

```bash
sbatch --account=<account> --qos=<qos> benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/karolina_one_node_qexp.sbatch
```

Outputs are written under:

```text
artifacts/experiments/pmg_gasm_karolina_qexp_one_node_p4_l1_omega7/
```

After the array finishes, refresh the comparison table with:

```bash
./.venv/bin/python benchmarks/experiment_pmg_gasm_3D_hetero_SSR_P4_L1/summarize_karolina_qexp.py
```

The generated tables are `summary.md` and `summary.tsv` in the output directory.
