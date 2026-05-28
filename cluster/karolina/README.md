# Karolina Runner

This harness runs the self-contained PETSc SSR engine on Karolina. It defaults
to the maintained baseline profile and writes all artifacts into the chosen
`RUN_ROOT`.

Default login reminder:

```bash
ssh it4i-karolina-login1
```

Use Karolina, not Barbora, for these jobs.

## Submit

```bash
cd cluster/karolina
RUN_ROOT=/mnt/proj1/fta-26-40/petsc_ssr_$(date +%Y%m%d_%H%M%S) \
  NODE_COUNTS="1 2" \
  TIME_LIMIT=00:30:00 \
  ./submit_scaling.sh
```

## Collect

```bash
./collect_results.py "$RUN_ROOT"
```
