# Karolina Scaling Diagnosis

The PETSc-native tangent assembly path is not the part that breaks the current
scaling. In the 2026-05-12/13 Karolina runs, tangent assembly improved almost
linearly up to 256 ranks, while PMG-shell coarse work and preconditioner apply
became much more expensive once the run crossed nodes.

## One-node scaling plus 256-rank two-node point

| ranks | nodes | wall s | linear solve s | PC setup s | PC apply s | tangent s | force s | linear iters | PMG fine s | PMG mid s | PMG coarse hypre s | transfer s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 419.95 | 194.76 | 50.06 | 145.59 | 28.00 | 12.32 | 899 | 112.75 | 4.41 | 2.73 | 1.09 |
| 64 | 1 | 252.06 | 113.62 | 28.08 | 84.20 | 13.55 | 10.36 | 950 | 62.75 | 2.55 | 4.53 | 0.91 |
| 128 | 1 | 183.60 | 79.69 | 18.92 | 60.14 | 7.52 | 9.84 | 1022 | 37.71 | 2.18 | 11.04 | 0.31 |
| 256 | 2 | 369.73 | 170.01 | 117.06 | 149.54 | 3.74 | 25.57 | 999 | 21.95 | 5.15 | 108.15 | 3.34 |

The 128 -> 256 rank regression is therefore mainly in the PMG coarse solve and
PMG setup/apply path. Assembly keeps improving, and orthogonalization improves.

## Same total ranks, packed on one node vs split across two nodes

| total ranks | layout | wall s | linear solve s | PC setup s | PC apply s | tangent s | force s | PMG coarse hypre s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1x32 | 419.95 | 194.76 | 50.06 | 145.59 | 28.00 | 12.32 | 2.73 |
| 32 | 2x16 | 453.84 | 229.88 | 49.67 | 168.45 | 27.96 | 12.90 | 18.31 |
| 64 | 1x64 | 252.06 | 113.62 | 28.08 | 84.20 | 13.55 | 10.36 | 4.53 |
| 64 | 2x32 | 303.74 | 150.87 | 37.69 | 120.02 | 13.54 | 14.63 | 32.67 |
| 128 | 1x128 | 183.60 | 79.69 | 18.92 | 60.14 | 7.52 | 9.84 | 11.04 |
| 128 | 2x64 | 267.93 | 140.07 | 43.07 | 118.12 | 7.55 | 14.70 | 61.29 |

At fixed MPI rank count, splitting across two nodes leaves tangent assembly
nearly unchanged but makes PMG coarse hypre, preconditioner apply, and force
communication worse. This points to node-boundary communication and the coarse
AMG phase, not to the new matrix assembly backend.

## What to test next

The next screen should keep the PETSc-native AIJ assembly and compare
preconditioners that reduce or avoid the fragile PMG-shell coarse hypre phase:

- PMG-shell control with hypre coarse solve.
- PMG-shell with `PCREDUNDANT` + MUMPS coarse solve, especially one redundant
  coarse group per node.
- Direct AIJ hypre with PMIS/ext+i-mm and accepted-step preconditioner rebuild.
- Direct AIJ GAMG with lower-communication coarsening options and interpolation
  reuse.
- MATIS/BDDC with local ILU subsolves.
- MATIS/BDDC with approximate local GAMG subsolves.

Use generated layouts `1x128`, `2x64`, and `2x128` to separate pure rank-count
effects from communication effects.
