/*
   Modular wrapper for the standalone P4 indirect SSR solver.

   The implementation is split into ordered fragments so the production code is
   readable by subsystem while still compiling as one translation unit. Keeping
   one translation unit preserves the proven static helper visibility and avoids
   any numerical or performance change from the refactor itself.
*/
#include "context.c.inc"
#include "../io/problem_manifest.c.inc"
#include "../profiling/stats.c.inc"
#include "../algorithms/registry.c.inc"
#include "public_engine.c.inc"
#include "../linear/pmg_shell.c.inc"
#include "../linear/deflation_krylov.c.inc"
#include "../reporting/reporting.c.inc"
#include "../nonlinear/newton_fixed_load.c.inc"
#include "../nonlinear/newton_indirect_ssr.c.inc"
#include "../continuation/continuation_indirect_ssr.c.inc"
#include "../continuation/continuation_direct_ssr.c.inc"
#include "../cython/cython_api.c.inc"
#include "../replay/replay_debug.c.inc"
#include "cli_runner.c.inc"
