from __future__ import annotations

from petsc_ssr import ProblemSpec, SsrContext, SsrOptions


def main() -> int:
    problem = ProblemSpec.l1_slope(refine_levels=0)
    options = SsrOptions.current_baseline(omega_max=7.0e6)
    with SsrContext(problem, options, output_dir=".local/tmp/l1_omega7") as ctx:
        result = ctx.run()
        if ctx.rank == 0:
            summary = result.summary
            print(
                f"steps={summary.get('accepted_steps')} omega={float(summary.get('omega_last', 0.0)):.8e} "
                f"lambda={float(summary.get('lambda_last', 0.0)):.8e} linear={summary.get('total_linear_its')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
