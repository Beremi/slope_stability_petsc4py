from __future__ import annotations

import argparse
from pathlib import Path

from petsc_ssr.context import SsrContext
from petsc_ssr.options import SsrOptions
from petsc_ssr.problem import ProblemSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone PETSc SSR engine.")
    parser.add_argument("--problem-json", type=Path)
    parser.add_argument("--problem-py", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--element-degree", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--refine-levels", type=int, default=0)
    parser.add_argument("--use-box-mesh", action="store_true")
    parser.add_argument("--omega-max", type=float, default=7.0e6)
    parser.add_argument("--analysis", choices=["ssr", "ll"], default="ssr")
    parser.add_argument("--continuation-method", choices=["indirect", "direct"], default=None)
    parser.add_argument("--lambda-ell", type=float, default=1.0)
    parser.add_argument("--d-t-min", type=float, default=1.0e-3)
    parser.add_argument("--d-omega-ini-scale", type=float, default=0.2)
    parser.add_argument("--continuation-step-max", type=int, default=100)
    parser.add_argument("--linear-rtol", type=float, default=1.0e-1)
    parser.add_argument("--ksp-max-it", type=int, default=200)
    parser.add_argument("--pc-variant", default="pmg")
    parser.add_argument("--deflation", choices=["true", "false"], default="true")
    parser.add_argument("--algorithm", choices=["c", "python-loop"], default="c")
    parser.add_argument("--output-dir", type=Path, default=Path(".local/tmp/ssr_engine_run"))
    parser.add_argument("--petsc-opt", action="append", default=[], help="Additional raw PETSc option token.")
    return parser


def problem_from_args(args: argparse.Namespace) -> ProblemSpec:
    if args.problem_json:
        return ProblemSpec.from_json(args.problem_json)
    if args.problem_py:
        return ProblemSpec.from_python_file(args.problem_py)
    if args.use_box_mesh:
        problem = ProblemSpec.tiny_box()
        return ProblemSpec(
            name=problem.name,
            mesh_path=problem.mesh_path,
            dimension=problem.dimension,
            element_degree=args.element_degree,
            refine_levels=problem.refine_levels,
            boundary=problem.boundary,
            materials=problem.materials,
            use_box_mesh=problem.use_box_mesh,
            metadata=problem.metadata,
        )
    problem = ProblemSpec.l1_slope(refine_levels=args.refine_levels, mesh_path=args.mesh)
    return ProblemSpec(
        name=problem.name,
        mesh_path=problem.mesh_path,
        dimension=problem.dimension,
        element_degree=args.element_degree,
        refine_levels=problem.refine_levels,
        boundary=problem.boundary,
        materials=problem.materials,
        use_box_mesh=problem.use_box_mesh,
        metadata=problem.metadata,
    )


def options_from_args(args: argparse.Namespace) -> SsrOptions:
    opts = SsrOptions.current_baseline(omega_max=args.omega_max)
    opts.analysis = args.analysis
    continuation_method = args.continuation_method or ("direct" if args.analysis == "ll" else "indirect")
    opts.continuation_method = continuation_method
    opts.continuation_algorithm = continuation_method
    opts.newton_algorithm = "fixed-load" if continuation_method == "direct" else "indirect-ssr"
    opts.lambda_ell = args.lambda_ell
    opts.d_t_min = args.d_t_min
    opts.d_omega_ini_scale = args.d_omega_ini_scale
    opts.continuation_step_max = args.continuation_step_max
    opts.linear.rtol = args.linear_rtol
    opts.linear.max_it = args.ksp_max_it
    opts.linear.deflation = args.deflation == "true"
    opts.pc_variant = args.pc_variant
    opts.petsc_options.extend(args.petsc_opt)
    return opts


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    problem = problem_from_args(args)
    options = options_from_args(args)
    with SsrContext(problem, options, output_dir=args.output_dir) as ctx:
        result = ctx.run_python_loop() if args.algorithm == "python-loop" else ctx.run()
        if ctx.rank == 0:
            summary = result.summary
            print(
                "SSR_ENGINE_RESULT "
                f"steps={summary.get('accepted_steps', '')} "
                f"omega_last={float(summary.get('omega_last', 0.0)):.8e} "
                f"lambda_last={float(summary.get('lambda_last', 0.0)):.8e} "
                f"newton_its={summary.get('total_newton_its', '')} "
                f"linear_its={summary.get('total_linear_its', '')} "
                f"csv={result.curve_csv}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
