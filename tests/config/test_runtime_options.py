from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from petsc_ssr.options import PmgOptions, SsrOptions
from petsc_ssr.problem import ProblemSpec
from petsc_ssr.runtime.options import (
    quote_option_tokens,
    read_options_file,
    resolve_run_option_tokens,
    resolve_run_options_string,
)


def test_runtime_options_file_reader_is_comment_and_quote_aware(tmp_path: Path) -> None:
    opts = tmp_path / "profile.opts"
    opts.write_text(
        """
# full-line comment
-ksp_type fgmres
-log_view :logs/petsc_log.txt # trailing comment
-custom_path "value with spaces"
""",
        encoding="utf-8",
    )

    assert read_options_file(opts) == [
        "-ksp_type",
        "fgmres",
        "-log_view",
        ":logs/petsc_log.txt",
        "-custom_path",
        "value with spaces",
    ]


def test_runtime_option_resolver_owns_artifact_paths_and_solution_policy(tmp_path: Path) -> None:
    options_file = tmp_path / "profile.opts"
    options_file.write_text("-ksp_converged_reason\n", encoding="utf-8")
    options = replace(SsrOptions.current_baseline(), pmg=replace(PmgOptions.current_baseline(), options_file=options_file))
    problem = ProblemSpec.tiny_box()

    tokens = resolve_run_option_tokens(problem, options, tmp_path / "run", write_solution_vtu=False)

    assert tokens[:1] == ["-ksp_converged_reason"]
    assert "-curve_csv" in tokens
    assert str(tmp_path / "run" / "data" / "continuation_curve.csv") in tokens
    assert "-summary_json" in tokens
    assert str(tmp_path / "run" / "data" / "summary.json") in tokens
    assert "-solution_vtk" not in tokens
    assert "-solution_binary" not in tokens


def test_runtime_option_string_quotes_paths_once(tmp_path: Path) -> None:
    options_file = tmp_path / "profile.opts"
    options_file.write_text("", encoding="utf-8")
    options = replace(SsrOptions.current_baseline(), pmg=replace(PmgOptions.current_baseline(), options_file=options_file))
    output_dir = tmp_path / "run with spaces"

    options_string = resolve_run_options_string(ProblemSpec.tiny_box(), options, output_dir, write_solution_vtu=True)

    assert quote_option_tokens(["-x", "value with spaces"]) == "-x 'value with spaces'"
    assert f"'{output_dir / 'data' / 'continuation_curve.csv'}'" in options_string
    assert f"'{output_dir / 'exports' / 'final_solution.vtu'}'" in options_string
