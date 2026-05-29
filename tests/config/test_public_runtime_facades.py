from __future__ import annotations

from petsc_ssr.options import SsrOptions as LegacySsrOptions
from petsc_ssr.runtime.options import SsrOptions, flatten_tokens
from petsc_ssr.runtime.runner import run_case, run_case_main


def test_runtime_options_facade_reexports_typed_option_model() -> None:
    assert SsrOptions is LegacySsrOptions
    assert flatten_tokens([["-ksp_type", "fgmres"], ["-pc_type", "gamg"]]) == [
        "-ksp_type",
        "fgmres",
        "-pc_type",
        "gamg",
    ]


def test_runtime_runner_facade_keeps_run_entrypoint_importable() -> None:
    assert run_case_main.__name__ == "main"
    assert callable(run_case)
