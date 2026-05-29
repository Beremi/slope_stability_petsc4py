from __future__ import annotations

import pytest

from petsc_ssr.config.validators import normalize_output_preset, reject_unknown_fields, validate_case_metadata


def test_shared_output_preset_normalizer_keeps_section_context() -> None:
    assert normalize_output_preset("Performance", section_name="[overrides.output]") == "performance"

    with pytest.raises(ValueError, match=r"\[overrides\.output\]\.preset"):
        normalize_output_preset("full-debug", section_name="[overrides.output]")


def test_shared_unknown_field_validator_reports_section() -> None:
    with pytest.raises(ValueError, match=r"\[linear\].*tolerance.*solver profiles"):
        reject_unknown_fields("[linear]", {"profile": "baseline", "tolerance": 0.1}, {"profile"}, "solver profiles own policy.")


def test_shared_case_metadata_validator_rejects_runtime_fields_and_structured_tags() -> None:
    with pytest.raises(ValueError, match="suite/launcher/artifact"):
        validate_case_metadata({"name": "bad", "ranks": [1, 2]}, mesh={}, physics={})

    with pytest.raises(ValueError, match="duplicate structured state"):
        validate_case_metadata(
            {"name": "bad", "tags": ["3d", "p4", "ssr"]},
            mesh={"asset": "3d_hetero_slope", "element": "P4"},
            physics={"mechanics": {"model": "mohr_coulomb_ssr"}},
        )
