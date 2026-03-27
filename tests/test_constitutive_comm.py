from __future__ import annotations

from slope_stability.constitutive.problem import ConstitutiveOperator


def test_local_comm_prefers_owned_override() -> None:
    operator = ConstitutiveOperator.__new__(ConstitutiveOperator)
    operator._owned_comm = "sentinel"
    assert operator._local_comm() == "sentinel"
