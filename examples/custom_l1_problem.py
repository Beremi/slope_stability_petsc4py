from __future__ import annotations

from petsc_ssr import BoundarySpec, MaterialSpec, ProblemSpec


def build_problem() -> ProblemSpec:
    return ProblemSpec.l1_slope(refine_levels=0).__class__(
        name="custom_l1_same_materials",
        mesh_path=ProblemSpec.l1_slope().mesh_path,
        refine_levels=0,
        boundary=BoundarySpec("rollers"),
        materials=(
            MaterialSpec(1, 15.0, 38.0, 0.0, 50000.0, 0.30, 22.0),
            MaterialSpec(2, 10.0, 35.0, 0.0, 50000.0, 0.30, 21.0),
            MaterialSpec(3, 18.0, 32.0, 0.0, 20000.0, 0.33, 20.0),
            MaterialSpec(4, 15.0, 30.0, 0.0, 10000.0, 0.33, 19.0),
        ),
        metadata={"element": "P4", "note": "scriptable example matching baseline"},
    )
