from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


ENGINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_L1_MESH = ENGINE_ROOT / "meshes" / "3d_hetero_slope" / "adaptive_family_a_l1.msh"


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    region: int
    c0: float
    phi_deg: float
    psi_deg: float
    young: float
    poisson: float
    gamma_sat: float
    gamma_unsat: float | None = None

    def option_tokens(self) -> list[str]:
        values = [
            self.c0,
            self.phi_deg,
            self.psi_deg,
            self.young,
            self.poisson,
            self.gamma_sat,
            self.gamma_sat if self.gamma_unsat is None else self.gamma_unsat,
        ]
        return [f"-material_region_{self.region}", ",".join(f"{value:.17g}" for value in values)]


@dataclass(frozen=True, slots=True)
class BoundarySpec:
    mode: str = "rollers"
    tag_base: int = 5
    tag_x_min: int = 2
    tag_x_max: int = 1
    tag_z_min: int = 3
    tag_z_max: int = 4

    def option_tokens(self) -> list[str]:
        return [
            "-mesh_bc_mode", self.mode,
            "-bc_tag_base", str(self.tag_base),
            "-bc_tag_x_min", str(self.tag_x_min),
            "-bc_tag_x_max", str(self.tag_x_max),
            "-bc_tag_z_min", str(self.tag_z_min),
            "-bc_tag_z_max", str(self.tag_z_max),
        ]


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    name: str
    mesh_path: Path
    dimension: int = 3
    element_degree: int = 4
    refine_levels: int = 0
    boundary: BoundarySpec = field(default_factory=BoundarySpec)
    materials: tuple[MaterialSpec, ...] = ()
    use_box_mesh: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError(f"Unsupported mechanics dimension={self.dimension}; expected 2 or 3")
        if self.element_degree not in (1, 2, 4):
            raise ValueError(f"Unsupported mechanics element_degree={self.element_degree}; expected 1, 2, or 4")

    @classmethod
    def l1_slope(cls, *, refine_levels: int = 0, mesh_path: str | Path | None = None) -> "ProblemSpec":
        return cls(
            name="3d_hetero_slope_l1",
            mesh_path=Path(mesh_path) if mesh_path is not None else DEFAULT_L1_MESH,
            dimension=3,
            element_degree=4,
            refine_levels=refine_levels,
            boundary=BoundarySpec("rollers"),
            materials=default_l1_materials(),
            metadata={"cell_label": "Cell Sets", "face_label": "Face Sets", "element": "P4"},
        )

    @classmethod
    def tiny_box(cls) -> "ProblemSpec":
        return cls(
            name="tiny_box",
            mesh_path=ENGINE_ROOT / "meshes" / "fixtures" / "tiny_box.msh",
            dimension=3,
            element_degree=4,
            boundary=BoundarySpec("rollers"),
            materials=default_l1_materials(),
            use_box_mesh=True,
            metadata={"element": "P4"},
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProblemSpec":
        materials = tuple(MaterialSpec(**item) for item in payload.get("materials", default_l1_material_dicts()))
        boundary_raw = payload.get("boundary", {"mode": "rollers"})
        return cls(
            name=str(payload.get("name", "scripted_problem")),
            mesh_path=Path(payload.get("mesh_path", DEFAULT_L1_MESH)),
            dimension=int(payload.get("dimension", payload.get("dim", 3))),
            element_degree=int(payload.get("element_degree", payload.get("degree", _degree_from_elem_type(payload.get("elem_type", "P4"))))),
            refine_levels=int(payload.get("refine_levels", 0)),
            boundary=BoundarySpec(**boundary_raw),
            materials=materials,
            use_box_mesh=bool(payload.get("use_box_mesh", False)),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ProblemSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_python_file(cls, path: str | Path, factory: str = "build_problem") -> "ProblemSpec":
        module = _load_module(Path(path))
        build = getattr(module, factory)
        problem = build()
        if isinstance(problem, cls):
            return problem
        if isinstance(problem, dict):
            return cls.from_dict(problem)
        raise TypeError(f"{factory}() must return ProblemSpec or dict, got {type(problem)!r}")

    def option_tokens(self) -> list[str]:
        tokens: list[str] = []
        if self.use_box_mesh:
            tokens.extend(["-use_box_mesh", "true"])
        else:
            tokens.extend(["-mesh", str(self.mesh_path)])
        tokens.extend(["-refine_levels", str(self.refine_levels)])
        tokens.extend(["-mechanics_dim", str(self.dimension)])
        tokens.extend(["-element_degree", str(self.element_degree)])
        tokens.extend(self.boundary.option_tokens())
        for material in self.materials:
            tokens.extend(material.option_tokens())
        seepage_pressure_csv = self.metadata.get("seepage_pressure_csv")
        if seepage_pressure_csv:
            tokens.extend(["-seepage_pressure_csv", str(seepage_pressure_csv)])
        seepage_grho = self.metadata.get("seepage_grho")
        if seepage_grho is not None:
            tokens.extend(["-seepage_grho", f"{float(seepage_grho):.17g}"])
        mechanics_bc_nodes_csv = self.metadata.get("mechanics_bc_nodes_csv")
        if mechanics_bc_nodes_csv:
            tokens.extend(["-mechanics_bc_nodes_csv", str(mechanics_bc_nodes_csv)])
        return tokens

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mesh_path"] = str(self.mesh_path)
        return payload


def default_l1_materials() -> tuple[MaterialSpec, ...]:
    return tuple(MaterialSpec(**item) for item in default_l1_material_dicts())


def default_l1_material_dicts() -> list[dict[str, float | int]]:
    return [
        {"region": 1, "c0": 15.0, "phi_deg": 38.0, "psi_deg": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 22.0, "gamma_unsat": 22.0},
        {"region": 2, "c0": 10.0, "phi_deg": 35.0, "psi_deg": 0.0, "young": 50000.0, "poisson": 0.30, "gamma_sat": 21.0, "gamma_unsat": 21.0},
        {"region": 3, "c0": 18.0, "phi_deg": 32.0, "psi_deg": 0.0, "young": 20000.0, "poisson": 0.33, "gamma_sat": 20.0, "gamma_unsat": 20.0},
        {"region": 4, "c0": 15.0, "phi_deg": 30.0, "psi_deg": 0.0, "young": 10000.0, "poisson": 0.33, "gamma_sat": 19.0, "gamma_unsat": 19.0},
    ]


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load problem module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _degree_from_elem_type(elem_type: object) -> int:
    text = str(elem_type).strip().upper()
    if text.startswith("P"):
        try:
            return int(text[1:])
        except ValueError:
            pass
    raise ValueError(f"Unsupported mechanics element type {elem_type!r}; expected P1, P2, or P4")
