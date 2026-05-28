from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve


ENGINE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMSOL_SEEPAGE_MESH = ENGINE_ROOT / "meshes" / "3d_hetero_seepage_transition" / "transition_default.msh"


@dataclass(slots=True)
class HydroMesh:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    elem_type: str
    nodesets: dict[str, np.ndarray] = field(default_factory=dict)
    boundary_groups: dict[str, np.ndarray] = field(default_factory=dict)
    region_names: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    @property
    def n_nodes(self) -> int:
        return int(self.coord.shape[1])

    @property
    def n_elements(self) -> int:
        return int(self.elem.shape[1])


@dataclass(slots=True)
class SeepageAssembly:
    dim: int
    n_nodes: int
    n_elem: int
    n_q: int
    n_int: int
    elem: np.ndarray
    weight: np.ndarray
    B: csr_matrix
    C: csr_matrix
    hatp: np.ndarray
    dphi: dict[str, np.ndarray]

    @property
    def hatphi_tiled(self) -> np.ndarray:
        return np.tile(self.hatp, (1, self.n_elem))


@dataclass(slots=True)
class HydroResult:
    mesh: HydroMesh
    pressure: np.ndarray
    grad_pressure: np.ndarray
    saturated_elements: np.ndarray
    history: dict[str, Any]
    assembly: SeepageAssembly
    q_free: np.ndarray
    prescribed_pressure: np.ndarray

    def summary(self) -> dict[str, Any]:
        criteria = list(self.history.get("criterion", []))
        init = dict(self.history.get("init_linear", {}))
        return {
            "nodes": int(self.mesh.n_nodes),
            "elements": int(self.mesh.n_elements),
            "elem_type": self.mesh.elem_type,
            "free_nodes": int(np.count_nonzero(self.q_free)),
            "dirichlet_nodes": int(self.q_free.size - np.count_nonzero(self.q_free)),
            "newton_iterations": int(self.history.get("iterations", 0)),
            "newton_converged": bool(self.history.get("converged", False)),
            "final_criterion": float(criteria[-1]) if criteria else float("nan"),
            "init_linear_iterations": int(init.get("iterations", 0)),
            "newton_linear_iterations": int(sum(int(v) for v in self.history.get("linear_iterations", []))),
            "pressure_min": float(np.min(self.pressure)) if self.pressure.size else 0.0,
            "pressure_max": float(np.max(self.pressure)) if self.pressure.size else 0.0,
            "saturated_elements": int(np.count_nonzero(self.saturated_elements)),
            "K_D_nnz": int(self.history.get("K_D_nnz", 0)),
            "runtime_seconds": float(self.history.get("runtime_seconds", 0.0)),
        }

    def write_outputs(self, output_dir: str | Path) -> tuple[Path, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / "hydro_summary.json"
        npz_path = out / "hydro_result.npz"
        summary_path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        np.savez_compressed(
            npz_path,
            coord=self.mesh.coord,
            elem=self.mesh.elem,
            pressure=self.pressure,
            grad_pressure=self.grad_pressure,
            saturated_elements=self.saturated_elements,
            q_free=self.q_free,
            prescribed_pressure=self.prescribed_pressure,
        )
        return summary_path, npz_path


def load_comsol_seepage_mesh(path: str | Path = DEFAULT_COMSOL_SEEPAGE_MESH, *, elem_type: str = "P2") -> HydroMesh:
    """Load the COMSOL seepage transition mesh without meshio.

    The source mesh is Gmsh 4.1 ASCII with physical point nodesets, surface
    groups, and volume regions. Only tetrahedra, boundary triangles, and point
    nodesets are needed for the seepage benchmark.
    """

    base = _load_gmsh41_simplex(Path(path))
    elem_key = str(elem_type).strip().upper()
    if elem_key == "P1":
        return HydroMesh(
            coord=base["coord"],
            elem=base["tet4"],
            surf=base["tri3"],
            elem_type="P1",
            nodesets=base["nodesets"],
            boundary_groups=base["boundary_groups"],
            region_names=base["region_names"],
        )
    if elem_key == "P2":
        coord, elem, surf = _elevate_tet4_mesh_to_tet10(base["coord"], base["tet4"], base["tri3"])
        nodesets = {name: _expand_nodeset(coord.shape[1], surf, nodes) for name, nodes in base["nodesets"].items()}
        return HydroMesh(
            coord=coord,
            elem=elem,
            surf=surf,
            elem_type="P2",
            nodesets=nodesets,
            boundary_groups=base["boundary_groups"],
            region_names=base["region_names"],
        )
    raise ValueError(f"Unsupported seepage element type {elem_type!r}; expected P1 or P2.")


def build_comsol_seepage_boundary(mesh: HydroMesh, *, water_unit_weight: float = 9.81) -> tuple[np.ndarray, np.ndarray]:
    q_w = np.ones(mesh.n_nodes, dtype=bool)
    pw_d = np.zeros(mesh.n_nodes, dtype=np.float64)
    y = np.asarray(mesh.coord[1, :], dtype=np.float64)
    for target, kind, level in (
        ("head_dry", "dry", None),
        ("head_porous", "constant_level", 55.0),
        ("head_free", "constant_level", 35.0),
    ):
        if target not in mesh.nodesets:
            raise KeyError(f"COMSOL seepage mesh does not contain nodeset {target!r}.")
        nodes = np.asarray(mesh.nodesets[target], dtype=np.int64)
        q_w[nodes] = False
        if kind == "dry":
            continue
        assert level is not None
        values = float(water_unit_weight) * np.maximum(float(level) - y, 0.0)
        pw_d[nodes] = np.maximum(pw_d[nodes], values[nodes])
    return q_w, pw_d


def solve_comsol_seepage(
    *,
    mesh_path: str | Path = DEFAULT_COMSOL_SEEPAGE_MESH,
    elem_type: str = "P2",
    linear_tolerance: float = 1.0e-10,
    linear_max_iter: int = 500,
    newton_max_it: int = 50,
    parse_only: bool = False,
) -> HydroResult | HydroMesh:
    start = perf_counter()
    mesh = load_comsol_seepage_mesh(mesh_path, elem_type=elem_type)
    if parse_only:
        return mesh
    q_w, pw_d = build_comsol_seepage_boundary(mesh, water_unit_weight=9.81)
    result = seepage_problem_3d(
        mesh.coord,
        mesh.elem,
        q_w,
        pw_d,
        grho=9.81,
        elem_type=mesh.elem_type,
        linear_tolerance=linear_tolerance,
        linear_max_iter=linear_max_iter,
        it_max=newton_max_it,
    )
    pw, grad_p, mater_sat, history, assembly = result
    history["runtime_seconds"] = perf_counter() - start
    return HydroResult(
        mesh=mesh,
        pressure=pw,
        grad_pressure=grad_p,
        saturated_elements=mater_sat,
        history=history,
        assembly=assembly,
        q_free=q_w,
        prescribed_pressure=pw_d,
    )


def print_hydro_result(result: HydroResult | HydroMesh) -> None:
    if isinstance(result, HydroMesh):
        print(
            "HYDRO_MESH "
            f"elem_type={result.elem_type} "
            f"nodes={result.n_nodes} "
            f"elements={result.n_elements} "
            f"surfaces={int(result.surf.shape[1])} "
            f"nodesets={','.join(sorted(result.nodesets))} "
            f"regions={','.join(str(name) for name in dict.fromkeys(result.region_names.tolist()))}"
        )
        return
    summary = result.summary()
    print(
        "HYDRO_RESULT "
        f"elem_type={summary['elem_type']} "
        f"nodes={summary['nodes']} "
        f"elements={summary['elements']} "
        f"free_nodes={summary['free_nodes']} "
        f"dirichlet_nodes={summary['dirichlet_nodes']} "
        f"newton_iterations={summary['newton_iterations']} "
        f"init_linear_iterations={summary['init_linear_iterations']} "
        f"newton_linear_iterations={summary['newton_linear_iterations']} "
        f"final_criterion={summary['final_criterion']:.8e} "
        f"pressure_min={summary['pressure_min']:.8e} "
        f"pressure_max={summary['pressure_max']:.8e} "
        f"saturated_elements={summary['saturated_elements']} "
        f"runtime={summary['runtime_seconds']:.6f}"
    )


def seepage_problem_3d(
    coord: np.ndarray,
    elem: np.ndarray,
    Q_w: np.ndarray,
    pw_D: np.ndarray,
    grho: float,
    *,
    elem_type: str = "P2",
    linear_tolerance: float = 1.0e-10,
    linear_max_iter: int = 500,
    it_max: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], SeepageAssembly]:
    del linear_tolerance, linear_max_iter
    assembly = assemble_auxiliary_matrices_3d(coord, elem, elem_type)
    assembly.dphi["_coord_y"] = np.asarray(coord[1, :], dtype=np.float64)
    eps = penalty_parameters_3d(coord, elem)
    eps_int = np.kron(eps, np.ones(assembly.n_q, dtype=np.float64))
    conduct0 = np.ones(assembly.n_int, dtype=np.float64)
    K_D, wc = _build_flow_stiffness(assembly, conduct0)
    f = _rhs_from_dirichlet(assembly, wc, pw_D, grho)
    q_free = np.asarray(Q_w, dtype=bool)
    pw_0 = np.zeros(assembly.n_nodes, dtype=np.float64)
    init_sol = spsolve(K_D[q_free][:, q_free], np.asarray(f[q_free], dtype=np.float64))
    pw_0[q_free] = np.asarray(init_sol, dtype=np.float64)
    pw_init = pw_0 + np.asarray(pw_D, dtype=np.float64)
    pw, history = newton_flow(
        pw_init,
        conduct0,
        Q_w,
        assembly,
        K_D,
        wc,
        eps_int,
        grho,
        it_max=it_max,
    )
    pw, grad_p, mater_sat = _finalize_seepage_outputs(assembly, pw, eps)
    history["K_D_nnz"] = int(K_D.nnz)
    history["init_linear"] = {"iterations": 1, "solver": "scipy_spsolve"}
    return pw, grad_p, mater_sat, history, assembly


def assemble_auxiliary_matrices_3d(coord: np.ndarray, elem: np.ndarray, elem_type: str) -> SeepageAssembly:
    xi, wf = quadrature_volume_3d(elem_type)
    hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d(elem_type, xi)
    elem = np.asarray(elem, dtype=np.int64)
    coord = np.asarray(coord, dtype=np.float64)
    n_p = int(elem.shape[0])
    n_elem = int(elem.shape[1])
    n_q = int(xi.shape[1])
    n_int = n_elem * n_q
    n_nodes = int(coord.shape[1])

    coord_x = coord[0, elem]
    coord_y = coord[1, elem]
    coord_z = coord[2, elem]

    dhat1_t = np.tile(dhat1, (1, n_elem))
    dhat2_t = np.tile(dhat2, (1, n_elem))
    dhat3_t = np.tile(dhat3, (1, n_elem))
    cx = np.repeat(coord_x, n_q, axis=1)
    cy = np.repeat(coord_y, n_q, axis=1)
    cz = np.repeat(coord_z, n_q, axis=1)

    j11 = np.sum(cx * dhat1_t, axis=0)
    j12 = np.sum(cy * dhat1_t, axis=0)
    j13 = np.sum(cz * dhat1_t, axis=0)
    j21 = np.sum(cx * dhat2_t, axis=0)
    j22 = np.sum(cy * dhat2_t, axis=0)
    j23 = np.sum(cz * dhat2_t, axis=0)
    j31 = np.sum(cx * dhat3_t, axis=0)
    j32 = np.sum(cy * dhat3_t, axis=0)
    j33 = np.sum(cz * dhat3_t, axis=0)

    det_j = j11 * (j22 * j33 - j23 * j32) - j12 * (j21 * j33 - j23 * j31) + j13 * (j21 * j32 - j22 * j31)
    inv_det = 1.0 / det_j

    dphi1 = ((j22 * j33 - j23 * j32) * dhat1_t - (j12 * j33 - j13 * j32) * dhat2_t + (j12 * j23 - j13 * j22) * dhat3_t) * inv_det
    dphi2 = (-(j21 * j33 - j23 * j31) * dhat1_t + (j11 * j33 - j13 * j31) * dhat2_t - (j11 * j23 - j13 * j21) * dhat3_t) * inv_det
    dphi3 = ((j21 * j32 - j22 * j31) * dhat1_t - (j11 * j32 - j12 * j31) * dhat2_t + (j11 * j22 - j12 * j21) * dhat3_t) * inv_det
    weight = np.tile(np.asarray(wf, dtype=np.float64), n_elem) * np.abs(det_j)

    hatphi = np.tile(hatp, (1, n_elem))
    int_ids = np.arange(n_int, dtype=np.int64)
    rows0 = 3 * int_ids
    rows1 = rows0 + 1
    rows2 = rows0 + 2

    b_rows: list[np.ndarray] = []
    b_cols: list[np.ndarray] = []
    b_vals: list[np.ndarray] = []
    c_rows: list[np.ndarray] = []
    c_cols: list[np.ndarray] = []
    c_vals: list[np.ndarray] = []

    for a in range(n_p):
        node_rep = np.repeat(elem[a, :], n_q)
        b_rows.extend((rows0, rows1, rows2))
        b_cols.extend((node_rep, node_rep, node_rep))
        b_vals.extend((dphi1[a, :], dphi2[a, :], dphi3[a, :]))
        c_rows.append(int_ids)
        c_cols.append(node_rep)
        c_vals.append(hatphi[a, :])

    B = coo_matrix(
        (np.concatenate(b_vals), (np.concatenate(b_rows), np.concatenate(b_cols))),
        shape=(3 * n_int, n_nodes),
    ).tocsr()
    B.eliminate_zeros()

    C = coo_matrix(
        (np.concatenate(c_vals), (np.concatenate(c_rows), np.concatenate(c_cols))),
        shape=(n_int, n_nodes),
    ).tocsr()
    C.eliminate_zeros()

    return SeepageAssembly(
        dim=3,
        n_nodes=n_nodes,
        n_elem=n_elem,
        n_q=n_q,
        n_int=n_int,
        elem=elem,
        weight=weight,
        B=B,
        C=C,
        hatp=hatp,
        dphi={"dphi1": dphi1, "dphi2": dphi2, "dphi3": dphi3},
    )


def newton_flow(
    pw_init: np.ndarray,
    conduct0: np.ndarray,
    Q_w: np.ndarray,
    assembly: SeepageAssembly,
    K_D: csr_matrix,
    wc: np.ndarray,
    eps_int: np.ndarray,
    grho: float,
    *,
    it_max: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, dict[str, Any]]:
    pw = np.asarray(pw_init, dtype=np.float64).copy()
    Q_w = np.asarray(Q_w, dtype=bool).ravel()
    eps_int = np.asarray(eps_int, dtype=np.float64).ravel()
    conduct0 = np.asarray(conduct0, dtype=np.float64).ravel()
    coord_y = np.asarray(assembly.dphi["_coord_y"], dtype=np.float64)
    denom = max(float(np.linalg.norm(pw_init)), 1.0e-14)
    history: dict[str, Any] = {
        "criterion": [],
        "linear_iterations": [],
        "linear_solve_time": [],
        "linear_solve_info": [],
        "converged": False,
    }

    it = 0
    while True:
        it += 1
        pw_int = _compute_pressure_at_integration_points(assembly, pw)
        perm_r = np.ones(assembly.n_int, dtype=np.float64)
        perm_r_der = np.zeros(assembly.n_int, dtype=np.float64)
        part1 = (pw_int < eps_int) & (pw_int > 0.0)
        part2 = pw_int <= 0.0
        perm_r[part1] = pw_int[part1] / eps_int[part1]
        perm_r[part2] = 0.0
        perm_r_der[part1] = 1.0 / eps_int[part1]

        K = _build_newton_matrix(assembly, conduct0, K_D, coord_y, perm_r_der, grho)
        f = _build_newton_rhs(assembly, wc, pw, coord_y, grho, perm_r)

        t0 = perf_counter()
        dp = np.zeros_like(pw)
        dp[Q_w] = spsolve(K[Q_w][:, Q_w], np.asarray(f[Q_w], dtype=np.float64))
        solve_time = perf_counter() - t0

        pw = pw + dp
        crit = float(np.linalg.norm(dp) / denom)
        history["criterion"].append(crit)
        history["linear_iterations"].append(1)
        history["linear_solve_time"].append(solve_time)
        history["linear_solve_info"].append({"iterations": 1, "solver": "scipy_spsolve"})
        if crit < tol:
            history["converged"] = True
            break
        if it >= int(it_max):
            break

    history["iterations"] = len(history["criterion"])
    return pw, history


def local_basis_volume_3d(elem_type: str, xi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    elem_key = str(elem_type).strip().upper()
    xi1 = xi[0, :]
    xi2 = xi[1, :]
    xi3 = xi[2, :]
    n_q = xi.shape[1]
    if elem_key == "P1":
        hatp = np.array([1 - xi1 - xi2 - xi3, xi1, xi2, xi3], dtype=np.float64)
        dhat1 = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float64)[:, None]
        dhat2 = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float64)[:, None]
        dhat3 = np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float64)[:, None]
        return hatp, dhat1, dhat2, dhat3
    if elem_key == "P2":
        xi0 = 1 - xi1 - xi2 - xi3
        hatp = np.array(
            [
                xi0 * (2 * xi0 - 1),
                xi1 * (2 * xi1 - 1),
                xi2 * (2 * xi2 - 1),
                xi3 * (2 * xi3 - 1),
                4 * xi0 * xi1,
                4 * xi1 * xi2,
                4 * xi0 * xi2,
                4 * xi1 * xi3,
                4 * xi2 * xi3,
                4 * xi0 * xi3,
            ],
            dtype=np.float64,
        )
        dhat1 = np.array(
            [
                -4 * xi0 + 1,
                4 * xi1 - 1,
                np.zeros(n_q),
                np.zeros(n_q),
                4 * (xi0 - xi1),
                4 * xi2,
                -4 * xi2,
                4 * xi3,
                np.zeros(n_q),
                -4 * xi3,
            ],
            dtype=np.float64,
        )
        dhat2 = np.array(
            [
                -4 * xi0 + 1,
                np.zeros(n_q),
                4 * xi2 - 1,
                np.zeros(n_q),
                -4 * xi1,
                4 * xi1,
                4 * (xi0 - xi2),
                np.zeros(n_q),
                4 * xi3,
                -4 * xi3,
            ],
            dtype=np.float64,
        )
        dhat3 = np.array(
            [
                -4 * xi0 + 1,
                np.zeros(n_q),
                np.zeros(n_q),
                4 * xi3 - 1,
                -4 * xi1,
                np.zeros(n_q),
                -4 * xi2,
                4 * xi1,
                4 * xi2,
                4 * (xi0 - xi3),
            ],
            dtype=np.float64,
        )
        return hatp, dhat1, dhat2, dhat3
    raise ValueError(f"Unsupported seepage element type {elem_type!r}.")


def quadrature_volume_3d(elem_type: str) -> tuple[np.ndarray, np.ndarray]:
    elem_key = str(elem_type).strip().upper()
    if elem_key == "P1":
        return np.array([[0.25], [0.25], [0.25]], dtype=np.float64), np.array([1.0 / 6.0], dtype=np.float64)
    if elem_key == "P2":
        xi = np.array(
            [
                [1 / 4, 0.0714285714285714, 0.785714285714286, 0.0714285714285714, 0.0714285714285714, 0.399403576166799, 0.100596423833201, 0.100596423833201, 0.399403576166799, 0.399403576166799, 0.100596423833201],
                [1 / 4, 0.0714285714285714, 0.0714285714285714, 0.785714285714286, 0.0714285714285714, 0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799],
                [1 / 4, 0.0714285714285714, 0.0714285714285714, 0.0714285714285714, 0.785714285714286, 0.100596423833201, 0.100596423833201, 0.399403576166799, 0.100596423833201, 0.399403576166799, 0.399403576166799],
            ],
            dtype=np.float64,
        )
        wf = np.array(
            [
                -0.013155555555555,
                0.007622222222222,
                0.007622222222222,
                0.007622222222222,
                0.007622222222222,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
                0.024888888888888,
            ],
            dtype=np.float64,
        )
        return xi, wf
    raise ValueError(f"Unsupported seepage element type {elem_type!r}.")


def penalty_parameters_3d(coord: np.ndarray, elem: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64)
    elem = np.asarray(elem, dtype=np.int64)
    grho = 9.81
    base = elem[:4, :]
    out = np.zeros(base.shape[1], dtype=np.float64)
    for i in range(base.shape[1]):
        ids = base[:, i]
        pts = coord[:, ids].T
        p1, p2, p3, p4 = pts
        l12 = float(np.linalg.norm(p1 - p2))
        l13 = float(np.linalg.norm(p1 - p3))
        l23 = float(np.linalg.norm(p2 - p3))
        l14 = float(np.linalg.norm(p1 - p4))
        l24 = float(np.linalg.norm(p2 - p4))
        l34 = float(np.sqrt((p3[0] - p4[0]) ** 2 + (p3[1] - p4[1]) ** 2 + (p2[2] - p4[2]) ** 2))
        out[i] = grho * min((l12, l13, l23, l14, l24, l34)) / 2.0
    return out


def _compute_pressure_at_integration_points(assembly: SeepageAssembly, pw: np.ndarray) -> np.ndarray:
    pw_e = np.asarray(pw[assembly.elem], dtype=np.float64)
    return np.sum(assembly.hatphi_tiled * np.kron(pw_e, np.ones((1, assembly.n_q), dtype=np.float64)), axis=0)


def _compute_gradient(assembly: SeepageAssembly, nodal_values: np.ndarray) -> np.ndarray:
    grad = assembly.B @ np.asarray(nodal_values, dtype=np.float64)
    return grad.reshape(assembly.dim, assembly.n_int, order="F")


def _build_flow_stiffness(assembly: SeepageAssembly, conduct0: np.ndarray) -> tuple[csr_matrix, np.ndarray]:
    wc = assembly.weight * np.asarray(conduct0, dtype=np.float64)
    K_D = (assembly.B.T @ (diags(np.repeat(wc, assembly.dim), format="csr") @ assembly.B)).tocsr()
    return K_D, wc


def _rhs_from_dirichlet(assembly: SeepageAssembly, wc: np.ndarray, pw_D: np.ndarray, grho: float) -> np.ndarray:
    grad_pw_D = assembly.B @ np.asarray(pw_D, dtype=np.float64)
    grad_y = assembly.B @ np.asarray(assembly.dphi["_coord_y"], dtype=np.float64)
    q3 = grad_pw_D + grho * grad_y
    return -assembly.B.T @ (np.repeat(wc, assembly.dim) * q3)


def _build_newton_matrix(
    assembly: SeepageAssembly,
    conduct0: np.ndarray,
    K_D: csr_matrix,
    coord_y: np.ndarray,
    perm_r_der: np.ndarray,
    grho: float,
) -> csr_matrix:
    coeff = np.asarray(conduct0, dtype=np.float64) * np.asarray(perm_r_der, dtype=np.float64) * assembly.weight
    grad_y = _compute_gradient(assembly, coord_y)
    c_coo = assembly.C.tocoo()
    row_base = np.asarray(c_coo.row, dtype=np.int64)
    rows_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    for c in range(assembly.dim):
        rows_parts.append(assembly.dim * row_base + c)
        data_parts.append(np.asarray(c_coo.data, dtype=np.float64) * (coeff * grho * grad_y[c, :])[row_base])
    EC = coo_matrix(
        (
            np.concatenate(data_parts),
            (np.concatenate(rows_parts), np.tile(np.asarray(c_coo.col, dtype=np.int64), assembly.dim)),
        ),
        shape=(assembly.dim * assembly.n_int, assembly.n_nodes),
    ).tocsr()
    return (K_D + assembly.B.T @ EC).tocsr()


def _build_newton_rhs(
    assembly: SeepageAssembly,
    wc: np.ndarray,
    pw: np.ndarray,
    coord_y: np.ndarray,
    grho: float,
    perm_r: np.ndarray,
) -> np.ndarray:
    grad_pw = _compute_gradient(assembly, pw)
    grad_y = _compute_gradient(assembly, coord_y)
    q3 = grad_pw + grho * np.asarray(perm_r, dtype=np.float64)[None, :] * grad_y
    return -assembly.B.T @ (np.repeat(wc, assembly.dim) * q3.reshape(-1, order="F"))


def _finalize_seepage_outputs(
    assembly: SeepageAssembly,
    pw: np.ndarray,
    eps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_p = _compute_gradient(assembly, pw)
    pw_int = _compute_pressure_at_integration_points(assembly, pw)
    if assembly.n_q > 1:
        int_pw_e = np.sum((pw_int * assembly.weight).reshape(assembly.n_elem, assembly.n_q), axis=1)
        int_e = np.sum(assembly.weight.reshape(assembly.n_elem, assembly.n_q), axis=1)
        pw_aver_e = int_pw_e / int_e
    else:
        pw_aver_e = pw_int
    mater_sat = pw_aver_e >= 0.1 * np.asarray(eps, dtype=np.float64)
    return pw, grad_p, mater_sat


def _load_gmsh41_simplex(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    physical_names: dict[tuple[int, int], str] = {}
    entity_names: dict[tuple[int, int], tuple[str, ...]] = {}
    node_tags_by_entity: dict[tuple[int, int], list[int]] = {}
    coords_by_tag: dict[int, tuple[float, float, float]] = {}
    tet_tags: list[tuple[int, int, int, int]] = []
    tri_tags: list[tuple[int, int, int]] = []
    point_nodes_by_name: dict[str, list[int]] = {}
    boundary_names: list[str] = []
    boundary_groups_raw: dict[str, list[int]] = {}
    region_names: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        it = iter(handle)
        for raw in it:
            marker = raw.strip()
            if marker == "$PhysicalNames":
                count = int(next(it).strip())
                for _ in range(count):
                    parts = shlex.split(next(it).strip())
                    dim = int(parts[0])
                    tag = int(parts[1])
                    physical_names[(dim, tag)] = _logical_name(parts[2])
                _expect_marker(next(it), "$EndPhysicalNames")
            elif marker == "$Entities":
                counts = [int(v) for v in next(it).split()]
                for dim, n_entities in enumerate(counts):
                    for _ in range(n_entities):
                        parts = next(it).split()
                        tag, names = _parse_entity_line(dim, parts, physical_names)
                        entity_names[(dim, tag)] = names
                _expect_marker(next(it), "$EndEntities")
            elif marker == "$Nodes":
                n_blocks = int(next(it).split()[0])
                for _ in range(n_blocks):
                    dim_s, entity_s, parametric_s, n_s = next(it).split()
                    dim = int(dim_s)
                    entity = int(entity_s)
                    parametric = int(parametric_s)
                    n_nodes = int(n_s)
                    tags = [int(next(it).strip()) for _ in range(n_nodes)]
                    node_tags_by_entity.setdefault((dim, entity), []).extend(tags)
                    for tag in tags:
                        coords = [float(v) for v in next(it).split()]
                        coords_by_tag[tag] = (coords[0], coords[1], coords[2])
                        if parametric:
                            pass
                _expect_marker(next(it), "$EndNodes")
            elif marker == "$Elements":
                n_blocks = int(next(it).split()[0])
                for _ in range(n_blocks):
                    dim_s, entity_s, type_s, n_s = next(it).split()
                    dim = int(dim_s)
                    entity = int(entity_s)
                    element_type = int(type_s)
                    n_elem = int(n_s)
                    names = entity_names.get((dim, entity), ())
                    for _ in range(n_elem):
                        values = [int(v) for v in next(it).split()]
                        nodes = values[1:]
                        if element_type == 4:
                            tet_tags.append((nodes[0], nodes[1], nodes[2], nodes[3]))
                            region_names.append(names[0] if names else "")
                        elif element_type == 2:
                            tri_tags.append((nodes[0], nodes[1], nodes[2]))
                            boundary_names.append(names[0] if names else "")
                            if names:
                                boundary_groups_raw.setdefault(names[0], []).append(len(tri_tags) - 1)
                        elif element_type == 15 and names:
                            for name in names:
                                point_nodes_by_name.setdefault(name, []).append(nodes[0])
                _expect_marker(next(it), "$EndElements")

    for key, tags in node_tags_by_entity.items():
        for name in entity_names.get(key, ()):
            if key[0] == 0:
                point_nodes_by_name.setdefault(name, []).extend(tags)

    sorted_tags = sorted(coords_by_tag)
    tag_to_idx = {tag: idx for idx, tag in enumerate(sorted_tags)}
    coord = np.asarray([coords_by_tag[tag] for tag in sorted_tags], dtype=np.float64).T
    tet4 = np.asarray([[tag_to_idx[tag] for tag in tet] for tet in tet_tags], dtype=np.int64).T
    tri3 = np.asarray([[tag_to_idx[tag] for tag in tri] for tri in tri_tags], dtype=np.int64).T
    nodesets = {
        name: np.unique([tag_to_idx[tag] for tag in tags if tag in tag_to_idx]).astype(np.int64)
        for name, tags in point_nodes_by_name.items()
    }
    boundary_groups = {name: np.asarray(indices, dtype=np.int64) for name, indices in boundary_groups_raw.items()}
    return {
        "coord": coord,
        "tet4": tet4,
        "tri3": tri3,
        "nodesets": nodesets,
        "boundary_groups": boundary_groups,
        "region_names": np.asarray(region_names, dtype=object),
    }


def _parse_entity_line(dim: int, parts: list[str], physical_names: dict[tuple[int, int], str]) -> tuple[int, tuple[str, ...]]:
    tag = int(parts[0])
    if dim == 0:
        offset = 4
    else:
        offset = 7
    n_phys = int(parts[offset])
    phys = [int(v) for v in parts[offset + 1 : offset + 1 + n_phys]]
    return tag, tuple(physical_names.get((dim, p), str(p)) for p in phys)


def _expect_marker(line: str, expected: str) -> None:
    got = line.strip()
    if got != expected:
        raise ValueError(f"Expected {expected}, got {got}")


def _logical_name(name: str) -> str:
    text = str(name).strip().strip('"')
    for prefix in ("region", "boundary", "nodeset", "boundary_geom"):
        if text.startswith(prefix + ":"):
            return text.split(":", 1)[1]
        if text.startswith(prefix + "_"):
            return text[len(prefix) + 1 :]
    return text


def _midpoint_node_index(
    coord: np.ndarray,
    edge_map: dict[tuple[int, int], int],
    extra_points: list[np.ndarray],
    a: int,
    b: int,
) -> int:
    i = int(a)
    j = int(b)
    key = (i, j) if i < j else (j, i)
    idx = edge_map.get(key)
    if idx is not None:
        return idx
    idx = int(coord.shape[1] + len(extra_points))
    edge_map[key] = idx
    extra_points.append(0.5 * (coord[:, key[0]] + coord[:, key[1]]))
    return idx


def _elevate_tet4_mesh_to_tet10(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coord_arr = np.asarray(coord, dtype=np.float64)
    tet4 = np.asarray(elem, dtype=np.int64)
    tri3 = np.asarray(surf, dtype=np.int64)
    edge_map: dict[tuple[int, int], int] = {}
    extra_points: list[np.ndarray] = []

    tet10 = np.empty((10, tet4.shape[1]), dtype=np.int64)
    tet10[:4, :] = tet4
    for idx in range(tet4.shape[1]):
        v0, v1, v2, v3 = (int(v) for v in tet4[:, idx])
        tet10[4, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v1)
        tet10[5, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v1, v2)
        tet10[6, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v2)
        tet10[7, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v1, v3)
        tet10[8, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v2, v3)
        tet10[9, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v3)

    tri6 = np.empty((6, tri3.shape[1]), dtype=np.int64)
    if tri3.shape[1]:
        tri6[:3, :] = tri3
        for idx in range(tri3.shape[1]):
            v0, v1, v2 = (int(v) for v in tri3[:, idx])
            tri6[3, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v1)
            tri6[4, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v1, v2)
            tri6[5, idx] = _midpoint_node_index(coord_arr, edge_map, extra_points, v0, v2)

    coord_new = np.hstack((coord_arr, np.column_stack(extra_points))) if extra_points else coord_arr.copy()
    return coord_new, tet10, tri6


def _expand_nodeset(n_nodes: int, surf: np.ndarray, base_nodes: np.ndarray) -> np.ndarray:
    selected = np.zeros(n_nodes, dtype=bool)
    selected[np.asarray(base_nodes, dtype=np.int64)] = True
    surf = np.asarray(surf, dtype=np.int64)
    if surf.size == 0:
        return np.flatnonzero(selected).astype(np.int64)
    for col in range(surf.shape[1]):
        corners = surf[:3, col]
        if np.all(selected[corners]):
            selected[surf[:, col]] = True
            continue
        for edge, local_nodes in ((np.array([0, 1]), (3,)), (np.array([1, 2]), (4,)), (np.array([0, 2]), (5,))):
            if np.all(selected[corners[edge]]):
                selected[surf[list(local_nodes), col]] = True
    return np.flatnonzero(selected).astype(np.int64)
