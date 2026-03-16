from __future__ import annotations

import argparse
import json

import numpy as np
from mpi4py import MPI

from slope_stability.constitutive.problem import ConstitutiveOperator
from slope_stability.core.simplex_lagrange import tetra_lagrange_node_tuples
from slope_stability.fem.distributed_tangent import assemble_owned_tangent_values, prepare_owned_tangent_pattern


MATERIAL = {
    "c0": 15.0,
    "phi": 30.0,
    "psi": 0.0,
    "young": 10000.0,
    "poisson": 0.33,
    "gamma_sat": 19.0,
    "gamma_unsat": 19.0,
}


def _elevated_tetra_mesh(order: int, tet4: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tuples = tetra_lagrange_node_tuples(int(order))
    coord_map: dict[tuple[float, float, float], int] = {}
    coord_list: list[np.ndarray] = []
    elem = np.empty((len(tuples), tet4.shape[1]), dtype=np.int64)

    for e in range(int(tet4.shape[1])):
        verts = np.asarray(vertices[:, tet4[:, e]], dtype=np.float64)
        for local_idx, counts in enumerate(tuples):
            bary = np.asarray(counts, dtype=np.float64) / float(order)
            point = verts @ bary
            key = tuple(np.round(point, 12).tolist())
            global_idx = coord_map.get(key)
            if global_idx is None:
                global_idx = len(coord_list)
                coord_map[key] = global_idx
                coord_list.append(point)
            elem[local_idx, e] = int(global_idx)

    coord = np.column_stack(coord_list) if coord_list else np.empty((3, 0), dtype=np.float64)
    return coord, elem


def _owned_ranges_from_owner_nodes(elem: np.ndarray, n_nodes: int, size: int) -> list[tuple[int, int]]:
    elem = np.asarray(elem, dtype=np.int64)
    owner_nodes = np.min(elem, axis=0)
    starts = [0]
    for rank in range(1, int(size)):
        cut = (rank * int(elem.shape[1])) // int(size)
        starts.append(int(owner_nodes[cut]))
    starts.append(int(n_nodes))
    ranges = [(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:], strict=False)]
    if any(stop <= start for start, stop in ranges):
        raise RuntimeError(f"Invalid owned-node ranges for MPI test: {ranges}")
    return ranges


def _build_chain_pattern(elem_type: str, *, size: int, rank: int, elems_per_rank: int):
    order = {"P1": 1, "P2": 2, "P4": 4}[str(elem_type).upper()]
    n_elem = max(int(size) * int(elems_per_rank), 4)
    n_vertices = int(n_elem) + 3
    t = np.arange(1, n_vertices + 1, dtype=np.float64)
    vertices = np.vstack((t, 0.1 * t * t, 0.01 * t * t * t))
    tet4 = np.vstack([np.arange(i, i + n_elem, dtype=np.int64) for i in range(4)])
    coord, elem = _elevated_tetra_mesh(order, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    owned_ranges = _owned_ranges_from_owner_nodes(elem, int(coord.shape[1]), int(size))
    pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        owned_ranges[int(rank)],
        elem_type=elem_type,
        include_unique=True,
        include_legacy_scatter=False,
        include_overlap_B=False,
    )
    return coord, elem, q_mask, pattern, owned_ranges


def _material_fields(n_int: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    young = float(MATERIAL["young"])
    poisson = float(MATERIAL["poisson"])
    shear = young / (2.0 * (1.0 + poisson))
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    lame = bulk - 2.0 * shear / 3.0
    return (
        np.full(n_int, float(MATERIAL["c0"]), dtype=np.float64),
        np.full(n_int, np.deg2rad(float(MATERIAL["phi"])), dtype=np.float64),
        np.full(n_int, np.deg2rad(float(MATERIAL["psi"])), dtype=np.float64),
        np.full(n_int, shear, dtype=np.float64),
        np.full(n_int, bulk, dtype=np.float64),
        np.full(n_int, lame, dtype=np.float64),
        np.ones(n_int, dtype=np.float64),
    )


def _evaluate_mode(pattern, q_mask, n_elem: int, n_q: int, n_nodes: int, mode: str):
    c0, phi, psi, shear, bulk, lame, weight = _material_fields(int(n_elem * n_q))
    builder = ConstitutiveOperator(
        B=None,
        c0=c0,
        phi=phi,
        psi=psi,
        Davis_type="B",
        shear=shear,
        bulk=bulk,
        lame=lame,
        WEIGHT=weight,
        n_strain=int(pattern.n_strain),
        n_int=int(n_elem * n_q),
        dim=int(pattern.dim),
        q_mask=q_mask,
    )
    builder.set_owned_tangent_pattern(
        pattern,
        use_compiled=False,
        tangent_kernel="rows",
        constitutive_mode=mode,
        use_compiled_constitutive=False,
    )
    rng = np.random.default_rng(12345)
    U = rng.standard_normal((int(pattern.dim), int(n_nodes)))
    builder.reduction(1.0)
    builder.constitutive_problem_stress_tangent(U)
    return {
        "S_local": np.asarray(builder._owned_local_S, dtype=np.float64).copy(),
        "DS_local": np.asarray(builder._owned_local_DS, dtype=np.float64).copy(),
        "F_local": np.asarray(builder.build_F_local(), dtype=np.float64).copy(),
        "tangent_values": np.asarray(
            assemble_owned_tangent_values(pattern, builder._owned_local_DS, use_compiled=False, kernel="rows"),
            dtype=np.float64,
        ).copy(),
    }


def _max_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def main() -> int:
    parser = argparse.ArgumentParser(description="MPI equivalence check for overlap/unique constitutive modes.")
    parser.add_argument("--elem-type", type=str, default="P2", choices=["P1", "P2", "P4"])
    parser.add_argument("--elems-per-rank", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1.0e-11)
    parser.add_argument("--atol", type=float, default=1.0e-11)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())

    coord, elem, q_mask, pattern, owned_ranges = _build_chain_pattern(
        str(args.elem_type),
        size=size,
        rank=rank,
        elems_per_rank=int(args.elems_per_rank),
    )

    remote_overlap_local = int(np.count_nonzero(~np.asarray(pattern.local_overlap_owner_mask, dtype=bool)))
    remote_overlap_global = int(comm.allreduce(remote_overlap_local, op=MPI.SUM))
    if size > 1 and remote_overlap_global <= 0:
        if rank == 0:
            print("No remote overlap integration points were generated; MPI equivalence test is invalid.")
        return 2

    n_elem = int(elem.shape[1])
    modes = {
        mode: _evaluate_mode(pattern, q_mask, n_elem, int(pattern.n_q), int(coord.shape[1]), mode)
        for mode in ("overlap", "unique_gather", "unique_exchange")
    }

    baseline = modes["overlap"]
    local_summary: dict[str, dict[str, float | bool]] = {}
    local_ok = True
    for mode in ("unique_gather", "unique_exchange"):
        candidate = modes[mode]
        mode_summary: dict[str, float | bool] = {}
        for field in ("S_local", "DS_local", "F_local", "tangent_values"):
            field_ok = bool(
                np.allclose(candidate[field], baseline[field], rtol=float(args.rtol), atol=float(args.atol))
            )
            mode_summary[f"{field}_ok"] = field_ok
            mode_summary[f"{field}_max_abs"] = _max_abs_delta(candidate[field], baseline[field])
            local_ok = local_ok and field_ok
        local_summary[mode] = mode_summary

    all_ok = bool(comm.allreduce(1 if local_ok else 0, op=MPI.MIN))
    global_summary: dict[str, dict[str, float | bool]] = {}
    for mode, mode_summary in local_summary.items():
        merged: dict[str, float | bool] = {}
        for key, value in mode_summary.items():
            if key.endswith("_ok"):
                merged[key] = bool(comm.allreduce(1 if bool(value) else 0, op=MPI.MIN))
            else:
                merged[key] = float(comm.allreduce(float(value), op=MPI.MAX))
        global_summary[mode] = merged

    if rank == 0:
        payload = {
            "elem_type": str(args.elem_type),
            "size": size,
            "owned_ranges": owned_ranges,
            "remote_overlap_global": remote_overlap_global,
            "results": global_summary,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
