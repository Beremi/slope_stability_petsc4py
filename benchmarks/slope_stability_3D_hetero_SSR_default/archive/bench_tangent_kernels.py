from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from slope_stability.core.simplex_lagrange import tetra_lagrange_node_tuples
from slope_stability.fem.distributed_elastic import find_overlap_partition
from slope_stability.fem.distributed_tangent import assemble_owned_tangent_values, prepare_owned_tangent_pattern
from slope_stability.io import load_mesh_file
from slope_stability.mesh import reorder_mesh_nodes


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "tangent_kernel_microbench"
DEFAULT_KERNELS = ("legacy", "rows")
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


def _build_small_pattern(elem_type: str):
    order = {"P1": 1, "P2": 2, "P4": 4}[str(elem_type).upper()]
    vertices = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tet4 = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    coord, elem = _elevated_tetra_mesh(order, tet4, vertices)
    q_mask = np.ones((3, coord.shape[1]), dtype=bool)
    return prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        np.zeros(elem.shape[1], dtype=np.int64),
        [MATERIAL],
        (0, coord.shape[1]),
        elem_type=elem_type,
        include_unique=True,
        include_overlap_B=False,
    )


def _build_l1_pattern(mesh_path: Path):
    mesh = load_mesh_file(mesh_path, elem_type="P4")
    return prepare_owned_tangent_pattern(
        np.asarray(mesh.coord, dtype=np.float64),
        np.asarray(mesh.elem, dtype=np.int64),
        np.asarray(mesh.q_mask, dtype=bool),
        np.asarray(mesh.material, dtype=np.int64),
        [MATERIAL],
        (0, int(mesh.coord.shape[1])),
        elem_type="P4",
        include_unique=True,
        include_overlap_B=False,
    )


def _build_virtual_rank_pattern(mesh_path: Path, *, node_ordering: str, virtual_ranks: int, selector: str):
    mesh = load_mesh_file(mesh_path, elem_type="P4")
    reordered = reorder_mesh_nodes(mesh.coord, mesh.elem, mesh.surf, mesh.q_mask, strategy=node_ordering)
    coord = np.asarray(reordered.coord, dtype=np.float64)
    elem = np.asarray(reordered.elem, dtype=np.int64)
    q_mask = np.asarray(reordered.q_mask, dtype=bool)
    material = np.asarray(mesh.material, dtype=np.int64)
    n_nodes = int(coord.shape[1])

    blocks: list[dict[str, int]] = []
    for rank in range(int(virtual_ranks)):
        node0 = (rank * n_nodes) // int(virtual_ranks)
        node1 = ((rank + 1) * n_nodes) // int(virtual_ranks)
        overlap_nodes, overlap_elements = find_overlap_partition(elem, (node0, node1))
        blocks.append(
            {
                "rank": int(rank),
                "node0": int(node0),
                "node1": int(node1),
                "owned_nodes": int(node1 - node0),
                "overlap_nodes": int(overlap_nodes.size),
                "overlap_elements": int(overlap_elements.size),
            }
        )

    selector_key = str(selector).strip().lower()
    if selector_key == "max_overlap":
        selected = max(blocks, key=lambda row: row["overlap_elements"])
    elif selector_key == "median_overlap":
        ordered = sorted(blocks, key=lambda row: row["overlap_elements"])
        selected = ordered[len(ordered) // 2]
    else:
        rank = int(selector_key)
        selected = next(row for row in blocks if row["rank"] == rank)

    pattern = prepare_owned_tangent_pattern(
        coord,
        elem,
        q_mask,
        material,
        [MATERIAL],
        (int(selected["node0"]), int(selected["node1"])),
        elem_type="P4",
        include_unique=True,
        include_overlap_B=False,
    )
    return pattern, {
        "mesh_path": str(Path(mesh_path).resolve()),
        "node_ordering": str(node_ordering),
        "virtual_ranks": int(virtual_ranks),
        "selector": str(selector),
        "selected_rank": int(selected["rank"]),
        "selected_owned_nodes": int(selected["owned_nodes"]),
        "selected_overlap_nodes": int(selected["overlap_nodes"]),
        "selected_overlap_elements": int(selected["overlap_elements"]),
        "blocks": blocks,
    }


def _benchmark_kernel(pattern, ds: np.ndarray, *, kernel: str, repeat: int, warmup: int, use_compiled: bool) -> dict[str, float]:
    for _ in range(int(warmup)):
        assemble_owned_tangent_values(pattern, ds, use_compiled=use_compiled, kernel=kernel)

    samples: list[float] = []
    for _ in range(int(repeat)):
        t0 = perf_counter()
        assemble_owned_tangent_values(pattern, ds, use_compiled=use_compiled, kernel=kernel)
        samples.append(perf_counter() - t0)

    arr = np.asarray(samples, dtype=np.float64)
    return {
        "samples": [float(v) for v in arr.tolist()],
        "min_s": float(arr.min()) if arr.size else 0.0,
        "max_s": float(arr.max()) if arr.size else 0.0,
        "mean_s": float(arr.mean()) if arr.size else 0.0,
        "median_s": float(median(arr.tolist())) if arr.size else 0.0,
        "stdev_s": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def _write_outputs(out_dir: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "kernel",
                "median_s",
                "mean_s",
                "min_s",
                "max_s",
                "stdev_s",
                "speedup_vs_legacy",
            ],
        )
        writer.writeheader()
        for kernel, metrics in payload["results"].items():
            writer.writerow(
                {
                    "kernel": kernel,
                    "median_s": metrics["median_s"],
                    "mean_s": metrics["mean_s"],
                    "min_s": metrics["min_s"],
                    "max_s": metrics["max_s"],
                    "stdev_s": metrics["stdev_s"],
                    "speedup_vs_legacy": metrics["speedup_vs_legacy"],
                }
            )
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbenchmark owned tangent assembly kernels.")
    parser.add_argument("--mode", choices=["small", "l1", "virtual-rank"], default="small")
    parser.add_argument("--elem-type", choices=["P1", "P2", "P4"], default="P4")
    parser.add_argument("--mesh-path", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--node-ordering", type=str, default="xyz")
    parser.add_argument("--virtual-ranks", type=int, default=8)
    parser.add_argument("--virtual-rank-selector", type=str, default="max_overlap")
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--kernels", nargs="+", default=list(DEFAULT_KERNELS))
    parser.add_argument("--use-compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.mode == "small":
        pattern = _build_small_pattern(args.elem_type)
        mode_info = None
    elif args.mode == "virtual-rank":
        pattern, mode_info = _build_virtual_rank_pattern(
            Path(args.mesh_path),
            node_ordering=args.node_ordering,
            virtual_ranks=int(args.virtual_ranks),
            selector=args.virtual_rank_selector,
        )
    else:
        pattern = _build_l1_pattern(Path(args.mesh_path))
        mode_info = None

    rng = np.random.default_rng(int(args.seed))
    ds = rng.standard_normal((pattern.n_strain * pattern.n_strain, pattern.local_int_indices.size))
    results: dict[str, dict[str, float | list[float]]] = {}

    for kernel in [str(v).lower() for v in args.kernels]:
        results[kernel] = _benchmark_kernel(
            pattern,
            ds,
            kernel=kernel,
            repeat=int(args.repeat),
            warmup=int(args.warmup),
            use_compiled=bool(args.use_compiled),
        )

    legacy_median = float(results["legacy"]["median_s"]) if "legacy" in results else None
    for kernel, metrics in results.items():
        if legacy_median is None or float(metrics["median_s"]) == 0.0:
            metrics["speedup_vs_legacy"] = 1.0 if kernel == "legacy" else 0.0
        else:
            metrics["speedup_vs_legacy"] = float(legacy_median) / float(metrics["median_s"])

    payload = {
        "mode": args.mode,
        "elem_type": args.elem_type if args.mode == "small" else "P4",
        "mesh_path": None if args.mode == "small" else str(Path(args.mesh_path).resolve()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "use_compiled": bool(args.use_compiled),
        "mode_info": mode_info,
        "pattern": {
            "n_overlap_elements": int(pattern.overlap_elements.size),
            "n_overlap_nodes": int(pattern.overlap_nodes.size),
            "n_local_rows": int(pattern.local_matrix_pattern.shape[0]),
            "nnz": int(pattern.local_matrix_pattern.nnz),
            "n_q": int(pattern.n_q),
            "n_p": int(pattern.n_p),
            "stats": dict(pattern.stats),
            "timings": dict(pattern.timings),
        },
        "results": results,
    }
    json_path, csv_path = _write_outputs(Path(args.out_dir), payload)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "results": results}, indent=2))


if __name__ == "__main__":
    main()
