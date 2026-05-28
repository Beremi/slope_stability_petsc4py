"""Standardized result export helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def _attempt_group_key(event: dict[str, object]) -> tuple[object, ...] | None:
    phase = str(event.get("phase", "")).strip().lower()
    if phase == "continuation":
        target_step = event.get("target_step")
        attempt_in_step = event.get("attempt_in_step")
        if target_step is None or attempt_in_step is None:
            return None
        return ("continuation", int(target_step), int(attempt_in_step))
    if phase == "init":
        init_stage = event.get("init_stage")
        init_attempt = event.get("init_attempt")
        if init_stage is None or init_attempt is None:
            return None
        return ("init", str(init_stage), int(init_attempt))
    return None


def _make_attempt_record(event: dict[str, object]) -> dict[str, object]:
    return {
        "phase": None if event.get("phase") is None else str(event.get("phase")),
        "continuation_kind": event.get("continuation_kind"),
        "target_step": None if event.get("target_step") is None else int(event.get("target_step")),
        "accepted_steps_before": None if event.get("accepted_steps") is None else int(event.get("accepted_steps")),
        "attempt_in_step": None if event.get("attempt_in_step") is None else int(event.get("attempt_in_step")),
        "init_stage": None if event.get("init_stage") is None else str(event.get("init_stage")),
        "init_attempt": None if event.get("init_attempt") is None else int(event.get("init_attempt")),
        "lambda_before": event.get("lambda_before"),
        "omega_target": event.get("omega_target"),
        "newton_iterations": [],
    }


def _build_progress_views(progress: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], dict[str, object]] = {}
    ordered: list[dict[str, object]] = []

    for event in progress:
        key = _attempt_group_key(event)
        if key is None:
            continue
        record = grouped.get(key)
        if record is None:
            record = _make_attempt_record(event)
            grouped[key] = record
            ordered.append(record)
        event_name = str(event.get("event", ""))
        if event_name == "newton_iteration":
            record["newton_iterations"].append(dict(event))
        elif event_name == "attempt_complete":
            record["attempt_complete"] = dict(event)
        elif event_name == "step_accepted":
            record["step_accepted"] = dict(event)

    step_map: dict[int, dict[str, object]] = {}
    for record in ordered:
        if record.get("phase") != "continuation" or record.get("target_step") is None:
            continue
        step_idx = int(record["target_step"])
        step_record = step_map.get(step_idx)
        if step_record is None:
            step_record = {
                "accepted_step": int(step_idx),
                "attempts": [],
            }
            step_map[step_idx] = step_record
        step_record["attempts"].append(record)
        if "step_accepted" in record:
            step_record["step_accepted"] = record["step_accepted"]

    step_records = [step_map[idx] for idx in sorted(step_map)]
    return ordered, step_records


def write_debug_bundle_h5(
    *,
    out_path: Path,
    config_text: str,
    run_info_path: Path,
    npz_path: Path,
    progress_path: Path | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_info_text = run_info_path.read_text(encoding="utf-8")
    progress_text = ""
    if progress_path is not None and progress_path.exists():
        progress_text = progress_path.read_text(encoding="utf-8")

    with np.load(npz_path, allow_pickle=True) as npz, h5py.File(out_path, "w") as h5:
        meta = h5.create_group("metadata")
        meta.create_dataset("config_toml", data=np.bytes_(config_text))
        meta.create_dataset("run_info_json", data=np.bytes_(run_info_text))
        meta.create_dataset("progress_jsonl", data=np.bytes_(progress_text))
        arrays = h5.create_group("arrays")
        for key in sorted(npz.files):
            _create_h5_dataset(arrays, key, np.asarray(npz[key]))
    return out_path


def write_history_json(
    *,
    out_path: Path,
    run_info_path: Path,
    npz_path: Path,
    progress_path: Path | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    progress = []
    if progress_path is not None and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as handle:
            progress = [json.loads(line) for line in handle if line.strip()]

    history: dict[str, object] = {
        "run_info": run_info.get("run_info", {}),
        "params": run_info.get("params", {}),
        "mesh": run_info.get("mesh", {}),
        "timings": run_info.get("timings", {}),
        "progress_events": progress,
    }
    if progress:
        attempt_records, step_records = _build_progress_views(progress)
        if attempt_records:
            history["attempt_records"] = attempt_records
        if step_records:
            history["step_records"] = step_records
    with np.load(npz_path, allow_pickle=True) as npz:
        for key in ("lambda_hist", "omega_hist", "Umax_hist"):
            if key in npz:
                history[key] = np.asarray(npz[key]).tolist()
        stats = {key[6:]: np.asarray(npz[key]).tolist() for key in npz.files if key.startswith("stats_")}
        if stats:
            history["stats"] = stats
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return out_path


def write_history_csv_tables(*, out_dir: Path, history_json_path: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads(history_json_path.read_text(encoding="utf-8"))
    written: list[Path] = []

    attempt_records = list(history.get("attempt_records", []))
    step_records = list(history.get("step_records", []))

    accepted_rows: list[dict[str, object]] = []
    for step_record in step_records:
        step_accepted = dict(step_record.get("step_accepted", {}))
        if not step_accepted:
            continue
        attempts = list(step_record.get("attempts", []))
        accepted_rows.append(
            {
                "accepted_step": step_record.get("accepted_step"),
                "omega_value": step_accepted.get("omega_value"),
                "d_omega": step_accepted.get("d_omega"),
                "lambda_value": step_accepted.get("lambda_value"),
                "d_lambda": step_accepted.get("d_lambda"),
                "u_max": step_accepted.get("u_max"),
                "step_attempt_count": step_accepted.get("step_attempt_count"),
                "step_newton_iterations": step_accepted.get("step_newton_iterations"),
                "step_linear_iterations": step_accepted.get("step_linear_iterations"),
                "line_search_iterations": step_accepted.get("line_search_iterations"),
                "line_search_fallback_count": sum(
                    int(bool(iteration.get("line_search_fallback_used", False)))
                    for attempt in attempts
                    for iteration in list(attempt.get("newton_iterations", []))
                ),
                "deflation_basis_dim_solve_max": step_accepted.get("deflation_basis_dim_solve_max"),
                "deflation_basis_dim_end_last": step_accepted.get("deflation_basis_dim_end_last"),
                "step_linear_solve_time_s": step_accepted.get("step_linear_solve_time"),
                "step_linear_preconditioner_time_s": step_accepted.get("step_linear_preconditioner_time"),
                "step_linear_orthogonalization_time_s": step_accepted.get("step_linear_orthogonalization_time"),
                "step_wall_time_s": step_accepted.get("step_wall_time"),
                "step_newton_relres_end": step_accepted.get("step_newton_relres_end"),
                "step_newton_relcorr_end": step_accepted.get("step_newton_relcorr_end"),
            }
        )
    if not accepted_rows:
        stats = dict(history.get("stats", {}))
        step_index = list(stats.get("step_index", []))
        lambda_hist = list(history.get("lambda_hist", []))
        u_hist = list(history.get("Umax_hist", []))
        for idx, step_value in enumerate(step_index):
            step_no = int(step_value)
            attempts = [record for record in step_records if int(record.get("accepted_step", -1)) == step_no]
            fallback_count = sum(
                int(bool(iteration.get("line_search_fallback_used", False)))
                for record in attempts
                for attempt in list(record.get("attempts", []))
                for iteration in list(attempt.get("newton_iterations", []))
            )
            lambda_value = _series_value(stats, "step_lambda", idx)
            prev_lambda = _history_step_value(lambda_hist, step_no - 1)
            accepted_rows.append(
                {
                    "accepted_step": step_no,
                    "omega_value": _series_value(stats, "step_omega", idx),
                    "d_omega": _series_value(stats, "step_d_omega", idx),
                    "lambda_value": lambda_value,
                    "d_lambda": (
                        None
                        if lambda_value is None or prev_lambda is None
                        else float(lambda_value) - float(prev_lambda)
                    ),
                    "u_max": _history_step_value(u_hist, step_no),
                    "step_attempt_count": _series_value(stats, "step_attempt_count", idx),
                    "step_newton_iterations": _series_value(stats, "step_newton_iterations", idx),
                    "step_linear_iterations": _series_value(stats, "step_linear_iterations", idx),
                    "line_search_iterations": _series_value(stats, "step_line_search_iterations", idx),
                    "line_search_fallback_count": fallback_count,
                    "deflation_basis_dim_solve_max": _series_value(stats, "step_deflation_basis_dim_solve_max", idx),
                    "deflation_basis_dim_end_last": _series_value(stats, "step_deflation_basis_dim_end_last", idx),
                    "step_linear_solve_time_s": _series_value(stats, "step_linear_solve_time", idx),
                    "step_linear_preconditioner_time_s": _series_value(stats, "step_linear_preconditioner_time", idx),
                    "step_linear_orthogonalization_time_s": _series_value(stats, "step_linear_orthogonalization_time", idx),
                    "step_wall_time_s": _series_value(stats, "step_wall_time", idx),
                    "step_newton_relres_end": _series_value(stats, "step_newton_relres_end", idx),
                    "step_newton_relcorr_end": _series_value(stats, "step_newton_relcorr_end", idx),
                }
            )
    if accepted_rows:
        path = out_dir / "accepted_continuation_steps.csv"
        _write_csv(path, accepted_rows)
        written.append(path)

    attempt_rows: list[dict[str, object]] = []
    for record in attempt_records:
        attempt_complete = dict(record.get("attempt_complete", {}))
        iterations = list(record.get("newton_iterations", []))
        last_iteration = dict(iterations[-1]) if iterations else {}
        newton_iterations = attempt_complete.get("newton_iterations")
        if newton_iterations is None:
            newton_iterations = len(iterations)
        linear_iterations = attempt_complete.get("linear_iterations")
        if linear_iterations is None:
            linear_iterations = sum(int(iteration.get("linear_iterations", 0) or 0) for iteration in iterations)
        line_search_iterations = attempt_complete.get("line_search_iterations")
        if line_search_iterations is None:
            line_search_iterations = sum(
                int(iteration.get("line_search_iterations", 0) or 0) for iteration in iterations
            )
        deflation_basis_dim_solve_max = attempt_complete.get("deflation_basis_dim_solve_max")
        if deflation_basis_dim_solve_max is None:
            deflation_basis_dim_solve_max = _max_present(iterations, "deflation_basis_dim_solve")
        deflation_basis_dim_end_last = attempt_complete.get("deflation_basis_dim_end_last")
        if deflation_basis_dim_end_last is None:
            deflation_basis_dim_end_last = last_iteration.get("deflation_basis_dim_end")
        linear_solve_time = attempt_complete.get("linear_solve_time")
        if linear_solve_time is None:
            linear_solve_time = sum(float(iteration.get("linear_solve_time", 0.0) or 0.0) for iteration in iterations)
        linear_preconditioner_time = attempt_complete.get("linear_preconditioner_time")
        if linear_preconditioner_time is None:
            linear_preconditioner_time = sum(
                float(iteration.get("linear_preconditioner_time", 0.0) or 0.0) for iteration in iterations
            )
        linear_orthogonalization_time = attempt_complete.get("linear_orthogonalization_time")
        if linear_orthogonalization_time is None:
            linear_orthogonalization_time = sum(
                float(iteration.get("linear_orthogonalization_time", 0.0) or 0.0) for iteration in iterations
            )
        attempt_wall_time = attempt_complete.get("attempt_wall_time")
        if attempt_wall_time is None:
            attempt_wall_time = sum(float(iteration.get("iteration_wall_time", 0.0) or 0.0) for iteration in iterations)
        attempt_rows.append(
            {
                "phase": record.get("phase"),
                "target_step": record.get("target_step"),
                "attempt_in_step": record.get("attempt_in_step"),
                "init_attempt": record.get("init_attempt"),
                "omega_target": record.get("omega_target"),
                "lambda_before": record.get("lambda_before"),
                "lambda_after": attempt_complete.get("lambda_after", record.get("lambda_before")),
                "newton_iterations": newton_iterations,
                "linear_iterations": linear_iterations,
                "line_search_iterations": line_search_iterations,
                "line_search_fallback_count": sum(
                    int(bool(iteration.get("line_search_fallback_used", False))) for iteration in iterations
                ),
                "line_search_mode": _first_present(iterations, "line_search_mode"),
                "deflation_basis_dim_solve_max": deflation_basis_dim_solve_max,
                "deflation_basis_dim_end_last": deflation_basis_dim_end_last,
                "linear_solve_time_s": linear_solve_time,
                "linear_preconditioner_time_s": linear_preconditioner_time,
                "linear_orthogonalization_time_s": linear_orthogonalization_time,
                "attempt_wall_time_s": attempt_wall_time,
                "final_rel_residual": last_iteration.get("rel_residual"),
                "final_stopping_value": last_iteration.get("stopping_value"),
                "final_alpha": last_iteration.get("alpha"),
                "final_status": last_iteration.get("status"),
            }
        )
    if attempt_rows:
        path = out_dir / "all_attempts_summary.csv"
        _write_csv(path, attempt_rows)
        written.append(path)

    newton_rows: list[dict[str, object]] = []
    for record in attempt_records:
        for iteration in list(record.get("newton_iterations", [])):
            newton_rows.append(
                {
                    "phase": record.get("phase"),
                    "target_step": record.get("target_step"),
                    "attempt_in_step": record.get("attempt_in_step"),
                    "init_attempt": record.get("init_attempt"),
                    "newton_iteration": iteration.get("iteration"),
                    "omega_target": record.get("omega_target"),
                    "lambda_value": iteration.get("lambda_value"),
                    "accepted_delta_lambda": iteration.get("accepted_delta_lambda"),
                    "rel_residual": iteration.get("rel_residual"),
                    "stopping_value": iteration.get("stopping_value"),
                    "alpha": iteration.get("alpha"),
                    "line_search_iterations": iteration.get("line_search_iterations"),
                    "line_search_mode": iteration.get("line_search_mode"),
                    "line_search_fallback_used": iteration.get("line_search_fallback_used"),
                    "linear_iterations": iteration.get("linear_iterations"),
                    "linear_solve_time_s": iteration.get("linear_solve_time"),
                    "linear_preconditioner_time_s": iteration.get("linear_preconditioner_time"),
                    "linear_orthogonalization_time_s": iteration.get("linear_orthogonalization_time"),
                    "iteration_wall_time_s": iteration.get("iteration_wall_time"),
                    "deflation_basis_dim_solve": iteration.get("deflation_basis_dim_solve"),
                    "deflation_basis_dim_end": iteration.get("deflation_basis_dim_end"),
                    "status": iteration.get("status"),
                }
            )
    if newton_rows:
        path = out_dir / "all_newton_iterations.csv"
        _write_csv(path, newton_rows)
        written.append(path)

    return written


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _first_present(rows: list[dict[str, object]], key: str) -> object:
    for row in rows:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _max_present(rows: list[dict[str, object]], key: str) -> object:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return max(values)


def _series_value(stats: dict[str, object], key: str, index: int) -> object:
    values = stats.get(key, [])
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return None
    return values[index]


def _history_step_value(values: list[object], step_no: int) -> object:
    index = int(step_no) - 1
    if index < 0 or index >= len(values):
        return None
    return values[index]


def write_vtu(
    out_path: Path,
    *,
    points: np.ndarray,
    cell_blocks: list[tuple[str, np.ndarray]],
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, np.ndarray] | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be (n_points, 3)")

    point_data = {} if point_data is None else dict(point_data)
    cell_data = {} if cell_data is None else dict(cell_data)

    connectivity_parts: list[np.ndarray] = []
    offsets_parts: list[np.ndarray] = []
    types_parts: list[np.ndarray] = []
    total = 0
    total_cells = 0
    for cell_type, block in cell_blocks:
        vtk_type, vtk_block = _normalize_cell_block(cell_type, block)
        connectivity_parts.append(vtk_block.reshape(-1))
        total += vtk_block.shape[1] * vtk_block.shape[0]
        offsets_parts.append(np.arange(vtk_block.shape[1], total + 1, vtk_block.shape[1], dtype=np.int64))
        types_parts.append(np.full(vtk_block.shape[0], vtk_type, dtype=np.uint8))
        total_cells += vtk_block.shape[0]

    connectivity = np.concatenate(connectivity_parts) if connectivity_parts else np.empty(0, dtype=np.int64)
    offsets = np.concatenate(offsets_parts) if offsets_parts else np.empty(0, dtype=np.int64)
    types = np.concatenate(types_parts) if types_parts else np.empty(0, dtype=np.uint8)

    vtk = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(vtk, "UnstructuredGrid")
    piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(points.shape[0]),
        NumberOfCells=str(total_cells),
    )

    point_data_node = ET.SubElement(piece, "PointData")
    for name, values in point_data.items():
        _append_data_array(point_data_node, name, np.asarray(values))

    cell_data_node = ET.SubElement(piece, "CellData")
    for name, values in cell_data.items():
        _append_data_array(cell_data_node, name, np.asarray(values))

    points_node = ET.SubElement(piece, "Points")
    _append_data_array(points_node, None, np.asarray(points, dtype=np.float64), n_components=3)

    cells_node = ET.SubElement(piece, "Cells")
    _append_data_array(cells_node, "connectivity", connectivity.astype(np.int64))
    _append_data_array(cells_node, "offsets", offsets.astype(np.int64))
    _append_data_array(cells_node, "types", types.astype(np.uint8))

    tree = ET.ElementTree(vtk)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _normalize_cell_block(cell_type: str, block: np.ndarray) -> tuple[int, np.ndarray]:
    block = np.asarray(block, dtype=np.int64)
    if block.ndim != 2:
        raise ValueError(f"Cell block {cell_type!r} must be 2D.")

    if cell_type == "triangle":
        if block.shape[1] != 3:
            raise ValueError("triangle block must have width 3")
        return 5, block
    if cell_type == "triangle6":
        if block.shape[1] != 6:
            raise ValueError("triangle6 block must have width 6")
        # Internal order: [v0, v1, v2, e12, e20, e01]
        return 22, block[:, [0, 1, 2, 5, 3, 4]]
    if cell_type == "tetra":
        if block.shape[1] != 4:
            raise ValueError("tetra block must have width 4")
        return 10, block
    if cell_type == "tetra10":
        if block.shape[1] != 10:
            raise ValueError("tetra10 block must have width 10")
        # Internal order: [v0, v1, v2, v3, e01, e12, e02, e13, e23, e03]
        return 24, block[:, [0, 1, 2, 3, 4, 5, 6, 9, 7, 8]]
    if cell_type == "VTK_LAGRANGE_TRIANGLE":
        if block.shape[1] != 15:
            raise ValueError("VTK_LAGRANGE_TRIANGLE block must have width 15")
        return 69, block
    if cell_type == "VTK_LAGRANGE_TETRAHEDRON":
        if block.shape[1] != 35:
            raise ValueError("VTK_LAGRANGE_TETRAHEDRON block must have width 35")
        return 71, block
    raise ValueError(f"Unsupported cell_type {cell_type!r}")


def _append_data_array(node: ET.Element, name: str | None, values: np.ndarray, n_components: int | None = None) -> None:
    arr = np.asarray(values)
    if arr.ndim == 1:
        components = 1 if n_components is None else n_components
        flat = arr.reshape(-1, components) if components > 1 else arr.reshape(-1, 1)
    elif arr.ndim == 2:
        flat = arr
        components = arr.shape[1]
    else:
        raise ValueError("Only 1D or 2D arrays can be exported to VTU.")

    vtk_type = _vtk_type_for_dtype(arr.dtype)
    attrib = {"type": vtk_type, "format": "ascii"}
    if name is not None:
        attrib["Name"] = name
    if components > 1:
        attrib["NumberOfComponents"] = str(components)
    data = ET.SubElement(node, "DataArray", attrib=attrib)
    if components == 1:
        data.text = _format_ascii(flat[:, 0])
    else:
        data.text = _format_ascii(flat.reshape(-1))


def _create_h5_dataset(group: h5py.Group, name: str, values: np.ndarray) -> None:
    arr = np.asarray(values)
    if arr.dtype.kind in {"U", "S", "O"}:
        text = np.asarray(arr, dtype=str)
        group.create_dataset(name, data=np.char.encode(text, encoding="utf-8"))
        return
    group.create_dataset(name, data=arr)


def _format_ascii(values: np.ndarray) -> str:
    arr = np.asarray(values).reshape(-1)
    if np.issubdtype(arr.dtype, np.integer):
        return " ".join(str(int(v)) for v in arr)
    return " ".join(f"{float(v):.16e}" for v in arr)


def _vtk_type_for_dtype(dtype: np.dtype) -> str:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return "Float64" if dt.itemsize >= 8 else "Float32"
    if np.issubdtype(dt, np.unsignedinteger):
        if dt.itemsize <= 1:
            return "UInt8"
        if dt.itemsize <= 2:
            return "UInt16"
        if dt.itemsize <= 4:
            return "UInt32"
        return "UInt64"
    if np.issubdtype(dt, np.integer):
        if dt.itemsize <= 1:
            return "Int8"
        if dt.itemsize <= 2:
            return "Int16"
        if dt.itemsize <= 4:
            return "Int32"
        return "Int64"
    raise TypeError(f"Unsupported dtype {dtype!r} for VTU export.")
