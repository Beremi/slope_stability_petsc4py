"""Standardized result export helpers."""

from __future__ import annotations

from pathlib import Path
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np


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
    with np.load(npz_path, allow_pickle=True) as npz:
        for key in ("lambda_hist", "omega_hist", "Umax_hist"):
            if key in npz:
                history[key] = np.asarray(npz[key]).tolist()
        stats = {key[6:]: np.asarray(npz[key]).tolist() for key in npz.files if key.startswith("stats_")}
        if stats:
            history["stats"] = stats
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return out_path


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
