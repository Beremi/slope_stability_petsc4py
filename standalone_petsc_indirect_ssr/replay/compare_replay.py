#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _hist_from_log(path: Path) -> dict[str, list[float]]:
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    pat = re.compile(r'DEFLATED_MATLAB_DFGMRES label="REPLAY (d[WV])" it=\d+ rel=([0-9.eE+-]+)')
    out = {"dW": [], "dV": []}
    for label, value in pat.findall(text):
        out[label].append(float(value))
    return out


def _samples_from_log(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    initial_pat = re.compile(
        r'DEFLATED_MATLAB_DFGMRES_INITIAL label="REPLAY (d[WV])" '
        r'basis_cols=\d+ rhs_norm=([0-9.eE+-]+) beta=([0-9.eE+-]+) rel=([0-9.eE+-]+)'
    )
    sample_pat = re.compile(
        r'DEFLATED_MATLAB_DFGMRES_SAMPLE label="REPLAY (d[WV])" it=1 h00=([0-9.eE+-]+) h10=([0-9.eE+-]+)'
    )
    out: dict[str, dict[str, float]] = {"dW": {}, "dV": {}}
    for label, rhs_norm, beta, rel in initial_pat.findall(text):
        out[label].update({"rhs_norm": float(rhs_norm), "beta": float(beta), "initial_rel": float(rel)})
    for label, h00, h10 in sample_pat.findall(text):
        out[label].update({"h00": float(h00), "h10": float(h10)})
    return out


def _result_line(path: Path) -> str:
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith("REPLAY_RESULT "):
            return line
    return ""


def _diagnostic_lines(path: Path) -> list[str]:
    text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", "replace")
    prefixes = ("REPLAY_MAP_CHECK ", "REPLAY_VEC_ROUNDTRIP ", "REPLAY_VEC_DIFF ")
    return [line for line in text.splitlines() if line.startswith(prefixes)]


def _fmt_hist(values: list[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _probe_scalars(sample_dir: Path, expected: dict, label: str) -> dict[str, float]:
    probe = expected.get(f"{label}_probe")
    if not isinstance(probe, dict):
        return {}
    json_name = probe.get("json")
    if not json_name:
        return {}
    path = sample_dir / str(json_name)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    scalars = payload.get("scalars", {})
    h_col0 = scalars.get("h_col0", [])
    out: dict[str, float] = {}
    for key in ("b_norm", "beta", "initial_rel"):
        if key in scalars:
            out[key if key != "b_norm" else "rhs_norm"] = float(scalars[key])
    if isinstance(h_col0, list) and len(h_col0) >= 2:
        out["h00"] = float(h_col0[0])
        out["h10"] = float(h_col0[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_dir", type=Path)
    ap.add_argument("logs", nargs="+", type=Path)
    ns = ap.parse_args()

    meta = json.loads((ns.sample_dir / "meta.json").read_text())
    expected = meta["expected"]
    print(f"sample={ns.sample_dir}")
    print(f"omega={meta['omega']:.8e} lambda={meta['lambda']:.8e} basis_cols={meta['basis_cols']}")
    for key in ("dW", "dV"):
        print(f"{key} petsc4py iters={expected[f'{key}_iterations']} hist={_fmt_hist(expected[f'{key}_reported_residual_history'])}")
        probe = _probe_scalars(ns.sample_dir, expected, key)
        if probe:
            print(
                f"{key} petsc4py sample "
                f"rhs_norm={probe.get('rhs_norm', float('nan')):.6e} "
                f"beta={probe.get('beta', float('nan')):.6e} "
                f"rel={probe.get('initial_rel', float('nan')):.6e} "
                f"h00={probe.get('h00', float('nan')):.6e} "
                f"h10={probe.get('h10', float('nan')):.6e}"
            )
    for log in ns.logs:
        hist = _hist_from_log(log)
        samples = _samples_from_log(log)
        print(f"\nlog={log}")
        for key in ("dW", "dV"):
            print(f"{key} C iters={len(hist[key])} hist={_fmt_hist(hist[key])}")
            sample = samples.get(key, {})
            if sample:
                print(
                    f"{key} C sample "
                    f"rhs_norm={sample.get('rhs_norm', float('nan')):.6e} "
                    f"beta={sample.get('beta', float('nan')):.6e} "
                    f"rel={sample.get('initial_rel', float('nan')):.6e} "
                    f"h00={sample.get('h00', float('nan')):.6e} "
                    f"h10={sample.get('h10', float('nan')):.6e}"
                )
        line = _result_line(log)
        if line:
            print(line)
        for line in _diagnostic_lines(log):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
