"""Benchmark log artifact helpers.

PETSc's ``-options_left`` diagnostics are emitted through normal output in the
runner configurations used by the suite tools.  The helpers here keep that
stdout parsing local to benchmark reporting and materialize a stable
``logs/options_left.txt`` artifact for post-run checks.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence


_CLEAN_MARKERS = ("there are no unused options", "no unused options")
_CHECK_MARKERS = ("options left", "unused option", "not used")
_PETSC_EVENT_ROW_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_./:-]*)\s+([0-9]+)\s+([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\b"
)
_PETSC_EVENT_HEADER_NAMES = {"event", "stage", "summary", "objects", "memory", "flop", "time"}


def materialize_suite_log_artifacts(output_dir: str | Path, *, command: Sequence[str] = ()) -> None:
    """Write derived log artifacts that PETSc does not write directly."""

    out = Path(output_dir)
    logs_dir = out / "logs"
    stdout = logs_dir / "stdout.txt"
    if not stdout.exists() or not _command_requests_options_left(command):
        return

    options_left = logs_dir / "options_left.txt"
    if options_left.exists() and options_left.stat().st_size > 0:
        return

    text = stdout.read_text(encoding="utf-8", errors="replace")
    status = classify_options_left_text(text)
    options_left.write_text(_render_options_left_artifact(status, text), encoding="utf-8")


def options_left_status(output_dir: str | Path) -> str:
    """Return ``clean``, ``check``, ``unknown``, or ``missing`` for a run root."""

    out = Path(output_dir)
    options_left = out / "logs" / "options_left.txt"
    if options_left.exists():
        text = options_left.read_text(encoding="utf-8", errors="replace")
        status = _status_field(text)
        if status is not None:
            return status
        return classify_options_left_text(text)

    stdout = out / "logs" / "stdout.txt"
    if not stdout.exists():
        return "missing"
    return classify_options_left_text(stdout.read_text(encoding="utf-8", errors="replace"))


def petsc_log_top_events(output_dir: str | Path, *, limit: int = 10) -> list[dict[str, object]]:
    """Return the top PETSc log events by inclusive event time for one run root."""

    log_path = Path(output_dir) / "logs" / "petsc_log.txt"
    if not log_path.exists():
        return []
    events = parse_petsc_log_events(log_path.read_text(encoding="utf-8", errors="replace"))
    return events[: max(0, int(limit))]


def parse_petsc_log_events(text: str) -> list[dict[str, object]]:
    """Parse PETSc ``-log_view`` event rows into aggregate event timings.

    PETSc's human log view has changed column decoration across releases, but
    event rows consistently begin with an event name, a call count, and a time
    column.  This parser intentionally ignores the rest of the line and keeps
    reports lightweight; the full log remains the authoritative artifact.
    """

    aggregate: dict[str, dict[str, object]] = {}
    for raw in text.splitlines():
        match = _PETSC_EVENT_ROW_RE.match(raw)
        if match is None:
            continue
        name = match.group(1)
        if name.strip().lower() in _PETSC_EVENT_HEADER_NAMES:
            continue
        count = int(match.group(2))
        time_s = float(match.group(3))
        if count <= 0 or time_s < 0.0:
            continue
        event = aggregate.setdefault(name, {"event": name, "count": 0, "time_s": 0.0})
        event["count"] = int(event["count"]) + count
        event["time_s"] = float(event["time_s"]) + time_s
    return sorted(aggregate.values(), key=lambda item: (-float(item["time_s"]), str(item["event"])))


def classify_options_left_text(text: str) -> str:
    lower = text.lower()
    if any(marker in lower for marker in _CLEAN_MARKERS):
        return "clean"
    if any(marker in lower for marker in _CHECK_MARKERS):
        return "check"
    return "unknown"


def _command_requests_options_left(command: Sequence[str]) -> bool:
    return any(str(token) in {"-options_left", "--petsc-opt=-options_left"} for token in command)


def _status_field(text: str) -> str | None:
    for raw in text.splitlines():
        line = raw.strip().lower()
        if not line.startswith("status:"):
            continue
        status = line.split(":", 1)[1].strip()
        if status in {"clean", "check", "unknown", "missing"}:
            return status
    return None


def _render_options_left_artifact(status: str, stdout_text: str) -> str:
    lines = [
        f"status: {status}",
        "source: stdout.txt",
        "",
    ]
    if status == "clean":
        lines.append("There are no unused options.")
    elif status == "check":
        lines.extend(_interesting_lines(stdout_text))
    else:
        lines.append("No PETSc options-left marker was found in captured stdout.")
    return "\n".join(lines).rstrip() + "\n"


def _interesting_lines(text: str) -> list[str]:
    selected: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        lower = line.lower()
        if line and any(marker in lower for marker in _CHECK_MARKERS):
            selected.append(line)
    return selected or ["PETSc reported unused options; inspect logs/stdout.txt for details."]
