from __future__ import annotations

from pathlib import Path

from petsc_ssr.benchmarks.logs import parse_petsc_log_events, petsc_log_top_events


def test_parse_petsc_log_events_aggregates_and_sorts_by_time() -> None:
    text = """
Event                Count      Time (sec)
MatMult                 20      1.5000e+00
SSR_PMGApply             4      3.2500e+00
MatMult                 10      7.5000e-01
Event                    1      9.9000e+01
"""

    rows = parse_petsc_log_events(text)

    assert rows[0] == {"event": "SSR_PMGApply", "count": 4, "time_s": 3.25}
    assert rows[1] == {"event": "MatMult", "count": 30, "time_s": 2.25}
    assert all(row["event"] != "Event" for row in rows)


def test_petsc_log_top_events_reads_standard_suite_log_path(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "petsc_log.txt").write_text(
        """
KSPSolve                 2      4.0000e+00
PCApply                 10      2.0000e+00
MatMult                 40      1.0000e+00
""",
        encoding="utf-8",
    )

    assert petsc_log_top_events(tmp_path, limit=2) == [
        {"event": "KSPSolve", "count": 2, "time_s": 4.0},
        {"event": "PCApply", "count": 10, "time_s": 2.0},
    ]
