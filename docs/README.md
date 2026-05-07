# Documentation

This directory contains current architecture, config, benchmark, and study notes for the
asset-first slope-stability runtime. Historical notes are retained when they preserve useful
debug or design context.

## Active Guides

- [New Benchmark On A New Geometry](new-benchmark-new-geometry-guide.md)
- [Config Case Matrix](config-case-matrix.md)
- [3D Configuration Scheme](config-scheme-3d.md)
- [Computational Path](computational-path.md)
- [Problem Definition Audit](problem-definition-audit.md)

Start with the new-benchmark guide when adding a new problem, mesh, or benchmark. Use the
computational path note when tracing a case config through asset resolution, mesh building,
solver execution, and exports.

## Historical Notes

Older phase summaries and solver notes may mention paths or experiments that are no longer
active. Treat the current source tree, `meshes/<asset>/definition.py`, and benchmark
`case.toml` files as authoritative.
