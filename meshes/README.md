# Mesh Assets

This directory is a temporary problem-family sorting layer.

Current intent:

- keep one canonical mesh file per mesh variant
- keep material and boundary-condition tagging in the mesh whenever practical
- keep extra derived setup in Python `definition.py` files

Each subfolder may contain:

- one or more mesh files
- `definition.py` with temporary metadata and setup hints

Longer term, this should converge toward a more uniform mesh-plus-metadata contract.
