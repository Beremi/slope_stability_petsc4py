# Converter Index

This index records the deterministic scripts that created the canonical mesh variants from
legacy mesh or text-bundle inputs.

| Asset | Legacy source | Script | Canonical outputs | Notes |
| --- | --- | --- | --- | --- |
| [2d_homo_slope](2d_homo_slope) | generated homogeneous 2D slope definition | [convert_to_msh.py](2d_homo_slope/legacy/convert_to_msh.py) | `h1.0.msh`, `h0.5.msh` | Rebuilds from the old generated geometry recipe kept under `legacy/source/2d_generated_homo/`. |
| [2d_sloan2013](2d_sloan2013) | generated Sloan 2013 2D seepage mesh | [convert_to_msh.py](2d_sloan2013/legacy/convert_to_msh.py) | `default.msh` | Rebuilds from `src/slope_stability/mesh/sloan2013_2d.py`; no extra raw mesh files were part of the old asset. |
| [2d_kozinec](2d_kozinec) | MATLAB-style text bundle | [convert_to_msh.py](2d_kozinec/legacy/convert_to_msh.py) | `default.msh` | Reads the text bundle now stored in `legacy/source/`. |
| [2d_luzec](2d_luzec) | MATLAB-style text bundle | [convert_to_msh.py](2d_luzec/legacy/convert_to_msh.py) | `default.msh` | Reads the text bundle now stored in `legacy/source/`. |
| [2d_franz_dam](2d_franz_dam) | MATLAB-style text bundle | [convert_to_msh.py](2d_franz_dam/legacy/convert_to_msh.py) | `default.msh` | Reads the text bundle now stored in `legacy/source/`. |
| [3d_homo_slope](3d_homo_slope) | old `3d_homo_ssr` and `3d_homo_ll` tagged Gmsh meshes | [retag_to_canonical.py](3d_homo_slope/legacy/retag_to_canonical.py) | `adaptive_family_a_*.msh`, `uniform_family_a.msh`, `adaptive_family_b_*.msh`, `uniform_family_b.msh` | Retags old numeric physical groups and collapses homogeneous regions to one logical region. |
| [3d_hetero_slope](3d_hetero_slope) | old `3d_hetero_ssr` and `3d_hetero_ll` tagged Gmsh meshes | [retag_to_canonical.py](3d_hetero_slope/legacy/retag_to_canonical.py) | `adaptive_family_a_*.msh`, `uniform_family_a.msh`, `adaptive_family_b_*.msh`, `uniform_family_b.msh` | Retags old numeric physical groups to logical regions and supports. |
| [3d_hetero_seepage](3d_hetero_seepage) | old water-level-tagged Gmsh meshes | [retag_to_canonical.py](3d_hetero_seepage/legacy/retag_to_canonical.py) | `family_*.msh`, `concave_family_*.msh` | Retags old physical groups and extracts canonical seepage node supports. |
| [3d_hetero_seepage_transition](3d_hetero_seepage_transition) | old COMSOL-exported transition mesh | [convert_to_msh.py](3d_hetero_seepage_transition/legacy/convert_to_msh.py) | `transition_default.msh` | Retags the COMSOL `.msh` and recreates hydraulic node sets from the old transition recipe. |
| [3d_siopt](3d_siopt) | old tagged Gmsh SIOPT meshes | [retag_to_canonical.py](3d_siopt/legacy/retag_to_canonical.py) | `reference_l0.msh`, `reference_l1.msh`, `reference_l5.msh` | Retags old numeric physical groups to logical region and support names. |

## Rerun Pattern

Run converter scripts from the repository root:

```bash
PYTHONPATH=src python meshes/3d_homo_slope/legacy/retag_to_canonical.py
```

Each script writes the canonical `.msh` files back into its parent asset directory.
