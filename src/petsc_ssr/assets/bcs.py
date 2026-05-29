"""Public boundary-condition spec types and normalization helpers."""

from __future__ import annotations

from .api import DirichletBCSpec, HeadBCSpec, HydraulicStateSpec, MechanicsSpec, NeumannBCSpec, ProfileSpec, SeepageSpec
from .factories import build_seepage_spec

__all__ = [
    "DirichletBCSpec",
    "HeadBCSpec",
    "HydraulicStateSpec",
    "MechanicsSpec",
    "NeumannBCSpec",
    "ProfileSpec",
    "SeepageSpec",
    "build_seepage_spec",
]
