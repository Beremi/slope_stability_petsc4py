# Curved Boundaries

Curved and high-order boundary geometry belongs to mesh assets. A case should
select an asset variant and element order; it should not carry coordinate
patches, generated midpoint tables, or solver-side geometry fixes.

Current asset metadata uses `boundary_geometry` declarations that name declared
boundary supports. Validation rejects geometry patches that refer to unknown
supports, and manifest generation records the declared relationship before the
native engine starts.

The target native contract is DMPlex-native:

- boundary supports are represented by DMLabels;
- high-order coordinates live in the DMPlex coordinate section;
- generated surface nodes are projected or supplied by the asset pipeline;
- Neumann and seepage flux rules reference boundary labels and optional geometry
  patches, not coordinate-matched CSV rows.

This keeps curved P2/P4 tetrahedral boundaries and future mesh adaptation on the
same distributed path as the rest of the solver.
