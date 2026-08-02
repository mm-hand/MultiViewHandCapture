# Cup asset provenance

This directory is copied without geometry changes from the object bundled at:

`UltraDexGrasp/third_party/DGN_2k_processed/DGN_2k/processed_data/`
`mujoco_Cole_Hardware_Mug_Classic_Blue/`

The runtime visual is `mesh/simplified.obj`. Its upstream SHA-256 is
`2ca38c4ad436015a0bcf7e94dad7289547178472ac6ce122067459613acde0ca`.
It has 2,000 vertices, 4,000 faces, is watertight, and is a single connected
component, so the handle is part of the same mesh as the cup body.

The copied directory also retains the upstream normalized mesh, info metadata,
and COACD URDF/convex pieces for provenance and future collision refinement.
The current simulator follows UltraDexGrasp's object loading behavior: it uses
the simplified mesh for rendering and one convex collision generated from that
same file.

No standalone license file was present beside this processed object in the
local DGN bundle. Treat it as a local research asset and verify the upstream
DGN/source-object redistribution terms before publishing it.
