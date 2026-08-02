# Object asset provenance

`bowl/` is a project-local copy of UltraDexGrasp's
`asset/object_mesh/bowl/` example object. It is included so the default
pick-and-place scene has the same mesh object without any runtime dependency
on the UltraDexGrasp checkout.

The source project is licensed under Apache-2.0. A copy is stored at
`../robot/LICENSE-UltraDexGrasp-Apache-2.0.txt`.

Runtime code uses `bowl/mesh/simplified.obj` for both the visual mesh and a
convex collision shape, matching the original environment's default object.

`cup/` is a project-local, byte-for-byte copy of the processed
`mujoco_Cole_Hardware_Mug_Classic_Blue` object shipped in UltraDexGrasp's DGN
object bundle. Its handle and body are one connected visual mesh. The complete
processed object directory is bundled here so runtime does not depend on the
UltraDexGrasp checkout.

`can/` and `box/` are original procedural assets generated locally by
`generate_builtin_objects.py`; they were not downloaded. Each directory
contains its own provenance note. The generator is deterministic and uses only
Python's standard library.
