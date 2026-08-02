# Portable NERO + capture-MMHand robot asset

This directory is the complete runtime robot bundle used by the SAPIEN
teleoperation viewer. The URDF contains only relative paths beneath this
directory; moving `teleop_maniskill/` does not require an UltraDexGrasp
checkout or the original `assets/mmhand` directory.

## Imported sources

- NERO arm kinematics, inertials, mount, visual DAE files, and collision STL
  files were imported from UltraDexGrasp's
  `asset/nero_mmhand_orig_urdf/nero_mmhand_orig_combined_sapien.urdf` and
  `asset/nero_mmhand_orig_urdf/meshes/nero/` on 2026-07-31.
- The hand's complete 32-link kinematic tree, inertials, mesh frames, mesh
  scales, and 21 active joints were imported from this capture project's
  `assets/mmhand/urdf/hand.urdf` and `assets/mmhand/meshes/` on 2026-07-31.
- `nero_capture_mmhand.urdf` is a modified combination. Hand link and fixed
  joint names are namespaced to avoid `base_link` collisions, runtime mesh
  paths are made local, and display materials are split into palm, joint-base,
  phalanx, and fingertip colors. Active joint names and all geometry and
  kinematic transforms remain unchanged. The fixed hand mount transform is
  copied from the UltraDexGrasp combined model. Sixteen capture links contain
  singular inertia tensors due to rounded zero entries; the generated URDF
  adds only `1e-9 kg m^2` to their three diagonal entries so SAPIEN/PhysX can
  load them while retaining the source mass, center of mass, and practical
  dynamics. All other inertial values are copied verbatim.

The NERO visual geometry uses the original DAE files so their embedded CAD
materials are retained. The corresponding STL files are used only for
collision. MMHand retains the capture URDF's original hybrid geometry: 27
high-resolution STL visual meshes; 12 STL collision meshes for the palm and
finger bases; and 45 cylinder/sphere collision primitives on the articulated
phalanges and fingertips.

`MANIFEST.json` records the exact byte size and SHA-256 digest of every file
required by the robot asset. `prepare_full_fidelity_urdf.py validate` checks
the 42-link, 28-DOF, local-path contract without reading either import source.

## Licenses and redistribution caution

- `LICENSE-UltraDexGrasp-Apache-2.0.txt` is the Apache-2.0 license shipped at
  the root of the UltraDexGrasp checkout from which the NERO model was
  imported. This file is preserved with the modified combined URDF.
- `LICENSE-MMHand-dex-retargeting-MIT.txt` is the third-party license notice
  already distributed with this project's MMHand asset bundle. It contains
  the dex-retargeting code and asset MIT notices.

Neither source bundle contains a mesh-specific notice that explicitly states
the ownership or redistribution terms of AgileX NERO's OEM CAD files or the
MMHand hardware CAD independently of the enclosing repository licenses.
Consequently, this bundle is suitable for the current local research use, but
those OEM asset rights should be confirmed with the model providers before
publishing or redistributing the mesh files outside the project.
