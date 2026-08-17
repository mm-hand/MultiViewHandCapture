# MANUS Core SDK 3.1.1 assets

This directory contains the Linux Integrated subset downloaded from the
official `MANUS_Core_3.1.1_SDK.zip` release:

- `ManusSDK_v3.1.1/include/`: unmodified official C/C++ headers;
- `ManusSDK_v3.1.1/lib/libManusSDK_Integrated.so`: unmodified runtime;
- `ManusSDK_v3.1.1/license/license.txt`: upstream license;
- `manus_sdk_bridge.cpp`: this project's small C ABI wrapper;
- `libmanus_sdk_bridge.so`: locally built wrapper used by Python.

Official archive URL:

`https://static.manus-meta.com/resources/manus_core_3/sdk/MANUS_Core_3.1.1_SDK.zip`

Checksums:

- archive SHA-256: `c5ccd3c42a501107ec79f70d8450a486fbc3925c5c1e18e606114d09f2d9d24a`
- extracted Integrated library SHA-256:
  `0e67141b97b64c089c3bbdab47980ca9822c4de19adea810a2f68722adcb3fe3`

Rebuild the wrapper after changing it with:

```bash
make -C input/manus/assets
```

The wrapper calls `CoreSdk_InitializeIntegrated()`, registers the Raw Skeleton
callback, and calls `CoreSdk_InitializeCoordinateSystemWithVUH(..., true)` with
`unitScale=1.0`. Therefore positions crossing this boundary are metres in the
WORLD/GLOBAL frame. It calls
`CoreSdk_GetRawSkeletonNodeCount()` and
`CoreSdk_GetRawSkeletonNodeInfoArray()` for every returned topology; the fixed
25-row layout is only an adapter fallback/test reference.

The wrapper also exposes the official Integrated glove-calibration sequence:
step discovery, start/run/stop/finish, and calibration blob import/export.
`track.py` uses these calls through `input/manus/calibration.py` before any
tracking UI or retargeting is started.

Runtime prerequisites for Integrated mode:

- an SDK Integrated feature on the connected MANUS license key;
- no second MANUS Core/Core Integrated instance using the same devices;
- Linux read/write permission for MANUS USB/hidraw devices (USB vendor `3325`).

Official references:

- https://docs.manus-meta.com/latest/Resources/
- https://docs.manus-meta.com/3.1.0/Software/Skeletons/
