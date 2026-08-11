# WiLoR runtime assets

Copied from `../wilor_onnx_runtime`; the source and copied SHA-256 values were
verified equal on 2026-08-12.

```text
detector_fp16.onnx  036a78496dd65aba7507f68f7707bf190a876940edc6c38e4aaaf6278d48df32
wilor_fp16.onnx     42f0920e54c86a1b789e31c1363767b07a816dccd99b0a867313f01d673c65de
mano_faces.npy      9538bf89adad23d94074fe34e2d3f00b4512d2992a9e209f5dca7bac4add8692
```

`joint_regressor.npy` is the Float32 `mano.J_regressor` initializer extracted
from the copied WiLoR graph. The two ONNX files are stored through Git LFS.
Licensing and attribution are in `WILOR_MODEL_LICENSE.txt` and
`THIRD_PARTY_NOTICES.md`.
