# MultiView Hand Capture

一个扁平、可直接运行的双目 3D 手部跟踪脚本。

处理顺序：MediaPipe 21 点检测 → 双目三角化与异常值拦截 → 3D 点 One Euro
滤波 → ±20% 骨长约束 → MCP/PIP/DIP 屈伸角非负约束 → 角度 One Euro
滤波 → 最小旋转修正 21 个 3D 点。连续坏帧会保留最后结果 3 帧，之后隐藏并
重置滤波器。

## 使用

```bash
python -m pip install -r requirements.txt
python calibrate.py   # 已有 stereo_params.json 时无需重复标定
python track.py
```

相机编号、左右画面旋转、棋盘格尺寸和滤波参数都在 `config.py`。标定时使用
`pattern.png`，让棋盘格同时完整出现在两个画面中；按 `C` 采样，至少 10 对后
按 `Q` 计算并保存。

`StereoProcessor.process_frame(frame)` 的主要输出为：

```python
{
    "found": True,
    "stale": False,
    "handedness": "Left",             # 按未镜像的左相机视角
    "keypoint_absolute": ...,          # (21, 3)，毫米
    "keypoint_relative": ...,          # (21, 3)，掌心坐标，米
    "phase": "GESTURE TRACKING - Left Hand",
    "quality": {
        "reprojection_error": 0.8,
        "rejected_reason": None,       # detection/reprojection/depth/hand-size...
    },
}
```

本地回归录像放在 `test_data/`，不会提交到 Git。运行全部测试：

```bash
python -m unittest -v test_hand_capture.py
```
