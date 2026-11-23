# 第三方软件与开源协议声明

本项目（**YoloStudio**）采用 **GNU General Public License v3.0 (`GPL-3.0-only`)** 开源协议，详见根目录 [LICENSE](LICENSE)。

## 主要第三方依赖

- **Ultralytics YOLO (`ultralytics`)**：`AGPL-3.0`（用于 YOLO26 训练、验证、推理、导出及本地 SAM / SAM 3 推理）。
- **PyQt6 (`PyQt6`)**：`GPL-3.0` / 商业双许可（用于原生桌面图形界面及标注与复核工作台）。
- **PyTorch / TorchVision**：`BSD-3-Clause`（深度学习计算与 GPU 加速引擎）。
- **OpenCV (`opencv-contrib-python-headless`)**：`Apache-2.0`（视频解码、图像处理与抽帧去重）。
- **NumPy、Pillow、PyYAML、Pydantic、google-genai、openai**：各自遵循其原始开源协议。
- **可选 SAM 3 文本自动标注依赖（Ultralytics `CLIP`、`ftfy`、`regex`、`timm`）**：各自遵循其原始开源协议。

> 说明：本项目的「2. 标注工作台」采用原生 PyQt6 实现，不再内嵌 X-AnyLabeling 或依赖 `PyQt6-WebEngine`；生成的 `.json` 标注文件继续 100% 兼容标准 LabelMe / X-AnyLabeling 矩形标注格式。
