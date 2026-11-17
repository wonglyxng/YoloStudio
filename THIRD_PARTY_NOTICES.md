# 第三方软件与开源协议声明

本项目（**YoloStudio**）已与 **X-AnyLabeling** 进行单进程源码级集成，整体仓库及分发产物遵循 **GNU General Public License v3.0 (`GPL-3.0-only`)**，详见根目录 [LICENSE](LICENSE)。

## 1. X-AnyLabeling

- **上游仓库**：<https://github.com/CVHub520/X-AnyLabeling>
- **引入路径**：`third_party/X-AnyLabeling/`
- **锁定版本**：`v4.3.3`（上游提交 `a8acb46`）
- **作者与维护者**：Wei Wang / CVHub
- **开源协议**：`GPL-3.0-only`（原仓库许可文本保留于 `third_party/X-AnyLabeling/LICENSE`）
- **集成方式说明**：
  - 本仓库完整保留了 `third_party/X-AnyLabeling/` 源码目录（仅移除了嵌套的 `.git` 目录，以便作为单一项目统一受控管理），未改动原版标注画布与 AI 辅助标注核心逻辑。
  - 在运行环境（`.venv`）中通过可编辑源码包形式安装 `third_party/X-AnyLabeling[gpu]`，并由 [src/yolo_studio/gui/tab_labeling.py](src/yolo_studio/gui/tab_labeling.py) 在 `PyQt6` 单进程中内嵌托管原版 `anylabeling.views.mainwindow.MainWindow`。

## 2. 其他主要依赖

- **Ultralytics YOLO (`ultralytics`)**：`AGPL-3.0`（用于 YOLO26 训练、验证、推理与导出）。
- **PyQt6 / PyQt6-WebEngine**：`GPL-3.0` / 商业双许可。
- **ONNX Runtime GPU (`onnxruntime-gpu`)**：`MIT`（用于 X-AnyLabeling GPU AI 辅助标注推理）。
- **OpenCV、PyTorch、NumPy、Pillow、Pydantic、google-genai、openai**：各自遵循其原始开源协议。
