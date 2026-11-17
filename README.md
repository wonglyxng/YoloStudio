# YOLO26 视频抽帧与训练工具 (YoloStudio)

面向 **录制视频 → 抽帧 → 内嵌 X-AnyLabeling 标注 → MLLM 自动纠错清洗 → YOLO26 训练 / 验证 / 推理 / 导出** 的单进程一体化桌面 GUI 工具（同时提供 CLI 命令行入口）。

## 核心功能

1. **视频抽帧**：支持单视频或文件夹批量抽帧，提供按秒间隔、按帧步长、按目标帧数均匀抽帧、起止时间裁剪、相似帧去重、分辨率限制与实时画面预览。
2. **单进程内嵌 X-AnyLabeling 标注工作台**：在「2. 标注工作台 (X-AnyLabeling)」标签页内直接托管原版 X-AnyLabeling（`PyQt6` + `onnxruntime-gpu`），无需额外寻找或启动 `.exe`；支持人工画框与 SAM / YOLO 等 GPU AI 辅助标注，并与前后标签页一键联动打开目录、导入 `classes.txt` 或定位单张图片。
3. **数据集转换与校验**：同时支持 **X-AnyLabeling 原生 `.json`** 与 **导出的 YOLO `.txt` + `classes.txt`**，自动校验、统计类别并划分为 `train/val/test` 生成 `data.yaml`。
4. **MLLM 自动标注纠错与补漏**：支持 **基准模型低阈值推理** 或 **已有 `.json`** 两种来源，集成 **Google Gemini API (`google-genai`)** 与 **OpenAI 兼容多模态接口**（通义千问 Qwen-VL / 硅基流动 / 本地 Ollama / vLLM），自动绘制编号框 + 局部裁剪图送审，剔除误检、修正错标类、发现/补充漏标并分流至 `auto_passed` / `llm_fixed` / `need_review`，支持界面对比预览及双击结果行直接跳转内嵌工作台精修。
5. **YOLO26 目标检测训练与评估导出**：支持 `yolo26n.pt` / `yolo26s.pt` / `yolo26m.pt` / `yolo26l.pt` / `yolo26x.pt` 及自定义 `.pt` 权重，实时展示指标与训练日志，支持安全停止、断点续训、验证推理预览及一键导出 `ONNX`、`TorchScript`、`TensorRT`。

## 开源协议

因与原版 **X-AnyLabeling** 进行单进程源码级集成，本项目整体采用 **GNU General Public License v3.0 (`GPL-3.0-only`)** 开源协议，详见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 快速启动

双击运行 `run_gui.bat`，或执行：

```powershell
.venv\Scripts\python.exe main.py
```

安装或更新依赖（含 `PyQt6`、`PyQt6-WebEngine` 及本地可编辑安装的 `third_party/X-AnyLabeling[gpu]`）：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

详细使用说明请参考：
- [001_录制视频抽帧工具使用说明.md](docs/001_录制视频抽帧工具使用说明.md)
- [002_YOLO26训练使用说明.md](docs/002_YOLO26训练使用说明.md)
