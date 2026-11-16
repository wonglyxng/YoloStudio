# YOLO26 视频抽帧与训练工具 (YoloStudio)

面向 **录制视频 → 抽帧 → X-AnyLabeling 标注 → MLLM 自动纠错清洗 → YOLO26 训练 / 验证 / 推理 / 导出** 的一体化桌面 GUI 工具（同时提供 CLI 命令行入口）。

## 核心功能

1. **视频抽帧**：支持单视频或文件夹批量抽帧，提供按秒间隔、按帧步长、按目标帧数均匀抽帧、起止时间裁剪、相似帧去重、分辨率限制与实时画面预览。
2. **X-AnyLabeling 标注衔接**：一键打开 X-AnyLabeling 指向抽帧目录；同时支持 **X-AnyLabeling 原生 `.json`** 与 **导出的 YOLO `.txt` + `classes.txt`**，自动校验、统计类别并划分为 `train/val/test` 生成 `data.yaml`。
3. **MLLM 自动标注纠错与补漏**：支持 **基准模型低阈值推理** 或 **已有 `.json`** 两种来源，集成 **Google Gemini API (`google-genai`)** 与 **OpenAI 兼容多模态接口**（通义千问 Qwen-VL / 硅基流动 / 本地 Ollama / vLLM），自动绘制编号框 + 局部裁剪图送审，剔除误检、修正错标类、发现/补充漏标并分流至 `auto_passed` / `llm_fixed` / `need_review`，支持界面对比预览与一键打开 X-AnyLabeling 复核。
4. **YOLO26 目标检测训练**：支持 `yolo26n.pt` / `yolo26s.pt` / `yolo26m.pt` / `yolo26l.pt` / `yolo26x.pt` 及自定义 `.pt` 权重，实时展示 Loss、Precision、Recall、mAP50、mAP50-95 与训练日志，支持安全停止与断点续训。
5. **验证、推理与导出**：支持模型验证评估、图片/视频推理结果可视化预览，以及一键导出 `ONNX`、`TorchScript`、`TensorRT`。

## 快速启动

双击运行 `run_gui.bat`，或执行：

```powershell
.venv\Scripts\python.exe main.py
```

安装或更新新增依赖（`google-genai`、`openai`、`pydantic`）：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

详细使用说明请参考：
- [001_录制视频抽帧工具使用说明.md](docs/001_录制视频抽帧工具使用说明.md)
- [002_YOLO26训练使用说明.md](docs/002_YOLO26训练使用说明.md)
