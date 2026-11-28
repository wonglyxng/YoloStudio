<p align="center">
  <img src="src/yolo_studio/resources/logo.png" alt="YoloStudio Logo" width="112" />
</p>

# YOLO26 视频抽帧与训练工具 (YoloStudio)

面向 **录制视频 → 抽帧 → 原生 PyQt6 标注与复核工作台 → 自动智能标注（LLM / SAM 3） → YOLO26 训练 / 验证 / 推理 / 导出** 的一体化桌面 GUI 工具（同时提供 CLI 命令行入口）。

## 核心功能

1. **视频抽帧**：支持单视频或文件夹批量抽帧，提供按秒间隔、按帧步长、按目标帧数均匀抽帧、起止时间裁剪、相似帧去重、分辨率限制与实时画面预览。
2. **原生 PyQt6 轻量标注与复核工作台**：在「2. 标注工作台」内置纯 PyQt6 高速矩形标注与复核画布（零外部软件或 `PyQt6-WebEngine` 依赖，秒级启动），支持以鼠标或选中框为锚点无级缩放（≥ 200% 自动像素锐化）、抓手模式（`H`）/拖空白/右键拖拽/中键/全局`Space+左键`自由平移、拖拽与两点点击双模式画框（`R`/`W`）+ 边缘自动滚屏、**本地 SAM 2.1 多点正负提示点选一键标注（`Q`，左键前景/右键排除武器特效，`Enter` 落框，支持半透明物体轮廓与鼠标悬停实时预览，内置图像特征 LRU 缓存与热激活切图预编码，同图连点与收框近实时响应）**、**SAM 2.1 连续帧向后追踪传播（标好当前帧后一键向后追踪多帧，支持冲突合并与切镜头防爆框）**、自适应控制点与方向键微调、`Tab` 轮选重叠框、`Ctrl+C`/`Ctrl+V` 跨图粘贴、**`Shift+V` / `←上帧` 一键复用上一张图框（跨分辨率等比映射与自动防重）**、**`X` 全模式常显十字准星（中心留空细准星，颜色/线型/虚线长度/描边可在设置中切换）**、**`B` 暗部/对比度非破坏性临时视图增强（支持 Gamma/对比度/亮度实时滑块微调）**、`L` 显隐标签、`1~9` 速切类别、**`C` 可搜索类别快速选择器（含可选“画完自动搜类”与回车新建）**、**单框隐藏（`V`）/锁定（`K`，防误拖防误删）与按类别显隐/Solo 只看某一类**、**目标框填充/描边透明度实时滑块（含一键“仅描边”模式）**、**类别全目录重命名/合并（自动备份原标注）**、**`?`/`F1` 快捷键速查面板**、「保持视野」连续帧精修、右键快捷菜单、自动保存、`F` 一键确认通过并跳下一张、AI 漏标提醒横幅、本地 SAM 2.1 一键贴边收框、**单框局部放大 AI 精修（`G`，读取右上角全局 AI 设置仅上传单目标局部图，极省 Token）**与**轻量级模型反查脏标注/漏标（使用已训练 `.pt` 权重离线扫描当前标注集，按异常严重度置顶疑似漏标/错类/误标/偏移图，画布非破坏性叠加预测幽灵框并支持一键采纳漏标）**；读写 100% 兼容标准 LabelMe / X-AnyLabeling `.json` 格式，并与前后标签页一键联动打开目录、导入 `classes.txt` 或定位单张图片。
3. **全局设置中心与自动智能标注（零样本冷启动、SAM 3 文本自动标注、预标质检与标准收框）**：右上角常驻 **「⚙ 设置」** 弹窗（快捷键 `Ctrl+,`），集中管理跨页共享的 **AI 大模型配置**（Google Gemini API / OpenAI 官方 API / OpenAI 兼容接口、思考/推理级别防呆与一键连接测试）、**本地视觉模型配置**（SAM 2.1 / SAM 3 默认权重与 GPU 加速偏好）以及 **通用设置**（工作台默认偏好、记住上次目录与恢复默认），供「2. 标注工作台」与「3. 自动智能标注」全局共享；第 3 页支持 **基准模型预标 + LLM 质检精修**、**已有标注 (`.json`) + LLM 质检精修**、**LLM 零样本直接标注（冷启动，无需模型）** 以及 **SAM 3 文本自动标注（纯本地零 API 费用 / 可选叠加 LLM 二次质检）** 四种模式，自动绘制编号框 + 局部裁剪图送检，剔除误检、修正错类、发现/补充漏标，并支持 **LLM 标记松框 / LLM 局部精修收框 / 本地 SAM 像素级标准收框**（含安全门禁与虚线原框对比），结果分流至 `auto_passed` / `llm_fixed` / `need_review`，支持分流筛选、加载历史报告回看及双击结果行直接跳转标注工作台精修。
4. **数据集转换与校验**：同时支持 **工作台标准 `.json`（兼容 LabelMe / X-AnyLabeling）** 与 **YOLO `.txt` + `classes.txt`**，自动校验、统计类别并划分为 `train/val/test` 生成 `data.yaml`。
5. **YOLO26 目标检测训练与评估导出**：支持 `yolo26n.pt` / `yolo26s.pt` / `yolo26m.pt` / `yolo26l.pt` / `yolo26x.pt` 及自定义 `.pt` 权重，内置场景超参数预设（通用自然场景 / 游戏与 UI 弱增强 / 小样本微调）、优化器与数据增强细粒度调节及任意附加参数透传，训练日志完整输出提交配置与实际生效超参数，支持安全停止、断点续训、验证推理预览；评估与导出页可自动读取 `.pt` 权重的训练尺寸与类别顺序并同步提示，独立配置导出尺寸（方形/矩形）、`batch`、`dynamic`、`nms` 模式、`opset`、`simplify` 与 `FP16`，并同步提供 CLI `export` 子命令支持导出 `ONNX`、`TorchScript`、`TensorRT` 等部署格式。

## 开源协议

本项目采用 **GNU General Public License v3.0 (`GPL-3.0-only`)** 开源协议，详见 [LICENSE](LICENSE) 与 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 快速启动

双击运行 `run_gui.bat`，或执行：

```powershell
.venv\Scripts\python.exe main.py
```

安装或更新依赖：

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> 注意：`ultralytics` 依赖 `opencv-python`，本项目使用 `opencv-contrib-python-headless`，两者都会写入同一个 `cv2` 包而互相覆盖。安装或更新依赖后请执行以下命令，只保留后者（它已包含本项目需要的全部 OpenCV 功能）：
>
> ```powershell
> .venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-contrib-python-headless
> .venv\Scripts\python.exe -m pip install --no-deps opencv-contrib-python-headless
> ```
>
> 之后 `pip check` 会提示 `ultralytics` 缺少 `opencv-python`，可以忽略。

## 独立便携整合包发布（面向无 Python 电脑）

针对未安装 Python 或 CUDA Toolkit 的目标 Windows x64 电脑，可执行以下命令构建免安装绿色便携整合包（含无黑框原生启动器 `YoloStudio.exe` 与 `YoloStudio-CLI.cmd`）：

```powershell
# 日常代码迭代极速发版（约 55 秒）：仅编译核心 .pyd + 启动器并输出 ~5.6 MiB 热更新补丁包
.\scripts\build_portable.ps1 --bump patch --patch-only --variant cpu --protect pyd --archive 7z

# 依赖未变更时复用已预编译 runtime/ 底座快速产出完整包（配合 --compression fast 提速归档）
.\scripts\build_portable.ps1 --mode clean --variant gpu --protect pyd --reuse-runtime --download-weights yolo26n.pt yolo26s.pt --archive 7z --compression fast --patch-archive

# 首次全新构建完整包（默认开启 --protect pyd 防查看防篡改；对外按 GPLv3 开源分发请附加 --protect none）
.\scripts\build_portable.ps1 --mode clean --variant gpu --protect pyd --download-weights yolo26n.pt yolo26s.pt --archive 7z
```

## 项目文档

完整索引见 [项目文档索引.md](项目文档索引.md)：
- [001_录制视频抽帧工具使用说明.md](docs/001_录制视频抽帧工具使用说明.md)
- [002_YOLO26训练使用说明.md](docs/002_YOLO26训练使用说明.md)
- [003_独立便携整合包构建与发布说明.md](docs/003_独立便携整合包构建与发布说明.md)
