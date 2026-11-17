"""无头批处理命令行启动脚本，覆盖视频抽帧、数据集构建、YOLO26 训练与 MLLM 标注纠错。

职责与真实入口：
    仓库根目录 CLI 兼容入口脚本，将 `src/` 加入 `sys.path` 后委托 `yolo_studio.cli` 执行
    `extract`、`dataset`、`train`、`review` 子命令（详见 `docs/001_录制视频抽帧工具使用说明.md`
    与 `docs/002_YOLO26训练使用说明.md`）。同时透传导出 `build_parser` 与 `main`。

运行环境与失败方式：
    需 Python >= 3.11 及对应子命令运行库；输入输出契约与退出码同 `yolo_studio.cli`。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 需先将 src 加入 sys.path 再导入内部包，以下保留 noqa: E402 / F401 指令
from yolo_studio.cli import build_parser, main  # noqa: E402, F401


if __name__ == "__main__":
    raise SystemExit(main())
