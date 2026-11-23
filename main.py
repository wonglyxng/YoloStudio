"""YOLO26 训练工具桌面图形界面启动脚本。

职责与真实入口：
    仓库根目录与独立便携整合包（`app/main.py`）共用的桌面 GUI 启动脚本，
    供 `run_gui.bat`、`.venv\\Scripts\\python.exe main.py` 或便携包无黑框启动器 `YoloStudio.exe` 调用；
    将 `src/` 加入 `sys.path` 后，经 `yolo_studio.app.resolve_workspace_root` 解析可写工作区并启动主窗口
    （操作流程见 `README.md`、`docs/001_录制视频抽帧工具使用说明.md` 与 `docs/003_独立便携整合包构建与发布说明.md`）。

运行环境与失败方式：
    需 Python >= 3.11 与 `PyQt6` 依赖；
    有控制台环境启动失败时抛出异常并返回非零退出码；`pythonw.exe` 无控制台环境启动失败时写出崩溃日志并弹窗提示后返回 `1`。
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path


def main() -> int:
    """将应用 `src/` 加入 `sys.path`，解析可写工作区并启动桌面应用事件循环。"""
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from yolo_studio.app import (
        main as run_app,
        report_startup_crash,
        resolve_workspace_root,
    )

    workspace_root = resolve_workspace_root(project_root)
    try:
        return run_app(workspace_root=workspace_root)
    except Exception as exc:
        raw_stderr = getattr(sys.stderr, "original", sys.stderr)
        if raw_stderr is None:
            report_startup_crash(exc, workspace_root=workspace_root, show_dialog=True)
            return 1
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
