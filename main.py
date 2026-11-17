"""YOLO26 训练工具桌面图形界面启动脚本（单进程内嵌 X-AnyLabeling）。

职责与真实入口：
    仓库根目录桌面 GUI 启动脚本，供 `run_gui.bat` 或 `.venv\\Scripts\\python.exe main.py` 直接运行；
    将仓库 `src/` 加入 `sys.path` 后，以仓库根目录为默认工作区委托 `yolo_studio.app.main` 启动主窗口
    （操作流程见 `README.md` 与 `docs/001_录制视频抽帧工具使用说明.md`）。

运行环境与失败方式：
    需 Python >= 3.11、`PyQt6`、`PyQt6-WebEngine` 及本地 `third_party/X-AnyLabeling` 依赖；
    缺少依赖或无法初始化桌面图形会话时抛出异常并返回非零退出码。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """将仓库 `src/` 加入 `sys.path` 并以仓库根目录启动桌面应用事件循环。"""
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if src_dir.is_dir() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from yolo_studio.app import main as run_app

    return run_app(workspace_root=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
