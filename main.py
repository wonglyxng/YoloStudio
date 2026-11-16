"""YOLO26 训练工具桌面图形界面启动脚本。

职责与真实入口：
    作为桌面 GUI 入口脚本，提供 `main()` 函数并在直接运行时启动应用
    （常见启动方式为 `run_gui.bat` 或 `.venv\\Scripts\\python.exe main.py`，
    操作流程见 `README.md` 与 `docs/001_录制视频抽帧工具使用说明.md`）。

运行环境与依赖：
    需 Python >= 3.11、已安装 `PySide6` 等依赖以及可用的本地桌面图形会话。

输入、输出与协作边界：
    无命令行业务参数；将仓库 `src/` 加入 `sys.path` 后委托
    `yolo_studio.gui.main_window.MainWindow` 展示五步工作流界面并读写工作区文件。
    自身仅负责路径注入与 Qt 事件循环生命周期，不承担业务逻辑与后台任务调度。

失败方式：
    缺少 `PySide6` 等依赖或当前环境无法初始化图形界面时抛出导入或 Qt 异常并返回非零退出码。
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """初始化路径并启动桌面主窗口事件循环。

    副作用：将仓库 `src/` 插入 `sys.path`（若尚未存在），创建 `QApplication`
    与 `MainWindow` 并阻塞至用户关闭主窗口，随后返回 Qt 退出码。
    """
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from PySide6.QtWidgets import QApplication

    from yolo_studio.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO26 Training Studio")
    window = MainWindow(workspace_root=project_root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
