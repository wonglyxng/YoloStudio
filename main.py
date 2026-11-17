"""YOLO26 训练工具桌面图形界面启动脚本（单进程内嵌 X-AnyLabeling）。

职责与真实入口：
    作为桌面 GUI 入口脚本，提供 `main()` 函数并在直接运行时启动应用
    （常见启动方式为 `run_gui.bat` 或 `.venv\\Scripts\\python.exe main.py`，
    操作流程见 `README.md` 与 `docs/001_录制视频抽帧工具使用说明.md`）。

运行环境与依赖：
    需 Python >= 3.11、已安装 `PyQt6`、`PyQt6-WebEngine`、本地可编辑安装的
    `third_party/X-AnyLabeling[gpu]` 及可用的本地桌面图形会话。

输入、输出与协作边界：
    无命令行业务参数；将仓库 `src/` 与 `third_party/X-AnyLabeling` 加入 `sys.path`、
    注入 CUDA DLL 搜索路径并设置 `AA_ShareOpenGLContexts` 后，初始化 X-AnyLabeling 资源，
    再委托 `yolo_studio.gui.main_window.MainWindow` 展示六步工作流界面。

失败方式：
    缺少 `PyQt6` 等依赖或当前环境无法初始化图形界面时抛出导入或 Qt 异常并返回非零退出码。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    """初始化运行环境、Qt 资源及桌面主窗口事件循环。

    副作用：将仓库 `src/` 与 `third_party/X-AnyLabeling` 插入 `sys.path`，配置 CUDA DLL 路径、
    `AA_ShareOpenGLContexts`、`QApplication`、X-AnyLabeling 翻译/主题资源及 `MainWindow`，
    并阻塞至用户关闭主窗口，随后返回 Qt 退出码。
    """
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    xal_dir = project_root / "third_party" / "X-AnyLabeling"
    for extra_path in (xal_dir, src_dir):
        if extra_path.is_dir() and str(extra_path) not in sys.path:
            sys.path.insert(0, str(extra_path))

    # 与原版 X-AnyLabeling 入口保持一致，降低内嵌标注工作台中科学计算库多线程冲突并抑制无关 ICC 日志；
    # 该设置会让 torch 默认只用 1 个 CPU 线程，训练/推理入口通过
    # `core.trainer.ensure_torch_cpu_threads()` 恢复 torch 线程数，不受此影响
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.gui.icc=false")

    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication

    from yolo_studio.gui.tab_labeling import (
        ensure_cuda_dll_search_paths,
        ensure_xanylabeling_runtime_initialized,
    )

    ensure_cuda_dll_search_paths()
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO26 Training Studio")
    ensure_xanylabeling_runtime_initialized(app)

    from yolo_studio.gui.main_window import MainWindow

    window = MainWindow(workspace_root=project_root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
