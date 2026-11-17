"""YoloStudio 桌面图形界面应用入口模块（单进程内嵌 X-AnyLabeling）。

职责与真实入口：
    提供 `main(workspace_root)`，既作为 `pyproject.toml` 的 GUI 入口点 `yolo-studio`，
    也供仓库根目录脚本 `main.py` 委托调用（操作流程见 `README.md` 与 `docs/` 使用说明）。

运行环境与依赖：
    需 Python >= 3.11、已安装 `PyQt6`、`PyQt6-WebEngine`、`x-anylabeling-cvhub`（或仓库内
    `third_party/X-AnyLabeling` 源码树）及可用的本地桌面图形会话。

输入、输出与协作边界：
    `workspace_root` 指定默认数据/训练产物相对目录，省略时优先使用源码仓库根目录（若位于源码树中），
    否则使用当前工作目录 `Path.cwd()`。启动流程配置线程环境变量、CUDA DLL 搜索路径、
    `AA_ShareOpenGLContexts` 与 X-AnyLabeling 运行时资源后，交由 `gui.main_window.MainWindow` 接管。

失败方式：
    缺少 `PyQt6` 等依赖或当前环境无法初始化图形界面时抛出导入或 Qt 异常并返回非零退出码。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_default_workspace() -> Path:
    """若当前模块位于包含 `third_party/X-AnyLabeling` 的源码仓库内则返回仓库根，否则返回当前工作目录。"""
    repo_candidate = Path(__file__).resolve().parents[2]
    if (repo_candidate / "third_party" / "X-AnyLabeling").is_dir():
        return repo_candidate
    return Path.cwd()


def main(workspace_root: Path | None = None) -> int:
    """初始化运行环境、Qt 资源及桌面主窗口事件循环。

    副作用：若源码树下存在 `third_party/X-AnyLabeling` 则将其插入 `sys.path`，设置科学计算线程环境变量、
    CUDA DLL 搜索路径、`AA_ShareOpenGLContexts`、`QApplication`、X-AnyLabeling 翻译/主题资源及 `MainWindow`，
    阻塞至用户关闭主窗口后返回 Qt 退出码。
    """
    root = (workspace_root or _resolve_default_workspace()).resolve()
    xal_dir = root / "third_party" / "X-AnyLabeling"
    if xal_dir.is_dir() and str(xal_dir) not in sys.path:
        sys.path.insert(0, str(xal_dir))

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

    window = MainWindow(workspace_root=root)
    window.show()
    return app.exec()
