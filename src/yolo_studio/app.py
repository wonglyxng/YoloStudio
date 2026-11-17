"""YoloStudio 桌面图形界面应用入口模块（单进程内嵌 X-AnyLabeling）。

职责与真实入口：
    提供 `main(workspace_root)`、`resolve_workspace_root(preferred_root)` 与无控制台启动异常兜底函数，
    既作为 `pyproject.toml` 的 GUI 入口点 `yolo-studio`，也供仓库根目录脚本 `main.py` 及独立便携包启动器调用
    （操作流程见 `README.md` 与 `docs/` 使用说明）。

运行环境与依赖：
    需 Python >= 3.11、已安装 `PyQt6`、`PyQt6-WebEngine`、`x-anylabeling-cvhub`（或本地
    `third_party/X-AnyLabeling` 源码树）及可用的本地桌面图形会话。

输入、输出与协作边界：
    `workspace_root` 指定默认数据/训练产物相对目录；未指定或通过 `resolve_workspace_root` 解析时，
    按 `YOLO_STUDIO_WORKSPACE` 环境变量 → 便携包同级 `workspace/` 目录 → 源码根目录/当前工作目录的顺序确定，
    并在目标位置不可写时自动回退至用户个人目录下的 `YoloStudio` 工作区。
    启动流程配置线程环境变量、CUDA DLL 搜索路径、`AA_ShareOpenGLContexts` 与 X-AnyLabeling 运行时资源后，
    交由 `gui.main_window.MainWindow` 接管。

失败方式：
    缺少 `PyQt6` 等依赖或当前环境无法初始化图形界面时抛出导入或 Qt 异常并返回非零退出码。
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import traceback
from pathlib import Path

WORKSPACE_ENV_VAR = "YOLO_STUDIO_WORKSPACE"


def _code_root_candidate() -> Path:
    """返回包含 `src/yolo_studio/app.py` 的应用代码根目录（源码仓库根或便携包的 `app/` 目录）。"""
    return Path(__file__).resolve().parents[2]


def _portable_bundle_workspace(base_dir: Path) -> Path | None:
    """若 `base_dir` 为独立便携整合包内的 `app/` 子目录，则返回同级 `<便携包根>/workspace` 路径。

    识别条件：
        `base_dir` 目录名为 `app`（忽略大小写），且其父目录下存在 `runtime/python.exe`（Windows）
        或 `runtime/bin/python3`（类 Unix）。这样可将用户抽帧、数据集与训练产物同 `app/` 代码目录隔离，
        支持后续直接覆盖 `app/` 完成免重装代码更新。
    """
    resolved = base_dir.resolve()
    if resolved.name.lower() != "app":
        return None
    bundle_root = resolved.parent
    if (bundle_root / "runtime" / "python.exe").is_file() or (
        bundle_root / "runtime" / "bin" / "python3"
    ).is_file():
        return bundle_root / "workspace"
    return None


def _ensure_writable_directory(directory: Path) -> bool:
    """创建目录并通过真实临时文件写入验证写权限，成功返回 `True`，无权限或路径非法返回 `False`。

    设计原因：
        Windows 上 `os.access(..., os.W_OK)` 仅检查只读文件属性标志，不校验 NTFS ACL（例如非管理员访问
        `C:\\Program Files` 会误判为可写），因此必须实际创建并立即删除探针文件来确认可写性。
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=directory, prefix=".yolo_studio_probe_", delete=True
        ):
            pass
        return True
    except OSError:
        return False


def _fallback_user_workspace() -> Path:
    """返回位于当前登录用户个人目录下的可写回退工作区路径。"""
    home = Path.home()
    docs_dir = home / "Documents"
    candidates = [
        (docs_dir / "YoloStudio") if docs_dir.is_dir() else (home / "YoloStudio"),
        home / "YoloStudio",
        Path(tempfile.gettempdir()) / "YoloStudio",
    ]
    for candidate in candidates:
        if _ensure_writable_directory(candidate):
            return candidate.resolve()
    return candidates[0].resolve()


def resolve_workspace_root(preferred_root: Path | None = None) -> Path:
    """解析并确保返回一个可写的默认工作区根目录。

    优先级顺序：
        1. 环境变量 `YOLO_STUDIO_WORKSPACE`（非空白字符串时优先采用）；
        2. 传入的 `preferred_root`（若其位于便携整合包的 `app/` 目录，则自动映射为同级 `workspace/`）；
        3. 当前模块所在源码根（若包含 `third_party/X-AnyLabeling`，便携包下同样映射至 `workspace/`）或 `Path.cwd()`。

    副作用与回退行为：
        若选定目录不存在会自动创建；若因系统权限限制（如解压在只读目录）无法写入，
        自动回退至 `~/Documents/YoloStudio`（或 `~/YoloStudio`），确保抽帧、数据集构建与训练不会因只读目录报错。
    """
    env_val = os.environ.get(WORKSPACE_ENV_VAR, "").strip()
    if env_val:
        candidate = Path(env_val).expanduser().resolve()
    elif preferred_root is not None:
        portable_ws = _portable_bundle_workspace(preferred_root)
        candidate = (portable_ws or preferred_root).resolve()
    else:
        code_root = _code_root_candidate()
        if (code_root / "third_party" / "X-AnyLabeling").is_dir():
            portable_ws = _portable_bundle_workspace(code_root)
            candidate = (portable_ws or code_root).resolve()
        else:
            candidate = Path.cwd().resolve()

    if _ensure_writable_directory(candidate):
        return candidate
    return _fallback_user_workspace()


def report_startup_crash(
    exc: BaseException,
    workspace_root: Path | None = None,
    show_dialog: bool | None = None,
) -> Path | None:
    """将启动期未捕获异常写入崩溃日志，并在 Windows 无控制台模式下弹出错误提示框。

    参数与行为：
        `workspace_root` 为可选工作区目录；未传或不可写时回退至用户可写工作区下的 `logs/startup_crash.log`。
        `show_dialog` 为 `None` 时仅在 Windows 且标准错误流缺失（如 `pythonw.exe` 启动）且非 `QT_QPA_PLATFORM=offscreen`
        自动化测试环境下调用原生 `MessageBoxW` 提示用户，避免双击启动失败时静默无响应。
    """
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_file: Path | None = None
    try:
        base = resolve_workspace_root(workspace_root)
        log_dir = base / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "startup_crash.log"
        log_file.write_text(formatted, encoding="utf-8")
    except OSError:
        log_file = None

    if show_dialog is None:
        raw_stderr = getattr(sys.stderr, "original", sys.stderr)
        is_offscreen = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
        show_dialog = os.name == "nt" and raw_stderr is None and not is_offscreen

    if show_dialog and os.name == "nt":
        try:
            hint = f"\n\n详细错误日志已保存至:\n{log_file}" if log_file else ""
            msg = f"YoloStudio 启动失败: {exc}{hint}"
            ctypes.windll.user32.MessageBoxW(0, msg, "YoloStudio 启动错误", 0x10)
        except Exception:
            pass
    return log_file


def main(workspace_root: Path | None = None) -> int:
    """初始化运行环境、Qt 资源及桌面主窗口事件循环。

    副作用：若应用代码根或工作区下存在 `third_party/X-AnyLabeling` 则将其插入 `sys.path`，
    设置科学计算线程环境变量、CUDA DLL 搜索路径、`AA_ShareOpenGLContexts`、`QApplication`、
    X-AnyLabeling 翻译/主题资源及 `MainWindow`，阻塞至用户关闭主窗口后返回 Qt 退出码。
    """
    root = (
        workspace_root.resolve()
        if workspace_root is not None
        else resolve_workspace_root()
    )
    code_root = _code_root_candidate()
    for candidate_base in (code_root, root):
        xal_dir = candidate_base / "third_party" / "X-AnyLabeling"
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
