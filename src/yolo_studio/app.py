"""YoloStudio 桌面图形界面应用入口模块。

职责与真实入口：
    提供 `main(workspace_root)`、`resolve_workspace_root(preferred_root)` 与无控制台启动异常兜底函数，
    既作为 `pyproject.toml` 的 GUI 入口点 `yolo-studio`，也供仓库根目录脚本 `main.py` 及独立便携包启动器调用
    （操作流程见 `README.md` 与 `docs/` 使用说明）。

运行环境与依赖：
    需 Python >= 3.11、已安装 `PyQt6` 及可用的本地桌面图形会话。

输入、输出与协作边界：
    `workspace_root` 指定默认数据/训练产物相对目录；未指定或通过 `resolve_workspace_root` 解析时，
    按 `YOLO_STUDIO_WORKSPACE` 环境变量 → 便携包同级 `workspace/` 目录 → 源码根目录/当前工作目录的顺序确定，
    并在目标位置不可写时自动回退至用户个人目录下的 `YoloStudio` 工作区。
    启动流程配置线程环境变量、CUDA DLL 搜索路径与启动欢迎图后，交由 `gui.main_window.MainWindow` 接管。

失败方式：
    缺少 `PyQt6` 等依赖或当前环境无法初始化图形界面时抛出导入或 Qt 异常并返回非零退出码。
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path

from yolo_studio._version import __version__

WORKSPACE_ENV_VAR = "YOLO_STUDIO_WORKSPACE"
_IN_RUNTIME_EXCEPTHOOK = False


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
        3. 当前模块所在源码根（若包含 `main.py` 或 `src/yolo_studio`，便携包下同样映射至 `workspace/`）或 `Path.cwd()`。

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
        if (code_root / "main.py").is_file() or (code_root / "src" / "yolo_studio").is_dir():
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


def install_runtime_excepthook(
    workspace_root: Path | None = None,
    show_dialog: bool | None = None,
) -> None:
    """安装进程级未捕获异常钩子，防止 PyQt6 槽函数异常触发 C++ 层 `qFatal`/`abort` 导致应用无声闪退。

    副作用与线程约束：
        替换 `sys.excepthook`；放行 `KeyboardInterrupt` 与 `SystemExit`；
        其余未捕获异常将带时间戳追加写入 `<workspace>/logs/runtime_error.log`，
        并在非无头桌面会话的主线程中弹出中文错误对话框提示错误摘要与日志路径。
        内部通过 `_IN_RUNTIME_EXCEPTHOOK` 防止异常弹窗重入递归。
    """
    prev_hook = sys.excepthook

    def _runtime_hook(exc_type, exc_value, exc_tb) -> None:
        global _IN_RUNTIME_EXCEPTHOOK
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            prev_hook(exc_type, exc_value, exc_tb)
            return

        formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log_file: Path | None = None
        try:
            base = resolve_workspace_root(workspace_root)
            log_dir = base / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "runtime_error.log"
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"\n===== {stamp} (YoloStudio v{__version__}) =====\n{formatted}\n")
        except OSError:
            log_file = None

        raw_stderr = getattr(sys.stderr, "original", sys.stderr)
        if raw_stderr is not None:
            try:
                raw_stderr.write(formatted)
                raw_stderr.flush()
            except Exception:
                pass

        if _IN_RUNTIME_EXCEPTHOOK:
            return
        _IN_RUNTIME_EXCEPTHOOK = True
        try:
            should_show = show_dialog
            if should_show is None:
                is_offscreen = os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen"
                should_show = (not is_offscreen) and (threading.current_thread() is threading.main_thread())
            if should_show:
                from PyQt6.QtWidgets import QApplication, QMessageBox

                qapp = QApplication.instance()
                if isinstance(qapp, QApplication):
                    hint = f"\n\n完整错误日志已保存至:\n{log_file}" if log_file else ""
                    QMessageBox.critical(
                        qapp.activeWindow(),
                        "YoloStudio 运行时异常",
                        f"操作执行时发生未预期错误，已拦截以防止程序闪退：\n\n{exc_type.__name__}: {exc_value}{hint}",
                    )
        except Exception:
            pass
        finally:
            _IN_RUNTIME_EXCEPTHOOK = False

    sys.excepthook = _runtime_hook


APP_USER_MODEL_ID = "YoloStudio.Desktop"
APP_ICON_FILENAME = "logo.ico"
APP_LOGO_FILENAME = "logo.png"
APP_SPLASH_SUBTITLE = "一站式 YOLO26 抽帧、智能标注与训练导出工作台"


def _configure_windows_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> None:
    """在 Windows 桌面会话中注册显式 `AppUserModelID`，确保任务栏独立显示 YoloStudio 图标而非 Python 默认图标。

    调用时机：
        须在创建 `QApplication` 与显示任何顶层窗口之前调用；非 Windows 平台或 `offscreen` 无头测试环境直接跳过。
    """
    if os.name != "nt" or os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id))
    except Exception:
        pass


def _find_package_resource(
    filename: str,
    code_root: Path,
    workspace_root: Path,
    *,
    package_dir: Path | None = None,
) -> Path | None:
    """按应用代码目录、工作区及当前包目录顺序查找 `yolo_studio/resources/<filename>`，未找到返回 `None`。"""
    candidates: list[Path] = []
    for base in (code_root, workspace_root):
        candidates.append(base / "src" / "yolo_studio" / "resources" / filename)
        candidates.append(base / "yolo_studio" / "resources" / filename)
    if package_dir is not None:
        candidates.append(package_dir / "resources" / filename)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_file():
            return candidate
    return None


def _find_app_icon_path(
    code_root: Path,
    workspace_root: Path,
    *,
    package_dir: Path | None = None,
) -> Path | None:
    """查找 YoloStudio 自有 `resources/logo.ico`，不存在时返回 `None`。"""
    return _find_package_resource(
        APP_ICON_FILENAME,
        code_root,
        workspace_root,
        package_dir=package_dir,
    )


def _find_app_logo_path(
    code_root: Path,
    workspace_root: Path,
    *,
    package_dir: Path | None = None,
) -> Path | None:
    """查找 YoloStudio 自有高分辨率透明底 `resources/logo.png`，用于高分屏启动欢迎图与窗口大图图标。"""
    return _find_package_resource(
        APP_LOGO_FILENAME,
        code_root,
        workspace_root,
        package_dir=package_dir,
    )


def _build_app_icon(icon_path: Path | None, logo_path: Path | None = None) -> object | None:
    """组合多尺寸 `.ico`（优化小尺寸锐度）与高清 `.png` 构建 `QIcon`；均不可用时返回 `None`。"""
    from PyQt6.QtGui import QIcon

    icon = QIcon()
    if icon_path is not None and icon_path.is_file():
        icon = QIcon(str(icon_path))
    if logo_path is not None and logo_path.is_file():
        icon.addFile(str(logo_path))
    return None if icon.isNull() else icon


def _render_startup_splash_pixmap(
    dpr: float,
    icon_path: Path | None,
    logo_path: Path | None = None,
) -> object:
    """按目标设备像素比 `dpr` 绘制高分屏圆角启动欢迎图 `QPixmap`（供桌面启动与离线自检共用）。"""
    from PyQt6.QtCore import QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap

    scale = max(1.0, float(dpr))
    width, height = 508, 210
    pixmap = QPixmap(int(round(width * scale)), int(round(height * scale)))
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    path = QPainterPath()
    path.addRoundedRect(QRectF(2, 2, width - 4, height - 4), 12.0, 12.0)
    painter.fillPath(path, QColor("#161b22"))
    painter.setPen(QColor("#30363d"))
    painter.drawPath(path)

    logo_size = 56
    logo_pix = QPixmap()
    if logo_path is not None and logo_path.is_file():
        raw_logo = QPixmap(str(logo_path))
        if not raw_logo.isNull():
            phys_px = int(round(logo_size * scale))
            logo_pix = raw_logo.scaled(
                phys_px,
                phys_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_pix.setDevicePixelRatio(scale)
    if logo_pix.isNull() and icon_path is not None and icon_path.is_file():
        logo_pix = QIcon(str(icon_path)).pixmap(logo_size, logo_size)

    text_x = 28
    if not logo_pix.isNull():
        painter.drawPixmap(QRectF(26, 30, logo_size, logo_size), logo_pix, QRectF(logo_pix.rect()))
        text_x = 96

    title_font = QFont("Microsoft YaHei UI", 16, QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.setPen(QColor("#f0f6fc"))
    painter.drawText(QRectF(text_x, 30, width - text_x - 24, 32), f"YoloStudio v{__version__}")

    sub_font = QFont("Microsoft YaHei UI", 9)
    painter.setFont(sub_font)
    painter.setPen(QColor("#8b949e"))
    painter.drawText(
        QRectF(text_x, 66, width - text_x - 24, 28),
        APP_SPLASH_SUBTITLE,
    )

    painter.setPen(QColor("#0969da"))
    painter.drawLine(28, 148, width - 28, 148)
    painter.end()
    return pixmap


def _create_startup_splash(
    app: object,
    icon_path: Path | None,
    logo_path: Path | None = None,
) -> object | None:
    """在非无头桌面环境下立即绘制并显示轻量启动欢迎图（`QSplashScreen`），消除冷启动视觉空白期。"""
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        return None

    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import QApplication, QSplashScreen

    qapp = app if isinstance(app, QApplication) else QApplication.instance()
    screen = qapp.primaryScreen() if qapp is not None else None
    dpr = float(screen.devicePixelRatio()) if screen is not None else 1.0

    pixmap = _render_startup_splash_pixmap(dpr, icon_path, logo_path=logo_path)

    splash = QSplashScreen(
        pixmap,
        Qt.WindowType.SplashScreen | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint,
    )
    splash.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    status_align = int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
    splash.showMessage("      正在加载工作区主界面...", status_align, QColor("#c9d1d9"))
    splash.show()
    if qapp is not None:
        qapp.processEvents()
    return splash


def main(workspace_root: Path | None = None) -> int:
    """初始化运行环境、瞬时启动欢迎图及桌面主窗口事件循环。

    性能优化设计：
        1. `QApplication` 就绪后立即弹出轻量 `QSplashScreen`（约 `0.15s` 上屏反馈）；
        2. 主窗口采用 `lazy_init=True` 构建：原生 PyQt6 标注工作台零重型外部依赖同步瞬时就绪，
           `torch` 计算设备检测推迟至主窗口显示后由后台 `DeviceProbeWorker` 异步完成，使首屏在亚秒级直接可交互。
    """
    root = (
        workspace_root.resolve()
        if workspace_root is not None
        else resolve_workspace_root()
    )
    code_root = _code_root_candidate()
    package_dir = Path(__file__).resolve().parent

    # 限制 GUI 主线程中 BLAS/OpenMP 盲目抢核并抑制无关 ICC 日志；
    # 训练/推理入口会通过 `core.trainer.ensure_torch_cpu_threads()` 恢复 torch 线程数，不受此影响
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.gui.icc=false")

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from yolo_studio.gui.tab_labeling import ensure_cuda_dll_search_paths

    ensure_cuda_dll_search_paths()
    _configure_windows_app_user_model_id()

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO26 Training Studio")
    app.setApplicationVersion(__version__)
    install_runtime_excepthook(workspace_root=root)
    icon_path = _find_app_icon_path(code_root, root, package_dir=package_dir)
    logo_path = _find_app_logo_path(code_root, root, package_dir=package_dir)
    app_icon = _build_app_icon(icon_path, logo_path=logo_path)
    if app_icon is not None:
        app.setWindowIcon(app_icon)

    splash = _create_startup_splash(app, icon_path, logo_path=logo_path)

    from yolo_studio.gui.main_window import MainWindow

    window = MainWindow(workspace_root=root, lazy_init=True)
    if app_icon is not None:
        window.setWindowIcon(app_icon)
    window.show()
    if splash is not None:
        splash.finish(window)

    QTimer.singleShot(50, window.start_background_warmup)
    return app.exec()
