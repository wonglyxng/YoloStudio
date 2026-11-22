"""标签页 2：单进程内嵌原版 X-AnyLabeling 标注工作台界面。

职责：
    在 YoloStudio 主进程与主窗口内直接托管原版 `anylabeling.views.mainwindow.MainWindow`
    （含完整画布、标注工具栏、标签/形状/文件面板与 AI 自动标注引擎），并提供顶部工作流联动栏，
    与「1. 视频抽帧」「3. LLM 智能标注」「4. 数据集准备」双向衔接。

运行环境与协作边界：
    运行于 PyQt6 主线程，依赖 `third_party/X-AnyLabeling`（`x-anylabeling-cvhub`）源码包及
    `PyQt6` / `PyQt6-WebEngine` / `onnxruntime-gpu`。
    初始化需先调用 `ensure_xanylabeling_runtime_initialized` 注册 Qt 资源、翻译与 CUDA DLL 搜索路径。
    内嵌子窗口保留独立 `menuBar()` 与 `statusBar()`，但关闭生命周期由外层 `MainWindow.closeEvent`
    通过 `prepare_for_close()` 统一调度。
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QFileSystemWatcher, QLocale, QSize, Qt, QTimer, QTranslator, pyqtSignal as Signal
from PyQt6.QtGui import QAction, QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_RUNTIME_INITIALIZED = False
_INSTALLED_TRANSLATOR: QTranslator | None = None
_LOADED_CONFIG: dict | None = None
_ADDED_DLL_HANDLES: list[object] = []


def ensure_cuda_dll_search_paths() -> None:
    """在 Windows 上将 PyTorch 自带的 CUDA/cuDNN 目录加入 DLL 搜索路径。

    说明：
        当进程先启动 `onnxruntime-gpu` 且未提前 `import torch` 时，Windows 可能找不到
        `cudnn64_9.dll` 导致退回 `CPUExecutionProvider`。本函数幂等补充 `torch\\lib` 搜索目录。
    """
    if os.name != "nt":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            torch_lib = Path(spec.origin).resolve().parent / "lib"
            if torch_lib.is_dir():
                lib_str = str(torch_lib)
                current_path = os.environ.get("PATH", "")
                if lib_str not in current_path.split(os.pathsep):
                    os.environ["PATH"] = lib_str + os.pathsep + current_path
                if hasattr(os, "add_dll_directory"):
                    _ADDED_DLL_HANDLES.append(os.add_dll_directory(lib_str))
    except Exception:
        # 找不到或仅 CPU 环境时忽略，不影响 CPU 回退运行
        pass


def ensure_xanylabeling_runtime_initialized(app: QApplication) -> dict:
    """幂等初始化 X-AnyLabeling 所需的配置、Qt 资源、语言翻译与主题调色板。

    调用条件：
        必须在已创建 `QApplication` 之后、构造内嵌工作台窗口之前调用；
        重复调用时直接复用已加载的缓存配置，不重复安装 `QTranslator`。
    """
    global _RUNTIME_INITIALIZED, _INSTALLED_TRANSLATOR, _LOADED_CONFIG
    if _RUNTIME_INITIALIZED and _LOADED_CONFIG is not None:
        return _LOADED_CONFIG

    ensure_cuda_dll_search_paths()

    from anylabeling import config as anylabeling_config
    from anylabeling.config import get_config, get_work_directory, set_work_directory
    from anylabeling.resources import resources  # noqa: F401  # 注册图标与翻译 Qt 资源
    from anylabeling.views.labeling.utils.theme import init_theme

    work_dir = os.path.expanduser("~")
    set_work_directory(work_dir)
    rc_path = os.path.join(get_work_directory(), ".xanylabelingrc")
    anylabeling_config.current_config_file = rc_path
    config = get_config(rc_path, show_msg=False)

    # 仅初始化内部主题字典供图标/控件查询，不覆盖 YoloStudio 顶层应用样式表
    theme_mode = config.get("theme", "light")
    if theme_mode not in ("light", "dark"):
        theme_mode = "light"
    init_theme("light" if theme_mode == "auto" else theme_mode)

    # 优先加载用户配置的语言，否则按系统区域（默认尝试 zh_CN）加载内嵌翻译
    language = config.get("language") or QLocale.system().name()
    translator = QTranslator(app)
    if translator.load(f":/languages/translations/{language}.qm") or translator.load(
        ":/languages/translations/zh_CN.qm"
    ):
        app.installTranslator(translator)
        _INSTALLED_TRANSLATOR = translator

    _LOADED_CONFIG = config
    _RUNTIME_INITIALIZED = True
    return config


def _create_embedded_window_class():
    """延迟定义继承自原版 `MainWindow` 的内嵌宿主机类，避免模块导入阶段触发 Qt 资源依赖。"""
    from anylabeling.views.labeling.utils.theme import get_app_stylesheet
    from anylabeling.views.mainwindow import MainWindow as XalMainWindow

    class _EmbeddedXAnyLabelingWindow(XalMainWindow):
        """内嵌于标签页中的 X-AnyLabeling 主窗口包装类。

        状态与副作用：
            持有 `_allow_close` 关闭放行开关与 `title_changed` 信号；
            通过 `Qt.WindowType.Widget` 嵌入外层布局，并随标签页可见状态动态切换内部 `QAction`
            的快捷键上下文：当前页可见时启用原生 `WindowShortcut`（确保挂载在菜单/工具栏上的
            `R`/`A`/`D` 等全部快捷键正常响应），切离隐藏时收起为 `WidgetShortcut` 防止跨页误触。
        """

        title_changed = Signal(str)

        def __init__(self, app: QApplication, config: dict) -> None:
            self._allow_close = False
            super().__init__(app, config=config)
            self.setWindowFlags(Qt.WindowType.Widget)
            self.setStyleSheet(get_app_stylesheet())
            # 初始挂载时尚未显示，先收起快捷键，待所在标签页真正进入可见态再激活
            self.set_shortcuts_active(False)

        def set_shortcuts_active(self, active: bool) -> None:
            """按工作台标签页是否可见切换内部 `QAction` 的快捷键作用域。

            原因说明：
                X-AnyLabeling 将绝大多数快捷键 `QAction` 仅注册在 `QMenu` 与 `ToolBar` 上，
                并未同时 `addAction` 到画布或外层容器；若强设为 `WidgetWithChildrenShortcut`，
                当焦点位于兄弟节点 `Canvas` 时 Qt `QShortcutMap` 不会匹配，导致快捷键全部失效。
                因此在工作台可见时恢复原生 `WindowShortcut`，隐藏时切为 `WidgetShortcut` 实现跨页隔离。
            """
            ctx = (
                Qt.ShortcutContext.WindowShortcut
                if active
                else Qt.ShortcutContext.WidgetShortcut
            )
            for action in self.findChildren(QAction):
                # 排除独立的非模态顶层子窗口（如点云/分类弹窗），但保留 QMenu（Popup 亦带 Window 标志位）
                obj = action.parent()
                while obj is not None and not isinstance(obj, QWidget):
                    obj = obj.parent()
                if (
                    isinstance(obj, QWidget)
                    and obj is not self
                    and obj.isWindow()
                    and not isinstance(obj, QMenu)
                ):
                    continue
                action.setShortcutContext(ctx)

        def _restore_window_geometry(self) -> None:
            """内嵌模式下由外层标签页布局接管尺寸，忽略独立窗口位置/大小恢复。"""
            return

        def setWindowTitle(self, title: str) -> None:
            """截获标题变更并向外层工作台栏发送通知，避免影响宿主的独立窗口状态。"""
            super().setWindowTitle(title)
            self.title_changed.emit(title)

        def closeEvent(self, event: QCloseEvent) -> None:
            """拦截非宿主触发的关闭（如语言/主题重启提示），仅在宿主退出阶段放行保存检查。"""
            if not self._allow_close:
                event.ignore()
                QMessageBox.information(
                    self,
                    "提示",
                    "内嵌标注工作台由 YoloStudio 主窗口统一托管；若刚修改语言或主题设置，请关闭并重新打开 YoloStudio 生效。",
                )
                return
            super().closeEvent(event)

        def prepare_for_close(self) -> bool:
            """由宿主退出流程调用：触发未保存标注检查并持久化 X-AnyLabeling 状态。"""
            self._allow_close = True
            try:
                fake_event = QCloseEvent()
                super().closeEvent(fake_event)
                return fake_event.isAccepted()
            finally:
                self._allow_close = False

    return _EmbeddedXAnyLabelingWindow


def _resolve_xal_view(xal_window: object | None) -> object | None:
    """安全读取 X-AnyLabeling 主窗口内的核心标注画布视图（`labeling_widget.view`）。"""
    if xal_window is None:
        return None
    wrapper = getattr(xal_window, "labeling_widget", None)
    return getattr(wrapper, "view", None)


def _apply_labels_to_xal_view(lw: object, labels: list[str]) -> bool:
    """将类别列表注入 X-AnyLabeling 视图的配置、右侧唯一标签列表与弹窗历史补全词库。

    版本耦合与防御说明：
        对应 `third_party/X-AnyLabeling/anylabeling/views/labeling/label_widget.py` 的内部结构；
        对 `_config`、`load_labels` 与 `label_dialog.add_label_history` 均先做属性/类型校验，
        避免未来升级第三方源码导致未捕获的 `AttributeError` 或 `TypeError`。
        至少成功调用 `load_labels` 时返回 `True`。
    """
    load_fn = getattr(lw, "load_labels", None)
    if not callable(load_fn):
        return False

    cfg = getattr(lw, "_config", None)
    if isinstance(cfg, dict):
        cfg["labels"] = list(labels)

    load_fn(labels, clear_existing=True)

    dialog = getattr(lw, "label_dialog", None)
    add_history = getattr(dialog, "add_label_history", None)
    if callable(add_history):
        for label in labels:
            add_history(label, update_last_label=False)
    return True


class _ElidedStatusLabel(QLabel):
    """固定最小宽度并自动居中省略过长路径的单行状态标签。

    设计原因：
        X-AnyLabeling 切换图片时会调用 `setWindowTitle` 传入包含完整绝对路径、分辨率与序号的长标题；
        普通 `QLabel` 的 `minimumSizeHint` 等于全文像素宽度，会把顶层主窗口动态撑宽。
        本控件将水平策略设为 `Ignored` 并固定紧凑 `sizeHint`，超宽时自动省略中间路径且通过悬浮提示展示全文。
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(80)
        if text:
            self.setText(text)

    def full_text(self) -> str:
        """返回未经省略截断的完整状态文本。"""
        return self._full_text

    def setText(self, text: str) -> None:  # type: ignore[override]
        """记录完整文本并刷新悬浮提示与省略显示（保留 type: ignore 指令）。"""
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._refresh_elided()

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(80, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(240, super().sizeHint().height())

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_elided()

    def _refresh_elided(self) -> None:
        avail = max(40, self.contentsRect().width())
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, avail
        )
        super().setText(elided)


class LabelingStudioTab(QWidget):
    """单进程内嵌 X-AnyLabeling 标注工作台标签页（标签页 2）。

    内部状态与资源：
        持有内嵌主窗口 `_xal_window`、当前跟踪的目录 `_current_dir`、最近预加载的类别文件
        `_last_classes_file` 以及带 `400 ms` 防抖的 `QFileSystemWatcher` 目录监听器。
    """

    send_to_dataset = Signal(str, str)  # 参数：(标注图片目录, classes.txt 路径)
    send_to_review = Signal(str)  # 参数：(图片目录)
    annotations_changed = Signal(str)  # 参数：(发生 .json 写入/修改的目录)

    def __init__(
        self,
        workspace_root: Path,
        parent: QWidget | None = None,
        lazy_mount: bool = False,
    ) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self._lazy_mount = lazy_mount
        self._mounting = False
        self._mount_failed = False
        self._xal_window = None
        self._current_dir: Path | None = None
        self._last_classes_file: Path | None = None

        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_watched_dir_changed)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(400)
        self._debounce_timer.timeout.connect(self._emit_annotations_changed)

        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 顶部紧凑联动栏：尽量节省高度，将屏幕空间留给 X-AnyLabeling 画布
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        bar.setStyleSheet(
            "QFrame { background-color: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; }"
        )
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 5, 10, 5)
        bar_layout.setSpacing(8)

        self.status_label = _ElidedStatusLabel(
            "工作台就绪：可打开抽帧目录进行人工画框或 AI 辅助标注（快捷键 R 画框，Ctrl+S 保存）"
        )
        self.status_label.setStyleSheet("border: none; color: #24292f; font-weight: 500;")
        bar_layout.addWidget(self.status_label, 1)

        self.open_dir_btn = QPushButton("打开图片目录...")
        self.load_classes_btn = QPushButton("导入 classes.txt...")
        self.refresh_btn = QPushButton("刷新目录")
        self.to_dataset_btn = QPushButton("发送到「4. 数据集准备」→")
        self.to_dataset_btn.setObjectName("primaryBtn")
        self.to_review_btn = QPushButton("发送到「3. LLM 智能标注」→")

        for btn in (
            self.open_dir_btn,
            self.load_classes_btn,
            self.refresh_btn,
            self.to_review_btn,
            self.to_dataset_btn,
        ):
            # 防止点击顶部联动按钮后滞留键盘焦点，避免按 Space 平移画布时误触按钮点击
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            bar_layout.addWidget(btn)
        root.addWidget(bar)

        self._host_container = QVBoxLayout()
        self._host_container.setContentsMargins(0, 0, 0, 0)
        root.addLayout(self._host_container, 1)

        if self._lazy_mount:
            self._show_lazy_placeholder()
        else:
            self._mount_embedded_xanylabeling()

        self.open_dir_btn.clicked.connect(self._on_pick_and_open_dir)
        self.load_classes_btn.clicked.connect(self._on_pick_and_load_classes)
        self.refresh_btn.clicked.connect(self._on_refresh_dir)
        self.to_dataset_btn.clicked.connect(self._on_send_to_dataset)
        self.to_review_btn.clicked.connect(self._on_send_to_review)

    def _clear_host_container(self) -> None:
        """清空工作台宿主容器内的占位面板或旧组件。"""
        while self._host_container.count():
            item = self._host_container.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _show_lazy_placeholder(self) -> None:
        """在延迟挂载模式下展示轻量占位视图，避免首屏启动时同步加载整个 X-AnyLabeling。"""
        self._clear_host_container()
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        hint_lbl = QLabel("正在准备内嵌 X-AnyLabeling 标注工作台...")
        hint_lbl.setStyleSheet("color: #57606a; font-size: 14px; font-weight: 500;")
        load_btn = QPushButton("立即加载标注工作台")
        load_btn.clicked.connect(lambda: self.ensure_mounted())
        layout.addWidget(hint_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(load_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._host_container.addWidget(box, 1)

    def ensure_mounted(self) -> bool:
        """幂等确保内嵌 X-AnyLabeling 主窗口已完成初始化与挂载；成功返回 `True`。

        重入保护：
            挂载期间设置 `_mounting=True` 并切换等待光标，防止事件刷新引发 `showEvent` 递归调用。
        """
        if self._xal_window is not None:
            return True
        if self._mounting:
            return False

        self._mounting = True
        self.status_label.setText("正在加载内嵌 X-AnyLabeling 标注工作台，请稍候...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            QApplication.processEvents()
            self._mount_embedded_xanylabeling()
            if self._xal_window is not None:
                self._mount_failed = False
                is_active = self.isVisible()
                self._xal_window.set_shortcuts_active(is_active)
                if is_active:
                    QTimer.singleShot(0, self._focus_canvas_if_ready)
                self.status_label.setText(
                    "工作台就绪：可打开抽帧目录进行人工画框或 AI 辅助标注（快捷键 R 画框，Ctrl+S 保存）"
                )
                return True
            self._mount_failed = True
            return False
        finally:
            QApplication.restoreOverrideCursor()
            self._mounting = False

    def _mount_embedded_xanylabeling(self) -> None:
        """初始化并挂载原版 X-AnyLabeling 主窗口；若加载失败则在页内呈现错误说明与重试按钮。"""
        self._clear_host_container()

        app = QApplication.instance()
        if not isinstance(app, QApplication):
            self._mount_failed = True
            self._show_mount_error("未检测到有效 QApplication 实例，无法挂载内嵌标注工作台。")
            return

        try:
            config = ensure_xanylabeling_runtime_initialized(app)
            embedded_cls = _create_embedded_window_class()
            self._xal_window = embedded_cls(app, config)
            self._xal_window.title_changed.connect(self._on_xal_title_changed)
            self._host_container.addWidget(self._xal_window, 1)
            self._xal_window.set_shortcuts_active(self.isVisible())
            self._mount_failed = False
        except Exception as exc:
            self._xal_window = None
            self._mount_failed = True
            self._show_mount_error(f"初始化内嵌 X-AnyLabeling 失败：{exc}")

    def _show_mount_error(self, message: str) -> None:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        err_lbl = QLabel(message)
        err_lbl.setWordWrap(True)
        err_lbl.setStyleSheet("color: #cf222e; font-weight: 600;")
        retry_btn = QPushButton("重试加载标注工作台")
        retry_btn.clicked.connect(self._mount_embedded_xanylabeling)
        layout.addWidget(err_lbl)
        layout.addWidget(retry_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._host_container.addWidget(box, 1)

    def _label_widget(self):
        return _resolve_xal_view(self._xal_window)

    def _focus_canvas_if_ready(self) -> None:
        """在工作台可见且未聚焦于内部文本输入框时，将键盘焦点移交给画布以响应直接按键。"""
        if not self.isVisible() or self._xal_window is None:
            return
        focused = QApplication.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit)) and self._xal_window.isAncestorOf(
            focused
        ):
            return
        lw = self._label_widget()
        if lw is None:
            return
        canvas = getattr(lw, "canvas", None)
        if isinstance(canvas, QWidget) and canvas.isEnabled():
            canvas.setFocus(Qt.FocusReason.OtherFocusReason)
        elif isinstance(lw, QWidget):
            lw.setFocus(Qt.FocusReason.OtherFocusReason)

    def showEvent(self, event) -> None:  # type: ignore[override]
        """切换进入工作台标签页时按需触发首次挂载，激活快捷键并将焦点移交画布（保留 type: ignore 指令）。"""
        super().showEvent(event)
        if self._xal_window is not None:
            self._xal_window.set_shortcuts_active(True)
            QTimer.singleShot(0, self._focus_canvas_if_ready)
        elif not self._mount_failed and not self._mounting:
            QTimer.singleShot(0, self._auto_mount_on_show)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        """切离工作台标签页时收起内部快捷键，避免在其他标签页输入或操作时误触（保留 type: ignore 指令）。"""
        super().hideEvent(event)
        if self._xal_window is not None:
            self._xal_window.set_shortcuts_active(False)

    def _auto_mount_on_show(self) -> None:
        """在标签页切换绘制完成后异步完成工作台首次挂载。"""
        if self.isVisible() and self._xal_window is None and not self._mount_failed:
            self.ensure_mounted()

    def _on_xal_title_changed(self, title: str) -> None:
        if title:
            self.status_label.setText(title)

    def _watch_directory(self, dir_path: Path) -> None:
        existing = self._watcher.directories()
        if existing:
            self._watcher.removePaths(existing)
        if dir_path.is_dir():
            self._watcher.addPath(str(dir_path))

    def _on_watched_dir_changed(self, _path: str) -> None:
        self._debounce_timer.start()

    def _emit_annotations_changed(self) -> None:
        cur = self.current_directory()
        if cur is not None:
            self.annotations_changed.emit(str(cur))

    def current_directory(self) -> Path | None:
        """返回工作台当前打开的图片目录；若尚未打开任何目录则返回 `None`。"""
        lw = self._label_widget()
        if lw is not None:
            if getattr(lw, "filename", None):
                p = Path(str(lw.filename))
                if p.exists():
                    return p if p.is_dir() else p.parent
            if getattr(lw, "last_open_dir", None):
                p = Path(str(lw.last_open_dir))
                if p.is_dir():
                    return p
        return self._current_dir

    def load_classes_file(self, classes_file: Path) -> int:
        """从 `classes.txt` 读取类别列表并同步注入右侧标签列表与弹窗历史补全词库。

        返回值：
            成功导入的非空类别数量；工作台未就绪或文件不存在时返回 `0`。
        """
        if self._xal_window is None:
            self.ensure_mounted()
        lw = self._label_widget()
        if lw is None or not classes_file.is_file():
            return 0
        labels = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
        if not labels:
            return 0

        if not _apply_labels_to_xal_view(lw, labels):
            return 0
        self._last_classes_file = classes_file.resolve()
        return len(labels)

    def open_directory(
        self,
        dir_path: Path,
        classes_file: Path | None = None,
        select_image: Path | None = None,
    ) -> bool:
        """在内嵌 X-AnyLabeling 中打开指定图片目录，并可选预载类别与定位到单张图片。

        调用条件与失败/取消行为：
            尚未挂载时自动调用 `ensure_mounted()`；切换前会调用 `may_continue()`，
            若当前图片有未保存修改且用户取消保存弹窗，直接返回 `False` 并不覆盖当前画布状态。
        """
        if self._xal_window is None:
            self.ensure_mounted()
        lw = self._label_widget()
        if lw is None:
            QMessageBox.warning(self, "工作台未就绪", "内嵌 X-AnyLabeling 尚未完成初始化。")
            return False

        target_dir = dir_path.resolve()
        if not target_dir.is_dir():
            QMessageBox.warning(self, "目录不存在", f"图片目录不存在：\n{target_dir}")
            return False

        if not lw.may_continue():
            return False

        if classes_file is not None and classes_file.is_file():
            self.load_classes_file(classes_file)
        elif (target_dir / "classes.txt").is_file():
            self.load_classes_file(target_dir / "classes.txt")

        # 若传了目标单图且属于目标目录，先载入文件列表再精确定位该图片，避免重复加载首图
        lw.import_image_folder(str(target_dir), load=(select_image is None))
        if select_image is not None and select_image.is_file():
            lw.load_file(str(select_image.resolve()))

        self._current_dir = target_dir
        self._watch_directory(target_dir)
        is_active = self.isVisible()
        self._xal_window.set_shortcuts_active(is_active)
        if is_active:
            QTimer.singleShot(0, self._focus_canvas_if_ready)
        self.status_label.setText(f"当前标注目录：{target_dir}")
        return True

    def _on_pick_and_open_dir(self) -> None:
        start = str(self.current_directory() or (self.workspace_root / "data" / "raw_frames"))
        chosen = QFileDialog.getExistingDirectory(self, "选择待标注图片目录", start)
        if chosen:
            self.open_directory(Path(chosen))

    def _on_pick_and_load_classes(self) -> None:
        start = str(self.current_directory() or self.workspace_root)
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择 classes.txt 类别清单", start, "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if chosen:
            count = self.load_classes_file(Path(chosen))
            if count > 0:
                QMessageBox.information(self, "已导入类别", f"已向工作台加载 {count} 个类别。")
            else:
                QMessageBox.warning(self, "类别为空", "所选文件中未读取到有效类别名称。")

    def _on_refresh_dir(self) -> None:
        cur = self.current_directory()
        if cur is None or not cur.is_dir():
            QMessageBox.information(self, "提示", "当前尚未打开任何图片目录。")
            return
        lw = self._label_widget()
        cur_file = Path(str(lw.filename)) if lw and getattr(lw, "filename", None) else None
        self.open_directory(
            cur,
            classes_file=self._last_classes_file,
            select_image=cur_file if cur_file and cur_file.is_file() else None,
        )

    def _on_send_to_dataset(self) -> None:
        cur = self.current_directory()
        if cur is None:
            QMessageBox.warning(self, "提示", "请先在工作台中打开待标注的图片目录。")
            return
        cls_path = ""
        if (cur / "classes.txt").is_file():
            cls_path = str((cur / "classes.txt").resolve())
        elif self._last_classes_file and self._last_classes_file.is_file():
            cls_path = str(self._last_classes_file)
        self.send_to_dataset.emit(str(cur), cls_path)

    def _on_send_to_review(self) -> None:
        cur = self.current_directory()
        if cur is None:
            QMessageBox.warning(self, "提示", "请先在工作台中打开图片目录。")
            return
        self.send_to_review.emit(str(cur))

    def prepare_for_close(self) -> bool:
        """在主窗口退出前询问未保存的标注修改并保存 X-AnyLabeling 状态。"""
        if self._xal_window is None:
            return True
        return self._xal_window.prepare_for_close()
