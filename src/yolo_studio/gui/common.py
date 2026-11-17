"""PyQt6 界面共享辅助控件、跨页计算任务互斥协调器与本地路径/图像工具函数。

职责：
    提供路径选择器行控件、系统文件管理器跳转、等比缩略图加载、紧凑型指标卡片构建，
    以及防止多标签页同时启动本地模型任务争抢显存的 `GpuTaskGuard`。

运行环境与协作边界：
    Python >= 3.11 + `PyQt6`；供 `gui` 各标签页与 `MainWindow` 复用，不持有模块级可变全局状态，也不自行创建后台线程。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

# 长任务日志框（`QPlainTextEdit.setMaximumBlockCount`）保留的最大行数；超出后自动丢弃最早的行，
# 避免长时间训练日志无限增长拖慢界面
LOG_MAX_LINES = 5000


def open_local_path(path: Path) -> None:
    """在系统文件管理器中打开本地目录或文件；若目标不存在则回退打开其父目录。"""
    target = path if path.exists() else path.parent
    if target.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))


def load_pixmap_scaled(image_path: Path, max_w: int, max_h: int) -> QPixmap | None:
    """加载本地图像文件并按宽高比缩放至不超过 `(max_w, max_h)` 像素；无效时返回 `None`。"""
    if not image_path.is_file():
        return None
    pixmap = QPixmap(str(image_path))
    if pixmap.isNull():
        return None
    return pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class PathPickerRow(QWidget):
    """包含路径输入框与“浏览...”按钮的水平复合控件，支持 `dir` 目录模式与文件选择模式。"""

    def __init__(
        self,
        placeholder: str = "",
        mode: str = "dir",
        file_filter: str = "All Files (*.*)",
        default_path: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.line_edit = QLineEdit(default_path)
        self.line_edit.setPlaceholderText(placeholder)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setFixedWidth(76)
        self.browse_btn.clicked.connect(self._on_browse)

        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_btn)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def setText(self, value: str) -> None:
        self.line_edit.setText(value)

    def path(self) -> Path:
        return Path(self.text())

    def _on_browse(self) -> None:
        start_dir = self.text() or str(Path.cwd())
        if self.mode == "dir":
            chosen = QFileDialog.getExistingDirectory(self, "选择文件夹", start_dir)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "选择文件", start_dir, self.file_filter)
        if chosen:
            self.line_edit.setText(chosen)


def make_metric_card(title: str, initial_value: str = "-") -> tuple[QWidget, QLabel]:
    """创建紧凑型训练指标展示卡片并返回 `(卡片控件, 数值标签)`。"""
    card = QWidget()
    card.setObjectName("metricCard")
    layout = QHBoxLayout(card)
    layout.setContentsMargins(10, 6, 10, 6)

    title_lbl = QLabel(f"{title}:")
    title_lbl.setStyleSheet("color: #666666; font-weight: 500;")
    val_lbl = QLabel(initial_value)
    val_lbl.setStyleSheet("font-weight: 700; font-size: 13px;")
    layout.addWidget(title_lbl)
    layout.addStretch(1)
    layout.addWidget(val_lbl)
    return card, val_lbl


class GpuTaskGuard:
    """跨标签页本地重计算 / GPU 模型任务互斥协调器。

    职责与设计原因：
        训练、基准模型低阈值推理、验证、推理预览与模型导出均会加载本地 PyTorch / Ultralytics 模型；
        若用户在训练未结束时切换到其他标签页启动推理或验证，极易引发 CUDA 显存溢出（OOM）。
        本协调器由 `MainWindow` 创建并注入相关标签页，仅在主线程中调用。

    状态与无死锁保证：
        内部记录最近登记的 `_active_worker` 与任务名 `_active_name`；
        占用判定直接以 `_active_worker.isRunning()` 为准，无需依赖易遗漏的手工释放回调，
        即使后台线程异常退出也不会残留锁定状态。
    """

    def __init__(self) -> None:
        self._active_worker: QThread | None = None
        self._active_name: str = ""

    def active_task_name(self) -> str | None:
        """若当前有登记的后台模型任务仍在运行则返回其名称，否则返回 `None`。"""
        if self._active_worker is not None and self._active_worker.isRunning():
            return self._active_name
        return None

    def try_acquire(self, parent: QWidget, task_name: str) -> bool:
        """在主线程检查是否允许启动 `task_name`；若有冲突任务正在运行则弹窗提示并返回 `False`。"""
        busy_name = self.active_task_name()
        if busy_name is not None:
            QMessageBox.warning(
                parent,
                "计算设备占用中",
                f"当前「{busy_name}」正在后台运行并占用本地计算设备。\n"
                f"请等待其完成或手动停止后再启动「{task_name}」，以免显存溢出。",
            )
            return False
        return True

    def bind(self, worker: QThread, task_name: str) -> None:
        """登记即将启动或刚启动的后台模型工作线程。"""
        self._active_worker = worker
        self._active_name = task_name
