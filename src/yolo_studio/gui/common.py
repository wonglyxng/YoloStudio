"""PySide6 界面共享辅助控件与本地路径/图像工具函数。

职责：
    提供路径选择器行控件、系统文件管理器跳转、等比缩略图加载与紧凑型指标卡片构建。

运行环境与协作边界：
    Python >= 3.11 + `PySide6`；供 `gui` 各标签页复用，不持有全局状态或启动后台线程。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)


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
