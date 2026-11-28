"""标注工作台可搜索的类别快速选择与新建弹窗。

职责：
    解决类别超过 9 个时 `1~9` 数字快捷键覆盖不全的问题：提供即时输入过滤的类别列表，
    支持空格分词多关键字过滤、大小写不敏感匹配、键盘 `↑/↓` 移动选中、`Enter` 一键确认，
    以及在输入全新类别名时直接回车新建并应用。

运行环境与协作边界：
    运行于 PyQt6 主线程，仅依赖 `PyQt6` 与 `core.annotation_io.label_color_rgb`；
    不直接修改工作台数据，调用方在 `exec() == Accepted` 后通过 `selected_label()` 获取结果。
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtGui import QColor, QIcon, QKeyEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.annotation_io import label_color_rgb

_CREATE_NEW_ROLE = Qt.ItemDataRole.UserRole + 1


def _make_swatch_icon(rgb: tuple[int, int, int], size: int = 12) -> QIcon:
    """绘制用于类别列表项的圆角色块图标。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(*rgb))
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)
    painter.end()
    return QIcon(pix)


def filter_classes_by_query(classes: list[str], query: str) -> list[str]:
    """按用户输入关键字过滤并排序类别清单（纯函数，便于测试）。

    匹配与排序规则：
        将 `query` 按空白拆分为多个词元，要求类别名（忽略大小写）包含全部词元；
        排序优先展示：完全匹配 > 前缀匹配 > 原列表顺序。
    """
    tokens = [tok for tok in query.strip().lower().split() if tok]
    if not tokens:
        return list(classes)
    full_q = query.strip().lower()
    exact: list[str] = []
    prefix: list[str] = []
    partial: list[str] = []
    for cls_name in classes:
        low = cls_name.lower()
        if not all(tok in low for tok in tokens):
            continue
        if low == full_q:
            exact.append(cls_name)
        elif low.startswith(full_q):
            prefix.append(cls_name)
        else:
            partial.append(cls_name)
    return exact + prefix + partial


class ClassPickerDialog(QDialog):
    """可搜索类别快速选择与新建弹窗（用于 `C` 快捷选类、双击改类与画框后自动选类）。

    键盘交互：
        搜索框始终保持焦点，输入即实时过滤；按 `↑`/`↓` 直接在候选列表中切换高亮，
        按 `Enter` 立即确认当前高亮项（或新建输入的新类别），按 `Esc` 取消。
    """

    def __init__(
        self,
        classes: list[str],
        *,
        current_label: str = "",
        counts: dict[str, int] | None = None,
        title: str = "选择或搜索目标类别",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._all_classes: list[str] = list(classes)
        self._current_label: str = current_label.strip()
        self._counts: dict[str, int] = dict(counts or {})
        self._selected_result: str = ""

        self.setWindowTitle(title)
        self.setMinimumWidth(360)
        self.resize(380, 420)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        tip = QLabel("输入关键字搜索已有类别，↑↓ 切换、Enter 确认；输入新名称可直接新建：")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #57606a; font-size: 12px;")
        lay.addWidget(tip)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索类别名（支持空格多关键词）或输入新类别...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.installEventFilter(self)
        lay.addWidget(self.search_edit)

        self.class_list = QListWidget()
        self.class_list.setStyleSheet(
            "QListWidget { border: 1px solid #d0d7de; border-radius: 5px; font-size: 12px; }"
            "QListWidget::item { padding: 4px 6px; }"
            "QListWidget::item:selected { background-color: #ddf4ff; color: #0550ae; font-weight: 600; }"
        )
        lay.addWidget(self.class_list, 1)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_btn = btns.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_btn.setText("确定")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        self.search_edit.textChanged.connect(self._rebuild_list)
        self.class_list.itemDoubleClicked.connect(lambda _item: self._on_accept())

        self._rebuild_list("")
        self.search_edit.setFocus()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        """在搜索框聚焦时支持直接用上下方向键与回车选择候选类别。"""
        if obj is self.search_edit and event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                count = self.class_list.count()
                if count > 0:
                    cur = self.class_list.currentRow()
                    step = 1 if key == Qt.Key.Key_Down else -1
                    nxt = (cur + step) % count if cur >= 0 else (0 if step > 0 else count - 1)
                    self.class_list.setCurrentRow(nxt)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._on_accept()
                return True
        return super().eventFilter(obj, event)

    def _rebuild_list(self, query: str) -> None:
        self.class_list.clear()
        matched = filter_classes_by_query(self._all_classes, query)
        index_map = {name: i for i, name in enumerate(self._all_classes)}

        preselect_row = -1
        for cls_name in matched:
            orig_idx = index_map.get(cls_name, -1)
            prefix = f"[{orig_idx + 1}] " if 0 <= orig_idx < 9 else "    "
            cnt = self._counts.get(cls_name, 0)
            suffix = f"  ({cnt})" if cnt > 0 else ""
            item = QListWidgetItem(_make_swatch_icon(label_color_rgb(cls_name)), f"{prefix}{cls_name}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, cls_name)
            item.setData(_CREATE_NEW_ROLE, False)
            self.class_list.addItem(item)
            if not query.strip() and cls_name == self._current_label:
                preselect_row = self.class_list.count() - 1

        clean_query = query.strip()
        if clean_query and clean_query not in self._all_classes:
            create_item = QListWidgetItem(
                _make_swatch_icon(label_color_rgb(clean_query)),
                f"+ 新建类别「{clean_query}」",
            )
            create_item.setData(Qt.ItemDataRole.UserRole, clean_query)
            create_item.setData(_CREATE_NEW_ROLE, True)
            self.class_list.addItem(create_item)

        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(preselect_row if preselect_row >= 0 else 0)
        self.ok_btn.setEnabled(self.class_list.count() > 0 or bool(clean_query))

    def _resolve_candidate(self) -> str:
        item = self.class_list.currentItem()
        if item is not None:
            val = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if val:
                return val
        return self.search_edit.text().strip()

    def _on_accept(self) -> None:
        chosen = self._resolve_candidate()
        if not chosen:
            QMessageBox.warning(self, "提示", "类别名称不能为空。")
            return
        self._selected_result = chosen
        self.accept()

    def selected_label(self) -> str:
        """返回最终确认的类别名称（未确认时返回当前候选）。"""
        return self._selected_result or self._resolve_candidate()
