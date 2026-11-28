"""标注工作台「重命名 / 合并类别」确认弹窗。

职责：
    替代原先的单行输入框：让用户输入新类别名（带已有类别自动补全），选择作用范围
    （整个目录 / 仅当前图片），并在确认前明确展示将影响的图片数、框数、是否为合并以及备份位置。

运行环境与协作边界：
    运行于 PyQt6 主线程，只负责收集用户选择，不读写任何文件；实际改写由
    `gui.tab_labeling.LabelingStudioTab._rename_class` 调用 `core.label_batch_ops` 完成。
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class ClassRenameDialog(QDialog):
    """重命名或合并类别的确认弹窗。

    构造参数中的计数由调用方预先统计：`dir_file_count` / `dir_shape_count` 为当前目录中
    除当前图片以外含旧类别的文件数与形状数，`current_shape_count` 为当前图片（含未保存改动）中的框数。
    `has_directory=False`（尚未打开目录）时只提供“仅当前图片”范围。
    """

    def __init__(
        self,
        old_name: str,
        existing_classes: list[str],
        *,
        dir_file_count: int,
        dir_shape_count: int,
        current_shape_count: int,
        has_directory: bool,
        backup_hint: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.old_name = old_name
        self._existing = [c for c in existing_classes if c != old_name]
        self.setWindowTitle("重命名 / 合并类别")
        self.setMinimumWidth(460)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.addWidget(QLabel(f"将类别「<b>{old_name}</b>」改为："))

        self.name_edit = QLineEdit(old_name)
        self.name_edit.selectAll()
        self.name_edit.setPlaceholderText("输入新类别名；输入已有类别名即为合并")
        completer = QCompleter(self._existing, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.name_edit.setCompleter(completer)
        lay.addWidget(self.name_edit)

        total_files = dir_file_count + (1 if current_shape_count > 0 else 0)
        total_shapes = dir_shape_count + current_shape_count
        self.dir_radio = QRadioButton(
            f"整个目录（推荐）：{total_files} 张图片，共 {total_shapes} 个框"
        )
        self.cur_radio = QRadioButton(f"仅当前图片：{current_shape_count} 个框")
        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self.dir_radio)
        self._scope_group.addButton(self.cur_radio)
        self.dir_radio.setEnabled(has_directory)
        if has_directory:
            self.dir_radio.setChecked(True)
        else:
            self.cur_radio.setChecked(True)
        lay.addWidget(self.dir_radio)
        lay.addWidget(self.cur_radio)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.info_label)
        self._backup_hint = backup_hint

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_btn = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        lay.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.name_edit.textChanged.connect(self._refresh_state)
        self.dir_radio.toggled.connect(self._refresh_state)
        self._refresh_state()

    def new_name(self) -> str:
        """返回去除首尾空白后的新类别名。"""
        return self.name_edit.text().strip()

    def apply_to_directory(self) -> bool:
        """是否作用于整个目录（否则仅当前图片）。"""
        return self.dir_radio.isChecked()

    def is_merge(self) -> bool:
        """新类别名是否为已有的其它类别（即合并操作）。"""
        return self.new_name() in self._existing

    def _refresh_state(self, *_args: object) -> None:
        name = self.new_name()
        valid = bool(name) and name != self.old_name
        self.ok_btn.setEnabled(valid)
        self.ok_btn.setText("合并" if valid and self.is_merge() else "重命名")
        lines: list[str] = []
        if not name:
            lines.append("<span style='color:#cf222e'>类别名不能为空。</span>")
        elif name == self.old_name:
            lines.append("<span style='color:#57606a'>请输入与原名不同的类别名。</span>")
        elif self.is_merge():
            lines.append(
                f"<span style='color:#9a6700'>「{name}」已存在：将把「{self.old_name}」的框全部并入「{name}」，"
                f"并从类别清单中移除「{self.old_name}」。</span>"
            )
        if valid and self.apply_to_directory():
            lines.append(
                f"<span style='color:#57606a'>改写前会自动备份原 .json 到：{self._backup_hint}</span>"
            )
        elif valid:
            lines.append(
                "<span style='color:#9a6700'>仅修改当前图片，目录中其它图片仍保留旧类别名，"
                "训练时会被当作两个不同类别。</span>"
            )
        self.info_label.setText("<br>".join(lines))
