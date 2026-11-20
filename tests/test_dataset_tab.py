"""「4. 数据集准备」标签页（`DatasetTab`）界面交互测试。

职责：
    在离线 `offscreen` Qt 环境下验证数据集页的划分比例归一化提示、classes.txt 与 need_review 选项联动、
    长路径输入框 tooltip/光标约定、真实后台扫描与生成流程、覆盖已有划分前的确认、
    类别表数值排序与验证集缺类高亮，以及参数修改后结果过期与“前往训练”按钮的失效。

运行环境与协作边界：
    通过 `pytest` 运行，依赖 `PyQt6` 与 `Pillow`；只实例化 `DatasetTab`，不加载内嵌 X-AnyLabeling，
    不访问外部网络。确认对话框通过 monkeypatch 替换 `QMessageBox.question`，不弹出可见窗口。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from PIL import Image
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from yolo_studio.core.dataset_converter import AnnotationFormat, DatasetBuildResult
from yolo_studio.gui.common import PathPickerRow
from yolo_studio.gui.tab_dataset import DatasetTab


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """初始化模块级无头 `QApplication`；与内嵌工作台测试共用进程时需提前开启 OpenGL 上下文共享。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if existing is not None:
        return existing
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication([])


def _make_labeled_dir(root: Path, count: int = 10) -> Path:
    """生成带 X-AnyLabeling `.json` 标注的图片目录：前 count-1 张含 person，仅最后一张含 rare。"""
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        img = root / f"f_{idx:02d}.jpg"
        Image.new("RGB", (200, 100), color=(40, 120, 200)).save(img)
        label = "rare" if idx == count - 1 else "person"
        payload = {
            "shapes": [{"label": label, "points": [[20, 10], [120, 60]], "shape_type": "rectangle"}],
            "imagePath": img.name,
            "imageWidth": 200,
            "imageHeight": 100,
        }
        img.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _wait_idle(app: QApplication, tab: DatasetTab, timeout: float = 30.0) -> None:
    """等待后台扫描/生成线程结束并处理排队的跨线程信号。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if not tab._is_busy():
            return
        time.sleep(0.02)
    raise AssertionError("后台任务超时未结束")


def test_ratio_hint_format_hint_and_exclude_option(qt_app: QApplication, tmp_path: Path) -> None:
    """验证比例合计不为 1 时给出归一化占比、Val 为 0 时警示，以及格式/递归选项的文案联动。"""
    tab = DatasetTab(tmp_path)
    assert "合计" not in tab.ratio_hint.text()

    tab.test_spin.setValue(0.1)
    assert "73%" in tab.ratio_hint.text() and "合计 1.10" in tab.ratio_hint.text()

    tab.val_spin.setValue(0.0)
    assert "Val 为 0" in tab.ratio_hint.text()

    tab.format_combo.setCurrentIndex(tab.format_combo.findData(AnnotationFormat.YOLO_TXT))
    assert "建议" in tab.classes_label.text()
    tab.format_combo.setCurrentIndex(0)
    assert "可选" in tab.classes_label.text()

    assert tab.recursive_chk.isChecked()  # 默认递归，与标注工作台列图范围一致
    assert tab.exclude_need_review_chk.isEnabled()
    tab.recursive_chk.setChecked(False)
    assert not tab.exclude_need_review_chk.isEnabled()


def test_scope_hint_next_to_images_dir(qt_app: QApplication, tmp_path: Path) -> None:
    """验证递归开关位于图片目录分组内，并按目录结构实时提示图片范围（子目录有图未勾选递归时警示）。"""
    root = tmp_path / "pilot"
    _make_labeled_dir(root / "video-001", count=3)
    _make_labeled_dir(root / "video-002", count=2)
    _make_labeled_dir(root / "need_review", count=1)
    (root / "清单.jsonl").write_text("{}", encoding="utf-8")

    tab = DatasetTab(tmp_path)
    source_group = tab.images_picker.parentWidget()
    assert tab.recursive_chk.parentWidget() is source_group  # 与图片目录同在分组 1

    tab.recursive_chk.setChecked(False)
    tab.set_frames_dir(str(root))
    tab._refresh_scope_counts()
    assert "顶层没有图片" in tab.scope_hint.text() and "6 张" in tab.scope_hint.text()
    assert "请勾选「包含子目录」" in tab.scope_hint.text()

    tab.recursive_chk.setChecked(True)
    assert "共 5 张" in tab.scope_hint.text() and "排除 need_review 中的 1 张" in tab.scope_hint.text()
    tab.exclude_need_review_chk.setChecked(False)
    assert "共 6 张" in tab.scope_hint.text()

    tab.set_frames_dir(str(tmp_path / "missing"))
    tab._refresh_scope_counts()
    assert "不存在" in tab.scope_hint.text()


def test_path_picker_tooltip_and_cursor(qt_app: QApplication) -> None:
    """验证程序写入长路径后光标回到开头，tooltip 同步为完整路径，并转发 textChanged 信号。"""
    picker = PathPickerRow()
    received: list[str] = []
    picker.textChanged.connect(received.append)
    long_path = "C:/" + "/".join(f"very_long_folder_{i}" for i in range(12))
    picker.setText(long_path)
    assert picker.line_edit.cursorPosition() == 0
    assert picker.line_edit.toolTip() == long_path
    assert received[-1] == long_path


def test_scan_build_confirm_overwrite_and_stale(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """走通真实后台扫描与生成：覆盖确认、类别表、自动同步 data.yaml、参数修改后结果过期与按钮失效。"""
    raw_dir = _make_labeled_dir(tmp_path / "raw")
    out_dir = tmp_path / "ds"
    tab = DatasetTab(tmp_path)
    tab.set_frames_dir(str(raw_dir))
    tab.output_ds_picker.setText(str(out_dir))

    tab._on_scan()
    _wait_idle(qt_app, tab)
    assert tab.stat_cards[0].value_label.text() == "10"
    assert tab.class_table.rowCount() == 2
    assert "扫描完成" in tab.summary_label.text()

    yaml_paths: list[str] = []
    go_train: list[bool] = []
    tab.send_data_yaml.connect(yaml_paths.append)
    tab.request_train_tab.connect(lambda: go_train.append(True))

    tab._on_build()  # 首次生成：输出目录为空，不应弹出确认
    _wait_idle(qt_app, tab)
    assert len(yaml_paths) == 1 and Path(yaml_paths[0]).is_file()
    assert tab.send_train_btn.isEnabled()
    headers = [tab.class_table.horizontalHeaderItem(c).text() for c in range(tab.class_table.columnCount())]
    assert headers[-1] == "Train / Val"

    asked: list[str] = []

    def _deny(*args: object, **_kwargs: object) -> QMessageBox.StandardButton:
        asked.append(str(args[2]))
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", _deny)
    tab._on_build()  # 已有划分：用户拒绝后不应启动后台任务
    assert asked and "images/train" in asked[0]
    assert not tab._is_busy()

    tab.send_train_btn.click()
    assert go_train == [True] and len(yaml_paths) == 2

    tab.seed_spin.setValue(7)
    assert not tab.send_train_btn.isEnabled()
    assert "过期" in tab.summary_label.text()

    tab.mark_annotations_changed(str(raw_dir))
    assert "已更新标注文件" in tab.summary_label.text()


def test_class_table_numeric_sort_and_missing_val_highlight(qt_app: QApplication, tmp_path: Path) -> None:
    """验证类别表按数值排序、偏少类别高亮，以及验证集缺失类别的单元格标红和告警计数。"""
    tab = DatasetTab(tmp_path)
    classes = [f"c{i}" for i in range(11)]
    counts = {name: (100 if i == 0 else i) for i, name in enumerate(classes)}
    tab._populate_class_table(classes, counts)

    tab.class_table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
    ordered = [int(tab.class_table.item(r, 2).text()) for r in range(tab.class_table.rowCount())]
    assert ordered == sorted(ordered, reverse=True)
    low_row = next(r for r in range(tab.class_table.rowCount()) if tab.class_table.item(r, 1).text() == "c1")
    assert tab.class_table.item(low_row, 1).background().color().name() == "#fff4e5"

    result = DatasetBuildResult(
        output_dir=tmp_path,
        data_yaml_path=tmp_path / "data.yaml",
        classes_txt_path=tmp_path / "classes.txt",
        classes=["a", "b"],
        train_count=4,
        val_count=1,
        test_count=0,
        total_boxes=5,
        class_counts={"a": 4, "b": 1},
        split_class_counts={"train": {"a": 3, "b": 1}, "val": {"a": 1}, "test": {}},
    )
    tab._running_revision = tab._param_revision
    tab._on_build_done(result)
    val_col = tab.class_table.columnCount() - 1
    b_row = next(r for r in range(tab.class_table.rowCount()) if tab.class_table.item(r, 1).text() == "b")
    assert tab.class_table.item(b_row, val_col).text() == "1 / 0"
    assert tab.class_table.item(b_row, val_col).background().color().name() == "#fde2e1"
    assert tab.stat_cards[5].value_label.text() == "1"
    assert "缺少以下类别" in tab.log_box.toPlainText()
