"""配置标签页表单标签与下拉框/行内控件间距回归测试。

职责：
    在离线 `offscreen` Qt 环境下验证共享布局辅助函数 `configure_form_layout` / `align_form_labels`，
    以及五个配置标签页（视频抽帧、MLLM 标注纠错、数据集准备、YOLO26 训练、验证推理与导出）中：
    1. 所有 `QFormLayout` 标签均采用左对齐，且右侧下拉框/输入框紧邻最长标签紧凑对齐；
    2. 同页多分组在挂载到 `MainWindow` 并继承样式表字号后，所有标签左边缘对齐、右侧字段列沿同页最长标签垂直对齐且切换动态行不抖动；
    3. 行内多控件组合（如「格式:」+ 下拉框、「起:/止:」、「低阈值Conf:/IoU:/ImgSz:」、「Conf:/IoU:」）
       中的标签不会被布局均分拉宽，与紧随其后的控件始终保持合适距离。

运行环境与协作边界：
    通过 `pytest` 离线运行，依赖 `PyQt6`；使用 `MainWindow(lazy_init=True)` 构造完整标签页但不加载内嵌
    X-AnyLabeling 或触发真实 `torch` 硬件探测，不访问外部网络。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QSettings, Qt
from PyQt6.QtWidgets import QApplication, QFormLayout, QHBoxLayout, QLabel, QSizePolicy, QWidget

from yolo_studio.core.trainer import DeviceInfo
from yolo_studio.gui.common import FORM_LABEL_SPACING, align_form_labels, configure_form_layout
from yolo_studio.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """初始化模块级无头 `QApplication`；与内嵌工作台测试共用进程时提前开启 OpenGL 上下文共享。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if existing is not None:
        return existing
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication([])


def _make_device_info(gpu_count: int = 1) -> DeviceInfo:
    """构造无硬件探测开销的模拟计算设备快照。"""
    gpus = tuple((str(i), f"Mock RTX 40{7 - i}0", 12.0 - i * 4.0) for i in range(gpu_count))
    return DeviceInfo(
        cuda_available=gpu_count > 0,
        device_count=gpu_count,
        device_name=gpus[0][1] if gpus else "Mock CPU",
        vram_gb=gpus[0][2] if gpus else 0.0,
        recommended_device="0" if gpus else "cpu",
        gpu_devices=gpus,
        probing=False,
    )


def _visual_label_to_field_gap(label: QLabel, field_left: int) -> int:
    """计算标签文字实际右边缘到右侧控件左边缘的水平像素距离。"""
    geom = label.geometry()
    if label.alignment() & Qt.AlignmentFlag.AlignRight:
        text_right = geom.x() + geom.width()
    else:
        text_right = geom.x() + label.fontMetrics().horizontalAdvance(label.text())
    return field_left - text_right


def _find_inline_pair_gap(container: QWidget, label_text: str, target: QWidget) -> int:
    """在同一父容器的水平行内查找指定 `QLabel` 文字右端到紧邻目标控件左端的水平间距（像素）。"""
    for lbl in container.findChildren(QLabel):
        if lbl.text() == label_text and lbl.parentWidget() is target.parentWidget():
            lbl_geom = lbl.geometry()
            text_right = lbl_geom.x() + lbl.fontMetrics().horizontalAdvance(lbl.text())
            return target.geometry().x() - text_right
    raise AssertionError(f"未在容器中找到行内标签 {label_text!r}")


def test_align_form_labels_realigns_after_stylesheet_font_change(qt_app: QApplication) -> None:
    """验证 `align_form_labels` 在父容器套用放大字号样式表后自动重算并保持跨组列宽一致。"""
    host = QWidget()
    outer = QHBoxLayout(host)
    box1 = QWidget()
    box2 = QWidget()
    f1 = QFormLayout(box1)
    f2 = QFormLayout(box2)
    configure_form_layout(f1)
    configure_form_layout(f2)
    f1.addRow("超长说明标签名称(可选):", QWidget())
    f2.addRow("短标签:", QWidget())
    outer.addWidget(box1)
    outer.addWidget(box2)

    initial_w = align_form_labels([f1, f2])
    assert initial_w > 0

    # 套用明显更大的字号，模拟挂入 MainWindow 继承 APP_STYLESHEET 的场景
    host.setStyleSheet("QWidget { font-size: 20px; }")
    host.show()
    qt_app.processEvents()

    lbl1 = f1.itemAt(0, QFormLayout.ItemRole.LabelRole).widget()
    lbl2 = f2.itemAt(0, QFormLayout.ItemRole.LabelRole).widget()
    assert isinstance(lbl1, QLabel) and isinstance(lbl2, QLabel)
    assert bool(lbl1.alignment() & Qt.AlignmentFlag.AlignLeft)
    assert bool(lbl2.alignment() & Qt.AlignmentFlag.AlignLeft)
    assert lbl1.minimumWidth() == lbl2.minimumWidth()
    assert lbl1.minimumWidth() > initial_w
    assert lbl2.minimumWidth() == lbl1.sizeHint().width()
    host.close()


def test_all_config_tabs_form_and_inline_label_spacing(qt_app: QApplication, tmp_path: Path) -> None:
    """验证五个配置标签页的表单标签左对齐、右侧控件沿最长标签跨组对齐及行内标签不被拉伸。"""
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    win = MainWindow(workspace_root=tmp_path, lazy_init=True, settings=settings)
    try:
        win._on_device_probed(_make_device_info(gpu_count=1))
        win.resize(1380, 860)
        win.show()
        qt_app.processEvents()

        config_tabs = [
            (0, win.extract_tab),
            (2, win.review_tab),
            (3, win.dataset_tab),
            (4, win.train_tab),
            (5, win.eval_tab),
        ]
        for tab_idx, tab in config_tabs:
            win.tabs.setCurrentIndex(tab_idx)
            qt_app.processEvents()

            for form in tab.findChildren(QFormLayout):
                assert bool(form.labelAlignment() & Qt.AlignmentFlag.AlignLeft)
                label_xs: set[int] = set()
                field_xs: set[int] = set()
                for row in range(form.rowCount()):
                    lbl_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    fld_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                    if lbl_item is None or fld_item is None:
                        continue
                    lbl = lbl_item.widget()
                    if not isinstance(lbl, QLabel) or not lbl.text() or not lbl.isVisible():
                        continue
                    assert bool(lbl.alignment() & Qt.AlignmentFlag.AlignLeft)
                    label_xs.add(lbl.geometry().x())
                    field_xs.add(fld_item.geometry().x())
                    # 标签盒子右端到字段左端应恰好为 horizontalSpacing
                    box_gap = fld_item.geometry().x() - (lbl.geometry().x() + lbl.geometry().width())
                    assert box_gap == form.horizontalSpacing()
                assert len(label_xs) == 1
                assert len(field_xs) == 1

        # 验证多分组页面的字段列左边缘沿同页最长标签精确垂直对齐（含样式表生效后的最终布局）
        # 1) Tab 3（MLLM 纠错）：切出 OpenAI Base URL 与 SAM 隐藏行后三组依然对齐，且最长标签紧邻控件
        win.tabs.setCurrentIndex(2)
        win.review_tab.provider_combo.setCurrentIndex(1)
        win.review_tab.refine_mode_combo.setCurrentIndex(win.review_tab.refine_mode_combo.count() - 1)
        qt_app.processEvents()
        review_forms = (win.review_tab._src_form, win.review_tab._llm_form, win.review_tab._pol_form)
        review_field_xs = {
            form.itemAt(0, QFormLayout.ItemRole.FieldRole).geometry().x() for form in review_forms
        }
        assert len(review_field_xs) == 1
        review_min_gap = min(
            _visual_label_to_field_gap(
                form.itemAt(r, QFormLayout.ItemRole.LabelRole).widget(),
                form.itemAt(r, QFormLayout.ItemRole.FieldRole).geometry().x(),
            )
            for form in review_forms
            for r in range(form.rowCount())
            if form.itemAt(r, QFormLayout.ItemRole.LabelRole) is not None
            and isinstance(form.itemAt(r, QFormLayout.ItemRole.LabelRole).widget(), QLabel)
            and form.itemAt(r, QFormLayout.ItemRole.LabelRole).widget().isVisible()
            and form.itemAt(r, QFormLayout.ItemRole.LabelRole).widget().text()
        )
        assert FORM_LABEL_SPACING - 2 <= review_min_gap <= FORM_LABEL_SPACING + 4

        # 2) Tab 4（数据集准备）：来源分组与划分分组字段列沿最长标签精确对齐
        win.tabs.setCurrentIndex(3)
        qt_app.processEvents()
        ds_forms = (win.dataset_tab._source_form, win.dataset_tab._split_form)
        ds_field_xs = {
            form.itemAt(0, QFormLayout.ItemRole.FieldRole).geometry().x() for form in ds_forms
        }
        assert len(ds_field_xs) == 1
        assert FORM_LABEL_SPACING - 2 <= _visual_label_to_field_gap(
            win.dataset_tab.classes_label,
            win.dataset_tab.classes_picker.geometry().x(),
        ) <= FORM_LABEL_SPACING + 4

        # 3) Tab 6（验证、推理与导出）：三组字段列沿最长标签精确对齐
        win.tabs.setCurrentIndex(5)
        qt_app.processEvents()
        eval_forms = win.eval_tab.findChildren(QFormLayout)
        eval_field_xs = {
            form.itemAt(0, QFormLayout.ItemRole.FieldRole).geometry().x()
            for form in eval_forms
        }
        assert len(eval_field_xs) == 1

        # 验证此前被 QHBoxLayout 均分拉宽的行内「标签 ↔ 下拉框/微调框」均已恢复紧凑间距
        win.tabs.setCurrentIndex(0)
        qt_app.processEvents()
        assert 2 <= _find_inline_pair_gap(win.extract_tab, "起:", win.extract_tab.start_sec_spin) <= 12
        assert 2 <= _find_inline_pair_gap(win.extract_tab, "止:", win.extract_tab.end_sec_spin) <= 12
        assert 2 <= _find_inline_pair_gap(win.extract_tab, "格式:", win.extract_tab.format_combo) <= 12
        assert 2 <= _find_inline_pair_gap(win.extract_tab, "JPG质量:", win.extract_tab.quality_spin) <= 12

        win.tabs.setCurrentIndex(2)
        qt_app.processEvents()
        assert 2 <= _find_inline_pair_gap(win.review_tab, "低阈值Conf:", win.review_tab.yolo_conf_spin) <= 12
        assert 2 <= _find_inline_pair_gap(win.review_tab, "IoU:", win.review_tab.yolo_iou_spin) <= 12
        assert 2 <= _find_inline_pair_gap(win.review_tab, "ImgSz:", win.review_tab.yolo_imgsz_spin) <= 12

        # 多卡 GPU 环境下，MLLM 页基准推理参数行的紧凑下拉框仅增加极小最小宽，不把表单撑出横向滚动条，
        # 同时下拉弹出列表保留充足宽度以完整展示显卡型号
        single_gpu_min_w = win.review_tab.yolo_params_widget.minimumSizeHint().width()
        win._on_device_probed(_make_device_info(gpu_count=2))
        qt_app.processEvents()
        gpu_combo = win.review_tab.yolo_device_ctrl.gpu_combo
        assert gpu_combo.isVisible() is True
        assert gpu_combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
        assert gpu_combo.minimumWidth() <= 80
        assert gpu_combo.view().minimumWidth() >= 240
        multi_gpu_min_w = win.review_tab.yolo_params_widget.minimumSizeHint().width()
        assert multi_gpu_min_w - single_gpu_min_w <= gpu_combo.minimumWidth() + 12

        win.tabs.setCurrentIndex(5)
        qt_app.processEvents()
        assert 2 <= _find_inline_pair_gap(win.eval_tab, "Conf:", win.eval_tab.conf_spin) <= 12
        assert 2 <= _find_inline_pair_gap(win.eval_tab, "IoU:", win.eval_tab.iou_spin) <= 12
    finally:
        win.close()
