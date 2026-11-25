"""PyQt6 原生轻量标注与复核工作台及跨页联动集成测试。

职责：
    在离线 `offscreen` Qt 环境下验证主窗口六标签页装配、原生 PyQt6 标注工作台画布交互
    （画框、选框移动/调大小、数字键与列表改类、复制、删除、撤销重做、AI 漏标提醒条、
    `F` 一键复核确认、本地 SAM 收框结果回填与撤回、自动保存与未保存拦截）、
    快捷键作用域隔离、`open_directory` 精确目录/单图/类别加载、跨标签页联动跳转、
    跨页 GPU 任务互斥与 ONNX 导出 FP16 置灰、异步视频探测、LLM 智能标注对比图缓存，
    以及关闭主窗口时等待后台线程安全停止后再退出。

运行环境与协作边界：
    通过 `pytest` 运行，仅依赖 `PyQt6` 与 `Pillow`（完全不依赖 X-AnyLabeling 或 `PyQt6-WebEngine`）；
    不访问外部网络，也不弹出可见桌面窗口。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Event

import pytest
from PIL import Image
from PyQt6.QtCore import QSettings, Qt, QThread
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGroupBox, QMessageBox

from yolo_studio._version import __version__
from yolo_studio.gui.main_window import MainWindow
from yolo_studio.gui.tab_labeling import ensure_cuda_dll_search_paths


def _make_window(tmp_path: Path, *, lazy_init: bool = False) -> MainWindow:
    """构造使用临时独立 `QSettings` 的 `MainWindow`，防止自动化测试读写宿主机真实用户配置。"""
    settings = QSettings(str(tmp_path / "_test_settings.ini"), QSettings.Format.IniFormat)
    return MainWindow(workspace_root=tmp_path, lazy_init=lazy_init, settings=settings)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """初始化模块级无头 `QApplication`。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    ensure_cuda_dll_search_paths()
    existing = QApplication.instance()
    if existing is None:
        app = QApplication([])
    else:
        app = existing
    return app


def _write_sample_frame(path: Path, size: tuple[int, int] = (160, 120)) -> None:
    """在临时目录生成用于测试加载的 JPEG 图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(60, 130, 180)).save(path)


def test_mainwindow_mounts_native_workbench_and_scopes_shortcuts(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证主窗口装配 6 个标签页，且原生标注工作台组合快捷键随可见态自动启停隔离。"""
    win = _make_window(tmp_path)
    win.show()
    qt_app.processEvents()
    try:
        assert win.windowTitle() == f"YoloStudio v{__version__}"
        assert win.tabs.count() == 6
        assert win.tabs.tabText(1) == "2. 标注工作台"
        assert win.tabs.tabText(2) == "3. LLM 智能标注"
        assert win.labeling_tab.to_review_btn.text() == "→智能标注"
        assert "3. LLM 智能标注" in win.labeling_tab.to_review_btn.toolTip()
        review_group_titles = [
            box.title() for box in win.review_tab.findChildren(QGroupBox)
        ]
        assert "2. 大语言模型 (LLM) 配置" in review_group_titles
        assert win.labeling_tab.ensure_mounted() is True

        # 初始位于第 1 页（视频抽帧），工作台不可见，快捷键处于停用隔离态
        actions = list(win.labeling_tab._scoped_actions)
        assert len(actions) >= 4
        assert all(not act.isEnabled() for act in actions)

        # 切入第 2 页（标注工作台）后激活工作台内快捷键
        win.tabs.setCurrentIndex(1)
        qt_app.processEvents()
        assert all(
            act.isEnabled() and act.shortcutContext() == Qt.ShortcutContext.WidgetWithChildrenShortcut
            for act in actions
        )

        # 切离第 2 页后再次停用隔离
        win.tabs.setCurrentIndex(2)
        qt_app.processEvents()
        assert all(not act.isEnabled() for act in actions)
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_native_workbench_keyboard_shortcuts_and_tab_isolation(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """用真实按键验证原生工作台 R/W/E/A/D 快捷键生效、搜索框打字不误触、切至其他标签页不串键。"""
    frames_dir = tmp_path / "shortcut_frames"
    _write_sample_frame(frames_dir / "f1.jpg")
    _write_sample_frame(frames_dir / "f2.jpg")

    win = _make_window(tmp_path)
    win.show()
    QTest.qWaitForWindowExposed(win)
    win.activateWindow()
    qt_app.processEvents()
    try:
        win._on_open_in_workbench(str(frames_dir))
        qt_app.processEvents()

        assert win.tabs.currentIndex() == 1
        tab = win.labeling_tab
        assert qt_app.focusWidget() is tab.canvas
        assert tab.current_image_path() is not None
        assert tab.current_image_path().name == "f1.jpg"

        # 1) 工作台内按 D 切下一张、按 A 切回上一张、按 R 进入画矩形框模式、按 E 回到编辑模式
        QTest.keyClick(tab.canvas, Qt.Key.Key_D)
        qt_app.processEvents()
        assert tab.current_image_path().name == "f2.jpg"

        QTest.keyClick(tab.canvas, Qt.Key.Key_A)
        qt_app.processEvents()
        assert tab.current_image_path().name == "f1.jpg"

        QTest.keyClick(tab.canvas, Qt.Key.Key_R)
        qt_app.processEvents()
        assert tab.canvas.drawing() is True

        QTest.keyClick(tab.canvas, Qt.Key.Key_E)
        qt_app.processEvents()
        assert tab.canvas.editing() is True

        # 2) 聚焦工作台内文件搜索框打字 d、r 时走文本输入，不触发翻页或模式切换
        tab.file_search.setFocus()
        qt_app.processEvents()
        QTest.keyClick(tab.file_search, Qt.Key.Key_D)
        QTest.keyClick(tab.file_search, Qt.Key.Key_R)
        qt_app.processEvents()
        assert tab.file_search.text() == "dr"
        assert tab.canvas.editing() is True
        tab.file_search.clear()
        qt_app.processEvents()

        # 3) 切回第 1 页（视频抽帧）后按 D、R 不影响工作台状态
        win.tabs.setCurrentIndex(0)
        win.extract_tab.start_btn.setFocus()
        qt_app.processEvents()
        QTest.keyClick(win.extract_tab.start_btn, Qt.Key.Key_D)
        QTest.keyClick(win.extract_tab.start_btn, Qt.Key.Key_R)
        qt_app.processEvents()
        assert tab.current_image_path().name == "f1.jpg"
        assert tab.canvas.editing() is True
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_switching_next_image_with_long_path_does_not_resize_mainwindow(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证切换长路径/高分辨率图片时顶部省略状态栏不会抬高 minimumSizeHint 或动态改变主窗口尺寸。"""
    deep_dir = tmp_path / ("长路径测试目录_" * 6)
    _write_sample_frame(deep_dir / "img_001.jpg", size=(640, 480))
    long_name = "img_002_" + ("very_long_frame_filename_" * 5) + ".jpg"
    _write_sample_frame(deep_dir / long_name, size=(2560, 1440))

    win = _make_window(tmp_path)
    win.show()
    qt_app.processEvents()
    try:
        initial_size = win.size()
        initial_min_w = win.minimumSizeHint().width()
        assert win.labeling_tab.status_label.minimumSizeHint().width() == 80

        win.tabs.setCurrentIndex(1)
        assert win.labeling_tab.open_directory(deep_dir) is True
        qt_app.processEvents()
        size_after_first = win.size()
        assert size_after_first == initial_size

        assert win.labeling_tab.open_next_image() is True
        qt_app.processEvents()

        # 切换到超长路径的第二张大图后，主窗口尺寸与最小宽度保持不变，且全文保存在 full_text/toolTip 中
        assert win.size() == initial_size
        assert win.minimumSizeHint().width() == initial_min_w
        assert long_name in win.labeling_tab.status_label.full_text()
        assert long_name in win.labeling_tab.status_label.toolTip()
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_native_workbench_open_directory_classes_and_cross_tab_routing(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证 `open_directory`、`classes.txt` 预载、单图定位及数据集/智能标注页双向联动信号。"""
    frames_dir = tmp_path / "frames"
    img_a = frames_dir / "frame_001.jpg"
    img_b = frames_dir / "frame_002.jpg"
    _write_sample_frame(img_a)
    _write_sample_frame(img_b)

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\nhelmet\n", encoding="utf-8")

    win = _make_window(tmp_path)
    try:
        # 从「3. LLM 智能标注」触发跳转至「2. 标注工作台」并定位第二张图
        win.tabs.setCurrentWidget(win.review_tab)
        win.review_tab.open_in_workbench.emit(
            str(frames_dir), str(classes_file), str(img_b)
        )

        assert win.tabs.currentIndex() == 1
        assert win.labeling_tab.current_directory() == frames_dir.resolve()
        assert win.labeling_tab.unique_label_list.count() == 2
        assert win.labeling_tab.current_image_path() == img_b.resolve()

        # 从「2. 标注工作台」发回「4. 数据集准备」与「3. LLM 智能标注」
        win.labeling_tab.send_to_dataset.emit(str(frames_dir), str(classes_file))
        assert win.tabs.currentWidget() is win.dataset_tab
        assert win.tabs.tabText(win.tabs.currentIndex()) == "4. 数据集准备"
        assert Path(win.dataset_tab.images_picker.text()).resolve() == frames_dir.resolve()
        assert Path(win.dataset_tab.classes_picker.text()).resolve() == classes_file.resolve()

        win.labeling_tab.send_to_review.emit(str(frames_dir))
        assert win.tabs.currentWidget() is win.review_tab
        assert win.tabs.tabText(win.tabs.currentIndex()) == "3. LLM 智能标注"
        assert Path(win.review_tab.images_picker.text()).resolve() == frames_dir.resolve()
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


class _CancellableThread(QThread):
    """模拟支持协作取消的后台任务线程，收到 `cancel()` 后在下一个检查点退出。"""

    def __init__(self) -> None:
        super().__init__()
        self._cancel = Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        while not self._cancel.wait(0.05):
            pass


def test_close_waits_for_running_workers(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证关闭主窗口时会请求后台线程停止，并在线程结束后才真正关闭，不在运行中析构 `QThread`。"""
    win = _make_window(tmp_path)
    worker = _CancellableThread()
    win.train_tab.worker = worker
    worker.start()
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.StandardButton.Yes)
    )
    win.show()
    try:
        win.close()
        # 首次关闭被忽略并进入等待；线程收到取消信号后退出，定时器随后触发真正关闭
        assert win._close_wait_timer.isActive() or not worker.isRunning()
        deadline = time.monotonic() + 5.0
        while win.isVisible() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.02)
        assert not worker.isRunning()
        assert not win.isVisible()
    finally:
        worker.cancel()
        worker.wait(2000)


def test_native_workbench_full_edit_review_undo_sam_and_unsaved_guard(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证原生工作台的画框、几何微调、数字键改类、复制、撤销重做、AI 漏标横幅、SAM 收框回调与未保存拦截。"""
    frames_dir = tmp_path / "review_batch"
    img1 = frames_dir / "01_need_review.jpg"
    img2 = frames_dir / "02_clean.jpg"
    _write_sample_frame(img1, size=(400, 300))
    _write_sample_frame(img2, size=(400, 300))

    # 为第 1 张图写入含 LLM 漏标提示与审核说明的初始 .json
    init_json = {
        "version": "2.4.4",
        "flags": {"has_missing_hint": True, "missing_note": "画面右侧疑似漏标安全帽"},
        "shapes": [
            {
                "label": "worker",
                "score": 0.62,
                "points": [[20.0, 30.0], [140.0, 220.0]],
                "description": "relabeled: 修正为工人",
                "shape_type": "rectangle",
                "flags": {"llm_added": False, "box_refined": False},
            }
        ],
        "imagePath": img1.name,
        "imageData": None,
        "imageHeight": 300,
        "imageWidth": 400,
    }
    img1.with_suffix(".json").write_text(json.dumps(init_json, ensure_ascii=False), encoding="utf-8")

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("worker\nhelmet\nvest\n", encoding="utf-8")

    win = _make_window(tmp_path)
    win.show()
    qt_app.processEvents()
    changed_dirs: list[str] = []
    win.labeling_tab.annotations_changed.connect(changed_dirs.append)
    try:
        tab = win.labeling_tab
        win.tabs.setCurrentWidget(tab)
        assert tab.open_directory(frames_dir, classes_file=classes_file) is True
        qt_app.processEvents()

        # 1) 类别加载、AI 漏标提醒条自动亮起、对象列表与详情展示审核理由
        assert tab.unique_label_list.count() == 3
        assert tab.review_banner.isHidden() is False
        assert "漏标安全帽" in tab.review_banner_label.text()
        assert tab.shape_list.count() == 1
        tab.canvas.select_box(0, emit_signal=True)
        assert "修正为工人" in tab.box_detail_label.text()
        assert "0.620" in tab.box_detail_label.text()

        # 2) 数字键 2 将选中框快速改类为 helmet；画新框、几何微调、复制框与撤销/重做
        QTest.keyClick(tab.canvas, Qt.Key.Key_2)
        qt_app.processEvents()
        assert tab._doc is not None and tab._doc.boxes[0].label == "helmet"

        tab.canvas.box_drawn.emit(200.0, 40.0, 300.0, 160.0)
        assert len(tab._doc.boxes) == 2
        tab.canvas.box_geometry_committed.emit(1, 210.0, 50.0, 310.0, 170.0)
        assert tab._doc.boxes[1].x1 == 210.0

        tab._duplicate_selected_box()
        assert len(tab._doc.boxes) == 3

        tab._undo()
        assert len(tab._doc.boxes) == 2
        tab._redo()
        assert len(tab._doc.boxes) == 3

        # 3) 模拟本地 SAM 收框结果回填与一键 Ctrl+Z 撤回
        tab._on_sam_refine_finished(str(img1.resolve()), {0: (25.0, 35.0, 130.0, 210.0)})
        assert tab._doc.boxes[0].x1 == 25.0
        assert tab._doc.boxes[0].flags.get("box_refined") is True
        tab._undo()
        assert tab._doc.boxes[0].x1 == 20.0

        # 4) 关闭自动保存后，测试未保存修改在取消关闭时拦截
        tab.auto_save_chk.setChecked(False)
        monkeypatch.setattr(
            QMessageBox, "question", staticmethod(lambda *_a, **_k: QMessageBox.StandardButton.Cancel)
        )
        assert tab.prepare_for_close() is False

        # 5) 按 F 一键「确认通过→下张」：清除漏标提醒、保存 .json 并发出变更通知、自动跳至第 2 张图
        tab.auto_save_chk.setChecked(True)
        QTest.keyClick(tab.canvas, Qt.Key.Key_F)
        qt_app.processEvents()
        assert tab.current_image_path() == img2.resolve()
        assert len(changed_dirs) >= 1

        saved_payload = json.loads(img1.with_suffix(".json").read_text(encoding="utf-8"))
        assert saved_payload["flags"]["verified"] is True
        assert saved_payload["flags"]["has_missing_hint"] is False
        assert len(saved_payload["shapes"]) == 3
        assert saved_payload["shapes"][0]["label"] == "helmet"

        # 第 2 张 0 框图按 F 同样写出显式空图负样本 .json
        QTest.keyClick(tab.canvas, Qt.Key.Key_F)
        qt_app.processEvents()
        empty_payload = json.loads(img2.with_suffix(".json").read_text(encoding="utf-8"))
        assert empty_payload["shapes"] == []
        assert empty_payload["flags"]["verified"] is True
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_annotation_canvas_mouse_draw_move_resize_overlap_and_zoom(
    qt_app: QApplication,
) -> None:
    """验证原生画布真实鼠标拖拽画框、8 手柄调框、整框平移、重叠框小面积优先点选与缩放。"""
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QColor, QPixmap

    from yolo_studio.core.annotation_io import RectAnnotation
    from yolo_studio.gui.annotation_canvas import AnnotationCanvasView

    canvas = AnnotationCanvasView()
    canvas.resize(600, 450)
    canvas.show()
    qt_app.processEvents()
    try:
        pix = QPixmap(400, 300)
        pix.fill(QColor("#334155"))
        canvas.set_image(pix, 400, 300)
        canvas.set_zoom(1.0)
        qt_app.processEvents()

        # 1) 绘制模式真实鼠标拉框触发 box_drawn
        drawn: list[tuple[float, float, float, float]] = []
        canvas.box_drawn.connect(lambda x1, y1, x2, y2: drawn.append((x1, y1, x2, y2)))
        canvas.set_mode(AnnotationCanvasView.MODE_DRAW)
        p_start = canvas.mapFromScene(QPointF(40.0, 50.0))
        p_end = canvas.mapFromScene(QPointF(160.0, 170.0))
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p_start)
        QTest.mouseMove(canvas.viewport(), p_end)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p_end)
        qt_app.processEvents()
        assert len(drawn) == 1
        assert drawn[0][0] == pytest.approx(40.0, abs=2.0)
        assert drawn[0][2] == pytest.approx(160.0, abs=2.0)

        # 2) 重叠框：大框包含小框，点击重叠区域优先选中面积更小的小框
        big_box = RectAnnotation("person", 20.0, 20.0, 300.0, 260.0)
        small_box = RectAnnotation("helmet", 100.0, 40.0, 180.0, 110.0)
        canvas.set_mode(AnnotationCanvasView.MODE_EDIT)
        canvas.set_boxes([big_box, small_box], selected_index=-1)
        hit_pt = canvas.mapFromScene(QPointF(140.0, 75.0))
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, hit_pt)
        qt_app.processEvents()
        assert canvas.selected_index() == 1

        # 3) 拖动已选中框的右下角手柄 (se) 扩大尺寸，验证提交几何变更
        committed: list[tuple[int, float, float, float, float]] = []
        canvas.box_geometry_committed.connect(
            lambda idx, x1, y1, x2, y2: committed.append((idx, x1, y1, x2, y2))
        )
        se_pt = canvas.mapFromScene(QPointF(180.0, 110.0))
        se_target = canvas.mapFromScene(QPointF(210.0, 140.0))
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, se_pt)
        QTest.mouseMove(canvas.viewport(), se_target)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, se_target)
        qt_app.processEvents()
        assert len(committed) == 1
        assert committed[0][0] == 1
        assert committed[0][3] == pytest.approx(210.0, abs=2.0)
        assert committed[0][4] == pytest.approx(140.0, abs=2.0)

        # 4) 缩放倍率与适应窗口正常工作
        canvas.zoom_by(1.5)
        assert canvas.zoom_factor() == pytest.approx(1.5, abs=0.05)
        canvas.fit_to_window()
        assert canvas.zoom_factor() > 0.0
    finally:
        canvas.close()


def test_gpu_guard_shared_device_and_eval_onnx_half_toggle(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证主窗口跨页共享 `DeviceInfo` 与 `GpuTaskGuard` 互斥保护，以及 `EvalTab` 的 ONNX 半精度自动置灰。"""
    warnings_shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda _p, title, text: warnings_shown.append(f"{title}:{text}")),
    )

    win = _make_window(tmp_path)
    worker = _CancellableThread()
    try:
        # 三个标签页共享同一份硬件快照与互斥协调器
        assert win.train_tab.device_info is win.device_info
        assert win.review_tab.device_info is win.device_info
        assert win.eval_tab.device_info is win.device_info
        assert win.train_tab.gpu_guard is win.gpu_guard

        # 默认导出格式为 ONNX：FP16 复选框应自动置灰且未勾选；切换至 TorchScript 时仅在启用 GPU 下可用
        assert win.eval_tab.format_combo.currentData() == "onnx"
        assert not win.eval_tab.half_chk.isEnabled()
        assert not win.eval_tab.half_chk.isChecked()
        win.eval_tab.format_combo.setCurrentIndex(1)  # torchscript
        if win.eval_tab.device_ctrl.is_using_gpu():
            assert win.eval_tab.half_chk.isEnabled()
            win.eval_tab.half_chk.setChecked(True)
            # 在 TorchScript 下手动关闭 GPU 开关切为 CPU，FP16 亦自动取消并置灰
            win.eval_tab.device_ctrl.gpu_chk.setChecked(False)
            assert not win.eval_tab.half_chk.isEnabled()
            assert not win.eval_tab.half_chk.isChecked()
            win.eval_tab.device_ctrl.gpu_chk.setChecked(True)
            assert win.eval_tab.half_chk.isEnabled()
            win.eval_tab.half_chk.setChecked(True)
        else:
            assert not win.eval_tab.half_chk.isEnabled()
        win.eval_tab.format_combo.setCurrentIndex(0)  # 切回 onnx 自动取消勾选并置灰
        assert not win.eval_tab.half_chk.isEnabled()
        assert not win.eval_tab.half_chk.isChecked()

        # 绑定运行中的后台模型任务时，其他任务请求应被拦截；线程退出后自动放行
        worker.start()
        win.gpu_guard.bind(worker, "YOLO26 模型训练")
        assert win.gpu_guard.try_acquire(win, "模型推理预览") is False
        assert len(warnings_shown) == 1 and "YOLO26 模型训练" in warnings_shown[0]

        worker.cancel()
        worker.wait(2000)
        assert win.gpu_guard.try_acquire(win, "模型推理预览") is True
    finally:
        worker.cancel()
        worker.wait(2000)
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_async_video_probe_and_review_preview_cache(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证抽帧页通过后台线程异步探测视频列表，且审核页切换结果行复用 LRU 预览缓存。"""
    import cv2
    import numpy as np

    from yolo_studio.core import mllm_reviewer
    from yolo_studio.core.mllm_reviewer import CandidateBox, ImageReviewRecord

    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (160, 120))
    assert writer.isOpened()
    for _ in range(15):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    img_path = tmp_path / "rev.jpg"
    _write_sample_frame(img_path)

    render_calls = {"count": 0}
    orig_render = mllm_reviewer.render_review_comparison

    def _counting_render(rec: ImageReviewRecord, max_width: int = 1200) -> bytes:
        render_calls["count"] += 1
        return orig_render(rec, max_width=max_width)

    monkeypatch.setattr("yolo_studio.gui.tab_review.render_review_comparison", _counting_render)

    win = _make_window(tmp_path)
    try:
        # 1) 验证异步视频探测线程不阻塞主线程并将结果填入列表
        win.extract_tab._add_video_paths([video_path])
        assert win.extract_tab.probe_worker is not None
        deadline = time.monotonic() + 5.0
        while win.extract_tab.probe_worker.isRunning() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.02)
        qt_app.processEvents()
        assert win.extract_tab.video_list.count() == 1
        assert "160x120" in win.extract_tab.video_list.item(0).text()

        # 2) 验证审核结果行重复点击时直接命中内存 LRU 缓存，不重复渲染
        rec = ImageReviewRecord(
            image_path=img_path,
            output_image_path=img_path,
            output_json_path=None,
            triage_bucket="auto_passed",
            before_boxes=[CandidateBox(box_id=0, label="person", x1=10, y1=10, x2=80, y2=80)],
            after_boxes=[CandidateBox(box_id=0, label="person", x1=10, y1=10, x2=80, y2=80, status="kept")],
            kept_count=1,
        )
        win.review_tab._on_progress(1, 1, rec)
        win.review_tab._on_row_selected(0)
        win.review_tab._on_row_selected(0)
        assert render_calls["count"] == 1
    finally:
        if win.extract_tab.probe_worker and win.extract_tab.probe_worker.isRunning():
            win.extract_tab.probe_worker.cancel()
            win.extract_tab.probe_worker.wait(2000)
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_lazy_init_fast_startup_and_async_device_probe_and_cross_tab_open(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证 `lazy_init=True` 首屏秒开、原生工作台即刻就绪、异步探测无损刷新设备状态及跨页联动打开目录。"""
    frames_dir = tmp_path / "lazy_frames"
    _write_sample_frame(frames_dir / "f001.jpg")

    win = _make_window(tmp_path, lazy_init=True)
    try:
        # 1) 首屏秒开阶段：原生标注工作台无需重型依赖即刻就绪，设备处于后台探测态
        assert win.labeling_tab.ensure_mounted() is True
        assert win.device_info.probing is True
        assert "正在后台检测" in win.train_tab.hw_label.text()

        # 模拟用户在探测返回前手动改写了第 6 页的运行设备
        win.eval_tab.device_ctrl.set_device("custom:1")

        # 2) 启动后台硬件探测并等待信号回传
        win.start_background_warmup()
        assert win._device_probe_worker is not None
        deadline = time.monotonic() + 10.0
        while win._device_probe_worker.isRunning() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.02)
        qt_app.processEvents()

        assert win.device_info.probing is False
        assert win.train_tab.device_info is win.device_info
        assert win.labeling_tab.device_info is win.device_info
        assert "正在后台检测" not in win.train_tab.hw_label.text()
        assert win.train_tab.device_ctrl.device() == win.device_info.recommended_device
        # 用户手动修改过的设备值不被后台探测覆盖
        assert win.eval_tab.device_ctrl.device() == "custom:1"

        # 3) 从第 1 页发送抽帧目录时，自动跳转第 2 页原生工作台并打开目录
        assert win.tabs.currentIndex() == 0
        win.extract_tab.send_frames_dir.emit(str(frames_dir))
        assert win.tabs.currentIndex() == 1
        assert win.labeling_tab.current_directory() == frames_dir.resolve()
    finally:
        if win._device_probe_worker and win._device_probe_worker.isRunning():
            win._device_probe_worker.wait(3000)
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_train_tab_hyperparameter_presets_and_config_builder(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证训练标签页场景预设联动、手动微调自动切换“自定义”状态，以及附加超参数组装与语法校验。"""
    from yolo_studio.gui.tab_train import TrainTab

    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("names: [icon]\n", encoding="utf-8")

    tab = TrainTab(workspace_root=tmp_path)
    try:
        tab.set_data_yaml(str(yaml_file))
        # 默认为通用自然场景预设
        assert tab.preset_combo.currentData() == "default"
        assert tab.mosaic_spin.value() == pytest.approx(1.0)
        assert tab.nbs_spin.value() == 64

        # 切换到游戏/UI 弱增强预设，所有关联控件自动同步
        game_idx = tab.preset_combo.findData("game_ui")
        assert game_idx >= 0
        tab.preset_combo.setCurrentIndex(game_idx)
        qt_app.processEvents()

        assert tab.optimizer_combo.currentData() == "AdamW"
        assert tab.lr0_spin.value() == pytest.approx(0.001)
        assert tab.lrf_spin.value() == pytest.approx(0.05)
        assert tab.nbs_spin.value() == 16
        assert tab.mosaic_spin.value() == pytest.approx(0.0)
        assert tab.scale_spin.value() == pytest.approx(0.15)
        assert tab.translate_spin.value() == pytest.approx(0.03)
        assert tab.hsv_s_spin.value() == pytest.approx(0.2)

        # 手动修改单个增强参数后，预设下拉框自动切为 custom
        tab.scale_spin.setValue(0.20)
        qt_app.processEvents()
        assert tab.preset_combo.currentData() == "custom"

        # 填写附加超参数并构建 TrainConfig
        tab.epochs_spin.setValue(160)
        tab.patience_spin.setValue(160)
        tab.imgsz_combo.setCurrentText("960")
        tab.batch_spin.setValue(4)
        tab.workers_spin.setValue(0)
        tab.amp_chk.setChecked(False)
        tab.plots_chk.setChecked(False)
        tab.extra_args_edit.setPlainText("box=7.5, weight_decay=0.0005")

        cfg = tab.build_train_config()
        assert cfg.optimizer == "AdamW"
        assert cfg.lr0 == pytest.approx(0.001)
        assert cfg.lrf == pytest.approx(0.05)
        assert cfg.nbs == 16
        assert cfg.mosaic == pytest.approx(0.0)
        assert cfg.scale == pytest.approx(0.20)
        assert cfg.amp is False
        assert cfg.plots is False
        assert cfg.extra_args == {"box": 7.5, "weight_decay": 0.0005}

        # 非法附加参数在构建阶段立即抛出中文 ValueError
        tab.extra_args_edit.setPlainText("invalid_arg_without_sep")
        with pytest.raises(ValueError, match="分隔符"):
            tab.build_train_config()
    finally:
        tab.close()


def test_train_tab_guard_states_extra_validation_and_collapsible_panel(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证防呆联动禁用态（auto 置灰 LR0、自定义权重置灰官方模型、Mosaic=0 置灰 CloseMosaic）、实时附加校验与增强区折叠。"""
    from yolo_studio.gui.tab_train import TrainTab

    tab = TrainTab(workspace_root=tmp_path)
    try:
        # 1) 默认 optimizer=auto 时 LR0 自动置灰并带提示标签
        assert tab.optimizer_combo.currentData() == "auto"
        assert tab.lr0_spin.isEnabled() is False
        assert "自动" in tab.lr0_label.text()

        # 切换为 AdamW 后 LR0 恢复可编辑
        adamw_idx = tab.optimizer_combo.findData("AdamW")
        tab.optimizer_combo.setCurrentIndex(adamw_idx)
        qt_app.processEvents()
        assert tab.lr0_spin.isEnabled() is True
        assert tab.lr0_label.text() == "LR0:"

        # 2) 自定义权重路径非空时置灰官方预训练模型下拉框，清空后恢复
        assert tab.model_combo.isEnabled() is True
        tab.custom_model_edit.setText(str(tmp_path / "custom.pt"))
        qt_app.processEvents()
        assert tab.model_combo.isEnabled() is False
        tab.custom_model_edit.clear()
        qt_app.processEvents()
        assert tab.model_combo.isEnabled() is True

        # 3) Mosaic 为 0 时置灰 CloseMosaic，调大后恢复
        tab.mosaic_spin.setValue(0.0)
        qt_app.processEvents()
        assert tab.close_mosaic_spin.isEnabled() is False
        tab.mosaic_spin.setValue(0.5)
        qt_app.processEvents()
        assert tab.close_mosaic_spin.isEnabled() is True

        # 4) 折叠与展开增强区并展示摘要
        assert tab.aug_body.isHidden() is False
        tab.toggle_aug_btn.click()
        qt_app.processEvents()
        assert tab.aug_body.isHidden() is True
        assert "展开" in tab.toggle_aug_btn.text()
        assert "Mosaic=0.50" in tab.aug_summary_label.text()
        tab.toggle_aug_btn.click()
        qt_app.processEvents()
        assert tab.aug_body.isHidden() is False

        # 5) 附加参数实时状态校验：合法项、覆盖提醒、语法错误
        assert tab.extra_args_status_label.isHidden() is True
        tab.extra_args_edit.setPlainText("box=7.5")
        qt_app.processEvents()
        assert tab.extra_args_status_label.isHidden() is False
        assert "已解析 1 项" in tab.extra_args_status_label.text()

        tab.extra_args_edit.setPlainText("box=7.5, mosaic=0.2")
        qt_app.processEvents()
        assert "覆盖上方界面设置: mosaic" in tab.extra_args_status_label.text()

        tab.extra_args_edit.setPlainText("broken_without_sep")
        qt_app.processEvents()
        assert "语法错误" in tab.extra_args_status_label.text()
    finally:
        tab.close()


def test_train_tab_import_export_json_reset_and_qsettings_persistence(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证训练参数 JSON 导入回填、往返导出、恢复默认，以及在异步硬件探测场景下的 QSettings 持久记忆。"""
    from PyQt6.QtCore import QSettings

    from yolo_studio.core.trainer import DeviceInfo, parse_train_arguments_text
    from yolo_studio.gui.tab_train import TrainTab

    ini_path = tmp_path / "train_settings.ini"
    settings = QSettings(str(ini_path), QSettings.Format.IniFormat)

    gpu_info = DeviceInfo(
        cuda_available=True,
        device_count=1,
        device_name="Mock RTX 3060",
        vram_gb=6.0,
        recommended_device="0",
        probing=False,
    )
    probing_info = DeviceInfo(
        cuda_available=False,
        device_count=0,
        device_name="正在后台检测计算设备...",
        vram_gb=0.0,
        recommended_device="cpu",
        probing=True,
    )

    sample_json = """
    "arguments": {
      "data": "C:\\\\other_project\\\\data.yaml",
      "epochs": 160,
      "imgsz": 960,
      "batch": 4,
      "nbs": 16,
      "device": 0,
      "workers": 0,
      "project": "C:\\\\other_project\\\\runs",
      "name": "gpu-training",
      "exist_ok": false,
      "pretrained": true,
      "optimizer": "AdamW",
      "lr0": 0.001,
      "lrf": 0.05,
      "warmup_epochs": 3.0,
      "patience": 160,
      "seed": 0,
      "deterministic": true,
      "amp": false,
      "cache": false,
      "plots": false,
      "save": true,
      "val": true,
      "mosaic": 0.0,
      "mixup": 0.0,
      "cutmix": 0.0,
      "copy_paste": 0.0,
      "degrees": 0.0,
      "shear": 0.0,
      "perspective": 0.0,
      "translate": 0.03,
      "scale": 0.15,
      "flipud": 0.0,
      "fliplr": 0.5,
      "hsv_h": 0.01,
      "hsv_s": 0.2,
      "hsv_v": 0.2,
      "box": 7.5
    },
    """

    tab1 = TrainTab(workspace_root=tmp_path, device_info=gpu_info)
    try:
        orig_yaml = tab1.yaml_picker.text()
        applied, extra_keys = tab1.import_arguments_from_text(sample_json, include_paths=False)
        assert "epochs" in applied and "optimizer" in applied
        assert extra_keys == ["box"]
        # 路径默认受保护不覆盖，且自动识别为 game_ui 预设
        assert tab1.yaml_picker.text() == orig_yaml
        assert tab1.preset_combo.currentData() == "game_ui"
        assert tab1.epochs_spin.value() == 160
        assert tab1.imgsz_combo.currentText() == "960"
        assert tab1.batch_spin.value() == 4
        assert tab1.nbs_spin.value() == 16
        assert tab1.workers_spin.value() == 0
        assert tab1.amp_chk.isChecked() is False
        assert tab1.plots_chk.isChecked() is False
        assert tab1.close_mosaic_spin.value() == 0
        assert tab1.close_mosaic_spin.isEnabled() is False

        # 导出 JSON 可无损往返解析
        exported = tab1.export_arguments_json()
        re_parsed = parse_train_arguments_text(exported)
        assert re_parsed["epochs"] == 160
        assert re_parsed["optimizer"] == "AdamW"
        assert re_parsed["box"] == 7.5

        # 折叠增强区并持久化
        tab1.set_aug_panel_expanded(False)
        tab1.save_settings(settings)
        settings.sync()
    finally:
        tab1.close()

    # 模拟下次以 lazy_init (probing=True) 启动并恢复持久化，再完成异步 GPU 探测
    tab2 = TrainTab(workspace_root=tmp_path, device_info=probing_info)
    try:
        tab2.restore_settings(settings)
        tab2.apply_device_info(gpu_info)

        assert tab2.preset_combo.currentData() == "game_ui"
        assert tab2.epochs_spin.value() == 160
        assert tab2.batch_spin.value() == 4  # 不被探测返回的默认 16 覆盖
        assert tab2.amp_chk.isChecked() is False  # 不被探测返回的默认 True 覆盖
        assert tab2.device_ctrl.device() == "0"
        assert tab2.aug_body.isHidden() is True
        assert '"box": 7.5' in tab2.extra_args_edit.toPlainText()

        # 一键恢复默认
        tab2.reset_defaults()
        assert tab2.preset_combo.currentData() == "default"
        assert tab2.epochs_spin.value() == 100
        assert tab2.batch_spin.value() == 16
        assert tab2.amp_chk.isChecked() is True
        assert tab2.lr0_spin.isEnabled() is False
        assert tab2.extra_args_edit.toPlainText() == ""
    finally:
        tab2.close()


def test_compute_device_control_across_tabs_4_5_6_and_multi_gpu_and_amp_fp16_guards(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证统一设备控件在单卡/多卡/无卡状态下的展示与交互，以及 4/5/6 页的联动防呆与梯度累积换算。"""
    from yolo_studio.core.trainer import DeviceInfo
    from yolo_studio.gui.tab_eval import EvalTab
    from yolo_studio.gui.tab_review import MllmReviewTab
    from yolo_studio.gui.tab_train import TrainTab

    single_gpu = DeviceInfo(
        cuda_available=True,
        device_count=1,
        device_name="Mock RTX 3060",
        vram_gb=6.0,
        recommended_device="0",
        gpu_devices=(("0", "Mock RTX 3060", 6.0),),
    )
    multi_gpu = DeviceInfo(
        cuda_available=True,
        device_count=2,
        device_name="Mock RTX 4090",
        vram_gb=24.0,
        recommended_device="0",
        gpu_devices=(("0", "Mock RTX 4090", 24.0), ("1", "Mock RTX 3060", 6.0)),
    )
    cpu_only = DeviceInfo(
        cuda_available=False,
        device_count=0,
        device_name="CPU",
        vram_gb=0.0,
        recommended_device="cpu",
        gpu_devices=(),
    )

    # 1) 标签页 5（训练页）：单卡隐藏多卡下拉、切 CPU 联动置灰 AMP 且切回 GPU 恢复、实时梯度累积换算、多卡可选 0,1
    train_tab = TrainTab(workspace_root=tmp_path, device_info=single_gpu)
    try:
        assert train_tab.device_ctrl.gpu_chk.isChecked() is True
        assert train_tab.device_ctrl.gpu_combo.isHidden() is True
        assert train_tab.device_ctrl.device() == "0"
        assert train_tab.amp_chk.isEnabled() is True
        assert train_tab.amp_chk.isChecked() is True

        # 梯度累积实时换算：默认 batch=16, nbs=64 -> 每 4 步；改 batch=4, nbs=16 -> 每 4 步；batch=-1 -> AutoBatch
        assert train_tab.accum_hint_label.text() == "每 4 步更新"
        train_tab.batch_spin.setValue(4)
        train_tab.nbs_spin.setValue(32)
        assert train_tab.accum_hint_label.text() == "每 8 步更新"
        train_tab.batch_spin.setValue(-1)
        assert "AutoBatch" in train_tab.accum_hint_label.text()

        # 取消勾选 GPU -> 切为 CPU，AMP 自动取消并置灰；再勾回 GPU -> 自动恢复 AMP 勾选状态
        train_tab.device_ctrl.gpu_chk.setChecked(False)
        qt_app.processEvents()
        assert train_tab.device_ctrl.device() == "cpu"
        assert train_tab.amp_chk.isEnabled() is False
        assert train_tab.amp_chk.isChecked() is False
        assert "CPU" in train_tab.hw_label.text()

        train_tab.device_ctrl.gpu_chk.setChecked(True)
        qt_app.processEvents()
        assert train_tab.device_ctrl.device() == "0"
        assert train_tab.amp_chk.isEnabled() is True
        assert train_tab.amp_chk.isChecked() is True

        # JSON 导入 device="cpu" 自动联动顶部 GPU 开关
        train_tab.import_arguments_from_text('{"device": "cpu", "epochs": 20}')
        assert train_tab.device_ctrl.gpu_chk.isChecked() is False
        assert train_tab.device_ctrl.device() == "cpu"

        # 切换为多卡硬件快照并重置：自动显现多卡下拉并支持全部多卡并行 (0,1)
        train_tab.apply_device_info(multi_gpu)
        train_tab.device_ctrl.reset_to_recommended()
        assert train_tab.device_ctrl.gpu_combo.isHidden() is False
        all_idx = train_tab.device_ctrl.gpu_combo.findData("0,1")
        assert all_idx >= 0
        train_tab.device_ctrl.gpu_combo.setCurrentIndex(all_idx)
        assert train_tab.device_ctrl.device() == "0,1"
    finally:
        train_tab.close()

    # 2) 标签页 3（LLM 智能标注页）：单卡行内紧凑开关，切至已有 .json 模式时随推理参数一并禁用；无卡环境置灰 CPU
    review_tab = MllmReviewTab(workspace_root=tmp_path, device_info=single_gpu)
    try:
        assert review_tab.yolo_device_ctrl.device() == "0"
        assert review_tab.yolo_device_ctrl.status_label.isHidden() is True
        review_tab.source_mode_combo.setCurrentIndex(1)  # EXISTING_JSON
        qt_app.processEvents()
        assert review_tab.yolo_device_ctrl.isEnabled() is False
        review_tab.source_mode_combo.setCurrentIndex(0)  # YOLO_PREDICT
        qt_app.processEvents()
        assert review_tab.yolo_device_ctrl.isEnabled() is True

        review_tab.apply_device_info(cpu_only)
        assert review_tab.yolo_device_ctrl.gpu_chk.isEnabled() is False
        assert review_tab.yolo_device_ctrl.device() == "cpu"
    finally:
        review_tab.close()

    # 3) 标签页 6（验证/推理/导出页）：全局运行设备开关 + 切 CPU 联动置灰 TorchScript FP16 导出
    eval_tab = EvalTab(workspace_root=tmp_path, device_info=single_gpu)
    try:
        assert eval_tab.device_ctrl.device() == "0"
        eval_tab.format_combo.setCurrentIndex(1)  # torchscript
        qt_app.processEvents()
        assert eval_tab.half_chk.isEnabled() is True
        eval_tab.half_chk.setChecked(True)

        eval_tab.device_ctrl.gpu_chk.setChecked(False)
        qt_app.processEvents()
        assert eval_tab.device_ctrl.device() == "cpu"
        assert eval_tab.half_chk.isEnabled() is False
        assert eval_tab.half_chk.isChecked() is False

        eval_tab.device_ctrl.gpu_chk.setChecked(True)
        qt_app.processEvents()
        assert eval_tab.half_chk.isEnabled() is True
    finally:
        eval_tab.close()


def test_review_tab_box_refine_filters_history_report_and_persistence(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证第 3 页边框修正模式联动、高置信度阈值角色切换、环境变量提示、分流筛选、历史报告加载与持久化记忆。"""
    import json

    from yolo_studio.core.mllm_reviewer import (
        BoxDecision,
        BoxRefineMode,
        CandidateBox,
        MllmProvider,
        MllmReviewConfig,
        MllmReviewResponse,
        ReviewSourceMode,
        run_mllm_review,
    )
    from yolo_studio.core.trainer import DeviceInfo
    from yolo_studio.gui.tab_review import MllmReviewTab

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ini_path = tmp_path / "review_settings.ini"
    settings = QSettings(str(ini_path), QSettings.Format.IniFormat)
    gpu_info = DeviceInfo(
        cuda_available=True,
        device_count=1,
        device_name="Mock RTX 3060",
        vram_gb=6.0,
        recommended_device="0",
        gpu_devices=(("0", "Mock RTX 3060", 6.0),),
    )

    # 先离线跑出一份包含 auto_passed / llm_fixed(含收框) / need_review 的真实审核报告
    in_dir = tmp_path / "imgs"
    out_dir = tmp_path / "reviewed"
    for name in ("a_pass.jpg", "b_tight.jpg", "c_need.jpg"):
        p = in_dir / name
        _write_sample_frame(p, (400, 200))
        payload = {
            "shapes": [
                {
                    "label": "target",
                    "score": 0.45,
                    "points": [[40.0, 20.0], [160.0, 180.0]],
                    "shape_type": "rectangle",
                }
            ],
            "imageWidth": 400,
            "imageHeight": 200,
        }
        p.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    class _StubClient:
        def __init__(self) -> None:
            self.idx = 0
            self.resps = [
                MllmReviewResponse(box_reviews=[BoxDecision(box_id=0, action="keep", box_quality="tight")]),
                MllmReviewResponse(
                    box_reviews=[
                        BoxDecision(
                            box_id=0,
                            action="keep",
                            box_quality="loose",
                            refined_box_2d=[200, 220, 820, 780],
                            reason="tighten",
                        )
                    ]
                ),
                MllmReviewResponse(
                    box_reviews=[BoxDecision(box_id=0, action="keep", box_quality="tight")],
                    has_missing=True,
                    missing_note="missed one",
                ),
            ]

        def review_image(self, *_args, **_kwargs):
            r = self.resps[self.idx]
            self.idx += 1
            return r

    run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.EXISTING_JSON,
            allowed_classes=["target"],
            max_workers=1,
            box_refine_mode=BoxRefineMode.MLLM_CROP,
        ),
        client_override=_StubClient(),
    )

    tab1 = MllmReviewTab(workspace_root=tmp_path, device_info=gpu_info)
    try:
        # 1) 默认输出目录、工作模式说明、环境变量缺失提示、目录计数与按模式自动隐藏无关置灰行
        assert Path(tab1.output_picker.text()) == (tmp_path / "data" / "smart_labeled").resolve()
        assert tab1.start_btn.text() == "开始 LLM 智能标注"
        assert "基准 .pt" in tab1.source_mode_hint_label.text()
        assert "未检测到环境变量 GEMINI_API_KEY" in tab1.api_key_status_label.text()
        assert tab1.high_conf_spin.minimumWidth() >= 100  # 防止 Win11 并排箭头截断两位小数
        assert tab1.ann_picker.isHidden() is True
        assert tab1.base_url_edit.isHidden() is True

        tab1.set_images_dir(str(in_dir))
        assert "3 张图片" in tab1.images_count_label.text()

        # 1b) 递归开关紧贴图片目录、默认勾选；取消后子目录有图而顶层无图时醒目提示
        assert tab1.recursive_chk.isChecked()
        assert tab1.recursive_chk.parentWidget() is tab1.images_picker.parentWidget()
        nested_root = tmp_path / "nested_frames"
        _write_sample_frame(nested_root / "video-001" / "f1.jpg")
        _write_sample_frame(nested_root / "video-002" / "f2.jpg")
        tab1.set_images_dir(str(nested_root))
        assert "共 2 张图片（含子目录）" in tab1.images_count_label.text()
        tab1.recursive_chk.setChecked(False)
        tab1._refresh_images_count()
        assert "请勾选「包含子目录」" in tab1.images_count_label.text()
        tab1.recursive_chk.setChecked(True)
        tab1.set_images_dir(str(in_dir))

        # 切到 EXISTING_JSON 隐藏权重与推理参数行、显示已有标注目录行并更新模式提示；切到 OpenAI 兼容接口显示 Base URL
        tab1.source_mode_combo.setCurrentIndex(1)
        compat_idx = tab1.provider_combo.findData(MllmProvider.OPENAI_COMPAT)
        assert compat_idx >= 0
        tab1.provider_combo.setCurrentIndex(compat_idx)
        qt_app.processEvents()
        assert ".json" in tab1.source_mode_hint_label.text()
        assert tab1.weights_picker.isHidden() is True
        assert tab1.yolo_params_widget.isHidden() is True
        assert tab1.ann_picker.isHidden() is False
        assert tab1.base_url_edit.isHidden() is False
        tab1.source_mode_combo.setCurrentIndex(0)
        tab1.provider_combo.setCurrentIndex(0)
        qt_app.processEvents()

        # 2) 默认开启漏标检查 + 局部裁剪 -> 阈值角色为“优先裁剪 <”；关掉裁剪 -> 禁用；关掉漏标 -> 恢复“免审阈值 ≥”
        assert "优先裁剪" in tab1.high_conf_label.text()
        tab1.crops_chk.setChecked(False)
        qt_app.processEvents()
        assert tab1.high_conf_spin.isEnabled() is False
        tab1.check_missing_chk.setChecked(False)
        qt_app.processEvents()
        assert tab1.high_conf_spin.isEnabled() is True
        assert "免审阈值" in tab1.high_conf_label.text()

        # 2b) 小批量试跑 5 张快捷按钮与上限微调框联动
        assert tab1.max_images_spin.value() == 0
        tab1.trial_5_btn.setChecked(True)
        qt_app.processEvents()
        assert tab1.max_images_spin.value() == 5
        assert tab1._build_current_config().max_images == 5
        tab1.max_images_spin.setValue(2)
        qt_app.processEvents()
        assert tab1.trial_5_btn.isChecked() is False
        assert "仅处理前 2 张" in tab1.images_count_label.text()

        # 3) 边框紧贴修正模式联动与动态人话说明：SAM_LOCAL 显示 SAM 行，MLLM_CROP 强制勾选并锁定局部裁剪
        assert tab1.sam_row_widget.isHidden() is True
        assert "保持原有框坐标不变" in tab1.refine_hint_label.text()
        sam_idx = tab1.refine_mode_combo.findData(BoxRefineMode.SAM_LOCAL)
        tab1.refine_mode_combo.setCurrentIndex(sam_idx)
        qt_app.processEvents()
        assert tab1.sam_row_widget.isHidden() is False
        assert "SAM" in tab1.refine_hint_label.text()

        crop_idx = tab1.refine_mode_combo.findData(BoxRefineMode.MLLM_CROP)
        tab1.refine_mode_combo.setCurrentIndex(crop_idx)
        qt_app.processEvents()
        assert tab1.sam_row_widget.isHidden() is True
        assert tab1.crops_chk.isChecked() is True
        assert tab1.crops_chk.isEnabled() is False
        assert "局部放大图" in tab1.refine_hint_label.text()

        # 4) 加载历史审核报告并验证分流筛选、对比预览、逐框判定理由详情与一键精修按钮
        tab1.load_history_report(out_dir / "review_report.json")
        qt_app.processEvents()
        assert tab1.result_table.rowCount() == 3
        assert "全部 (3)" in tab1.filter_buttons["all"].text()
        assert "已删改/收框 (1)" in tab1.filter_buttons["llm_fixed"].text()
        assert "待复核 (1)" in tab1.filter_buttons["need_review"].text()
        assert tab1.send_ds_btn.isEnabled() is True
        assert tab1.open_selected_btn.isEnabled() is False

        # 筛选仅看待复核：只显示第 2 行，前两行隐藏
        tab1.set_result_filter("need_review")
        assert tab1.result_table.isRowHidden(0) is True
        assert tab1.result_table.isRowHidden(1) is True
        assert tab1.result_table.isRowHidden(2) is False
        # 单击含收框记录的行（row=1），能正常渲染对比图、展示收框详情并激活“在工作台精修此图”按钮
        tab1.set_result_filter("all")
        tab1._on_row_selected(1)
        assert tab1.preview_label.pixmap() is not None and not tab1.preview_label.pixmap().isNull()
        assert "b_tight.jpg" in tab1.detail_label.text()
        assert "已收框" in tab1.detail_label.text()
        assert tab1.open_selected_btn.isEnabled() is True

        emitted: list[tuple[str, str, str]] = []
        tab1.open_in_workbench.connect(lambda d, c, f: emitted.append((d, c, f)))
        tab1.open_selected_btn.click()
        assert len(emitted) == 1
        assert emitted[0][2].endswith("b_tight.jpg")

        # 5) 持久化策略（含 max_images；手动填入的明文 API Key 不落盘）
        tab1.refine_mode_combo.setCurrentIndex(sam_idx)
        tab1.sam_refine_all_chk.setChecked(True)
        tab1.max_images_spin.setValue(5)
        tab1.classes_edit.setText("hero, boss")
        tab1.guideline_edit.setText("紧贴主体边缘")
        tab1.api_key_edit.setText("secret-key-should-not-persist")
        tab1.save_settings(settings)
        settings.sync()
    finally:
        tab1.close()

    tab2 = MllmReviewTab(workspace_root=tmp_path, device_info=gpu_info)
    try:
        tab2.restore_settings(settings)
        assert tab2.refine_mode_combo.currentData() == BoxRefineMode.SAM_LOCAL
        assert tab2.sam_refine_all_chk.isChecked() is True
        assert tab2.sam_row_widget.isHidden() is False
        assert tab2.max_images_spin.value() == 5
        assert tab2.trial_5_btn.isChecked() is True
        assert tab2.classes_edit.text() == "hero, boss"
        assert tab2.guideline_edit.text() == "紧贴主体边缘"
        assert tab2.api_key_edit.text() == ""  # 明文 API Key 绝不持久化
    finally:
        tab2.close()


def test_annotation_canvas_anchor_zoom_bounds_sharpness_and_pan_channels(
    qt_app: QApplication,
) -> None:
    """验证鼠标锚点缩放、高倍像素锐化、软边界防拖丢、多通道平移、防抖阈值与小框自适应手柄。"""
    import math
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QColor, QPixmap, QWheelEvent

    from yolo_studio.core.annotation_io import RectAnnotation
    from yolo_studio.gui.annotation_canvas import AnnotationCanvasView

    canvas = AnnotationCanvasView()
    canvas.resize(900, 600)
    canvas.show()
    qt_app.processEvents()
    try:
        pix = QPixmap(1920, 1080)
        pix.fill(QColor("#334155"))
        canvas.set_image(pix, 1920, 1080)
        qt_app.processEvents()

        # 1) 在图像右上角边缘点 (1850, 60) 连续滚轮放大 10 次，目标始终留在指针下方（误差 < 1.5px）
        target_scene = QPointF(1850.0, 60.0)
        target_vp = canvas.mapFromScene(target_scene)
        for _ in range(10):
            wev = QWheelEvent(
                QPointF(target_vp),
                QPointF(canvas.viewport().mapToGlobal(target_vp)),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            canvas.wheelEvent(wev)
            qt_app.processEvents()
        after_scene = canvas._map_view_to_scene_f(QPointF(target_vp))
        drift = math.hypot(after_scene.x() - target_scene.x(), after_scene.y() - target_scene.y())
        assert canvas.zoom_factor() >= AnnotationCanvasView.PIXEL_SHARP_ZOOM
        assert drift < 1.5
        # 高倍自动切换最近邻清晰像素渲染，缩回低倍后恢复平滑抗锯齿
        assert canvas._pixmap_item.transformationMode() == Qt.TransformationMode.FastTransformation
        canvas.set_zoom(1.0)
        qt_app.processEvents()
        assert canvas._pixmap_item.transformationMode() == Qt.TransformationMode.SmoothTransformation

        # 2) 充裕画布留白允许把右下角拖至视口中心，且过界平移受软边界保护不会拖丢图像
        canvas.set_zoom(2.0)
        qt_app.processEvents()
        canvas._set_view_center(QPointF(1920.0, 1080.0), clamp=True)
        qt_app.processEvents()
        center_corner = canvas._get_view_center_scene()
        assert center_corner.x() == pytest.approx(1920.0, abs=2.0)
        assert center_corner.y() == pytest.approx(1080.0, abs=2.0)
        canvas._set_view_center(QPointF(9999.0, 9999.0), clamp=True)
        qt_app.processEvents()
        assert canvas._get_view_center_scene().x() <= 1922.0
        assert canvas._get_view_center_scene().y() <= 1082.0

        # 3) 编辑模式拖拽空白 >=4px 平移画布且保留选中框；轻点空白 <4px 才取消选中
        box = RectAnnotation("person", 200.0, 200.0, 420.0, 420.0)
        small_box = RectAnnotation("dot", 600.0, 250.0, 616.0, 266.0)
        canvas.set_boxes([box, small_box], selected_index=0)
        canvas.set_zoom(1.0)
        canvas._set_view_center(QPointF(960.0, 540.0), clamp=True)
        qt_app.processEvents()
        before_center = canvas._get_view_center_scene()

        blank_p1 = QPoint(80, 80)
        blank_p2 = QPoint(180, 160)
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, blank_p1)
        QTest.mouseMove(canvas.viewport(), blank_p2)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, blank_p2)
        qt_app.processEvents()
        after_pan_center = canvas._get_view_center_scene()
        assert canvas.selected_index() == 0
        assert math.hypot(after_pan_center.x() - before_center.x(), after_pan_center.y() - before_center.y()) > 80.0

        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(80, 80))
        qt_app.processEvents()
        assert canvas.selected_index() == -1

        # 4) 抓手模式 (MODE_PAN) 下即使按在目标框上拖动，也是平移画布而不误移框
        canvas.select_box(0, emit_signal=False)
        canvas.set_mode(AnnotationCanvasView.MODE_PAN)
        assert canvas.panning_mode() is True
        committed: list[tuple[int, float, float, float, float]] = []
        canvas.box_geometry_committed.connect(lambda *a: committed.append(a))
        box_vp = canvas.mapFromScene(QPointF(300.0, 300.0))
        box_vp_end = QPoint(box_vp.x() + 60, box_vp.y() + 40)
        center_before_pan = canvas._get_view_center_scene()
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, box_vp)
        QTest.mouseMove(canvas.viewport(), box_vp_end)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, box_vp_end)
        qt_app.processEvents()
        assert committed == []
        assert (canvas._get_view_center_scene() - center_before_pan).manhattanLength() > 50.0

        # 5) 右键拖动平移 vs 右键轻点发上下文菜单信号
        menu_requests: list[tuple[QPoint, int]] = []
        canvas.context_menu_requested.connect(lambda gp, idx: menu_requests.append((gp, idx)))
        canvas.set_mode(AnnotationCanvasView.MODE_EDIT)
        center_before_rpan = canvas._get_view_center_scene()
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, QPoint(400, 300))
        QTest.mouseMove(canvas.viewport(), QPoint(320, 240))
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, QPoint(320, 240))
        qt_app.processEvents()
        assert (canvas._get_view_center_scene() - center_before_rpan).manhattanLength() > 50.0
        assert menu_requests == []

        box0_vp = canvas.mapFromScene(QPointF(310.0, 310.0))
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, box0_vp)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, box0_vp)
        qt_app.processEvents()
        assert len(menu_requests) == 1 and menu_requests[0][1] == 0

        # 6) 编辑模式轻点框产生 2px 手抖不触发坐标修改；极小框隐藏四边中点手柄；画框模式屏蔽手柄命中
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, box0_vp)
        QTest.mouseMove(canvas.viewport(), QPoint(box0_vp.x() + 2, box0_vp.y() + 1))
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(box0_vp.x() + 2, box0_vp.y() + 1))
        qt_app.processEvents()
        assert committed == []

        handles_small = canvas._visible_handles_for_box(small_box.x1, small_box.y1, small_box.x2, small_box.y2)
        assert set(handles_small.keys()) == {"nw", "ne", "se", "sw"}
        canvas.set_mode(AnnotationCanvasView.MODE_DRAW)
        assert canvas._hit_test_handle(QPointF(box.x1, box.y1)) is None
    finally:
        canvas.close()


def test_annotation_canvas_two_click_draw_and_edge_scroll(
    qt_app: QApplication,
) -> None:
    """验证两点点击画框（含中途滚轮缩放与中键平移）、右键/Esc 取消草稿及边缘自动滚屏。"""
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QColor, QPixmap, QWheelEvent

    from yolo_studio.gui.annotation_canvas import AnnotationCanvasView

    canvas = AnnotationCanvasView()
    canvas.resize(640, 480)
    canvas.show()
    qt_app.processEvents()
    try:
        pix = QPixmap(1600, 1200)
        pix.fill(QColor("#1e293b"))
        canvas.set_image(pix, 1600, 1200)
        canvas.set_mode(AnnotationCanvasView.MODE_DRAW)
        qt_app.processEvents()

        drawn: list[tuple[float, float, float, float]] = []
        canvas.box_drawn.connect(lambda *a: drawn.append(a))

        # 1) 两点点击画框：点击第一点进入 waiting_second，中途滚轮放大并中键平移后点击第二点完成成框
        p1_vp = canvas.mapFromScene(QPointF(200.0, 200.0))
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p1_vp)
        qt_app.processEvents()
        assert canvas._draw_state == "waiting_second"

        zoom_vp = canvas.mapFromScene(QPointF(600.0, 500.0))
        wev = QWheelEvent(
            QPointF(zoom_vp),
            QPointF(canvas.viewport().mapToGlobal(zoom_vp)),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        canvas.wheelEvent(wev)
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(320, 240))
        QTest.mouseMove(canvas.viewport(), QPoint(260, 200))
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, QPoint(260, 200))
        qt_app.processEvents()
        assert canvas._draw_state == "waiting_second"

        p2_vp = canvas.mapFromScene(QPointF(700.0, 620.0))
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p2_vp)
        qt_app.processEvents()
        assert canvas._draw_state == "idle"
        assert len(drawn) == 1
        assert drawn[0][0] == pytest.approx(200.0, abs=3.0)
        assert drawn[0][2] == pytest.approx(700.0, abs=3.0)

        # 2) 右键轻点直接取消尚未完成的两点草稿，不发射 box_drawn
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(200, 200))
        assert canvas._draw_state == "waiting_second"
        QTest.mouseClick(canvas.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier, QPoint(240, 240))
        qt_app.processEvents()
        assert canvas._draw_state == "idle"
        assert len(drawn) == 1

        # 3) 拖拽画框接近右下边缘时触发自动滚屏并扩展视野
        canvas.set_zoom(1.5)
        canvas._set_view_center(QPointF(500.0, 400.0), clamp=True)
        qt_app.processEvents()
        start_vp = QPoint(300, 240)
        edge_vp = QPoint(canvas.viewport().width() - 4, canvas.viewport().height() - 4)
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start_vp)
        QTest.mouseMove(canvas.viewport(), edge_vp)
        center_before_scroll = canvas._get_view_center_scene()
        canvas._on_edge_scroll_tick()
        qt_app.processEvents()
        center_after_scroll = canvas._get_view_center_scene()
        assert center_after_scroll.x() > center_before_scroll.x()
        assert center_after_scroll.y() > center_before_scroll.y()
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, edge_vp)
    finally:
        canvas.close()


def test_workbench_global_space_pan_nudge_tab_clipboard_keep_view_and_cache(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证全局 Space 平移与防粘滞、H/L/+/- 快捷键、列表定位、Tab 轮选、方向键微调、跨图粘贴、保持视野及图片缓存。"""
    import yolo_studio.gui.tab_labeling as tab_labeling_mod
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QFocusEvent
    from PyQt6.QtCore import QEvent

    frames_dir = tmp_path / "pro_frames"
    f1 = frames_dir / "frame_001.jpg"
    f2 = frames_dir / "frame_002.jpg"
    _write_sample_frame(f1, size=(800, 600))
    _write_sample_frame(f2, size=(800, 600))
    f1.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": "2.4.4",
                "flags": {},
                "shapes": [
                    {
                        "label": "worker",
                        "points": [[40.0, 50.0], [180.0, 220.0]],
                        "shape_type": "rectangle",
                        "flags": {},
                    },
                    {
                        "label": "helmet",
                        "points": [[620.0, 440.0], [740.0, 560.0]],
                        "shape_type": "rectangle",
                        "flags": {},
                    },
                ],
                "imagePath": f1.name,
                "imageData": None,
                "imageHeight": 600,
                "imageWidth": 800,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("worker\nhelmet\n", encoding="utf-8")

    load_calls: list[str] = []
    orig_loader = tab_labeling_mod.load_image_pixmap

    def _counting_loader(path: Path):
        load_calls.append(path.name)
        return orig_loader(path)

    monkeypatch.setattr(tab_labeling_mod, "load_image_pixmap", _counting_loader)

    win = _make_window(tmp_path)
    win.show()
    QTest.qWaitForWindowExposed(win)
    win.activateWindow()
    qt_app.processEvents()
    try:
        tab = win.labeling_tab
        win.tabs.setCurrentWidget(tab)
        assert tab.open_directory(frames_dir, classes_file=classes_file) is True
        qt_app.processEvents()
        assert load_calls == ["frame_001.jpg"]

        # 1) 即使焦点在左侧文件列表，按住 Space + 左键仍可拖动画布，且失焦自动复位防粘滞
        tab.canvas.set_zoom(2.0)
        tab.canvas._set_view_center(QPointF(400.0, 300.0), clamp=True)
        tab.file_list.setFocus()
        qt_app.processEvents()
        QTest.keyPress(tab.file_list, Qt.Key.Key_Space)
        qt_app.processEvents()
        assert tab.canvas._space_pressed is True
        c_before = tab.canvas._get_view_center_scene()
        QTest.mousePress(tab.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(300, 220))
        QTest.mouseMove(tab.canvas.viewport(), QPoint(220, 160))
        QTest.mouseRelease(tab.canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(220, 160))
        qt_app.processEvents()
        assert (tab.canvas._get_view_center_scene() - c_before).manhattanLength() > 30.0
        tab.canvas.focusOutEvent(QFocusEvent(QEvent.Type.FocusOut))
        assert tab.canvas._space_pressed is False

        # 2) H 抓手模式、L 显隐标签、+/- 缩放与 0 适应窗口
        QTest.keyClick(tab.canvas, Qt.Key.Key_H)
        assert tab.canvas.panning_mode() is True
        assert tab.pan_mode_btn.isChecked() is True
        QTest.keyClick(tab.canvas, Qt.Key.Key_E)
        assert tab.canvas.editing() is True

        QTest.keyClick(tab.canvas, Qt.Key.Key_L)
        assert tab.show_labels_chk.isChecked() is False
        QTest.keyClick(tab.canvas, Qt.Key.Key_L)
        assert tab.show_labels_chk.isChecked() is True

        z0 = tab.canvas.zoom_factor()
        QTest.keyClick(tab.canvas, Qt.Key.Key_Equal)
        assert tab.canvas.zoom_factor() > z0
        QTest.keyClick(tab.canvas, Qt.Key.Key_Minus)
        assert tab.canvas.zoom_factor() == pytest.approx(z0, abs=0.05)

        # 3) 放大后点击右侧目标列表第二项，画布自动将该远端框移入视口中心安全区
        tab.canvas.set_zoom(2.5)
        tab.canvas._set_view_center(QPointF(100.0, 120.0), clamp=True)
        qt_app.processEvents()
        tab.shape_list.setCurrentRow(1)
        qt_app.processEvents()
        vis_center = tab.canvas._get_view_center_scene()
        assert vis_center.x() == pytest.approx(680.0, abs=15.0)
        assert vis_center.y() == pytest.approx(500.0, abs=15.0)

        # 4) Tab / Shift+Tab 轮选框与方向键 1px / Shift+方向键 10px 微调（连续微调合并撤回）
        tab.canvas.setFocus()
        qt_app.processEvents()
        QTest.keyClick(tab.canvas, Qt.Key.Key_Tab)
        qt_app.processEvents()
        assert tab.canvas.selected_index() == 0
        assert tab._doc is not None
        orig_x1 = tab._doc.boxes[0].x1
        QTest.keyClick(tab.canvas, Qt.Key.Key_Right)
        assert tab._doc.boxes[0].x1 == pytest.approx(orig_x1 + 1.0)
        QTest.keyClick(tab.canvas, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        assert tab._doc.boxes[0].x1 == pytest.approx(orig_x1 + 11.0)
        tab._undo()
        assert tab._doc.boxes[0].x1 == pytest.approx(orig_x1)

        # 5) 选中第 0 个框 Ctrl+C 复制，按 D 切下一张，Ctrl+V 原位跨图粘贴
        tab.canvas.select_box(0, emit_signal=True)
        tab._copy_boxes_to_clipboard()
        QTest.keyClick(tab.canvas, Qt.Key.Key_D)
        qt_app.processEvents()
        assert tab.current_image_path() == f2.resolve()
        assert load_calls == ["frame_001.jpg", "frame_002.jpg"]
        tab._paste_boxes_from_clipboard()
        assert len(tab._doc.boxes) == 1
        assert tab._doc.boxes[0].label == "worker"
        assert tab._doc.boxes[0].x1 == pytest.approx(orig_x1)

        # 6) 开启「保持视野」切回第 1 张：保留缩放与观察中心，且命中图片缓存不重复解码
        tab.keep_view_chk.setChecked(True)
        tab.canvas.set_zoom(2.2)
        tab.canvas._set_view_center(QPointF(620.0, 460.0), clamp=True)
        qt_app.processEvents()
        QTest.keyClick(tab.canvas, Qt.Key.Key_A)
        qt_app.processEvents()
        assert tab.current_image_path() == f1.resolve()
        assert tab.canvas.zoom_factor() == pytest.approx(2.2, abs=0.05)
        assert tab.canvas._get_view_center_scene().x() == pytest.approx(620.0, abs=3.0)
        assert tab.canvas._get_view_center_scene().y() == pytest.approx(460.0, abs=3.0)
        assert load_calls == ["frame_001.jpg", "frame_002.jpg"]
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_workbench_two_column_layout_file_list_jump_class_mgmt_and_hud(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证单行顶栏、两栏大画布与三段右栏布局、文件列表省略/进度条/跳页/下未标、按需详情卡、类别重命名移除及轻量 HUD。"""
    frames_dir = tmp_path / "batch_frames"
    imgs = [
        frames_dir / ("vcd411ba60735d5c3ab3620b8e77c507be_f0000000000.png"),
        frames_dir / ("vcd411ba60735d5c3ab3620b8e77c507be_f0000000060.png"),
        frames_dir / ("vcd411ba60735d5c3ab3620b8e77c507be_f0000000120.png"),
    ]
    for p in imgs:
        _write_sample_frame(p, size=(1280, 720))

    # 第 1 张已标注（含框），第 2 张显式空图负样本，第 3 张未标注
    imgs[0].with_suffix(".json").write_text(
        json.dumps(
            {
                "version": "2.4.4",
                "flags": {"verified": True},
                "shapes": [
                    {
                        "label": "hero",
                        "points": [[100.0, 120.0], [260.0, 320.0]],
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                ],
                "imagePath": imgs[0].name,
                "imageData": None,
                "imageHeight": 720,
                "imageWidth": 1280,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    imgs[1].with_suffix(".json").write_text(
        json.dumps(
            {
                "version": "2.4.4",
                "flags": {},
                "shapes": [],
                "imagePath": imgs[1].name,
                "imageData": None,
                "imageHeight": 720,
                "imageWidth": 1280,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    win = _make_window(tmp_path)
    win.show()
    qt_app.processEvents()
    try:
        tab = win.labeling_tab
        win.tabs.setCurrentWidget(tab)
        qt_app.processEvents()

        # 1) 两栏主布局 + 右栏三段垂直分割 + 原底栏缩放控制条完整保留 + 未选框时详情卡隐藏
        assert tab.main_splitter.count() == 2
        assert tab.right_splitter.count() == 3
        assert win.minimumSizeHint().width() <= 1380
        assert tab.fit_btn.isVisible() is True
        assert tab.zoom_100_btn.isVisible() is True
        assert tab.zoom_menu_btn.isVisible() is True
        assert tab.hint_info_label.isVisible() is True
        assert tab.box_detail_panel.isHidden() is True

        # 2) 打开目录后：文件列表启用中部省略、无横向滚动条、状态圆点与尾部框数、进度条更新
        assert tab.open_directory(frames_dir) is True
        qt_app.processEvents()
        assert tab.file_list.textElideMode() == Qt.TextElideMode.ElideMiddle
        assert tab.file_list.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        assert tab.file_list.count() == 3
        assert tab.file_list.item(0).text().endswith("[1]")
        assert tab.file_list.item(1).text().endswith("[0]")
        assert not tab.file_list.item(0).icon().isNull()
        assert tab.progress_bar.maximum() == 3
        assert tab.progress_bar.value() == 2  # 已标(1) + 空图(1)

        # 3) 「下未标」自动跳过已标第1张和空图第2张，直达未标第3张
        assert tab.current_image_path() == imgs[0].resolve()
        assert tab.open_next_unlabeled_image() is True
        qt_app.processEvents()
        assert tab.current_image_path() == imgs[2].resolve()

        # 4) 页码输入「1」回车跳回第 1 张，不触发数字键 1 改类且焦点自动交还画布
        tab.jump_index_edit.setFocus()
        QTest.keyClicks(tab.jump_index_edit, "1")
        QTest.keyClick(tab.jump_index_edit, Qt.Key.Key_Return)
        qt_app.processEvents()
        assert tab.current_image_path() == imgs[0].resolve()
        assert tab.jump_index_edit.text() == ""
        assert qt_app.focusWidget() is tab.canvas

        # 5) 选中框后详情卡按需展开；重命名类别同步更新当前图框且支持 Ctrl+Z 撤回；支持移除类别
        tab.canvas.select_box(0, emit_signal=True)
        qt_app.processEvents()
        assert tab.box_detail_panel.isHidden() is False
        assert tab._rename_class("hero", "player") is True
        assert tab._doc is not None and tab._doc.boxes[0].label == "player"
        assert "player" in tab._classes
        tab._undo()
        assert tab._doc.boxes[0].label == "hero"

        tab._merge_classes(["unused_cls"], replace_order=False)
        assert "unused_cls" in tab._classes
        assert tab._remove_class_from_list("unused_cls") is True
        assert "unused_cls" not in tab._classes

        # 6) 模式切换触发画布轻量 HUD 悬浮提示，两点画框提示可正常渲染而不异常
        QTest.keyClick(tab.canvas, Qt.Key.Key_R)
        qt_app.processEvents()
        assert "画框模式" in tab.canvas._hud_message
        tab.canvas.show_hud_message("")
        assert tab.canvas._hud_message == ""
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_annotation_canvas_hidpi_physical_zoom_and_anchor(
    qt_app: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 150% 高 DPI 下物理缩放倍率、策略 A 适应窗口、1:1 点对点与滚轮锚点精确对齐。"""
    import math
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QColor, QPixmap, QWheelEvent

    from yolo_studio.gui.annotation_canvas import AnnotationCanvasView

    canvas = AnnotationCanvasView()
    canvas.resize(1024, 725)
    canvas.show()
    qt_app.processEvents()
    try:
        monkeypatch.setattr(canvas, "_display_dpr", lambda: 1.5)
        pix = QPixmap(1368, 800)
        pix.fill(QColor("#334155"))
        canvas.set_image(pix, 1368, 800)
        qt_app.processEvents()

        # 1) 策略 A：适应窗口尽量填满逻辑画布，并如实显示物理缩放倍率 (~110%)
        expected_logical_fit = (canvas.viewport().width() - 24) / 1368.0
        assert canvas._logical_scale() == pytest.approx(expected_logical_fit, abs=0.01)
        assert canvas.zoom_factor() == pytest.approx(expected_logical_fit * 1.5, abs=0.02)
        assert canvas.zoom_factor() > 1.0

        # 2) 点击 1:1 (set_zoom(1.0)) 时，物理倍率为 100%，底层逻辑倍率为 1/1.5，做到真正点对点 1:1
        canvas.set_zoom(1.0)
        qt_app.processEvents()
        assert canvas.zoom_factor() == pytest.approx(1.0, abs=1e-4)
        assert canvas._logical_scale() == pytest.approx(1.0 / 1.5, abs=1e-4)
        assert canvas._pixmap_item.transformationMode() == Qt.TransformationMode.SmoothTransformation

        # 3) 物理 200% 即触发最近邻锐利像素渲染
        canvas.set_zoom(2.0)
        qt_app.processEvents()
        assert canvas.zoom_factor() == pytest.approx(2.0, abs=1e-4)
        assert canvas._logical_scale() == pytest.approx(2.0 / 1.5, abs=1e-4)
        assert canvas._pixmap_item.transformationMode() == Qt.TransformationMode.FastTransformation

        # 4) 150% DPI 下滚轮在边缘点 (1280, 720) 连续缩放 8 次，锚点保持稳定不飘移
        canvas.fit_to_window()
        qt_app.processEvents()
        target_scene = QPointF(1280.0, 720.0)
        target_vp = canvas.mapFromScene(target_scene)
        for _ in range(8):
            wev = QWheelEvent(
                QPointF(target_vp),
                QPointF(canvas.viewport().mapToGlobal(target_vp)),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
            canvas.wheelEvent(wev)
            qt_app.processEvents()
        after_scene = canvas._map_view_to_scene_f(QPointF(target_vp))
        drift = math.hypot(after_scene.x() - target_scene.x(), after_scene.y() - target_scene.y())
        assert drift < 1.5
    finally:
        canvas.close()


def test_mainwindow_uses_title_bar_status_without_bottom_statusbar(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证主窗口取消底部状态栏，并将计算设备就绪等系统提示展示到窗口标题栏上。"""
    from PyQt6.QtWidgets import QStatusBar

    from yolo_studio.core.trainer import DeviceInfo

    win = _make_window(tmp_path, lazy_init=True)
    win.show()
    qt_app.processEvents()
    try:
        assert win.findChild(QStatusBar) is None
        assert win.windowTitle() == f"YoloStudio v{__version__}"

        gpu_info = DeviceInfo(
            cuda_available=True,
            device_count=1,
            device_name="Mock RTX 4070",
            vram_gb=12.0,
            recommended_device="0",
            probing=False,
        )
        win._on_device_probed(gpu_info)
        qt_app.processEvents()
        assert win.findChild(QStatusBar) is None
        assert "Mock RTX 4070" in win.windowTitle()
        assert f"YoloStudio v{__version__}" in win.windowTitle()

        win._clear_title_status()
        assert win.windowTitle() == f"YoloStudio v{__version__}"
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()
