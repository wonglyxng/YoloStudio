"""PyQt6 单进程内嵌 X-AnyLabeling 工作台与跨页联动集成测试。

职责：
    在离线 `offscreen` Qt 环境下验证主窗口六标签页装配、X-AnyLabeling 内嵌宿主初始化与桥接适配层、
    快捷键作用域隔离、`open_directory` 精确目录/单图/类别加载、跨标签页联动跳转、
    跨页 GPU 任务互斥与 ONNX 导出 FP16 置灰、异步视频探测、MLLM 对比图缓存，
    以及关闭主窗口时等待后台线程安全停止后再退出。

运行环境与协作边界：
    通过 `pytest` 运行，依赖 `PyQt6`、`PyQt6-WebEngine`、`Pillow` 与本地安装的
    `third_party/X-AnyLabeling`（`anylabeling` 包）；不访问外部网络，也不弹出可见桌面窗口
    （关闭确认对话框在测试中被替换为直接返回“是”）。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from threading import Event

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XAL_ROOT = PROJECT_ROOT / "third_party" / "X-AnyLabeling"
if XAL_ROOT.is_dir() and str(XAL_ROOT) not in sys.path:
    sys.path.insert(0, str(XAL_ROOT))

from PyQt6.QtCore import QCoreApplication, Qt, QThread  # noqa: E402
from PyQt6.QtGui import QAction  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from yolo_studio.gui.main_window import MainWindow  # noqa: E402
from yolo_studio.gui.tab_labeling import (  # noqa: E402
    ensure_cuda_dll_search_paths,
    ensure_xanylabeling_runtime_initialized,
)


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """初始化模块级无头 `QApplication` 与 X-AnyLabeling 运行时资源。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    ensure_cuda_dll_search_paths()
    existing = QApplication.instance()
    if existing is None:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication([])
    else:
        app = existing
    ensure_xanylabeling_runtime_initialized(app)
    return app


def _write_sample_frame(path: Path, size: tuple[int, int] = (160, 120)) -> None:
    """在临时目录生成用于测试加载的 JPEG 图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(60, 130, 180)).save(path)


def test_mainwindow_mounts_embedded_xanylabeling_and_scopes_shortcuts(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证主窗口装配 6 个标签页，且内嵌 X-AnyLabeling 快捷键随可见态切换 WindowShortcut / WidgetShortcut。"""
    win = MainWindow(workspace_root=tmp_path)
    win.show()
    qt_app.processEvents()
    try:
        assert win.tabs.count() == 6
        assert win.tabs.tabText(1) == "2. 标注工作台 (X-AnyLabeling)"
        assert win.labeling_tab._xal_window is not None

        # 初始位于第 1 页（视频抽帧），工作台不可见，快捷键处于收起隔离态
        actions = [a for a in win.labeling_tab._xal_window.findChildren(QAction) if a.shortcuts()]
        assert len(actions) > 10
        assert all(
            action.shortcutContext() == Qt.ShortcutContext.WidgetShortcut
            for action in actions
        )

        # 切入第 2 页（标注工作台）后激活原生 WindowShortcut
        win.tabs.setCurrentIndex(1)
        qt_app.processEvents()
        assert all(
            action.shortcutContext() == Qt.ShortcutContext.WindowShortcut
            for action in actions
        )

        # 切离第 2 页后再次收起为 WidgetShortcut
        win.tabs.setCurrentIndex(2)
        qt_app.processEvents()
        assert all(
            action.shortcutContext() == Qt.ShortcutContext.WidgetShortcut
            for action in actions
        )
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_embedded_workbench_keyboard_shortcuts_and_tab_isolation(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """用真实按键验证内嵌工作台 R/A/D 快捷键生效、搜索框打字不误触、切至其他标签页不串键。"""
    frames_dir = tmp_path / "shortcut_frames"
    _write_sample_frame(frames_dir / "f1.jpg")
    _write_sample_frame(frames_dir / "f2.jpg")

    win = MainWindow(workspace_root=tmp_path)
    win.show()
    QTest.qWaitForWindowExposed(win)
    win.activateWindow()
    qt_app.processEvents()
    try:
        win.dataset_tab.set_frames_dir(str(frames_dir))
        win.dataset_tab._on_open_in_workbench()
        qt_app.processEvents()

        assert win.tabs.currentIndex() == 1
        lw = win.labeling_tab._label_widget()
        assert lw is not None
        assert qt_app.focusWidget() is lw.canvas
        assert Path(str(lw.filename)).name == "f1.jpg"

        # 1) 工作台内按 D 切下一张、按 A 切回上一张、按 R 进入画矩形框模式
        QTest.keyClick(lw.canvas, Qt.Key.Key_D)
        qt_app.processEvents()
        assert Path(str(lw.filename)).name == "f2.jpg"

        QTest.keyClick(lw.canvas, Qt.Key.Key_A)
        qt_app.processEvents()
        assert Path(str(lw.filename)).name == "f1.jpg"

        QTest.keyClick(lw.canvas, Qt.Key.Key_R)
        qt_app.processEvents()
        assert lw.canvas.drawing() is True
        assert lw.canvas.create_mode == "rectangle"

        # 2) 聚焦工作台内文件搜索框打字 d、r 时走文本输入，不触发翻页或模式切换
        lw.set_edit_mode()
        lw.file_search.setFocus()
        qt_app.processEvents()
        QTest.keyClick(lw.file_search, Qt.Key.Key_D)
        QTest.keyClick(lw.file_search, Qt.Key.Key_R)
        qt_app.processEvents()
        assert lw.file_search.text() == "dr"
        assert lw.canvas.editing() is True
        lw.file_search.clear()
        qt_app.processEvents()

        # 3) 切回第 1 页（视频抽帧）后按 D、R 不影响工作台状态
        win.tabs.setCurrentIndex(0)
        win.extract_tab.start_btn.setFocus()
        qt_app.processEvents()
        QTest.keyClick(win.extract_tab.start_btn, Qt.Key.Key_D)
        QTest.keyClick(win.extract_tab.start_btn, Qt.Key.Key_R)
        qt_app.processEvents()
        assert Path(str(lw.filename)).name == "f1.jpg"
        assert lw.canvas.editing() is True
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


def test_embedded_workbench_open_directory_classes_and_cross_tab_routing(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证 `open_directory`、`classes.txt` 预载、单图定位及数据集/纠错页双向联动信号。"""
    frames_dir = tmp_path / "frames"
    img_a = frames_dir / "frame_001.jpg"
    img_b = frames_dir / "frame_002.jpg"
    _write_sample_frame(img_a)
    _write_sample_frame(img_b)

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("person\nhelmet\n", encoding="utf-8")

    win = MainWindow(workspace_root=tmp_path)
    try:
        # 从「3. 数据集准备」触发跳转至「2. 标注工作台」并定位第二张图
        win.tabs.setCurrentIndex(2)
        win.review_tab.open_in_workbench.emit(
            str(frames_dir), str(classes_file), str(img_b)
        )

        assert win.tabs.currentIndex() == 1
        assert win.labeling_tab.current_directory() == frames_dir.resolve()

        lw = win.labeling_tab._label_widget()
        assert lw is not None
        assert lw.unique_label_list.count() == 2
        assert Path(str(lw.filename)).resolve() == img_b.resolve()

        # 从「2. 标注工作台」发回「3. 数据集准备」与「4. MLLM 标注纠错」
        win.labeling_tab.send_to_dataset.emit(str(frames_dir), str(classes_file))
        assert win.tabs.currentIndex() == 2
        assert Path(win.dataset_tab.images_picker.text()).resolve() == frames_dir.resolve()
        assert Path(win.dataset_tab.classes_picker.text()).resolve() == classes_file.resolve()

        win.labeling_tab.send_to_review.emit(str(frames_dir))
        assert win.tabs.currentIndex() == 3
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
    win = MainWindow(workspace_root=tmp_path)
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


def test_xal_label_bridge_and_defensive_fallback(qt_app: QApplication, tmp_path: Path) -> None:
    """验证 X-AnyLabeling 类别桥接函数同步写入配置、唯一标签列表与弹窗历史词库，且对不兼容对象安全返回 False。"""
    from yolo_studio.gui.tab_labeling import _apply_labels_to_xal_view

    assert _apply_labels_to_xal_view(object(), ["person"]) is False

    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("worker\nvest\ngoggles\n", encoding="utf-8")

    win = MainWindow(workspace_root=tmp_path)
    try:
        count = win.labeling_tab.load_classes_file(classes_file)
        assert count == 3
        lw = win.labeling_tab._label_widget()
        assert lw is not None
        assert lw._config["labels"] == ["worker", "vest", "goggles"]
        assert lw.unique_label_list.count() == 3
    finally:
        assert win.labeling_tab.prepare_for_close() is True
        win.close()


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

    win = MainWindow(workspace_root=tmp_path)
    worker = _CancellableThread()
    try:
        # 三个标签页共享同一份硬件快照与互斥协调器
        assert win.train_tab.device_info is win.device_info
        assert win.review_tab.device_info is win.device_info
        assert win.eval_tab.device_info is win.device_info
        assert win.train_tab.gpu_guard is win.gpu_guard

        # 默认导出格式为 ONNX：FP16 复选框应自动置灰且未勾选；切换至 TorchScript 后启用
        assert win.eval_tab.format_combo.currentData() == "onnx"
        assert not win.eval_tab.half_chk.isEnabled()
        assert not win.eval_tab.half_chk.isChecked()
        win.eval_tab.format_combo.setCurrentIndex(1)  # torchscript
        assert win.eval_tab.half_chk.isEnabled()
        win.eval_tab.half_chk.setChecked(True)
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

    win = MainWindow(workspace_root=tmp_path)
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


def test_lazy_init_fast_startup_and_async_device_probe_and_on_demand_mount(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证 `lazy_init=True` 首屏不挂载工作台、异步探测无损刷新设备状态，且跨页跳转时自动按需挂载。"""
    frames_dir = tmp_path / "lazy_frames"
    _write_sample_frame(frames_dir / "f001.jpg")

    win = MainWindow(workspace_root=tmp_path, lazy_init=True)
    try:
        # 1) 首屏秒开阶段：内嵌工作台处于未挂载占位态，设备处于后台探测态
        assert win.labeling_tab._xal_window is None
        assert win.device_info.probing is True
        assert "正在后台检测" in win.train_tab.hw_label.text()

        # 模拟用户在探测返回前手动改写了第 6 页的设备输入框
        win.eval_tab.device_edit.setText("custom:1")

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
        assert "正在后台检测" not in win.train_tab.hw_label.text()
        assert win.train_tab.device_edit.text() == win.device_info.recommended_device
        # 用户手动修改过的设备值不被后台探测覆盖
        assert win.eval_tab.device_edit.text() == "custom:1"

        # 3) 从第 1 页发送抽帧目录时，第 2 页自动按需挂载内嵌 X-AnyLabeling 并打开目录
        assert win.tabs.currentIndex() == 0
        win.extract_tab.send_frames_dir.emit(str(frames_dir))
        assert win.labeling_tab._xal_window is not None
        assert win.tabs.currentIndex() == 1
        assert win.labeling_tab.current_directory() == frames_dir.resolve()
    finally:
        if win._device_probe_worker and win._device_probe_worker.isRunning():
            win._device_probe_worker.wait(3000)
        assert win.labeling_tab.prepare_for_close() is True
        win.close()
