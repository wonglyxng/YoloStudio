"""PyQt6 桌面主窗口模块，统筹六步工作流标签页及跨页数据流转。

职责：
    组装“1. 视频抽帧 → 2. 标注工作台 → 3. LLM 标注清洗 → 4. 数据集准备 →
    5. YOLO26 训练 → 6. 验证、推理与导出”六个标签页，协调内嵌工作台联动与跨页路径同步。

运行环境与协作边界：
    需 PyQt6 主事件循环与已创建的 `QApplication`；由 `main.py` 创建并显示，
    内部持有六个标签页实例及 `QSettings("YOLO26Studio", "VideoToYOLO26")` 持久化对象。
    退出窗口时先经 `LabelingStudioTab.prepare_for_close()` 确认未保存标注修改，未放行则阻止关闭；
    仍有后台任务运行时请求其安全停止，并在全部线程结束后才真正关闭。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, QThread, QTimer
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QMessageBox, QTabWidget, QVBoxLayout, QWidget

from yolo_studio._version import __version__
from yolo_studio.core.trainer import DeviceInfo, detect_compute_device
from yolo_studio.gui.common import GpuTaskGuard
from yolo_studio.gui.tab_dataset import DatasetTab
from yolo_studio.gui.tab_eval import EvalTab
from yolo_studio.gui.tab_extract import ExtractTab
from yolo_studio.gui.tab_labeling import LabelingStudioTab
from yolo_studio.gui.tab_review import MllmReviewTab
from yolo_studio.gui.tab_train import TrainTab
from yolo_studio.gui.workers import DeviceProbeWorker

APP_STYLESHEET = """
QMainWindow, QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}
QGroupBox {
    font-weight: 600;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    padding: 6px 12px;
    border-radius: 5px;
    border: 1px solid #c6cdd5;
    background-color: #f6f8fa;
}
QPushButton:hover {
    background-color: #eaeef2;
}
QPushButton:disabled {
    color: #999999;
    background-color: #f2f2f2;
}
QPushButton#primaryBtn {
    background-color: #0969da;
    color: #ffffff;
    border: 1px solid #085ec4;
    font-weight: 600;
}
QPushButton#primaryBtn:hover {
    background-color: #085ec4;
}
QPushButton#primaryBtn:disabled {
    background-color: #8bb8ee;
    border-color: #8bb8ee;
}
QWidget#metricCard {
    border: 1px solid #d8dee4;
    border-radius: 6px;
    background-color: #f8fafc;
}
"""


class MainWindow(QMainWindow):
    """YOLO26 训练工具桌面顶层主窗口（单进程整合内嵌 X-AnyLabeling）。

    状态与生命周期：
        持有 `workspace_root`、`QSettings` 及六个工作流标签页；
        关闭时先委派内嵌 `LabelingStudioTab` 核查未保存标注再持久化主窗口状态。
    """

    def __init__(
        self,
        workspace_root: Path,
        lazy_init: bool = False,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self._lazy_init = lazy_init
        self._device_probe_worker: DeviceProbeWorker | None = None
        self.settings = settings or QSettings("YOLO26Studio", "VideoToYOLO26")
        self.setWindowTitle(f"YoloStudio v{__version__}")
        self.resize(1380, 860)
        self.setStyleSheet(APP_STYLESHEET)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        if lazy_init:
            self.device_info = DeviceInfo(
                cuda_available=False,
                device_count=0,
                device_name="正在后台检测计算设备...",
                vram_gb=0.0,
                recommended_device="cpu",
                probing=True,
            )
        else:
            self.device_info = detect_compute_device()
        self.gpu_guard = GpuTaskGuard()

        self.tabs = QTabWidget()
        self.extract_tab = ExtractTab(workspace_root)
        self.labeling_tab = LabelingStudioTab(workspace_root, lazy_mount=lazy_init)
        self.dataset_tab = DatasetTab(workspace_root)
        self.review_tab = MllmReviewTab(
            workspace_root, device_info=self.device_info, gpu_guard=self.gpu_guard
        )
        self.train_tab = TrainTab(
            workspace_root, device_info=self.device_info, gpu_guard=self.gpu_guard
        )
        self.eval_tab = EvalTab(
            workspace_root, device_info=self.device_info, gpu_guard=self.gpu_guard
        )

        self.tabs.addTab(self.extract_tab, "1. 视频抽帧")
        self.tabs.addTab(self.labeling_tab, "2. 标注工作台")
        # 标签页按“采集 → 标注 → 质检 → 组装 → 训练 → 评估”排列；页面跳转一律按控件对象切换，
        # 不依赖序号，调整顺序时只需修改此处与界面文案中的编号
        self.tabs.addTab(self.review_tab, "3. LLM 标注清洗")
        self.tabs.addTab(self.dataset_tab, "4. 数据集准备")
        self.tabs.addTab(self.train_tab, "5. YOLO26 训练")
        self.tabs.addTab(self.eval_tab, "6. 验证、推理与导出")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        # 跨标签页工作流信号连接
        self.extract_tab.send_frames_dir.connect(self._on_frames_ready)
        self.labeling_tab.send_to_dataset.connect(self._on_labeling_to_dataset)
        self.labeling_tab.send_to_review.connect(self._on_labeling_to_review)
        self.labeling_tab.annotations_changed.connect(self._on_annotations_changed)
        self.review_tab.open_in_workbench.connect(self._on_open_in_workbench)
        self.review_tab.send_cleaned_dataset.connect(self._on_cleaned_ready)
        self.dataset_tab.send_data_yaml.connect(self._on_yaml_ready)
        self.dataset_tab.request_train_tab.connect(
            lambda: self.tabs.setCurrentWidget(self.train_tab)
        )
        self.train_tab.send_best_weights.connect(self._on_weights_ready)
        self.train_tab.request_eval_tab.connect(
            lambda: self.tabs.setCurrentWidget(self.eval_tab)
        )

        self._restore_settings()

        # 关闭时等待后台线程结束的轮询定时器（仅在用户确认停止任务或等待后台初始化线程收尾时启动）
        self._close_wait_timer = QTimer(self)
        self._close_wait_timer.setInterval(200)
        self._close_wait_timer.timeout.connect(self._poll_workers_then_close)

    def start_background_warmup(self) -> None:
        """在主窗口显示上屏后幂等启动后台异步硬件探测（加载 `torch` 并刷新 CUDA/CPU 状态）。"""
        if not self.device_info.probing:
            return
        if self._device_probe_worker is not None and self._device_probe_worker.isRunning():
            return
        self._device_probe_worker = DeviceProbeWorker()
        self._device_probe_worker.finished_info.connect(self._on_device_probed)
        self._device_probe_worker.start()

    def _on_device_probed(self, info: DeviceInfo) -> None:
        """接收后台硬件探测结果，原地更新共享 `device_info` 并刷新第 3/5/6 标签页控件。"""
        self.device_info.cuda_available = info.cuda_available
        self.device_info.device_count = info.device_count
        self.device_info.device_name = info.device_name
        self.device_info.vram_gb = info.vram_gb
        self.device_info.recommended_device = info.recommended_device
        self.device_info.gpu_devices = info.gpu_devices
        self.device_info.probing = False

        self.review_tab.apply_device_info(self.device_info)
        self.train_tab.apply_device_info(self.device_info)
        self.eval_tab.apply_device_info(self.device_info)

        summary = (
            f"计算设备就绪: {info.device_name} ({info.vram_gb:.1f} GB)"
            if info.cuda_available
            else "计算设备就绪: CPU 模式"
        )
        self.statusBar().showMessage(summary, 5000)

    def _on_frames_ready(self, frames_dir: str) -> None:
        """同步抽帧输出目录至数据集页与纠错页；用户仍停留在抽帧页时自动切至内嵌标注工作台。

        抽帧完成与点击「发送到标注工作台」都会触发此槽；两种情况下用户通常都处于抽帧页。
        """
        self.dataset_tab.set_frames_dir(frames_dir)
        self.review_tab.set_images_dir(frames_dir)
        if self.tabs.currentWidget() is self.extract_tab:
            self._on_open_in_workbench(frames_dir, "", "")

    def _on_open_in_workbench(
        self, target_dir: str, classes_file: str = "", select_image: str = ""
    ) -> None:
        """切换至「2. 标注工作台」并在内嵌 X-AnyLabeling 中打开目标目录、类别与可选指定图片。"""
        cls_path = Path(classes_file) if classes_file else None
        img_path = Path(select_image) if select_image else None
        if self.labeling_tab.open_directory(
            Path(target_dir), classes_file=cls_path, select_image=img_path
        ):
            self.tabs.setCurrentWidget(self.labeling_tab)

    def _on_labeling_to_dataset(self, images_dir: str, classes_file: str) -> None:
        """将工作台当前标注目录与类别清单同步至「4. 数据集准备」并切换过去。

        内嵌 X-AnyLabeling 打开目录时会递归列出子目录中的图片，因此同步时开启递归扫描，
        保证数据集页统计与生成的图片范围和工作台中标注的范围一致。
        """
        self.dataset_tab.set_dataset_source(images_dir, classes_file, recursive=True)
        self.tabs.setCurrentWidget(self.dataset_tab)

    def _on_labeling_to_review(self, images_dir: str) -> None:
        """将工作台当前图片目录同步至「3. LLM 标注清洗」并切换过去。"""
        self.review_tab.set_images_dir(images_dir)
        self.tabs.setCurrentWidget(self.review_tab)

    def _on_annotations_changed(self, dir_path: str) -> None:
        """工作台内保存或更新 `.json` 后通知数据集页；由数据集页判断是否同目录并标记统计结果过期。"""
        self.dataset_tab.mark_annotations_changed(dir_path)

    def _on_cleaned_ready(self, images_dir: str, classes_txt: str, recursive: bool) -> None:
        """将 MLLM 清洗输出目录与类别清单填入「4. 数据集准备」并切换至该页。"""
        self.dataset_tab.set_dataset_source(images_dir, classes_txt, recursive)
        self.tabs.setCurrentWidget(self.dataset_tab)

    def _on_yaml_ready(self, yaml_path: str) -> None:
        """将生成的 `data.yaml` 路径同步至「5. YOLO26 训练」。"""
        self.train_tab.set_data_yaml(yaml_path)

    def _on_weights_ready(self, weights_path: str, yaml_path: str) -> None:
        """将训练产出的 `best.pt` 权重与 `data.yaml` 同步至「6. 验证、推理与导出」和「3. LLM 标注清洗」。

        仅同步路径不主动切换标签页；用户点击训练页发送按钮时由 `TrainTab.request_eval_tab` 独立触发切页。
        """
        self.eval_tab.set_weights_and_yaml(weights_path, yaml_path)
        self.review_tab.set_weights_path(weights_path)

    def _restore_settings(self) -> None:
        """从持久化 `QSettings` 恢复上次保存的工作区目录、MLLM 策略与训练超参数偏好（如存在）。"""
        last_dir = self.settings.value("last_frames_dir", "")
        if (
            isinstance(last_dir, str)
            and last_dir
            and "pytest-of-" not in last_dir.lower()
            and Path(last_dir).is_dir()
        ):
            self.dataset_tab.set_frames_dir(last_dir)
            self.review_tab.set_images_dir(last_dir)
        self.review_tab.restore_settings(self.settings)
        self.train_tab.restore_settings(self.settings)

    def _running_workers(self) -> list[QThread]:
        """收集各标签页中仍在运行的后台 `QThread`（属性不存在或尚未创建时忽略）。"""
        candidates = [
            getattr(self.extract_tab, "probe_worker", None),
            getattr(self.extract_tab, "worker", None),
            getattr(self.dataset_tab, "scan_worker", None),
            getattr(self.dataset_tab, "build_worker", None),
            getattr(self.review_tab, "worker", None),
            getattr(self.review_tab, "conn_test_worker", None),
            getattr(self.train_tab, "worker", None),
            getattr(self.eval_tab, "val_worker", None),
            getattr(self.eval_tab, "pred_worker", None),
            getattr(self.eval_tab, "export_worker", None),
        ]
        return [w for w in candidates if isinstance(w, QThread) and w.isRunning()]

    def _running_readonly_probes(self) -> list[QThread]:
        """收集仍在收尾的只读后台探测线程（硬件探测与 `.pt` 权重元数据读取）。"""
        probes: list[QThread] = []
        if self._device_probe_worker is not None and self._device_probe_worker.isRunning():
            probes.append(self._device_probe_worker)
        get_inspect = getattr(self.eval_tab, "running_inspect_workers", None)
        if callable(get_inspect):
            probes.extend(
                w for w in get_inspect() if isinstance(w, QThread) and w.isRunning()
            )
        return probes

    def _poll_workers_then_close(self) -> None:
        """等待后台任务及只读探测线程全部结束后再次触发关闭；由 `_close_wait_timer` 周期调用。"""
        running = self._running_workers()
        probes = self._running_readonly_probes()
        if running:
            self.statusBar().showMessage(f"正在等待 {len(running)} 个后台任务安全停止后退出...")
            return
        if probes:
            self.statusBar().showMessage("正在等待后台只读探测收尾后退出...")
            return
        self._close_wait_timer.stop()
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭主窗口：先核查内嵌标注工作台未保存改动，再处理仍在运行的后台任务，最后持久化目录状态。

        后台任务处理：
            存在运行中的业务 `QThread` 时询问用户；确认后对支持取消的任务调用 `cancel()`，
            禁用界面并忽略本次关闭事件，由定时器轮询到全部线程结束后再次调用 `close()`。
            若仅有只读的后台硬件/权重元数据探测线程仍在收尾，则免打扰直接通过定时器等待其退出，
            避免析构运行中的 `QThread`。
        """
        if self._close_wait_timer.isActive():
            event.ignore()
            return
        if not self.labeling_tab.prepare_for_close():
            event.ignore()
            return

        running = self._running_workers()
        if running:
            reply = QMessageBox.question(
                self,
                "后台任务仍在运行",
                f"当前有 {len(running)} 个后台任务正在运行（如训练、抽帧或 MLLM 审核）。\n"
                "是否请求停止并在其安全结束后退出？\n"
                "训练会在当前轮次结束并保存权重后停止；验证、推理、导出需等待完成。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            event.ignore()
            if reply != QMessageBox.StandardButton.Yes:
                return
            for worker in running:
                cancel = getattr(worker, "cancel", None)
                if callable(cancel):
                    cancel()
            self.tabs.setEnabled(False)
            self.statusBar().showMessage("正在等待后台任务安全停止后退出...")
            self._close_wait_timer.start()
            return

        if self._running_readonly_probes():
            event.ignore()
            self.tabs.setEnabled(False)
            self.statusBar().showMessage("正在等待后台只读探测收尾后退出...")
            self._close_wait_timer.start()
            return

        cur_frames = self.dataset_tab.images_picker.text()
        if cur_frames and "pytest-of-" not in cur_frames.lower():
            self.settings.setValue("last_frames_dir", cur_frames)
        self.review_tab.save_settings(self.settings)
        self.train_tab.save_settings(self.settings)
        super().closeEvent(event)
