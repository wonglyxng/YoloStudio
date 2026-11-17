"""PyQt6 桌面主窗口模块，统筹六步工作流标签页及跨页数据流转。

职责：
    组装“1. 视频抽帧 → 2. 标注工作台 (X-AnyLabeling) → 3. 数据集准备 → 4. MLLM 标注纠错 →
    5. YOLO26 训练 → 6. 验证、推理与导出”六个标签页，协调内嵌工作台联动与跨页路径同步。

运行环境与协作边界：
    需 PyQt6 主事件循环与已创建的 `QApplication`；由 `main.py` 创建并显示，
    内部持有六个标签页实例及 `QSettings("YOLO26Studio", "VideoToYOLO26")` 持久化对象。
    退出窗口时先经 `LabelingStudioTab.prepare_for_close()` 确认未保存标注修改，未放行则阻止关闭。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from yolo_studio.gui.tab_dataset import DatasetTab
from yolo_studio.gui.tab_eval import EvalTab
from yolo_studio.gui.tab_extract import ExtractTab
from yolo_studio.gui.tab_labeling import LabelingStudioTab
from yolo_studio.gui.tab_review import MllmReviewTab
from yolo_studio.gui.tab_train import TrainTab

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

    def __init__(self, workspace_root: Path) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self.settings = QSettings("YOLO26Studio", "VideoToYOLO26")
        self.setWindowTitle(
            "YOLO26 视频抽帧与训练工具 (Video → 内嵌 X-AnyLabeling → MLLM纠错 → YOLO26)"
        )
        self.resize(1380, 860)
        self.setStyleSheet(APP_STYLESHEET)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.extract_tab = ExtractTab(workspace_root)
        self.labeling_tab = LabelingStudioTab(workspace_root)
        self.dataset_tab = DatasetTab(workspace_root)
        self.review_tab = MllmReviewTab(workspace_root)
        self.train_tab = TrainTab(workspace_root)
        self.eval_tab = EvalTab(workspace_root)

        self.tabs.addTab(self.extract_tab, "1. 视频抽帧")
        self.tabs.addTab(self.labeling_tab, "2. 标注工作台 (X-AnyLabeling)")
        self.tabs.addTab(self.dataset_tab, "3. 数据集准备")
        self.tabs.addTab(self.review_tab, "4. MLLM 标注纠错")
        self.tabs.addTab(self.train_tab, "5. YOLO26 训练")
        self.tabs.addTab(self.eval_tab, "6. 验证、推理与导出")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        # 跨标签页工作流信号连接
        self.extract_tab.send_frames_dir.connect(self._on_frames_ready)
        self.labeling_tab.send_to_dataset.connect(self._on_labeling_to_dataset)
        self.labeling_tab.send_to_review.connect(self._on_labeling_to_review)
        self.labeling_tab.annotations_changed.connect(self._on_annotations_changed)
        self.dataset_tab.open_in_workbench.connect(self._on_open_in_workbench)
        self.review_tab.open_in_workbench.connect(self._on_open_in_workbench)
        self.review_tab.send_cleaned_dataset.connect(self._on_cleaned_ready)
        self.dataset_tab.send_data_yaml.connect(self._on_yaml_ready)
        self.train_tab.send_best_weights.connect(self._on_weights_ready)

        self._restore_settings()

    def _on_frames_ready(self, frames_dir: str) -> None:
        """同步抽帧输出目录至数据集页与纠错页，并在主动跳转或处于首页时切至内嵌标注工作台。"""
        self.dataset_tab.set_frames_dir(frames_dir)
        self.review_tab.set_images_dir(frames_dir)
        if self.sender() == self.extract_tab.send_next_btn or self.tabs.currentIndex() == 0:
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
            self.tabs.setCurrentIndex(1)

    def _on_labeling_to_dataset(self, images_dir: str, classes_file: str) -> None:
        """将工作台当前标注目录与类别清单同步至「3. 数据集准备」并切换过去。"""
        self.dataset_tab.set_dataset_source(images_dir, classes_file, recursive=False)
        self.tabs.setCurrentIndex(2)

    def _on_labeling_to_review(self, images_dir: str) -> None:
        """将工作台当前图片目录同步至「4. MLLM 标注纠错」并切换过去。"""
        self.review_tab.set_images_dir(images_dir)
        self.tabs.setCurrentIndex(3)

    def _on_annotations_changed(self, dir_path: str) -> None:
        """工作台内保存或更新 `.json` 后，若数据集页当前指向同目录则更新状态提示。"""
        cur_ds_dir = self.dataset_tab.images_picker.text()
        if not cur_ds_dir:
            return
        try:
            if Path(cur_ds_dir).resolve() == Path(dir_path).resolve():
                self.dataset_tab.summary_label.setText(
                    "检测到「2. 标注工作台」已更新标注文件，可点击「扫描标注统计」刷新分布。"
                )
        except OSError:
            pass

    def _on_cleaned_ready(self, images_dir: str, classes_txt: str, recursive: bool) -> None:
        """将 MLLM 清洗输出目录与类别清单填入「3. 数据集准备」并切换至该页。"""
        self.dataset_tab.set_dataset_source(images_dir, classes_txt, recursive)
        self.tabs.setCurrentIndex(2)

    def _on_yaml_ready(self, yaml_path: str) -> None:
        """将生成的 `data.yaml` 路径同步至「5. YOLO26 训练」。"""
        self.train_tab.set_data_yaml(yaml_path)

    def _on_weights_ready(self, weights_path: str, yaml_path: str) -> None:
        """将训练产出的 `best.pt` 权重与 `data.yaml` 同步至「6. 验证、推理与导出」和「4. MLLM 标注纠错」。"""
        self.eval_tab.set_weights_and_yaml(weights_path, yaml_path)
        self.review_tab.set_weights_path(weights_path)

    def _restore_settings(self) -> None:
        """从持久化 `QSettings` 恢复上次保存的工作区目录偏好（如存在）。"""
        last_dir = self.settings.value("last_frames_dir", "")
        if isinstance(last_dir, str) and last_dir and Path(last_dir).is_dir():
            self.dataset_tab.set_frames_dir(last_dir)
            self.review_tab.set_images_dir(last_dir)

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭主窗口前先核查内嵌标注工作台未保存改动，再持久化目录状态。"""
        if not self.labeling_tab.prepare_for_close():
            event.ignore()
            return
        cur_frames = self.dataset_tab.images_picker.text()
        if cur_frames:
            self.settings.setValue("last_frames_dir", cur_frames)
        super().closeEvent(event)
