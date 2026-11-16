"""PySide6 桌面主窗口模块，统筹五步工作流标签页及跨页数据流转。

职责：
    组装“1. 视频抽帧 → 2. X-AnyLabeling 标注与数据集 → 3. MLLM 标注纠错 →
    4. YOLO26 训练 → 5. 验证、推理与导出”五个标签页，协调跨标签页路径同步与用户设置持久化。

运行环境与协作边界：
    需 PySide6 主事件循环与已创建的 `QApplication`；由 `main.py` 创建并显示，
    内部持有五个标签页实例及 `QSettings("YOLO26Studio", "VideoToYOLO26")` 持久化对象。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from yolo_studio.gui.tab_dataset import DatasetTab
from yolo_studio.gui.tab_eval import EvalTab
from yolo_studio.gui.tab_extract import ExtractTab
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
    """YOLO26 训练工具桌面顶层主窗口。

    状态与持久化：
        持有 `workspace_root`、`QSettings` 及五个工作流标签页；
        启动时恢复上次记录的 `xanylabeling_exe` 路径，关闭窗口时将非空路径写回配置。
    """

    def __init__(self, workspace_root: Path) -> None:
        super().__init__()
        self.workspace_root = workspace_root
        self.settings = QSettings("YOLO26Studio", "VideoToYOLO26")
        self.setWindowTitle(
            "YOLO26 视频抽帧与训练工具 (Video → X-AnyLabeling → MLLM纠错 → YOLO26)"
        )
        self.resize(1300, 820)
        self.setStyleSheet(APP_STYLESHEET)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.extract_tab = ExtractTab(workspace_root)
        self.dataset_tab = DatasetTab(workspace_root)
        self.review_tab = MllmReviewTab(workspace_root)
        self.train_tab = TrainTab(workspace_root)
        self.eval_tab = EvalTab(workspace_root)

        self.review_tab.set_xal_exe_provider(self.dataset_tab.get_xal_path)

        self.tabs.addTab(self.extract_tab, "1. 视频抽帧")
        self.tabs.addTab(self.dataset_tab, "2. X-AnyLabeling 标注与数据集")
        self.tabs.addTab(self.review_tab, "3. MLLM 标注纠错")
        self.tabs.addTab(self.train_tab, "4. YOLO26 训练")
        self.tabs.addTab(self.eval_tab, "5. 验证、推理与导出")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        # 跨标签页工作流信号连接
        self.extract_tab.send_frames_dir.connect(self._on_frames_ready)
        self.review_tab.send_cleaned_dataset.connect(self._on_cleaned_ready)
        self.dataset_tab.send_data_yaml.connect(self._on_yaml_ready)
        self.train_tab.send_best_weights.connect(self._on_weights_ready)

        self._restore_settings()

    def _on_frames_ready(self, frames_dir: str) -> None:
        """同步抽帧输出目录至数据集页与 MLLM 纠错页，并在主动跳转或处于首页时切至标签页 2。"""
        self.dataset_tab.set_frames_dir(frames_dir)
        self.review_tab.set_images_dir(frames_dir)
        if self.sender() == self.extract_tab.send_next_btn or self.tabs.currentIndex() == 0:
            self.tabs.setCurrentIndex(1)

    def _on_cleaned_ready(self, images_dir: str, classes_txt: str, recursive: bool) -> None:
        """将 MLLM 清洗输出目录与类别清单填入标签页 2 并切换至该页。"""
        self.dataset_tab.set_dataset_source(images_dir, classes_txt, recursive)
        self.tabs.setCurrentIndex(1)

    def _on_yaml_ready(self, yaml_path: str) -> None:
        """将生成的 `data.yaml` 路径同步至标签页 4（YOLO26 训练）。"""
        self.train_tab.set_data_yaml(yaml_path)

    def _on_weights_ready(self, weights_path: str, yaml_path: str) -> None:
        """将训练产出的 `best.pt` 权重与 `data.yaml` 同步至标签页 5（验证推理）和标签页 3（MLLM 纠错）。"""
        self.eval_tab.set_weights_and_yaml(weights_path, yaml_path)
        self.review_tab.set_weights_path(weights_path)

    def _restore_settings(self) -> None:
        """从持久化 `QSettings` 恢复上次保存的 X-AnyLabeling 可执行路径。"""
        saved_xal = self.settings.value("xanylabeling_exe", "")
        if isinstance(saved_xal, str) and saved_xal:
            self.dataset_tab.set_xal_path(saved_xal)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """窗口关闭时将非空的 X-AnyLabeling 路径写入 `QSettings`（保留 type: ignore 指令）。"""
        xal_path = self.dataset_tab.get_xal_path()
        if xal_path:
            self.settings.setValue("xanylabeling_exe", xal_path)
        super().closeEvent(event)
