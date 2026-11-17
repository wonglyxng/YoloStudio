"""标签页 3：标注统计检验与 YOLO26 数据集准备界面。

职责：
    与「2. 标注工作台」联动打开标注图片目录，扫描并展示标注分布与告警，
    并按 train/val/test 比例生成 YOLO26 标准数据集与 `data.yaml`。

运行环境与协作边界：
    运行于 PyQt6 主线程；扫描与构建任务委托给 `DatasetScanWorker` / `DatasetBuildWorker`，
    通过 `open_in_workbench` 请求主窗口切换至内嵌标注工作台，
    生成结果通过 `send_data_yaml` 信号同步至 `MainWindow` 与训练标签页。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.dataset_converter import (
    AnnotationFormat,
    DatasetBuildConfig,
    DatasetBuildResult,
    DatasetScanSummary,
)
from yolo_studio.gui.common import PathPickerRow, open_local_path
from yolo_studio.gui.workers import DatasetBuildWorker, DatasetScanWorker


class DatasetTab(QWidget):
    """标注校验与 YOLO26 数据集构建标签页（标签页 3）。

    内部状态：
        持有后台线程 `scan_worker`、`build_worker` 及最近一次成功生成的 `_last_yaml_path`。
    """

    send_data_yaml = Signal(str)
    open_in_workbench = Signal(str, str)  # 信号参数：(图片目录, 可选 classes.txt 路径)

    def __init__(self, workspace_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.scan_worker: DatasetScanWorker | None = None
        self.build_worker: DatasetBuildWorker | None = None
        self._last_yaml_path: str = ""
        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(10)

        # 分组 1：标注源目录与内嵌工作台联动
        xal_group = QGroupBox("1. 标注源目录与工作台联动")
        xal_form = QFormLayout(xal_group)

        default_frames = str((self.workspace_root / "data" / "raw_frames").resolve())
        self.images_picker = PathPickerRow(
            placeholder="选择待标注或待转换的图片目录（与抽帧输出一致）",
            mode="dir",
            default_path=default_frames,
        )
        xal_form.addRow("图片目录:", self.images_picker)

        xal_btn_row = QHBoxLayout()
        self.open_workbench_btn = QPushButton("在「2. 标注工作台」中打开标注 →")
        self.open_workbench_btn.setObjectName("primaryBtn")
        xal_tip = QLabel("提示：支持 X-AnyLabeling 同目录保存的原生 .json 以及导出的 YOLO .txt；工作台保存后可直接点下方扫描统计。")
        xal_tip.setWordWrap(True)
        xal_tip.setStyleSheet("color: #666666;")
        xal_btn_row.addWidget(self.open_workbench_btn)
        xal_form.addRow("", xal_btn_row)
        xal_form.addRow("", xal_tip)
        left.addWidget(xal_group)

        # 分组 2：标注格式转换与 Train/Val/Test 划分配置
        ds_group = QGroupBox("2. 标注转换与 Train/Val 数据集划分")
        ds_form = QFormLayout(ds_group)

        self.ann_picker = PathPickerRow(
            placeholder="留空则默认与「图片目录」相同（支持 .json 或导出的 .txt）",
            mode="dir",
        )
        ds_form.addRow("标注目录(可选):", self.ann_picker)

        self.format_combo = QComboBox()
        self.format_combo.addItem("自动识别 (.json 或 YOLO .txt)", AnnotationFormat.AUTO)
        self.format_combo.addItem("X-AnyLabeling 原生 .json", AnnotationFormat.XANYLABELING_JSON)
        self.format_combo.addItem("YOLO 导出 .txt + classes.txt", AnnotationFormat.YOLO_TXT)
        ds_form.addRow("标注格式:", self.format_combo)

        self.classes_picker = PathPickerRow(
            placeholder="可选 classes.txt（定义类别顺序，留空则自动提取）",
            mode="file",
            file_filter="文本文件 (*.txt);;所有文件 (*.*)",
        )
        ds_form.addRow("classes.txt(可选):", self.classes_picker)

        default_ds_out = str((self.workspace_root / "datasets" / "custom_yolo26").resolve())
        self.output_ds_picker = PathPickerRow(
            placeholder="输出 YOLO26 标准数据集目录",
            mode="dir",
            default_path=default_ds_out,
        )
        ds_form.addRow("数据集输出目录:", self.output_ds_picker)

        ratio_row = QHBoxLayout()
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.1, 0.99)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.8)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 0.9)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 0.9)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)

        ratio_row.addWidget(QLabel("Train:"))
        ratio_row.addWidget(self.train_spin)
        ratio_row.addWidget(QLabel("Val:"))
        ratio_row.addWidget(self.val_spin)
        ratio_row.addWidget(QLabel("Test:"))
        ratio_row.addWidget(self.test_spin)
        ratio_row.addWidget(QLabel("Seed:"))
        ratio_row.addWidget(self.seed_spin)
        ds_form.addRow("划分比例:", ratio_row)

        opt_row = QHBoxLayout()
        self.include_empty_chk = QCheckBox("保留未标注图片作为负样本")
        self.recursive_chk = QCheckBox("递归扫描子目录")
        opt_row.addWidget(self.include_empty_chk)
        opt_row.addWidget(self.recursive_chk)
        ds_form.addRow("附加选项:", opt_row)

        left.addWidget(ds_group)

        btn_row = QHBoxLayout()
        self.scan_btn = QPushButton("扫描标注统计")
        self.build_btn = QPushButton("生成 YOLO26 数据集 (data.yaml)")
        self.build_btn.setObjectName("primaryBtn")
        self.open_ds_btn = QPushButton("打开数据集目录")
        self.send_train_btn = QPushButton("发送到「5. YOLO26 训练」→")
        self.send_train_btn.setEnabled(False)
        btn_row.addWidget(self.scan_btn)
        btn_row.addWidget(self.build_btn)
        btn_row.addWidget(self.open_ds_btn)
        btn_row.addWidget(self.send_train_btn)
        left.addLayout(btn_row)

        # 右侧面板：类别分布表与扫描/构建日志
        right = QVBoxLayout()
        right.setSpacing(10)

        stat_group = QGroupBox("3. 标注统计与校验结果")
        stat_layout = QVBoxLayout(stat_group)

        self.summary_label = QLabel("点击「扫描标注统计」查看图片数、标注框数和类别分布。")
        self.summary_label.setWordWrap(True)
        stat_layout.addWidget(self.summary_label)

        self.class_table = QTableWidget(0, 3)
        self.class_table.setHorizontalHeaderLabels(["类别 ID", "类别名称", "标注框数量"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        stat_layout.addWidget(self.class_table, 2)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("转换日志与告警信息将在此显示...")
        stat_layout.addWidget(self.log_box, 1)

        right.addWidget(stat_group)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        # 信号与槽连接
        self.open_workbench_btn.clicked.connect(self._on_open_in_workbench)
        self.scan_btn.clicked.connect(self._on_scan)
        self.build_btn.clicked.connect(self._on_build)
        self.open_ds_btn.clicked.connect(lambda: open_local_path(self.output_ds_picker.path()))
        self.send_train_btn.clicked.connect(
            lambda: self.send_data_yaml.emit(self._last_yaml_path) if self._last_yaml_path else None
        )

    def set_frames_dir(self, path_str: str) -> None:
        """跨页写入：更新待标注/转换的图片源目录输入框。"""
        if path_str:
            self.images_picker.setText(path_str)

    def set_dataset_source(
        self, images_dir: str, classes_file: str = "", recursive: bool = False
    ) -> None:
        """跨页写入：接收 MLLM 清洗结果目录、类别文件并同步递归扫描开关，清空独立标注目录栏。"""
        if images_dir:
            self.images_picker.setText(images_dir)
            self.ann_picker.setText("")
        if classes_file:
            self.classes_picker.setText(classes_file)
        self.recursive_chk.setChecked(recursive)

    def _build_config(self) -> DatasetBuildConfig:
        images_text = self.images_picker.text()
        if not images_text:
            raise ValueError("请先指定图片目录。")
        ann_text = self.ann_picker.text()
        cls_text = self.classes_picker.text()
        out_text = self.output_ds_picker.text()
        if not out_text:
            raise ValueError("请先指定数据集输出目录。")

        return DatasetBuildConfig(
            images_dir=Path(images_text),
            annotations_dir=Path(ann_text) if ann_text else None,
            output_dir=Path(out_text),
            annotation_format=self.format_combo.currentData(),
            classes_file=Path(cls_text) if cls_text else None,
            train_ratio=self.train_spin.value(),
            val_ratio=self.val_spin.value(),
            test_ratio=self.test_spin.value(),
            include_unannotated=self.include_empty_chk.isChecked(),
            seed=self.seed_spin.value(),
            copy_images=True,
            recursive=self.recursive_chk.isChecked(),
        )

    def _on_open_in_workbench(self) -> None:
        images_text = self.images_picker.text()
        if not images_text or not Path(images_text).is_dir():
            QMessageBox.warning(self, "提示", "请先选择有效的图片目录。")
            return
        cls_text = self.classes_picker.text()
        self.open_in_workbench.emit(images_text, cls_text)
        self.log_box.append(f"已在「2. 标注工作台」中打开目录: {images_text}")

    def _on_scan(self) -> None:
        """校验参数后禁用扫描/生成按钮并在后台线程执行只读扫描。"""
        try:
            cfg = self._build_config()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        self.scan_btn.setEnabled(False)
        self.build_btn.setEnabled(False)
        self.log_box.clear()
        self.summary_label.setText("正在扫描图片和标注文件...")

        self.scan_worker = DatasetScanWorker(cfg)
        self.scan_worker.finished_summary.connect(self._on_scan_done)
        self.scan_worker.failed.connect(self._on_worker_failed)
        self.scan_worker.start()

    def _on_scan_done(self, summary: DatasetScanSummary) -> None:
        self.scan_btn.setEnabled(True)
        self.build_btn.setEnabled(True)

        fmt_name = "X-AnyLabeling JSON" if summary.detected_format == "json" else "YOLO TXT"
        self.summary_label.setText(
            f"识别格式: {fmt_name} | 图片总数: {summary.total_images} | "
            f"已标注: {summary.annotated_images} | 未标注: {summary.unannotated_images} | "
            f"总框数: {summary.total_boxes} | 类别数: {len(summary.classes)}"
        )
        self._populate_class_table(summary.classes, summary.class_counts)
        if summary.warnings:
            self.log_box.append("发现以下告警：")
            for w in summary.warnings[:50]:
                self.log_box.append(f"- {w}")
        else:
            self.log_box.append("扫描完成，未发现异常。")

    def _on_build(self) -> None:
        """校验参数后在后台线程执行数据集转换、划分与 `data.yaml` 写出。"""
        try:
            cfg = self._build_config()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        self.scan_btn.setEnabled(False)
        self.build_btn.setEnabled(False)
        self.log_box.clear()
        self.summary_label.setText("正在转换标注并生成 YOLO26 数据集...")

        self.build_worker = DatasetBuildWorker(cfg)
        self.build_worker.finished_result.connect(self._on_build_done)
        self.build_worker.failed.connect(self._on_worker_failed)
        self.build_worker.start()

    def _on_build_done(self, result: DatasetBuildResult) -> None:
        """恢复按钮状态、展示划分统计并自动向训练页派发新生成的 `data.yaml` 路径。"""
        self.scan_btn.setEnabled(True)
        self.build_btn.setEnabled(True)
        self._last_yaml_path = str(result.data_yaml_path)
        self.send_train_btn.setEnabled(True)

        self.summary_label.setText(
            f"数据集已生成！Train: {result.train_count} | Val: {result.val_count} | "
            f"Test: {result.test_count} | 总标注框: {result.total_boxes}"
        )
        self._populate_class_table(result.classes, result.class_counts)

        self.log_box.append(f"data.yaml 已生成: {result.data_yaml_path}")
        self.log_box.append(f"classes.txt 已生成: {result.classes_txt_path}")
        for w in result.warnings[:50]:
            self.log_box.append(f"[警告] {w}")

        self.send_data_yaml.emit(self._last_yaml_path)

    def _on_worker_failed(self, err: str) -> None:
        self.scan_btn.setEnabled(True)
        self.build_btn.setEnabled(True)
        self.summary_label.setText(f"处理失败: {err}")
        QMessageBox.critical(self, "数据集处理错误", err)

    def _populate_class_table(self, classes: list[str], counts: dict[str, int]) -> None:
        self.class_table.setRowCount(len(classes))
        for idx, name in enumerate(classes):
            self.class_table.setItem(idx, 0, QTableWidgetItem(str(idx)))
            self.class_table.setItem(idx, 1, QTableWidgetItem(name))
            self.class_table.setItem(idx, 2, QTableWidgetItem(str(counts.get(name, 0))))
