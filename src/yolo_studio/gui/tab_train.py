"""标签页 5：YOLO26 模型训练界面。

职责：
    展示本机计算设备（CUDA GPU/CPU）状态，配置 `data.yaml`、预训练权重、超参数与保存目录，
    并在后台启动训练后实时刷新 Epoch 指标卡片与 Ultralytics 日志流。

运行环境与协作边界：
    运行于 PyQt6 主线程；训练任务委托给 `gui.workers.TrainWorker`，
    在训练产出有效 `best.pt` 时通过 `send_best_weights` 信号向主窗口同步给标签页 4 与标签页 6。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.trainer import (
    YOLO26_DETECT_MODELS,
    DeviceInfo,
    EpochMetrics,
    TrainConfig,
    TrainResult,
    detect_compute_device,
)
from yolo_studio.gui.common import (
    GpuTaskGuard,
    LOG_MAX_LINES,
    PathPickerRow,
    make_metric_card,
    open_local_path,
)
from yolo_studio.gui.workers import TrainWorker


class TrainTab(QWidget):
    """YOLO26 模型训练功能标签页（标签页 5）。

    内部状态：
        持有硬件探测结果 `device_info`、跨页计算设备互斥协调器 `gpu_guard`、训练工作线程 `worker`、
        最近实验目录 `_last_save_dir` 以及最近产出的最佳权重路径 `_last_best_pt`。
    """

    send_best_weights = Signal(str, str)  # 信号参数：(weights_path, data_yaml)

    def __init__(
        self,
        workspace_root: Path,
        parent: QWidget | None = None,
        device_info: DeviceInfo | None = None,
        gpu_guard: GpuTaskGuard | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.device_info: DeviceInfo = device_info or detect_compute_device()
        self.gpu_guard = gpu_guard
        self.worker: TrainWorker | None = None
        self._last_save_dir: Path = workspace_root / "runs" / "detect"
        self._last_best_pt: str = ""
        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(10)

        # 计算设备状态横幅
        hw_group = QGroupBox("1. 计算设备状态")
        hw_layout = QHBoxLayout(hw_group)
        if self.device_info.cuda_available:
            hw_text = (
                f"已就绪 GPU: {self.device_info.device_name} "
                f"({self.device_info.vram_gb:.1f} GB 显存) | CUDA 可用"
            )
            hw_color = "#1f7a1f"
        else:
            hw_text = "未检测到可用 CUDA GPU，将使用 CPU 训练（速度较慢）"
            hw_color = "#b25900"
        self.hw_label = QLabel(hw_text)
        self.hw_label.setStyleSheet(f"color: {hw_color}; font-weight: 600;")
        hw_layout.addWidget(self.hw_label)
        left.addWidget(hw_group)

        # 模型与训练超参数配置
        cfg_group = QGroupBox("2. 模型与训练参数")
        form = QFormLayout(cfg_group)
        form.setSpacing(8)

        default_yaml = str((self.workspace_root / "datasets" / "custom_yolo26" / "data.yaml").resolve())
        self.yaml_picker = PathPickerRow(
            placeholder="选择数据集 data.yaml 文件",
            mode="file",
            file_filter="YAML 文件 (*.yaml *.yml);;所有文件 (*.*)",
            default_path=default_yaml,
        )
        form.addRow("data.yaml:", self.yaml_picker)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        for model_file, desc in YOLO26_DETECT_MODELS:
            self.model_combo.addItem(desc, model_file)
        self.model_combo.setCurrentIndex(1)  # 默认选用均衡推荐的 yolo26s.pt
        self.custom_pt_btn = QPushButton("选择本地 .pt")
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.custom_pt_btn)
        form.addRow("预训练模型:", model_row)

        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setPlaceholderText("留空使用上方选择的官方 YOLO26 权重，或填入本地 .pt 路径")
        form.addRow("自定义权重路径:", self.custom_model_edit)

        grid_row1 = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(100)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.setEditable(True)
        self.imgsz_combo.addItems(["320", "480", "640", "800", "960", "1024", "1280"])
        self.imgsz_combo.setCurrentText("640")

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 512)
        self.batch_spin.setValue(16 if self.device_info.cuda_available else 4)
        self.batch_spin.setToolTip("设为 -1 可让 Ultralytics 自动探测最大可用 Batch Size")

        grid_row1.addWidget(QLabel("Epochs:"))
        grid_row1.addWidget(self.epochs_spin)
        grid_row1.addWidget(QLabel("ImgSz:"))
        grid_row1.addWidget(self.imgsz_combo)
        grid_row1.addWidget(QLabel("Batch(-1=自动):"))
        grid_row1.addWidget(self.batch_spin)
        form.addRow("核心超参数:", grid_row1)

        grid_row2 = QHBoxLayout()
        self.lr0_spin = QDoubleSpinBox()
        self.lr0_spin.setDecimals(4)
        self.lr0_spin.setRange(0.0001, 0.5)
        self.lr0_spin.setSingleStep(0.001)
        self.lr0_spin.setValue(0.01)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(50)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)

        self.device_edit = QLineEdit(self.device_info.recommended_device)
        self.device_edit.setFixedWidth(70)

        grid_row2.addWidget(QLabel("LR0:"))
        grid_row2.addWidget(self.lr0_spin)
        grid_row2.addWidget(QLabel("Patience:"))
        grid_row2.addWidget(self.patience_spin)
        grid_row2.addWidget(QLabel("Workers:"))
        grid_row2.addWidget(self.workers_spin)
        grid_row2.addWidget(QLabel("Device:"))
        grid_row2.addWidget(self.device_edit)
        form.addRow("优化器与硬件:", grid_row2)

        opt_row = QHBoxLayout()
        self.amp_chk = QCheckBox("混合精度 AMP")
        self.amp_chk.setChecked(self.device_info.cuda_available)
        self.cache_chk = QCheckBox("内存缓存图片 (cache)")
        self.resume_chk = QCheckBox("断点续训 (resume)")
        self.resume_chk.setToolTip(
            "从上次中断的训练继续：优先使用「自定义权重路径」中选择的 last.pt，"
            "否则使用「保存根目录/实验名称/weights/last.pt」；新训练不会覆盖已有实验目录，会自动递增编号"
        )
        opt_row.addWidget(self.amp_chk)
        opt_row.addWidget(self.cache_chk)
        opt_row.addWidget(self.resume_chk)
        form.addRow("训练选项:", opt_row)

        default_proj = str((self.workspace_root / "runs" / "detect").resolve())
        self.project_picker = PathPickerRow(
            placeholder="训练结果保存根目录", mode="dir", default_path=default_proj
        )
        form.addRow("保存根目录:", self.project_picker)

        self.run_name_edit = QLineEdit("yolo26_train")
        form.addRow("实验名称:", self.run_name_edit)

        left.addWidget(cfg_group)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("开始 YOLO26 训练")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("安全停止训练")
        self.stop_btn.setEnabled(False)
        self.open_run_btn = QPushButton("打开训练结果目录")
        self.send_eval_btn = QPushButton("将 best.pt 发送到「6. 验证与推理」→")
        self.send_eval_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.open_run_btn)
        btn_row.addWidget(self.send_eval_btn)
        left.addLayout(btn_row)

        # 右侧面板：实时训练指标卡片与日志输出
        right = QVBoxLayout()
        right.setSpacing(10)

        metrics_group = QGroupBox("3. 实时训练指标")
        metrics_grid = QGridLayout(metrics_group)
        metrics_grid.setSpacing(6)

        c1, self.val_epoch = make_metric_card("Epoch", "0 / 0")
        c2, self.val_box_loss = make_metric_card("Box Loss", "-")
        c3, self.val_cls_loss = make_metric_card("Cls Loss", "-")
        c4, self.val_precision = make_metric_card("Precision", "-")
        c5, self.val_recall = make_metric_card("Recall", "-")
        c6, self.val_map50 = make_metric_card("mAP50", "-")
        c7, self.val_map50_95 = make_metric_card("mAP50-95", "-")

        metrics_grid.addWidget(c1, 0, 0, 1, 2)
        metrics_grid.addWidget(c2, 1, 0)
        metrics_grid.addWidget(c3, 1, 1)
        metrics_grid.addWidget(c4, 2, 0)
        metrics_grid.addWidget(c5, 2, 1)
        metrics_grid.addWidget(c6, 3, 0)
        metrics_grid.addWidget(c7, 3, 1)
        right.addWidget(metrics_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        right.addWidget(self.progress_bar)

        log_group = QGroupBox("4. 训练日志")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Ultralytics YOLO26 实时日志将在此输出...")
        log_layout.addWidget(self.log_box)
        right.addWidget(log_group, 1)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        # 信号与槽连接
        self.custom_pt_btn.clicked.connect(self._on_pick_custom_pt)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_run_btn.clicked.connect(lambda: open_local_path(self._last_save_dir))
        self.send_eval_btn.clicked.connect(
            lambda: self.send_best_weights.emit(self._last_best_pt, self.yaml_picker.text())
            if self._last_best_pt
            else None
        )

    def set_data_yaml(self, yaml_path: str) -> None:
        """跨页写入：更新训练集配置文件 `data.yaml` 路径。"""
        if yaml_path:
            self.yaml_picker.setText(yaml_path)

    def _on_pick_custom_pt(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO26 权重文件", str(self.workspace_root), "PyTorch 权重 (*.pt);;所有文件 (*.*)"
        )
        if chosen:
            self.custom_model_edit.setText(chosen)

    def _on_start(self) -> None:
        """校验 `data.yaml` 与 `imgsz` 后锁定运行按钮、清空日志并启动 `TrainWorker`。"""
        yaml_text = self.yaml_picker.text()
        if not yaml_text or not Path(yaml_text).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 data.yaml 文件。")
            return

        try:
            imgsz = int(self.imgsz_combo.currentText().strip())
        except ValueError:
            QMessageBox.warning(self, "提示", "ImgSz 必须是正整数。")
            return

        model_name = self.custom_model_edit.text().strip() or str(self.model_combo.currentData())
        proj_dir = Path(self.project_picker.text() or (self.workspace_root / "runs" / "detect"))
        run_name = self.run_name_edit.text().strip() or "yolo26_train"
        self._last_save_dir = proj_dir / run_name

        cfg = TrainConfig(
            data_yaml=Path(yaml_text),
            model_name=model_name,
            epochs=self.epochs_spin.value(),
            imgsz=imgsz,
            batch=self.batch_spin.value(),
            device=self.device_edit.text().strip() or "0",
            lr0=self.lr0_spin.value(),
            patience=self.patience_spin.value(),
            workers=self.workers_spin.value(),
            amp=self.amp_chk.isChecked(),
            cache=self.cache_chk.isChecked(),
            project_dir=proj_dir,
            run_name=run_name,
            resume=self.resume_chk.isChecked(),
        )

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "YOLO26 模型训练"):
            return

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.send_eval_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self.log_box.appendPlainText("正在启动 YOLO26 训练任务...")

        self.worker = TrainWorker(cfg)
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.worker, "YOLO26 模型训练")
        self.worker.epoch_updated.connect(self._on_epoch_updated)
        self.worker.log_line.connect(self._on_log_line)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stop(self) -> None:
        """向训练线程请求安全停训（当前 Epoch 完成后保存权重退出）。"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.stop_btn.setEnabled(False)
            self.log_box.appendPlainText("正在请求停止训练，请等待当前 Epoch 完成保存...")

    def _on_epoch_updated(self, m: EpochMetrics) -> None:
        pct = int(min(100, (m.epoch / max(1, m.total_epochs)) * 100))
        self.progress_bar.setValue(pct)
        self.val_epoch.setText(f"{m.epoch} / {m.total_epochs}")
        self.val_box_loss.setText(f"{m.box_loss:.4f}")
        self.val_cls_loss.setText(f"{m.cls_loss:.4f}")
        self.val_precision.setText(f"{m.precision:.4f}")
        self.val_recall.setText(f"{m.recall:.4f}")
        self.val_map50.setText(f"{m.map50:.4f}")
        self.val_map50_95.setText(f"{m.map50_95:.4f}")

    def _on_log_line(self, line: str) -> None:
        self.log_box.appendPlainText(line)

    def _on_finished(self, res: TrainResult) -> None:
        """恢复界面状态并记录产物目录；若生成了 `best.pt` 则自动通知下游标签页。"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._last_save_dir = res.save_dir
        if not res.stopped_early:
            self.progress_bar.setValue(100)

        status_text = "训练已提前停止" if res.stopped_early else "训练完成"
        self.log_box.appendPlainText(f"\n=== {status_text} ===")
        self.log_box.appendPlainText(f"结果目录: {res.save_dir}")
        if res.best_weights:
            self._last_best_pt = str(res.best_weights)
            self.send_eval_btn.setEnabled(True)
            self.log_box.appendPlainText(f"最佳权重: {res.best_weights}")
            self.send_best_weights.emit(self._last_best_pt, self.yaml_picker.text())
        if res.last_weights:
            self.log_box.appendPlainText(f"最新权重: {res.last_weights}")

    def _on_failed(self, err: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_box.appendPlainText(f"\n[错误] {err}")
        QMessageBox.critical(self, "训练错误", err)
