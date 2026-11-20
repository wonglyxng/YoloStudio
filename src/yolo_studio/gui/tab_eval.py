"""标签页 6：模型验证、推理预览与导出界面。

职责：
    对训练好的 YOLO26 `.pt` 权重执行验证集指标评估（Val）、图片/目录/视频推理可视化预览，
    以及导出 `ONNX`、`TorchScript` 或 `TensorRT (.engine)` 部署模型。

运行环境与协作边界：
    运行于 PyQt6 主线程；三类耗时任务分别委托给 `ValidateWorker`、`PredictWorker`、
    `ExportWorker`，并通过 `_set_busy()` 互斥锁住执行按钮，避免同一时刻并发占用模型或 GPU。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.dataset_converter import IMAGE_EXTENSIONS
from yolo_studio.core.trainer import DeviceInfo, detect_compute_device
from yolo_studio.gui.common import (
    ComputeDeviceControl,
    GpuTaskGuard,
    LOG_MAX_LINES,
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    load_pixmap_scaled,
    open_local_path,
)
from yolo_studio.gui.workers import ExportWorker, PredictWorker, ValidateWorker


class EvalTab(QWidget):
    """模型验证、推理测试与部署导出标签页（标签页 6）。

    内部状态：
        持有硬件信息 `device_info`、跨页计算设备互斥协调器 `gpu_guard`，以及三个互斥运行的后台工作线程引用
        `val_worker`、`pred_worker`、`export_worker`。
    """

    def __init__(
        self,
        workspace_root: Path,
        parent: QWidget | None = None,
        device_info: DeviceInfo | None = None,
        gpu_guard: GpuTaskGuard | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.device_info = device_info or detect_compute_device()
        self.gpu_guard = gpu_guard
        self.val_worker: ValidateWorker | None = None
        self.pred_worker: PredictWorker | None = None
        self.export_worker: ExportWorker | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(10)

        # 分组 1：模型权重、运行设备与验证配置
        model_group = QGroupBox("1. 模型权重与运行设备")
        m_form = QFormLayout(model_group)
        configure_form_layout(m_form)

        self.weights_picker = PathPickerRow(
            placeholder="选择训练好的 best.pt 权重",
            mode="file",
            file_filter="PyTorch 权重 (*.pt);;所有文件 (*.*)",
        )
        m_form.addRow("模型权重 (.pt):", self.weights_picker)

        self.yaml_picker = PathPickerRow(
            placeholder="选择数据集 data.yaml（用于验证指标评估）",
            mode="file",
            file_filter="YAML 文件 (*.yaml *.yml);;所有文件 (*.*)",
        )
        m_form.addRow("data.yaml:", self.yaml_picker)

        self.device_ctrl = ComputeDeviceControl(
            self.device_info,
            show_status_text=True,
            allow_multi_gpu=False,
            checkbox_text="启用 GPU 加速",
            compact_status=True,
        )
        m_form.addRow("运行设备:", self.device_ctrl)

        hw_row = QHBoxLayout()
        hw_row.setSpacing(6)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(128, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.val_btn = QPushButton("运行验证 (Val)")
        hw_row.addWidget(QLabel("ImgSz:"))
        hw_row.addWidget(self.imgsz_spin)
        hw_row.addStretch(1)
        hw_row.addWidget(self.val_btn)
        m_form.addRow("验证设置:", hw_row)
        left.addWidget(model_group)

        # 分组 2：推理效果测试
        pred_group = QGroupBox("2. 推理效果测试")
        p_form = QFormLayout(pred_group)
        configure_form_layout(p_form)

        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择测试图片、图片文件夹或测试视频")
        self.pick_file_btn = QPushButton("选图片/视频")
        self.pick_dir_btn = QPushButton("选文件夹")
        src_row.addWidget(self.source_edit, 1)
        src_row.addWidget(self.pick_file_btn)
        src_row.addWidget(self.pick_dir_btn)
        p_form.addRow("测试输入:", src_row)

        default_pred_out = str((self.workspace_root / "runs" / "predict_preview").resolve())
        self.pred_out_picker = PathPickerRow(
            placeholder="推理结果输出目录", mode="dir", default_path=default_pred_out
        )
        p_form.addRow("输出目录:", self.pred_out_picker)

        thr_row = QHBoxLayout()
        thr_row.setSpacing(6)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.predict_btn = QPushButton("运行推理预览")
        self.predict_btn.setObjectName("primaryBtn")
        self.open_pred_btn = QPushButton("打开输出目录")
        thr_row.addWidget(QLabel("Conf:"))
        thr_row.addWidget(self.conf_spin)
        thr_row.addSpacing(12)
        thr_row.addWidget(QLabel("IoU:"))
        thr_row.addWidget(self.iou_spin)
        thr_row.addStretch(1)
        thr_row.addWidget(self.predict_btn)
        thr_row.addWidget(self.open_pred_btn)
        p_form.addRow("阈值与执行:", thr_row)
        left.addWidget(pred_group)

        # 分组 3：模型格式导出
        exp_group = QGroupBox("3. 模型导出 (部署格式)")
        e_form = QFormLayout(exp_group)
        configure_form_layout(e_form)

        exp_row = QHBoxLayout()
        exp_row.setSpacing(6)
        self.format_combo = QComboBox()
        self.format_combo.addItem("ONNX (通用推理推荐)", "onnx")
        self.format_combo.addItem("TorchScript (.torchscript)", "torchscript")
        self.format_combo.addItem("TensorRT Engine (.engine, 需本机 TensorRT)", "engine")
        self.half_chk = QCheckBox("FP16 半精度 (half)")
        self.export_btn = QPushButton("导出模型")
        exp_row.addWidget(self.format_combo, 1)
        exp_row.addWidget(self.half_chk)
        exp_row.addWidget(self.export_btn)
        e_form.addRow("导出格式:", exp_row)
        align_form_labels([m_form, p_form, e_form])
        left.addWidget(exp_group)

        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("验证指标、推理结果路径与导出日志将在此显示...")
        left.addWidget(self.log_box, 1)

        # 右侧面板：推理结果图像预览
        right = QVBoxLayout()
        preview_group = QGroupBox("4. 推理结果预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("运行图片推理后，将在此显示带检测框的结果图")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 360)
        self.preview_label.setStyleSheet(
            "border: 1px dashed #999999; border-radius: 6px; background: #1e1e1e; color: #cccccc;"
        )
        preview_layout.addWidget(self.preview_label, 1)
        right.addWidget(preview_group)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        # 信号与槽连接
        self.pick_file_btn.clicked.connect(self._on_pick_file)
        self.pick_dir_btn.clicked.connect(self._on_pick_dir)
        self.val_btn.clicked.connect(self._on_validate)
        self.predict_btn.clicked.connect(self._on_predict)
        self.open_pred_btn.clicked.connect(lambda: open_local_path(self.pred_out_picker.path()))
        self.export_btn.clicked.connect(self._on_export)
        self.format_combo.currentIndexChanged.connect(self._on_export_format_changed)
        self.device_ctrl.device_changed.connect(self._on_export_format_changed)
        self._on_export_format_changed()

    def _on_export_format_changed(self, *_args: object) -> None:
        """同步半精度选项状态：ONNX 导出或 CPU 运行模式下均不支持 FP16，需在界面上置灰提示。"""
        is_onnx = str(self.format_combo.currentData()) == "onnx"
        using_gpu = self.device_ctrl.is_using_gpu() and not self.device_info.probing
        if is_onnx:
            self.half_chk.setChecked(False)
            self.half_chk.setEnabled(False)
            self.half_chk.setToolTip(
                "ONNX 导出固定使用 CPU 以规避图追踪设备冲突，不支持 FP16 半精度；选择 TorchScript 或 TensorRT 时可启用"
            )
        elif not using_gpu:
            self.half_chk.setChecked(False)
            self.half_chk.setEnabled(False)
            self.half_chk.setToolTip(
                "当前运行设备为 CPU 模式，不支持 FP16 半精度导出；请在上方启用 GPU 加速后再开启"
            )
        else:
            self.half_chk.setEnabled(True)
            self.half_chk.setToolTip("以 FP16 半精度导出权重（需在 GPU 模式下执行）")

    def apply_device_info(self, info: DeviceInfo) -> None:
        """异步硬件探测完成后刷新运行设备开关与导出 FP16 可用状态（保留用户已手动修改的选择）。"""
        self.device_info = info
        self.device_ctrl.apply_device_info(info)
        self._on_export_format_changed()

    def set_weights_and_yaml(self, weights_path: str, yaml_path: str) -> None:
        """跨页写入：同步更新待评估权重与 `data.yaml` 路径。"""
        if weights_path:
            self.weights_picker.setText(weights_path)
        if yaml_path:
            self.yaml_picker.setText(yaml_path)

    def _on_pick_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "选择测试图片或视频",
            str(self.workspace_root),
            "媒体文件 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.flv *.wmv *.ts *.m4v *.webm);;所有文件 (*.*)",
        )
        if chosen:
            self.source_edit.setText(chosen)

    def _on_pick_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择测试图片目录", str(self.workspace_root))
        if chosen:
            self.source_edit.setText(chosen)

    def _set_busy(self, busy: bool) -> None:
        """互斥启用或禁用验证、推理与导出三个执行按钮，防止后台任务冲突。"""
        self.val_btn.setEnabled(not busy)
        self.predict_btn.setEnabled(not busy)
        self.export_btn.setEnabled(not busy)

    def _on_validate(self) -> None:
        """校验权重与 `data.yaml` 后在后台启动 `ValidateWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())
        w = self.weights_picker.text()
        y = self.yaml_picker.text()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 .pt 权重文件。")
            return
        if not y or not Path(y).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 data.yaml 文件。")
            return

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "模型验证 (Val)"):
            return

        self._set_busy(True)
        self.log_box.clear()
        self.log_box.appendPlainText("开始执行模型验证...")

        self.val_worker = ValidateWorker(
            Path(w),
            Path(y),
            imgsz=self.imgsz_spin.value(),
            device=self.device_ctrl.device(),
        )
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.val_worker, "模型验证 (Val)")
        self.val_worker.log_line.connect(self.log_box.appendPlainText)
        self.val_worker.finished_metrics.connect(self._on_val_done)
        self.val_worker.failed.connect(self._on_failed)
        self.val_worker.start()

    def _on_val_done(self, metrics: dict[str, float]) -> None:
        self._set_busy(False)
        self.log_box.appendPlainText("\n=== 验证完成 ===")
        for k, v in metrics.items():
            self.log_box.appendPlainText(f"{k}: {v:.4f}")

    def _on_predict(self) -> None:
        """校验权重与测试输入路径后在后台启动 `PredictWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())
        w = self.weights_picker.text()
        src = self.source_edit.text().strip()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 .pt 权重文件。")
            return
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "提示", "请选择有效的测试输入文件或目录。")
            return

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "模型推理预览"):
            return

        out_dir = Path(self.pred_out_picker.text() or (self.workspace_root / "runs" / "predict_preview"))
        self._set_busy(True)
        self.log_box.clear()
        self.log_box.appendPlainText("开始推理测试...")

        self.pred_worker = PredictWorker(
            weights_path=Path(w),
            source_path=Path(src),
            output_dir=out_dir,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            imgsz=self.imgsz_spin.value(),
            device=self.device_ctrl.device(),
        )
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.pred_worker, "模型推理预览")
        self.pred_worker.log_line.connect(self.log_box.appendPlainText)
        self.pred_worker.finished_outputs.connect(self._on_predict_done)
        self.pred_worker.failed.connect(self._on_failed)
        self.pred_worker.start()

    def _on_predict_done(self, outputs: list[Path]) -> None:
        self._set_busy(False)
        self.log_box.appendPlainText(f"\n推理完成，共生成 {len(outputs)} 个结果文件。")
        for p in outputs[:20]:
            self.log_box.appendPlainText(f"- {p}")
        for p in outputs:
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                pix = load_pixmap_scaled(
                    p,
                    max(360, self.preview_label.width() - 12),
                    max(300, self.preview_label.height() - 12),
                )
                if pix:
                    self.preview_label.setPixmap(pix)
                break

    def _on_export(self) -> None:
        """校验 `.pt` 权重路径后按所选格式在后台启动 `ExportWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())
        w = self.weights_picker.text()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 .pt 权重文件。")
            return

        fmt = str(self.format_combo.currentData())
        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, f"模型导出 ({fmt})"):
            return

        self._set_busy(True)
        self.log_box.clear()
        self.log_box.appendPlainText(f"开始导出为 {fmt} 格式...")

        self.export_worker = ExportWorker(
            weights_path=Path(w),
            export_format=fmt,
            imgsz=self.imgsz_spin.value(),
            half=self.half_chk.isChecked() and self.device_ctrl.is_using_gpu(),
            device=self.device_ctrl.device(),
        )
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.export_worker, f"模型导出 ({fmt})")
        self.export_worker.log_line.connect(self.log_box.appendPlainText)
        self.export_worker.finished_path.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_failed)
        self.export_worker.start()

    def _on_export_done(self, exported_path: str) -> None:
        self._set_busy(False)
        self.log_box.appendPlainText(f"\n=== 导出成功 ===\n文件路径: {exported_path}")
        QMessageBox.information(self, "导出成功", f"模型已导出到:\n{exported_path}")

    def _on_failed(self, err: str) -> None:
        self._set_busy(False)
        self.log_box.appendPlainText(f"\n[错误] {err}")
        QMessageBox.critical(self, "执行错误", err)
