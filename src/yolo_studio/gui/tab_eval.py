"""标签页 5：模型验证、推理预览与导出界面。

职责：
    对训练好的 YOLO26 `.pt` 权重执行验证集指标评估（Val）、图片/目录/视频推理可视化预览，
    以及导出 `ONNX`、`TorchScript` 或 `TensorRT (.engine)` 部署模型。

运行环境与协作边界：
    运行于 PySide6 主线程；三类耗时任务分别委托给 `ValidateWorker`、`PredictWorker`、
    `ExportWorker`，并通过 `_set_busy()` 互斥锁住执行按钮，避免同一时刻并发占用模型或 GPU。
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.trainer import detect_compute_device
from yolo_studio.gui.common import PathPickerRow, load_pixmap_scaled, open_local_path
from yolo_studio.gui.workers import ExportWorker, PredictWorker, ValidateWorker


class EvalTab(QWidget):
    """模型验证、推理测试与部署导出标签页（标签页 5）。

    内部状态：
        持有硬件信息 `device_info` 以及三个互斥运行的后台工作线程引用
        `val_worker`、`pred_worker`、`export_worker`。
    """

    def __init__(self, workspace_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.device_info = detect_compute_device()
        self.val_worker: ValidateWorker | None = None
        self.pred_worker: PredictWorker | None = None
        self.export_worker: ExportWorker | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(10)

        # 分组 1：模型权重与验证配置
        model_group = QGroupBox("1. 模型权重与验证配置")
        m_form = QFormLayout(model_group)

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

        hw_row = QHBoxLayout()
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(128, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.device_edit = QLineEdit(self.device_info.recommended_device)
        self.device_edit.setFixedWidth(70)
        self.val_btn = QPushButton("运行验证 (Val)")
        hw_row.addWidget(QLabel("ImgSz:"))
        hw_row.addWidget(self.imgsz_spin)
        hw_row.addWidget(QLabel("Device:"))
        hw_row.addWidget(self.device_edit)
        hw_row.addWidget(self.val_btn)
        m_form.addRow("验证设置:", hw_row)
        left.addWidget(model_group)

        # 分组 2：推理效果测试
        pred_group = QGroupBox("2. 推理效果测试")
        p_form = QFormLayout(pred_group)

        src_row = QHBoxLayout()
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
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.predict_btn = QPushButton("运行推理预览")
        self.predict_btn.setObjectName("primaryBtn")
        self.open_pred_btn = QPushButton("打开输出目录")
        thr_row.addWidget(QLabel("Conf:"))
        thr_row.addWidget(self.conf_spin)
        thr_row.addWidget(QLabel("IoU:"))
        thr_row.addWidget(self.iou_spin)
        thr_row.addWidget(self.predict_btn)
        thr_row.addWidget(self.open_pred_btn)
        p_form.addRow("阈值与执行:", thr_row)
        left.addWidget(pred_group)

        # 分组 3：模型格式导出
        exp_group = QGroupBox("3. 模型导出 (部署格式)")
        e_form = QFormLayout(exp_group)

        exp_row = QHBoxLayout()
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
        left.addWidget(exp_group)

        self.log_box = QTextEdit()
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
            "媒体文件 (*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)",
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
        w = self.weights_picker.text()
        y = self.yaml_picker.text()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 .pt 权重文件。")
            return
        if not y or not Path(y).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 data.yaml 文件。")
            return

        self._set_busy(True)
        self.log_box.clear()
        self.log_box.append("开始执行模型验证...")

        self.val_worker = ValidateWorker(
            Path(w),
            Path(y),
            imgsz=self.imgsz_spin.value(),
            device=self.device_edit.text().strip() or "0",
        )
        self.val_worker.log_line.connect(self.log_box.append)
        self.val_worker.finished_metrics.connect(self._on_val_done)
        self.val_worker.failed.connect(self._on_failed)
        self.val_worker.start()

    def _on_val_done(self, metrics: dict[str, float]) -> None:
        self._set_busy(False)
        self.log_box.append("\n=== 验证完成 ===")
        for k, v in metrics.items():
            self.log_box.append(f"{k}: {v:.4f}")

    def _on_predict(self) -> None:
        """校验权重与测试输入路径后在后台启动 `PredictWorker`。"""
        w = self.weights_picker.text()
        src = self.source_edit.text().strip()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 .pt 权重文件。")
            return
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "提示", "请选择有效的测试输入文件或目录。")
            return

        out_dir = Path(self.pred_out_picker.text() or (self.workspace_root / "runs" / "predict_preview"))
        self._set_busy(True)
        self.log_box.clear()
        self.log_box.append("开始推理测试...")

        self.pred_worker = PredictWorker(
            weights_path=Path(w),
            source_path=Path(src),
            output_dir=out_dir,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            imgsz=self.imgsz_spin.value(),
            device=self.device_edit.text().strip() or "0",
        )
        self.pred_worker.log_line.connect(self.log_box.append)
        self.pred_worker.finished_outputs.connect(self._on_predict_done)
        self.pred_worker.failed.connect(self._on_failed)
        self.pred_worker.start()

    def _on_predict_done(self, outputs: list[Path]) -> None:
        self._set_busy(False)
        self.log_box.append(f"\n推理完成，共生成 {len(outputs)} 个结果文件。")
        for p in outputs[:20]:
            self.log_box.append(f"- {p}")
        for p in outputs:
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
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
        w = self.weights_picker.text()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 .pt 权重文件。")
            return

        fmt = str(self.format_combo.currentData())
        self._set_busy(True)
        self.log_box.clear()
        self.log_box.append(f"开始导出为 {fmt} 格式...")

        self.export_worker = ExportWorker(
            weights_path=Path(w),
            export_format=fmt,
            imgsz=self.imgsz_spin.value(),
            half=self.half_chk.isChecked(),
            device=self.device_edit.text().strip() or "0",
        )
        self.export_worker.log_line.connect(self.log_box.append)
        self.export_worker.finished_path.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_failed)
        self.export_worker.start()

    def _on_export_done(self, exported_path: str) -> None:
        self._set_busy(False)
        self.log_box.append(f"\n=== 导出成功 ===\n文件路径: {exported_path}")
        QMessageBox.information(self, "导出成功", f"模型已导出到:\n{exported_path}")

    def _on_failed(self, err: str) -> None:
        self._set_busy(False)
        self.log_box.append(f"\n[错误] {err}")
        QMessageBox.critical(self, "执行错误", err)
