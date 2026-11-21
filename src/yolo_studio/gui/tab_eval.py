"""标签页 6：模型验证、推理预览与导出界面。

职责：
    对训练好的 YOLO26 `.pt` 权重执行验证集指标评估（Val）、图片/目录/视频推理可视化预览，
    并在选择权重后自动后台只读解析训练输入尺寸、类别顺序与检测头能力以提示用户、同步尺寸，
    同时在独立导出面板中支持导出尺寸（方形/矩形）、`batch`、`dynamic`、`nms` 模式、`opset`、
    `simplify` 与 `FP16` 等完整部署导出参数配置与输入/输出张量实时预览。

运行环境与协作边界：
    运行于 PyQt6 主线程；权重元数据探测由 `WeightsInspectWorker` 在后台只读执行，
    验证、推理、导出三类重计算任务分别委托给 `ValidateWorker`、`PredictWorker`、`ExportWorker`，
    并通过 `_set_busy()` 与 `GpuTaskGuard` 互斥保护，避免并发占用 GPU 显存。
"""

from __future__ import annotations

import html
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.dataset_converter import IMAGE_EXTENSIONS
from yolo_studio.core.trainer import (
    DYNAMIC_EXPORT_FORMATS,
    EXPORT_NMS_KEY_TO_VALUE,
    EXPORT_NMS_MODES,
    EXPORT_OPSET_OPTIONS,
    ONNX_BACKED_EXPORT_FORMATS,
    DeviceInfo,
    ExportConfig,
    WeightsMetadata,
    describe_export_io,
    detect_compute_device,
    format_class_order,
    format_class_order_lines,
    format_imgsz_text,
    format_weights_metadata_log,
)
from yolo_studio.gui.common import (
    LOG_MAX_LINES,
    ComputeDeviceControl,
    GpuTaskGuard,
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    load_pixmap_scaled,
    open_local_path,
)
from yolo_studio.gui.workers import (
    ExportWorker,
    PredictWorker,
    ValidateWorker,
    WeightsInspectWorker,
)


def _show_class_order_dialog(parent: QWidget, meta: WeightsMetadata) -> None:
    """弹出对话框展示 `.pt` 权重的训练尺寸、检测头能力与完整有序类别清单，支持一键复制。"""
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"权重元数据与类别顺序 - {meta.weights_path.name}")
    dlg.resize(520, 440)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(8)

    head_desc = (
        "双分支检测头（支持 raw / end2end / embedded）"
        if meta.supports_end2end and meta.one2many_available
        else (
            "仅保留 one2one 端到端无 NMS 分支"
            if meta.supports_end2end
            else "标准 one2many 检测头"
        )
    )
    summary = QLabel(
        f"权重文件：{meta.weights_path.name}\n"
        f"任务类型：{meta.task or '未记录'}  |  "
        f"训练输入尺寸：{format_imgsz_text(meta.train_imgsz)}  |  "
        f"类别总数：{meta.num_classes}\n"
        f"检测头能力：{head_desc}"
    )
    summary.setWordWrap(True)
    layout.addWidget(summary)

    text_box = QPlainTextEdit()
    text_box.setReadOnly(True)
    text_box.setPlainText(format_class_order_lines(meta.class_names))
    layout.addWidget(text_box, 1)

    btn_row = QHBoxLayout()
    copy_indexed_btn = QPushButton("复制编号清单 (0: name)")
    copy_plain_btn = QPushButton("复制纯类别名 (classes.txt)")
    close_btn = QPushButton("关闭")

    def _copy_text(content: str, label: str) -> None:
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(content)
            QMessageBox.information(dlg, "已复制", f"已将{label}复制到剪贴板。")

    copy_indexed_btn.clicked.connect(
        lambda: _copy_text(format_class_order_lines(meta.class_names), "编号类别顺序")
    )
    copy_plain_btn.clicked.connect(
        lambda: _copy_text(
            "\n".join(name for _, name in meta.class_names), "纯类别名（每行一个）"
        )
    )
    close_btn.clicked.connect(dlg.accept)

    btn_row.addWidget(copy_indexed_btn)
    btn_row.addWidget(copy_plain_btn)
    btn_row.addStretch(1)
    btn_row.addWidget(close_btn)
    layout.addLayout(btn_row)

    dlg.exec()


class EvalTab(QWidget):
    """模型验证、推理测试与部署导出标签页（标签页 6）。

    内部状态与并发边界：
        - 持有共享硬件快照 `device_info`、跨页计算任务互斥协调器 `gpu_guard`；
        - 持有验证/推理/导出三个互斥后台线程（`val_worker`、`pred_worker`、`export_worker`）；
        - 持有只读权重元数据探测线程 `inspect_worker`（及尚未自然退出的旧探测线程列表
          `_retired_inspect_workers`），通过单调递增 `_inspect_request_id` 丢弃过期回调，
          并用 `_inspected_file_sig`（路径 + 修改时间 + 大小）缓存避免重复反序列化同一权重。
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
        self.inspect_worker: WeightsInspectWorker | None = None
        self._retired_inspect_workers: list[WeightsInspectWorker] = []

        self._weights_metadata: WeightsMetadata | None = None
        self._inspected_file_sig: tuple[str, int, int] | None = None
        self._inspect_request_id: int = 0
        self._syncing_export_dims: bool = False
        self._busy: bool = False

        # 用户在路径输入框手动键入时的短防抖定时器（180ms），程序写入路径时直接立即探测
        self._inspect_timer = QTimer(self)
        self._inspect_timer.setSingleShot(True)
        self._inspect_timer.setInterval(180)
        self._inspect_timer.timeout.connect(self._trigger_weights_inspect)

        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(10)

        # 分组 1：模型权重、自动元数据提示、运行设备与验证设置
        model_group = QGroupBox("1. 模型权重与运行设备")
        self._model_form = QFormLayout(model_group)
        configure_form_layout(self._model_form)

        self.weights_picker = PathPickerRow(
            placeholder="选择训练好的 best.pt 权重（自动读取训练尺寸与类别顺序）",
            mode="file",
            file_filter="PyTorch 权重 (*.pt);;所有文件 (*.*)",
        )
        self._model_form.addRow("模型权重 (.pt):", self.weights_picker)

        meta_row = QHBoxLayout()
        meta_row.setSpacing(6)
        self.weights_info_label = QLabel("选择 .pt 权重后将自动读取训练尺寸与类别顺序")
        self.weights_info_label.setWordWrap(True)
        self.weights_info_label.setStyleSheet("color: #666666;")
        self.view_classes_btn = QPushButton("类别清单")
        self.view_classes_btn.setEnabled(False)
        self.view_classes_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.view_classes_btn.setToolTip("查看并复制 .pt 权重记录的完整类别索引顺序")
        meta_row.addWidget(self.weights_info_label, 1)
        meta_row.addWidget(self.view_classes_btn)
        self._model_form.addRow("权重信息:", meta_row)

        self.yaml_picker = PathPickerRow(
            placeholder="选择数据集 data.yaml（用于验证指标评估）",
            mode="file",
            file_filter="YAML 文件 (*.yaml *.yml);;所有文件 (*.*)",
        )
        self._model_form.addRow("data.yaml:", self.yaml_picker)

        self.device_ctrl = ComputeDeviceControl(
            self.device_info,
            show_status_text=True,
            allow_multi_gpu=False,
            checkbox_text="启用 GPU 加速",
            compact_status=True,
        )
        self._model_form.addRow("运行设备:", self.device_ctrl)

        hw_row = QHBoxLayout()
        hw_row.setSpacing(6)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.imgsz_spin.setToolTip(
            "用于『运行验证 (Val)』与『推理效果测试』的输入尺寸；模型导出尺寸请在下方第 3 组独立设置"
        )
        self.val_btn = QPushButton("运行验证 (Val)")
        hw_row.addWidget(QLabel("ImgSz:"))
        hw_row.addWidget(self.imgsz_spin)
        hw_row.addStretch(1)
        hw_row.addWidget(self.val_btn)
        self._model_form.addRow("验证/推理:", hw_row)
        left.addWidget(model_group)

        # 分组 2：推理效果测试
        pred_group = QGroupBox("2. 推理效果测试")
        self._pred_form = QFormLayout(pred_group)
        configure_form_layout(self._pred_form)

        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("选择测试图片、图片文件夹或测试视频")
        self.pick_file_btn = QPushButton("选图片/视频")
        self.pick_dir_btn = QPushButton("选文件夹")
        src_row.addWidget(self.source_edit, 1)
        src_row.addWidget(self.pick_file_btn)
        src_row.addWidget(self.pick_dir_btn)
        self._pred_form.addRow("测试输入:", src_row)

        default_pred_out = str((self.workspace_root / "runs" / "predict_preview").resolve())
        self.pred_out_picker = PathPickerRow(
            placeholder="推理结果输出目录", mode="dir", default_path=default_pred_out
        )
        self._pred_form.addRow("输出目录:", self.pred_out_picker)

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
        self._pred_form.addRow("阈值与执行:", thr_row)
        left.addWidget(pred_group)

        # 分组 3：模型格式导出（独立导出尺寸、batch、dynamic、NMS 模式、opset、simplify 与 IO 预览）
        exp_group = QGroupBox("3. 模型导出 (部署格式)")
        self._exp_form = QFormLayout(exp_group)
        configure_form_layout(self._exp_form)

        # 行 3.1：导出格式、FP16 半精度与 ONNX 图简化
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        self.format_combo = QComboBox()
        self.format_combo.addItem("ONNX (通用推理推荐)", "onnx")
        self.format_combo.addItem("TorchScript (.torchscript)", "torchscript")
        self.format_combo.addItem("TensorRT Engine (.engine, 需本机 TensorRT)", "engine")
        self.half_chk = QCheckBox("FP16 半精度 (half)")
        self.simplify_chk = QCheckBox("简化图 (simplify)")
        self.simplify_chk.setChecked(True)
        self.simplify_chk.setToolTip("导出 ONNX/TensorRT 时使用 onnxslim 简化中间计算图（推荐开启）")
        fmt_row.addWidget(self.format_combo, 1)
        fmt_row.addWidget(self.half_chk)
        fmt_row.addWidget(self.simplify_chk)
        self._exp_form.addRow("导出格式:", fmt_row)

        # 行 3.2：独立导出输入尺寸（支持方形联动与矩形高×宽，以及一键同步权重训练尺寸）
        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        self.export_h_spin = QSpinBox()
        self.export_h_spin.setRange(64, 4096)
        self.export_h_spin.setSingleStep(32)
        self.export_h_spin.setValue(640)
        self.export_h_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_h_spin.setToolTip("导出模型输入图像高度 H（像素，建议为 32 的倍数）")

        self.export_w_spin = QSpinBox()
        self.export_w_spin.setRange(64, 4096)
        self.export_w_spin.setSingleStep(32)
        self.export_w_spin.setValue(640)
        self.export_w_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_w_spin.setToolTip("导出模型输入图像宽度 W（像素，取消『方形联动』后可设矩形如 544×960）")

        self.export_square_chk = QCheckBox("方形联动")
        self.export_square_chk.setChecked(True)
        self.export_square_chk.setToolTip("勾选时宽(W)自动跟随高(H)；取消勾选可导出非正方形输入（如 544×960）")

        self.sync_train_imgsz_btn = QPushButton("用训练尺寸")
        self.sync_train_imgsz_btn.setEnabled(False)
        self.sync_train_imgsz_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.sync_train_imgsz_btn.setToolTip("选择 .pt 权重并读取到训练尺寸后，可一键恢复为权重训练尺寸")

        size_row.addWidget(QLabel("高(H):"))
        size_row.addWidget(self.export_h_spin)
        size_row.addSpacing(6)
        size_row.addWidget(QLabel("宽(W):"))
        size_row.addWidget(self.export_w_spin)
        size_row.addSpacing(6)
        size_row.addWidget(self.export_square_chk)
        size_row.addStretch(1)
        size_row.addWidget(self.sync_train_imgsz_btn)
        self._exp_form.addRow("导出尺寸:", size_row)

        # 行 3.3：Batch、Dynamic 与 ONNX Opset
        batch_row = QHBoxLayout()
        batch_row.setSpacing(6)
        self.export_batch_spin = QSpinBox()
        self.export_batch_spin.setRange(1, 128)
        self.export_batch_spin.setValue(1)
        self.export_batch_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_batch_spin.setToolTip("静态批大小；TensorRT 开启动态输入时建议设为预期最大批大小")

        self.dynamic_chk = QCheckBox("动态尺寸/批次 (dynamic)")
        self.dynamic_chk.setToolTip(
            "开启后导出的输入维度为 (batch, 3, height, width)，支持推理时传入不同批大小与分辨率（32 的倍数）"
        )

        self.export_opset_combo = QComboBox()
        for label, ver in EXPORT_OPSET_OPTIONS:
            self.export_opset_combo.addItem(label, ver)
        self.export_opset_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_opset_combo.setToolTip("指定 ONNX Opset 版本（留默认『自动』则由 Ultralytics 择优选择）")

        batch_row.addWidget(QLabel("Batch:"))
        batch_row.addWidget(self.export_batch_spin)
        batch_row.addSpacing(10)
        batch_row.addWidget(self.dynamic_chk)
        batch_row.addSpacing(10)
        batch_row.addWidget(QLabel("Opset:"))
        batch_row.addWidget(self.export_opset_combo)
        batch_row.addStretch(1)
        self._exp_form.addRow("批次与算子:", batch_row)

        # 行 3.4：NMS 后处理输出模式与 MaxDet
        nms_row = QHBoxLayout()
        nms_row.setSpacing(6)
        self.nms_mode_combo = QComboBox()
        for key, label, _nms_val in EXPORT_NMS_MODES:
            self.nms_mode_combo.addItem(label, key)
        self.nms_mode_combo.setToolTip(
            "raw(nms=None): 原始输出 [B, 4+nc, anchors]，需推理端外部 NMS；\n"
            "end2end(nms=False): YOLO26 原生无 NMS 端到端头，输出 [B, max_det, 6]；\n"
            "embedded(nms=True): 在计算图内嵌入 NMS，输出 [B, max_det, 6]。"
        )

        self.export_max_det_spin = QSpinBox()
        self.export_max_det_spin.setRange(1, 3000)
        self.export_max_det_spin.setSingleStep(50)
        self.export_max_det_spin.setValue(300)
        self.export_max_det_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_max_det_spin.setToolTip("端到端或图内嵌入 NMS 模式下单图最大输出检测框数量")

        nms_row.addWidget(self.nms_mode_combo, 1)
        nms_row.addSpacing(8)
        nms_row.addWidget(QLabel("MaxDet:"))
        nms_row.addWidget(self.export_max_det_spin)
        self._exp_form.addRow("NMS 模式:", nms_row)

        # 行 3.5：图内嵌入 NMS 专属阈值行（仅选择 embedded 时展开）
        self.nms_thresh_row_widget = QWidget()
        nms_thr_layout = QHBoxLayout(self.nms_thresh_row_widget)
        nms_thr_layout.setContentsMargins(0, 0, 0, 0)
        nms_thr_layout.setSpacing(6)
        self.export_nms_conf_spin = QDoubleSpinBox()
        self.export_nms_conf_spin.setRange(0.01, 0.99)
        self.export_nms_conf_spin.setSingleStep(0.05)
        self.export_nms_conf_spin.setValue(0.25)
        self.export_nms_conf_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_nms_conf_spin.setToolTip("图内嵌入 NMS 的置信度过滤阈值（仅 embedded 模式生效）")

        self.export_nms_iou_spin = QDoubleSpinBox()
        self.export_nms_iou_spin.setRange(0.05, 0.99)
        self.export_nms_iou_spin.setSingleStep(0.05)
        self.export_nms_iou_spin.setValue(0.50)
        self.export_nms_iou_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_nms_iou_spin.setToolTip("图内嵌入 NMS 的 IoU 去重阈值（仅 embedded 模式生效）")

        nms_thr_layout.addWidget(QLabel("NMS Conf:"))
        nms_thr_layout.addWidget(self.export_nms_conf_spin)
        nms_thr_layout.addSpacing(12)
        nms_thr_layout.addWidget(QLabel("NMS IoU:"))
        nms_thr_layout.addWidget(self.export_nms_iou_spin)
        nms_thr_layout.addStretch(1)
        self._exp_form.addRow("嵌入阈值:", self.nms_thresh_row_widget)

        # 行 3.6：实时预期输入/输出张量说明与「导出模型」执行按钮
        io_row = QHBoxLayout()
        io_row.setSpacing(8)
        self.export_io_label = QLabel()
        self.export_io_label.setWordWrap(True)
        self.export_io_label.setTextFormat(Qt.TextFormat.RichText)
        self.export_btn = QPushButton("导出模型")
        self.export_btn.setObjectName("primaryBtn")
        self.export_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        io_row.addWidget(self.export_io_label, 1)
        io_row.addWidget(self.export_btn)
        self._exp_form.addRow("输入/输出:", io_row)

        align_form_labels([self._model_form, self._pred_form, self._exp_form])
        left.addWidget(exp_group)

        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("权重元数据摘要、验证指标、推理结果路径与导出日志将在此显示...")
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
        self.weights_picker.textChanged.connect(self._on_weights_text_changed)
        self.view_classes_btn.clicked.connect(self._on_view_classes)
        self.pick_file_btn.clicked.connect(self._on_pick_file)
        self.pick_dir_btn.clicked.connect(self._on_pick_dir)
        self.val_btn.clicked.connect(self._on_validate)
        self.predict_btn.clicked.connect(self._on_predict)
        self.open_pred_btn.clicked.connect(lambda: open_local_path(self.pred_out_picker.path()))
        self.export_btn.clicked.connect(self._on_export)

        self.format_combo.currentIndexChanged.connect(self._on_export_format_changed)
        self.device_ctrl.device_changed.connect(self._on_export_format_changed)
        self.half_chk.toggled.connect(self._refresh_export_io_preview)
        self.simplify_chk.toggled.connect(self._refresh_export_io_preview)
        self.export_h_spin.valueChanged.connect(self._on_export_h_changed)
        self.export_w_spin.valueChanged.connect(self._refresh_export_io_preview)
        self.export_square_chk.toggled.connect(self._on_export_square_toggled)
        self.sync_train_imgsz_btn.clicked.connect(self._on_sync_train_imgsz)
        self.export_batch_spin.valueChanged.connect(self._refresh_export_io_preview)
        self.dynamic_chk.toggled.connect(self._on_dynamic_toggled)
        self.export_opset_combo.currentIndexChanged.connect(self._refresh_export_io_preview)
        self.nms_mode_combo.currentIndexChanged.connect(self._on_export_format_changed)
        self.export_max_det_spin.valueChanged.connect(self._refresh_export_io_preview)
        self.export_nms_conf_spin.valueChanged.connect(self._refresh_export_io_preview)
        self.export_nms_iou_spin.valueChanged.connect(self._refresh_export_io_preview)

        self._on_export_square_toggled(self.export_square_chk.isChecked())
        self._on_export_format_changed()

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭标签页时停止防抖定时器，避免关闭后触发新的权重探测。"""
        self._inspect_timer.stop()
        super().closeEvent(event)

    def running_inspect_workers(self) -> list[WeightsInspectWorker]:
        """返回仍在后台运行的权重元数据探测线程列表，供主窗口关闭前安全等待。"""
        workers: list[WeightsInspectWorker] = []
        if self.inspect_worker is not None and self.inspect_worker.isRunning():
            workers.append(self.inspect_worker)
        for old in self._retired_inspect_workers:
            if old.isRunning():
                workers.append(old)
        return workers

    def current_weights_metadata(self) -> WeightsMetadata | None:
        """返回当前已成功读取的 `.pt` 权重元数据快照；尚未加载或读取失败时返回 `None`。"""
        return self._weights_metadata

    def _on_weights_text_changed(self, _text: str) -> None:
        """权重路径变化后通过短防抖调度只读元数据探测，防止手输路径过程中频繁启动线程。"""
        self._inspect_timer.start()

    def _clear_weights_metadata(self, message: str, *, color: str = "#666666") -> None:
        self._weights_metadata = None
        self._inspected_file_sig = None
        self.weights_info_label.setText(message)
        self.weights_info_label.setStyleSheet(f"color: {color};")
        self.weights_info_label.setToolTip(message)
        self.view_classes_btn.setEnabled(False)
        self.sync_train_imgsz_btn.setEnabled(False)
        self.sync_train_imgsz_btn.setToolTip("选择 .pt 权重并读取到训练尺寸后，可一键恢复为权重训练尺寸")
        self._refresh_export_io_preview()

    def _trigger_weights_inspect(self) -> None:
        """校验当前权重路径并在后台启动 `WeightsInspectWorker` 读取训练尺寸与类别顺序。"""
        self._inspect_timer.stop()
        raw = self.weights_picker.text()
        if not raw:
            self._inspect_request_id += 1
            self._clear_weights_metadata("选择 .pt 权重后将自动读取训练尺寸与类别顺序")
            return

        path = Path(raw)
        if not path.is_file():
            self._inspect_request_id += 1
            self._clear_weights_metadata("未找到有效的 .pt 权重文件", color="#b25900")
            return

        try:
            st = path.stat()
            sig = (str(path.resolve()), st.st_mtime_ns, st.st_size)
        except OSError:
            sig = (str(path), 0, 0)

        if self._inspected_file_sig == sig and self._weights_metadata is not None:
            return

        self._inspect_request_id += 1
        req_id = self._inspect_request_id
        self._weights_metadata = None
        self._inspected_file_sig = None
        self.view_classes_btn.setEnabled(False)
        self.sync_train_imgsz_btn.setEnabled(False)
        self.weights_info_label.setText("正在后台读取权重元数据（训练尺寸与类别顺序）...")
        self.weights_info_label.setStyleSheet("color: #0969da;")
        self.weights_info_label.setToolTip("正在读取 .pt 权重文件头信息...")
        self._refresh_export_io_preview()

        if self.inspect_worker is not None and self.inspect_worker.isRunning():
            self._retired_inspect_workers.append(self.inspect_worker)

        worker = WeightsInspectWorker(path, request_id=req_id)
        self.inspect_worker = worker
        worker.finished_metadata.connect(
            lambda rid, meta, file_sig=sig: self._on_weights_inspected(rid, meta, file_sig)
        )
        worker.failed.connect(self._on_weights_inspect_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_inspect_worker(w))
        worker.start()

    def _cleanup_inspect_worker(self, worker: WeightsInspectWorker) -> None:
        if worker in self._retired_inspect_workers:
            self._retired_inspect_workers.remove(worker)

    def _apply_train_imgsz_to_controls(self, train_imgsz: tuple[int, int]) -> None:
        """将权重记录的 `(高, 宽)` 同步写入验证/推理 `ImgSz` 与独立导出尺寸控件。"""
        h, w = train_imgsz
        clamped_val = max(self.imgsz_spin.minimum(), min(self.imgsz_spin.maximum(), max(h, w)))
        self.imgsz_spin.setValue(clamped_val)

        clamped_h = max(self.export_h_spin.minimum(), min(self.export_h_spin.maximum(), h))
        clamped_w = max(self.export_w_spin.minimum(), min(self.export_w_spin.maximum(), w))
        self._syncing_export_dims = True
        try:
            is_square = clamped_h == clamped_w
            self.export_square_chk.setChecked(is_square)
            self.export_h_spin.setValue(clamped_h)
            self.export_w_spin.setValue(clamped_w)
            self.export_w_spin.setEnabled(not is_square)
        finally:
            self._syncing_export_dims = False

    def _on_weights_inspected(
        self, request_id: int, meta: WeightsMetadata, file_sig: tuple[str, int, int]
    ) -> None:
        """接收权重元数据探测结果：更新界面提示、自动同步训练尺寸并在日志区输出完整摘要。"""
        if request_id != self._inspect_request_id:
            return

        self._weights_metadata = meta
        self._inspected_file_sig = file_sig
        self.view_classes_btn.setEnabled(meta.num_classes > 0)

        cls_brief = format_class_order(meta.class_names, max_items=5)
        if meta.train_imgsz is not None:
            self._apply_train_imgsz_to_controls(meta.train_imgsz)
            size_text = format_imgsz_text(meta.train_imgsz)
            self.sync_train_imgsz_btn.setEnabled(True)
            self.sync_train_imgsz_btn.setToolTip(f"一键恢复导出尺寸为权重训练尺寸: {size_text}")
            self.weights_info_label.setText(
                f"训练尺寸: {size_text}（已自动同步） | 类别({meta.num_classes}): {cls_brief}"
            )
            self.weights_info_label.setStyleSheet("color: #1f7a1f; font-weight: 500;")
        else:
            self.sync_train_imgsz_btn.setEnabled(False)
            self.sync_train_imgsz_btn.setToolTip("当前权重未记录训练尺寸，无法自动同步")
            self.weights_info_label.setText(
                f"训练尺寸: 未记录（请手动确认尺寸） | 类别({meta.num_classes}): {cls_brief}"
            )
            self.weights_info_label.setStyleSheet("color: #b25900; font-weight: 500;")

        tooltip_lines = [
            f"权重文件: {meta.weights_path.name}",
            f"任务类型: {meta.task or '未记录'}",
            f"训练输入尺寸: {format_imgsz_text(meta.train_imgsz)}",
            f"类别总数: {meta.num_classes}",
            "完整类别顺序:",
            format_class_order_lines(meta.class_names),
        ]
        self.weights_info_label.setToolTip("\n".join(tooltip_lines))

        self.log_box.appendPlainText(format_weights_metadata_log(meta))
        if meta.train_imgsz is not None:
            self.log_box.appendPlainText(
                f"  [自动同步] 已将验证/推理尺寸与导出尺寸同步为权重训练尺寸: {format_imgsz_text(meta.train_imgsz)}"
            )
        else:
            self.log_box.appendPlainText(
                "  [提示] 该权重未记录训练输入尺寸，请手动确认验证与导出尺寸。"
            )
        self._refresh_export_io_preview()

    def _on_weights_inspect_failed(self, request_id: int, err: str) -> None:
        if request_id != self._inspect_request_id:
            return
        self._clear_weights_metadata(f"未能自动解析权重元数据：{err}", color="#b25900")
        self.log_box.appendPlainText(f"[提示] 读取权重元数据失败: {err}")

    def _on_view_classes(self) -> None:
        if self._weights_metadata is not None:
            _show_class_order_dialog(self, self._weights_metadata)

    def _on_sync_train_imgsz(self) -> None:
        """一键将导出尺寸恢复为当前 `.pt` 权重记录的训练尺寸。"""
        if self._weights_metadata is None or self._weights_metadata.train_imgsz is None:
            return
        h, w = self._weights_metadata.train_imgsz
        clamped_h = max(self.export_h_spin.minimum(), min(self.export_h_spin.maximum(), h))
        clamped_w = max(self.export_w_spin.minimum(), min(self.export_w_spin.maximum(), w))
        self._syncing_export_dims = True
        try:
            is_square = clamped_h == clamped_w
            self.export_square_chk.setChecked(is_square)
            self.export_h_spin.setValue(clamped_h)
            self.export_w_spin.setValue(clamped_w)
            self.export_w_spin.setEnabled(not is_square)
        finally:
            self._syncing_export_dims = False
        self._refresh_export_io_preview()

    def _on_export_square_toggled(self, checked: bool) -> None:
        """切换方形联动开关：勾选时宽自动对齐高并禁用编辑，取消勾选即可设非正方形尺寸。"""
        self.export_w_spin.setEnabled(not checked)
        if checked and not self._syncing_export_dims:
            self._syncing_export_dims = True
            try:
                self.export_w_spin.setValue(self.export_h_spin.value())
            finally:
                self._syncing_export_dims = False
        self._refresh_export_io_preview()

    def _on_export_h_changed(self, value: int) -> None:
        if self.export_square_chk.isChecked() and not self._syncing_export_dims:
            self._syncing_export_dims = True
            try:
                self.export_w_spin.setValue(value)
            finally:
                self._syncing_export_dims = False
        self._refresh_export_io_preview()

    def _on_dynamic_toggled(self, _checked: bool) -> None:
        self._refresh_export_batch_tooltip()
        self._refresh_export_io_preview()

    def _refresh_export_batch_tooltip(self) -> None:
        """按导出格式与动态输入开关刷新 `Batch` 微调框的语义提示。"""
        fmt = str(self.format_combo.currentData() or "onnx")
        is_dynamic = self.dynamic_chk.isChecked() and self.dynamic_chk.isEnabled()
        if fmt == "engine" and is_dynamic:
            self.export_batch_spin.setToolTip(
                "TensorRT 开启动态输入时，Batch 表示推理期允许的最大批大小（建议设为 8 或 16）"
            )
        elif fmt == "onnx" and is_dynamic:
            self.export_batch_spin.setToolTip(
                "ONNX 开启动态输入后导出维度为动态 batch，此数值仅作导出追踪参考"
            )
        else:
            self.export_batch_spin.setToolTip("静态批大小；TensorRT 开启动态输入时建议设为预期最大批大小")

    def _refresh_export_btn_state(self) -> None:
        """同步「导出模型」按钮可用态：忙碌期间或 TensorRT 处于 CPU 模式时置灰防误点。"""
        fmt = str(self.format_combo.currentData() or "onnx")
        using_gpu = self.device_ctrl.is_using_gpu() and not self.device_info.probing
        blocked_engine_cpu = fmt == "engine" and not using_gpu
        self.export_btn.setEnabled((not self._busy) and (not blocked_engine_cpu))
        if blocked_engine_cpu:
            self.export_btn.setToolTip(
                "TensorRT (.engine) 导出需要可用的 CUDA GPU，请在上方运行设备中启用 GPU 加速后再导出"
            )
        else:
            self.export_btn.setToolTip("按当前配置导出部署模型（产物生成在源权重同目录）")

    def _on_export_format_changed(self, *_args: object) -> None:
        """同步导出面板各项防呆状态：FP16、ONNX 专属参数（opset/simplify）、dynamic、NMS 阈值及导出按钮。"""
        fmt = str(self.format_combo.currentData() or "onnx")
        is_onnx = fmt == "onnx"
        is_onnx_backed = fmt in ONNX_BACKED_EXPORT_FORMATS
        supports_dynamic = fmt in DYNAMIC_EXPORT_FORMATS
        using_gpu = self.device_ctrl.is_using_gpu() and not self.device_info.probing

        # 1) FP16 半精度联动
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

        # 2) ONNX 中间图专属选项（simplify、opset）
        self.simplify_chk.setEnabled(is_onnx_backed)
        self.export_opset_combo.setEnabled(is_onnx_backed)
        if is_onnx_backed:
            self.simplify_chk.setToolTip("导出 ONNX/TensorRT 时使用 onnxslim 简化中间计算图（推荐开启）")
            self.export_opset_combo.setToolTip(
                "指定 ONNX Opset 版本（默认『自动』由 Ultralytics 择优选择）"
            )
        else:
            self.simplify_chk.setToolTip(f"当前导出格式 ({fmt}) 不使用 ONNX simplify 选项")
            self.export_opset_combo.setToolTip(f"当前导出格式 ({fmt}) 不使用 ONNX opset 选项")

        # 3) dynamic 动态尺寸支持联动
        self.dynamic_chk.setEnabled(supports_dynamic)
        if not supports_dynamic:
            self.dynamic_chk.setChecked(False)

        # 4) NMS 模式联动：MaxDet 与嵌入 NMS 阈值行
        nms_key = str(self.nms_mode_combo.currentData() or "raw")
        uses_max_det = nms_key in {"end2end", "embedded"}
        is_embedded = nms_key == "embedded"
        self.export_max_det_spin.setEnabled(uses_max_det)
        if uses_max_det:
            self.export_max_det_spin.setToolTip("端到端或图内嵌入 NMS 模式下单图最大输出检测框数量")
        else:
            self.export_max_det_spin.setToolTip("原始输出模式 (raw) 输出全部候选框，不使用 MaxDet")
        self._exp_form.setRowVisible(self.nms_thresh_row_widget, is_embedded)

        self._refresh_export_batch_tooltip()
        self._refresh_export_btn_state()
        self._refresh_export_io_preview()

    def build_export_config(self, *, allow_missing_weights: bool = False) -> ExportConfig:
        """从界面导出面板读取并组装完整的 `ExportConfig` 配置对象。"""
        w_text = self.weights_picker.text()
        if not allow_missing_weights and (not w_text or not Path(w_text).is_file()):
            raise ValueError("请选择有效的 .pt 权重文件。")

        fmt = str(self.format_combo.currentData() or "onnx")
        h = self.export_h_spin.value()
        w = h if self.export_square_chk.isChecked() else self.export_w_spin.value()
        imgsz: int | tuple[int, int] = h if h == w else (h, w)

        nms_key = str(self.nms_mode_combo.currentData() or "raw")
        nms_val = EXPORT_NMS_KEY_TO_VALUE.get(nms_key, None)

        opset_data = self.export_opset_combo.currentData()
        opset = int(opset_data) if (opset_data is not None and self.export_opset_combo.isEnabled()) else None

        conf_val = self.export_nms_conf_spin.value() if nms_val is True else None
        iou_val = self.export_nms_iou_spin.value() if nms_val is True else None

        return ExportConfig(
            weights_path=Path(w_text) if w_text else Path("model.pt"),
            export_format=fmt,
            imgsz=imgsz,
            batch=self.export_batch_spin.value(),
            dynamic=self.dynamic_chk.isChecked() and self.dynamic_chk.isEnabled(),
            nms=nms_val,
            max_det=self.export_max_det_spin.value(),
            conf=conf_val,
            iou=iou_val,
            opset=opset,
            simplify=self.simplify_chk.isChecked() if self.simplify_chk.isEnabled() else True,
            half=self.half_chk.isChecked() and self.half_chk.isEnabled() and self.device_ctrl.is_using_gpu(),
            device=self.device_ctrl.device(),
        )

    def _refresh_export_io_preview(self, *_args: object) -> None:
        """根据当前导出面板控件值与已读取权重元数据，实时刷新输入/输出张量说明与风险提醒。"""
        cfg = self.build_export_config(allow_missing_weights=True)
        io_text, warnings = describe_export_io(cfg, metadata=self._weights_metadata)

        html_parts = [
            f"<span style='color:#333333;'>{html.escape(io_text)}</span>",
        ]
        for warn in warnings[:2]:
            html_parts.append(
                f"<br><span style='color:#b25900; font-weight:500;'>[提示] {html.escape(warn)}</span>"
            )
        if len(warnings) > 2:
            html_parts.append(
                f"<br><span style='color:#b25900;'>（另有 {len(warnings) - 2} 条提示，悬停查看全部）</span>"
            )
        self.export_io_label.setText("".join(html_parts))

        tip_lines = [io_text, *(f"[提示] {w}" for w in warnings)]
        self.export_io_label.setToolTip("\n".join(tip_lines))

    def apply_device_info(self, info: DeviceInfo) -> None:
        """异步硬件探测完成后刷新运行设备开关与导出 FP16 可用状态（保留用户已手动修改的选择）。"""
        self.device_info = info
        self.device_ctrl.apply_device_info(info)
        self._on_export_format_changed()

    def set_weights_and_yaml(self, weights_path: str, yaml_path: str) -> None:
        """跨页写入：同步更新待评估权重与 `data.yaml` 路径，并立即触发权重元数据探测。"""
        if weights_path:
            self.weights_picker.setText(weights_path)
            self._trigger_weights_inspect()
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
        self._busy = busy
        self.val_btn.setEnabled(not busy)
        self.predict_btn.setEnabled(not busy)
        self._refresh_export_btn_state()

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
        """校验 `.pt` 权重与独立导出参数后在后台启动 `ExportWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())

        try:
            cfg = self.build_export_config(allow_missing_weights=False)
        except ValueError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return

        if cfg.export_format == "engine" and cfg.device.strip().lower() == "cpu":
            QMessageBox.warning(
                self,
                "提示",
                "TensorRT (.engine) 导出需要可用的 CUDA GPU，请在上方运行设备中启用 GPU 加速后再导出。",
            )
            return

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(
            self, f"模型导出 ({cfg.export_format})"
        ):
            return

        self._set_busy(True)
        self.log_box.clear()
        self.log_box.appendPlainText(f"开始导出为 {cfg.export_format} 格式...")

        self.export_worker = ExportWorker(cfg)
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.export_worker, f"模型导出 ({cfg.export_format})")
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
