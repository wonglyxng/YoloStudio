"""标签页 6：模型验证、推理预览与导出界面。

职责：
    对训练好的 YOLO26 `.pt` 权重或已导出部署模型（`.onnx` / `.torchscript` / `.engine`）
    执行验证集指标评估（Val）与图片/目录/视频推理可视化预览，并在选择模型后自动后台只读解析
    尺寸、类别顺序、动态维度与检测头能力以提示用户和自动联动；
    支持在推理测试中选择检测头模式（`one2many` / `end2end`）、输入填充方式（最小矩形 / 固定方形 /
    按导出尺寸固定填充）及预览绘制样式（紧凑防重叠半透明 / Ultralytics 默认、框线挡位、类别与置信度开关），
    提供「对齐导出参数」、「测刚导出模型 / 切回 .pt 权重」快捷联动及结果图放大查看能力；
    同时在独立导出面板中支持导出尺寸（方形/矩形）、`batch`、`dynamic`、`nms` 模式、`opset`、
    `simplify` 与 `FP16` 等完整部署导出参数配置与输入/输出张量实时预览。

运行环境与协作边界：
    运行于 PyQt6 主线程；模型元数据探测由 `WeightsInspectWorker` 在后台只读执行，
    验证、推理、导出三类重计算任务分别委托给 `ValidateWorker`、`PredictWorker`、`ExportWorker`，
    并通过 `_set_busy()` 与 `GpuTaskGuard` 互斥保护，避免并发占用 GPU 显存。
"""

from __future__ import annotations

import html
from pathlib import Path

from PyQt6.QtCore import QSize, QTimer, Qt, pyqtSignal as Signal
from PyQt6.QtGui import QCloseEvent, QMouseEvent, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.dataset_converter import IMAGE_EXTENSIONS
from yolo_studio.core.trainer import (
    DEFAULT_EXPORT_PADDING_VALUE,
    DEFAULT_EXPORT_PREPROCESS_MODE,
    DEFAULT_PREDICT_LINE_WIDTH_MODE,
    DEFAULT_PREDICT_RENDER_STYLE,
    DYNAMIC_EXPORT_FORMATS,
    EXPORT_NMS_KEY_TO_VALUE,
    EXPORT_NMS_MODES,
    EXPORT_OPSET_OPTIONS,
    EXPORT_PREPROCESS_MODES,
    ONNX_BACKED_EXPORT_FORMATS,
    ONNX_DEPLOY_SCHEMA_VERSION,
    PREDICT_HEAD_KEY_TO_NMS,
    PREDICT_HEAD_MODES,
    PREDICT_LINE_WIDTH_MODES,
    PREDICT_PAD_MODES,
    PREDICT_RENDER_STYLES,
    DeviceInfo,
    ExportConfig,
    PredictRenderConfig,
    WeightsMetadata,
    describe_export_io,
    detect_compute_device,
    detect_model_format,
    format_class_order,
    format_class_order_lines,
    format_imgsz_text,
    format_predict_render_summary,
    format_preprocess_summary,
    format_weights_metadata_log,
    is_exported_model_path,
)
from yolo_studio.gui.common import (
    LOG_MAX_LINES,
    ComputeDeviceControl,
    GpuTaskGuard,
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    open_local_path,
)
from yolo_studio.gui.workers import (
    ExportWorker,
    PredictWorker,
    ValidateWorker,
    WeightsInspectWorker,
)


def _show_class_order_dialog(parent: QWidget, meta: WeightsMetadata) -> None:
    """弹出对话框展示模型文件的输入尺寸、检测头能力与完整有序类别清单，支持一键复制。"""
    dlg = QDialog(parent)
    dlg.setWindowTitle(f"模型元数据与类别顺序 - {meta.weights_path.name}")
    dlg.resize(520, 440)

    layout = QVBoxLayout(dlg)
    layout.setSpacing(8)

    if meta.is_exported:
        mode_map = {
            "end2end": "已固化 YOLO26 one2one 端到端无 NMS",
            "embedded": "已固化图内嵌入 NMS",
            "raw": "已固化原始 one2many（外部 NMS）",
        }
        head_desc = mode_map.get(meta.export_nms_mode, "未记录（由导出模型内部决定）")
        dyn_desc = (
            "动态"
            if meta.export_dynamic is True
            else ("静态" if meta.export_dynamic is False else "动态性未知")
        )
        pre_line = (
            f"\n预处理契约：{format_preprocess_summary(meta.preprocess_mode, meta.preprocess_padding_value, auto_rect=meta.preprocess_auto_rect)} (RGB, 0~1)"
            if meta.preprocess_mode
            else ""
        )
        summary_text = (
            f"模型文件：{meta.weights_path.name} ({meta.model_format.upper()} 导出模型)\n"
            f"任务类型：{meta.task or '未记录'}  |  "
            f"导出输入尺寸：{format_imgsz_text(meta.train_imgsz)} ({dyn_desc})  |  "
            f"类别总数：{meta.num_classes}\n"
            f"固化输出头：{head_desc}{pre_line}"
        )
    else:
        head_desc = (
            "双分支检测头（支持 raw / end2end / embedded）"
            if meta.supports_end2end and meta.one2many_available
            else (
                "仅保留 one2one 端到端无 NMS 分支"
                if meta.supports_end2end
                else "标准 one2many 检测头"
            )
        )
        summary_text = (
            f"权重文件：{meta.weights_path.name}\n"
            f"任务类型：{meta.task or '未记录'}  |  "
            f"训练输入尺寸：{format_imgsz_text(meta.train_imgsz)}  |  "
            f"类别总数：{meta.num_classes}\n"
            f"检测头能力：{head_desc}"
        )

    summary = QLabel(summary_text)
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


class _WrappingLabel(QLabel):
    """在嵌套水平/表单布局中按实际宽度精确撑开高度的自动换行标签。

    设计原因：
        普通 `QLabel(wordWrap=True)` 放入 `QFormLayout` 内部的 `QHBoxLayout`（与按钮同行）时，
        Qt 往往按单行高度或按钮高度估算行高，导致 2~3 行文本在分组框底部被裁切，同时未折行的
        长文案 `minimumSizeHint` 会把窗口横向撑宽。本控件收紧水平最小宽度并在 `setText` /
        `resizeEvent` / `showEvent` 中按当前实际宽度重算 `heightForWidth` 更新 `minimumHeight`，
        确保多行说明完整显示且随文案变短自动回缩。
    """

    def __init__(
        self,
        text: str = "",
        *,
        rich_text: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("", parent)
        self._last_height_hint = 0
        self._syncing_height = False
        self.setWordWrap(True)
        if rich_text:
            self.setTextFormat(Qt.TextFormat.RichText)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        if text:
            self.setText(text)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(80, max(self._last_height_hint, 20))

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(240, max(self._last_height_hint, 20))

    def setText(self, text: str) -> None:  # type: ignore[override]
        super().setText(text)
        self._sync_wrapped_height()

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_wrapped_height()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._sync_wrapped_height()

    def _sync_wrapped_height(self) -> None:
        if self._syncing_height:
            return
        avail_w = self.width()
        if avail_w < 80:
            return
        self._syncing_height = True
        try:
            # 不可通过 setMinimumHeight 锁死高度，否则 QLabel.heightForWidth 会被旧最小高度钳制导致无法回缩
            needed = super().heightForWidth(avail_w)
            if needed <= 0:
                needed = super().sizeHint().height()
            # 额外留 4px 缓冲，避免高 DPI 缩放下末行基线或下伸部被分组框底边裁切
            target_h = max(20, needed + 4)
            if target_h != self._last_height_hint:
                self._last_height_hint = target_h
                self.updateGeometry()
        finally:
            self._syncing_height = False


class _ScaledPreviewLabel(QLabel):
    """支持随窗口或分割条拖拽自动等比缩放，并可双击触发放大查看原图的推理结果预览标签。"""

    double_clicked = Signal()

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self._source_pixmap: QPixmap | None = None
        self._image_path: Path | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet(
            "border: 1px dashed #999999; border-radius: 6px; background: #1e1e1e; color: #cccccc;"
        )

    def set_preview_image(self, image_path: Path) -> bool:
        """从本地文件路径加载原始推理结果图并按当前控件尺寸缩放展示；加载失败返回 `False`。"""
        if not image_path.is_file():
            return False
        pix = QPixmap(str(image_path))
        if pix.isNull():
            return False
        self._source_pixmap = pix
        self._image_path = image_path
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{image_path.name} ({pix.width()}×{pix.height()})\n双击可放大查看原图")
        self._apply_scaled()
        return True

    def image_path(self) -> Path | None:
        """返回当前展示的结果图路径；尚未加载图片时返回 `None`。"""
        return self._image_path

    def source_pixmap(self) -> QPixmap | None:
        """返回当前缓存的原始分辨率 `QPixmap`；尚未加载图片时返回 `None`。"""
        return self._source_pixmap

    def _apply_scaled(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target_w = max(160, self.width() - 12)
        target_h = max(120, self.height() - 12)
        scaled = self._source_pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._source_pixmap is not None:
            self._apply_scaled()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._source_pixmap is not None
            and not self._source_pixmap.isNull()
        ):
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class _PreviewZoomDialog(QDialog):
    """推理结果图全分辨率缩放查看对话框，便于细检密集小目标框与置信度。"""

    def __init__(
        self,
        pixmap: QPixmap,
        image_path: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_pixmap = pixmap
        self._image_path = image_path
        self._zoom: float = 1.0
        self._fit_mode: bool = True

        title = f"推理结果放大查看 - {image_path.name}" if image_path else "推理结果放大查看"
        self.setWindowTitle(f"{title} ({pixmap.width()}×{pixmap.height()})")
        self.resize(1120, 740)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.fit_btn = QPushButton("适应窗口")
        self.actual_btn = QPushButton("1:1 原尺寸")
        self.zoom_out_btn = QPushButton("缩小 (-)")
        self.zoom_in_btn = QPushButton("放大 (+)")
        self.scale_label = QLabel("100%")
        self.scale_label.setStyleSheet("color: #444444; font-weight: 500;")
        self.open_ext_btn = QPushButton("系统打开")
        self.open_ext_btn.setEnabled(image_path is not None and image_path.is_file())
        close_btn = QPushButton("关闭")

        bar.addWidget(self.fit_btn)
        bar.addWidget(self.actual_btn)
        bar.addWidget(self.zoom_out_btn)
        bar.addWidget(self.zoom_in_btn)
        bar.addSpacing(8)
        bar.addWidget(self.scale_label)
        bar.addStretch(1)
        bar.addWidget(QLabel("提示: Ctrl+滚轮缩放"))
        bar.addSpacing(8)
        bar.addWidget(self.open_ext_btn)
        bar.addWidget(close_btn)
        root.addLayout(bar)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setStyleSheet("background: #1e1e1e; border: 1px solid #444444;")
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.img_label)
        root.addWidget(self.scroll, 1)

        self.fit_btn.clicked.connect(self._fit_to_window)
        self.actual_btn.clicked.connect(lambda: self._set_zoom(1.0))
        self.zoom_out_btn.clicked.connect(lambda: self._set_zoom(self._zoom / 1.25))
        self.zoom_in_btn.clicked.connect(lambda: self._set_zoom(self._zoom * 1.25))
        self.open_ext_btn.clicked.connect(self._on_open_external)
        close_btn.clicked.connect(self.accept)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._fit_mode:
            QTimer.singleShot(0, self._fit_to_window)

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._fit_mode:
            self._fit_to_window()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._set_zoom(self._zoom * 1.15)
            elif delta < 0:
                self._set_zoom(self._zoom / 1.15)
            event.accept()
            return
        super().wheelEvent(event)

    def _fit_to_window(self) -> None:
        self._fit_mode = True
        vp = self.scroll.viewport()
        avail_w = max(200, vp.width() - 16)
        avail_h = max(160, vp.height() - 16)
        pw = max(1, self._source_pixmap.width())
        ph = max(1, self._source_pixmap.height())
        ratio = min(avail_w / pw, avail_h / ph)
        self._apply_zoom(max(0.15, min(6.0, ratio)))

    def _set_zoom(self, scale: float) -> None:
        self._fit_mode = False
        self._apply_zoom(max(0.15, min(6.0, float(scale))))

    def _apply_zoom(self, scale: float) -> None:
        self._zoom = scale
        target_w = max(1, round(self._source_pixmap.width() * scale))
        target_h = max(1, round(self._source_pixmap.height() * scale))
        scaled = self._source_pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.img_label.setPixmap(scaled)
        self.img_label.resize(scaled.size())
        self.scale_label.setText(f"{round(scale * 100)}%")

    def _on_open_external(self) -> None:
        if self._image_path is not None and self._image_path.is_file():
            open_local_path(self._image_path)


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
        self._syncing_eval_controls: bool = False
        self._preserve_export_dims_once: bool = False
        self._last_pt_weights_path: Path | None = None
        self._last_exported_path: Path | None = None
        self._busy: bool = False

        # 用户在路径输入框手动键入时的短防抖定时器（180ms），程序写入路径时直接立即探测
        self._inspect_timer = QTimer(self)
        self._inspect_timer.setSingleShot(True)
        self._inspect_timer.setInterval(180)
        self._inspect_timer.timeout.connect(self._trigger_weights_inspect)

        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)

        # 左侧面板：配置分组放入可滚动容器，避免小屏或展开隐藏行时挤压裁切底部说明文字
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.left_scroll.setMinimumWidth(560)
        left_content = QWidget()
        left = QVBoxLayout(left_content)
        left.setContentsMargins(0, 0, 4, 0)
        left.setSpacing(10)

        # 分组 1：模型文件（支持 .pt 与已导出模型）、自动元数据提示、运行设备与验证/推理共享设置
        model_group = QGroupBox("1. 模型权重与运行设备")
        self._model_form = QFormLayout(model_group)
        configure_form_layout(self._model_form)

        self.weights_picker = PathPickerRow(
            placeholder="选择 .pt 权重或已导出模型 (.onnx / .torchscript / .engine)",
            mode="file",
            file_filter=(
                "YOLO 模型 (*.pt *.onnx *.torchscript *.engine);;"
                "PyTorch 权重 (*.pt);;"
                "ONNX 模型 (*.onnx);;"
                "TorchScript (*.torchscript);;"
                "TensorRT Engine (*.engine);;"
                "所有文件 (*.*)"
            ),
        )
        self._model_form.addRow("模型文件:", self.weights_picker)

        meta_row = QHBoxLayout()
        meta_row.setSpacing(6)
        self.weights_info_label = _WrappingLabel(
            "选择 .pt 权重或已导出模型后将自动读取尺寸、检测头与类别顺序",
            rich_text=False,
        )
        self.weights_info_label.setStyleSheet("color: #666666;")
        self.view_classes_btn = QPushButton("类别清单")
        self.view_classes_btn.setEnabled(False)
        self.view_classes_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.view_classes_btn.setToolTip("查看并复制模型文件记录的完整类别索引顺序")
        meta_row.addWidget(self.weights_info_label, 1)
        meta_row.addWidget(self.view_classes_btn, 0, Qt.AlignmentFlag.AlignVCenter)
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
        self.eval_head_label = QLabel("检测头:")
        self.eval_head_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.eval_head_combo = QComboBox()
        self.eval_head_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.eval_head_combo.setMinimumContentsLength(12)
        for key, label, _nms_val in PREDICT_HEAD_MODES:
            self.eval_head_combo.addItem(label, key)
        self.eval_head_combo.setToolTip(
            "仅对 PyTorch (.pt) 权重生效，同时作用于『运行验证 (Val)』与『推理效果测试』：\n"
            "one2many (nms=None): PT 默认头 + 外部 NMS；\n"
            "end2end (nms=False): YOLO26 原生 one2one 无 NMS 端到端头（可复现 end2end 导出效果）。\n"
            "加载已导出模型（.onnx 等）时自动按模型固化模式锁定。"
        )
        self.val_btn = QPushButton("运行验证 (Val)")
        self.val_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        hw_row.addWidget(QLabel("ImgSz:"))
        hw_row.addWidget(self.imgsz_spin)
        hw_row.addSpacing(8)
        hw_row.addWidget(self.eval_head_label)
        hw_row.addWidget(self.eval_head_combo, 1)
        hw_row.addSpacing(6)
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

        pad_row = QHBoxLayout()
        pad_row.setSpacing(6)
        self.pred_pad_combo = QComboBox()
        self.pred_pad_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.pred_pad_combo.setMinimumContentsLength(12)
        for key, label in PREDICT_PAD_MODES:
            self.pred_pad_combo.addItem(label, key)
        self.pred_pad_combo.setToolTip(
            "控制推理测试的图像缩放与补边方式：\n"
            "• 最小矩形填充 (rect=True): PT 默认，等比缩放后短边仅补齐至 32 倍数（如 576×960）；\n"
            "• 固定方形填充 (rect=False): 补灰边凑满 ImgSz×ImgSz，模拟静态方形导出；\n"
            "• 按导出尺寸固定填充 (H×W，rect=False): 按下方第 3 组导出尺寸（支持矩形）补灰边，精确模拟静态导出。"
        )
        self.sync_export_pred_btn = QPushButton("对齐导出参数")
        self.sync_export_pred_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.sync_export_pred_btn.setToolTip(
            "一键将检测头、输入尺寸与填充模式对齐下方「3. 模型导出」当前配置，用于复现导出后效果"
        )
        pad_row.addWidget(self.pred_pad_combo, 1)
        pad_row.addWidget(self.sync_export_pred_btn)
        self._pred_form.addRow("输入填充:", pad_row)

        style_row = QHBoxLayout()
        style_row.setSpacing(6)
        self.pred_render_style_combo = QComboBox()
        self.pred_render_style_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.pred_render_style_combo.setMinimumContentsLength(10)
        for key, label in PREDICT_RENDER_STYLES:
            self.pred_render_style_combo.addItem(label, key)
        self.pred_render_style_combo.setToolTip(
            "控制推理预览结果图的框线与标签绘制方式：\n"
            "• 紧凑防重叠 (半透明，推荐): 按目标密度自适应细框与小字号，半透明标签底色并自动防重叠错开，置信度保持 0.94 格式；\n"
            "• Ultralytics 默认: 使用 Ultralytics 原生 r.plot() 风格绘制。"
        )
        self.pred_lw_label = QLabel("线宽:")
        self.pred_lw_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.pred_line_width_combo = QComboBox()
        for key, label in PREDICT_LINE_WIDTH_MODES:
            self.pred_line_width_combo.addItem(label, key)
        self.pred_line_width_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.pred_line_width_combo.setToolTip(
            "控制推理框线粗细与紧凑模式字号大小：\n"
            "自动：随分辨率与目标密集度自适应；极细/适中/较粗：按固定挡位渲染。"
        )
        self.pred_show_labels_chk = QCheckBox("类别")
        self.pred_show_labels_chk.setChecked(True)
        self.pred_show_labels_chk.setToolTip("在检测框标签中显示类别名称（取消且同时取消置信度时仅画框）")
        self.pred_show_conf_chk = QCheckBox("置信度")
        self.pred_show_conf_chk.setChecked(True)
        self.pred_show_conf_chk.setToolTip("在检测框标签中显示置信度小数（如 0.94）")
        style_row.addWidget(self.pred_render_style_combo, 1)
        style_row.addSpacing(6)
        style_row.addWidget(self.pred_lw_label)
        style_row.addWidget(self.pred_line_width_combo)
        style_row.addSpacing(8)
        style_row.addWidget(self.pred_show_labels_chk)
        style_row.addWidget(self.pred_show_conf_chk)
        self._pred_form.addRow("预览样式:", style_row)

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

        pred_info_row = QHBoxLayout()
        pred_info_row.setSpacing(8)
        self.pred_summary_label = _WrappingLabel(rich_text=True)
        self.switch_model_btn = QPushButton()
        self.switch_model_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.switch_model_btn.setVisible(False)
        pred_info_row.addWidget(self.pred_summary_label, 1)
        pred_info_row.addWidget(self.switch_model_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._pred_form.addRow("实际参数:", pred_info_row)
        left.addWidget(pred_group)

        # 分组 3：模型格式导出（独立导出尺寸、batch、dynamic、NMS 模式、opset、simplify 与 IO 预览）
        exp_group = QGroupBox("3. 模型导出 (部署格式)")
        self._exp_form = QFormLayout(exp_group)
        configure_form_layout(self._exp_form)

        # 行 3.1：导出格式、FP16 半精度与 ONNX 图简化
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        self.format_combo = QComboBox()
        self.format_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.format_combo.setMinimumContentsLength(10)
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
        self.nms_mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.nms_mode_combo.setMinimumContentsLength(12)
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

        # 行 3.5b：写入 ONNX 部署元数据的预处理策略（居中填充 / 左上靠边 / 直接拉伸 与补边灰度）
        pre_row = QHBoxLayout()
        pre_row.setSpacing(6)
        self.export_preprocess_combo = QComboBox()
        self.export_preprocess_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.export_preprocess_combo.setMinimumContentsLength(12)
        for key, label, _rm, _pa in EXPORT_PREPROCESS_MODES:
            self.export_preprocess_combo.addItem(label, key)
        self.export_preprocess_combo.setToolTip(
            "写入 ONNX 自身元数据 (metadata_props) 的预处理契约，供下游推理端自动读取（不改动计算图）：\n"
            "• 居中填充 (letterbox_center，推荐): YOLO 标准等比缩放 + 四周居中补边；\n"
            "• 左上靠边填充 (letterbox_top_left): 等比缩放贴左上角，仅在右下补边；\n"
            "• 直接拉伸填充 (stretch): 不保持宽高比，直接拉伸至导出高×宽（无补边）。"
        )
        self.export_pad_value_label = QLabel("补边值:")
        self.export_pad_value_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_pad_value_spin = QSpinBox()
        self.export_pad_value_spin.setRange(0, 255)
        self.export_pad_value_spin.setValue(DEFAULT_EXPORT_PADDING_VALUE)
        self.export_pad_value_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_pad_value_spin.setToolTip(
            "LetterBox 补边像素灰度值（0=纯黑，114=YOLO 标准灰边，255=纯白；直接拉伸模式不生效）"
        )
        pre_row.addWidget(self.export_preprocess_combo, 1)
        pre_row.addSpacing(8)
        pre_row.addWidget(self.export_pad_value_label)
        pre_row.addWidget(self.export_pad_value_spin)
        self._exp_form.addRow("预处理:", pre_row)

        # 行 3.6：实时预期输入/输出张量说明与「导出模型」执行按钮
        io_row = QHBoxLayout()
        io_row.setSpacing(8)
        self.export_io_label = _WrappingLabel(rich_text=True)
        self.export_btn = QPushButton("导出模型")
        self.export_btn.setObjectName("primaryBtn")
        self.export_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        io_row.addWidget(self.export_io_label, 1)
        io_row.addWidget(self.export_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        self._exp_form.addRow("输入/输出:", io_row)

        # 限制下拉框最小宽度避免小屏撑出横向滚动条，同时保证展开弹出列表时可完整阅读长选项文案
        for combo in (
            self.eval_head_combo,
            self.pred_pad_combo,
            self.pred_render_style_combo,
            self.format_combo,
            self.nms_mode_combo,
            self.export_preprocess_combo,
        ):
            combo.view().setMinimumWidth(320)

        align_form_labels([self._model_form, self._pred_form, self._exp_form])
        left.addWidget(exp_group)
        left.addStretch(1)
        self.left_scroll.setWidget(left_content)

        # 右侧面板：垂直可拖拽分割条，上方为推理结果预览，下方为运行日志
        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setChildrenCollapsible(False)
        self.right_splitter.setHandleWidth(8)
        self.right_splitter.setMinimumWidth(360)

        preview_group = QGroupBox("4. 推理结果预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 6, 8, 8)
        preview_layout.setSpacing(6)
        preview_bar = QHBoxLayout()
        preview_bar.setContentsMargins(0, 0, 0, 0)
        preview_bar.setSpacing(6)
        self.preview_info_label = QLabel("双击结果图或点击「放大查看」可按原图分辨率缩放检视")
        self.preview_info_label.setStyleSheet("color: #666666; font-size: 12px;")
        self.preview_info_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.zoom_preview_btn = QPushButton("放大查看")
        self.zoom_preview_btn.setEnabled(False)
        self.zoom_preview_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.zoom_preview_btn.setToolTip("在独立缩放窗口中查看当前推理结果原图（支持 1:1 与 Ctrl+滚轮缩放）")
        preview_bar.addWidget(self.preview_info_label, 1)
        preview_bar.addWidget(self.zoom_preview_btn)
        preview_layout.addLayout(preview_bar)
        self.preview_label = _ScaledPreviewLabel("运行图片推理后，将在此显示带检测框的结果图")
        preview_layout.addWidget(self.preview_label, 1)
        self.right_splitter.addWidget(preview_group)

        log_group = QGroupBox("5. 运行日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(8, 8, 8, 8)
        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(140)
        self.log_box.setPlaceholderText("权重元数据摘要、验证指标、推理结果路径与导出日志将在此显示...")
        log_layout.addWidget(self.log_box, 1)
        self.right_splitter.addWidget(log_group)

        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 2)
        self.right_splitter.setSizes([460, 300])

        self.main_splitter.addWidget(self.left_scroll)
        self.main_splitter.addWidget(self.right_splitter)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([830, 520])
        root.addWidget(self.main_splitter)

        # 信号与槽连接
        self.weights_picker.textChanged.connect(self._on_weights_text_changed)
        self.view_classes_btn.clicked.connect(self._on_view_classes)
        self.imgsz_spin.valueChanged.connect(self._on_eval_param_changed)
        self.eval_head_combo.currentIndexChanged.connect(self._on_eval_param_changed)
        self.pred_pad_combo.currentIndexChanged.connect(self._on_eval_param_changed)
        self.pred_render_style_combo.currentIndexChanged.connect(self._on_eval_param_changed)
        self.pred_line_width_combo.currentIndexChanged.connect(self._on_eval_param_changed)
        self.pred_show_labels_chk.toggled.connect(self._on_eval_param_changed)
        self.pred_show_conf_chk.toggled.connect(self._on_eval_param_changed)
        self.sync_export_pred_btn.clicked.connect(self._on_sync_export_to_predict)
        self.switch_model_btn.clicked.connect(self._on_switch_model)
        self.conf_spin.valueChanged.connect(self._on_eval_param_changed)
        self.iou_spin.valueChanged.connect(self._on_eval_param_changed)
        self.zoom_preview_btn.clicked.connect(self._on_zoom_preview)
        self.preview_label.double_clicked.connect(self._on_zoom_preview)

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
        self.export_preprocess_combo.currentIndexChanged.connect(self._on_export_format_changed)
        self.export_pad_value_spin.valueChanged.connect(self._refresh_export_io_preview)

        self._on_export_square_toggled(self.export_square_chk.isChecked())
        self._on_export_format_changed()
        self._refresh_eval_controls_state()

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭标签页时停止防抖定时器，避免关闭后触发新的模型探测。"""
        self._inspect_timer.stop()
        super().closeEvent(event)

    def running_inspect_workers(self) -> list[WeightsInspectWorker]:
        """返回仍在后台运行的模型元数据探测线程列表，供主窗口关闭前安全等待。"""
        workers: list[WeightsInspectWorker] = []
        if self.inspect_worker is not None and self.inspect_worker.isRunning():
            workers.append(self.inspect_worker)
        for old in self._retired_inspect_workers:
            if old.isRunning():
                workers.append(old)
        return workers

    def current_weights_metadata(self) -> WeightsMetadata | None:
        """返回当前已成功读取的模型元数据快照；尚未加载或读取失败时返回 `None`。"""
        return self._weights_metadata

    def _current_model_is_exported(self) -> bool:
        """判断当前上方选择的模型是否为已导出部署模型（`.onnx` / `.torchscript` / `.engine`）。"""
        if self._weights_metadata is not None:
            return self._weights_metadata.is_exported
        raw = self.weights_picker.text().strip()
        return bool(raw) and is_exported_model_path(Path(raw))

    def _current_model_format(self) -> str:
        """返回当前上方所选模型文件的格式键名（`pt` / `onnx` / `torchscript` / `engine`）。"""
        if self._weights_metadata is not None:
            return self._weights_metadata.model_format
        raw = self.weights_picker.text().strip()
        return detect_model_format(Path(raw)) if raw else "pt"

    def _on_weights_text_changed(self, _text: str) -> None:
        """模型路径变化后先立即刷新界面防呆状态，再通过短防抖调度只读元数据探测。"""
        self._refresh_eval_controls_state()
        self._inspect_timer.start()

    def _clear_weights_metadata(self, message: str, *, color: str = "#666666") -> None:
        self._weights_metadata = None
        self._inspected_file_sig = None
        self._preserve_export_dims_once = False
        self.weights_info_label.setText(message)
        self.weights_info_label.setStyleSheet(f"color: {color};")
        self.weights_info_label.setToolTip(message)
        self.view_classes_btn.setEnabled(False)
        self.sync_train_imgsz_btn.setEnabled(False)
        self.sync_train_imgsz_btn.setToolTip("选择 .pt 权重并读取到训练尺寸后，可一键恢复为权重训练尺寸")
        self._refresh_export_io_preview()
        self._refresh_eval_controls_state()

    def _trigger_weights_inspect(self) -> None:
        """校验当前模型路径并在后台启动 `WeightsInspectWorker` 读取尺寸、检测头与类别顺序。"""
        self._inspect_timer.stop()
        raw = self.weights_picker.text()
        if not raw:
            self._inspect_request_id += 1
            self._clear_weights_metadata("选择 .pt 权重或已导出模型后将自动读取尺寸、检测头与类别顺序")
            return

        path = Path(raw)
        if not path.is_file():
            self._inspect_request_id += 1
            self._clear_weights_metadata("未找到有效的模型文件", color="#b25900")
            return

        try:
            st = path.stat()
            sig = (str(path.resolve()), st.st_mtime_ns, st.st_size)
        except OSError:
            sig = (str(path), 0, 0)

        if self._inspected_file_sig == sig and self._weights_metadata is not None:
            self._refresh_eval_controls_state()
            return

        self._inspect_request_id += 1
        req_id = self._inspect_request_id
        self._weights_metadata = None
        self._inspected_file_sig = None
        self.view_classes_btn.setEnabled(False)
        self.sync_train_imgsz_btn.setEnabled(False)
        self.weights_info_label.setText("正在后台读取模型元数据（输入尺寸、检测头与类别顺序）...")
        self.weights_info_label.setStyleSheet("color: #0969da;")
        self.weights_info_label.setToolTip("正在只读解析模型文件头信息...")
        self._refresh_export_io_preview()
        self._refresh_eval_controls_state()

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
        """将 `.pt` 权重记录的 `(高, 宽)` 同步写入验证/推理 `ImgSz` 与独立导出尺寸控件。"""
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
        """接收模型元数据探测结果：更新界面提示、自动同步尺寸与防呆状态并在日志区输出摘要。"""
        if request_id != self._inspect_request_id:
            return

        self._weights_metadata = meta
        self._inspected_file_sig = file_sig
        self.view_classes_btn.setEnabled(meta.num_classes > 0)

        cls_brief = format_class_order(meta.class_names, max_items=5)
        if meta.is_exported:
            self.sync_train_imgsz_btn.setEnabled(False)
            self.sync_train_imgsz_btn.setToolTip("当前选中为已导出部署模型，请切回 .pt 权重后再执行导出")
            if meta.train_imgsz is not None:
                h, w = meta.train_imgsz
                clamped_val = max(
                    self.imgsz_spin.minimum(), min(self.imgsz_spin.maximum(), max(h, w))
                )
                self.imgsz_spin.setValue(clamped_val)
            size_text = format_imgsz_text(meta.train_imgsz)
            dyn_text = (
                "动态"
                if meta.export_dynamic is True
                else ("静态" if meta.export_dynamic is False else "动态性未知")
            )
            mode_brief = {
                "end2end": "one2one 无NMS",
                "embedded": "图内嵌入NMS",
                "raw": "one2many 外部NMS",
            }.get(meta.export_nms_mode, "固化输出头")
            pre_brief = (
                f" | 预处理: {format_preprocess_summary(meta.preprocess_mode, meta.preprocess_padding_value)}"
                if meta.preprocess_mode
                else ""
            )
            self.weights_info_label.setText(
                f"[{meta.model_format.upper()} 导出模型] 输入: {size_text} ({dyn_text}) | "
                f"输出: {mode_brief}{pre_brief} | 类别({meta.num_classes}): {cls_brief}"
            )
            self.weights_info_label.setStyleSheet("color: #0969da; font-weight: 500;")
            tooltip_lines = [
                f"模型文件: {meta.weights_path.name} ({meta.model_format.upper()} 导出模型)",
                f"任务类型: {meta.task or '未记录'}",
                f"导出输入尺寸: {size_text} ({dyn_text})",
                f"固化输出头: {mode_brief}",
            ]
            if meta.preprocess_mode:
                tooltip_lines.append(
                    f"预处理契约: {format_preprocess_summary(meta.preprocess_mode, meta.preprocess_padding_value, auto_rect=meta.preprocess_auto_rect)} (RGB, 0~1)"
                )
            tooltip_lines.extend(
                [
                    f"类别总数: {meta.num_classes}",
                    "完整类别顺序:",
                    format_class_order_lines(meta.class_names),
                ]
            )
            self.weights_info_label.setToolTip("\n".join(tooltip_lines))

            self.log_box.appendPlainText(format_weights_metadata_log(meta))
            if meta.export_dynamic is False and meta.train_imgsz is not None:
                self.log_box.appendPlainText(
                    f"  [自动锁定] 该静态 {meta.model_format.upper()} 模型固定按 {size_text} 与其固化输出头运行验证/推理。"
                )
        else:
            self._last_pt_weights_path = meta.weights_path
            preserve_export = self._preserve_export_dims_once
            self._preserve_export_dims_once = False
            if meta.train_imgsz is not None:
                if preserve_export:
                    h, w = meta.train_imgsz
                    clamped_val = max(
                        self.imgsz_spin.minimum(), min(self.imgsz_spin.maximum(), max(h, w))
                    )
                    self.imgsz_spin.setValue(clamped_val)
                else:
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
                if preserve_export:
                    self.log_box.appendPlainText(
                        f"  [切回权重] 已保留下方自定义导出尺寸；验证/推理尺寸同步为: {format_imgsz_text(meta.train_imgsz)}"
                    )
                else:
                    self.log_box.appendPlainText(
                        f"  [自动同步] 已将验证/推理尺寸与导出尺寸同步为权重训练尺寸: {format_imgsz_text(meta.train_imgsz)}"
                    )
            else:
                self.log_box.appendPlainText(
                    "  [提示] 该权重未记录训练输入尺寸，请手动确认验证与导出尺寸。"
                )

        self._refresh_export_io_preview()
        self._refresh_eval_controls_state()

    def _on_weights_inspect_failed(self, request_id: int, err: str) -> None:
        if request_id != self._inspect_request_id:
            return
        self._clear_weights_metadata(f"未能自动解析模型元数据：{err}", color="#b25900")
        self.log_box.appendPlainText(f"[提示] 读取模型元数据失败: {err}")

    def _on_view_classes(self) -> None:
        if self._weights_metadata is not None:
            _show_class_order_dialog(self, self._weights_metadata)

    def _on_sync_train_imgsz(self) -> None:
        """一键将导出尺寸恢复为当前 `.pt` 权重记录的训练尺寸。"""
        if (
            self._weights_metadata is None
            or self._weights_metadata.is_exported
            or self._weights_metadata.train_imgsz is None
        ):
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

    def _on_eval_param_changed(self, *_args: object) -> None:
        """检测头、尺寸、填充方式或阈值变化后刷新推理联动防呆与实际参数摘要。"""
        if self._syncing_eval_controls:
            return
        self._refresh_eval_controls_state()

    def _resolve_eval_head_nms(self) -> bool | None:
        """根据当前模型与检测头下拉框返回传给 Ultralytics 的 `nms` 参数值（已导出模型返回 `None`）。"""
        if self._current_model_is_exported():
            return None
        meta = self._weights_metadata
        if meta is not None and not meta.supports_end2end:
            return None
        if meta is not None and meta.supports_end2end and not meta.one2many_available:
            return False
        head_key = str(self.eval_head_combo.currentData() or "one2many")
        return PREDICT_HEAD_KEY_TO_NMS.get(head_key, None)

    def build_predict_render_config(self) -> PredictRenderConfig:
        """从界面「预览样式」行组装 `PredictRenderConfig` 配置对象。"""
        style = str(self.pred_render_style_combo.currentData() or DEFAULT_PREDICT_RENDER_STYLE)
        line_width_mode = str(
            self.pred_line_width_combo.currentData() or DEFAULT_PREDICT_LINE_WIDTH_MODE
        )
        return PredictRenderConfig(
            style=style,
            line_width_mode=line_width_mode,
            show_labels=self.pred_show_labels_chk.isChecked(),
            show_conf=self.pred_show_conf_chk.isChecked(),
            label_alpha=0.68,
            avoid_overlap=True,
        )

    def build_predict_params(self) -> dict[str, object]:
        """汇总当前「2. 推理效果测试」实际将提交的参数字典（含检测头、输入尺寸、预处理、样式与阈值）。"""
        meta = self._weights_metadata
        is_exported = self._current_model_is_exported()
        static_exported = is_exported and meta is not None and meta.export_dynamic is False

        preprocess_mode = DEFAULT_EXPORT_PREPROCESS_MODE
        padding_value: int | None = DEFAULT_EXPORT_PADDING_VALUE

        if is_exported:
            if meta is not None and meta.preprocess_mode:
                preprocess_mode = meta.preprocess_mode
                padding_value = (
                    None
                    if preprocess_mode == "stretch"
                    else (
                        meta.preprocess_padding_value
                        if meta.preprocess_padding_value is not None
                        else DEFAULT_EXPORT_PADDING_VALUE
                    )
                )
            if meta is not None and meta.train_imgsz is not None and (
                static_exported or meta.deploy_schema == ONNX_DEPLOY_SCHEMA_VERSION
            ):
                mh, mw = meta.train_imgsz
                imgsz: int | tuple[int, int] = mh if mh == mw else (mh, mw)
                rect = bool(meta.export_dynamic and meta.preprocess_auto_rect)
            else:
                pad_mode = str(self.pred_pad_combo.currentData() or "rect")
                if pad_mode == "export":
                    eh = self.export_h_spin.value()
                    ew = eh if self.export_square_chk.isChecked() else self.export_w_spin.value()
                    imgsz = eh if eh == ew else (eh, ew)
                    rect = False
                elif pad_mode == "square":
                    imgsz = self.imgsz_spin.value()
                    rect = False
                else:
                    imgsz = self.imgsz_spin.value()
                    rect = True
            if preprocess_mode == "stretch":
                rect = False
        else:
            pad_mode = str(self.pred_pad_combo.currentData() or "rect")
            if pad_mode == "export":
                eh = self.export_h_spin.value()
                ew = eh if self.export_square_chk.isChecked() else self.export_w_spin.value()
                imgsz = eh if eh == ew else (eh, ew)
                rect = False
                if str(self.format_combo.currentData() or "onnx") == "onnx":
                    preprocess_mode = str(
                        self.export_preprocess_combo.currentData() or DEFAULT_EXPORT_PREPROCESS_MODE
                    )
                    padding_value = (
                        None if preprocess_mode == "stretch" else self.export_pad_value_spin.value()
                    )
            elif pad_mode == "square":
                imgsz = self.imgsz_spin.value()
                rect = False
            else:
                imgsz = self.imgsz_spin.value()
                rect = True

        nms_val = self._resolve_eval_head_nms()
        if is_exported:
            uses_external_iou = meta is None or meta.export_nms_mode in {None, "raw"}
            max_det = 300
        else:
            effective_end2end = nms_val is False
            uses_external_iou = not effective_end2end
            max_det = self.export_max_det_spin.value() if effective_end2end else 300

        return {
            "imgsz": imgsz,
            "rect": rect,
            "nms": nms_val,
            "max_det": max_det,
            "conf": self.conf_spin.value(),
            "iou": self.iou_spin.value(),
            "uses_external_iou": uses_external_iou,
            "preprocess_mode": preprocess_mode,
            "padding_value": padding_value,
            "render_config": self.build_predict_render_config(),
        }

    def _refresh_eval_controls_state(self) -> None:
        """同步模型验证与推理测试区域的防呆联动（检测头、尺寸/填充、IoU、对齐与切换按钮及参数摘要）。"""
        meta = self._weights_metadata
        is_exported = self._current_model_is_exported()
        static_exported = is_exported and meta is not None and meta.export_dynamic is False

        self._syncing_eval_controls = True
        try:
            # 1) 检测头下拉框联动与锁定
            if is_exported:
                self.eval_head_combo.setEnabled(False)
                if meta is not None and meta.export_nms_mode == "end2end":
                    idx = self.eval_head_combo.findData("end2end")
                    if idx >= 0:
                        self.eval_head_combo.setCurrentIndex(idx)
                    self.eval_head_combo.setToolTip(
                        "当前已导出模型已固化为 YOLO26 one2one 端到端无 NMS 输出，不可切换"
                    )
                elif meta is not None and meta.export_nms_mode == "embedded":
                    idx = self.eval_head_combo.findData("one2many")
                    if idx >= 0:
                        self.eval_head_combo.setCurrentIndex(idx)
                    self.eval_head_combo.setToolTip(
                        "当前已导出模型已固化为图内嵌入 NMS 输出，推理期不可切换"
                    )
                elif meta is not None and meta.export_nms_mode == "raw":
                    idx = self.eval_head_combo.findData("one2many")
                    if idx >= 0:
                        self.eval_head_combo.setCurrentIndex(idx)
                    self.eval_head_combo.setToolTip(
                        "当前已导出模型已固化为原始 one2many 输出（需外部 NMS），不可切换"
                    )
                else:
                    self.eval_head_combo.setToolTip("已导出模型的检测头已在导出时固化，推理期不可切换")
            elif meta is not None and not meta.supports_end2end:
                idx = self.eval_head_combo.findData("one2many")
                if idx >= 0:
                    self.eval_head_combo.setCurrentIndex(idx)
                self.eval_head_combo.setEnabled(False)
                self.eval_head_combo.setToolTip(
                    "当前 .pt 权重未包含 YOLO26 one2one 端到端分支，仅支持 one2many + 外部 NMS"
                )
            elif meta is not None and meta.supports_end2end and not meta.one2many_available:
                idx = self.eval_head_combo.findData("end2end")
                if idx >= 0:
                    self.eval_head_combo.setCurrentIndex(idx)
                self.eval_head_combo.setEnabled(False)
                self.eval_head_combo.setToolTip("当前 .pt 权重仅保留 one2one 端到端无 NMS 分支")
            else:
                self.eval_head_combo.setEnabled(True)
                self.eval_head_combo.setToolTip(
                    "仅对 PyTorch (.pt) 权重生效，同时作用于『运行验证 (Val)』与『推理效果测试』：\n"
                    "one2many (nms=None): PT 默认头 + 外部 NMS；\n"
                    "end2end (nms=False): YOLO26 原生 one2one 无 NMS 端到端头（可复现 end2end 导出效果）。"
                )

            # 2) 静态导出模型或 Stretch 拉伸导出模型锁定 ImgSz / 输入填充下拉框
            exported_stretch = is_exported and meta is not None and meta.preprocess_mode == "stretch"
            if (
                is_exported
                and meta is not None
                and meta.export_dynamic
                and meta.deploy_schema == ONNX_DEPLOY_SCHEMA_VERSION
            ):
                self.imgsz_spin.setEnabled(False)
                self.pred_pad_combo.setEnabled(False)
                self.imgsz_spin.setToolTip("动态 ONNX 推理采用模型记录的参考尺寸")
                self.pred_pad_combo.setToolTip("按 ONNX 中的预处理策略和步长计算每帧填充")
            elif static_exported:
                self.imgsz_spin.setEnabled(False)
                self.imgsz_spin.setToolTip(
                    f"当前静态 {meta.model_format.upper() if meta else ''} 模型固定使用导出尺寸 "
                    f"{format_imgsz_text(meta.train_imgsz if meta else None)}，推理/验证将自动沿用该固定尺寸"
                )
                self.pred_pad_combo.setEnabled(False)
                if exported_stretch:
                    self.pred_pad_combo.setToolTip(
                        "该导出模型元数据声明采用直接拉伸 (Stretch) 预处理，推理时将直接拉伸至导出尺寸"
                    )
                else:
                    self.pred_pad_combo.setToolTip(
                        "静态导出模型固定使用导出尺寸进行预处理 (rect=False)，无需手动选择填充模式"
                    )
            else:
                self.imgsz_spin.setEnabled(True)
                self.imgsz_spin.setToolTip(
                    "用于『运行验证 (Val)』与『推理效果测试』的输入尺寸；模型导出尺寸请在下方第 3 组独立设置"
                )
                if exported_stretch:
                    self.pred_pad_combo.setEnabled(False)
                    self.pred_pad_combo.setToolTip(
                        "该导出模型元数据声明采用直接拉伸 (Stretch) 预处理，不进行 LetterBox 补边"
                    )
                else:
                    self.pred_pad_combo.setEnabled(True)
                    self.pred_pad_combo.setToolTip(
                        "控制推理测试的图像缩放与补边方式：\n"
                        "• 最小矩形填充 (rect=True): PT 默认，等比缩放后短边仅补齐至 32 倍数（如 576×960）；\n"
                        "• 固定方形填充 (rect=False): 补灰边凑满 ImgSz×ImgSz，模拟静态方形导出；\n"
                        "• 按导出尺寸固定填充 (H×W，rect=False): 按下方第 3 组导出尺寸（含预处理策略）固定处理，精确模拟静态导出。"
                    )
        finally:
            self._syncing_eval_controls = False

        # 3) IoU 阈值与「对齐导出参数」按钮防呆联动
        params = self.build_predict_params()
        uses_external_iou = bool(params["uses_external_iou"])
        self.iou_spin.setEnabled(uses_external_iou)
        if uses_external_iou:
            self.iou_spin.setToolTip("外部 NMS 去重 IoU 阈值（one2many / raw 原始输出模式生效）")
        else:
            self.iou_spin.setToolTip(
                "当前为 YOLO26 one2one 端到端无 NMS 或图内嵌入 NMS 模式，不使用外部 IoU 阈值"
            )

        self.sync_export_pred_btn.setEnabled((not is_exported) and (not self._busy))
        if is_exported:
            self.sync_export_pred_btn.setToolTip("当前已直接加载导出模型，无需模拟导出参数")
        else:
            self.sync_export_pred_btn.setToolTip(
                "一键将检测头、输入尺寸与填充模式对齐下方「3. 模型导出」当前配置，用于复现导出后效果"
            )

        # 4) 导出产物 ↔ .pt 源权重一键切换按钮
        if (
            not is_exported
            and self._last_exported_path is not None
            and self._last_exported_path.is_file()
        ):
            self.switch_model_btn.setText("测刚导出模型")
            self.switch_model_btn.setToolTip(
                f"一键切换上方模型为刚导出的 {self._last_exported_path.name}，直接实测导出后效果"
            )
            self.switch_model_btn.setVisible(True)
            self.switch_model_btn.setEnabled(not self._busy)
        elif (
            is_exported
            and self._last_pt_weights_path is not None
            and self._last_pt_weights_path.is_file()
        ):
            self.switch_model_btn.setText("切回 .pt 权重")
            self.switch_model_btn.setToolTip(
                f"一键切回源 PyTorch 权重 {self._last_pt_weights_path.name} 以继续对比或重新导出"
            )
            self.switch_model_btn.setVisible(True)
            self.switch_model_btn.setEnabled(not self._busy)
        else:
            self.switch_model_btn.setVisible(False)

        self._refresh_action_btn_states()
        self._refresh_pred_summary(params)

    def _refresh_pred_summary(self, params: dict[str, object] | None = None) -> None:
        """刷新「2. 推理效果测试」底部的实际生效参数摘要，并在与下方导出配置不一致时提醒。"""
        if params is None:
            params = self.build_predict_params()

        meta = self._weights_metadata
        is_exported = self._current_model_is_exported()
        fmt_upper = self._current_model_format().upper()

        imgsz_val = params["imgsz"]
        size_pair = (
            (int(imgsz_val[0]), int(imgsz_val[1]))
            if isinstance(imgsz_val, tuple)
            else (int(imgsz_val), int(imgsz_val))  # type: ignore[arg-type]
        )
        size_text = format_imgsz_text(size_pair)
        rect_flag = bool(params["rect"])
        conf_val = float(params["conf"])  # type: ignore[arg-type]
        iou_val = float(params["iou"])  # type: ignore[arg-type]
        max_det_val = int(params["max_det"])  # type: ignore[arg-type]
        pre_mode = str(params.get("preprocess_mode") or DEFAULT_EXPORT_PREPROCESS_MODE)
        pre_pad = params.get("padding_value")
        pre_pad_int = int(pre_pad) if isinstance(pre_pad, int) else None

        if is_exported:
            mode_map = {
                "end2end": "one2one 无 NMS (已固化)",
                "embedded": "图内嵌入 NMS (已固化)",
                "raw": "one2many 外部 NMS (已固化)",
            }
            head_txt = mode_map.get(meta.export_nms_mode if meta else None, "导出模型固化头")
            if pre_mode == "stretch":
                pad_txt = "直接拉伸 Stretch"
            elif meta is not None and meta.export_dynamic is False:
                pad_txt = (
                    f"{format_preprocess_summary(pre_mode, pre_pad_int)} (静态)"
                    if (pre_mode != DEFAULT_EXPORT_PREPROCESS_MODE or pre_pad_int != DEFAULT_EXPORT_PADDING_VALUE)
                    else "固定尺寸填充 (静态)"
                )
            else:
                pad_txt = "最小矩形填充 (rect=True)" if rect_flag else "固定尺寸填充 (rect=False)"
                if pre_mode != DEFAULT_EXPORT_PREPROCESS_MODE or pre_pad_int != DEFAULT_EXPORT_PADDING_VALUE:
                    pad_txt = f"{format_preprocess_summary(pre_mode, pre_pad_int)}, {pad_txt}"
            model_tag = f"{fmt_upper} 导出模型"
        else:
            if params["nms"] is False:
                head_txt = f"one2one 端到端无 NMS (MaxDet={max_det_val})"
            else:
                head_txt = "one2many + 外部 NMS"
            if pre_mode == "stretch":
                pad_txt = "直接拉伸 Stretch (rect=False)"
            else:
                pad_txt = "最小矩形填充 (rect=True)" if rect_flag else "固定尺寸填充 (rect=False)"
                if pre_mode != DEFAULT_EXPORT_PREPROCESS_MODE or pre_pad_int != DEFAULT_EXPORT_PADDING_VALUE:
                    pad_txt = f"{format_preprocess_summary(pre_mode, pre_pad_int)}, {pad_txt}"
            model_tag = "PT 权重"

        thr_txt = (
            f"Conf={conf_val:.2f}, IoU={iou_val:.2f}"
            if bool(params["uses_external_iou"])
            else f"Conf={conf_val:.2f} (无外部 IoU)"
        )
        raw_render_cfg = params.get("render_config")
        render_cfg = (
            raw_render_cfg
            if isinstance(raw_render_cfg, PredictRenderConfig)
            else self.build_predict_render_config()
        )
        render_txt = format_predict_render_summary(render_cfg)
        summary_line = (
            f"{model_tag}  |  {head_txt}  |  输入: {size_text} {pad_txt}  |  {thr_txt}  |  样式: {render_txt}"
        )

        hints: list[str] = []
        if is_exported:
            if meta is not None and meta.export_nms_mode == "embedded":
                hints.append("图内嵌入 NMS 模型已固化导出阈值，调低此处 Conf 不会召回已被图内过滤的目标。")
        else:
            exp_cfg = self.build_export_config(allow_missing_weights=True)
            exp_imgsz = exp_cfg.imgsz
            exp_pair = (
                (int(exp_imgsz[0]), int(exp_imgsz[1]))
                if isinstance(exp_imgsz, tuple)
                else (
                    (int(exp_imgsz), int(exp_imgsz))
                    if isinstance(exp_imgsz, int)
                    else size_pair
                )
            )
            diffs: list[str] = []
            if (exp_cfg.nms is False) != (params["nms"] is False):
                diffs.append("检测头")
            if not exp_cfg.dynamic:
                if rect_flag:
                    diffs.append("填充方式")
                if size_pair != exp_pair:
                    diffs.append("输入尺寸")
            if exp_cfg.export_format == "onnx":
                exp_pad = None if exp_cfg.preprocess_mode == "stretch" else exp_cfg.padding_value
                if exp_cfg.preprocess_mode != pre_mode or exp_pad != pre_pad_int:
                    diffs.append("预处理")
            if diffs:
                diff_str = "/".join(diffs)
                hints.append(
                    f"与下方导出配置存在差异（{diff_str}），可点击「对齐导出参数」模拟导出后效果。"
                )

        html_parts = [f"<span style='color:#333333;'>{html.escape(summary_line)}</span>"]
        for hint in hints:
            html_parts.append(
                f"<br><span style='color:#b25900; font-weight:500;'>[提示] {html.escape(hint)}</span>"
            )
        self.pred_summary_label.setText("".join(html_parts))
        self.pred_summary_label.setToolTip("\n".join([summary_line, *(f"[提示] {h}" for h in hints)]))

    def _on_sync_export_to_predict(self) -> None:
        """一键将推理测试的检测头、输入尺寸、填充模式与必要阈值对齐下方第 3 组导出配置。"""
        if self._current_model_is_exported():
            return

        exp_cfg = self.build_export_config(allow_missing_weights=True)
        meta = self._weights_metadata

        self._syncing_eval_controls = True
        try:
            # 1) 对齐检测头（导出 end2end -> one2one；raw / embedded -> one2many）
            if exp_cfg.nms is False:
                if meta is None or meta.supports_end2end:
                    idx = self.eval_head_combo.findData("end2end")
                    if idx >= 0:
                        self.eval_head_combo.setCurrentIndex(idx)
            else:
                if meta is None or meta.one2many_available:
                    idx = self.eval_head_combo.findData("one2many")
                    if idx >= 0:
                        self.eval_head_combo.setCurrentIndex(idx)

            # 2) 对齐输入尺寸与填充模式（含自定义预处理策略模拟）
            if isinstance(exp_cfg.imgsz, tuple):
                ref_dim = max(exp_cfg.imgsz)
            elif isinstance(exp_cfg.imgsz, int):
                ref_dim = exp_cfg.imgsz
            else:
                ref_dim = self.imgsz_spin.value()
            clamped = max(self.imgsz_spin.minimum(), min(self.imgsz_spin.maximum(), ref_dim))
            self.imgsz_spin.setValue(clamped)

            has_custom_onnx_pre = exp_cfg.export_format == "onnx" and (
                exp_cfg.preprocess_mode != DEFAULT_EXPORT_PREPROCESS_MODE
                or exp_cfg.padding_value != DEFAULT_EXPORT_PADDING_VALUE
            )
            target_pad = "rect" if (exp_cfg.dynamic and not has_custom_onnx_pre) else "export"
            pad_idx = self.pred_pad_combo.findData(target_pad)
            if pad_idx >= 0:
                self.pred_pad_combo.setCurrentIndex(pad_idx)

            # 3) 图内嵌入 NMS 模式下顺带对齐 Conf / IoU 便于模拟导出过滤阈值
            if exp_cfg.nms is True:
                if exp_cfg.conf is not None:
                    self.conf_spin.setValue(exp_cfg.conf)
                if exp_cfg.iou is not None:
                    self.iou_spin.setValue(exp_cfg.iou)
        finally:
            self._syncing_eval_controls = False

        self._refresh_eval_controls_state()
        params = self.build_predict_params()
        imgsz_val = params["imgsz"]
        size_pair = (
            (int(imgsz_val[0]), int(imgsz_val[1]))
            if isinstance(imgsz_val, tuple)
            else (int(imgsz_val), int(imgsz_val))  # type: ignore[arg-type]
        )
        head_label = "end2end (one2one 无 NMS)" if params["nms"] is False else "one2many (外部 NMS)"
        pad_label = "最小矩形 (rect=True)" if params["rect"] else "固定尺寸填充 (rect=False)"
        self.log_box.appendPlainText(
            f"[对齐导出] 推理测试已同步下方导出配置：检测头={head_label}，"
            f"输入={format_imgsz_text(size_pair)}，{pad_label}。"
        )

    def _on_switch_model(self) -> None:
        """在最近成功导出的部署模型与源 `.pt` 权重之间一键来回切换并立即触发元数据探测。"""
        if self._current_model_is_exported():
            target = self._last_pt_weights_path
            preserve_export = True
        else:
            cur_raw = self.weights_picker.text().strip()
            if cur_raw and Path(cur_raw).is_file() and not is_exported_model_path(Path(cur_raw)):
                self._last_pt_weights_path = Path(cur_raw)
            target = self._last_exported_path
            preserve_export = False

        if target is None or not target.is_file():
            self._refresh_eval_controls_state()
            return
        self._preserve_export_dims_once = preserve_export
        self.weights_picker.setText(str(target))
        self._trigger_weights_inspect()

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
        """同步「导出模型」按钮可用态：忙碌、已选中非 `.pt` 导出模型或 TensorRT 处于 CPU 模式时置灰防误点。"""
        fmt = str(self.format_combo.currentData() or "onnx")
        using_gpu = self.device_ctrl.is_using_gpu() and not self.device_info.probing
        blocked_engine_cpu = fmt == "engine" and not using_gpu
        blocked_exported_model = self._current_model_is_exported()
        self.export_btn.setEnabled(
            (not self._busy) and (not blocked_engine_cpu) and (not blocked_exported_model)
        )
        if blocked_exported_model:
            self.export_btn.setToolTip(
                "模型导出仅支持 PyTorch (.pt) 源权重；当前上方已选中导出模型，请切回 .pt 权重后再导出"
            )
        elif blocked_engine_cpu:
            self.export_btn.setToolTip(
                "TensorRT (.engine) 导出需要可用的 CUDA GPU，请在上方运行设备中启用 GPU 加速后再导出"
            )
        else:
            self.export_btn.setToolTip("按当前配置导出部署模型（产物生成在源权重同目录）")

    def _refresh_action_btn_states(self) -> None:
        """统一同步验证、推理预览与导出三个执行按钮的防呆可用状态。"""
        using_gpu = self.device_ctrl.is_using_gpu() and not self.device_info.probing
        blocked_engine_model_cpu = self._current_model_format() == "engine" and not using_gpu
        can_eval = (not self._busy) and (not blocked_engine_model_cpu)
        self.val_btn.setEnabled(can_eval)
        self.predict_btn.setEnabled(can_eval)
        if blocked_engine_model_cpu:
            tip = "TensorRT (.engine) 模型仅支持在 CUDA GPU 上运行，请在上方启用 GPU 加速"
            self.val_btn.setToolTip(tip)
            self.predict_btn.setToolTip(tip)
        else:
            self.val_btn.setToolTip("使用当前所选模型、尺寸与检测头在验证集上评估 mAP / Precision / Recall")
            self.predict_btn.setToolTip("按当前模型、检测头、填充模式与阈值执行推理并预览结果")
        self._refresh_export_btn_state()

    def _on_export_format_changed(self, *_args: object) -> None:
        """同步导出面板各项防呆状态：FP16、ONNX 专属参数（opset/simplify）、dynamic、NMS 阈值及执行按钮。"""
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

        # 5) ONNX 部署预处理元数据控件联动（仅 ONNX 启用；stretch 无补边值）
        pre_key = str(self.export_preprocess_combo.currentData() or DEFAULT_EXPORT_PREPROCESS_MODE)
        self.export_preprocess_combo.setEnabled(is_onnx)
        uses_pad_val = is_onnx and pre_key != "stretch"
        self.export_pad_value_label.setEnabled(uses_pad_val)
        self.export_pad_value_spin.setEnabled(uses_pad_val)
        if not is_onnx:
            tip = f"自定义预处理元数据写入仅对 ONNX 格式生效，当前格式 ({fmt}) 不适用"
            self.export_preprocess_combo.setToolTip(tip)
            self.export_pad_value_spin.setToolTip(tip)
        else:
            self.export_preprocess_combo.setToolTip(
                "写入 ONNX 自身元数据 (metadata_props) 的预处理契约，供下游推理端自动读取（不改动计算图）：\n"
                "• 居中填充 (letterbox_center，推荐): YOLO 标准等比缩放 + 四周居中补边；\n"
                "• 左上靠边填充 (letterbox_top_left): 等比缩放贴左上角，仅在右下补边；\n"
                "• 直接拉伸填充 (stretch): 不保持宽高比，直接拉伸至导出高×宽（无补边）。"
            )
            if pre_key == "stretch":
                self.export_pad_value_spin.setToolTip("直接拉伸 (Stretch) 模式不进行补边，不使用补边灰度值")
            else:
                self.export_pad_value_spin.setToolTip(
                    "LetterBox 补边像素灰度值（0=纯黑，114=YOLO 标准灰边，255=纯白）"
                )

        self._refresh_export_batch_tooltip()
        self._refresh_action_btn_states()
        self._refresh_export_io_preview()

    def build_export_config(self, *, allow_missing_weights: bool = False) -> ExportConfig:
        """从界面导出面板读取并组装完整的 `ExportConfig` 配置对象。"""
        w_text = self.weights_picker.text()
        if not allow_missing_weights:
            if not w_text or not Path(w_text).is_file():
                raise ValueError("请选择有效的 .pt 权重文件。")
            if is_exported_model_path(Path(w_text)):
                raise ValueError("模型导出仅支持 PyTorch (.pt) 源权重，请先切回 .pt 权重。")

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

        preprocess_mode = (
            str(self.export_preprocess_combo.currentData() or DEFAULT_EXPORT_PREPROCESS_MODE)
            if self.export_preprocess_combo.isEnabled()
            else DEFAULT_EXPORT_PREPROCESS_MODE
        )
        padding_value = (
            self.export_pad_value_spin.value()
            if self.export_pad_value_spin.isEnabled()
            else DEFAULT_EXPORT_PADDING_VALUE
        )

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
            preprocess_mode=preprocess_mode,
            padding_value=padding_value,
        )

    def _refresh_export_io_preview(self, *_args: object) -> None:
        """根据当前导出面板控件值与已读取权重元数据，实时刷新输入/输出张量说明及推理参数摘要。"""
        cfg = self.build_export_config(allow_missing_weights=True)
        is_exported = self._current_model_is_exported()
        meta_for_export = None if is_exported else self._weights_metadata
        io_text, warnings = describe_export_io(cfg, metadata=meta_for_export)
        if is_exported:
            warnings = [
                "当前上方选中的是已导出模型，模型导出功能仅支持 PyTorch (.pt) 源权重。",
                *warnings,
            ]

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

        if hasattr(self, "pred_summary_label"):
            self._refresh_pred_summary()

    def apply_device_info(self, info: DeviceInfo) -> None:
        """异步硬件探测完成后刷新运行设备开关与导出 FP16 可用状态（保留用户已手动修改的选择）。"""
        self.device_info = info
        self.device_ctrl.apply_device_info(info)
        self._on_export_format_changed()

    def set_weights_and_yaml(self, weights_path: str, yaml_path: str) -> None:
        """跨页写入：同步更新待评估模型与 `data.yaml` 路径，并立即触发模型元数据探测。"""
        if weights_path:
            self._preserve_export_dims_once = False
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
        """互斥启用或禁用验证、推理与导出三个执行按钮及快捷按钮，防止后台任务冲突。"""
        self._busy = busy
        self._refresh_action_btn_states()
        self.sync_export_pred_btn.setEnabled((not busy) and (not self._current_model_is_exported()))
        if self.switch_model_btn.isVisible():
            self.switch_model_btn.setEnabled(not busy)

    def _on_validate(self) -> None:
        """校验模型文件与 `data.yaml` 后在后台启动 `ValidateWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())
        w = self.weights_picker.text()
        y = self.yaml_picker.text()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的模型文件（.pt / .onnx / .torchscript / .engine）。")
            return
        if not y or not Path(y).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的 data.yaml 文件。")
            return
        device = self.device_ctrl.device()
        if detect_model_format(Path(w)) == "engine" and device.strip().lower() == "cpu":
            QMessageBox.warning(
                self,
                "提示",
                "TensorRT (.engine) 模型仅支持在 CUDA GPU 上验证，请在上方启用 GPU 加速后再运行。",
            )
            return

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "模型验证 (Val)"):
            return

        meta = self._weights_metadata
        if (
            meta is not None
            and meta.is_exported
            and meta.export_dynamic is False
            and meta.train_imgsz is not None
        ):
            mh, mw = meta.train_imgsz
            val_imgsz: int | tuple[int, int] = mh if mh == mw else (mh, mw)
        else:
            val_imgsz = self.imgsz_spin.value()

        self._set_busy(True)
        self.log_box.clear()
        self.log_box.appendPlainText("开始执行模型验证...")

        self.val_worker = ValidateWorker(
            Path(w),
            Path(y),
            imgsz=val_imgsz,
            device=device,
            nms=self._resolve_eval_head_nms(),
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
        """校验模型文件与测试输入路径后在后台启动 `PredictWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())
        w = self.weights_picker.text()
        src = self.source_edit.text().strip()
        if not w or not Path(w).is_file():
            QMessageBox.warning(self, "提示", "请选择有效的模型文件（.pt / .onnx / .torchscript / .engine）。")
            return
        if not src or not Path(src).exists():
            QMessageBox.warning(self, "提示", "请选择有效的测试输入文件或目录。")
            return
        device = self.device_ctrl.device()
        if detect_model_format(Path(w)) == "engine" and device.strip().lower() == "cpu":
            QMessageBox.warning(
                self,
                "提示",
                "TensorRT (.engine) 模型仅支持在 CUDA GPU 上推理，请在上方启用 GPU 加速后再运行。",
            )
            return

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "模型推理预览"):
            return

        params = self.build_predict_params()
        out_dir = Path(self.pred_out_picker.text() or (self.workspace_root / "runs" / "predict_preview"))
        self._set_busy(True)
        self.log_box.clear()
        self.log_box.appendPlainText("开始推理测试...")

        self.pred_worker = PredictWorker(
            weights_path=Path(w),
            source_path=Path(src),
            output_dir=out_dir,
            conf=float(params["conf"]),  # type: ignore[arg-type]
            iou=float(params["iou"]),  # type: ignore[arg-type]
            imgsz=params["imgsz"],  # type: ignore[arg-type]
            device=device,
            nms=params["nms"],  # type: ignore[arg-type]
            rect=bool(params["rect"]),
            max_det=int(params["max_det"]),  # type: ignore[arg-type]
            preprocess_mode=str(params.get("preprocess_mode") or DEFAULT_EXPORT_PREPROCESS_MODE),
            padding_value=(
                int(params["padding_value"])
                if isinstance(params.get("padding_value"), int)
                else None
            ),
            render_config=(
                params["render_config"]
                if isinstance(params.get("render_config"), PredictRenderConfig)
                else self.build_predict_render_config()
            ),
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
                if self.preview_label.set_preview_image(p):
                    pix = self.preview_label.source_pixmap()
                    dims = f" ({pix.width()}×{pix.height()})" if pix is not None else ""
                    self.preview_info_label.setText(f"{p.name}{dims} · 双击图片可放大查看")
                    self.preview_info_label.setToolTip(str(p))
                    self.zoom_preview_btn.setEnabled(True)
                    break

    def _on_zoom_preview(self) -> None:
        """打开独立缩放对话框，全分辨率查看当前推理预览结果图。"""
        pix = self.preview_label.source_pixmap()
        if pix is None or pix.isNull():
            return
        dlg = _PreviewZoomDialog(pix, image_path=self.preview_label.image_path(), parent=self)
        dlg.exec()

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

        self._last_pt_weights_path = cfg.weights_path
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
        exp_path = Path(exported_path)
        if exp_path.is_file():
            self._last_exported_path = exp_path
        self._set_busy(False)
        self._refresh_eval_controls_state()
        self.log_box.appendPlainText(f"\n=== 导出成功 ===\n文件路径: {exported_path}")
        self.log_box.appendPlainText(
            "[提示] 可点击「2. 推理效果测试」中的「测刚导出模型」一键切换并实测该导出模型的推理效果。"
        )
        QMessageBox.information(self, "导出成功", f"模型已导出到:\n{exported_path}")

    def _on_failed(self, err: str) -> None:
        self._set_busy(False)
        self.log_box.appendPlainText(f"\n[错误] {err}")
        QMessageBox.critical(self, "执行错误", err)
