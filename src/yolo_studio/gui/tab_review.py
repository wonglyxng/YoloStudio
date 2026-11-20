"""标签页 3：基准模型自动标注与 MLLM 纠错补漏界面。

职责：
    配置候选框来源（基准 `.pt` 低阈值推理或已有 X-AnyLabeling `.json`）、MLLM 接口与预设、
    连通性测试、边框紧贴标准化修正（标记过松 / MLLM 局部图精修 / 本地 SAM 像素级收紧）与审核分流策略；
    右侧提供带实时计数的分流状态筛选、语义配色结果表与运行日志（同区双页签）、自适应缩放的 Before/After 对比图、
    历史报告加载，并支持一键跳转「2. 标注工作台」精修或发送清洗结果至数据集准备页；
    左右两栏置于可拖拽的水平分割条中，用户可按屏幕宽度调整比例。

运行环境与协作边界：
    运行于 PyQt6 主线程；后台审核由 `MllmReviewWorker` 执行，接口连通性测试由
    `MllmConnectionTestWorker` 执行，`render_review_comparison` 在点击行时即时渲染对比图并做 LRU 缓存，
    通过 `open_in_workbench` 与 `send_cleaned_dataset` 信号与 `MainWindow` 协同；
    持久化配置通过 `save_settings` / `restore_settings` 接受外部注入的 `QSettings`（默认不保存明文 API Key）。
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QColor, QFont, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.mllm_reviewer import (
    BOX_REFINE_MODE_HINTS,
    BOX_REFINE_MODE_ITEMS,
    BUCKET_AUTO_PASSED,
    BUCKET_LLM_FIXED,
    BUCKET_NEED_REVIEW,
    GEMINI_MODELS,
    OPENAI_PRESETS,
    SAM_REFINE_MODELS,
    BoxRefineMode,
    ImageReviewRecord,
    MllmProvider,
    MllmReviewConfig,
    MllmReviewResult,
    ReviewSourceMode,
    count_review_images,
    format_record_detail_summary,
    load_review_report,
    render_review_comparison,
)
from yolo_studio.core.trainer import DeviceInfo, detect_compute_device
from yolo_studio.gui.common import (
    LOG_MAX_LINES,
    ComputeDeviceControl,
    GpuTaskGuard,
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    open_local_path,
)
from yolo_studio.gui.workers import MllmConnectionTestWorker, MllmReviewWorker

# 左右对比预览 JPEG 字节流的 LRU 缓存容量上限（按结果表行号索引，每次重新开始审核或加载报告时清空）
_PREVIEW_CACHE_MAX = 64
_REVIEW_SETTINGS_KEY = "review_tab/state_json"


class _ScaledPreviewLabel(QLabel):
    """支持随窗口或分割条拖拽自动按比例重缩放的对比图预览标签。"""

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet(
            "border: 1px dashed #999999; border-radius: 6px; background: #1e1e1e; color: #cccccc;"
        )

    def set_preview_pixmap(self, pixmap: QPixmap | None) -> None:
        """设置原始高分辨率对比图并立即按当前控件尺寸等比缩放显示；传 `None` 时恢复占位文案。"""
        if pixmap is None or pixmap.isNull():
            self._source_pixmap = None
            self.setPixmap(QPixmap())
            self.setText(self._placeholder)
            return
        self._source_pixmap = pixmap
        self._apply_scaled()

    def clear_preview(self, message: str | None = None) -> None:
        """清空当前图像并显示占位提示或指定文本。"""
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(message if message is not None else self._placeholder)

    def _apply_scaled(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target_w = max(240, self.width() - 10)
        target_h = max(160, self.height() - 10)
        scaled = self._source_pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._source_pixmap is not None:
            self._apply_scaled()


class MllmReviewTab(QWidget):
    """MLLM 自动标注纠错、补漏与边框紧贴修正功能标签页（标签页 3）。

    内部状态：
        持有硬件探测信息 `device_info`、跨页计算设备互斥协调器 `gpu_guard`、后台审核线程 `worker`、
        连通性探测线程 `conn_test_worker`、逐图审核记录 `_records`、对比图 LRU 缓存 `_preview_cache`、
        当前表格筛选器 `_active_filter`、配置持久化脏标记 `_settings_dirty`、
        最近输出目录 `_last_output_dir` 与类别文件 `_last_classes_txt`。
    """

    send_cleaned_dataset = Signal(str, str, bool)  # 信号参数：(输出目录, classes.txt 路径, 是否按子目录递归扫描)
    open_in_workbench = Signal(str, str, str)  # 信号参数：(目标目录, classes.txt 路径, 可选指定图片路径)

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
        self.worker: MllmReviewWorker | None = None
        self.conn_test_worker: MllmConnectionTestWorker | None = None
        self._records: list[ImageReviewRecord] = []
        self._preview_cache: OrderedDict[int, bytes] = OrderedDict()
        self._active_filter: str = "all"
        self._selected_row: int = -1
        self._restoring_state: bool = False
        self._settings_dirty: bool = False
        self._last_output_dir: Path = workspace_root / "data" / "reviewed_frames"
        self._last_classes_txt: str = ""
        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(150)
        self._count_timer.timeout.connect(self._refresh_images_count)
        self._init_ui()

    def apply_device_info(self, info: DeviceInfo) -> None:
        """异步硬件探测完成后刷新基准推理设备开关与提示（保留用户已手动修改的选择）。"""
        self.device_info = info
        self.yolo_device_ctrl.apply_device_info(info)

    def set_images_dir(self, path_str: str) -> None:
        """跨页写入：同步待纠错图片目录输入框并立即刷新图片计数。"""
        if path_str:
            self.images_picker.setText(path_str)
            self._refresh_images_count()

    def set_weights_path(self, path_str: str) -> None:
        """跨页写入：将新训练的 `best.pt` 路径更新至基准模型输入框。"""
        if path_str:
            self.weights_picker.setText(path_str)

    @staticmethod
    def _tune_form(form: QFormLayout) -> None:
        """统一左侧三组表单的标签右对齐与留白节奏，保证标签紧邻右侧控件并缓解拥挤感。"""
        configure_form_layout(
            form,
            horizontal_spacing=10,
            vertical_spacing=8,
            margins=(12, 10, 12, 12),
        )

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 左右两栏放入可拖拽的水平分割条，用户可按屏幕宽度自行调整参数区与结果区比例
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(10)

        # 左侧面板：滚动表单区 + 底部固定操作栏
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 0, 4, 0)
        left.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        form_stack = QVBoxLayout(scroll_content)
        form_stack.setContentsMargins(0, 0, 6, 0)
        form_stack.setSpacing(12)

        # 1. 待纠错数据来源与基准模型
        src_group = QGroupBox("1. 待纠错数据来源与基准模型")
        s_form = QFormLayout(src_group)
        self._tune_form(s_form)
        self._src_form = s_form

        self.source_mode_combo = QComboBox()
        self.source_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source_mode_combo.setMinimumContentsLength(12)
        self.source_mode_combo.addItem(
            "基准模型低阈值推理 + MLLM 纠错（推荐）", ReviewSourceMode.YOLO_PREDICT
        )
        self.source_mode_combo.addItem(
            "读取已有 X-AnyLabeling .json + MLLM 复核", ReviewSourceMode.EXISTING_JSON
        )
        self.source_mode_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        s_form.addRow("工作模式:", self.source_mode_combo)

        default_frames = str((self.workspace_root / "data" / "raw_frames").resolve())
        img_dir_box = QVBoxLayout()
        img_dir_box.setContentsMargins(0, 0, 0, 0)
        img_dir_box.setSpacing(3)
        self.images_picker = PathPickerRow(
            placeholder="待自动标注或复核的图片目录",
            mode="dir",
            default_path=default_frames,
        )
        # 递归开关决定从图片目录取哪些图片，作为图片目录的从属选项紧贴输入框放置；
        # 默认开启，与内嵌标注工作台（递归列出子目录图片）及数据集页保持一致
        self.recursive_chk = QCheckBox("包含子目录")
        self.recursive_chk.setChecked(True)
        self.recursive_chk.setToolTip(
            "默认开启，与标注工作台一致（工作台打开目录时同样包含子目录中的图片）；"
            "输出目录位于图片目录内部时会自动跳过，避免重复审核上一轮结果；只想处理顶层图片时取消勾选"
        )
        self.images_count_label = QLabel()
        self.images_count_label.setStyleSheet("color: #57606a; font-size: 12px;")
        self.images_count_label.setWordWrap(True)
        img_dir_box.addWidget(self.images_picker)
        img_dir_box.addWidget(self.recursive_chk)
        img_dir_box.addWidget(self.images_count_label)
        s_form.addRow("图片目录:", img_dir_box)

        self.weights_picker = PathPickerRow(
            placeholder="选择基准模型 best.pt 权重（先手动标一批训练得出）",
            mode="file",
            file_filter="PyTorch 权重 (*.pt);;所有文件 (*.*)",
        )
        s_form.addRow("基准模型 (.pt):", self.weights_picker)

        self.yolo_params_widget = QWidget()
        yolo_row = QHBoxLayout(self.yolo_params_widget)
        yolo_row.setContentsMargins(0, 0, 0, 0)
        yolo_row.setSpacing(4)
        self.yolo_conf_spin = QDoubleSpinBox()
        self.yolo_conf_spin.setRange(0.01, 0.95)
        self.yolo_conf_spin.setSingleStep(0.02)
        self.yolo_conf_spin.setValue(0.12)
        self.yolo_conf_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.yolo_conf_spin.setToolTip("建议设低（0.08~0.15）以尽量召回潜在漏标，再由 MLLM 筛除误检")
        self.yolo_iou_spin = QDoubleSpinBox()
        self.yolo_iou_spin.setRange(0.10, 0.95)
        self.yolo_iou_spin.setSingleStep(0.05)
        self.yolo_iou_spin.setValue(0.60)
        self.yolo_iou_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.yolo_imgsz_spin = QSpinBox()
        self.yolo_imgsz_spin.setRange(128, 2048)
        self.yolo_imgsz_spin.setSingleStep(32)
        self.yolo_imgsz_spin.setValue(640)
        self.yolo_imgsz_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.yolo_device_ctrl = ComputeDeviceControl(
            self.device_info,
            show_status_text=False,
            allow_multi_gpu=False,
            checkbox_text="GPU 加速",
            compact_status=True,
        )
        yolo_row.addWidget(QLabel("低阈值Conf:"))
        yolo_row.addWidget(self.yolo_conf_spin)
        yolo_row.addSpacing(4)
        yolo_row.addWidget(QLabel("IoU:"))
        yolo_row.addWidget(self.yolo_iou_spin)
        yolo_row.addSpacing(4)
        yolo_row.addWidget(QLabel("ImgSz:"))
        yolo_row.addWidget(self.yolo_imgsz_spin)
        yolo_row.addSpacing(6)
        yolo_row.addWidget(self.yolo_device_ctrl, 1)
        s_form.addRow("基准推理参数:", self.yolo_params_widget)

        self.ann_picker = PathPickerRow(
            placeholder="可选：已有 .json 目录（留空则同图片目录）",
            mode="dir",
        )
        s_form.addRow("已有标注目录:", self.ann_picker)

        default_out = str((self.workspace_root / "data" / "reviewed_frames").resolve())
        self.output_picker = PathPickerRow(
            placeholder="清洗后图片与 X-AnyLabeling .json 输出目录",
            mode="dir",
            default_path=default_out,
        )
        s_form.addRow("清洗输出目录:", self.output_picker)
        form_stack.addWidget(src_group)

        # 2. MLLM 接口与模型配置
        llm_group = QGroupBox("2. 多模态大语言模型 (MLLM) 配置")
        l_form = QFormLayout(llm_group)
        self._tune_form(l_form)
        self._llm_form = l_form

        self.provider_combo = QComboBox()
        self.provider_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.provider_combo.setMinimumContentsLength(12)
        self.provider_combo.addItem("Google Gemini API (google-genai)", MllmProvider.GEMINI)
        self.provider_combo.addItem(
            "OpenAI 兼容接口 (Qwen-VL / 硅基流动 / Ollama / vLLM)", MllmProvider.OPENAI_COMPAT
        )
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        l_form.addRow("模型接口:", self.provider_combo)

        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(12)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        l_form.addRow("常用预设:", self.preset_combo)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("OpenAI 兼容接口 Base URL（Gemini 模式无需填写）")
        l_form.addRow("Base URL:", self.base_url_edit)

        self.model_edit = QLineEdit("gemini-3.8-flash")
        l_form.addRow("模型名称:", self.model_edit)

        key_box = QVBoxLayout()
        key_box.setContentsMargins(0, 0, 0, 0)
        key_box.setSpacing(4)
        key_row = QHBoxLayout()
        key_row.setSpacing(6)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("留空则自动读取环境变量 GEMINI_API_KEY / OPENAI_API_KEY")
        self.show_key_chk = QCheckBox("显示")
        self.show_key_chk.toggled.connect(
            lambda checked: self.api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        self.test_conn_btn = QPushButton("测试连接")
        self.test_conn_btn.setStyleSheet("padding: 4px 10px;")
        self.test_conn_btn.setToolTip("发送极简纯文本探针测试当前接口、Key 与模型名是否畅通（不上传图片）")
        key_row.addWidget(self.api_key_edit, 1)
        key_row.addWidget(self.show_key_chk)
        key_row.addWidget(self.test_conn_btn)
        self.api_key_status_label = QLabel()
        self.api_key_status_label.setStyleSheet("color: #57606a; font-size: 12px;")
        key_box.addLayout(key_row)
        key_box.addWidget(self.api_key_status_label)
        l_form.addRow("API Key:", key_box)
        form_stack.addWidget(llm_group)

        # 3. 纠错、补漏、边框紧贴修正与分流策略
        pol_group = QGroupBox("3. 纠错、补漏、边框修正与分流策略")
        p_form = QFormLayout(pol_group)
        self._tune_form(p_form)
        self._pol_form = p_form

        cls_row = QHBoxLayout()
        cls_row.setSpacing(6)
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("英文逗号分隔，如 person, helmet（留空自动从模型或 .json 提取）")
        self.import_classes_btn = QPushButton("导入 classes.txt")
        self.import_classes_btn.setStyleSheet("padding: 4px 8px;")
        self.import_classes_btn.setToolTip("从本地 classes.txt 读取类别名称并填充到左侧输入框")
        cls_row.addWidget(self.classes_edit, 1)
        cls_row.addWidget(self.import_classes_btn)
        p_form.addRow("目标类别清单:", cls_row)

        self.guideline_edit = QLineEdit()
        self.guideline_edit.setPlaceholderText("可选标注规则，如：远景模糊人影不标；框需紧贴主体不含特效背景")
        p_form.addRow("辅助判定规则:", self.guideline_edit)

        refine_box = QVBoxLayout()
        refine_box.setContentsMargins(0, 0, 0, 0)
        refine_box.setSpacing(3)
        self.refine_mode_combo = QComboBox()
        self.refine_mode_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.refine_mode_combo.setMinimumContentsLength(12)
        for mode, label in BOX_REFINE_MODE_ITEMS:
            self.refine_mode_combo.addItem(label, mode)
        self.refine_mode_combo.setToolTip(
            "解决预训练模型框画偏大、包进多余背景的问题：可仅标记过松框转人工、由 MLLM 在局部放大图精修，或调用本地 SAM 像素级收紧"
        )
        self.refine_hint_label = QLabel()
        self.refine_hint_label.setWordWrap(True)
        self.refine_hint_label.setStyleSheet("color: #57606a; font-size: 12px;")
        refine_box.addWidget(self.refine_mode_combo)
        refine_box.addWidget(self.refine_hint_label)
        p_form.addRow("边框紧贴修正:", refine_box)

        # SAM 参数行：仅在选中 `SAM_LOCAL` 时展示，其余模式通过 setRowVisible 折叠保持界面清爽
        self.sam_row_widget = QWidget()
        sam_row = QHBoxLayout(self.sam_row_widget)
        sam_row.setContentsMargins(0, 0, 0, 0)
        sam_row.setSpacing(6)
        self.sam_model_combo = QComboBox()
        self.sam_model_combo.setEditable(True)
        self.sam_model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sam_model_combo.setMinimumContentsLength(10)
        for sam_file, sam_desc in SAM_REFINE_MODELS:
            self.sam_model_combo.addItem(sam_desc, sam_file)
        self.sam_model_combo.setCurrentIndex(0)
        self.pick_sam_btn = QPushButton("选 .pt")
        self.pick_sam_btn.setStyleSheet("padding: 4px 8px;")
        self.sam_refine_all_chk = QCheckBox("全部保留框收紧")
        self.sam_refine_all_chk.setToolTip(
            "未勾选（默认）：仅对 MLLM 判定过松的框调用 SAM 像素级收紧；勾选：对全部保留/改类框统一执行 SAM 收紧"
        )
        sam_row.addWidget(self.sam_model_combo, 1)
        sam_row.addWidget(self.pick_sam_btn)
        sam_row.addWidget(self.sam_refine_all_chk)
        p_form.addRow("SAM 分割模型:", self.sam_row_widget)
        self._sam_row_label = p_form.labelForField(self.sam_row_widget)

        triage_row = QHBoxLayout()
        triage_row.setSpacing(6)
        self.high_conf_label = QLabel("免审阈值 ≥")
        self.high_conf_spin = QDoubleSpinBox()
        self.high_conf_spin.setRange(0.10, 1.00)
        self.high_conf_spin.setSingleStep(0.05)
        self.high_conf_spin.setValue(0.85)
        # 留足 Win11 原生并排上下箭头的宽度，防止两位小数被截断
        self.high_conf_spin.setMinimumWidth(104)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(2)
        self.workers_spin.setMinimumWidth(84)
        triage_row.addWidget(self.high_conf_label)
        triage_row.addWidget(self.high_conf_spin)
        triage_row.addSpacing(18)
        triage_row.addWidget(QLabel("并发数:"))
        triage_row.addWidget(self.workers_spin)
        triage_row.addStretch(1)
        p_form.addRow("阈值与并发:", triage_row)

        # 处理数量单独成行：试跑开关与上限分离于阈值行，避免一行塞入过多控件
        limit_row = QHBoxLayout()
        limit_row.setSpacing(6)
        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(0, 100000)
        self.max_images_spin.setValue(0)
        self.max_images_spin.setSpecialValueText("全部")
        self.max_images_spin.setSuffix(" 张")
        self.max_images_spin.setMinimumWidth(104)
        self.max_images_spin.setToolTip(
            "0（显示为“全部”）表示处理目录中所有图片；可设为 5 等小数值先小批量试跑验证规则与收框效果，避免误耗 API 额度"
        )
        self.trial_5_btn = QPushButton("试跑 5 张")
        self.trial_5_btn.setCheckable(True)
        self.trial_5_btn.setStyleSheet(
            "QPushButton { padding: 3px 10px; font-size: 12px; }"
            "QPushButton:checked { background-color: #0969da; color: #ffffff; border-color: #085ec4; font-weight: 600; }"
        )
        self.trial_5_btn.setToolTip("一键切换仅审核前 5 张图用于快速试跑，再次点击恢复处理全部图片")
        limit_hint = QLabel("首次调整规则或收框模式时，建议先试跑验证效果")
        limit_hint.setStyleSheet("color: #8c959f; font-size: 12px;")
        limit_row.addWidget(self.max_images_spin)
        limit_row.addWidget(self.trial_5_btn)
        limit_row.addSpacing(8)
        limit_row.addWidget(limit_hint)
        limit_row.addStretch(1)
        p_form.addRow("处理数量:", limit_row)

        self.check_missing_chk = QCheckBox("检查漏标目标")
        self.check_missing_chk.setChecked(True)
        self.add_missing_chk = QCheckBox("尝试直接补充漏标框")
        self.add_missing_chk.setChecked(True)
        self.crops_chk = QCheckBox("附带局部裁剪放大图")
        self.crops_chk.setChecked(True)
        self.triage_dirs_chk = QCheckBox("按结果分建子目录")
        self.triage_dirs_chk.setChecked(True)
        self.triage_dirs_chk.setToolTip("在输出目录下自动分建 auto_passed / llm_fixed / need_review 三个子目录")

        def _make_check_grid(boxes: list[QCheckBox]) -> QWidget:
            """按统一的 3 列等宽网格排列复选框，保证“纠错选项”与“输出选项”两行纵向对齐。"""
            holder = QWidget()
            grid = QGridLayout(holder)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(12)
            for col, box in enumerate(boxes):
                grid.addWidget(box, 0, col)
            for col in range(3):
                grid.setColumnStretch(col, 1)
            return holder

        p_form.addRow(
            "纠错选项:", _make_check_grid([self.check_missing_chk, self.add_missing_chk, self.crops_chk])
        )
        p_form.addRow("输出选项:", _make_check_grid([self.triage_dirs_chk]))
        align_form_labels([s_form, l_form, p_form])
        form_stack.addWidget(pol_group)
        form_stack.addStretch(1)

        scroll.setWidget(scroll_content)
        left.addWidget(scroll, 1)

        # 底部操作栏：主操作（开始/停止）靠左，结果后续操作靠右，并用分隔线与表单区隔开
        btn_sep = QFrame()
        btn_sep.setFrameShape(QFrame.Shape.HLine)
        btn_sep.setStyleSheet("color: #d0d7de;")
        left.addWidget(btn_sep)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("开始 MLLM 纠错清洗")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setMinimumWidth(160)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.open_need_review_btn = QPushButton("在工作台打开待复核目录")
        self.open_out_btn = QPushButton("打开输出目录")
        self.send_ds_btn = QPushButton("发送到「4. 数据集准备」→")
        self.send_ds_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.open_out_btn)
        btn_row.addWidget(self.open_need_review_btn)
        btn_row.addWidget(self.send_ds_btn)
        left.addLayout(btn_row)

        # 右侧面板：进度与汇总、分流筛选栏 + 历史报告加载、结果列表/运行日志页签 + 自适应对比预览
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(4, 0, 0, 0)
        right.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        right.addWidget(self.progress_bar)

        self.summary_label = QLabel("就绪：配置基准模型与 MLLM 后点击「开始 MLLM 纠错清洗」，或加载历史审核报告继续复核。")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #424a53;")
        right.addWidget(self.summary_label)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        self.filter_buttons: dict[str, QPushButton] = {}
        filter_specs = (
            ("all", "全部"),
            (BUCKET_AUTO_PASSED, "自动通过"),
            (BUCKET_LLM_FIXED, "已删改/收框"),
            (BUCKET_NEED_REVIEW, "待复核"),
            ("error", "出错"),
        )
        filter_btn_qss = (
            "QPushButton { padding: 3px 10px; font-size: 12px; }"
            "QPushButton:checked { background-color: #0969da; color: #ffffff; border-color: #085ec4; font-weight: 600; }"
        )
        for key, label in filter_specs:
            btn = QPushButton(f"{label} (0)")
            btn.setCheckable(True)
            btn.setChecked(key == "all")
            btn.setStyleSheet(filter_btn_qss)
            btn.clicked.connect(lambda _checked, k=key: self.set_result_filter(k))
            self.filter_buttons[key] = btn
            filter_row.addWidget(btn)
        filter_row.addStretch(1)
        self.load_report_btn = QPushButton("加载历史报告...")
        self.load_report_btn.setStyleSheet("padding: 3px 10px; font-size: 12px;")
        self.load_report_btn.setToolTip("读取历史清洗目录中的 review_report.json，直接还原结果表与对比预览继续复核")
        filter_row.addWidget(self.load_report_btn)
        right.addLayout(filter_row)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)

        # 结果列表与运行日志合并为同一区域的两个页签，省出一整块纵向空间留给对比预览
        table_group = QGroupBox("4. 审核结果与日志")
        table_group.setToolTip("单击结果行查看对比图与判定理由，双击直达「2. 标注工作台」精修")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(8, 8, 8, 8)
        self.result_tabs = QTabWidget()
        self.result_tabs.setDocumentMode(True)

        self.result_table = QTableWidget(0, 8)
        self.result_table.setHorizontalHeaderLabels(
            ["图片", "分流结果", "保留", "改类", "删误检", "收框", "补漏", "说明"]
        )
        header = self.result_table.horizontalHeader()
        header.setMinimumSectionSize(44)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.result_table.setColumnWidth(0, 155)
        self.result_table.setColumnWidth(1, 88)
        for num_col in (2, 3, 4, 5, 6):
            header.setSectionResizeMode(num_col, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        self.result_table.verticalHeader().setDefaultSectionSize(28)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.currentCellChanged.connect(self._on_row_selected)
        self.result_table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self.result_tabs.addTab(self.result_table, "结果列表")

        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("MLLM 逐张审核日志将在此输出...")
        self.result_tabs.addTab(self.log_box, "运行日志")
        table_layout.addWidget(self.result_tabs)
        splitter.addWidget(table_group)

        preview_group = QGroupBox("5. 纠错前后对比预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.setSpacing(6)
        self.legend_label = QLabel(
            "框色图例："
            '<span style="color:#16a34a; font-weight:600;">■</span> 保留&nbsp;&nbsp;&nbsp;'
            '<span style="color:#ea580c; font-weight:600;">■</span> 改类&nbsp;&nbsp;&nbsp;'
            '<span style="color:#ca8a04; font-weight:600;">■</span> 补漏&nbsp;&nbsp;&nbsp;'
            '<span style="color:#0284c7; font-weight:600;">■</span> 已收框(虚线为原框)&nbsp;&nbsp;&nbsp;'
            '<span style="color:#c026d3; font-weight:600;">■</span> 过松/超门禁待修'
        )
        self.legend_label.setTextFormat(Qt.TextFormat.RichText)
        self.legend_label.setStyleSheet("color: #57606a; font-size: 12px;")
        preview_layout.addWidget(self.legend_label)

        self.preview_label = _ScaledPreviewLabel(
            "单击结果行对比「纠错前 vs 纠错后」标注框\n双击行或点击右下按钮可在「2. 标注工作台」精修"
        )
        preview_layout.addWidget(self.preview_label, 1)

        detail_row = QHBoxLayout()
        detail_row.setContentsMargins(0, 2, 0, 0)
        detail_row.setSpacing(10)
        self.detail_label = QLabel("未选中图片：单击上方结果行可查看 MLLM 逐框判定理由与收框明细。")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: #424a53; font-size: 12px;")
        self.open_selected_btn = QPushButton("在工作台精修此图 →")
        self.open_selected_btn.setStyleSheet("padding: 4px 12px; font-size: 12px;")
        self.open_selected_btn.setEnabled(False)
        self.open_selected_btn.setToolTip("跳转到「2. 标注工作台」并精确打开当前选中的图片微调标注框")
        detail_row.addWidget(self.detail_label, 1)
        detail_row.addWidget(self.open_selected_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        preview_layout.addLayout(detail_row)
        splitter.addWidget(preview_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([260, 400])
        right.addWidget(splitter, 1)

        left_widget.setMinimumWidth(560)
        right_widget.setMinimumWidth(440)
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(right_widget)
        self.main_splitter.setStretchFactor(0, 5)
        self.main_splitter.setStretchFactor(1, 4)
        self.main_splitter.setSizes([5000, 4000])
        root.addWidget(self.main_splitter)

        # 信号与槽连接
        self.images_picker.line_edit.textChanged.connect(self._schedule_images_count)
        self.recursive_chk.toggled.connect(self._schedule_images_count)
        self.output_picker.line_edit.textChanged.connect(self._schedule_images_count)
        self.max_images_spin.valueChanged.connect(self._on_max_images_changed)
        self.trial_5_btn.toggled.connect(self._on_trial_5_toggled)
        self.test_conn_btn.clicked.connect(self._on_test_connection)
        self.import_classes_btn.clicked.connect(self._on_import_classes_file)
        self.pick_sam_btn.clicked.connect(self._on_pick_sam_pt)
        self.load_report_btn.clicked.connect(self._on_load_report_dialog)
        self.open_selected_btn.clicked.connect(self._on_open_selected_in_xal)
        self.api_key_edit.textChanged.connect(self._refresh_api_key_status)
        self.base_url_edit.textChanged.connect(self._refresh_api_key_status)
        self.refine_mode_combo.currentIndexChanged.connect(self._sync_strategy_states)
        self.sam_refine_all_chk.toggled.connect(self._sync_strategy_states)
        self.check_missing_chk.toggled.connect(self._sync_strategy_states)
        self.crops_chk.toggled.connect(self._sync_strategy_states)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_need_review_btn.clicked.connect(self._on_open_need_review_in_xal)
        self.open_out_btn.clicked.connect(lambda: open_local_path(self.output_picker.path()))
        self.send_ds_btn.clicked.connect(self._on_send_to_dataset)

        self._bind_settings_dirty_watchers()
        self._on_source_mode_changed()
        self._on_provider_changed()
        self._sync_strategy_states()
        self._refresh_api_key_status()
        self._refresh_images_count()

    def _bind_settings_dirty_watchers(self) -> None:
        """监听配置改动置位持久化脏标记，避免空打开/关闭窗口或测试覆写用户偏好。"""
        for combo in (
            self.source_mode_combo,
            self.provider_combo,
            self.preset_combo,
            self.refine_mode_combo,
        ):
            combo.currentIndexChanged.connect(self._mark_settings_dirty)
        self.sam_model_combo.currentTextChanged.connect(self._mark_settings_dirty)
        for edit in (self.base_url_edit, self.model_edit, self.classes_edit, self.guideline_edit):
            edit.textChanged.connect(self._mark_settings_dirty)
        for spin in (
            self.yolo_conf_spin,
            self.yolo_iou_spin,
            self.yolo_imgsz_spin,
            self.high_conf_spin,
            self.workers_spin,
            self.max_images_spin,
        ):
            spin.valueChanged.connect(self._mark_settings_dirty)
        for chk in (
            self.check_missing_chk,
            self.add_missing_chk,
            self.crops_chk,
            self.triage_dirs_chk,
            self.recursive_chk,
            self.sam_refine_all_chk,
        ):
            chk.toggled.connect(self._mark_settings_dirty)

    def _mark_settings_dirty(self, *_args: object) -> None:
        if not self._restoring_state:
            self._settings_dirty = True

    def _schedule_images_count(self, *_args: object) -> None:
        """目录或递归勾选变更时防抖刷新待审核图片计数，避免连续输入路径时频繁扫盘。"""
        self._count_timer.start()

    def _refresh_images_count(self) -> None:
        """立即统计并展示待纠错图片数、递归提示及试跑上限提示（口径同实际审核的 `collect_review_images`）。

        未勾选递归时额外统计子目录图片：顶层没有图片而子目录有图片时以橙色提示勾选递归，
        这是按视频或分流子目录组织数据时最常见的配置遗漏；递归时排除位于输入目录内部的输出目录。
        """
        raw_dir = self.images_picker.text()
        if not raw_dir:
            self._set_count_hint("请选择待纠错图片目录", "#9a6700")
            return
        img_dir = Path(raw_dir)
        if not img_dir.is_dir():
            self._set_count_hint("当前目录不存在，请重新选择或先完成抽帧", "#9a6700")
            return
        out_text = self.output_picker.text()
        out_dir = Path(out_text) if out_text else None
        recursive = self.recursive_chk.isChecked()
        nested_total = count_review_images(img_dir, recursive=True, output_dir=out_dir)
        top_total = count_review_images(img_dir, recursive=False)
        total = nested_total if recursive else top_total
        if total == 0:
            if not recursive and nested_total > 0:
                self._set_count_hint(
                    f"顶层没有图片，子目录中有 {nested_total} 张，请勾选「包含子目录」。", "#b25900"
                )
            else:
                self._set_count_hint("该目录及子目录中没有图片", "#9a6700")
            return
        msg = f"共 {total} 张图片（含子目录）" if recursive else f"共 {total} 张图片（仅顶层）"
        if not recursive and nested_total > top_total:
            msg += f"；子目录另有 {nested_total - top_total} 张，勾选「包含子目录」可一并处理"
        limit = self.max_images_spin.value()
        if limit > 0 and total > limit:
            msg += f"（已设试跑上限：本次仅审核前 {limit} 张）"
        self._set_count_hint(msg, "#1a7f37")

    def _set_count_hint(self, text: str, color: str) -> None:
        self.images_count_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.images_count_label.setText(text)

    def _on_trial_5_toggled(self, checked: bool) -> None:
        """一键在“仅试跑前 5 张”与“处理全部图片”之间切换。"""
        target = 5 if checked else 0
        if self.max_images_spin.value() != target:
            self.max_images_spin.setValue(target)

    def _on_max_images_changed(self, value: int) -> None:
        """上限数值变化时同步快捷按钮勾选状态与目录计数提示。"""
        self.trial_5_btn.blockSignals(True)
        self.trial_5_btn.setChecked(value == 5)
        self.trial_5_btn.blockSignals(False)
        self._refresh_images_count()

    def _on_source_mode_changed(self) -> None:
        """按候选框来源模式启用/禁用并动态折叠无关表单行，减少无效置灰项干扰。"""
        is_yolo = self.source_mode_combo.currentData() == ReviewSourceMode.YOLO_PREDICT
        self.weights_picker.setEnabled(is_yolo)
        self.yolo_conf_spin.setEnabled(is_yolo)
        self.yolo_iou_spin.setEnabled(is_yolo)
        self.yolo_imgsz_spin.setEnabled(is_yolo)
        self.yolo_device_ctrl.setEnabled(is_yolo)
        self.ann_picker.setEnabled(not is_yolo)
        self._src_form.setRowVisible(self.weights_picker, is_yolo)
        self._src_form.setRowVisible(self.yolo_params_widget, is_yolo)
        self._src_form.setRowVisible(self.ann_picker, not is_yolo)

    def _on_provider_changed(self) -> None:
        """切换 Gemini 与 OpenAI 兼容接口时更新常用预设并按需折叠 Base URL 行。"""
        provider = self.provider_combo.currentData()
        is_gemini = provider == MllmProvider.GEMINI
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        if is_gemini:
            for model_id, label in GEMINI_MODELS:
                self.preset_combo.addItem(label, (model_id, ""))
            self.base_url_edit.setEnabled(False)
            self.base_url_edit.clear()
            self.model_edit.setText(GEMINI_MODELS[0][0])
        else:
            for name, url, model_id in OPENAI_PRESETS:
                self.preset_combo.addItem(name, (model_id, url))
            self.base_url_edit.setEnabled(True)
            self.base_url_edit.setText(OPENAI_PRESETS[0][1])
            self.model_edit.setText(OPENAI_PRESETS[0][2])
        self._llm_form.setRowVisible(self.base_url_edit, not is_gemini)
        self.preset_combo.blockSignals(False)
        self._refresh_api_key_status()

    def _on_preset_changed(self) -> None:
        data = self.preset_combo.currentData()
        if not data:
            return
        model_id, url = data
        self.model_edit.setText(model_id)
        if url:
            self.base_url_edit.setText(url)
        self._refresh_api_key_status()

    def _refresh_api_key_status(self, *_args: object) -> None:
        """实时检测手动 Key 或系统环境变量是否已就绪，减少启动任务后才报鉴权失败的挫败感。"""
        provider = self.provider_combo.currentData()
        manual_key = self.api_key_edit.text().strip()
        if manual_key:
            self.api_key_status_label.setStyleSheet("color: #1a7f37; font-size: 12px;")
            self.api_key_status_label.setText("已使用上方手动填写的 API Key（默认不落盘明文保存）")
            return

        if provider == MllmProvider.GEMINI:
            has_env = bool(os.environ.get("GEMINI_API_KEY", "").strip())
            if has_env:
                self.api_key_status_label.setStyleSheet("color: #1a7f37; font-size: 12px;")
                self.api_key_status_label.setText("已检测到环境变量 GEMINI_API_KEY，上方输入框可保持留空")
            else:
                self.api_key_status_label.setStyleSheet("color: #9a6700; font-size: 12px;")
                self.api_key_status_label.setText("未检测到环境变量 GEMINI_API_KEY，请在上方填写或配置环境变量")
            return

        has_env = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        base_url = self.base_url_edit.text().strip().lower()
        is_local = any(host in base_url for host in ("localhost", "127.0.0.1"))
        if has_env:
            self.api_key_status_label.setStyleSheet("color: #1a7f37; font-size: 12px;")
            self.api_key_status_label.setText("已检测到环境变量 OPENAI_API_KEY，上方输入框可保持留空")
        elif is_local:
            self.api_key_status_label.setStyleSheet("color: #0969da; font-size: 12px;")
            self.api_key_status_label.setText("当前为本地服务地址（如 Ollama / vLLM），API Key 可留空直接连接")
        else:
            self.api_key_status_label.setStyleSheet("color: #9a6700; font-size: 12px;")
            self.api_key_status_label.setText("未检测到环境变量 OPENAI_API_KEY，请在上方填写或配置环境变量")

    def _sync_strategy_states(self, *_args: object) -> None:
        """同步边框修正模式、漏标开关、局部裁剪与高置信度阈值之间的防呆联动状态。"""
        refine_mode: BoxRefineMode = self.refine_mode_combo.currentData() or BoxRefineMode.OFF
        is_sam = refine_mode == BoxRefineMode.SAM_LOCAL
        is_mllm_crop = refine_mode == BoxRefineMode.MLLM_CROP

        self._pol_form.setRowVisible(self.sam_row_widget, is_sam)
        if is_sam and self.sam_refine_all_chk.isChecked():
            self.refine_hint_label.setText(
                "已开启全量收紧：本地 SAM 对所有保留/改类框统一贴边收框（零额外 API 费），超门禁自动回退原框转待复核。"
            )
        else:
            self.refine_hint_label.setText(BOX_REFINE_MODE_HINTS.get(refine_mode, ""))

        # MLLM 局部放大图精修必须依赖局部裁剪图坐标系
        if is_mllm_crop:
            prev_restore = self._restoring_state
            self._restoring_state = True
            try:
                self.crops_chk.setChecked(True)
            finally:
                self._restoring_state = prev_restore
            self.crops_chk.setEnabled(False)
            self.crops_chk.setToolTip("「MLLM 局部放大图精修」模式必须附带局部裁剪图以提供高精度局部坐标系")
        else:
            self.crops_chk.setEnabled(True)
            self.crops_chk.setToolTip("将低置信度候选框外扩 20% 裁剪放大后一并送审，显著提升小目标纠错准确率")

        check_missing = self.check_missing_chk.isChecked()
        self.add_missing_chk.setEnabled(check_missing)

        # 精准反映 high_conf_skip 在当前策略组合下的真实生效角色，并用方向符号（≥ / <）消除理解歧义
        needs_mllm_for_refine = refine_mode in {BoxRefineMode.FLAG_LOOSE, BoxRefineMode.MLLM_CROP} or (
            is_sam and not self.sam_refine_all_chk.isChecked()
        )
        can_skip_mllm = (not check_missing) and (not needs_mllm_for_refine)
        crops_active = self.crops_chk.isChecked() and (not is_mllm_crop)

        if can_skip_mllm:
            self.high_conf_label.setText("免审阈值 ≥")
            self.high_conf_spin.setEnabled(True)
            self.high_conf_spin.setToolTip(
                "当前未开启漏标检查：整张图所有框置信度均 ≥ 此值时直接通过并跳过 MLLM 调用，省钱提速"
            )
        elif crops_active:
            self.high_conf_label.setText("优先裁剪 <")
            self.high_conf_spin.setEnabled(True)
            self.high_conf_spin.setToolTip(
                "当前已开启漏标或收框评估（不会跳过整图 MLLM）；单图目标超过 8 个时，优先为置信度 < 此值的框生成局部放大图"
            )
        else:
            self.high_conf_label.setText("免审阈值(停用):")
            self.high_conf_spin.setEnabled(False)
            self.high_conf_spin.setToolTip(
                "当前开启了漏标检查/局部精修且无需按置信度筛裁剪，此阈值暂不生效；取消漏标检查后恢复高置信度免审"
            )

    def _resolve_sam_model_name(self) -> str:
        """从 SAM 模型组合框提取权重文件名或自定义 `.pt` 路径。"""
        text = self.sam_model_combo.currentText().strip()
        idx = self.sam_model_combo.currentIndex()
        if idx >= 0 and text == self.sam_model_combo.itemText(idx):
            data = self.sam_model_combo.itemData(idx)
            if data:
                return str(data)
        return text or "sam2.1_t.pt"

    def _on_pick_sam_pt(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择本地 SAM 权重文件", str(self.workspace_root), "PyTorch 权重 (*.pt);;所有文件 (*.*)"
        )
        if chosen:
            self.sam_model_combo.setCurrentText(chosen)

    def _on_import_classes_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "导入 classes.txt 类别清单", str(self.workspace_root), "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if not chosen:
            return
        try:
            classes = [
                line.strip()
                for line in Path(chosen).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            QMessageBox.warning(self, "读取失败", f"无法读取类别文件：{exc}")
            return
        if not classes:
            QMessageBox.warning(self, "提示", "所选 classes.txt 中没有有效类别名称。")
            return
        self.classes_edit.setText(", ".join(classes))

    def _build_current_config(self) -> MllmReviewConfig:
        """按当前表单控件组装 `MllmReviewConfig`（不校验目录磁盘存在性，供测试连接与审核启动复用）。"""
        raw_classes = self.classes_edit.text().replace("，", ",")
        allowed_classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        ann_text = self.ann_picker.text()
        weights_text = self.weights_picker.text()
        refine_mode: BoxRefineMode = self.refine_mode_combo.currentData() or BoxRefineMode.OFF
        return MllmReviewConfig(
            images_dir=Path(self.images_picker.text() or "."),
            output_dir=Path(self.output_picker.text() or (self.workspace_root / "data" / "reviewed_frames")),
            source_mode=self.source_mode_combo.currentData(),
            annotations_dir=Path(ann_text) if ann_text else None,
            weights_path=Path(weights_text) if weights_text else None,
            yolo_conf=self.yolo_conf_spin.value(),
            yolo_iou=self.yolo_iou_spin.value(),
            yolo_imgsz=self.yolo_imgsz_spin.value(),
            yolo_device=self.yolo_device_ctrl.device(),
            provider=self.provider_combo.currentData(),
            model_name=self.model_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            base_url=self.base_url_edit.text().strip(),
            allowed_classes=allowed_classes,
            custom_guideline=self.guideline_edit.text().strip(),
            high_conf_skip=self.high_conf_spin.value(),
            check_missing=self.check_missing_chk.isChecked(),
            add_missing_boxes=self.check_missing_chk.isChecked() and self.add_missing_chk.isChecked(),
            include_crops=self.crops_chk.isChecked(),
            max_workers=self.workers_spin.value(),
            max_images=self.max_images_spin.value(),
            triage_subfolders=self.triage_dirs_chk.isChecked(),
            recursive=self.recursive_chk.isChecked(),
            box_refine_mode=refine_mode,
            sam_model_name=self._resolve_sam_model_name(),
            sam_refine_all_kept=self.sam_refine_all_chk.isChecked(),
        )

    def _on_test_connection(self) -> None:
        """在后台线程执行 MLLM 轻量连通性测试并在状态栏与日志反馈诊断结果。"""
        if self.conn_test_worker is not None and self.conn_test_worker.isRunning():
            return
        cfg = self._build_current_config()
        self.test_conn_btn.setEnabled(False)
        self.test_conn_btn.setText("测试中...")
        self.api_key_status_label.setStyleSheet("color: #0969da; font-size: 12px;")
        self.api_key_status_label.setText("正在向 MLLM 接口发送连通性探针...")

        self.conn_test_worker = MllmConnectionTestWorker(cfg)
        self.conn_test_worker.finished_test.connect(self._on_test_connection_done)
        self.conn_test_worker.start()

    def _on_test_connection_done(self, ok: bool, message: str) -> None:
        self.test_conn_btn.setEnabled(True)
        self.test_conn_btn.setText("测试连接")
        color = "#1a7f37" if ok else "#cf222e"
        self.api_key_status_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 500;")
        self.api_key_status_label.setText(message)
        self.log_box.appendPlainText(f"[连接测试] {message}")
        if not ok:
            self.result_tabs.setCurrentWidget(self.log_box)
            QMessageBox.warning(self, "MLLM 连接测试失败", message)

    def save_settings(self, settings: QSettings, *, force: bool = False) -> None:
        """将当前 MLLM 纠错策略配置序列化写入 `settings`（默认不保存明文 API Key）。"""
        if not force and not self._settings_dirty:
            return
        provider: MllmProvider = self.provider_combo.currentData() or MllmProvider.GEMINI
        source_mode: ReviewSourceMode = self.source_mode_combo.currentData() or ReviewSourceMode.YOLO_PREDICT
        refine_mode: BoxRefineMode = self.refine_mode_combo.currentData() or BoxRefineMode.OFF
        state: dict[str, object] = {
            "version": 1,
            "source_mode": source_mode.value,
            "yolo_conf": self.yolo_conf_spin.value(),
            "yolo_iou": self.yolo_iou_spin.value(),
            "yolo_imgsz": self.yolo_imgsz_spin.value(),
            "provider": provider.value,
            "base_url": self.base_url_edit.text().strip(),
            "model_name": self.model_edit.text().strip(),
            "classes": self.classes_edit.text().strip(),
            "guideline": self.guideline_edit.text().strip(),
            "high_conf_skip": self.high_conf_spin.value(),
            "max_workers": self.workers_spin.value(),
            "max_images": self.max_images_spin.value(),
            "check_missing": self.check_missing_chk.isChecked(),
            "add_missing": self.add_missing_chk.isChecked(),
            "include_crops": self.crops_chk.isChecked(),
            "triage_dirs": self.triage_dirs_chk.isChecked(),
            # 旧版键名 recursive 保存的多为旧默认值 False，改用新键名使“默认递归”对已有用户生效
            "recursive_subdirs": self.recursive_chk.isChecked(),
            "box_refine_mode": refine_mode.value,
            "sam_model_name": self._resolve_sam_model_name(),
            "sam_refine_all": self.sam_refine_all_chk.isChecked(),
        }
        settings.setValue(_REVIEW_SETTINGS_KEY, json.dumps(state, ensure_ascii=False))

    def restore_settings(self, settings: QSettings) -> None:
        """从 `settings` 恢复上次保存的 MLLM 纠错策略配置；数据缺失或损坏时静默保持默认。"""
        raw = settings.value(_REVIEW_SETTINGS_KEY, "")
        if not isinstance(raw, str) or not raw.strip():
            return
        try:
            state = json.loads(raw)
        except Exception:
            return
        if not isinstance(state, dict):
            return

        self._restoring_state = True
        try:
            src_val = str(state.get("source_mode") or "")
            for i in range(self.source_mode_combo.count()):
                item = self.source_mode_combo.itemData(i)
                if isinstance(item, ReviewSourceMode) and item.value == src_val:
                    self.source_mode_combo.setCurrentIndex(i)
                    break

            if "yolo_conf" in state:
                self.yolo_conf_spin.setValue(float(state["yolo_conf"]))
            if "yolo_iou" in state:
                self.yolo_iou_spin.setValue(float(state["yolo_iou"]))
            if "yolo_imgsz" in state:
                self.yolo_imgsz_spin.setValue(int(state["yolo_imgsz"]))

            prov_val = str(state.get("provider") or "")
            for i in range(self.provider_combo.count()):
                item = self.provider_combo.itemData(i)
                if isinstance(item, MllmProvider) and item.value == prov_val:
                    self.provider_combo.setCurrentIndex(i)
                    break

            if state.get("base_url"):
                self.base_url_edit.setText(str(state["base_url"]))
            if state.get("model_name"):
                self.model_edit.setText(str(state["model_name"]))
            if "classes" in state:
                self.classes_edit.setText(str(state.get("classes") or ""))
            if "guideline" in state:
                self.guideline_edit.setText(str(state.get("guideline") or ""))
            if "high_conf_skip" in state:
                self.high_conf_spin.setValue(float(state["high_conf_skip"]))
            if "max_workers" in state:
                self.workers_spin.setValue(int(state["max_workers"]))
            if "max_images" in state:
                self.max_images_spin.setValue(max(0, int(state["max_images"])))
            if "check_missing" in state:
                self.check_missing_chk.setChecked(bool(state["check_missing"]))
            if "add_missing" in state:
                self.add_missing_chk.setChecked(bool(state["add_missing"]))
            if "include_crops" in state:
                self.crops_chk.setChecked(bool(state["include_crops"]))
            if "triage_dirs" in state:
                self.triage_dirs_chk.setChecked(bool(state["triage_dirs"]))
            if "recursive_subdirs" in state:
                self.recursive_chk.setChecked(bool(state["recursive_subdirs"]))

            refine_val = str(state.get("box_refine_mode") or "")
            for i in range(self.refine_mode_combo.count()):
                item = self.refine_mode_combo.itemData(i)
                if isinstance(item, BoxRefineMode) and item.value == refine_val:
                    self.refine_mode_combo.setCurrentIndex(i)
                    break

            sam_model = str(state.get("sam_model_name") or "").strip()
            if sam_model:
                idx = self.sam_model_combo.findData(sam_model)
                if idx >= 0:
                    self.sam_model_combo.setCurrentIndex(idx)
                else:
                    self.sam_model_combo.setCurrentText(sam_model)
            if "sam_refine_all" in state:
                self.sam_refine_all_chk.setChecked(bool(state["sam_refine_all"]))
        except Exception:
            pass
        finally:
            self._restoring_state = False

        self._on_source_mode_changed()
        self._sync_strategy_states()
        self._refresh_api_key_status()
        self._refresh_images_count()

    def _on_start(self) -> None:
        """校验图片目录、权重与参数，重置表格与记录缓存后启动 `MllmReviewWorker`。"""
        images_text = self.images_picker.text()
        out_text = self.output_picker.text()
        if not images_text or not Path(images_text).is_dir():
            QMessageBox.warning(self, "提示", "请选择有效的待纠错图片目录。")
            return
        if not out_text:
            QMessageBox.warning(self, "提示", "请选择清洗输出目录。")
            return

        cfg = self._build_current_config()
        uses_local_gpu_model = (
            cfg.source_mode == ReviewSourceMode.YOLO_PREDICT
            or cfg.box_refine_mode == BoxRefineMode.SAM_LOCAL
        )
        if uses_local_gpu_model and self.device_info.probing:
            self.apply_device_info(detect_compute_device())
            cfg.yolo_device = self.yolo_device_ctrl.device()

        if cfg.source_mode == ReviewSourceMode.YOLO_PREDICT and (
            not cfg.weights_path or not cfg.weights_path.is_file()
        ):
            QMessageBox.warning(self, "提示", "请选择有效的基准模型 .pt 权重文件。")
            return

        if uses_local_gpu_model and self.gpu_guard is not None:
            task_label = (
                "MLLM 基准模型推理 / SAM 收框审核"
                if cfg.box_refine_mode == BoxRefineMode.SAM_LOCAL
                else "MLLM 基准模型推理审核"
            )
            if not self.gpu_guard.try_acquire(self, task_label):
                return

        self._settings_dirty = True
        self._last_output_dir = cfg.output_dir
        self._records.clear()
        self._preview_cache.clear()
        self._selected_row = -1
        self.preview_label.clear_preview()
        self.detail_label.setText("未选中图片：单击上方结果行可查看 MLLM 逐框判定理由与收框明细。")
        self.detail_label.setToolTip("")
        self.open_selected_btn.setEnabled(False)
        self.result_table.setRowCount(0)
        self.result_tabs.setCurrentWidget(self.result_table)
        self._refresh_filter_counts()
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.send_ds_btn.setEnabled(False)
        self.summary_label.setText("正在运行候选框提取与 MLLM 纠错清洗...")

        self.worker = MllmReviewWorker(cfg)
        if uses_local_gpu_model and self.gpu_guard is not None:
            self.gpu_guard.bind(self.worker, "MLLM 基准模型推理审核")
        self.worker.progress.connect(self._on_progress)
        self.worker.log_line.connect(self.log_box.appendPlainText)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stop(self) -> None:
        """向正在运行的审核线程发送取消标志并禁用停止按钮防止重复点击。"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.stop_btn.setEnabled(False)
            self.log_box.appendPlainText("正在停止 MLLM 纠错任务...")

    def _record_matches_filter(self, rec: ImageReviewRecord, filter_key: str) -> bool:
        if filter_key == "all":
            return True
        if filter_key == "error":
            return bool(rec.error)
        return rec.triage_bucket == filter_key

    def set_result_filter(self, filter_key: str) -> None:
        """按分流状态筛选右侧审核结果表（保留行号与 `_records` 1:1 对应关系）。"""
        self._active_filter = filter_key
        for key, btn in self.filter_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(key == filter_key)
            btn.blockSignals(False)
        for row, rec in enumerate(self._records):
            self.result_table.setRowHidden(row, not self._record_matches_filter(rec, filter_key))

    def _refresh_filter_counts(self) -> None:
        """按当前 `_records` 更新顶部 5 个分流筛选按钮的实时计数。"""
        total = len(self._records)
        passed = sum(1 for r in self._records if r.triage_bucket == BUCKET_AUTO_PASSED)
        fixed = sum(1 for r in self._records if r.triage_bucket == BUCKET_LLM_FIXED)
        review = sum(1 for r in self._records if r.triage_bucket == BUCKET_NEED_REVIEW)
        errors = sum(1 for r in self._records if r.error)
        labels = {
            "all": f"全部 ({total})",
            BUCKET_AUTO_PASSED: f"自动通过 ({passed})",
            BUCKET_LLM_FIXED: f"已删改/收框 ({fixed})",
            BUCKET_NEED_REVIEW: f"待复核 ({review})",
            "error": f"出错 ({errors})",
        }
        for key, text in labels.items():
            if key in self.filter_buttons:
                self.filter_buttons[key].setText(text)

    def _append_record_row(self, rec: ImageReviewRecord) -> None:
        """向结果表追加一条语义配色记录，并按当前筛选态决定可见性。"""
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)

        bucket_meta = {
            BUCKET_AUTO_PASSED: ("自动通过", "#1f7a1f"),
            BUCKET_LLM_FIXED: ("已删改/收框", "#0969da"),
            BUCKET_NEED_REVIEW: ("待人工复核", "#b25900"),
        }
        status_label, status_color = bucket_meta.get(rec.triage_bucket, (rec.triage_bucket, "#333333"))
        if rec.error:
            status_label = "审核出错"
            status_color = "#cf222e"

        note_parts: list[str] = []
        if rec.error:
            note_parts.append(rec.error)
        if rec.loose_flagged_count:
            note_parts.append(f"{rec.loose_flagged_count} 框过松/超门禁")
        if rec.missing_note and rec.missing_note not in note_parts:
            note_parts.append(rec.missing_note)
        if not note_parts:
            note_parts.append(format_record_detail_summary(rec))
        note = "；".join(note_parts)

        img_item = QTableWidgetItem(rec.image_path.name)
        img_item.setData(Qt.ItemDataRole.UserRole, row)
        status_item = QTableWidgetItem(status_label)
        status_item.setForeground(QColor(status_color))
        bold_font = QFont(status_item.font())
        bold_font.setBold(True)
        status_item.setFont(bold_font)
        note_item = QTableWidgetItem(note)
        note_item.setToolTip(note)

        self.result_table.setItem(row, 0, img_item)
        self.result_table.setItem(row, 1, status_item)
        self.result_table.setItem(row, 2, QTableWidgetItem(str(rec.kept_count)))
        self.result_table.setItem(row, 3, QTableWidgetItem(str(rec.relabeled_count)))
        self.result_table.setItem(row, 4, QTableWidgetItem(str(rec.deleted_count)))
        self.result_table.setItem(row, 5, QTableWidgetItem(str(rec.refined_count)))
        self.result_table.setItem(row, 6, QTableWidgetItem(str(rec.added_count)))
        self.result_table.setItem(row, 7, note_item)
        self.result_table.setRowHidden(row, not self._record_matches_filter(rec, self._active_filter))

    def _on_progress(self, cur: int, total: int, rec: ImageReviewRecord) -> None:
        """追加单张审核记录到表格并刷新进度条与筛选计数。"""
        self.progress_bar.setValue(int(min(100, (cur / max(1, total)) * 100)))
        self._records.append(rec)
        self._append_record_row(rec)
        self._refresh_filter_counts()

    def load_history_report(self, report_or_dir: Path) -> MllmReviewResult:
        """加载历史清洗报告 `review_report.json` 并还原右侧表格、计数、对比预览与下游联动路径。"""
        res = load_review_report(report_or_dir)
        self._records = list(res.records)
        self._preview_cache.clear()
        self._selected_row = -1
        self.preview_label.clear_preview()
        self.detail_label.setText("未选中图片：单击上方结果行可查看 MLLM 逐框判定理由与收框明细。")
        self.detail_label.setToolTip("")
        self.open_selected_btn.setEnabled(False)
        self.result_table.setRowCount(0)
        for rec in self._records:
            self._append_record_row(rec)
        self._refresh_filter_counts()
        self.set_result_filter("all")

        self._last_output_dir = res.output_dir
        self.output_picker.setText(str(res.output_dir))
        self._last_classes_txt = str(res.classes_txt_path) if res.classes_txt_path else ""
        self.send_ds_btn.setEnabled(res.total_images > 0)
        self.progress_bar.setValue(100 if res.total_images > 0 else 0)
        refined_total = sum(r.refined_count for r in res.records)
        self.summary_label.setText(
            f"已加载历史报告（共 {res.total_images} 张）：自动通过 {res.auto_passed_count} | "
            f"已删改/收框 {res.llm_fixed_count}（收紧 {refined_total} 框）| "
            f"待人工复核 {res.need_review_count} | 出错 {res.error_count}"
        )
        self.log_box.appendPlainText(f"[历史报告] 已加载: {res.report_path}")
        return res

    def _on_load_report_dialog(self) -> None:
        start_dir = self.output_picker.text() or str(self.workspace_root)
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "选择历史 MLLM 审核报告 (review_report.json)",
            start_dir,
            "审核报告 (review_report.json);;JSON 文件 (*.json);;所有文件 (*.*)",
        )
        if not chosen:
            return
        try:
            self.load_history_report(Path(chosen))
        except (FileNotFoundError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "加载失败", str(exc))

    def _get_comparison_jpeg(self, row: int, rec: ImageReviewRecord) -> bytes:
        """获取单行审核对比预览 JPEG 字节流；命中 LRU 缓存时免除重复磁盘解码与绘图。"""
        cached = self._preview_cache.get(row)
        if cached is not None:
            self._preview_cache.move_to_end(row)
            return cached
        jpeg_bytes = render_review_comparison(rec, max_width=1100)
        self._preview_cache[row] = jpeg_bytes
        while len(self._preview_cache) > _PREVIEW_CACHE_MAX:
            self._preview_cache.popitem(last=False)
        return jpeg_bytes

    def _on_row_selected(self, row: int, *_args: object) -> None:
        if row < 0 or row >= len(self._records):
            self._selected_row = -1
            self.open_selected_btn.setEnabled(False)
            return
        self._selected_row = row
        rec = self._records[row]
        detail_text = format_record_detail_summary(rec)
        full_detail = f"【{rec.image_path.name}】{detail_text}"
        self.detail_label.setText(full_detail)
        self.detail_label.setToolTip(full_detail)
        self.open_selected_btn.setEnabled(True)
        try:
            jpeg_bytes = self._get_comparison_jpeg(row, rec)
            pix = QPixmap()
            pix.loadFromData(jpeg_bytes, "JPG")
            if not pix.isNull():
                self.preview_label.set_preview_pixmap(pix)
        except Exception as exc:
            self.preview_label.clear_preview(f"预览渲染失败: {exc}")

    def _on_open_selected_in_xal(self) -> None:
        """点击预览区右下角按钮时，直接将当前选中行图片送入内嵌标注工作台精修。"""
        if self._selected_row < 0 or self._selected_row >= len(self._records):
            return
        self._on_row_double_clicked(self._selected_row, 0)

    def _on_finished(self, res: MllmReviewResult) -> None:
        """恢复界面状态、记录产物路径、更新汇总栏并在存在结果时允许发送至数据集页。"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._last_output_dir = res.output_dir
        self._last_classes_txt = str(res.classes_txt_path) if res.classes_txt_path else ""
        self.send_ds_btn.setEnabled(res.total_images > 0)

        refined_total = sum(r.refined_count for r in res.records)
        state = "已停止" if res.cancelled else "清洗完成"
        self.summary_label.setText(
            f"{state}（共 {res.total_images} 张）：自动通过 {res.auto_passed_count} | "
            f"已删改/收框 {res.llm_fixed_count}（收紧 {refined_total} 框）| "
            f"待人工复核 {res.need_review_count} | 出错 {res.error_count}"
        )
        self._refresh_filter_counts()
        if res.report_path:
            self.log_box.appendPlainText(f"\n审核报告已保存: {res.report_path}")

    def _on_failed(self, err: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.summary_label.setText(f"执行失败: {err}")
        # 日志与结果列表共用页签区域，失败时自动切到“运行日志”便于直接查看出错上下文
        self.result_tabs.setCurrentWidget(self.log_box)
        QMessageBox.critical(self, "MLLM 纠错失败", err)

    def _on_row_double_clicked(self, row: int, _col: int) -> None:
        """双击结果行时跳转到内嵌标注工作台并定位到对应输出图片。"""
        if row < 0 or row >= len(self._records):
            return
        rec = self._records[row]
        target_img = (
            rec.output_image_path
            if (rec.output_image_path and rec.output_image_path.is_file())
            else rec.image_path
        )
        if not target_img.is_file():
            QMessageBox.warning(self, "提示", f"图片文件不存在：\n{target_img}")
            return
        out_dir = self.output_picker.path()
        cls_file = out_dir / "classes.txt"
        cls_str = str(cls_file) if cls_file.is_file() else self._last_classes_txt
        self.open_in_workbench.emit(str(target_img.parent), cls_str, str(target_img))

    def _on_open_need_review_in_xal(self) -> None:
        out_dir = self.output_picker.path()
        need_dir = out_dir / "need_review"
        target_dir = need_dir if need_dir.is_dir() else out_dir
        if not target_dir.is_dir():
            QMessageBox.warning(self, "提示", f"目录不存在，请先运行清洗或检查路径:\n{target_dir}")
            return
        cls_file = out_dir / "classes.txt"
        cls_str = str(cls_file) if cls_file.is_file() else self._last_classes_txt
        self.open_in_workbench.emit(str(target_dir), cls_str, "")

    def _on_send_to_dataset(self) -> None:
        out_dir = self.output_picker.path()
        if not out_dir.is_dir():
            QMessageBox.warning(self, "提示", f"清洗输出目录不存在:\n{out_dir}")
            return
        cls_file = out_dir / "classes.txt"
        cls_str = str(cls_file) if cls_file.is_file() else self._last_classes_txt
        self.send_cleaned_dataset.emit(str(out_dir), cls_str, self.triage_dirs_chk.isChecked())
