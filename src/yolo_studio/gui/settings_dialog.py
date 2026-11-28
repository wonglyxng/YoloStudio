"""桌面端全局「设置」分栏对话框（AI 大模型 / 本地视觉模型 / 通用设置）。

职责：
    提供主窗口右上角「⚙ 设置」（及 `Ctrl+,`）弹出的集中配置界面，采用左侧分组导航 + 右侧堆栈表单：
    - **AI 大模型**：配置模型接口、内置预设、Base URL、模型名称、思考/推理级别防呆联动、
      API Key 与环境变量检测状态以及一键连通性测试；
    - **本地视觉模型**：配置全局共享的 SAM 2.1 默认权重、SAM 3 权重路径与默认 GPU 加速偏好；
    - **通用设置**：配置标注工作台默认行为（自动保存、显示标签、保持视野、十字准星颜色/线型/描边）、记住上次目录与恢复默认。

运行环境与协作边界：
    运行于 PyQt6 主线程，编辑期间仅操作临时控件副本；点击「应用」或「确定」时才原子写回
    `AppSettingsStore`（并由调用方或 `applied` 信号持久化至 `QSettings`），点击「取消」不修改任何全局状态。
    连通性测试在后台 `MllmConnectionTestWorker` 线程中运行，关闭对话框前自动安全等待其收尾。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QCloseEvent, QColor, QIcon, QPainter, QPen, QPixmap, QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.mllm_reviewer import (
    GEMINI_MODELS,
    OPENAI_MODELS,
    OPENAI_PRESETS,
    REASONING_EFFORT_HINTS,
    REASONING_EFFORT_ITEMS,
    SAM_REFINE_MODELS,
    MllmProvider,
    ReasoningEffort,
    resolve_effective_reasoning_effort,
    resolve_sam3_weights,
    supported_reasoning_efforts,
)
from yolo_studio.core.trainer import DeviceInfo
from yolo_studio.gui.annotation_canvas import CROSSHAIR_COLOR_PRESETS, normalize_crosshair_color_key
from yolo_studio.gui.app_settings import (
    AppPreferences,
    AppSettingsStore,
    build_llm_probe_config,
    describe_api_key_status,
)
from yolo_studio.gui.common import (
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    install_combobox_wheel_guard,
    open_local_path,
)
from yolo_studio.gui.workers import MllmConnectionTestWorker

CUSTOM_PRESET_KEY = "__custom__"
CUSTOM_PRESET_LABEL = "自定义（当前手动指定）"

PAGE_GENERAL = "general"
PAGE_AI = "ai"
PAGE_LOCAL_MODELS = "local_models"

_NAV_PAGES: tuple[tuple[str, str], ...] = (
    (PAGE_AI, "AI 大模型"),
    (PAGE_LOCAL_MODELS, "本地视觉模型"),
    (PAGE_GENERAL, "通用设置"),
)


class SettingsDialog(QDialog):
    """YoloStudio 全局设置分栏对话框（AI 大模型 / 本地视觉模型 / 通用设置）。

    内部状态：
        持有共享 `settings_store` 引用、`workspace_root`、`device_info`、连通性测试线程
        `conn_test_worker` 及表单预设重入标志 `_applying_preset` / `_loading_from_store`。
        确认或点击「应用」后发射 `settings_applied` 信号，便于主窗口即时持久化 `QSettings`。
    """

    settings_applied = Signal()

    def __init__(
        self,
        settings_store: AppSettingsStore,
        workspace_root: Path,
        *,
        device_info: DeviceInfo | None = None,
        initial_page: str = PAGE_AI,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        install_combobox_wheel_guard()
        self.settings_store = settings_store
        self.workspace_root = workspace_root
        self.device_info = device_info
        self.conn_test_worker: MllmConnectionTestWorker | None = None
        self._applying_preset = False
        self._loading_from_store = False
        self._effort_fallback_notice = ""

        self.setWindowTitle("设置 — YoloStudio")
        self.setMinimumSize(760, 520)
        self.resize(800, 560)

        self._init_ui()
        self._load_from_preferences(self.settings_store.preferences)
        self.select_page(initial_page)

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 10)
        root.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(12)

        # 左侧分组导航栏
        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(148)
        self.nav_list.setSpacing(2)
        self.nav_list.setStyleSheet(
            "QListWidget { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px 4px; }"
            "QListWidget::item { padding: 8px 10px; border-radius: 5px; color: #24292f; }"
            "QListWidget::item:selected { background: #ddf4ff; color: #0550ae; font-weight: 600; }"
            "QListWidget::item:hover:!selected { background: #eaeef2; }"
        )
        for key, label in _NAV_PAGES:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.nav_list.addItem(item)
        body.addWidget(self.nav_list)

        # 右侧配置页面栈
        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self._build_ai_page())
        self.page_stack.addWidget(self._build_local_models_page())
        self.page_stack.addWidget(self._build_general_page())
        body.addWidget(self.page_stack, 1)
        root.addLayout(body, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #d0d7de;")
        root.addWidget(sep)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.reset_defaults_btn = QPushButton("恢复默认设置")
        self.reset_defaults_btn.setToolTip("将 AI、本地模型与通用偏好恢复为默认值（不影响任何图片与标注文件）")
        self.cancel_btn = QPushButton("取消")
        self.apply_btn = QPushButton("应用")
        self.ok_btn = QPushButton("确定")
        self.ok_btn.setObjectName("primaryBtn")
        self.ok_btn.setDefault(True)
        btn_row.addWidget(self.reset_defaults_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.ok_btn)
        root.addLayout(btn_row)

        # 信号绑定
        self.nav_list.currentRowChanged.connect(self.page_stack.setCurrentIndex)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.base_url_edit.textChanged.connect(self._sync_preset_combo_from_fields)
        self.base_url_edit.textChanged.connect(self._refresh_api_key_status)
        self.model_edit.textChanged.connect(self._sync_preset_combo_from_fields)
        self.model_edit.textChanged.connect(self._sync_reasoning_effort_guard)
        self.reasoning_effort_combo.currentIndexChanged.connect(
            lambda *_args: self._sync_reasoning_effort_guard(clear_notice=True)
        )
        self.api_key_edit.textChanged.connect(self._refresh_api_key_status)
        self.test_conn_btn.clicked.connect(self._on_test_connection)

        self.pick_sam_btn.clicked.connect(self._on_pick_sam_pt)
        self.sam3_weights_picker.line_edit.textChanged.connect(self._refresh_sam3_status)
        self.open_weights_dir_btn.clicked.connect(self._on_open_weights_dir)
        self.open_workspace_btn.clicked.connect(lambda: open_local_path(self.workspace_root))
        self.general_open_weights_btn.clicked.connect(self._on_open_weights_dir)

        self.reset_defaults_btn.clicked.connect(self._on_reset_defaults)
        self.cancel_btn.clicked.connect(self.reject)
        self.apply_btn.clicked.connect(self._on_apply)
        self.ok_btn.clicked.connect(self._on_ok)

    @staticmethod
    def _wrap_scroll(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_ai_page(self) -> QWidget:
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)

        group = QGroupBox("AI 大模型全局配置")
        form = QFormLayout(group)
        configure_form_layout(form, vertical_spacing=9, margins=(12, 12, 12, 12))
        self._ai_form = form

        self.provider_combo = QComboBox()
        self.provider_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.provider_combo.setMinimumContentsLength(12)
        self.provider_combo.addItem("Google Gemini API (google-genai)", MllmProvider.GEMINI)
        self.provider_combo.addItem("OpenAI 官方 API (api.openai.com)", MllmProvider.OPENAI)
        self.provider_combo.addItem(
            "OpenAI 兼容接口 (Qwen-VL / 硅基流动 / Ollama / vLLM)", MllmProvider.OPENAI_COMPAT
        )
        form.addRow("模型接口:", self.provider_combo)

        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(12)
        form.addRow("常用预设:", self.preset_combo)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("OpenAI 兼容接口 Base URL（Gemini / OpenAI 官方模式无需填写）")
        form.addRow("Base URL:", self.base_url_edit)

        self.model_edit = QLineEdit(GEMINI_MODELS[0][0])
        form.addRow("模型名称:", self.model_edit)

        effort_box = QVBoxLayout()
        effort_box.setContentsMargins(0, 0, 0, 0)
        effort_box.setSpacing(3)
        self.reasoning_effort_combo = QComboBox()
        self.reasoning_effort_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.reasoning_effort_combo.setMinimumContentsLength(12)
        for effort, effort_label in REASONING_EFFORT_ITEMS:
            self.reasoning_effort_combo.addItem(effort_label, effort)
        self.reasoning_effort_hint_label = QLabel()
        self.reasoning_effort_hint_label.setWordWrap(True)
        self.reasoning_effort_hint_label.setStyleSheet("color: #57606a; font-size: 12px;")
        effort_box.addWidget(self.reasoning_effort_combo)
        effort_box.addWidget(self.reasoning_effort_hint_label)
        form.addRow("思考/推理级别:", effort_box)

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
        self.api_key_status_label.setWordWrap(True)
        self.api_key_status_label.setStyleSheet("color: #57606a; font-size: 12px;")
        key_box.addLayout(key_row)
        key_box.addWidget(self.api_key_status_label)
        form.addRow("API Key:", key_box)

        align_form_labels([form])
        lay.addWidget(group)

        scope_tip = QLabel(
            "说明：此处配置由「2. 标注工作台（单框 AI 精修 G）」与「3. 自动智能标注（批量标注/质检）」全局共享。"
            "为保障凭据安全，手动填写的 API Key 仅在当前进程运行期间有效，退出后不会明文落盘；推荐在系统环境变量中配置 GEMINI_API_KEY 或 OPENAI_API_KEY。"
        )
        scope_tip.setWordWrap(True)
        scope_tip.setStyleSheet("color: #57606a; font-size: 12px; padding: 0 4px;")
        lay.addWidget(scope_tip)
        lay.addStretch(1)
        return self._wrap_scroll(content)

    def _build_local_models_page(self) -> QWidget:
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)

        sam_group = QGroupBox("本地 SAM 视觉分割模型")
        s_form = QFormLayout(sam_group)
        configure_form_layout(s_form, vertical_spacing=9, margins=(12, 12, 12, 12))
        self._local_sam_form = s_form

        sam2_box = QVBoxLayout()
        sam2_box.setContentsMargins(0, 0, 0, 0)
        sam2_box.setSpacing(3)
        sam2_row = QHBoxLayout()
        sam2_row.setContentsMargins(0, 0, 0, 0)
        sam2_row.setSpacing(6)
        self.sam_model_combo = QComboBox()
        self.sam_model_combo.setEditable(True)
        self.sam_model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.sam_model_combo.setMinimumContentsLength(10)
        for sam_file, sam_desc in SAM_REFINE_MODELS:
            self.sam_model_combo.addItem(sam_desc, sam_file)
        self.pick_sam_btn = QPushButton("选 .pt")
        self.pick_sam_btn.setStyleSheet("padding: 4px 10px;")
        sam2_row.addWidget(self.sam_model_combo, 1)
        sam2_row.addWidget(self.pick_sam_btn)
        sam2_hint = QLabel(
            "同时用于「2. 标注工作台」的 SAM 点选标注 (Q) / SAM 收框，以及「3. 自动智能标注」的本地 SAM 像素级收紧。"
        )
        sam2_hint.setWordWrap(True)
        sam2_hint.setStyleSheet("color: #57606a; font-size: 12px;")
        sam2_box.addLayout(sam2_row)
        sam2_box.addWidget(sam2_hint)
        s_form.addRow("SAM 2.1 模型:", sam2_box)

        sam3_box = QVBoxLayout()
        sam3_box.setContentsMargins(0, 0, 0, 0)
        sam3_box.setSpacing(3)
        self.sam3_weights_picker = PathPickerRow(
            placeholder="可选：留空则自动识别 weights/sam3.pt 或 sam3.1_multiplex.pt",
            mode="file",
            file_filter="PyTorch 权重 (*.pt);;所有文件 (*.*)",
        )
        self.sam3_status_label = QLabel()
        self.sam3_status_label.setWordWrap(True)
        self.sam3_status_label.setStyleSheet("color: #57606a; font-size: 12px;")
        sam3_box.addWidget(self.sam3_weights_picker)
        sam3_box.addWidget(self.sam3_status_label)
        s_form.addRow("SAM 3 默认权重:", sam3_box)

        weights_row = QHBoxLayout()
        weights_row.setContentsMargins(0, 0, 0, 0)
        weights_row.setSpacing(6)
        self.open_weights_dir_btn = QPushButton("打开 weights 权重目录")
        self.open_weights_dir_btn.setStyleSheet("padding: 4px 10px;")
        weights_row.addWidget(self.open_weights_dir_btn)
        weights_row.addStretch(1)
        s_form.addRow("权重目录:", weights_row)
        lay.addWidget(sam_group)

        dev_group = QGroupBox("计算设备偏好")
        d_form = QFormLayout(dev_group)
        configure_form_layout(d_form, vertical_spacing=9, margins=(12, 12, 12, 12))
        self._local_dev_form = d_form

        self.hw_summary_label = QLabel(self._format_device_summary())
        self.hw_summary_label.setWordWrap(True)
        self.hw_summary_label.setStyleSheet("color: #24292f; font-size: 12px;")
        d_form.addRow("当前硬件:", self.hw_summary_label)

        self.default_prefer_gpu_chk = QCheckBox("优先启用 GPU 加速（检测到 NVIDIA 显卡时默认使用 GPU）")
        self.default_prefer_gpu_chk.setToolTip(
            "取消勾选后，工作台本地 SAM 点选/收框将优先使用 CPU 运行，适合显存紧张时临时避让"
        )
        d_form.addRow("设备策略:", self.default_prefer_gpu_chk)

        align_form_labels([s_form, d_form])
        lay.addWidget(dev_group)
        lay.addStretch(1)
        return self._wrap_scroll(content)

    def _build_general_page(self) -> QWidget:
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)

        wb_group = QGroupBox("标注工作台默认偏好")
        wb_form = QFormLayout(wb_group)
        configure_form_layout(wb_form, vertical_spacing=9, margins=(12, 12, 12, 12))
        self._gen_wb_form = wb_form

        self.wb_auto_save_chk = QCheckBox("默认开启「自动保存」（切图或关闭前自动保存 .json）")
        self.wb_show_labels_chk = QCheckBox("默认显示目标框类别与置信度角标 (L)")
        self.wb_keep_view_chk = QCheckBox("默认开启「保持视野」（切图保留缩放倍率与观察位置）")
        wb_form.addRow("自动保存:", self.wb_auto_save_chk)
        wb_form.addRow("画布显示:", self.wb_show_labels_chk)
        wb_form.addRow("连续帧视野:", self.wb_keep_view_chk)

        crosshair_box = QVBoxLayout()
        crosshair_box.setContentsMargins(0, 0, 0, 0)
        crosshair_box.setSpacing(3)
        self.wb_crosshair_color_combo = QComboBox()
        self.wb_crosshair_color_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.wb_crosshair_color_combo.setMinimumContentsLength(14)
        for color_key, color_name, color_rgb in CROSSHAIR_COLOR_PRESETS:
            self.wb_crosshair_color_combo.addItem(
                self._make_color_swatch_icon(color_rgb), color_name, color_key
            )
        crosshair_style_row = QHBoxLayout()
        crosshair_style_row.setContentsMargins(0, 0, 0, 0)
        crosshair_style_row.setSpacing(10)
        self.wb_crosshair_dashed_combo = QComboBox()
        self.wb_crosshair_dashed_combo.addItem("细虚线（默认，更轻）", True)
        self.wb_crosshair_dashed_combo.addItem("细实线（对齐更连贯）", False)
        self.wb_crosshair_halo_chk = QCheckBox("黑色描边（复杂/高亮背景更清晰）")
        crosshair_style_row.addWidget(self.wb_crosshair_dashed_combo)
        crosshair_style_row.addWidget(self.wb_crosshair_halo_chk)
        crosshair_style_row.addStretch(1)
        crosshair_hint = QLabel(
            "画框 (R)、SAM 点选 (Q) 与准星 (X) 共用此样式；线条在光标处留空，交点由鼠标指针标示。"
        )
        crosshair_hint.setWordWrap(True)
        crosshair_hint.setStyleSheet("color: #57606a; font-size: 12px;")
        crosshair_box.addWidget(self.wb_crosshair_color_combo)
        crosshair_box.addLayout(crosshair_style_row)
        crosshair_box.addWidget(crosshair_hint)
        wb_form.addRow("十字准星样式:", crosshair_box)
        lay.addWidget(wb_group)

        app_group = QGroupBox("工作区与启动行为")
        app_form = QFormLayout(app_group)
        configure_form_layout(app_form, vertical_spacing=9, margins=(12, 12, 12, 12))
        self._gen_app_form = app_form

        self.remember_last_dir_chk = QCheckBox("启动时自动记住并恢复上次打开的图片目录")
        app_form.addRow("历史目录:", self.remember_last_dir_chk)

        ws_row = QHBoxLayout()
        ws_row.setContentsMargins(0, 0, 0, 0)
        ws_row.setSpacing(6)
        self.workspace_path_edit = QLineEdit(str(self.workspace_root.resolve()))
        self.workspace_path_edit.setReadOnly(True)
        self.open_workspace_btn = QPushButton("打开目录")
        self.open_workspace_btn.setStyleSheet("padding: 4px 10px;")
        self.general_open_weights_btn = QPushButton("权重目录")
        self.general_open_weights_btn.setStyleSheet("padding: 4px 10px;")
        ws_row.addWidget(self.workspace_path_edit, 1)
        ws_row.addWidget(self.open_workspace_btn)
        ws_row.addWidget(self.general_open_weights_btn)
        app_form.addRow("当前工作区:", ws_row)

        align_form_labels([wb_form, app_form])
        lay.addWidget(app_group)
        lay.addStretch(1)
        return self._wrap_scroll(content)

    @staticmethod
    def _make_color_swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
        """生成带深色描边的 14×14 纯色方块图标，用于在下拉框中直观预览准星颜色。"""
        pix = QPixmap(14, 14)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor(36, 41, 47), 1.0))
            painter.setBrush(QColor(*rgb))
            painter.drawRoundedRect(1, 1, 12, 12, 2, 2)
        finally:
            painter.end()
        return QIcon(pix)

    def select_page(self, page_key: str) -> None:
        """按页面键名（`ai` / `local_models` / `general`）切换左侧导航与右侧堆栈页。"""
        target = (page_key or "").strip().lower()
        for idx in range(self.nav_list.count()):
            item = self.nav_list.item(idx)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) == target:
                self.nav_list.setCurrentRow(idx)
                return
        self.nav_list.setCurrentRow(0)

    def _format_device_summary(self) -> str:
        info = self.device_info
        if info is None:
            return "未传入硬件快照（运行模型时将自动检测）"
        if info.probing:
            return "正在后台检测计算设备..."
        if info.cuda_available:
            return f"已检测到 GPU：{info.device_name}（约 {info.vram_gb:.1f} GB 显存，推荐设备 {info.recommended_device}）"
        return "未检测到可用 NVIDIA CUDA 显卡，当前将使用 CPU 模式运行本地模型"

    def _load_from_preferences(self, prefs: AppPreferences) -> None:
        self._loading_from_store = True
        try:
            idx = self.provider_combo.findData(prefs.provider)
            if idx >= 0:
                self.provider_combo.setCurrentIndex(idx)
            self._on_provider_changed()

            if prefs.provider == MllmProvider.OPENAI_COMPAT:
                if prefs.base_url:
                    self.base_url_edit.setText(prefs.base_url)
                if prefs.model_name:
                    self.model_edit.setText(prefs.model_name)
            else:
                if prefs.model_name:
                    self.model_edit.setText(prefs.model_name)
            self._sync_preset_combo_from_fields()

            eff_idx = self.reasoning_effort_combo.findData(prefs.reasoning_effort)
            if eff_idx >= 0:
                self.reasoning_effort_combo.setCurrentIndex(eff_idx)
            self.api_key_edit.setText(prefs.api_key)

            sam_idx = self.sam_model_combo.findData(prefs.sam_model_name)
            if sam_idx >= 0:
                self.sam_model_combo.setCurrentIndex(sam_idx)
            else:
                self.sam_model_combo.setCurrentText(prefs.sam_model_name)
            self.sam3_weights_picker.setText(prefs.sam3_weights_path)
            self.default_prefer_gpu_chk.setChecked(prefs.default_prefer_gpu)

            self.wb_auto_save_chk.setChecked(prefs.workbench_auto_save)
            self.wb_show_labels_chk.setChecked(prefs.workbench_show_labels)
            self.wb_keep_view_chk.setChecked(prefs.workbench_keep_view)
            color_idx = self.wb_crosshair_color_combo.findData(
                normalize_crosshair_color_key(prefs.workbench_crosshair_color)
            )
            self.wb_crosshair_color_combo.setCurrentIndex(max(0, color_idx))
            dashed_idx = self.wb_crosshair_dashed_combo.findData(bool(prefs.workbench_crosshair_dashed))
            self.wb_crosshair_dashed_combo.setCurrentIndex(max(0, dashed_idx))
            self.wb_crosshair_halo_chk.setChecked(bool(prefs.workbench_crosshair_halo))
            self.remember_last_dir_chk.setChecked(prefs.remember_last_dir)
        finally:
            self._loading_from_store = False

        self._effort_fallback_notice = ""
        self._sync_reasoning_effort_guard()
        self._refresh_api_key_status()
        self._refresh_sam3_status()

    def _resolve_sam_model_name(self) -> str:
        text = self.sam_model_combo.currentText().strip()
        idx = self.sam_model_combo.currentIndex()
        if idx >= 0 and text == self.sam_model_combo.itemText(idx):
            data = self.sam_model_combo.itemData(idx)
            if data:
                return str(data)
        return text or "sam2.1_t.pt"

    def build_preferences(self) -> AppPreferences:
        """按对话框当前表单控件组装一份 `AppPreferences` 快照（尚未写回 `settings_store`）。"""
        provider: MllmProvider = self.provider_combo.currentData() or MllmProvider.GEMINI
        effort: ReasoningEffort = self.reasoning_effort_combo.currentData() or ReasoningEffort.AUTO
        return AppPreferences(
            provider=provider,
            base_url=self.base_url_edit.text().strip() if provider == MllmProvider.OPENAI_COMPAT else "",
            model_name=self.model_edit.text().strip(),
            custom_model=self.preset_combo.currentData() == CUSTOM_PRESET_KEY,
            reasoning_effort=effort,
            api_key=self.api_key_edit.text().strip(),
            sam_model_name=self._resolve_sam_model_name(),
            sam3_weights_path=self.sam3_weights_picker.text().strip(),
            workbench_auto_save=self.wb_auto_save_chk.isChecked(),
            workbench_show_labels=self.wb_show_labels_chk.isChecked(),
            workbench_keep_view=self.wb_keep_view_chk.isChecked(),
            workbench_crosshair_color=normalize_crosshair_color_key(
                self.wb_crosshair_color_combo.currentData()
            ),
            workbench_crosshair_dashed=bool(self.wb_crosshair_dashed_combo.currentData()),
            workbench_crosshair_halo=self.wb_crosshair_halo_chk.isChecked(),
            remember_last_dir=self.remember_last_dir_chk.isChecked(),
            default_prefer_gpu=self.default_prefer_gpu_chk.isChecked(),
        )

    def _on_provider_changed(self) -> None:
        provider: MllmProvider = self.provider_combo.currentData() or MllmProvider.GEMINI
        is_gemini = provider == MllmProvider.GEMINI
        is_openai_official = provider == MllmProvider.OPENAI
        is_compat = provider == MllmProvider.OPENAI_COMPAT

        self._applying_preset = True
        self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()
            if is_gemini:
                for model_id, label in GEMINI_MODELS:
                    self.preset_combo.addItem(label, (model_id, ""))
                self.base_url_edit.setEnabled(False)
                self.base_url_edit.clear()
                self.model_edit.setText(GEMINI_MODELS[0][0])
            elif is_openai_official:
                for model_id, label in OPENAI_MODELS:
                    self.preset_combo.addItem(label, (model_id, ""))
                self.base_url_edit.setEnabled(False)
                self.base_url_edit.clear()
                self.model_edit.setText(OPENAI_MODELS[0][0])
            else:
                for name, url, model_id in OPENAI_PRESETS:
                    self.preset_combo.addItem(name, (model_id, url))
                self.base_url_edit.setEnabled(True)
                self.base_url_edit.setText(OPENAI_PRESETS[0][1])
                self.model_edit.setText(OPENAI_PRESETS[0][2])
            self.preset_combo.addItem(CUSTOM_PRESET_LABEL, CUSTOM_PRESET_KEY)
            self.preset_combo.setCurrentIndex(0)
            self._ai_form.setRowVisible(self.base_url_edit, is_compat)
        finally:
            self.preset_combo.blockSignals(False)
            self._applying_preset = False
        self._refresh_api_key_status()
        self._sync_reasoning_effort_guard()

    def _on_preset_changed(self) -> None:
        data = self.preset_combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        model_id, url = data
        self._applying_preset = True
        try:
            self.model_edit.setText(model_id)
            if url:
                self.base_url_edit.setText(url)
        finally:
            self._applying_preset = False
        self._refresh_api_key_status()
        self._sync_reasoning_effort_guard()

    def _sync_preset_combo_from_fields(self, *_args: object) -> None:
        if self._applying_preset:
            return
        provider: MllmProvider = self.provider_combo.currentData() or MllmProvider.GEMINI
        cur_model = self.model_edit.text().strip()
        cur_url = self.base_url_edit.text().strip().rstrip("/")
        matched_idx = -1
        for idx in range(self.preset_combo.count()):
            data = self.preset_combo.itemData(idx)
            if not isinstance(data, tuple) or len(data) != 2:
                continue
            preset_model, preset_url = data
            if provider == MllmProvider.OPENAI_COMPAT:
                if preset_model == cur_model and preset_url.strip().rstrip("/") == cur_url:
                    matched_idx = idx
                    break
            elif preset_model == cur_model:
                matched_idx = idx
                break
        if matched_idx < 0:
            matched_idx = self.preset_combo.findData(CUSTOM_PRESET_KEY)
        if matched_idx >= 0 and self.preset_combo.currentIndex() != matched_idx:
            self.preset_combo.blockSignals(True)
            try:
                self.preset_combo.setCurrentIndex(matched_idx)
            finally:
                self.preset_combo.blockSignals(False)

    def _sync_reasoning_effort_guard(self, *_args: object, clear_notice: bool = False) -> None:
        provider: MllmProvider = self.provider_combo.currentData() or MllmProvider.GEMINI
        model_name = self.model_edit.text().strip()
        supported = supported_reasoning_efforts(provider, model_name)
        supported_set = set(supported)

        combo_model = self.reasoning_effort_combo.model()
        if isinstance(combo_model, QStandardItemModel):
            for idx in range(self.reasoning_effort_combo.count()):
                effort_data = self.reasoning_effort_combo.itemData(idx)
                item = combo_model.item(idx)
                if item is not None and isinstance(effort_data, ReasoningEffort):
                    item.setEnabled(effort_data in supported_set)

        self.reasoning_effort_combo.setEnabled(len(supported) > 1)

        cur_effort: ReasoningEffort = self.reasoning_effort_combo.currentData() or ReasoningEffort.AUTO
        if cur_effort not in supported_set:
            effective, notice = resolve_effective_reasoning_effort(provider, model_name, cur_effort)
            target_idx = self.reasoning_effort_combo.findData(effective)
            if target_idx >= 0:
                self.reasoning_effort_combo.blockSignals(True)
                try:
                    self.reasoning_effort_combo.setCurrentIndex(target_idx)
                finally:
                    self.reasoning_effort_combo.blockSignals(False)
            self._effort_fallback_notice = "" if self._loading_from_store else (notice or "")
        elif clear_notice:
            self._effort_fallback_notice = ""

        self._refresh_reasoning_effort_hint()

    def _refresh_reasoning_effort_hint(self) -> None:
        effort: ReasoningEffort = self.reasoning_effort_combo.currentData() or ReasoningEffort.AUTO
        provider: MllmProvider = self.provider_combo.currentData() or MllmProvider.GEMINI
        model_name = self.model_edit.text().strip()
        model_lower = model_name.lower()
        supported = supported_reasoning_efforts(provider, model_name)

        if supported == (ReasoningEffort.AUTO,):
            base_text = "当前为非思考直出模型，不支持思考/推理级别调节（已自动锁定为模型默认）。"
        elif effort == ReasoningEffort.AUTO:
            if provider == MllmProvider.GEMINI:
                if "flash-lite" in model_lower:
                    base_text = "当前 Gemini 模型默认处于 minimal 极简思考（高吞吐低延迟）。"
                elif "gemini-4-hydrogen" in model_lower or "gemini-4-neon" in model_lower:
                    base_text = "当前 Gemini 4 模型服务端默认思考级别为 high（追求更快速度可调至 low / medium）。"
                elif "gemini-3.8" in model_lower or "gemini-4" in model_lower:
                    base_text = "当前 Gemini 模型服务端默认思考级别为 medium（均衡模式）。"
                else:
                    base_text = "不显式传入 thinking_level，遵循 Gemini 服务端模型默认思考级别。"
            else:
                if model_lower.startswith(("o1", "o3", "o4")):
                    base_text = "当前 OpenAI o 系列模型服务端默认推理级别为 medium。"
                elif model_lower.startswith("gpt-6"):
                    base_text = "当前 GPT-6.2 原生推理模型遵循服务端默认推理配置（最低可手动调至 low，不支持 none）。"
                elif model_lower.startswith("gpt-5.1") or model_lower.startswith("gpt-5.2"):
                    base_text = "当前模型服务端默认推理级别为 none（低延迟直出，需思考可手动调高）。"
                elif model_lower.startswith("gpt-5"):
                    base_text = "遵循当前模型服务端默认推理配置（通常为 medium 或模型默认）。"
                else:
                    base_text = REASONING_EFFORT_HINTS[ReasoningEffort.AUTO]
        else:
            base_text = REASONING_EFFORT_HINTS.get(effort, "")

        if len(supported) > 1 and len(supported) < len(REASONING_EFFORT_ITEMS):
            names = [
                "默认" if item == ReasoningEffort.AUTO else ("极简" if item == ReasoningEffort.OFF else item.value)
                for item in supported
            ]
            base_text = f"{base_text} 支持：{' / '.join(names)}。"

        if self._effort_fallback_notice:
            self.reasoning_effort_hint_label.setText(f"{self._effort_fallback_notice}；{base_text}")
        else:
            self.reasoning_effort_hint_label.setText(base_text)

    def _refresh_api_key_status(self, *_args: object) -> None:
        text, color = describe_api_key_status(self.build_preferences())
        self.api_key_status_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.api_key_status_label.setText(text)

    def _refresh_sam3_status(self, *_args: object) -> None:
        raw = self.sam3_weights_picker.text().strip()
        if raw:
            p = Path(raw)
            if p.is_file() and p.stat().st_size > 0:
                self.sam3_status_label.setStyleSheet("color: #1a7f37; font-size: 12px;")
                self.sam3_status_label.setText(f"已指定有效 SAM 3 权重：{p.name}")
            else:
                self.sam3_status_label.setStyleSheet("color: #cf222e; font-size: 12px;")
                self.sam3_status_label.setText("指定的 SAM 3 权重文件不存在或为空，请重新选择")
            return
        found = resolve_sam3_weights(workspace_root=self.workspace_root)
        if found is not None:
            self.sam3_status_label.setStyleSheet("color: #1a7f37; font-size: 12px;")
            self.sam3_status_label.setText(f"已自动发现本地权重：{found}")
        else:
            self.sam3_status_label.setStyleSheet("color: #57606a; font-size: 12px;")
            self.sam3_status_label.setText("尚未在 weights/ 目录检测到 sam3.pt（可留空，使用 SAM 3 时再放入）")

    def _on_pick_sam_pt(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择本地 SAM 2.1 权重文件", str(self.workspace_root), "PyTorch 权重 (*.pt);;所有文件 (*.*)"
        )
        if chosen:
            self.sam_model_combo.setCurrentText(chosen)

    def _on_open_weights_dir(self) -> None:
        weights_dir = self.workspace_root / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        open_local_path(weights_dir)

    def _on_test_connection(self) -> None:
        if self.conn_test_worker is not None and self.conn_test_worker.isRunning():
            return
        cfg = build_llm_probe_config(self.build_preferences(), self.workspace_root)
        self.test_conn_btn.setEnabled(False)
        self.test_conn_btn.setText("测试中...")
        self.api_key_status_label.setStyleSheet("color: #0969da; font-size: 12px;")
        self.api_key_status_label.setText("正在向 LLM 接口发送连通性探针...")

        self.conn_test_worker = MllmConnectionTestWorker(cfg)
        self.conn_test_worker.finished_test.connect(self._on_test_connection_done)
        self.conn_test_worker.start()

    def _on_test_connection_done(self, ok: bool, message: str) -> None:
        self.test_conn_btn.setEnabled(True)
        self.test_conn_btn.setText("测试连接")
        color = "#1a7f37" if ok else "#cf222e"
        self.api_key_status_label.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 500;")
        self.api_key_status_label.setText(message)
        if not ok:
            QMessageBox.warning(self, "LLM 连接测试失败", message)

    def _on_reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self,
            "恢复默认设置",
            "确定将 AI 大模型、本地 SAM 模型与通用偏好恢复为默认值吗？\n（不会删除任何图片、标注或模型权重文件）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._load_from_preferences(AppPreferences())

    def apply_changes(self) -> None:
        """将当前对话框表单内容原子提交至 `settings_store` 并发出 `settings_applied` 信号。"""
        prefs = self.build_preferences()
        self.settings_store.set_preferences(prefs, mark_dirty=True)
        self.settings_applied.emit()

    def _on_apply(self) -> None:
        self.apply_changes()

    def _on_ok(self) -> None:
        self.apply_changes()
        self.accept()

    def done(self, result: int) -> None:  # type: ignore[override]
        if self.conn_test_worker is not None and self.conn_test_worker.isRunning():
            self.conn_test_worker.wait(2000)
        super().done(result)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        if self.conn_test_worker is not None and self.conn_test_worker.isRunning():
            self.conn_test_worker.wait(2000)
        super().closeEvent(event)
