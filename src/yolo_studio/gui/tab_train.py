"""标签页 5：YOLO26 模型训练界面。

职责：
    展示本机计算设备（CUDA GPU/CPU）状态，配置 `data.yaml`、预训练权重、场景超参数预设、
    优化器、数据增强与任意附加参数，支持配置持久化记忆、训练参数 JSON 一键导入/复制与防呆联动，
    并在后台启动训练后实时刷新 Epoch 指标卡片与 Ultralytics 日志流（含提交配置与实际生效超参数明细）。

运行环境与协作边界：
    运行于 PyQt6 主线程；训练任务委托给 `gui.workers.TrainWorker`，
    在训练产出有效 `best.pt` 时通过 `send_best_weights` 信号向主窗口同步给标签页 4 与标签页 6；
    持久化读写通过 `save_settings` / `restore_settings` 接受主窗口或外部注入的 `QSettings` 实例。
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.trainer import (
    TRAIN_HYP_PRESETS,
    TRAIN_PRESET_ITEMS,
    TRAIN_UI_MANAGED_KEYS,
    YOLO26_DETECT_MODELS,
    YOLO26_OPTIMIZERS,
    DeviceInfo,
    EpochMetrics,
    TrainConfig,
    TrainResult,
    build_train_kwargs,
    detect_compute_device,
    parse_extra_train_args,
    parse_train_arguments_text,
    split_known_and_extra_train_args,
)
from yolo_studio.gui.common import (
    LOG_MAX_LINES,
    GpuTaskGuard,
    PathPickerRow,
    make_metric_card,
    open_local_path,
)
from yolo_studio.gui.workers import TrainWorker

_CUSTOM_PRESET_KEY = "custom"
_SETTINGS_STATE_KEY = "train_tab/state_json"


def _make_double_spin(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 2,
    step: float = 0.05,
    tooltip: str = "",
    compact: bool = False,
) -> QDoubleSpinBox:
    """构建统一样式与精度的浮点超参数微调框；`compact=True` 时采用等宽网格友好的伸缩策略。"""
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    if compact:
        spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        spin.setMinimumWidth(60)
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


def _make_grid_label(text: str) -> QLabel:
    """创建右对齐的紧凑网格字段标签，保证多行网格列视觉齐整。"""
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def _as_bool(value: object, default: bool) -> bool:
    """将 JSON 导入或持久化中的布尔兼容值安全转换为 `bool`。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes", "on"}:
            return True
        if low in {"false", "0", "no", "off"}:
            return False
    return default


class _ImportTrainArgsDialog(QDialog):
    """用于粘贴或加载外部训练参数 JSON / 日志片段并回填界面的轻量对话框。"""

    def __init__(self, workspace_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.parsed_args: dict[str, object] = {}
        self.include_paths: bool = False
        self.setWindowTitle("导入训练超参数 JSON")
        self.resize(580, 400)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        hint = QLabel(
            '可粘贴完整 JSON、外部项目的 "arguments": { ... } 片段、本工具训练日志或 key=value 文本。\n'
            "界面已支持的超参数将自动回填到对应控件，其余长尾参数会自动归入「附加参数」。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #4b5563;")
        layout.addWidget(hint)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            '"arguments": {\n'
            '  "epochs": 160,\n'
            '  "imgsz": 960,\n'
            '  "optimizer": "AdamW",\n'
            '  "lr0": 0.001,\n'
            '  "mosaic": 0.0\n'
            "}"
        )
        layout.addWidget(self.text_edit, 1)

        self.include_paths_chk = QCheckBox(
            "同时导入路径、权重与实验名（data / project / name / model，默认关闭以保护当前工作区路径）"
        )
        self.include_paths_chk.setChecked(False)
        layout.addWidget(self.include_paths_chk)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("从文件加载 (.json)...")
        load_btn.clicked.connect(self._on_load_file)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("解析并回填到界面")
        ok_btn.setObjectName("primaryBtn")
        ok_btn.clicked.connect(self._on_confirm)

        btn_row.addWidget(load_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _on_load_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "选择训练参数 JSON 文件",
            str(self.workspace_root),
            "JSON / 文本文件 (*.json *.txt *.yaml *.yml);;所有文件 (*.*)",
        )
        if not chosen:
            return
        try:
            content = Path(chosen).read_text(encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "读取失败", f"无法读取文件：{exc}")
            return
        self.text_edit.setPlainText(content)

    def _on_confirm(self) -> None:
        try:
            self.parsed_args = parse_train_arguments_text(self.text_edit.toPlainText())
        except ValueError as exc:
            QMessageBox.warning(self, "参数格式错误", str(exc))
            return
        self.include_paths = self.include_paths_chk.isChecked()
        self.accept()


class TrainTab(QWidget):
    """YOLO26 模型训练功能标签页（标签页 5）。

    内部状态：
        持有硬件探测结果 `device_info`、跨页计算设备互斥协调器 `gpu_guard`、训练工作线程 `worker`、
        预设/持久化批量写入重入锁 `_applying_preset` / `_restoring_state`、用户修改脏标记 `_settings_dirty`、
        硬件自适应项覆盖标记（`_auto_device_text`、`_auto_batch_value`、`_amp_user_overridden`），
        以及最近实验目录 `_last_save_dir` 与最近最佳权重路径 `_last_best_pt`。
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
        self._applying_preset: bool = False
        self._restoring_state: bool = False
        self._settings_dirty: bool = False
        self._aug_expanded: bool = True
        self._last_save_dir: Path = workspace_root / "runs" / "detect"
        self._last_best_pt: str = ""
        self._auto_device_text: str = self.device_info.recommended_device
        self._auto_batch_value: int = 16 if self.device_info.cuda_available else 4
        self._amp_user_overridden: bool = False
        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        form_stack = QVBoxLayout(scroll_content)
        form_stack.setContentsMargins(0, 0, 4, 0)
        form_stack.setSpacing(8)

        # 1. 计算设备状态横幅（限制最大垂直高度，避免空占上半屏空间）
        hw_group = QGroupBox("1. 计算设备状态")
        hw_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        hw_layout = QHBoxLayout(hw_group)
        self.hw_label = QLabel()
        self._refresh_hw_banner()
        hw_layout.addWidget(self.hw_label)
        form_stack.addWidget(hw_group)

        # 2. 模型、预设与核心训练参数
        cfg_group = QGroupBox("2. 模型与核心训练参数")
        form = QFormLayout(cfg_group)
        form.setSpacing(6)

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
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(12)
        for model_file, desc in YOLO26_DETECT_MODELS:
            self.model_combo.addItem(desc, model_file)
        self.model_combo.setCurrentIndex(1)  # 默认选用均衡推荐的 yolo26s.pt
        self.custom_pt_btn = QPushButton("选择本地 .pt")
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.custom_pt_btn)
        form.addRow("预训练模型:", model_row)

        self.custom_model_edit = QLineEdit()
        self.custom_model_edit.setClearButtonEnabled(True)
        self.custom_model_edit.setPlaceholderText("留空使用上方选择的官方 YOLO26 权重，或填入本地 .pt 路径")
        form.addRow("自定义权重路径:", self.custom_model_edit)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        self.preset_combo = QComboBox()
        self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(10)
        for preset_key, preset_label in TRAIN_PRESET_ITEMS:
            self.preset_combo.addItem(preset_label, preset_key)
        self.preset_combo.addItem("自定义 (当前手动微调)", _CUSTOM_PRESET_KEY)
        self.preset_combo.setToolTip(
            "一键切换优化器与数据增强超参数组合：如「游戏画面 / UI 弱增强」会自动切换为 AdamW (LR0=0.001, NBS=16) 并关闭 Mosaic 与畸变增强"
        )
        compact_btn_style = "padding: 4px 8px;"
        self.import_json_btn = QPushButton("导入 JSON...")
        self.import_json_btn.setStyleSheet(compact_btn_style)
        self.import_json_btn.setToolTip('粘贴或从文件加载外部 "arguments" JSON 配置并自动回填到界面控件')
        self.export_json_btn = QPushButton("复制 JSON")
        self.export_json_btn.setStyleSheet(compact_btn_style)
        self.export_json_btn.setToolTip("将当前整套训练超参数复制为标准 JSON 片段，便于留档或分享")
        self.reset_defaults_btn = QPushButton("恢复默认")
        self.reset_defaults_btn.setStyleSheet(compact_btn_style)
        self.reset_defaults_btn.setToolTip("重置全部训练超参数与增强项为通用默认配置")
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.import_json_btn)
        preset_row.addWidget(self.export_json_btn)
        preset_row.addWidget(self.reset_defaults_btn)
        form.addRow("超参数预设:", preset_row)

        # 核心规模与硬件：统一 2 行 × 4 组网格对齐，避免 10 位 Seed 撑宽导致上下列错位
        core_grid_widget = QWidget()
        core_grid = QGridLayout(core_grid_widget)
        core_grid.setContentsMargins(0, 0, 0, 0)
        core_grid.setHorizontalSpacing(8)
        core_grid.setVerticalSpacing(6)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(100)
        self.epochs_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.epochs_spin.setMinimumWidth(64)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.setEditable(True)
        self.imgsz_combo.addItems(["320", "480", "640", "800", "960", "1024", "1280"])
        self.imgsz_combo.setCurrentText("640")
        self.imgsz_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.imgsz_combo.setMinimumWidth(64)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 512)
        self.batch_spin.setValue(16 if self.device_info.cuda_available else 4)
        self.batch_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.batch_spin.setMinimumWidth(64)
        self.batch_spin.setToolTip("设为 -1 可让 Ultralytics 自动探测最大可用 Batch Size")

        self.nbs_spin = QSpinBox()
        self.nbs_spin.setRange(1, 1024)
        self.nbs_spin.setValue(64)
        self.nbs_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.nbs_spin.setMinimumWidth(64)
        self.nbs_spin.setToolTip(
            "名义批大小 (Nominal Batch Size)：梯度累积步数 ≈ round(NBS / Batch)；小数据集小 Batch 下调小至 16 可加快参数更新频率"
        )

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 5000)
        self.patience_spin.setValue(50)
        self.patience_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.patience_spin.setMinimumWidth(64)
        self.patience_spin.setToolTip("早停容忍轮数：验证集指标连续 N 轮无提升则提前停训；设为 0 或等于 Epochs 可禁用早停")

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)
        self.workers_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.workers_spin.setMinimumWidth(64)
        self.workers_spin.setToolTip("DataLoader 工作进程数：Windows 下如遇多进程内存开销问题可设为 0")

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(0)
        self.seed_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.seed_spin.setMinimumWidth(64)
        self.seed_spin.setToolTip("随机种子 (seed)，配合确定性算子提升实验可复现性")

        self.device_edit = QLineEdit(self.device_info.recommended_device)
        self.device_edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.device_edit.setMinimumWidth(64)
        self.device_edit.setToolTip("训练设备：如 0（0 号 CUDA GPU）、0,1（多卡）或 cpu")

        core_grid.addWidget(_make_grid_label("Epochs:"), 0, 0)
        core_grid.addWidget(self.epochs_spin, 0, 1)
        core_grid.addWidget(_make_grid_label("ImgSz:"), 0, 2)
        core_grid.addWidget(self.imgsz_combo, 0, 3)
        core_grid.addWidget(_make_grid_label("Batch(-1=自动):"), 0, 4)
        core_grid.addWidget(self.batch_spin, 0, 5)
        core_grid.addWidget(_make_grid_label("NBS:"), 0, 6)
        core_grid.addWidget(self.nbs_spin, 0, 7)

        core_grid.addWidget(_make_grid_label("Patience:"), 1, 0)
        core_grid.addWidget(self.patience_spin, 1, 1)
        core_grid.addWidget(_make_grid_label("Workers:"), 1, 2)
        core_grid.addWidget(self.workers_spin, 1, 3)
        core_grid.addWidget(_make_grid_label("Seed:"), 1, 4)
        core_grid.addWidget(self.seed_spin, 1, 5)
        core_grid.addWidget(_make_grid_label("Device:"), 1, 6)
        core_grid.addWidget(self.device_edit, 1, 7)
        for col in (1, 3, 5, 7):
            core_grid.setColumnStretch(col, 1)
        form.addRow("规模与硬件:", core_grid_widget)

        opt_sched_row = QHBoxLayout()
        opt_sched_row.setSpacing(6)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.optimizer_combo.setMinimumContentsLength(10)
        for opt_name, opt_desc in YOLO26_OPTIMIZERS:
            self.optimizer_combo.addItem(opt_desc, opt_name)
        self.optimizer_combo.setToolTip(
            "注意：选 auto 时 Ultralytics 会按类别数与步数自动选定优化器并覆盖 LR0；如需严格固定 LR0 请显式选择 AdamW 或 SGD"
        )

        self.lr0_label = QLabel("LR0:")
        self.lr0_spin = _make_double_spin(
            0.0001,
            0.5,
            0.01,
            decimals=4,
            step=0.001,
            tooltip="初始学习率：SGD 常用 0.01，AdamW 常用 0.001（需将优化器设为非 auto 才严格按此值生效）",
        )
        self.lrf_spin = _make_double_spin(
            0.001,
            1.0,
            0.01,
            decimals=3,
            step=0.01,
            tooltip="末期学习率衰减系数：最终学习率 = LR0 × LRF",
        )
        self.warmup_spin = _make_double_spin(
            0.0,
            50.0,
            3.0,
            decimals=1,
            step=0.5,
            tooltip="学习率预热轮数 (warmup_epochs)",
        )

        opt_sched_row.addWidget(self.optimizer_combo, 1)
        opt_sched_row.addWidget(self.lr0_label)
        opt_sched_row.addWidget(self.lr0_spin)
        opt_sched_row.addWidget(QLabel("LRF:"))
        opt_sched_row.addWidget(self.lrf_spin)
        opt_sched_row.addWidget(QLabel("Warmup:"))
        opt_sched_row.addWidget(self.warmup_spin)
        form.addRow("优化器与学习率:", opt_sched_row)

        # 训练与复现选项：统一 2 行 × 4 列网格对齐
        opts_widget = QWidget()
        opts_grid = QGridLayout(opts_widget)
        opts_grid.setContentsMargins(0, 0, 0, 0)
        opts_grid.setHorizontalSpacing(10)
        opts_grid.setVerticalSpacing(4)

        self.amp_chk = QCheckBox("混合精度 AMP")
        self.amp_chk.setChecked(self.device_info.cuda_available)
        self.cos_lr_chk = QCheckBox("余弦退火 (cos_lr)")
        self.cache_chk = QCheckBox("内存缓存 (cache)")
        self.resume_chk = QCheckBox("断点续训 (resume)")
        self.resume_chk.setToolTip(
            "从上次中断的训练继续：优先使用「自定义权重路径」中选择的 last.pt，"
            "否则使用「保存根目录/实验名称/weights/last.pt」；新训练不会覆盖已有实验目录，会自动递增编号"
        )
        self.deterministic_chk = QCheckBox("确定性算子 (deterministic)")
        self.deterministic_chk.setChecked(True)
        self.plots_chk = QCheckBox("保存评估图表 (plots)")
        self.plots_chk.setChecked(True)
        self.val_chk = QCheckBox("训练期验证 (val)")
        self.val_chk.setChecked(True)

        opts_grid.addWidget(self.amp_chk, 0, 0)
        opts_grid.addWidget(self.cos_lr_chk, 0, 1)
        opts_grid.addWidget(self.cache_chk, 0, 2)
        opts_grid.addWidget(self.resume_chk, 0, 3)
        opts_grid.addWidget(self.deterministic_chk, 1, 0)
        opts_grid.addWidget(self.plots_chk, 1, 1)
        opts_grid.addWidget(self.val_chk, 1, 2)
        for col in range(4):
            opts_grid.setColumnStretch(col, 1)
        form.addRow("训练与复现:", opts_widget)

        default_proj = str((self.workspace_root / "runs" / "detect").resolve())
        self.project_picker = PathPickerRow(
            placeholder="训练结果保存根目录", mode="dir", default_path=default_proj
        )
        form.addRow("保存根目录:", self.project_picker)

        self.run_name_edit = QLineEdit("yolo26_train")
        form.addRow("实验名称:", self.run_name_edit)

        form_stack.addWidget(cfg_group)

        # 3. 数据增强与附加超参数（支持一键折叠/展开 + 3 行 × 5 组纵向齐整网格 + 实时校验）
        aug_group = QGroupBox("3. 数据增强与附加超参数")
        aug_outer = QVBoxLayout(aug_group)
        aug_outer.setSpacing(6)

        aug_header = QHBoxLayout()
        aug_header.setContentsMargins(0, 0, 0, 0)
        self.aug_summary_label = QLabel()
        self.aug_summary_label.setStyleSheet("color: #57606a; font-size: 12px;")
        self.toggle_aug_btn = QPushButton("收起 ▲")
        self.toggle_aug_btn.setFixedHeight(24)
        self.toggle_aug_btn.setStyleSheet("padding: 2px 10px; font-size: 12px;")
        self.toggle_aug_btn.setToolTip("折叠或展开数据增强与附加超参数编辑区域（仅隐藏控件，不影响参数生效）")
        aug_header.addWidget(self.aug_summary_label, 1)
        aug_header.addWidget(self.toggle_aug_btn)
        aug_outer.addLayout(aug_header)

        self.aug_body = QWidget()
        aug_body_layout = QVBoxLayout(self.aug_body)
        aug_body_layout.setContentsMargins(0, 2, 0, 0)
        aug_body_layout.setSpacing(6)

        aug_grid = QGridLayout()
        aug_grid.setContentsMargins(0, 0, 0, 0)
        aug_grid.setHorizontalSpacing(6)
        aug_grid.setVerticalSpacing(6)

        self.mosaic_spin = _make_double_spin(
            0.0, 1.0, 1.0, compact=True, tooltip="Mosaic 四图拼接概率 (固定视角/UI 检测建议 0.0)"
        )
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 5000)
        self.close_mosaic_spin.setValue(10)
        self.close_mosaic_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.close_mosaic_spin.setMinimumWidth(60)
        self.close_mosaic_spin.setToolTip("最后 N 轮关闭 Mosaic 增强 (close_mosaic)")
        self.mixup_spin = _make_double_spin(0.0, 1.0, 0.0, compact=True, tooltip="MixUp 图像混合概率")
        self.fliplr_spin = _make_double_spin(0.0, 1.0, 0.5, compact=True, tooltip="水平左右翻转概率 (fliplr)")
        self.flipud_spin = _make_double_spin(0.0, 1.0, 0.0, compact=True, tooltip="垂直上下翻转概率 (flipud)")

        self.translate_spin = _make_double_spin(
            0.0, 0.9, 0.1, step=0.02, compact=True, tooltip="图像平移比例 (translate)"
        )
        self.scale_spin = _make_double_spin(
            0.0, 0.95, 0.5, step=0.05, compact=True, tooltip="图像缩放增益比例 (scale)"
        )
        self.degrees_spin = _make_double_spin(
            0.0, 180.0, 0.0, decimals=1, step=5.0, compact=True, tooltip="随机旋转角度范围 (+/- degrees)"
        )
        self.shear_spin = _make_double_spin(
            0.0, 45.0, 0.0, decimals=1, step=2.0, compact=True, tooltip="随机错切角度范围 (+/- shear)"
        )
        self.perspective_spin = _make_double_spin(
            0.0,
            0.001,
            0.0,
            decimals=4,
            step=0.0001,
            compact=True,
            tooltip="随机透视变换系数 (perspective, 0~0.001)",
        )

        self.hsv_h_spin = _make_double_spin(
            0.0, 1.0, 0.015, decimals=3, step=0.005, compact=True, tooltip="HSV 色调抖动幅度 (hsv_h)"
        )
        self.hsv_s_spin = _make_double_spin(
            0.0, 1.0, 0.7, step=0.05, compact=True, tooltip="HSV 饱和度抖动幅度 (hsv_s)"
        )
        self.hsv_v_spin = _make_double_spin(
            0.0, 1.0, 0.4, step=0.05, compact=True, tooltip="HSV 明度抖动幅度 (hsv_v)"
        )
        self.cutmix_spin = _make_double_spin(0.0, 1.0, 0.0, compact=True, tooltip="CutMix 区域混合概率 (cutmix)")
        self.copy_paste_spin = _make_double_spin(
            0.0, 1.0, 0.0, compact=True, tooltip="目标复制粘贴增强比例 (copy_paste)"
        )

        aug_matrix: list[list[tuple[str, QWidget]]] = [
            [
                ("Mosaic:", self.mosaic_spin),
                ("CloseMosaic:", self.close_mosaic_spin),
                ("MixUp:", self.mixup_spin),
                ("CutMix:", self.cutmix_spin),
            ],
            [
                ("Translate:", self.translate_spin),
                ("Scale:", self.scale_spin),
                ("Degrees:", self.degrees_spin),
                ("Shear:", self.shear_spin),
            ],
            [
                ("Perspective:", self.perspective_spin),
                ("FlipLR:", self.fliplr_spin),
                ("FlipUD:", self.flipud_spin),
                ("CopyPaste:", self.copy_paste_spin),
            ],
            [
                ("HSV-H:", self.hsv_h_spin),
                ("HSV-S:", self.hsv_s_spin),
                ("HSV-V:", self.hsv_v_spin),
            ],
        ]
        for row_idx, row_items in enumerate(aug_matrix):
            for col_group, (lbl_text, widget) in enumerate(row_items):
                aug_grid.addWidget(_make_grid_label(lbl_text), row_idx, col_group * 2)
                aug_grid.addWidget(widget, row_idx, col_group * 2 + 1)
        for col_group in range(4):
            aug_grid.setColumnStretch(col_group * 2 + 1, 1)
        aug_body_layout.addLayout(aug_grid)

        extra_form = QFormLayout()
        extra_form.setContentsMargins(0, 2, 0, 0)
        extra_form.setSpacing(4)
        self.extra_args_edit = QPlainTextEdit()
        self.extra_args_edit.setFixedHeight(48)
        self.extra_args_edit.setPlaceholderText(
            "可选：透传或覆盖任意 Ultralytics 参数（支持 JSON 或 key=value，如 box=7.5, weight_decay=0.0005, freeze=10）"
        )
        self.extra_args_status_label = QLabel()
        self.extra_args_status_label.setStyleSheet("color: #6e7781; font-size: 11px;")
        self.extra_args_status_label.setVisible(False)
        extra_box = QVBoxLayout()
        extra_box.setContentsMargins(0, 0, 0, 0)
        extra_box.setSpacing(2)
        extra_box.addWidget(self.extra_args_edit)
        extra_box.addWidget(self.extra_args_status_label)
        extra_form.addRow("附加参数:", extra_box)
        aug_body_layout.addLayout(extra_form)

        aug_outer.addWidget(self.aug_body)
        form_stack.addWidget(aug_group)
        form_stack.addStretch(1)
        scroll.setWidget(scroll_content)
        left.addWidget(scroll, 1)

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

        metrics_group = QGroupBox("4. 实时训练指标")
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

        log_group = QGroupBox("5. 训练日志")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Ultralytics YOLO26 训练超参数与实时日志将在此输出...")
        log_layout.addWidget(self.log_box)
        right.addWidget(log_group, 1)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        # 信号与槽连接
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.import_json_btn.clicked.connect(self._on_import_json_dialog)
        self.export_json_btn.clicked.connect(self._on_export_json_clicked)
        self.reset_defaults_btn.clicked.connect(self._on_reset_defaults_clicked)
        self.toggle_aug_btn.clicked.connect(self._on_toggle_aug_panel)
        self.custom_pt_btn.clicked.connect(self._on_pick_custom_pt)
        self.custom_model_edit.textChanged.connect(self._on_control_state_inputs_changed)
        self.optimizer_combo.currentIndexChanged.connect(self._on_control_state_inputs_changed)
        self.mosaic_spin.valueChanged.connect(self._on_control_state_inputs_changed)
        self.extra_args_edit.textChanged.connect(self._on_extra_args_changed)
        self.amp_chk.toggled.connect(self._on_amp_toggled)
        self._bind_preset_dirty_watchers()
        self._bind_settings_dirty_watchers()

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_run_btn.clicked.connect(lambda: open_local_path(self._last_save_dir))
        self.send_eval_btn.clicked.connect(
            lambda: self.send_best_weights.emit(self._last_best_pt, self.yaml_picker.text())
            if self._last_best_pt
            else None
        )

        # 初始同步防呆联动与增强摘要
        self._sync_control_states()
        self._refresh_extra_args_status()

    def _bind_preset_dirty_watchers(self) -> None:
        """监听受预设管理的控件修改事件；用户手动改动任一超参数时自动将预设下拉框标记为“自定义”。"""
        self.optimizer_combo.currentIndexChanged.connect(self._mark_preset_custom)
        self.cos_lr_chk.toggled.connect(self._mark_preset_custom)
        for spin in (
            self.nbs_spin,
            self.lr0_spin,
            self.lrf_spin,
            self.warmup_spin,
            self.mosaic_spin,
            self.close_mosaic_spin,
            self.mixup_spin,
            self.cutmix_spin,
            self.copy_paste_spin,
            self.degrees_spin,
            self.translate_spin,
            self.scale_spin,
            self.shear_spin,
            self.perspective_spin,
            self.flipud_spin,
            self.fliplr_spin,
            self.hsv_h_spin,
            self.hsv_s_spin,
            self.hsv_v_spin,
        ):
            spin.valueChanged.connect(self._mark_preset_custom)

    def _bind_settings_dirty_watchers(self) -> None:
        """监听训练配置变更以置位持久化脏标记，避免未做修改的窗口或测试覆写已有用户配置。"""
        for combo in (self.model_combo, self.preset_combo, self.optimizer_combo):
            combo.currentIndexChanged.connect(self._mark_settings_dirty)
        self.imgsz_combo.currentTextChanged.connect(self._mark_settings_dirty)
        for edit in (self.custom_model_edit, self.device_edit, self.run_name_edit):
            edit.textChanged.connect(self._mark_settings_dirty)
        self.extra_args_edit.textChanged.connect(self._mark_settings_dirty)
        for chk in (
            self.amp_chk,
            self.cos_lr_chk,
            self.cache_chk,
            self.resume_chk,
            self.deterministic_chk,
            self.plots_chk,
            self.val_chk,
        ):
            chk.toggled.connect(self._mark_settings_dirty)
        for spin in (
            self.epochs_spin,
            self.batch_spin,
            self.nbs_spin,
            self.lr0_spin,
            self.lrf_spin,
            self.warmup_spin,
            self.patience_spin,
            self.workers_spin,
            self.seed_spin,
            self.mosaic_spin,
            self.close_mosaic_spin,
            self.mixup_spin,
            self.cutmix_spin,
            self.copy_paste_spin,
            self.degrees_spin,
            self.translate_spin,
            self.scale_spin,
            self.shear_spin,
            self.perspective_spin,
            self.flipud_spin,
            self.fliplr_spin,
            self.hsv_h_spin,
            self.hsv_s_spin,
            self.hsv_v_spin,
        ):
            spin.valueChanged.connect(self._mark_settings_dirty)

    def _mark_settings_dirty(self, *_args: object) -> None:
        if not self._restoring_state:
            self._settings_dirty = True
        self._refresh_aug_summary()

    def _on_amp_toggled(self, _checked: bool) -> None:
        if not self._restoring_state:
            self._amp_user_overridden = True

    def _mark_preset_custom(self, *_args: object) -> None:
        if self._applying_preset or self._restoring_state:
            return
        custom_idx = self.preset_combo.findData(_CUSTOM_PRESET_KEY)
        if custom_idx >= 0 and self.preset_combo.currentIndex() != custom_idx:
            self._applying_preset = True
            try:
                self.preset_combo.setCurrentIndex(custom_idx)
            finally:
                self._applying_preset = False

    def _on_preset_changed(self, _index: int) -> None:
        if self._applying_preset or self._restoring_state:
            return
        preset_key = str(self.preset_combo.currentData() or "")
        if preset_key and preset_key != _CUSTOM_PRESET_KEY:
            self.apply_preset(preset_key)

    def _on_control_state_inputs_changed(self, *_args: object) -> None:
        self._sync_control_states()

    def _sync_control_states(self) -> None:
        """统一维护防呆联动禁用态：auto 优化器置灰 LR0、自定义权重置灰官方下拉框、Mosaic=0 置灰 CloseMosaic。"""
        is_auto_opt = str(self.optimizer_combo.currentData() or "auto").lower() == "auto"
        self.lr0_spin.setEnabled(not is_auto_opt)
        self.lr0_label.setText("LR0(自动):" if is_auto_opt else "LR0:")
        if is_auto_opt:
            self.lr0_spin.setToolTip(
                "当前优化器为 auto，Ultralytics 将按类别数与步数自动计算并覆盖 LR0；切换到 AdamW / SGD 后可手动编辑"
            )
        else:
            self.lr0_spin.setToolTip("初始学习率：当前优化器已显式指定，训练时将严格按此值生效")

        has_custom_model = bool(self.custom_model_edit.text().strip())
        self.model_combo.setEnabled(not has_custom_model)
        self.model_combo.setToolTip(
            "已指定自定义权重路径，训练时将优先加载自定义 .pt；清空自定义路径后恢复选用官方模型"
            if has_custom_model
            else "选择官方 YOLO26 目标检测预训练模型"
        )

        mosaic_enabled = self.mosaic_spin.value() > 0.0
        self.close_mosaic_spin.setEnabled(mosaic_enabled)
        self.close_mosaic_spin.setToolTip(
            "最后 N 轮关闭 Mosaic 增强 (close_mosaic)"
            if mosaic_enabled
            else "当前 Mosaic 概率为 0，已彻底关闭拼接增强，CloseMosaic 不再生效"
        )
        self._refresh_aug_summary()

    def _on_toggle_aug_panel(self) -> None:
        self.set_aug_panel_expanded(not self._aug_expanded)
        if not self._restoring_state:
            self._settings_dirty = True

    def set_aug_panel_expanded(self, expanded: bool) -> None:
        """展开或折叠「3. 数据增强与附加超参数」编辑区，并同步按钮文案与摘要行。"""
        self._aug_expanded = bool(expanded)
        self.aug_body.setVisible(self._aug_expanded)
        self.toggle_aug_btn.setText("收起 ▲" if self._aug_expanded else "展开 ▼")
        self._refresh_aug_summary()

    def _refresh_aug_summary(self) -> None:
        """刷新数据增强分组顶部的摘要文案：折叠时展示关键增强参数快照，展开时展示简短引导。"""
        if not self._aug_expanded:
            extra_note = " | 含附加参数" if self.extra_args_edit.toPlainText().strip() else ""
            self.aug_summary_label.setText(
                f"当前生效: Mosaic={self.mosaic_spin.value():.2f}, "
                f"Scale={self.scale_spin.value():.2f}, "
                f"Translate={self.translate_spin.value():.2f}, "
                f"HSV=({self.hsv_h_spin.value():.3f}, {self.hsv_s_spin.value():.2f}, {self.hsv_v_spin.value():.2f})"
                f"{extra_note}"
            )
        else:
            self.aug_summary_label.setText("可细粒度调节数据增强强度，或通过下方「附加参数」透传任意 Ultralytics 超参数")

    def _on_extra_args_changed(self) -> None:
        self._refresh_extra_args_status()
        self._refresh_aug_summary()

    def _refresh_extra_args_status(self) -> None:
        """实时校验「附加参数」输入并在下方以颜色状态条提示解析项数、覆盖冲突或语法错误。"""
        raw = self.extra_args_edit.toPlainText().strip()
        if not raw:
            self.extra_args_status_label.setVisible(False)
            self.extra_args_status_label.setText("")
            return
        try:
            parsed = parse_extra_train_args(raw)
        except ValueError as exc:
            self.extra_args_status_label.setStyleSheet("color: #cf222e; font-size: 11px; font-weight: 500;")
            self.extra_args_status_label.setText(f"语法错误: {exc}")
            self.extra_args_status_label.setVisible(True)
            return

        overridden = [k for k in parsed if k in TRAIN_UI_MANAGED_KEYS]
        if overridden:
            names = ", ".join(overridden)
            self.extra_args_status_label.setStyleSheet("color: #9a6700; font-size: 11px; font-weight: 500;")
            self.extra_args_status_label.setText(
                f"已解析 {len(parsed)} 项附加参数（注意：以下键将覆盖上方界面设置: {names}）"
            )
        else:
            self.extra_args_status_label.setStyleSheet("color: #1a7f37; font-size: 11px;")
            self.extra_args_status_label.setText(f"已解析 {len(parsed)} 项附加参数")
        self.extra_args_status_label.setVisible(True)

    def apply_preset(self, preset_key: str) -> None:
        """将指定预设字典中的优化器与数据增强参数同步写入界面控件。

        调用约束：
            `preset_key` 须存在于 `TRAIN_HYP_PRESETS`，否则抛出 `ValueError`。
            写入期间置位 `_applying_preset`，防止控件 `valueChanged` 反向把预设改回“自定义”。
        """
        if preset_key not in TRAIN_HYP_PRESETS:
            raise ValueError(f"未知的超参数预设: {preset_key}")
        preset = TRAIN_HYP_PRESETS[preset_key]
        self._applying_preset = True
        try:
            combo_idx = self.preset_combo.findData(preset_key)
            if combo_idx >= 0 and self.preset_combo.currentIndex() != combo_idx:
                self.preset_combo.setCurrentIndex(combo_idx)

            opt_idx = self.optimizer_combo.findData(str(preset["optimizer"]))
            if opt_idx >= 0:
                self.optimizer_combo.setCurrentIndex(opt_idx)

            self.lr0_spin.setValue(float(preset["lr0"]))  # type: ignore[arg-type]
            self.lrf_spin.setValue(float(preset["lrf"]))  # type: ignore[arg-type]
            self.warmup_spin.setValue(float(preset["warmup_epochs"]))  # type: ignore[arg-type]
            self.nbs_spin.setValue(int(preset["nbs"]))  # type: ignore[call-overload]
            self.cos_lr_chk.setChecked(bool(preset["cos_lr"]))

            self.mosaic_spin.setValue(float(preset["mosaic"]))  # type: ignore[arg-type]
            self.close_mosaic_spin.setValue(int(preset["close_mosaic"]))  # type: ignore[call-overload]
            self.mixup_spin.setValue(float(preset["mixup"]))  # type: ignore[arg-type]
            self.cutmix_spin.setValue(float(preset["cutmix"]))  # type: ignore[arg-type]
            self.copy_paste_spin.setValue(float(preset["copy_paste"]))  # type: ignore[arg-type]
            self.degrees_spin.setValue(float(preset["degrees"]))  # type: ignore[arg-type]
            self.translate_spin.setValue(float(preset["translate"]))  # type: ignore[arg-type]
            self.scale_spin.setValue(float(preset["scale"]))  # type: ignore[arg-type]
            self.shear_spin.setValue(float(preset["shear"]))  # type: ignore[arg-type]
            self.perspective_spin.setValue(float(preset["perspective"]))  # type: ignore[arg-type]
            self.flipud_spin.setValue(float(preset["flipud"]))  # type: ignore[arg-type]
            self.fliplr_spin.setValue(float(preset["fliplr"]))  # type: ignore[arg-type]
            self.hsv_h_spin.setValue(float(preset["hsv_h"]))  # type: ignore[arg-type]
            self.hsv_s_spin.setValue(float(preset["hsv_s"]))  # type: ignore[arg-type]
            self.hsv_v_spin.setValue(float(preset["hsv_v"]))  # type: ignore[arg-type]
        finally:
            self._applying_preset = False
        if not self._restoring_state:
            self._settings_dirty = True
        self._sync_control_states()

    def _apply_known_args_to_widgets(self, known: dict[str, object]) -> list[str]:
        """将已识别的 Ultralytics 键回填到对应界面控件，返回实际应用的字段名列表。"""
        applied: list[str] = []

        def _set_int(spin: QSpinBox, key: str) -> None:
            if key in known and known[key] is not None:
                spin.setValue(int(known[key]))  # type: ignore[call-overload]
                applied.append(key)

        def _set_float(spin: QDoubleSpinBox, key: str) -> None:
            if key in known and known[key] is not None:
                spin.setValue(float(known[key]))  # type: ignore[arg-type]
                applied.append(key)

        def _set_bool(chk: QCheckBox, key: str) -> None:
            if key in known and known[key] is not None:
                chk.setChecked(_as_bool(known[key], chk.isChecked()))
                applied.append(key)

        if "data" in known and known["data"]:
            self.yaml_picker.setText(str(known["data"]))
            applied.append("data")
        if "project" in known and known["project"]:
            self.project_picker.setText(str(known["project"]))
            applied.append("project")
        if "name" in known and known["name"]:
            self.run_name_edit.setText(str(known["name"]))
            applied.append("name")
        if "model" in known and known["model"]:
            model_val = str(known["model"])
            idx = self.model_combo.findData(model_val)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
                self.custom_model_edit.clear()
            else:
                self.custom_model_edit.setText(model_val)
            applied.append("model")

        _set_int(self.epochs_spin, "epochs")
        if "imgsz" in known and known["imgsz"] is not None:
            self.imgsz_combo.setCurrentText(str(int(known["imgsz"])))  # type: ignore[call-overload]
            applied.append("imgsz")
        _set_int(self.batch_spin, "batch")
        _set_int(self.nbs_spin, "nbs")
        _set_int(self.patience_spin, "patience")
        _set_int(self.workers_spin, "workers")
        _set_int(self.seed_spin, "seed")

        if "device" in known and known["device"] is not None:
            self.device_edit.setText(str(known["device"]))
            applied.append("device")

        if "optimizer" in known and known["optimizer"] is not None:
            opt_text = str(known["optimizer"]).strip()
            idx = self.optimizer_combo.findData(opt_text)
            if idx < 0:
                # 忽略大小写匹配已知优化器名
                for i in range(self.optimizer_combo.count()):
                    if str(self.optimizer_combo.itemData(i)).lower() == opt_text.lower():
                        idx = i
                        break
            if idx >= 0:
                self.optimizer_combo.setCurrentIndex(idx)
                applied.append("optimizer")

        _set_float(self.lr0_spin, "lr0")
        _set_float(self.lrf_spin, "lrf")
        _set_float(self.warmup_spin, "warmup_epochs")

        _set_bool(self.amp_chk, "amp")
        if "amp" in applied:
            self._amp_user_overridden = True
        _set_bool(self.cos_lr_chk, "cos_lr")
        _set_bool(self.cache_chk, "cache")
        _set_bool(self.resume_chk, "resume")
        _set_bool(self.deterministic_chk, "deterministic")
        _set_bool(self.plots_chk, "plots")
        _set_bool(self.val_chk, "val")

        _set_float(self.mosaic_spin, "mosaic")
        _set_int(self.close_mosaic_spin, "close_mosaic")
        _set_float(self.mixup_spin, "mixup")
        _set_float(self.cutmix_spin, "cutmix")
        _set_float(self.copy_paste_spin, "copy_paste")
        _set_float(self.degrees_spin, "degrees")
        _set_float(self.translate_spin, "translate")
        _set_float(self.scale_spin, "scale")
        _set_float(self.shear_spin, "shear")
        _set_float(self.perspective_spin, "perspective")
        _set_float(self.flipud_spin, "flipud")
        _set_float(self.fliplr_spin, "fliplr")
        _set_float(self.hsv_h_spin, "hsv_h")
        _set_float(self.hsv_s_spin, "hsv_s")
        _set_float(self.hsv_v_spin, "hsv_v")
        return applied

    def _sync_preset_combo_from_widgets(self) -> None:
        """若当前控件数值与某个内置预设完全一致则选中该预设，否则设为“自定义”。"""
        matched_key = _CUSTOM_PRESET_KEY
        current_opt = str(self.optimizer_combo.currentData() or "auto")
        for preset_key, preset in TRAIN_HYP_PRESETS.items():
            if (
                current_opt == str(preset["optimizer"])
                and abs(self.lr0_spin.value() - float(preset["lr0"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.lrf_spin.value() - float(preset["lrf"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.warmup_spin.value() - float(preset["warmup_epochs"])) < 1e-6  # type: ignore[arg-type]
                and self.nbs_spin.value() == int(preset["nbs"])  # type: ignore[call-overload]
                and self.cos_lr_chk.isChecked() == bool(preset["cos_lr"])
                and abs(self.mosaic_spin.value() - float(preset["mosaic"])) < 1e-6  # type: ignore[arg-type]
                and self.close_mosaic_spin.value() == int(preset["close_mosaic"])  # type: ignore[call-overload]
                and abs(self.mixup_spin.value() - float(preset["mixup"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.cutmix_spin.value() - float(preset["cutmix"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.copy_paste_spin.value() - float(preset["copy_paste"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.degrees_spin.value() - float(preset["degrees"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.translate_spin.value() - float(preset["translate"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.scale_spin.value() - float(preset["scale"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.shear_spin.value() - float(preset["shear"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.perspective_spin.value() - float(preset["perspective"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.flipud_spin.value() - float(preset["flipud"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.fliplr_spin.value() - float(preset["fliplr"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.hsv_h_spin.value() - float(preset["hsv_h"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.hsv_s_spin.value() - float(preset["hsv_s"])) < 1e-6  # type: ignore[arg-type]
                and abs(self.hsv_v_spin.value() - float(preset["hsv_v"])) < 1e-6  # type: ignore[arg-type]
            ):
                matched_key = preset_key
                break
        idx = self.preset_combo.findData(matched_key)
        if idx >= 0:
            self._applying_preset = True
            try:
                self.preset_combo.setCurrentIndex(idx)
            finally:
                self._applying_preset = False

    def import_arguments_from_text(
        self, raw: str, *, include_paths: bool = False
    ) -> tuple[list[str], list[str]]:
        """解析训练参数 JSON / 键值对文本并自动回填到界面控件，返回 `(已回填界面字段, 附加参数键)`。

        调用与失败行为：
            `raw` 语法非法或不含任何有效键时抛出中文 `ValueError`，不改动当前界面状态。
            界面没有专用控件的长尾参数会以紧凑 JSON 写入 `extra_args_edit`；若导入中显式指定 `mosaic=0`
            且未指定 `close_mosaic`，同步将 `close_mosaic` 归零保持语义一致。
        """
        parsed = parse_train_arguments_text(raw)
        known, extra = split_known_and_extra_train_args(parsed, include_paths=include_paths)
        if not known and not extra:
            raise ValueError("解析到的参数均属于路径字段且未勾选导入路径，没有可回填的超参数。")

        # 单独处理未知 optimizer 名称：若不在下拉框内则移入附加参数以免丢失
        if "optimizer" in known and known["optimizer"] is not None:
            opt_val = str(known["optimizer"]).strip()
            known_opts = {name.lower() for name, _ in YOLO26_OPTIMIZERS}
            if opt_val.lower() not in known_opts:
                extra["optimizer"] = known.pop("optimizer")

        if "mosaic" in known and float(known["mosaic"]) == 0.0 and "close_mosaic" not in known:  # type: ignore[arg-type]
            known["close_mosaic"] = 0

        self._applying_preset = True
        try:
            applied = self._apply_known_args_to_widgets(known)
            if extra:
                self.extra_args_edit.setPlainText(json.dumps(extra, ensure_ascii=False))
            else:
                self.extra_args_edit.clear()
        finally:
            self._applying_preset = False

        self._sync_preset_combo_from_widgets()
        self._sync_control_states()
        self._refresh_extra_args_status()
        self._settings_dirty = True
        return applied, list(extra.keys())

    def export_arguments_json(self) -> str:
        """将当前界面全部训练超参数序列化为带 `"arguments"` 顶层包装的缩进 JSON 文本。

        与 `build_train_config` 的区别：
            导出不要求 `data.yaml` 磁盘文件已存在，便于在准备数据前先导出或备份调好的超参数配置；
            但若 `ImgSz` 非正整数或「附加参数」语法有误仍会抛出中文 `ValueError`。
        """
        cfg = self._assemble_train_config(require_existing_yaml=False)
        kwargs = build_train_kwargs(cfg)
        lines = json.dumps(kwargs, ensure_ascii=False, indent=2, default=str).splitlines()
        if lines:
            lines[0] = f'"arguments": {lines[0]}'
        return "\n".join(lines)

    def reset_defaults(self) -> None:
        """将全部训练超参数与数据增强控件重置为通用默认状态，并恢复本机硬件推荐设备/批次/AMP。"""
        self._applying_preset = True
        try:
            self.model_combo.setCurrentIndex(1)
            self.custom_model_edit.clear()
            self.epochs_spin.setValue(100)
            self.imgsz_combo.setCurrentText("640")
            self._auto_batch_value = 16 if self.device_info.cuda_available else 4
            self.batch_spin.setValue(self._auto_batch_value)
            self.patience_spin.setValue(50)
            self.workers_spin.setValue(4)
            self.seed_spin.setValue(0)
            self._auto_device_text = self.device_info.recommended_device
            self.device_edit.setText(self._auto_device_text)
            self._amp_user_overridden = False
            self.amp_chk.setChecked(self.device_info.cuda_available)
            self.cache_chk.setChecked(False)
            self.resume_chk.setChecked(False)
            self.deterministic_chk.setChecked(True)
            self.plots_chk.setChecked(True)
            self.val_chk.setChecked(True)
            self.run_name_edit.setText("yolo26_train")
            self.extra_args_edit.clear()
        finally:
            self._applying_preset = False

        self.apply_preset("default")
        self._refresh_extra_args_status()
        self._settings_dirty = True

    def _on_import_json_dialog(self) -> None:
        dlg = _ImportTrainArgsDialog(self.workspace_root, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            applied, extra_keys = self.import_arguments_from_text(
                dlg.text_edit.toPlainText(), include_paths=dlg.include_paths
            )
        except ValueError as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return

        extra_msg = f"\n归入「附加参数」({len(extra_keys)} 项): {', '.join(extra_keys)}" if extra_keys else ""
        QMessageBox.information(
            self,
            "导入完成",
            f"已成功回填 {len(applied)} 项界面超参数。{extra_msg}",
        )

    def _on_export_json_clicked(self) -> None:
        try:
            text = self.export_arguments_json()
        except ValueError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        self.export_json_btn.setText("已复制 ✓")
        QTimer.singleShot(1800, lambda: self.export_json_btn.setText("复制 JSON"))

    def _on_reset_defaults_clicked(self) -> None:
        self.reset_defaults()

    def save_settings(self, settings: QSettings, *, force: bool = False) -> None:
        """将当前训练页超参数配置序列化写入 `settings`；未做改动且 `force=False` 时跳过写入。

        设计原因：
            通过 `_settings_dirty` 保护仅在用户实际改动训练页配置后落盘，避免空打开/关闭窗口
            或自动化测试误覆盖用户平时调好的持久化配置。
        """
        if not force and not self._settings_dirty:
            return
        dev_text = self.device_edit.text().strip()
        state: dict[str, object] = {
            "version": 1,
            "preset": str(self.preset_combo.currentData() or _CUSTOM_PRESET_KEY),
            "model_file": str(self.model_combo.currentData() or "yolo26s.pt"),
            "custom_model": self.custom_model_edit.text().strip(),
            "epochs": self.epochs_spin.value(),
            "imgsz": self.imgsz_combo.currentText().strip() or "640",
            "batch": self.batch_spin.value(),
            "batch_is_auto": self.batch_spin.value() == self._auto_batch_value,
            "nbs": self.nbs_spin.value(),
            "optimizer": str(self.optimizer_combo.currentData() or "auto"),
            "lr0": self.lr0_spin.value(),
            "lrf": self.lrf_spin.value(),
            "warmup_epochs": self.warmup_spin.value(),
            "patience": self.patience_spin.value(),
            "workers": self.workers_spin.value(),
            "seed": self.seed_spin.value(),
            "custom_device": "" if dev_text == self._auto_device_text else dev_text,
            "amp": self.amp_chk.isChecked(),
            "amp_user_overridden": self._amp_user_overridden,
            "cos_lr": self.cos_lr_chk.isChecked(),
            "cache": self.cache_chk.isChecked(),
            "deterministic": self.deterministic_chk.isChecked(),
            "plots": self.plots_chk.isChecked(),
            "val": self.val_chk.isChecked(),
            "mosaic": self.mosaic_spin.value(),
            "close_mosaic": self.close_mosaic_spin.value(),
            "mixup": self.mixup_spin.value(),
            "cutmix": self.cutmix_spin.value(),
            "copy_paste": self.copy_paste_spin.value(),
            "degrees": self.degrees_spin.value(),
            "translate": self.translate_spin.value(),
            "scale": self.scale_spin.value(),
            "shear": self.shear_spin.value(),
            "perspective": self.perspective_spin.value(),
            "flipud": self.flipud_spin.value(),
            "fliplr": self.fliplr_spin.value(),
            "hsv_h": self.hsv_h_spin.value(),
            "hsv_s": self.hsv_s_spin.value(),
            "hsv_v": self.hsv_v_spin.value(),
            "extra_args_text": self.extra_args_edit.toPlainText(),
            "aug_expanded": self._aug_expanded,
            "run_name": self.run_name_edit.text().strip() or "yolo26_train",
        }
        settings.setValue(_SETTINGS_STATE_KEY, json.dumps(state, ensure_ascii=False))

    def restore_settings(self, settings: QSettings) -> None:
        """从 `settings` 恢复上次保存的训练页超参数配置；数据缺失或损坏时静默保持默认。

        硬件兼容处理：
            若上次未自定义设备，继续跟随当前机器自动探测；若上次保存的自定义 GPU 设备在当前无 GPU
            环境下不可用，则安全回退为本机推荐设备。恢复过程不置位 `_settings_dirty`。
        """
        raw = settings.value(_SETTINGS_STATE_KEY, "")
        if not isinstance(raw, str) or not raw.strip():
            return
        try:
            state = json.loads(raw)
        except Exception:
            return
        if not isinstance(state, dict):
            return

        self._restoring_state = True
        self._applying_preset = True
        try:
            model_file = str(state.get("model_file", ""))
            model_idx = self.model_combo.findData(model_file)
            if model_idx >= 0:
                self.model_combo.setCurrentIndex(model_idx)
            if "custom_model" in state:
                self.custom_model_edit.setText(str(state.get("custom_model") or ""))

            self._apply_known_args_to_widgets(state)

            if state.get("batch_is_auto"):
                self.batch_spin.setValue(self._auto_batch_value)
            else:
                # 用户此前手动改过 batch，标记自动基线失效以防异步探测返回时将其覆盖
                self._auto_batch_value = -999

            self._amp_user_overridden = bool(state.get("amp_user_overridden", False))
            if not self._amp_user_overridden:
                self.amp_chk.setChecked(self.device_info.cuda_available)

            custom_dev = str(state.get("custom_device") or "").strip()
            if custom_dev:
                # 仅在当前设备兼容（非探测中且无 GPU 时不恢复 CUDA 卡号）时应用自定义设备
                if self.device_info.probing or self.device_info.cuda_available or custom_dev.lower() == "cpu":
                    self.device_edit.setText(custom_dev)

            if "run_name" in state and state.get("run_name"):
                self.run_name_edit.setText(str(state["run_name"]))
            if "extra_args_text" in state:
                self.extra_args_edit.setPlainText(str(state.get("extra_args_text") or ""))
            if "aug_expanded" in state:
                self.set_aug_panel_expanded(bool(state["aug_expanded"]))

            preset_key = str(state.get("preset") or _CUSTOM_PRESET_KEY)
            p_idx = self.preset_combo.findData(preset_key)
            if p_idx >= 0:
                self.preset_combo.setCurrentIndex(p_idx)
            else:
                self._sync_preset_combo_from_widgets()
        except Exception:
            pass
        finally:
            self._applying_preset = False
            self._restoring_state = False

        self._sync_control_states()
        self._refresh_extra_args_status()

    def _refresh_hw_banner(self) -> None:
        """按当前 `device_info` 状态更新顶部计算设备提示横幅文案与配色。"""
        if self.device_info.probing:
            hw_text = "正在后台检测计算设备 (CUDA GPU / CPU)..."
            hw_color = "#0969da"
        elif self.device_info.cuda_available:
            hw_text = (
                f"已就绪 GPU: {self.device_info.device_name} "
                f"({self.device_info.vram_gb:.1f} GB 显存) | CUDA 可用"
            )
            hw_color = "#1f7a1f"
        else:
            hw_text = "未检测到可用 CUDA GPU，将使用 CPU 训练（速度较慢）"
            hw_color = "#b25900"
        self.hw_label.setText(hw_text)
        self.hw_label.setStyleSheet(f"color: {hw_color}; font-weight: 600;")

    def apply_device_info(self, info: DeviceInfo) -> None:
        """异步硬件探测完成后刷新横幅与尚未被用户手动改动的默认训练参数。"""
        was_probing = self.device_info.probing
        self.device_info = info
        self._refresh_hw_banner()

        self._restoring_state = True
        try:
            cur_dev = self.device_edit.text().strip()
            if cur_dev == self._auto_device_text or (not info.cuda_available and cur_dev.isdigit()):
                self.device_edit.setText(info.recommended_device)
            self._auto_device_text = info.recommended_device

            target_batch = 16 if info.cuda_available else 4
            if self.batch_spin.value() == self._auto_batch_value:
                self.batch_spin.setValue(target_batch)
                self._auto_batch_value = target_batch

            if (
                was_probing
                and not self._amp_user_overridden
                and (self.worker is None or not self.worker.isRunning())
            ):
                self.amp_chk.setChecked(info.cuda_available)
        finally:
            self._restoring_state = False

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

    def _assemble_train_config(self, *, require_existing_yaml: bool) -> TrainConfig:
        """从当前界面控件提取并校验字段，组装 `TrainConfig`。"""
        yaml_text = self.yaml_picker.text()
        if require_existing_yaml and (not yaml_text or not Path(yaml_text).is_file()):
            raise ValueError("请选择有效的 data.yaml 文件。")

        try:
            imgsz = int(self.imgsz_combo.currentText().strip())
            if imgsz <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("ImgSz 必须是正整数。") from exc

        extra_args = parse_extra_train_args(self.extra_args_edit.toPlainText())

        model_name = self.custom_model_edit.text().strip() or str(self.model_combo.currentData())
        proj_dir = Path(self.project_picker.text() or (self.workspace_root / "runs" / "detect"))
        run_name = self.run_name_edit.text().strip() or "yolo26_train"
        yaml_path = Path(yaml_text) if yaml_text else (self.workspace_root / "datasets" / "custom_yolo26" / "data.yaml")

        return TrainConfig(
            data_yaml=yaml_path,
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
            optimizer=str(self.optimizer_combo.currentData() or "auto"),
            lrf=self.lrf_spin.value(),
            warmup_epochs=self.warmup_spin.value(),
            nbs=self.nbs_spin.value(),
            cos_lr=self.cos_lr_chk.isChecked(),
            mosaic=self.mosaic_spin.value(),
            close_mosaic=self.close_mosaic_spin.value(),
            mixup=self.mixup_spin.value(),
            cutmix=self.cutmix_spin.value(),
            copy_paste=self.copy_paste_spin.value(),
            degrees=self.degrees_spin.value(),
            translate=self.translate_spin.value(),
            scale=self.scale_spin.value(),
            shear=self.shear_spin.value(),
            perspective=self.perspective_spin.value(),
            flipud=self.flipud_spin.value(),
            fliplr=self.fliplr_spin.value(),
            hsv_h=self.hsv_h_spin.value(),
            hsv_s=self.hsv_s_spin.value(),
            hsv_v=self.hsv_v_spin.value(),
            seed=self.seed_spin.value(),
            deterministic=self.deterministic_chk.isChecked(),
            plots=self.plots_chk.isChecked(),
            val=self.val_chk.isChecked(),
            extra_args=extra_args,
        )

    def build_train_config(self) -> TrainConfig:
        """校验当前界面表单并组装完整的 `TrainConfig`；输入非法时抛出带中文说明的 `ValueError`。"""
        return self._assemble_train_config(require_existing_yaml=True)

    def _on_start(self) -> None:
        """校验表单与附加超参数后锁定运行按钮、清空日志并启动 `TrainWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())

        try:
            cfg = self.build_train_config()
        except ValueError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return

        self._settings_dirty = True
        self._last_save_dir = cfg.project_dir / cfg.run_name

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
