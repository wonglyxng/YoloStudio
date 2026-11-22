"""标签页 5：YOLO26 模型训练界面。

职责：
    展示本机计算设备（CUDA GPU/CPU）状态，配置 `data.yaml`、预训练权重、场景超参数预设、
    优化器、数据增强与任意附加参数，支持配置持久化记忆、训练参数 JSON 一键导入/复制与防呆联动，
    并在后台启动训练后实时刷新 Epoch 指标卡片、训练时长与 Ultralytics 日志流（含提交配置与实际生效超参数明细）。

运行环境与协作边界：
    运行于 PyQt6 主线程；训练任务委托给 `gui.workers.TrainWorker`，
    在训练产出有效 `best.pt` 时通过 `send_best_weights` 信号向主窗口同步给标签页 3 与标签页 6，
    点击「将 best.pt 发送到「6. 验证与推理」→」时再次同步并通过 `request_eval_tab` 请求主窗口切换至标签页 6；
    持久化读写通过 `save_settings` / `restore_settings` 接受主窗口或外部注入的 `QSettings` 实例。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QCloseEvent
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
    ComputeDeviceControl,
    GpuTaskGuard,
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    make_metric_card,
    open_local_path,
)
from yolo_studio.gui.workers import TrainWorker

_CUSTOM_PRESET_KEY = "custom"
_SETTINGS_STATE_KEY = "train_tab/state_json"
_BEST_EVAL_METRIC_KEYS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)
_BEST_EVAL_CARD_IDLE_QSS = (
    "QWidget#bestEvalCard { border: 1px solid #d8dee4; border-radius: 6px; background-color: #f8fafc; }"
)
_BEST_EVAL_CARD_READY_QSS = (
    "QWidget#bestEvalCard { border: 1px solid #86efac; border-radius: 6px; background-color: #f0fdf4; }"
)


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


class _BatchSpinBox(QSpinBox):
    """支持 `-1`（GPU 自动探测 AutoBatch）与正整数（`>= 1`）的批大小微调框，自动跳过无效值 `0`。"""

    def stepBy(self, steps: int) -> None:
        cur = self.value()
        if steps < 0 and cur == 1 and self.minimum() == -1:
            self.setValue(-1)
            return
        if steps > 0 and cur == -1:
            self.setValue(1)
            return
        super().stepBy(steps)
        if self.value() == 0:
            self.setValue(1 if steps >= 0 else (self.minimum() if self.minimum() < 0 else 1))


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


def format_elapsed_duration(seconds: float) -> str:
    """将已用训练秒数格式化为固定宽度 `HH:MM:SS` 文本。

    边界行为：
        非有限值（`NaN`/`inf`）或非正数按 `0` 秒返回 `"00:00:00"`；
        超过 99 小时后小时位继续自然递增（如 `"100:00:00"`），不回绕归零。
    """
    if not math.isfinite(seconds) or seconds <= 0:
        total_sec = 0
    else:
        total_sec = int(seconds)
    hours, rem = divmod(total_sec, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
        持有硬件探测结果 `device_info`、统一设备控制组件 `device_ctrl`、跨页计算设备互斥协调器 `gpu_guard`、
        训练工作线程 `worker`、预设/持久化批量写入重入锁 `_applying_preset` / `_restoring_state`、
        用户修改脏标记 `_settings_dirty`、硬件自适应项覆盖标记（`_auto_batch_value`、`_amp_user_overridden`、
        `_amp_saved_before_cpu`）、基于 `time.monotonic()` 单调时钟域的训练启动时间戳 `_train_started_at`
        与实时刷新定时器 `_elapsed_timer`、`best.pt` 最终评估就绪标记 `_has_best_eval_metrics`，
        以及最近实验目录 `_last_save_dir` 与最近最佳权重路径 `_last_best_pt`。
    """

    send_best_weights = Signal(str, str)  # 信号参数：(weights_path, data_yaml)
    request_eval_tab = Signal()  # 请求主窗口切换到「6. 验证、推理与导出」，不携带数据

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
        self._auto_batch_value: int = 16 if self.device_info.cuda_available else 4
        self._amp_user_overridden: bool = False
        self._amp_saved_before_cpu: bool | None = None
        self._amp_disabled_for_cpu: bool = False
        self._batch_cpu_clamped: bool = False
        self._batch_was_auto_before_cpu: bool = False
        # 训练计时器状态：使用 `time.monotonic()` 单调时钟域，避免系统时间校时影响已用时长计算
        self._train_started_at: float | None = None
        self._has_best_eval_metrics: bool = False
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(500)
        self._init_ui()
        self._elapsed_timer.timeout.connect(self._refresh_elapsed_time)

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

        # 1. 计算设备（GPU 加速开关 + 硬件状态合一，多卡时自动展开选卡下拉框）
        hw_group = QGroupBox("1. 计算设备")
        hw_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        hw_layout = QHBoxLayout(hw_group)
        self.device_ctrl = ComputeDeviceControl(
            self.device_info,
            show_status_text=True,
            allow_multi_gpu=True,
            checkbox_text="启用 GPU 加速",
            compact_status=False,
        )
        # 保留 `hw_label` 别名指向内部状态标签，兼容外部检查与测试引用
        self.hw_label = self.device_ctrl.status_label
        hw_layout.addWidget(self.device_ctrl, 1)
        form_stack.addWidget(hw_group)

        # 2. 模型、预设与核心训练参数
        cfg_group = QGroupBox("2. 模型与核心训练参数")
        form = QFormLayout(cfg_group)
        configure_form_layout(form, vertical_spacing=6)

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

        self.batch_spin = _BatchSpinBox()
        self.batch_spin.setRange(-1, 512)
        self.batch_spin.setSpecialValueText("自动")
        self.batch_spin.setValue(16 if self.device_info.cuda_available else 4)
        self.batch_spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.batch_spin.setMinimumWidth(64)
        self.batch_spin.setToolTip("设为 -1（显示为“自动”）可让 Ultralytics 自动探测最大可用 Batch Size")

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

        self.accum_hint_label = QLabel()
        self.accum_hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.accum_hint_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.accum_hint_label.setMinimumWidth(64)
        self.accum_hint_label.setStyleSheet(
            "color: #57606a; background: #f3f4f6; border: 1px solid #d8dee4; "
            "border-radius: 4px; padding: 2px 6px; font-size: 12px;"
        )

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
        core_grid.addWidget(_make_grid_label("梯度累积:"), 1, 6)
        core_grid.addWidget(self.accum_hint_label, 1, 7)
        for col in (1, 3, 5, 7):
            core_grid.setColumnStretch(col, 1)
        form.addRow("规模与批次:", core_grid_widget)

        opt_sched_row = QHBoxLayout()
        opt_sched_row.setSpacing(4)
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
        self.lr0_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.lrf_spin = _make_double_spin(
            0.001,
            1.0,
            0.01,
            decimals=3,
            step=0.01,
            tooltip="末期学习率衰减系数：最终学习率 = LR0 × LRF",
        )
        self.lrf_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.warmup_spin = _make_double_spin(
            0.0,
            50.0,
            3.0,
            decimals=1,
            step=0.5,
            tooltip="学习率预热轮数 (warmup_epochs)",
        )
        self.warmup_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        opt_sched_row.addWidget(self.optimizer_combo, 1)
        opt_sched_row.addSpacing(6)
        opt_sched_row.addWidget(self.lr0_label)
        opt_sched_row.addWidget(self.lr0_spin)
        opt_sched_row.addSpacing(6)
        opt_sched_row.addWidget(QLabel("LRF:"))
        opt_sched_row.addWidget(self.lrf_spin)
        opt_sched_row.addSpacing(6)
        opt_sched_row.addWidget(QLabel("Warmup:"))
        opt_sched_row.addWidget(self.warmup_spin)
        form.addRow("优化器与学习率:", opt_sched_row)

        # 训练与复现选项：统一 2 行 × 4 列网格对齐，下方按需展开断点续训生效范围提示
        opts_widget = QWidget()
        opts_vbox = QVBoxLayout(opts_widget)
        opts_vbox.setContentsMargins(0, 0, 0, 0)
        opts_vbox.setSpacing(4)
        opts_grid = QGridLayout()
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
        opts_vbox.addLayout(opts_grid)

        self.resume_hint_label = QLabel(
            "断点续训模式：优先加载「自定义权重路径」或「保存根目录/实验名称/weights/last.pt」，"
            "已置灰的超参数与增强项将沿用断点原配置，仅运行与验证类参数可覆盖。"
        )
        self.resume_hint_label.setWordWrap(True)
        self.resume_hint_label.setStyleSheet(
            "color: #0969da; background: #eff6ff; border: 1px solid #bfdbfe; "
            "border-radius: 4px; padding: 4px 8px; font-size: 12px;"
        )
        self.resume_hint_label.setVisible(False)
        opts_vbox.addWidget(self.resume_hint_label)
        form.addRow("训练与复现:", opts_widget)

        default_proj = str((self.workspace_root / "runs" / "detect").resolve())
        self.project_picker = PathPickerRow(
            placeholder="训练结果保存根目录", mode="dir", default_path=default_proj
        )
        form.addRow("保存根目录:", self.project_picker)

        self.run_name_edit = QLineEdit("yolo26_train")
        form.addRow("实验名称:", self.run_name_edit)
        align_form_labels([form])

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
        configure_form_layout(extra_form, horizontal_spacing=8, vertical_spacing=4, margins=(0, 2, 0, 0))
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
        self.send_eval_btn.setToolTip(
            "训练产出 best.pt 后已自动同步到「6. 验证、推理与导出」与「3. LLM 标注清洗」；点击再次同步并切换过去"
        )
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
        c_time, self.val_elapsed = make_metric_card("训练时长", "-")
        c2, self.val_box_loss = make_metric_card("Box Loss", "-")
        c3, self.val_cls_loss = make_metric_card("Cls Loss", "-")
        c4, self.val_precision = make_metric_card("Precision", "-")
        c5, self.val_recall = make_metric_card("Recall", "-")
        c6, self.val_map50 = make_metric_card("mAP50", "-")
        c7, self.val_map50_95 = make_metric_card("mAP50-95", "-")

        metrics_grid.addWidget(c1, 0, 0)
        metrics_grid.addWidget(c_time, 0, 1)
        metrics_grid.addWidget(c2, 1, 0)
        metrics_grid.addWidget(c3, 1, 1)
        metrics_grid.addWidget(c4, 2, 0)
        metrics_grid.addWidget(c5, 2, 1)
        metrics_grid.addWidget(c6, 3, 0)
        metrics_grid.addWidget(c7, 3, 1)
        self.best_eval_card = self._build_best_eval_card()
        metrics_grid.addWidget(self.best_eval_card, 4, 0, 1, 2)
        metrics_grid.setColumnStretch(0, 1)
        metrics_grid.setColumnStretch(1, 1)
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
        self.device_ctrl.device_changed.connect(self._on_device_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.import_json_btn.clicked.connect(self._on_import_json_dialog)
        self.export_json_btn.clicked.connect(self._on_export_json_clicked)
        self.reset_defaults_btn.clicked.connect(self._on_reset_defaults_clicked)
        self.toggle_aug_btn.clicked.connect(self._on_toggle_aug_panel)
        self.custom_pt_btn.clicked.connect(self._on_pick_custom_pt)
        self.custom_model_edit.textChanged.connect(self._on_control_state_inputs_changed)
        self.optimizer_combo.currentIndexChanged.connect(self._on_control_state_inputs_changed)
        self.mosaic_spin.valueChanged.connect(self._on_control_state_inputs_changed)
        self.resume_chk.toggled.connect(self._on_control_state_inputs_changed)
        self.val_chk.toggled.connect(self._on_control_state_inputs_changed)
        self.batch_spin.valueChanged.connect(self._on_batch_value_changed)
        self.nbs_spin.valueChanged.connect(self._refresh_accum_hint)
        self.extra_args_edit.textChanged.connect(self._on_extra_args_changed)
        self.amp_chk.toggled.connect(self._on_amp_toggled)
        self._bind_preset_dirty_watchers()
        self._bind_settings_dirty_watchers()

        # 记录受断点续训 (resume=True) 忽略的控件及其默认提示文案，切回正常训练时完整恢复
        self._resume_ignored_widgets: tuple[tuple[QWidget, str], ...] = tuple(
            (w, w.toolTip())
            for w in (
                self.preset_combo,
                self.import_json_btn,
                self.epochs_spin,
                self.nbs_spin,
                self.optimizer_combo,
                self.lrf_spin,
                self.warmup_spin,
                self.cos_lr_chk,
                self.seed_spin,
                self.deterministic_chk,
                self.mosaic_spin,
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
            )
        )

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_run_btn.clicked.connect(lambda: open_local_path(self._last_save_dir))
        self.send_eval_btn.clicked.connect(self._on_send_to_eval)

        # 初始同步防呆联动、梯度累积换算与增强摘要
        self._sync_control_states()
        self._refresh_accum_hint()
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
        for edit in (self.custom_model_edit, self.run_name_edit):
            edit.textChanged.connect(self._mark_settings_dirty)
        self.device_ctrl.device_changed.connect(self._mark_settings_dirty)
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

    def _on_device_changed(self, _device: str) -> None:
        self._sync_control_states()

    def _on_control_state_inputs_changed(self, *_args: object) -> None:
        self._sync_control_states()

    def _on_batch_value_changed(self, value: int) -> None:
        """禁止 `Batch` 停留在无效值 `0`，并实时刷新梯度累积换算提示。"""
        if value == 0:
            self.batch_spin.setValue(1)
            return
        self._refresh_accum_hint()

    def _refresh_accum_hint(self, *_args: object) -> None:
        """按当前 `Batch` 与 `NBS` 实时换算梯度累积步数并刷新只读提示标签。"""
        batch_val = self.batch_spin.value()
        nbs_val = self.nbs_spin.value()
        if batch_val <= 0:
            self.accum_hint_label.setText("自动 (AutoBatch)")
            self.accum_hint_label.setToolTip(
                "Batch=-1 时由 Ultralytics 先自动探测最大可用 Batch，再按 max(round(NBS / Batch), 1) 计算梯度累积步数"
            )
            return
        steps = max(round(nbs_val / batch_val), 1)
        self.accum_hint_label.setText(f"每 {steps} 步更新")
        self.accum_hint_label.setToolTip(
            f"当前 NBS={nbs_val}、Batch={batch_val}，Ultralytics 每累积 {steps} 个 Batch 执行一次权重更新 "
            "（公式：max(round(NBS / Batch), 1)）"
        )

    def _sync_control_states(self) -> None:
        """统一维护防呆联动禁用态：断点续训置灰忽略项、auto 置灰 LR0、自定义权重置灰官方模型、
        Mosaic=0 置灰 CloseMosaic、关闭验证置灰 Patience、CPU 置灰 AMP 并禁用 Batch=-1 自动探测。
        """
        is_resume = self.resume_chk.isChecked()
        probing = self.device_info.probing
        using_gpu = self.device_ctrl.is_using_gpu()
        resume_tip = (
            "断点续训 (resume) 模式下该参数将沿用 last.pt 原配置，不会覆盖；取消勾选断点续训后可编辑"
        )

        self.resume_hint_label.setVisible(is_resume)

        # 1) 受断点续训忽略的超参数与数据增强控件统一置灰，避免用户误改却不生效
        for widget, default_tip in self._resume_ignored_widgets:
            widget.setEnabled(not is_resume)
            widget.setToolTip(resume_tip if is_resume else default_tip)

        # 2) 优化器为 auto 或断点续训时置灰 LR0
        is_auto_opt = str(self.optimizer_combo.currentData() or "auto").lower() == "auto"
        self.lr0_spin.setEnabled((not is_auto_opt) and (not is_resume))
        self.lr0_label.setText("LR0(自动):" if is_auto_opt else "LR0:")
        if is_resume:
            self.lr0_spin.setToolTip(resume_tip)
        elif is_auto_opt:
            self.lr0_spin.setToolTip(
                "当前优化器为 auto，Ultralytics 将按类别数与步数自动计算并覆盖 LR0；切换到 AdamW / SGD 后可手动编辑"
            )
        else:
            self.lr0_spin.setToolTip("初始学习率：当前优化器已显式指定，训练时将严格按此值生效")

        # 3) 指定自定义权重或开启断点续训时置灰官方预训练模型下拉框
        has_custom_model = bool(self.custom_model_edit.text().strip())
        self.model_combo.setEnabled((not has_custom_model) and (not is_resume))
        if is_resume:
            self.model_combo.setToolTip(
                "已启用断点续训，将优先加载「自定义权重路径」或「保存根目录/实验名称/weights/last.pt」，不使用官方预训练模型"
            )
        elif has_custom_model:
            self.model_combo.setToolTip(
                "已指定自定义权重路径，训练时将优先加载自定义 .pt；清空自定义路径后恢复选用官方模型"
            )
        else:
            self.model_combo.setToolTip("选择官方 YOLO26 目标检测预训练模型")

        # 4) CloseMosaic：续训时 Ultralytics 允许覆盖，正常训练仅在 Mosaic > 0 时生效
        if is_resume:
            self.close_mosaic_spin.setEnabled(True)
            self.close_mosaic_spin.setToolTip(
                "最后 N 轮关闭 Mosaic 增强 (close_mosaic)；断点续训时允许覆盖此参数"
            )
        else:
            mosaic_enabled = self.mosaic_spin.value() > 0.0
            self.close_mosaic_spin.setEnabled(mosaic_enabled)
            self.close_mosaic_spin.setToolTip(
                "最后 N 轮关闭 Mosaic 增强 (close_mosaic)"
                if mosaic_enabled
                else "当前 Mosaic 概率为 0，已彻底关闭拼接增强，CloseMosaic 不再生效"
            )

        # 5) 关闭训练期验证 (val=False) 时早停 Patience 不会触发
        val_enabled = self.val_chk.isChecked()
        self.patience_spin.setEnabled(val_enabled)
        self.patience_spin.setToolTip(
            "早停容忍轮数：验证集指标连续 N 轮无提升则提前停训；设为 0 或等于 Epochs 可禁用早停"
            if val_enabled
            else "已关闭训练期验证 (val=False)，训练中途不评估验证集，早停 Patience 不会生效"
        )

        # 6) CPU 模式不支持 AutoBatch (-1)，限制 Batch >= 1；切回 GPU 时恢复 -1 可选及原偏好
        if not probing and not using_gpu:
            if not self._batch_cpu_clamped:
                prev_restoring = self._restoring_state
                self._restoring_state = True
                try:
                    was_auto = self.batch_spin.value() <= 0
                    self._batch_was_auto_before_cpu = was_auto
                    self.batch_spin.setMinimum(1)
                    self.batch_spin.setSpecialValueText("")
                    if was_auto:
                        self.batch_spin.setValue(4)
                finally:
                    self._restoring_state = prev_restoring
                self._batch_cpu_clamped = True
            self.batch_spin.setToolTip(
                "CPU 模式不支持 AutoBatch (-1)，请指定正整数批大小（建议 2~8）；启用 GPU 加速后恢复 -1 自动探测"
            )
        else:
            if self._batch_cpu_clamped:
                prev_restoring = self._restoring_state
                self._restoring_state = True
                try:
                    self.batch_spin.setMinimum(-1)
                    self.batch_spin.setSpecialValueText("自动")
                    if self._batch_was_auto_before_cpu:
                        self.batch_spin.setValue(-1)
                        self._batch_was_auto_before_cpu = False
                finally:
                    self._restoring_state = prev_restoring
                self._batch_cpu_clamped = False
            self.batch_spin.setToolTip(
                "设为 -1（显示为“自动”）可让 Ultralytics 自动探测最大可用 Batch Size"
            )

        # 7) CPU 模式或探测期间联动置灰混合精度 AMP，重新启用 GPU 时恢复用户原有偏好；续训时同样置灰
        if probing:
            self.amp_chk.setEnabled(False)
            self.amp_chk.setToolTip("正在检测计算设备，探测完成后自动更新混合精度 AMP 可用状态")
        elif not using_gpu:
            if not self._amp_disabled_for_cpu:
                self._amp_saved_before_cpu = self.amp_chk.isChecked()
                self._amp_disabled_for_cpu = True
                prev_restoring = self._restoring_state
                self._restoring_state = True
                try:
                    self.amp_chk.setChecked(False)
                finally:
                    self._restoring_state = prev_restoring
            self.amp_chk.setEnabled(False)
            self.amp_chk.setToolTip("当前为 CPU 模式，不支持 CUDA 混合精度 AMP；在顶部启用 GPU 加速后可开启")
        else:
            if self._amp_disabled_for_cpu:
                if self._amp_saved_before_cpu is not None:
                    prev_restoring = self._restoring_state
                    self._restoring_state = True
                    try:
                        self.amp_chk.setChecked(self._amp_saved_before_cpu)
                    finally:
                        self._restoring_state = prev_restoring
                self._amp_saved_before_cpu = None
                self._amp_disabled_for_cpu = False
            self.amp_chk.setEnabled(not is_resume)
            self.amp_chk.setToolTip(
                resume_tip
                if is_resume
                else "开启自动混合精度 (FP16 AMP) 可节省显存并提速；如需严格对标 FP32 数值可关闭"
            )

        self._refresh_accum_hint()
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

        if "device" in known and known["device"] is not None:
            self.device_ctrl.set_device(str(known["device"]), user_action=not self._restoring_state)
            applied.append("device")

        _set_int(self.epochs_spin, "epochs")
        if "imgsz" in known and known["imgsz"] is not None:
            self.imgsz_combo.setCurrentText(str(int(known["imgsz"])))  # type: ignore[call-overload]
            applied.append("imgsz")
        if "batch" in known and known["batch"] is not None:
            raw_batch = int(known["batch"])  # type: ignore[call-overload]
            if raw_batch == 0:
                raw_batch = 1
            if raw_batch < 0 and self.batch_spin.minimum() > raw_batch:
                self.batch_spin.setMinimum(-1)
                self.batch_spin.setSpecialValueText("自动")
                self._batch_cpu_clamped = False
            self.batch_spin.setValue(raw_batch)
            applied.append("batch")
        _set_int(self.nbs_spin, "nbs")
        _set_int(self.patience_spin, "patience")
        _set_int(self.workers_spin, "workers")
        _set_int(self.seed_spin, "seed")

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
            self._batch_cpu_clamped = False
            self._batch_was_auto_before_cpu = False
            self.batch_spin.setMinimum(-1)
            self.batch_spin.setSpecialValueText("自动")
            self.batch_spin.setValue(self._auto_batch_value)
            self.patience_spin.setValue(50)
            self.workers_spin.setValue(4)
            self.seed_spin.setValue(0)
            self.device_ctrl.reset_to_recommended()
            self._amp_user_overridden = False
            self._amp_saved_before_cpu = None
            self._amp_disabled_for_cpu = False
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
        saved_amp = (
            self._amp_saved_before_cpu
            if (self._amp_disabled_for_cpu and self._amp_saved_before_cpu is not None)
            else self.amp_chk.isChecked()
        )
        saved_batch = (
            -1
            if (self._batch_cpu_clamped and self._batch_was_auto_before_cpu)
            else self.batch_spin.value()
        )
        state: dict[str, object] = {
            "version": 1,
            "preset": str(self.preset_combo.currentData() or _CUSTOM_PRESET_KEY),
            "model_file": str(self.model_combo.currentData() or "yolo26s.pt"),
            "custom_model": self.custom_model_edit.text().strip(),
            "epochs": self.epochs_spin.value(),
            "imgsz": self.imgsz_combo.currentText().strip() or "640",
            "batch": saved_batch,
            "batch_is_auto": saved_batch == self._auto_batch_value,
            "nbs": self.nbs_spin.value(),
            "optimizer": str(self.optimizer_combo.currentData() or "auto"),
            "lr0": self.lr0_spin.value(),
            "lrf": self.lrf_spin.value(),
            "warmup_epochs": self.warmup_spin.value(),
            "patience": self.patience_spin.value(),
            "workers": self.workers_spin.value(),
            "seed": self.seed_spin.value(),
            "custom_device": self.device_ctrl.device() if self.device_ctrl.is_user_modified() else "",
            "amp": saved_amp,
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
            self._amp_disabled_for_cpu = False
            self._amp_saved_before_cpu = None
            self._batch_cpu_clamped = False
            self._batch_was_auto_before_cpu = False
            self.batch_spin.setMinimum(-1)
            self.batch_spin.setSpecialValueText("自动")

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
                self.device_ctrl.set_device(custom_dev, user_action=True)

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
        self._refresh_accum_hint()
        self._refresh_extra_args_status()

    def apply_device_info(self, info: DeviceInfo) -> None:
        """异步硬件探测完成后刷新顶部设备控件与尚未被用户手动改动的默认训练参数。"""
        was_probing = self.device_info.probing
        self.device_info = info

        self._restoring_state = True
        try:
            self.device_ctrl.apply_device_info(info)

            target_batch = 16 if info.cuda_available else 4
            if self.batch_spin.value() == self._auto_batch_value:
                self.batch_spin.setValue(target_batch)
                self._auto_batch_value = target_batch

            if (
                was_probing
                and not self._amp_user_overridden
                and (self.worker is None or not self.worker.isRunning())
            ):
                self.amp_chk.setChecked(self.device_ctrl.is_using_gpu())
        finally:
            self._restoring_state = False

        self._sync_control_states()
        self._refresh_accum_hint()

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

        batch_val = self.batch_spin.value()
        if batch_val == 0:
            raise ValueError("Batch 不能为 0，请设为 -1（GPU 自动探测）或正整数。")

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
            device=self.device_ctrl.device(),
            lr0=self.lr0_spin.value(),
            patience=self.patience_spin.value(),
            workers=self.workers_spin.value(),
            amp=self.amp_chk.isChecked() and self.device_ctrl.is_using_gpu(),
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

    def _disconnect_worker(self, worker: TrainWorker | None) -> None:
        """断开已完成训练线程的信号连接，防止历史实例的残留队列事件污染新一次训练的指标与日志。"""
        if worker is None:
            return
        for sig in (worker.epoch_updated, worker.log_line, worker.finished_result, worker.failed):
            try:
                sig.disconnect()
            except TypeError:
                pass

    def _build_best_eval_card(self) -> QWidget:
        """构建紧凑的 `best.pt` 最终评估摘要卡片，独立于上方最后一轮实时指标卡片展示。"""
        card = QWidget()
        card.setObjectName("bestEvalCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setStyleSheet(_BEST_EVAL_CARD_IDLE_QSS)
        card.setToolTip(
            "训练结束后 Ultralytics 对最佳权重 best.pt 单独执行的完整验证指标；独立展示，不覆盖上方最后一轮指标"
        )

        outer = QVBoxLayout(card)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        self.best_eval_title_label = QLabel("best.pt 最终评估")
        self.best_eval_title_label.setStyleSheet("color: #57606a; font-weight: 600; font-size: 12px;")
        self.best_eval_status_label = QLabel("训练结束后更新（不覆盖上方最后一轮）")
        self.best_eval_status_label.setStyleSheet("color: #8c959f; font-size: 11px;")
        header.addWidget(self.best_eval_title_label)
        header.addStretch(1)
        header.addWidget(self.best_eval_status_label)
        outer.addLayout(header)

        values_row = QHBoxLayout()
        values_row.setContentsMargins(0, 0, 0, 0)
        values_row.setSpacing(10)

        def _make_pair(caption: str, tip: str) -> QLabel:
            pair = QHBoxLayout()
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(4)
            cap_lbl = QLabel(f"{caption}:")
            cap_lbl.setStyleSheet("color: #666666; font-weight: 500; font-size: 12px;")
            cap_lbl.setToolTip(tip)
            val_lbl = QLabel("-")
            val_lbl.setStyleSheet("font-weight: 700; font-size: 13px;")
            val_lbl.setToolTip(tip)
            pair.addWidget(cap_lbl)
            pair.addWidget(val_lbl)
            pair.addStretch(1)
            values_row.addLayout(pair, 1)
            return val_lbl

        self.val_best_precision = _make_pair("P", "best.pt 最终评估精确率 (Precision)")
        self.val_best_recall = _make_pair("R", "best.pt 最终评估召回率 (Recall)")
        self.val_best_map50 = _make_pair("mAP50", "best.pt 最终评估 mAP@0.50")
        self.val_best_map50_95 = _make_pair("mAP50-95", "best.pt 最终评估 mAP@0.50:0.95")
        outer.addLayout(values_row)
        return card

    def _reset_best_eval_metrics(self) -> None:
        """清空 `best.pt` 最终评估摘要卡片数值并恢复待评估状态样式。"""
        self._has_best_eval_metrics = False
        self.best_eval_card.setStyleSheet(_BEST_EVAL_CARD_IDLE_QSS)
        self.best_eval_title_label.setStyleSheet("color: #57606a; font-weight: 600; font-size: 12px;")
        self.best_eval_status_label.setText("训练结束后更新（不覆盖上方最后一轮）")
        self.best_eval_status_label.setStyleSheet("color: #8c959f; font-size: 11px;")
        for lbl in (
            self.val_best_precision,
            self.val_best_recall,
            self.val_best_map50,
            self.val_best_map50_95,
        ):
            lbl.setText("-")

    def _set_best_eval_metrics(
        self, *, precision: float, recall: float, map50: float, map50_95: float
    ) -> None:
        """写入 `best.pt` 最终评估四项验证指标并切换高亮完成样式，不影响上方最后一轮卡片。"""
        self._has_best_eval_metrics = True
        self.best_eval_card.setStyleSheet(_BEST_EVAL_CARD_READY_QSS)
        self.best_eval_title_label.setStyleSheet("color: #15803d; font-weight: 700; font-size: 12px;")
        self.best_eval_status_label.setText("独立于上方最后一轮指标")
        self.best_eval_status_label.setStyleSheet("color: #166534; font-size: 11px;")
        self.val_best_precision.setText(f"{precision:.4f}")
        self.val_best_recall.setText(f"{recall:.4f}")
        self.val_best_map50.setText(f"{map50:.4f}")
        self.val_best_map50_95.setText(f"{map50_95:.4f}")

    def _reset_live_metrics(self, *, total_epochs: int = 0, initial_elapsed: str = "-") -> None:
        """立即重置右侧全部实时训练指标卡片、`best.pt` 最终评估摘要与进度条，清除上次训练残留。

        参数说明：
            `total_epochs`：本次训练计划总轮数；大于 0 时将 `Epoch` 重置为 `0 / <total_epochs>`，
                为 0（如空闲初始态或断点续训尚待读取断点内总轮数）时重置为 `0 / 0`。
            `initial_elapsed`：`训练时长` 卡片的重置显示文本（新训练启动时传入 `"00:00:00"`）。
        """
        self.progress_bar.setValue(0)
        self.val_epoch.setText(f"0 / {total_epochs}" if total_epochs > 0 else "0 / 0")
        self.val_elapsed.setText(initial_elapsed)
        self.val_box_loss.setText("-")
        self.val_cls_loss.setText("-")
        self.val_precision.setText("-")
        self.val_recall.setText("-")
        self.val_map50.setText("-")
        self.val_map50_95.setText("-")
        self._reset_best_eval_metrics()

    def _start_elapsed_timer(self) -> None:
        """以当前 `time.monotonic()` 单调时钟为起点启动训练时长刷新定时器，并立即归零显示 `00:00:00`。"""
        self._train_started_at = time.monotonic()
        self.val_elapsed.setText("00:00:00")
        self._elapsed_timer.start()

    def _refresh_elapsed_time(self) -> str:
        """按单调时钟计算当前已用训练时长并刷新卡片文本；尚未启动计时时原样返回当前显示文本。"""
        if self._train_started_at is None:
            return self.val_elapsed.text()
        elapsed_text = format_elapsed_duration(time.monotonic() - self._train_started_at)
        if self.val_elapsed.text() != elapsed_text:
            self.val_elapsed.setText(elapsed_text)
        return elapsed_text

    def _stop_elapsed_timer(self) -> str | None:
        """停止训练时长定时器并在停表前完成末次刷新，返回定格的 `HH:MM:SS` 文案（未计时时返回 `None`）。"""
        self._elapsed_timer.stop()
        if self._train_started_at is None:
            return None
        return self._refresh_elapsed_time()

    def closeEvent(self, event: QCloseEvent) -> None:
        """关闭标签页时停止实时计时定时器，避免关闭后定时器继续触发。"""
        self._elapsed_timer.stop()
        super().closeEvent(event)

    def _resolve_initial_total_epochs(self, cfg: TrainConfig) -> int:
        """计算新训练启动瞬间的 `Epoch` 卡片分母；断点续训时由断点内配置决定，启动瞬间返回 `0`。"""
        if cfg.resume:
            return 0
        raw_epochs = build_train_kwargs(cfg).get("epochs", cfg.epochs)
        if isinstance(raw_epochs, bool):
            return cfg.epochs if cfg.epochs > 0 else 0
        try:
            parsed = int(raw_epochs)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            parsed = cfg.epochs
        return parsed if parsed > 0 else 0

    def _on_start(self) -> None:
        """校验表单与附加超参数后锁定运行按钮、立即重置全部实时指标与计时器、清空日志并启动 `TrainWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())

        try:
            cfg = self.build_train_config()
        except ValueError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "YOLO26 模型训练"):
            return

        self._settings_dirty = True
        self._last_save_dir = cfg.project_dir / cfg.run_name
        self._last_best_pt = ""
        self._disconnect_worker(self.worker)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.send_eval_btn.setEnabled(False)
        self._reset_live_metrics(
            total_epochs=self._resolve_initial_total_epochs(cfg),
            initial_elapsed="00:00:00",
        )
        self._start_elapsed_timer()
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

    def _on_send_to_eval(self) -> None:
        """将最近训练产出的 `best.pt` 与当前 `data.yaml` 再次同步至下游页并请求主窗口切换至标签页 6。

        说明：
            无有效 `_last_best_pt` 时静默返回；仅用户主动点击发送按钮时发射 `request_eval_tab`，
            训练完成自动同步（`_on_finished`）不会强制切页，以免打断用户查看训练日志与最终评估指标。
        """
        if not self._last_best_pt:
            return
        self.send_best_weights.emit(self._last_best_pt, self.yaml_picker.text())
        self.request_eval_tab.emit()

    def _on_epoch_updated(self, m: EpochMetrics) -> None:
        """更新右侧指标卡片：常规轮次更新实时/最后一轮卡片，`final_eval` 仅更新下方 `best.pt` 最终评估栏。"""
        self._refresh_elapsed_time()
        if m.is_final_eval:
            self._set_best_eval_metrics(
                precision=m.precision,
                recall=m.recall,
                map50=m.map50,
                map50_95=m.map50_95,
            )
            return

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
        """恢复界面状态、停表定格训练时长、兜底补齐 `best.pt` 评估指标并通知下游标签页。"""
        elapsed_text = self._stop_elapsed_timer()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._last_save_dir = res.save_dir
        if not res.stopped_early:
            self.progress_bar.setValue(100)

        # 多卡 DDP 或未走主进程 `final_eval` 回调时，兜底从 `TrainResult.metrics` 填充 `best.pt` 最终评估指标
        if (
            not self._has_best_eval_metrics
            and res.best_weights is not None
            and any(k in res.metrics for k in _BEST_EVAL_METRIC_KEYS)
        ):
            self._set_best_eval_metrics(
                precision=float(res.metrics.get("metrics/precision(B)", 0.0)),
                recall=float(res.metrics.get("metrics/recall(B)", 0.0)),
                map50=float(res.metrics.get("metrics/mAP50(B)", 0.0)),
                map50_95=float(res.metrics.get("metrics/mAP50-95(B)", 0.0)),
            )

        status_text = "训练已提前停止" if res.stopped_early else "训练完成"
        self.log_box.appendPlainText(f"\n=== {status_text} ===")
        if elapsed_text is not None:
            self.log_box.appendPlainText(f"训练时长: {elapsed_text}")
        if self._has_best_eval_metrics:
            self.log_box.appendPlainText(
                "best.pt 最终评估: "
                f"P={self.val_best_precision.text()}, "
                f"R={self.val_best_recall.text()}, "
                f"mAP50={self.val_best_map50.text()}, "
                f"mAP50-95={self.val_best_map50_95.text()}"
            )
        self.log_box.appendPlainText(f"结果目录: {res.save_dir}")
        if res.best_weights:
            self._last_best_pt = str(res.best_weights)
            self.send_eval_btn.setEnabled(True)
            self.log_box.appendPlainText(f"最佳权重: {res.best_weights}")
            self.send_best_weights.emit(self._last_best_pt, self.yaml_picker.text())
        if res.last_weights:
            self.log_box.appendPlainText(f"最新权重: {res.last_weights}")

    def _on_failed(self, err: str) -> None:
        """训练异常退出时停表定格训练时长、恢复按钮状态并弹窗提示错误。"""
        elapsed_text = self._stop_elapsed_timer()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if elapsed_text is not None:
            self.log_box.appendPlainText(f"\n训练时长: {elapsed_text}")
            self.log_box.appendPlainText(f"[错误] {err}")
        else:
            self.log_box.appendPlainText(f"\n[错误] {err}")
        QMessageBox.critical(self, "训练错误", err)
