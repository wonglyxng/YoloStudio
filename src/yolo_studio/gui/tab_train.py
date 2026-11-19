"""标签页 5：YOLO26 模型训练界面。

职责：
    展示本机计算设备（CUDA GPU/CPU）状态，配置 `data.yaml`、预训练权重、场景超参数预设、
    优化器、数据增强与任意附加参数，并在后台启动训练后实时刷新 Epoch 指标卡片与 Ultralytics 日志流
    （含提交阶段与 Trainer 初始化后的实际生效超参数明细）。

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
    YOLO26_DETECT_MODELS,
    YOLO26_OPTIMIZERS,
    DeviceInfo,
    EpochMetrics,
    TrainConfig,
    TrainResult,
    detect_compute_device,
    parse_extra_train_args,
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


def _make_double_spin(
    minimum: float,
    maximum: float,
    value: float,
    *,
    decimals: int = 2,
    step: float = 0.05,
    tooltip: str = "",
) -> QDoubleSpinBox:
    """构建统一样式与精度的浮点超参数微调框。"""
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


class TrainTab(QWidget):
    """YOLO26 模型训练功能标签页（标签页 5）。

    内部状态：
        持有硬件探测结果 `device_info`、跨页计算设备互斥协调器 `gpu_guard`、训练工作线程 `worker`、
        预设批量写入重入锁 `_applying_preset`、最近实验目录 `_last_save_dir` 及最近最佳权重路径 `_last_best_pt`。
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
        self._last_save_dir: Path = workspace_root / "runs" / "detect"
        self._last_best_pt: str = ""
        self._auto_device_text: str = self.device_info.recommended_device
        self._auto_batch_value: int = 16 if self.device_info.cuda_available else 4
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

        self.preset_combo = QComboBox()
        for preset_key, preset_label in TRAIN_PRESET_ITEMS:
            self.preset_combo.addItem(preset_label, preset_key)
        self.preset_combo.addItem("自定义 (当前手动微调)", _CUSTOM_PRESET_KEY)
        self.preset_combo.setToolTip(
            "一键切换优化器与数据增强超参数组合：如「游戏画面 / UI 弱增强」会自动切换为 AdamW (LR0=0.001, NBS=16) 并关闭 Mosaic 与畸变增强"
        )
        form.addRow("超参数预设:", self.preset_combo)

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

        self.nbs_spin = QSpinBox()
        self.nbs_spin.setRange(1, 1024)
        self.nbs_spin.setValue(64)
        self.nbs_spin.setToolTip(
            "名义批大小 (Nominal Batch Size)：梯度累积步数 ≈ round(NBS / Batch)；小数据集小 Batch 下调小至 16 可加快参数更新频率"
        )

        grid_row1.addWidget(QLabel("Epochs:"))
        grid_row1.addWidget(self.epochs_spin)
        grid_row1.addWidget(QLabel("ImgSz:"))
        grid_row1.addWidget(self.imgsz_combo)
        grid_row1.addWidget(QLabel("Batch(-1=自动):"))
        grid_row1.addWidget(self.batch_spin)
        grid_row1.addWidget(QLabel("NBS:"))
        grid_row1.addWidget(self.nbs_spin)
        form.addRow("规模与批次:", grid_row1)

        opt_sched_row = QHBoxLayout()
        self.optimizer_combo = QComboBox()
        for opt_name, opt_desc in YOLO26_OPTIMIZERS:
            self.optimizer_combo.addItem(opt_desc, opt_name)
        self.optimizer_combo.setToolTip(
            "注意：选 auto 时 Ultralytics 会按类别数与步数自动选定优化器并覆盖 LR0；如需严格固定 LR0 请显式选择 AdamW 或 SGD"
        )

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
        opt_sched_row.addWidget(QLabel("LR0:"))
        opt_sched_row.addWidget(self.lr0_spin)
        opt_sched_row.addWidget(QLabel("LRF:"))
        opt_sched_row.addWidget(self.lrf_spin)
        opt_sched_row.addWidget(QLabel("Warmup:"))
        opt_sched_row.addWidget(self.warmup_spin)
        form.addRow("优化器与学习率:", opt_sched_row)

        hw_row = QHBoxLayout()
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 5000)
        self.patience_spin.setValue(50)
        self.patience_spin.setToolTip("早停容忍轮数：验证集指标连续 N 轮无提升则提前停训；设为 0 或等于 Epochs 可禁用早停")

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(4)
        self.workers_spin.setToolTip("DataLoader 工作进程数：Windows 下如遇多进程内存开销问题可设为 0")

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(0)
        self.seed_spin.setToolTip("随机种子 (seed)，配合确定性训练提升实验可复现性")

        self.device_edit = QLineEdit(self.device_info.recommended_device)
        self.device_edit.setFixedWidth(70)

        hw_row.addWidget(QLabel("Patience:"))
        hw_row.addWidget(self.patience_spin)
        hw_row.addWidget(QLabel("Workers:"))
        hw_row.addWidget(self.workers_spin)
        hw_row.addWidget(QLabel("Seed:"))
        hw_row.addWidget(self.seed_spin)
        hw_row.addWidget(QLabel("Device:"))
        hw_row.addWidget(self.device_edit)
        form.addRow("早停与硬件:", hw_row)

        opt_row1 = QHBoxLayout()
        self.amp_chk = QCheckBox("混合精度 AMP")
        self.amp_chk.setChecked(self.device_info.cuda_available)
        self.cos_lr_chk = QCheckBox("余弦退火 (cos_lr)")
        self.cache_chk = QCheckBox("内存缓存 (cache)")
        self.resume_chk = QCheckBox("断点续训 (resume)")
        self.resume_chk.setToolTip(
            "从上次中断的训练继续：优先使用「自定义权重路径」中选择的 last.pt，"
            "否则使用「保存根目录/实验名称/weights/last.pt」；新训练不会覆盖已有实验目录，会自动递增编号"
        )
        opt_row1.addWidget(self.amp_chk)
        opt_row1.addWidget(self.cos_lr_chk)
        opt_row1.addWidget(self.cache_chk)
        opt_row1.addWidget(self.resume_chk)
        form.addRow("训练选项:", opt_row1)

        opt_row2 = QHBoxLayout()
        self.deterministic_chk = QCheckBox("确定性算子 (deterministic)")
        self.deterministic_chk.setChecked(True)
        self.plots_chk = QCheckBox("保存评估图表 (plots)")
        self.plots_chk.setChecked(True)
        self.val_chk = QCheckBox("训练期验证 (val)")
        self.val_chk.setChecked(True)
        opt_row2.addWidget(self.deterministic_chk)
        opt_row2.addWidget(self.plots_chk)
        opt_row2.addWidget(self.val_chk)
        opt_row2.addStretch(1)
        form.addRow("复现与输出:", opt_row2)

        default_proj = str((self.workspace_root / "runs" / "detect").resolve())
        self.project_picker = PathPickerRow(
            placeholder="训练结果保存根目录", mode="dir", default_path=default_proj
        )
        form.addRow("保存根目录:", self.project_picker)

        self.run_name_edit = QLineEdit("yolo26_train")
        form.addRow("实验名称:", self.run_name_edit)

        form_stack.addWidget(cfg_group)

        # 3. 数据增强与任意附加超参数
        aug_group = QGroupBox("3. 数据增强与附加超参数")
        aug_form = QFormLayout(aug_group)
        aug_form.setSpacing(6)

        aug_row1 = QHBoxLayout()
        self.mosaic_spin = _make_double_spin(0.0, 1.0, 1.0, tooltip="Mosaic 四图拼接概率 (固定视角/UI 检测建议 0.0)")
        self.close_mosaic_spin = QSpinBox()
        self.close_mosaic_spin.setRange(0, 5000)
        self.close_mosaic_spin.setValue(10)
        self.close_mosaic_spin.setToolTip("最后 N 轮关闭 Mosaic 增强 (close_mosaic)")
        self.mixup_spin = _make_double_spin(0.0, 1.0, 0.0, tooltip="MixUp 图像混合概率")
        self.fliplr_spin = _make_double_spin(0.0, 1.0, 0.5, tooltip="水平左右翻转概率 (fliplr)")
        self.flipud_spin = _make_double_spin(0.0, 1.0, 0.0, tooltip="垂直上下翻转概率 (flipud)")
        aug_row1.addWidget(QLabel("Mosaic:"))
        aug_row1.addWidget(self.mosaic_spin)
        aug_row1.addWidget(QLabel("CloseMosaic:"))
        aug_row1.addWidget(self.close_mosaic_spin)
        aug_row1.addWidget(QLabel("MixUp:"))
        aug_row1.addWidget(self.mixup_spin)
        aug_row1.addWidget(QLabel("FlipLR:"))
        aug_row1.addWidget(self.fliplr_spin)
        aug_row1.addWidget(QLabel("FlipUD:"))
        aug_row1.addWidget(self.flipud_spin)
        aug_form.addRow("拼接与翻转:", aug_row1)

        aug_row2 = QHBoxLayout()
        self.translate_spin = _make_double_spin(0.0, 0.9, 0.1, step=0.02, tooltip="图像平移比例 (translate)")
        self.scale_spin = _make_double_spin(0.0, 0.95, 0.5, step=0.05, tooltip="图像缩放增益比例 (scale)")
        self.degrees_spin = _make_double_spin(0.0, 180.0, 0.0, decimals=1, step=5.0, tooltip="随机旋转角度范围 (+/- degrees)")
        self.shear_spin = _make_double_spin(0.0, 45.0, 0.0, decimals=1, step=2.0, tooltip="随机错切角度范围 (+/- shear)")
        self.perspective_spin = _make_double_spin(
            0.0, 0.001, 0.0, decimals=4, step=0.0001, tooltip="随机透视变换系数 (perspective, 0~0.001)"
        )
        aug_row2.addWidget(QLabel("Translate:"))
        aug_row2.addWidget(self.translate_spin)
        aug_row2.addWidget(QLabel("Scale:"))
        aug_row2.addWidget(self.scale_spin)
        aug_row2.addWidget(QLabel("Degrees:"))
        aug_row2.addWidget(self.degrees_spin)
        aug_row2.addWidget(QLabel("Shear:"))
        aug_row2.addWidget(self.shear_spin)
        aug_row2.addWidget(QLabel("Persp:"))
        aug_row2.addWidget(self.perspective_spin)
        aug_form.addRow("几何变换:", aug_row2)

        aug_row3 = QHBoxLayout()
        self.hsv_h_spin = _make_double_spin(0.0, 1.0, 0.015, decimals=3, step=0.005, tooltip="HSV 色调抖动幅度 (hsv_h)")
        self.hsv_s_spin = _make_double_spin(0.0, 1.0, 0.7, step=0.05, tooltip="HSV 饱和度抖动幅度 (hsv_s)")
        self.hsv_v_spin = _make_double_spin(0.0, 1.0, 0.4, step=0.05, tooltip="HSV 明度抖动幅度 (hsv_v)")
        self.cutmix_spin = _make_double_spin(0.0, 1.0, 0.0, tooltip="CutMix 区域混合概率 (cutmix)")
        self.copy_paste_spin = _make_double_spin(0.0, 1.0, 0.0, tooltip="目标复制粘贴增强比例 (copy_paste)")
        aug_row3.addWidget(QLabel("HSV-H:"))
        aug_row3.addWidget(self.hsv_h_spin)
        aug_row3.addWidget(QLabel("HSV-S:"))
        aug_row3.addWidget(self.hsv_s_spin)
        aug_row3.addWidget(QLabel("HSV-V:"))
        aug_row3.addWidget(self.hsv_v_spin)
        aug_row3.addWidget(QLabel("CutMix:"))
        aug_row3.addWidget(self.cutmix_spin)
        aug_row3.addWidget(QLabel("CopyPaste:"))
        aug_row3.addWidget(self.copy_paste_spin)
        aug_form.addRow("色彩与贴图:", aug_row3)

        self.extra_args_edit = QPlainTextEdit()
        self.extra_args_edit.setFixedHeight(50)
        self.extra_args_edit.setPlaceholderText(
            '可选：透传或覆盖任意 Ultralytics 参数（支持 JSON 或 key=value，如 box=7.5, weight_decay=0.0005, freeze=10）'
        )
        aug_form.addRow("附加参数:", self.extra_args_edit)

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
        self._bind_preset_dirty_watchers()
        self.custom_pt_btn.clicked.connect(self._on_pick_custom_pt)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_run_btn.clicked.connect(lambda: open_local_path(self._last_save_dir))
        self.send_eval_btn.clicked.connect(
            lambda: self.send_best_weights.emit(self._last_best_pt, self.yaml_picker.text())
            if self._last_best_pt
            else None
        )

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

    def _mark_preset_custom(self, *_args: object) -> None:
        if self._applying_preset:
            return
        custom_idx = self.preset_combo.findData(_CUSTOM_PRESET_KEY)
        if custom_idx >= 0 and self.preset_combo.currentIndex() != custom_idx:
            self._applying_preset = True
            try:
                self.preset_combo.setCurrentIndex(custom_idx)
            finally:
                self._applying_preset = False

    def _on_preset_changed(self, _index: int) -> None:
        if self._applying_preset:
            return
        preset_key = str(self.preset_combo.currentData() or "")
        if preset_key and preset_key != _CUSTOM_PRESET_KEY:
            self.apply_preset(preset_key)

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

        if self.device_edit.text().strip() == self._auto_device_text:
            self.device_edit.setText(info.recommended_device)
        self._auto_device_text = info.recommended_device

        target_batch = 16 if info.cuda_available else 4
        if self.batch_spin.value() == self._auto_batch_value:
            self.batch_spin.setValue(target_batch)
        self._auto_batch_value = target_batch

        if was_probing and (self.worker is None or not self.worker.isRunning()):
            self.amp_chk.setChecked(info.cuda_available)

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

    def build_train_config(self) -> TrainConfig:
        """校验当前界面表单并组装完整的 `TrainConfig`；输入非法时抛出带中文说明的 `ValueError`。"""
        yaml_text = self.yaml_picker.text()
        if not yaml_text or not Path(yaml_text).is_file():
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

        return TrainConfig(
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

    def _on_start(self) -> None:
        """校验表单与附加超参数后锁定运行按钮、清空日志并启动 `TrainWorker`。"""
        if self.device_info.probing:
            self.apply_device_info(detect_compute_device())

        try:
            cfg = self.build_train_config()
        except ValueError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return

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
