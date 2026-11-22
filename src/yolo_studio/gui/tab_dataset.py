"""标签页 4：标注统计检验与 YOLO26 数据集准备界面。

职责：
    读取标注工作台或 LLM 智能标注产出的图片与标注，扫描并展示标注分布与告警，
    并按 train/val/test 比例生成 YOLO26 标准数据集与 `data.yaml`。
    左侧按“数据来源 / 划分与输出”分组：前者决定读取哪些图片与标注（扫描只依赖它），后者只影响生成结果。
    界面负责参数输入、划分比例归一化预览、覆盖已有划分前的确认、
    类别分布（含各划分框数与验证集缺类提示）展示，以及参数修改后的结果过期标记。

运行环境与协作边界：
    运行于 PyQt6 主线程；扫描与构建任务委托给 `DatasetScanWorker` / `DatasetBuildWorker`，
    不支持中途取消，运行期间只显示不定长忙碌进度条。
    生成结果通过 `send_data_yaml` 信号同步至 `MainWindow` 与训练标签页，
    `request_train_tab` 仅请求主窗口切换到训练页，本模块不直接操作其他标签页。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.dataset_converter import (
    AnnotationFormat,
    DatasetBuildConfig,
    DatasetBuildResult,
    DatasetScanSummary,
    ImageCountPreview,
    existing_split_dirs,
    preview_image_counts,
)
from yolo_studio.core.mllm_reviewer import BUCKET_NEED_REVIEW
from yolo_studio.gui.common import (
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    open_local_path,
)
from yolo_studio.gui.workers import DatasetBuildWorker, DatasetScanWorker

# 日志框中最多逐条展示的告警数量；超出部分只给出剩余条数，避免海量告警拖慢界面
MAX_WARNINGS_SHOWN = 50
# 类别框数低于“最多类别框数 × 该比例”时视为明显不均衡并以橙色高亮
LOW_SHARE_RATIO = 0.1

# 状态提示等级 → 文字颜色
_STATUS_COLORS = {
    "info": "#555555",
    "busy": "#0969da",
    "ok": "#1f7a1f",
    "warn": "#b25900",
    "error": "#cf222e",
}
_WARN_BG = QColor("#fff4e5")
_ERROR_BG = QColor("#fde2e1")

_CLASSES_PLACEHOLDER_DEFAULT = "可选 classes.txt（定义类别顺序，留空则自动提取）"
_CLASSES_PLACEHOLDER_YOLO = (
    "YOLO .txt 只含类别编号，建议指定；留空时在标注/图片目录中查找 classes.txt，找不到则用 class_0, class_1..."
)


class _StatCard(QWidget):
    """统计区的小型指标卡片：上方灰色标题、下方加粗数值，标题与数值均可随视图切换更新。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        # QWidget 子类需显式开启样式背景，主窗口样式表中 #metricCard 的边框与底色才会生效
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(2)
        self.title_label = QLabel("-")
        self.title_label.setStyleSheet("color: #666666;")
        self.value_label = QLabel("-")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.set_value("-", "-")

    def set_value(self, title: str, value: str, color: str | None = None) -> None:
        """更新卡片标题与数值；`color` 为空时使用默认文字颜色。"""
        self.title_label.setText(title)
        self.value_label.setText(value)
        color_css = f"color: {color};" if color else ""
        self.value_label.setStyleSheet(f"font-weight: 700; font-size: 15px; {color_css}")


class _SortableItem(QTableWidgetItem):
    """按数值键排序的表格单元格，避免“10”排在“9”前面这类按字符串排序的问题。"""

    def __init__(self, text: str, sort_key: float) -> None:
        super().__init__(text)
        self._sort_key = sort_key
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, _SortableItem):
            return self._sort_key < other._sort_key
        return super().__lt__(other)


class DatasetTab(QWidget):
    """标注校验与 YOLO26 数据集构建标签页（标签页 4）。

    内部状态（均只在主线程读写）：
        `scan_worker` / `build_worker`：最近一次启动的后台线程，供主窗口关闭时等待退出。
        `_param_revision`：参数版本号，任何会影响扫描或生成结果的输入变化都会递增。
        `_running_revision`：当前后台任务启动时的参数版本号。
        `_result_revision`：右侧统计结果对应的参数版本号；`None` 表示尚无结果。
            与 `_param_revision` 不一致时结果已过期，界面会提示重新扫描或生成。
        `_last_yaml_path` / `_yaml_revision`：最近一次成功生成的 `data.yaml` 及其参数版本号；
            仅当版本号与当前参数一致时才允许“前往训练”，避免把与当前设置不符的数据集当作最新结果。
        `_busy`：是否有扫描或生成任务在运行；运行期间禁用扫描/生成/前往训练按钮。
        `_scope_counts`：当前图片目录的图片计数预览（顶层/递归/need_review 内），仅用于图片目录下方的数量提示；
            目录为空或不存在时为 `None`，图片目录变化后经防抖定时器重新计算。
    """

    send_data_yaml = Signal(str)
    request_train_tab = Signal()  # 请求主窗口切换到「5. YOLO26 训练」，不携带数据

    def __init__(self, workspace_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.scan_worker: DatasetScanWorker | None = None
        self.build_worker: DatasetBuildWorker | None = None
        self._last_yaml_path: str = ""
        self._param_revision = 0
        self._running_revision = 0
        self._result_revision: int | None = None
        self._yaml_revision: int | None = None
        self._busy = False
        self._scope_counts: ImageCountPreview | None = None
        # 图片目录输入的防抖定时器：停止输入 300 ms 后才遍历目录计数
        self._scope_timer = QTimer(self)
        self._scope_timer.setSingleShot(True)
        self._scope_timer.setInterval(300)
        self._scope_timer.timeout.connect(self._refresh_scope_counts)
        self._init_ui()

    # ------------------------------------------------------------------ 界面构建

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(10)

        left.addWidget(self._build_source_group())
        left.addWidget(self._build_split_group())
        align_form_labels([self._source_form, self._split_form])
        # 配置分组按内容高度排列，剩余空间放在分组与操作按钮之间，使操作栏与右侧面板底部对齐
        left.addStretch(1)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # 不定长模式：后台任务不上报进度，仅表示正在运行
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setFixedHeight(6)
        self.busy_bar.setVisible(False)
        left.addWidget(self.busy_bar)

        btn_row = QHBoxLayout()
        self.build_btn = QPushButton("生成 YOLO26 数据集 (data.yaml)")
        self.build_btn.setObjectName("primaryBtn")
        self.open_ds_btn = QPushButton("打开数据集目录")
        self.send_train_btn = QPushButton("前往「5. YOLO26 训练」→")
        self.send_train_btn.setEnabled(False)
        self.send_train_btn.setToolTip("生成成功后 data.yaml 已自动同步到训练页；点击再次同步并切换过去")
        btn_row.addWidget(self.build_btn)
        btn_row.addWidget(self.open_ds_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.send_train_btn)
        left.addLayout(btn_row)

        root.addLayout(left, 3)
        root.addWidget(self._build_stats_group(), 2)

        self._connect_signals()
        self._update_ratio_hint()
        self._update_format_hint()
        self._update_exclude_option()
        self._refresh_scope_counts()

    def _build_source_group(self) -> QGroupBox:
        """分组 1“数据来源”：图片目录及子目录开关、标注目录、标注格式、classes.txt 与扫描统计入口。

        分组原则：本组只放“从哪里读取图片与标注”的输入参数，「扫描标注统计」只依赖这些参数，
        因此扫描按钮放在本组末尾；划分、输出等只影响生成结果的参数放在分组 2。
        递归与排除开关决定从图片目录中取哪些图片，作为图片目录的从属选项紧贴输入框放置（不设行标签），
        并实时显示图片数。递归默认开启：内嵌 X-AnyLabeling 打开目录时同样递归列出子目录图片，
        按视频分建子目录的抽帧结果与 LLM 智能标注分流结果也都位于子目录；平铺目录开启递归结果不变，
        未复核的 need_review 默认排除。
        """
        group = QGroupBox("1. 数据来源")
        form = QFormLayout(group)
        configure_form_layout(form)
        self._source_form = form

        default_frames = str((self.workspace_root / "data" / "raw_frames").resolve())
        self.images_picker = PathPickerRow(
            placeholder="选择待转换的图片目录（与抽帧输出或标注工作台打开的目录一致）",
            mode="dir",
            default_path=default_frames,
        )
        form.addRow("图片目录:", self.images_picker)

        self.recursive_chk = QCheckBox("包含子目录")
        self.recursive_chk.setChecked(True)
        self.recursive_chk.setToolTip(
            "默认开启，与标注工作台一致（工作台打开目录时同样包含子目录中的图片）；"
            "只想使用目录顶层图片时取消勾选"
        )
        self.exclude_need_review_chk = QCheckBox()
        self.exclude_need_review_chk.setChecked(True)
        self.exclude_need_review_chk.setToolTip(
            "LLM 智能标注结果中的 need_review 含 LLM 补框与处理出错样本，人工复核完成前不应进入训练；"
            "确认已在工作台复核后可取消勾选。仅在勾选「包含子目录」时生效。"
        )
        self.scope_hint = QLabel()
        self.scope_hint.setWordWrap(True)
        scope_row = QHBoxLayout()
        scope_row.addWidget(self.recursive_chk)
        scope_row.addWidget(self.exclude_need_review_chk)
        scope_row.addStretch(1)
        scope_box = QVBoxLayout()
        scope_box.setSpacing(2)
        scope_box.addLayout(scope_row)
        scope_box.addWidget(self.scope_hint)
        form.addRow("", scope_box)

        self.ann_picker = PathPickerRow(
            placeholder="留空则默认与「图片目录」相同（支持 .json 或导出的 .txt）",
            mode="dir",
        )
        form.addRow("标注目录(可选):", self.ann_picker)

        self.format_combo = QComboBox()
        self.format_combo.addItem("自动识别 (.json 或 YOLO .txt)", AnnotationFormat.AUTO)
        self.format_combo.addItem("X-AnyLabeling 原生 .json", AnnotationFormat.XANYLABELING_JSON)
        self.format_combo.addItem("YOLO 导出 .txt + classes.txt", AnnotationFormat.YOLO_TXT)
        form.addRow("标注格式:", self.format_combo)

        self.classes_label = QLabel("classes.txt(可选):")
        self.classes_picker = PathPickerRow(
            placeholder=_CLASSES_PLACEHOLDER_DEFAULT,
            mode="file",
            file_filter="文本文件 (*.txt);;所有文件 (*.*)",
        )
        form.addRow(self.classes_label, self.classes_picker)

        scan_row = QHBoxLayout()
        self.scan_btn = QPushButton("扫描标注统计")
        self.scan_btn.setToolTip("只读扫描图片与标注文件，统计类别分布并列出告警，不写入任何文件")
        scan_tip = QLabel("在工作台保存标注后，点此刷新右侧分布")
        scan_tip.setStyleSheet("color: #666666;")
        scan_row.addWidget(self.scan_btn)
        scan_row.addWidget(scan_tip)
        scan_row.addStretch(1)
        form.addRow("", scan_row)
        return group

    def _build_split_group(self) -> QGroupBox:
        """分组 2“划分与输出”：输出目录、Train/Val/Test 比例、随机种子与负样本选项，只影响生成结果。"""
        group = QGroupBox("2. 划分与输出")
        form = QFormLayout(group)
        configure_form_layout(form)
        self._split_form = form

        default_ds_out = str((self.workspace_root / "datasets" / "custom_yolo26").resolve())
        self.output_ds_picker = PathPickerRow(
            placeholder="输出 YOLO26 标准数据集目录",
            mode="dir",
            default_path=default_ds_out,
        )
        form.addRow("数据集输出目录:", self.output_ds_picker)

        self.train_spin = self._make_ratio_spin(0.1, 1.0, 0.8)
        self.val_spin = self._make_ratio_spin(0.0, 0.9, 0.2)
        self.test_spin = self._make_ratio_spin(0.0, 0.9, 0.0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.seed_spin.setToolTip("随机种子：输入与比例不变时，相同种子得到相同的划分结果")

        ratio_row = QHBoxLayout()
        for text, spin in (
            ("Train", self.train_spin),
            ("Val", self.val_spin),
            ("Test", self.test_spin),
        ):
            ratio_row.addWidget(QLabel(f"{text}:"))
            ratio_row.addWidget(spin)
            ratio_row.addSpacing(8)
        ratio_row.addStretch(1)

        self.ratio_hint = QLabel()
        self.ratio_hint.setWordWrap(True)
        ratio_box = QVBoxLayout()
        ratio_box.setSpacing(2)
        ratio_box.addLayout(ratio_row)
        ratio_box.addWidget(self.ratio_hint)
        form.addRow("划分比例:", ratio_box)

        # 随机种子单独成行，避免与三项比例挤在一行把左栏撑宽、压缩右侧统计面板
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed_spin)
        seed_hint = QLabel("相同种子与输入得到相同划分")
        seed_hint.setStyleSheet("color: #666666;")
        seed_row.addWidget(seed_hint)
        seed_row.addStretch(1)
        form.addRow("随机种子:", seed_row)

        self.include_empty_chk = QCheckBox("保留无标注文件的图片作为负样本")
        self.include_empty_chk.setToolTip(
            "仅影响完全没有 .json/.txt 的图片；已保存但无目标的标注文件视为人工确认的负样本，始终纳入"
        )
        form.addRow("附加选项:", self.include_empty_chk)
        return group

    @staticmethod
    def _make_ratio_spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(value)
        # 按控件自身 sizeHint 定宽：高 DPI 下箭头按钮更宽，写死像素宽度会裁掉数值
        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return spin

    def _build_stats_group(self) -> QGroupBox:
        """分组 3：状态提示、指标卡片、类别分布表与下方的扫描/转换日志。"""
        group = QGroupBox("3. 标注统计与校验结果")
        layout = QVBoxLayout(group)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        cards_grid = QGridLayout()
        cards_grid.setSpacing(6)
        self.stat_cards = [_StatCard() for _ in range(6)]
        for idx, card in enumerate(self.stat_cards):
            cards_grid.addWidget(card, idx // 3, idx % 3)
        layout.addLayout(cards_grid)

        self.class_table = QTableWidget(0, 0)
        self.class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.class_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.class_table.verticalHeader().setVisible(False)
        self.class_table.setAlternatingRowColors(True)
        self._set_table_columns(with_splits=False, with_test=False)
        layout.addWidget(self.class_table, 3)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("扫描、转换日志与告警信息将在此显示...")
        layout.addWidget(self.log_box, 2)

        self._reset_stats_view()
        return group

    def _connect_signals(self) -> None:
        self.scan_btn.clicked.connect(self._on_scan)
        self.build_btn.clicked.connect(self._on_build)
        self.open_ds_btn.clicked.connect(lambda: open_local_path(self.output_ds_picker.path()))
        self.send_train_btn.clicked.connect(self._on_go_train)

        for picker in (self.images_picker, self.ann_picker, self.classes_picker, self.output_ds_picker):
            picker.textChanged.connect(self._on_params_changed)
        self.format_combo.currentIndexChanged.connect(self._on_params_changed)
        self.format_combo.currentIndexChanged.connect(self._update_format_hint)
        for spin in (self.train_spin, self.val_spin, self.test_spin, self.seed_spin):
            spin.valueChanged.connect(self._on_params_changed)
        for spin in (self.train_spin, self.val_spin, self.test_spin):
            spin.valueChanged.connect(self._update_ratio_hint)
        for chk in (self.include_empty_chk, self.recursive_chk, self.exclude_need_review_chk):
            chk.toggled.connect(self._on_params_changed)
        self.recursive_chk.toggled.connect(self._update_exclude_option)
        self.images_picker.textChanged.connect(self._schedule_scope_refresh)
        self.recursive_chk.toggled.connect(self._render_scope_hint)
        self.exclude_need_review_chk.toggled.connect(self._render_scope_hint)

    # ------------------------------------------------------------------ 跨页写入接口

    def set_frames_dir(self, path_str: str) -> None:
        """跨页写入：更新待标注/转换的图片源目录输入框；路径变化会使已有统计结果过期。"""
        if path_str:
            self.images_picker.setText(path_str)

    def set_dataset_source(
        self, images_dir: str, classes_file: str = "", recursive: bool = True
    ) -> None:
        """跨页写入：接收 LLM 智能标注结果目录、类别文件并同步递归扫描开关，清空独立标注目录栏。"""
        if images_dir:
            self.images_picker.setText(images_dir)
            self.ann_picker.setText("")
        if classes_file:
            self.classes_picker.setText(classes_file)
        self.recursive_chk.setChecked(recursive)

    def mark_annotations_changed(self, dir_path: str) -> None:
        """工作台保存标注后的通知入口：若与当前图片目录相同，则使统计结果过期并提示重新扫描。

        只比较解析后的绝对路径；路径无法解析（如含非法字符）时静默忽略，不影响工作台保存流程。
        """
        cur_dir = self.images_picker.text()
        if not cur_dir:
            return
        try:
            same = Path(cur_dir).resolve() == Path(dir_path).resolve()
        except OSError:
            return
        if not same:
            return
        self._param_revision += 1
        self._refresh_send_btn()
        self._set_status(
            "检测到「2. 标注工作台」已更新标注文件，可点击「扫描标注统计」刷新分布。", "warn"
        )

    # ------------------------------------------------------------------ 参数联动

    def _on_params_changed(self, *_args: object) -> None:
        """任一输入变化时递增参数版本号；已有结果时提示过期并禁止前往训练页。"""
        self._param_revision += 1
        self._refresh_send_btn()
        if self._result_revision is not None and not self._is_busy():
            self._set_status("参数已修改，下方统计结果已过期，请重新扫描或生成数据集。", "warn")

    def _update_ratio_hint(self, *_args: object) -> None:
        """根据三项比例实时显示归一化后的实际划分占比，以及合计不为 1、Val 为 0 等提示。"""
        tr, va, te = self.train_spin.value(), self.val_spin.value(), self.test_spin.value()
        total = tr + va + te
        if total <= 0:
            self.ratio_hint.setText("三项比例之和必须大于 0。")
            self.ratio_hint.setStyleSheet(f"color: {_STATUS_COLORS['error']};")
            return
        text = f"实际划分 ≈ Train {tr / total:.0%} / Val {va / total:.0%} / Test {te / total:.0%}"
        level = "info"
        if abs(total - 1.0) > 1e-6:
            text += f"（合计 {total:.2f}，将按比例归一化）"
            level = "warn"
        if va <= 0:
            text += "；Val 为 0 时 data.yaml 的 val 将指向训练集，验证指标会虚高"
            level = "warn"
        self.ratio_hint.setText(text)
        self.ratio_hint.setStyleSheet(f"color: {_STATUS_COLORS[level]};")

    def _update_format_hint(self, *_args: object) -> None:
        """选择 YOLO .txt 格式时强调 classes.txt 的作用，其余格式恢复为可选提示。"""
        if self.format_combo.currentData() == AnnotationFormat.YOLO_TXT:
            self.classes_label.setText("classes.txt(建议):")
            self.classes_picker.setPlaceholderText(_CLASSES_PLACEHOLDER_YOLO)
        else:
            self.classes_label.setText("classes.txt(可选):")
            self.classes_picker.setPlaceholderText(_CLASSES_PLACEHOLDER_DEFAULT)

    def _update_exclude_option(self, *_args: object) -> None:
        """排除 need_review 选项仅在递归扫描时生效：未开启递归时置灰（紧邻递归开关，无需额外说明）。"""
        self.exclude_need_review_chk.setEnabled(self.recursive_chk.isChecked())
        self.exclude_need_review_chk.setText(f"排除未复核的 {BUCKET_NEED_REVIEW} 子目录")

    def _schedule_scope_refresh(self, *_args: object) -> None:
        """图片目录变化时延迟重新计数，避免逐字输入路径时反复遍历磁盘。"""
        self._scope_timer.start()

    def _refresh_scope_counts(self) -> None:
        """在主线程遍历图片目录统计顶层/递归图片数并缓存，随后刷新数量提示。

        `preview_image_counts` 只按扩展名计数并设有条目上限，正常目录耗时为毫秒级；
        目录为空或不存在时缓存置为 `None`，提示区显示对应状态。
        """
        text = self.images_picker.text()
        if not text or not Path(text).is_dir():
            self._scope_counts = None
        else:
            self._scope_counts = preview_image_counts(Path(text), (BUCKET_NEED_REVIEW,))
        self._render_scope_hint()

    def _render_scope_hint(self, *_args: object) -> None:
        """根据缓存计数与当前勾选状态，显示扫描统计与生成数据集所用的候选图片数量。

        候选图片按扩展名计数，生成时是否纳入还取决于标注情况与负样本选项，因此只表述为“共 N 张图片”。

        重点场景：顶层没有图片而子目录有图片且未勾选递归时，以醒目颜色提示勾选，
        这是“按视频/分流子目录组织的数据集”最常见的配置遗漏。
        """
        text = self.images_picker.text()
        counts = self._scope_counts
        if not text:
            self.scope_hint.setText("")
            return
        if counts is None:
            self._set_scope_hint("目录不存在或无法访问。", "error")
            return
        more = "+" if counts.truncated else ""
        nested = counts.total - counts.top_level
        if self.recursive_chk.isChecked():
            kept = counts.total - (counts.excluded if self.exclude_need_review_chk.isChecked() else 0)
            msg = f"共 {kept}{more} 张图片（含子目录）"
            if self.exclude_need_review_chk.isChecked() and counts.excluded:
                msg += f"；已排除 {BUCKET_NEED_REVIEW} 中的 {counts.excluded} 张"
            self._set_scope_hint(msg, "ok" if kept else "error")
        elif counts.top_level == 0 and nested > 0:
            self._set_scope_hint(
                f"顶层没有图片，子目录中有 {nested}{more} 张，请勾选「包含子目录」。", "warn"
            )
        elif counts.top_level == 0:
            self._set_scope_hint("该目录及子目录中没有图片。", "error")
        elif nested > 0:
            self._set_scope_hint(
                f"共 {counts.top_level} 张图片（仅顶层）；子目录另有 {nested}{more} 张，勾选「包含子目录」可一并纳入。",
                "info",
            )
        else:
            self._set_scope_hint(f"共 {counts.top_level}{more} 张图片。", "ok")

    def _set_scope_hint(self, text: str, level: str) -> None:
        self.scope_hint.setText(text)
        self.scope_hint.setStyleSheet(f"color: {_STATUS_COLORS[level]};")

    def _refresh_send_btn(self) -> None:
        fresh = bool(self._last_yaml_path) and self._yaml_revision == self._param_revision
        self.send_train_btn.setEnabled(fresh and not self._is_busy())

    # ------------------------------------------------------------------ 配置与动作

    def _build_config(self) -> DatasetBuildConfig:
        images_text = self.images_picker.text()
        if not images_text:
            raise ValueError("请先指定图片目录。")
        ann_text = self.ann_picker.text()
        cls_text = self.classes_picker.text()
        out_text = self.output_ds_picker.text()
        if not out_text:
            raise ValueError("请先指定数据集输出目录。")
        if self.train_spin.value() + self.val_spin.value() + self.test_spin.value() <= 0:
            raise ValueError("train/val/test 比例之和必须大于 0。")

        return DatasetBuildConfig(
            images_dir=Path(images_text),
            annotations_dir=Path(ann_text) if ann_text else None,
            output_dir=Path(out_text),
            annotation_format=self.format_combo.currentData(),
            classes_file=Path(cls_text) if cls_text else None,
            train_ratio=self.train_spin.value(),
            val_ratio=self.val_spin.value(),
            test_ratio=self.test_spin.value(),
            include_unannotated=self.include_empty_chk.isChecked(),
            seed=self.seed_spin.value(),
            copy_images=True,
            recursive=self.recursive_chk.isChecked(),
            exclude_subdirs=(
                (BUCKET_NEED_REVIEW,) if self.exclude_need_review_chk.isChecked() else ()
            ),
        )

    def _on_scan(self) -> None:
        """校验参数后进入忙碌状态并在后台线程执行只读扫描。"""
        try:
            cfg = self._build_config()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return

        self._set_busy(True)
        self.log_box.clear()
        self._set_status("正在扫描图片和标注文件...", "busy")

        self.scan_worker = DatasetScanWorker(cfg)
        self.scan_worker.finished_summary.connect(self._on_scan_done)
        self.scan_worker.failed.connect(self._on_worker_failed)
        self.scan_worker.start()

    def _on_scan_done(self, summary: DatasetScanSummary) -> None:
        """展示扫描统计；若扫描期间参数又被修改，结果展示后立即标记为过期。"""
        self._set_busy(False)
        self._result_revision = self._running_revision

        fmt_name = "X-AnyLabeling JSON" if summary.detected_format == "json" else "YOLO TXT"
        warn_count = len(summary.warnings)
        cards = self.stat_cards
        cards[0].set_value("图片总数", str(summary.total_images))
        cards[1].set_value("已标注", str(summary.annotated_images), _STATUS_COLORS["ok"])
        cards[2].set_value(
            f"未标注（显式负样本 {summary.negative_images}）",
            str(summary.unannotated_images),
            _STATUS_COLORS["warn"] if summary.unannotated_images > summary.negative_images else None,
        )
        cards[3].set_value("总框数", str(summary.total_boxes))
        cards[4].set_value("类别数", str(len(summary.classes)))
        cards[5].set_value("告警", str(warn_count), _STATUS_COLORS["warn"] if warn_count else None)

        self._populate_class_table(summary.classes, summary.class_counts)

        status = f"扫描完成，识别格式: {fmt_name}"
        if summary.excluded_images:
            status += f"；已按设置排除 {summary.excluded_images} 张图片"
        if warn_count:
            status += f"；有 {warn_count} 条告警，详见下方日志"
        self._finish_status(status, "warn" if warn_count else "ok")

        if summary.warnings:
            self.log_box.append("发现以下告警：")
            self._append_warnings(summary.warnings, prefix="- ")
        else:
            self.log_box.append("扫描完成，未发现异常。")

    def _on_build(self) -> None:
        """校验参数、确认覆盖已有划分目录后，在后台线程执行转换、划分与 `data.yaml` 写出。"""
        try:
            cfg = self._build_config()
        except Exception as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        if not self._confirm_overwrite(cfg.output_dir):
            return

        self._set_busy(True)
        self.log_box.clear()
        self._set_status("正在转换标注并生成 YOLO26 数据集...", "busy")

        self.build_worker = DatasetBuildWorker(cfg)
        self.build_worker.finished_result.connect(self._on_build_done)
        self.build_worker.failed.connect(self._on_worker_failed)
        self.build_worker.start()

    def _confirm_overwrite(self, output_dir: Path) -> bool:
        """输出目录中已有非空划分子目录时弹窗确认；用户取消返回 `False`，无可覆盖内容直接返回 `True`。"""
        existing = existing_split_dirs(output_dir)
        if not existing:
            return True
        lines = []
        for d in existing:
            try:
                lines.append(f"  • {d.relative_to(output_dir).as_posix()}")
            except ValueError:
                lines.append(f"  • {d}")
        reply = QMessageBox.question(
            self,
            "覆盖已有数据集",
            "输出目录中已存在以下数据集划分，生成时将被清空并重建：\n\n"
            + "\n".join(lines)
            + f"\n\n输出目录: {output_dir}\n确定继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _on_build_done(self, result: DatasetBuildResult) -> None:
        """展示划分统计与各划分类别分布，并自动向训练页派发新生成的 `data.yaml` 路径。"""
        self._set_busy(False)
        self._result_revision = self._running_revision
        self._yaml_revision = self._running_revision
        self._last_yaml_path = str(result.data_yaml_path)

        missing_val = self._classes_missing_in_split(result, "val") if result.val_count else []
        warn_count = len(result.warnings) + (1 if missing_val else 0)
        cards = self.stat_cards
        cards[0].set_value("Train 图片", str(result.train_count))
        cards[1].set_value(
            "Val 图片", str(result.val_count), _STATUS_COLORS["warn"] if not result.val_count else None
        )
        cards[2].set_value("Test 图片", str(result.test_count))
        cards[3].set_value("总框数", str(result.total_boxes))
        cards[4].set_value("类别数", str(len(result.classes)))
        cards[5].set_value("告警", str(warn_count), _STATUS_COLORS["warn"] if warn_count else None)

        self._populate_class_table(
            result.classes,
            result.class_counts,
            split_counts=result.split_class_counts,
            with_test=result.test_count > 0,
        )

        self.log_box.append(f"data.yaml 已生成: {result.data_yaml_path}")
        self.log_box.append(f"classes.txt 已生成: {result.classes_txt_path}")
        self.log_box.append("data.yaml 路径已自动同步到「5. YOLO26 训练」。")
        if not result.val_count:
            self.log_box.append("[警告] 验证集为空，data.yaml 的 val 已指向训练集，验证指标不能反映泛化效果。")
        if missing_val:
            self.log_box.append(
                f"[警告] 验证集中缺少以下类别，无法评估其精度: {', '.join(missing_val)}；"
                "可调大 Val 比例或更换 Seed 后重新生成。"
            )
        self._append_warnings(result.warnings, prefix="[警告] ")

        status = f"数据集已生成；有 {warn_count} 条告警，详见下方日志" if warn_count else "数据集已生成"
        self._finish_status(status, "warn" if warn_count else "ok")
        self._refresh_send_btn()
        self.send_data_yaml.emit(self._last_yaml_path)

    def _on_go_train(self) -> None:
        """重新同步最近一次生成的 `data.yaml` 并请求主窗口切换到训练页。"""
        if not self._last_yaml_path or self._yaml_revision != self._param_revision:
            return
        self.send_data_yaml.emit(self._last_yaml_path)
        self.request_train_tab.emit()

    def _on_worker_failed(self, err: str) -> None:
        self._set_busy(False)
        self._set_status(f"处理失败: {err}", "error")
        QMessageBox.critical(self, "数据集处理错误", err)

    # ------------------------------------------------------------------ 展示辅助

    def _is_busy(self) -> bool:
        return self._busy

    def _set_busy(self, busy: bool) -> None:
        """切换后台任务运行状态；进入忙碌时记录当前参数版本号，供结果返回后判断是否过期。"""
        self._busy = busy
        if busy:
            self._running_revision = self._param_revision
        self.scan_btn.setEnabled(not busy)
        self.build_btn.setEnabled(not busy)
        self.busy_bar.setVisible(busy)
        self._refresh_send_btn()

    def _set_status(self, text: str, level: str = "info") -> None:
        self.summary_label.setText(text)
        self.summary_label.setStyleSheet(f"color: {_STATUS_COLORS.get(level, _STATUS_COLORS['info'])};")

    def _finish_status(self, text: str, level: str) -> None:
        """任务完成后的状态提示；若运行期间参数已被修改，则改为过期提示。"""
        if self._result_revision != self._param_revision:
            self._set_status(f"{text}，但运行期间参数已修改，结果已过期，请重新扫描或生成。", "warn")
        else:
            self._set_status(text + "。", level)

    def _append_warnings(self, warnings: list[str], prefix: str) -> None:
        for w in warnings[:MAX_WARNINGS_SHOWN]:
            self.log_box.append(f"{prefix}{w}")
        hidden = len(warnings) - MAX_WARNINGS_SHOWN
        if hidden > 0:
            self.log_box.append(f"……另有 {hidden} 条告警未显示。")

    def _reset_stats_view(self) -> None:
        for card, title in zip(self.stat_cards, ("图片总数", "已标注", "未标注", "总框数", "类别数", "告警")):
            card.set_value(title, "-")
        self._set_status("点击「扫描标注统计」查看图片数、标注框数和类别分布。", "info")

    @staticmethod
    def _classes_missing_in_split(result: DatasetBuildResult, split: str) -> list[str]:
        split_counts = result.split_class_counts.get(split, {})
        return [
            name
            for name in result.classes
            if result.class_counts.get(name, 0) > 0 and split_counts.get(name, 0) == 0
        ]

    def _set_table_columns(self, with_splits: bool, with_test: bool) -> None:
        """设置类别表列；各划分框数合并为一列（如 `14 / 4`），避免右侧窄面板中类别名称被挤压。"""
        headers = ["ID", "类别名称", "框数", "占比"]
        if with_splits:
            headers.append("Train / Val / Test" if with_test else "Train / Val")
        self.class_table.setColumnCount(len(headers))
        self.class_table.setHorizontalHeaderLabels(headers)
        header = self.class_table.horizontalHeader()
        header.setMinimumSectionSize(36)
        for col in range(len(headers)):
            mode = QHeaderView.ResizeMode.Stretch if col == 1 else QHeaderView.ResizeMode.ResizeToContents
            header.setSectionResizeMode(col, mode)

    def _populate_class_table(
        self,
        classes: list[str],
        counts: dict[str, int],
        split_counts: dict[str, dict[str, int]] | None = None,
        with_test: bool = False,
    ) -> None:
        """填充类别分布表并高亮异常：无框类别标红、明显偏少类别标橙、验证集缺失的类别标红。"""
        with_splits = split_counts is not None
        self.class_table.setSortingEnabled(False)  # 填充期间关闭排序，避免行在写入过程中被重排
        self.class_table.clearContents()
        self._set_table_columns(with_splits, with_test)
        self.class_table.setRowCount(len(classes))

        total = sum(counts.get(name, 0) for name in classes)
        max_count = max((counts.get(name, 0) for name in classes), default=0)
        for row, name in enumerate(classes):
            count = counts.get(name, 0)
            share = count / total if total else 0.0
            id_item = _SortableItem(str(row), row)
            name_item = QTableWidgetItem(name)
            count_item = _SortableItem(str(count), count)
            share_item = _SortableItem(f"{share:.1%}", share)
            row_items = [id_item, name_item, count_item, share_item]

            row_bg: QColor | None = None
            if count == 0:
                row_bg = _ERROR_BG
                name_item.setToolTip("该类别没有任何标注框（可能仅在 classes.txt 中定义）")
            elif max_count and count < max_count * LOW_SHARE_RATIO:
                row_bg = _WARN_BG
                name_item.setToolTip(
                    f"框数不足最多类别的 {LOW_SHARE_RATIO:.0%}，类别明显不均衡，建议补充样本"
                )

            if with_splits:
                splits = ("train", "val") + (("test",) if with_test else ())
                per_split = [split_counts.get(s, {}).get(name, 0) for s in splits]
                val_n = per_split[1]
                # 合并列按验证集框数排序，便于把验证集缺失或偏少的类别排到一起
                item = _SortableItem(" / ".join(str(n) for n in per_split), val_n)
                item.setToolTip("、".join(f"{s}: {n}" for s, n in zip(splits, per_split)))
                if val_n == 0 and count > 0:
                    item.setBackground(_ERROR_BG)
                    item.setToolTip(item.toolTip() + "\n验证集中没有该类别，无法评估其精度")
                row_items.append(item)

            for col, item in enumerate(row_items):
                if row_bg is not None and item.background().style() == Qt.BrushStyle.NoBrush:
                    item.setBackground(row_bg)
                self.class_table.setItem(row, col, item)

        self.class_table.setSortingEnabled(True)
        self.class_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
