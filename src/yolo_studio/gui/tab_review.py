"""标签页 4：基准模型自动标注与 MLLM 纠错补漏界面。

职责：
    配置候选框来源（基准 `.pt` 低阈值推理或已有 X-AnyLabeling `.json`）、MLLM 接口与预设、
    审核分流策略；展示逐图分流结果表格、纠错前后对比图与实时日志，
    并支持直接跳转「2. 标注工作台」打开 `need_review` 目录或定位单张图片复核，以及发送清洗结果至数据集准备页。

运行环境与协作边界：
    运行于 PyQt6 主线程；后台审核由 `MllmReviewWorker` 执行，`render_review_comparison`
    在点击行时即时渲染对比图，通过 `open_in_workbench` 与 `send_cleaned_dataset` 信号与 `MainWindow` 协同。
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.mllm_reviewer import (
    GEMINI_MODELS,
    OPENAI_PRESETS,
    ImageReviewRecord,
    MllmProvider,
    MllmReviewConfig,
    MllmReviewResult,
    ReviewSourceMode,
    render_review_comparison,
)
from yolo_studio.core.trainer import DeviceInfo, detect_compute_device
from yolo_studio.gui.common import (
    LOG_MAX_LINES,
    ComputeDeviceControl,
    GpuTaskGuard,
    PathPickerRow,
    open_local_path,
)
from yolo_studio.gui.workers import MllmReviewWorker

# 左右对比预览 JPEG 字节流的 LRU 缓存容量上限（按结果表行号索引，每次重新开始审核时清空）
_PREVIEW_CACHE_MAX = 64


class MllmReviewTab(QWidget):
    """MLLM 自动标注纠错与补漏功能标签页（标签页 4）。

    内部状态：
        持有硬件探测信息 `device_info`、跨页计算设备互斥协调器 `gpu_guard`、后台工作线程 `worker`、
        已完成的逐图审核记录列表 `_records`、对比图字节缓存 `_preview_cache`、
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
        self._records: list[ImageReviewRecord] = []
        self._preview_cache: OrderedDict[int, bytes] = OrderedDict()
        self._last_output_dir: Path = workspace_root / "data" / "reviewed_frames"
        self._last_classes_txt: str = ""
        self._init_ui()

    def apply_device_info(self, info: DeviceInfo) -> None:
        """异步硬件探测完成后刷新基准推理设备开关与提示（保留用户已手动修改的选择）。"""
        self.device_info = info
        self.yolo_device_ctrl.apply_device_info(info)

    def set_images_dir(self, path_str: str) -> None:
        """跨页写入：同步待纠错图片目录输入框。"""
        if path_str:
            self.images_picker.setText(path_str)

    def set_weights_path(self, path_str: str) -> None:
        """跨页写入：将新训练的 `best.pt` 路径更新至基准模型输入框。"""
        if path_str:
            self.weights_picker.setText(path_str)

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        # 左侧面板：输入来源、MLLM 接口与清洗分流参数配置
        left = QVBoxLayout()
        left.setSpacing(8)

        # 1. 待纠错数据来源与基准模型
        src_group = QGroupBox("1. 待纠错数据来源与基准模型")
        s_form = QFormLayout(src_group)
        s_form.setSpacing(6)

        self.source_mode_combo = QComboBox()
        self.source_mode_combo.addItem(
            "基准模型低阈值推理 + MLLM 纠错（推荐）", ReviewSourceMode.YOLO_PREDICT
        )
        self.source_mode_combo.addItem(
            "读取已有 X-AnyLabeling .json + MLLM 复核", ReviewSourceMode.EXISTING_JSON
        )
        self.source_mode_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        s_form.addRow("工作模式:", self.source_mode_combo)

        default_frames = str((self.workspace_root / "data" / "raw_frames").resolve())
        self.images_picker = PathPickerRow(
            placeholder="待自动标注或复核的图片目录",
            mode="dir",
            default_path=default_frames,
        )
        s_form.addRow("图片目录:", self.images_picker)

        self.weights_picker = PathPickerRow(
            placeholder="选择基准模型 best.pt 权重（先手动标一批训练得出）",
            mode="file",
            file_filter="PyTorch 权重 (*.pt);;所有文件 (*.*)",
        )
        s_form.addRow("基准模型 (.pt):", self.weights_picker)

        yolo_row = QHBoxLayout()
        self.yolo_conf_spin = QDoubleSpinBox()
        self.yolo_conf_spin.setRange(0.01, 0.95)
        self.yolo_conf_spin.setSingleStep(0.02)
        self.yolo_conf_spin.setValue(0.12)
        self.yolo_conf_spin.setToolTip("建议设低（0.08~0.15）以尽量召回潜在漏标，再由 MLLM 筛除误检")
        self.yolo_iou_spin = QDoubleSpinBox()
        self.yolo_iou_spin.setRange(0.10, 0.95)
        self.yolo_iou_spin.setSingleStep(0.05)
        self.yolo_iou_spin.setValue(0.60)
        self.yolo_imgsz_spin = QSpinBox()
        self.yolo_imgsz_spin.setRange(128, 2048)
        self.yolo_imgsz_spin.setSingleStep(32)
        self.yolo_imgsz_spin.setValue(640)
        self.yolo_device_ctrl = ComputeDeviceControl(
            self.device_info,
            show_status_text=False,
            allow_multi_gpu=False,
            checkbox_text="GPU 加速",
            compact_status=True,
        )
        yolo_row.addWidget(QLabel("低阈值Conf:"))
        yolo_row.addWidget(self.yolo_conf_spin)
        yolo_row.addWidget(QLabel("IoU:"))
        yolo_row.addWidget(self.yolo_iou_spin)
        yolo_row.addWidget(QLabel("ImgSz:"))
        yolo_row.addWidget(self.yolo_imgsz_spin)
        yolo_row.addWidget(self.yolo_device_ctrl)
        s_form.addRow("基准推理参数:", yolo_row)

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
        left.addWidget(src_group)

        # 2. MLLM 接口与模型配置
        llm_group = QGroupBox("2. 多模态大语言模型 (MLLM) 配置")
        l_form = QFormLayout(llm_group)
        l_form.setSpacing(6)

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Google Gemini API (google-genai)", MllmProvider.GEMINI)
        self.provider_combo.addItem(
            "OpenAI 兼容接口 (Qwen-VL / 硅基流动 / Ollama / vLLM)", MllmProvider.OPENAI_COMPAT
        )
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        l_form.addRow("模型接口:", self.provider_combo)

        self.preset_combo = QComboBox()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        l_form.addRow("常用预设:", self.preset_combo)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("OpenAI 兼容接口 Base URL（Gemini 模式无需填写）")
        l_form.addRow("Base URL:", self.base_url_edit)

        self.model_edit = QLineEdit("gemini-3.8-flash")
        l_form.addRow("模型名称:", self.model_edit)

        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("留空则自动读取环境变量 GEMINI_API_KEY / OPENAI_API_KEY")
        self.show_key_chk = QCheckBox("显示")
        self.show_key_chk.toggled.connect(
            lambda checked: self.api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(self.api_key_edit, 1)
        key_row.addWidget(self.show_key_chk)
        l_form.addRow("API Key:", key_row)
        left.addWidget(llm_group)

        # 3. 纠错、补漏与分流策略
        pol_group = QGroupBox("3. 纠错、补漏与分流策略")
        p_form = QFormLayout(pol_group)
        p_form.setSpacing(6)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("英文逗号分隔，如 person, helmet（留空自动从模型或 .json 提取）")
        p_form.addRow("目标类别清单:", self.classes_edit)

        self.guideline_edit = QLineEdit()
        self.guideline_edit.setPlaceholderText("可选标注规则，如：远景模糊人影不标；反光背心只标穿在身上的")
        p_form.addRow("辅助判定规则:", self.guideline_edit)

        triage_row = QHBoxLayout()
        self.high_conf_spin = QDoubleSpinBox()
        self.high_conf_spin.setRange(0.10, 1.00)
        self.high_conf_spin.setSingleStep(0.05)
        self.high_conf_spin.setValue(0.85)
        self.high_conf_spin.setToolTip("当未开启漏标巡检且所有框置信度均高于此值时，跳过 MLLM 调用")
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(2)
        triage_row.addWidget(QLabel("高置信度免审阈值:"))
        triage_row.addWidget(self.high_conf_spin)
        triage_row.addWidget(QLabel("并发数:"))
        triage_row.addWidget(self.workers_spin)
        p_form.addRow("效率与并发:", triage_row)

        chk_row1 = QHBoxLayout()
        self.check_missing_chk = QCheckBox("检查漏标目标")
        self.check_missing_chk.setChecked(True)
        self.add_missing_chk = QCheckBox("尝试直接补充漏标框")
        self.add_missing_chk.setChecked(True)
        self.crops_chk = QCheckBox("附带局部裁剪放大图")
        self.crops_chk.setChecked(True)
        chk_row1.addWidget(self.check_missing_chk)
        chk_row1.addWidget(self.add_missing_chk)
        chk_row1.addWidget(self.crops_chk)
        p_form.addRow("纠错选项:", chk_row1)

        self.triage_dirs_chk = QCheckBox(
            "按结果分建子目录 (auto_passed / llm_fixed / need_review)"
        )
        self.triage_dirs_chk.setChecked(True)
        self.recursive_chk = QCheckBox("递归扫描输入子目录")
        chk_row2 = QHBoxLayout()
        chk_row2.addWidget(self.triage_dirs_chk)
        chk_row2.addWidget(self.recursive_chk)
        p_form.addRow("分流与目录:", chk_row2)
        left.addWidget(pol_group)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("开始 MLLM 纠错清洗")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.open_need_review_btn = QPushButton("在工作台打开待复核目录")
        self.open_out_btn = QPushButton("打开输出目录")
        self.send_ds_btn = QPushButton("发送到「3. 数据集准备」→")
        self.send_ds_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.open_need_review_btn)
        btn_row.addWidget(self.open_out_btn)
        btn_row.addWidget(self.send_ds_btn)
        left.addLayout(btn_row)

        # 右侧面板：汇总统计、逐图结果表、Before/After 对比预览与审核日志
        right = QVBoxLayout()
        right.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        right.addWidget(self.progress_bar)

        self.summary_label = QLabel("就绪：配置基准模型与 MLLM 后点击「开始 MLLM 纠错清洗」。")
        self.summary_label.setWordWrap(True)
        right.addWidget(self.summary_label)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(
            ["图片", "分流结果", "保留", "改类", "删误检", "补漏", "说明"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.currentCellChanged.connect(self._on_row_selected)
        self.result_table.cellDoubleClicked.connect(self._on_row_double_clicked)
        splitter.addWidget(self.result_table)

        self.preview_label = QLabel("单击表格行对比「纠错前 vs 纠错后」标注框；双击行可在「2. 标注工作台」定位该图精修")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(230)
        self.preview_label.setStyleSheet(
            "border: 1px dashed #999999; border-radius: 6px; background: #1e1e1e; color: #cccccc;"
        )
        splitter.addWidget(self.preview_label)

        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(140)
        self.log_box.setPlaceholderText("MLLM 逐张审核日志将在此输出...")
        splitter.addWidget(self.log_box)

        splitter.setSizes([210, 270, 120])
        right.addWidget(splitter, 1)

        root.addLayout(left, 5)
        root.addLayout(right, 4)

        # 信号与槽连接
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_need_review_btn.clicked.connect(self._on_open_need_review_in_xal)
        self.open_out_btn.clicked.connect(lambda: open_local_path(self.output_picker.path()))
        self.send_ds_btn.clicked.connect(self._on_send_to_dataset)
        self.check_missing_chk.toggled.connect(self.add_missing_chk.setEnabled)

        self._on_source_mode_changed()
        self._on_provider_changed()

    def _on_source_mode_changed(self) -> None:
        is_yolo = self.source_mode_combo.currentData() == ReviewSourceMode.YOLO_PREDICT
        self.weights_picker.setEnabled(is_yolo)
        self.yolo_conf_spin.setEnabled(is_yolo)
        self.yolo_iou_spin.setEnabled(is_yolo)
        self.yolo_imgsz_spin.setEnabled(is_yolo)
        self.yolo_device_ctrl.setEnabled(is_yolo)
        self.ann_picker.setEnabled(not is_yolo)

    def _on_provider_changed(self) -> None:
        provider = self.provider_combo.currentData()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        if provider == MllmProvider.GEMINI:
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
        self.preset_combo.blockSignals(False)

    def _on_preset_changed(self) -> None:
        data = self.preset_combo.currentData()
        if not data:
            return
        model_id, url = data
        self.model_edit.setText(model_id)
        if url:
            self.base_url_edit.setText(url)

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

        source_mode: ReviewSourceMode = self.source_mode_combo.currentData()
        if source_mode == ReviewSourceMode.YOLO_PREDICT and self.device_info.probing:
            self.apply_device_info(detect_compute_device())
        weights_text = self.weights_picker.text()
        if source_mode == ReviewSourceMode.YOLO_PREDICT and (
            not weights_text or not Path(weights_text).is_file()
        ):
            QMessageBox.warning(self, "提示", "请选择有效的基准模型 .pt 权重文件。")
            return

        raw_classes = self.classes_edit.text().replace("，", ",")
        allowed_classes = [c.strip() for c in raw_classes.split(",") if c.strip()]

        ann_text = self.ann_picker.text()
        self._last_output_dir = Path(out_text)

        cfg = MllmReviewConfig(
            images_dir=Path(images_text),
            output_dir=self._last_output_dir,
            source_mode=source_mode,
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
            triage_subfolders=self.triage_dirs_chk.isChecked(),
            recursive=self.recursive_chk.isChecked(),
        )

        if source_mode == ReviewSourceMode.YOLO_PREDICT and self.gpu_guard is not None:
            if not self.gpu_guard.try_acquire(self, "MLLM 基准模型推理审核"):
                return

        self._records.clear()
        self._preview_cache.clear()
        self.result_table.setRowCount(0)
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.send_ds_btn.setEnabled(False)
        self.summary_label.setText("正在运行基准模型推理与 MLLM 纠错清洗...")

        self.worker = MllmReviewWorker(cfg)
        if source_mode == ReviewSourceMode.YOLO_PREDICT and self.gpu_guard is not None:
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

    def _on_progress(self, cur: int, total: int, rec: ImageReviewRecord) -> None:
        """追加单张审核记录到表格并刷新整体进度条。"""
        self.progress_bar.setValue(int(min(100, (cur / max(1, total)) * 100)))
        self._records.append(rec)
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)

        bucket_map = {
            "auto_passed": "自动通过",
            "llm_fixed": "已删改错标",
            "need_review": "待人工复核",
        }
        note = rec.error or rec.missing_note or ("高置信度跳过" if rec.skipped_mllm else "")
        self.result_table.setItem(row, 0, QTableWidgetItem(rec.image_path.name))
        self.result_table.setItem(row, 1, QTableWidgetItem(bucket_map.get(rec.triage_bucket, rec.triage_bucket)))
        self.result_table.setItem(row, 2, QTableWidgetItem(str(rec.kept_count)))
        self.result_table.setItem(row, 3, QTableWidgetItem(str(rec.relabeled_count)))
        self.result_table.setItem(row, 4, QTableWidgetItem(str(rec.deleted_count)))
        self.result_table.setItem(row, 5, QTableWidgetItem(str(rec.added_count)))
        self.result_table.setItem(row, 6, QTableWidgetItem(note))

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

    def _on_row_selected(self, row: int, *_args) -> None:
        if row < 0 or row >= len(self._records):
            return
        rec = self._records[row]
        try:
            jpeg_bytes = self._get_comparison_jpeg(row, rec)
            pix = QPixmap()
            pix.loadFromData(jpeg_bytes, "JPG")
            if not pix.isNull():
                scaled = pix.scaled(
                    max(360, self.preview_label.width() - 10),
                    max(200, self.preview_label.height() - 10),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)
        except Exception as exc:
            self.preview_label.setText(f"预览渲染失败: {exc}")

    def _on_finished(self, res: MllmReviewResult) -> None:
        """恢复界面状态、记录产物路径、更新汇总栏并在存在结果时允许发送至数据集页。"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._last_output_dir = res.output_dir
        self._last_classes_txt = str(res.classes_txt_path) if res.classes_txt_path else ""
        self.send_ds_btn.setEnabled(res.total_images > 0)

        state = "已停止" if res.cancelled else "清洗完成"
        self.summary_label.setText(
            f"{state}（共 {res.total_images} 张）：自动通过 {res.auto_passed_count} | "
            f"已删改错标 {res.llm_fixed_count} | 待人工复核 {res.need_review_count} | 出错 {res.error_count}"
        )
        if res.report_path:
            self.log_box.appendPlainText(f"\n审核报告已保存: {res.report_path}")

    def _on_failed(self, err: str) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.summary_label.setText(f"执行失败: {err}")
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
        out_dir = str(self._last_output_dir)
        cls_path = self._last_classes_txt
        if not cls_path and (self._last_output_dir / "classes.txt").is_file():
            cls_path = str(self._last_output_dir / "classes.txt")
        self.send_cleaned_dataset.emit(out_dir, cls_path, self.triage_dirs_chk.isChecked())
