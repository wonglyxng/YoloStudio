"""PyQt6 界面共享辅助控件、跨页计算任务互斥协调器与本地路径/图像工具函数。

职责：
    提供路径选择器行控件、友好型 GPU/CPU 计算设备切换控件、表单标签对齐与间距规范工具、
    系统文件管理器跳转、等比缩略图加载、紧凑型指标卡片构建，
    以及防止多标签页同时启动本地模型任务争抢显存的 `GpuTaskGuard`。

运行环境与协作边界：
    Python >= 3.11 + `PyQt6`；供 `gui` 各标签页与 `MainWindow` 复用，不持有模块级可变全局状态，也不自行创建后台线程。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QObject, QThread, Qt, QUrl, pyqtSignal as Signal
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QWidget,
)

if TYPE_CHECKING:
    from yolo_studio.core.trainer import DeviceInfo

# 长任务日志框（`QPlainTextEdit.setMaximumBlockCount`）保留的最大行数；超出后自动丢弃最早的行，
# 避免长时间训练日志无限增长拖慢界面
LOG_MAX_LINES = 5000

# 表单标签与右侧字段控件（下拉框、输入框、微调框等）之间的默认水平间距（像素）
FORM_LABEL_SPACING = 10


def configure_form_layout(
    form: QFormLayout,
    *,
    horizontal_spacing: int = FORM_LABEL_SPACING,
    vertical_spacing: int | None = None,
    margins: tuple[int, int, int, int] | None = None,
) -> None:
    """统一表单标签对齐方式与列间距，使每行标签紧邻其右侧控件保持合适距离。

    设计原因：
        Windows 原生 `QFormLayout` 默认将标签左对齐；当同一分组存在较长标签（如 `最长边(0=原尺寸):`、
        `classes.txt(可选):`）时，短标签与右侧下拉框/输入框之间会被拉出大片空白。
        改为右对齐并设定固定 `horizontal_spacing` 后，所有标签右端均与字段左边缘保持恒定距离。
    """
    form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    form.setHorizontalSpacing(horizontal_spacing)
    if vertical_spacing is not None:
        form.setVerticalSpacing(vertical_spacing)
    if margins is not None:
        form.setContentsMargins(*margins)


class _FormLabelAligner(QObject):
    """监听样式表/字体继承变化并自动重算跨分组 `QFormLayout` 统一标签列宽的事件过滤器。

    设计原因：
        各标签页在 `__init__` 构造表单时尚未挂载到 `MainWindow`，此时标签尚未继承主窗口样式表中的字号；
        挂入 `MainWindow` 并首次完成样式应用（`Polish` / `StyleChange` / `FontChange`）后，字体宽度会变化。
        本对象绑定到 `forms[0]` 的生命周期，在字体或样式生效后按最新 `sizeHint` 自动重算并保持像素级对齐。
    """

    _TRIGGER_EVENTS = (
        QEvent.Type.Polish,
        QEvent.Type.FontChange,
        QEvent.Type.StyleChange,
    )

    def __init__(self, forms: Sequence[QFormLayout], extra_padding: int = 2) -> None:
        super().__init__(forms[0])
        self._forms = tuple(forms)
        self._extra_padding = max(0, extra_padding)
        self._syncing = False
        self._last_width = -1

        watched: set[int] = set()
        for form in self._forms:
            parent_widget = form.parentWidget()
            if parent_widget is not None and id(parent_widget) not in watched:
                parent_widget.installEventFilter(self)
                watched.add(id(parent_widget))

    def sync(self) -> int:
        """基于当前字体/样式计算并同步全部关联表单的非空标签最小宽度，返回最新宽度（像素）。"""
        if self._syncing:
            return max(0, self._last_width)
        self._syncing = True
        try:
            labels: list[QLabel] = []
            for form in self._forms:
                for row in range(form.rowCount()):
                    item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    widget = item.widget() if item is not None else None
                    if isinstance(widget, QLabel) and widget.text():
                        labels.append(widget)

            if not labels:
                self._last_width = 0
                return 0

            # 先清零旧 minimumWidth，防止上一次对齐值污染当前字体下的自然宽度测量
            for label in labels:
                label.setMinimumWidth(0)
            max_width = max(label.sizeHint().width() for label in labels)
            if max_width <= 0:
                self._last_width = 0
                return 0

            target_width = max_width + self._extra_padding
            for label in labels:
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                label.setMinimumWidth(target_width)
            self._last_width = target_width
            return target_width
        finally:
            self._syncing = False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() in self._TRIGGER_EVENTS:
            self.sync()
        return super().eventFilter(watched, event)


def align_form_labels(forms: Sequence[QFormLayout], *, extra_padding: int = 2) -> int:
    """将同页多个 `QFormLayout` 的非空行标签统一为一致的右对齐列宽并返回当前最小宽度（像素）。

    调用时机与约束：
        应在相关 `QFormLayout` 的全部表单行添加完成后调用（支持包含当前处于隐藏状态的可切换行，
        如 `Base URL:`、`SAM 分割模型:`），避免用户切换模式显示/隐藏行时整个表单左右抖动；
        内部会挂接 `_FormLabelAligner` 在样式表与字体继承生效后自动重算，确保跨分组控件左边缘精确对齐。
    """
    valid_forms = [f for f in forms if f is not None]
    if not valid_forms:
        return 0
    aligner = _FormLabelAligner(valid_forms, extra_padding=extra_padding)
    return aligner.sync()


def open_local_path(path: Path) -> None:
    """在系统文件管理器中打开本地目录或文件；若目标不存在则回退打开其父目录。"""
    target = path if path.exists() else path.parent
    if target.exists():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))


def load_pixmap_scaled(image_path: Path, max_w: int, max_h: int) -> QPixmap | None:
    """加载本地图像文件并按宽高比缩放至不超过 `(max_w, max_h)` 像素；无效时返回 `None`。"""
    if not image_path.is_file():
        return None
    pixmap = QPixmap(str(image_path))
    if pixmap.isNull():
        return None
    return pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


class PathPickerRow(QWidget):
    """包含路径输入框与“浏览...”按钮的水平复合控件，支持 `dir` 目录模式与文件选择模式。

    长路径显示约定：
        通过 `setText` 或“浏览...”写入路径后，光标统一回到开头，使各输入框都从盘符开始显示，
        避免有的显示开头、有的被滚动到结尾；输入框 tooltip 始终同步为完整路径，便于查看被截断的部分。
    """

    textChanged = Signal(str)  # 输入框文本变化（含程序写入与用户编辑），参数为未去空白的原始文本

    def __init__(
        self,
        placeholder: str = "",
        mode: str = "dir",
        file_filter: str = "All Files (*.*)",
        default_path: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.file_filter = file_filter

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.line_edit = QLineEdit(default_path)
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.setCursorPosition(0)
        self.line_edit.setToolTip(default_path)
        self.line_edit.textChanged.connect(self._on_text_changed)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setFixedWidth(76)
        self.browse_btn.clicked.connect(self._on_browse)

        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_btn)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def setText(self, value: str) -> None:
        """程序写入路径并把光标移回开头，使长路径从盘符开始显示。"""
        self.line_edit.setText(value)
        self.line_edit.setCursorPosition(0)

    def setPlaceholderText(self, value: str) -> None:
        self.line_edit.setPlaceholderText(value)

    def path(self) -> Path:
        return Path(self.text())

    def _on_text_changed(self, value: str) -> None:
        self.line_edit.setToolTip(value.strip())
        self.textChanged.emit(value)

    def _on_browse(self) -> None:
        start_dir = self.text() or str(Path.cwd())
        if self.mode == "dir":
            chosen = QFileDialog.getExistingDirectory(self, "选择文件夹", start_dir)
        else:
            chosen, _ = QFileDialog.getOpenFileName(self, "选择文件", start_dir, self.file_filter)
        if chosen:
            self.setText(chosen)


def make_metric_card(title: str, initial_value: str = "-") -> tuple[QWidget, QLabel]:
    """创建紧凑型训练指标展示卡片并返回 `(卡片控件, 数值标签)`。"""
    card = QWidget()
    card.setObjectName("metricCard")
    layout = QHBoxLayout(card)
    layout.setContentsMargins(10, 6, 10, 6)

    title_lbl = QLabel(f"{title}:")
    title_lbl.setStyleSheet("color: #666666; font-weight: 500;")
    val_lbl = QLabel(initial_value)
    val_lbl.setStyleSheet("font-weight: 700; font-size: 13px;")
    layout.addWidget(title_lbl)
    layout.addStretch(1)
    layout.addWidget(val_lbl)
    return card, val_lbl


class GpuTaskGuard:
    """跨标签页本地重计算 / GPU 模型任务互斥协调器。

    职责与设计原因：
        训练、基准模型低阈值推理、验证、推理预览与模型导出均会加载本地 PyTorch / Ultralytics 模型；
        若用户在训练未结束时切换到其他标签页启动推理或验证，极易引发 CUDA 显存溢出（OOM）。
        本协调器由 `MainWindow` 创建并注入相关标签页，仅在主线程中调用。

    状态与无死锁保证：
        内部记录最近登记的 `_active_worker` 与任务名 `_active_name`；
        占用判定直接以 `_active_worker.isRunning()` 为准，无需依赖易遗漏的手工释放回调，
        即使后台线程异常退出也不会残留锁定状态。
    """

    def __init__(self) -> None:
        self._active_worker: QThread | None = None
        self._active_name: str = ""

    def active_task_name(self) -> str | None:
        """若当前有登记的后台模型任务仍在运行则返回其名称，否则返回 `None`。"""
        if self._active_worker is not None and self._active_worker.isRunning():
            return self._active_name
        return None

    def try_acquire(self, parent: QWidget, task_name: str) -> bool:
        """在主线程检查是否允许启动 `task_name`；若有冲突任务正在运行则弹窗提示并返回 `False`。"""
        busy_name = self.active_task_name()
        if busy_name is not None:
            QMessageBox.warning(
                parent,
                "计算设备占用中",
                f"当前「{busy_name}」正在后台运行并占用本地计算设备。\n"
                f"请等待其完成或手动停止后再启动「{task_name}」，以免显存溢出。",
            )
            return False
        return True

    def bind(self, worker: QThread, task_name: str) -> None:
        """登记即将启动或刚启动的后台模型工作线程。"""
        self._active_worker = worker
        self._active_name = task_name


_CUDA_INDEX_RE = re.compile(r"^\d+(,\d+)*$")


def _resolve_gpu_list(info: "DeviceInfo") -> list[tuple[str, str, float]]:
    """从 `DeviceInfo` 提取标准化的 `(设备编号, 型号名, 显存 GiB)` 列表；缺少 `gpu_devices` 明细时自动兜底。"""
    if not info.cuda_available or info.device_count <= 0:
        return []
    if info.gpu_devices:
        return list(info.gpu_devices)
    gpus: list[tuple[str, str, float]] = [("0", info.device_name, info.vram_gb)]
    for idx in range(1, info.device_count):
        gpus.append((str(idx), f"CUDA GPU {idx}", 0.0))
    return gpus


class ComputeDeviceControl(QWidget):
    """用户友好的 GPU/CPU 计算设备切换与硬件状态一体化控件（供标签页 3 / 5 / 6 共用）。

    职责与展示模式：
        用直观的「启用 GPU 加速」开关替代晦涩的 `Device: 0` 文本输入框：
        - 单卡环境：仅展示 GPU 开关 + 型号/显存状态（无多余选卡下拉框）；
        - 多卡环境（`device_count > 1`）：勾选 GPU 开关时自动浮现显卡选择下拉框，
          `allow_multi_gpu=True` 时额外提供「全部多卡并行 (0,1,...)」选项；
        - 无 GPU 环境：自动取消勾选并置灰开关，安全回退为 `cpu`；
        - 后台探测期（`probing=True`）：暂时禁用开关并蓝字提示正在检测，探测完毕自动就绪。

    内部状态与并发约束：
        仅在 PyQt6 主线程中访问；通过 `_user_modified`、`_pending_device` 与 `_custom_override`
        保证异步硬件探测完成时既能自动启用本机 GPU，又不会覆写用户已显式指定的 CPU 或自定义设备。
    """

    device_changed = Signal(str)  # 信号参数：切换后最终传给 Ultralytics 的 device 字符串

    def __init__(
        self,
        device_info: "DeviceInfo",
        *,
        show_status_text: bool = True,
        allow_multi_gpu: bool = False,
        checkbox_text: str = "启用 GPU 加速",
        compact_status: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.device_info = device_info
        self._show_status_text = show_status_text
        self._allow_multi_gpu = allow_multi_gpu
        self._compact_status = compact_status
        self._updating: bool = False
        self._user_modified: bool = False
        self._pending_device: str | None = None
        self._custom_override: str = ""
        self._gpu_list: list[tuple[str, str, float]] = []
        self._last_emitted_device: str = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6 if (compact_status and not show_status_text) else 8)

        self.gpu_chk = QCheckBox(checkbox_text)
        self.gpu_chk.setStyleSheet("font-weight: 600;")

        self.gpu_combo = QComboBox()
        self.gpu_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.gpu_combo.setVisible(False)

        self.status_label = QLabel()
        self.status_label.setVisible(show_status_text)

        layout.addWidget(self.gpu_chk)
        if show_status_text:
            self.gpu_combo.setMinimumContentsLength(12)
            layout.addWidget(self.gpu_combo)
            layout.addWidget(self.status_label, 1)
        else:
            # 无状态文案的紧凑模式（如推理参数行）：多卡选卡下拉框按剩余空间弹性缩放，
            # 防止撑出横向滚动条，同时保证下拉弹出列表有足够宽度完整显示显卡型号
            if compact_status:
                self.gpu_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                self.gpu_combo.setMinimumWidth(64)
                self.gpu_combo.setMaximumWidth(180)
                self.gpu_combo.view().setMinimumWidth(260)
            else:
                self.gpu_combo.setMinimumContentsLength(12)
            layout.addWidget(self.gpu_combo, 100)
            layout.addStretch(1)

        self.gpu_chk.toggled.connect(self._on_gpu_toggled)
        self.gpu_combo.currentIndexChanged.connect(self._on_gpu_combo_changed)

        self._rebuild_from_device_info(initial=True)
        self._last_emitted_device = self.device()

    def device(self) -> str:
        """返回最终应传递给 Ultralytics 的计算设备标识（如 `'0'`、`'0,1'` 或 `'cpu'`）。"""
        if self._custom_override:
            return self._custom_override
        if self.device_info.probing:
            if self._pending_device:
                return self._pending_device
            return self.device_info.recommended_device or "cpu"
        if not self.device_info.cuda_available or not self.gpu_chk.isChecked():
            return "cpu"
        if len(self._gpu_list) > 1:
            selected = str(self.gpu_combo.currentData() or "").strip()
            if selected:
                return selected
        return self._gpu_list[0][0] if self._gpu_list else (self.device_info.recommended_device or "0")

    def is_using_gpu(self) -> bool:
        """判断当前生效设备是否为 GPU（非 `cpu`）。"""
        return self.device().strip().lower() != "cpu"

    def is_user_modified(self) -> bool:
        """返回用户或导入流程是否已手动改变过默认设备选择。"""
        return self._user_modified

    def reset_to_recommended(self) -> None:
        """清除手动覆盖并恢复为当前本机硬件推荐设备。"""
        self._user_modified = False
        self._pending_device = None
        self._custom_override = ""
        self._rebuild_from_device_info(initial=False)
        self._emit_if_changed()

    def set_device(self, device_str: str, *, user_action: bool = True) -> None:
        """按 Ultralytics 设备字符串（如 `'0'`、`'cpu'`、`'0,1'`）同步开关与选卡状态。

        调用与兜底约束：
            若仍处于异步硬件探测期（`probing=True`），先暂存至 `_pending_device`，待 `apply_device_info`
            到达后再依据真实硬件落地；若无 CUDA GPU 而传入纯数字 CUDA 卡号，则安全回退为 `'cpu'`。
        """
        dev = str(device_str).strip()
        if not dev:
            self.reset_to_recommended()
            return

        if user_action:
            self._user_modified = True

        if self.device_info.probing:
            self._pending_device = dev
            if dev.lower() != "cpu" and not _CUDA_INDEX_RE.fullmatch(dev):
                self._custom_override = dev
            else:
                self._custom_override = ""
            self._refresh_visuals()
            self._emit_if_changed()
            return

        self._apply_device_string_now(dev)
        self._refresh_visuals()
        self._emit_if_changed()

    def apply_device_info(self, info: "DeviceInfo") -> None:
        """接收最新硬件探测结果，刷新选项与状态文案，同时尊重用户既有选择。"""
        self.device_info = info
        self._rebuild_from_device_info(initial=False)
        self._emit_if_changed()

    def _populate_gpu_combo(self) -> None:
        prev_data = str(self.gpu_combo.currentData() or "")
        self.gpu_combo.blockSignals(True)
        try:
            self.gpu_combo.clear()
            for dev_id, name, vram in self._gpu_list:
                vram_txt = f" ({vram:.1f} GB)" if vram > 0 else ""
                self.gpu_combo.addItem(f"GPU {dev_id}: {name}{vram_txt}", dev_id)
            if self._allow_multi_gpu and len(self._gpu_list) > 1:
                all_ids = ",".join(dev_id for dev_id, _, _ in self._gpu_list)
                self.gpu_combo.addItem(f"全部多卡并行 ({all_ids})", all_ids)
            if prev_data:
                idx = self.gpu_combo.findData(prev_data)
                if idx >= 0:
                    self.gpu_combo.setCurrentIndex(idx)
        finally:
            self.gpu_combo.blockSignals(False)

    def _apply_device_string_now(self, dev: str) -> None:
        """在非探测状态下将具体 `dev` 字符串映射至复选框、下拉框或 `_custom_override`。"""
        self._updating = True
        try:
            if dev.lower() == "cpu":
                self._custom_override = ""
                self.gpu_chk.setChecked(False)
                return

            if self.device_info.cuda_available and self._gpu_list:
                self.gpu_chk.setChecked(True)
                if len(self._gpu_list) == 1 and dev == self._gpu_list[0][0]:
                    self._custom_override = ""
                    return
                idx = self.gpu_combo.findData(dev)
                if idx >= 0:
                    self.gpu_combo.setCurrentIndex(idx)
                    self._custom_override = ""
                    return
                self._custom_override = dev
                return

            # 无可用 CUDA GPU：纯数字 CUDA 索引安全回退为 cpu，非纯数字非常规设备保留自定义覆盖
            self.gpu_chk.setChecked(False)
            if _CUDA_INDEX_RE.fullmatch(dev):
                self._custom_override = ""
            else:
                self._custom_override = dev
        finally:
            self._updating = False

    def _rebuild_from_device_info(self, *, initial: bool) -> None:
        self._gpu_list = _resolve_gpu_list(self.device_info)
        self._populate_gpu_combo()

        self._updating = True
        try:
            if self.device_info.probing:
                self.gpu_chk.setEnabled(False)
                self.gpu_chk.setChecked(False)
            elif not self.device_info.cuda_available:
                self.gpu_chk.setEnabled(False)
                self.gpu_chk.setChecked(False)
                if self._pending_device:
                    pending = self._pending_device
                    self._pending_device = None
                    self._apply_device_string_now(pending)
                elif self._custom_override and _CUDA_INDEX_RE.fullmatch(self._custom_override):
                    self._custom_override = ""
            else:
                self.gpu_chk.setEnabled(True)
                if self._pending_device is not None:
                    pending = self._pending_device
                    self._pending_device = None
                    self._apply_device_string_now(pending)
                elif initial or not self._user_modified:
                    self._custom_override = ""
                    self.gpu_chk.setChecked(True)
                    rec_idx = self.gpu_combo.findData(self.device_info.recommended_device)
                    if rec_idx >= 0:
                        self.gpu_combo.setCurrentIndex(rec_idx)
        finally:
            self._updating = False

        self._refresh_visuals()

    def _on_gpu_toggled(self, checked: bool) -> None:
        if self._updating:
            return
        self._user_modified = True
        self._pending_device = None
        if not checked:
            self._custom_override = ""
        self._refresh_visuals()
        self._emit_if_changed()

    def _on_gpu_combo_changed(self, _index: int) -> None:
        if self._updating:
            return
        self._user_modified = True
        self._pending_device = None
        self._custom_override = ""
        self._refresh_visuals()
        self._emit_if_changed()

    def _refresh_visuals(self) -> None:
        show_combo = (
            not self.device_info.probing
            and self.device_info.cuda_available
            and len(self._gpu_list) > 1
            and self.gpu_chk.isChecked()
            and not self._custom_override
        )
        self.gpu_combo.setVisible(show_combo)

        if self._custom_override:
            text = f"自定义设备: {self._custom_override}"
            color = "#6f42c1"
            tip = f"当前使用自定义设备标识: {self._custom_override}（取消/重选开关可恢复自动识别）"
        elif self.device_info.probing:
            text = (
                "正在检测设备..."
                if self._compact_status
                else "正在后台检测计算设备 (CUDA GPU / CPU)..."
            )
            color = "#0969da"
            tip = "正在后台检测本地 CUDA GPU 与显存信息，完成后将自动就绪"
        elif self.device_info.cuda_available:
            first_name = self.device_info.device_name
            first_vram = self.device_info.vram_gb
            if self.gpu_chk.isChecked():
                color = "#1f7a1f"
                if len(self._gpu_list) > 1:
                    text = (
                        f"CUDA 可用 ({len(self._gpu_list)} 张 GPU)"
                        if self._compact_status
                        else f"已就绪 {len(self._gpu_list)} 张 CUDA GPU | CUDA 可用"
                    )
                    tip = f"当前使用 GPU 设备 ({self.device()})；可从旁侧下拉列表切换显卡"
                else:
                    text = (
                        f"{first_name} ({first_vram:.1f} GB)"
                        if self._compact_status
                        else f"已就绪 GPU: {first_name} ({first_vram:.1f} GB 显存) | CUDA 可用"
                    )
                    tip = f"使用 GPU 加速：{first_name} ({first_vram:.1f} GB 显存，device={self.device()})"
            else:
                color = "#b25900"
                text = (
                    "CPU 模式（已关闭 GPU）"
                    if self._compact_status
                    else "已手动切换为 CPU 模式（速度较慢，重新勾选左侧开关可恢复 GPU 加速）"
                )
                tip = f"当前已关闭 GPU 加速（device=cpu）；本机可用显卡：{first_name} ({first_vram:.1f} GB)"
        else:
            color = "#b25900"
            text = (
                "CPU 模式（无可用 CUDA GPU）"
                if self._compact_status
                else "未检测到可用 CUDA GPU，将使用 CPU 运行（速度较慢）"
            )
            tip = "未检测到可用 CUDA GPU 或当前为 CPU 运行环境，固定使用 CPU 模式（device=cpu）"

        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        self.gpu_chk.setToolTip(tip)
        self.gpu_combo.setToolTip(self.gpu_combo.currentText() or tip)
        self.status_label.setToolTip(tip)

    def _emit_if_changed(self) -> None:
        current = self.device()
        if current != self._last_emitted_device:
            self._last_emitted_device = current
            self.device_changed.emit(current)
