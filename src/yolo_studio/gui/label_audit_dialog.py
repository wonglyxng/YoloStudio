"""标注工作台「模型反查脏标注 / 漏标」配置与后台扫描对话框。

职责：
    面向「2. 标注工作台」提供轻量级迷你 FiftyOne 入口：选择已训练的 YOLO `.pt` 权重与
    敏感度阈值后，在后台 `QThread` 中调用 `core.label_auditor.audit_annotations` 扫描当前目录，
    实时汇报扫描进度与四类异常计数（疑似漏标 / 疑似错类 / 疑似误标 / 疑似偏移），
    并在扫描完成后将 `LabelAuditResult` 回传给工作台按异常度排序精修。

运行环境与协作边界：
    运行于 PyQt6 主线程，遵守 `GpuTaskGuard` 互斥保护以免与训练/评估抢占 GPU 显存；
    关闭弹窗前自动取消并等待后台扫描线程安全收尾；仅执行只读模型推理与标注比对，
    绝不直接修改任何磁盘 `.json` 或图片文件。
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Callable

from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.label_auditor import (
    LabelAuditConfig,
    LabelAuditResult,
    PredictedBox,
    audit_annotations,
)
from yolo_studio.gui.common import (
    GpuTaskGuard,
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    install_combobox_wheel_guard,
)


class _LabelAuditWorker(QThread):
    """在后台线程中运行本地 YOLO 权重反查标注集，支持随时安全取消与逐张进度回报。"""

    progress = Signal(int, int, str)  # 参数：(当前序号, 总数, 当前图片文件名)
    finished_result = Signal(object)  # 参数：LabelAuditResult
    failed = Signal(str)

    def __init__(
        self,
        config: LabelAuditConfig,
        *,
        predictor_override: Callable[[Path], tuple[list[PredictedBox], list[str]]] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._predictor_override = predictor_override
        self._cancel_event = Event()

    def cancel(self) -> None:
        """请求安全中止后续图片反查（可重复调用）。"""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = audit_annotations(
                self.config,
                predictor_override=self._predictor_override,
                progress_callback=lambda cur, total, p: self.progress.emit(cur, total, p.name),
                cancel_event=self._cancel_event,
            )
            self.finished_result.emit(result)
        except Exception as exc:
            self.failed.emit(f"模型反查执行失败：{exc}")


class LabelAuditDialog(QDialog):
    """模型反查脏标注 / 漏标配置与进度弹窗。

    返回值与数据获取：
        当用户完成扫描并点击「在工作台按异常排序查看」确认后，弹窗以 `Accepted` 结束，
        调用方可通过 `audit_result` 属性获取 `LabelAuditResult` 并激活工作台异常排序视图。
    """

    def __init__(
        self,
        images: list[Path],
        workspace_root: Path,
        *,
        default_weights: str = "",
        device: str = "0",
        gpu_guard: GpuTaskGuard | None = None,
        predictor_override: Callable[[Path], tuple[list[PredictedBox], list[str]]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        install_combobox_wheel_guard()
        self._images = [p.resolve() for p in images]
        self.workspace_root = workspace_root
        self._device = device
        self.gpu_guard = gpu_guard
        self._predictor_override = predictor_override
        self.worker: _LabelAuditWorker | None = None
        self.audit_result: LabelAuditResult | None = None

        self.setWindowTitle("模型反查脏标注与漏标")
        self.setMinimumWidth(520)
        self.resize(560, 390)

        self._init_ui(default_weights)

    def _init_ui(self, default_weights: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        intro = QLabel(
            f"使用本地已训练的 YOLO 权重扫描当前目录（共 {len(self._images)} 张图片），"
            "自动定位疑似漏标、疑似错类、疑似误标与偏移框，并按异常严重度置顶排序到工作台精修。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #57606a; font-size: 12px;")
        root.addWidget(intro)

        form = QFormLayout()
        configure_form_layout(form, vertical_spacing=8)

        self.weights_picker = PathPickerRow(
            placeholder="选择已训练的 best.pt 权重（训练完成后自动同步）",
            mode="file",
            file_filter="PyTorch 权重 (*.pt);;所有文件 (*.*)",
            default_path=default_weights or str((self.workspace_root / "runs").resolve()),
        )
        if default_weights and Path(default_weights).is_file():
            self.weights_picker.setText(default_weights)
        form.addRow("反查模型 (.pt):", self.weights_picker)

        self.only_annotated_chk = QCheckBox("仅检查已有 .json 标注的图片（推荐，聚焦脏标注与漏标）")
        self.only_annotated_chk.setChecked(True)
        self.only_annotated_chk.setToolTip(
            "勾选时仅扫描已生成 .json 的图片（含 0 框空图）；取消勾选则连同未标注图片一并扫描疑似漏标"
        )
        form.addRow("扫描范围:", self.only_annotated_chk)

        self.missing_conf_spin = QDoubleSpinBox()
        self.missing_conf_spin.setRange(0.10, 0.95)
        self.missing_conf_spin.setSingleStep(0.05)
        self.missing_conf_spin.setDecimals(2)
        self.missing_conf_spin.setValue(0.50)
        self.missing_conf_spin.setMinimumWidth(100)
        self.missing_conf_spin.setToolTip("模型预测置信度不低于该值时，才报告疑似漏标、错类或偏移框")
        form.addRow("高置信报警阈值:", self.missing_conf_spin)

        self.pred_conf_spin = QDoubleSpinBox()
        self.pred_conf_spin.setRange(0.05, 0.80)
        self.pred_conf_spin.setSingleStep(0.05)
        self.pred_conf_spin.setDecimals(2)
        self.pred_conf_spin.setValue(0.25)
        self.pred_conf_spin.setMinimumWidth(100)
        self.pred_conf_spin.setToolTip("推理最低保留阈值：若某人工框在此阈值下仍无任何模型预测重叠，记为疑似误标")
        form.addRow("误标排查阈值:", self.pred_conf_spin)

        self.match_iou_spin = QDoubleSpinBox()
        self.match_iou_spin.setRange(0.25, 0.85)
        self.match_iou_spin.setSingleStep(0.05)
        self.match_iou_spin.setDecimals(2)
        self.match_iou_spin.setValue(0.50)
        self.match_iou_spin.setMinimumWidth(100)
        self.match_iou_spin.setToolTip("人工框与模型框认定空间重合的 IoU 阈值；低于该值且 >= 0.20 的同类高置信框视为偏移")
        form.addRow("空间匹配 IoU:", self.match_iou_spin)

        self.imgsz_combo = QComboBox()
        self.imgsz_combo.setEditable(True)
        for sz in ("320", "480", "640", "800", "960", "1280"):
            self.imgsz_combo.addItem(sz)
        self.imgsz_combo.setCurrentText("640")
        self.imgsz_combo.setMinimumWidth(100)
        form.addRow("推理尺寸 (imgsz):", self.imgsz_combo)

        align_form_labels([form])
        root.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(1, len(self._images)))
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        root.addWidget(self.progress_bar)

        self.summary_label = QLabel("就绪：选择权重后点击「开始扫描反查」。")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #24292f; font-size: 12px;")
        root.addWidget(self.summary_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("开始扫描反查")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.apply_btn = QPushButton("在工作台查看异常排序")
        self.apply_btn.setObjectName("primaryBtn")
        self.apply_btn.setEnabled(False)
        self.close_btn = QPushButton("关闭")

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.apply_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)

    def _set_running(self, running: bool) -> None:
        self.weights_picker.setEnabled(not running)
        self.only_annotated_chk.setEnabled(not running)
        self.missing_conf_spin.setEnabled(not running)
        self.pred_conf_spin.setEnabled(not running)
        self.match_iou_spin.setEnabled(not running)
        self.imgsz_combo.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.apply_btn.setEnabled(False)

    def _parse_imgsz(self) -> int:
        raw = self.imgsz_combo.currentText().strip()
        try:
            val = int(raw)
        except ValueError:
            val = 640
        return max(160, min(2560, val))

    def _on_start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self._images:
            QMessageBox.warning(self, "无图片可反查", "当前标注目录暂无图片，请先打开图片目录。")
            return

        weights_raw = self.weights_picker.text().strip()
        weights_path = Path(weights_raw) if weights_raw else Path()
        if self._predictor_override is None and (not weights_raw or not weights_path.is_file()):
            QMessageBox.warning(
                self,
                "请选择模型权重",
                "请先选择有效的已训练 YOLO `.pt` 权重文件（如 runs/.../weights/best.pt）。",
            )
            return

        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "工作台模型反查脏标注"):
            return

        cfg = LabelAuditConfig(
            weights_path=weights_path,
            images=list(self._images),
            only_annotated=self.only_annotated_chk.isChecked(),
            pred_conf=float(self.pred_conf_spin.value()),
            missing_conf=float(self.missing_conf_spin.value()),
            match_iou=float(self.match_iou_spin.value()),
            imgsz=self._parse_imgsz(),
            device=self._device,
        )
        self.audit_result = None
        self.progress_bar.setRange(0, max(1, len(self._images)))
        self.progress_bar.setValue(0)
        self.summary_label.setStyleSheet("color: #0969da; font-size: 12px;")
        self.summary_label.setText("正在加载模型并扫描标注目录...")
        self._set_running(True)

        self.worker = _LabelAuditWorker(cfg, predictor_override=self._predictor_override)
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.worker, "工作台模型反查脏标注")
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.stop_btn.setEnabled(False)
            self.summary_label.setText("正在停止反查并汇总已扫描结果...")

    def _on_progress(self, cur: int, total: int, name: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(cur)
        self.summary_label.setText(f"正在反查 ({cur}/{total})：{name}")

    def _on_finished(self, result: Any) -> None:
        self._set_running(False)
        if not isinstance(result, LabelAuditResult):
            return
        self.audit_result = result
        if result.scanned_images == 0:
            self.summary_label.setStyleSheet("color: #9a6700; font-size: 12px;")
            self.summary_label.setText(
                f"未找到可反查的图片（已跳过 {result.skipped_unannotated} 张未生成 .json 的图片）。"
                "如需检查未标图片，请取消勾选「仅检查已有 .json 标注的图片」。"
            )
            self.apply_btn.setEnabled(False)
            return

        status_prefix = "已手动停止 · " if result.cancelled else "反查完成 · "
        if result.flagged_images == 0:
            self.summary_label.setStyleSheet("color: #1a7f37; font-size: 12px; font-weight: 500;")
            self.summary_label.setText(
                f"{status_prefix}共扫描 {result.scanned_images} 张图片，未发现高置信疑似漏标、错类或误标 ✓"
            )
            self.apply_btn.setEnabled(True)
            return

        self.summary_label.setStyleSheet("color: #9a6700; font-size: 12px; font-weight: 600;")
        self.summary_label.setText(
            f"{status_prefix}共扫描 {result.scanned_images} 张，检出 {result.flagged_images} 张疑似异常图："
            f"漏标 {result.total_missing} · 错类 {result.total_wrong_label} · "
            f"误标 {result.total_spurious} · 偏移 {result.total_misaligned}"
        )
        self.apply_btn.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.summary_label.setStyleSheet("color: #cf222e; font-size: 12px;")
        self.summary_label.setText(msg)
        QMessageBox.warning(self, "模型反查失败", msg)

    def _stop_and_wait_worker(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)

    def done(self, result: int) -> None:  # type: ignore[override]
        self._stop_and_wait_worker()
        super().done(result)

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._stop_and_wait_worker()
        super().closeEvent(event)
