"""标注工作台「SAM 2.1 连续帧向后追踪」配置与后台进度弹窗。

职责：
    在标注工作台中标好当前起始帧后，将当前帧全部框或单个选中框通过本地 SAM 2.1 视觉记忆库
    沿图片列表向后连续追踪传播 `N` 帧，支持配置帧数快捷预设、已有标注冲突合并策略与随时停止。

运行环境与协作边界：
    运行于 PyQt6 主线程，后台由 `_SamTrackWorker(QThread)` 执行 `core.sam_tracker.propagate_boxes_with_sam2`
    并遵守 `GpuTaskGuard` 互斥保护；关闭窗口前自动取消并安全等待后台线程收尾。
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal as Signal
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.annotation_io import RectAnnotation
from yolo_studio.core.sam_tracker import (
    SamTrackConfig,
    SamTrackResult,
    TrackConflictPolicy,
    is_sam2_compatible_weight,
    propagate_boxes_with_sam2,
)
from yolo_studio.gui.common import (
    GpuTaskGuard,
    align_form_labels,
    configure_form_layout,
    install_combobox_wheel_guard,
)


class _SamTrackWorker(QThread):
    """在后台线程中运行 SAM 2.1 连续帧追踪传播，支持实时进度上报与安全中止。"""

    progress = Signal(int, int, str)  # 参数：(当前帧序号, 总目标帧数, 状态文案)
    finished_result = Signal(object)  # 参数：SamTrackResult
    failed = Signal(str)

    def __init__(
        self,
        config: SamTrackConfig,
        *,
        tracker_override: Callable[
            [Path, list[Path], list[RectAnnotation]],
            list[list[tuple[float, float, float, float] | None]],
        ]
        | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._tracker_override = tracker_override
        self._cancel_event = Event()

    def cancel(self) -> None:
        """请求中止后续帧追踪（已写入磁盘的帧保留，可重复调用）。"""
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = propagate_boxes_with_sam2(
                self.config,
                tracker_override=self._tracker_override,
                progress_callback=lambda cur, total, p, stage: self.progress.emit(
                    cur, total, f"{stage}：{p.name}"
                ),
                cancel_event=self._cancel_event,
            )
            self.finished_result.emit(result)
        except Exception as exc:
            self.failed.emit(f"SAM 2.1 连续帧追踪失败：{exc}")


class SamTrackDialog(QDialog):
    """SAM 2.1 连续帧向后追踪配置与进度对话框。

    属性约定：
        追踪完成后（含中途停止但已写入部分帧），调用方可通过 `track_result` 读取 `SamTrackResult`
        并刷新工作台文件列表状态。
    """

    def __init__(
        self,
        start_image: Path,
        following_images: list[Path],
        all_boxes: list[RectAnnotation],
        *,
        selected_box: RectAnnotation | None = None,
        prefer_selected: bool = False,
        sam_model_name: str = "sam2.1_t.pt",
        device: str = "0",
        gpu_guard: GpuTaskGuard | None = None,
        tracker_override: Callable[
            [Path, list[Path], list[RectAnnotation]],
            list[list[tuple[float, float, float, float] | None]],
        ]
        | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        install_combobox_wheel_guard()
        self._start_image = start_image.resolve()
        self._following_images = [p.resolve() for p in following_images]
        self._all_boxes = [b.clone() for b in all_boxes]
        self._selected_box = selected_box.clone() if selected_box is not None else None
        self._sam_model_name = (sam_model_name or "").strip() or "sam2.1_t.pt"
        self._device = device
        self.gpu_guard = gpu_guard
        self._tracker_override = tracker_override
        self.worker: _SamTrackWorker | None = None
        self.track_result: SamTrackResult | None = None

        self.setWindowTitle("SAM 2.1 连续帧向后追踪")
        self.setMinimumWidth(500)
        self.resize(540, 340)

        self._init_ui(prefer_selected)

    def _init_ui(self, prefer_selected: bool) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        max_frames = len(self._following_images)
        intro = QLabel(
            f"以当前帧「{self._start_image.name}」为视觉记忆起点，沿图片列表向后自动追踪传播目标框"
            f"（后续还有 {max_frames} 帧；适合连续抽帧场景，目标出画自动停框，切镜头自动防爆框）。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #57606a; font-size: 12px;")
        root.addWidget(intro)

        form = QFormLayout()
        configure_form_layout(form, vertical_spacing=8)

        self.seed_scope_combo = QComboBox()
        self.seed_scope_combo.addItem(f"当前图全部目标框 ({len(self._all_boxes)} 个)", "all")
        if self._selected_box is not None:
            self.seed_scope_combo.addItem(
                f"仅当前选中框（「{self._selected_box.label}」）", "selected"
            )
            if prefer_selected:
                self.seed_scope_combo.setCurrentIndex(1)
        form.addRow("追踪目标:", self.seed_scope_combo)

        count_row = QHBoxLayout()
        count_row.setSpacing(5)
        self.frame_count_spin = QSpinBox()
        self.frame_count_spin.setRange(1, max(1, max_frames))
        self.frame_count_spin.setValue(min(10, max(1, max_frames)))
        self.frame_count_spin.setMinimumWidth(88)
        self.frame_count_spin.setSuffix(" 帧")
        count_row.addWidget(self.frame_count_spin)
        for preset_n in (5, 10, 20):
            if preset_n <= max_frames:
                btn = QPushButton(f"{preset_n}")
                btn.setFixedWidth(36)
                btn.setStyleSheet("padding: 2px 4px; font-size: 11px;")
                btn.clicked.connect(lambda _c=False, n=preset_n: self.frame_count_spin.setValue(n))
                count_row.addWidget(btn)
        if max_frames > 0:
            all_btn = QPushButton(f"全部({max_frames})")
            all_btn.setStyleSheet("padding: 2px 6px; font-size: 11px;")
            all_btn.clicked.connect(lambda: self.frame_count_spin.setValue(max_frames))
            count_row.addWidget(all_btn)
        count_row.addStretch(1)
        form.addRow("向后追踪帧数:", count_row)

        self.conflict_combo = QComboBox()
        self.conflict_combo.addItem(
            "追加不重叠框（推荐，保留目标帧原有标注）", TrackConflictPolicy.APPEND_NON_OVERLAP
        )
        self.conflict_combo.addItem(
            "跳过已有标注帧（不改动已有框，但维持记忆不断链）", TrackConflictPolicy.SKIP_LABELED
        )
        self.conflict_combo.addItem(
            "覆盖已有矩形框（用本轮追踪结果替换）", TrackConflictPolicy.OVERWRITE
        )
        form.addRow("已有标注处理:", self.conflict_combo)

        self.model_info_label = QLabel(
            f"{Path(self._sam_model_name).name} · 设备: {self._device}（可在右上角「设置 → 本地视觉模型」更换）"
        )
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: #57606a; font-size: 12px;")
        form.addRow("使用模型:", self.model_info_label)

        align_form_labels([form])
        root.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(1, self.frame_count_spin.value()))
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m 帧")
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪：确认帧数与冲突策略后点击「开始向后追踪」。")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #24292f; font-size: 12px;")
        root.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("开始向后追踪")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.done_btn = QPushButton("完成并返回工作台")
        self.done_btn.setObjectName("primaryBtn")
        self.done_btn.setVisible(False)
        self.close_btn = QPushButton("关闭")

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.done_btn)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.done_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self._on_close_clicked)

    def _set_running(self, running: bool) -> None:
        self.seed_scope_combo.setEnabled(not running)
        self.frame_count_spin.setEnabled(not running)
        self.conflict_combo.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _selected_seed_boxes(self) -> list[RectAnnotation]:
        if self.seed_scope_combo.currentData() == "selected" and self._selected_box is not None:
            return [self._selected_box.clone()]
        return [b.clone() for b in self._all_boxes]

    def _on_start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        seeds = self._selected_seed_boxes()
        if not seeds or not self._following_images:
            QMessageBox.warning(self, "无法开始追踪", "当前帧缺少可追踪目标框，或后续已无更多图片。")
            return
        if self._tracker_override is None and not is_sam2_compatible_weight(self._sam_model_name):
            QMessageBox.warning(
                self,
                "需要 SAM 2.1 权重",
                f"当前全局本地 SAM 权重为「{Path(self._sam_model_name).name}」，"
                "连续帧记忆追踪仅支持 SAM 2 / SAM 2.1 系列（如 sam2.1_t.pt / sam2.1_s.pt）。\n"
                "请先在右上角「设置 → 本地视觉模型」中切换。",
            )
            return
        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "工作台 SAM 2.1 连续帧追踪"):
            return

        count = min(self.frame_count_spin.value(), len(self._following_images))
        targets = self._following_images[:count]
        policy: TrackConflictPolicy = (
            self.conflict_combo.currentData() or TrackConflictPolicy.APPEND_NON_OVERLAP
        )
        cfg = SamTrackConfig(
            start_image=self._start_image,
            target_images=targets,
            seed_boxes=seeds,
            model_name=self._sam_model_name,
            device=self._device,
            conflict_policy=policy,
        )
        self.track_result = None
        self.done_btn.setVisible(False)
        self.progress_bar.setRange(0, max(1, count))
        self.progress_bar.setValue(0)
        self.status_label.setStyleSheet("color: #0969da; font-size: 12px;")
        self.status_label.setText(f"正在初始化 SAM 2.1 记忆库并向后追踪 {count} 帧...")
        self._set_running(True)

        self.worker = _SamTrackWorker(cfg, tracker_override=self._tracker_override)
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.worker, "工作台 SAM 2.1 连续帧追踪")
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("正在停止追踪并汇总已写入的帧...")

    def _on_progress(self, cur: int, total: int, text: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(cur)
        self.status_label.setText(text)

    def _on_finished(self, result: Any) -> None:
        self._set_running(False)
        if not isinstance(result, SamTrackResult):
            return
        self.track_result = result
        prefix = "已中止 · " if result.cancelled else "追踪完成 · "
        self.status_label.setStyleSheet("color: #1a7f37; font-size: 12px; font-weight: 600;")
        self.status_label.setText(
            f"{prefix}共处理 {result.processed_frames} 帧：写入 {result.written_frames} 帧"
            f"（共新增/更新 {result.total_added_boxes} 个框），跳过 {result.skipped_frames} 帧，"
            f"目标出画/未检出 {result.lost_frames} 帧。"
        )
        self.done_btn.setVisible(True)
        self.close_btn.setVisible(False)

    def _on_failed(self, msg: str) -> None:
        self._set_running(False)
        self.status_label.setStyleSheet("color: #cf222e; font-size: 12px;")
        self.status_label.setText(msg)
        QMessageBox.warning(self, "SAM 2.1 连续帧追踪失败", msg)

    def _on_close_clicked(self) -> None:
        if self.track_result is not None and self.track_result.written_frames > 0:
            self.accept()
        else:
            self.reject()

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
