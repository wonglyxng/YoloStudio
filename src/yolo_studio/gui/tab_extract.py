"""标签页 1：录制视频抽帧界面。

职责：
    提供视频文件与目录选择、抽帧参数配置、后台抽帧进度与实时缩略图预览，
    并在抽帧完成或手动点击转发时向主窗口派发输出目录信号。

运行环境与协作边界：
    运行于 PyQt6 主线程；耗时抽帧委托给 `gui.workers.FrameExtractWorker`（调用
    `core.frame_extractor`），通过 `send_frames_dir` 信号与 `gui.main_window.MainWindow` 协同。
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.frame_extractor import (
    ExtractionConfig,
    ExtractionMode,
    ExtractionProgress,
    ExtractionResult,
)
from yolo_studio.gui.common import PathPickerRow, load_pixmap_scaled, open_local_path
from yolo_studio.gui.workers import FrameExtractWorker, VideoProbeWorker


class ExtractTab(QWidget):
    """视频抽帧功能标签页（标签页 1）。

    内部状态：
        持有 `workspace_root`、当前待处理视频路径列表 `_video_paths`、
        后台视频探测线程 `probe_worker` 以及后台抽帧线程 `worker` 引用。
    """

    send_frames_dir = Signal(str)

    def __init__(self, workspace_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.worker: FrameExtractWorker | None = None
        self.probe_worker: VideoProbeWorker | None = None
        self._video_paths: list[Path] = []
        self._init_ui()

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setSpacing(12)

        # 左侧面板：视频选择与抽帧参数配置
        left = QVBoxLayout()
        left.setSpacing(10)

        video_group = QGroupBox("1. 选择录制视频")
        video_layout = QVBoxLayout(video_group)

        btn_row = QHBoxLayout()
        self.add_files_btn = QPushButton("添加视频文件")
        self.add_folder_btn = QPushButton("添加视频文件夹")
        self.remove_btn = QPushButton("移除选中")
        self.clear_btn = QPushButton("清空列表")
        btn_row.addWidget(self.add_files_btn)
        btn_row.addWidget(self.add_folder_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.clear_btn)
        video_layout.addLayout(btn_row)

        self.video_list = QListWidget()
        self.video_list.setMinimumHeight(140)
        video_layout.addWidget(self.video_list)
        left.addWidget(video_group)

        param_group = QGroupBox("2. 抽帧参数设置")
        form = QFormLayout(param_group)
        form.setSpacing(8)

        default_frames_dir = str((self.workspace_root / "data" / "raw_frames").resolve())
        self.output_picker = PathPickerRow(
            placeholder="选择抽帧图片输出目录", mode="dir", default_path=default_frames_dir
        )
        form.addRow("输出目录:", self.output_picker)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("按时间间隔抽帧 (每隔 N 秒)", ExtractionMode.INTERVAL_SECONDS)
        self.mode_combo.addItem("按固定帧步长抽帧 (每隔 N 帧)", ExtractionMode.EVERY_N_FRAMES)
        self.mode_combo.addItem("按目标总帧数均匀抽帧", ExtractionMode.TARGET_COUNT)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("抽帧策略:", self.mode_combo)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.05, 3600.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" 秒")
        form.addRow("时间间隔:", self.interval_spin)

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 100000)
        self.frame_step_spin.setValue(30)
        self.frame_step_spin.setSuffix(" 帧")
        form.addRow("帧步长:", self.frame_step_spin)

        self.target_count_spin = QSpinBox()
        self.target_count_spin.setRange(1, 100000)
        self.target_count_spin.setValue(100)
        self.target_count_spin.setSuffix(" 张/视频")
        form.addRow("目标帧数:", self.target_count_spin)

        time_row = QHBoxLayout()
        self.start_sec_spin = QDoubleSpinBox()
        self.start_sec_spin.setRange(0.0, 86400.0)
        self.start_sec_spin.setValue(0.0)
        self.start_sec_spin.setSuffix(" 秒")
        self.end_sec_spin = QDoubleSpinBox()
        self.end_sec_spin.setRange(0.0, 86400.0)
        self.end_sec_spin.setValue(0.0)
        self.end_sec_spin.setSuffix(" 秒 (0=到结尾)")
        time_row.addWidget(QLabel("起:"))
        time_row.addWidget(self.start_sec_spin)
        time_row.addWidget(QLabel("止:"))
        time_row.addWidget(self.end_sec_spin)
        form.addRow("截取时间段:", time_row)

        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.0, 50.0)
        self.dedup_spin.setSingleStep(0.5)
        self.dedup_spin.setValue(2.0)
        self.dedup_spin.setToolTip("相邻保存帧灰度平均像素差低于该值时自动跳过，0 表示不去重")
        form.addRow("相似帧去重阈值:", self.dedup_spin)

        resize_row = QHBoxLayout()
        self.max_dim_spin = QSpinBox()
        self.max_dim_spin.setRange(0, 8192)
        self.max_dim_spin.setSingleStep(160)
        self.max_dim_spin.setValue(0)
        self.max_dim_spin.setToolTip("限制导出图片最长边像素，0 表示保持视频原始分辨率")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpg", "png"])
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(10, 100)
        self.quality_spin.setValue(95)
        resize_row.addWidget(self.max_dim_spin)
        resize_row.addWidget(QLabel("格式:"))
        resize_row.addWidget(self.format_combo)
        resize_row.addWidget(QLabel("JPG质量:"))
        resize_row.addWidget(self.quality_spin)
        form.addRow("最长边(0=原尺寸):", resize_row)

        naming_row = QHBoxLayout()
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("可选文件名前缀，如 cam1_")
        self.subfolder_chk = QCheckBox("按视频名分建子目录")
        naming_row.addWidget(self.prefix_edit, 1)
        naming_row.addWidget(self.subfolder_chk)
        form.addRow("命名选项:", naming_row)

        left.addWidget(param_group)

        action_row = QHBoxLayout()
        self.start_btn = QPushButton("开始抽帧")
        self.start_btn.setObjectName("primaryBtn")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.open_dir_btn = QPushButton("打开输出目录")
        self.send_next_btn = QPushButton("发送到「2. 标注工作台」→")
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.stop_btn)
        action_row.addWidget(self.open_dir_btn)
        action_row.addWidget(self.send_next_btn)
        left.addLayout(action_row)

        # 右侧面板：实时进度条与抽帧预览
        right = QVBoxLayout()
        right.setSpacing(10)

        preview_group = QGroupBox("3. 实时抽帧预览与进度")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_label = QLabel("抽帧开始后将在此显示最新提取的画面")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(360, 260)
        self.preview_label.setStyleSheet(
            "border: 1px dashed #999999; border-radius: 6px; background: #1e1e1e; color: #cccccc;"
        )
        preview_layout.addWidget(self.preview_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        preview_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("就绪：请添加录制视频后点击「开始抽帧」。")
        self.status_label.setWordWrap(True)
        preview_layout.addWidget(self.status_label)

        right.addWidget(preview_group)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        # 信号与槽连接
        self.add_files_btn.clicked.connect(self._on_add_files)
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.clear_btn.clicked.connect(self._on_clear_videos)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_dir_btn.clicked.connect(lambda: open_local_path(self.output_picker.path()))
        self.send_next_btn.clicked.connect(lambda: self.send_frames_dir.emit(self.output_picker.text()))
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        self.interval_spin.setEnabled(mode == ExtractionMode.INTERVAL_SECONDS)
        self.frame_step_spin.setEnabled(mode == ExtractionMode.EVERY_N_FRAMES)
        self.target_count_spin.setEnabled(mode == ExtractionMode.TARGET_COUNT)

    def _set_list_controls_enabled(self, enabled: bool, allow_clear: bool = True) -> None:
        """同步视频列表增删按钮与开始抽帧按钮的可用性，防止探测或抽帧期间并发改动列表。"""
        self.add_files_btn.setEnabled(enabled)
        self.add_folder_btn.setEnabled(enabled)
        self.remove_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(allow_clear)
        self.start_btn.setEnabled(enabled)

    def _add_video_paths(self, raw_paths: list[Path]) -> None:
        """启动后台 `VideoProbeWorker` 扫描并探测视频元信息，避免大量视频阻塞界面。"""
        if self.probe_worker and self.probe_worker.isRunning():
            return
        existing = {p.resolve() for p in self._video_paths}
        self._set_list_controls_enabled(False)
        self.status_label.setText("正在后台扫描并读取视频元信息...")

        self.probe_worker = VideoProbeWorker(raw_paths, existing)
        self.probe_worker.item_probed.connect(self._on_video_probed)
        self.probe_worker.finished_probing.connect(self._on_probe_finished)
        self.probe_worker.failed.connect(self._on_probe_failed)
        self.probe_worker.start()

    def _on_video_probed(self, vp: Path, text: str) -> None:
        """接收后台探测完成的单个视频并追加至列表；若探测已被取消则忽略残留队列信号。"""
        if self.sender() is not self.probe_worker or (
            self.probe_worker and self.probe_worker.cancel_event.is_set()
        ):
            return
        if any(p.resolve() == vp.resolve() for p in self._video_paths):
            return
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, str(vp))
        self.video_list.addItem(item)
        self._video_paths.append(vp)

    def _on_probe_finished(self, added: int) -> None:
        if self.sender() is not self.probe_worker:
            return
        self._set_list_controls_enabled(True)
        self.status_label.setText(
            f"已添加 {added} 个视频（当前列表共 {len(self._video_paths)} 个）。"
        )

    def _on_probe_failed(self, err: str) -> None:
        if self.sender() is not self.probe_worker:
            return
        self._set_list_controls_enabled(True)
        self.status_label.setText(f"读取视频列表失败: {err}")

    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择视频文件",
            str(self.workspace_root),
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.ts *.m4v *.webm);;所有文件 (*.*)",
        )
        if files:
            self._add_video_paths([Path(f) for f in files])

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择包含视频的文件夹", str(self.workspace_root))
        if folder:
            self._add_video_paths([Path(folder)])

    def _on_remove_selected(self) -> None:
        for item in self.video_list.selectedItems():
            row = self.video_list.row(item)
            self.video_list.takeItem(row)
            if 0 <= row < len(self._video_paths):
                self._video_paths.pop(row)

    def _on_clear_videos(self) -> None:
        if self.probe_worker and self.probe_worker.isRunning():
            self.probe_worker.cancel()
        self.video_list.clear()
        self._video_paths.clear()
        self._set_list_controls_enabled(True)

    def _on_start(self) -> None:
        """校验视频列表与输出目录后切换按钮状态并启动后台 `FrameExtractWorker`。"""
        if not self._video_paths:
            QMessageBox.warning(self, "提示", "请先添加至少一个视频文件。")
            return
        out_text = self.output_picker.text()
        if not out_text:
            QMessageBox.warning(self, "提示", "请选择输出目录。")
            return

        end_sec = self.end_sec_spin.value()
        config = ExtractionConfig(
            output_dir=Path(out_text),
            mode=self.mode_combo.currentData(),
            interval_seconds=self.interval_spin.value(),
            every_n_frames=self.frame_step_spin.value(),
            target_count=self.target_count_spin.value(),
            start_sec=self.start_sec_spin.value(),
            end_sec=end_sec if end_sec > 0 else None,
            dedup_threshold=self.dedup_spin.value(),
            max_dimension=self.max_dim_spin.value(),
            image_format=self.format_combo.currentText(),
            jpeg_quality=self.quality_spin.value(),
            prefix=self.prefix_edit.text().strip(),
            create_video_subfolder=self.subfolder_chk.isChecked(),
        )

        self._set_list_controls_enabled(False, allow_clear=False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在初始化抽帧...")

        self.worker = FrameExtractWorker(list(self._video_paths), config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stop(self) -> None:
        """向正在运行的抽帧线程发送取消请求，由核心循环安全退出。"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status_label.setText("正在停止抽帧...")

    def _on_progress(self, prog: ExtractionProgress) -> None:
        """在主线程刷新进度条、状态文字及最新抽取帧缩略图。"""
        pct = int(min(100, (prog.current_frame / max(1, prog.total_frames)) * 100))
        self.progress_bar.setValue(pct)
        self.status_label.setText(
            f"视频 [{prog.video_index}/{prog.total_videos}] {prog.video_name} | "
            f"帧 {prog.current_frame}/{prog.total_frames} | "
            f"已保存 {prog.saved_total} 张 (去重跳过 {prog.skipped_similar} 张)"
        )
        if prog.last_saved_path:
            pix = load_pixmap_scaled(
                prog.last_saved_path,
                max(320, self.preview_label.width() - 12),
                max(220, self.preview_label.height() - 12),
            )
            if pix:
                self.preview_label.setPixmap(pix)

    def _on_finished(self, result: ExtractionResult) -> None:
        """恢复按钮可用状态、展示完成摘要，并在保存了至少 1 帧时向后续页同步输出目录。"""
        self._set_list_controls_enabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100 if not result.cancelled else self.progress_bar.value())

        state = "已手动停止" if result.cancelled else "抽帧完成"
        msg = (
            f"{state}：共处理 {result.videos_processed} 个视频，"
            f"保存 {result.frames_saved} 张图片（去重跳过 {result.frames_skipped_similar} 张）。"
        )
        if result.errors:
            msg += "\n警告:\n" + "\n".join(result.errors[:5])
        self.status_label.setText(msg)

        if result.frames_saved > 0:
            self.send_frames_dir.emit(str(result.output_dir))

    def _on_failed(self, err: str) -> None:
        self._set_list_controls_enabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"抽帧失败: {err}")
        QMessageBox.critical(self, "抽帧错误", err)
