"""标签页 2：原生 PyQt6 轻量标注与复核工作台界面。

职责：
    提供面向 YOLO 目标检测（矩形框）与 AI 复核闭环的高性能原生 PyQt6 标注工作台：
    左侧图片文件列表（搜索/状态筛选/进度统计）、中央可缩放平移交互画布（`R/W` 画框、
    8 控制点精修、撤销重做、AI 漏标提醒横幅）、右侧类别管理与目标框详情面板
    （展示 LLM/SAM 审核理由与置信度，支持一键本地 SAM 像素级贴边收框），
    并与「1. 视频抽帧」「3. LLM 智能标注」「4. 数据集准备」实现双向路径与变更信号联动。

运行环境与协作边界：
    运行于 PyQt6 主线程，仅依赖 `PyQt6`、`Pillow` 及 `core.annotation_io`；
    完全不依赖 X-AnyLabeling 或 `PyQt6-WebEngine`。本地 SAM 贴边收框按需在后台 `QThread`
    中延迟加载 `core.mllm_reviewer.UltralyticsSamBoxRefiner` 并遵守 `GpuTaskGuard` 互斥保护。
    保存格式 100% 兼容标准 LabelMe / X-AnyLabeling `.json` 矩形标注协议。
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QEvent, QFileSystemWatcher, QObject, QSize, Qt, QThread, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QAction, QColor, QIcon, QKeyEvent, QKeySequence, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.annotation_io import (
    ImageAnnotationDoc,
    ImageStatusSummary,
    RectAnnotation,
    inspect_image_status,
    label_color_rgb,
    load_classes_txt,
    load_image_annotation,
    save_classes_txt,
    save_image_annotation,
    scan_labeling_directory,
)
from yolo_studio.core.trainer import DeviceInfo, detect_compute_device
from yolo_studio.gui.annotation_canvas import AnnotationCanvasView, load_image_pixmap
from yolo_studio.gui.common import GpuTaskGuard

_ADDED_DLL_HANDLES: list[object] = []
_MAX_UNDO_STEPS = 80


def ensure_cuda_dll_search_paths() -> None:
    """在 Windows 上将 PyTorch 自带的 CUDA/cuDNN 目录加入 DLL 搜索路径。

    说明：
        在桌面进程提前补充 `torch\\lib` 目录，避免某些扩展库在 `import torch` 前加载 CUDA 运行时失败。
    """
    if os.name != "nt":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            torch_lib = Path(spec.origin).resolve().parent / "lib"
            if torch_lib.is_dir():
                lib_str = str(torch_lib)
                current_path = os.environ.get("PATH", "")
                if lib_str not in current_path.split(os.pathsep):
                    os.environ["PATH"] = lib_str + os.pathsep + current_path
                if hasattr(os, "add_dll_directory"):
                    _ADDED_DLL_HANDLES.append(os.add_dll_directory(lib_str))
    except Exception:
        pass


class _ElidedStatusLabel(QLabel):
    """按可用宽度自动中部省略长路径文本的状态标签，防止长文件名撑大顶层窗口。

    布局约束：
        固定返回较小的最小宽度提示（`80 px`）并采用 `Ignored` 水平尺寸策略，
        确保窗口尺寸永远由用户控制而非被动态图片路径推宽；完整文本始终保留在 `toolTip()` 中。
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(80)
        self.setText(text)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().minimumSizeHint()
        return QSize(80, hint.height())

    def sizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().sizeHint()
        return QSize(min(hint.width(), 420), hint.height())

    def full_text(self) -> str:
        return self._full_text

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._full_text = text or ""
        self.setToolTip(self._full_text)
        self._apply_elided_text()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_elided_text()

    def _apply_elided_text(self) -> None:
        avail = max(40, self.contentsRect().width())
        elided = self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, avail
        )
        super().setText(elided)


def _make_color_icon(rgb: tuple[int, int, int], size: int = 12) -> QIcon:
    """绘制用于类别/框列表项的圆角色块图标。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(*rgb))
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)
    painter.end()
    return QIcon(pix)


class _LabelPickerDialog(QDialog):
    """简洁的类别选择与新建弹窗（用于未选类别时落框或双击改类）。"""

    def __init__(
        self,
        classes: list[str],
        *,
        current_label: str = "",
        title: str = "选择或输入目标类别",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(QLabel("选择已有类别，或直接输入新类别名称后回车确认："))

        self.combo = QComboBox()
        self.combo.setEditable(True)
        self.combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for cls_name in classes:
            self.combo.addItem(_make_color_icon(label_color_rgb(cls_name)), cls_name)
        if current_label:
            idx = self.combo.findText(current_label)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
            else:
                self.combo.setEditText(current_label)
        elif classes:
            self.combo.setCurrentIndex(0)
        layout.addWidget(self.combo)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
        btns.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        if self.combo.lineEdit() is not None:
            self.combo.lineEdit().selectAll()
            self.combo.lineEdit().setFocus()

    def _on_accept(self) -> None:
        if not self.selected_label():
            QMessageBox.warning(self, "提示", "类别名称不能为空。")
            return
        self.accept()

    def selected_label(self) -> str:
        return self.combo.currentText().strip()


class _SamBoxRefineWorker(QThread):
    """在后台线程中使用本地 SAM 2.1 对工作台中的指定框执行像素级轮廓收紧。"""

    refined_ready = Signal(str, dict)  # 参数：(image_path_str, {box_idx: (x1, y1, x2, y2)})
    failed = Signal(str)

    def __init__(
        self,
        image_path: Path,
        target_boxes: list[tuple[int, RectAnnotation]],
        *,
        device: str = "0",
        model_name: str = "sam2.1_t.pt",
        refiner_cache: dict[tuple[str, str], Any] | None = None,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.target_boxes = [(idx, b.clone()) for idx, b in target_boxes]
        self.device = device
        self.model_name = model_name
        self._refiner_cache = refiner_cache if refiner_cache is not None else {}

    def run(self) -> None:
        try:
            import cv2
            import numpy as np

            from yolo_studio.core.mllm_reviewer import CandidateBox, UltralyticsSamBoxRefiner

            raw_bytes = np.fromfile(str(self.image_path), dtype=np.uint8)
            img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                self.failed.emit(f"无法读取图像像素：{self.image_path.name}")
                return

            cache_key = (self.model_name, self.device)
            refiner = self._refiner_cache.get(cache_key)
            if refiner is None:
                refiner = UltralyticsSamBoxRefiner(model_name=self.model_name, device=self.device)
                self._refiner_cache[cache_key] = refiner

            cands = [
                CandidateBox(
                    box_id=idx,
                    label=box.label,
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    confidence=box.score,
                )
                for idx, box in self.target_boxes
            ]
            refined_map = refiner.refine_boxes(img_bgr, cands)
            valid: dict[int, tuple[float, float, float, float]] = {}
            for idx, coords in refined_map.items():
                if coords is not None:
                    x1, y1, x2, y2 = coords
                    if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0:
                        valid[idx] = (float(x1), float(y1), float(x2), float(y2))
            self.refined_ready.emit(str(self.image_path.resolve()), valid)
        except Exception as exc:
            self.failed.emit(f"本地 SAM 贴边收紧失败：{exc}")


class LabelingStudioTab(QWidget):
    """原生 PyQt6 轻量标注与复核工作台标签页（标签页 2）。

    内部状态与资源：
        持有当前目录 `_current_dir`、全量图片清单 `_all_images`、按路径索引的轻量状态表 `_status_map`、
        当前已加载文档 `_doc`、未保存脏标记 `_dirty`、每图撤销/重做栈 `_undo_stack` / `_redo_stack`、
        类别列表 `_classes`、当前默认绘制类别 `_active_label`、可选后台 SAM 收框线程 `sam_worker`
        以及带 `400 ms` 防抖的 `QFileSystemWatcher` 目录监听器。
    """

    send_to_dataset = Signal(str, str)  # 参数：(标注图片目录, classes.txt 路径)
    send_to_review = Signal(str)  # 参数：(图片目录)
    annotations_changed = Signal(str)  # 参数：(发生 .json 写入/修改的目录)

    def __init__(
        self,
        workspace_root: Path,
        parent: QWidget | None = None,
        lazy_mount: bool = False,
        device_info: DeviceInfo | None = None,
        gpu_guard: GpuTaskGuard | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self._lazy_mount = lazy_mount
        self.device_info = device_info
        self.gpu_guard = gpu_guard

        self._current_dir: Path | None = None
        self._last_classes_file: Path | None = None
        self._all_images: list[Path] = []
        self._filtered_images: list[Path] = []
        self._status_map: dict[Path, ImageStatusSummary] = {}

        self._doc: ImageAnnotationDoc | None = None
        self._dirty: bool = False
        self._undo_stack: list[tuple[list[RectAnnotation], dict[str, Any]]] = []
        self._redo_stack: list[tuple[list[RectAnnotation], dict[str, Any]]] = []

        self._classes: list[str] = []
        self._active_label: str = ""
        self._syncing_lists: bool = False
        self._suppress_watch_until: float = 0.0

        self.sam_worker: _SamBoxRefineWorker | None = None
        self._sam_refiner_cache: dict[tuple[str, str], Any] = {}
        self._scoped_actions: list[QAction] = []

        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._on_watched_dir_changed)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(400)
        self._debounce_timer.timeout.connect(self._emit_annotations_changed)

        self._init_ui()
        self._init_shortcuts()
        self._set_shortcuts_active(self.isVisible())
        self._refresh_action_states()

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 1) 顶部工作流联动栏
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        bar.setStyleSheet(
            "QFrame { background-color: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; }"
        )
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(10, 5, 10, 5)
        bar_layout.setSpacing(8)

        self.status_label = _ElidedStatusLabel(
            "工作台就绪：打开图片目录即可快速画框与复核（R/W 画框 · E 编辑 · A/D 切图 · 1-9 切类 · F 确认通过 · Ctrl+S 保存）"
        )
        self.status_label.setStyleSheet("border: none; color: #24292f; font-weight: 500;")
        bar_layout.addWidget(self.status_label, 1)

        self.open_dir_btn = QPushButton("打开图片目录...")
        self.load_classes_btn = QPushButton("导入 classes.txt...")
        self.refresh_btn = QPushButton("刷新目录")
        self.to_review_btn = QPushButton("发送到「3. LLM 智能标注」→")
        self.to_dataset_btn = QPushButton("发送到「4. 数据集准备」→")
        self.to_dataset_btn.setObjectName("primaryBtn")

        for btn in (
            self.open_dir_btn,
            self.load_classes_btn,
            self.refresh_btn,
            self.to_review_btn,
            self.to_dataset_btn,
        ):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            bar_layout.addWidget(btn)
        root.addWidget(bar)

        # 2) 精简画布与复核操作工具栏
        tools = QFrame()
        tools.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #d8dee4; border-radius: 6px; }"
            "QPushButton { padding: 4px 10px; font-size: 12px; }"
            "QPushButton:checked { background-color: #ddf4ff; border-color: #0969da; color: #0550ae; font-weight: 600; }"
        )
        t_lay = QHBoxLayout(tools)
        t_lay.setContentsMargins(8, 4, 8, 4)
        t_lay.setSpacing(6)

        self.edit_mode_btn = QPushButton("编辑 (E)")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.setChecked(True)
        self.draw_mode_btn = QPushButton("画框 (R)")
        self.draw_mode_btn.setCheckable(True)
        self.dup_box_btn = QPushButton("复制框")
        self.del_box_btn = QPushButton("删除框 (Del)")
        self.undo_btn = QPushButton("撤销")
        self.redo_btn = QPushButton("重做")

        self.prev_img_btn = QPushButton("◀ 上一张 (A)")
        self.next_img_btn = QPushButton("下一张 (D) ▶")
        self.save_btn = QPushButton("保存 (Ctrl+S)")
        self.verify_next_btn = QPushButton("确认通过→下张 (F)")
        self.verify_next_btn.setToolTip("标记当前图已人工复核（0 框时保存为显式空图负样本）、清除漏标提醒并自动切换下一张")

        self.sam_refine_btn = QPushButton("SAM 贴边收框")
        self.sam_refine_btn.setToolTip("调用本地 SAM 2.1 对当前选中框（未选中时对整图全部框）执行像素级轮廓贴边收紧")

        self.fit_btn = QPushButton("适应窗口 (0)")
        self.zoom_100_btn = QPushButton("1:1")
        self.show_labels_chk = QCheckBox("显示标签")
        self.show_labels_chk.setChecked(True)
        self.auto_save_chk = QCheckBox("自动保存")
        self.auto_save_chk.setChecked(True)
        self.auto_save_chk.setToolTip("切图或退出前自动保存已修改的 .json 标注")

        for w in (
            self.edit_mode_btn,
            self.draw_mode_btn,
            self.dup_box_btn,
            self.del_box_btn,
            self.undo_btn,
            self.redo_btn,
            self.prev_img_btn,
            self.next_img_btn,
            self.save_btn,
            self.verify_next_btn,
            self.sam_refine_btn,
            self.fit_btn,
            self.zoom_100_btn,
            self.show_labels_chk,
            self.auto_save_chk,
        ):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        for w in (self.edit_mode_btn, self.draw_mode_btn, self.dup_box_btn, self.del_box_btn):
            t_lay.addWidget(w)
        t_lay.addWidget(self._make_vsep())
        t_lay.addWidget(self.undo_btn)
        t_lay.addWidget(self.redo_btn)
        t_lay.addWidget(self._make_vsep())
        t_lay.addWidget(self.prev_img_btn)
        t_lay.addWidget(self.next_img_btn)
        t_lay.addWidget(self.save_btn)
        t_lay.addWidget(self.verify_next_btn)
        t_lay.addWidget(self._make_vsep())
        t_lay.addWidget(self.sam_refine_btn)
        t_lay.addWidget(self._make_vsep())
        t_lay.addWidget(self.fit_btn)
        t_lay.addWidget(self.zoom_100_btn)
        t_lay.addWidget(self.show_labels_chk)
        t_lay.addWidget(self.auto_save_chk)
        t_lay.addStretch(1)
        root.addWidget(tools)

        # 3) 主体三栏可调分割区
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # --- 左栏：文件列表与筛选 ---
        left_panel = QFrame()
        left_panel.setMinimumWidth(210)
        left_panel.setMaximumWidth(380)
        left_lay = QVBoxLayout(left_panel)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(5)

        left_title = QLabel("图片文件列表")
        left_title.setStyleSheet("font-weight: 600; color: #24292f;")
        left_lay.addWidget(left_title)

        self.file_search = QLineEdit()
        self.file_search.setPlaceholderText("搜索文件名或子目录...")
        self.file_search.setClearButtonEnabled(True)
        left_lay.addWidget(self.file_search)

        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItem("全部图片", "all")
        self.status_filter_combo.addItem("仅看未标注", "unlabeled")
        self.status_filter_combo.addItem("仅看已标注", "labeled")
        self.status_filter_combo.addItem("仅看空图 (0框)", "empty")
        self.status_filter_combo.addItem("仅看含漏标/待复核提醒", "review")
        self.status_filter_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        left_lay.addWidget(self.status_filter_combo)

        self.progress_summary_label = QLabel("尚未打开图片目录")
        self.progress_summary_label.setStyleSheet("color: #57606a; font-size: 12px;")
        self.progress_summary_label.setWordWrap(True)
        left_lay.addWidget(self.progress_summary_label)

        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.setStyleSheet(
            "QListWidget { border: 1px solid #d0d7de; border-radius: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 3px 4px; }"
        )
        left_lay.addWidget(self.file_list, 1)
        splitter.addWidget(left_panel)

        # --- 中栏：AI 复核提醒横幅 + 核心画布 + 底部微状态栏 ---
        center_panel = QWidget()
        center_lay = QVBoxLayout(center_panel)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(4)

        self.review_banner = QFrame()
        self.review_banner.setStyleSheet(
            "QFrame { background-color: #fff8c5; border: 1px solid #d4a72c; border-radius: 5px; }"
        )
        banner_lay = QHBoxLayout(self.review_banner)
        banner_lay.setContentsMargins(10, 4, 10, 4)
        banner_lay.setSpacing(8)
        self.review_banner_label = QLabel("")
        self.review_banner_label.setWordWrap(True)
        self.review_banner_label.setStyleSheet("border: none; color: #6f4e00; font-weight: 500; font-size: 12px;")
        self.clear_hint_btn = QPushButton("已核查，清除提醒")
        self.clear_hint_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear_hint_btn.setStyleSheet("padding: 2px 8px; font-size: 12px;")
        banner_lay.addWidget(self.review_banner_label, 1)
        banner_lay.addWidget(self.clear_hint_btn)
        self.review_banner.setVisible(False)
        center_lay.addWidget(self.review_banner)

        self.canvas = AnnotationCanvasView()
        center_lay.addWidget(self.canvas, 1)

        footer = QFrame()
        footer.setStyleSheet(
            "QFrame { background-color: #f6f8fa; border: 1px solid #d8dee4; border-radius: 4px; }"
            "QLabel { border: none; color: #57606a; font-size: 12px; }"
        )
        f_lay = QHBoxLayout(footer)
        f_lay.setContentsMargins(8, 3, 8, 3)
        f_lay.setSpacing(14)
        self.mode_badge_label = QLabel("模式: 编辑")
        self.mode_badge_label.setStyleSheet("border: none; color: #0969da; font-weight: 600; font-size: 12px;")
        self.img_info_label = QLabel("分辨率: -")
        self.cursor_info_label = QLabel("坐标: -")
        self.zoom_info_label = QLabel("缩放: 100%")
        self.save_state_label = QLabel("")
        f_lay.addWidget(self.mode_badge_label)
        f_lay.addWidget(self.img_info_label)
        f_lay.addWidget(self.cursor_info_label)
        f_lay.addWidget(self.zoom_info_label)
        f_lay.addStretch(1)
        f_lay.addWidget(self.save_state_label)
        center_lay.addWidget(footer)
        splitter.addWidget(center_panel)

        # --- 右栏：类别清单 + 当前图目标框列表 + 框详情 ---
        right_panel = QFrame()
        right_panel.setMinimumWidth(240)
        right_panel.setMaximumWidth(400)
        right_lay = QVBoxLayout(right_panel)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.setSpacing(6)

        cls_hdr = QHBoxLayout()
        cls_title = QLabel("类别清单 (按 1-9 速切)")
        cls_title.setStyleSheet("font-weight: 600; color: #24292f;")
        self.save_classes_btn = QPushButton("保存 classes.txt")
        self.save_classes_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.save_classes_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        cls_hdr.addWidget(cls_title)
        cls_hdr.addStretch(1)
        cls_hdr.addWidget(self.save_classes_btn)
        right_lay.addLayout(cls_hdr)

        add_cls_row = QHBoxLayout()
        add_cls_row.setSpacing(4)
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("输入新类别名称后回车...")
        self.add_class_btn = QPushButton("+ 添加")
        self.add_class_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_class_btn.setStyleSheet("padding: 4px 8px; font-size: 12px;")
        add_cls_row.addWidget(self.new_class_edit, 1)
        add_cls_row.addWidget(self.add_class_btn)
        right_lay.addLayout(add_cls_row)

        self.unique_label_list = QListWidget()
        self.unique_label_list.setStyleSheet(
            "QListWidget { border: 1px solid #d0d7de; border-radius: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 3px 4px; }"
        )
        right_lay.addWidget(self.unique_label_list, 2)

        obj_hdr = QHBoxLayout()
        self.obj_title_label = QLabel("当前图目标框 (0)")
        self.obj_title_label.setStyleSheet("font-weight: 600; color: #24292f;")
        self.clear_boxes_btn = QPushButton("清空全部框")
        self.clear_boxes_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear_boxes_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        obj_hdr.addWidget(self.obj_title_label)
        obj_hdr.addStretch(1)
        obj_hdr.addWidget(self.clear_boxes_btn)
        right_lay.addLayout(obj_hdr)

        self.shape_list = QListWidget()
        self.shape_list.setStyleSheet(
            "QListWidget { border: 1px solid #d0d7de; border-radius: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 3px 4px; }"
        )
        right_lay.addWidget(self.shape_list, 3)

        self.box_detail_label = QLabel("未选中目标框：单击画布或列表项可查看坐标、AI 置信度与审核理由。")
        self.box_detail_label.setWordWrap(True)
        self.box_detail_label.setStyleSheet(
            "QLabel { background-color: #f6f8fa; border: 1px solid #d8dee4; border-radius: 5px; "
            "padding: 6px; color: #24292f; font-size: 12px; }"
        )
        right_lay.addWidget(self.box_detail_label)

        box_ops = QHBoxLayout()
        box_ops.setSpacing(6)
        self.change_cls_btn = QPushButton("改类别")
        self.change_cls_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.change_cls_btn.setStyleSheet("padding: 4px 8px; font-size: 12px;")
        self.box_del_inline_btn = QPushButton("删除选中")
        self.box_del_inline_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.box_del_inline_btn.setStyleSheet("padding: 4px 8px; font-size: 12px;")
        box_ops.addWidget(self.change_cls_btn, 1)
        box_ops.addWidget(self.box_del_inline_btn, 1)
        right_lay.addLayout(box_ops)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([240, 820, 270])
        root.addWidget(splitter, 1)

        # 事件信号连接
        self.open_dir_btn.clicked.connect(self._on_pick_and_open_dir)
        self.load_classes_btn.clicked.connect(self._on_pick_and_load_classes)
        self.refresh_btn.clicked.connect(self._on_refresh_dir)
        self.to_dataset_btn.clicked.connect(self._on_send_to_dataset)
        self.to_review_btn.clicked.connect(self._on_send_to_review)

        self.edit_mode_btn.clicked.connect(lambda: self.canvas.set_mode(AnnotationCanvasView.MODE_EDIT))
        self.draw_mode_btn.clicked.connect(lambda: self.canvas.set_mode(AnnotationCanvasView.MODE_DRAW))
        self.dup_box_btn.clicked.connect(self._duplicate_selected_box)
        self.del_box_btn.clicked.connect(self._delete_selected_box)
        self.box_del_inline_btn.clicked.connect(self._delete_selected_box)
        self.change_cls_btn.clicked.connect(self._prompt_relabel_selected_box)
        self.clear_boxes_btn.clicked.connect(self._clear_all_boxes)
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)
        self.prev_img_btn.clicked.connect(self.open_prev_image)
        self.next_img_btn.clicked.connect(self.open_next_image)
        self.save_btn.clicked.connect(lambda: self.save_current_annotation(force=True))
        self.verify_next_btn.clicked.connect(self._verify_and_next)
        self.clear_hint_btn.clicked.connect(self._clear_missing_hint)
        self.sam_refine_btn.clicked.connect(self._on_sam_refine_clicked)
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.zoom_100_btn.clicked.connect(lambda: self.canvas.set_zoom(1.0))
        self.show_labels_chk.toggled.connect(self.canvas.set_show_labels)

        self.file_search.textChanged.connect(self._apply_file_filter)
        self.status_filter_combo.currentIndexChanged.connect(self._apply_file_filter)
        self.file_list.currentRowChanged.connect(self._on_file_row_changed)
        self.file_list.customContextMenuRequested.connect(self._on_file_list_context_menu)

        self.add_class_btn.clicked.connect(self._on_add_class_from_input)
        self.new_class_edit.returnPressed.connect(self._on_add_class_from_input)
        self.save_classes_btn.clicked.connect(self._on_save_classes_txt)
        self.unique_label_list.currentRowChanged.connect(self._on_class_row_changed)
        self.unique_label_list.itemDoubleClicked.connect(self._on_class_double_clicked)

        self.shape_list.currentRowChanged.connect(self._on_shape_row_changed)
        self.shape_list.itemDoubleClicked.connect(lambda _item: self._prompt_relabel_selected_box())

        self.canvas.box_drawn.connect(self._on_canvas_box_drawn)
        self.canvas.box_geometry_committed.connect(self._on_canvas_box_geometry_committed)
        self.canvas.selection_changed.connect(self._on_canvas_selection_changed)
        self.canvas.mode_changed.connect(self._on_canvas_mode_changed)
        self.canvas.zoom_changed.connect(self._on_canvas_zoom_changed)
        self.canvas.cursor_pos_changed.connect(self._on_canvas_cursor_changed)
        self.canvas.box_double_clicked.connect(lambda _idx: self._prompt_relabel_selected_box())

        # 在主要非文本子控件上安装键盘过滤器，保证单键快捷键在画布与列表上均响应，而在输入框内绝不拦截打字
        for target in (self.canvas, self.file_list, self.shape_list, self.unique_label_list):
            target.installEventFilter(self)

    @staticmethod
    def _make_vsep() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color: #d0d7de;")
        return line

    def _init_shortcuts(self) -> None:
        """注册带修饰键的组合快捷键并限定作用域于工作台内部（避免跨标签页串键）。"""
        combos: list[tuple[str, Any]] = [
            ("Ctrl+S", lambda: self.save_current_annotation(force=True)),
            ("Ctrl+Z", self._undo),
            ("Ctrl+Y", self._redo),
            ("Ctrl+Shift+Z", self._redo),
            ("Ctrl+D", self._duplicate_selected_box),
        ]
        for seq, handler in combos:
            act = QAction(self)
            act.setShortcut(QKeySequence(seq))
            act.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            act.triggered.connect(handler)
            self.addAction(act)
            self._scoped_actions.append(act)

    def _set_shortcuts_active(self, active: bool) -> None:
        """同步工作台快捷键启停状态：切离工作台标签页时停用，防止任何情况下误触。"""
        ctx = (
            Qt.ShortcutContext.WidgetWithChildrenShortcut
            if active
            else Qt.ShortcutContext.WidgetShortcut
        )
        for act in self._scoped_actions:
            act.setEnabled(active)
            act.setShortcutContext(ctx)

    def _is_text_input_focused(self) -> bool:
        focused = QApplication.focusWidget()
        return isinstance(focused, (QLineEdit, QTextEdit, QPlainTextEdit))

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if (
            event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and self.isVisible()
            and not self._is_text_input_focused()
        ):
            if self._handle_workbench_keypress(event):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if self.isVisible() and not self._is_text_input_focused():
            if self._handle_workbench_keypress(event):
                event.accept()
                return
        super().keyPressEvent(event)

    def _handle_workbench_keypress(self, event: QKeyEvent) -> bool:
        """处理工作台单键快捷键；仅在无 Ctrl/Alt 修饰键时生效，返回 `True` 表示已消费。"""
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier):
            return False
        key = event.key()
        if key in (Qt.Key.Key_R, Qt.Key.Key_W):
            self.canvas.set_mode(AnnotationCanvasView.MODE_DRAW)
            return True
        if key == Qt.Key.Key_E:
            self.canvas.set_mode(AnnotationCanvasView.MODE_EDIT)
            return True
        if key in (Qt.Key.Key_A, Qt.Key.Key_Left):
            self.open_prev_image()
            return True
        if key in (Qt.Key.Key_D, Qt.Key.Key_Right):
            self.open_next_image()
            return True
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.canvas.selected_index() >= 0:
                self._delete_selected_box()
                return True
            return False
        if key == Qt.Key.Key_F:
            self._verify_and_next()
            return True
        if key == Qt.Key.Key_0:
            self.canvas.fit_to_window()
            return True
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
            cls_idx = key - Qt.Key.Key_1
            if cls_idx < len(self._classes):
                self._apply_class_shortcut(cls_idx)
                return True
        return False

    def _focus_canvas_if_ready(self) -> None:
        if not self.isVisible() or self._is_text_input_focused():
            return
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._set_shortcuts_active(True)
        QTimer.singleShot(0, self._focus_canvas_if_ready)

    def hideEvent(self, event) -> None:  # type: ignore[override]
        super().hideEvent(event)
        self._set_shortcuts_active(False)

    # ------------------------------------------------------------------
    # 对外兼容公共接口
    # ------------------------------------------------------------------
    def ensure_mounted(self) -> bool:
        """原生工作台随主窗口同步创建完成，调用本方法始终幂等返回 `True`。"""
        return True

    def apply_device_info(self, info: DeviceInfo) -> None:
        """接收主窗口后台硬件探测结果，供本地 SAM 贴边收框自动选择 GPU 或 CPU。"""
        self.device_info = info

    def current_directory(self) -> Path | None:
        """返回工作台当前打开的图片目录；若尚未打开任何目录则返回 `None`。"""
        return self._current_dir

    def current_image_path(self) -> Path | None:
        """返回当前正在查看的图片绝对路径；尚未加载图片时返回 `None`。"""
        return self._doc.image_path if self._doc is not None else None

    def load_classes_file(self, classes_file: Path) -> int:
        """从 `classes.txt` 读取类别列表并同步更新右侧类别清单。"""
        labels = load_classes_txt(classes_file)
        if not labels:
            return 0
        self._merge_classes(labels, replace_order=True)
        self._last_classes_file = classes_file.resolve()
        return len(labels)

    def open_directory(
        self,
        dir_path: Path,
        classes_file: Path | None = None,
        select_image: Path | None = None,
    ) -> bool:
        """打开指定图片目录（递归子目录），预载类别清单并可选直接定位到单张图片。"""
        target_dir = dir_path.resolve()
        if not target_dir.is_dir():
            QMessageBox.warning(self, "目录不存在", f"图片目录不存在：\n{target_dir}")
            return False

        if not self._may_continue():
            return False

        # 已通过未保存保护校验，清空旧文档引用，确保刷新同目录同单图时强制重读磁盘最新 .json
        self._doc = None
        self._dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()

        images = scan_labeling_directory(target_dir, recursive=True)
        self._current_dir = target_dir
        self._all_images = images
        self._status_map = {}

        discovered_labels: list[str] = []
        for img_path in images:
            summary = inspect_image_status(img_path)
            self._status_map[img_path] = summary
            for lbl in summary.labels:
                if lbl not in discovered_labels:
                    discovered_labels.append(lbl)

        # 按“显式 classes_file > 目录内 classes.txt > 已有 .json 出现的类别”顺序合并类别
        loaded_from_file: list[str] = []
        if classes_file is not None and classes_file.is_file():
            loaded_from_file = load_classes_txt(classes_file)
            if loaded_from_file:
                self._last_classes_file = classes_file.resolve()
        elif (target_dir / "classes.txt").is_file():
            loaded_from_file = load_classes_txt(target_dir / "classes.txt")
            if loaded_from_file:
                self._last_classes_file = (target_dir / "classes.txt").resolve()

        if loaded_from_file:
            merged = list(loaded_from_file)
            for lbl in discovered_labels:
                if lbl not in merged:
                    merged.append(lbl)
            self._merge_classes(merged, replace_order=True)
        elif discovered_labels:
            self._merge_classes(discovered_labels, replace_order=False)

        self._watch_directory(target_dir)

        # 重置筛选以保证指定目标单图可见
        self._syncing_lists = True
        try:
            self.file_search.clear()
            self.status_filter_combo.setCurrentIndex(0)
        finally:
            self._syncing_lists = False
        self._apply_file_filter(keep_current=False)

        if not images:
            self._doc = None
            self._dirty = False
            self._undo_stack.clear()
            self._redo_stack.clear()
            self.canvas.clear_image()
            self._refresh_shape_list()
            self._refresh_review_banner()
            self._refresh_footer_and_title()
            self._refresh_action_states()
            self.status_label.setText(f"当前目录无图片：{target_dir}")
            return True

        target_img = images[0]
        if select_image is not None:
            resolved_sel = select_image.resolve()
            if resolved_sel in self._status_map:
                target_img = resolved_sel
            elif resolved_sel.is_file():
                # 兜底：若目标单图不在扫描结果中，按同文件加入列表首位
                self._all_images.insert(0, resolved_sel)
                self._status_map[resolved_sel] = inspect_image_status(resolved_sel)
                self._apply_file_filter(keep_current=False)
                target_img = resolved_sel

        self._select_image_in_list(target_img)
        if self.isVisible():
            QTimer.singleShot(0, self._focus_canvas_if_ready)
        return True

    def open_next_image(self) -> bool:
        """在当前筛选文件列表中切换到下一张图片。"""
        if not self._filtered_images:
            return False
        cur_row = self.file_list.currentRow()
        if cur_row < 0:
            self.file_list.setCurrentRow(0)
            return True
        if cur_row + 1 < len(self._filtered_images):
            self.file_list.setCurrentRow(cur_row + 1)
            return True
        self.save_state_label.setText("已是列表最后一张")
        return False

    def open_prev_image(self) -> bool:
        """在当前筛选文件列表中切换到上一张图片。"""
        if not self._filtered_images:
            return False
        cur_row = self.file_list.currentRow()
        if cur_row > 0:
            self.file_list.setCurrentRow(cur_row - 1)
            return True
        self.save_state_label.setText("已是列表第一张")
        return False

    def save_current_annotation(self, *, force: bool = False) -> bool:
        """保存当前图片标注到同名 `.json`；无改动且非 `force` 时直接返回 `True`。"""
        if self._doc is None:
            return True
        if not self._dirty and not force:
            return True

        import time

        self._suppress_watch_until = time.monotonic() + 0.8
        try:
            save_image_annotation(self._doc)
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", f"无法写入标注文件：\n{self._doc.json_path}\n{exc}")
            return False

        self._dirty = False
        self._update_single_image_status(self._doc.image_path)
        self._refresh_footer_and_title()
        self.save_state_label.setText(f"已保存 {self._doc.json_path.name}")
        if self._current_dir is not None:
            self.annotations_changed.emit(str(self._current_dir))
        return True

    def prepare_for_close(self) -> bool:
        """在主窗口退出前处理未保存标注修改并安全停止可能运行的 SAM 收框线程。"""
        if not self._may_continue():
            return False
        if self.sam_worker is not None and self.sam_worker.isRunning():
            self.sam_worker.wait(3000)
        return True

    # ------------------------------------------------------------------
    # 目录监听与未保存保护
    # ------------------------------------------------------------------
    def _watch_directory(self, dir_path: Path) -> None:
        existing = self._watcher.directories()
        if existing:
            self._watcher.removePaths(existing)
        if dir_path.is_dir():
            self._watcher.addPath(str(dir_path))

    def _on_watched_dir_changed(self, _path: str) -> None:
        import time

        if time.monotonic() < self._suppress_watch_until:
            return
        self._debounce_timer.start()

    def _emit_annotations_changed(self) -> None:
        if self._current_dir is not None:
            self.annotations_changed.emit(str(self._current_dir))

    def _may_continue(self) -> bool:
        """切图、切换目录或关闭前核查未保存改动：自动保存开启时直接静默保存，否则弹出询问。"""
        if self._doc is None or not self._dirty:
            return True
        if self.auto_save_chk.isChecked():
            return self.save_current_annotation(force=True)
        reply = QMessageBox.question(
            self,
            "保存标注修改",
            f"当前图片「{self._doc.image_path.name}」的标注尚未保存，是否先保存？",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Save:
            return self.save_current_annotation(force=True)
        if reply == QMessageBox.StandardButton.Discard:
            self._dirty = False
            return True
        return False

    # ------------------------------------------------------------------
    # 图片加载与文件列表渲染
    # ------------------------------------------------------------------
    def _format_file_item_text(self, img_path: Path, summary: ImageStatusSummary) -> str:
        rel = img_path.name
        if self._current_dir is not None:
            try:
                rel = str(img_path.relative_to(self._current_dir)).replace("\\", "/")
            except ValueError:
                rel = img_path.name

        key = summary.status_key
        if key == "review":
            prefix = f"! [{summary.box_count}]"
        elif key == "labeled":
            prefix = f"✓ [{summary.box_count}]"
        elif key == "empty":
            prefix = "○ [0]"
        else:
            prefix = "·"
        return f"{prefix}  {rel}"

    def _style_file_item(self, item: QListWidgetItem, summary: ImageStatusSummary) -> None:
        key = summary.status_key
        if key == "review":
            item.setForeground(QColor("#9a6700"))
        elif key == "labeled":
            item.setForeground(QColor("#1a7f37"))
        elif key == "empty":
            item.setForeground(QColor("#0969da"))
        else:
            item.setForeground(QColor("#57606a"))

    def _apply_file_filter(self, *_args: Any, keep_current: bool = True) -> None:
        if self._syncing_lists:
            return
        cur_img = self._doc.image_path if (keep_current and self._doc is not None) else None
        keyword = self.file_search.text().strip().lower()
        mode = self.status_filter_combo.currentData() or "all"

        filtered: list[Path] = []
        for img_path in self._all_images:
            summary = self._status_map.get(img_path) or inspect_image_status(img_path)
            if mode != "all" and summary.status_key != mode:
                continue
            if keyword:
                rel_str = str(img_path).lower()
                if self._current_dir is not None:
                    try:
                        rel_str = str(img_path.relative_to(self._current_dir)).lower()
                    except ValueError:
                        pass
                if keyword not in rel_str:
                    continue
            filtered.append(img_path)

        self._filtered_images = filtered
        self._syncing_lists = True
        try:
            self.file_list.clear()
            target_row = -1
            for idx, img_path in enumerate(filtered):
                summary = self._status_map[img_path]
                item = QListWidgetItem(self._format_file_item_text(img_path, summary))
                item.setData(Qt.ItemDataRole.UserRole, str(img_path))
                item.setToolTip(str(img_path))
                self._style_file_item(item, summary)
                self.file_list.addItem(item)
                if cur_img is not None and img_path == cur_img:
                    target_row = idx
            if target_row >= 0:
                self.file_list.setCurrentRow(target_row)
        finally:
            self._syncing_lists = False
        self._refresh_progress_summary()

    def _select_image_in_list(self, img_path: Path) -> None:
        resolved = img_path.resolve()
        for row, p in enumerate(self._filtered_images):
            if p == resolved:
                if self.file_list.currentRow() == row:
                    self._load_image(resolved)
                else:
                    self.file_list.setCurrentRow(row)
                return
        self._load_image(resolved)

    def _on_file_row_changed(self, row: int) -> None:
        if self._syncing_lists:
            return
        if not (0 <= row < len(self._filtered_images)):
            return
        target = self._filtered_images[row]
        if self._doc is not None and self._doc.image_path == target:
            return
        if not self._may_continue():
            # 取消切图：回滚文件列表选中行到当前文档
            if self._doc is not None:
                self._syncing_lists = True
                try:
                    for idx, p in enumerate(self._filtered_images):
                        if p == self._doc.image_path:
                            self.file_list.setCurrentRow(idx)
                            break
                finally:
                    self._syncing_lists = False
            return
        self._load_image(target)

    def _load_image(self, image_path: Path) -> None:
        pix, w, h = load_image_pixmap(image_path)
        doc = load_image_annotation(image_path, image_size=(w, h) if (w > 0 and h > 0) else None)
        self._doc = doc
        self._dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()

        # 把当前图片新出现的类别纳入右侧清单
        new_labels = [b.label for b in doc.boxes if b.label and b.label not in self._classes]
        if new_labels:
            self._merge_classes(new_labels, replace_order=False)

        self.canvas.set_image(pix, doc.image_width, doc.image_height)
        self.canvas.set_boxes(doc.boxes, selected_index=-1)
        self._refresh_class_counts()
        self._refresh_shape_list()
        self._refresh_review_banner()
        self._refresh_footer_and_title()
        self._refresh_action_states()
        self._refresh_progress_summary()

    def _update_single_image_status(self, image_path: Path) -> None:
        resolved = image_path.resolve()
        summary = inspect_image_status(resolved)
        self._status_map[resolved] = summary
        for row, p in enumerate(self._filtered_images):
            if p == resolved:
                item = self.file_list.item(row)
                if item is not None:
                    item.setText(self._format_file_item_text(resolved, summary))
                    self._style_file_item(item, summary)
                break
        self._refresh_progress_summary()

    def _refresh_progress_summary(self) -> None:
        total = len(self._all_images)
        if total == 0:
            self.progress_summary_label.setText(
                "当前目录暂无图片" if self._current_dir else "尚未打开图片目录"
            )
            return
        labeled = sum(1 for s in self._status_map.values() if s.status_key == "labeled")
        empty = sum(1 for s in self._status_map.values() if s.status_key == "empty")
        review = sum(1 for s in self._status_map.values() if s.status_key == "review")
        unlabeled = total - labeled - empty - review
        pos = self.file_list.currentRow() + 1 if self.file_list.currentRow() >= 0 else 0
        shown = len(self._filtered_images)
        parts = [f"{pos}/{shown} 张" if shown == total else f"筛选 {pos}/{shown} (共 {total})"]
        parts.append(f"已标 {labeled}")
        if empty:
            parts.append(f"空图 {empty}")
        if review:
            parts.append(f"待复核 {review}")
        if unlabeled:
            parts.append(f"未标 {unlabeled}")
        self.progress_summary_label.setText(" · ".join(parts))

    def _refresh_review_banner(self) -> None:
        if self._doc is None:
            self.review_banner.setVisible(False)
            return
        if self._doc.has_missing_hint or self._doc.missing_note:
            note = self._doc.missing_note or "上游 LLM 怀疑本图可能存在漏标或待确认目标，请人工检查。"
            self.review_banner_label.setText(f"⚠ AI 复核提醒：{note}")
            self.review_banner.setVisible(True)
        else:
            self.review_banner.setVisible(False)

    def _refresh_footer_and_title(self) -> None:
        if self._doc is None:
            self.img_info_label.setText("分辨率: -")
            self.save_state_label.setText("")
            return
        dirty_mark = " *" if self._dirty else ""
        self.status_label.setText(f"{self._doc.image_path}{dirty_mark}")
        self.img_info_label.setText(
            f"分辨率: {self._doc.image_width}×{self._doc.image_height} · 目标: {len(self._doc.boxes)}"
        )
        if self._dirty:
            self.save_state_label.setText("未保存*")
            self.save_state_label.setStyleSheet("border: none; color: #cf222e; font-weight: 600; font-size: 12px;")
        elif self._doc.has_json_file:
            mark = "已复核 ✓" if self._doc.is_verified else "已保存"
            self.save_state_label.setText(mark)
            self.save_state_label.setStyleSheet("border: none; color: #1a7f37; font-size: 12px;")
        else:
            self.save_state_label.setText("未生成 .json")
            self.save_state_label.setStyleSheet("border: none; color: #57606a; font-size: 12px;")

    def _refresh_action_states(self) -> None:
        has_img = self._doc is not None
        has_sel = has_img and self.canvas.selected_index() >= 0
        has_boxes = has_img and self._doc is not None and len(self._doc.boxes) > 0
        sam_busy = self.sam_worker is not None and self.sam_worker.isRunning()

        self.dup_box_btn.setEnabled(has_sel)
        self.del_box_btn.setEnabled(has_sel)
        self.box_del_inline_btn.setEnabled(has_sel)
        self.change_cls_btn.setEnabled(has_sel)
        self.clear_boxes_btn.setEnabled(has_boxes)
        self.undo_btn.setEnabled(len(self._undo_stack) > 0)
        self.redo_btn.setEnabled(len(self._redo_stack) > 0)
        self.save_btn.setEnabled(has_img)
        self.verify_next_btn.setEnabled(has_img)
        self.prev_img_btn.setEnabled(len(self._filtered_images) > 1)
        self.next_img_btn.setEnabled(len(self._filtered_images) > 1)
        self.sam_refine_btn.setEnabled(has_boxes and not sam_busy)
        self.sam_refine_btn.setText("SAM 收紧中..." if sam_busy else "SAM 贴边收框")

    # ------------------------------------------------------------------
    # 类别管理
    # ------------------------------------------------------------------
    def _merge_classes(self, labels: list[str], *, replace_order: bool = False) -> None:
        if replace_order:
            deduped: list[str] = []
            seen: set[str] = set()
            for raw in labels:
                name = str(raw).strip()
                if name and name not in seen:
                    seen.add(name)
                    deduped.append(name)
            self._classes = deduped
        else:
            for raw in labels:
                name = str(raw).strip()
                if name and name not in self._classes:
                    self._classes.append(name)

        if self._classes and self._active_label not in self._classes:
            self._active_label = self._classes[0]
        self.canvas.set_active_label(self._active_label)
        self._refresh_class_counts()

    def _refresh_class_counts(self) -> None:
        counts: dict[str, int] = {}
        if self._doc is not None:
            for b in self._doc.boxes:
                counts[b.label] = counts.get(b.label, 0) + 1

        self._syncing_lists = True
        try:
            self.unique_label_list.clear()
            active_row = -1
            for idx, cls_name in enumerate(self._classes):
                shortcut_tag = f"[{idx + 1}] " if idx < 9 else "    "
                cnt = counts.get(cls_name, 0)
                text = f"{shortcut_tag}{cls_name}  ({cnt})" if cnt > 0 else f"{shortcut_tag}{cls_name}"
                item = QListWidgetItem(_make_color_icon(label_color_rgb(cls_name)), text)
                item.setData(Qt.ItemDataRole.UserRole, cls_name)
                self.unique_label_list.addItem(item)
                if cls_name == self._active_label:
                    active_row = idx
            if active_row >= 0:
                self.unique_label_list.setCurrentRow(active_row)
        finally:
            self._syncing_lists = False

    def _on_add_class_from_input(self) -> None:
        name = self.new_class_edit.text().strip()
        if not name:
            return
        self.new_class_edit.clear()
        self._merge_classes([name], replace_order=False)
        self._set_active_class(name)
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def _set_active_class(self, cls_name: str) -> None:
        if cls_name not in self._classes:
            self._classes.append(cls_name)
        self._active_label = cls_name
        self.canvas.set_active_label(cls_name)
        self._refresh_class_counts()

    def _on_class_row_changed(self, row: int) -> None:
        if self._syncing_lists:
            return
        if 0 <= row < len(self._classes):
            self._active_label = self._classes[row]
            self.canvas.set_active_label(self._active_label)

    def _on_class_double_clicked(self, item: QListWidgetItem) -> None:
        cls_name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not cls_name:
            return
        self._set_active_class(cls_name)
        sel = self.canvas.selected_index()
        if self._doc is not None and 0 <= sel < len(self._doc.boxes):
            self._relabel_box(sel, cls_name)

    def _apply_class_shortcut(self, cls_idx: int) -> None:
        if not (0 <= cls_idx < len(self._classes)):
            return
        cls_name = self._classes[cls_idx]
        self._set_active_class(cls_name)
        sel = self.canvas.selected_index()
        if self._doc is not None and 0 <= sel < len(self._doc.boxes):
            self._relabel_box(sel, cls_name)

    def _on_save_classes_txt(self) -> None:
        if not self._classes:
            QMessageBox.information(self, "提示", "当前类别清单为空，请先添加或导入类别。")
            return
        default_path = (
            (self._current_dir / "classes.txt")
            if self._current_dir is not None
            else (self.workspace_root / "classes.txt")
        )
        chosen, _ = QFileDialog.getSaveFileName(
            self, "保存 classes.txt 类别清单", str(default_path), "文本文件 (*.txt)"
        )
        if not chosen:
            return
        saved = save_classes_txt(Path(chosen), self._classes)
        self._last_classes_file = saved.resolve()
        self.save_state_label.setText(f"已导出 {saved.name}")

    # ------------------------------------------------------------------
    # 撤销/重做栈与标注编辑操作
    # ------------------------------------------------------------------
    def _push_undo_snapshot(self) -> None:
        if self._doc is None:
            return
        snap = ([b.clone() for b in self._doc.boxes], copy.deepcopy(self._doc.flags))
        self._undo_stack.append(snap)
        if len(self._undo_stack) > _MAX_UNDO_STEPS:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _apply_boxes_change(self, selected_idx: int = -1) -> None:
        if self._doc is None:
            return
        self._dirty = True
        self.canvas.set_boxes(self._doc.boxes, selected_index=selected_idx)
        self._refresh_class_counts()
        self._refresh_shape_list(selected_idx=selected_idx)
        self._refresh_review_banner()
        self._refresh_footer_and_title()
        self._refresh_action_states()

    def _undo(self) -> None:
        if self._doc is None or not self._undo_stack:
            return
        cur_snap = ([b.clone() for b in self._doc.boxes], copy.deepcopy(self._doc.flags))
        self._redo_stack.append(cur_snap)
        prev_boxes, prev_flags = self._undo_stack.pop()
        self._doc.boxes = prev_boxes
        self._doc.flags = prev_flags
        self._apply_boxes_change(selected_idx=-1)

    def _redo(self) -> None:
        if self._doc is None or not self._redo_stack:
            return
        cur_snap = ([b.clone() for b in self._doc.boxes], copy.deepcopy(self._doc.flags))
        self._undo_stack.append(cur_snap)
        next_boxes, next_flags = self._redo_stack.pop()
        self._doc.boxes = next_boxes
        self._doc.flags = next_flags
        self._apply_boxes_change(selected_idx=-1)

    def _on_canvas_box_drawn(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self._doc is None:
            return
        label = self._active_label
        if not label:
            dlg = _LabelPickerDialog(self._classes, title="为新标注框指定类别", parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            label = dlg.selected_label()
            self._set_active_class(label)

        self._push_undo_snapshot()
        new_box = RectAnnotation(label=label, x1=x1, y1=y1, x2=x2, y2=y2)
        if self._doc.image_width > 0 and self._doc.image_height > 0:
            new_box = new_box.clamped(self._doc.image_width, self._doc.image_height)
        self._doc.boxes.append(new_box)
        self._apply_boxes_change(selected_idx=len(self._doc.boxes) - 1)

    def _on_canvas_box_geometry_committed(
        self, idx: int, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        if self._doc is None or not (0 <= idx < len(self._doc.boxes)):
            return
        self._push_undo_snapshot()
        old = self._doc.boxes[idx]
        updated = RectAnnotation(
            label=old.label,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=old.score,
            description=old.description,
            difficult=old.difficult,
            group_id=old.group_id,
            flags=copy.deepcopy(old.flags),
        )
        if self._doc.image_width > 0 and self._doc.image_height > 0:
            updated = updated.clamped(self._doc.image_width, self._doc.image_height)
        self._doc.boxes[idx] = updated
        self._apply_boxes_change(selected_idx=idx)

    def _relabel_box(self, idx: int, new_label: str) -> None:
        if self._doc is None or not (0 <= idx < len(self._doc.boxes)):
            return
        clean = new_label.strip()
        if not clean or self._doc.boxes[idx].label == clean:
            return
        self._push_undo_snapshot()
        self._doc.boxes[idx].label = clean
        if clean not in self._classes:
            self._classes.append(clean)
        self._apply_boxes_change(selected_idx=idx)

    def _prompt_relabel_selected_box(self) -> None:
        idx = self.canvas.selected_index()
        if self._doc is None or not (0 <= idx < len(self._doc.boxes)):
            return
        cur_label = self._doc.boxes[idx].label
        dlg = _LabelPickerDialog(
            self._classes, current_label=cur_label, title="修改选中框类别", parent=self
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            chosen = dlg.selected_label()
            self._set_active_class(chosen)
            self._relabel_box(idx, chosen)

    def _duplicate_selected_box(self) -> None:
        idx = self.canvas.selected_index()
        if self._doc is None or not (0 <= idx < len(self._doc.boxes)):
            return
        self._push_undo_snapshot()
        src = self._doc.boxes[idx]
        offset = 14.0
        bw = src.width
        bh = src.height
        nx1 = src.x1 + offset
        ny1 = src.y1 + offset
        if self._doc.image_width > 0 and nx1 + bw > self._doc.image_width:
            nx1 = max(0.0, src.x1 - offset)
        if self._doc.image_height > 0 and ny1 + bh > self._doc.image_height:
            ny1 = max(0.0, src.y1 - offset)
        dup = RectAnnotation(
            label=src.label,
            x1=nx1,
            y1=ny1,
            x2=nx1 + bw,
            y2=ny1 + bh,
        )
        self._doc.boxes.append(dup)
        self._apply_boxes_change(selected_idx=len(self._doc.boxes) - 1)

    def _delete_selected_box(self) -> None:
        idx = self.canvas.selected_index()
        if self._doc is None or not (0 <= idx < len(self._doc.boxes)):
            return
        self._push_undo_snapshot()
        self._doc.boxes.pop(idx)
        next_sel = min(idx, len(self._doc.boxes) - 1)
        self._apply_boxes_change(selected_idx=next_sel)

    def _clear_all_boxes(self) -> None:
        if self._doc is None or not self._doc.boxes:
            return
        self._push_undo_snapshot()
        self._doc.boxes.clear()
        self._apply_boxes_change(selected_idx=-1)

    def _clear_missing_hint(self) -> None:
        if self._doc is None:
            return
        self._push_undo_snapshot()
        self._doc.flags["has_missing_hint"] = False
        self._doc.flags["missing_note"] = ""
        self._apply_boxes_change(selected_idx=self.canvas.selected_index())

    def _verify_and_next(self) -> None:
        """标记当前图已复核通过（自动清除漏标提醒并保存 .json，0 框时保存为显式空图负样本），再自动切下一张。"""
        if self._doc is None:
            return
        self._doc.flags["verified"] = True
        self._doc.flags["has_missing_hint"] = False
        self._doc.flags["missing_note"] = ""
        self._dirty = True
        if self.save_current_annotation(force=True):
            self._refresh_review_banner()
            self.open_next_image()

    # ------------------------------------------------------------------
    # 本地 SAM 像素级贴边收紧
    # ------------------------------------------------------------------
    def _on_sam_refine_clicked(self) -> None:
        if self._doc is None or not self._doc.boxes:
            return
        if self.sam_worker is not None and self.sam_worker.isRunning():
            return
        if self.gpu_guard is not None and not self.gpu_guard.try_acquire(self, "工作台本地 SAM 收框"):
            return

        sel = self.canvas.selected_index()
        if 0 <= sel < len(self._doc.boxes):
            targets = [(sel, self._doc.boxes[sel])]
        else:
            targets = list(enumerate(self._doc.boxes))

        dev_info = self.device_info
        if dev_info is None or dev_info.probing:
            dev_info = detect_compute_device()
            self.device_info = dev_info
        device = dev_info.recommended_device

        self.sam_worker = _SamBoxRefineWorker(
            self._doc.image_path,
            targets,
            device=device,
            model_name="sam2.1_t.pt",
            refiner_cache=self._sam_refiner_cache,
        )
        if self.gpu_guard is not None:
            self.gpu_guard.bind(self.sam_worker, "工作台本地 SAM 收框")
        self.sam_worker.refined_ready.connect(self._on_sam_refine_finished)
        self.sam_worker.failed.connect(self._on_sam_refine_failed)
        self.sam_worker.finished.connect(self._refresh_action_states)
        self._refresh_action_states()
        self.save_state_label.setText(f"正在用本地 SAM 2.1 收紧 {len(targets)} 个框...")
        self.sam_worker.start()

    def _on_sam_refine_finished(self, img_path_str: str, refined_map: dict) -> None:
        if self._doc is None or str(self._doc.image_path.resolve()) != img_path_str:
            return
        if not refined_map:
            self.save_state_label.setText("SAM 未检出更紧贴的有效轮廓，保持原框")
            return
        self._push_undo_snapshot()
        changed = 0
        for idx, coords in refined_map.items():
            if 0 <= idx < len(self._doc.boxes):
                x1, y1, x2, y2 = coords
                old = self._doc.boxes[idx]
                flags = copy.deepcopy(old.flags)
                flags["box_refined"] = True
                self._doc.boxes[idx] = RectAnnotation(
                    label=old.label,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=old.score,
                    description=old.description,
                    difficult=old.difficult,
                    group_id=old.group_id,
                    flags=flags,
                ).clamped(self._doc.image_width, self._doc.image_height)
                changed += 1
        self._apply_boxes_change(selected_idx=self.canvas.selected_index())
        self.save_state_label.setText(f"SAM 已收紧 {changed} 个框（Ctrl+Z 可一键撤回）")

    def _on_sam_refine_failed(self, msg: str) -> None:
        self.save_state_label.setText("SAM 收框失败")
        QMessageBox.warning(self, "本地 SAM 收框失败", msg)

    # ------------------------------------------------------------------
    # 右侧目标框列表与画布双向同步
    # ------------------------------------------------------------------
    def _refresh_shape_list(self, selected_idx: int = -1) -> None:
        self._syncing_lists = True
        try:
            self.shape_list.clear()
            boxes = self._doc.boxes if self._doc is not None else []
            self.obj_title_label.setText(f"当前图目标框 ({len(boxes)})")
            for idx, b in enumerate(boxes):
                score_part = f" · {b.score:.2f}" if b.score is not None else ""
                size_part = f" [{int(round(b.width))}×{int(round(b.height))}]"
                tag = f"#{idx + 1} {b.label}{score_part}{size_part}"
                item = QListWidgetItem(_make_color_icon(label_color_rgb(b.label)), tag)
                if b.description:
                    item.setToolTip(b.description)
                self.shape_list.addItem(item)
            if 0 <= selected_idx < len(boxes):
                self.shape_list.setCurrentRow(selected_idx)
        finally:
            self._syncing_lists = False
        self._refresh_box_detail(selected_idx)

    def _refresh_box_detail(self, idx: int) -> None:
        if self._doc is None or not (0 <= idx < len(self._doc.boxes)):
            self.box_detail_label.setText(
                "未选中目标框：单击画布或列表项可查看坐标、AI 置信度与审核理由；按数字键 1-9 可速改类别。"
            )
            return
        b = self._doc.boxes[idx]
        lines = [
            f"类别：{b.label}   尺寸：{int(round(b.width))} × {int(round(b.height))} px",
            f"坐标：({b.x1:.1f}, {b.y1:.1f}) → ({b.x2:.1f}, {b.y2:.1f})",
        ]
        meta_tags: list[str] = []
        if b.score is not None:
            meta_tags.append(f"置信度 {b.score:.3f}")
        if isinstance(b.flags, dict) and b.flags.get("llm_added"):
            meta_tags.append("LLM 补标")
        if isinstance(b.flags, dict) and b.flags.get("box_refined"):
            meta_tags.append("已收框")
        if meta_tags:
            lines.append("标签：" + " · ".join(meta_tags))
        if b.description:
            lines.append(f"审核说明：{b.description}")
        self.box_detail_label.setText("\n".join(lines))

    def _on_shape_row_changed(self, row: int) -> None:
        if self._syncing_lists:
            return
        self.canvas.select_box(row, emit_signal=False)
        self._refresh_box_detail(row)
        self._refresh_action_states()

    def _on_canvas_selection_changed(self, idx: int) -> None:
        self._syncing_lists = True
        try:
            if 0 <= idx < self.shape_list.count():
                self.shape_list.setCurrentRow(idx)
            else:
                self.shape_list.clearSelection()
                self.shape_list.setCurrentRow(-1)
        finally:
            self._syncing_lists = False
        self._refresh_box_detail(idx)
        self._refresh_action_states()

    def _on_canvas_mode_changed(self, mode: str) -> None:
        is_draw = mode == AnnotationCanvasView.MODE_DRAW
        self.edit_mode_btn.setChecked(not is_draw)
        self.draw_mode_btn.setChecked(is_draw)
        self.mode_badge_label.setText("模式: 画矩形框 (Esc 返回编辑)" if is_draw else "模式: 编辑")
        self.mode_badge_label.setStyleSheet(
            "border: none; color: #cf222e; font-weight: 600; font-size: 12px;"
            if is_draw
            else "border: none; color: #0969da; font-weight: 600; font-size: 12px;"
        )

    def _on_canvas_zoom_changed(self, scale: float) -> None:
        self.zoom_info_label.setText(f"缩放: {int(round(scale * 100))}%")

    def _on_canvas_cursor_changed(self, x: int, y: int) -> None:
        self.cursor_info_label.setText(f"坐标: ({x}, {y})" if x >= 0 and y >= 0 else "坐标: -")

    # ------------------------------------------------------------------
    # 文件列表右键菜单与顶部工作流按钮
    # ------------------------------------------------------------------
    def _on_file_list_context_menu(self, pos) -> None:
        item = self.file_list.itemAt(pos)
        if item is None:
            return
        img_path = Path(str(item.data(Qt.ItemDataRole.UserRole)))
        menu = QMenu(self)
        reveal_act = menu.addAction("在文件管理器中定位")
        clear_json_act = menu.addAction("删除同名 .json 标注文件")
        chosen = menu.exec(self.file_list.mapToGlobal(pos))
        if chosen is reveal_act:
            self._reveal_in_file_manager(img_path)
        elif chosen is clear_json_act:
            json_p = img_path.with_suffix(".json")
            if json_p.is_file():
                try:
                    json_p.unlink()
                except OSError as exc:
                    QMessageBox.warning(self, "删除失败", str(exc))
                    return
                if self._doc is not None and self._doc.image_path == img_path.resolve():
                    self._load_image(img_path)
                self._update_single_image_status(img_path)
                if self._current_dir is not None:
                    self.annotations_changed.emit(str(self._current_dir))

    @staticmethod
    def _reveal_in_file_manager(path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path.parent)])
        except OSError:
            pass

    def _on_pick_and_open_dir(self) -> None:
        start = str(self.current_directory() or (self.workspace_root / "data" / "raw_frames"))
        chosen = QFileDialog.getExistingDirectory(self, "选择待标注图片目录", start)
        if chosen:
            self.open_directory(Path(chosen))

    def _on_pick_and_load_classes(self) -> None:
        start = str(self.current_directory() or self.workspace_root)
        chosen, _ = QFileDialog.getOpenFileName(
            self, "选择 classes.txt 类别清单", start, "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if chosen:
            count = self.load_classes_file(Path(chosen))
            if count > 0:
                QMessageBox.information(self, "已导入类别", f"已向工作台加载 {count} 个类别。")
            else:
                QMessageBox.warning(self, "类别为空", "所选文件中未读取到有效类别名称。")

    def _on_refresh_dir(self) -> None:
        cur = self.current_directory()
        if cur is None or not cur.is_dir():
            QMessageBox.information(self, "提示", "当前尚未打开任何图片目录。")
            return
        cur_file = self.current_image_path()
        self.open_directory(
            cur,
            classes_file=self._last_classes_file,
            select_image=cur_file if cur_file and cur_file.is_file() else None,
        )

    def _on_send_to_dataset(self) -> None:
        cur = self.current_directory()
        if cur is None:
            QMessageBox.warning(self, "提示", "请先在工作台中打开待标注的图片目录。")
            return
        if not self._may_continue():
            return
        cls_path = ""
        if (cur / "classes.txt").is_file():
            cls_path = str((cur / "classes.txt").resolve())
        elif self._last_classes_file and self._last_classes_file.is_file():
            cls_path = str(self._last_classes_file)
        self.send_to_dataset.emit(str(cur), cls_path)

    def _on_send_to_review(self) -> None:
        cur = self.current_directory()
        if cur is None:
            QMessageBox.warning(self, "提示", "请先在工作台中打开图片目录。")
            return
        if not self._may_continue():
            return
        self.send_to_review.emit(str(cur))
