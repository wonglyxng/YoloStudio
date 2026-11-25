"""标签页 2：原生 PyQt6 轻量标注与复核工作台界面。

职责：
    提供面向 YOLO 目标检测（矩形框）与 AI 复核闭环的高性能原生 PyQt6 标注工作台：
    左侧图片文件列表（搜索/状态筛选/进度统计）、中央可精准锚点缩放与自由平移交互画布
    （`R/W` 拖拽或两点画框、`H` 抓手平移、`E` 编辑与 8 控制点精修、方向键微调、`Tab` 轮选、
    `Ctrl+C/V` 跨图粘贴、撤销重做、AI 漏标提醒横幅、切图保持视野与图片内存缓存）、
    右侧类别管理与目标框详情面板（展示 LLM/SAM 审核理由与置信度，支持一键本地 SAM 像素级贴边收框），
    并与「1. 视频抽帧」「3. 自动智能标注」「4. 数据集准备」实现双向路径与变更信号联动。

运行环境与协作边界：
    运行于 PyQt6 主线程，仅依赖 `PyQt6`、`Pillow` 及 `core.annotation_io`；
    完全不依赖 X-AnyLabeling 或 `PyQt6-WebEngine`。本地 SAM 贴边收框按需在后台 `QThread`
    中延迟加载 `core.mllm_reviewer.UltralyticsSamBoxRefiner` 并遵守 `GpuTaskGuard` 互斥保护。
    保存格式 100% 兼容标准 LabelMe / X-AnyLabeling `.json` 矩形标注协议。
"""

from __future__ import annotations

from collections import OrderedDict
import copy
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QEvent, QFileSystemWatcher, QObject, QPoint, QPointF, QRect, QSize, Qt, QThread, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QIntValidator, QKeyEvent, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
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
from yolo_studio.gui.common import GpuTaskGuard, install_combobox_wheel_guard

_ADDED_DLL_HANDLES: list[object] = []
_MAX_UNDO_STEPS = 80
_PIXMAP_CACHE_MAX = 8


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

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        min_width: int = 80,
    ) -> None:
        super().__init__(parent)
        self._full_text = ""
        self._fixed_min_w = max(0, int(min_width))
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(self._fixed_min_w)
        self.setText(text)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().minimumSizeHint()
        return QSize(self._fixed_min_w, hint.height())

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


class _CompactFooterLabel(QLabel):
    """展示完整底栏状态文案但不随文本长度撑大窗口最小宽度的标签。"""

    def __init__(
        self,
        text: str = "",
        *,
        min_width: int = 40,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self._fixed_min_w = max(0, int(min_width))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().minimumSizeHint()
        return QSize(min(self._fixed_min_w, hint.width()), hint.height())


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


def _make_status_dot_icon(
    status_key: str, *, verified: bool = False, size: int = 12
) -> QIcon:
    """绘制用于图片文件列表项的彩色状态圆点图标，清晰区分已标/待复核/空图/未标。"""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cx = size / 2.0
    if status_key == "labeled":
        if verified:
            painter.setPen(QPen(QColor("#2da44e"), 1.2))
            painter.setBrush(QColor("#1a7f37"))
            painter.drawEllipse(QPointF(cx, cx), 4.0, 4.0)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1a7f37"))
            painter.drawEllipse(QPointF(cx, cx), 3.6, 3.6)
    elif status_key == "review":
        painter.setPen(QPen(QColor("#9a6700"), 1.0))
        painter.setBrush(QColor("#d4a72c"))
        painter.drawEllipse(QPointF(cx, cx), 3.8, 3.8)
    elif status_key == "empty":
        painter.setPen(QPen(QColor("#0969da"), 1.7))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cx), 3.4, 3.4)
    else:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#afb8c1"))
        painter.drawEllipse(QPointF(cx, cx), 2.8, 2.8)
    painter.end()
    return QIcon(pix)


class _EmptyHintListWidget(QListWidget):
    """在列表为空时居中绘制浅灰引导提示的 `QListWidget`，消除大面积空白呆板感。"""

    def __init__(self, empty_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._empty_text = empty_text

    def set_empty_text(self, text: str) -> None:
        if text != self._empty_text:
            self._empty_text = text
            self.viewport().update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self.count() > 0 or not self._empty_text:
            return
        vp = self.viewport()
        painter = QPainter(vp)
        painter.setPen(QColor("#8c959f"))
        font = QFont(self.font())
        font.setPointSize(9)
        painter.setFont(font)
        rect = QRect(12, 12, max(40, vp.width() - 24), max(24, vp.height() - 24))
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._empty_text,
        )
        painter.end()


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
        install_combobox_wheel_guard()
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
        self._clipboard_boxes: list[RectAnnotation] = []
        self._clipboard_source_image: Path | None = None
        self._pixmap_cache: OrderedDict[tuple[Path, int, int], tuple[QPixmap, int, int]] = OrderedDict()
        self._last_nudge_key: tuple[Path, int] | None = None
        self._last_nudge_ts: float = 0.0

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

        # 1) 单行精简主工具栏（模式分段胶囊 + 框操作 + 翻页/通过 + 视图偏好 + 右端工作流出口）
        tools = QFrame()
        tools.setObjectName("workbenchTools")
        tools.setStyleSheet(
            "QFrame#workbenchTools { background-color: #ffffff; border: 1px solid #d8dee4; border-radius: 6px; }"
            "QPushButton { padding: 3px 7px; font-size: 12px; }"
            "QCheckBox { font-size: 12px; spacing: 3px; }"
            "QPushButton#segEditBtn { border-top-right-radius: 0px; border-bottom-right-radius: 0px; }"
            "QPushButton#segDrawBtn { border-radius: 0px; border-left: none; border-right: none; }"
            "QPushButton#segPanBtn { border-top-left-radius: 0px; border-bottom-left-radius: 0px; }"
            "QPushButton#segEditBtn:checked { background-color: #ddf4ff; border-color: #0969da; color: #0550ae; font-weight: 600; }"
            "QPushButton#segDrawBtn:checked { background-color: #ffebe9; border-color: #cf222e; color: #a40e26; font-weight: 600; }"
            "QPushButton#segPanBtn:checked { background-color: #fff8c5; border-color: #bf8700; color: #7d4e00; font-weight: 600; }"
            "QPushButton#verifyNextBtn { background-color: #eafaf1; border-color: #2da44e; color: #1a7f37; font-weight: 600; }"
            "QPushButton#verifyNextBtn:hover { background-color: #dafbe1; }"
            "QPushButton#verifyNextBtn:disabled { background-color: #f2f2f2; border-color: #d0d7de; color: #999999; }"
            "QPushButton#accentActionBtn { padding: 3px 8px; font-size: 12px; }"
        )
        t_lay = QHBoxLayout(tools)
        t_lay.setContentsMargins(6, 4, 6, 4)
        t_lay.setSpacing(4)

        self.open_dir_btn = QPushButton("打开目录")
        self.open_dir_btn.setToolTip("打开待标注图片目录（自动递归加载子目录与已有 .json）")

        mode_seg = QWidget()
        mode_seg_lay = QHBoxLayout(mode_seg)
        mode_seg_lay.setContentsMargins(0, 0, 0, 0)
        mode_seg_lay.setSpacing(0)
        self.edit_mode_btn = QPushButton("编辑(E)")
        self.edit_mode_btn.setObjectName("segEditBtn")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.setChecked(True)
        self.edit_mode_btn.setToolTip("编辑模式 (E)：点选/移动框、拖拽控制点、方向键微调，拖拽空白处可平移")
        self.draw_mode_btn = QPushButton("画框(R)")
        self.draw_mode_btn.setObjectName("segDrawBtn")
        self.draw_mode_btn.setCheckable(True)
        self.draw_mode_btn.setToolTip("画矩形框模式 (R / W)：支持按住拖拽成框或轻点起点+终点两点画框")
        self.pan_mode_btn = QPushButton("抓手(H)")
        self.pan_mode_btn.setObjectName("segPanBtn")
        self.pan_mode_btn.setCheckable(True)
        self.pan_mode_btn.setToolTip("抓手平移模式 (H)：左键直接拖动画布，不误碰目标框（也可随时右键/中键/空格+左键平移）")
        mode_seg_lay.addWidget(self.edit_mode_btn)
        mode_seg_lay.addWidget(self.draw_mode_btn)
        mode_seg_lay.addWidget(self.pan_mode_btn)

        self.dup_box_btn = QPushButton("复制")
        self.dup_box_btn.setToolTip("复制当前选中框 (Ctrl+D)")
        self.del_box_btn = QPushButton("删除")
        self.del_box_btn.setToolTip("删除当前选中框 (Del / Backspace)")
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.setToolTip("撤销上一步修改 (Ctrl+Z)")
        self.redo_btn = QPushButton("重做")
        self.redo_btn.setToolTip("重做修改 (Ctrl+Y)")

        self.prev_img_btn = QPushButton("◀")
        self.prev_img_btn.setFixedWidth(26)
        self.prev_img_btn.setToolTip("上一张图片 (A)")
        self.next_img_btn = QPushButton("▶")
        self.next_img_btn.setFixedWidth(26)
        self.next_img_btn.setToolTip("下一张图片 (D)")
        self.save_btn = QPushButton("保存")
        self.save_btn.setToolTip("立即保存当前图片标注到同名 .json (Ctrl+S)")
        self.verify_next_btn = QPushButton("✓ 通过(F)")
        self.verify_next_btn.setObjectName("verifyNextBtn")
        self.verify_next_btn.setToolTip("标记当前图已人工复核（0 框时保存为显式空图负样本）、清除漏标提醒并自动切换下一张 (F)")

        self.sam_refine_btn = QPushButton("SAM收框")
        self.sam_refine_btn.setToolTip("调用本地 SAM 2.1 对当前选中框（未选中时对整图全部框）执行像素级轮廓贴边收紧")

        self.keep_view_chk = QCheckBox("保持视野")
        self.keep_view_chk.setChecked(False)
        self.keep_view_chk.setToolTip("切图时保留当前的缩放倍率与局部观察位置（适合固定机位连续帧局部精修）")
        self.show_labels_chk = QCheckBox("标签")
        self.show_labels_chk.setChecked(True)
        self.show_labels_chk.setToolTip("显示/隐藏目标框类别与置信度角标 (快捷键 L)")
        self.auto_save_chk = QCheckBox("自动保存")
        self.auto_save_chk.setChecked(True)
        self.auto_save_chk.setToolTip("切图或退出前自动保存已修改的 .json 标注")

        self.to_review_btn = QPushButton("→智能标注")
        self.to_review_btn.setToolTip("将当前图片目录发送到「3. 自动智能标注」")
        self.to_dataset_btn = QPushButton("→数据集")
        self.to_dataset_btn.setObjectName("accentActionBtn")
        self.to_dataset_btn.setToolTip("将当前图片目录与类别清单发送到「4. 数据集准备」")

        for w in (
            self.open_dir_btn,
            self.edit_mode_btn,
            self.draw_mode_btn,
            self.pan_mode_btn,
            self.dup_box_btn,
            self.del_box_btn,
            self.undo_btn,
            self.redo_btn,
            self.prev_img_btn,
            self.next_img_btn,
            self.save_btn,
            self.verify_next_btn,
            self.sam_refine_btn,
            self.keep_view_chk,
            self.show_labels_chk,
            self.auto_save_chk,
            self.to_review_btn,
            self.to_dataset_btn,
        ):
            w.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        t_lay.addWidget(self.open_dir_btn)
        t_lay.addWidget(self._make_vsep())
        t_lay.addWidget(mode_seg)
        t_lay.addWidget(self._make_vsep())
        t_lay.addWidget(self.dup_box_btn)
        t_lay.addWidget(self.del_box_btn)
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
        t_lay.addWidget(self.keep_view_chk)
        t_lay.addWidget(self.show_labels_chk)
        t_lay.addWidget(self.auto_save_chk)
        t_lay.addStretch(1)
        t_lay.addWidget(self.to_review_btn)
        t_lay.addWidget(self.to_dataset_btn)
        root.addWidget(tools)

        # 2) 两栏主分割区：左侧超大标注画布 + 右侧三段可调卡片栏
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)

        # --- 左栏：AI 复核提醒横幅 + 核心画布 + 底部栏（保持原有底栏不变） ---
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
            "QPushButton { padding: 2px 7px; font-size: 11px; min-height: 20px; }"
        )
        f_lay = QHBoxLayout(footer)
        f_lay.setContentsMargins(8, 3, 8, 3)
        f_lay.setSpacing(8)
        self.mode_badge_label = _CompactFooterLabel("模式: 编辑", min_width=64)
        self.mode_badge_label.setStyleSheet("border: none; color: #0969da; font-weight: 600; font-size: 12px;")
        self.img_info_label = _CompactFooterLabel("分辨率: -", min_width=64)
        self.cursor_info_label = _CompactFooterLabel("坐标: -", min_width=52)
        self.hint_info_label = _ElidedStatusLabel(
            "滚轮缩放 · 拖空白/右键/空格平移 · 方向键微调选中框",
            min_width=0,
        )
        self.hint_info_label.setStyleSheet("border: none; color: #8c959f; font-size: 11px;")
        self.zoom_info_label = QLabel("缩放: 100%", self)
        self.zoom_info_label.setVisible(False)
        self.save_state_label = _CompactFooterLabel("", min_width=32)

        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedWidth(26)
        self.zoom_out_btn.setToolTip("缩小画布 (- / Ctrl+-)")
        self.zoom_menu_btn = QPushButton("100% ▾")
        self.zoom_menu_btn.setToolTip("点击选择预设缩放倍率")
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(26)
        self.zoom_in_btn.setToolTip("放大画布 (+ / Ctrl+=)")
        self.fit_btn = QPushButton("适应 (0)")
        self.fit_btn.setToolTip("将整张图片居中适配到画布可视区域 (0 / Ctrl+0 / 双击空白)")
        self.zoom_100_btn = QPushButton("1:1")
        self.zoom_100_btn.setToolTip("切换至 100% 原始像素比例 (Ctrl+1)")
        for zb in (
            self.zoom_out_btn,
            self.zoom_menu_btn,
            self.zoom_in_btn,
            self.fit_btn,
            self.zoom_100_btn,
        ):
            zb.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        f_lay.addWidget(self.mode_badge_label)
        f_lay.addWidget(self.img_info_label)
        f_lay.addWidget(self.cursor_info_label)
        f_lay.addWidget(self.hint_info_label, 1)
        f_lay.addWidget(self.save_state_label)
        f_lay.addWidget(self._make_vsep())
        f_lay.addWidget(self.zoom_out_btn)
        f_lay.addWidget(self.zoom_menu_btn)
        f_lay.addWidget(self.zoom_in_btn)
        f_lay.addWidget(self.fit_btn)
        f_lay.addWidget(self.zoom_100_btn)
        center_lay.addWidget(footer)
        self.main_splitter.addWidget(center_panel)

        # --- 右栏：三段可调白底圆角卡片（类别清单 / 当前图目标框 / 图片文件列表） ---
        right_panel = QWidget()
        right_panel.setMinimumWidth(270)
        right_panel.setMaximumWidth(440)
        right_panel.setStyleSheet(
            "QFrame#sideCard { background-color: #ffffff; border: 1px solid #d8dee4; border-radius: 6px; }"
        )
        right_outer = QVBoxLayout(right_panel)
        right_outer.setContentsMargins(0, 0, 0, 0)
        right_outer.setSpacing(0)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.right_splitter.setChildrenCollapsible(False)

        # 卡片 1：类别清单（导入/保存/添加/右键重命名与移除）
        cls_card = QFrame()
        cls_card.setObjectName("sideCard")
        cls_lay = QVBoxLayout(cls_card)
        cls_lay.setContentsMargins(8, 6, 8, 6)
        cls_lay.setSpacing(5)

        cls_hdr = QHBoxLayout()
        cls_hdr.setSpacing(4)
        cls_title = QLabel("类别清单 (1-9)")
        cls_title.setStyleSheet("font-weight: 600; color: #24292f;")
        self.load_classes_btn = QPushButton("导入")
        self.load_classes_btn.setToolTip("导入外部 classes.txt 类别清单...")
        self.save_classes_btn = QPushButton("保存")
        self.save_classes_btn.setToolTip("导出保存为 classes.txt...")
        for btn in (self.load_classes_btn, self.save_classes_btn):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet("padding: 2px 7px; font-size: 11px;")
        cls_hdr.addWidget(cls_title)
        cls_hdr.addStretch(1)
        cls_hdr.addWidget(self.load_classes_btn)
        cls_hdr.addWidget(self.save_classes_btn)
        cls_lay.addLayout(cls_hdr)

        add_cls_row = QHBoxLayout()
        add_cls_row.setSpacing(4)
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("输入新类别名称后回车...")
        self.add_class_btn = QPushButton("+ 添加")
        self.add_class_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_class_btn.setStyleSheet("padding: 3px 8px; font-size: 12px;")
        add_cls_row.addWidget(self.new_class_edit, 1)
        add_cls_row.addWidget(self.add_class_btn)
        cls_lay.addLayout(add_cls_row)

        self.unique_label_list = _EmptyHintListWidget(
            "暂无类别\n在上方输入名称回车添加，或点击「导入」"
        )
        self.unique_label_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.unique_label_list.setStyleSheet(
            "QListWidget { border: 1px solid #d0d7de; border-radius: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 3px 4px; }"
        )
        cls_lay.addWidget(self.unique_label_list, 1)
        self.right_splitter.addWidget(cls_card)

        # 卡片 2：当前图目标框 + 按需展开的紧凑详情卡
        obj_card = QFrame()
        obj_card.setObjectName("sideCard")
        obj_lay = QVBoxLayout(obj_card)
        obj_lay.setContentsMargins(8, 6, 8, 6)
        obj_lay.setSpacing(5)

        obj_hdr = QHBoxLayout()
        self.obj_title_label = QLabel("当前图目标框 (0)")
        self.obj_title_label.setStyleSheet("font-weight: 600; color: #24292f;")
        self.clear_boxes_btn = QPushButton("清空")
        self.clear_boxes_btn.setToolTip("清空当前图片全部目标框")
        self.clear_boxes_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.clear_boxes_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        obj_hdr.addWidget(self.obj_title_label)
        obj_hdr.addStretch(1)
        obj_hdr.addWidget(self.clear_boxes_btn)
        obj_lay.addLayout(obj_hdr)

        self.shape_list = _EmptyHintListWidget(
            "当前图暂无目标框\n按 R 进入画框模式，拖拽或两点点击即可标注"
        )
        self.shape_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.shape_list.setStyleSheet(
            "QListWidget { border: 1px solid #d0d7de; border-radius: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 3px 4px; }"
        )
        obj_lay.addWidget(self.shape_list, 1)

        self.box_detail_panel = QFrame()
        self.box_detail_panel.setStyleSheet(
            "QFrame { background-color: #f6f8fa; border: 1px solid #d8dee4; border-radius: 5px; }"
        )
        detail_lay = QVBoxLayout(self.box_detail_panel)
        detail_lay.setContentsMargins(6, 5, 6, 5)
        detail_lay.setSpacing(4)
        self.box_detail_label = QLabel("")
        self.box_detail_label.setWordWrap(True)
        self.box_detail_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.box_detail_label.setMinimumWidth(60)
        self.box_detail_label.setStyleSheet("border: none; color: #24292f; font-size: 11px;")
        detail_lay.addWidget(self.box_detail_label)

        box_ops = QHBoxLayout()
        box_ops.setSpacing(4)
        self.change_cls_btn = QPushButton("改类别")
        self.focus_box_btn = QPushButton("聚焦")
        self.focus_box_btn.setToolTip("在画布中放大并居中查看当前选中框")
        self.box_del_inline_btn = QPushButton("删除")
        for btn in (self.change_cls_btn, self.focus_box_btn, self.box_del_inline_btn):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setStyleSheet("padding: 3px 6px; font-size: 11px;")
            box_ops.addWidget(btn, 1)
        detail_lay.addLayout(box_ops)
        self.box_detail_panel.setVisible(False)
        obj_lay.addWidget(self.box_detail_panel)
        self.right_splitter.addWidget(obj_card)

        # 卡片 3：图片文件列表（中部省略文件名 + 彩色圆点 + 进度条 + 页码直达 + 下一未标）
        file_card = QFrame()
        file_card.setObjectName("sideCard")
        file_lay = QVBoxLayout(file_card)
        file_lay.setContentsMargins(8, 6, 8, 6)
        file_lay.setSpacing(5)

        file_hdr = QHBoxLayout()
        file_hdr.setSpacing(5)
        file_title = QLabel("图片列表")
        file_title.setStyleSheet("font-weight: 600; color: #24292f;")
        self.status_label = _ElidedStatusLabel("未打开目录")
        self.status_label.setStyleSheet("border: none; color: #57606a; font-size: 11px;")
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setToolTip("重新扫描当前目录中的图片与 .json 标注状态")
        self.refresh_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.refresh_btn.setStyleSheet("padding: 2px 7px; font-size: 11px;")
        file_hdr.addWidget(file_title)
        file_hdr.addWidget(self.status_label, 1)
        file_hdr.addWidget(self.refresh_btn)
        file_lay.addLayout(file_hdr)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        self.file_search = QLineEdit()
        self.file_search.setPlaceholderText("搜索文件名...")
        self.file_search.setClearButtonEnabled(True)
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItem("全部", "all")
        self.status_filter_combo.addItem("未标注", "unlabeled")
        self.status_filter_combo.addItem("已标注", "labeled")
        self.status_filter_combo.addItem("空图 (0框)", "empty")
        self.status_filter_combo.addItem("待复核", "review")
        self.status_filter_combo.setMinimumWidth(88)
        self.status_filter_combo.setMaximumWidth(112)
        self.status_filter_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        filter_row.addWidget(self.file_search, 1)
        filter_row.addWidget(self.status_filter_combo)
        file_lay.addLayout(filter_row)

        progress_row = QHBoxLayout()
        progress_row.setSpacing(4)
        self.progress_summary_label = _CompactFooterLabel("尚未打开目录", min_width=60)
        self.progress_summary_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.progress_summary_label.setStyleSheet("color: #57606a; font-size: 11px;")
        self.jump_index_edit = QLineEdit()
        self.jump_index_edit.setFixedWidth(48)
        self.jump_index_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.jump_index_edit.setPlaceholderText("跳页")
        self.jump_index_edit.setValidator(QIntValidator(1, 999999, self))
        self.jump_index_edit.setToolTip("输入当前列表序号 (1..N) 后回车快速跳转")
        self.jump_index_edit.setStyleSheet("padding: 1px 4px; font-size: 11px;")
        self.next_unlabeled_btn = QPushButton("下未标")
        self.next_unlabeled_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.next_unlabeled_btn.setToolTip("跳转到下一张未标注或待复核图片（支持循环查找）")
        self.next_unlabeled_btn.setStyleSheet("padding: 2px 6px; font-size: 11px;")
        progress_row.addWidget(self.progress_summary_label, 1)
        progress_row.addWidget(self.jump_index_edit)
        progress_row.addWidget(self.next_unlabeled_btn)
        file_lay.addLayout(progress_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: none; border-radius: 2px; background-color: #eaeef2; }"
            "QProgressBar::chunk { background-color: #2da44e; border-radius: 2px; }"
        )
        file_lay.addWidget(self.progress_bar)

        self.file_list = _EmptyHintListWidget(
            "尚未打开图片目录\n点击左上角「打开目录」选择图片文件夹"
        )
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.file_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.setStyleSheet(
            "QListWidget { border: 1px solid #d0d7de; border-radius: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 3px 4px; }"
        )
        file_lay.addWidget(self.file_list, 1)
        self.right_splitter.addWidget(file_card)
        self.right_splitter.setSizes([155, 200, 330])

        right_outer.addWidget(self.right_splitter)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setSizes([1040, 310])
        root.addWidget(self.main_splitter, 1)

        # 事件信号连接
        self.open_dir_btn.clicked.connect(self._on_pick_and_open_dir)
        self.load_classes_btn.clicked.connect(self._on_pick_and_load_classes)
        self.refresh_btn.clicked.connect(self._on_refresh_dir)
        self.to_dataset_btn.clicked.connect(self._on_send_to_dataset)
        self.to_review_btn.clicked.connect(self._on_send_to_review)

        self.edit_mode_btn.clicked.connect(lambda: self.canvas.set_mode(AnnotationCanvasView.MODE_EDIT))
        self.draw_mode_btn.clicked.connect(lambda: self.canvas.set_mode(AnnotationCanvasView.MODE_DRAW))
        self.pan_mode_btn.clicked.connect(lambda: self.canvas.set_mode(AnnotationCanvasView.MODE_PAN))
        self.dup_box_btn.clicked.connect(self._duplicate_selected_box)
        self.del_box_btn.clicked.connect(self._delete_selected_box)
        self.box_del_inline_btn.clicked.connect(self._delete_selected_box)
        self.change_cls_btn.clicked.connect(self._prompt_relabel_selected_box)
        self.focus_box_btn.clicked.connect(self._focus_selected_box)
        self.clear_boxes_btn.clicked.connect(self._clear_all_boxes)
        self.undo_btn.clicked.connect(self._undo)
        self.redo_btn.clicked.connect(self._redo)
        self.prev_img_btn.clicked.connect(self.open_prev_image)
        self.next_img_btn.clicked.connect(self.open_next_image)
        self.save_btn.clicked.connect(lambda: self.save_current_annotation(force=True))
        self.verify_next_btn.clicked.connect(self._verify_and_next)
        self.clear_hint_btn.clicked.connect(self._clear_missing_hint)
        self.sam_refine_btn.clicked.connect(self._on_sam_refine_clicked)
        self.keep_view_chk.toggled.connect(self._on_keep_view_toggled)
        self.zoom_out_btn.clicked.connect(lambda: self.canvas.zoom_by(1.0 / 1.25))
        self.zoom_in_btn.clicked.connect(lambda: self.canvas.zoom_by(1.25))
        self.zoom_menu_btn.clicked.connect(self._on_zoom_menu_clicked)
        self.fit_btn.clicked.connect(self.canvas.fit_to_window)
        self.zoom_100_btn.clicked.connect(lambda: self.canvas.set_zoom(1.0))
        self.show_labels_chk.toggled.connect(self.canvas.set_show_labels)

        self.file_search.textChanged.connect(self._apply_file_filter)
        self.status_filter_combo.currentIndexChanged.connect(self._apply_file_filter)
        self.jump_index_edit.returnPressed.connect(self._on_jump_to_index)
        self.next_unlabeled_btn.clicked.connect(self.open_next_unlabeled_image)
        self.file_list.currentRowChanged.connect(self._on_file_row_changed)
        self.file_list.customContextMenuRequested.connect(self._on_file_list_context_menu)

        self.add_class_btn.clicked.connect(self._on_add_class_from_input)
        self.new_class_edit.returnPressed.connect(self._on_add_class_from_input)
        self.save_classes_btn.clicked.connect(self._on_save_classes_txt)
        self.unique_label_list.currentRowChanged.connect(self._on_class_row_changed)
        self.unique_label_list.itemDoubleClicked.connect(self._on_class_double_clicked)
        self.unique_label_list.customContextMenuRequested.connect(self._on_class_list_context_menu)

        self.shape_list.currentRowChanged.connect(self._on_shape_row_changed)
        self.shape_list.itemDoubleClicked.connect(self._on_shape_double_clicked)
        self.shape_list.customContextMenuRequested.connect(self._on_shape_list_context_menu)

        self.canvas.box_drawn.connect(self._on_canvas_box_drawn)
        self.canvas.box_geometry_committed.connect(self._on_canvas_box_geometry_committed)
        self.canvas.selection_changed.connect(self._on_canvas_selection_changed)
        self.canvas.mode_changed.connect(self._on_canvas_mode_changed)
        self.canvas.zoom_changed.connect(self._on_canvas_zoom_changed)
        self.canvas.cursor_pos_changed.connect(self._on_canvas_cursor_changed)
        self.canvas.box_double_clicked.connect(lambda _idx: self._prompt_relabel_selected_box())
        self.canvas.context_menu_requested.connect(self._on_canvas_context_menu)

        # 在工作台自身与主要非文本子控件上安装键盘过滤器，保证全局空格平移与单键快捷键响应且绝不拦截文本输入
        for target in (self, self.canvas, self.file_list, self.shape_list, self.unique_label_list):
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
            ("Ctrl+C", self._copy_boxes_to_clipboard),
            ("Ctrl+V", self._paste_boxes_from_clipboard),
            ("Ctrl+=", lambda: self.canvas.zoom_by(1.25)),
            ("Ctrl++", lambda: self.canvas.zoom_by(1.25)),
            ("Ctrl+-", lambda: self.canvas.zoom_by(1.0 / 1.25)),
            ("Ctrl+0", self.canvas.fit_to_window),
            ("Ctrl+1", lambda: self.canvas.set_zoom(1.0)),
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

    def _is_list_navigation_widget(self, obj: QObject) -> bool:
        return obj in (
            self.file_list,
            self.shape_list,
            self.unique_label_list,
            self.status_filter_combo,
        )

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if not self.isVisible() or self._is_text_input_focused():
            return super().eventFilter(obj, event)

        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            mods = event.modifiers()
            key = event.key()
            has_cmd = bool(
                mods
                & (
                    Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                    | Qt.KeyboardModifier.MetaModifier
                )
            )
            if key == Qt.Key.Key_Space and not has_cmd:
                if not event.isAutoRepeat():
                    self.canvas.set_space_pressed(True)
                return True
            if key == Qt.Key.Key_Escape and not has_cmd:
                if self.canvas.cancel_active_draft():
                    return True
                if not self.canvas.editing():
                    self.canvas.set_mode(AnnotationCanvasView.MODE_EDIT)
                    return True
                if self.canvas.selected_index() >= 0:
                    self.canvas.select_box(-1, emit_signal=True)
                    return True
            if (
                key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab)
                and not has_cmd
                and not self._is_list_navigation_widget(obj)
                and self._doc is not None
                and self._doc.boxes
            ):
                reverse = key == Qt.Key.Key_Backtab or bool(
                    mods & Qt.KeyboardModifier.ShiftModifier
                )
                self._cycle_selected_box(reverse=reverse)
                return True
            if self._handle_workbench_keypress(event, target_obj=obj):
                return True

        elif event.type() == QEvent.Type.KeyRelease and isinstance(event, QKeyEvent):
            if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
                self.canvas.set_space_pressed(False)
                return True

        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if self.isVisible() and not self._is_text_input_focused():
            if self._handle_workbench_keypress(event, target_obj=self):
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if (
            self.isVisible()
            and not self._is_text_input_focused()
            and event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
        ):
            self.canvas.set_space_pressed(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _handle_workbench_keypress(
        self, event: QKeyEvent, *, target_obj: QObject | None = None
    ) -> bool:
        """处理工作台单键与方向微调快捷键；仅在无 Ctrl/Alt/Meta 修饰键时生效，返回 `True` 表示已消费。"""
        mods = event.modifiers()
        if mods & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False
        key = event.key()
        if key in (Qt.Key.Key_R, Qt.Key.Key_W):
            self.canvas.set_mode(AnnotationCanvasView.MODE_DRAW)
            return True
        if key == Qt.Key.Key_E:
            self.canvas.set_mode(AnnotationCanvasView.MODE_EDIT)
            return True
        if key == Qt.Key.Key_H:
            self.canvas.set_mode(AnnotationCanvasView.MODE_PAN)
            return True
        if key == Qt.Key.Key_L:
            self.show_labels_chk.toggle()
            return True
        if key == Qt.Key.Key_A:
            self.open_prev_image()
            return True
        if key == Qt.Key.Key_D:
            self.open_next_image()
            return True

        allow_box_nudge = (
            target_obj is None or not self._is_list_navigation_widget(target_obj)
        ) and self.canvas.selected_index() >= 0
        if allow_box_nudge and key in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            step = 10 if (mods & Qt.KeyboardModifier.ShiftModifier) else 1
            dx = -step if key == Qt.Key.Key_Left else (step if key == Qt.Key.Key_Right else 0)
            dy = -step if key == Qt.Key.Key_Up else (step if key == Qt.Key.Key_Down else 0)
            self._nudge_selected_box(dx, dy)
            return True

        if key == Qt.Key.Key_Left:
            self.open_prev_image()
            return True
        if key == Qt.Key.Key_Right:
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
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.canvas.zoom_by(1.25)
            return True
        if key == Qt.Key.Key_Minus:
            self.canvas.zoom_by(1.0 / 1.25)
            return True
        if key == Qt.Key.Key_0:
            self.canvas.fit_to_window()
            return True
        if Qt.Key.Key_1 <= key <= Qt.Key.Key_9 and not (
            mods & Qt.KeyboardModifier.ShiftModifier
        ):
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
        self.canvas.set_space_pressed(False)
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
        if self._current_dir != target_dir:
            self._pixmap_cache.clear()
        self._doc = None
        self._dirty = False
        self._last_nudge_key = None
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

    def open_next_unlabeled_image(self) -> bool:
        """循环跳转到当前筛选列表中的下一张未标注或待复核图片。"""
        total = len(self._filtered_images)
        if total == 0:
            return False
        cur_row = self.file_list.currentRow()
        start = 0 if cur_row < 0 else (cur_row + 1) % total
        for step in range(total):
            idx = (start + step) % total
            img_path = self._filtered_images[idx]
            summary = self._status_map.get(img_path) or inspect_image_status(img_path)
            if summary.status_key in ("unlabeled", "review"):
                if idx == cur_row:
                    self.save_state_label.setText("当前已是唯一未标注/待复核图片")
                else:
                    self.file_list.setCurrentRow(idx)
                self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
                return True
        self.save_state_label.setText("当前列表已全部标注完成 ✓")
        self.canvas.show_hud_message("当前列表已全部标注完成 ✓", color="#2da44e")
        return False

    def _on_jump_to_index(self) -> None:
        raw = self.jump_index_edit.text().strip()
        if not raw or not self._filtered_images:
            self.jump_index_edit.clear()
            self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        try:
            num = int(raw)
        except ValueError:
            self.jump_index_edit.clear()
            self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        target_row = max(0, min(len(self._filtered_images) - 1, num - 1))
        self.jump_index_edit.clear()
        self.file_list.setCurrentRow(target_row)
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def save_current_annotation(self, *, force: bool = False) -> bool:
        """保存当前图片标注到同名 `.json`；无改动且非 `force` 时直接返回 `True`。"""
        if self._doc is None:
            return True
        if not self._dirty and not force:
            return True

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
        if key in ("labeled", "review"):
            return f"{rel}  [{summary.box_count}]"
        if key == "empty":
            return f"{rel}  [0]"
        return rel

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
        if not self._all_images:
            if self._current_dir is None:
                self.file_list.set_empty_text(
                    "尚未打开图片目录\n点击左上角「打开目录」选择图片文件夹"
                )
            else:
                self.file_list.set_empty_text(
                    "当前目录暂无支持的图片\n支持 jpg / png / bmp / webp 等格式"
                )
        elif not filtered:
            self.file_list.set_empty_text("当前筛选条件下无匹配图片\n请调整搜索词或状态筛选")
        self._syncing_lists = True
        try:
            self.file_list.clear()
            target_row = -1
            for idx, img_path in enumerate(filtered):
                summary = self._status_map[img_path]
                item = QListWidgetItem(
                    _make_status_dot_icon(summary.status_key, verified=summary.is_verified),
                    self._format_file_item_text(img_path, summary),
                )
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

    def _load_image_pixmap_cached(self, image_path: Path) -> tuple[QPixmap, int, int]:
        """按路径与文件修改时间缓存近期浏览图片，减少往返切图解码延迟（最多保留 8 张）。"""
        resolved = image_path.resolve()
        try:
            st = resolved.stat()
            cache_key = (resolved, int(st.st_mtime_ns), int(st.st_size))
        except OSError:
            return load_image_pixmap(resolved)

        cached = self._pixmap_cache.get(cache_key)
        if cached is not None:
            self._pixmap_cache.move_to_end(cache_key)
            return cached

        loaded = load_image_pixmap(resolved)
        if not loaded[0].isNull():
            self._pixmap_cache[cache_key] = loaded
            while len(self._pixmap_cache) > _PIXMAP_CACHE_MAX:
                self._pixmap_cache.popitem(last=False)
        return loaded

    def _load_image(self, image_path: Path) -> None:
        preserve_view = (
            self.keep_view_chk.isChecked()
            and self.canvas.image_width > 0
            and self.canvas.image_height > 0
        )
        prev_scale = 1.0
        prev_center = QPointF()
        prev_fit = True
        prev_size = (0, 0)
        if preserve_view:
            prev_scale, prev_center, prev_fit = self.canvas.get_view_state()
            prev_size = (self.canvas.image_width, self.canvas.image_height)

        pix, w, h = self._load_image_pixmap_cached(image_path)
        doc = load_image_annotation(image_path, image_size=(w, h) if (w > 0 and h > 0) else None)
        self._doc = doc
        self._dirty = False
        self._last_nudge_key = None
        self._undo_stack.clear()
        self._redo_stack.clear()

        # 把当前图片新出现的类别纳入右侧清单
        new_labels = [b.label for b in doc.boxes if b.label and b.label not in self._classes]
        if new_labels:
            self._merge_classes(new_labels, replace_order=False)

        self.canvas.set_image(
            pix,
            doc.image_width,
            doc.image_height,
            auto_fit=not preserve_view,
        )
        if (
            preserve_view
            and not pix.isNull()
            and doc.image_width > 0
            and doc.image_height > 0
        ):
            if prev_fit:
                self.canvas.fit_to_window()
            else:
                if (doc.image_width, doc.image_height) == prev_size:
                    target_center = prev_center
                else:
                    target_center = QPointF(
                        prev_center.x() * float(doc.image_width) / max(1, prev_size[0]),
                        prev_center.y() * float(doc.image_height) / max(1, prev_size[1]),
                    )
                self.canvas.restore_view_state(prev_scale, target_center, fit_mode=False)

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
                    item.setIcon(
                        _make_status_dot_icon(summary.status_key, verified=summary.is_verified)
                    )
                    item.setText(self._format_file_item_text(resolved, summary))
                    self._style_file_item(item, summary)
                break
        self._refresh_progress_summary()

    def _refresh_progress_summary(self) -> None:
        total = len(self._all_images)
        if total == 0:
            self.progress_summary_label.setText(
                "当前目录无图片" if self._current_dir else "尚未打开目录"
            )
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_bar.setToolTip("")
            return
        labeled = sum(1 for s in self._status_map.values() if s.status_key == "labeled")
        empty = sum(1 for s in self._status_map.values() if s.status_key == "empty")
        review = sum(1 for s in self._status_map.values() if s.status_key == "review")
        unlabeled = total - labeled - empty - review
        done_count = labeled + empty
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done_count)
        pct = int(round(done_count * 100.0 / total))
        self.progress_bar.setToolTip(f"标注完成度：{done_count}/{total} ({pct}%)")

        pos = self.file_list.currentRow() + 1 if self.file_list.currentRow() >= 0 else 0
        shown = len(self._filtered_images)
        parts = [f"{pos}/{shown}" if shown == total else f"筛 {pos}/{shown} (共{total})"]
        parts.append(f"已标 {labeled}")
        if empty:
            parts.append(f"空 {empty}")
        if review:
            parts.append(f"待核 {review}")
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
        self.focus_box_btn.setEnabled(has_sel)
        self.clear_boxes_btn.setEnabled(has_boxes)
        self.undo_btn.setEnabled(len(self._undo_stack) > 0)
        self.redo_btn.setEnabled(len(self._redo_stack) > 0)
        self.save_btn.setEnabled(has_img)
        self.verify_next_btn.setEnabled(has_img)
        self.prev_img_btn.setEnabled(len(self._filtered_images) > 1)
        self.next_img_btn.setEnabled(len(self._filtered_images) > 1)
        self.jump_index_edit.setEnabled(len(self._filtered_images) > 0)
        self.next_unlabeled_btn.setEnabled(len(self._filtered_images) > 0)
        self.sam_refine_btn.setEnabled(has_boxes and not sam_busy)
        self.sam_refine_btn.setText("SAM收紧中" if sam_busy else "SAM收框")

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

    def _on_class_list_context_menu(self, pos: QPoint) -> None:
        item = self.unique_label_list.itemAt(pos)
        if item is None:
            return
        cls_name = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not cls_name:
            return
        menu = QMenu(self)
        use_act = menu.addAction(f"设为当前绘制类别「{cls_name}」")
        rename_act = menu.addAction("重命名类别...")
        menu.addSeparator()
        remove_act = menu.addAction("从清单移除")
        chosen = menu.exec(self.unique_label_list.mapToGlobal(pos))
        if chosen is use_act:
            self._set_active_class(cls_name)
        elif chosen is rename_act:
            self._prompt_rename_class(cls_name)
        elif chosen is remove_act:
            self._remove_class_from_list(cls_name)

    def _prompt_rename_class(self, old_name: str) -> None:
        new_name, ok = QInputDialog.getText(
            self, "重命名类别", f"将类别「{old_name}」重命名为：", text=old_name
        )
        if ok:
            self._rename_class(old_name, new_name)

    def _rename_class(self, old_name: str, new_name: str) -> bool:
        """重命名右侧类别并同步更新当前图片同名框（支持 Ctrl+Z 撤回）。"""
        clean = new_name.strip()
        if not clean or clean == old_name or old_name not in self._classes:
            return False
        if clean in self._classes:
            QMessageBox.warning(self, "重命名冲突", f"类别「{clean}」已存在。")
            return False

        idx = self._classes.index(old_name)
        self._classes[idx] = clean
        if self._active_label == old_name:
            self._active_label = clean
            self.canvas.set_active_label(clean)

        if self._doc is not None and any(b.label == old_name for b in self._doc.boxes):
            self._push_undo_snapshot()
            for box in self._doc.boxes:
                if box.label == old_name:
                    box.label = clean
            self._apply_boxes_change(selected_idx=self.canvas.selected_index())
        else:
            self._refresh_class_counts()
        self.save_state_label.setText(f"已重命名类别为「{clean}」")
        return True

    def _remove_class_from_list(self, cls_name: str) -> bool:
        """从右侧快捷类别清单移除指定类别（不删除画布已有目标框）。"""
        if cls_name not in self._classes:
            return False
        in_cur_doc = self._doc is not None and any(b.label == cls_name for b in self._doc.boxes)
        if in_cur_doc:
            reply = QMessageBox.question(
                self,
                "移除类别确认",
                f"当前图片中仍有目标使用类别「{cls_name}」。\n"
                "是否仅将其从右侧快捷清单移除？（画面已有框会保留）",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False

        self._classes.remove(cls_name)
        if self._active_label == cls_name:
            self._active_label = self._classes[0] if self._classes else ""
            self.canvas.set_active_label(self._active_label)
        self._refresh_class_counts()
        return True

    # ------------------------------------------------------------------
    # 撤销/重做栈与标注编辑操作
    # ------------------------------------------------------------------
    def _push_undo_snapshot(self) -> None:
        if self._doc is None:
            return
        self._last_nudge_key = None
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
        self._last_nudge_key = None
        cur_snap = ([b.clone() for b in self._doc.boxes], copy.deepcopy(self._doc.flags))
        self._redo_stack.append(cur_snap)
        prev_boxes, prev_flags = self._undo_stack.pop()
        self._doc.boxes = prev_boxes
        self._doc.flags = prev_flags
        self._apply_boxes_change(selected_idx=-1)

    def _redo(self) -> None:
        if self._doc is None or not self._redo_stack:
            return
        self._last_nudge_key = None
        cur_snap = ([b.clone() for b in self._doc.boxes], copy.deepcopy(self._doc.flags))
        self._undo_stack.append(cur_snap)
        next_boxes, next_flags = self._redo_stack.pop()
        self._doc.boxes = next_boxes
        self._doc.flags = next_flags
        self._apply_boxes_change(selected_idx=-1)

    def _cycle_selected_box(self, *, reverse: bool = False) -> None:
        """按 Tab / Shift+Tab 在当前图目标框间循环切换选中并确保新选中框处于可视区。"""
        if self._doc is None or not self._doc.boxes:
            return
        total = len(self._doc.boxes)
        cur = self.canvas.selected_index()
        if not (0 <= cur < total):
            next_idx = total - 1 if reverse else 0
        else:
            next_idx = (cur - 1) % total if reverse else (cur + 1) % total
        self.canvas.select_box(next_idx, emit_signal=True)
        self.canvas.ensure_box_visible(next_idx)

    def _nudge_selected_box(self, dx: int, dy: int) -> None:
        """使用方向键对选中框执行像素级平移（600 ms 内连续微调同一框合并为单次撤销记录）。"""
        idx = self.canvas.selected_index()
        if self._doc is None or not (0 <= idx < len(self._doc.boxes)):
            return
        old = self._doc.boxes[idx]
        bw = old.width
        bh = old.height
        nx1 = old.x1 + float(dx)
        ny1 = old.y1 + float(dy)
        if self._doc.image_width > 0:
            nx1 = max(0.0, min(float(self._doc.image_width) - bw, nx1))
        else:
            nx1 = max(0.0, nx1)
        if self._doc.image_height > 0:
            ny1 = max(0.0, min(float(self._doc.image_height) - bh, ny1))
        else:
            ny1 = max(0.0, ny1)

        if abs(nx1 - old.x1) < 1e-6 and abs(ny1 - old.y1) < 1e-6:
            return

        now = time.monotonic()
        nudge_key = (self._doc.image_path, idx)
        if (
            self._last_nudge_key != nudge_key
            or (now - self._last_nudge_ts) > 0.6
            or not self._undo_stack
        ):
            self._push_undo_snapshot()
        self._last_nudge_key = nudge_key
        self._last_nudge_ts = now

        self._doc.boxes[idx] = RectAnnotation(
            label=old.label,
            x1=nx1,
            y1=ny1,
            x2=nx1 + bw,
            y2=ny1 + bh,
            score=old.score,
            description=old.description,
            difficult=old.difficult,
            group_id=old.group_id,
            flags=copy.deepcopy(old.flags),
        )
        self._apply_boxes_change(selected_idx=idx)
        self.canvas.ensure_box_visible(idx)

    def _copy_boxes_to_clipboard(self) -> None:
        """复制当前选中框（无选中时复制整图全部框）到工作台剪贴板，支持跨图粘贴。"""
        if self._doc is None or not self._doc.boxes:
            self.save_state_label.setText("当前图无可复制目标框")
            return
        sel = self.canvas.selected_index()
        if 0 <= sel < len(self._doc.boxes):
            self._clipboard_boxes = [self._doc.boxes[sel].clone()]
            msg = f"已复制 1 个「{self._doc.boxes[sel].label}」框"
        else:
            self._clipboard_boxes = [b.clone() for b in self._doc.boxes]
            msg = f"已复制整图 {len(self._clipboard_boxes)} 个框"
        self._clipboard_source_image = self._doc.image_path
        self.save_state_label.setText(msg)

    def _paste_boxes_from_clipboard(self) -> None:
        """将剪贴板中的目标框粘贴到当前图片（同图偏移 14px，跨图原位保留并裁剪边界）。"""
        if self._doc is None or not self._clipboard_boxes:
            self.save_state_label.setText("剪贴板暂无可粘贴目标框")
            return

        same_image = self._clipboard_source_image == self._doc.image_path
        offset = 14.0
        pasted: list[RectAnnotation] = []
        unseen_labels: list[str] = []

        for src in self._clipboard_boxes:
            bw = src.width
            bh = src.height
            if same_image:
                nx1 = src.x1 + offset
                ny1 = src.y1 + offset
                if self._doc.image_width > 0 and nx1 + bw > self._doc.image_width:
                    nx1 = max(0.0, src.x1 - offset)
                if self._doc.image_height > 0 and ny1 + bh > self._doc.image_height:
                    ny1 = max(0.0, src.y1 - offset)
                candidate = RectAnnotation(
                    label=src.label,
                    x1=nx1,
                    y1=ny1,
                    x2=nx1 + bw,
                    y2=ny1 + bh,
                )
            else:
                candidate = RectAnnotation(
                    label=src.label,
                    x1=src.x1,
                    y1=src.y1,
                    x2=src.x2,
                    y2=src.y2,
                )
            if self._doc.image_width > 0 and self._doc.image_height > 0:
                candidate = candidate.clamped(self._doc.image_width, self._doc.image_height)
            if candidate.width < AnnotationCanvasView.MIN_BOX_SIZE_PX or candidate.height < AnnotationCanvasView.MIN_BOX_SIZE_PX:
                continue
            pasted.append(candidate)
            if candidate.label and candidate.label not in self._classes and candidate.label not in unseen_labels:
                unseen_labels.append(candidate.label)

        if not pasted:
            self.save_state_label.setText("粘贴框超出当前图像有效区域")
            return

        self._push_undo_snapshot()
        if unseen_labels:
            self._merge_classes(unseen_labels, replace_order=False)
        self._doc.boxes.extend(pasted)
        self._apply_boxes_change(selected_idx=len(self._doc.boxes) - 1)
        self.canvas.ensure_box_visible(len(self._doc.boxes) - 1)
        self.save_state_label.setText(f"已粘贴 {len(pasted)} 个目标框")

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
            self.box_detail_label.setText("")
            self.box_detail_panel.setVisible(False)
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
        self.box_detail_panel.setVisible(True)

    def _focus_selected_box(self) -> None:
        idx = self.canvas.selected_index()
        if idx >= 0:
            self.canvas.focus_box(idx)

    def _on_shape_row_changed(self, row: int) -> None:
        if self._syncing_lists:
            return
        self.canvas.select_box(row, emit_signal=False)
        if row >= 0:
            self.canvas.ensure_box_visible(row)
        self._refresh_box_detail(row)
        self._refresh_action_states()

    def _on_shape_double_clicked(self, _item: QListWidgetItem) -> None:
        row = self.shape_list.currentRow()
        if row >= 0:
            self.canvas.focus_box(row)
            self._prompt_relabel_selected_box()

    def _on_shape_list_context_menu(self, pos: QPoint) -> None:
        item = self.shape_list.itemAt(pos)
        if item is None or self._doc is None:
            return
        row = self.shape_list.row(item)
        if not (0 <= row < len(self._doc.boxes)):
            return
        self.shape_list.setCurrentRow(row)
        menu = QMenu(self)
        focus_act = menu.addAction("聚焦居中查看")
        relabel_act = menu.addAction("修改类别...")
        menu.addSeparator()
        dup_act = menu.addAction("复制框 (Ctrl+D)")
        copy_act = menu.addAction("复制到剪贴板 (Ctrl+C)")
        sam_act = menu.addAction("SAM 贴边收框")
        sam_busy = self.sam_worker is not None and self.sam_worker.isRunning()
        sam_act.setEnabled(not sam_busy)
        menu.addSeparator()
        del_act = menu.addAction("删除框 (Del)")
        chosen = menu.exec(self.shape_list.mapToGlobal(pos))
        if chosen is focus_act:
            self.canvas.focus_box(row)
        elif chosen is relabel_act:
            self._prompt_relabel_selected_box()
        elif chosen is dup_act:
            self._duplicate_selected_box()
        elif chosen is copy_act:
            self._copy_boxes_to_clipboard()
        elif chosen is sam_act:
            self._on_sam_refine_clicked()
        elif chosen is del_act:
            self._delete_selected_box()

    def _on_canvas_context_menu(self, global_pos: QPoint, hit_idx: int) -> None:
        if self._doc is None:
            return
        menu = QMenu(self)
        if 0 <= hit_idx < len(self._doc.boxes):
            self.canvas.select_box(hit_idx, emit_signal=True)
            relabel_act = menu.addAction("修改类别...")
            focus_act = menu.addAction("聚焦居中查看")
            menu.addSeparator()
            dup_act = menu.addAction("复制框 (Ctrl+D)")
            copy_act = menu.addAction("复制到剪贴板 (Ctrl+C)")
            sam_act = menu.addAction("SAM 贴边收框")
            sam_busy = self.sam_worker is not None and self.sam_worker.isRunning()
            sam_act.setEnabled(not sam_busy)
            menu.addSeparator()
            del_act = menu.addAction("删除框 (Del)")
            chosen = menu.exec(global_pos)
            if chosen is relabel_act:
                self._prompt_relabel_selected_box()
            elif chosen is focus_act:
                self.canvas.focus_box(hit_idx)
            elif chosen is dup_act:
                self._duplicate_selected_box()
            elif chosen is copy_act:
                self._copy_boxes_to_clipboard()
            elif chosen is sam_act:
                self._on_sam_refine_clicked()
            elif chosen is del_act:
                self._delete_selected_box()
            return

        edit_act = menu.addAction("编辑模式 (E)")
        edit_act.setCheckable(True)
        edit_act.setChecked(self.canvas.editing())
        draw_act = menu.addAction("画矩形框 (R)")
        draw_act.setCheckable(True)
        draw_act.setChecked(self.canvas.drawing())
        pan_act = menu.addAction("抓手平移 (H)")
        pan_act.setCheckable(True)
        pan_act.setChecked(self.canvas.panning_mode())
        menu.addSeparator()
        paste_act = menu.addAction("粘贴目标框 (Ctrl+V)")
        paste_act.setEnabled(bool(self._clipboard_boxes))
        copy_all_act = menu.addAction("复制整图所有框 (Ctrl+C)")
        copy_all_act.setEnabled(bool(self._doc.boxes))
        menu.addSeparator()
        fit_act = menu.addAction("适应窗口 (0)")
        zoom_100_act = menu.addAction("1:1 原始像素")
        toggle_lbl_act = menu.addAction("显示目标标签 (L)")
        toggle_lbl_act.setCheckable(True)
        toggle_lbl_act.setChecked(self.show_labels_chk.isChecked())

        chosen = menu.exec(global_pos)
        if chosen is edit_act:
            self.canvas.set_mode(AnnotationCanvasView.MODE_EDIT)
        elif chosen is draw_act:
            self.canvas.set_mode(AnnotationCanvasView.MODE_DRAW)
        elif chosen is pan_act:
            self.canvas.set_mode(AnnotationCanvasView.MODE_PAN)
        elif chosen is paste_act:
            self._paste_boxes_from_clipboard()
        elif chosen is copy_all_act:
            self.canvas.select_box(-1, emit_signal=True)
            self._copy_boxes_to_clipboard()
        elif chosen is fit_act:
            self.canvas.fit_to_window()
        elif chosen is zoom_100_act:
            self.canvas.set_zoom(1.0)
        elif chosen is toggle_lbl_act:
            self.show_labels_chk.toggle()

    def _on_zoom_menu_clicked(self) -> None:
        menu = QMenu(self)
        fit_act = menu.addAction("适应窗口 (0)")
        menu.addSeparator()
        presets: list[tuple[QAction, float]] = []
        for pct in (50, 100, 200, 400, 800):
            label = "100% (1:1)" if pct == 100 else f"{pct}%"
            presets.append((menu.addAction(label), pct / 100.0))
        chosen = menu.exec(self.zoom_menu_btn.mapToGlobal(QPoint(0, 0)))
        if chosen is None:
            return
        if chosen is fit_act:
            self.canvas.fit_to_window()
            return
        for act, scale in presets:
            if chosen is act:
                self.canvas.set_zoom(scale)
                return

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

    def _on_keep_view_toggled(self, checked: bool) -> None:
        if not (self.isVisible() and self._doc is not None):
            return
        if checked:
            self.canvas.show_hud_message("已开启「保持视野」：切图将保留缩放与观察位置", color="#0969da")
        else:
            self.canvas.show_hud_message(
                "已关闭「保持视野」：切图将自动适应窗口", color="#57606a", duration_ms=1200
            )

    def _on_canvas_mode_changed(self, mode: str) -> None:
        is_draw = mode == AnnotationCanvasView.MODE_DRAW
        is_pan = mode == AnnotationCanvasView.MODE_PAN
        is_edit = not is_draw and not is_pan
        self.edit_mode_btn.setChecked(is_edit)
        self.draw_mode_btn.setChecked(is_draw)
        self.pan_mode_btn.setChecked(is_pan)
        if is_draw:
            self.mode_badge_label.setText("模式: 画框 (拖/两点 · Esc退出)")
            self.mode_badge_label.setStyleSheet(
                "border: none; color: #cf222e; font-weight: 600; font-size: 12px;"
            )
            if self.isVisible() and self._doc is not None:
                self.canvas.show_hud_message(
                    "画框模式：按住拖拽，或轻点起点+终点成框（Esc 退出）", color="#cf222e"
                )
        elif is_pan:
            self.mode_badge_label.setText("模式: 抓手平移 (Esc返回)")
            self.mode_badge_label.setStyleSheet(
                "border: none; color: #9a6700; font-weight: 600; font-size: 12px;"
            )
            if self.isVisible() and self._doc is not None:
                self.canvas.show_hud_message(
                    "抓手模式：左键直接拖动画布，不误碰目标框", color="#d4a72c"
                )
        else:
            self.mode_badge_label.setText("模式: 编辑")
            self.mode_badge_label.setStyleSheet(
                "border: none; color: #0969da; font-weight: 600; font-size: 12px;"
            )
            if self.isVisible() and self._doc is not None:
                self.canvas.show_hud_message(
                    "编辑模式：点选/调框，拖拽空白处平移，方向键微调",
                    color="#0969da",
                    duration_ms=1200,
                )

    def _on_canvas_zoom_changed(self, scale: float) -> None:
        pct = int(round(scale * 100))
        self.zoom_info_label.setText(f"缩放: {pct}%")
        self.zoom_menu_btn.setText(f"{pct}% ▾")

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
