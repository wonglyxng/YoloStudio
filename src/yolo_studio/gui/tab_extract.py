"""标签页 1：录制视频抽帧界面。

职责：
    提供录制视频的拖拽/浏览导入、多选移除与元信息展示、抽帧策略与命名规则配置、
    选中视频只读预览、抽帧前产出量预估、后台多视频总进度监控与实时缩略图预览，
    并在抽帧完成或手动点击转发时向主窗口派发输出目录同步或切换工作台信号。

运行环境与协作边界：
    运行于 PyQt6 主线程；视频扫描委托给 `gui.workers.VideoProbeWorker`，
    视频预览帧抓取委托给 `gui.workers.VideoFramePreviewWorker`，
    耗时批量抽帧委托给 `gui.workers.FrameExtractWorker`（调用 `core.frame_extractor`）；
    通过 `frames_dir_synced`（静默同步）与 `send_frames_dir`（同步并跳转标注工作台）
    信号与 `gui.main_window.MainWindow` 协同。
"""

from __future__ import annotations

from collections import OrderedDict
import time
from pathlib import Path

from PyQt6.QtCore import QEvent, QMimeData, QRect, QSize, QThread, QTimer, Qt, pyqtSignal as Signal
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QImage,
    QKeyEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
    QResizeEvent,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
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
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.frame_extractor import (
    VIDEO_EXTENSIONS,
    ExtractionConfig,
    ExtractionMode,
    ExtractionProgress,
    ExtractionResult,
    VideoInfo,
    compile_naming_rule,
    estimate_video_frames,
    format_duration_hms,
    preview_naming_rule,
)
from yolo_studio.gui.common import (
    PathPickerRow,
    align_form_labels,
    configure_form_layout,
    open_local_path,
)
from yolo_studio.gui.workers import FrameExtractWorker, VideoFramePreviewWorker, VideoProbeWorker

# 列表项扩展角色：主角色 Qt.ItemDataRole.UserRole 仍保持存放绝对路径字符串
_ROLE_VIDEO_INFO = Qt.ItemDataRole.UserRole + 1
_ROLE_PROBE_FAILED = Qt.ItemDataRole.UserRole + 2
_ROLE_EST_FRAMES = Qt.ItemDataRole.UserRole + 3

# 预览帧内存缓存上限（按 `(视频绝对路径, 取帧秒数)` 缓存，避免切回列表重复解码）
_PREVIEW_CACHE_MAX_ITEMS = 32

# 状态消息等级对应的前景色
_STATUS_COLORS = {
    "info": "#57606a",
    "busy": "#0969da",
    "ok": "#1f7a1f",
    "warn": "#b25900",
    "error": "#cf222e",
}

# 命名规则常用快捷预设：(菜单文案, 填入规则字符串)
_NAMING_PRESETS: tuple[tuple[str, str], ...] = (
    ("默认内容哈希 + 毫秒  vc{hash}_f{ms:010d}", ""),
    ("视频名 + 6位序号  {video}_{seq:06d}", "{video}_{seq:06d}"),
    ("视频名 + 毫秒时间戳  {video}_f{ms:010d}", "{video}_f{ms:010d}"),
    ("视频名 + 原视频帧号  {video}_fr{frame:06d}", "{video}_fr{frame:06d}"),
    ("短哈希 + 6位序号  vc{hash:.8}_{seq:06d}", "vc{hash:.8}_{seq:06d}"),
)


def _extract_droppable_video_paths(mime_data: QMimeData | None) -> list[Path]:
    """从拖拽事件 `QMimeData` 中提取本地文件夹或受支持的视频文件路径列表。"""
    if mime_data is None or not mime_data.hasUrls():
        return []
    paths: list[Path] = []
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        candidate = Path(url.toLocalFile())
        if candidate.is_dir() or (
            candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS
        ):
            paths.append(candidate)
    return paths


class _VideoListWidget(QListWidget):
    """支持文件/文件夹拖放、多选、`Delete` 快捷键与空态引导画面的视频列表控件。

    协作边界：
        继承标准 `QListWidget`，完整保留 `addItem`、`item`、`count`、`selectedItems` 等接口；
        文件拖入与按键删除仅通过信号通知宿主标签页处理，不在控件内部直接修改业务路径列表。
    """

    paths_dropped = Signal(object)  # 参数：list[Path]
    remove_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._drag_active = False
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setUniformItemSizes(True)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setSpacing(2)
        self.setMinimumHeight(170)
        self.setStyleSheet(
            "QListWidget {"
            "  border: 1px solid #d0d7de;"
            "  border-radius: 6px;"
            "  background-color: #ffffff;"
            "  padding: 4px;"
            "  outline: 0;"
            "}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if _extract_droppable_video_paths(event.mimeData()):
            self._drag_active = True
            self.viewport().update()
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if _extract_droppable_video_paths(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # type: ignore[override]
        self._drag_active = False
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._drag_active = False
        self.viewport().update()
        paths = _extract_droppable_video_paths(event.mimeData())
        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()
            return
        event.ignore()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selectedItems():
            self.remove_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self.count() > 0 and not self._drag_active:
            return

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        vp_rect = self.viewport().rect().adjusted(10, 10, -10, -10)
        if vp_rect.width() < 60 or vp_rect.height() < 40:
            return

        if self._drag_active:
            painter.fillRect(vp_rect, QColor(234, 245, 255, 215))
            pen = QPen(QColor("#0969da"), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRoundedRect(vp_rect, 8, 8)
            title_font = QFont(self.font())
            title_font.setBold(True)
            painter.setFont(title_font)
            painter.setPen(QColor("#0969da"))
            painter.drawText(vp_rect, Qt.AlignmentFlag.AlignCenter, "松开鼠标即可导入视频或目录")
            return

        pen = QPen(QColor("#d0d7de"), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QColor("#fafbfc"))
        painter.drawRoundedRect(vp_rect, 8, 8)

        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSizeF(max(9.5, title_font.pointSizeF()))
        sub_font = QFont(self.font())
        sub_font.setPointSizeF(max(8.5, sub_font.pointSizeF() - 0.8))

        center_y = vp_rect.center().y()
        top_rect = QRect(vp_rect.left() + 12, center_y - 22, vp_rect.width() - 24, 22)
        bot_rect = QRect(vp_rect.left() + 12, center_y + 2, vp_rect.width() - 24, 20)

        painter.setFont(title_font)
        painter.setPen(QColor("#4b5563"))
        painter.drawText(top_rect, Qt.AlignmentFlag.AlignCenter, "拖拽视频文件或文件夹到此处")

        painter.setFont(sub_font)
        painter.setPen(QColor("#7b8490"))
        painter.drawText(
            bot_rect,
            Qt.AlignmentFlag.AlignCenter,
            "支持 .mp4 / .avi / .mov / .mkv 等格式，或点击上方「添加」按钮",
        )


class _VideoItemDelegate(QStyledItemDelegate):
    """两行卡片式视频列表项绘制代理：展示文件名、时长/预估徽章与规格摘要。"""

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:  # type: ignore[override]
        return QSize(max(120, option.rect.width()), 54)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:  # type: ignore[override]
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = option.rect.adjusted(2, 1, -2, -1)
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        if is_selected:
            painter.setPen(QPen(QColor("#b6d7ff"), 1))
            painter.setBrush(QColor("#eaf4ff"))
            painter.drawRoundedRect(rect, 5, 5)
            accent = QRect(rect.left() + 1, rect.top() + 5, 3, max(8, rect.height() - 10))
            painter.fillRect(accent, QColor("#0969da"))
        elif is_hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#f6f8fa"))
            painter.drawRoundedRect(rect, 5, 5)
        else:
            painter.setPen(QPen(QColor("#eef1f4"), 1))
            painter.drawLine(rect.left() + 6, rect.bottom(), rect.right() - 6, rect.bottom())

        info = index.data(_ROLE_VIDEO_INFO)
        probe_failed = bool(index.data(_ROLE_PROBE_FAILED))
        path_str = str(index.data(Qt.ItemDataRole.UserRole) or "")

        # 兼容外部仅插入普通文本 QListWidgetItem（如单元测试手工 addItem）的场景
        if not isinstance(info, VideoInfo) and not probe_failed:
            painter.setPen(QColor("#1f2328"))
            text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
            text_rect = rect.adjusted(10, 0, -10, 0)
            elided = QFontMetrics(painter.font()).elidedText(
                text, Qt.TextElideMode.ElideMiddle, max(40, text_rect.width())
            )
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)
            painter.restore()
            return

        content_rect = rect.adjusted(10, 5, -10, -5)
        top_h = 22
        bot_h = max(16, content_rect.height() - top_h)
        top_rect = QRect(content_rect.left(), content_rect.top(), content_rect.width(), top_h)
        bot_rect = QRect(content_rect.left(), content_rect.top() + top_h + 1, content_rect.width(), bot_h)

        # 准备第一行右侧徽章（预估抽取帧数 + 视频时长）
        badge_specs: list[tuple[str, str, str]] = []
        est_frames = index.data(_ROLE_EST_FRAMES)
        if isinstance(info, VideoInfo):
            if isinstance(est_frames, int):
                badge_specs.append((f"≈ {est_frames:,} 帧", "#edf4ff", "#0969da"))
            badge_specs.append((format_duration_hms(info.duration_sec), "#f0f2f5", "#24292f"))

        badge_font = QFont(option.font)
        badge_font.setPointSizeF(max(8.0, badge_font.pointSizeF() - 1.0))
        badge_fm = QFontMetrics(badge_font)

        right_cursor = top_rect.right()
        for text, bg_hex, fg_hex in reversed(badge_specs):
            bw = badge_fm.horizontalAdvance(text) + 14
            bh = 19
            bx = right_cursor - bw
            by = top_rect.top() + (top_rect.height() - bh) // 2
            if bx - 8 < top_rect.left() + 80:
                break
            b_rect = QRect(bx, by, bw, bh)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(bg_hex))
            painter.drawRoundedRect(b_rect, 4, 4)
            painter.setFont(badge_font)
            painter.setPen(QColor(fg_hex))
            painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, text)
            right_cursor = bx - 6

        # 第一行左侧：加粗视频文件名
        title_font = QFont(option.font)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#1f2328"))
        name_text = info.path.name if isinstance(info, VideoInfo) else Path(path_str).name
        name_avail = max(40, right_cursor - top_rect.left() - 4)
        elided_name = QFontMetrics(title_font).elidedText(
            name_text, Qt.TextElideMode.ElideMiddle, name_avail
        )
        painter.drawText(
            QRect(top_rect.left(), top_rect.top(), name_avail, top_rect.height()),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            elided_name,
        )

        # 第二行：规格摘要与父目录名
        sub_font = QFont(option.font)
        sub_font.setPointSizeF(max(8.2, sub_font.pointSizeF() - 1.0))
        painter.setFont(sub_font)
        if probe_failed:
            painter.setPen(QColor("#cf222e"))
            parent_name = Path(path_str).parent.name or path_str
            sub_text = f"读取元信息失败（可能文件损坏或编解码不受支持）  ·  目录: {parent_name}"
        else:
            painter.setPen(QColor("#57606a"))
            assert isinstance(info, VideoInfo)
            sub_text = (
                f"{info.width}×{info.height}  ·  {info.fps:.1f} fps  ·  "
                f"{info.frame_count:,} 帧  ·  目录: {info.path.parent.name}"
            )
        elided_sub = QFontMetrics(sub_font).elidedText(
            sub_text, Qt.TextElideMode.ElideRight, bot_rect.width()
        )
        painter.drawText(
            bot_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided_sub
        )

        painter.restore()


class _VideoPreviewCanvas(QLabel):
    """支持随分栏与窗口尺寸变化自动等比缩放渲染的柔和视频帧预览画布。"""

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(placeholder, parent)
        self._placeholder = placeholder
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(360, 210)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setWordWrap(True)
        self._apply_empty_style()

    def _apply_empty_style(self) -> None:
        self.setStyleSheet(
            "border: 1px dashed #5d6673; border-radius: 8px; "
            "background-color: #23272e; color: #b8c0cc; padding: 12px;"
        )

    def _apply_media_style(self) -> None:
        self.setStyleSheet(
            "border: 1px solid #424a57; border-radius: 8px; "
            "background-color: #1f242b; color: #d0d7de; padding: 4px;"
        )

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        """写入原始预览图并立即按当前画布尺寸等比平滑缩放展示。"""
        if pixmap.isNull():
            return
        self._source_pixmap = pixmap
        self._apply_media_style()
        self._apply_scaled()

    def clear_preview(self, placeholder: str | None = None) -> None:
        """清空当前画面并恢复默认柔和深灰空态引导文字。"""
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self.setText(placeholder or self._placeholder)
        self._apply_empty_style()

    def has_preview_image(self) -> bool:
        """返回当前画布是否已加载有效预览帧。"""
        return self._source_pixmap is not None and not self._source_pixmap.isNull()

    def _apply_scaled(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target_w = max(180, self.width() - 12)
        target_h = max(120, self.height() - 12)
        scaled = self._source_pixmap.scaled(
            target_w,
            target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._source_pixmap is not None and not self._source_pixmap.isNull():
            self._apply_scaled()


class _WrappingLabel(QLabel):
    """按实际宽度重算折行高度的标签，确保嵌套布局中多行状态与预览提示不被上下裁切。"""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__("", parent)
        self._last_height_hint = 0
        self._syncing_height = False
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        if text:
            self.setText(text)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(80, max(self._last_height_hint, 20))

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(240, max(self._last_height_hint, 20))

    def setText(self, text: str) -> None:  # type: ignore[override]
        super().setText(text)
        self._sync_wrapped_height()

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_wrapped_height()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._sync_wrapped_height()

    def changeEvent(self, event: QEvent) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._sync_wrapped_height()

    def _sync_wrapped_height(self) -> None:
        if self._syncing_height:
            return
        avail_w = self.width()
        if avail_w < 80:
            return
        self._syncing_height = True
        try:
            text = self.text()
            fm_rect = self.fontMetrics().boundingRect(
                QRect(0, 0, avail_w, 10000),
                int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                text,
            )
            qt_needed = super().heightForWidth(avail_w)
            needed = max(qt_needed, fm_rect.height(), super().sizeHint().height())
            target_h = max(20, needed + 6)
            if target_h != self._last_height_hint:
                self._last_height_hint = target_h
                self.updateGeometry()
                parent = self.parentWidget()
                if parent is not None:
                    parent.updateGeometry()
        finally:
            self._syncing_height = False


class _SummaryCard(QWidget):
    """右侧任务预估与运行状态卡片（标题 + 主数值 + 精简副说明）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)

        self.title_label = QLabel("-")
        self.title_label.setStyleSheet("color: #57606a; font-size: 12px;")
        self.value_label = QLabel("-")
        self.value_label.setStyleSheet("font-weight: 700; font-size: 15px; color: #1f2328;")
        self.detail_label = QLabel("-")
        self.detail_label.setStyleSheet("color: #6e7781; font-size: 11px;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_content(
        self, title: str, value: str, detail: str = "", *, color: str | None = None
    ) -> None:
        """更新卡片标题、主数值与底部副说明，支持按语义指定主数值颜色。"""
        self.title_label.setText(title)
        self.value_label.setText(value)
        color_css = f"color: {color};" if color else "color: #1f2328;"
        self.value_label.setStyleSheet(f"font-weight: 700; font-size: 15px; {color_css}")
        if detail:
            self.detail_label.setText(detail)
            self.detail_label.setVisible(True)
        else:
            self.detail_label.clear()
            self.detail_label.setVisible(False)


class ExtractTab(QWidget):
    """视频抽帧功能标签页（标签页 1）。

    内部状态与线程安全边界：
        所有状态仅在 PyQt6 主线程读写：
        - `_video_paths` 与 `video_list` 行一一对应，配合 `_video_info_map` 驱动预估与富文本列表展示；
        - `probe_worker` 负责异步扫描并探测视频元信息；
        - `preview_worker` / `_retired_preview_workers` 负责只读读取选中视频某时间点的预览画面，
          通过 `_preview_request_id` 丢弃过期回调，配合 `_preview_cache` 避免重复解码；
        - `worker` 负责批量抽帧，运行期锁定配置并按全部视频总帧数权重计算平滑单调总进度。
    """

    send_frames_dir = Signal(str)
    frames_dir_synced = Signal(str)

    def __init__(self, workspace_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self.worker: FrameExtractWorker | None = None
        self.probe_worker: VideoProbeWorker | None = None
        self.preview_worker: VideoFramePreviewWorker | None = None
        self._retired_preview_workers: list[VideoFramePreviewWorker] = []

        self._video_paths: list[Path] = []
        self._video_info_map: dict[Path, VideoInfo] = {}
        self._video_probe_failed: set[Path] = set()
        self._preview_cache: OrderedDict[tuple[Path, float], tuple[QPixmap, float]] = OrderedDict()

        self._controls_locked: bool = False
        self._allow_clear_when_locked: bool = False
        self._preview_request_id: int = 0
        self._extract_started_at: float = 0.0
        self._last_progress_pct: int = 0
        self._batch_video_frames: list[int] = []
        self._showing_extract_summary: bool = False

        # 起始时间调整后的预览防抖定时器（避免拖动/连击微调框时频繁起解码线程）
        self._preview_debounce_timer = QTimer(self)
        self._preview_debounce_timer.setSingleShot(True)
        self._preview_debounce_timer.setInterval(260)
        self._preview_debounce_timer.timeout.connect(self._request_selected_video_preview)

        # 抽帧期间每秒刷新“任务耗时”卡片的定时器
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([760, 560])

        root.addWidget(self.splitter)

        self._connect_signals()
        self._on_mode_changed()
        self._on_format_changed()
        self._refresh_naming_preview()
        self._refresh_estimates()
        self._refresh_list_action_states()
        self._set_status_message("就绪：请添加录制视频后点击「开始抽帧」。", level="info")
        self._set_preview_badge("就绪", "info")

    def _build_left_panel(self) -> QWidget:
        """构建左侧「视频列表 + 抽帧参数 + 操作按钮栏」面板。"""
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.setContentsMargins(0, 4, 4, 0)
        left.setSpacing(10)

        # 1. 视频列表分组
        self.video_group = QGroupBox("1. 选择录制视频")
        video_layout = QVBoxLayout(self.video_group)
        video_layout.setSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.add_files_btn = QPushButton("添加视频文件")
        self.add_folder_btn = QPushButton("添加视频文件夹")
        self.remove_btn = QPushButton("移除选中")
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.setObjectName("dangerBtn")
        btn_row.addWidget(self.add_files_btn)
        btn_row.addWidget(self.add_folder_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.remove_btn)
        btn_row.addWidget(self.clear_btn)
        video_layout.addLayout(btn_row)

        self.video_list = _VideoListWidget()
        self.video_list.setItemDelegate(_VideoItemDelegate(self.video_list))
        self.video_list.customContextMenuRequested.connect(self._on_list_context_menu)
        video_layout.addWidget(self.video_list, 1)
        left.addWidget(self.video_group, 1)

        # 2. 参数设置分组
        param_group = QGroupBox("2. 抽帧参数设置")
        form = QFormLayout(param_group)
        configure_form_layout(form, vertical_spacing=8)

        default_frames_dir = str((self.workspace_root / "data" / "raw_frames").resolve())
        self.output_picker = PathPickerRow(
            placeholder="选择抽帧图片输出目录", mode="dir", default_path=default_frames_dir
        )
        form.addRow("输出目录:", self.output_picker)

        # 策略下拉框与生效数值框同行，消除 2 行常驻置灰占位
        strategy_container = QWidget()
        strategy_row = QHBoxLayout(strategy_container)
        strategy_row.setContentsMargins(0, 0, 0, 0)
        strategy_row.setSpacing(6)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("按时间间隔抽帧 (每隔 N 秒)", ExtractionMode.INTERVAL_SECONDS)
        self.mode_combo.addItem("按固定帧步长抽帧 (每隔 N 帧)", ExtractionMode.EVERY_N_FRAMES)
        self.mode_combo.addItem("按目标总帧数均匀抽帧", ExtractionMode.TARGET_COUNT)
        self.mode_combo.setMinimumWidth(210)

        self.strategy_param_label = QLabel("间隔:")
        self.strategy_param_stack = QStackedWidget()
        self.strategy_param_stack.setFixedWidth(136)
        self.strategy_param_stack.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.05, 3600.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" 秒")

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 100000)
        self.frame_step_spin.setValue(30)
        self.frame_step_spin.setSuffix(" 帧")

        self.target_count_spin = QSpinBox()
        self.target_count_spin.setRange(1, 100000)
        self.target_count_spin.setValue(100)
        self.target_count_spin.setSuffix(" 张/视频")

        self.strategy_param_stack.addWidget(self.interval_spin)
        self.strategy_param_stack.addWidget(self.frame_step_spin)
        self.strategy_param_stack.addWidget(self.target_count_spin)

        strategy_row.addWidget(self.mode_combo)
        strategy_row.addSpacing(6)
        strategy_row.addWidget(self.strategy_param_label)
        strategy_row.addWidget(self.strategy_param_stack)
        strategy_row.addStretch(1)
        form.addRow("抽帧策略:", strategy_container)

        # 截取时间段
        time_container = QWidget()
        time_row = QHBoxLayout(time_container)
        time_row.setContentsMargins(0, 0, 0, 0)
        time_row.setSpacing(6)

        self.start_sec_spin = QDoubleSpinBox()
        self.start_sec_spin.setRange(0.0, 86400.0)
        self.start_sec_spin.setValue(0.0)
        self.start_sec_spin.setSuffix(" 秒")
        self.start_sec_spin.setFixedWidth(108)
        self.start_sec_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.end_sec_spin = QDoubleSpinBox()
        self.end_sec_spin.setRange(0.0, 86400.0)
        self.end_sec_spin.setValue(0.0)
        self.end_sec_spin.setSuffix(" 秒")
        self.end_sec_spin.setSpecialValueText("0.00 秒")
        self.end_sec_spin.setFixedWidth(108)
        self.end_sec_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.time_hint_label = QLabel("止为 0 表示抽到结尾")
        self.time_hint_label.setStyleSheet("color: #57606a; font-size: 12px;")

        time_row.addWidget(QLabel("起:"))
        time_row.addWidget(self.start_sec_spin)
        time_row.addSpacing(10)
        time_row.addWidget(QLabel("止:"))
        time_row.addWidget(self.end_sec_spin)
        time_row.addSpacing(6)
        time_row.addWidget(self.time_hint_label)
        time_row.addStretch(1)
        form.addRow("截取时间段:", time_container)

        # 轻量水平分割线，区分采样范围与过滤/输出规格
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #e5e9ef; border: none;")
        form.addRow(divider)

        # 相似帧去重 + 内联说明
        dedup_container = QWidget()
        dedup_row = QHBoxLayout(dedup_container)
        dedup_row.setContentsMargins(0, 0, 0, 0)
        dedup_row.setSpacing(8)

        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.0, 50.0)
        self.dedup_spin.setSingleStep(0.5)
        self.dedup_spin.setValue(2.0)
        self.dedup_spin.setSpecialValueText("0.00 (不去重)")
        self.dedup_spin.setFixedWidth(148)
        self.dedup_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.dedup_spin.setToolTip("相邻保存帧灰度平均像素差低于该值时自动跳过，0 表示不去重")

        self.dedup_hint_label = QLabel("像素差阈值，0=不去重，越大过滤越严（推荐 1.5~3.0）")
        self.dedup_hint_label.setStyleSheet("color: #57606a; font-size: 12px;")
        dedup_row.addWidget(self.dedup_spin)
        dedup_row.addWidget(self.dedup_hint_label, 1)
        form.addRow("相似帧去重:", dedup_container)

        # 缩放最长边 + 格式 + JPG质量
        resize_container = QWidget()
        resize_row = QHBoxLayout(resize_container)
        resize_row.setContentsMargins(0, 0, 0, 0)
        resize_row.setSpacing(6)

        self.max_dim_spin = QSpinBox()
        self.max_dim_spin.setRange(0, 8192)
        self.max_dim_spin.setSingleStep(160)
        self.max_dim_spin.setValue(0)
        self.max_dim_spin.setSuffix(" px")
        self.max_dim_spin.setSpecialValueText("0 px (原尺寸)")
        self.max_dim_spin.setFixedWidth(152)
        self.max_dim_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.max_dim_spin.setToolTip("限制导出图片最长边像素，0 表示保持视频原始分辨率")

        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpg", "png"])
        self.format_combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self.quality_label = QLabel("JPG质量:")
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(10, 100)
        self.quality_spin.setValue(95)
        self.quality_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        resize_row.addWidget(self.max_dim_spin)
        resize_row.addSpacing(10)
        resize_row.addWidget(QLabel("格式:"))
        resize_row.addWidget(self.format_combo)
        resize_row.addSpacing(10)
        resize_row.addWidget(self.quality_label)
        resize_row.addWidget(self.quality_spin)
        resize_row.addStretch(1)
        form.addRow("缩放最长边:", resize_container)

        # 命名规则 + 快捷预设 + 子目录勾选
        naming_container = QWidget()
        naming_row = QHBoxLayout(naming_container)
        naming_row.setContentsMargins(0, 0, 0, 0)
        naming_row.setSpacing(6)

        self.naming_rule_edit = QLineEdit()
        self.naming_rule_edit.setPlaceholderText(
            "留空默认 vc{hash}_f{ms:010d}；可填前缀或占位符模板"
        )
        self.naming_rule_edit.setToolTip(
            "帧图片命名规则（不含扩展名）：\n"
            "• 留空：默认按 vc + 视频SHA256前32位 + _f + 10位毫秒时长（如 vc9f86d0..._f0000001500.jpg）\n"
            "• 普通文本（不含占位符，如 cam1_）：作为文件名前缀，输出 {前缀}{视频名}_{6位序号}\n"
            "• 占位符模板：支持 {video}、{hash}(32位)、{hash_full}(64位)、{ms}(毫秒)、{frame}(帧号)、{seq}/{index}(保存序号)"
        )
        # 保留旧属性别名，兼容外部脚本或测试读取
        self.prefix_edit = self.naming_rule_edit

        self.naming_preset_btn = QPushButton("预设")
        self.naming_preset_btn.setFixedWidth(74)
        self.naming_preset_btn.setToolTip("选择常用帧文件命名模板快速填入")
        preset_menu = QMenu(self.naming_preset_btn)
        for label, rule_tpl in _NAMING_PRESETS:
            action = preset_menu.addAction(label)
            action.triggered.connect(lambda _chk=False, tpl=rule_tpl: self._apply_naming_preset(tpl))
        self.naming_preset_btn.setMenu(preset_menu)

        self.subfolder_chk = QCheckBox("按视频名分建子目录")

        naming_row.addWidget(self.naming_rule_edit, 1)
        naming_row.addWidget(self.naming_preset_btn)
        naming_row.addWidget(self.subfolder_chk)
        form.addRow("命名规则:", naming_container)

        self.naming_preview_label = _WrappingLabel()
        self.naming_preview_label.setStyleSheet("color: #57606a; font-size: 12px;")
        form.addRow("", self.naming_preview_label)

        align_form_labels([form])
        left.addWidget(param_group)

        # 底部流程控制区
        flow_row = QHBoxLayout()
        flow_row.setContentsMargins(2, 0, 2, 0)
        self.auto_open_workbench_chk = QCheckBox("抽帧完成后自动进入「2. 标注工作台」")
        self.auto_open_workbench_chk.setStyleSheet("color: #57606a; font-size: 12px;")
        self.auto_open_workbench_chk.setToolTip(
            "默认抽帧完成后停留在当前页查看统计摘要并高亮右侧流转按钮；勾选后无错完成时自动切至标注工作台"
        )
        flow_row.addWidget(self.auto_open_workbench_chk)
        flow_row.addStretch(1)
        left.addLayout(flow_row)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.start_btn = QPushButton("开始抽帧")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setMinimumWidth(116)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setFixedWidth(74)
        self.stop_btn.setEnabled(False)
        self.open_dir_btn = QPushButton("打开输出目录")
        self.send_next_btn = QPushButton("发送到「2. 标注工作台」→")
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.stop_btn)
        action_row.addStretch(1)
        action_row.addWidget(self.open_dir_btn)
        action_row.addWidget(self.send_next_btn)
        left.addLayout(action_row)

        return left_widget

    def _build_right_panel(self) -> QWidget:
        """构建右侧「画面预览 + 多视频总进度 + 任务预估/监控卡片」面板。"""
        right_widget = QWidget()
        right = QVBoxLayout(right_widget)
        right.setContentsMargins(4, 4, 0, 0)
        right.setSpacing(10)

        preview_group = QGroupBox("3. 实时抽帧预览与进度")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(10)

        # 顶部来源提示与状态徽章
        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self.preview_caption_label = QLabel("未选择视频")
        self.preview_caption_label.setStyleSheet("color: #57606a; font-size: 12px;")
        self.preview_caption_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        self.preview_badge_label = QLabel("就绪")
        self.preview_badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_badge_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        header_row.addWidget(self.preview_caption_label, 1)
        header_row.addWidget(self.preview_badge_label)
        preview_layout.addLayout(header_row)

        # 16:9 友好柔和预览画布（不再被拉成细长黑框）
        self.preview_label = _VideoPreviewCanvas(
            "添加并选中视频后可在此预览画面；抽帧开始后实时显示最新提取的帧"
        )
        preview_layout.addWidget(self.preview_label, 3)

        # 进度条 + 右侧独立进度文字标签，避免蓝灰分界处内嵌文本对比度撕裂
        progress_row = QHBoxLayout()
        progress_row.setSpacing(10)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFormat("0%")

        self.progress_text_label = QLabel("0%")
        self.progress_text_label.setStyleSheet("color: #24292f; font-size: 12px; font-weight: 600;")
        self.progress_text_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.progress_text_label)
        preview_layout.addLayout(progress_row)

        # 状态提示条（使用 #extractStatusBox 限定选择器，避免子 QLabel 继承产生双边框）
        status_box = QWidget()
        status_box.setObjectName("extractStatusBox")
        status_box.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        status_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        status_box.setStyleSheet(
            "#extractStatusBox {"
            "  background-color: #f6f8fa;"
            "  border: 1px solid #e5e9ef;"
            "  border-radius: 6px;"
            "}"
        )
        status_box_layout = QVBoxLayout(status_box)
        status_box_layout.setContentsMargins(10, 7, 10, 7)
        self.status_label = _WrappingLabel()
        status_box_layout.addWidget(self.status_label)
        preview_layout.addWidget(status_box)

        # 2×2 任务预估 / 实时监控卡片
        cards_grid = QGridLayout()
        cards_grid.setSpacing(8)
        self._card_videos = _SummaryCard()
        self._card_frames = _SummaryCard()
        self._card_dedup = _SummaryCard()
        self._card_output = _SummaryCard()
        self.summary_cards = (
            self._card_videos,
            self._card_frames,
            self._card_dedup,
            self._card_output,
        )
        cards_grid.addWidget(self._card_videos, 0, 0)
        cards_grid.addWidget(self._card_frames, 0, 1)
        cards_grid.addWidget(self._card_dedup, 1, 0)
        cards_grid.addWidget(self._card_output, 1, 1)
        preview_layout.addLayout(cards_grid)

        right.addWidget(preview_group, 1)
        return right_widget

    def _connect_signals(self) -> None:
        self.add_files_btn.clicked.connect(self._on_add_files)
        self.add_folder_btn.clicked.connect(self._on_add_folder)
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.clear_btn.clicked.connect(self._on_clear_videos)

        self.video_list.paths_dropped.connect(self._on_paths_dropped)
        self.video_list.remove_requested.connect(self._on_remove_selected)
        self.video_list.itemSelectionChanged.connect(self._on_selection_changed)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.interval_spin.valueChanged.connect(self._on_sampling_param_changed)
        self.frame_step_spin.valueChanged.connect(self._on_sampling_param_changed)
        self.target_count_spin.valueChanged.connect(self._on_sampling_param_changed)
        self.start_sec_spin.valueChanged.connect(self._on_start_sec_changed)
        self.end_sec_spin.valueChanged.connect(self._on_sampling_param_changed)
        self.dedup_spin.valueChanged.connect(self._on_output_param_changed)
        self.max_dim_spin.valueChanged.connect(self._on_output_param_changed)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.quality_spin.valueChanged.connect(self._on_output_param_changed)
        self.output_picker.textChanged.connect(self._on_output_param_changed)
        self.naming_rule_edit.textChanged.connect(self._refresh_naming_preview)
        self.subfolder_chk.toggled.connect(self._refresh_naming_preview)

        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.open_dir_btn.clicked.connect(lambda: open_local_path(self.output_picker.path()))
        self.send_next_btn.clicked.connect(self._on_send_next_clicked)

    # ------------------------------------------------------------------ 公共查询与窗口关闭辅助

    def running_preview_workers(self) -> list[QThread]:
        """返回仍在后台运行的只读视频预览解码线程，供主窗口退出前安全等待。"""
        workers: list[QThread] = []
        if self.preview_worker is not None and self.preview_worker.isRunning():
            workers.append(self.preview_worker)
        for old in self._retired_preview_workers:
            if old.isRunning():
                workers.append(old)
        return workers

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """关闭标签页时停掉内部定时器并等待活跃的只读预览线程收尾。"""
        self._preview_debounce_timer.stop()
        self._elapsed_timer.stop()
        for worker in self.running_preview_workers():
            worker.wait(500)
        super().closeEvent(event)

    # ------------------------------------------------------------------ 参数与联动响应

    def _on_mode_changed(self) -> None:
        """切换抽帧策略时原地切换数值控件页并更新各独立 SpinBox 启用态。"""
        mode = self.mode_combo.currentData()
        if mode == ExtractionMode.INTERVAL_SECONDS:
            self.strategy_param_stack.setCurrentIndex(0)
            self.strategy_param_label.setText("间隔:")
        elif mode == ExtractionMode.EVERY_N_FRAMES:
            self.strategy_param_stack.setCurrentIndex(1)
            self.strategy_param_label.setText("步长:")
        else:
            self.strategy_param_stack.setCurrentIndex(2)
            self.strategy_param_label.setText("目标:")

        self.interval_spin.setEnabled(mode == ExtractionMode.INTERVAL_SECONDS)
        self.frame_step_spin.setEnabled(mode == ExtractionMode.EVERY_N_FRAMES)
        self.target_count_spin.setEnabled(mode == ExtractionMode.TARGET_COUNT)
        self._on_sampling_param_changed()

    def _on_format_changed(self) -> None:
        """按输出图片格式联动 JPG 质量微调框：PNG 为无损格式时置灰禁用以防误解。"""
        is_jpg = self.format_combo.currentText().strip().lower() == "jpg"
        self.quality_label.setEnabled(is_jpg)
        self.quality_spin.setEnabled(is_jpg)
        tip = (
            "JPG 有损压缩质量（10~100，数值越大画质越好、体积越大）"
            if is_jpg
            else "PNG 为无损格式，不适用 JPG 压缩质量设置；切换为 jpg 格式后生效"
        )
        self.quality_label.setToolTip(tip)
        self.quality_spin.setToolTip(tip)
        self._refresh_naming_preview()
        self._on_output_param_changed()

    def _apply_naming_preset(self, template: str) -> None:
        """将菜单选择的常用命名规则填入输入框并聚焦。"""
        self.naming_rule_edit.setText(template)
        self.naming_rule_edit.setFocus()

    def _refresh_naming_preview(self) -> None:
        """按当前命名规则、输出图片格式与子目录勾选状态实时刷新示例路径。"""
        raw_rule = self.naming_rule_edit.text().strip()
        image_format = self.format_combo.currentText().strip() or "jpg"
        use_subfolder = self.subfolder_chk.isChecked()
        try:
            sample = preview_naming_rule(
                naming_rule=raw_rule,
                image_format=image_format,
                create_video_subfolder=use_subfolder,
            )
        except ValueError as exc:
            self.naming_preview_label.setStyleSheet("color: #cf222e; font-size: 12px;")
            self.naming_preview_label.setText(str(exc))
            return
        self.naming_preview_label.setStyleSheet("color: #57606a; font-size: 12px;")
        if not raw_rule:
            self.naming_preview_label.setText(f"默认命名示例：{sample}")
        else:
            self.naming_preview_label.setText(f"命名示例：{sample}")

    def _on_start_sec_changed(self) -> None:
        self._on_sampling_param_changed()
        if not self._is_extracting():
            self._preview_debounce_timer.start()

    def _on_sampling_param_changed(self) -> None:
        self._showing_extract_summary = False
        self._set_send_next_highlighted(False)
        self._refresh_estimates()

    def _on_output_param_changed(self) -> None:
        self._showing_extract_summary = False
        self._set_send_next_highlighted(False)
        if not self._is_extracting():
            self._refresh_pre_extract_cards()

    def _is_extracting(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _build_estimation_config(self) -> ExtractionConfig:
        """构造当前参数快照，用于静态预估每个视频的抽帧数量。"""
        out_text = self.output_picker.text() or str(self.workspace_root / "data" / "raw_frames")
        end_sec = self.end_sec_spin.value()
        return ExtractionConfig(
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
            naming_rule=self.naming_rule_edit.text().strip(),
            create_video_subfolder=self.subfolder_chk.isChecked(),
        )

    def _refresh_estimates(self) -> None:
        """同步刷新列表分组标题统计、各行预估抽取帧数与右侧抽帧前预估卡片。"""
        cfg = self._build_estimation_config()
        total_est = 0
        total_duration = 0.0
        known_info_count = 0

        for row in range(self.video_list.count()):
            item = self.video_list.item(row)
            if item is None:
                continue
            info = item.data(_ROLE_VIDEO_INFO)
            if isinstance(info, VideoInfo):
                est = estimate_video_frames(info, cfg)
                item.setData(_ROLE_EST_FRAMES, est)
                total_est += est
                total_duration += max(0.0, info.duration_sec)
                known_info_count += 1
            else:
                item.setData(_ROLE_EST_FRAMES, None)

        self.video_list.viewport().update()

        count = len(self._video_paths)
        if count <= 0:
            self.video_group.setTitle("1. 选择录制视频")
        elif known_info_count > 0:
            self.video_group.setTitle(
                f"1. 选择录制视频（共 {count} 个 · 总时长 {format_duration_hms(total_duration)}）"
            )
        else:
            self.video_group.setTitle(f"1. 选择录制视频（共 {count} 个）")

        if not self._is_extracting() and not self._showing_extract_summary:
            self._refresh_pre_extract_cards(
                known_info_count=known_info_count,
                total_duration=total_duration,
                total_est=total_est,
            )

    def _refresh_pre_extract_cards(
        self,
        *,
        known_info_count: int | None = None,
        total_duration: float | None = None,
        total_est: int | None = None,
    ) -> None:
        """在未启动抽帧时展示任务规模预估与输出配置摘要卡片。"""
        count = len(self._video_paths)
        if known_info_count is None or total_duration is None or total_est is None:
            cfg = self._build_estimation_config()
            known_info_count = 0
            total_duration = 0.0
            total_est = 0
            for vp in self._video_paths:
                info = self._video_info_map.get(vp.resolve())
                if info is not None:
                    known_info_count += 1
                    total_duration += max(0.0, info.duration_sec)
                    total_est += estimate_video_frames(info, cfg)

        if count <= 0:
            self._card_videos.set_content("待处理视频", "0 个", "请先添加录制视频")
            self._card_frames.set_content("预估抽取", "-", "添加视频后实时估算")
        else:
            dur_detail = (
                f"总时长 {format_duration_hms(total_duration)}"
                if known_info_count > 0
                else "等待读取视频元信息"
            )
            self._card_videos.set_content(
                "待处理视频", f"{count} 个", dur_detail, color="#0969da"
            )
            if known_info_count > 0:
                end_sec = self.end_sec_spin.value()
                if end_sec > 0 and end_sec <= self.start_sec_spin.value():
                    self._card_frames.set_content(
                        "预估抽取", "时间段无效", "结束时间需大于开始时间", color="#cf222e"
                    )
                else:
                    self._card_frames.set_content(
                        "预估抽取",
                        f"≈ {total_est:,} 张",
                        "未计入相似帧去重",
                        color="#1f7a1f" if total_est > 0 else "#b25900",
                    )
            else:
                self._card_frames.set_content("预估抽取", "-", "无有效元数据")

        dedup_val = self.dedup_spin.value()
        if dedup_val <= 0:
            self._card_dedup.set_content("相似帧去重", "已关闭", "保留全部采样帧")
        else:
            self._card_dedup.set_content(
                "相似帧去重", f"阈值 {dedup_val:.2f}", "自动过滤静止重复画面"
            )

        fmt = self.format_combo.currentText().strip().upper()
        fmt_txt = f"{fmt} Q{self.quality_spin.value()}" if fmt == "JPG" else fmt
        max_dim = self.max_dim_spin.value()
        size_txt = "原尺寸" if max_dim <= 0 else f"≤{max_dim}px"
        out_path = self.output_picker.text()
        out_leaf = Path(out_path).name if out_path else "未设置"
        self._card_output.set_content("输出规格", f"{fmt_txt} · {size_txt}", f"目录: {out_leaf}")

    # ------------------------------------------------------------------ 列表维护与状态控制

    def _refresh_list_action_states(self) -> None:
        """按当前锁定状态、列表数量与选中态刷新按钮可用性，空列表时禁用「开始抽帧」。"""
        if self._controls_locked:
            self.add_files_btn.setEnabled(False)
            self.add_folder_btn.setEnabled(False)
            self.remove_btn.setEnabled(False)
            self.clear_btn.setEnabled(self._allow_clear_when_locked)
            self.start_btn.setEnabled(False)
            return

        has_videos = bool(self._video_paths)
        self.add_files_btn.setEnabled(True)
        self.add_folder_btn.setEnabled(True)
        self.remove_btn.setEnabled(bool(self.video_list.selectedItems()))
        self.clear_btn.setEnabled(has_videos)
        self.start_btn.setEnabled(has_videos)
        self.start_btn.setToolTip(
            "开始对当前列表中的视频执行抽帧"
            if has_videos
            else "请先添加至少一个视频文件后再开始抽帧"
        )

    def _set_list_controls_enabled(self, enabled: bool, allow_clear: bool = True) -> None:
        """同步视频列表增删按钮与开始抽帧按钮的可用性，防止探测或抽帧期间并发改动列表。"""
        self._controls_locked = not enabled
        self._allow_clear_when_locked = allow_clear if not enabled else False
        self._refresh_list_action_states()

    def _set_status_message(self, text: str, *, level: str = "info") -> None:
        color = _STATUS_COLORS.get(level, "#57606a")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 13px;")
        self.status_label.setText(text)

    def _set_progress_display(self, value: int, text: str) -> None:
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(text)
        self.progress_text_label.setText(text)

    def _set_preview_badge(self, text: str, level: str = "info") -> None:
        styles = {
            "info": ("#f0f2f5", "#57606a"),
            "preview": ("#eaf5ff", "#0969da"),
            "busy": ("#eaf5ff", "#0969da"),
            "ok": ("#e6f6ea", "#1f7a1f"),
            "warn": ("#fff4e5", "#b25900"),
            "error": ("#fdeceb", "#cf222e"),
        }
        bg, fg = styles.get(level, styles["info"])
        self.preview_badge_label.setText(text)
        self.preview_badge_label.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border-radius: 4px; "
            "padding: 2px 8px; font-size: 11px; font-weight: 600;"
        )

    def _set_send_next_highlighted(self, highlighted: bool) -> None:
        target_name = "accentActionBtn" if highlighted else ""
        if self.send_next_btn.objectName() != target_name:
            self.send_next_btn.setObjectName(target_name)
            style = self.send_next_btn.style()
            if style is not None:
                style.unpolish(self.send_next_btn)
                style.polish(self.send_next_btn)
            self.send_next_btn.update()

    # ------------------------------------------------------------------ 视频添加、探测与选中预览

    def _add_video_paths(self, raw_paths: list[Path]) -> None:
        """启动后台 `VideoProbeWorker` 扫描并探测视频元信息，避免大量视频阻塞界面。"""
        if self.probe_worker and self.probe_worker.isRunning():
            self._set_status_message("正在扫描上一批视频，请稍候...", level="busy")
            return
        if self._is_extracting():
            return

        existing = {p.resolve() for p in self._video_paths}
        self._set_list_controls_enabled(False)
        self._set_status_message("正在后台扫描并读取视频元信息...", level="busy")

        self.probe_worker = VideoProbeWorker(raw_paths, existing)
        self.probe_worker.item_info_probed.connect(self._on_video_info_probed)
        self.probe_worker.finished_probing.connect(self._on_probe_finished)
        self.probe_worker.failed.connect(self._on_probe_failed)
        self.probe_worker.start()

    def _on_video_info_probed(self, vp: Path, info: object, text: str) -> None:
        """接收后台探测完成的单个视频信息并追加至列表；若探测已被取消则忽略残留队列信号。"""
        if self.sender() is not self.probe_worker or (
            self.probe_worker and self.probe_worker.cancel_event.is_set()
        ):
            return
        resolved = vp.resolve()
        if any(p.resolve() == resolved for p in self._video_paths):
            return

        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, str(vp))
        item.setToolTip(str(vp))
        if isinstance(info, VideoInfo):
            self._video_info_map[resolved] = info
            self._video_probe_failed.discard(resolved)
            item.setData(_ROLE_VIDEO_INFO, info)
            item.setData(_ROLE_PROBE_FAILED, False)
        else:
            self._video_probe_failed.add(resolved)
            item.setData(_ROLE_VIDEO_INFO, None)
            item.setData(_ROLE_PROBE_FAILED, True)

        self.video_list.addItem(item)
        self._video_paths.append(vp)
        self._refresh_estimates()

    def _on_probe_finished(self, added: int) -> None:
        if self.sender() is not self.probe_worker:
            return
        self._set_list_controls_enabled(True)
        self._showing_extract_summary = False
        self._refresh_estimates()
        self._set_status_message(
            f"已添加 {added} 个视频（当前列表共 {len(self._video_paths)} 个）。",
            level="ok" if added > 0 else "info",
        )
        # 若列表已有条目但尚未选中任何项，自动选中第 1 条并预览首帧画面
        if self.video_list.count() > 0 and not self.video_list.selectedItems():
            self.video_list.setCurrentRow(0)
        elif self.video_list.selectedItems() and not self.preview_label.has_preview_image():
            self._request_selected_video_preview()

    def _on_probe_failed(self, err: str) -> None:
        if self.sender() is not self.probe_worker:
            return
        self._set_list_controls_enabled(True)
        self._set_status_message(f"读取视频列表失败: {err}", level="error")

    def _on_paths_dropped(self, paths: object) -> None:
        if isinstance(paths, list) and paths and not self._is_extracting():
            self._add_video_paths([Path(p) for p in paths])

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
        folder = QFileDialog.getExistingDirectory(
            self, "选择包含视频的文件夹", str(self.workspace_root)
        )
        if folder:
            self._add_video_paths([Path(folder)])

    def _on_selection_changed(self) -> None:
        self._refresh_list_action_states()
        if not self._is_extracting():
            self._request_selected_video_preview()

    def _on_remove_selected(self) -> None:
        if self._controls_locked:
            return
        rows = sorted(
            {self.video_list.row(item) for item in self.video_list.selectedItems()},
            reverse=True,
        )
        for row in rows:
            item = self.video_list.takeItem(row)
            del item
            if 0 <= row < len(self._video_paths):
                removed = self._video_paths.pop(row)
                resolved = removed.resolve()
                self._video_info_map.pop(resolved, None)
                self._video_probe_failed.discard(resolved)

        self._showing_extract_summary = False
        self._refresh_estimates()
        self._refresh_list_action_states()
        if not self._video_paths:
            self._preview_request_id += 1
            self.preview_label.clear_preview()
            self.preview_caption_label.setText("未选择视频")
            self._set_preview_badge("就绪", "info")
            self._set_status_message("就绪：请添加录制视频后点击「开始抽帧」。", level="info")
        elif self.video_list.selectedItems():
            self._request_selected_video_preview()
        else:
            self.preview_caption_label.setText("点击左侧视频可预览抽帧起点画面")
            self._set_preview_badge("就绪", "info")

    def _on_clear_videos(self) -> None:
        if self.probe_worker and self.probe_worker.isRunning():
            self.probe_worker.cancel()
        self.video_list.clear()
        self._video_paths.clear()
        self._video_info_map.clear()
        self._video_probe_failed.clear()
        self._showing_extract_summary = False
        self._set_send_next_highlighted(False)
        self._set_list_controls_enabled(True)
        self._refresh_estimates()
        self._preview_request_id += 1
        self.preview_label.clear_preview()
        self.preview_caption_label.setText("未选择视频")
        self._set_preview_badge("就绪", "info")
        self._set_progress_display(0, "0%")
        self._set_status_message("已清空视频列表，可重新拖拽或添加视频。", level="info")

    def _on_list_context_menu(self, pos) -> None:
        item = self.video_list.itemAt(pos)
        if item is None:
            return
        path_str = str(item.data(Qt.ItemDataRole.UserRole) or "")
        menu = QMenu(self)
        act_preview = menu.addAction("刷新该视频画面预览")
        act_open_folder = menu.addAction("打开所在文件夹")
        menu.addSeparator()
        act_remove = menu.addAction("从列表中移除选中")
        act_remove.setEnabled(not self._controls_locked)

        chosen = menu.exec(self.video_list.viewport().mapToGlobal(pos))
        if chosen is act_preview:
            self.video_list.setCurrentItem(item)
            self._request_selected_video_preview()
        elif chosen is act_open_folder and path_str:
            p = Path(path_str)
            open_local_path(p.parent if p.parent.exists() else p)
        elif chosen is act_remove:
            self._on_remove_selected()

    def _request_selected_video_preview(self) -> None:
        """在后台抓帧预览当前选中视频在起始时间点（`start_sec`）的画面，优先命中内存缓存。"""
        if self._is_extracting():
            return
        selected = self.video_list.selectedItems()
        if not selected:
            return
        item = selected[0]
        raw_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not raw_path:
            return
        video_path = Path(raw_path)

        try:
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                self._preview_request_id += 1
                self.preview_label.clear_preview("当前条目未包含可解码的视频画面")
                self.preview_caption_label.setText(video_path.name)
                self._set_preview_badge("无画面", "info")
                return
        except OSError:
            return

        target_sec = round(max(0.0, self.start_sec_spin.value()), 1)
        resolved = video_path.resolve()
        if resolved in self._video_probe_failed:
            self.preview_label.clear_preview("视频文件读取失败，无法解码预览画面")
            self.preview_caption_label.setText(video_path.name)
            self._set_preview_badge("读取失败", "warn")
            return

        cache_key = (resolved, target_sec)
        cached = self._preview_cache.get(cache_key)
        if cached is not None:
            self._preview_cache.move_to_end(cache_key)
            pix, actual_sec = cached
            self._show_video_preview_pixmap(video_path, pix, actual_sec)
            return

        self._preview_request_id += 1
        req_id = self._preview_request_id
        self.preview_caption_label.setText(f"正在加载画面预览: {video_path.name} ...")
        self._set_preview_badge("加载中", "preview")

        if self.preview_worker is not None and self.preview_worker.isRunning():
            self._retired_preview_workers.append(self.preview_worker)

        worker = VideoFramePreviewWorker(
            video_path,
            timestamp_sec=target_sec,
            request_id=req_id,
            max_dimension=960,
        )
        self.preview_worker = worker
        worker.preview_ready.connect(
            lambda rid, vp, sec, qimg, key=cache_key: self._on_video_preview_ready(
                rid, vp, sec, qimg, key
            )
        )
        worker.preview_failed.connect(self._on_video_preview_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_preview_worker(w))
        worker.start()

    def _cleanup_preview_worker(self, worker: VideoFramePreviewWorker) -> None:
        if worker in self._retired_preview_workers:
            self._retired_preview_workers.remove(worker)

    def _on_video_preview_ready(
        self,
        request_id: int,
        video_path: Path,
        actual_sec: float,
        image: object,
        cache_key: tuple[Path, float],
    ) -> None:
        if not isinstance(image, QImage) or image.isNull():
            return
        pix = QPixmap.fromImage(image)
        if pix.isNull():
            return
        self._preview_cache[cache_key] = (pix, actual_sec)
        self._preview_cache.move_to_end(cache_key)
        while len(self._preview_cache) > _PREVIEW_CACHE_MAX_ITEMS:
            self._preview_cache.popitem(last=False)

        if request_id != self._preview_request_id or self._is_extracting():
            return
        self._show_video_preview_pixmap(video_path, pix, actual_sec)

    def _show_video_preview_pixmap(self, video_path: Path, pix: QPixmap, actual_sec: float) -> None:
        self.preview_label.set_preview_pixmap(pix)
        info = self._video_info_map.get(video_path.resolve())
        res_txt = f"{info.width}×{info.height}" if info is not None else f"{pix.width()}×{pix.height()}"
        self.preview_caption_label.setText(
            f"预览: {video_path.name}  ({res_txt} · {format_duration_hms(actual_sec)})"
        )
        self.preview_caption_label.setToolTip(str(video_path))
        self._set_preview_badge("视频预览", "preview")

    def _on_video_preview_failed(self, request_id: int, err: str) -> None:
        if request_id != self._preview_request_id or self._is_extracting():
            return
        self.preview_label.clear_preview(err)
        self._set_preview_badge("无画面", "warn")

    # ------------------------------------------------------------------ 抽帧执行与多视频进度

    def _on_start(self) -> None:
        """校验视频列表、输出目录、截取时间段与命名规则后启动后台 `FrameExtractWorker`。"""
        if not self._video_paths:
            QMessageBox.warning(self, "提示", "请先添加至少一个视频文件。")
            return
        out_text = self.output_picker.text()
        if not out_text:
            QMessageBox.warning(self, "提示", "请选择输出目录。")
            return

        start_sec = self.start_sec_spin.value()
        end_sec = self.end_sec_spin.value()
        if end_sec > 0 and end_sec <= start_sec:
            QMessageBox.warning(
                self,
                "提示",
                "截取结束时间必须大于开始时间（或将结束时间设为 0 表示抽到视频结尾）。",
            )
            return

        naming_rule = self.naming_rule_edit.text().strip()
        try:
            compile_naming_rule(naming_rule=naming_rule)
        except ValueError as exc:
            QMessageBox.warning(self, "提示", str(exc))
            return

        config = ExtractionConfig(
            output_dir=Path(out_text),
            mode=self.mode_combo.currentData(),
            interval_seconds=self.interval_spin.value(),
            every_n_frames=self.frame_step_spin.value(),
            target_count=self.target_count_spin.value(),
            start_sec=start_sec,
            end_sec=end_sec if end_sec > 0 else None,
            dedup_threshold=self.dedup_spin.value(),
            max_dimension=self.max_dim_spin.value(),
            image_format=self.format_combo.currentText(),
            jpeg_quality=self.quality_spin.value(),
            naming_rule=naming_rule,
            create_video_subfolder=self.subfolder_chk.isChecked(),
        )

        # 按当前列表记录每个视频的帧数权重，用于计算多视频单调总进度
        self._batch_video_frames = []
        for vp in self._video_paths:
            info = self._video_info_map.get(vp.resolve())
            self._batch_video_frames.append(info.frame_count if info and info.frame_count > 0 else 0)

        self._last_progress_pct = 0
        self._extract_started_at = time.monotonic()
        self._showing_extract_summary = True
        self._set_send_next_highlighted(False)
        self._set_list_controls_enabled(False, allow_clear=False)
        self.stop_btn.setEnabled(True)
        self._set_progress_display(0, "总进度 0%")
        self.preview_caption_label.setText("正在初始化视频抽帧任务...")
        self._set_preview_badge("抽帧中", "busy")
        self._set_status_message("正在初始化抽帧...", level="busy")

        total_vids = len(self._video_paths)
        self._card_videos.set_content("视频进度", f"0 / {total_vids}", "准备开始...", color="#0969da")
        self._card_frames.set_content("已保存帧数", "0 张", "等待抽帧输出", color="#1f7a1f")
        self._card_dedup.set_content("去重跳过", "0 张", "跳过相似静止帧")
        self._card_output.set_content("任务耗时", "00:00", "正在运行中")

        self._elapsed_timer.start()
        self.worker = FrameExtractWorker(list(self._video_paths), config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_result.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_stop(self) -> None:
        """向正在运行的抽帧线程发送取消请求，由核心循环在安全检查点退出。"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._set_preview_badge("停止中", "warn")
            self._set_status_message("正在安全停止抽帧，请稍候...", level="warn")

    def _compute_overall_percent(self, prog: ExtractionProgress) -> int:
        """计算覆盖全部待处理视频的平滑单调总进度（0~100）。"""
        vid_idx = max(1, prog.video_index)
        if (
            len(self._batch_video_frames) >= prog.total_videos
            and prog.total_videos > 0
            and all(f > 0 for f in self._batch_video_frames[: prog.total_videos])
        ):
            weights = self._batch_video_frames[: prog.total_videos]
            total_all = max(1, sum(weights))
            completed = sum(weights[: vid_idx - 1])
            cur_cap = weights[vid_idx - 1]
            cur_frames = min(max(0, prog.current_frame), cur_cap)
            pct = int(min(100, ((completed + cur_frames) / total_all) * 100))
        else:
            cur_frac = min(1.0, max(0, prog.current_frame) / max(1, prog.total_frames))
            pct = int(min(100, ((vid_idx - 1 + cur_frac) / max(1, prog.total_videos)) * 100))

        self._last_progress_pct = max(self._last_progress_pct, pct)
        return self._last_progress_pct

    def _current_elapsed_seconds(self) -> float:
        """返回当前抽帧任务的已耗时（秒）；未记录有效起点时安全返回 `0.0`。"""
        if self._extract_started_at <= 0.0:
            return 0.0
        return max(0.0, time.monotonic() - self._extract_started_at)

    def _on_elapsed_tick(self) -> None:
        if not self._is_extracting():
            self._elapsed_timer.stop()
            return
        self._card_output.set_content(
            "任务耗时", format_duration_hms(self._current_elapsed_seconds()), "正在运行中"
        )

    def _on_progress(self, prog: ExtractionProgress) -> None:
        """在主线程刷新多视频总进度条、监控卡片、状态文字及最新抽取帧预览画面。"""
        pct = self._compute_overall_percent(prog)
        self._set_progress_display(pct, f"总进度 {pct}%  ({prog.video_index}/{prog.total_videos})")

        self._set_status_message(
            f"视频 [{prog.video_index}/{prog.total_videos}] {prog.video_name}  ·  "
            f"帧 {prog.current_frame:,}/{prog.total_frames:,}  ·  "
            f"已保存 {prog.saved_total:,} 张（去重跳过 {prog.skipped_similar:,} 张）",
            level="busy",
        )

        self._card_videos.set_content(
            "视频进度",
            f"{prog.video_index} / {prog.total_videos}",
            prog.video_name,
            color="#0969da",
        )
        self._card_frames.set_content(
            "已保存帧数",
            f"{prog.saved_total:,} 张",
            "实时写入磁盘",
            color="#1f7a1f",
        )
        examined = prog.saved_total + prog.skipped_similar
        skip_rate = int(round((prog.skipped_similar / examined) * 100)) if examined > 0 else 0
        self._card_dedup.set_content(
            "去重跳过",
            f"{prog.skipped_similar:,} 张",
            f"跳过率 {skip_rate}%" if examined > 0 else "暂无重复帧",
        )
        self._card_output.set_content(
            "任务耗时", format_duration_hms(self._current_elapsed_seconds()), "正在运行中"
        )

        if prog.last_saved_path is not None and prog.last_saved_path.is_file():
            pix = QPixmap(str(prog.last_saved_path))
            if not pix.isNull():
                self.preview_label.set_preview_pixmap(pix)
                self.preview_caption_label.setText(
                    f"最新抽帧: {prog.last_saved_path.name}  ({prog.video_name})"
                )
                self.preview_caption_label.setToolTip(str(prog.last_saved_path))

    def _on_finished(self, result: ExtractionResult) -> None:
        """恢复按钮状态、展示完成摘要，静默同步输出目录并按需进入标注工作台。"""
        self._elapsed_timer.stop()
        self._set_list_controls_enabled(True)
        self.stop_btn.setEnabled(False)
        self._showing_extract_summary = True

        elapsed = self._current_elapsed_seconds()
        if not result.cancelled:
            self._set_progress_display(100, "100%  已完成")
        else:
            cur_val = self.progress_bar.value()
            self._set_progress_display(cur_val, f"{cur_val}%  已停止")

        total_vids = len(self._video_paths)
        examined = result.frames_saved + result.frames_skipped_similar
        skip_rate = int(round((result.frames_skipped_similar / examined) * 100)) if examined > 0 else 0

        if result.cancelled:
            badge_text, badge_level, status_level = "已停止", "warn", "warn"
            state = "已手动停止"
        elif result.errors:
            badge_text, badge_level, status_level = "有警告", "warn", "warn"
            state = "抽帧完成（含警告）"
        else:
            badge_text, badge_level, status_level = "已完成", "ok", "ok"
            state = "抽帧完成"

        self._set_preview_badge(badge_text, badge_level)
        self._card_videos.set_content(
            "视频进度",
            f"{result.videos_processed} / {max(total_vids, result.videos_processed)}",
            state,
            color="#1f7a1f" if not result.cancelled and not result.errors else "#b25900",
        )
        self._card_frames.set_content(
            "已保存帧数",
            f"{result.frames_saved:,} 张",
            f"目录: {result.output_dir.name}",
            color="#1f7a1f" if result.frames_saved > 0 else "#b25900",
        )
        self._card_dedup.set_content(
            "去重跳过",
            f"{result.frames_skipped_similar:,} 张",
            f"跳过率 {skip_rate}%",
        )
        self._card_output.set_content(
            "任务耗时",
            format_duration_hms(elapsed),
            state,
        )

        msg = (
            f"{state}：共处理 {result.videos_processed} 个视频，"
            f"保存 {result.frames_saved:,} 张图片（去重跳过 {result.frames_skipped_similar:,} 张）。"
        )
        if result.frames_saved > 0 and not (
            self.auto_open_workbench_chk.isChecked() and not result.cancelled and not result.errors
        ):
            msg += " 可直接点击「发送到『2. 标注工作台』→」开始标注。"
        if result.errors:
            msg += "\n警告:\n" + "\n".join(result.errors[:5])
        self._set_status_message(msg, level=status_level)

        if result.frames_saved > 0:
            self.frames_dir_synced.emit(str(result.output_dir))
            self._set_send_next_highlighted(True)
            if self.auto_open_workbench_chk.isChecked() and not result.cancelled and not result.errors:
                self.send_frames_dir.emit(str(result.output_dir))

    def _on_failed(self, err: str) -> None:
        self._elapsed_timer.stop()
        self._set_list_controls_enabled(True)
        self.stop_btn.setEnabled(False)
        self._showing_extract_summary = True
        self._set_preview_badge("出错", "error")
        self._set_status_message(f"抽帧失败: {err}", level="error")
        elapsed = self._current_elapsed_seconds()
        self._card_output.set_content("任务耗时", format_duration_hms(elapsed), "任务异常终止", color="#cf222e")
        QMessageBox.critical(self, "抽帧错误", err)

    def _on_send_next_clicked(self) -> None:
        self._set_send_next_highlighted(False)
        self.send_frames_dir.emit(self.output_picker.text())
