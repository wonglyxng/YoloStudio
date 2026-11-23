"""原生 PyQt6 矩形目标检测标注画布组件。

职责：
    提供基于 `QGraphicsView` 的高性能图像显示与矩形框交互画布，支持无级缩放平移、
    全画布十字准星绘制、8 控制点精准调框、整框平移、重叠框小面积优先点选、
    AI 置信度与审核角标渲染以及屏幕自适应恒定线宽/手柄绘制。

运行环境与协作边界：
    运行于 PyQt6 主线程，依赖 `PyQt6` 与 `core.annotation_io`；
    仅维护当前图片的视觉显示与交互草稿状态，不直接读写磁盘 `.json` 文件，
    所有持久变更通过信号交由 `gui.tab_labeling.LabelingStudioTab` 统一入撤销栈并落盘。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal as Signal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFont,
    QFontMetricsF,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QTransform,
    QWheelEvent,
)
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QWidget

from yolo_studio.core.annotation_io import RectAnnotation, label_color_rgb

HANDLE_CURSORS: dict[str, Qt.CursorShape] = {
    "nw": Qt.CursorShape.SizeFDiagCursor,
    "se": Qt.CursorShape.SizeFDiagCursor,
    "ne": Qt.CursorShape.SizeBDiagCursor,
    "sw": Qt.CursorShape.SizeBDiagCursor,
    "n": Qt.CursorShape.SizeVerCursor,
    "s": Qt.CursorShape.SizeVerCursor,
    "e": Qt.CursorShape.SizeHorCursor,
    "w": Qt.CursorShape.SizeHorCursor,
}


def load_image_pixmap(image_path: Path) -> tuple[QPixmap, int, int]:
    """加载图片为 `QPixmap` 并返回 `(pixmap, width, height)`；支持中文路径并以 Pillow 作为解码兜底。"""
    pix = QPixmap(str(image_path))
    if not pix.isNull():
        return pix, pix.width(), pix.height()

    try:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            data = rgb.tobytes("raw", "RGB")
            qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)
            return pix, w, h
    except Exception:
        return QPixmap(), 0, 0


class AnnotationCanvasView(QGraphicsView):
    """支持矩形框绘制、8 手柄编辑、滚轮缩放与拖拽平移的原生标注画布。

    坐标系约定：
        场景坐标系与原图像素坐标系 1:1 对应，左上角为 `(0, 0)`，右下角为 `(image_width, image_height)`。
    """

    MODE_EDIT = "edit"
    MODE_DRAW = "draw"

    MIN_ZOOM = 0.05
    MAX_ZOOM = 25.0
    MIN_BOX_SIZE_PX = 2.0

    # 画布交互信号
    box_drawn = Signal(float, float, float, float)  # 新绘制矩形框 (x1, y1, x2, y2)
    box_geometry_committed = Signal(int, float, float, float, float)  # 拖拽/调大小提交 (index, x1, y1, x2, y2)
    selection_changed = Signal(int)  # 选中框索引变更，无选中为 -1
    mode_changed = Signal(str)  # 交互模式变更 ("edit" | "draw")
    zoom_changed = Signal(float)  # 缩放倍率变更 (1.0 = 100%)
    cursor_pos_changed = Signal(int, int)  # 鼠标图像像素坐标变更（移出图像时发 -1, -1）
    box_double_clicked = Signal(int)  # 双击已有框（用于快速改类别）

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(0.0)
        self._scene.addItem(self._pixmap_item)

        self._image_width: int = 0
        self._image_height: int = 0
        self._boxes: list[RectAnnotation] = []
        self._selected_idx: int = -1
        self._hover_idx: int = -1
        self._hover_handle: str | None = None
        self._mode: str = self.MODE_EDIT
        self._active_label: str = ""
        self._show_labels: bool = True

        # 绘制 / 拖拽 / 平移运行态
        self._pending_fit: bool = False
        self._space_pressed: bool = False
        self._panning: bool = False
        self._pan_last_view_pos: QPoint = QPoint()

        self._is_drawing_draft: bool = False
        self._draw_start: QPointF = QPointF()
        self._draw_current: QPointF = QPointF()

        self._dragging_box_idx: int = -1
        self._dragging_handle: str | None = None
        self._drag_origin_rect: tuple[float, float, float, float] | None = None
        self._drag_start_scene: QPointF = QPointF()
        self._drag_preview_rect: tuple[float, float, float, float] | None = None

        self._cursor_scene_pos: QPointF | None = None

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setBackgroundBrush(QBrush(QColor("#1e232a")))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

    # ------------------------------------------------------------------
    # 公共只读状态与配置方法
    # ------------------------------------------------------------------
    @property
    def image_width(self) -> int:
        return self._image_width

    @property
    def image_height(self) -> int:
        return self._image_height

    @property
    def mode(self) -> str:
        return self._mode

    def drawing(self) -> bool:
        """当前是否处于绘制矩形框模式。"""
        return self._mode == self.MODE_DRAW

    def editing(self) -> bool:
        """当前是否处于选择与编辑矩形框模式。"""
        return self._mode == self.MODE_EDIT

    def selected_index(self) -> int:
        return self._selected_idx

    def zoom_factor(self) -> float:
        return float(self.transform().m11())

    def set_active_label(self, label: str) -> None:
        """设置当前默认绘制类别名称（仅影响草稿框预览颜色与角标显示）。"""
        self._active_label = label.strip()
        if self._is_drawing_draft:
            self.viewport().update()

    def set_show_labels(self, visible: bool) -> None:
        """切换是否渲染矩形框左上角标签与置信度角标。"""
        self._show_labels = bool(visible)
        self.viewport().update()

    def set_mode(self, mode: str) -> None:
        """在编辑模式（`edit`）与绘制矩形框模式（`draw`）之间切换并重置未落地的草稿框。"""
        normalized = self.MODE_DRAW if mode == self.MODE_DRAW else self.MODE_EDIT
        changed = normalized != self._mode
        self._mode = normalized
        self._cancel_transient_interactions()
        self._update_cursor_for_hover()
        self.viewport().update()
        if changed:
            self.mode_changed.emit(self._mode)

    def set_image(self, pixmap: QPixmap, width: int = 0, height: int = 0) -> None:
        """加载新图像像素并重置视口自适应缩放。"""
        self._cancel_transient_interactions()
        self._boxes = []
        self._selected_idx = -1
        self._hover_idx = -1
        self._hover_handle = None

        if pixmap.isNull():
            self.clear_image()
            return

        self._pixmap_item.setPixmap(pixmap)
        self._image_width = int(width or pixmap.width())
        self._image_height = int(height or pixmap.height())
        # 在图像四周留出少量边距便于拖拉贴边控制手柄与画布平移
        margin = max(60.0, max(self._image_width, self._image_height) * 0.12)
        self._scene.setSceneRect(
            -margin,
            -margin,
            self._image_width + margin * 2.0,
            self._image_height + margin * 2.0,
        )
        self.fit_to_window()
        self._pending_fit = self.viewport().width() <= 20 or self.viewport().height() <= 20
        self.viewport().update()

    def clear_image(self) -> None:
        """清空当前画布图像与全部矩形框。"""
        self._cancel_transient_interactions()
        self._pixmap_item.setPixmap(QPixmap())
        self._image_width = 0
        self._image_height = 0
        self._boxes = []
        self._selected_idx = -1
        self._hover_idx = -1
        self._hover_handle = None
        self._scene.setSceneRect(0, 0, 1, 1)
        self.resetTransform()
        self.zoom_changed.emit(1.0)
        self.viewport().update()

    def set_boxes(self, boxes: list[RectAnnotation], selected_index: int = -1) -> None:
        """整体刷新画布显示的矩形框列表并同步选中索引（不触发 `selection_changed`）。"""
        self._cancel_transient_interactions()
        self._boxes = [b.clone() for b in boxes]
        if 0 <= selected_index < len(self._boxes):
            self._selected_idx = selected_index
        else:
            self._selected_idx = -1
        self._hover_idx = -1
        self._hover_handle = None
        self.viewport().update()

    def select_box(self, idx: int, *, emit_signal: bool = True) -> None:
        """选中指定索引的矩形框（越界或 `< 0` 则清除选中）。"""
        target = idx if 0 <= idx < len(self._boxes) else -1
        if target == self._selected_idx:
            return
        self._selected_idx = target
        self.viewport().update()
        if emit_signal:
            self.selection_changed.emit(self._selected_idx)

    def fit_to_window(self) -> None:
        """将当前图片按比例完整居中适配进可视视口。"""
        if self._image_width <= 0 or self._image_height <= 0:
            self.resetTransform()
            self.zoom_changed.emit(1.0)
            return
        vw = max(1, self.viewport().width() - 24)
        vh = max(1, self.viewport().height() - 24)
        scale = min(vw / float(self._image_width), vh / float(self._image_height))
        scale = max(self.MIN_ZOOM, min(self.MAX_ZOOM, scale))
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(self._image_width / 2.0, self._image_height / 2.0)
        self.zoom_changed.emit(scale)
        self.viewport().update()

    def set_zoom(self, scale: float) -> None:
        """设置绝对缩放倍率（自动裁剪至 `[MIN_ZOOM, MAX_ZOOM]`）。"""
        clamped = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(scale)))
        self.setTransform(QTransform.fromScale(clamped, clamped))
        self.zoom_changed.emit(clamped)
        self.viewport().update()

    def zoom_by(self, factor: float) -> None:
        """按相对倍率放大或缩小当前画布。"""
        cur = self.zoom_factor()
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, cur * factor))
        if abs(target - cur) < 1e-6:
            return
        actual_factor = target / cur
        self.scale(actual_factor, actual_factor)
        self.zoom_changed.emit(self.zoom_factor())
        self.viewport().update()

    # ------------------------------------------------------------------
    # 坐标与命中测试辅助
    # ------------------------------------------------------------------
    def _clamp_point(self, pt: QPointF) -> QPointF:
        if self._image_width <= 0 or self._image_height <= 0:
            return QPointF(max(0.0, pt.x()), max(0.0, pt.y()))
        return QPointF(
            max(0.0, min(float(self._image_width), pt.x())),
            max(0.0, min(float(self._image_height), pt.y())),
        )

    def _inv_scale(self) -> float:
        return 1.0 / max(self.zoom_factor(), 1e-4)

    @staticmethod
    def _box_handles(x1: float, y1: float, x2: float, y2: float) -> dict[str, QPointF]:
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        return {
            "nw": QPointF(x1, y1),
            "n": QPointF(mx, y1),
            "ne": QPointF(x2, y1),
            "e": QPointF(x2, my),
            "se": QPointF(x2, y2),
            "s": QPointF(mx, y2),
            "sw": QPointF(x1, y2),
            "w": QPointF(x1, my),
        }

    def _hit_test_handle(self, scene_pos: QPointF) -> str | None:
        if not (0 <= self._selected_idx < len(self._boxes)):
            return None
        b = self._boxes[self._selected_idx]
        tol = 8.5 * self._inv_scale()
        best_name: str | None = None
        best_dist_sq = tol * tol
        for name, hpt in self._box_handles(b.x1, b.y1, b.x2, b.y2).items():
            dx = scene_pos.x() - hpt.x()
            dy = scene_pos.y() - hpt.y()
            d2 = dx * dx + dy * dy
            if d2 <= best_dist_sq:
                best_dist_sq = d2
                best_name = name
        return best_name

    def _hit_test_box(self, scene_pos: QPointF) -> int:
        """命中检测：优先选中包含该点的面积最小框，避免大目标框遮死内部小目标框。"""
        pad = 4.0 * self._inv_scale()
        best_idx = -1
        best_area = float("inf")
        px, py = scene_pos.x(), scene_pos.y()
        for idx, b in enumerate(self._boxes):
            if (b.x1 - pad) <= px <= (b.x2 + pad) and (b.y1 - pad) <= py <= (b.y2 + pad):
                area = b.area
                if area < best_area:
                    best_area = area
                    best_idx = idx
        return best_idx

    def _cancel_transient_interactions(self) -> None:
        self._panning = False
        self._is_drawing_draft = False
        self._dragging_box_idx = -1
        self._dragging_handle = None
        self._drag_origin_rect = None
        self._drag_preview_rect = None

    def _update_cursor_for_hover(self) -> None:
        if self._panning:
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return
        if self._space_pressed:
            self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            return
        if self._mode == self.MODE_DRAW:
            self.viewport().setCursor(QCursor(Qt.CursorShape.CrossCursor))
            return
        if self._hover_handle and self._hover_handle in HANDLE_CURSORS:
            self.viewport().setCursor(QCursor(HANDLE_CURSORS[self._hover_handle]))
            return
        if self._hover_idx >= 0:
            self.viewport().setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
            return
        self.viewport().setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _compute_resized_rect(
        self,
        origin: tuple[float, float, float, float],
        handle: str,
        scene_pos: QPointF,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = origin
        pt = self._clamp_point(scene_pos)
        if "w" in handle:
            x1 = pt.x()
        if "e" in handle:
            x2 = pt.x()
        if "n" in handle:
            y1 = pt.y()
        if "s" in handle:
            y2 = pt.y()
        lx, rx = (x1, x2) if x1 <= x2 else (x2, x1)
        ty, by = (y1, y2) if y1 <= y2 else (y2, y1)
        return lx, ty, rx, by

    def _compute_moved_rect(
        self,
        origin: tuple[float, float, float, float],
        delta: QPointF,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = origin
        bw = x2 - x1
        bh = y2 - y1
        nx1 = x1 + delta.x()
        ny1 = y1 + delta.y()
        if self._image_width > 0:
            nx1 = max(0.0, min(float(self._image_width) - bw, nx1))
        if self._image_height > 0:
            ny1 = max(0.0, min(float(self._image_height) - bh, ny1))
        return nx1, ny1, nx1 + bw, ny1 + bh

    # ------------------------------------------------------------------
    # Qt 事件处理
    # ------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._pending_fit and self.viewport().width() > 20 and self.viewport().height() > 20:
            self._pending_fit = False
            self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._image_width <= 0:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.18 if delta > 0 else (1.0 / 1.18)
        self.zoom_by(factor)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self._update_cursor_for_hover()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self._is_drawing_draft or self._dragging_box_idx >= 0:
                self._cancel_transient_interactions()
                self.viewport().update()
                event.accept()
                return
            if self._mode == self.MODE_DRAW:
                self.set_mode(self.MODE_EDIT)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self._panning = False
            self._update_cursor_for_hover()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._image_width <= 0:
            super().mousePressEvent(event)
            return

        btn = event.button()
        # 右键取消正在画的草稿框；其他情况右键/中键/Space+左键启动平移
        if btn == Qt.MouseButton.RightButton and self._is_drawing_draft:
            self._is_drawing_draft = False
            self.viewport().update()
            event.accept()
            return
        if btn in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton) or (
            btn == Qt.MouseButton.LeftButton and self._space_pressed
        ):
            self._panning = True
            self._pan_last_view_pos = event.position().toPoint()
            self._update_cursor_for_hover()
            event.accept()
            return

        if btn != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        if self._mode == self.MODE_DRAW:
            clamped = self._clamp_point(scene_pos)
            self._is_drawing_draft = True
            self._draw_start = clamped
            self._draw_current = clamped
            self.viewport().update()
            event.accept()
            return

        # 编辑模式：先检测当前选中框的 8 手柄，再检测全部矩形框
        handle = self._hit_test_handle(scene_pos)
        if handle is not None and 0 <= self._selected_idx < len(self._boxes):
            b = self._boxes[self._selected_idx]
            self._dragging_box_idx = self._selected_idx
            self._dragging_handle = handle
            self._drag_origin_rect = (b.x1, b.y1, b.x2, b.y2)
            self._drag_preview_rect = self._drag_origin_rect
            self._drag_start_scene = scene_pos
            self.viewport().update()
            event.accept()
            return

        hit_idx = self._hit_test_box(scene_pos)
        if hit_idx >= 0:
            self.select_box(hit_idx, emit_signal=True)
            b = self._boxes[hit_idx]
            self._dragging_box_idx = hit_idx
            self._dragging_handle = None
            self._drag_origin_rect = (b.x1, b.y1, b.x2, b.y2)
            self._drag_preview_rect = self._drag_origin_rect
            self._drag_start_scene = scene_pos
            self.viewport().update()
            event.accept()
            return

        self.select_box(-1, emit_signal=True)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        view_pos = event.position().toPoint()
        scene_pos = self.mapToScene(view_pos)
        self._cursor_scene_pos = scene_pos

        if self._image_width > 0 and self._image_height > 0:
            ix, iy = int(scene_pos.x()), int(scene_pos.y())
            if 0 <= ix < self._image_width and 0 <= iy < self._image_height:
                self.cursor_pos_changed.emit(ix, iy)
            else:
                self.cursor_pos_changed.emit(-1, -1)

        if self._panning:
            delta = view_pos - self._pan_last_view_pos
            self._pan_last_view_pos = view_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if self._is_drawing_draft:
            self._draw_current = self._clamp_point(scene_pos)
            self.viewport().update()
            event.accept()
            return

        if self._dragging_box_idx >= 0 and self._drag_origin_rect is not None:
            if self._dragging_handle is not None:
                self._drag_preview_rect = self._compute_resized_rect(
                    self._drag_origin_rect, self._dragging_handle, scene_pos
                )
            else:
                delta_scene = scene_pos - self._drag_start_scene
                self._drag_preview_rect = self._compute_moved_rect(self._drag_origin_rect, delta_scene)
            self.viewport().update()
            event.accept()
            return

        if self._mode == self.MODE_DRAW:
            self._hover_idx = -1
            self._hover_handle = None
            self._update_cursor_for_hover()
            self.viewport().update()
            return

        # 悬停高亮与手柄光标反馈
        prev_hover = (self._hover_idx, self._hover_handle)
        self._hover_handle = self._hit_test_handle(scene_pos)
        self._hover_idx = self._selected_idx if self._hover_handle else self._hit_test_box(scene_pos)
        self._update_cursor_for_hover()
        if (self._hover_idx, self._hover_handle) != prev_hover:
            self.viewport().update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._panning:
            self._panning = False
            self._update_cursor_for_hover()
            event.accept()
            return

        if self._is_drawing_draft and event.button() == Qt.MouseButton.LeftButton:
            self._is_drawing_draft = False
            end_pt = self._clamp_point(self.mapToScene(event.position().toPoint()))
            x1, x2 = sorted((self._draw_start.x(), end_pt.x()))
            y1, y2 = sorted((self._draw_start.y(), end_pt.y()))
            self.viewport().update()
            if (x2 - x1) >= self.MIN_BOX_SIZE_PX and (y2 - y1) >= self.MIN_BOX_SIZE_PX:
                self.box_drawn.emit(x1, y1, x2, y2)
            event.accept()
            return

        if self._dragging_box_idx >= 0 and event.button() == Qt.MouseButton.LeftButton:
            idx = self._dragging_box_idx
            origin = self._drag_origin_rect
            preview = self._drag_preview_rect
            self._cancel_transient_interactions()
            self.viewport().update()
            if (
                origin is not None
                and preview is not None
                and (preview[2] - preview[0]) >= self.MIN_BOX_SIZE_PX
                and (preview[3] - preview[1]) >= self.MIN_BOX_SIZE_PX
                and any(abs(a - b) >= 0.5 for a, b in zip(origin, preview))
            ):
                self.box_geometry_committed.emit(idx, preview[0], preview[1], preview[2], preview[3])
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._image_width <= 0 or event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        if self._mode == self.MODE_EDIT:
            hit_idx = self._hit_test_box(scene_pos)
            if hit_idx >= 0:
                self.select_box(hit_idx, emit_signal=True)
                self.box_double_clicked.emit(hit_idx)
                event.accept()
                return
            self.fit_to_window()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._cursor_scene_pos = None
        self._hover_idx = -1
        self._hover_handle = None
        self.cursor_pos_changed.emit(-1, -1)
        self.viewport().update()
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # 高清晰度屏幕自适应前景绘制
    # ------------------------------------------------------------------
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        super().drawForeground(painter, rect)
        if self._image_width <= 0 or self._image_height <= 0:
            return

        inv = self._inv_scale()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 绘制图像浅灰细边轮廓（深色图片与深色背景之间更易识别边界）
        border_pen = QPen(QColor(255, 255, 255, 55), 1.0 * inv)
        painter.setPen(border_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(0.0, 0.0, float(self._image_width), float(self._image_height)))

        # 绘制全部已有矩形框（选中框置顶后绘，确保手柄与高亮不被覆盖）
        order = [i for i in range(len(self._boxes)) if i != self._selected_idx]
        if 0 <= self._selected_idx < len(self._boxes):
            order.append(self._selected_idx)

        for idx in order:
            box = self._boxes[idx]
            is_sel = idx == self._selected_idx
            is_hover = idx == self._hover_idx
            if is_sel and self._dragging_box_idx == idx and self._drag_preview_rect is not None:
                x1, y1, x2, y2 = self._drag_preview_rect
            else:
                x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            self._paint_box(painter, box, x1, y1, x2, y2, inv, is_sel=is_sel, is_hover=is_hover)

        # 绘制模式：绘制全画布十字辅助准星与实时草稿框
        if self._mode == self.MODE_DRAW:
            if self._cursor_scene_pos is not None:
                cx = max(0.0, min(float(self._image_width), self._cursor_scene_pos.x()))
                cy = max(0.0, min(float(self._image_height), self._cursor_scene_pos.y()))
                guide_pen = QPen(QColor(255, 255, 255, 170), 1.0 * inv, Qt.PenStyle.DashLine)
                painter.setPen(guide_pen)
                painter.drawLine(QPointF(cx, 0.0), QPointF(cx, float(self._image_height)))
                painter.drawLine(QPointF(0.0, cy), QPointF(float(self._image_width), cy))

            if self._is_drawing_draft:
                dx1, dx2 = sorted((self._draw_start.x(), self._draw_current.x()))
                dy1, dy2 = sorted((self._draw_start.y(), self._draw_current.y()))
                r, g, b = label_color_rgb(self._active_label or "draft")
                draft_rect = QRectF(dx1, dy1, dx2 - dx1, dy2 - dy1)
                painter.setBrush(QBrush(QColor(r, g, b, 45)))
                painter.setPen(QPen(QColor(r, g, b, 240), 2.0 * inv, Qt.PenStyle.DashLine))
                painter.drawRect(draft_rect)
                size_tip = f"{int(round(dx2 - dx1))}×{int(round(dy2 - dy1))}"
                if self._active_label:
                    size_tip = f"{self._active_label}  {size_tip}"
                self._paint_tag_pill(painter, dx1, dy1, size_tip, QColor(r, g, b, 230), inv)

        painter.restore()

    def _paint_box(
        self,
        painter: QPainter,
        box: RectAnnotation,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        inv: float,
        *,
        is_sel: bool,
        is_hover: bool,
    ) -> None:
        r, g, b = label_color_rgb(box.label)
        rect_f = QRectF(x1, y1, x2 - x1, y2 - y1)

        fill_alpha = 58 if is_sel else (42 if is_hover else 24)
        painter.setBrush(QBrush(QColor(r, g, b, fill_alpha)))

        if is_sel:
            # 双层描边（白底 + 彩色主线），保证在任意明暗背景上都清晰锐利
            painter.setPen(QPen(QColor(255, 255, 255, 225), 4.0 * inv))
            painter.drawRect(rect_f)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(r, g, b, 255), 2.2 * inv))
            painter.drawRect(rect_f)
        else:
            line_w = 2.4 * inv if is_hover else 1.8 * inv
            painter.setPen(QPen(QColor(r, g, b, 240), line_w))
            painter.drawRect(rect_f)

        if self._show_labels:
            tag = box.label
            if box.score is not None:
                tag += f" {box.score:.2f}"
            if isinstance(box.flags, dict) and box.flags.get("llm_added"):
                tag += " ★"
            elif isinstance(box.flags, dict) and box.flags.get("box_refined"):
                tag += " ⌖"
            self._paint_tag_pill(painter, x1, y1, tag, QColor(r, g, b, 232), inv)

        if is_sel:
            hs = 4.6 * inv
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(r, g, b), 1.6 * inv))
            for hpt in self._box_handles(x1, y1, x2, y2).values():
                painter.drawRect(QRectF(hpt.x() - hs, hpt.y() - hs, hs * 2.0, hs * 2.0))

    def _paint_tag_pill(
        self,
        painter: QPainter,
        x1: float,
        y1: float,
        text: str,
        bg_color: QColor,
        inv: float,
    ) -> None:
        """以恒定屏幕像素字号绘制标签角标（靠近图像顶部时自动内折至框内左上角）。"""
        painter.save()
        painter.translate(x1, y1)
        painter.scale(inv, inv)
        font = QFont("Microsoft YaHei UI", 9, QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = QFontMetricsF(font)
        pad_x = 5.0
        pad_y = 2.0
        tw = fm.horizontalAdvance(text) + pad_x * 2.0
        th = fm.height() + pad_y * 2.0
        # 若左上角距离图像顶边足够高则绘制在框外上方，否则内折进框内避免出界
        top_y = -th - 1.0 if (y1 / max(inv, 1e-6)) >= (th + 2.0) else 1.0
        pill = QRectF(0.0, top_y, tw, th)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(pill, 3.0, 3.0)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
        painter.restore()
