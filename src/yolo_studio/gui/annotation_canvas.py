"""原生 PyQt6 矩形目标检测标注画布组件。

职责：
    提供基于 `QGraphicsView` 的高性能图像显示与矩形框交互画布，支持鼠标锚点无级缩放、
    高倍像素锐化显示、多通道顺滑平移（抓手模式 / 拖空白 / 右键拖拽 / 中键 / 空格+左键）、
    拖拽+两点双模式画框、边缘自动滚屏、防误拖与自适应控制点、全画布双色高对比十字准星、
    重叠框小面积优先点选、AI 置信度与审核角标渲染以及右键上下文菜单定位。

运行环境与协作边界：
    运行于 PyQt6 主线程，依赖 `PyQt6` 与 `core.annotation_io`；
    仅维护当前图片的视觉显示与交互草稿状态，不直接读写磁盘 `.json` 文件，
    所有持久变更通过信号交由 `gui.tab_labeling.LabelingStudioTab` 统一入撤销栈并落盘。
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QFocusEvent,
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
    """支持精确锚点缩放、自由平移与双模式矩形标注的原生画布。

    坐标系约定：
        场景坐标系与原图像素坐标系 1:1 对应，左上角为 `(0, 0)`，右下角为 `(image_width, image_height)`。
    """

    MODE_EDIT = "edit"
    MODE_DRAW = "draw"
    MODE_PAN = "pan"

    MIN_ZOOM = 0.05
    MAX_ZOOM = 25.0
    MIN_BOX_SIZE_PX = 2.0
    DRAG_THRESHOLD_PX = 4.0
    HANDLE_DRAG_THRESHOLD_PX = 2.5
    PIXEL_SHARP_ZOOM = 2.0
    EDGE_SCROLL_MARGIN_PX = 28.0

    # 画布交互信号
    box_drawn = Signal(float, float, float, float)  # 新绘制矩形框 (x1, y1, x2, y2)
    box_geometry_committed = Signal(int, float, float, float, float)  # 拖拽/调大小提交 (index, x1, y1, x2, y2)
    selection_changed = Signal(int)  # 选中框索引变更，无选中为 -1
    mode_changed = Signal(str)  # 交互模式变更 ("edit" | "draw" | "pan")
    zoom_changed = Signal(float)  # 缩放倍率变更 (1.0 = 100%)
    cursor_pos_changed = Signal(int, int)  # 鼠标图像像素坐标变更（移出图像时发 -1, -1）
    box_double_clicked = Signal(int)  # 双击已有框（用于快速改类别）
    context_menu_requested = Signal(QPoint, int)  # 右键上下文菜单请求 (全局坐标, 命中框索引/-1)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(0.0)
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
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

        # 视口缩放与自适应状态
        self._fit_mode: bool = True
        self._pending_fit: bool = False
        self._zoom_anchor_cache: tuple[QPointF, QPointF] | None = None
        self._cached_center_scene: QPointF | None = None

        # 平移状态（统一基于起点差值计算，避免累积漂移）
        self._space_pressed: bool = False
        self._panning: bool = False
        self._pan_trigger: str = ""
        self._pan_start_view_pos: QPointF = QPointF()
        self._pan_start_center_scene: QPointF = QPointF()
        self._right_press_view_pos: QPointF | None = None
        self._right_press_global_pos: QPoint = QPoint()
        self._right_dragged: bool = False

        # 绘制状态（支持拖拽与两点点击双模式）
        self._draw_state: str = "idle"  # "idle" | "pressing" | "waiting_second"
        self._draw_press_view_pos: QPointF = QPointF()
        self._draw_dragged: bool = False
        self._draw_start: QPointF = QPointF()
        self._draw_current: QPointF = QPointF()

        # 编辑拖拽状态（带最小移动阈值防手抖误改）
        self._edit_action: str = "none"  # "none" | "pending_handle" | "dragging_handle" | "pending_move" | "moving_box" | "pending_blank_pan" | "blank_panning"
        self._edit_press_view_pos: QPointF = QPointF()
        self._dragging_box_idx: int = -1
        self._dragging_handle: str | None = None
        self._drag_origin_rect: tuple[float, float, float, float] | None = None
        self._drag_start_scene: QPointF = QPointF()
        self._drag_preview_rect: tuple[float, float, float, float] | None = None

        self._cursor_scene_pos: QPointF | None = None
        self._last_cursor_view_pos: QPointF | None = None

        self._edge_scroll_timer = QTimer(self)
        self._edge_scroll_timer.setInterval(25)
        self._edge_scroll_timer.timeout.connect(self._on_edge_scroll_tick)

        # 轻量顶部悬浮胶囊提示状态
        self._hud_message: str = ""
        self._hud_accent: QColor = QColor("#0969da")
        self._hud_timer = QTimer(self)
        self._hud_timer.setSingleShot(True)
        self._hud_timer.timeout.connect(self._clear_hud_message)

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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

    def panning_mode(self) -> bool:
        """当前是否处于抓手平移模式。"""
        return self._mode == self.MODE_PAN

    def is_fit_mode(self) -> bool:
        """当前视图是否处于自适应窗口状态。"""
        return self._fit_mode

    def selected_index(self) -> int:
        return self._selected_idx

    def zoom_factor(self) -> float:
        return float(self.transform().m11())

    def set_active_label(self, label: str) -> None:
        """设置当前默认绘制类别名称（仅影响草稿框预览颜色与角标显示）。"""
        self._active_label = label.strip()
        if self._draw_state != "idle":
            self.viewport().update()

    def set_show_labels(self, visible: bool) -> None:
        """切换是否渲染矩形框左上角标签与置信度角标。"""
        self._show_labels = bool(visible)
        self.viewport().update()

    def show_hud_message(
        self,
        text: str,
        *,
        color: str = "#0969da",
        duration_ms: int = 1600,
    ) -> None:
        """在画布顶部中央展示轻量悬浮胶囊提示；传入空文本可立即关闭。"""
        clean = (text or "").strip()
        self._hud_message = clean
        self._hud_accent = QColor(color)
        if clean:
            self._hud_timer.start(max(300, int(duration_ms)))
        else:
            self._hud_timer.stop()
        self.viewport().update()

    def _clear_hud_message(self) -> None:
        if self._hud_message:
            self._hud_message = ""
            self.viewport().update()

    def set_space_pressed(self, pressed: bool) -> None:
        """由工作台或画布同步空格键按住状态，支持任意焦点位置按住空格拖动画布。"""
        flag = bool(pressed)
        if flag == self._space_pressed:
            return
        self._space_pressed = flag
        if not flag and self._panning and self._pan_trigger == "space":
            self._stop_pan()
        self._update_cursor_for_hover()

    def set_mode(self, mode: str) -> None:
        """在编辑模式（`edit`）、画框模式（`draw`）与抓手模式（`pan`）之间切换。"""
        if mode == self.MODE_DRAW:
            normalized = self.MODE_DRAW
        elif mode == self.MODE_PAN:
            normalized = self.MODE_PAN
        else:
            normalized = self.MODE_EDIT
        changed = normalized != self._mode
        self._mode = normalized
        self._cancel_transient_interactions()
        self._update_cursor_for_hover()
        self.viewport().update()
        if changed:
            self.mode_changed.emit(self._mode)

    def cancel_active_draft(self) -> bool:
        """取消当前未落地的画框草稿或框编辑拖拽；存在活跃草稿时返回 `True`。"""
        had_active = (
            self._draw_state != "idle"
            or self._edit_action != "none"
            or self._panning
        )
        if not had_active:
            return False
        self._cancel_transient_interactions()
        self._update_cursor_for_hover()
        self.viewport().update()
        return True

    def set_image(
        self,
        pixmap: QPixmap,
        width: int = 0,
        height: int = 0,
        *,
        auto_fit: bool = True,
    ) -> None:
        """加载新图像像素并重置瞬态交互；`auto_fit=True` 时自适应居中显示。"""
        self._cancel_transient_interactions()
        self._zoom_anchor_cache = None
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
        self._update_scene_bounds(self.zoom_factor())
        if auto_fit:
            self.fit_to_window()
            self._pending_fit = self.viewport().width() <= 20 or self.viewport().height() <= 20
        else:
            self._pending_fit = False
        self._update_cursor_for_hover()
        self.viewport().update()

    def clear_image(self) -> None:
        """清空当前画布图像与全部矩形框。"""
        self._cancel_transient_interactions()
        self._zoom_anchor_cache = None
        self._cached_center_scene = None
        self._hud_timer.stop()
        self._hud_message = ""
        self._pixmap_item.setPixmap(QPixmap())
        self._image_width = 0
        self._image_height = 0
        self._boxes = []
        self._selected_idx = -1
        self._hover_idx = -1
        self._hover_handle = None
        self._fit_mode = True
        self._pending_fit = False
        self._scene.setSceneRect(0, 0, 1, 1)
        self.resetTransform()
        self._update_pixmap_transform_mode(1.0)
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

    # ------------------------------------------------------------------
    # 精准锚点缩放、动态边界与视口状态管理
    # ------------------------------------------------------------------
    def _map_view_to_scene_f(self, view_pos: QPointF) -> QPointF:
        inv, ok = self.viewportTransform().inverted()
        if ok:
            return inv.map(QPointF(view_pos))
        return self.mapToScene(QPoint(round(view_pos.x()), round(view_pos.y())))

    def _get_view_center_scene(self) -> QPointF:
        return self._map_view_to_scene_f(
            QPointF(self.viewport().width() / 2.0, self.viewport().height() / 2.0)
        )

    def _update_scene_bounds(self, scale: float | None = None) -> None:
        """按当前缩放与视口尺寸动态计算充裕的场景留白，保证图像四角皆可居中查看。"""
        if self._image_width <= 0 or self._image_height <= 0:
            self._scene.setSceneRect(0, 0, 1, 1)
            return
        s = max(scale if scale is not None else self.zoom_factor(), self.MIN_ZOOM)
        vw_scene = max(1.0, float(self.viewport().width())) / s
        vh_scene = max(1.0, float(self.viewport().height())) / s
        pad_x = max(vw_scene, float(self._image_width) * 0.6, 200.0)
        pad_y = max(vh_scene, float(self._image_height) * 0.6, 200.0)
        self._scene.setSceneRect(
            -pad_x,
            -pad_y,
            float(self._image_width) + pad_x * 2.0,
            float(self._image_height) + pad_y * 2.0,
        )

    def _set_view_center(self, scene_pt: QPointF, *, clamp: bool = True) -> QPointF:
        """将视口中心移动至指定场景坐标；开启软边界时限制在图像范围内防止拖丢，返回实际中心。"""
        if clamp and self._image_width > 0 and self._image_height > 0:
            target = QPointF(
                max(0.0, min(float(self._image_width), scene_pt.x())),
                max(0.0, min(float(self._image_height), scene_pt.y())),
            )
        else:
            target = QPointF(scene_pt)
        self._cached_center_scene = QPointF(target)
        self.centerOn(target)
        return target

    def _update_pixmap_transform_mode(self, scale: float) -> None:
        """低倍率平滑抗锯齿，高倍率（≥ 200%）自动切换最近邻锐利像素边界。"""
        use_smooth = scale < self.PIXEL_SHARP_ZOOM
        mode = (
            Qt.TransformationMode.SmoothTransformation
            if use_smooth
            else Qt.TransformationMode.FastTransformation
        )
        if self._pixmap_item.transformationMode() != mode:
            self._pixmap_item.setTransformationMode(mode)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, use_smooth)

    def _calc_fit_scale(self) -> float:
        if self._image_width <= 0 or self._image_height <= 0:
            return 1.0
        vw = max(1, self.viewport().width() - 24)
        vh = max(1, self.viewport().height() - 24)
        scale = min(vw / float(self._image_width), vh / float(self._image_height))
        return max(self.MIN_ZOOM, min(self.MAX_ZOOM, scale))

    def _default_zoom_anchor_view(self) -> QPointF:
        """按钮/快捷键缩放的默认锚点：有可见选中框时优先以该框为中心，否则以当前视口中心为准。"""
        vp_center = QPointF(self.viewport().width() / 2.0, self.viewport().height() / 2.0)
        if not (0 <= self._selected_idx < len(self._boxes)):
            return vp_center
        b = self._boxes[self._selected_idx]
        tl = self._map_view_to_scene_f(QPointF(0.0, 0.0))
        br = self._map_view_to_scene_f(
            QPointF(float(self.viewport().width()), float(self.viewport().height()))
        )
        vis_rect = QRectF(tl, br).normalized()
        box_rect = QRectF(b.x1, b.y1, max(1.0, b.width), max(1.0, b.height))
        if vis_rect.intersects(box_rect):
            box_center = QPointF((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)
            mapped = self.viewportTransform().map(box_center)
            return QPointF(
                max(0.0, min(float(self.viewport().width()), mapped.x())),
                max(0.0, min(float(self.viewport().height()), mapped.y())),
            )
        return vp_center

    def fit_to_window(self) -> None:
        """将当前图片按比例完整居中适配进可视视口。"""
        self._zoom_anchor_cache = None
        if self._image_width <= 0 or self._image_height <= 0:
            self._fit_mode = True
            self.resetTransform()
            self._update_pixmap_transform_mode(1.0)
            self.zoom_changed.emit(1.0)
            return
        scale = self._calc_fit_scale()
        self._fit_mode = True
        self.setTransform(QTransform.fromScale(scale, scale))
        self._update_pixmap_transform_mode(scale)
        self._update_scene_bounds(scale)
        self._set_view_center(QPointF(self._image_width / 2.0, self._image_height / 2.0), clamp=True)
        self.zoom_changed.emit(self.zoom_factor())
        self.viewport().update()

    def zoom_to(self, scale: float, anchor_view_pos: QPointF | None = None) -> None:
        """缩放到指定倍率并精确保持 `anchor_view_pos` 下的场景点位置不变。"""
        clamped = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(scale)))
        if self._image_width <= 0 or self._image_height <= 0:
            self._zoom_anchor_cache = None
            self._fit_mode = False
            self.setTransform(QTransform.fromScale(clamped, clamped))
            self._update_pixmap_transform_mode(clamped)
            self.zoom_changed.emit(self.zoom_factor())
            self.viewport().update()
            return

        anchor_vp = (
            QPointF(anchor_view_pos)
            if anchor_view_pos is not None
            else self._default_zoom_anchor_view()
        )
        # 连续在同一点缩放时复用初始浮点场景锚点，防止 QGraphicsView 整数滚动条舍入累积漂移
        if (
            self._zoom_anchor_cache is not None
            and (anchor_vp - self._zoom_anchor_cache[0]).manhattanLength() < 0.75
        ):
            anchor_scene = QPointF(self._zoom_anchor_cache[1])
        else:
            anchor_scene = self._map_view_to_scene_f(anchor_vp)

        self._fit_mode = False
        self.setTransform(QTransform.fromScale(clamped, clamped))
        self._update_pixmap_transform_mode(clamped)
        self._update_scene_bounds(clamped)

        vp_center = QPointF(self.viewport().width() / 2.0, self.viewport().height() / 2.0)
        offset_vp = anchor_vp - vp_center
        desired_center = QPointF(
            anchor_scene.x() - offset_vp.x() / clamped,
            anchor_scene.y() - offset_vp.y() / clamped,
        )
        actual_center = self._set_view_center(desired_center, clamp=True)
        if (actual_center - desired_center).manhattanLength() < 1e-3:
            self._zoom_anchor_cache = (QPointF(anchor_vp), QPointF(anchor_scene))
        else:
            self._zoom_anchor_cache = None
        self.zoom_changed.emit(self.zoom_factor())
        self.viewport().update()

    def set_zoom(self, scale: float, anchor_view_pos: QPointF | None = None) -> None:
        """设置绝对缩放倍率（自动裁剪至 `[MIN_ZOOM, MAX_ZOOM]` 并锚定焦点）。"""
        self.zoom_to(scale, anchor_view_pos=anchor_view_pos)

    def zoom_by(self, factor: float, anchor_view_pos: QPointF | None = None) -> None:
        """按相对倍率放大或缩小当前画布（默认锚定可见选中框或视口中心）。"""
        cur = self.zoom_factor()
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, cur * float(factor)))
        if abs(target - cur) < 1e-6:
            return
        self.zoom_to(target, anchor_view_pos=anchor_view_pos)

    def get_view_state(self) -> tuple[float, QPointF, bool]:
        """导出当前缩放倍率、视口中心场景坐标及自适应标记，供切图保持视野使用。"""
        return self.zoom_factor(), self._get_view_center_scene(), self._fit_mode

    def restore_view_state(
        self,
        scale: float,
        center_scene: QPointF,
        fit_mode: bool = False,
    ) -> None:
        """恢复之前保存的缩放与中心位置；若原处于自适应状态则按新图尺寸重新适配。"""
        self._zoom_anchor_cache = None
        if self._image_width <= 0 or self._image_height <= 0:
            return
        if fit_mode:
            self.fit_to_window()
            return
        clamped = max(self.MIN_ZOOM, min(self.MAX_ZOOM, float(scale)))
        self._fit_mode = False
        self.setTransform(QTransform.fromScale(clamped, clamped))
        self._update_pixmap_transform_mode(clamped)
        self._update_scene_bounds(clamped)
        self._set_view_center(center_scene, clamp=True)
        self.zoom_changed.emit(self.zoom_factor())
        self.viewport().update()

    def ensure_box_visible(self, idx: int) -> None:
        """若目标框中心落于当前视口安全区外，则平滑移至视口中央。"""
        if self._image_width <= 0 or not (0 <= idx < len(self._boxes)):
            return
        b = self._boxes[idx]
        bcx = (b.x1 + b.x2) / 2.0
        bcy = (b.y1 + b.y2) / 2.0
        tl = self._map_view_to_scene_f(QPointF(0.0, 0.0))
        br = self._map_view_to_scene_f(
            QPointF(float(self.viewport().width()), float(self.viewport().height()))
        )
        vis_w = max(1.0, br.x() - tl.x())
        vis_h = max(1.0, br.y() - tl.y())
        pad_x = vis_w * 0.15
        pad_y = vis_h * 0.15
        if (
            bcx < tl.x() + pad_x
            or bcx > br.x() - pad_x
            or bcy < tl.y() + pad_y
            or bcy > br.y() - pad_y
        ):
            self._zoom_anchor_cache = None
            self._fit_mode = False
            self._set_view_center(QPointF(bcx, bcy), clamp=True)
            self.viewport().update()

    def focus_box(self, idx: int) -> None:
        """将指定目标框放大并居中至当前视口（适合列表跳转或右键精准精修）。"""
        if self._image_width <= 0 or not (0 <= idx < len(self._boxes)):
            return
        self._zoom_anchor_cache = None
        b = self._boxes[idx]
        bw = max(8.0, b.width)
        bh = max(8.0, b.height)
        vw = max(1.0, float(self.viewport().width()))
        vh = max(1.0, float(self.viewport().height()))
        box_scale = min((vw * 0.55) / bw, (vh * 0.55) / bh)
        target_s = max(self._calc_fit_scale(), min(8.0, self.MAX_ZOOM, box_scale))
        self._fit_mode = False
        self.setTransform(QTransform.fromScale(target_s, target_s))
        self._update_pixmap_transform_mode(target_s)
        self._update_scene_bounds(target_s)
        self._set_view_center(QPointF((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0), clamp=True)
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

    def _visible_handles_for_box(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> dict[str, QPointF]:
        """根据当前屏幕渲染尺寸动态返回可用手柄，避免极小框的边中点挤占内部拖动热区。"""
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        scale = max(self.zoom_factor(), 1e-4)
        sw = (x2 - x1) * scale
        sh = (y2 - y1) * scale
        handles: dict[str, QPointF] = {
            "nw": QPointF(x1, y1),
            "ne": QPointF(x2, y1),
            "se": QPointF(x2, y2),
            "sw": QPointF(x1, y2),
        }
        if sw >= 32.0:
            handles["n"] = QPointF(mx, y1)
            handles["s"] = QPointF(mx, y2)
        if sh >= 32.0:
            handles["e"] = QPointF(x2, my)
            handles["w"] = QPointF(x1, my)
        return handles

    def _hit_test_handle(self, scene_pos: QPointF) -> str | None:
        if self._mode != self.MODE_EDIT or not (0 <= self._selected_idx < len(self._boxes)):
            return None
        b = self._boxes[self._selected_idx]
        scale = max(self.zoom_factor(), 1e-4)
        min_dim_px = min(b.width * scale, b.height * scale)
        tol = (6.0 if min_dim_px < 22.0 else 8.5) * self._inv_scale()
        best_name: str | None = None
        best_dist_sq = tol * tol
        for name, hpt in self._visible_handles_for_box(b.x1, b.y1, b.x2, b.y2).items():
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

    def _reset_edit_drag_state(self) -> None:
        self._edit_action = "none"
        self._dragging_box_idx = -1
        self._dragging_handle = None
        self._drag_origin_rect = None
        self._drag_preview_rect = None

    def _start_pan(self, view_pos: QPointF, trigger: str) -> None:
        self._zoom_anchor_cache = None
        self._panning = True
        self._pan_trigger = trigger
        self._pan_start_view_pos = QPointF(view_pos)
        self._pan_start_center_scene = self._get_view_center_scene()
        self._fit_mode = False
        self._update_cursor_for_hover()

    def _update_pan(self, view_pos: QPointF) -> None:
        scale = max(self.zoom_factor(), 1e-4)
        delta_vp = QPointF(view_pos) - self._pan_start_view_pos
        target_center = QPointF(
            self._pan_start_center_scene.x() - delta_vp.x() / scale,
            self._pan_start_center_scene.y() - delta_vp.y() / scale,
        )
        self._set_view_center(target_center, clamp=True)

    def _stop_pan(self) -> None:
        self._panning = False
        self._pan_trigger = ""

    def _stop_edge_scroll(self) -> None:
        if self._edge_scroll_timer.isActive():
            self._edge_scroll_timer.stop()

    def _update_edge_scroll_state(self, view_pos: QPointF) -> None:
        self._last_cursor_view_pos = QPointF(view_pos)
        should_run = (
            (self._mode == self.MODE_DRAW and self._draw_state == "pressing" and self._draw_dragged)
            or (
                self._mode == self.MODE_EDIT
                and self._edit_action in ("dragging_handle", "moving_box")
            )
        )
        if not should_run:
            self._stop_edge_scroll()
            return
        vw = float(self.viewport().width())
        vh = float(self.viewport().height())
        m = self.EDGE_SCROLL_MARGIN_PX
        near_edge = (
            view_pos.x() < m
            or view_pos.x() > (vw - m)
            or view_pos.y() < m
            or view_pos.y() > (vh - m)
        )
        if near_edge and not self._edge_scroll_timer.isActive():
            self._edge_scroll_timer.start()
        elif not near_edge and self._edge_scroll_timer.isActive():
            self._edge_scroll_timer.stop()

    def _on_edge_scroll_tick(self) -> None:
        """拖框或调框接近视口边缘时平滑滚屏，支持高倍放大下顺畅跨屏精修。"""
        if self._last_cursor_view_pos is None or self._image_width <= 0:
            self._stop_edge_scroll()
            return
        is_draw_drag = (
            self._mode == self.MODE_DRAW and self._draw_state == "pressing" and self._draw_dragged
        )
        is_edit_drag = (
            self._mode == self.MODE_EDIT and self._edit_action in ("dragging_handle", "moving_box")
        )
        if not (is_draw_drag or is_edit_drag):
            self._stop_edge_scroll()
            return

        vp = self._last_cursor_view_pos
        vw = float(self.viewport().width())
        vh = float(self.viewport().height())
        m = self.EDGE_SCROLL_MARGIN_PX
        vx = 0.0
        vy = 0.0
        if vp.x() < m:
            vx = -max(3.0, min(16.0, (m - vp.x()) * 0.5))
        elif vp.x() > vw - m:
            vx = max(3.0, min(16.0, (vp.x() - (vw - m)) * 0.5))
        if vp.y() < m:
            vy = -max(3.0, min(16.0, (m - vp.y()) * 0.5))
        elif vp.y() > vh - m:
            vy = max(3.0, min(16.0, (vp.y() - (vh - m)) * 0.5))
        if vx == 0.0 and vy == 0.0:
            self._stop_edge_scroll()
            return

        scale = max(self.zoom_factor(), 1e-4)
        cur_center = self._get_view_center_scene()
        self._zoom_anchor_cache = None
        self._fit_mode = False
        self._set_view_center(
            QPointF(cur_center.x() + vx / scale, cur_center.y() + vy / scale),
            clamp=True,
        )
        scene_pos = self._map_view_to_scene_f(vp)
        self._cursor_scene_pos = scene_pos
        self._emit_cursor_coords(scene_pos)
        if is_draw_drag:
            self._draw_current = self._clamp_point(scene_pos)
        elif self._drag_origin_rect is not None:
            if self._edit_action == "dragging_handle" and self._dragging_handle is not None:
                self._drag_preview_rect = self._compute_resized_rect(
                    self._drag_origin_rect, self._dragging_handle, scene_pos
                )
            else:
                self._drag_preview_rect = self._compute_moved_rect(
                    self._drag_origin_rect, scene_pos - self._drag_start_scene
                )
        self.viewport().update()

    def _cancel_transient_interactions(self) -> None:
        self._stop_edge_scroll()
        self._stop_pan()
        self._right_press_view_pos = None
        self._right_dragged = False
        self._draw_state = "idle"
        self._draw_dragged = False
        self._reset_edit_drag_state()

    def _update_cursor_for_hover(self) -> None:
        if self._panning:
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return
        if self._space_pressed or self._mode == self.MODE_PAN:
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

    def _emit_cursor_coords(self, scene_pos: QPointF) -> None:
        if self._image_width > 0 and self._image_height > 0:
            ix, iy = int(scene_pos.x()), int(scene_pos.y())
            if 0 <= ix < self._image_width and 0 <= iy < self._image_height:
                self.cursor_pos_changed.emit(ix, iy)
            else:
                self.cursor_pos_changed.emit(-1, -1)

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

    def _emit_drawn_box_if_valid(self, p1: QPointF, p2: QPointF) -> None:
        x1, x2 = sorted((p1.x(), p2.x()))
        y1, y2 = sorted((p1.y(), p2.y()))
        if (x2 - x1) >= self.MIN_BOX_SIZE_PX and (y2 - y1) >= self.MIN_BOX_SIZE_PX:
            self.box_drawn.emit(x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Qt 事件处理
    # ------------------------------------------------------------------
    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        self._zoom_anchor_cache = None
        old_center = (
            QPointF(self._cached_center_scene)
            if self._cached_center_scene is not None
            else self._get_view_center_scene()
        )
        super().resizeEvent(event)
        if self._image_width <= 0 or self._image_height <= 0:
            return
        if (self._pending_fit or self._fit_mode) and (
            self.viewport().width() > 20 and self.viewport().height() > 20
        ):
            self._pending_fit = False
            self.fit_to_window()
        else:
            self._update_scene_bounds(self.zoom_factor())
            self._set_view_center(old_center, clamp=True)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # type: ignore[override]
        # 失焦时自动复位空格按住态并停止拖拽按压，防止切窗后粘滞
        self.set_space_pressed(False)
        self._stop_edge_scroll()
        if self._panning:
            self._stop_pan()
            self._update_cursor_for_hover()
        super().focusOutEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._image_width <= 0:
            super().wheelEvent(event)
            return
        angle_y = event.angleDelta().y()
        if angle_y == 0:
            angle_y = event.angleDelta().x()
        if angle_y != 0:
            steps = float(angle_y) / 120.0
        else:
            pixel_y = event.pixelDelta().y() or event.pixelDelta().x()
            if pixel_y == 0:
                return
            steps = float(pixel_y) / 60.0
        factor = max(0.5, min(2.0, 1.18**steps))
        anchor_vp = QPointF(event.position())
        self.zoom_by(factor, anchor_view_pos=anchor_vp)
        scene_pos = self._map_view_to_scene_f(anchor_vp)
        self._cursor_scene_pos = scene_pos
        self._emit_cursor_coords(scene_pos)
        if self._draw_state in ("pressing", "waiting_second"):
            self._draw_current = self._clamp_point(scene_pos)
            self.viewport().update()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space_pressed(True)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self._draw_state != "idle" or self._edit_action != "none":
                self.cancel_active_draft()
                event.accept()
                return
            if self._mode != self.MODE_EDIT:
                self.set_mode(self.MODE_EDIT)
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.set_space_pressed(False)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if self._image_width <= 0:
            super().mousePressEvent(event)
            return

        btn = event.button()
        view_pos = QPointF(event.position())
        self._last_cursor_view_pos = view_pos

        # 1) 中键或 Space+左键：全模式立即启动抓手平移（保留两点画框已定第一点）
        if btn == Qt.MouseButton.MiddleButton or (
            btn == Qt.MouseButton.LeftButton and self._space_pressed
        ):
            trigger = "middle" if btn == Qt.MouseButton.MiddleButton else "space"
            self._start_pan(view_pos, trigger)
            event.accept()
            return

        # 2) 右键：按下先记录起点，拖动 ≥4px 平移画布；轻点松手取消草稿或弹菜单
        if btn == Qt.MouseButton.RightButton:
            self._right_press_view_pos = view_pos
            self._right_press_global_pos = event.globalPosition().toPoint()
            self._right_dragged = False
            event.accept()
            return

        if btn != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        # 3) 抓手模式：左键直接拖动画布
        if self._mode == self.MODE_PAN:
            self._start_pan(view_pos, "pan_mode")
            event.accept()
            return

        scene_pos = self._map_view_to_scene_f(view_pos)

        # 4) 画框模式：支持按住拖框 + 单击两点画框
        if self._mode == self.MODE_DRAW:
            clamped = self._clamp_point(scene_pos)
            if self._draw_state == "waiting_second":
                start_pt = self._draw_start
                self._draw_state = "idle"
                self._draw_dragged = False
                self._stop_edge_scroll()
                self.viewport().update()
                self._emit_drawn_box_if_valid(start_pt, clamped)
                event.accept()
                return
            self._draw_state = "pressing"
            self._draw_press_view_pos = view_pos
            self._draw_dragged = False
            self._draw_start = clamped
            self._draw_current = clamped
            self.viewport().update()
            event.accept()
            return

        # 5) 编辑模式：先检手柄，再检框，最后落入空白平移待判定
        self._edit_press_view_pos = view_pos
        handle = self._hit_test_handle(scene_pos)
        if handle is not None and 0 <= self._selected_idx < len(self._boxes):
            b = self._boxes[self._selected_idx]
            self._edit_action = "pending_handle"
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
            self._edit_action = "pending_move"
            self._dragging_box_idx = hit_idx
            self._dragging_handle = None
            self._drag_origin_rect = (b.x1, b.y1, b.x2, b.y2)
            self._drag_preview_rect = self._drag_origin_rect
            self._drag_start_scene = scene_pos
            self.viewport().update()
            event.accept()
            return

        self._edit_action = "pending_blank_pan"
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        view_pos = QPointF(event.position())
        self._last_cursor_view_pos = view_pos
        scene_pos = self._map_view_to_scene_f(view_pos)
        self._cursor_scene_pos = scene_pos
        self._emit_cursor_coords(scene_pos)

        # 右键按住移动超过阈值时激活平滑平移
        if (
            self._right_press_view_pos is not None
            and (event.buttons() & Qt.MouseButton.RightButton)
            and not self._panning
        ):
            dist = math.hypot(
                view_pos.x() - self._right_press_view_pos.x(),
                view_pos.y() - self._right_press_view_pos.y(),
            )
            if dist >= self.DRAG_THRESHOLD_PX:
                self._right_dragged = True
                self._start_pan(self._right_press_view_pos, "right")

        if self._panning:
            self._update_pan(view_pos)
            event.accept()
            return

        if self._mode == self.MODE_DRAW:
            if self._draw_state == "pressing":
                if not self._draw_dragged:
                    dist = math.hypot(
                        view_pos.x() - self._draw_press_view_pos.x(),
                        view_pos.y() - self._draw_press_view_pos.y(),
                    )
                    if dist >= self.DRAG_THRESHOLD_PX:
                        self._draw_dragged = True
                self._draw_current = self._clamp_point(scene_pos)
                if self._draw_dragged:
                    self._update_edge_scroll_state(view_pos)
                self.viewport().update()
                event.accept()
                return
            if self._draw_state == "waiting_second":
                self._draw_current = self._clamp_point(scene_pos)
                self.viewport().update()
                event.accept()
                return
            self._hover_idx = -1
            self._hover_handle = None
            self._update_cursor_for_hover()
            self.viewport().update()
            return

        if self._mode == self.MODE_PAN:
            self._hover_idx = -1
            self._hover_handle = None
            self._update_cursor_for_hover()
            return

        # 编辑模式拖拽判定与更新
        if self._edit_action == "pending_blank_pan":
            dist = math.hypot(
                view_pos.x() - self._edit_press_view_pos.x(),
                view_pos.y() - self._edit_press_view_pos.y(),
            )
            if dist >= self.DRAG_THRESHOLD_PX:
                self._edit_action = "blank_panning"
                self._start_pan(self._edit_press_view_pos, "blank")
                self._update_pan(view_pos)
            event.accept()
            return

        if self._edit_action in ("pending_handle", "dragging_handle") and self._drag_origin_rect is not None:
            if self._edit_action == "pending_handle":
                dist = math.hypot(
                    view_pos.x() - self._edit_press_view_pos.x(),
                    view_pos.y() - self._edit_press_view_pos.y(),
                )
                if dist < self.HANDLE_DRAG_THRESHOLD_PX:
                    event.accept()
                    return
                self._edit_action = "dragging_handle"
            self._drag_preview_rect = self._compute_resized_rect(
                self._drag_origin_rect, self._dragging_handle or "", scene_pos
            )
            self._update_edge_scroll_state(view_pos)
            self.viewport().update()
            event.accept()
            return

        if self._edit_action in ("pending_move", "moving_box") and self._drag_origin_rect is not None:
            if self._edit_action == "pending_move":
                dist = math.hypot(
                    view_pos.x() - self._edit_press_view_pos.x(),
                    view_pos.y() - self._edit_press_view_pos.y(),
                )
                if dist < self.DRAG_THRESHOLD_PX:
                    event.accept()
                    return
                self._edit_action = "moving_box"
            self._drag_preview_rect = self._compute_moved_rect(
                self._drag_origin_rect, scene_pos - self._drag_start_scene
            )
            self._update_edge_scroll_state(view_pos)
            self.viewport().update()
            event.accept()
            return

        # 编辑模式悬停高亮与手柄光标反馈
        prev_hover = (self._hover_idx, self._hover_handle)
        self._hover_handle = self._hit_test_handle(scene_pos)
        self._hover_idx = self._selected_idx if self._hover_handle else self._hit_test_box(scene_pos)
        self._update_cursor_for_hover()
        if (self._hover_idx, self._hover_handle) != prev_hover:
            self.viewport().update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        btn = event.button()
        view_pos = QPointF(event.position())
        scene_pos = self._map_view_to_scene_f(view_pos)

        # 右键松手：拖动过则收尾平移；轻点则取消草稿或发右键菜单信号
        if btn == Qt.MouseButton.RightButton and self._right_press_view_pos is not None:
            was_dragged = self._right_dragged
            self._right_press_view_pos = None
            self._right_dragged = False
            if was_dragged or (self._panning and self._pan_trigger == "right"):
                self._stop_pan()
                self._update_cursor_for_hover()
                event.accept()
                return
            if self._draw_state != "idle":
                self._draw_state = "idle"
                self._draw_dragged = False
                self._stop_edge_scroll()
                self._update_cursor_for_hover()
                self.viewport().update()
                event.accept()
                return
            if self._edit_action in ("pending_handle", "dragging_handle", "pending_move", "moving_box"):
                self._reset_edit_drag_state()
                self._stop_edge_scroll()
                self._update_cursor_for_hover()
                self.viewport().update()
                event.accept()
                return
            hit_idx = self._hit_test_box(scene_pos)
            self.context_menu_requested.emit(event.globalPosition().toPoint(), hit_idx)
            event.accept()
            return

        # 空格/中键/抓手模式平移收尾
        if self._panning and self._pan_trigger in ("middle", "space", "pan_mode"):
            self._stop_pan()
            self._update_cursor_for_hover()
            event.accept()
            return

        if btn != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return

        # 画框模式左键松手：拖拽≥4px 直接落框；轻点则锁定第一角点等待第二点
        if self._mode == self.MODE_DRAW and self._draw_state == "pressing":
            self._stop_edge_scroll()
            end_pt = self._clamp_point(scene_pos)
            dist = math.hypot(
                view_pos.x() - self._draw_press_view_pos.x(),
                view_pos.y() - self._draw_press_view_pos.y(),
            )
            if self._draw_dragged or dist >= self.DRAG_THRESHOLD_PX:
                start_pt = self._draw_start
                self._draw_state = "idle"
                self._draw_dragged = False
                self.viewport().update()
                self._emit_drawn_box_if_valid(start_pt, end_pt)
            else:
                self._draw_state = "waiting_second"
                self._draw_dragged = False
                self._draw_current = end_pt
                self.viewport().update()
            event.accept()
            return

        # 编辑模式左键松手
        if self._mode == self.MODE_EDIT:
            self._stop_edge_scroll()
            if self._edit_action == "pending_blank_pan":
                self._edit_action = "none"
                self.select_box(-1, emit_signal=True)
                self._update_cursor_for_hover()
                event.accept()
                return
            if self._edit_action == "blank_panning":
                self._edit_action = "none"
                self._stop_pan()
                self._update_cursor_for_hover()
                event.accept()
                return
            if self._edit_action in ("pending_handle", "pending_move"):
                self._reset_edit_drag_state()
                self._update_cursor_for_hover()
                self.viewport().update()
                event.accept()
                return
            if self._edit_action in ("dragging_handle", "moving_box"):
                idx = self._dragging_box_idx
                origin = self._drag_origin_rect
                preview = self._drag_preview_rect
                self._reset_edit_drag_state()
                self._update_cursor_for_hover()
                self.viewport().update()
                if (
                    origin is not None
                    and preview is not None
                    and (preview[2] - preview[0]) >= self.MIN_BOX_SIZE_PX
                    and (preview[3] - preview[1]) >= self.MIN_BOX_SIZE_PX
                    and any(abs(a - b) >= 0.5 for a, b in zip(origin, preview))
                ):
                    self.box_geometry_committed.emit(
                        idx, preview[0], preview[1], preview[2], preview[3]
                    )
                event.accept()
                return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._image_width <= 0 or event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        self._stop_edge_scroll()
        self._reset_edit_drag_state()
        scene_pos = self._map_view_to_scene_f(QPointF(event.position()))
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
        if self._mode == self.MODE_PAN:
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

        # 绘制图像双层精致边界（外层微阴影 + 内层浅灰细边），兼顾白底截图与暗底画面
        img_rect = QRectF(0.0, 0.0, float(self._image_width), float(self._image_height))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(0, 0, 0, 95), 2.4 * inv))
        painter.drawRect(img_rect)
        painter.setPen(QPen(QColor(255, 255, 255, 78), 1.0 * inv))
        painter.drawRect(img_rect)

        # 绘制全部已有矩形框（选中框置顶后绘，确保手柄与高亮不被覆盖）
        order = [i for i in range(len(self._boxes)) if i != self._selected_idx]
        if 0 <= self._selected_idx < len(self._boxes):
            order.append(self._selected_idx)

        for idx in order:
            box = self._boxes[idx]
            is_sel = idx == self._selected_idx
            is_hover = idx == self._hover_idx
            is_active_edit = (
                is_sel
                and self._edit_action in ("dragging_handle", "moving_box")
                and self._dragging_box_idx == idx
                and self._drag_preview_rect is not None
            )
            if is_active_edit and self._drag_preview_rect is not None:
                x1, y1, x2, y2 = self._drag_preview_rect
            else:
                x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            self._paint_box(
                painter,
                box,
                x1,
                y1,
                x2,
                y2,
                inv,
                is_sel=is_sel,
                is_hover=is_hover,
                show_live_size=is_active_edit,
            )

        # 绘制模式：双色高对比十字准星与实时草稿框
        if self._mode == self.MODE_DRAW:
            if self._cursor_scene_pos is not None:
                cx = max(0.0, min(float(self._image_width), self._cursor_scene_pos.x()))
                cy = max(0.0, min(float(self._image_height), self._cursor_scene_pos.y()))
                painter.setPen(QPen(QColor(0, 0, 0, 145), 2.2 * inv, Qt.PenStyle.SolidLine))
                painter.drawLine(QPointF(cx, 0.0), QPointF(cx, float(self._image_height)))
                painter.drawLine(QPointF(0.0, cy), QPointF(float(self._image_width), cy))
                painter.setPen(QPen(QColor(255, 255, 255, 225), 1.0 * inv, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(cx, 0.0), QPointF(cx, float(self._image_height)))
                painter.drawLine(QPointF(0.0, cy), QPointF(float(self._image_width), cy))

            if self._draw_state != "idle":
                dx1, dx2 = sorted((self._draw_start.x(), self._draw_current.x()))
                dy1, dy2 = sorted((self._draw_start.y(), self._draw_current.y()))
                r, g, b = label_color_rgb(self._active_label or "draft")
                draft_rect = QRectF(dx1, dy1, dx2 - dx1, dy2 - dy1)
                painter.setBrush(QBrush(QColor(r, g, b, 45)))
                painter.setPen(QPen(QColor(r, g, b, 240), 2.0 * inv, Qt.PenStyle.DashLine))
                painter.drawRect(draft_rect)
                # 两点画框等待第二点时标出锚点圆点，提示继续点击终点
                if self._draw_state == "waiting_second":
                    dot_r = 4.0 * inv
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.setPen(QPen(QColor(r, g, b, 255), 1.6 * inv))
                    painter.drawEllipse(self._draw_start, dot_r, dot_r)
                size_tip = f"{int(round(dx2 - dx1))}×{int(round(dy2 - dy1))}"
                if self._active_label:
                    size_tip = f"{self._active_label}  {size_tip}"
                self._paint_tag_pill(painter, dx1, dy1, size_tip, QColor(r, g, b, 230), inv)

        painter.restore()
        self._paint_hud_overlay(painter)

    def _paint_hud_overlay(self, painter: QPainter) -> None:
        """在画布视口顶部中央按需绘制轻量悬浮提示胶囊（不遮挡目标且不拦截鼠标事件）。"""
        if self._mode == self.MODE_DRAW and self._draw_state == "waiting_second":
            text = "已定起点 · 点击第二角点完成（滚轮缩放 / 中键平移，右键或 Esc 取消）"
            dot_color = QColor("#f85149")
        elif self._hud_message:
            text = self._hud_message
            dot_color = QColor(self._hud_accent)
        else:
            return

        vw = self.viewport().width()
        if vw < 120:
            return

        painter.save()
        painter.setWorldMatrixEnabled(False)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont("Microsoft YaHei UI", 9, QFont.Weight.Medium)
        painter.setFont(font)
        fm = QFontMetricsF(font)
        max_text_w = max(80.0, float(vw) - 56.0)
        display_text = fm.elidedText(text, Qt.TextElideMode.ElideRight, max_text_w)
        tw = fm.horizontalAdvance(display_text)
        th = fm.height()
        pill_w = tw + 30.0
        pill_h = th + 10.0
        px = (float(vw) - pill_w) / 2.0
        py = 10.0
        pill = QRectF(px, py, pill_w, pill_h)

        painter.setBrush(QBrush(QColor(22, 27, 34, 212)))
        painter.setPen(QPen(QColor(255, 255, 255, 42), 1.0))
        painter.drawRoundedRect(pill, pill_h / 2.0, pill_h / 2.0)

        dot_r = 3.5
        dot_cx = px + 12.0
        dot_cy = py + pill_h / 2.0
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(QPointF(dot_cx, dot_cy), dot_r, dot_r)

        painter.setPen(QPen(QColor(240, 246, 252)))
        text_rect = QRectF(px + 20.0, py, tw + 6.0, pill_h)
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            display_text,
        )
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
        show_live_size: bool = False,
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

        if show_live_size:
            size_text = f"{box.label}  {int(round(x2 - x1))}×{int(round(y2 - y1))}"
            self._paint_tag_pill(painter, x1, y1, size_text, QColor(r, g, b, 240), inv)
        elif self._show_labels:
            tag = box.label
            if box.score is not None:
                tag += f" {box.score:.2f}"
            if isinstance(box.flags, dict) and box.flags.get("llm_added"):
                tag += " ★"
            elif isinstance(box.flags, dict) and box.flags.get("box_refined"):
                tag += " ⌖"
            self._paint_tag_pill(painter, x1, y1, tag, QColor(r, g, b, 232), inv)

        # 仅在编辑模式下渲染选中框控制手柄，避免画框/抓手模式下误导拖动
        if is_sel and self._mode == self.MODE_EDIT:
            scale = max(self.zoom_factor(), 1e-4)
            min_dim_px = min((x2 - x1) * scale, (y2 - y1) * scale)
            hs = (3.4 if min_dim_px < 22.0 else 4.6) * inv
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(r, g, b), 1.6 * inv))
            for hpt in self._visible_handles_for_box(x1, y1, x2, y2).values():
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
