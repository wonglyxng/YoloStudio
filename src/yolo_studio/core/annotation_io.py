"""原生标注工作台 `.json` 标注数据读写、状态索引与类别工具模块。

职责：
    读写标准 LabelMe / X-AnyLabeling 兼容矩形标注 `.json` 文件（含 `shapes`、AI 置信度 `score`、
    审核说明 `description` 与图像级 `flags` 如 `has_missing_hint` / `missing_note` / `verified`），
    提供目录图片扫描、轻量标注状态摘要、`classes.txt` 读写与稳定类别配色。

运行环境与协作边界：
    纯 Python 标准库 + `Pillow`，不依赖 PyQt6；供 `gui.tab_labeling` 原生标注工作台及单元测试调用。
    与 `core.dataset_converter` 和 `core.mllm_reviewer` 保持 100% `.json` 格式双向互通；
    读取历史文件若包含非矩形形状（如 `polygon` / `rotation`），将其无损保存在 `preserved_shapes`
    并在写出时原样保留，不静默丢弃用户历史数据。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from yolo_studio.core.dataset_converter import find_images

# 默认输出格式版本号，与现有 LabelMe / X-AnyLabeling 工具链保持完全兼容
ANNOTATION_JSON_VERSION = "2.4.4"

# 高对比度高可读类别调色板（RGB），兼顾浅色与深色画面边缘辨识度
_CLASS_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (22, 163, 74),    # 翠绿
    (37, 99, 235),    # 宝蓝
    (234, 88, 12),    # 亮橙
    (220, 38, 38),    # 绯红
    (147, 51, 234),   # 亮紫
    (8, 145, 178),    # 青蓝
    (202, 138, 4),    # 琥珀黄
    (219, 39, 119),   # 玫红
    (13, 148, 136),   # 蓝绿
    (79, 70, 229),    # 靛蓝
    (194, 65, 12),    # 深橙
    (101, 163, 13),   # 橄榄绿
)


def label_color_rgb(label: str) -> tuple[int, int, int]:
    """根据类别名称返回稳定且高对比度的 `(R, G, B)` 配色（同一名称跨会话保持一致）。"""
    key = (label or "").strip()
    if not key:
        return (100, 116, 139)
    digest = hashlib.md5(key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:2], "big") % len(_CLASS_PALETTE_RGB)
    return _CLASS_PALETTE_RGB[idx]


def get_image_dimensions(image_path: Path) -> tuple[int, int]:
    """读取图片像素宽高 `(width, height)`；文件无法解码时返回 `(0, 0)`。"""
    try:
        with Image.open(image_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


@dataclass
class RectAnnotation:
    """单个像素级矩形标注框。

    坐标系与约束：
        `x1, y1, x2, y2` 为原图像素坐标系下的左上角与右下角坐标（原点位于图像左上角，单位 px）；
        `score` 为可选的 AI 检测置信度（`[0.0, 1.0]`，人工手画时为 `None`）；
        `description` 存储 LLM/SAM 3 审核理由或人工备注；`flags` 保留 `llm_added` / `box_refined` 等标记。
    """

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = None
    description: str = ""
    difficult: bool = False
    group_id: int | None = None
    flags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lx, rx = float(self.x1), float(self.x2)
        ty, by = float(self.y1), float(self.y2)
        self.x1, self.x2 = (lx, rx) if lx <= rx else (rx, lx)
        self.y1, self.y2 = (ty, by) if ty <= by else (by, ty)
        self.label = str(self.label).strip()
        if self.score is not None:
            try:
                self.score = float(self.score)
            except (TypeError, ValueError):
                self.score = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clamped(self, img_w: int, img_h: int) -> RectAnnotation:
        """返回裁剪到 `[0, img_w] × [0, img_h]` 图像边界内的新框副本。"""
        max_w = float(max(0, img_w))
        max_h = float(max(0, img_h))
        x1 = max(0.0, min(max_w, self.x1)) if max_w > 0 else max(0.0, self.x1)
        x2 = max(0.0, min(max_w, self.x2)) if max_w > 0 else max(0.0, self.x2)
        y1 = max(0.0, min(max_h, self.y1)) if max_h > 0 else max(0.0, self.y1)
        y2 = max(0.0, min(max_h, self.y2)) if max_h > 0 else max(0.0, self.y2)
        return RectAnnotation(
            label=self.label,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=self.score,
            description=self.description,
            difficult=self.difficult,
            group_id=self.group_id,
            flags=copy.deepcopy(self.flags),
        )

    def clone(self) -> RectAnnotation:
        """深拷贝当前矩形框对象（用于撤销/重做快照及框复制）。"""
        return RectAnnotation(
            label=self.label,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            score=self.score,
            description=self.description,
            difficult=self.difficult,
            group_id=self.group_id,
            flags=copy.deepcopy(self.flags),
        )

    def to_shape_dict(self) -> dict[str, Any]:
        """序列化为标准 LabelMe / X-AnyLabeling 兼容的 `shape` 字典。"""
        return {
            "label": self.label,
            "score": round(self.score, 4) if self.score is not None else None,
            "points": [[round(self.x1, 2), round(self.y1, 2)], [round(self.x2, 2), round(self.y2, 2)]],
            "group_id": self.group_id,
            "description": self.description,
            "difficult": bool(self.difficult),
            "shape_type": "rectangle",
            "flags": copy.deepcopy(self.flags) if isinstance(self.flags, dict) else {},
        }


@dataclass
class ImageAnnotationDoc:
    """单张图片对应的完整标注文档状态。

    状态归属与兼容性：
        `boxes` 为当前画布可编辑的矩形框列表；`preserved_shapes` 保存读取自历史 `.json` 的
        非矩形形状原始字典，保存时原样写回以防误删；`has_json_file` 区分“尚未标注”与“显式空图负样本”。
    """

    image_path: Path
    json_path: Path
    image_width: int
    image_height: int
    boxes: list[RectAnnotation] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    preserved_shapes: list[dict[str, Any]] = field(default_factory=list)
    version: str = ANNOTATION_JSON_VERSION
    has_json_file: bool = False

    @property
    def has_missing_hint(self) -> bool:
        return bool(self.flags.get("has_missing_hint"))

    @property
    def missing_note(self) -> str:
        val = self.flags.get("missing_note")
        return str(val).strip() if val else ""

    @property
    def is_verified(self) -> bool:
        return bool(self.flags.get("verified"))

    @property
    def has_issue(self) -> bool:
        """是否被人工标记为“待确认问题”（含仅有备注文本的情况）。"""
        return bool(self.flags.get("issue_flag")) or bool(self.issue_note)

    @property
    def issue_note(self) -> str:
        val = self.flags.get("issue_note")
        return str(val).strip() if val else ""


@dataclass(frozen=True)
class ImageStatusSummary:
    """文件列表栏使用的轻量单图状态摘要（读取 `.json` 文本而不解码像素）。"""

    image_path: Path
    json_path: Path
    has_json: bool
    box_count: int
    has_missing_hint: bool
    is_verified: bool
    labels: tuple[str, ...] = ()
    has_issue: bool = False
    issue_note: str = ""

    @property
    def status_key(self) -> str:
        """归类状态：`review`（含漏标提示或问题标记）、`labeled`（含框）、`empty`（显式 0 框负样本）、`unlabeled`（无标注文件）。"""
        if self.has_missing_hint or self.has_issue:
            return "review"
        if not self.has_json:
            return "unlabeled"
        if self.box_count > 0:
            return "labeled"
        return "empty"


def resolve_annotation_json_path(image_path: Path) -> Path:
    """返回图片同目录下的同名 `.json` 标注路径。"""
    return image_path.with_suffix(".json")


def _parse_rect_shape(shape: dict[str, Any]) -> RectAnnotation | None:
    """将单个 `rectangle` shape 字典解析为 `RectAnnotation`；结构非法时返回 `None`。"""
    label = str(shape.get("label", "")).strip()
    points = shape.get("points") or []
    if not label or not isinstance(points, list) or len(points) < 2:
        return None
    try:
        xs = [float(pt[0]) for pt in points]
        ys = [float(pt[1]) for pt in points]
    except (TypeError, ValueError, IndexError):
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
        return None
    raw_flags = shape.get("flags")
    flags = copy.deepcopy(raw_flags) if isinstance(raw_flags, dict) else {}
    group_id = shape.get("group_id")
    if group_id is not None:
        try:
            group_id = int(group_id)
        except (TypeError, ValueError):
            group_id = None
    return RectAnnotation(
        label=label,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=shape.get("score"),
        description=str(shape.get("description") or "").strip(),
        difficult=bool(shape.get("difficult", False)),
        group_id=group_id,
        flags=flags,
    )


def load_image_annotation(
    image_path: Path,
    *,
    image_size: tuple[int, int] | None = None,
) -> ImageAnnotationDoc:
    """加载单张图片及其同名 `.json` 标注文档。

    失败容错：
        同名 `.json` 不存在、内容损坏或非字典结构时返回空 `boxes` 文档，不抛出异常中断切图流程。
    """
    resolved_img = image_path.resolve()
    json_path = resolve_annotation_json_path(resolved_img)
    img_w, img_h = image_size if image_size is not None else get_image_dimensions(resolved_img)

    if not json_path.is_file():
        return ImageAnnotationDoc(
            image_path=resolved_img,
            json_path=json_path,
            image_width=img_w,
            image_height=img_h,
            has_json_file=False,
        )

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ImageAnnotationDoc(
            image_path=resolved_img,
            json_path=json_path,
            image_width=img_w,
            image_height=img_h,
            has_json_file=True,
        )

    if not isinstance(data, dict):
        return ImageAnnotationDoc(
            image_path=resolved_img,
            json_path=json_path,
            image_width=img_w,
            image_height=img_h,
            has_json_file=True,
        )

    if img_w <= 0:
        img_w = int(data.get("imageWidth") or 0)
    if img_h <= 0:
        img_h = int(data.get("imageHeight") or 0)

    boxes: list[RectAnnotation] = []
    preserved: list[dict[str, Any]] = []
    for raw_shape in data.get("shapes") or []:
        if not isinstance(raw_shape, dict):
            continue
        shape_type = str(raw_shape.get("shape_type") or "rectangle").strip().lower()
        if shape_type == "rectangle":
            parsed = _parse_rect_shape(raw_shape)
            if parsed is not None:
                boxes.append(parsed.clamped(img_w, img_h) if (img_w > 0 and img_h > 0) else parsed)
        else:
            preserved.append(copy.deepcopy(raw_shape))

    raw_flags = data.get("flags")
    flags = copy.deepcopy(raw_flags) if isinstance(raw_flags, dict) else {}
    version = str(data.get("version") or ANNOTATION_JSON_VERSION)

    return ImageAnnotationDoc(
        image_path=resolved_img,
        json_path=json_path,
        image_width=img_w,
        image_height=img_h,
        boxes=boxes,
        flags=flags,
        preserved_shapes=preserved,
        version=version,
        has_json_file=True,
    )


def save_image_annotation(doc: ImageAnnotationDoc) -> Path:
    """将 `ImageAnnotationDoc` 原子写入同名 `.json` 文件并更新 `doc.has_json_file = True`。

    原子写入策略：
        先在目标目录创建临时文件写入 UTF-8 编码 JSON，再用 `os.replace` 替换目标路径，
        防止中途异常造成已有标注文件截断损坏。
    """
    img_w, img_h = doc.image_width, doc.image_height
    if img_w <= 0 or img_h <= 0:
        img_w, img_h = get_image_dimensions(doc.image_path)
        doc.image_width, doc.image_height = img_w, img_h

    shapes: list[dict[str, Any]] = []
    for box in doc.boxes:
        clamped = box.clamped(img_w, img_h) if (img_w > 0 and img_h > 0) else box
        if not clamped.label or clamped.width < 1.0 or clamped.height < 1.0:
            continue
        shapes.append(clamped.to_shape_dict())
    if doc.preserved_shapes:
        shapes.extend(copy.deepcopy(doc.preserved_shapes))

    payload: dict[str, Any] = {
        "version": doc.version or ANNOTATION_JSON_VERSION,
        "flags": copy.deepcopy(doc.flags) if isinstance(doc.flags, dict) else {},
        "shapes": shapes,
        "imagePath": doc.image_path.name,
        "imageData": None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w),
    }

    target = doc.json_path
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}_", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp_path, target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    doc.has_json_file = True
    return target


def inspect_image_status(image_path: Path) -> ImageStatusSummary:
    """快速扫描单张图片的标注状态摘要（仅解析 JSON 文本，不读取图像像素）。"""
    resolved_img = image_path.resolve()
    json_path = resolve_annotation_json_path(resolved_img)
    if not json_path.is_file():
        return ImageStatusSummary(
            image_path=resolved_img,
            json_path=json_path,
            has_json=False,
            box_count=0,
            has_missing_hint=False,
            is_verified=False,
        )

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ImageStatusSummary(
            image_path=resolved_img,
            json_path=json_path,
            has_json=True,
            box_count=0,
            has_missing_hint=False,
            is_verified=False,
        )

    if not isinstance(data, dict):
        return ImageStatusSummary(
            image_path=resolved_img,
            json_path=json_path,
            has_json=True,
            box_count=0,
            has_missing_hint=False,
            is_verified=False,
        )

    shapes = data.get("shapes") or []
    labels: list[str] = []
    valid_count = 0
    for sh in shapes:
        if not isinstance(sh, dict):
            continue
        lbl = str(sh.get("label") or "").strip()
        pts = sh.get("points") or []
        if lbl and isinstance(pts, list) and len(pts) >= 2:
            valid_count += 1
            if lbl not in labels:
                labels.append(lbl)

    flags = data.get("flags") if isinstance(data.get("flags"), dict) else {}
    issue_note = str(flags.get("issue_note") or "").strip()
    has_issue = bool(flags.get("issue_flag")) or bool(issue_note)
    return ImageStatusSummary(
        image_path=resolved_img,
        json_path=json_path,
        has_json=True,
        box_count=valid_count,
        has_missing_hint=bool(flags.get("has_missing_hint")),
        is_verified=bool(flags.get("verified")),
        labels=tuple(labels),
        has_issue=has_issue,
        issue_note=issue_note,
    )


def scan_labeling_directory(directory: Path, *, recursive: bool = True) -> list[Path]:
    """扫描待标注目录中的所有图片（默认递归子目录，排序稳定，与数据集准备页保持一致）。"""
    if not directory.is_dir():
        return []
    return find_images(directory.resolve(), recursive=recursive)


def load_classes_txt(classes_file: Path) -> list[str]:
    """读取 `classes.txt` 中去重保序的非空类别名称列表；文件不存在或不可读时返回空列表。"""
    if not classes_file.is_file():
        return []
    try:
        lines = classes_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for line in lines:
        name = line.strip()
        if name and name not in seen:
            seen.add(name)
            labels.append(name)
    return labels


def save_classes_txt(classes_file: Path, labels: list[str]) -> Path:
    """将去重保序的非空类别列表写入 `classes.txt`（UTF-8 编码）。"""
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in labels:
        name = str(raw).strip()
        if name and name not in seen:
            seen.add(name)
            deduped.append(name)
    classes_file.parent.mkdir(parents=True, exist_ok=True)
    classes_file.write_text("\n".join(deduped) + ("\n" if deduped else ""), encoding="utf-8")
    return classes_file
