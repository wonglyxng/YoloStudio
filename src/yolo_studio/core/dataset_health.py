"""纯离线数据集标注健康检查引擎（类似 Roboflow Health Check）。

职责：
    在不依赖任何 `.pt` 模型权重的前提下，高速扫描当前标注目录中的全部图片与 `.json` 文件，
    汇总标注覆盖率、类别分布不平衡指标，并定位常见脏标注问题：
    1. 损坏或无法解析的 `.json` 文件；
    2. 过小目标框（宽或高小于阈值像素，易在 YOLO 下采样中丢失）；
    3. 极端长宽比框（疑似手抖拉歪或误拖线框）；
    4. 同图同类别高度重合重复框（`IoU >= duplicate_iou`）；
    5. 人工标记的「待确认问题」或上游 AI 漏标提醒图片；
    6. 未登记在当前类别清单中的孤立类别与零样本类别。

运行环境与协作边界：
    纯 Python 核心模块，不依赖 PyQt6；只读取 `.json` 文本而不解码图片像素，扫描数千张图耗时毫秒级。
    所有问题条目均携带目标图片路径与可选目标框索引，供工作台一键跳转并聚焦修复。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Iterable

from yolo_studio.core.annotation_io import RectAnnotation, resolve_annotation_json_path


class HealthIssueKind(str, Enum):
    """健康检查识别出的异常类型。"""

    CORRUPT_JSON = "corrupt_json"
    DUPLICATE_BOX = "duplicate_box"
    TINY_BOX = "tiny_box"
    EXTREME_ASPECT = "extreme_aspect"
    UNREGISTERED_LABEL = "unregistered_label"
    IMAGE_ISSUE_FLAG = "image_issue_flag"


HEALTH_ISSUE_LABELS: dict[HealthIssueKind, str] = {
    HealthIssueKind.CORRUPT_JSON: "损坏 JSON",
    HealthIssueKind.DUPLICATE_BOX: "高度重合框",
    HealthIssueKind.TINY_BOX: "过小目标框",
    HealthIssueKind.EXTREME_ASPECT: "极端长宽比",
    HealthIssueKind.UNREGISTERED_LABEL: "未登记类别",
    HealthIssueKind.IMAGE_ISSUE_FLAG: "待确认问题",
}

_SEVERITY_RANK: dict[HealthIssueKind, int] = {
    HealthIssueKind.CORRUPT_JSON: 0,
    HealthIssueKind.IMAGE_ISSUE_FLAG: 1,
    HealthIssueKind.DUPLICATE_BOX: 2,
    HealthIssueKind.UNREGISTERED_LABEL: 3,
    HealthIssueKind.TINY_BOX: 4,
    HealthIssueKind.EXTREME_ASPECT: 5,
}


@dataclass(frozen=True)
class HealthIssue:
    """单条可一键跳转定位的标注健康异常记录。"""

    kind: HealthIssueKind
    image_path: Path
    box_index: int | None
    label: str
    summary: str


@dataclass(frozen=True)
class ClassHealthStat:
    """单个类别在当前数据集中的样本分布统计。"""

    label: str
    box_count: int
    image_count: int
    share_ratio: float
    sample_image: Path | None
    is_underrepresented: bool
    is_unregistered: bool


@dataclass
class DatasetHealthReport:
    """整目录数据集标注健康体检汇总报告。"""

    total_images: int = 0
    labeled_images: int = 0
    empty_images: int = 0
    unlabeled_images: int = 0
    issue_flagged_images: int = 0
    corrupt_json_count: int = 0
    total_boxes: int = 0
    imbalance_ratio: float = 0.0
    class_stats: list[ClassHealthStat] = field(default_factory=list)
    zero_count_classes: list[str] = field(default_factory=list)
    issues: list[HealthIssue] = field(default_factory=list)


def _rect_iou_coords(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def analyze_dataset_health(
    image_paths: Iterable[Path],
    *,
    known_classes: Iterable[str] = (),
    min_box_px: float = 10.0,
    max_aspect_ratio: float = 8.0,
    duplicate_iou: float = 0.85,
    low_count_threshold: int = 5,
) -> DatasetHealthReport:
    """扫描一组图片路径对应的标注 `.json` 并生成数据集健康体检报告（只读无副作用）。

    参数单位与阈值：
        - `min_box_px`：矩形宽或高低于此值（px）视为过小目标框；
        - `max_aspect_ratio`：`max(w/h, h/w) >= max_aspect_ratio` 视为极端长宽比；
        - `duplicate_iou`：同图同类别任意两框 IoU 达到此值视为高度重合重复框；
        - `low_count_threshold`：类别总框数低于此值标记为样本偏少预警。
    """
    images = [Path(p).resolve() for p in image_paths]
    known_list = [str(c).strip() for c in known_classes if str(c).strip()]
    known_set = set(known_list)

    report = DatasetHealthReport(total_images=len(images))
    cls_box_counts: dict[str, int] = {}
    cls_img_counts: dict[str, int] = {}
    cls_first_img: dict[str, Path] = {}
    issues: list[HealthIssue] = []

    for img_path in images:
        json_path = resolve_annotation_json_path(img_path)
        if not json_path.is_file():
            report.unlabeled_images += 1
            continue

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            report.corrupt_json_count += 1
            issues.append(
                HealthIssue(
                    kind=HealthIssueKind.CORRUPT_JSON,
                    image_path=img_path,
                    box_index=None,
                    label="",
                    summary=f"标注文件无法解析：{exc}",
                )
            )
            continue

        if not isinstance(data, dict):
            report.corrupt_json_count += 1
            issues.append(
                HealthIssue(
                    kind=HealthIssueKind.CORRUPT_JSON,
                    image_path=img_path,
                    box_index=None,
                    label="",
                    summary="标注文件根结构不是 JSON 对象",
                )
            )
            continue

        flags = data.get("flags") if isinstance(data.get("flags"), dict) else {}
        has_issue_flag = bool(flags.get("issue_flag")) or bool(str(flags.get("issue_note") or "").strip())
        has_missing = bool(flags.get("has_missing_hint")) or bool(str(flags.get("missing_note") or "").strip())
        if has_issue_flag or has_missing:
            report.issue_flagged_images += 1
            note_parts: list[str] = []
            issue_note = str(flags.get("issue_note") or "").strip()
            missing_note = str(flags.get("missing_note") or "").strip()
            if has_issue_flag:
                note_parts.append(issue_note or "人工标记待确认")
            if has_missing:
                note_parts.append(f"AI 复核：{missing_note or '疑似漏标'}")
            issues.append(
                HealthIssue(
                    kind=HealthIssueKind.IMAGE_ISSUE_FLAG,
                    image_path=img_path,
                    box_index=None,
                    label="",
                    summary="；".join(note_parts),
                )
            )

        rects: list[tuple[int, RectAnnotation]] = []
        rect_idx = 0
        for raw_shape in data.get("shapes") or []:
            if not isinstance(raw_shape, dict):
                continue
            shape_type = str(raw_shape.get("shape_type") or "rectangle").strip().lower()
            if shape_type != "rectangle":
                continue
            label = str(raw_shape.get("label") or "").strip()
            pts = raw_shape.get("points") or []
            if not label or not isinstance(pts, list) or len(pts) < 2:
                continue
            try:
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[1][0]), float(pts[1][1])
            except (TypeError, ValueError, IndexError):
                continue
            box = RectAnnotation(label=label, x1=x1, y1=y1, x2=x2, y2=y2)
            rects.append((rect_idx, box))
            rect_idx += 1

        if rects:
            report.labeled_images += 1
        else:
            report.empty_images += 1

        seen_in_img: set[str] = set()
        for b_idx, box in rects:
            report.total_boxes += 1
            cls_box_counts[box.label] = cls_box_counts.get(box.label, 0) + 1
            if box.label not in seen_in_img:
                seen_in_img.add(box.label)
                cls_img_counts[box.label] = cls_img_counts.get(box.label, 0) + 1
            cls_first_img.setdefault(box.label, img_path)

            if known_set and box.label not in known_set:
                issues.append(
                    HealthIssue(
                        kind=HealthIssueKind.UNREGISTERED_LABEL,
                        image_path=img_path,
                        box_index=b_idx,
                        label=box.label,
                        summary=f"框 #{b_idx + 1} 使用了未在类别清单登记的类别「{box.label}」",
                    )
                )

            bw, bh = box.width, box.height
            if bw < min_box_px or bh < min_box_px:
                issues.append(
                    HealthIssue(
                        kind=HealthIssueKind.TINY_BOX,
                        image_path=img_path,
                        box_index=b_idx,
                        label=box.label,
                        summary=f"框 #{b_idx + 1} 尺寸仅 {bw:.1f}×{bh:.1f} px（<{min_box_px:.0f} px）",
                    )
                )
            elif min(bw, bh) > 0.0:
                ratio = max(bw / bh, bh / bw)
                if ratio >= max_aspect_ratio:
                    issues.append(
                        HealthIssue(
                            kind=HealthIssueKind.EXTREME_ASPECT,
                            image_path=img_path,
                            box_index=b_idx,
                            label=box.label,
                            summary=f"框 #{b_idx + 1} 长宽比 {ratio:.1f}:1（{bw:.0f}×{bh:.0f} px）",
                        )
                    )

        # 同图同类重复框检测（O(k^2)，单图框数通常几十个以内）
        for i in range(len(rects)):
            idx_a, box_a = rects[i]
            coords_a = (box_a.x1, box_a.y1, box_a.x2, box_a.y2)
            for j in range(i + 1, len(rects)):
                idx_b, box_b = rects[j]
                if box_a.label != box_b.label:
                    continue
                iou = _rect_iou_coords(coords_a, (box_b.x1, box_b.y1, box_b.x2, box_b.y2))
                if iou >= duplicate_iou:
                    issues.append(
                        HealthIssue(
                            kind=HealthIssueKind.DUPLICATE_BOX,
                            image_path=img_path,
                            box_index=idx_b,
                            label=box_b.label,
                            summary=f"框 #{idx_a + 1} 与 #{idx_b + 1} 高度重合（IoU={iou:.2f}）",
                        )
                    )

    # 类别统计与不平衡比
    all_labels = list(dict.fromkeys(list(cls_box_counts.keys())))
    all_labels.sort(key=lambda lbl: (-cls_box_counts[lbl], lbl))
    total_boxes_safe = max(1, report.total_boxes)
    for lbl in all_labels:
        cnt = cls_box_counts[lbl]
        report.class_stats.append(
            ClassHealthStat(
                label=lbl,
                box_count=cnt,
                image_count=cls_img_counts.get(lbl, 0),
                share_ratio=cnt / total_boxes_safe,
                sample_image=cls_first_img.get(lbl),
                is_underrepresented=cnt < low_count_threshold,
                is_unregistered=bool(known_set and lbl not in known_set),
            )
        )
    report.zero_count_classes = [c for c in known_list if cls_box_counts.get(c, 0) == 0]
    if len(report.class_stats) >= 2:
        max_cnt = report.class_stats[0].box_count
        min_cnt = max(1, report.class_stats[-1].box_count)
        report.imbalance_ratio = round(max_cnt / min_cnt, 2)

    issues.sort(key=lambda it: (_SEVERITY_RANK.get(it.kind, 99), it.image_path.name, it.box_index or -1))
    report.issues = issues
    return report
