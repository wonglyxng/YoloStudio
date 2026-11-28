"""基于已训练 YOLO 权重的标注质量反查与漏标/脏标注发现引擎（轻量级迷你 FiftyOne）。

职责：
    使用本地已训练的 YOLO `.pt` 权重对标注目录中的图片执行离线推理，并将模型预测框与
    人工 `.json` 真值框进行多轮空间-类别匹配，自动识别四类最影响训练 mAP 的高频数据问题：
    1. **疑似漏标 (`MISSING`)**：模型以高置信度检出目标，但人工未标注；
    2. **疑似错类 (`WRONG_LABEL`)**：模型高置信度预测框与人工框高度重合，但类别不一致；
    3. **疑似误标 (`SPURIOUS`)**：人工标注了模型已知类别的目标框，但模型在低阈值下也完全无空间支持；
    4. **疑似偏移 (`MISALIGNED`)**：同类别高置信度预测框与人工框存在部分重叠但 IoU 明显偏低（框过松/偏歪）。

运行环境与协作边界：
    纯 Python 核心模块，不依赖 PyQt6；实际推理时延迟导入 `ultralytics.YOLO` 与 `cv2`，
    采用 `np.fromfile + cv2.imdecode` 以兼容 Windows 中文路径。支持注入 `predictor_override`
    实现零权重离线单元测试；本模块只读图片与 `.json`，绝不直接修改任何磁盘标注文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import numpy as np

from yolo_studio.core.annotation_io import RectAnnotation, load_image_annotation


class AuditIssueKind(str, Enum):
    """单条标注质量反查异常类型。"""

    MISSING = "missing"
    WRONG_LABEL = "wrong_label"
    SPURIOUS = "spurious"
    MISALIGNED = "misaligned"


@dataclass(frozen=True)
class PredictedBox:
    """模型对单张图输出的像素坐标预测框。

    坐标系约定：
        `x1, y1, x2, y2` 为原图左上角原点的像素坐标，`confidence` 取值 `0.0 ~ 1.0`。
    """

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True)
class AuditIssue:
    """单条可解释的模型反查异常记录。

    字段约定：
        - `gt_index`：对应当前图人工标注框列表中的 0-based 索引；疑似漏标无对应人工框时为 `None`。
        - `pred_box`：关联的模型预测框 `(x1, y1, x2, y2)`；疑似误标无模型支持框时为 `None`。
        - `score`：单条异常严重度，越大越值得人工优先核查。
    """

    kind: AuditIssueKind
    score: float
    gt_index: int | None = None
    gt_label: str = ""
    pred_label: str = ""
    pred_conf: float = 0.0
    pred_box: tuple[float, float, float, float] | None = None
    iou: float = 0.0
    summary: str = ""


@dataclass
class ImageAuditReport:
    """单张图片的反查诊断汇总。"""

    image_path: Path
    has_json: bool
    gt_count: int
    pred_count: int
    issues: list[AuditIssue] = field(default_factory=list)
    anomaly_score: float = 0.0

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    @property
    def missing_count(self) -> int:
        return sum(1 for item in self.issues if item.kind == AuditIssueKind.MISSING)

    @property
    def wrong_label_count(self) -> int:
        return sum(1 for item in self.issues if item.kind == AuditIssueKind.WRONG_LABEL)

    @property
    def spurious_count(self) -> int:
        return sum(1 for item in self.issues if item.kind == AuditIssueKind.SPURIOUS)

    @property
    def misaligned_count(self) -> int:
        return sum(1 for item in self.issues if item.kind == AuditIssueKind.MISALIGNED)

    def compact_badge(self) -> str:
        """生成用于图片列表展示的紧凑异常角标（例如 `漏1 错类1`）。"""
        parts: list[str] = []
        if self.missing_count:
            parts.append(f"漏{self.missing_count}")
        if self.wrong_label_count:
            parts.append(f"错类{self.wrong_label_count}")
        if self.spurious_count:
            parts.append(f"误标{self.spurious_count}")
        if self.misaligned_count:
            parts.append(f"偏移{self.misaligned_count}")
        return " ".join(parts)

    def headline_summary(self) -> str:
        """生成用于画布顶部诊断横幅的单行可读摘要。"""
        if not self.issues:
            return "模型反查未发现明显异常"
        counts: list[str] = []
        if self.missing_count:
            counts.append(f"疑似漏标 {self.missing_count}")
        if self.wrong_label_count:
            counts.append(f"疑似错类 {self.wrong_label_count}")
        if self.spurious_count:
            counts.append(f"疑似误标 {self.spurious_count}")
        if self.misaligned_count:
            counts.append(f"疑似偏移 {self.misaligned_count}")
        top_notes = "；".join(item.summary for item in self.issues[:3] if item.summary)
        if len(self.issues) > 3:
            top_notes += f" 等共 {len(self.issues)} 项"
        return f"模型反查：{' · '.join(counts)}（{top_notes}）"


@dataclass
class LabelAuditConfig:
    """模型反查任务配置。"""

    weights_path: Path
    images: list[Path]
    only_annotated: bool = True
    pred_conf: float = 0.25
    missing_conf: float = 0.50
    match_iou: float = 0.50
    low_iou: float = 0.20
    imgsz: int = 640
    device: str = "0"


@dataclass
class LabelAuditResult:
    """整批图片的模型反查汇总结果（`reports` 按 `anomaly_score` 降序排列）。"""

    weights_path: Path
    model_classes: list[str]
    scanned_images: int
    skipped_unannotated: int
    cancelled: bool
    reports: list[ImageAuditReport] = field(default_factory=list)
    report_map: dict[Path, ImageAuditReport] = field(default_factory=dict)

    @property
    def flagged_images(self) -> int:
        return sum(1 for r in self.reports if r.has_issues)

    @property
    def total_missing(self) -> int:
        return sum(r.missing_count for r in self.reports)

    @property
    def total_wrong_label(self) -> int:
        return sum(r.wrong_label_count for r in self.reports)

    @property
    def total_spurious(self) -> int:
        return sum(r.spurious_count for r in self.reports)

    @property
    def total_misaligned(self) -> int:
        return sum(r.misaligned_count for r in self.reports)


def _box_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """计算两个 `(x1, y1, x2, y2)` 矩形框的交并比 IoU。"""
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


def _norm_label(label: str) -> str:
    return (label or "").strip().lower()


def compare_image_predictions(
    image_path: Path,
    gt_boxes: list[RectAnnotation],
    predictions: list[PredictedBox],
    *,
    has_json: bool = True,
    known_classes: set[str] | None = None,
    missing_conf: float = 0.50,
    match_iou: float = 0.50,
    low_iou: float = 0.20,
) -> ImageAuditReport:
    """将单张图的人工真值框与模型预测框进行多轮匹配，输出可解释异常清单与综合异常分。

    防误报原则：
        1. 先完成同类别高 IoU 匹配，确保正常框不被跨类或邻近预测干扰；
        2. 若提供了 `known_classes`（模型训练过的类别集合），仅对模型已知类别的人工框判定
           `SPURIOUS`（疑似误标），防止新增未训练类别被误报为脏标注；
        3. 若预测框与任意人工框已有 `IoU >= match_iou`，绝不重复报为漏标。
    """
    n_gt = len(gt_boxes)
    n_pred = len(predictions)
    gt_xyxy = [(float(b.x1), float(b.y1), float(b.x2), float(b.y2)) for b in gt_boxes]
    pred_xyxy = [(float(p.x1), float(p.y1), float(p.x2), float(p.y2)) for p in predictions]
    gt_norm = [_norm_label(b.label) for b in gt_boxes]
    pred_norm = [_norm_label(p.label) for p in predictions]
    known_norm = {_norm_label(c) for c in known_classes} if known_classes else None

    iou_mat: list[list[float]] = [
        [_box_iou_xyxy(gt_xyxy[gi], pred_xyxy[pi]) for pi in range(n_pred)]
        for gi in range(n_gt)
    ]

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    supported_gt: set[int] = set()
    issues: list[AuditIssue] = []

    # 第 1 轮：同类别高重合优先匹配（正常一致）
    same_cls_pairs: list[tuple[float, float, int, int]] = []
    for gi in range(n_gt):
        for pi in range(n_pred):
            if gt_norm[gi] == pred_norm[pi] and iou_mat[gi][pi] >= match_iou:
                same_cls_pairs.append((iou_mat[gi][pi], predictions[pi].confidence, gi, pi))
    same_cls_pairs.sort(reverse=True)
    for _iou, _conf, gi, pi in same_cls_pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        supported_gt.add(gi)

    # 第 2 轮：跨类高重合匹配 -> 疑似错类（高置信报告；低置信仅视为弱支持避免误报误标）
    cross_cls_pairs: list[tuple[float, float, int, int]] = []
    for gi in range(n_gt):
        if gi in matched_gt:
            continue
        for pi in range(n_pred):
            if pi in matched_pred:
                continue
            if gt_norm[gi] != pred_norm[pi] and iou_mat[gi][pi] >= match_iou:
                cross_cls_pairs.append((iou_mat[gi][pi], predictions[pi].confidence, gi, pi))
    cross_cls_pairs.sort(reverse=True)
    for iou_val, conf_val, gi, pi in cross_cls_pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        supported_gt.add(gi)
        if conf_val >= missing_conf:
            p = predictions[pi]
            g = gt_boxes[gi]
            issues.append(
                AuditIssue(
                    kind=AuditIssueKind.WRONG_LABEL,
                    score=float(conf_val * (0.65 + 0.35 * iou_val)),
                    gt_index=gi,
                    gt_label=g.label,
                    pred_label=p.label,
                    pred_conf=conf_val,
                    pred_box=pred_xyxy[pi],
                    iou=iou_val,
                    summary=f"疑似错类：标 {g.label} → 预测 {p.label} ({conf_val:.2f})",
                )
            )

    # 第 3 轮：同类别中等重叠（low_iou <= IoU < match_iou）-> 疑似框偏移/过松过紧
    shift_pairs: list[tuple[float, float, int, int]] = []
    for gi in range(n_gt):
        if gi in matched_gt:
            continue
        for pi in range(n_pred):
            if pi in matched_pred:
                continue
            if gt_norm[gi] == pred_norm[pi] and low_iou <= iou_mat[gi][pi] < match_iou:
                shift_pairs.append((iou_mat[gi][pi], predictions[pi].confidence, gi, pi))
    shift_pairs.sort(reverse=True)
    for iou_val, conf_val, gi, pi in shift_pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        supported_gt.add(gi)
        if conf_val >= missing_conf:
            p = predictions[pi]
            g = gt_boxes[gi]
            issues.append(
                AuditIssue(
                    kind=AuditIssueKind.MISALIGNED,
                    score=float(conf_val * max(0.35, 1.0 - iou_val) * 0.85),
                    gt_index=gi,
                    gt_label=g.label,
                    pred_label=p.label,
                    pred_conf=conf_val,
                    pred_box=pred_xyxy[pi],
                    iou=iou_val,
                    summary=f"疑似偏移：{g.label} (IoU {iou_val:.2f}, 置信 {conf_val:.2f})",
                )
            )

    # 第 4 轮：剩余高置信预测框且与全部人工框均无高重合 -> 疑似漏标
    for pi in range(n_pred):
        if pi in matched_pred:
            continue
        p = predictions[pi]
        if p.confidence < missing_conf:
            continue
        max_gt_iou = max((iou_mat[gi][pi] for gi in range(n_gt)), default=0.0)
        if max_gt_iou >= match_iou:
            continue
        issues.append(
            AuditIssue(
                kind=AuditIssueKind.MISSING,
                score=float(p.confidence),
                gt_index=None,
                gt_label="",
                pred_label=p.label,
                pred_conf=float(p.confidence),
                pred_box=pred_xyxy[pi],
                iou=max_gt_iou,
                summary=f"疑似漏标：{p.label} ({p.confidence:.2f})",
            )
        )

    # 第 5 轮：剩余人工框完全缺乏模型低阈值支持（且属于模型已知类别）-> 疑似误标
    for gi in range(n_gt):
        if gi in supported_gt:
            continue
        if known_norm is not None and gt_norm[gi] not in known_norm:
            continue
        max_pred_iou = max((iou_mat[gi][pi] for pi in range(n_pred)), default=0.0)
        if max_pred_iou >= low_iou:
            continue
        g = gt_boxes[gi]
        issues.append(
            AuditIssue(
                kind=AuditIssueKind.SPURIOUS,
                score=0.65,
                gt_index=gi,
                gt_label=g.label,
                pred_label="",
                pred_conf=0.0,
                pred_box=None,
                iou=max_pred_iou,
                summary=f"疑似误标：{g.label}（模型未检出支持）",
            )
        )

    issues.sort(key=lambda item: item.score, reverse=True)
    if issues:
        anomaly_score = issues[0].score + 0.15 * sum(item.score for item in issues[1:])
    else:
        anomaly_score = 0.0

    return ImageAuditReport(
        image_path=image_path.resolve(),
        has_json=has_json,
        gt_count=n_gt,
        pred_count=n_pred,
        issues=issues,
        anomaly_score=round(float(anomaly_score), 4),
    )


def _read_bgr_unicode_safe(image_path: Path) -> np.ndarray | None:
    """兼容 Windows 中文路径读取 BGR 图像数组，失败时返回 `None`。"""
    try:
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def audit_annotations(
    config: LabelAuditConfig,
    *,
    predictor_override: Callable[[Path], tuple[list[PredictedBox], list[str]]] | None = None,
    progress_callback: Callable[[int, int, Path], None] | None = None,
    cancel_event: Event | None = None,
) -> LabelAuditResult:
    """批量运行本地 YOLO 模型反查标注目录中的疑似漏标、错类、误标与偏移框。

    输入与失败行为：
        - 若未提供 `predictor_override`，要求 `config.weights_path` 指向存在的 `.pt` 文件，
          缺失时抛出 `FileNotFoundError`。
        - 收到 `cancel_event.set()` 时安全中止后续图片扫描，并保留已完成图片的诊断报告。
    """
    target_images: list[Path] = []
    skipped_unannotated = 0
    for raw_path in config.images:
        resolved = raw_path.resolve()
        if not resolved.is_file():
            continue
        has_json = resolved.with_suffix(".json").is_file()
        if config.only_annotated and not has_json:
            skipped_unannotated += 1
            continue
        target_images.append(resolved)

    model = None
    model_classes: list[str] = []
    if predictor_override is None:
        if not config.weights_path or not config.weights_path.is_file():
            raise FileNotFoundError(f"反查模型权重文件不存在: {config.weights_path}")
        from ultralytics import YOLO

        from yolo_studio.core.trainer import ensure_torch_cpu_threads

        ensure_torch_cpu_threads()
        model = YOLO(str(config.weights_path))
        if isinstance(model.names, dict):
            model_classes = [str(v) for _, v in sorted(model.names.items())]
        elif isinstance(model.names, (list, tuple)):
            model_classes = [str(v) for v in model.names]

    effective_pred_conf = max(0.01, min(float(config.pred_conf), float(config.missing_conf)))
    reports: list[ImageAuditReport] = []
    report_map: dict[Path, ImageAuditReport] = {}
    cancelled = False
    total = len(target_images)

    for idx, img_path in enumerate(target_images, start=1):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        if progress_callback is not None:
            progress_callback(idx, total, img_path)

        doc = load_image_annotation(img_path)
        preds: list[PredictedBox] = []
        if predictor_override is not None:
            preds, override_classes = predictor_override(img_path)
            if override_classes and not model_classes:
                model_classes = list(override_classes)
        else:
            img_bgr = _read_bgr_unicode_safe(img_path)
            if img_bgr is not None and model is not None:
                results = model.predict(
                    source=img_bgr,
                    conf=effective_pred_conf,
                    iou=0.60,
                    imgsz=max(160, int(config.imgsz)),
                    device=str(config.device),
                    verbose=False,
                )
                if results and results[0].boxes is not None:
                    r_boxes = results[0].boxes
                    xyxy_list = r_boxes.xyxy.cpu().tolist()
                    cls_list = r_boxes.cls.cpu().tolist()
                    conf_list = r_boxes.conf.cpu().tolist()
                    for xyxy, cls_id, conf_val in zip(xyxy_list, cls_list, conf_list):
                        cid = int(cls_id)
                        if isinstance(model.names, dict):
                            lbl = str(model.names.get(cid, f"class_{cid}"))
                        elif isinstance(model.names, (list, tuple)) and 0 <= cid < len(model.names):
                            lbl = str(model.names[cid])
                        else:
                            lbl = f"class_{cid}"
                        preds.append(
                            PredictedBox(
                                label=lbl,
                                x1=float(xyxy[0]),
                                y1=float(xyxy[1]),
                                x2=float(xyxy[2]),
                                y2=float(xyxy[3]),
                                confidence=float(conf_val),
                            )
                        )

        known_set = set(model_classes) if model_classes else None
        report = compare_image_predictions(
            img_path,
            doc.boxes,
            preds,
            has_json=doc.has_json_file,
            known_classes=known_set,
            missing_conf=float(config.missing_conf),
            match_iou=float(config.match_iou),
            low_iou=float(config.low_iou),
        )
        reports.append(report)
        report_map[img_path] = report

    reports.sort(key=lambda r: (-r.anomaly_score, str(r.image_path)))
    return LabelAuditResult(
        weights_path=config.weights_path,
        model_classes=model_classes,
        scanned_images=len(reports),
        skipped_unannotated=skipped_unannotated,
        cancelled=cancelled,
        reports=reports,
        report_map=report_map,
    )
