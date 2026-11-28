"""基于 SAM 2 / SAM 2.1 视觉记忆库的连续帧目标追踪与跨帧标注传播引擎。

职责：
    面向标注工作台“已标好当前帧 → 向后连续追踪传播 N 帧”高频提效场景，直接复用本地
    `sam2.1_t.pt` / `sam2.1_s.pt` / `sam2.1_b.pt` / `sam2.1_l.pt` 权重与 Ultralytics
    `SAM2VideoPredictor` 记忆编码器，将起始帧中的一个或多个目标框（连同类别）沿图片序列
    向后逐帧传播，提取紧贴外接矩形并按冲突策略写回标准 X-AnyLabeling `.json`。

运行环境与协作边界：
    纯 Python 核心模块，不依赖 PyQt6；真实追踪时延迟导入 `torch` 与 `SAM2VideoPredictor`，
    采用 `np.fromfile + cv2.imdecode` 兼容 Windows 中文路径。支持注入 `tracker_override`
    进行零权重离线单元测试；支持通过 `cancel_event` 随时中止，结束时自动释放记忆库显存。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Any, Callable

import cv2
import numpy as np

from yolo_studio.core.annotation_io import (
    RectAnnotation,
    load_image_annotation,
    save_image_annotation,
)


class TrackConflictPolicy(str, Enum):
    """连续帧追踪写回目标帧 `.json` 时遇到已有标注框的合并策略。"""

    APPEND_NON_OVERLAP = "append"
    SKIP_LABELED = "skip"
    OVERWRITE = "overwrite"


@dataclass
class SamTrackConfig:
    """SAM 2.1 连续帧向后追踪任务配置。"""

    start_image: Path
    target_images: list[Path]
    seed_boxes: list[RectAnnotation]
    model_name: str = "sam2.1_t.pt"
    device: str = "0"
    conflict_policy: TrackConflictPolicy = TrackConflictPolicy.APPEND_NON_OVERLAP
    min_box_size_px: float = 4.0
    min_area_ratio: float = 0.08
    max_area_ratio: float = 6.0
    dedup_iou: float = 0.50
    mark_as_tracked: bool = True


@dataclass
class FrameTrackOutcome:
    """单个目标帧的追踪传播落盘明细。"""

    image_path: Path
    status: str  # "written" | "skipped_labeled" | "empty_lost" | "read_error"
    tracked_count: int = 0
    added_count: int = 0
    skipped_dup_count: int = 0


@dataclass
class SamTrackResult:
    """整批连续帧 SAM 2.1 向后追踪汇总结果。"""

    start_image: Path
    model_source: str
    seed_count: int
    processed_frames: int
    written_frames: int
    skipped_frames: int
    lost_frames: int
    total_added_boxes: int
    cancelled: bool
    outcomes: list[FrameTrackOutcome] = field(default_factory=list)


class _FrameSequenceDataset:
    """为 `SAM2VideoPredictor` 提供逐帧图片序列所需的最小只读状态协议。"""

    def __init__(self, num_frames: int) -> None:
        self.mode = "video"
        self.frames = int(num_frames)
        self.nf = int(num_frames)
        self.frame = 0


def _read_bgr_unicode_safe(image_path: Path) -> np.ndarray | None:
    """兼容 Windows 中文路径读取 BGR 图像数组，读取失败返回 `None`。"""
    try:
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _box_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
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


def _mask_to_tight_xyxy(
    binary_mask: np.ndarray,
    img_w: int,
    img_h: int,
    *,
    min_size_px: float = 4.0,
) -> tuple[float, float, float, float] | None:
    """从 SAM 2 追踪输出的单目标二值掩码提取最大有效连通域的外接矩形，过滤零散噪点。"""
    if binary_mask.ndim != 2 or not np.any(binary_mask):
        return None
    mh, mw = binary_mask.shape[:2]
    mask = binary_mask
    if mh != img_h or mw != img_w:
        mask = cv2.resize(
            binary_mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST
        ) > 0.5

    # 若存在多个离散碎块，保留面积占主连通域 >= 15% 的块，过滤远处孤立噪声像素
    num_labels, comp_map, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    filtered = mask
    if num_labels > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(areas.max()) if len(areas) > 0 else 0
        if max_area > 0:
            keep_ids = [
                cid + 1 for cid, area in enumerate(areas) if int(area) >= max(4, int(max_area * 0.15))
            ]
            if keep_ids:
                filtered = np.isin(comp_map, keep_ids)

    ys, xs = np.nonzero(filtered)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = max(0.0, float(int(xs.min())))
    y1 = max(0.0, float(int(ys.min())))
    x2 = min(float(img_w), float(int(xs.max()) + 1))
    y2 = min(float(img_h), float(int(ys.max()) + 1))
    if (x2 - x1) < min_size_px or (y2 - y1) < min_size_px:
        return None
    return (x1, y1, x2, y2)


def _passes_area_gate(
    xyxy: tuple[float, float, float, float],
    seed_area: float,
    img_w: int,
    img_h: int,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
) -> bool:
    """校验追踪框面积是否仍在合理变动范围内，防止切镜头时 SAM 2 发散框住整张背景。"""
    bw = max(0.0, xyxy[2] - xyxy[0])
    bh = max(0.0, xyxy[3] - xyxy[1])
    area = bw * bh
    if area <= 0.0:
        return False
    img_area = max(1.0, float(img_w * img_h))
    # 若原框原本就不是全图级大框，追踪结果突然膨胀覆盖超过 92% 画面视为异常发散
    if seed_area < img_area * 0.65 and area > img_area * 0.92:
        return False
    if seed_area > 1.0:
        ratio = area / seed_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            return False
    return True


def is_sam2_compatible_weight(model_name: str) -> bool:
    """判断权重名或文件名是否属于支持记忆追踪的 SAM 2 / SAM 2.1 权重。"""
    name = Path(model_name.strip()).name.lower()
    if not name:
        return True
    return "sam2" in name


def _apply_tracked_boxes_to_frame(
    img_path: Path,
    img_size: tuple[int, int] | None,
    seed_boxes: list[RectAnnotation],
    raw_coords: list[tuple[float, float, float, float] | None],
    config: SamTrackConfig,
) -> FrameTrackOutcome:
    """将单帧追踪坐标按面积门禁与冲突策略合并写回对应 `.json`，返回本帧明细。"""
    doc = load_image_annotation(img_path, image_size=img_size)
    img_w = doc.image_width or (img_size[0] if img_size else 0)
    img_h = doc.image_height or (img_size[1] if img_size else 0)

    valid_tracked: list[RectAnnotation] = []
    for seed, xyxy in zip(seed_boxes, raw_coords):
        if xyxy is None:
            continue
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        if img_w > 0 and img_h > 0:
            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))
        if (x2 - x1) < config.min_box_size_px or (y2 - y1) < config.min_box_size_px:
            continue
        seed_area = max(1.0, seed.width * seed.height)
        if img_w > 0 and img_h > 0 and not _passes_area_gate(
            (x1, y1, x2, y2),
            seed_area,
            img_w,
            img_h,
            min_area_ratio=config.min_area_ratio,
            max_area_ratio=config.max_area_ratio,
        ):
            continue
        flags: dict[str, Any] = {"box_refined": True}
        desc = ""
        if config.mark_as_tracked:
            flags["sam_tracked"] = True
            desc = "sam_track: SAM 2.1 连续帧追踪传播"
        valid_tracked.append(
            RectAnnotation(
                label=seed.label,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                description=desc,
                difficult=seed.difficult,
                group_id=seed.group_id,
                flags=flags,
            )
        )

    if config.conflict_policy == TrackConflictPolicy.SKIP_LABELED and doc.boxes:
        return FrameTrackOutcome(
            image_path=img_path,
            status="skipped_labeled",
            tracked_count=len(valid_tracked),
            added_count=0,
        )

    if not valid_tracked:
        return FrameTrackOutcome(
            image_path=img_path,
            status="empty_lost",
            tracked_count=0,
            added_count=0,
        )

    skipped_dup = 0
    if config.conflict_policy == TrackConflictPolicy.OVERWRITE:
        doc.boxes = valid_tracked
        added_count = len(valid_tracked)
    else:
        existing = list(doc.boxes)
        to_add: list[RectAnnotation] = []
        for cand in valid_tracked:
            cand_xy = (cand.x1, cand.y1, cand.x2, cand.y2)
            if any(
                ex.label == cand.label
                and _box_iou_xyxy((ex.x1, ex.y1, ex.x2, ex.y2), cand_xy) >= config.dedup_iou
                for ex in existing
            ):
                skipped_dup += 1
                continue
            to_add.append(cand)
            existing.append(cand)
        if not to_add:
            return FrameTrackOutcome(
                image_path=img_path,
                status="skipped_labeled",
                tracked_count=len(valid_tracked),
                added_count=0,
                skipped_dup_count=skipped_dup,
            )
        doc.boxes.extend(to_add)
        added_count = len(to_add)

    save_image_annotation(doc)
    return FrameTrackOutcome(
        image_path=img_path,
        status="written",
        tracked_count=len(valid_tracked),
        added_count=added_count,
        skipped_dup_count=skipped_dup,
    )


def propagate_boxes_with_sam2(
    config: SamTrackConfig,
    *,
    tracker_override: Callable[
        [Path, list[Path], list[RectAnnotation]],
        list[list[tuple[float, float, float, float] | None]],
    ]
    | None = None,
    progress_callback: Callable[[int, int, Path, str], None] | None = None,
    cancel_event: Event | None = None,
) -> SamTrackResult:
    """使用 SAM 2.1 记忆库将 `config.seed_boxes` 从 `start_image` 向后连续追踪到 `target_images`。

    约束与异常：
        - `seed_boxes` 或 `target_images` 为空时直接返回空汇总结果；
        - 真实推理要求权重属于 SAM 2 / SAM 2.1 系列（否则抛出 `ValueError` 给出清晰提示）；
        - 收到 `cancel_event.set()` 时立即停止后续帧推理并保留已写盘的帧结果，退出前清理显存。
    """
    start_resolved = config.start_image.resolve()
    targets = [p.resolve() for p in config.target_images if p.resolve().is_file()]
    seeds = [b.clone() for b in config.seed_boxes]
    if not seeds or not targets:
        return SamTrackResult(
            start_image=start_resolved,
            model_source=config.model_name,
            seed_count=len(seeds),
            processed_frames=0,
            written_frames=0,
            skipped_frames=0,
            lost_frames=0,
            total_added_boxes=0,
            cancelled=False,
        )

    outcomes: list[FrameTrackOutcome] = []
    cancelled = False
    total = len(targets)

    # 离线测试注入路径
    if tracker_override is not None:
        all_preds = tracker_override(start_resolved, targets, seeds)
        for idx, target_path in enumerate(targets, start=1):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            if progress_callback is not None:
                progress_callback(idx, total, target_path, "正在写入追踪结果")
            raw_coords = all_preds[idx - 1] if (idx - 1) < len(all_preds) else []
            out = _apply_tracked_boxes_to_frame(target_path, None, seeds, raw_coords, config)
            outcomes.append(out)
        return _summarize_track_result(start_resolved, config.model_name, len(seeds), outcomes, cancelled)

    # 真实 SAM2VideoPredictor 推理路径
    model_name = (config.model_name or "").strip() or "sam2.1_t.pt"
    if not is_sam2_compatible_weight(model_name):
        raise ValueError(
            f"当前全局本地 SAM 权重为「{Path(model_name).name}」，仅 SAM 2 / SAM 2.1 系列"
            "（如 sam2.1_t.pt / sam2.1_s.pt / sam2.1_b.pt）支持连续帧记忆追踪；"
            "请先在右上角「设置 → 本地视觉模型」中切换为 SAM 2.1 权重。"
        )

    import torch
    from ultralytics.models.sam import SAM2VideoPredictor
    from ultralytics.utils import ops

    from yolo_studio.core.trainer import ensure_torch_cpu_threads, resolve_pretrained_weights

    ensure_torch_cpu_threads()
    resolved_weight = resolve_pretrained_weights(model_name)

    start_bgr = _read_bgr_unicode_safe(start_resolved)
    if start_bgr is None:
        raise RuntimeError(f"无法读取起始帧像素：{start_resolved}")

    overrides: dict[str, Any] = {
        "model": str(resolved_weight),
        "conf": 0.25,
        "task": "segment",
        "mode": "predict",
        "imgsz": 1024,
        "device": str(config.device),
        "save": False,
        "verbose": False,
    }
    predictor = SAM2VideoPredictor(overrides=overrides)
    predictor.setup_model(model=None, verbose=False)
    predictor.imgsz = [1024, 1024]
    predictor.model.set_imgsz(predictor.imgsz)
    predictor._bb_feat_sizes = [
        [int(x / (predictor.stride * s)) for x in predictor.imgsz] for s in (0.25, 0.5, 1.0)
    ]
    predictor.dataset = _FrameSequenceDataset(total + 1)
    predictor.init_state(predictor)

    init_bboxes = [[float(b.x1), float(b.y1), float(b.x2), float(b.y2)] for b in seeds]
    try:
        with torch.inference_mode():
            # 第 0 帧：将起始帧目标框编码进 SAM 2 记忆库
            if progress_callback is not None:
                progress_callback(0, total, start_resolved, "正在编码起始帧目标记忆")
            predictor.dataset.frame = 0
            predictor.batch = ([str(start_resolved)], [start_bgr], [""])
            predictor.im = None
            predictor.features = None
            im0 = predictor.preprocess([start_bgr])
            predictor.inference(im0, bboxes=init_bboxes)

            # 第 1..N 帧：沿工作台帧序列逐帧向后传播
            for idx, target_path in enumerate(targets, start=1):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                if progress_callback is not None:
                    progress_callback(idx, total, target_path, f"正在追踪第 {idx}/{total} 帧")

                frame_bgr = _read_bgr_unicode_safe(target_path)
                if frame_bgr is None:
                    outcomes.append(FrameTrackOutcome(image_path=target_path, status="read_error"))
                    continue

                img_h, img_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
                predictor.dataset.frame = idx
                predictor.batch = ([str(target_path)], [frame_bgr], [""])
                predictor.im = None
                predictor.features = None
                im = predictor.preprocess([frame_bgr])
                pred_masks, _pred_scores = predictor.inference(im)

                dst_shape = (int(im.shape[2]), int(im.shape[3]))
                src_shape = (img_h, img_w)
                gain = min(i / o for i, o in zip(dst_shape, src_shape))
                ratio = [
                    round(o * gain) * m / i / o
                    for o, m, i in zip(src_shape, pred_masks.shape[1:], dst_shape)
                ]
                scaled = ops.scale_masks(
                    pred_masks[None].float(), src_shape, (ratio, (0, 0)), padding=False
                )[0]
                mask_threshold = float(getattr(predictor.model, "mask_threshold", 0.0))
                masks_bin = (scaled > mask_threshold).detach().cpu().numpy()

                frame_coords: list[tuple[float, float, float, float] | None] = []
                for obj_i in range(len(seeds)):
                    if obj_i < len(masks_bin):
                        frame_coords.append(
                            _mask_to_tight_xyxy(
                                masks_bin[obj_i],
                                img_w,
                                img_h,
                                min_size_px=config.min_box_size_px,
                            )
                        )
                    else:
                        frame_coords.append(None)

                outcome = _apply_tracked_boxes_to_frame(
                    target_path, (img_w, img_h), seeds, frame_coords, config
                )
                outcomes.append(outcome)
    finally:
        # 彻底释放记忆库显存引用，避免退出追踪后长时间占用 GPU
        try:
            predictor.inference_state = None
            predictor.reset_image()
        except Exception:
            pass
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return _summarize_track_result(start_resolved, str(resolved_weight), len(seeds), outcomes, cancelled)


def _summarize_track_result(
    start_image: Path,
    model_source: str,
    seed_count: int,
    outcomes: list[FrameTrackOutcome],
    cancelled: bool,
) -> SamTrackResult:
    return SamTrackResult(
        start_image=start_image,
        model_source=model_source,
        seed_count=seed_count,
        processed_frames=len(outcomes),
        written_frames=sum(1 for o in outcomes if o.status == "written"),
        skipped_frames=sum(1 for o in outcomes if o.status == "skipped_labeled"),
        lost_frames=sum(1 for o in outcomes if o.status == "empty_lost"),
        total_added_boxes=sum(o.added_count for o in outcomes),
        cancelled=cancelled,
        outcomes=outcomes,
    )
