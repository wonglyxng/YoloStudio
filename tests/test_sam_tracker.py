"""SAM 2.1 连续帧向后追踪传播引擎单元测试。

职责：
    离线验证 `core.sam_tracker` 的三种已有标注冲突策略（追加不重叠、跳过已有标注、覆盖矩形框并保留非矩形形状）、
    面积门禁防切镜头全图爆框、掩码连通域噪点过滤、非 SAM 2 权重提示以及中途取消安全收尾。

运行环境：
    通过 `pytest` 离线运行，不加载真实权重或访问外部网络。
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image

from yolo_studio.core.annotation_io import RectAnnotation, load_image_annotation
from yolo_studio.core.sam_tracker import (
    SamTrackConfig,
    TrackConflictPolicy,
    _mask_to_tight_xyxy,
    is_sam2_compatible_weight,
    propagate_boxes_with_sam2,
)


def _write_frame(path: Path, size: tuple[int, int] = (400, 300)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(40, 60, 80)).save(path)


def test_sam_tracker_mask_cleanup_and_weight_compatibility() -> None:
    """验证追踪掩码连通域去噪与 SAM 2 权重兼容性校验。"""
    mask = np.zeros((200, 300), dtype=bool)
    mask[50:120, 80:160] = True  # 主目标块 70x80=5600 px
    mask[2:6, 2:6] = True  # 远处 16px 噪点应过滤
    assert _mask_to_tight_xyxy(mask, 300, 200) == (80.0, 50.0, 160.0, 120.0)
    assert _mask_to_tight_xyxy(np.zeros((200, 300), dtype=bool), 300, 200) is None

    assert is_sam2_compatible_weight("sam2.1_t.pt") is True
    assert is_sam2_compatible_weight("weights/sam2_b.pt") is True
    assert is_sam2_compatible_weight("mobile_sam.pt") is False
    assert is_sam2_compatible_weight("sam_b.pt") is False


def test_propagate_boxes_conflict_policies_area_gate_and_cancel(tmp_path: Path) -> None:
    """验证追加不重叠/跳过已标/覆盖模式、爆框门禁过滤、保留非矩形形状与取消中止。"""
    f0 = tmp_path / "f0.jpg"
    f1 = tmp_path / "f1.jpg"
    f2 = tmp_path / "f2.jpg"
    f3 = tmp_path / "f3.jpg"
    for p in (f0, f1, f2, f3):
        _write_frame(p, (400, 300))

    # f1 预先含 1 个与追踪框高度重合的 player 框 + 1 个非矩形多边形（覆盖模式也应保留非矩形）
    f1.with_suffix(".json").write_text(
        json.dumps(
            {
                "version": "2.5.4",
                "flags": {},
                "shapes": [
                    {
                        "label": "player",
                        "points": [[52.0, 42.0], [132.0, 122.0]],
                        "group_id": None,
                        "description": "manual",
                        "shape_type": "rectangle",
                        "flags": {},
                    },
                    {
                        "label": "zone",
                        "points": [[10.0, 10.0], [30.0, 10.0], [20.0, 30.0]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {},
                    },
                ],
                "imagePath": f1.name,
                "imageHeight": 300,
                "imageWidth": 400,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    seeds = [
        RectAnnotation(label="player", x1=50.0, y1=40.0, x2=130.0, y2=120.0),
        RectAnnotation(label="monster", x1=220.0, y1=60.0, x2=300.0, y2=140.0),
    ]

    # f1: player 与已有高度重合(去重跳过), monster 为新框(追加)
    # f2: player 正常移动, monster 发生异常全图爆框 (0,0,398,298) 应被面积门禁拦截
    # f3: 两目标均出画返回 None -> empty_lost
    scripted = [
        [(52.0, 42.0, 132.0, 122.0), (225.0, 62.0, 305.0, 142.0)],
        [(60.0, 45.0, 140.0, 125.0), (0.0, 0.0, 398.0, 298.0)],
        [None, None],
    ]

    # 1) APPEND_NON_OVERLAP 模式
    res_append = propagate_boxes_with_sam2(
        SamTrackConfig(
            start_image=f0,
            target_images=[f1, f2, f3],
            seed_boxes=seeds,
            conflict_policy=TrackConflictPolicy.APPEND_NON_OVERLAP,
        ),
        tracker_override=lambda _s, _t, _b: scripted,
    )
    assert res_append.processed_frames == 3
    assert res_append.written_frames == 2
    assert res_append.lost_frames == 1
    assert res_append.total_added_boxes == 2  # f1 新增 monster + f2 新增 player
    doc1 = load_image_annotation(f1)
    assert [b.label for b in doc1.boxes] == ["player", "monster"]
    assert doc1.boxes[1].flags.get("sam_tracked") is True
    assert len(doc1.preserved_shapes) == 1  # 多边形无损保留
    doc2 = load_image_annotation(f2)
    assert [b.label for b in doc2.boxes] == ["player"]  # 爆框 monster 被安全门禁过滤
    assert not f3.with_suffix(".json").exists()

    # 2) SKIP_LABELED 模式：f1 已有框跳过写入
    res_skip = propagate_boxes_with_sam2(
        SamTrackConfig(
            start_image=f0,
            target_images=[f1],
            seed_boxes=seeds,
            conflict_policy=TrackConflictPolicy.SKIP_LABELED,
        ),
        tracker_override=lambda _s, _t, _b: [[(60.0, 40.0, 140.0, 120.0), (240.0, 70.0, 320.0, 150.0)]],
    )
    assert res_skip.skipped_frames == 1
    assert res_skip.written_frames == 0

    # 3) OVERWRITE 模式：替换全部矩形框但仍无损保留多边形
    res_over = propagate_boxes_with_sam2(
        SamTrackConfig(
            start_image=f0,
            target_images=[f1],
            seed_boxes=[seeds[1]],
            conflict_policy=TrackConflictPolicy.OVERWRITE,
        ),
        tracker_override=lambda _s, _t, _b: [[(230.0, 70.0, 310.0, 150.0)]],
    )
    assert res_over.written_frames == 1
    doc1_over = load_image_annotation(f1)
    assert [b.label for b in doc1_over.boxes] == ["monster"]
    assert len(doc1_over.preserved_shapes) == 1

    # 4) 取消事件中止后续帧，且非 SAM 2 权重在真实路径下抛出明确 ValueError
    cancel_evt = Event()
    res_cancel = propagate_boxes_with_sam2(
        SamTrackConfig(start_image=f0, target_images=[f2, f3], seed_boxes=seeds),
        tracker_override=lambda _s, _t, _b: scripted[1:],
        progress_callback=lambda cur, _tot, _p, _st: cancel_evt.set() if cur >= 1 else None,
        cancel_event=cancel_evt,
    )
    assert res_cancel.cancelled is True
    assert res_cancel.processed_frames == 1

    with pytest.raises(ValueError, match="SAM 2"):
        propagate_boxes_with_sam2(
            SamTrackConfig(
                start_image=f0,
                target_images=[f1],
                seed_boxes=seeds,
                model_name="mobile_sam.pt",
            )
        )
