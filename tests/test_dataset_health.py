"""`core.dataset_health` 离线数据集标注健康体检引擎单元测试。

覆盖损坏 JSON、过小框、极端长宽比、同图同类高度重合重复框、未登记类别、零样本类别、
人工待确认问题/AI 漏标提醒统计以及类别不平衡比计算。
"""

from __future__ import annotations

import json
from pathlib import Path

from yolo_studio.core.dataset_health import HealthIssueKind, analyze_dataset_health


def _touch_img(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"img")
    return path


def _write_ann(img: Path, shapes: list[dict], *, flags: dict | None = None) -> None:
    payload = {
        "version": "2.5.4",
        "flags": flags or {},
        "shapes": shapes,
        "imagePath": img.name,
        "imageHeight": 300,
        "imageWidth": 400,
    }
    img.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _rect(label: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"label": label, "points": [[x1, y1], [x2, y2]], "shape_type": "rectangle", "flags": {}}


def test_analyze_dataset_health_detects_all_issue_kinds_and_imbalance(tmp_path: Path) -> None:
    """验证健康体检能够准确识别全部 6 类异常、零样本类别与类别不平衡比。"""
    imgs = [_touch_img(tmp_path / f"f{i}.jpg") for i in range(5)]

    # f0: 2 个高度重合的 hero 框 + 1 个过小 monster 框 + 1 个极端长宽比 hero 框 + 1 个未登记 ghost 框
    _write_ann(
        imgs[0],
        [
            _rect("hero", 20, 20, 120, 120),
            _rect("hero", 22, 22, 120, 120),     # 与 #0 高度重合 (IoU > 0.9)
            _rect("monster", 200, 50, 206, 56),  # 6x6 < 10px 过小框
            _rect("hero", 10, 200, 190, 212),    # 180x12 比例 15:1 极端长宽比
            _rect("ghost", 300, 100, 360, 180),  # 未登记类别
        ],
    )
    # f1: 人工标记待确认问题 + 3 个 hero 框（拉开不平衡比）
    _write_ann(
        imgs[1],
        [
            _rect("hero", 10, 10, 50, 50),
            _rect("hero", 60, 10, 100, 50),
            _rect("hero", 110, 10, 150, 50),
        ],
        flags={"issue_flag": True, "issue_note": "边缘遮挡需复核"},
    )
    # f2: 显式空图负样本
    _write_ann(imgs[2], [])
    # f3: 损坏 JSON
    imgs[3].with_suffix(".json").write_text("{broken json", encoding="utf-8")
    # f4: 无 JSON（未标注）

    report = analyze_dataset_health(
        imgs,
        known_classes=["hero", "monster", "npc_unused"],
        min_box_px=10.0,
        max_aspect_ratio=8.0,
        duplicate_iou=0.85,
        low_count_threshold=5,
    )

    assert report.total_images == 5
    assert report.labeled_images == 2
    assert report.empty_images == 1
    assert report.unlabeled_images == 1
    assert report.corrupt_json_count == 1
    assert report.issue_flagged_images == 1
    assert report.total_boxes == 8
    assert report.zero_count_classes == ["npc_unused"]
    # hero=6, monster=1, ghost=1 -> 不平衡比 6.0
    assert report.imbalance_ratio == 6.0

    kinds = {it.kind for it in report.issues}
    assert kinds == {
        HealthIssueKind.CORRUPT_JSON,
        HealthIssueKind.IMAGE_ISSUE_FLAG,
        HealthIssueKind.DUPLICATE_BOX,
        HealthIssueKind.UNREGISTERED_LABEL,
        HealthIssueKind.TINY_BOX,
        HealthIssueKind.EXTREME_ASPECT,
    }
    stats_by_label = {s.label: s for s in report.class_stats}
    assert stats_by_label["hero"].is_underrepresented is False
    assert stats_by_label["monster"].is_underrepresented is True
    assert stats_by_label["ghost"].is_unregistered is True
