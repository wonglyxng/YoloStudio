"""基于已训练 YOLO 权重的标注质量反查引擎单元测试。

职责：
    离线验证 `core.label_auditor` 的四类异常匹配算法（疑似漏标、疑似错类、疑似误标、疑似偏移）、
    模型未知类别防误报保护、综合异常严重度排序、仅扫已标图过滤与随时取消收尾。

运行环境：
    通过 `pytest` 离线运行，不加载真实 `.pt` 权重也不访问网络。
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event

from PIL import Image

from yolo_studio.core.annotation_io import RectAnnotation
from yolo_studio.core.label_auditor import (
    AuditIssueKind,
    LabelAuditConfig,
    PredictedBox,
    audit_annotations,
    compare_image_predictions,
)


def _write_img_and_json(
    path: Path,
    boxes: list[tuple[str, float, float, float, float]] | None,
    size: tuple[int, int] = (400, 300),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(50, 70, 90)).save(path)
    if boxes is not None:
        payload = {
            "version": "2.5.4",
            "flags": {},
            "shapes": [
                {
                    "label": lbl,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                }
                for lbl, x1, y1, x2, y2 in boxes
            ],
            "imagePath": path.name,
            "imageHeight": size[1],
            "imageWidth": size[0],
        }
        path.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )


def test_compare_image_predictions_detects_four_issue_kinds_and_ignores_unknown_class(
    tmp_path: Path,
) -> None:
    """验证四类异常精准识别、正常框不误报，以及模型未训练过的新类不误报为疑似误标。"""
    img_p = tmp_path / "sample.jpg"
    gt_boxes = [
        # 0: 正常匹配框（player）
        RectAnnotation(label="player", x1=10.0, y1=10.0, x2=90.0, y2=90.0),
        # 1: 疑似错类（人工标 player，模型高置信预测 monster）
        RectAnnotation(label="player", x1=110.0, y1=10.0, x2=190.0, y2=90.0),
        # 2: 疑似框偏移（同为 monster，IoU 约 0.30 落在 [0.20, 0.50) 区间）
        RectAnnotation(label="monster", x1=210.0, y1=10.0, x2=290.0, y2=90.0),
        # 3: 疑似误标（属于模型已知类别 loot，但完全无模型预测支持）
        RectAnnotation(label="loot", x1=10.0, y1=150.0, x2=70.0, y2=210.0),
        # 4: 模型未知的新增类别 boss（不应被误报为疑似误标）
        RectAnnotation(label="boss", x1=110.0, y1=150.0, x2=190.0, y2=230.0),
    ]
    preds = [
        PredictedBox(label="player", x1=12.0, y1=12.0, x2=88.0, y2=88.0, confidence=0.92),
        PredictedBox(label="monster", x1=112.0, y1=12.0, x2=188.0, y2=88.0, confidence=0.89),
        PredictedBox(label="monster", x1=245.0, y1=10.0, x2=345.0, y2=90.0, confidence=0.82),
        # 疑似漏标（无任何人工框重合，高置信 0.91）
        PredictedBox(label="loot", x1=280.0, y1=160.0, x2=360.0, y2=240.0, confidence=0.91),
        # 低置信预测噪声（0.30 < 0.50 不应报漏标）
        PredictedBox(label="player", x1=320.0, y1=250.0, x2=380.0, y2=290.0, confidence=0.30),
    ]

    report = compare_image_predictions(
        img_p,
        gt_boxes,
        preds,
        has_json=True,
        known_classes={"player", "monster", "loot"},
        missing_conf=0.50,
        match_iou=0.50,
        low_iou=0.20,
    )

    assert report.has_issues is True
    assert report.missing_count == 1
    assert report.wrong_label_count == 1
    assert report.misaligned_count == 1
    assert report.spurious_count == 1
    assert report.anomaly_score > 0.90

    kinds_by_gt = {item.gt_index: item.kind for item in report.issues}
    assert kinds_by_gt[None] == AuditIssueKind.MISSING
    assert kinds_by_gt[1] == AuditIssueKind.WRONG_LABEL
    assert kinds_by_gt[2] == AuditIssueKind.MISALIGNED
    assert kinds_by_gt[3] == AuditIssueKind.SPURIOUS
    assert 0 not in kinds_by_gt  # 正常匹配框无报警
    assert 4 not in kinds_by_gt  # 模型未学过的新类 boss 不误报

    badge = report.compact_badge()
    assert "漏1" in badge and "错类1" in badge and "误标1" in badge and "偏移1" in badge
    assert "疑似漏标 1" in report.headline_summary()


def test_audit_annotations_filters_unannotated_sorts_and_cancels(tmp_path: Path) -> None:
    """验证批量反查跳过未标图开关、按异常严重度降序排列以及取消事件安全中止。"""
    img_clean = tmp_path / "a_clean.jpg"
    img_spurious = tmp_path / "b_spurious.jpg"
    img_severe = tmp_path / "c_severe.jpg"
    img_unlabeled = tmp_path / "d_unlabeled.jpg"

    _write_img_and_json(img_clean, [("player", 20.0, 20.0, 120.0, 120.0)])
    _write_img_and_json(img_spurious, [("player", 20.0, 20.0, 120.0, 120.0)])
    _write_img_and_json(img_severe, [("player", 20.0, 20.0, 120.0, 120.0)])
    _write_img_and_json(img_unlabeled, None)  # 无 .json

    preds_by_name = {
        img_clean.name: [PredictedBox("player", 22.0, 22.0, 118.0, 118.0, 0.95)],
        img_spurious.name: [],
        img_severe.name: [
            PredictedBox("monster", 22.0, 22.0, 118.0, 118.0, 0.90),
            PredictedBox("loot", 240.0, 140.0, 340.0, 240.0, 0.92),
        ],
        img_unlabeled.name: [PredictedBox("monster", 100.0, 80.0, 220.0, 200.0, 0.88)],
    }

    def _fake_pred(p: Path) -> tuple[list[PredictedBox], list[str]]:
        return preds_by_name[p.name], ["player", "monster", "loot"]

    # 1) 默认 only_annotated=True：跳过无 .json 的 img_unlabeled，且按异常严重度严格降序
    cfg = LabelAuditConfig(
        weights_path=tmp_path / "best.pt",
        images=[img_clean, img_spurious, img_severe, img_unlabeled],
        only_annotated=True,
    )
    res = audit_annotations(cfg, predictor_override=_fake_pred)
    assert res.scanned_images == 3
    assert res.skipped_unannotated == 1
    assert res.flagged_images == 2
    assert res.cancelled is False
    assert [r.image_path.name for r in res.reports] == [
        img_severe.name,
        img_spurious.name,
        img_clean.name,
    ]
    assert res.total_missing == 1
    assert res.total_wrong_label == 1
    assert res.total_spurious == 1

    # 2) only_annotated=False：一并扫描未标图并检出其中高置信漏标
    cfg_all = LabelAuditConfig(
        weights_path=tmp_path / "best.pt",
        images=[img_clean, img_unlabeled],
        only_annotated=False,
    )
    res_all = audit_annotations(cfg_all, predictor_override=_fake_pred)
    assert res_all.scanned_images == 2
    assert res_all.skipped_unannotated == 0
    assert res_all.report_map[img_unlabeled.resolve()].missing_count == 1

    # 3) 取消事件生效：扫完第 1 张即中止
    cancel_evt = Event()

    def _on_progress(cur: int, _total: int, _path: Path) -> None:
        if cur >= 1:
            cancel_evt.set()

    res_cancelled = audit_annotations(
        cfg,
        predictor_override=_fake_pred,
        progress_callback=_on_progress,
        cancel_event=cancel_evt,
    )
    assert res_cancelled.cancelled is True
    assert res_cancelled.scanned_images == 1
