"""`core.label_batch_ops` 目录级类别批量重命名/合并的单元测试。

覆盖使用情况统计、改名与合并、非矩形形状与扩展字段无损保留、备份目录与清单、
损坏文件计入失败清单，以及新旧名非法时不做任何改写。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from yolo_studio.core.label_batch_ops import (
    make_label_backup_dir,
    rename_label_in_json_files,
    scan_label_usage,
)


def _write_json(path: Path, shapes: list[dict], **extra: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": "2.5.4", "flags": {"k": 1}, "shapes": shapes, "imagePath": path.with_suffix(".jpg").name}
    payload.update(extra)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _rect(label: str, **extra: object) -> dict:
    shape = {"label": label, "points": [[1, 1], [10, 10]], "shape_type": "rectangle", "flags": {}}
    shape.update(extra)
    return shape


def test_scan_rename_merge_backup_and_failures(tmp_path: Path) -> None:
    """验证统计、跨文件改名（含子目录与多边形）、扩展字段保留、备份清单与损坏文件处理。"""
    root = tmp_path / "frames"
    a = _write_json(root / "a.json", [_rect("hero", score=0.9, custom_field="keep"), _rect("monster")])
    b = _write_json(
        root / "sub" / "b.json",
        [
            _rect("hero"),
            {"label": "hero", "points": [[0, 0], [5, 0], [3, 4]], "shape_type": "polygon"},
        ],
        extraTopKey="keep-me",
    )
    c = _write_json(root / "c.json", [_rect("monster")])
    broken = root / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    missing = root / "missing.json"

    usage = scan_label_usage([a, b, c, broken, missing], " hero ")
    assert usage.files == [a, b]
    assert usage.shape_count == 3

    backup = make_label_backup_dir(tmp_path / "ws", now=datetime(2026, 11, 29, 0, 10, 0))
    assert backup.name == "label_rename_20261129_001000"
    res = rename_label_in_json_files(
        [a, b, c, broken, missing], "hero", "player", backup_dir=backup, base_dir=root
    )
    assert sorted(p.name for p in res.changed_files) == ["a.json", "b.json"]
    assert res.changed_shapes == 3
    assert [p.name for p, _why in res.failed] == ["broken.json"]
    assert res.backup_dir == backup

    data_a = json.loads(a.read_text(encoding="utf-8"))
    assert [s["label"] for s in data_a["shapes"]] == ["player", "monster"]
    assert data_a["shapes"][0]["custom_field"] == "keep"
    assert data_a["flags"] == {"k": 1}
    data_b = json.loads(b.read_text(encoding="utf-8"))
    assert [s["label"] for s in data_b["shapes"]] == ["player", "player"]
    assert data_b["extraTopKey"] == "keep-me"
    # 未含旧类别的文件不被备份或改写
    assert json.loads(c.read_text(encoding="utf-8"))["shapes"][0]["label"] == "monster"

    # 备份保留子目录结构且内容为改写前原文
    assert json.loads((backup / "a.json").read_text(encoding="utf-8"))["shapes"][0]["label"] == "hero"
    assert (backup / "sub" / "b.json").is_file()
    assert not (backup / "c.json").exists()
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["old_label"] == "hero" and manifest["new_label"] == "player"
    assert len(manifest["files"]) == 2

    # 合并：monster 并入已存在的 player
    res_merge = rename_label_in_json_files([a, b, c], "monster", "player", backup_dir=None)
    assert res_merge.changed_shapes == 2 and res_merge.backup_dir is None
    assert scan_label_usage([a, b, c], "monster").shape_count == 0
    assert scan_label_usage([a, b, c], "player").shape_count == 5

    # 同一秒再次生成备份目录时追加序号，不覆盖已有备份
    again = make_label_backup_dir(tmp_path / "ws", now=datetime(2026, 11, 29, 0, 10, 0))
    assert again.name == "label_rename_20261129_001000_2"

    # 新旧名为空或相同时不做任何改写
    assert rename_label_in_json_files([a], "player", "player", backup_dir=None).changed_files == []
    assert rename_label_in_json_files([a], "player", "  ", backup_dir=None).changed_files == []
