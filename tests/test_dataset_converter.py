"""X-AnyLabeling（`.json` 与 YOLO `.txt`）数据集转换引擎单元测试。

职责：
    验证 `scan_dataset` 与 `build_yolo_dataset` 对自动格式识别、归一化坐标换算、
    `classes.txt` 读取及 `train`/`val` 划分的正确性；并覆盖重复构建清理旧划分、
    显式负样本保留、损坏标注跳过、递归排除子目录、同名图片不跨目录误配标注
    以及输出目录位置校验等回归场景；同时覆盖按划分统计类别框数与已有划分目录探测。

运行环境与协作边界：
    通过 `pytest` 在临时目录（`tmp_path`）中运行，依赖 `Pillow` 与 `PyYAML`，
    直接测试 `yolo_studio.core.dataset_converter`；不启动 GUI 或训练引擎。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from yolo_studio.core.dataset_converter import (
    AnnotationFormat,
    DatasetBuildConfig,
    build_yolo_dataset,
    existing_split_dirs,
    scan_dataset,
)


def _write_dummy_image(path: Path, size: tuple[int, int] = (200, 100)) -> None:
    """在指定路径生成纯色测试图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(40, 120, 200)).save(path)


def test_convert_xanylabeling_json_auto(tmp_path: Path) -> None:
    """验证自动识别 X-AnyLabeling `.json` 标注、过滤未标注图片及归一化 YOLO 坐标换算。"""
    raw_dir = tmp_path / "raw_frames"
    for idx in range(5):
        img_path = raw_dir / f"frame_{idx:03d}.jpg"
        _write_dummy_image(img_path, (200, 100))
        if idx < 4:
            payload = {
                "version": "2.4.0",
                "shapes": [
                    {
                        "label": "person" if idx % 2 == 0 else "helmet",
                        "points": [[20.0, 10.0], [120.0, 60.0]],
                        "shape_type": "rectangle",
                    }
                ],
                "imagePath": img_path.name,
                "imageHeight": 100,
                "imageWidth": 200,
            }
            img_path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "yolo_ds"
    cfg = DatasetBuildConfig(
        images_dir=raw_dir,
        output_dir=out_dir,
        annotation_format=AnnotationFormat.AUTO,
        train_ratio=0.75,
        val_ratio=0.25,
        include_unannotated=False,
        seed=123,
    )

    summary = scan_dataset(cfg)
    assert summary.detected_format == "json"
    assert summary.total_images == 5
    assert summary.annotated_images == 4
    assert summary.unannotated_images == 1
    assert set(summary.classes) == {"person", "helmet"}

    res = build_yolo_dataset(cfg)
    assert res.train_count == 3
    assert res.val_count == 1
    assert res.total_boxes == 4
    assert res.data_yaml_path.is_file()

    yaml_data = yaml.safe_load(res.data_yaml_path.read_text(encoding="utf-8"))
    assert yaml_data["nc"] == 2
    assert set(yaml_data["names"].values()) == {"person", "helmet"}

    label_files = list((out_dir / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 3
    first_line = label_files[0].read_text(encoding="utf-8").strip().split()
    assert len(first_line) == 5
    assert abs(float(first_line[1]) - 0.35) < 1e-4  # 中心点 X 归一化值：(20+120)/(2*200) = 0.35
    assert abs(float(first_line[2]) - 0.35) < 1e-4  # 中心点 Y 归一化值：(10+60)/(2*100) = 0.35
    assert abs(float(first_line[3]) - 0.50) < 1e-4  # 宽度归一化值：100/200 = 0.50
    assert abs(float(first_line[4]) - 0.50) < 1e-4  # 高度归一化值：50/100 = 0.50


def test_convert_yolo_txt_with_classes(tmp_path: Path) -> None:
    """验证独立标注目录中 YOLO `.txt` 与 `classes.txt` 的映射及数据集构建。"""
    img_dir = tmp_path / "images"
    ann_dir = tmp_path / "labels_raw"
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / "classes.txt").write_text("car\nbus\n", encoding="utf-8")

    for idx in range(4):
        _write_dummy_image(img_dir / f"cam_{idx}.jpg", (320, 240))
        (ann_dir / f"cam_{idx}.txt").write_text("0 0.5 0.5 0.25 0.25\n1 0.2 0.2 0.1 0.1\n", encoding="utf-8")

    out_dir = tmp_path / "ds_txt"
    cfg = DatasetBuildConfig(
        images_dir=img_dir,
        annotations_dir=ann_dir,
        output_dir=out_dir,
        annotation_format=AnnotationFormat.AUTO,
        train_ratio=0.5,
        val_ratio=0.5,
    )
    res = build_yolo_dataset(cfg)
    assert res.classes == ["car", "bus"]
    assert res.train_count == 2
    assert res.val_count == 2
    assert res.total_boxes == 8


def _write_json_ann(img_path: Path, shapes: list[dict], size: tuple[int, int] = (200, 100)) -> None:
    """在图片同目录写出 X-AnyLabeling `.json` 标注。"""
    payload = {
        "version": "2.4.4",
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageWidth": size[0],
        "imageHeight": size[1],
    }
    img_path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")


def _rect(label: str) -> dict:
    return {"label": label, "points": [[20.0, 10.0], [120.0, 60.0]], "shape_type": "rectangle"}


def test_rebuild_clears_previous_splits(tmp_path: Path) -> None:
    """验证对同一输出目录重复构建时不残留旧划分文件，同一图片不会同时出现在 train 与 val。"""
    raw_dir = tmp_path / "raw"
    for idx in range(10):
        img = raw_dir / f"f_{idx:02d}.jpg"
        _write_dummy_image(img)
        _write_json_ann(img, [_rect("person")])

    out_dir = tmp_path / "ds"
    (out_dir / "labels").mkdir(parents=True)
    (out_dir / "labels" / "train.cache").write_text("stale", encoding="utf-8")
    for seed in (1, 2):
        build_yolo_dataset(
            DatasetBuildConfig(images_dir=raw_dir, output_dir=out_dir, train_ratio=0.7, val_ratio=0.3, seed=seed)
        )

    train = {p.stem for p in (out_dir / "images" / "train").iterdir()}
    val = {p.stem for p in (out_dir / "images" / "val").iterdir()}
    assert len(train) + len(val) == 10
    assert not (train & val)
    assert len(list((out_dir / "labels" / "train").iterdir())) == len(train)
    assert not (out_dir / "labels" / "train.cache").exists()


def test_explicit_negative_kept_and_corrupt_json_skipped(tmp_path: Path) -> None:
    """验证空 shapes 的显式负样本默认纳入、损坏 JSON 仅告警跳过、无标注文件的图片默认排除。"""
    raw_dir = tmp_path / "raw"
    for idx in range(3):
        img = raw_dir / f"pos_{idx}.jpg"
        _write_dummy_image(img)
        _write_json_ann(img, [_rect("car")])
    neg = raw_dir / "neg.jpg"
    _write_dummy_image(neg)
    _write_json_ann(neg, [])
    bad = raw_dir / "bad.jpg"
    _write_dummy_image(bad)
    bad.with_suffix(".json").write_text("{not json", encoding="utf-8")
    _write_dummy_image(raw_dir / "nolabel.jpg")

    cfg = DatasetBuildConfig(images_dir=raw_dir, output_dir=tmp_path / "ds", train_ratio=0.5, val_ratio=0.5)
    summary = scan_dataset(cfg)
    assert summary.annotated_images == 3
    assert summary.negative_images == 1
    assert any("bad.json" in w for w in summary.warnings)

    res = build_yolo_dataset(cfg)
    assert res.train_count + res.val_count == 4  # 3 个正样本 + 1 个显式负样本
    neg_labels = [p for p in (tmp_path / "ds" / "labels").rglob("neg.txt")]
    assert len(neg_labels) == 1 and neg_labels[0].read_text(encoding="utf-8") == ""


def test_recursive_exclude_and_no_cross_dir_label_mismatch(tmp_path: Path) -> None:
    """验证递归扫描可排除 need_review，且同名图片不会回退匹配到平铺目录中别的图片的标注。"""
    root = tmp_path / "reviewed"
    a = root / "auto_passed" / "x.jpg"
    _write_dummy_image(a)
    _write_json_ann(a, [_rect("person")])
    n = root / "need_review" / "y.jpg"
    _write_dummy_image(n)
    _write_json_ann(n, [_rect("person")])
    # 两个子目录中的同名图片 dup.jpg：仅 sub1 下有自己的标注，根目录有一份平铺的 dup.json
    s1 = root / "sub1" / "dup.jpg"
    s2 = root / "sub2" / "dup.jpg"
    _write_dummy_image(s1)
    _write_dummy_image(s2)
    _write_json_ann(s1, [_rect("person")])
    (root / "dup.json").write_text(
        json.dumps({"shapes": [_rect("car")], "imageWidth": 200, "imageHeight": 100}), encoding="utf-8"
    )

    summary = scan_dataset(
        DatasetBuildConfig(images_dir=root, recursive=True, exclude_subdirs=("need_review",))
    )
    names = {s.image_path.relative_to(root).as_posix(): s for s in summary.samples}
    assert "need_review/y.jpg" not in names
    assert summary.excluded_images == 1
    assert names["sub2/dup.jpg"].source_label_path is None
    assert "car" not in summary.class_counts


def test_build_rejects_output_inside_recursive_images_dir(tmp_path: Path) -> None:
    """验证递归扫描时输出目录位于图片目录内部会被拒绝，避免把旧数据集重新当作输入。"""
    raw_dir = tmp_path / "raw"
    img = raw_dir / "a.jpg"
    _write_dummy_image(img)
    _write_json_ann(img, [_rect("person")])
    try:
        build_yolo_dataset(
            DatasetBuildConfig(images_dir=raw_dir, output_dir=raw_dir / "ds", recursive=True)
        )
    except ValueError as exc:
        assert "内部" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应拒绝位于图片目录内部的输出目录")


def test_build_reports_split_class_counts_and_existing_split_dirs(tmp_path: Path) -> None:
    """验证生成结果按划分统计类别框数（与写出的标签一致），以及 `existing_split_dirs` 只列出非空划分目录。"""
    raw_dir = tmp_path / "raw"
    for idx in range(10):
        img = raw_dir / f"f_{idx:02d}.jpg"
        _write_dummy_image(img)
        _write_json_ann(img, [_rect("person"), _rect("car" if idx % 2 else "person")])

    out_dir = tmp_path / "ds"
    assert existing_split_dirs(out_dir) == []
    (out_dir / "images" / "train").mkdir(parents=True)  # 空目录不算可覆盖内容
    assert existing_split_dirs(out_dir) == []

    res = build_yolo_dataset(
        DatasetBuildConfig(images_dir=raw_dir, output_dir=out_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    )
    assert set(res.split_class_counts) == {"train", "val", "test"}
    for name, total in res.class_counts.items():
        assert sum(res.split_class_counts[s].get(name, 0) for s in ("train", "val", "test")) == total
    val_boxes = sum(
        len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
        for p in (out_dir / "labels" / "val").glob("*.txt")
    )
    assert val_boxes == sum(res.split_class_counts["val"].values())

    found = {p.relative_to(out_dir).as_posix() for p in existing_split_dirs(out_dir)}
    assert found == {f"{k}/{s}" for k in ("images", "labels") for s in ("train", "val", "test")}
