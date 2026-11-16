"""X-AnyLabeling（`.json` 与 YOLO `.txt`）数据集转换引擎单元测试。

职责：
    验证 `scan_dataset` 与 `build_yolo_dataset` 对自动格式识别、归一化坐标换算、
    `classes.txt` 读取及 `train`/`val` 划分的正确性。

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
