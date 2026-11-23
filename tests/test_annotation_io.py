"""原生标注工作台 `.json` 标注数据读写与状态摘要单元测试。

职责：
    验证 `core.annotation_io` 对标准 LabelMe / X-AnyLabeling 兼容 `.json` 文件的读写往返、
    坐标规范化与边界裁剪、空图负样本保存、历史非矩形 shape 无损保留、AI 审核元信息透传、
    损坏文件容错、`classes.txt` 读写，以及与 `core.dataset_converter.scan_dataset` 的端到端互通。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from yolo_studio.core.annotation_io import (
    ImageAnnotationDoc,
    RectAnnotation,
    inspect_image_status,
    label_color_rgb,
    load_classes_txt,
    load_image_annotation,
    save_classes_txt,
    save_image_annotation,
    scan_labeling_directory,
)
from yolo_studio.core.dataset_converter import AnnotationFormat, DatasetBuildConfig, scan_dataset


def _make_img(path: Path, size: tuple[int, int] = (320, 240)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(40, 90, 140)).save(path)
    return path


def test_rect_annotation_normalization_clamping_and_roundtrip_with_dataset_converter(tmp_path: Path) -> None:
    """验证反向坐标规范化、边界裁剪、AI 审核元信息往返及与数据集扫描引擎的 100% 互通。"""
    img = _make_img(tmp_path / "中文子目录" / "测试帧_001.jpg", (400, 300))

    doc = load_image_annotation(img)
    assert doc.has_json_file is False
    assert doc.image_width == 400 and doc.image_height == 300
    assert doc.boxes == []
    assert inspect_image_status(img).status_key == "unlabeled"

    # 故意传入反向坐标及略超右边界坐标，验证自动规范化与裁剪
    box1 = RectAnnotation(
        label="安全帽",
        x1=180.0,
        y1=200.0,
        x2=60.0,
        y2=50.0,
        score=0.9134,
        description="kept: 佩戴清晰",
        flags={"llm_added": False, "box_refined": True},
    )
    box2 = RectAnnotation(
        label="人物",
        x1=220.0,
        y1=20.0,
        x2=450.0,
        y2=290.0,
        description="llm_added: 补标漏检行人",
        flags={"llm_added": True, "box_refined": False},
    )
    doc.boxes = [box1, box2]
    doc.flags = {"has_missing_hint": True, "missing_note": "左上角疑似有遮挡人物"}

    saved_json = save_image_annotation(doc)
    assert saved_json.is_file()
    assert doc.has_json_file is True

    reloaded = load_image_annotation(img)
    assert reloaded.has_json_file is True
    assert reloaded.has_missing_hint is True
    assert "左上角" in reloaded.missing_note
    assert len(reloaded.boxes) == 2
    assert reloaded.boxes[0].label == "安全帽"
    assert reloaded.boxes[0].x1 == 60.0 and reloaded.boxes[0].x2 == 180.0
    assert reloaded.boxes[0].y1 == 50.0 and reloaded.boxes[0].y2 == 200.0
    assert reloaded.boxes[0].score == 0.9134
    assert reloaded.boxes[0].flags["box_refined"] is True
    assert reloaded.boxes[1].x2 == 400.0  # 超界坐标已自动裁剪至图像宽度
    assert reloaded.boxes[1].flags["llm_added"] is True

    status = inspect_image_status(img)
    assert status.status_key == "review"
    assert status.box_count == 2
    assert status.labels == ("安全帽", "人物")

    # 端到端喂给数据集扫描引擎，确认 100% 兼容解析
    summary = scan_dataset(
        DatasetBuildConfig(
            images_dir=tmp_path,
            annotation_format=AnnotationFormat.AUTO,
            recursive=True,
        )
    )
    assert summary.detected_format == "json"
    assert summary.total_images == 1
    assert summary.annotated_images == 1
    assert summary.total_boxes == 2
    assert set(summary.classes) == {"安全帽", "人物"}


def test_preserves_non_rectangle_shapes_and_supports_empty_negative_sample(tmp_path: Path) -> None:
    """验证修改并保存矩形框时无损保留历史多边形形状，以及 0 框保存为显式空图负样本。"""
    img_poly = _make_img(tmp_path / "with_poly.jpg", (300, 200))
    poly_shape = {
        "label": "road",
        "points": [[10.0, 10.0], [100.0, 10.0], [100.0, 90.0]],
        "shape_type": "polygon",
        "flags": {},
    }
    raw_payload = {
        "version": "2.4.4",
        "flags": {},
        "shapes": [
            poly_shape,
            {
                "label": "car",
                "points": [[50.0, 40.0], [150.0, 120.0]],
                "shape_type": "rectangle",
            },
        ],
        "imagePath": "with_poly.jpg",
        "imageData": None,
        "imageHeight": 200,
        "imageWidth": 300,
    }
    img_poly.with_suffix(".json").write_text(json.dumps(raw_payload), encoding="utf-8")

    doc = load_image_annotation(img_poly)
    assert len(doc.boxes) == 1
    assert len(doc.preserved_shapes) == 1
    doc.boxes[0].label = "truck"
    save_image_annotation(doc)

    written = json.loads(img_poly.with_suffix(".json").read_text(encoding="utf-8"))
    types = [s["shape_type"] for s in written["shapes"]]
    assert "rectangle" in types and "polygon" in types
    rect_s = next(s for s in written["shapes"] if s["shape_type"] == "rectangle")
    assert rect_s["label"] == "truck"

    # 显式空图负样本
    img_empty = _make_img(tmp_path / "empty_bg.jpg", (200, 150))
    empty_doc = load_image_annotation(img_empty)
    empty_doc.flags["verified"] = True
    save_image_annotation(empty_doc)
    st = inspect_image_status(img_empty)
    assert st.has_json is True
    assert st.box_count == 0
    assert st.is_verified is True
    assert st.status_key == "empty"


def test_corrupted_json_tolerance_classes_txt_and_palette_stability(tmp_path: Path) -> None:
    """验证损坏 JSON 安全容错、递归扫图、`classes.txt` 去重读写与类别颜色稳定性。"""
    img = _make_img(tmp_path / "sub" / "broken.jpg", (160, 120))
    img.with_suffix(".json").write_text("{not valid json", encoding="utf-8")

    doc = load_image_annotation(img)
    assert isinstance(doc, ImageAnnotationDoc)
    assert doc.boxes == []
    assert doc.image_width == 160
    assert inspect_image_status(img).box_count == 0

    imgs = scan_labeling_directory(tmp_path, recursive=True)
    assert imgs == [img.resolve()]

    cls_file = tmp_path / "classes.txt"
    save_classes_txt(cls_file, ["person", " helmet ", "person", "", "vest"])
    assert load_classes_txt(cls_file) == ["person", "helmet", "vest"]

    assert label_color_rgb("person") == label_color_rgb("person")
    assert len(label_color_rgb("安全帽")) == 3
