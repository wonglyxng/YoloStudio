"""X-AnyLabeling 标注校验与 YOLO26 数据集构建引擎。

职责：
    扫描图片目录及其对应的 X-AnyLabeling `.json` 或 YOLO `.txt` 标注，统计类别与框数分布，
    并按比例拆分为 `train`、`val`、`test` 子集生成标准 YOLO 目录结构与 `data.yaml`。

运行环境：
    Python >= 3.11，依赖 `Pillow` 与 `PyYAML`。

协作对象与职责边界：
    为 `cli._cmd_dataset`、`gui.workers.DatasetScanWorker` / `DatasetBuildWorker` 与
    `gui.tab_dataset.DatasetTab` 提供扫描与转换能力；`IMAGE_EXTENSIONS` 同时被
    `core.mllm_reviewer` 复用。仅负责边界框转换（矩形框及多边形/旋转框的外接矩形）、
    数据集拆分与目录写入，不修改原始标注文件，不执行训练。
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class AnnotationFormat(str, Enum):
    """X-AnyLabeling 导出的输入标注格式枚举。"""

    AUTO = "auto"
    XANYLABELING_JSON = "json"
    YOLO_TXT = "txt"


@dataclass
class BoundingBox:
    """归一化 YOLO 目标检测框。

    坐标系与取值范围：
        `cx`、`cy` 为中心点相对图像宽高的归一化坐标，`w`、`h` 为归一化宽高，
        四个字段均已被裁剪到 `[0.0, 1.0]` 区间。
    """

    class_name: str
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class LabeledSample:
    """单张图片及其解析出的边界框列表；未匹配到标注文件时 `source_label_path` 为 `None`。"""

    image_path: Path
    source_label_path: Path | None
    boxes: list[BoundingBox] = field(default_factory=list)


@dataclass
class DatasetScanSummary:
    """构建训练集前的只读扫描统计报告。"""

    detected_format: str
    total_images: int
    annotated_images: int
    unannotated_images: int
    total_boxes: int
    classes: list[str]
    class_counts: dict[str, int]
    samples: list[LabeledSample]
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetBuildConfig:
    """生成 YOLO26 数据集目录与 `data.yaml` 的配置参数。

    关键字段说明：
        `annotations_dir`：标注目录；为 `None` 时默认与 `images_dir` 相同。
        `train_ratio` / `val_ratio` / `test_ratio`：划分权重，划分时会按三者之和归一化。
        `include_unannotated`：是否将无标注图片作为纯负样本纳入拆分输出。
        `copy_images`：为 `True` 时复制源图片，为 `False` 时移动源图片到目标数据集目录。
        `recursive`：是否递归扫描 `images_dir` 子目录（适配 MLLM 清洗分流后的多子目录）。
    """

    images_dir: Path
    annotations_dir: Path | None = None
    output_dir: Path = Path("datasets/custom_yolo26")
    annotation_format: AnnotationFormat = AnnotationFormat.AUTO
    classes_file: Path | None = None
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    include_unannotated: bool = False
    seed: int = 42
    copy_images: bool = True
    recursive: bool = False


@dataclass
class DatasetBuildResult:
    """YOLO 数据集生成完成后的产物路径与分布统计摘要。"""

    output_dir: Path
    data_yaml_path: Path
    classes_txt_path: Path
    classes: list[str]
    train_count: int
    val_count: int
    test_count: int
    total_boxes: int
    class_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def _find_images(images_dir: Path, recursive: bool) -> list[Path]:
    iterator = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_classes_file(classes_file: Path | None, search_dirs: list[Path]) -> list[str]:
    candidates: list[Path] = []
    if classes_file is not None and classes_file.is_file():
        candidates.append(classes_file)
    for d in search_dirs:
        if d and d.is_dir():
            for name in ("classes.txt", "labels.txt"):
                candidate = d / name
                if candidate.is_file():
                    candidates.append(candidate)

    for path in candidates:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        names = [line for line in lines if line]
        if names:
            return names
    return []


def _get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size  # 返回 (宽, 高) 像素尺寸


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_xanylabeling_json(
    json_path: Path, image_path: Path, warnings: list[str]
) -> list[BoundingBox]:
    """解析单个 X-AnyLabeling `.json` 文件，将矩形/多边形/旋转框转换为归一化外接框。"""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_w = data.get("imageWidth")
    img_h = data.get("imageHeight")
    if not img_w or not img_h:
        img_w, img_h = _get_image_size(image_path)

    img_w = float(img_w)
    img_h = float(img_h)
    if img_w <= 0 or img_h <= 0:
        warnings.append(f"无效图像尺寸，跳过: {json_path.name}")
        return []

    boxes: list[BoundingBox] = []
    for shape in data.get("shapes", []):
        label = str(shape.get("label", "")).strip()
        if not label:
            continue
        shape_type = str(shape.get("shape_type", "rectangle")).lower()
        points = shape.get("points") or []

        if shape_type in {"rectangle", "polygon", "rotation"} and len(points) >= 2:
            xs = [float(pt[0]) for pt in points]
            ys = [float(pt[1]) for pt in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            x_min = max(0.0, min(img_w, x_min))
            x_max = max(0.0, min(img_w, x_max))
            y_min = max(0.0, min(img_h, y_min))
            y_max = max(0.0, min(img_h, y_max))

            bw = x_max - x_min
            bh = y_max - y_min
            if bw <= 1.0 or bh <= 1.0:
                warnings.append(f"过小/无效框已忽略 ({json_path.name}, label={label})")
                continue

            cx = _clamp01((x_min + x_max) / (2.0 * img_w))
            cy = _clamp01((y_min + y_max) / (2.0 * img_h))
            nw = _clamp01(bw / img_w)
            nh = _clamp01(bh / img_h)
            boxes.append(BoundingBox(class_name=label, cx=cx, cy=cy, w=nw, h=nh))
        else:
            warnings.append(
                f"跳过非矩形/不支持的标注类型 '{shape_type}' ({json_path.name}, label={label})"
            )

    return boxes


def _parse_yolo_txt(
    txt_path: Path,
    known_classes: list[str],
    warnings: list[str],
) -> list[BoundingBox]:
    """解析单个 YOLO `.txt` 标签文件；类别索引越界时会就地扩展 `known_classes`。"""
    boxes: list[BoundingBox] = []
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            warnings.append(f"格式错误已跳过 {txt_path.name}:{line_no} -> '{line}'")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            warnings.append(f"数值解析失败 {txt_path.name}:{line_no} -> '{line}'")
            continue

        if cls_id < 0:
            warnings.append(f"非法类别 ID ({cls_id}) {txt_path.name}:{line_no}")
            continue
        if w <= 0 or h <= 0:
            warnings.append(f"宽高 <= 0 已忽略 {txt_path.name}:{line_no}")
            continue

        while cls_id >= len(known_classes):
            known_classes.append(f"class_{len(known_classes)}")

        boxes.append(
            BoundingBox(
                class_name=known_classes[cls_id],
                cx=_clamp01(cx),
                cy=_clamp01(cy),
                w=_clamp01(w),
                h=_clamp01(h),
            )
        )
    return boxes


def _resolve_label_file(
    image_path: Path,
    images_dir: Path,
    annotations_dir: Path,
    suffix: str,
) -> Path | None:
    try:
        rel_parent = image_path.parent.relative_to(images_dir)
    except ValueError:
        rel_parent = Path(".")

    candidates = [
        annotations_dir / rel_parent / f"{image_path.stem}{suffix}",
        annotations_dir / f"{image_path.stem}{suffix}",
        image_path.with_suffix(suffix),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def scan_dataset(config: DatasetBuildConfig) -> DatasetScanSummary:
    """只读扫描图片与 X-AnyLabeling 标注文件并返回统计摘要，不写出任何文件。

    格式识别与异常处理：
        `images_dir` 或指定的 `annotations_dir` 不存在时抛出 `FileNotFoundError`；
        未发现图片时抛出 `ValueError`。当标注格式为 `AUTO` 时，先抽样前 50 张图片统计
        `.json` 与 `.txt` 命中数（若全未命中再顺序扫描至首个命中），命中数相同时优先 `.json`。
    """
    if not config.images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {config.images_dir}")

    ann_dir = config.annotations_dir if config.annotations_dir else config.images_dir
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")

    images = _find_images(config.images_dir, recursive=config.recursive)
    warnings: list[str] = []
    if not images:
        raise ValueError(f"在目录中未找到图片文件: {config.images_dir}")

    known_classes = _load_classes_file(
        config.classes_file, [ann_dir, config.images_dir, ann_dir.parent]
    )

    # 若格式为 AUTO，根据已有标注文件命中数量自动推断格式
    fmt = config.annotation_format
    if fmt == AnnotationFormat.AUTO:
        json_hits = 0
        txt_hits = 0
        for img in images[:50]:
            if _resolve_label_file(img, config.images_dir, ann_dir, ".json"):
                json_hits += 1
            if _resolve_label_file(img, config.images_dir, ann_dir, ".txt"):
                txt_hits += 1
        if json_hits == 0 and txt_hits == 0:
            # 抽样前 50 张未命中时继续全量探测，避免标注靠后导致误判
            for img in images:
                if _resolve_label_file(img, config.images_dir, ann_dir, ".json"):
                    json_hits += 1
                if _resolve_label_file(img, config.images_dir, ann_dir, ".txt"):
                    txt_hits += 1
                if json_hits or txt_hits:
                    break
        fmt = (
            AnnotationFormat.XANYLABELING_JSON
            if json_hits >= txt_hits
            else AnnotationFormat.YOLO_TXT
        )

    if fmt == AnnotationFormat.YOLO_TXT and not known_classes:
        warnings.append("未找到 classes.txt，将使用 class_0, class_1... 作为临时类别名称。")

    samples: list[LabeledSample] = []
    class_counter: Counter[str] = Counter()
    discovered_classes: list[str] = list(known_classes)
    discovered_set = set(discovered_classes)

    annotated_count = 0
    unannotated_count = 0

    for img_path in images:
        if fmt == AnnotationFormat.XANYLABELING_JSON:
            label_path = _resolve_label_file(img_path, config.images_dir, ann_dir, ".json")
            boxes = (
                _parse_xanylabeling_json(label_path, img_path, warnings)
                if label_path
                else []
            )
        else:
            label_path = _resolve_label_file(img_path, config.images_dir, ann_dir, ".txt")
            boxes = (
                _parse_yolo_txt(label_path, discovered_classes, warnings)
                if label_path
                else []
            )
            discovered_set = set(discovered_classes)

        for box in boxes:
            class_counter[box.class_name] += 1
            if box.class_name not in discovered_set:
                discovered_classes.append(box.class_name)
                discovered_set.add(box.class_name)

        if boxes:
            annotated_count += 1
        else:
            unannotated_count += 1

        samples.append(
            LabeledSample(image_path=img_path, source_label_path=label_path, boxes=boxes)
        )

    return DatasetScanSummary(
        detected_format=fmt.value,
        total_images=len(images),
        annotated_images=annotated_count,
        unannotated_images=unannotated_count,
        total_boxes=sum(class_counter.values()),
        classes=discovered_classes,
        class_counts=dict(class_counter),
        samples=samples,
        warnings=warnings,
    )


def _split_samples(
    samples: list[LabeledSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[LabeledSample], list[LabeledSample], list[LabeledSample]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("train/val/test 比例之和必须大于 0。")

    tr = train_ratio / total_ratio
    va = val_ratio / total_ratio

    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 1:
        return shuffled, [], []

    n_train = max(1, int(round(n * tr)))
    if val_ratio > 0 and n_train >= n:
        n_train = n - 1

    remaining = n - n_train
    if test_ratio <= 0:
        n_val = remaining
        n_test = 0
    else:
        val_share = va / (1.0 - tr) if tr < 1.0 else 0.5
        n_val = int(round(remaining * val_share))
        n_test = remaining - n_val

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val : n_train + n_val + n_test]
    return train_set, val_set, test_set


def _unique_stem(path: Path, used_names: set[str]) -> str:
    stem = path.stem
    candidate = stem
    idx = 1
    while candidate in used_names:
        candidate = f"{stem}_{idx}"
        idx += 1
    used_names.add(candidate)
    return candidate


def build_yolo_dataset(config: DatasetBuildConfig) -> DatasetBuildResult:
    """转换标注、按比例划分为 train/val/test，并写出 Ultralytics `data.yaml`。

    输入校验与失败行为：
        先调用 `scan_dataset` 校验输入目录；若未检测到有效类别、无符合条件的样本，
        或三者划分比例之和 `<= 0`，均抛出 `ValueError` 且不写入数据集划分目录。

    输出产物与副作用：
        在 `config.output_dir` 下创建 `images/{train,val,test}`、`labels/{train,val,test}`、
        `classes.txt` 与 `data.yaml`（若 `val_set` 为空，`data.yaml` 的 `val` 回退指向 `images/train`；
        若 `test_set` 为空则不创建 `test` 目录与配置项）。
        当 `config.copy_images=False` 时会移动源图片文件并在目标已存在时先覆盖删除。
    """
    summary = scan_dataset(config)
    if not summary.classes:
        raise ValueError("未检测到任何目标类别，请先确认标注文件中包含有效标注。")

    selected = [
        s for s in summary.samples if config.include_unannotated or len(s.boxes) > 0
    ]
    if not selected:
        raise ValueError("没有任何包含有效标注框的图片，无法构建训练数据集。")

    class_to_id = {name: idx for idx, name in enumerate(summary.classes)}
    train_set, val_set, test_set = _split_samples(
        selected, config.train_ratio, config.val_ratio, config.test_ratio, config.seed
    )

    out_dir = config.output_dir.resolve()
    for split_name, split_items in (("train", train_set), ("val", val_set), ("test", test_set)):
        if not split_items and split_name == "test":
            continue
        img_out_dir = out_dir / "images" / split_name
        lbl_out_dir = out_dir / "labels" / split_name
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        used_stems: set[str] = set()
        for sample in split_items:
            stem = _unique_stem(sample.image_path, used_stems)
            dst_img = img_out_dir / f"{stem}{sample.image_path.suffix.lower()}"
            dst_lbl = lbl_out_dir / f"{stem}.txt"

            if config.copy_images:
                shutil.copy2(sample.image_path, dst_img)
            else:
                if dst_img.exists():
                    dst_img.unlink()
                shutil.move(str(sample.image_path), str(dst_img))

            label_lines = [
                f"{class_to_id[b.class_name]} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}"
                for b in sample.boxes
            ]
            dst_lbl.write_text("\n".join(label_lines), encoding="utf-8")

    classes_txt_path = out_dir / "classes.txt"
    classes_txt_path.write_text("\n".join(summary.classes) + "\n", encoding="utf-8")

    yaml_payload: dict[str, object] = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val" if val_set else "images/train",
        "nc": len(summary.classes),
        "names": {idx: name for idx, name in enumerate(summary.classes)},
    }
    if test_set:
        yaml_payload["test"] = "images/test"

    data_yaml_path = out_dir / "data.yaml"
    data_yaml_path.write_text(
        yaml.safe_dump(yaml_payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    return DatasetBuildResult(
        output_dir=out_dir,
        data_yaml_path=data_yaml_path,
        classes_txt_path=classes_txt_path,
        classes=summary.classes,
        train_count=len(train_set),
        val_count=len(val_set),
        test_count=len(test_set),
        total_boxes=summary.total_boxes,
        class_counts=summary.class_counts,
        warnings=summary.warnings,
    )
