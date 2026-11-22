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
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "val", "test")


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
    """单张图片及其解析出的边界框列表；未匹配到标注文件时 `source_label_path` 为 `None`。

    `is_negative` 为 `True` 表示存在标注文件、其中确实没有任何目标且解析过程未跳过任何形状或行
    （空 `.txt` 或 `shapes=[]` 的 `.json`），即人工确认过的背景负样本；
    因格式错误、不支持的形状被跳过而导致无框的样本不视为负样本。
    """

    image_path: Path
    source_label_path: Path | None
    boxes: list[BoundingBox] = field(default_factory=list)
    is_negative: bool = False


@dataclass
class DatasetScanSummary:
    """构建训练集前的只读扫描统计报告。

    计数口径：
        `annotated_images` 为至少含 1 个有效框的图片数；`unannotated_images` 为无有效框的图片数，
        其中 `negative_images` 张属于显式负样本（见 `LabeledSample.is_negative`）；
        `excluded_images` 为递归扫描时因位于 `exclude_subdirs` 中而被排除、不计入 `total_images` 的图片数。
    """

    detected_format: str
    total_images: int
    annotated_images: int
    unannotated_images: int
    total_boxes: int
    classes: list[str]
    class_counts: dict[str, int]
    samples: list[LabeledSample]
    warnings: list[str] = field(default_factory=list)
    negative_images: int = 0
    excluded_images: int = 0


@dataclass
class DatasetBuildConfig:
    """生成 YOLO26 数据集目录与 `data.yaml` 的配置参数。

    关键字段说明：
        `annotations_dir`：标注目录；为 `None` 时默认与 `images_dir` 相同。
        `train_ratio` / `val_ratio` / `test_ratio`：划分权重，划分时会按三者之和归一化。
        `include_unannotated`：是否将**没有任何标注文件**的图片作为纯负样本纳入拆分输出；
            存在标注文件但无目标的显式负样本（见 `LabeledSample.is_negative`）始终纳入，不受此开关影响。
        `copy_images`：为 `True` 时复制源图片，为 `False` 时移动源图片到目标数据集目录。
        `recursive`：是否递归扫描 `images_dir` 子目录（适配 LLM 智能标注分流后的多子目录）。
        `exclude_subdirs`：仅在 `recursive=True` 时生效，相对 `images_dir` 路径中任一层目录名
            命中该集合的图片会被排除（例如排除尚未人工复核的 `need_review`）。
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
    exclude_subdirs: tuple[str, ...] = ()


@dataclass
class DatasetBuildResult:
    """YOLO 数据集生成完成后的产物路径与分布统计摘要。

    关键字段说明：
        `class_counts`：扫描阶段全部样本（含未被纳入划分的样本）中各类别的标注框数。
        `split_class_counts`：实际写入各划分的标注框数，键为 `train` / `val` / `test`，
            值为“类别名称 → 框数”；三个划分键始终存在，某类别在该划分中没有框时不出现在内层字典中。
            用于检查验证集是否缺失某些类别。
    """

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
    split_class_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def find_images(
    images_dir: Path, recursive: bool, exclude_subdirs: tuple[str, ...] | set[str] = ()
) -> list[Path]:
    """按路径排序返回目录中的图片文件列表（只读）。

    `recursive=True` 时递归扫描子目录，并排除相对 `images_dir` 路径中任一层目录名命中
    `exclude_subdirs` 的图片；非递归模式只扫描顶层，`exclude_subdirs` 不生效。
    同时供 `core.mllm_reviewer` 复用。
    """
    if not recursive:
        return sorted(
            p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    excluded = set(exclude_subdirs)
    found: list[Path] = []
    for p in images_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if excluded and any(part in excluded for part in p.parent.relative_to(images_dir).parts):
            continue
        found.append(p)
    return sorted(found)


def _count_excluded_images(images_dir: Path, exclude_subdirs: tuple[str, ...]) -> int:
    """统计递归扫描时因 `exclude_subdirs` 被排除的图片数量，用于告警提示。"""
    if not exclude_subdirs:
        return 0
    total = len(find_images(images_dir, recursive=True))
    kept = len(find_images(images_dir, recursive=True, exclude_subdirs=exclude_subdirs))
    return total - kept


@dataclass
class ImageCountPreview:
    """图片目录的轻量计数预览，供界面在选择目录后即时提示扫描范围（只读，不解析标注）。

    字段说明：
        `top_level`：目录顶层（不含子目录）的图片数。
        `total`：含所有子目录的图片总数（含 `excluded`）。
        `excluded`：位于 `exclude_subdirs` 命中目录中的图片数，口径与 `find_images` 一致。
        `truncated`：累计文件数达到 `limit` 后提前停止，此时各计数为下限值。
    """

    top_level: int = 0
    total: int = 0
    excluded: int = 0
    truncated: bool = False


def preview_image_counts(
    images_dir: Path, exclude_subdirs: tuple[str, ...] = (), limit: int = 200_000
) -> ImageCountPreview:
    """遍历目录统计顶层与递归图片数，供界面实时显示“勾选递归扫描后会多出多少张图片”。

    只按扩展名计数，不打开图片或标注文件；遍历过的文件条目达到 `limit` 时停止并标记 `truncated`，
    避免误选磁盘根目录等超大目录时长时间卡住主线程。目录不存在或无法读取时返回全零结果。
    """
    result = ImageCountPreview()
    excluded = set(exclude_subdirs)
    seen = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(images_dir):
            rel_parts = Path(dirpath).relative_to(images_dir).parts
            in_excluded = bool(excluded) and any(part in excluded for part in rel_parts)
            for name in filenames:
                seen += 1
                if seen > limit:
                    result.truncated = True
                    return result
                if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                result.total += 1
                if not rel_parts:
                    result.top_level += 1
                elif in_excluded:
                    result.excluded += 1
    except OSError:
        return ImageCountPreview()
    return result


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
    """解析单个 X-AnyLabeling `.json` 文件，将矩形/多边形/旋转框转换为归一化外接框。

    文件损坏或不是合法 JSON 时记录告警并返回空列表，不中断整次扫描。
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        warnings.append(f"标注文件损坏或无法读取，已跳过 {json_path.name}: {exc}")
        return []
    if not isinstance(data, dict):
        warnings.append(f"标注文件结构无效，已跳过: {json_path.name}")
        return []
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
    for shape in data.get("shapes") or []:
        label = str(shape.get("label", "")).strip()
        if not label:
            warnings.append(f"空标签形状已忽略 ({json_path.name})")
            continue
        shape_type = str(shape.get("shape_type", "rectangle")).lower()
        points = shape.get("points") or []

        if shape_type in {"rectangle", "polygon", "rotation"} and len(points) >= 2:
            try:
                xs = [float(pt[0]) for pt in points]
                ys = [float(pt[1]) for pt in points]
            except (TypeError, ValueError, IndexError):
                warnings.append(f"坐标格式错误已忽略 ({json_path.name}, label={label})")
                continue
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
    """解析单个 YOLO `.txt` 标签文件；类别索引越界时会就地扩展 `known_classes`。

    文件无法按 UTF-8 读取时记录告警并返回空列表。
    """
    boxes: list[BoundingBox] = []
    try:
        lines = txt_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"标签文件无法读取，已跳过 {txt_path.name}: {exc}")
        return []
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


def resolve_label_file(
    image_path: Path,
    images_dir: Path,
    annotations_dir: Path,
    suffix: str,
    allow_flat_fallback: bool = True,
) -> Path | None:
    """按优先级查找图片对应的标注文件，未找到返回 `None`。

    查找顺序：`annotations_dir/<相对子目录>/<stem><suffix>` → `annotations_dir/<stem><suffix>`
    （仅 `allow_flat_fallback=True` 时）→ 与图片同目录的 `<stem><suffix>`。
    递归扫描中若存在同名（同 `stem`）图片，调用方应传 `allow_flat_fallback=False`，
    避免子目录图片误配到属于另一张同名图片的平铺标注。同时供 `core.mllm_reviewer` 复用。
    """
    try:
        rel_parent = image_path.parent.relative_to(images_dir)
    except ValueError:
        rel_parent = Path(".")

    candidates = [annotations_dir / rel_parent / f"{image_path.stem}{suffix}"]
    if allow_flat_fallback or rel_parent == Path("."):
        candidates.append(annotations_dir / f"{image_path.stem}{suffix}")
    candidates.append(image_path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def stem_collisions(images: list[Path]) -> set[str]:
    """返回在图片列表中出现不止一次的文件 `stem` 集合，用于关闭平铺标注回退匹配。"""
    counts = Counter(p.stem for p in images)
    return {stem for stem, n in counts.items() if n > 1}


def scan_dataset(config: DatasetBuildConfig) -> DatasetScanSummary:
    """只读扫描图片与 X-AnyLabeling 标注文件并返回统计摘要，不写出任何文件。

    格式识别与异常处理：
        `images_dir` 或指定的 `annotations_dir` 不存在时抛出 `FileNotFoundError`；
        未发现图片时抛出 `ValueError`。当标注格式为 `AUTO` 时，先抽样前 50 张图片统计
        `.json` 与 `.txt` 命中数（若全未命中再顺序扫描至首个命中），命中数相同时优先 `.json`。
        单个标注文件损坏只记录告警，不中断扫描。

    递归扫描约束：
        `config.recursive=True` 时按 `config.exclude_subdirs` 排除子目录并在告警中注明排除数量；
        同名（同 `stem`）图片不使用平铺标注回退匹配，防止跨子目录误配标注。
    """
    if not config.images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {config.images_dir}")

    ann_dir = config.annotations_dir if config.annotations_dir else config.images_dir
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")

    exclude = tuple(config.exclude_subdirs) if config.recursive else ()
    images = find_images(config.images_dir, recursive=config.recursive, exclude_subdirs=exclude)
    warnings: list[str] = []
    excluded_count = _count_excluded_images(config.images_dir, exclude)
    if excluded_count:
        warnings.append(
            f"已按设置排除子目录 {list(exclude)} 中的 {excluded_count} 张图片（未纳入统计与数据集）。"
        )
    if not images:
        raise ValueError(f"在目录中未找到图片文件: {config.images_dir}")

    known_classes = _load_classes_file(
        config.classes_file, [ann_dir, config.images_dir, ann_dir.parent]
    )
    colliding_stems = stem_collisions(images)

    def _resolve(img: Path, suffix: str) -> Path | None:
        return resolve_label_file(
            img,
            config.images_dir,
            ann_dir,
            suffix,
            allow_flat_fallback=img.stem not in colliding_stems,
        )

    # 若格式为 AUTO，根据已有标注文件命中数量自动推断格式
    fmt = config.annotation_format
    if fmt == AnnotationFormat.AUTO:
        json_hits = 0
        txt_hits = 0
        for img in images[:50]:
            if _resolve(img, ".json"):
                json_hits += 1
            if _resolve(img, ".txt"):
                txt_hits += 1
        if json_hits == 0 and txt_hits == 0:
            # 抽样前 50 张未命中时继续全量探测，避免标注靠后导致误判
            for img in images:
                if _resolve(img, ".json"):
                    json_hits += 1
                if _resolve(img, ".txt"):
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
    negative_count = 0

    for img_path in images:
        warn_before = len(warnings)
        if fmt == AnnotationFormat.XANYLABELING_JSON:
            label_path = _resolve(img_path, ".json")
            boxes = (
                _parse_xanylabeling_json(label_path, img_path, warnings)
                if label_path
                else []
            )
        else:
            label_path = _resolve(img_path, ".txt")
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

        # 仅当标注文件存在、无框且解析过程没有跳过任何内容时，才视为人工确认的显式负样本
        is_negative = label_path is not None and not boxes and len(warnings) == warn_before
        if boxes:
            annotated_count += 1
        else:
            unannotated_count += 1
            if is_negative:
                negative_count += 1

        samples.append(
            LabeledSample(
                image_path=img_path,
                source_label_path=label_path,
                boxes=boxes,
                is_negative=is_negative,
            )
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
        negative_images=negative_count,
        excluded_images=excluded_count,
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


def _is_within(path: Path, parent: Path) -> bool:
    """判断已解析的 `path` 是否位于 `parent` 之内（含相等）。"""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_location(
    config: DatasetBuildConfig,
    out_dir: Path,
    split_dirs: list[Path],
    samples: list[LabeledSample],
) -> None:
    """在清理旧划分目录前校验输出位置，防止误删源图片或把输出重新扫描为输入。"""
    if config.recursive and _is_within(out_dir, config.images_dir.resolve()):
        raise ValueError(
            f"递归扫描时数据集输出目录不能位于图片目录内部，否则重复构建会把旧数据集当作输入: {out_dir}"
        )
    for sample in samples:
        src = sample.image_path.resolve()
        if any(_is_within(src, d) for d in split_dirs):
            raise ValueError(
                f"源图片位于将被重建的数据集划分目录中，请更换输出目录: {sample.image_path}"
            )


def _clear_previous_splits(out_dir: Path, split_dirs: list[Path]) -> None:
    """删除上一次构建残留的划分子目录与 Ultralytics 标签缓存，仅作用于 `out_dir` 下的固定子路径。"""
    for d in split_dirs:
        if d.is_dir():
            shutil.rmtree(d)
    labels_root = out_dir / "labels"
    if labels_root.is_dir():
        for cache_file in labels_root.glob("*.cache"):
            cache_file.unlink(missing_ok=True)


def existing_split_dirs(output_dir: Path) -> list[Path]:
    """只读列出 `output_dir` 下已存在且非空、会被 `build_yolo_dataset` 清理重建的划分子目录。

    用途：
        供界面在生成前提示用户确认覆盖；返回空列表表示本次构建不会删除任何已有划分文件。
        与 `_clear_previous_splits` 的清理范围保持一致（`images|labels` × `train|val|test`），
        不包含 `labels/*.cache` 这类可再生缓存。目录不存在或无法读取时视为无可覆盖内容。
    """
    found: list[Path] = []
    for kind in ("images", "labels"):
        for split in SPLIT_NAMES:
            d = output_dir / kind / split
            try:
                if d.is_dir() and any(d.iterdir()):
                    found.append(d)
            except OSError:
                continue
    return found


def build_yolo_dataset(config: DatasetBuildConfig) -> DatasetBuildResult:
    """转换标注、按比例划分为 train/val/test，并写出 Ultralytics `data.yaml`。

    输入校验与失败行为：
        先调用 `scan_dataset` 校验输入目录；若未检测到有效类别、无符合条件的样本，
        或三者划分比例之和 `<= 0`，均抛出 `ValueError` 且不写入数据集划分目录。

    输出产物与副作用：
        在 `config.output_dir` 下创建 `images/{train,val,test}`、`labels/{train,val,test}`、
        `classes.txt` 与 `data.yaml`（若 `val_set` 为空，`data.yaml` 的 `val` 回退指向 `images/train`；
        若 `test_set` 为空则不创建 `test` 目录与配置项）。
        写入前会**删除**上述六个划分子目录及 `labels/*.cache`，保证重复构建时不残留旧划分文件
        （否则同一图片可能同时出现在 train 与 val 中造成评估泄漏）；`output_dir` 中的其他文件不受影响。
        当 `config.copy_images=False` 时会移动源图片文件并在目标已存在时先覆盖删除。

    安全约束：
        若任一源图片位于将被清理的划分子目录内，或递归扫描时 `output_dir` 位于 `images_dir` 内部，
        抛出 `ValueError` 且不删除任何文件。
    """
    summary = scan_dataset(config)
    if not summary.classes:
        raise ValueError("未检测到任何目标类别，请先确认标注文件中包含有效标注。")

    selected = [
        s
        for s in summary.samples
        if s.boxes or s.is_negative or config.include_unannotated
    ]
    if not selected:
        raise ValueError("没有任何包含有效标注框的图片，无法构建训练数据集。")

    class_to_id = {name: idx for idx, name in enumerate(summary.classes)}
    train_set, val_set, test_set = _split_samples(
        selected, config.train_ratio, config.val_ratio, config.test_ratio, config.seed
    )

    out_dir = config.output_dir.resolve()
    split_dirs = [out_dir / kind / split for kind in ("images", "labels") for split in SPLIT_NAMES]
    _validate_output_location(config, out_dir, split_dirs, summary.samples)
    _clear_previous_splits(out_dir, split_dirs)

    split_class_counts: dict[str, dict[str, int]] = {}
    for split_name, split_items in (("train", train_set), ("val", val_set), ("test", test_set)):
        split_counter: Counter[str] = Counter(b.class_name for s in split_items for b in s.boxes)
        split_class_counts[split_name] = dict(split_counter)
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
        split_class_counts=split_class_counts,
    )
