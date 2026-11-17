"""无头批处理命令行工具，覆盖视频抽帧、数据集构建、YOLO26 训练与 MLLM 标注纠错。

职责与真实入口：
    提供 `main()` 与子命令 `extract`、`dataset`、`train`、`review`
    （命令示例见 `docs/001_录制视频抽帧工具使用说明.md` 与 `docs/002_YOLO26训练使用说明.md`）。

运行环境与协作对象：
    需 Python >= 3.11 及对应子命令的运行库（`opencv-python`、`Pillow`、`PyYAML`、
    `ultralytics`、`torch`、`google-genai` / `openai`、`pydantic`）；
    `review` 子命令调用远程多模态接口时还需通过 `--api-key` 或环境变量
    `GEMINI_API_KEY` / `OPENAI_API_KEY` 提供有效凭据。
    本脚本仅负责参数解析、配置构建与摘要输出，具体执行委托给 `yolo_studio.core` 各引擎，
    不启动 PyQt6 GUI。

输入输出与失败方式：
    按子命令从本地视频、图片、标注或 `data.yaml` 读取输入，并在指定输出目录写出帧图、
    YOLO 数据集（含 `data.yaml` 与 `classes.txt`）、训练权重或 MLLM 清洗分流结果与报告。
    参数非法时由 `argparse` 直接终止；`extract` 出现视频错误或 `review` 存在审核异常时返回
    退出码 `1`；缺失必需文件、权重或凭据时核心引擎将抛出异常并导致进程非零退出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 需先将 src 加入 sys.path 再导入内部包，以下保留 noqa: E402 指令
from yolo_studio.core.dataset_converter import (  # noqa: E402
    AnnotationFormat,
    DatasetBuildConfig,
    build_yolo_dataset,
)
from yolo_studio.core.frame_extractor import (  # noqa: E402
    ExtractionConfig,
    ExtractionMode,
    extract_frames,
)
from yolo_studio.core.mllm_reviewer import (  # noqa: E402
    BUCKET_NEED_REVIEW,
    MllmProvider,
    MllmReviewConfig,
    ReviewSourceMode,
    run_mllm_review,
)
from yolo_studio.core.trainer import TrainConfig, train_yolo26  # noqa: E402


def _cmd_extract(args: argparse.Namespace) -> int:
    """执行 `extract` 子命令并将抽帧异常打印到 stderr；无错误返回 0，否则返回 1。"""
    cfg = ExtractionConfig(
        output_dir=Path(args.output),
        mode=ExtractionMode(args.mode),
        interval_seconds=args.interval,
        every_n_frames=args.step,
        target_count=args.count,
        dedup_threshold=args.dedup,
        max_dimension=args.max_dim,
        image_format=args.format,
    )
    res = extract_frames([Path(p) for p in args.inputs], cfg)
    print(f"处理视频: {res.videos_processed}, 保存帧数: {res.frames_saved}, 去重跳过: {res.frames_skipped_similar}")
    for err in res.errors:
        print(f"[ERROR] {err}", file=sys.stderr)
    return 0 if not res.errors else 1


def _cmd_dataset(args: argparse.Namespace) -> int:
    """执行 `dataset` 子命令生成 YOLO 数据集并输出划分统计，成功返回 0。"""
    cfg = DatasetBuildConfig(
        images_dir=Path(args.images),
        annotations_dir=Path(args.annotations) if args.annotations else None,
        output_dir=Path(args.output),
        annotation_format=AnnotationFormat(args.format),
        classes_file=Path(args.classes) if args.classes else None,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        include_unannotated=args.include_empty,
        seed=args.seed,
        recursive=args.recursive,
        exclude_subdirs=() if args.include_need_review else (BUCKET_NEED_REVIEW,),
    )
    res = build_yolo_dataset(cfg)
    for warning in res.warnings[:50]:
        print(f"[WARN] {warning}", file=sys.stderr)
    print(
        f"数据集生成成功: {res.data_yaml_path}\n"
        f"Train={res.train_count}, Val={res.val_count}, Test={res.test_count}, Classes={res.classes}"
    )
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    """执行 `train` 子命令启动 YOLO26 训练并将实时日志打印到标准输出，成功返回 0。"""
    cfg = TrainConfig(
        data_yaml=Path(args.data),
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project_dir=Path(args.project),
        run_name=args.name,
        resume=args.resume,
        exist_ok=args.exist_ok,
    )
    res = train_yolo26(cfg, log_callback=print)
    print(f"训练完成，输出目录: {res.save_dir}, best.pt: {res.best_weights}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """执行 `review` 子命令运行 MLLM 纠错清洗；无单图错误返回 0，否则返回 1。"""
    classes = [c.strip() for c in args.classes.replace("，", ",").split(",") if c.strip()]
    cfg = MllmReviewConfig(
        images_dir=Path(args.images),
        output_dir=Path(args.output),
        source_mode=ReviewSourceMode(args.source_mode),
        annotations_dir=Path(args.annotations) if args.annotations else None,
        weights_path=Path(args.weights) if args.weights else None,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.imgsz,
        yolo_device=args.device,
        provider=MllmProvider(args.provider),
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        allowed_classes=classes,
        custom_guideline=args.guideline,
        high_conf_skip=args.high_conf_skip,
        check_missing=not args.no_check_missing,
        add_missing_boxes=not args.no_add_missing,
        include_crops=not args.no_crops,
        max_workers=args.workers,
        triage_subfolders=not args.flat,
    )
    res = run_mllm_review(cfg, log_callback=print)
    print(
        f"MLLM 清洗完成: 总计={res.total_images}, 自动通过={res.auto_passed_count}, "
        f"已删改={res.llm_fixed_count}, 待人工复核={res.need_review_count}, 出错={res.error_count}"
    )
    return 0 if res.error_count == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """构建包含 `extract`、`dataset`、`train`、`review` 四个子命令的 CLI 解析器。"""
    parser = argparse.ArgumentParser(description="YOLO26 视频抽帧与训练 CLI 工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="从录制视频批量抽帧")
    p_ext.add_argument("inputs", nargs="+", help="视频文件或包含视频的目录")
    p_ext.add_argument("-o", "--output", default="data/raw_frames", help="输出目录")
    p_ext.add_argument("--mode", choices=["interval_seconds", "every_n_frames", "target_count"], default="interval_seconds")
    p_ext.add_argument("--interval", type=float, default=1.0, help="按秒抽帧间隔")
    p_ext.add_argument("--step", type=int, default=30, help="按帧步长")
    p_ext.add_argument("--count", type=int, default=100, help="每个视频目标帧数")
    p_ext.add_argument("--dedup", type=float, default=2.0, help="相似帧去重阈值 (0=关闭)")
    p_ext.add_argument("--max-dim", type=int, default=0, help="最长边缩放像素 (0=原尺寸)")
    p_ext.add_argument("--format", choices=["jpg", "png"], default="jpg")
    p_ext.set_defaults(func=_cmd_extract)

    p_ds = sub.add_parser("dataset", help="将 X-AnyLabeling 标注转换为 YOLO26 数据集")
    p_ds.add_argument("-i", "--images", required=True, help="抽帧图片目录")
    p_ds.add_argument("-a", "--annotations", default="", help="标注目录（留空默认同图片目录）")
    p_ds.add_argument("-o", "--output", default="datasets/custom_yolo26", help="输出数据集目录")
    p_ds.add_argument("--format", choices=["auto", "json", "txt"], default="auto")
    p_ds.add_argument("--classes", default="", help="可选 classes.txt 路径")
    p_ds.add_argument("--train-ratio", type=float, default=0.8)
    p_ds.add_argument("--val-ratio", type=float, default=0.2)
    p_ds.add_argument("--test-ratio", type=float, default=0.0)
    p_ds.add_argument("--include-empty", action="store_true", help="同时纳入没有任何标注文件的图片作为负样本")
    p_ds.add_argument("--seed", type=int, default=42)
    p_ds.add_argument("-r", "--recursive", action="store_true", help="递归扫描图片子目录（如 MLLM 清洗分流目录）")
    p_ds.add_argument(
        "--include-need-review",
        action="store_true",
        help="递归扫描时同时纳入 need_review 子目录（默认排除未人工复核的样本）",
    )
    p_ds.set_defaults(func=_cmd_dataset)

    p_tr = sub.add_parser("train", help="启动 YOLO26 模型训练")
    p_tr.add_argument("-d", "--data", required=True, help="data.yaml 路径")
    p_tr.add_argument("-m", "--model", default="yolo26s.pt", help="YOLO26 权重文件；续训时可直接指定 last.pt")
    p_tr.add_argument("--epochs", type=int, default=100)
    p_tr.add_argument("--imgsz", type=int, default=640)
    p_tr.add_argument("--batch", type=int, default=16)
    p_tr.add_argument("--device", default="0")
    p_tr.add_argument("--project", default="runs/detect")
    p_tr.add_argument("--name", default="yolo26_train")
    p_tr.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    p_tr.add_argument("--exist-ok", action="store_true", help="允许覆盖同名实验目录（默认自动递增编号）")
    p_tr.set_defaults(func=_cmd_train)

    p_rv = sub.add_parser("review", help="基准模型低阈值推理 + MLLM 纠错清洗")
    p_rv.add_argument("-i", "--images", required=True, help="待纠错图片目录")
    p_rv.add_argument("-o", "--output", default="data/reviewed_frames", help="清洗后输出目录")
    p_rv.add_argument("--source-mode", choices=["yolo_predict", "existing_json"], default="yolo_predict")
    p_rv.add_argument("-w", "--weights", default="", help="基准模型 .pt 路径 (yolo_predict 模式)")
    p_rv.add_argument("-a", "--annotations", default="", help="已有 .json 目录 (existing_json 模式)")
    p_rv.add_argument("--yolo-conf", type=float, default=0.12)
    p_rv.add_argument("--yolo-iou", type=float, default=0.60)
    p_rv.add_argument("--imgsz", type=int, default=640)
    p_rv.add_argument("--device", default="0")
    p_rv.add_argument("--provider", choices=["gemini", "openai_compat"], default="gemini")
    p_rv.add_argument("--model", default="gemini-3.8-flash")
    p_rv.add_argument("--api-key", default="")
    p_rv.add_argument("--base-url", default="")
    p_rv.add_argument("--classes", default="", help="逗号分隔的类别名称")
    p_rv.add_argument("--guideline", default="", help="附加标注判定规则")
    p_rv.add_argument("--high-conf-skip", type=float, default=0.85)
    p_rv.add_argument("--no-check-missing", action="store_true")
    p_rv.add_argument("--no-add-missing", action="store_true")
    p_rv.add_argument("--no-crops", action="store_true")
    p_rv.add_argument("--workers", type=int, default=2)
    p_rv.add_argument("--flat", action="store_true", help="不按状态分建子目录")
    p_rv.set_defaults(func=_cmd_review)

    return parser


def main() -> int:
    """解析命令行参数并分派执行对应的子命令处理函数。"""
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
