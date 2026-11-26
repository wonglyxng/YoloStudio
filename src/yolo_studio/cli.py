"""无头批处理命令行入口模块，覆盖视频抽帧、数据集构建、YOLO26 训练、模型导出与 LLM 智能标注。

职责与真实入口：
    提供 `build_parser()` 与 `main(argv)`，既作为已安装命令 `yolo-studio-cli` 的入口点，
    也供仓库根目录脚本 `cli.py` 直接委托调用（命令示例见 `docs/001_录制视频抽帧工具使用说明.md`
    与 `docs/002_YOLO26训练使用说明.md`）。

运行环境与协作对象：
    需 Python >= 3.11 及对应子命令的运行库（`opencv-contrib-python-headless`、`Pillow`、`PyYAML`、
    `ultralytics`、`torch`、`google-genai` / `openai`、`pydantic`）；
    `review` 子命令调用远程多模态接口时还需通过 `--api-key` 或环境变量
    `GEMINI_API_KEY` / `OPENAI_API_KEY` 提供有效凭据。
    本模块按子命令延迟导入具体核心引擎，使 `--help` 与轻量命令免于加载无关重型依赖；不启动 PyQt6 GUI。

输入输出与失败方式：
    按子命令从本地视频、图片、标注、`data.yaml` 或 `.pt` 权重读取输入，并在指定输出目录或权重同目录写出
    帧图、YOLO 数据集（含 `data.yaml` 与 `classes.txt`）、训练权重、部署导出文件（`.onnx`/`.torchscript`/
    `.engine` 等）或 LLM 智能标注分流结果与报告。
    参数非法时由 `argparse` 直接终止或返回退出码 `2`；`extract` 出现视频错误或 `review` 存在单图处理异常时返回
    退出码 `1`；缺失必需文件、权重或凭据时核心引擎抛出异常并导致进程非零退出。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from yolo_studio._version import __version__


def _cmd_extract(args: argparse.Namespace) -> int:
    """执行 `extract` 子命令并将抽帧异常打印到 stderr；无错误返回 0，否则返回 1。"""
    from yolo_studio.core.frame_extractor import (
        ExtractionConfig,
        ExtractionMode,
        extract_frames,
    )

    cfg = ExtractionConfig(
        output_dir=Path(args.output),
        mode=ExtractionMode(args.mode),
        interval_seconds=args.interval,
        every_n_frames=args.step,
        target_count=args.count,
        start_sec=args.start,
        end_sec=args.end if args.end > 0 else None,
        dedup_threshold=args.dedup,
        max_dimension=args.max_dim,
        image_format=args.format,
        jpeg_quality=args.quality,
        prefix=args.prefix,
        naming_rule=args.naming_rule,
        create_video_subfolder=args.subfolder,
    )
    res = extract_frames([Path(p) for p in args.inputs], cfg)
    print(
        f"处理视频: {res.videos_processed}, 保存帧数: {res.frames_saved}, 去重跳过: {res.frames_skipped_similar}"
    )
    for err in res.errors:
        print(f"[ERROR] {err}", file=sys.stderr)
    return 0 if not res.errors else 1


def _cmd_dataset(args: argparse.Namespace) -> int:
    """执行 `dataset` 子命令生成 YOLO 数据集并输出划分统计，成功返回 0。"""
    from yolo_studio.core.dataset_converter import (
        AnnotationFormat,
        DatasetBuildConfig,
        build_yolo_dataset,
    )
    from yolo_studio.core.mllm_reviewer import BUCKET_NEED_REVIEW

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


def _build_cli_train_config(args: argparse.Namespace) -> "object":
    """按『默认值 → `--preset` 预设 → 显式 CLI 选项 → `--extra-arg` 附加项』的优先级构建 `TrainConfig`。

    失败方式：
        `--extra-arg` 语法非法或预设名无效时抛出 `ValueError`。
    """
    from yolo_studio.core.trainer import (
        TrainConfig,
        apply_hyp_preset,
        parse_extra_train_args,
    )

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
    if getattr(args, "preset", ""):
        apply_hyp_preset(cfg, args.preset)

    override_fields = (
        "optimizer",
        "lr0",
        "lrf",
        "warmup_epochs",
        "nbs",
        "patience",
        "workers",
        "seed",
        "cos_lr",
        "amp",
        "cache",
        "deterministic",
        "plots",
        "val",
        "mosaic",
        "close_mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    )
    for field_name in override_fields:
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(cfg, field_name, val)

    merged_extra: dict[str, object] = {}
    for raw_item in getattr(args, "extra_arg", None) or []:
        merged_extra.update(parse_extra_train_args(raw_item))
    cfg.extra_args = merged_extra
    return cfg


def _cli_print_line(line: str) -> None:
    """将训练日志行直接写入原始标准输出流，保留前导缩进并避免被当前线程捕获器二次剥离空格。"""
    target = getattr(sys.stdout, "original", None) or sys.stdout
    if target is not None:
        target.write(f"{line}\n")
        target.flush()


def _cmd_train(args: argparse.Namespace) -> int:
    """执行 `train` 子命令启动 YOLO26 训练并将超参数与实时日志打印到标准输出，成功返回 0。"""
    from yolo_studio.core.trainer import TrainConfig, train_yolo26

    try:
        cfg = _build_cli_train_config(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    assert isinstance(cfg, TrainConfig)
    res = train_yolo26(cfg, log_callback=_cli_print_line)
    print(f"训练完成，输出目录: {res.save_dir}, best.pt: {res.best_weights}")
    return 0


def _build_cli_export_config(args: argparse.Namespace) -> "object":
    """解析并校验 `export` 子命令参数，构建对应的 `ExportConfig`。

    失败方式：
        `--imgsz`、`--nms`、`--batch`、`--opset`、`--conf`、`--iou`、`--workspace`、
        `--preprocess`、`--pad-value` 或 `--extra-arg` 参数非法时抛出中文 `ValueError`。
    """
    from yolo_studio.core.trainer import (
        ExportConfig,
        build_export_kwargs,
        normalize_export_format,
        normalize_export_imgsz,
        normalize_padding_value,
        parse_export_nms_mode,
        parse_export_preprocess_mode,
        parse_extra_train_args,
    )

    imgsz: int | tuple[int, int] | None = None
    if getattr(args, "imgsz", None):
        pair = normalize_export_imgsz(args.imgsz)
        imgsz = pair[0] if pair[0] == pair[1] else pair

    merged_extra: dict[str, object] = {}
    for raw_item in getattr(args, "extra_arg", None) or []:
        merged_extra.update(parse_extra_train_args(raw_item))

    cfg = ExportConfig(
        weights_path=Path(args.weights),
        export_format=normalize_export_format(args.format),
        imgsz=imgsz,
        batch=int(args.batch),
        dynamic=bool(args.dynamic),
        nms=parse_export_nms_mode(args.nms),
        max_det=int(args.max_det),
        conf=args.conf,
        iou=args.iou,
        opset=args.opset,
        simplify=bool(args.simplify),
        half=bool(args.half),
        device=str(args.device),
        workspace=args.workspace,
        preprocess_mode=parse_export_preprocess_mode(getattr(args, "preprocess", None)),
        padding_value=normalize_padding_value(getattr(args, "pad_value", None)),
        preprocess_auto_rect=getattr(args, "preprocess_auto_rect", None),
        extra_args=merged_extra,
    )
    # 提前执行纯参数校验，避免等到加载模型后才暴露非法选项
    build_export_kwargs(cfg)
    return cfg


def _cmd_export(args: argparse.Namespace) -> int:
    """执行 `export` 子命令：读取 `.pt` 权重元数据并导出部署模型（或仅预览元数据与 IO 形状）。"""
    from yolo_studio.core.trainer import (
        ExportConfig,
        build_export_kwargs,
        describe_export_io,
        export_model_with_config,
        format_imgsz_text,
        format_submitted_export_config,
        format_weights_metadata_log,
        inspect_weights_metadata,
    )

    try:
        cfg = _build_cli_export_config(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    assert isinstance(cfg, ExportConfig)
    if getattr(args, "inspect_only", False):
        meta = inspect_weights_metadata(cfg.weights_path)
        _cli_print_line(format_weights_metadata_log(meta))
        export_kwargs, build_warnings = build_export_kwargs(cfg, metadata=meta)
        io_summary, io_warnings = describe_export_io(cfg, metadata=meta)
        all_warnings: list[str] = []
        if cfg.imgsz is None and meta.train_imgsz is not None:
            all_warnings.append(
                f"未显式指定 --imgsz，将自动沿用权重训练尺寸: {format_imgsz_text(meta.train_imgsz)}。"
            )
        for item in [*build_warnings, *io_warnings]:
            if item not in all_warnings:
                all_warnings.append(item)
        _cli_print_line(
            format_submitted_export_config(
                export_kwargs, io_summary=io_summary, warnings=all_warnings
            )
        )
        return 0

    exported = export_model_with_config(cfg, log_callback=_cli_print_line)
    print(f"导出完成，输出文件: {exported}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    """执行 `review` 子命令运行 LLM 智能标注与可选边框紧贴修正；无单图错误返回 0，否则返回 1。"""
    from yolo_studio.core.mllm_reviewer import (
        BoxRefineMode,
        MllmProvider,
        MllmReviewConfig,
        ReasoningEffort,
        ReviewSourceMode,
        run_mllm_review,
    )

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
        reasoning_effort=ReasoningEffort(args.reasoning_effort),
        allowed_classes=classes,
        custom_guideline=args.guideline,
        high_conf_skip=args.high_conf_skip,
        check_missing=not args.no_check_missing,
        add_missing_boxes=not args.no_add_missing,
        include_crops=not args.no_crops,
        max_workers=args.workers,
        max_images=max(0, args.max_images),
        triage_subfolders=not args.flat,
        recursive=args.recursive,
        resume_from_existing=bool(args.resume),
        max_consecutive_errors=max(0, int(args.max_consecutive_errors)),
        box_refine_mode=BoxRefineMode(args.box_refine),
        sam_model_name=args.sam_model,
        sam_refine_all_kept=args.sam_refine_all,
        sam3_enable_llm_review=args.sam3_llm_review,
    )
    res = run_mllm_review(cfg, log_callback=print)
    refined_total = sum(r.refined_count for r in res.records)
    resume_suffix = f", 续传跳过={res.resumed_skipped_count}" if res.resumed_skipped_count > 0 else ""
    print(
        f"LLM 智能标注完成: 总计={res.total_images}{resume_suffix}, 自动通过={res.auto_passed_count}, "
        f"已删改/收框={res.llm_fixed_count}(收紧 {refined_total} 框), "
        f"待人工复核={res.need_review_count}, 出错={res.error_count}"
    )
    if res.aborted_reason:
        print(f"[熔断停机] {res.aborted_reason}")
    return 0 if (res.error_count == 0 and not res.aborted_reason) else 1


def build_parser() -> argparse.ArgumentParser:
    """构建包含 `extract`、`dataset`、`train`、`export`、`review` 五个子命令的 CLI 解析器。"""
    parser = argparse.ArgumentParser(description=f"YOLO26 视频抽帧与训练 CLI 工具 (v{__version__})")
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"YoloStudio v{__version__}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="从录制视频批量抽帧")
    p_ext.add_argument("inputs", nargs="+", help="视频文件或包含视频的目录")
    p_ext.add_argument("-o", "--output", default="data/raw_frames", help="输出目录")
    p_ext.add_argument(
        "--mode",
        choices=["interval_seconds", "every_n_frames", "target_count", "random_total_count"],
        default="interval_seconds",
    )
    p_ext.add_argument("--interval", type=float, default=1.0, help="按秒抽帧间隔")
    p_ext.add_argument("--step", type=int, default=30, help="按帧步长")
    p_ext.add_argument(
        "--count",
        type=int,
        default=100,
        help="目标帧数（target_count 为每视频帧数，random_total_count 为所有视频合计总帧数）",
    )
    p_ext.add_argument("--start", type=float, default=0.0, help="截取起始时间（秒）")
    p_ext.add_argument("--end", type=float, default=0.0, help="截取结束时间（秒，0=到视频结尾）")
    # 交互式 CLI 与桌面 GUI 默认开启 2.0 静止帧去重，底层 ExtractionConfig API 默认 0.0
    p_ext.add_argument("--dedup", type=float, default=2.0, help="相似帧去重阈值 (0=关闭)")
    p_ext.add_argument("--max-dim", type=int, default=0, help="最长边缩放像素 (0=原尺寸)")
    p_ext.add_argument("--format", choices=["jpg", "png"], default="jpg")
    p_ext.add_argument("--quality", type=int, default=95, help="JPG 压缩质量 (1..100)")
    p_ext.add_argument(
        "--naming-rule",
        default="",
        help=(
            "自定义命名规则（留空默认 vc{hash}_f{ms:010d}；填普通前缀如 cam1_ 按前缀+视频名+序号，"
            "或使用占位符 {video}/{hash}/{hash_full}/{ms}/{frame}/{seq}/{index}/{prefix}）"
        ),
    )
    p_ext.add_argument(
        "--prefix",
        default="",
        help="输出帧文件名前缀（兼容选项；未指定 --naming-rule 时生效，或在模板中以 {prefix} 引用）",
    )
    p_ext.add_argument("--subfolder", action="store_true", help="按视频名在输出目录下分建子目录")
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
    p_ds.add_argument("-r", "--recursive", action="store_true", help="递归扫描图片子目录（如 LLM 智能标注分流目录）")
    p_ds.add_argument(
        "--include-need-review",
        action="store_true",
        help="递归扫描时同时纳入 need_review 子目录（默认排除未人工复核的样本）",
    )
    p_ds.set_defaults(func=_cmd_dataset)

    p_tr = sub.add_parser("train", help="启动 YOLO26 模型训练")
    p_tr.add_argument("-d", "--data", required=True, help="data.yaml 路径")
    p_tr.add_argument("-m", "--model", default="yolo26s.pt", help="YOLO26 权重文件；续训时可直接指定 last.pt")
    p_tr.add_argument(
        "--preset",
        choices=("default", "game_ui", "small_sample_finetune"),
        default="",
        help="可选超参数预设：default(通用自然场景) / game_ui(游戏与UI弱增强) / small_sample_finetune(小样本微调)",
    )
    p_tr.add_argument("--epochs", type=int, default=100)
    p_tr.add_argument("--imgsz", type=int, default=640)
    p_tr.add_argument("--batch", type=int, default=16)
    p_tr.add_argument("--nbs", type=int, default=None, help="名义批大小 (梯度累积基准，默认 64，game_ui 预设 16)")
    p_tr.add_argument("--device", default="0")
    p_tr.add_argument("--workers", type=int, default=None, help="DataLoader 工作进程数 (默认 4)")
    p_tr.add_argument("--optimizer", default=None, help="优化器：auto / AdamW / SGD / MuSGD / Adam 等")
    p_tr.add_argument("--lr0", type=float, default=None, help="初始学习率 (需配合非 auto 优化器严格固定)")
    p_tr.add_argument("--lrf", type=float, default=None, help="最终学习率比例 (终值 = lr0 * lrf)")
    p_tr.add_argument("--warmup-epochs", type=float, default=None, help="学习率预热轮数")
    p_tr.add_argument("--patience", type=int, default=None, help="早停容忍轮数 (默认 50)")
    p_tr.add_argument("--seed", type=int, default=None, help="随机种子 (默认 0)")
    p_tr.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=None, help="是否启用余弦退火调度")
    p_tr.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="是否启用混合精度 AMP")
    p_tr.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None, help="是否将图片缓存至内存")
    p_tr.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=None, help="是否启用确定性算子"
    )
    p_tr.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None, help="是否保存训练评估图表")
    p_tr.add_argument("--val", action=argparse.BooleanOptionalAction, default=None, help="是否在训练期间运行验证")
    p_tr.add_argument("--mosaic", type=float, default=None, help="Mosaic 四图拼接增强概率 (0.0~1.0)")
    p_tr.add_argument("--close-mosaic", type=int, default=None, help="最后 N 轮关闭 Mosaic")
    p_tr.add_argument("--mixup", type=float, default=None, help="MixUp 混合增强概率")
    p_tr.add_argument("--cutmix", type=float, default=None, help="CutMix 混合增强概率")
    p_tr.add_argument("--copy-paste", type=float, default=None, help="目标复制粘贴增强比例")
    p_tr.add_argument("--degrees", type=float, default=None, help="随机旋转角度范围 (+/-)")
    p_tr.add_argument("--translate", type=float, default=None, help="随机平移比例")
    p_tr.add_argument("--scale", type=float, default=None, help="随机缩放增益比例")
    p_tr.add_argument("--shear", type=float, default=None, help="随机错切角度范围 (+/-)")
    p_tr.add_argument("--perspective", type=float, default=None, help="随机透视变换比例")
    p_tr.add_argument("--flipud", type=float, default=None, help="上下翻转概率")
    p_tr.add_argument("--fliplr", type=float, default=None, help="左右翻转概率")
    p_tr.add_argument("--hsv-h", type=float, default=None, help="HSV 色调抖动幅度")
    p_tr.add_argument("--hsv-s", type=float, default=None, help="HSV 饱和度抖动幅度")
    p_tr.add_argument("--hsv-v", type=float, default=None, help="HSV 明度抖动幅度")
    p_tr.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="透传任意其他 Ultralytics 参数（可多次指定，支持 JSON 或 key=value，如 --extra-arg box=7.5）",
    )
    p_tr.add_argument("--project", default="runs/detect")
    p_tr.add_argument("--name", default="yolo26_train")
    p_tr.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    p_tr.add_argument("--exist-ok", action="store_true", help="允许覆盖同名实验目录（默认自动递增编号）")
    p_tr.set_defaults(func=_cmd_train)

    p_rv = sub.add_parser("review", help="LLM 智能标注（基准模型预标 / 已有 .json / 零样本 / SAM 3 文本自动标注）")
    p_rv.add_argument("-i", "--images", required=True, help="待标注图片目录")
    p_rv.add_argument("-o", "--output", default="data/smart_labeled", help="智能标注输出目录")
    p_rv.add_argument(
        "--source-mode",
        choices=["yolo_predict", "existing_json", "llm_zero_shot", "sam3_text"],
        default="yolo_predict",
        help="候选标注来源：yolo_predict(基准模型预标) / existing_json(已有.json) / llm_zero_shot(零样本) / sam3_text(SAM 3文本自动标注)",
    )
    p_rv.add_argument(
        "-w",
        "--weights",
        default="",
        help="模型 .pt 路径（yolo_predict 传基准 best.pt；sam3_text 传本地 sam3.pt，留空自动识别 weights/sam3.pt）",
    )
    p_rv.add_argument("-a", "--annotations", default="", help="已有 .json 目录 (existing_json 模式)")
    p_rv.add_argument("--yolo-conf", type=float, default=0.12)
    p_rv.add_argument("--yolo-iou", type=float, default=0.60)
    p_rv.add_argument("--imgsz", type=int, default=640)
    p_rv.add_argument("--device", default="0")
    p_rv.add_argument("--provider", choices=["gemini", "openai", "openai_compat"], default="gemini")
    p_rv.add_argument("--model", default="gemini-4-hydrogen")
    p_rv.add_argument("--api-key", default="")
    p_rv.add_argument("--base-url", default="")
    p_rv.add_argument(
        "--reasoning-effort",
        choices=["auto", "off", "low", "medium", "high", "xhigh"],
        default="auto",
        help="思考/推理深度级别：auto(模型默认) / off(关闭或极简) / low / medium / high / xhigh",
    )
    p_rv.add_argument(
        "--classes",
        default="",
        help="逗号分隔的目标类别名称（llm_zero_shot 与 sam3_text 模式必填；sam3_text 支持 类别名=英文提示词）",
    )
    p_rv.add_argument(
        "--sam3-llm-review",
        action="store_true",
        help="仅 sam3_text 模式生效：叠加 LLM 按业务规则执行二次质检（默认不开启，纯本地免 API 费用）",
    )
    p_rv.add_argument("--guideline", default="", help="附加标注判定规则")
    p_rv.add_argument("--high-conf-skip", type=float, default=0.85)
    p_rv.add_argument("--no-check-missing", action="store_true")
    p_rv.add_argument("--no-add-missing", action="store_true")
    p_rv.add_argument("--no-crops", action="store_true")
    p_rv.add_argument(
        "--box-refine",
        choices=["off", "flag_loose", "mllm_crop", "sam_local"],
        default="off",
        help="边框紧贴标准化修正模式：off / flag_loose / mllm_crop / sam_local",
    )
    p_rv.add_argument("--sam-model", default="sam2.1_t.pt", help="sam_local 模式使用的 SAM 权重名或本地 .pt 路径")
    p_rv.add_argument(
        "--sam-refine-all",
        action="store_true",
        help="sam_local 模式下对全部保留框执行像素级收紧（默认仅收紧 LLM 判定过松的框）",
    )
    p_rv.add_argument("--workers", type=int, default=2)
    p_rv.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="最多处理前 N 张图片用于小批量试跑（0 表示处理目录中全部图片）",
    )
    p_rv.add_argument("--flat", action="store_true", help="不按状态分建子目录")
    p_rv.add_argument("-r", "--recursive", action="store_true", help="递归扫描输入子目录")
    p_rv.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用断点续传：复用输出目录 review_report.json 中已成功且输出完整的图片，仅补跑剩余与出错图片",
    )
    p_rv.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="连续网络/限流/服务端异常达到该阈值时自动熔断停机（默认 5；设为 0 关闭连续故障熔断）",
    )
    p_rv.set_defaults(func=_cmd_review)

    p_exp = sub.add_parser("export", help="查看权重元数据并导出部署模型 (ONNX / TorchScript / TensorRT 等)")
    p_exp.add_argument("-w", "--weights", required=True, help="待导出的 .pt 权重文件路径")
    p_exp.add_argument(
        "-f",
        "--format",
        default="onnx",
        help="目标导出格式：onnx / torchscript / engine (tensorrt) / openvino 等（默认 onnx）",
    )
    p_exp.add_argument(
        "--imgsz",
        default=None,
        help="导出输入尺寸：单整数（如 960）或 高,宽（如 544,960 / 544x960）；不指定时自动沿用 .pt 权重训练尺寸",
    )
    p_exp.add_argument(
        "--batch",
        type=int,
        default=1,
        help="静态批大小（TensorRT 开启动态输入时建议设为最大批大小，默认 1）",
    )
    p_exp.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用动态输入尺寸与动态批大小 (dynamic)",
    )
    p_exp.add_argument(
        "--nms",
        choices=("raw", "end2end", "embedded"),
        default="raw",
        help="输出头与 NMS 模式：raw(默认原始输出，外部NMS) / end2end(YOLO26原生无NMS) / embedded(图内嵌入NMS)",
    )
    p_exp.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="end2end 或 embedded 模式下单图最大输出检测框数（默认 300）",
    )
    p_exp.add_argument(
        "--conf",
        type=float,
        default=None,
        help="仅 embedded 模式生效：图内嵌入 NMS 的置信度阈值（默认 0.25）",
    )
    p_exp.add_argument(
        "--iou",
        type=float,
        default=None,
        help="仅 embedded 模式生效：图内嵌入 NMS 的 IoU 阈值（默认 0.50）",
    )
    p_exp.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset 版本号（7~22；留空自动选择，非 ONNX 系列格式自动忽略）",
    )
    p_exp.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 onnxslim 简化 ONNX 计算图（默认开启，非 ONNX 系列格式自动忽略）",
    )
    p_exp.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用 FP16 半精度导出（需 GPU 模式且非 ONNX CPU 导出）",
    )
    p_exp.add_argument(
        "--workspace",
        type=float,
        default=None,
        help="仅 TensorRT (engine) 生效：工作空间显存上限 (GiB)",
    )
    p_exp.add_argument("--device", default="0", help="导出计算设备（如 0 或 cpu；ONNX 固定使用 cpu）")
    p_exp.add_argument(
        "--preprocess",
        choices=("letterbox_center", "letterbox_top_left", "stretch"),
        default="letterbox_center",
        help="写入 ONNX 部署元数据的预处理策略：letterbox_center(居中填充，默认) / letterbox_top_left(左上靠边) / stretch(直接拉伸)",
    )
    p_exp.add_argument(
        "--pad-value",
        type=int,
        default=114,
        help="写入 ONNX 部署元数据的 LetterBox 补边灰度值（0~255，默认 114；stretch 模式忽略）",
    )
    p_exp.add_argument(
        "--preprocess-auto-rect",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="显式指定写入 ONNX 部署元数据的 auto_rect 标志（留空按 dynamic 与预处理策略自动推导）",
    )
    p_exp.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="透传任意其他 Ultralytics 导出参数（可多次指定，支持 JSON 或 key=value）",
    )
    p_exp.add_argument(
        "--inspect-only",
        action="store_true",
        help="仅读取并打印 .pt 权重训练尺寸、类别顺序与预期导出输入/输出形状，不执行实际导出",
    )
    p_exp.set_defaults(func=_cmd_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数并分派执行对应的子命令处理函数。"""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)
