"""Ultralytics YOLO26 模型训练、验证、推理预览、模型元数据探测与导出引擎。

职责：
    封装本地硬件探测、`.pt` 权重及已导出模型（`.onnx` / `.torchscript` / `.engine`）元数据只读解析
    （训练/导出尺寸、类别顺序、动态维度与检测头能力），以及 YOLO26 模型的训练、验证、
    图片/视频推理（支持检测头模式、填充方式及紧凑防重叠半透明绘制配置）和部署格式导出，
    并将训练、验证、推理与导出引擎的标准输出和 Epoch 指标实时桥接至调用方回调。

运行环境：
    Python >= 3.11，依赖 `torch`、`ultralytics`、`numpy`、`opencv-python`（可选 `Pillow` 用于紧凑 TrueType 标签渲染）；
    推理已导出 `.onnx` 模型时依赖 `onnxruntime`（或 `onnxruntime-gpu`）；
    使用预训练名称（如 `yolo26s.pt`）且本地缺失权重时，Ultralytics 会触发模型下载。

协作对象与职责边界：
    为 `cli._cmd_train`、`cli._cmd_export`、`gui.workers`（`TrainWorker`、`ValidateWorker`、
    `PredictWorker`、`ExportWorker`、`WeightsInspectWorker`）、`gui.tab_train.TrainTab`、
    `gui.tab_eval.EvalTab` 与 `gui.tab_review.MllmReviewTab` 提供底层能力。
    仅负责引擎调度与纯数据转换，不管理线程或 UI 控件。
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event

import cv2
import numpy as np

from yolo_studio.core.frame_extractor import VIDEO_EXTENSIONS
from yolo_studio.core.log_capture import capture_thread_output

YOLO26_DETECT_MODELS = [
    ("yolo26n.pt", "YOLO26n (Nano - 速度最快，显存占用最低)"),
    ("yolo26s.pt", "YOLO26s (Small - 速度与精度均衡，RTX 3060 推荐)"),
    ("yolo26m.pt", "YOLO26m (Medium - 更高精度)"),
    ("yolo26l.pt", "YOLO26l (Large - 高精度，建议适当减小 batch)"),
    ("yolo26x.pt", "YOLO26x (XLarge - 最高精度，显存需求较高)"),
]

YOLO26_OPTIMIZERS: list[tuple[str, str]] = [
    ("auto", "auto (自动选择，会覆盖自定义 LR0)"),
    ("AdamW", "AdamW (小样本/微调推荐，建议 LR0=0.001)"),
    ("SGD", "SGD (常规大数据集，建议 LR0=0.01)"),
    ("MuSGD", "MuSGD (混合动量 SGD)"),
    ("Adam", "Adam"),
    ("NAdam", "NAdam"),
    ("RAdam", "RAdam"),
    ("RMSProp", "RMSProp"),
]

TRAIN_PRESET_ITEMS: list[tuple[str, str]] = [
    ("default", "通用自然场景 (Ultralytics 默认)"),
    ("game_ui", "游戏画面 / UI 弱增强 (去畸变 + AdamW)"),
    ("small_sample_finetune", "小样本 / 小目标平稳微调 (AdamW + 余弦退火)"),
]

# 各训练场景预设对应的 `TrainConfig` 超参数覆盖表；未列出的字段（如路径、设备、轮数、批大小）保持用户当前设置
TRAIN_HYP_PRESETS: dict[str, dict[str, object]] = {
    "default": {
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "warmup_epochs": 3.0,
        "nbs": 64,
        "cos_lr": False,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
    },
    "game_ui": {
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.05,
        "warmup_epochs": 3.0,
        "nbs": 16,
        "cos_lr": False,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.03,
        "scale": 0.15,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.01,
        "hsv_s": 0.2,
        "hsv_v": 0.2,
    },
    "small_sample_finetune": {
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.05,
        "warmup_epochs": 3.0,
        "nbs": 16,
        "cos_lr": True,
        "mosaic": 0.5,
        "close_mosaic": 15,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.25,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.25,
    },
}


@dataclass
class DeviceInfo:
    """可用计算硬件与显存摘要；`vram_gb` 单位为 GiB，`probing=True` 表示后台异步探测尚在进行中。

    `gpu_devices` 按索引顺序记录本机可用 CUDA GPU 清单，每项为 `(设备编号字符串, 显卡名称, 显存 GiB)`；
    无可用 GPU 或尚未完成探测时为空元组。
    """

    cuda_available: bool
    device_count: int
    device_name: str
    vram_gb: float
    recommended_device: str
    probing: bool = False
    gpu_devices: tuple[tuple[str, str, float], ...] = ()


@dataclass
class TrainConfig:
    """YOLO26 训练任务超参数与路径配置。

    关键字段说明：
        核心与硬件：
            `batch`：批大小，设为 `-1` 时由 Ultralytics 自动探测最大可用批次。
            `nbs`：名义批大小（Nominal Batch Size），梯度累积步数近似为 `max(round(nbs / batch), 1)`。
            `workers`：DataLoader 工作进程数（Windows 下设为 `0` 可改为主进程直接加载）。
        优化器与学习率调度：
            `optimizer`：优化器名称（如 `"auto"`、`"AdamW"`、`"SGD"`）；注意设为 `"auto"` 时 Ultralytics
                会自动选定优化器并覆盖 `lr0`，显式指定非 `"auto"` 优化器时 `lr0` 按原值生效。
            `lr0` / `lrf`：初始学习率与最终学习率比例（终值 `lr0 * lrf`）；`warmup_epochs`：预热轮数；
            `cos_lr`：是否启用余弦退火调度；`patience`：验证指标无提升的早停容忍轮数（设为 `0` 或等于 `epochs` 关闭早停）。
        数据增强与复现性：
            `mosaic` / `close_mosaic` / `mixup` / `cutmix` / `copy_paste`：多图合成增强概率及末期关闭轮数。
            `degrees` / `translate` / `scale` / `shear` / `perspective` / `flipud` / `fliplr`：几何变换幅度与翻转概率。
            `hsv_h` / `hsv_s` / `hsv_v`：色调、饱和度与明度抖动比例。
            `seed` / `deterministic`：随机种子与确定性算子开关；`plots` / `save` / `val` / `pretrained`：工程运行开关。
        路径、续训与扩展参数：
            `project_dir` / `run_name`：训练产物保存根目录与子实验名。
            `exist_ok`：默认 `False`，同名实验目录已存在时自动递增为 `run_name2` 等，避免覆盖历史产物。
            `resume`：断点续训。优先使用非官方预训练的自定义 `.pt`（如 `last.pt`），否则查找
                `project_dir/run_name/weights/last.pt`，缺失时训练前抛出 `FileNotFoundError`。
            `extra_args`：透传给 `model.train()` 的任意附加参数字典，同名键优先级高于本配置类的同名字段。
    """

    data_yaml: Path
    model_name: str = "yolo26s.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    lr0: float = 0.01
    patience: int = 50
    workers: int = 4
    amp: bool = True
    cache: bool = False
    project_dir: Path = Path("runs/detect")
    run_name: str = "yolo26_train"
    resume: bool = False
    exist_ok: bool = False
    optimizer: str = "auto"
    lrf: float = 0.01
    warmup_epochs: float = 3.0
    nbs: int = 64
    cos_lr: bool = False
    mosaic: float = 1.0
    close_mosaic: int = 10
    mixup: float = 0.0
    cutmix: float = 0.0
    copy_paste: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: bool | float = 0.0
    fliplr: bool | float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    seed: int = 0
    deterministic: bool = True
    plots: bool = True
    pretrained: bool = True
    save: bool = True
    val: bool = True
    extra_args: dict[str, object] = field(default_factory=dict)


@dataclass
class EpochMetrics:
    """单个训练轮次或训练结束后 `best.pt` 最终评估提取的关键指标快照（`epoch` 从 1 开始计数）。

    字段说明：
        `is_final_eval`：为 `False` 时表示常规训练轮次结束快照；为 `True` 时表示来自 Ultralytics
            `BaseTrainer.final_eval()` 对 `best.pt` 执行的最终验证，此时 `epoch` 保持为实际已完成的
            最后一轮轮次（避免因 Ultralytics 临时 `epoch += 1` 记日志而多出 1 步），验证精度字段
            对应 `best.pt` 最终评估结果，不应用于覆盖最后一轮训练指标。
    """

    epoch: int
    total_epochs: int
    box_loss: float = 0.0
    cls_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    is_final_eval: bool = False


@dataclass
class TrainResult:
    """YOLO26 训练结束或安全停训后的结果摘要。"""

    save_dir: Path
    best_weights: Path | None
    last_weights: Path | None
    metrics: dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False


# 进程内缓存的本机硬件探测快照；首次调用 `detect_compute_device` 时写入，`refresh=True` 时重查
_CACHED_DEVICE_INFO: DeviceInfo | None = None


def ensure_torch_cpu_threads() -> None:
    """恢复 PyTorch CPU 算子的线程数。

    设计原因：
        桌面入口为降低主线程 BLAS/OpenMP 线程争抢设置了 `OMP_NUM_THREADS=1` 等环境变量，
        导致 torch 默认只用 1 个线程，CPU 训练/推理显著变慢。本函数在当前线程数为 1 时
        把 torch 进程级线程数恢复为约等于物理核心数（逻辑核数的一半，至少 1）。
        仅影响 torch，不影响 numpy/MKL 与 onnxruntime；可重复调用。
    """
    try:
        import torch

        if torch.get_num_threads() <= 1:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    except Exception:
        # 线程数设置失败不影响功能，仅保留默认值
        pass


def _resolve_resume_checkpoint(config: TrainConfig) -> Path:
    """确定断点续训使用的 `last.pt`，不存在时抛出 `FileNotFoundError`。"""
    official_names = {name for name, _ in YOLO26_DETECT_MODELS}
    candidate = Path(config.model_name)
    if candidate.name not in official_names and candidate.is_file():
        return candidate
    default_ckpt = config.project_dir / config.run_name / "weights" / "last.pt"
    if default_ckpt.is_file():
        return default_ckpt
    raise FileNotFoundError(
        f"断点续训需要上一次训练的 last.pt，未找到: {default_ckpt}。"
        "请确认实验名称与上次一致，或在自定义权重路径中直接选择要续训的 last.pt。"
    )


WEIGHTS_DIR_ENV_VAR = "YOLO_STUDIO_WEIGHTS_DIR"


def resolve_pretrained_weights(
    model_name: str, workspace_root: Path | None = None
) -> str:
    """优先将裸预训练权重名（如 `yolo26s.pt`）解析为本地预置文件绝对路径，未找到时原样返回。

    查找顺序：
        1. `model_name` 本身已是本地存在的非空文件时直接返回；
        2. 若 `model_name` 为不含目录层级的纯文件名，依次检索环境变量 `YOLO_STUDIO_WEIGHTS_DIR`、
           `workspace_root / "weights"`、`Path.cwd() / "weights"`、应用代码根 `weights/`
           以及独立便携包根 `<便携包根>/weights/` 下的同名文件；
        3. 全部未命中时原样返回 `model_name`，交由 Ultralytics 按默认机制下载或解析。
    """
    direct = Path(model_name)
    if direct.is_file() and direct.stat().st_size > 0:
        return str(direct.resolve())

    if direct.name != model_name or "/" in model_name or "\\" in model_name:
        return model_name

    code_root = Path(__file__).resolve().parents[3]
    search_dirs: list[Path] = []
    env_weights = os.environ.get(WEIGHTS_DIR_ENV_VAR, "").strip()
    if env_weights:
        search_dirs.append(Path(env_weights).expanduser())
    if workspace_root is not None:
        search_dirs.append(workspace_root / "weights")
    search_dirs.extend(
        [
            Path.cwd() / "weights",
            code_root / "weights",
            code_root.parent / "weights",
        ]
    )

    seen: set[Path] = set()
    for folder in search_dirs:
        try:
            resolved_dir = folder.resolve()
        except OSError:
            continue
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        candidate = resolved_dir / model_name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return str(candidate)

    return model_name


def detect_compute_device(refresh: bool = False) -> DeviceInfo:
    """检测本机 CUDA GPU 可用性并读取 0 号卡显存信息，无可用 GPU 时回退为 CPU。

    缓存与延迟导入：
        函数内部按需导入 `torch`，使无需 GPU 的轻量命令（如 `extract`、`dataset`）不必在模块加载期导入 PyTorch；
        探测结果缓存在模块级 `_CACHED_DEVICE_INFO` 中，传 `refresh=True` 时才重新查询底层设备。
    """
    global _CACHED_DEVICE_INFO
    if _CACHED_DEVICE_INFO is not None and not refresh:
        return _CACHED_DEVICE_INFO

    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        count = torch.cuda.device_count()
        gpu_list: list[tuple[str, str, float]] = []
        for idx in range(count):
            p = torch.cuda.get_device_properties(idx)
            gpu_list.append((str(idx), p.name, p.total_memory / (1024**3)))
        first_name, first_vram = gpu_list[0][1], gpu_list[0][2]
        info = DeviceInfo(
            cuda_available=True,
            device_count=count,
            device_name=first_name,
            vram_gb=first_vram,
            recommended_device="0",
            gpu_devices=tuple(gpu_list),
        )
    else:
        info = DeviceInfo(
            cuda_available=False,
            device_count=0,
            device_name="CPU (未检测到可用 CUDA GPU)",
            vram_gb=0.0,
            recommended_device="cpu",
            gpu_devices=(),
        )
    _CACHED_DEVICE_INFO = info
    return info


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        # 保留 type: ignore 指令供静态类型检查器识别运行时动态转换
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def apply_hyp_preset(config: TrainConfig, preset_name: str) -> TrainConfig:
    """按预设名称原地更新 `config` 中的优化器与数据增强超参数并返回 `config`。

    调用约束与失败行为：
        `preset_name` 须属于 `TRAIN_HYP_PRESETS`（`"default"`、`"game_ui"`、`"small_sample_finetune"`），
        否则抛出 `ValueError`。仅覆盖预设字典中声明的优化与增强字段，不改动 `data_yaml`、`model_name`、
        `epochs`、`imgsz`、`batch`、`device` 与保存目录等工程路径配置。
    """
    if preset_name not in TRAIN_HYP_PRESETS:
        valid = ", ".join(TRAIN_HYP_PRESETS.keys())
        raise ValueError(f"未知的训练超参数预设 '{preset_name}'，可选值: {valid}")
    for key, val in TRAIN_HYP_PRESETS[preset_name].items():
        setattr(config, key, val)
    return config


def _split_top_level_entries(text: str) -> list[str]:
    """按顶层逗号、分号或换行切分键值条目，忽略引号与括号内部的分隔符；引号或括号未闭合时抛出 `ValueError`。"""
    entries: list[str] = []
    buf: list[str] = []
    quote_char: str | None = None
    escaped = False
    depth = 0
    pairs = {"[": "]", "{": "}", "(": ")"}
    closing = {"]", "}", ")"}

    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if quote_char is not None:
            buf.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = None
            continue

        if ch in {"'", '"'}:
            quote_char = ch
            buf.append(ch)
        elif ch in pairs:
            depth += 1
            buf.append(ch)
        elif ch in closing:
            depth -= 1
            if depth < 0:
                raise ValueError("附加超参数存在多余的右括号。")
            buf.append(ch)
        elif depth == 0 and ch in {",", ";", "\n", "\r"}:
            piece = "".join(buf).strip()
            if piece:
                entries.append(piece)
            buf.clear()
        else:
            buf.append(ch)

    if quote_char is not None or depth != 0:
        raise ValueError("附加超参数存在未闭合的引号或括号。")
    tail = "".join(buf).strip()
    if tail:
        entries.append(tail)
    return entries


def _find_key_value_sep(entry: str) -> int:
    """查找单条键值表达式中顶层第一个 `=` 或 `:` 的索引，不存在时返回 `-1`。"""
    quote_char: str | None = None
    escaped = False
    depth = 0
    for idx, ch in enumerate(entry):
        if escaped:
            escaped = False
            continue
        if quote_char is not None:
            if ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = None
            continue
        if ch in {"'", '"'}:
            quote_char = ch
        elif ch in {"[", "{", "("}:
            depth += 1
        elif ch in {"]", "}", ")"}:
            depth = max(0, depth - 1)
        elif depth == 0 and ch in {"=", ":"}:
            return idx
    return -1


def _normalize_parsed_value(val: object) -> object:
    """将 `ast.literal_eval` 可能产生的元组递归规范为列表，并校验值属于受支持的基础或容器类型。"""
    if val is None or isinstance(val, (bool, int, float, str)):
        if isinstance(val, float) and not math.isfinite(val):
            raise ValueError("超参数浮点值不能为 NaN 或无穷大。")
        return val
    if isinstance(val, (list, tuple)):
        return [_normalize_parsed_value(item) for item in val]
    if isinstance(val, dict):
        out: dict[str, object] = {}
        for k, v in val.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"字典类型的超参数键必须为非空字符串: {k!r}")
            out[k] = _normalize_parsed_value(v)
        return out
    raise ValueError(f"不支持的超参数值类型: {type(val).__name__}")


def _parse_arg_value(raw_val: str) -> object:
    """将单个超参数字符串字面量解析为布尔、`None`、整数、浮点数、列表/字典或原始字符串。"""
    val_str = raw_val.strip()
    if not val_str:
        raise ValueError("超参数值不能为空。")
    lower = val_str.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None

    if val_str[0] not in {"'", '"', "[", "{", "("}:
        if re.fullmatch(r"[+-]?\d+", val_str):
            return int(val_str)
        try:
            num = float(val_str)
            if math.isfinite(num):
                return num
            raise ValueError("超参数浮点值不能为 NaN 或无穷大。")
        except ValueError as exc:
            if "NaN" in str(exc):
                raise
        return val_str

    cleaned = re.sub(r",\s*([}\]])", r"\1", val_str)
    for loader in (json.loads, ast.literal_eval):
        try:
            return _normalize_parsed_value(loader(cleaned))
        except Exception:
            continue
    raise ValueError(f"无法解析超参数值: {raw_val!r}")


def parse_extra_train_args(raw: str) -> dict[str, object]:
    """解析用户输入的附加训练超参数文本，返回可直接合入 `TrainConfig.extra_args` 的字典。

    支持的输入格式：
        1. JSON / Python 字典格式（兼容末尾多余逗号）：如 `'{"box": 7.5, "plots": false}'`；
        2. 无外层花括号的 JSON 键值片段：如 `'\"mosaic\": 0.0, \"scale\": 0.15,'`；
        3. 简洁键值对格式（支持逗号、分号或换行分隔）：如 `'box=7.5, weight_decay=0.0005, freeze=[0, 1]'`。

    失败行为：
        空白输入返回空字典 `{}`；语法错误、未闭合括号/引号或键名非合法 Python 标识符时抛出中文 `ValueError`。
    """
    stripped = raw.strip()
    if not stripped:
        return {}

    if stripped.startswith("{"):
        body = stripped[:-1].rstrip() if stripped.endswith(",") else stripped
        cleaned = re.sub(r",\s*([}\]])", r"\1", body)
        parsed_obj: object | None = None
        parsed_ok = False
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed_obj = loader(cleaned)
                parsed_ok = True
                break
            except Exception:
                continue
        if not parsed_ok or not isinstance(parsed_obj, dict):
            raise ValueError("附加超参数 JSON/字典格式有误，请检查花括号、引号与逗号是否配对。")

        result: dict[str, object] = {}
        for k, v in parsed_obj.items():
            if not isinstance(k, str) or not k.isidentifier():
                raise ValueError(f"非法的超参数名称: {k!r}（必须为合法的参数标识符）")
            result[k] = _normalize_parsed_value(v)
        return result

    entries = _split_top_level_entries(stripped)
    result = {}
    for entry in entries:
        sep_idx = _find_key_value_sep(entry)
        if sep_idx <= 0:
            raise ValueError(f"附加超参数条目缺少 '=' 或 ':' 分隔符: {entry!r}")
        raw_key = entry[:sep_idx].strip()
        if len(raw_key) >= 2 and raw_key[0] == raw_key[-1] and raw_key[0] in {"'", '"'}:
            key = raw_key[1:-1].strip()
        else:
            key = raw_key
        if not key or not key.isidentifier():
            raise ValueError(f"非法的超参数名称: {raw_key!r}（必须为合法的参数标识符）")
        raw_val = entry[sep_idx + 1 :]
        result[key] = _parse_arg_value(raw_val)
    return result


# 桌面训练界面拥有专用表单控件的 Ultralytics 参数键集合（用于附加参数覆盖提醒与 JSON 导入分流）
TRAIN_UI_MANAGED_KEYS: frozenset[str] = frozenset(
    {
        "data",
        "model",
        "epochs",
        "imgsz",
        "batch",
        "nbs",
        "device",
        "workers",
        "project",
        "name",
        "resume",
        "optimizer",
        "lr0",
        "lrf",
        "warmup_epochs",
        "cos_lr",
        "patience",
        "seed",
        "deterministic",
        "amp",
        "cache",
        "plots",
        "val",
        "mosaic",
        "close_mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "degrees",
        "shear",
        "perspective",
        "translate",
        "scale",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    }
)

# 与具体项目目录或模型文件绑定的路径/标识键；导入外部 JSON 参数时默认保护不被误覆盖，由用户按需勾选导入
TRAIN_PATH_KEYS: frozenset[str] = frozenset({"data", "project", "name", "model"})


def parse_train_arguments_text(raw: str) -> dict[str, object]:
    """解析用于界面导入的训练参数 JSON / 日志片段，返回扁平化的参数字典。

    兼容输入形式：
        1. 标准 JSON 对象：`{"epochs": 160, "optimizer": "AdamW", ...}`；
        2. 外层含 `"arguments"` 键的对象：`{"arguments": {"epochs": 160, ...}}`；
        3. 直接截取的配置/日志片段：`"arguments": { ... },` 或带 `=== ... ===`、`[提示] ...` 装饰行的训练日志；
        4. `key=value` / `key: value` 键值对文本。

    失败方式：
        输入为空、无有效键值对或语法错误时抛出中文 `ValueError`。
    """
    filtered_lines = [
        line
        for line in raw.splitlines()
        if not line.strip().startswith("===") and not line.strip().startswith("[提示]")
    ]
    cleaned = "\n".join(filtered_lines).strip()
    if not cleaned:
        raise ValueError("请输入要导入的训练参数 JSON 或键值对文本。")

    wrapped = re.match(r'^["\']?arguments["\']?\s*[:=]\s*(\{.*)$', cleaned, flags=re.DOTALL)
    if wrapped:
        cleaned = wrapped.group(1).strip()

    parsed = parse_extra_train_args(cleaned)
    if len(parsed) == 1 and "arguments" in parsed and isinstance(parsed["arguments"], dict):
        inner: dict[str, object] = {}
        for k, v in parsed["arguments"].items():
            if not isinstance(k, str) or not k.isidentifier():
                raise ValueError(f"非法的超参数名称: {k!r}（必须为合法的参数标识符）")
            inner[k] = _normalize_parsed_value(v)
        parsed = inner

    if not parsed:
        raise ValueError("未解析到任何有效的训练超参数。")
    return parsed


def split_known_and_extra_train_args(
    args_map: Mapping[str, object], *, include_paths: bool = False
) -> tuple[dict[str, object], dict[str, object]]:
    """将导入的参数字典拆分为『界面专用控件字段』与『附加参数字段』两部分。

    参数说明：
        `include_paths`：为 `False`（默认）时跳过 `data`、`project`、`name`、`model` 路径与实验名键，
            避免导入外部项目的超参数 JSON 时误覆盖当前工作区路径；为 `True` 时一并纳入 `known`。
        部分与界面默认语义完全一致且无需放入附加参数框的常规只读布尔项（如 `save=True`、`pretrained=True`、
        `exist_ok=False`）仅在其偏离本工具默认值时才保留在 `extra` 中，保持附加参数框整洁。
    """
    implicit_defaults: dict[str, object] = {
        "save": True,
        "pretrained": True,
        "exist_ok": False,
    }
    known: dict[str, object] = {}
    extra: dict[str, object] = {}
    for key, value in args_map.items():
        if key in TRAIN_PATH_KEYS and not include_paths:
            continue
        if key in TRAIN_UI_MANAGED_KEYS:
            known[key] = value
        elif key in implicit_defaults and value == implicit_defaults[key]:
            continue
        else:
            extra[key] = value
    return known, extra


def build_train_kwargs(config: TrainConfig) -> dict[str, object]:
    """将 `TrainConfig` 规范化转换为传递给 Ultralytics `model.train(**kwargs)` 的完整参数字典。

    合并优先级：
        先按结构化字段填充基础硬件、优化调度、复现开关与数据增强参数，最后以 `config.extra_args` 覆盖同名键。
    """
    train_kwargs: dict[str, object] = {
        "data": str(config.data_yaml),
        "epochs": int(config.epochs),
        "imgsz": int(config.imgsz),
        "batch": int(config.batch),
        "nbs": int(config.nbs),
        "device": config.device,
        "workers": int(config.workers),
        "project": str(config.project_dir.resolve()),
        "name": config.run_name,
        "exist_ok": bool(config.exist_ok),
        "pretrained": bool(config.pretrained),
        "resume": bool(config.resume),
        "optimizer": str(config.optimizer).strip() or "auto",
        "lr0": float(config.lr0),
        "lrf": float(config.lrf),
        "warmup_epochs": float(config.warmup_epochs),
        "cos_lr": bool(config.cos_lr),
        "patience": int(config.patience),
        "seed": int(config.seed),
        "deterministic": bool(config.deterministic),
        "amp": bool(config.amp),
        "cache": config.cache if isinstance(config.cache, str) else bool(config.cache),
        "plots": bool(config.plots),
        "save": bool(config.save),
        "val": bool(config.val),
        "mosaic": float(config.mosaic),
        "close_mosaic": int(config.close_mosaic),
        "mixup": float(config.mixup),
        "cutmix": float(config.cutmix),
        "copy_paste": float(config.copy_paste),
        "degrees": float(config.degrees),
        "shear": float(config.shear),
        "perspective": float(config.perspective),
        "translate": float(config.translate),
        "scale": config.scale if isinstance(config.scale, (list, tuple)) else float(config.scale),
        "flipud": float(config.flipud),
        "fliplr": float(config.fliplr),
        "hsv_h": float(config.hsv_h),
        "hsv_s": float(config.hsv_s),
        "hsv_v": float(config.hsv_v),
    }
    train_kwargs.update(config.extra_args)
    return train_kwargs


def format_submitted_hyperparams(
    train_kwargs: Mapping[str, object], *, resume: bool = False
) -> str:
    """将提交给 Ultralytics 的训练参数字典格式化为带提示说明的可读多行日志文本。"""
    json_lines = json.dumps(dict(train_kwargs), ensure_ascii=False, indent=2, default=str).splitlines()
    if json_lines:
        json_lines[0] = f'"arguments": {json_lines[0]}'
    lines = [
        "=== 训练超参数配置 (提交至 Ultralytics) ===",
        *json_lines,
    ]
    if resume:
        lines.append(
            "[提示] 已启用断点续训 (resume=True)，实际训练超参数将优先以断点权重 last.pt 中记录的配置为准。"
        )
    elif str(train_kwargs.get("optimizer", "auto")).strip().lower() == "auto":
        lines.append(
            '[提示] 当前 optimizer="auto"，Ultralytics 将按类别数与迭代步数自动选定优化器并重算 lr0；'
            "如需严格固定 lr0，请显式选择 AdamW / SGD 等优化器。"
        )
    return "\n".join(lines)


def format_effective_hyperparams(
    trainer: object, extra_keys: Sequence[str] = ()
) -> str:
    """在 `on_train_start` 阶段提取 `trainer.args` 经自动推导或断点恢复后的最终实际生效超参数并格式化。"""
    args_obj = getattr(trainer, "args", None)
    if isinstance(args_obj, Mapping):
        args_map: dict[str, object] = dict(args_obj)
    elif args_obj is not None and hasattr(args_obj, "__dict__"):
        args_map = dict(vars(args_obj))
    else:
        return ""

    if not args_map:
        return ""

    effective_amp = getattr(trainer, "amp", args_map.get("amp"))
    if isinstance(effective_amp, bool):
        args_map["amp"] = effective_amp
    accumulate = getattr(trainer, "accumulate", None)

    def _fmt_group(keys: Sequence[str]) -> str:
        parts: list[str] = []
        for k in keys:
            if k not in args_map:
                continue
            val = args_map[k]
            if k == "nbs" and accumulate is not None:
                parts.append(f"nbs={val} (accumulate={accumulate})")
            else:
                parts.append(f"{k}={val}")
        return ", ".join(parts)

    core_keys = ("epochs", "imgsz", "batch", "nbs", "device", "workers", "amp", "cache")
    opt_keys = (
        "optimizer",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "cos_lr",
        "patience",
    )
    aug_keys = (
        "mosaic",
        "close_mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "translate",
        "scale",
        "degrees",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    )
    eng_keys = (
        "seed",
        "deterministic",
        "pretrained",
        "plots",
        "save",
        "val",
        "exist_ok",
        "resume",
    )

    shown = set(core_keys) | set(opt_keys) | set(aug_keys) | set(eng_keys) | {"data", "project", "name"}
    custom_keys = [k for k in extra_keys if k not in shown and k in args_map]

    lines = ["=== 实际生效训练超参数 (Trainer 初始化完成) ==="]
    for label, group_keys in (
        ("[核心与硬件]", core_keys),
        ("[优化与调度]", opt_keys),
        ("[数据增强]  ", aug_keys),
        ("[工程与复现]", eng_keys),
    ):
        formatted = _fmt_group(group_keys)
        if formatted:
            lines.append(f"  {label} {formatted}")
    if custom_keys:
        lines.append(f"  [附加参数]   {_fmt_group(custom_keys)}")
    return "\n".join(lines)


def _emit_multiline_log(
    log_callback: Callable[[str], None] | None, text: str
) -> None:
    """将多行日志块逐行发送至 `log_callback`（忽略空文本块）。"""
    if log_callback is None or not text:
        return
    for line in text.splitlines():
        log_callback(line)


def train_yolo26(
    config: TrainConfig,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    cancel_event: Event | None = None,
) -> TrainResult:
    """使用 Ultralytics 启动 YOLO26 目标检测模型训练。

    调用条件与副作用：
        要求 `config.data_yaml` 存在（否则抛出 `FileNotFoundError`）；断点续训时还要求能解析到
        `last.pt`（见 `TrainConfig.resume`）。会在 `config.project_dir` 下的实验目录
        （`exist_ok=False` 时同名目录存在会自动递增）写出训练日志、指标曲线及 `weights/best.pt`、`last.pt`，
        实际目录以返回值 `TrainResult.save_dir` 为准。

    回调、超参数日志与安全停训约束：
        训练提交前先向 `log_callback` 输出本次传入的完整 `"arguments"` 参数字典；
        在 Ultralytics `on_train_start` 阶段（已完成 `AutoBatch`、优化器自动推导、AMP 校验与断点恢复）
        再输出最终实际生效的分类超参数摘要。
        通过 `on_train_epoch_end` 与 `on_fit_epoch_end` 回调派发 `EpochMetrics` 并检查 `cancel_event`；
        若收到取消信号，将设置 `trainer.stop = True`，等待当前轮次完成并保存断点权重后再退出，
        此时 `TrainResult.stopped_early=True`（训练已自然结束后才设置的取消信号不计入）。
        训练期间仅把**当前线程**的标准输出、标准错误与 `ultralytics` 日志按行转发到 `log_callback`
        （见 `core.log_capture`），不影响其他并发任务的输出。
    """
    from ultralytics import YOLO

    if not config.data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {config.data_yaml}")

    model_source = (
        str(_resolve_resume_checkpoint(config))
        if config.resume
        else resolve_pretrained_weights(config.model_name)
    )

    if log_callback:
        log_callback(f"{'断点续训，加载断点' if config.resume else '加载模型'}: {model_source}")
        log_callback(f"数据集配置: {config.data_yaml}")

    ensure_torch_cpu_threads()
    model = YOLO(model_source)
    stop_requested = {"value": False}
    # 追踪正常训练轮次与 `final_eval()` 额外回调的成对关系：
    # 正常每轮必定先触发 `on_train_epoch_end` 再触发 `on_fit_epoch_end`；
    # 训练结束后的 `final_eval()` 不经过 `on_train_epoch_end`，而是临时将 `trainer.epoch += 1`
    # 后再次触发 `on_fit_epoch_end`，据此可精确识别并在早停/跑满两种场景下保持真实完成轮次
    epoch_state = {"awaiting_fit_end": False, "last_completed_epoch": 0}
    extra_keys = tuple(config.extra_args.keys())

    def _on_train_start(trainer: object) -> None:
        effective_log = format_effective_hyperparams(trainer, extra_keys=extra_keys)
        _emit_multiline_log(log_callback, effective_log)

    def _check_cancel(trainer: object) -> None:
        if cancel_event and cancel_event.is_set() and not stop_requested["value"]:
            setattr(trainer, "stop", True)
            stop_requested["value"] = True
            if log_callback:
                log_callback("收到停止请求，将在当前轮次结束后安全停止训练...")

    def _on_train_epoch_end(trainer: object) -> None:
        epoch_state["awaiting_fit_end"] = True
        _check_cancel(trainer)

    def _on_fit_epoch_end(trainer: object) -> None:
        raw_epoch = int(getattr(trainer, "epoch", 0))
        epochs = int(getattr(trainer, "epochs", config.epochs))
        metrics_dict = getattr(trainer, "metrics", {}) or {}
        tloss = getattr(trainer, "tloss", None)

        if epoch_state["awaiting_fit_end"]:
            epoch_state["awaiting_fit_end"] = False
            is_final_eval = False
            epoch = raw_epoch + 1
            epoch_state["last_completed_epoch"] = epoch
        elif epoch_state["last_completed_epoch"] > 0 or (epochs > 0 and raw_epoch >= epochs):
            is_final_eval = True
            epoch = (
                epoch_state["last_completed_epoch"]
                if epoch_state["last_completed_epoch"] > 0
                else min(max(1, raw_epoch), epochs)
            )
        else:
            is_final_eval = False
            epoch = raw_epoch + 1
            epoch_state["last_completed_epoch"] = epoch

        box_loss = 0.0
        cls_loss = 0.0
        if not is_final_eval and tloss is not None:
            try:
                flat = tloss.detach().cpu().tolist()
                if isinstance(flat, list) and len(flat) >= 2:
                    box_loss = _safe_float(flat[0])
                    cls_loss = _safe_float(flat[1])
            except Exception:
                pass

        em = EpochMetrics(
            epoch=epoch,
            total_epochs=epochs,
            box_loss=box_loss,
            cls_loss=cls_loss,
            precision=_safe_float(metrics_dict.get("metrics/precision(B)", 0.0)),
            recall=_safe_float(metrics_dict.get("metrics/recall(B)", 0.0)),
            map50=_safe_float(metrics_dict.get("metrics/mAP50(B)", 0.0)),
            map50_95=_safe_float(metrics_dict.get("metrics/mAP50-95(B)", 0.0)),
            is_final_eval=is_final_eval,
        )
        if epoch_callback:
            epoch_callback(em)
        # 训练已自然进入 final_eval 阶段后，不再将迟到的取消信号计作提前停训
        if not is_final_eval:
            _check_cancel(trainer)

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_epoch_end", _on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    train_kwargs = build_train_kwargs(config)
    _emit_multiline_log(
        log_callback, format_submitted_hyperparams(train_kwargs, resume=config.resume)
    )

    with capture_thread_output(log_callback):
        results = model.train(**train_kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", config.project_dir / config.run_name))
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    final_metrics: dict[str, float] = {}
    results_dict = getattr(results, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            final_metrics[str(key)] = _safe_float(value)

    return TrainResult(
        save_dir=save_dir,
        best_weights=best_pt if best_pt.is_file() else None,
        last_weights=last_pt if last_pt.is_file() else None,
        metrics=final_metrics,
        stopped_early=stop_requested["value"],
    )


# 支持在验证与推理预览中直接加载的已导出部署模型后缀与标准格式名映射
EXPORTED_MODEL_SUFFIX_TO_FORMAT: dict[str, str] = {
    ".onnx": "onnx",
    ".torchscript": "torchscript",
    ".engine": "engine",
}
EXPORTED_MODEL_SUFFIXES: frozenset[str] = frozenset(EXPORTED_MODEL_SUFFIX_TO_FORMAT.keys())
SUPPORTED_EVAL_MODEL_SUFFIXES: frozenset[str] = frozenset({".pt", *EXPORTED_MODEL_SUFFIXES})

# PyTorch `.pt` 验证/推理可选检测头模式：(内部键名, 界面显示文案, 对应 Ultralytics `nms` 参数值)
PREDICT_HEAD_MODES: list[tuple[str, str, bool | None]] = [
    ("one2many", "one2many · 外部 NMS (PT 默认)", None),
    ("end2end", "end2end · YOLO26 one2one 无 NMS", False),
]
PREDICT_HEAD_KEY_TO_NMS: dict[str, bool | None] = {
    key: nms_val for key, _, nms_val in PREDICT_HEAD_MODES
}

# 推理效果测试输入尺寸与填充模式：(内部键名, 界面显示文案)
PREDICT_PAD_MODES: list[tuple[str, str]] = [
    ("rect", "最小矩形填充 (rect=True，PT 默认)"),
    ("square", "固定方形填充 (rect=False，模拟方形导出)"),
    ("export", "按导出尺寸固定填充 (H×W，rect=False)"),
]


def detect_model_format(model_path: Path) -> str:
    """根据模型文件扩展名返回规范化格式键名（`pt` / `onnx` / `torchscript` / `engine` 等）。"""
    suffix = model_path.suffix.lower()
    if suffix in EXPORTED_MODEL_SUFFIX_TO_FORMAT:
        return EXPORTED_MODEL_SUFFIX_TO_FORMAT[suffix]
    if suffix == ".pt" or not suffix:
        return "pt"
    return suffix.lstrip(".")


def is_exported_model_path(model_path: Path) -> bool:
    """判断路径是否为已导出的部署模型文件（`.onnx` / `.torchscript` / `.engine`）。"""
    return model_path.suffix.lower() in EXPORTED_MODEL_SUFFIXES


def _format_imgsz_arg(imgsz: int | tuple[int, int]) -> tuple[int | list[int], tuple[int, int]]:
    """将单整数或 `(高, 宽)` 元组校验并转换为 Ultralytics 可接受的 `imgsz` 参数及规范二元组。"""
    pair = normalize_export_imgsz(imgsz)
    h, w = pair
    return (h if h == w else [h, w]), pair


def build_val_kwargs(
    weights_path: Path,
    data_yaml: Path,
    *,
    imgsz: int | tuple[int, int] = 640,
    device: str = "0",
    nms: bool | None = None,
    rect: bool | None = None,
    metadata: WeightsMetadata | None = None,
) -> tuple[dict[str, object], list[str]]:
    """校验并组装传给 Ultralytics `model.val(**kwargs)` 的参数字典与提示列表。

    规则说明：
        - `.engine`（TensorRT）模型仅支持 CUDA GPU，在 CPU 设备模式下抛出 `ValueError`；
        - 已导出模型（`.onnx` / `.torchscript` / `.engine`）的检测头已在导出时固化，不传入 `nms` 覆盖；
        - 当传入非正方形 `(高, 宽)` 尺寸时，强制 `rect=False` 以避免矩形批排布与二元组 `imgsz` 冲突。
    """
    fmt = detect_model_format(weights_path)
    is_exported = is_exported_model_path(weights_path)
    dev_str = str(device).strip() or "0"
    if fmt == "engine" and dev_str.lower() == "cpu":
        raise ValueError("TensorRT (.engine) 模型仅支持在 CUDA GPU 上验证，请启用 GPU 加速。")

    imgsz_arg, pair = _format_imgsz_arg(imgsz)
    warnings: list[str] = []

    val_kwargs: dict[str, object] = {
        "data": str(data_yaml),
        "imgsz": imgsz_arg,
        "device": dev_str,
    }

    if is_exported:
        if nms is not None:
            warnings.append(f"已导出 {fmt.upper()} 模型的检测头已固化，忽略额外指定的 nms={nms}。")
        if metadata is not None and metadata.export_dynamic is False:
            val_kwargs["rect"] = False
            if metadata.train_imgsz is not None:
                mh, mw = metadata.train_imgsz
                if pair != metadata.train_imgsz:
                    warnings.append(
                        f"静态 {fmt.upper()} 模型固定使用导出尺寸 {format_imgsz_text(metadata.train_imgsz)}，"
                        "Ultralytics 验证时将自动沿用该固定尺寸。"
                    )
                val_kwargs["imgsz"] = mh if mh == mw else [mh, mw]
        elif rect is not None:
            val_kwargs["rect"] = bool(rect)
    else:
        if nms is not None:
            val_kwargs["nms"] = nms
            if nms is False and metadata is not None and not metadata.supports_end2end:
                warnings.append("当前 .pt 权重未检测到 one2one 端到端分支，验证将回退为 one2many 检测头。")
        elif metadata is not None and not metadata.one2many_available and metadata.supports_end2end:
            warnings.append("当前 .pt 权重仅保留 one2one 端到端分支，将按端到端无 NMS 执行验证。")

        if pair[0] != pair[1]:
            if rect is True:
                warnings.append("验证传入矩形输入尺寸时需关闭最小矩形批排布，已自动设置 rect=False。")
            val_kwargs["rect"] = False
        elif rect is not None:
            val_kwargs["rect"] = bool(rect)

    return val_kwargs, warnings


def format_submitted_val_config(
    weights_path: Path,
    val_kwargs: Mapping[str, object],
    *,
    metadata: WeightsMetadata | None = None,
    warnings: Sequence[str] = (),
) -> str:
    """将提交给 Ultralytics `model.val()` 的验证配置与提示格式化为可读多行日志。"""
    fmt = detect_model_format(weights_path)
    is_exported = is_exported_model_path(weights_path)
    if is_exported:
        mode_map = {
            "end2end": "已固化 one2one 端到端无 NMS",
            "embedded": "已固化图内嵌入 NMS",
            "raw": "已固化原始 one2many (外部 NMS)",
        }
        head_desc = mode_map.get(metadata.export_nms_mode if metadata else None, "由导出模型决定")
    elif val_kwargs.get("nms") is False:
        head_desc = "end2end (YOLO26 one2one 无 NMS)"
    else:
        head_desc = "one2many (外部 NMS，PT 默认)"

    payload: dict[str, object] = {
        "model": weights_path.name,
        "model_format": fmt,
        "head_mode": head_desc,
        **dict(val_kwargs),
    }
    json_lines = json.dumps(payload, ensure_ascii=False, indent=2, default=str).splitlines()
    if json_lines:
        json_lines[0] = f'"val_arguments": {json_lines[0]}'
    lines = ["=== 模型验证配置 (提交至 Ultralytics) ===", *json_lines]
    for warn in warnings:
        lines.append(f"[提示] {warn}")
    return "\n".join(lines)


def validate_model(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int | tuple[int, int] = 640,
    device: str = "0",
    *,
    nms: bool | None = None,
    rect: bool | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """在指定验证数据集上评估 `.pt` 权重或已导出部署模型（`.onnx` / `.torchscript` / `.engine`）并返回指标字典。

    参数与行为说明：
        - `imgsz` 支持正整数（方形）或 `(高, 宽)` 二元组；
        - `nms` 仅对 `.pt` 权重生效（`None`=默认 `one2many` + 外部 NMS，`False`=YOLO26 `one2one` 端到端头）；
        - `weights_path` 或 `data_yaml` 不存在时抛出 `FileNotFoundError`，非法设备或尺寸抛出 `ValueError`；
        - 执行前向 `log_callback` 输出本次验证配置，并将当前线程输出与 `ultralytics` 日志转发至 `log_callback`。
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {weights_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

    meta: WeightsMetadata | None = None
    try:
        meta = inspect_weights_metadata(weights_path)
    except Exception:
        meta = None

    val_kwargs, warnings = build_val_kwargs(
        weights_path,
        data_yaml,
        imgsz=imgsz,
        device=device,
        nms=nms,
        rect=rect,
        metadata=meta,
    )
    _emit_multiline_log(
        log_callback,
        format_submitted_val_config(weights_path, val_kwargs, metadata=meta, warnings=warnings),
    )

    from ultralytics import YOLO

    ensure_torch_cpu_threads()
    model = YOLO(str(weights_path))
    with capture_thread_output(log_callback):
        metrics = model.val(**val_kwargs)

    summary: dict[str, float] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for k, v in results_dict.items():
            summary[str(k)] = _safe_float(v)
    return summary


def _unique_pred_path(output_dir: Path, src: Path, used: set[str]) -> Path:
    """生成不重名的 `*_pred.jpg` 输出路径；同名不同扩展名（如 a.jpg 与 a.png）追加扩展名区分。"""
    base = f"{src.stem}_pred"
    if base.lower() in used:
        base = f"{src.stem}_{src.suffix.lstrip('.').lower()}_pred"
    candidate = base
    idx = 1
    while candidate.lower() in used:
        candidate = f"{base}_{idx}"
        idx += 1
    used.add(candidate.lower())
    return output_dir / f"{candidate}.jpg"


# ONNX 部署元数据契约版本与结构化 JSON 主键
ONNX_DEPLOY_SCHEMA_VERSION = "yolostudio.deploy.v1"
ONNX_DEPLOY_META_JSON_KEY = "yolostudio_deploy_meta"

# 导出预处理策略选项：(内部键名, 界面显示文案, resize_mode, padding_align)
EXPORT_PREPROCESS_MODES: list[tuple[str, str, str, str]] = [
    ("letterbox_center", "等比缩放 · 居中填充 (LetterBox，推荐)", "letterbox", "center"),
    ("letterbox_top_left", "等比缩放 · 左上靠边填充 (LetterBox)", "letterbox", "top_left"),
    ("stretch", "直接拉伸填充 (Stretch，不保持宽高比)", "stretch", "none"),
]
EXPORT_PREPROCESS_KEY_TO_SPEC: dict[str, tuple[str, str]] = {
    key: (resize_mode, padding_align)
    for key, _, resize_mode, padding_align in EXPORT_PREPROCESS_MODES
}
DEFAULT_EXPORT_PREPROCESS_MODE = "letterbox_center"
DEFAULT_EXPORT_PADDING_VALUE = 114


def parse_export_preprocess_mode(raw: str | None) -> str:
    """将界面键名、CLI 选项或别名规范化为导出预处理策略键名。

    可选规范值：
        - `"letterbox_center"`：保持宽高比缩放 + 居中填充（YOLO 标准默认）；
        - `"letterbox_top_left"`：保持宽高比缩放 + 左上角靠边填充；
        - `"stretch"`：直接拉伸至目标高×宽（不保持宽高比、不补边）。
    """
    if raw is None:
        return DEFAULT_EXPORT_PREPROCESS_MODE
    token = str(raw).strip().lower()
    if token in {"", "default", "letterbox", "center", "letterbox_center", "letterbox-center"}:
        return "letterbox_center"
    if token in {
        "top_left",
        "top-left",
        "topleft",
        "edge",
        "letterbox_top_left",
        "letterbox-top-left",
    }:
        return "letterbox_top_left"
    if token in {"stretch", "scale_fill", "scale-fill", "resize"}:
        return "stretch"
    raise ValueError(
        f"无效的预处理策略 '{raw}'，可选值: letterbox_center(居中填充)、"
        "letterbox_top_left(左上靠边填充)、stretch(直接拉伸)"
    )


def normalize_padding_value(raw: int | None) -> int:
    """校验并规范化 LetterBox 补边灰度值（`0 ~ 255` 整数，默认 `114`）。"""
    if raw is None:
        return DEFAULT_EXPORT_PADDING_VALUE
    if isinstance(raw, bool) or not isinstance(raw, int):
        try:
            if isinstance(raw, bool):
                raise ValueError
            val = int(raw)
        except Exception as exc:
            raise ValueError(f"补边灰度值必须为 0 ~ 255 的整数，当前值: {raw!r}") from exc
    else:
        val = raw
    if not (0 <= val <= 255):
        raise ValueError(f"补边灰度值必须在 0 ~ 255 之间，当前值: {val}")
    return val


def resolve_preprocess_spec(
    preprocess_mode: str | None,
    padding_value: int | None = DEFAULT_EXPORT_PADDING_VALUE,
) -> tuple[str, str, str, int | None]:
    """返回规范化四元组 `(mode_key, resize_mode, padding_align, effective_padding_value)`。"""
    mode_key = parse_export_preprocess_mode(preprocess_mode)
    resize_mode, padding_align = EXPORT_PREPROCESS_KEY_TO_SPEC[mode_key]
    pad_val = None if resize_mode == "stretch" else normalize_padding_value(padding_value)
    return mode_key, resize_mode, padding_align, pad_val


def format_preprocess_summary(
    preprocess_mode: str | None,
    padding_value: int | None = DEFAULT_EXPORT_PADDING_VALUE,
    *,
    auto_rect: bool | None = None,
) -> str:
    """将预处理策略格式化为简洁中文展示文本，用于预览、日志与元数据展示。"""
    mode_key, resize_mode, _align, pad_val = resolve_preprocess_spec(preprocess_mode, padding_value)
    if resize_mode == "stretch":
        base = "直接拉伸 Stretch (无补边)"
    elif mode_key == "letterbox_top_left":
        base = f"左上靠边 LetterBox (补边={pad_val})"
    else:
        base = f"居中 LetterBox (补边={pad_val})"
    if auto_rect is True and resize_mode == "letterbox":
        return f"{base}, auto_rect=True"
    if auto_rect is False and resize_mode == "letterbox":
        return f"{base}, auto_rect=False"
    return base


def _build_custom_preprocess_predictor(
    resize_mode: str,
    padding_align: str,
    padding_value: int,
) -> type:
    """动态构造按指定预处理策略（靠边填充/直接拉伸/自定义补边值）执行前处理与坐标还原的 `DetectionPredictor` 子类。"""
    from ultralytics.data.augment import LetterBox
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import ops

    is_stretch = resize_mode == "stretch"
    is_center = padding_align == "center"
    pad_val = int(padding_value)

    class _CustomPreprocessDetectionPredictor(DetectionPredictor):
        def pre_transform(self, im):
            same_shapes = len({x.shape for x in im}) == 1
            auto = (
                (not is_stretch)
                and same_shapes
                and self.args.rect
                and (
                    self.model.format == "pt"
                    or (getattr(self.model, "dynamic", False) and self.model.format != "imx")
                )
            )
            letterbox = LetterBox(
                self.imgsz,
                auto=auto,
                scale_fill=is_stretch,
                center=is_center,
                stride=self.model.stride,
                padding_value=pad_val,
            )
            return [letterbox(image=x) for x in im]

        def construct_result(self, pred, img, orig_img, img_path):
            if is_stretch:
                gain_y = img.shape[2] / orig_img.shape[0]
                gain_x = img.shape[3] / orig_img.shape[1]
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:],
                    pred[:, :4],
                    orig_img.shape,
                    ratio_pad=((gain_y, gain_x), (0.0, 0.0)),
                    padding=False,
                )
            elif not is_center:
                gain = min(img.shape[2] / orig_img.shape[0], img.shape[3] / orig_img.shape[1])
                new_h = max(round(orig_img.shape[0] * gain), 1)
                new_w = max(round(orig_img.shape[1] * gain), 1)
                gain_y = new_h / orig_img.shape[0]
                gain_x = new_w / orig_img.shape[1]
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:],
                    pred[:, :4],
                    orig_img.shape,
                    ratio_pad=((gain_y, gain_x), (0.0, 0.0)),
                    padding=True,
                )
            else:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])

    return _CustomPreprocessDetectionPredictor


def resolve_effective_predict_preprocess(
    *,
    is_exported: bool,
    preprocess_mode: str | None = None,
    padding_value: int | None = None,
    metadata: WeightsMetadata | None = None,
) -> tuple[str, str, str, int | None]:
    """解析推理阶段应执行的预处理策略；已导出模型未显式覆盖时优先沿用其内嵌元数据契约。"""
    raw_mode = preprocess_mode
    if raw_mode is None and is_exported and metadata is not None and metadata.preprocess_mode:
        raw_mode = metadata.preprocess_mode

    raw_pad = padding_value
    if raw_pad is None and is_exported and metadata is not None and metadata.preprocess_padding_value is not None:
        raw_pad = metadata.preprocess_padding_value

    return resolve_preprocess_spec(raw_mode, raw_pad)


# 推理预览绘制样式选项：(内部键名, 界面显示文案)
PREDICT_RENDER_STYLES: list[tuple[str, str]] = [
    ("compact", "紧凑防重叠 (半透明，推荐)"),
    ("ultralytics", "Ultralytics 默认"),
]
DEFAULT_PREDICT_RENDER_STYLE = "compact"

# 推理预览框线与字号挡位：(内部键名, 界面显示文案)
PREDICT_LINE_WIDTH_MODES: list[tuple[str, str]] = [
    ("auto", "自动"),
    ("thin", "极细"),
    ("medium", "适中"),
    ("bold", "较粗"),
]
DEFAULT_PREDICT_LINE_WIDTH_MODE = "auto"

# 与 Ultralytics 默认调色板一致的 20 色高区分度 BGR 类别色表（零依赖，避免触发 `ultralytics` 顶层导入副作用）
_CLASS_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 42, 4),
    (235, 219, 11),
    (243, 243, 243),
    (183, 223, 0),
    (104, 31, 17),
    (221, 111, 255),
    (79, 68, 255),
    (0, 237, 204),
    (68, 243, 0),
    (255, 0, 189),
    (255, 180, 0),
    (186, 0, 221),
    (255, 255, 0),
    (0, 192, 38),
    (179, 255, 1),
    (255, 36, 125),
    (104, 0, 123),
    (108, 27, 255),
    (47, 109, 252),
    (11, 255, 162),
)


@dataclass(frozen=True)
class PredictRenderConfig:
    """推理预览检测框与标签绘制样式配置。

    字段说明：
        `style`：绘制模式键名，`"compact"`（紧凑半透明 + 标签防重叠，默认）或 `"ultralytics"`（原版 `r.plot()`）。
        `line_width_mode`：框线与字号挡位，`"auto"`（按分辨率与目标密度自适应）、`"thin"`（极细）、
            `"medium"`（适中）或 `"bold"`（较粗）。
        `show_labels`：是否显示类别名称（如 `player`）。
        `show_conf`：是否显示置信度小数（固定保留两位小数，如 `0.94`）。
        `label_alpha`：紧凑模式下标签背景不透明度，范围 `[0.0, 1.0]`，默认 `0.68`（半透明兼顾可读与不遮挡）。
        `avoid_overlap`：紧凑模式下是否启用二维防重叠避让排版（默认开启）。
    """

    style: str = DEFAULT_PREDICT_RENDER_STYLE
    line_width_mode: str = DEFAULT_PREDICT_LINE_WIDTH_MODE
    show_labels: bool = True
    show_conf: bool = True
    label_alpha: float = 0.68
    avoid_overlap: bool = True


def parse_predict_render_style(raw: str | None) -> str:
    """将界面键名或别名规范化为推理绘制样式键名（`"compact"` 或 `"ultralytics"`）。"""
    if raw is None:
        return DEFAULT_PREDICT_RENDER_STYLE
    token = str(raw).strip().lower()
    if token in {"", "default", "compact", "dense", "smart"}:
        return "compact"
    if token in {"ultralytics", "native", "raw", "yolo"}:
        return "ultralytics"
    raise ValueError(f"无效的推理绘制样式 '{raw}'，可选值: compact(紧凑防重叠)、ultralytics(Ultralytics 默认)")


def parse_predict_line_width_mode(raw: str | None) -> str:
    """将界面键名或别名规范化为推理框线/字号挡位（`"auto"` / `"thin"` / `"medium"` / `"bold"`）。"""
    if raw is None:
        return DEFAULT_PREDICT_LINE_WIDTH_MODE
    token = str(raw).strip().lower()
    if token in {"", "auto", "default"}:
        return "auto"
    if token in {"thin", "fine", "small", "1"}:
        return "thin"
    if token in {"medium", "normal", "mid", "2"}:
        return "medium"
    if token in {"bold", "thick", "large", "3"}:
        return "bold"
    raise ValueError(f"无效的框线挡位 '{raw}'，可选值: auto(自动)、thin(极细)、medium(适中)、bold(较粗)")


def normalize_predict_render_config(config: PredictRenderConfig | None) -> PredictRenderConfig:
    """校验并规范化 `PredictRenderConfig`，返回字段合法的新配置实例。"""
    if config is None:
        return PredictRenderConfig()
    style = parse_predict_render_style(config.style)
    lw_mode = parse_predict_line_width_mode(config.line_width_mode)
    alpha = float(config.label_alpha)
    if not math.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
        raise ValueError(f"标签透明度 label_alpha 必须在 0.0 ~ 1.0 之间，当前值: {config.label_alpha!r}")
    return PredictRenderConfig(
        style=style,
        line_width_mode=lw_mode,
        show_labels=bool(config.show_labels),
        show_conf=bool(config.show_conf),
        label_alpha=alpha,
        avoid_overlap=bool(config.avoid_overlap),
    )


def format_predict_render_summary(config: PredictRenderConfig | None) -> str:
    """将 `PredictRenderConfig` 格式化为简洁中文展示文案，用于界面摘要与运行日志。"""
    cfg = normalize_predict_render_config(config)
    style_txt = "紧凑防重叠" if cfg.style == "compact" else "Ultralytics 默认"
    lw_map = {"auto": "自动", "thin": "极细", "medium": "适中", "bold": "较粗"}
    lw_txt = lw_map.get(cfg.line_width_mode, cfg.line_width_mode)
    if cfg.show_labels and cfg.show_conf:
        content_txt = "类别+置信度"
    elif cfg.show_labels:
        content_txt = "仅类别"
    elif cfg.show_conf:
        content_txt = "仅置信度"
    else:
        content_txt = "仅画框"
    return f"{style_txt} ({lw_txt}, {content_txt})"


def _get_class_color_bgr(cls_id: int) -> tuple[int, int, int]:
    """按类别编号返回稳定的 BGR 颜色三元组（与 Ultralytics 调色板保持一致，且不触发额外模块导入）。"""
    return _CLASS_PALETTE_BGR[int(cls_id) % len(_CLASS_PALETTE_BGR)]


def _to_numpy_2d_or_1d(value: object) -> np.ndarray | None:
    """将 Tensor / ndarray / 序列安全转换为 NumPy 数组；缺失或无法转换时返回 `None`。"""
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()  # type: ignore[union-attr]
        if hasattr(value, "cpu") and hasattr(value, "numpy"):
            return value.cpu().numpy()  # type: ignore[union-attr]
        return np.asarray(value)
    except Exception:
        return None


def _resolve_render_scale(
    img_h: int,
    img_w: int,
    *,
    num_boxes: int,
    median_box_dim: float,
    line_width_mode: str,
) -> tuple[int, int, float, int | None]:
    """根据画布分辨率、目标密度/中位尺寸与挡位计算 `(box_lw, font_px, cv_font_scale, ultra_lw)`。

    参数与单位：
        `img_h` / `img_w`：原图高与宽（像素）；`median_box_dim`：检测框宽高中位特征尺度（像素）。
    返回值说明：
        - `box_lw`：紧凑模式检测框线宽（像素，至少 `1`）；
        - `font_px`：紧凑模式 TrueType 字体像素高度（`10 ~ 28`）；
        - `cv_font_scale`：无 TrueType 字体回退 OpenCV `putText` 时的缩放系数；
        - `ultra_lw`：透传给 Ultralytics `plot(line_width=...)` / `predict(line_width=...)` 的线宽
          （`"auto"` 返回 `None` 沿用 Ultralytics 自身默认，其余挡位返回明确整数）。
    """
    avg_dim = max(1.0, (img_h + img_w) / 2.0)
    dense_or_small = num_boxes >= 12 or (0.0 < median_box_dim < 72.0)

    if line_width_mode == "thin":
        box_lw = 1
        font_px = max(10, min(18, round(avg_dim * 0.0115)))
        ultra_lw: int | None = 1
    elif line_width_mode == "medium":
        box_lw = max(2, min(4, round(avg_dim / 520.0)))
        font_px = max(12, min(24, round(avg_dim * 0.0155)))
        ultra_lw = 2
    elif line_width_mode == "bold":
        box_lw = max(3, min(5, round(avg_dim / 360.0)))
        font_px = max(14, min(28, round(avg_dim * 0.0195)))
        ultra_lw = 3
    else:
        # 自动挡：常规场景克制缩减，检测到密集/小目标画面时进一步收紧字号与线宽避免遮挡
        if dense_or_small:
            box_lw = 1 if avg_dim < 760 else (2 if avg_dim < 1800 else 3)
            font_px = max(11, min(20, round(avg_dim * 0.013)))
        else:
            box_lw = 1 if avg_dim < 600 else (2 if avg_dim < 1500 else 3)
            font_px = max(12, min(22, round(avg_dim * 0.015)))
        ultra_lw = None

    cv_font_scale = max(0.36, min(0.85, font_px / 28.0))
    return box_lw, font_px, cv_font_scale, ultra_lw


@lru_cache(maxsize=64)
def _load_compact_truetype_font(font_px: int, need_unicode: bool) -> object | None:
    """按字号与字符集需求加载清晰紧凑的本地 TrueType 字体并缓存；不可用时返回 `None` 交由 OpenCV 回退。"""
    try:
        from PIL import ImageFont
    except Exception:
        return None

    win_fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    candidates: list[Path] = []
    if need_unicode:
        candidates.extend(
            [
                win_fonts / "msyhbd.ttc",
                win_fonts / "msyh.ttc",
                win_fonts / "simhei.ttf",
                win_fonts / "simsun.ttc",
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
                Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
                Path("/System/Library/Fonts/PingFang.ttc"),
            ]
        )
    else:
        candidates.extend(
            [
                win_fonts / "segoeuib.ttf",
                win_fonts / "segoeui.ttf",
                win_fonts / "arialbd.ttf",
                win_fonts / "arial.ttf",
                win_fonts / "msyhbd.ttc",
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/System/Library/Fonts/Helvetica.ttc"),
            ]
        )

    # 仅在当前进程已加载 `ultralytics.utils` 时才顺带检索其字体缓存目录，避免触发 `ultralytics` 顶层日志副作用
    ultra_utils = sys.modules.get("ultralytics.utils")
    user_cfg_dir = getattr(ultra_utils, "USER_CONFIG_DIR", None) if ultra_utils is not None else None
    if user_cfg_dir is not None:
        cfg_dir = Path(user_cfg_dir)
        if need_unicode:
            candidates.append(cfg_dir / "Arial.Unicode.ttf")
        candidates.append(cfg_dir / "Arial.ttf")

    for path in candidates:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=int(font_px))
            except Exception:
                continue

    for bare_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(bare_name, size=int(font_px))
        except Exception:
            continue
    return None


def _rect_intersect_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> int:
    """计算两个轴对齐像素矩形 `(x1, y1, x2, y2)` 外扩 `pad` 像素后的交集面积。"""
    ix1 = max(a[0] - pad, b[0] - pad)
    iy1 = max(a[1] - pad, b[1] - pad)
    ix2 = min(a[2] + pad, b[2] + pad)
    iy2 = min(a[3] + pad, b[3] + pad)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def _layout_non_overlapping_labels(
    boxes: Sequence[tuple[int, int, int, int]],
    label_sizes: Sequence[tuple[int, int]],
    scores: Sequence[float],
    *,
    img_w: int,
    img_h: int,
    avoid_overlap: bool = True,
) -> list[tuple[int, int, int, int]]:
    """为一组检测框计算紧贴目标且尽量互不重叠的标签像素矩形 `(rx1, ry1, rx2, ry2)`。

    排版策略与坐标系：
        - 输入/输出均为原图像素坐标系 `(x1, y1, x2, y2)`，所有结果均钳制在 `[0, img_w] × [0, img_h]` 内；
        - 按行带从上到下、同行从左到右（同列高置信度优先）依次放置，使同排密集小目标自然形成
          “正上方 / 正下方”交替错落排布；
        - 候选优先级：框正上方（左/中/右）→ 框正下方（左/中/右）→ 框内顶/底（仅大框）→ 上下错层堆叠；
        - 综合惩罚已放置标签重叠（最高权重）、其他目标框遮挡、自身目标遮挡与偏移距离；
        - 当 `avoid_overlap=False` 时，仅按框正上方（顶部不足时退回框内顶部）并做画布边界钳制。
        - 返回列表顺序与输入 `boxes` 一一对应。
    """
    count = len(boxes)
    if count == 0:
        return []

    placements: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * count
    if not avoid_overlap:
        for i, ((x1, y1, _x2, _y2), (lw_px, lh_px)) in enumerate(zip(boxes, label_sizes)):
            eff_w = min(max(1, lw_px), max(1, img_w))
            eff_h = min(max(1, lh_px), max(1, img_h))
            raw_y = y1 - eff_h - 1 if y1 >= eff_h + 1 else y1 + 1
            cx = max(0, min(max(0, img_w - eff_w), x1))
            cy = max(0, min(max(0, img_h - eff_h), raw_y))
            placements[i] = (cx, cy, cx + eff_w, cy + eff_h)
        return placements

    median_h = int(np.median([max(1, sz[1]) for sz in label_sizes]))
    row_band = max(14, int(median_h * 1.4))
    order = sorted(
        range(count),
        key=lambda idx: (boxes[idx][1] // row_band, boxes[idx][0], -float(scores[idx])),
    )

    placed_rects: list[tuple[int, int, int, int]] = []
    gap = 2
    # 超高目标数时限制多层搜索预算，避免极端 max_det=300 拖慢批量或视频推理
    max_deep_search = 160

    for step, orig_idx in enumerate(order):
        x1, y1, x2, y2 = boxes[orig_idx]
        raw_lw, raw_lh = label_sizes[orig_idx]
        lw_px = min(max(1, raw_lw), max(1, img_w))
        lh_px = min(max(1, raw_lh), max(1, img_h))
        cx_mid = (x1 + x2 - lw_px) // 2

        candidates: list[tuple[float, int, int]] = [
            (0.0, x1, y1 - lh_px - 1),
            (0.3, cx_mid, y1 - lh_px - 1),
            (0.5, x2 - lw_px, y1 - lh_px - 1),
            (1.0, x1, y2 + 1),
            (1.3, cx_mid, y2 + 1),
            (1.5, x2 - lw_px, y2 + 1),
        ]
        if step < max_deep_search:
            if (y2 - y1) >= lh_px + 8 and (x2 - x1) >= max(24, int(lw_px * 0.55)):
                candidates.append((2.2, x1 + 2, y1 + 2))
                candidates.append((2.4, x1 + 2, y2 - lh_px - 2))
            for layer in range(1, 5):
                y_up = y1 - lh_px - 1 - layer * (lh_px + gap)
                y_dn = y2 + 1 + layer * (lh_px + gap)
                for align_idx, cx in enumerate((x1, cx_mid, x2 - lw_px)):
                    candidates.append((2.5 + layer * 1.2 + align_idx * 0.2, cx, y_up))
                    candidates.append((2.8 + layer * 1.2 + align_idx * 0.2, cx, y_dn))

        anchor_x = x1 + lw_px / 2.0
        anchor_y = y1 - lh_px / 2.0
        best_rect: tuple[int, int, int, int] = (0, 0, lw_px, lh_px)
        best_cost = float("inf")
        seen_xy: set[tuple[int, int]] = set()

        for tier_pen, raw_x, raw_y in candidates:
            cx = max(0, min(max(0, img_w - lw_px), int(raw_x)))
            cy = max(0, min(max(0, img_h - lh_px), int(raw_y)))
            if (cx, cy) in seen_xy:
                continue
            seen_xy.add((cx, cy))
            cand = (cx, cy, cx + lw_px, cy + lh_px)

            lbl_ov = 0
            for pr in placed_rects:
                if cand[0] - 1 < pr[2] + 1 and cand[2] + 1 > pr[0] - 1 and cand[1] - 1 < pr[3] + 1 and cand[3] + 1 > pr[1] - 1:
                    lbl_ov += _rect_intersect_area(cand, pr, pad=1)

            other_ov = 0
            if step < max_deep_search:
                for b_idx, ob in enumerate(boxes):
                    if b_idx == orig_idx:
                        continue
                    if cand[0] < ob[2] and cand[2] > ob[0] and cand[1] < ob[3] and cand[3] > ob[1]:
                        other_ov += _rect_intersect_area(cand, ob)

            self_ov = _rect_intersect_area(cand, (x1, y1, x2, y2))
            dist = math.hypot((cx + lw_px / 2.0) - anchor_x, (cy + lh_px / 2.0) - anchor_y)
            cost = (
                lbl_ov * 2000.0
                + other_ov * 0.8
                + self_ov * 1.2
                + tier_pen * 35.0
                + dist * 0.12
            )
            if cost < best_cost:
                best_cost = cost
                best_rect = cand
                if lbl_ov == 0 and other_ov == 0 and self_ov == 0 and tier_pen == 0.0:
                    break

        placed_rects.append(best_rect)
        placements[orig_idx] = best_rect

    return placements


def _call_result_plot_compat(
    result: object,
    *,
    show_labels: bool,
    show_conf: bool,
    line_width: int | None,
) -> np.ndarray:
    """兼容调用 Ultralytics `Results.plot(...)`；若遇到测试桩仅支持无参 `plot()` 时自动降级。"""
    plot_fn = getattr(result, "plot")
    plot_kwargs: dict[str, object] = {
        "labels": bool(show_labels),
        "conf": bool(show_conf),
    }
    if line_width is not None:
        plot_kwargs["line_width"] = int(line_width)
    try:
        return np.asarray(plot_fn(**plot_kwargs))
    except TypeError:
        return np.asarray(plot_fn())


def _normalize_bgr_canvas(raw_img: object) -> np.ndarray | None:
    """将 `Results.orig_img`（ndarray 或 Tensor）规范化为可写连续 3 通道 BGR uint8 副本。"""
    arr = _to_numpy_2d_or_1d(raw_img)
    if arr is None or arr.ndim < 2:
        return None
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[2] not in {1, 3, 4}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr.copy())


def _lookup_class_name(names: object, cls_id: int) -> str:
    """从 Ultralytics `names` 字典或序列解析类别名称，缺失时回退为编号字符串。"""
    if isinstance(names, Mapping):
        if cls_id in names:
            return str(names[cls_id])
        if str(cls_id) in names:
            return str(names[str(cls_id)])
    elif isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= cls_id < len(names):
            return str(names[cls_id])
    return str(cls_id)


def render_predict_result(
    result: object,
    render_config: PredictRenderConfig | None = None,
) -> np.ndarray:
    """将单帧 Ultralytics 推理结果渲染为 BGR `uint8` 图像。

    模式说明与回退行为：
        - `style="ultralytics"`：直接透传 `labels`、`conf`、`line_width` 调用 `result.plot(...)`；
        - `style="compact"`：针对目标检测框执行两阶段紧凑绘制：
          1. 按画布尺寸与目标密集度自适应细线宽与紧凑字号，统一先绘制全部检测框避免框线切穿文字；
          2. 置信度严格保留两位小数（如 `player 0.94`），按 `show_labels` / `show_conf` 组合文案；
          3. 使用 `_layout_non_overlapping_labels` 进行二维防重叠排版，以 `label_alpha` 半透明底色与
             1px 细描边绘制标签（错层远离原框时附带 1px 细引导线），并依据混合后局部亮度自动切换黑白高对比文字；
        - 若输入为缺少 `orig_img` 的轻量测试桩，或包含分割掩码/关键点/旋转框/分类概率等非纯检测结果，
          自动安全回退调用 `result.plot(...)`，不丢失额外视觉元素。
    """
    cfg = normalize_predict_render_config(render_config)

    # 先读取原图用于估算 Ultralytics 挡位线宽及判断是否可执行紧凑纯检测渲染
    canvas = _normalize_bgr_canvas(getattr(result, "orig_img", None))
    has_non_box_extras = any(
        getattr(result, attr, None) is not None for attr in ("masks", "keypoints", "obb", "probs")
    )
    if canvas is None:
        _, _, _, ultra_lw = _resolve_render_scale(
            720, 1280, num_boxes=0, median_box_dim=0.0, line_width_mode=cfg.line_width_mode
        )
        return _call_result_plot_compat(
            result,
            show_labels=cfg.show_labels,
            show_conf=cfg.show_conf,
            line_width=ultra_lw,
        )

    img_h, img_w = canvas.shape[:2]
    if cfg.style == "ultralytics" or has_non_box_extras:
        _, _, _, ultra_lw = _resolve_render_scale(
            img_h, img_w, num_boxes=0, median_box_dim=0.0, line_width_mode=cfg.line_width_mode
        )
        return _call_result_plot_compat(
            result,
            show_labels=cfg.show_labels,
            show_conf=cfg.show_conf,
            line_width=ultra_lw,
        )

    boxes_obj = getattr(result, "boxes", None)
    xyxy_arr = _to_numpy_2d_or_1d(getattr(boxes_obj, "xyxy", None)) if boxes_obj is not None else None
    if xyxy_arr is None or xyxy_arr.size == 0:
        return canvas
    if xyxy_arr.ndim == 1 and xyxy_arr.size == 4:
        xyxy_arr = xyxy_arr.reshape(1, 4)
    if xyxy_arr.ndim != 2 or xyxy_arr.shape[1] < 4:
        return canvas

    cls_arr = _to_numpy_2d_or_1d(getattr(boxes_obj, "cls", None))
    conf_arr = _to_numpy_2d_or_1d(getattr(boxes_obj, "conf", None))
    id_arr = (
        _to_numpy_2d_or_1d(getattr(boxes_obj, "id", None))
        if getattr(boxes_obj, "is_track", False)
        else None
    )
    names = getattr(result, "names", None)

    valid_boxes: list[tuple[int, int, int, int]] = []
    valid_cls: list[int] = []
    valid_scores: list[float] = []
    valid_labels: list[str] = []

    for i in range(xyxy_arr.shape[0]):
        rx1, ry1, rx2, ry2 = (float(xyxy_arr[i, 0]), float(xyxy_arr[i, 1]), float(xyxy_arr[i, 2]), float(xyxy_arr[i, 3]))
        x1 = max(0, min(img_w - 1, int(round(rx1))))
        y1 = max(0, min(img_h - 1, int(round(ry1))))
        x2 = max(0, min(img_w - 1, int(round(rx2))))
        y2 = max(0, min(img_h - 1, int(round(ry2))))
        if x2 <= x1 or y2 <= y1:
            continue

        c_id = int(round(float(cls_arr[i]))) if (cls_arr is not None and i < len(cls_arr)) else 0
        score = float(conf_arr[i]) if (conf_arr is not None and i < len(conf_arr)) else 1.0
        track_prefix = (
            f"id:{int(round(float(id_arr[i])))} "
            if (id_arr is not None and i < len(id_arr) and math.isfinite(float(id_arr[i])))
            else ""
        )
        cls_name = f"{track_prefix}{_lookup_class_name(names, c_id)}"

        if cfg.show_labels and cfg.show_conf:
            label_text = f"{cls_name} {score:.2f}"
        elif cfg.show_labels:
            label_text = cls_name
        elif cfg.show_conf:
            label_text = f"{score:.2f}"
        else:
            label_text = ""

        valid_boxes.append((x1, y1, x2, y2))
        valid_cls.append(c_id)
        valid_scores.append(score)
        valid_labels.append(label_text)

    if not valid_boxes:
        return canvas

    box_dims = [min(b[2] - b[0], b[3] - b[1]) for b in valid_boxes]
    median_dim = float(np.median(box_dims)) if box_dims else 0.0
    box_lw, font_px, cv_font_scale, _ = _resolve_render_scale(
        img_h,
        img_w,
        num_boxes=len(valid_boxes),
        median_box_dim=median_dim,
        line_width_mode=cfg.line_width_mode,
    )

    # 阶段 1：统一绘制全部检测框（按置信度从低到高，高置信度框边压在上方）
    draw_box_order = sorted(range(len(valid_boxes)), key=lambda idx: valid_scores[idx])
    for idx in draw_box_order:
        x1, y1, x2, y2 = valid_boxes[idx]
        color = _get_class_color_bgr(valid_cls[idx])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=box_lw, lineType=cv2.LINE_AA)

    if not any(valid_labels):
        return canvas

    # 阶段 2：测量标签文本尺寸并执行二维防重叠排版
    need_unicode = any(not text.isascii() for text in valid_labels if text)
    pil_font = _load_compact_truetype_font(font_px, need_unicode)
    pad_x = max(3, round(font_px * 0.28))
    pad_y = max(2, round(font_px * 0.16))

    label_indices: list[int] = []
    label_boxes: list[tuple[int, int, int, int]] = []
    label_sizes: list[tuple[int, int]] = []
    label_scores: list[float] = []
    text_metrics: list[tuple[int, int]] = []  # PIL 记录 (offset_x, offset_y)；OpenCV 记录 (text_h, baseline)

    for idx, text in enumerate(valid_labels):
        if not text:
            continue
        if pil_font is not None:
            bbox = pil_font.getbbox(text)  # type: ignore[union-attr]
            tw = max(1, int(bbox[2] - bbox[0]))
            th = max(1, int(bbox[3] - bbox[1]))
            text_metrics.append((int(bbox[0]), int(bbox[1])))
        else:
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, cv_font_scale, 1
            )
            tw = max(1, int(tw))
            th = max(1, int(th))
            text_metrics.append((th, int(baseline)))
        label_indices.append(idx)
        label_boxes.append(valid_boxes[idx])
        label_sizes.append((tw + pad_x * 2, th + pad_y * 2))
        label_scores.append(valid_scores[idx])

    placed_rects = _layout_non_overlapping_labels(
        label_boxes,
        label_sizes,
        label_scores,
        img_w=img_w,
        img_h=img_h,
        avoid_overlap=cfg.avoid_overlap,
    )

    # 阶段 3：在叠加层绘制半透明标签底色与错层细引导线，随后按 `label_alpha` 融合到画布
    alpha = cfg.label_alpha
    if alpha < 0.999:
        overlay = canvas.copy()
    else:
        overlay = canvas

    for item_i, orig_idx in enumerate(label_indices):
        x1, y1, x2, y2 = valid_boxes[orig_idx]
        rx1, ry1, rx2, ry2 = placed_rects[item_i]
        color = _get_class_color_bgr(valid_cls[orig_idx])
        # 标签因防重叠被推离原框超过 4 像素时，画一条 1px 细引导线明确归属关系
        if ry2 < y1 - 4:
            gx = max(x1 + 1, min(x2 - 1, (rx1 + rx2) // 2))
            cv2.line(overlay, (gx, ry2), (gx, y1), color, 1, cv2.LINE_AA)
        elif ry1 > y2 + 4:
            gx = max(x1 + 1, min(x2 - 1, (rx1 + rx2) // 2))
            cv2.line(overlay, (gx, y2), (gx, ry1), color, 1, cv2.LINE_AA)
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color, -1)

    if alpha < 0.999:
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)

    # 极细同色描边勾勒标签边界，兼顾半透明通透感与分组辨识度
    for item_i, orig_idx in enumerate(label_indices):
        rx1, ry1, rx2, ry2 = placed_rects[item_i]
        color = _get_class_color_bgr(valid_cls[orig_idx])
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), color, 1)

    # 依据融合后标签区域的 WCAG 相对亮度自动选高对比黑白文字（临界点约 0.22，保证亮青/黄底配深字锐利清晰）
    def _srgb_rel_luma(r_val: float, g_val: float, b_val: float) -> float:
        def _lin(ch: float) -> float:
            c = max(0.0, min(1.0, ch / 255.0))
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        return 0.2126 * _lin(r_val) + 0.7152 * _lin(g_val) + 0.0722 * _lin(b_val)

    txt_colors_rgb: list[tuple[int, int, int]] = []
    for item_i, orig_idx in enumerate(label_indices):
        rx1, ry1, rx2, ry2 = placed_rects[item_i]
        roi = canvas[ry1 : max(ry1 + 1, ry2), rx1 : max(rx1 + 1, rx2)]
        if roi.size > 0:
            mean_bgr = roi.mean(axis=(0, 1))
            rel_luma = _srgb_rel_luma(float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))
        else:
            b_c, g_c, r_c = _get_class_color_bgr(valid_cls[orig_idx])
            rel_luma = _srgb_rel_luma(float(r_c), float(g_c), float(b_c))
        txt_colors_rgb.append((16, 22, 32) if rel_luma >= 0.22 else (255, 255, 255))

    # 阶段 4：绘制不透明锐利文字（优先 Pillow TrueType，回退 OpenCV）
    if pil_font is not None:
        from PIL import Image, ImageDraw

        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for item_i, orig_idx in enumerate(label_indices):
            rx1, ry1, _rx2, _ry2 = placed_rects[item_i]
            ox, oy = text_metrics[item_i]
            draw.text(
                (rx1 + pad_x - ox, ry1 + pad_y - oy),
                valid_labels[orig_idx],
                font=pil_font,
                fill=txt_colors_rgb[item_i],
            )
        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    for item_i, orig_idx in enumerate(label_indices):
        rx1, ry1, _rx2, _ry2 = placed_rects[item_i]
        th, _baseline = text_metrics[item_i]
        r_t, g_t, b_t = txt_colors_rgb[item_i]
        cv2.putText(
            canvas,
            valid_labels[orig_idx],
            (rx1 + pad_x, ry1 + pad_y + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            cv_font_scale,
            (b_t, g_t, r_t),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def build_predict_kwargs(
    weights_path: Path,
    source_path: Path,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int | tuple[int, int] = 640,
    device: str = "0",
    nms: bool | None = None,
    rect: bool = True,
    max_det: int = 300,
    preprocess_mode: str | None = None,
    padding_value: int | None = None,
    render_config: PredictRenderConfig | None = None,
    is_video: bool = False,
    output_dir: Path | None = None,
    metadata: WeightsMetadata | None = None,
) -> tuple[dict[str, object], list[str]]:
    """校验并组装传给 Ultralytics `model.predict(**kwargs)` 的完整参数字典与提示列表。

    行为约束：
        - `.engine`（TensorRT）模型仅支持 CUDA GPU，在 CPU 设备模式下抛出 `ValueError`；
        - `.pt` 权重传入 `nms=False` 时启用 YOLO26 `one2one` 端到端无 NMS 头，`nms=None` 走 `one2many` + 外部 NMS；
        - `rect=True` 对 `.pt` 与动态导出模型启用最小矩形 LetterBox 填充，`rect=False` 强制按设定尺寸固定填充；
        - 已导出模型（`.onnx` / `.torchscript` / `.engine`）已固化检测头，不传 `nms` 覆盖；
          静态导出模型（`export_dynamic=False`）自动对齐导出尺寸并固定 `rect=False`；
        - 当预处理为 `stretch`（直接拉伸）时，强制 `rect=False`；
        - 视频推理且 `render_config.style == "compact"` 时设置 `save=False` 改由引擎逐帧渲染写出；
          其余视频推理场景交由 Ultralytics `save=True` 并透传 `show_labels` / `show_conf` / `line_width`。
    """
    fmt = detect_model_format(weights_path)
    is_exported = is_exported_model_path(weights_path)
    dev_str = str(device).strip() or "0"
    if fmt == "engine" and dev_str.lower() == "cpu":
        raise ValueError("TensorRT (.engine) 模型仅支持在 CUDA GPU 上推理，请启用 GPU 加速。")

    max_det_int = int(max_det)
    if max_det_int < 1:
        raise ValueError(f"最大检测框数 max_det 必须 >= 1，当前值: {max_det_int}")

    _mode_key, resize_mode, _pad_align, _pad_val = resolve_effective_predict_preprocess(
        is_exported=is_exported,
        preprocess_mode=preprocess_mode,
        padding_value=padding_value,
        metadata=metadata,
    )
    norm_render = normalize_predict_render_config(render_config) if render_config is not None else None

    imgsz_arg, pair = _format_imgsz_arg(imgsz)
    effective_rect = False if resize_mode == "stretch" else bool(rect)
    warnings: list[str] = []

    pred_kwargs: dict[str, object] = {
        "source": str(source_path),
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": imgsz_arg,
        "device": dev_str,
        "rect": effective_rect,
        "max_det": max_det_int,
        "stream": True,
    }

    if is_video:
        if norm_render is not None and norm_render.style == "compact":
            pred_kwargs["save"] = False
        else:
            pred_kwargs["save"] = True
            if output_dir is not None:
                pred_kwargs["project"] = str(output_dir.resolve())
                pred_kwargs["name"] = "predict"
                pred_kwargs["exist_ok"] = True
            if norm_render is not None:
                pred_kwargs["show_labels"] = norm_render.show_labels
                pred_kwargs["show_conf"] = norm_render.show_conf
                _, _, _, ultra_lw = _resolve_render_scale(
                    pair[0],
                    pair[1],
                    num_boxes=0,
                    median_box_dim=0.0,
                    line_width_mode=norm_render.line_width_mode,
                )
                if ultra_lw is not None:
                    pred_kwargs["line_width"] = ultra_lw
    else:
        pred_kwargs["save"] = False

    if is_exported:
        if nms is not None:
            warnings.append(f"已导出 {fmt.upper()} 模型的检测头已固化，忽略额外指定的 nms={nms}。")
        if metadata is not None and metadata.export_dynamic is False:
            pred_kwargs["rect"] = False
            if metadata.train_imgsz is not None:
                mh, mw = metadata.train_imgsz
                if pair != metadata.train_imgsz:
                    warnings.append(
                        f"静态 {fmt.upper()} 模型固定使用导出尺寸 {format_imgsz_text(metadata.train_imgsz)}，"
                        "推理时将自动按该固定尺寸填充。"
                    )
                pred_kwargs["imgsz"] = mh if mh == mw else [mh, mw]
        if metadata is not None and metadata.deploy_schema == ONNX_DEPLOY_SCHEMA_VERSION and metadata.export_dynamic:
            if metadata.train_imgsz is not None:
                height, width = metadata.train_imgsz
                pred_kwargs["imgsz"] = height if height == width else [height, width]
            if metadata.preprocess_auto_rect is not None:
                pred_kwargs["rect"] = resize_mode != "stretch" and metadata.preprocess_auto_rect
        if metadata is not None and metadata.export_nms_mode == "embedded":
            warnings.append(
                "当前导出模型已在计算图内嵌入 NMS；若导出时图内 NMS Conf 高于当前 Conf，无法召回已被图内滤除的目标。"
            )
    else:
        if nms is not None:
            pred_kwargs["nms"] = nms
            if nms is False and metadata is not None and not metadata.supports_end2end:
                warnings.append("当前 .pt 权重未检测到 one2one 端到端分支，推理将回退为 one2many 检测头。")
        elif metadata is not None and not metadata.one2many_available and metadata.supports_end2end:
            warnings.append("当前 .pt 权重仅保留 one2one 端到端分支，实际将按端到端无 NMS 推理。")

    return pred_kwargs, warnings


def format_submitted_predict_config(
    weights_path: Path,
    pred_kwargs: Mapping[str, object],
    *,
    metadata: WeightsMetadata | None = None,
    preprocess_mode: str | None = None,
    padding_value: int | None = None,
    render_config: PredictRenderConfig | None = None,
    warnings: Sequence[str] = (),
) -> str:
    """将提交给 Ultralytics `model.predict()` 的推理参数与提示格式化为可读多行日志。"""
    fmt = detect_model_format(weights_path)
    is_exported = is_exported_model_path(weights_path)
    if is_exported:
        mode_map = {
            "end2end": "已固化 one2one 端到端无 NMS",
            "embedded": "已固化图内嵌入 NMS",
            "raw": "已固化原始 one2many (外部 NMS)",
        }
        head_desc = mode_map.get(metadata.export_nms_mode if metadata else None, "由导出模型决定")
    elif pred_kwargs.get("nms") is False:
        head_desc = "end2end (YOLO26 one2one 无 NMS)"
    else:
        head_desc = "one2many (外部 NMS，PT 默认)"

    mode_key, resize_mode, pad_align, pad_val = resolve_effective_predict_preprocess(
        is_exported=is_exported,
        preprocess_mode=preprocess_mode,
        padding_value=padding_value,
        metadata=metadata,
    )
    if resize_mode == "stretch":
        pad_desc = "直接拉伸 Stretch (rect=False)"
    else:
        pad_desc = "最小矩形填充 (rect=True)" if pred_kwargs.get("rect") else "固定尺寸填充 (rect=False)"

    summary_dict: dict[str, object] = {
        "model": weights_path.name,
        "model_format": fmt,
        "head_mode": head_desc,
        "preprocess_mode": mode_key,
        "padding_align": pad_align,
        "padding_value": pad_val,
        "pad_mode": pad_desc,
        "imgsz": pred_kwargs.get("imgsz"),
        "rect": pred_kwargs.get("rect"),
        "conf": pred_kwargs.get("conf"),
        "iou": pred_kwargs.get("iou"),
        "max_det": pred_kwargs.get("max_det"),
        "device": pred_kwargs.get("device"),
    }
    if "nms" in pred_kwargs:
        summary_dict["nms"] = pred_kwargs["nms"]
    if render_config is not None:
        eff_render = normalize_predict_render_config(render_config)
        summary_dict["render_style"] = format_predict_render_summary(eff_render)

    json_lines = json.dumps(summary_dict, ensure_ascii=False, indent=2, default=str).splitlines()
    if json_lines:
        json_lines[0] = f'"predict_arguments": {json_lines[0]}'
    lines = ["=== 推理效果测试配置 (提交至 Ultralytics) ===", *json_lines]
    for warn in warnings:
        lines.append(f"[提示] {warn}")
    return "\n".join(lines)


def _read_video_fps(video_path: Path, default_fps: float = 25.0) -> float:
    """读取源视频帧率（fps）；读取失败或数值异常时回退为 `default_fps`。"""
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0.1 or fps > 240.0:
        return default_fps
    return fps


def _open_video_writer_safe(
    target_path: Path, fps: float, frame_w: int, frame_h: int
) -> tuple[cv2.VideoWriter | None, Path | None]:
    """打开 MP4 视频写入器；若目标路径含非 ASCII 导致 OpenCV 打不开，自动改写至临时 ASCII 文件。

    返回值说明：
        `(writer, temp_path)`；`writer` 为 `None` 表示当前环境无法创建 `mp4v` 写入器；
        若使用临时文件，`temp_path` 非空，调用方关闭 `writer` 后须负责将其移动到 `target_path`。
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    if str(target_path).isascii():
        writer = cv2.VideoWriter(str(target_path), fourcc, fps, (frame_w, frame_h))
        if writer.isOpened():
            return writer, None
        writer.release()

    tmp_file = NamedTemporaryFile(prefix="yolostudio_pred_", suffix=".mp4", delete=False)
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (frame_w, frame_h))
    if writer.isOpened():
        return writer, tmp_path
    writer.release()
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass
    return None, None


def predict_source(
    weights_path: Path,
    source_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int | tuple[int, int] = 640,
    device: str = "0",
    *,
    nms: bool | None = None,
    rect: bool = True,
    max_det: int = 300,
    preprocess_mode: str | None = None,
    padding_value: int | None = None,
    render_config: PredictRenderConfig | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> list[Path]:
    """对单张图片、图片目录或视频运行推理并将带框结果写入 `output_dir`。

    模型与参数支持：
        - 同时支持 `.pt` 源权重与已导出部署模型（`.onnx` / `.torchscript` / `.engine`）；
        - `imgsz` 支持正整数（方形）或 `(高, 宽)` 矩形二元组；
        - `nms` 仅对 `.pt` 权重生效（`None`=默认 `one2many` + 外部 NMS，`False`=YOLO26 `one2one` 无 NMS）；
        - `rect=True` 为最小矩形填充（PT 默认），`rect=False` 为严格按 `imgsz` 补灰边固定尺寸填充；
        - `preprocess_mode` / `padding_value` 支持复现 `letterbox_center` / `letterbox_top_left` / `stretch`
          预处理契约；已导出模型未显式传入时自动沿用其内嵌元数据中的预处理设置；
        - `render_config` 控制框线粗细、紧凑防重叠半透明样式及类别/置信度显示（默认启用紧凑半透明防重叠，
          置信度保持两位小数如 `0.94`）。

    输出与内存约束：
        紧凑模式下视频逐帧经 `render_predict_result` 写出至 `output_dir / "predict" / "<名>_pred.mp4"`
        （编码器不可用时安全回退 Ultralytics `save=True`）；Ultralytics 默认模式下视频交由 Ultralytics
        写出至 `output_dir / "predict"`；
        图片输入通过 `cv2.imencode` + `Path.write_bytes` 写出为 `*_pred.jpg` 以支持中文路径，
        同名图片自动消歧不互相覆盖。两种输入均以 `stream=True` 逐帧迭代，
        内存占用与视频长度或图片数量无关。模型或输入路径缺失时抛出 `FileNotFoundError`。
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"推理输入不存在: {source_path}")

    effective_render = normalize_predict_render_config(render_config)

    meta: WeightsMetadata | None = None
    try:
        meta = inspect_weights_metadata(weights_path)
    except Exception:
        meta = None

    is_exported = is_exported_model_path(weights_path)
    mode_key, resize_mode, pad_align, pad_val = resolve_effective_predict_preprocess(
        is_exported=is_exported,
        preprocess_mode=preprocess_mode,
        padding_value=padding_value,
        metadata=meta,
    )

    is_video = source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTENSIONS
    pred_kwargs, warnings = build_predict_kwargs(
        weights_path,
        source_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        nms=nms,
        rect=rect,
        max_det=max_det,
        preprocess_mode=mode_key,
        padding_value=pad_val,
        render_config=effective_render,
        is_video=is_video,
        output_dir=output_dir,
        metadata=meta,
    )
    _emit_multiline_log(
        log_callback,
        format_submitted_predict_config(
            weights_path,
            pred_kwargs,
            metadata=meta,
            preprocess_mode=mode_key,
            padding_value=pad_val,
            render_config=effective_render,
            warnings=warnings,
        ),
    )

    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_torch_cpu_threads()
    model = YOLO(str(weights_path))

    # 仅当预处理偏离 Ultralytics 默认居中灰边 114 LetterBox 且为目标检测任务时，启用自定义预处理预测器
    call_kwargs = dict(pred_kwargs)
    needs_custom_pre = mode_key != DEFAULT_EXPORT_PREPROCESS_MODE or (
        pad_val is not None and pad_val != DEFAULT_EXPORT_PADDING_VALUE
    )
    if needs_custom_pre:
        model_task = str(getattr(model, "task", "") or (meta.task if meta else "") or "detect").lower()
        if model_task == "detect":
            try:
                call_kwargs["predictor"] = _build_custom_preprocess_predictor(
                    resize_mode=resize_mode,
                    padding_align=pad_align,
                    padding_value=pad_val if pad_val is not None else DEFAULT_EXPORT_PADDING_VALUE,
                )
            except Exception as exc:
                if log_callback:
                    log_callback(f"[提示] 自定义预处理预测器加载失败，已回退默认预处理: {exc}")
        elif log_callback:
            log_callback(f"[提示] 当前任务 ({model_task}) 暂不支持自定义预处理预览，已使用 Ultralytics 默认预处理。")

    saved_outputs: list[Path] = []
    with capture_thread_output(log_callback):
        if is_video and effective_render.style == "compact":
            fps = _read_video_fps(source_path)
            target_video = output_dir / "predict" / f"{source_path.stem}_pred.mp4"
            writer: cv2.VideoWriter | None = None
            tmp_video_path: Path | None = None
            wrote_frames = 0
            fallback_needed = False
            try:
                for r in model.predict(**call_kwargs):
                    frame = render_predict_result(r, effective_render)
                    if frame is None or frame.ndim < 2:
                        continue
                    fh, fw = frame.shape[:2]
                    if writer is None:
                        writer, tmp_video_path = _open_video_writer_safe(target_video, fps, fw, fh)
                        if writer is None:
                            fallback_needed = True
                            break
                    writer.write(frame)
                    wrote_frames += 1
            finally:
                if writer is not None:
                    writer.release()

            if fallback_needed:
                if tmp_video_path is not None:
                    tmp_video_path.unlink(missing_ok=True)
                if log_callback:
                    log_callback("[提示] 当前环境无法初始化 MP4 视频编码器，已回退 Ultralytics 默认视频写出。")
                fb_kwargs = dict(call_kwargs)
                fb_kwargs["save"] = True
                fb_kwargs["project"] = str(output_dir.resolve())
                fb_kwargs["name"] = "predict"
                fb_kwargs["exist_ok"] = True
                fb_kwargs["show_labels"] = effective_render.show_labels
                fb_kwargs["show_conf"] = effective_render.show_conf
                save_dir: Path | None = None
                for r in model.predict(**fb_kwargs):
                    if save_dir is None and getattr(r, "save_dir", None):
                        save_dir = Path(r.save_dir)
                if save_dir is not None:
                    saved_outputs = sorted(save_dir.glob("*"))
            elif wrote_frames > 0:
                if tmp_video_path is not None and tmp_video_path.is_file():
                    target_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(tmp_video_path), str(target_video))
                if target_video.is_file():
                    saved_outputs.append(target_video)
            elif tmp_video_path is not None:
                tmp_video_path.unlink(missing_ok=True)
        elif is_video:
            save_dir = None
            for r in model.predict(**call_kwargs):
                if save_dir is None and getattr(r, "save_dir", None):
                    save_dir = Path(r.save_dir)
            if save_dir is not None:
                saved_outputs = sorted(save_dir.glob("*"))
        else:
            used_names: set[str] = set()
            for idx, r in enumerate(model.predict(**call_kwargs), start=1):
                plotted = render_predict_result(r, effective_render)
                src = Path(r.path) if getattr(r, "path", None) else Path(f"pred_{idx}.jpg")
                out_file = _unique_pred_path(output_dir, src, used_names)
                ok, encoded = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    out_file.write_bytes(encoded.tobytes())
                    saved_outputs.append(out_file)
    return saved_outputs


# 模型导出 NMS 模式选项：(内部键名, 界面显示文案, 对应 Ultralytics `nms` 参数值)
EXPORT_NMS_MODES: list[tuple[str, str, bool | None]] = [
    ("raw", "原始输出 · 外部 NMS (默认，兼容 YOLOv8/11)", None),
    ("end2end", "端到端无 NMS · YOLO26 原生 (输出 [B,N,6])", False),
    ("embedded", "图内嵌入 NMS (输出 [B,N,6])", True),
]
EXPORT_NMS_KEY_TO_VALUE: dict[str, bool | None] = {
    key: nms_val for key, _, nms_val in EXPORT_NMS_MODES
}
EXPORT_NMS_VALUE_TO_KEY: dict[bool | None, str] = {
    nms_val: key for key, _, nms_val in EXPORT_NMS_MODES
}

# ONNX Opset 界面下拉选项：(显示文案, 传给 Ultralytics 的值；`None` 表示自动选择)
EXPORT_OPSET_OPTIONS: list[tuple[str, int | None]] = [
    ("自动 (推荐)", None),
    *((str(ver), ver) for ver in range(11, 22)),
]

# 支持 ONNX 中间图专属参数（`opset`、`simplify`）的导出格式集合
ONNX_BACKED_EXPORT_FORMATS: frozenset[str] = frozenset(
    {"onnx", "engine", "ascend", "mnn", "rknn", "qnn", "deepx", "xilinx"}
)
# 支持动态输入尺寸与动态批大小（`dynamic=True`）的导出格式集合
DYNAMIC_EXPORT_FORMATS: frozenset[str] = frozenset(
    {"onnx", "torchscript", "engine", "openvino", "coreml", "mnn"}
)


@dataclass(frozen=True)
class WeightsMetadata:
    """从 `.pt` 权重或已导出部署模型（`.onnx` / `.torchscript` / `.engine`）只读解析出的元数据摘要。

    字段说明：
        `weights_path`：模型文件路径。
        `task`：任务类型（如 `"detect"`、`"segment"`、`"pose"`、`"obb"`、`"classify"`），缺失时为空字符串。
        `train_imgsz`：`.pt` 权重的训练输入尺寸或已导出模型的导出输入尺寸 `(高, 宽)` 像素元组；
            单边整数（如 `960`）规范为 `(960, 960)`，未记录有效尺寸时为 `None`。
        `class_names`：按类别索引升序排列的 `(类别编号, 类别名称)` 元组，例如 `((0, "person"), (1, "car"))`。
        `strides`：检测头步长元组（像素，默认 `(8, 16, 32)`），用于校验尺寸对齐与估算候选框数量。
        `channels`：模型输入通道数（通常为 `3`）。
        `supports_end2end`：检测头是否包含/固化 YOLO26 `one2one` 原生无 NMS 端到端分支（`nms=False` 可用）。
        `one2many_available`：检测头是否保留 `one2many` 原始分支（若已被融合移除或已导出为 `end2end`，则为 `False`）。
        `epoch`：`.pt` 权重保存时的训练轮次；预训练权重、导出模型或缺失时为 `None`。
        `model_format`：模型格式键名（`"pt"` / `"onnx"` / `"torchscript"` / `"engine"` 等）。
        `is_exported`：是否为已导出的部署模型（非 `.pt`）。
        `export_nms_mode`：已导出模型固化的后处理模式（`"raw"` / `"end2end"` / `"embedded"`），未知或 `.pt` 时为 `None`。
        `export_dynamic`：已导出模型是否支持动态输入维度；未知或 `.pt` 时为 `None`。
        `export_batch`：已导出模型记录的批大小；未知或 `.pt` 时为 `None`。
        `deploy_schema`：已导出 ONNX 模型写入的 YoloStudio 部署元数据协议版本（如 `"yolostudio.deploy.v1"`）。
        `preprocess_mode`：已导出 ONNX 模型元数据记录的预处理策略键名（`"letterbox_center"` / `"letterbox_top_left"` / `"stretch"`）。
        `preprocess_padding_value`：已导出 ONNX 模型元数据记录的 LetterBox 补边灰度（`0 ~ 255`；`stretch` 或未知为 `None`）。
        `preprocess_auto_rect`：已导出 ONNX 模型元数据记录的是否允许动态最小矩形缩放（`True` / `False` / `None`）。
    """

    weights_path: Path
    task: str = ""
    train_imgsz: tuple[int, int] | None = None
    class_names: tuple[tuple[int, str], ...] = ()
    strides: tuple[int, ...] = (8, 16, 32)
    channels: int = 3
    supports_end2end: bool = False
    one2many_available: bool = True
    epoch: int | None = None
    model_format: str = "pt"
    is_exported: bool = False
    export_nms_mode: str | None = None
    export_dynamic: bool | None = None
    export_batch: int | None = None
    deploy_schema: str | None = None
    preprocess_mode: str | None = None
    preprocess_padding_value: int | None = None
    preprocess_auto_rect: bool | None = None

    @property
    def max_stride(self) -> int:
        """返回检测头最大步长（像素，至少为 `32` 或实际最大步长）。"""
        return max(self.strides) if self.strides else 32

    @property
    def num_classes(self) -> int:
        """返回权重记录的类别总数；未记录类别时返回 `0`。"""
        return len(self.class_names)


@dataclass
class ExportConfig:
    """YOLO26 部署模型导出配置。

    字段说明：
        `weights_path`：待导出的源 `.pt` 权重文件路径。
        `export_format`：目标格式（如 `"onnx"`、`"torchscript"`、`"engine"` 等，别名 `tensorrt`/`trt` 自动归一为 `"engine"`）。
        `imgsz`：导出输入尺寸，支持单整数（方形，如 `960`）、`(高, 宽)` 二元组（如 `(544, 960)`）或 `None`；
            为 `None` 时自动沿用 `.pt` 权重中记录的训练尺寸（若权重未记录则交给 Ultralytics 默认）。
        `batch`：静态批大小（或 TensorRT 动态模式下的最大批大小），须 `>= 1`。
        `dynamic`：是否启用动态输入尺寸与动态批大小。
        `nms`：三态后处理开关：`None`=原始 `one2many` 输出（需外部 NMS），`False`=YOLO26 原生 `one2one`
            NMS-free 端到端头，`True`=图内嵌入 NMS 层。
        `max_det`：单图最大输出检测框数（仅 `nms=False` 或 `nms=True` 时生效），须 `>= 1`。
        `conf` / `iou`：仅 `nms=True`（图内嵌入 NMS）时生效的置信度与 IoU 阈值；为 `None` 时沿用 Ultralytics 默认值。
        `opset`：ONNX 算子集版本号（如 `12`、`17`）；`None` 表示由 Ultralytics 自动选择；非 ONNX 系列格式忽略。
        `simplify`：是否使用 `onnxslim` 简化 ONNX 计算图；非 ONNX 系列格式忽略。
        `half`：是否以 FP16 半精度导出（ONNX 在 CPU 上导出或设备为 CPU 时不支持并自动忽略）。
        `device`：导出计算设备标识（如 `"0"` 或 `"cpu"`；ONNX 导出固定切至 `"cpu"` 规避图追踪设备冲突）。
        `workspace`：TensorRT 工作空间大小（GiB，仅 `engine` 格式生效；为 `None` 时使用默认值）。
        `preprocess_mode`：写入 ONNX 部署元数据的预处理策略（`"letterbox_center"` / `"letterbox_top_left"` / `"stretch"`），
            仅作用于 ONNX 元数据，供下游推理端读取以统一前处理，不改动 Ultralytics 导出计算图。
        `padding_value`：写入 ONNX 部署元数据的 LetterBox 补边灰度值（`0 ~ 255`，默认 `114`；`stretch` 时忽略）。
        `preprocess_auto_rect`：写入 ONNX 部署元数据的 `auto_rect` 标志；为 `None` 时自动按
            `dynamic and preprocess_mode != "stretch"` 推导。
        `extra_args`：透传给 `model.export()` 的附加参数字典，同名键优先级高于结构化字段。
    """

    weights_path: Path
    export_format: str = "onnx"
    imgsz: int | tuple[int, int] | None = None
    batch: int = 1
    dynamic: bool = False
    nms: bool | None = None
    max_det: int = 300
    conf: float | None = None
    iou: float | None = None
    opset: int | None = None
    simplify: bool = True
    half: bool = False
    device: str = "0"
    workspace: float | None = None
    preprocess_mode: str = DEFAULT_EXPORT_PREPROCESS_MODE
    padding_value: int = DEFAULT_EXPORT_PADDING_VALUE
    preprocess_auto_rect: bool | None = None
    extra_args: dict[str, object] = field(default_factory=dict)


def normalize_export_format(export_format: str) -> str:
    """将导出格式名称规范化为小写标准键名，并将 `tensorrt`/`trt` 映射为 `engine`。"""
    fmt = str(export_format).strip().lower()
    if not fmt:
        raise ValueError("导出格式不能为空。")
    if fmt in {"tensorrt", "trt"}:
        return "engine"
    return fmt


def parse_export_nms_mode(raw: str | bool | None) -> bool | None:
    """将界面键名、CLI 选项或布尔值解析为 Ultralytics 三态 `nms` 参数（`None` / `False` / `True`）。"""
    if raw is None or isinstance(raw, bool):
        return raw
    token = str(raw).strip().lower()
    if token in {"raw", "none", "external", "default", ""}:
        return None
    if token in {"end2end", "false", "nms_free", "nms-free", "yolo26"}:
        return False
    if token in {"embedded", "true", "nms", "builtin"}:
        return True
    raise ValueError(
        f"无效的 NMS 模式 '{raw}'，可选值: raw(原始输出/外部NMS)、end2end(YOLO26端到端无NMS)、embedded(图内嵌入NMS)"
    )


def _extract_positive_int(value: object) -> int | None:
    """将非布尔整数或整数型浮点数安全转换为正整数，非法时返回 `None`。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        iv = int(value)
        return iv if iv > 0 else None
    return None


def _extract_imgsz_pair(raw: object) -> tuple[int, int] | None:
    """从权重配置字段中提取 `(高, 宽)` 正整数像素元组；无法解析时返回 `None`。"""
    single = _extract_positive_int(raw)
    if single is not None:
        return (single, single)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = list(raw)
        if len(items) == 1:
            one = _extract_positive_int(items[0])
            return (one, one) if one is not None else None
        if len(items) >= 2:
            h = _extract_positive_int(items[0])
            w = _extract_positive_int(items[1])
            if h is not None and w is not None:
                return (h, w)
    return None


def normalize_export_imgsz(raw: int | Sequence[int] | str) -> tuple[int, int]:
    """将单整数、二元组或字符串（如 `'960'`、`'544,960'`、`'544x960'`）规范为 `(高, 宽)` 像素元组。

    校验规则：
        高和宽均须为 `32 ~ 8192` 范围内的整数；格式或数值越界时抛出中文 `ValueError`。
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("导出尺寸不能为空。")
        parts = [p.strip() for p in re.split(r"[,xX×*:\s]+", text) if p.strip()]
        if len(parts) == 1:
            if not re.fullmatch(r"\d+", parts[0]):
                raise ValueError(f"非法的导出尺寸 '{raw}'，请输入正整数（如 960）或 高,宽（如 544,960）。")
            h = w = int(parts[0])
        elif len(parts) == 2:
            if not (re.fullmatch(r"\d+", parts[0]) and re.fullmatch(r"\d+", parts[1])):
                raise ValueError(f"非法的导出尺寸 '{raw}'，高与宽必须为正整数（如 544,960 或 544x960）。")
            h, w = int(parts[0]), int(parts[1])
        else:
            raise ValueError(f"非法的导出尺寸 '{raw}'，仅支持单个尺寸（如 960）或 高,宽 两个维度（如 544,960）。")
    else:
        pair = _extract_imgsz_pair(raw)
        if pair is None:
            raise ValueError(f"非法的导出尺寸 {raw!r}，请输入正整数或 (高, 宽) 正整数二元组。")
        h, w = pair

    if not (32 <= h <= 8192 and 32 <= w <= 8192):
        raise ValueError(f"导出尺寸 ({h}×{w}) 超出有效范围，高与宽须在 32 ~ 8192 像素之间。")
    return (h, w)


def align_imgsz_to_stride(imgsz: tuple[int, int], max_stride: int = 32) -> tuple[int, int]:
    """将 `(高, 宽)` 按检测头最大步长向上取整对齐（与 Ultralytics `check_imgsz` 规则一致）。"""
    stride = max(1, int(max_stride))
    h, w = imgsz
    aligned_h = max(stride, int(math.ceil(h / stride)) * stride)
    aligned_w = max(stride, int(math.ceil(w / stride)) * stride)
    return (aligned_h, aligned_w)


def format_imgsz_text(imgsz: tuple[int, int] | None) -> str:
    """将 `(高, 宽)` 像素元组格式化为简洁中文展示文本。"""
    if imgsz is None:
        return "未记录"
    h, w = imgsz
    return f"{h}×{w}" if h == w else f"{h}×{w} (高×宽)"


def _extract_class_names(raw: object) -> tuple[tuple[int, str], ...]:
    """将模型中的 `names`（字典或列表）提取为按类别索引升序排列的 `(id, name)` 元组。"""
    if isinstance(raw, Mapping):
        pairs: list[tuple[int, str]] = []
        for k, v in raw.items():
            try:
                idx = int(k)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            pairs.append((idx, str(v)))
        pairs.sort(key=lambda item: item[0])
        return tuple(pairs)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return tuple((idx, str(v)) for idx, v in enumerate(raw))
    return ()


def _extract_strides(raw: object) -> tuple[int, ...]:
    """提取模型检测头步长序列（像素），缺失或非法时回退标准三尺度 `(8, 16, 32)`。"""
    values: object = raw
    if hasattr(values, "tolist"):
        try:
            values = values.tolist()  # type: ignore[union-attr]
        except Exception:
            values = None
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        parsed = [iv for item in values if (iv := _extract_positive_int(item)) is not None]
        if parsed:
            return tuple(parsed)
    single = _extract_positive_int(values)
    if single is not None:
        return (single,)
    return (8, 16, 32)


def format_class_order(class_names: Sequence[tuple[int, str]], *, max_items: int = 0) -> str:
    """将有序 `(id, name)` 类别序列格式化为单行摘要，超限时按 `max_items` 截断并附总数。"""
    if not class_names:
        return "未记录"
    if max_items > 0 and len(class_names) > max_items:
        head = ", ".join(f"{idx}: {name}" for idx, name in class_names[:max_items])
        return f"{head}, ...（共 {len(class_names)} 类）"
    return ", ".join(f"{idx}: {name}" for idx, name in class_names)


def format_class_order_lines(class_names: Sequence[tuple[int, str]]) -> str:
    """将有序 `(id, name)` 类别序列格式化为逐行清单，便于弹窗展示或直接复制部署。"""
    if not class_names:
        return "（权重未记录类别名称）"
    return "\n".join(f"{idx}: {name}" for idx, name in class_names)


def _parse_metadata_literal(value: object) -> object:
    """将导出模型元数据中的字符串编码字段安全还原为 Python 基础/容器对象。"""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _probe_onnx_input_shape(
    model_path: Path,
) -> tuple[tuple[int, int] | None, int | None, int | None, bool | None]:
    """当 `.onnx` 文件未包含完整 Ultralytics 元数据时，只读探测首输入张量 `(imgsz, channels, batch, dynamic)`。"""
    try:
        import onnx

        proto = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return None, None, None, None

    if not proto.graph.input:
        return None, None, None, None
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    if len(dims) < 4:
        return None, None, None, None

    def _dim_int(dim_obj: object) -> tuple[int | None, bool]:
        param = str(getattr(dim_obj, "dim_param", "") or "").strip()
        val = int(getattr(dim_obj, "dim_value", 0) or 0)
        if param or val <= 0:
            return None, True
        return val, False

    n_val, n_dyn = _dim_int(dims[0])
    c_val, _ = _dim_int(dims[1])
    h_val, h_dyn = _dim_int(dims[2])
    w_val, w_dyn = _dim_int(dims[3])
    is_dynamic = bool(n_dyn or h_dyn or w_dyn)
    imgsz = (h_val, w_val) if (h_val is not None and w_val is not None) else None
    return imgsz, c_val, n_val, is_dynamic


def _inspect_exported_model_metadata(model_path: Path) -> WeightsMetadata:
    """只读解析已导出部署模型（`.onnx` / `.torchscript` / `.engine`）的内嵌元数据与输入维度。

    优先复用 Ultralytics `BaseBackend.read_metadata` 零权重加载读取内嵌配置；
    若 `.onnx` 缺失部分字段（如外部工具导出），再回退读取 ONNX 图首输入张量维度。
    文件为空或彻底不可解析时抛出中文 `ValueError`。
    """
    fmt = detect_model_format(model_path)
    if model_path.stat().st_size == 0:
        raise ValueError(f"导出模型文件为空: {model_path.name}")

    raw_meta: dict[str, object] = {}
    try:
        from ultralytics.nn.backends.base import BaseBackend

        loaded = BaseBackend.read_metadata(model_path)
        if isinstance(loaded, Mapping):
            raw_meta = dict(loaded)
    except Exception:
        raw_meta = {}

    parsed_args = _parse_metadata_literal(raw_meta.get("args"))
    args_map: Mapping[str, object] = parsed_args if isinstance(parsed_args, Mapping) else {}

    task = str(raw_meta.get("task") or "").strip()
    imgsz = _extract_imgsz_pair(_parse_metadata_literal(raw_meta.get("imgsz")))
    class_names = _extract_class_names(_parse_metadata_literal(raw_meta.get("names")))
    strides = _extract_strides(_parse_metadata_literal(raw_meta.get("stride")))
    channels = _extract_positive_int(_parse_metadata_literal(raw_meta.get("channels"))) or 3
    export_batch = _extract_positive_int(
        _parse_metadata_literal(raw_meta.get("batch"))
    ) or _extract_positive_int(args_map.get("batch"))

    raw_dynamic = args_map.get("dynamic") if "dynamic" in args_map else raw_meta.get("dynamic")
    parsed_dynamic = _parse_metadata_literal(raw_dynamic) if raw_dynamic is not None else None
    export_dynamic: bool | None = parsed_dynamic if isinstance(parsed_dynamic, bool) else None

    raw_end2end = _parse_metadata_literal(raw_meta.get("end2end"))
    raw_args_nms = args_map.get("nms") if "nms" in args_map else None

    export_nms_mode: str | None = None
    supports_end2end = False
    one2many_available = True
    if raw_end2end is True or raw_args_nms is False:
        export_nms_mode = "end2end"
        supports_end2end = True
        one2many_available = False
    elif raw_args_nms is True:
        export_nms_mode = "embedded"
        supports_end2end = False
        one2many_available = True
    elif raw_meta:
        export_nms_mode = "raw"
        supports_end2end = False
        one2many_available = True

    if fmt == "onnx" and (imgsz is None or export_dynamic is None):
        onnx_imgsz, onnx_ch, onnx_batch, onnx_dyn = _probe_onnx_input_shape(model_path)
        if imgsz is None and onnx_imgsz is not None:
            imgsz = onnx_imgsz
        if onnx_ch is not None and "channels" not in raw_meta:
            channels = onnx_ch
        if export_batch is None and onnx_batch is not None:
            export_batch = onnx_batch
        if export_dynamic is None and onnx_dyn is not None:
            export_dynamic = onnx_dyn

    # 解析 YoloStudio 写入的 ONNX 部署预处理契约（优先 JSON 总包，缺失时回退扁平键）
    deploy_schema: str | None = None
    preprocess_mode: str | None = None
    preprocess_padding_value: int | None = None
    preprocess_auto_rect: bool | None = None

    deploy_json_raw = raw_meta.get(ONNX_DEPLOY_META_JSON_KEY)
    deploy_obj: Mapping[str, object] = {}
    if isinstance(deploy_json_raw, str) and deploy_json_raw.strip():
        try:
            loaded_json = json.loads(deploy_json_raw)
            if isinstance(loaded_json, Mapping):
                deploy_obj = loaded_json
        except Exception:
            deploy_obj = {}

    if deploy_obj:
        schema_val = deploy_obj.get("schema")
        if isinstance(schema_val, str) and schema_val.strip():
            deploy_schema = schema_val.strip()

        pre_obj = deploy_obj.get("preprocess")
        if isinstance(pre_obj, Mapping):
            rm = str(pre_obj.get("resize_mode") or "").strip().lower()
            pa = str(pre_obj.get("padding_align") or "").strip().lower()
            if rm == "stretch":
                preprocess_mode = "stretch"
            elif rm == "letterbox":
                preprocess_mode = "letterbox_top_left" if pa == "top_left" else "letterbox_center"
            raw_pv = pre_obj.get("padding_value")
            if raw_pv is not None and not isinstance(raw_pv, bool):
                try:
                    preprocess_padding_value = normalize_padding_value(int(raw_pv))  # type: ignore[arg-type]
                except Exception:
                    preprocess_padding_value = None
            raw_ar = pre_obj.get("auto_rect")
            if isinstance(raw_ar, bool):
                preprocess_auto_rect = raw_ar

        # 部署声明记录实际导出输入和输出；原生字段继续服务于没有该声明的旧模型。
        inp = deploy_obj.get("input")
        if isinstance(inp, Mapping):
            reference = _extract_imgsz_pair([inp.get("height"), inp.get("width")])
            if reference is not None:
                imgsz = reference
            if isinstance(inp.get("dynamic"), bool):
                export_dynamic = inp["dynamic"]
        if not class_names:
            class_names = _extract_class_names(deploy_obj.get("classes"))
        post_obj = deploy_obj.get("postprocess")
        if isinstance(post_obj, Mapping):
            om = str(post_obj.get("output_mode") or "").strip().lower()
            if om in {"raw", "end2end", "embedded"}:
                export_nms_mode = om
                supports_end2end = om == "end2end"
                one2many_available = om != "end2end"

    if deploy_schema is None:
        raw_schema = raw_meta.get("deploy_schema")
        if isinstance(raw_schema, str) and raw_schema.strip():
            deploy_schema = raw_schema.strip()

    if preprocess_mode is None:
        flat_rm = str(raw_meta.get("preprocess_resize_mode") or "").strip().lower()
        flat_pa = str(raw_meta.get("preprocess_padding_align") or "").strip().lower()
        if flat_rm == "stretch":
            preprocess_mode = "stretch"
        elif flat_rm == "letterbox":
            preprocess_mode = "letterbox_top_left" if flat_pa == "top_left" else "letterbox_center"

    if preprocess_padding_value is None and preprocess_mode != "stretch":
        flat_pv = _parse_metadata_literal(raw_meta.get("preprocess_padding_value"))
        if isinstance(flat_pv, int) and not isinstance(flat_pv, bool):
            try:
                preprocess_padding_value = normalize_padding_value(flat_pv)
            except Exception:
                preprocess_padding_value = None

    if preprocess_auto_rect is None and "preprocess_auto_rect" in raw_meta:
        flat_ar = _parse_metadata_literal(raw_meta.get("preprocess_auto_rect"))
        if isinstance(flat_ar, bool):
            preprocess_auto_rect = flat_ar

    return WeightsMetadata(
        weights_path=model_path,
        task=task,
        train_imgsz=imgsz,
        class_names=class_names,
        strides=strides,
        channels=channels,
        supports_end2end=supports_end2end,
        one2many_available=one2many_available,
        epoch=None,
        model_format=fmt,
        is_exported=True,
        export_nms_mode=export_nms_mode,
        export_dynamic=export_dynamic,
        export_batch=export_batch,
        deploy_schema=deploy_schema,
        preprocess_mode=preprocess_mode,
        preprocess_padding_value=preprocess_padding_value,
        preprocess_auto_rect=preprocess_auto_rect,
    )


def inspect_weights_metadata(weights_path: Path) -> WeightsMetadata:
    """只读解析本地 `.pt` 权重或已导出部署模型（`.onnx` / `.torchscript` / `.engine`）的元数据摘要。

    调用条件与失败方式：
        - 要求 `weights_path` 为存在的本地文件，缺失时抛出 `FileNotFoundError`；
        - `.pt` 权重在 CPU 上只读反序列化，已导出模型直接解析内嵌元数据或 ONNX 图头，
          均不占用 GPU 显存且不修改源文件；
        - 若文件损坏或不是有效的模型文件，抛出中文 `ValueError`。
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {weights_path}")

    if is_exported_model_path(weights_path):
        return _inspect_exported_model_metadata(weights_path)

    ckpt: object
    try:
        from ultralytics.nn.tasks import torch_safe_load

        ckpt, _ = torch_safe_load(str(weights_path))
    except Exception:
        try:
            import torch

            ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"无法读取 .pt 权重元数据: {exc}") from exc

    model_obj: object | None = None
    train_args: Mapping[str, object] = {}
    ckpt_names: object = None
    epoch_val: int | None = None

    if isinstance(ckpt, Mapping):
        model_obj = ckpt.get("ema") or ckpt.get("model")
        raw_train_args = ckpt.get("train_args")
        if isinstance(raw_train_args, Mapping):
            train_args = raw_train_args
        elif raw_train_args is not None and hasattr(raw_train_args, "__dict__"):
            train_args = vars(raw_train_args)
        ckpt_names = ckpt.get("names")
        raw_epoch = ckpt.get("epoch")
        if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool) and raw_epoch >= 0:
            epoch_val = raw_epoch
    elif hasattr(ckpt, "names") or hasattr(ckpt, "args") or hasattr(ckpt, "model"):
        model_obj = ckpt
    else:
        raise ValueError(f"文件不是有效的 Ultralytics .pt 权重格式: {weights_path.name}")

    model_args: Mapping[str, object] = {}
    if model_obj is not None:
        raw_model_args = getattr(model_obj, "args", None)
        if isinstance(raw_model_args, Mapping):
            model_args = raw_model_args
        elif raw_model_args is not None and hasattr(raw_model_args, "__dict__"):
            model_args = vars(raw_model_args)

    train_imgsz = _extract_imgsz_pair(train_args.get("imgsz")) or _extract_imgsz_pair(
        model_args.get("imgsz")
    )
    task = str(
        train_args.get("task")
        or model_args.get("task")
        or (getattr(model_obj, "task", "") if model_obj is not None else "")
        or ""
    ).strip()

    class_names = _extract_class_names(
        getattr(model_obj, "names", None) if model_obj is not None else None
    )
    if not class_names:
        class_names = _extract_class_names(ckpt_names)

    strides = _extract_strides(
        getattr(model_obj, "stride", None) if model_obj is not None else None
    )

    channels = 3
    if model_obj is not None:
        model_yaml = getattr(model_obj, "yaml", None)
        if isinstance(model_yaml, Mapping):
            parsed_ch = _extract_positive_int(model_yaml.get("channels"))
            if parsed_ch is not None:
                channels = parsed_ch

    supports_end2end = False
    one2many_available = True
    if model_obj is not None:
        layers = getattr(model_obj, "model", None)
        head = None
        if layers is not None:
            try:
                head = layers[-1]
            except Exception:
                head = None
        if head is not None:
            supports_end2end = getattr(head, "one2one_cv2", None) is not None
            if hasattr(head, "cv2"):
                one2many_available = getattr(head, "cv2", None) is not None
        elif hasattr(model_obj, "end2end"):
            supports_end2end = bool(getattr(model_obj, "end2end", False))

    return WeightsMetadata(
        weights_path=weights_path,
        task=task,
        train_imgsz=train_imgsz,
        class_names=class_names,
        strides=strides,
        channels=channels,
        supports_end2end=supports_end2end,
        one2many_available=one2many_available,
        epoch=epoch_val,
        model_format="pt",
        is_exported=False,
    )


def format_weights_metadata_log(meta: WeightsMetadata) -> str:
    """将 `.pt` 权重或已导出模型元数据格式化为面向 GUI/CLI 日志区的多行中文摘要。"""
    task_text = meta.task or "未记录"
    imgsz_text = format_imgsz_text(meta.train_imgsz)

    if meta.is_exported:
        fmt_upper = meta.model_format.upper()
        dyn_desc = (
            "动态尺寸/批次"
            if meta.export_dynamic is True
            else ("静态固定尺寸" if meta.export_dynamic is False else "动态性未知")
        )
        batch_suffix = f", Batch={meta.export_batch}" if meta.export_batch is not None else ""
        mode_map = {
            "end2end": "已固化 YOLO26 one2one 端到端无 NMS（直出 [B,N,6]）",
            "embedded": "已固化图内嵌入 NMS（直出 [B,N,6]）",
            "raw": "已固化原始 one2many 输出（需外部 NMS）",
        }
        head_text = mode_map.get(meta.export_nms_mode, "未记录（由导出模型内部决定）")
        lines = [
            f"=== 导出模型元数据 ({meta.weights_path.name}, {fmt_upper}) ===",
            f"  任务类型: {task_text}",
            f"  导出输入尺寸: {imgsz_text}（{dyn_desc}{batch_suffix}）",
            f"  固化输出头: {head_text}",
        ]
        if meta.preprocess_mode:
            pre_desc = format_preprocess_summary(
                meta.preprocess_mode,
                meta.preprocess_padding_value,
                auto_rect=meta.preprocess_auto_rect,
            )
            schema_suffix = f" [{meta.deploy_schema}]" if meta.deploy_schema else ""
            lines.append(f"  预处理契约: {pre_desc}, RGB, 0~1{schema_suffix}")
        lines.append(f"  类别顺序 ({meta.num_classes} 类): {format_class_order(meta.class_names)}")
        return "\n".join(lines)

    if meta.supports_end2end and meta.one2many_available:
        head_text = "双分支检测头（支持 raw 外部NMS / end2end YOLO26无NMS / embedded 图内NMS）"
    elif meta.supports_end2end:
        head_text = "仅保留 one2one 端到端无 NMS 分支"
    else:
        head_text = "标准 one2many 检测头（支持 raw 外部NMS / embedded 图内NMS）"

    lines = [
        f"=== 权重元数据 ({meta.weights_path.name}) ===",
        f"  任务类型: {task_text}",
        f"  训练输入尺寸: {imgsz_text}",
        f"  检测头能力: {head_text}",
        f"  类别顺序 ({meta.num_classes} 类): {format_class_order(meta.class_names)}",
    ]
    return "\n".join(lines)


def _ultralytics_supports_quantize_arg() -> bool:
    """探测当前已安装的 Ultralytics 配置是否支持新版 `quantize` 精度参数。"""
    try:
        from ultralytics.utils import DEFAULT_CFG

        return hasattr(DEFAULT_CFG, "quantize")
    except Exception:
        return False


def resolve_export_imgsz(
    config: ExportConfig, metadata: WeightsMetadata | None = None
) -> tuple[int, int] | None:
    """解析本次导出应使用的 `(高, 宽)` 像素尺寸；未显式指定时回退为权重训练尺寸。"""
    if config.imgsz is not None:
        return normalize_export_imgsz(config.imgsz)
    if metadata is not None and metadata.train_imgsz is not None:
        return metadata.train_imgsz
    return None


def describe_export_io(
    config: ExportConfig, metadata: WeightsMetadata | None = None
) -> tuple[str, list[str]]:
    """根据导出配置与可选权重元数据，生成预期输入/输出张量说明及风险提示列表。

    返回值为 `(io_summary_text, warnings_list)`，供桌面界面实时预览与 CLI/导出日志复用。
    """
    fmt = normalize_export_format(config.export_format)
    warnings: list[str] = []

    raw_imgsz = resolve_export_imgsz(config, metadata)
    strides = metadata.strides if metadata is not None else (8, 16, 32)
    max_stride = metadata.max_stride if metadata is not None else 32
    channels = metadata.channels if metadata is not None else 3
    nc = metadata.num_classes if metadata is not None else 0

    if raw_imgsz is not None:
        aligned = align_imgsz_to_stride(raw_imgsz, max_stride=max_stride)
        if aligned != raw_imgsz:
            warnings.append(
                f"导出尺寸 {raw_imgsz[0]}×{raw_imgsz[1]} 不是步长 {max_stride} 的倍数，"
                f"Ultralytics 将自动向上对齐为 {aligned[0]}×{aligned[1]}。"
            )
        h, w = aligned
    else:
        h, w = (640, 640)

    if (
        metadata is not None
        and metadata.train_imgsz is not None
        and raw_imgsz is not None
        and raw_imgsz != metadata.train_imgsz
    ):
        warnings.append(
            f"当前导出尺寸 {format_imgsz_text(raw_imgsz)} 与权重训练尺寸 "
            f"{format_imgsz_text(metadata.train_imgsz)} 不一致，可能影响目标检测精度。"
        )

    export_device = "cpu" if fmt == "onnx" else str(config.device).strip() or "0"
    effective_fp16 = bool(config.half) and export_device.lower() != "cpu"
    dtype_label = "FP16" if effective_fp16 else "FP32"

    if fmt == "engine" and export_device.lower() == "cpu":
        warnings.append("TensorRT (.engine) 仅支持 CUDA GPU 导出，请在运行设备中启用 GPU 加速。")
    if config.dynamic and fmt == "engine" and int(config.batch) == 1:
        warnings.append("TensorRT 开启动态输入时，建议将 Batch 调大为推理期预期最大批大小（如 8 或 16）。")
    if config.dynamic and fmt not in DYNAMIC_EXPORT_FORMATS:
        warnings.append(f"导出格式 {fmt} 不支持 dynamic 动态输入，该选项将被忽略。")

    batch_dim = "batch" if config.dynamic and fmt in DYNAMIC_EXPORT_FORMATS else str(int(config.batch))
    if config.dynamic and fmt in DYNAMIC_EXPORT_FORMATS:
        in_desc = f"输入 images: ({batch_dim}, {channels}, height, width) [动态, 参考 {h}×{w}, {dtype_label}]"
    else:
        in_desc = f"输入 images: ({batch_dim}, {channels}, {h}, {w}) [{dtype_label}, NCHW]"

    nms_val = parse_export_nms_mode(config.nms)
    effective_end2end = False
    if nms_val is False:
        if metadata is not None and not metadata.supports_end2end:
            warnings.append(
                "当前权重检测头未检测到 YOLO26 one2one 端到端分支，选择 end2end 时将自动回退为原始 one2many 输出。"
            )
        else:
            effective_end2end = True
    elif metadata is not None and not metadata.one2many_available and metadata.supports_end2end:
        effective_end2end = True
        if nms_val is None or nms_val is True:
            warnings.append(
                "当前权重仅保留 one2one 端到端分支（one2many 已被融合移除），导出将直接输出端到端 [B,N,6] 结果。"
            )

    if effective_end2end:
        out_desc = (
            f"输出 output0: ({batch_dim}, {int(config.max_det)}, 6) "
            "[x1,y1,x2,y2,score,cls_id，YOLO26 无NMS直出]"
        )
    elif nms_val is True:
        conf_txt = f"{config.conf:.2f}" if config.conf is not None else "0.25"
        iou_txt = f"{config.iou:.2f}" if config.iou is not None else "0.50"
        out_desc = (
            f"输出 output0: ({batch_dim}, {int(config.max_det)}, 6) "
            f"[图内嵌入NMS, conf={conf_txt}, iou={iou_txt}]"
        )
    else:
        if config.dynamic and fmt in DYNAMIC_EXPORT_FORMATS:
            anchors_text = "anchors"
        else:
            anchors = sum(int(h // s) * int(w // s) for s in strides if s > 0)
            anchors_text = str(anchors)
        if nc > 0:
            ch_text = f"{4 + nc}"
            cls_hint = f"4+{nc}类"
        else:
            ch_text = "4+nc"
            cls_hint = "cx,cy,w,h+各类分数"
        out_desc = f"输出 output0: ({batch_dim}, {ch_text}, {anchors_text}) [{cls_hint}，需外部NMS]"

    mode_key, resize_mode, _pad_align, pad_val = resolve_preprocess_spec(
        config.preprocess_mode, config.padding_value
    )
    is_non_default_pre = mode_key != DEFAULT_EXPORT_PREPROCESS_MODE or (
        pad_val is not None and pad_val != DEFAULT_EXPORT_PADDING_VALUE
    )
    if fmt == "onnx":
        effective_auto_rect = (
            bool(config.preprocess_auto_rect)
            if config.preprocess_auto_rect is not None
            else (bool(config.dynamic) and resize_mode == "letterbox")
        )
        if resize_mode == "stretch":
            effective_auto_rect = False
        pre_desc = format_preprocess_summary(mode_key, pad_val, auto_rect=effective_auto_rect)
        if is_non_default_pre:
            warnings.append(
                "当前预处理策略偏离 YOLO 默认训练设置（居中 LetterBox + 灰边 114），"
                "将写入 ONNX 元数据供推理端按此执行，请注意评估检测精度变化。"
            )
        return f"{in_desc}  |  {out_desc}  |  预处理元数据: {pre_desc}", warnings

    if is_non_default_pre or config.preprocess_auto_rect is not None:
        warnings.append(f"自定义预处理元数据写入仅对 ONNX 格式生效，当前格式 '{fmt}' 已忽略。")

    return f"{in_desc}  |  {out_desc}", warnings


def build_export_kwargs(
    config: ExportConfig,
    *,
    metadata: WeightsMetadata | None = None,
    use_quantize_arg: bool | None = None,
) -> tuple[dict[str, object], list[str]]:
    """校验 `ExportConfig` 并生成可直接传给 `model.export(**kwargs)` 的参数字典与提示列表。

    格式兼容性保护：
        - `onnx` 固定使用 `device="cpu"` 以规避部分环境下的图追踪设备冲突；
        - `opset` 与 `simplify` 仅在 ONNX 系列后端格式中传入，避免 `torchscript` 触发 Ultralytics 参数校验异常；
        - 启用 FP16 且非 CPU 时，在支持 `quantize` 的 Ultralytics 8.4+ 环境中传 `quantize=16` 以消除 `half` 废弃告警；
        - 未显式传入 `imgsz` 时优先使用权重记录的 `train_imgsz`，两者均缺失时交由 Ultralytics 默认推导；
        - 同步校验 `preprocess_mode` 与 `padding_value` 合法性（不透传给 Ultralytics `model.export`，而在 ONNX 导出后写入元数据）。
    """
    fmt = normalize_export_format(config.export_format)
    warnings: list[str] = []

    batch = int(config.batch)
    if batch < 1:
        raise ValueError(f"导出批大小 batch 必须 >= 1，当前值: {batch}")

    max_det = int(config.max_det)
    if max_det < 1:
        raise ValueError(f"最大检测框数 max_det 必须 >= 1，当前值: {max_det}")

    nms_val = parse_export_nms_mode(config.nms)
    mode_key, _resize_mode, _pad_align, pad_val = resolve_preprocess_spec(
        config.preprocess_mode, config.padding_value
    )
    if mode_key != "stretch":
        normalize_padding_value(config.padding_value)

    opset: int | None = None
    if config.opset is not None:
        opset = int(config.opset)
        if not (7 <= opset <= 22):
            raise ValueError(f"ONNX opset 必须在 7 ~ 22 之间，当前值: {opset}")
        if nms_val is True and opset < 11:
            raise ValueError(f"图内嵌入 NMS (nms=True) 要求 ONNX opset >= 11，当前值: {opset}")

    if config.conf is not None:
        conf_val = float(config.conf)
        if not (0.0 < conf_val < 1.0):
            raise ValueError(f"嵌入 NMS 置信度阈值 conf 必须在 (0, 1) 之间，当前值: {conf_val}")
    if config.iou is not None:
        iou_val = float(config.iou)
        if not (0.0 < iou_val <= 1.0):
            raise ValueError(f"嵌入 NMS IoU 阈值 iou 必须在 (0, 1] 之间，当前值: {iou_val}")
    if config.workspace is not None:
        ws_val = float(config.workspace)
        if not math.isfinite(ws_val) or ws_val <= 0:
            raise ValueError(f"TensorRT workspace (GiB) 必须为正数，当前值: {config.workspace}")

    resolved_imgsz = resolve_export_imgsz(config, metadata)
    export_device = "cpu" if fmt == "onnx" else (str(config.device).strip() or "0")

    export_kwargs: dict[str, object] = {
        "format": fmt,
        "batch": batch,
        "device": export_device,
    }

    if resolved_imgsz is not None:
        h, w = resolved_imgsz
        export_kwargs["imgsz"] = h if h == w else [h, w]

    if fmt in DYNAMIC_EXPORT_FORMATS:
        export_kwargs["dynamic"] = bool(config.dynamic)
    elif config.dynamic:
        warnings.append(f"导出格式 '{fmt}' 不支持 dynamic=True，已忽略该选项。")

    if nms_val is not None:
        export_kwargs["nms"] = nms_val
        export_kwargs["max_det"] = max_det
        if nms_val is True:
            if config.conf is not None:
                export_kwargs["conf"] = float(config.conf)
            if config.iou is not None:
                export_kwargs["iou"] = float(config.iou)

    if fmt in ONNX_BACKED_EXPORT_FORMATS:
        export_kwargs["simplify"] = bool(config.simplify)
        if opset is not None:
            export_kwargs["opset"] = opset
    else:
        if opset is not None:
            warnings.append(f"导出格式 '{fmt}' 不支持 opset 参数，已忽略 opset={opset}。")
        if config.simplify is False:
            warnings.append(f"导出格式 '{fmt}' 不使用 simplify 参数，已忽略。")

    if fmt == "engine" and config.workspace is not None:
        export_kwargs["workspace"] = float(config.workspace)
    elif fmt != "engine" and config.workspace is not None:
        warnings.append(f"workspace 仅对 TensorRT (.engine) 生效，当前格式 '{fmt}' 已忽略。")

    if config.half:
        if export_device.lower() == "cpu":
            if fmt == "onnx":
                warnings.append("ONNX 固定在 CPU 上导出，不支持 FP16，半精度选项已忽略。")
            else:
                warnings.append("当前为 CPU 设备模式，不支持 FP16 半精度导出，已忽略。")
        else:
            use_q = _ultralytics_supports_quantize_arg() if use_quantize_arg is None else use_quantize_arg
            if use_q:
                export_kwargs["quantize"] = 16
            else:
                export_kwargs["half"] = True

    if fmt != "onnx" and (
        mode_key != DEFAULT_EXPORT_PREPROCESS_MODE
        or (pad_val is not None and pad_val != DEFAULT_EXPORT_PADDING_VALUE)
        or config.preprocess_auto_rect is not None
    ):
        warnings.append(f"自定义预处理元数据写入仅对 ONNX 格式生效，当前格式 '{fmt}' 已忽略。")

    if config.extra_args:
        export_kwargs.update(config.extra_args)

    return export_kwargs, warnings


def build_onnx_deploy_metadata(
    config: ExportConfig,
    *,
    metadata: WeightsMetadata | None = None,
    input_name: str = "images",
    output_name: str = "output0",
    input_dtype: str = "float32",
    actual_imgsz: tuple[int, int] | None = None,
) -> tuple[dict[str, object], dict[str, str]]:
    """根据 `ExportConfig` 与权重元数据生成写入 ONNX `metadata_props` 的部署契约字典与扁平键值表。

    协议版本：
        `yolostudio.deploy.v1`（详见 `docs/004_ONNX部署元数据契约说明.md`）。
        返回 `(deploy_payload, flat_props)`：
        - `deploy_payload` 为标准结构化字典（序列化后写入 `yolostudio_deploy_meta`）；
        - `flat_props` 为不与 Ultralytics / `BaseBackend` 冲突的 `map<str, str>` 键值表，
          可直接写入 ONNX `metadata_props` 供下游推理端读取。
    """
    max_stride = metadata.max_stride if metadata is not None else 32
    channels = metadata.channels if metadata is not None else 3
    task = (metadata.task if metadata is not None and metadata.task else "") or "detect"
    classes = [name for _, name in metadata.class_names] if metadata is not None else []

    if actual_imgsz is not None:
        h, w = actual_imgsz
    else:
        raw_imgsz = resolve_export_imgsz(config, metadata)
        h, w = align_imgsz_to_stride(raw_imgsz, max_stride=max_stride) if raw_imgsz is not None else (640, 640)

    mode_key, resize_mode, padding_align, pad_val = resolve_preprocess_spec(
        config.preprocess_mode, config.padding_value
    )
    is_dynamic = bool(config.dynamic)
    if resize_mode == "stretch":
        auto_rect = False
    elif config.preprocess_auto_rect is not None:
        auto_rect = bool(config.preprocess_auto_rect) and is_dynamic
    else:
        auto_rect = is_dynamic

    nms_val = parse_export_nms_mode(config.nms)
    effective_end2end = False
    if nms_val is False:
        effective_end2end = metadata is None or metadata.supports_end2end
    elif metadata is not None and not metadata.one2many_available and metadata.supports_end2end:
        effective_end2end = True

    if effective_end2end:
        output_mode = "end2end"
        nms_required = False
        box_format = "xyxy"
        max_det_out: int | None = int(config.max_det)
        embedded_conf: float | None = None
        embedded_iou: float | None = None
    elif nms_val is True:
        output_mode = "embedded"
        nms_required = False
        box_format = "xyxy"
        max_det_out = int(config.max_det)
        embedded_conf = float(config.conf) if config.conf is not None else 0.25
        embedded_iou = float(config.iou) if config.iou is not None else 0.50
    else:
        output_mode = "raw"
        nms_required = True
        box_format = "xywh"
        max_det_out = None
        embedded_conf = None
        embedded_iou = None

    deploy_payload: dict[str, object] = {
        "schema": ONNX_DEPLOY_SCHEMA_VERSION,
        "task": task,
        "input": {
            "name": input_name,
            "layout": "NCHW",
            "dtype": input_dtype,
            "batch": int(config.batch),
            "channels": int(channels),
            "height": int(h),
            "width": int(w),
            "dynamic": is_dynamic,
            "stride": int(max_stride),
            "color_format": "RGB",
            "value_range": [0.0, 1.0],
            "mean": [0.0, 0.0, 0.0],
            "std": [255.0, 255.0, 255.0],
        },
        "preprocess": {
            "mode": mode_key,
            "resize_mode": resize_mode,
            "padding_align": padding_align,
            "padding_value": pad_val,
            "auto_rect": auto_rect,
            "interpolation": "linear",
            "scale_up": True,
        },
        "postprocess": {
            "output_name": output_name,
            "output_mode": output_mode,
            "nms_required": nms_required,
            "box_format": box_format,
            "box_coord_space": "input_pixels",
            "max_det": max_det_out,
            "embedded_conf": embedded_conf,
            "embedded_iou": embedded_iou,
        },
        "classes": classes,
    }

    flat_props: dict[str, str] = {
        "deploy_schema": ONNX_DEPLOY_SCHEMA_VERSION,
        "preprocess_mode": mode_key,
        "preprocess_resize_mode": resize_mode,
        "preprocess_padding_align": padding_align,
        "preprocess_padding_value": "" if pad_val is None else str(pad_val),
        "preprocess_auto_rect": "true" if auto_rect else "false",
        "preprocess_color_format": "RGB",
        "preprocess_input_layout": "NCHW",
        "preprocess_normalize": "0_1",
        "preprocess_interpolation": "linear",
        "postprocess_output_mode": output_mode,
        "postprocess_nms_required": "true" if nms_required else "false",
        "postprocess_box_format": box_format,
        "postprocess_box_coord_space": "input_pixels",
        ONNX_DEPLOY_META_JSON_KEY: json.dumps(deploy_payload, ensure_ascii=False, separators=(",", ":")),
    }
    return deploy_payload, flat_props


def write_onnx_deploy_metadata(
    onnx_path: Path,
    config: ExportConfig,
    *,
    metadata: WeightsMetadata | None = None,
) -> dict[str, object]:
    """将预处理与后处理部署契约写入已导出的 `.onnx` 文件 `metadata_props` 中（保留 Ultralytics 原有字段）。

    调用条件与失败方式：
        - 要求 `onnx_path` 为有效的本地 ONNX 文件且已安装 `onnx` 库；
        - 自动从 ONNX 真实输入/输出节点提取张量名称、数据类型与静态分辨率，避免元数据与计算图不一致；
        - 文件缺失抛出 `FileNotFoundError`，ONNX 文件损坏或无法解析/保存时抛出中文 `ValueError`。
    """
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 模型文件不存在: {onnx_path}")

    try:
        import onnx
    except Exception as exc:
        raise ValueError(f"未安装 onnx 依赖，无法写入 ONNX 部署元数据: {exc}") from exc

    try:
        proto = onnx.load(str(onnx_path))
    except Exception as exc:
        raise ValueError(f"无法读取 ONNX 模型文件以写入元数据 ({onnx_path.name}): {exc}") from exc

    input_name = "images"
    output_name = "output0"
    input_dtype = "float32"
    actual_imgsz: tuple[int, int] | None = None

    elem_type_map = {1: "float32", 10: "float16", 2: "uint8", 11: "float64"}
    if proto.graph.input:
        in0 = proto.graph.input[0]
        if in0.name:
            input_name = in0.name
        tt = in0.type.tensor_type
        input_dtype = elem_type_map.get(int(tt.elem_type), "float32")
        dims = tt.shape.dim
        if len(dims) >= 4:
            h_dim = int(dims[2].dim_value or 0)
            w_dim = int(dims[3].dim_value or 0)
            if h_dim > 0 and w_dim > 0 and not dims[2].dim_param and not dims[3].dim_param:
                actual_imgsz = (h_dim, w_dim)
    if proto.graph.output and proto.graph.output[0].name:
        output_name = proto.graph.output[0].name

    deploy_payload, flat_props = build_onnx_deploy_metadata(
        config,
        metadata=metadata,
        input_name=input_name,
        output_name=output_name,
        input_dtype=input_dtype,
        actual_imgsz=actual_imgsz,
    )

    existing_props = {entry.key: entry for entry in proto.metadata_props}
    for key, val in flat_props.items():
        if key in existing_props:
            existing_props[key].value = val
        else:
            item = proto.metadata_props.add()
            item.key = key
            item.value = val

    # 在同目录完成序列化后再替换，写入失败时保留导出器生成的原文件以便诊断。
    with NamedTemporaryFile(dir=onnx_path.parent, suffix=".onnx", delete=False) as handle:
        staged = Path(handle.name)
    try:
        onnx.save(proto, str(staged))
        staged.replace(onnx_path)
    except Exception as exc:
        raise ValueError(f"写回 ONNX 部署元数据失败 ({onnx_path.name}): {exc}") from exc
    finally:
        staged.unlink(missing_ok=True)

    return deploy_payload


def format_submitted_export_config(
    export_kwargs: Mapping[str, object],
    *,
    io_summary: str = "",
    warnings: Sequence[str] = (),
) -> str:
    """将提交给 Ultralytics 的导出参数、预期输入/输出与告警格式化为可读日志文本。"""
    json_lines = json.dumps(dict(export_kwargs), ensure_ascii=False, indent=2, default=str).splitlines()
    if json_lines:
        json_lines[0] = f'"export_arguments": {json_lines[0]}'
    lines = [
        "=== 模型导出配置 (提交至 Ultralytics) ===",
        *json_lines,
    ]
    if io_summary:
        lines.append(f"[预期张量] {io_summary}")
    for warn in warnings:
        lines.append(f"[提示] {warn}")
    return "\n".join(lines)


def export_model_with_config(
    config: ExportConfig,
    log_callback: Callable[[str], None] | None = None,
) -> Path:
    """按 `ExportConfig` 执行 YOLO26 模型导出，并在导出前输出权重元数据与完整参数日志。

    行为说明：
        - 导出前先只读探测权重元数据（训练尺寸、类别顺序、检测头能力）并输出至 `log_callback`；
        - 若 `config.imgsz` 为 `None` 且权重记录了训练尺寸，则自动沿用训练尺寸；
        - 导出为 `.onnx` 时，自动向 ONNX `metadata_props` 追加写入预处理与后处理部署契约（保留 Ultralytics 原有字段）；
        - 产物生成在源权重同目录并返回其路径；权重缺失时抛出 `FileNotFoundError`，参数非法时抛出 `ValueError`。
    """
    if not config.weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {config.weights_path}")
    if is_exported_model_path(config.weights_path):
        raise ValueError(
            f"模型导出仅支持 PyTorch (.pt) 源权重，当前文件为已导出模型: {config.weights_path.name}"
        )

    from ultralytics import YOLO

    meta: WeightsMetadata | None = None
    try:
        meta = inspect_weights_metadata(config.weights_path)
        _emit_multiline_log(log_callback, format_weights_metadata_log(meta))
    except Exception as exc:
        if log_callback:
            log_callback(f"[提示] 预读取权重元数据失败（将继续执行导出）: {exc}")

    export_kwargs, build_warnings = build_export_kwargs(config, metadata=meta)
    io_summary, io_warnings = describe_export_io(config, metadata=meta)

    all_warnings: list[str] = []
    for item in [*build_warnings, *io_warnings]:
        if item not in all_warnings:
            all_warnings.append(item)

    if config.imgsz is None and meta is not None and meta.train_imgsz is not None:
        all_warnings.insert(
            0, f"未显式指定导出尺寸，已自动沿用权重训练尺寸: {format_imgsz_text(meta.train_imgsz)}。"
        )

    _emit_multiline_log(
        log_callback,
        format_submitted_export_config(
            export_kwargs, io_summary=io_summary, warnings=all_warnings
        ),
    )

    ensure_torch_cpu_threads()
    model = YOLO(str(config.weights_path))
    with capture_thread_output(log_callback):
        exported = model.export(**export_kwargs)
    exported_path = Path(str(exported))

    if exported_path.suffix.lower() == ".onnx":
        # 附加参数可以覆盖表单选项，元数据必须描述真正提交给导出器的设置。
        fields = ("imgsz", "batch", "dynamic", "nms", "max_det", "conf", "iou")
        effective = replace(config, **{key: export_kwargs[key] for key in fields if key in export_kwargs})
        payload = write_onnx_deploy_metadata(exported_path, effective, metadata=meta)
        pre_map = payload.get("preprocess", {})
        post_map = payload.get("postprocess", {})
        if log_callback and isinstance(pre_map, Mapping) and isinstance(post_map, Mapping):
            log_callback(
                f"[部署元数据] 已写入 ONNX ({ONNX_DEPLOY_SCHEMA_VERSION}): "
                f"preprocess={pre_map.get('resize_mode')}/{pre_map.get('padding_align')}"
                f"(pad={pre_map.get('padding_value')}, auto_rect={pre_map.get('auto_rect')}), "
                f"postprocess={post_map.get('output_mode')}/{post_map.get('box_format')}"
            )

    return exported_path


def export_model(
    weights_path: Path,
    export_format: str = "onnx",
    imgsz: int | tuple[int, int] | None = None,
    half: bool = False,
    device: str = "0",
    *,
    batch: int = 1,
    dynamic: bool = False,
    nms: bool | None = None,
    max_det: int = 300,
    conf: float | None = None,
    iou: float | None = None,
    opset: int | None = None,
    simplify: bool = True,
    workspace: float | None = None,
    preprocess_mode: str = DEFAULT_EXPORT_PREPROCESS_MODE,
    padding_value: int = DEFAULT_EXPORT_PADDING_VALUE,
    preprocess_auto_rect: bool | None = None,
    extra_args: Mapping[str, object] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> Path:
    """将已训练的 YOLO26 `.pt` 权重导出为指定部署格式（`onnx`、`torchscript`、`engine` 等）。

    当 `imgsz` 为 `None` 时自动沿用权重中记录的训练尺寸；当 `export_format == "onnx"` 时固定使用
    `"cpu"` 设备导出以规避图追踪设备冲突（此时忽略 FP16），并自动将 `preprocess_mode`、`padding_value`
    等预处理与后处理契约写入 ONNX `metadata_props`。产物生成在源权重同目录并返回其路径，
    权重缺失时抛出 `FileNotFoundError`，参数非法时抛出 `ValueError`。
    """
    cfg = ExportConfig(
        weights_path=weights_path,
        export_format=export_format,
        imgsz=imgsz,
        batch=batch,
        dynamic=dynamic,
        nms=nms,
        max_det=max_det,
        conf=conf,
        iou=iou,
        opset=opset,
        simplify=simplify,
        half=half,
        device=device,
        workspace=workspace,
        preprocess_mode=preprocess_mode,
        padding_value=padding_value,
        preprocess_auto_rect=preprocess_auto_rect,
        extra_args=dict(extra_args or {}),
    )
    return export_model_with_config(cfg, log_callback=log_callback)
