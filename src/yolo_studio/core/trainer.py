"""Ultralytics YOLO26 模型训练、验证、推理预览与导出引擎。

职责：
    封装本地硬件探测及 YOLO26 模型的训练、验证、图片/视频推理与部署格式导出，
    并将训练引擎的标准输出和 Epoch 指标实时桥接至调用方回调。

运行环境：
    Python >= 3.11，依赖 `torch`、`ultralytics` 与 `opencv-python`；
    使用预训练名称（如 `yolo26s.pt`）且本地缺失权重时，Ultralytics 会触发模型下载。

协作对象与职责边界：
    为 `cli._cmd_train`、`gui.workers`（`TrainWorker`、`ValidateWorker`、`PredictWorker`、
    `ExportWorker`）、`gui.tab_train.TrainTab`、`gui.tab_eval.EvalTab` 与
    `gui.tab_review.MllmReviewTab` 提供底层能力。仅负责引擎调度，不管理线程或 UI 控件。
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

import cv2

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
        桌面入口为兼容内嵌 X-AnyLabeling 设置了 `OMP_NUM_THREADS=1` 等环境变量，
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


def validate_model(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int = 640,
    device: str = "0",
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """在指定验证数据集上评估已训练的 YOLO26 权重并返回指标字典。

    `weights_path` 或 `data_yaml` 不存在时抛出 `FileNotFoundError`；
    当前线程的输出与 `ultralytics` 日志转发到 `log_callback`。
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

    ensure_torch_cpu_threads()
    model = YOLO(str(weights_path))
    with capture_thread_output(log_callback):
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)

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


def predict_source(
    weights_path: Path,
    source_path: Path,
    output_dir: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
    log_callback: Callable[[str], None] | None = None,
) -> list[Path]:
    """对单张图片、图片目录或视频运行推理并将带框结果写入 `output_dir`。

    视频输入交由 Ultralytics 写出至 `output_dir / "predict"`；
    图片输入通过 `cv2.imencode` + `Path.write_bytes` 写出为 `*_pred.jpg` 以支持中文路径，
    同名图片自动消歧不互相覆盖。两种输入均以 `stream=True` 逐帧迭代，
    内存占用与视频长度或图片数量无关。权重或输入路径缺失时抛出 `FileNotFoundError`。
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"推理输入不存在: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_torch_cpu_threads()
    model = YOLO(str(weights_path))

    saved_outputs: list[Path] = []
    with capture_thread_output(log_callback):
        is_video = source_path.is_file() and source_path.suffix.lower() in VIDEO_EXTENSIONS
        if is_video:
            save_dir: Path | None = None
            for r in model.predict(
                source=str(source_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                save=True,
                project=str(output_dir.resolve()),
                name="predict",
                exist_ok=True,
                stream=True,
            ):
                if save_dir is None and getattr(r, "save_dir", None):
                    save_dir = Path(r.save_dir)
            if save_dir is not None:
                saved_outputs = sorted(save_dir.glob("*"))
        else:
            used_names: set[str] = set()
            for idx, r in enumerate(
                model.predict(
                    source=str(source_path),
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    save=False,
                    stream=True,
                ),
                start=1,
            ):
                plotted = r.plot()
                src = Path(r.path) if getattr(r, "path", None) else Path(f"pred_{idx}.jpg")
                out_file = _unique_pred_path(output_dir, src, used_names)
                ok, encoded = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    out_file.write_bytes(encoded.tobytes())
                    saved_outputs.append(out_file)
    return saved_outputs


def export_model(
    weights_path: Path,
    export_format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    device: str = "0",
    log_callback: Callable[[str], None] | None = None,
) -> Path:
    """将已训练的 YOLO26 `.pt` 权重导出为指定部署格式（`onnx`、`torchscript`、`engine` 等）。

    当 `export_format == "onnx"` 时强制使用 `"cpu"` 设备导出，避免部分环境下 ONNX 图追踪设备冲突；
    CPU 导出不支持 FP16，此时会通过 `log_callback` 提示 `half` 被忽略。
    产物生成在源权重同目录并返回其路径，权重缺失时抛出 `FileNotFoundError`。
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    export_device = device if export_format != "onnx" else "cpu"
    if half and export_device == "cpu" and log_callback:
        log_callback("提示：ONNX 在 CPU 上导出，不支持 FP16，半精度选项将被忽略。")

    ensure_torch_cpu_threads()
    model = YOLO(str(weights_path))
    with capture_thread_output(log_callback):
        exported = model.export(
            format=export_format,
            imgsz=imgsz,
            half=half and export_device != "cpu",
            device=export_device,
        )
    return Path(str(exported))
