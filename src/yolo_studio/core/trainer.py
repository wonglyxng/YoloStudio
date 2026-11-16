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

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import torch

YOLO26_DETECT_MODELS = [
    ("yolo26n.pt", "YOLO26n (Nano - 速度最快，显存占用最低)"),
    ("yolo26s.pt", "YOLO26s (Small - 速度与精度均衡，RTX 3060 推荐)"),
    ("yolo26m.pt", "YOLO26m (Medium - 更高精度)"),
    ("yolo26l.pt", "YOLO26l (Large - 高精度，建议适当减小 batch)"),
    ("yolo26x.pt", "YOLO26x (XLarge - 最高精度，显存需求较高)"),
]


@dataclass
class DeviceInfo:
    """可用计算硬件与显存摘要；`vram_gb` 单位为 GiB。"""

    cuda_available: bool
    device_count: int
    device_name: str
    vram_gb: float
    recommended_device: str


@dataclass
class TrainConfig:
    """YOLO26 训练任务超参数与路径配置。

    关键字段说明：
        `batch`：批大小，设为 `-1` 时由 Ultralytics 自动探测最大可用批次。
        `lr0`：初始学习率；`patience`：早停容忍轮数；`workers`：DataLoader 工作进程数。
        `project_dir` / `run_name`：训练产物保存根目录与子实验名（默认开启 `exist_ok=True`）。
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
    extra_args: dict[str, object] = field(default_factory=dict)


@dataclass
class EpochMetrics:
    """单个训练轮次结束时提取的关键指标快照（`epoch` 为从 1 开始计数）。"""

    epoch: int
    total_epochs: int
    box_loss: float = 0.0
    cls_loss: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0


@dataclass
class TrainResult:
    """YOLO26 训练结束或安全停训后的结果摘要。"""

    save_dir: Path
    best_weights: Path | None
    last_weights: Path | None
    metrics: dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False


class _StreamForwarder(io.TextIOBase):
    """按行缓冲并转发标准输出/错误流至日志回调的适配流。

    状态与线程约束：
        内部 `_buffer` 保存尚未遇到换行符的半行内容（将 `\\r` 统一转为 `\\n`），
        调用方应在退出重定向块后调用 `flush()` 以刷出尾部残留行；
        本类非线程安全，仅应在单次 `redirect_stdout` / `redirect_stderr` 作用域内使用。
    """

    def __init__(self, log_callback: Callable[[str], None] | None) -> None:
        super().__init__()
        self._callback = log_callback
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s.replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            stripped = line.strip()
            if stripped and self._callback:
                self._callback(stripped)
        return len(s)

    def flush(self) -> None:
        if self._buffer.strip() and self._callback:
            self._callback(self._buffer.strip())
        self._buffer = ""


def detect_compute_device() -> DeviceInfo:
    """检测本机 CUDA GPU 可用性并读取 0 号卡显存信息，无可用 GPU 时回退为 CPU。"""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        return DeviceInfo(
            cuda_available=True,
            device_count=torch.cuda.device_count(),
            device_name=props.name,
            vram_gb=vram_gb,
            recommended_device="0",
        )
    return DeviceInfo(
        cuda_available=False,
        device_count=0,
        device_name="CPU (未检测到可用 CUDA GPU)",
        vram_gb=0.0,
        recommended_device="cpu",
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        # 保留 type: ignore 指令供静态类型检查器识别运行时动态转换
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def train_yolo26(
    config: TrainConfig,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    cancel_event: Event | None = None,
) -> TrainResult:
    """使用 Ultralytics 启动 YOLO26 目标检测模型训练。

    调用条件与副作用：
        要求 `config.data_yaml` 存在（否则抛出 `FileNotFoundError`）；
        会在 `config.project_dir / config.run_name` 下写出训练日志、指标曲线及 `weights/best.pt`、`last.pt`。

    回调与安全停训约束：
        通过 `on_train_epoch_end` 与 `on_fit_epoch_end` 回调派发 `EpochMetrics` 并检查 `cancel_event`；
        若收到取消信号，将设置 `trainer.stop = True`，等待当前轮次完成并保存断点权重后再退出。
        训练期间会重定向当前进程的 `stdout` 与 `stderr` 到 `log_callback`。
    """
    from ultralytics import YOLO

    if not config.data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {config.data_yaml}")

    if log_callback:
        log_callback(f"加载模型: {config.model_name}")
        log_callback(f"数据集配置: {config.data_yaml}")

    model = YOLO(config.model_name)

    def _check_cancel(trainer: object) -> None:
        if cancel_event and cancel_event.is_set():
            setattr(trainer, "stop", True)
            if log_callback:
                log_callback("收到停止请求，将在当前轮次结束后安全停止训练...")

    def _on_fit_epoch_end(trainer: object) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        epochs = int(getattr(trainer, "epochs", config.epochs))
        metrics_dict = getattr(trainer, "metrics", {}) or {}
        tloss = getattr(trainer, "tloss", None)

        box_loss = 0.0
        cls_loss = 0.0
        if tloss is not None:
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
        )
        if epoch_callback:
            epoch_callback(em)
        _check_cancel(trainer)

    model.add_callback("on_train_epoch_end", _check_cancel)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    train_kwargs: dict[str, object] = {
        "data": str(config.data_yaml),
        "epochs": int(config.epochs),
        "imgsz": int(config.imgsz),
        "batch": int(config.batch),
        "device": config.device,
        "lr0": float(config.lr0),
        "patience": int(config.patience),
        "workers": int(config.workers),
        "amp": bool(config.amp),
        "cache": bool(config.cache),
        "project": str(config.project_dir.resolve()),
        "name": config.run_name,
        "exist_ok": True,
        "resume": bool(config.resume),
    }
    train_kwargs.update(config.extra_args)

    forwarder = _StreamForwarder(log_callback)
    with redirect_stdout(forwarder), redirect_stderr(forwarder):
        results = model.train(**train_kwargs)
    forwarder.flush()

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
        stopped_early=bool(cancel_event and cancel_event.is_set()),
    )


def validate_model(
    weights_path: Path,
    data_yaml: Path,
    imgsz: int = 640,
    device: str = "0",
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, float]:
    """在指定验证数据集上评估已训练的 YOLO26 权重并返回指标字典。

    `weights_path` 或 `data_yaml` 不存在时抛出 `FileNotFoundError`。
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 不存在: {data_yaml}")

    model = YOLO(str(weights_path))
    forwarder = _StreamForwarder(log_callback)
    with redirect_stdout(forwarder), redirect_stderr(forwarder):
        metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    forwarder.flush()

    summary: dict[str, float] = {}
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for k, v in results_dict.items():
            summary[str(k)] = _safe_float(v)
    return summary


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
    图片输入通过 `cv2.imencode` + `Path.write_bytes` 写出为 `*_pred.jpg` 以支持中文路径。
    权重或输入路径缺失时抛出 `FileNotFoundError`。
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    if not source_path.exists():
        raise FileNotFoundError(f"推理输入不存在: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))

    forwarder = _StreamForwarder(log_callback)
    saved_outputs: list[Path] = []
    with redirect_stdout(forwarder), redirect_stderr(forwarder):
        is_video = source_path.is_file() and source_path.suffix.lower() in {
            ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".ts", ".m4v", ".webm"
        }
        if is_video:
            results = model.predict(
                source=str(source_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                save=True,
                project=str(output_dir.resolve()),
                name="predict",
                exist_ok=True,
            )
            if results:
                save_dir = Path(results[0].save_dir)
                saved_outputs = sorted(save_dir.glob("*"))
        else:
            results = model.predict(
                source=str(source_path),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                save=False,
            )
            for idx, r in enumerate(results, start=1):
                plotted = r.plot()
                src_name = Path(r.path).stem if getattr(r, "path", None) else f"pred_{idx}"
                out_file = output_dir / f"{src_name}_pred.jpg"
                ok, encoded = cv2.imencode(".jpg", plotted, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    out_file.write_bytes(encoded.tobytes())
                    saved_outputs.append(out_file)
    forwarder.flush()
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
    产物生成在源权重同目录并返回其路径，权重缺失时抛出 `FileNotFoundError`。
    """
    from ultralytics import YOLO

    if not weights_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    model = YOLO(str(weights_path))
    forwarder = _StreamForwarder(log_callback)
    with redirect_stdout(forwarder), redirect_stderr(forwarder):
        exported = model.export(
            format=export_format,
            imgsz=imgsz,
            half=half,
            device=device if export_format != "onnx" else "cpu",
        )
    forwarder.flush()
    return Path(str(exported))
