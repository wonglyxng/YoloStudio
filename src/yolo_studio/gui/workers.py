"""PyQt6 `QThread` 后台工作线程封装模块。

职责：
    将视频元信息探测、视频抽帧、标注扫描与转换、YOLO26 训练/验证/推理/导出及 MLLM 纠错清洗等耗时任务
    下沉至后台 `QThread` 执行，并通过 Qt `pyqtSignal` 安全回传进度与结果，避免阻塞主界面事件循环。

运行环境与协作边界：
    依赖 `PyQt6` 与 `yolo_studio.core` 各引擎；由各标签页在主线程实例化并启动。
    所有工作线程捕获未处理异常后通过 `failed` 信号传递字符串消息；
    支持取消的工作线程仅通过 `threading.Event` 请求安全停机，不强行终止线程。
"""

from __future__ import annotations

from pathlib import Path
from threading import Event

from PyQt6.QtCore import QThread, pyqtSignal as Signal

from yolo_studio.core.dataset_converter import (
    DatasetBuildConfig,
    DatasetBuildResult,
    DatasetScanSummary,
    build_yolo_dataset,
    scan_dataset,
)
from yolo_studio.core.frame_extractor import (
    ExtractionConfig,
    ExtractionResult,
    collect_videos,
    extract_frames,
    probe_video,
)
from yolo_studio.core.mllm_reviewer import (
    MllmReviewConfig,
    MllmReviewResult,
    run_mllm_review,
    test_mllm_connection,
)
from yolo_studio.core.trainer import (
    DeviceInfo,
    TrainConfig,
    TrainResult,
    detect_compute_device,
    export_model,
    predict_source,
    train_yolo26,
    validate_model,
)


class DeviceProbeWorker(QThread):
    """在后台线程执行 `detect_compute_device`（含首次 `import torch` 与 CUDA 探测），避免阻塞首屏显示。

    信号契约：
        `finished_info(DeviceInfo)`：探测完成后发射真实硬件摘要；发生异常时发射 CPU 回退摘要。
    """

    finished_info = Signal(object)

    def run(self) -> None:
        """线程入口：查询底层计算设备并通过信号回传 `DeviceInfo`。"""
        try:
            info = detect_compute_device()
        except Exception as exc:
            info = DeviceInfo(
                cuda_available=False,
                device_count=0,
                device_name=f"CPU (硬件探测异常已回退: {exc})",
                vram_gb=0.0,
                recommended_device="cpu",
                probing=False,
            )
        self.finished_info.emit(info)


class VideoProbeWorker(QThread):
    """在后台线程扫描目录并逐个读取视频头元信息，避免添加大量视频时卡住主界面。

    信号契约：
        `item_probed(Path, str)`：每探测完一个新视频发射一次，携带视频路径与列表展示文本。
        `finished_probing(int)`：全部探测完成或响应取消后发射本次新增的视频数量。
        `failed(str)`：发生意外未捕获异常时发射错误消息。
    """

    item_probed = Signal(object, str)
    finished_probing = Signal(int)
    failed = Signal(str)

    def __init__(self, raw_paths: list[Path], existing_resolved: set[Path]) -> None:
        super().__init__()
        self.raw_paths = list(raw_paths)
        self.existing_resolved = set(existing_resolved)
        self.cancel_event = Event()

    def cancel(self) -> None:
        """设置协同取消标志，停止后续视频探测。"""
        self.cancel_event.set()

    def run(self) -> None:
        """线程入口：展开视频路径、跳过已存在项并逐个调用 `probe_video` 回传结果。"""
        try:
            added = 0
            seen = set(self.existing_resolved)
            for vp in collect_videos(self.raw_paths):
                if self.cancel_event.is_set():
                    break
                resolved = vp.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    info = probe_video(vp)
                    text = (
                        f"{vp.name}  [{info.width}x{info.height}, {info.fps:.1f}fps, "
                        f"{info.duration_sec:.1f}s, {info.frame_count}帧]"
                    )
                except Exception:
                    text = f"{vp.name}  [读取元信息失败]"
                if self.cancel_event.is_set():
                    break
                added += 1
                self.item_probed.emit(vp, text)
            self.finished_probing.emit(added)
        except Exception as exc:
            self.failed.emit(str(exc))


class FrameExtractWorker(QThread):
    """在后台线程执行视频抽帧任务，并通过 `cancel_event` 支持安全中断。"""

    progress = Signal(object)
    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(self, video_paths: list[Path], config: ExtractionConfig) -> None:
        super().__init__()
        self.video_paths = video_paths
        self.config = config
        self.cancel_event = Event()

    def cancel(self) -> None:
        """设置协同取消标志，促使 `extract_frames` 在下一个检查点安全退出。"""
        self.cancel_event.set()

    def run(self) -> None:
        """线程入口：执行抽帧并通过 `progress` / `finished_result` / `failed` 通知主线程。"""
        try:
            res: ExtractionResult = extract_frames(
                self.video_paths,
                self.config,
                progress_callback=lambda p: self.progress.emit(p),
                cancel_event=self.cancel_event,
            )
            self.finished_result.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetScanWorker(QThread):
    """在后台线程只读扫描图片与 X-AnyLabeling 标注文件。"""

    finished_summary = Signal(object)
    failed = Signal(str)

    def __init__(self, config: DatasetBuildConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        """线程入口：执行 `scan_dataset` 并通过信号回传统计摘要或异常信息。"""
        try:
            summary: DatasetScanSummary = scan_dataset(self.config)
            self.finished_summary.emit(summary)
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetBuildWorker(QThread):
    """在后台线程转换标注、拆分集合并写出 `data.yaml`。"""

    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(self, config: DatasetBuildConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        """线程入口：执行 `build_yolo_dataset` 并通过信号回传结果或异常信息。"""
        try:
            res: DatasetBuildResult = build_yolo_dataset(self.config)
            self.finished_result.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainWorker(QThread):
    """在后台线程运行 YOLO26 模型训练，并通过 `cancel_event` 支持当前轮结束后停训。"""

    epoch_updated = Signal(object)
    log_line = Signal(str)
    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.cancel_event = Event()

    def cancel(self) -> None:
        """请求在当前 Epoch 结束后安全停止训练并保存权重。"""
        self.cancel_event.set()

    def run(self) -> None:
        """线程入口：运行 `train_yolo26` 并持续派发指标、日志及最终结果。"""
        try:
            res: TrainResult = train_yolo26(
                self.config,
                epoch_callback=lambda m: self.epoch_updated.emit(m),
                log_callback=lambda line: self.log_line.emit(line),
                cancel_event=self.cancel_event,
            )
            self.finished_result.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


class ValidateWorker(QThread):
    """在后台线程运行 YOLO26 验证集评估。"""

    log_line = Signal(str)
    finished_metrics = Signal(object)
    failed = Signal(str)

    def __init__(self, weights_path: Path, data_yaml: Path, imgsz: int, device: str) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.data_yaml = data_yaml
        self.imgsz = imgsz
        self.device = device

    def run(self) -> None:
        """线程入口：执行 `validate_model` 并派发日志及验证指标字典。"""
        try:
            metrics = validate_model(
                self.weights_path,
                self.data_yaml,
                imgsz=self.imgsz,
                device=self.device,
                log_callback=lambda line: self.log_line.emit(line),
            )
            self.finished_metrics.emit(metrics)
        except Exception as exc:
            self.failed.emit(str(exc))


class PredictWorker(QThread):
    """在后台线程运行 YOLO26 推理预览并写出渲染结果。"""

    log_line = Signal(str)
    finished_outputs = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        weights_path: Path,
        source_path: Path,
        output_dir: Path,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
    ) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.source_path = source_path
        self.output_dir = output_dir
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def run(self) -> None:
        """线程入口：执行 `predict_source` 并派发生成的输出文件路径列表。"""
        try:
            outputs = predict_source(
                self.weights_path,
                self.source_path,
                self.output_dir,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                log_callback=lambda line: self.log_line.emit(line),
            )
            self.finished_outputs.emit(outputs)
        except Exception as exc:
            self.failed.emit(str(exc))


class ExportWorker(QThread):
    """在后台线程执行 YOLO26 权重导出（ONNX / TorchScript / TensorRT 等）。"""

    log_line = Signal(str)
    finished_path = Signal(str)
    failed = Signal(str)

    def __init__(
        self, weights_path: Path, export_format: str, imgsz: int, half: bool, device: str
    ) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.export_format = export_format
        self.imgsz = imgsz
        self.half = half
        self.device = device

    def run(self) -> None:
        """线程入口：执行 `export_model` 并派发导出文件路径字符串。"""
        try:
            exported = export_model(
                self.weights_path,
                export_format=self.export_format,
                imgsz=self.imgsz,
                half=self.half,
                device=self.device,
                log_callback=lambda line: self.log_line.emit(line),
            )
            self.finished_path.emit(str(exported))
        except Exception as exc:
            self.failed.emit(str(exc))


class MllmReviewWorker(QThread):
    """在后台线程运行基准推理或 `.json` 加载及 MLLM 并发纠错任务。"""

    progress = Signal(int, int, object)
    log_line = Signal(str)
    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(self, config: MllmReviewConfig) -> None:
        super().__init__()
        self.config = config
        self.cancel_event = Event()

    def cancel(self) -> None:
        """设置协同取消标志，停止提交新的审核图片并尝试取消待执行任务。"""
        self.cancel_event.set()

    def run(self) -> None:
        """线程入口：执行 `run_mllm_review` 并派发单图进度、日志及最终审核汇总。"""
        try:
            res: MllmReviewResult = run_mllm_review(
                self.config,
                progress_callback=lambda cur, total, rec: self.progress.emit(cur, total, rec),
                log_callback=lambda line: self.log_line.emit(line),
                cancel_event=self.cancel_event,
            )
            self.finished_result.emit(res)
        except Exception as exc:
            self.failed.emit(str(exc))


class MllmConnectionTestWorker(QThread):
    """在后台线程执行 MLLM 接口轻量连通性探针，避免网络请求阻塞界面主线程。"""

    finished_test = Signal(bool, str)  # 信号参数：(是否连通, 诊断摘要)

    def __init__(self, config: MllmReviewConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        """线程入口：调用 `test_mllm_connection` 并将诊断结果派发至主线程。"""
        ok, msg = test_mllm_connection(self.config)
        self.finished_test.emit(ok, msg)
