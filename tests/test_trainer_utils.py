"""线程路由输出捕获、训练辅助函数与 CLI 入口单元测试。

职责：
    验证 `core.log_capture` 的 `\\r` 覆盖语义、线程隔离与 `ultralytics` 日志转发，
    `core.trainer` 中断点续训解析、推理输出文件名消歧与硬件探测缓存，
    以及 `yolo_studio.cli` 新增抽帧参数（起止时间、前缀、子目录、质量）的解析与端到端执行。

运行环境与协作边界：
    通过 `pytest` 运行；`core.trainer` 仅在调用 `detect_compute_device` 时按需加载 `torch`，
    测试不启动真实训练、不访问外部网络，也不加载任何 YOLO 权重。
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import pytest

from yolo_studio.core.log_capture import _LineAssembler, capture_thread_output
from yolo_studio.core.trainer import TrainConfig, _resolve_resume_checkpoint, _unique_pred_path


def test_line_assembler_treats_carriage_return_as_overwrite() -> None:
    """验证进度条式的 `\\r` 刷新只保留最终一版，普通换行逐行输出。"""
    lines: list[str] = []
    asm = _LineAssembler(lines.append)
    asm.feed("epoch 1/10  10%\r")
    asm.feed("epoch 1/10  50%\repoch 1/10 100%\n")
    asm.feed("done\r\nnext line")
    asm.flush()
    assert lines == ["epoch 1/10 100%", "done", "next line"]


def test_capture_is_thread_local_and_forwards_ultralytics_logger() -> None:
    """验证捕获仅作用于当前线程，其他线程输出不串入，且 `ultralytics` 日志被转发。"""
    logger = logging.getLogger("ultralytics")
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    captured: list[str] = []
    other_thread_lines: list[str] = []

    def _other() -> None:
        with capture_thread_output(other_thread_lines.append):
            print("来自其他线程")

    try:
        with capture_thread_output(captured.append):
            print("主线程输出")
            worker = threading.Thread(target=_other)
            worker.start()
            worker.join()
            logger.info("ultralytics 日志")
    finally:
        logger.removeHandler(handler)

    assert captured == ["主线程输出", "ultralytics 日志"]
    assert other_thread_lines == ["来自其他线程"]


def test_resume_checkpoint_resolution(tmp_path: Path) -> None:
    """验证续训优先使用自定义 last.pt，官方权重名回退到实验目录，缺失时抛出异常。"""
    data_yaml = tmp_path / "data.yaml"
    cfg = TrainConfig(data_yaml=data_yaml, model_name="yolo26s.pt", project_dir=tmp_path, run_name="exp", resume=True)
    with pytest.raises(FileNotFoundError):
        _resolve_resume_checkpoint(cfg)

    default_ckpt = tmp_path / "exp" / "weights" / "last.pt"
    default_ckpt.parent.mkdir(parents=True)
    default_ckpt.write_bytes(b"ckpt")
    assert _resolve_resume_checkpoint(cfg) == default_ckpt

    custom = tmp_path / "other_last.pt"
    custom.write_bytes(b"ckpt")
    cfg.model_name = str(custom)
    assert _resolve_resume_checkpoint(cfg) == custom


def test_unique_pred_path_disambiguates_same_stem(tmp_path: Path) -> None:
    """验证同名不同扩展名的图片推理结果不会互相覆盖。"""
    used: set[str] = set()
    first = _unique_pred_path(tmp_path, Path("a.jpg"), used)
    second = _unique_pred_path(tmp_path, Path("a.png"), used)
    assert first.name == "a_pred.jpg"
    assert second.name == "a_png_pred.jpg"


def test_detect_compute_device_caches_result() -> None:
    """验证 `detect_compute_device` 默认复用进程级缓存快照，传入 `refresh=True` 时可重新探测。"""
    from yolo_studio.core.trainer import detect_compute_device

    first = detect_compute_device()
    second = detect_compute_device()
    assert first is second
    refreshed = detect_compute_device(refresh=True)
    assert refreshed.recommended_device in {"0", "cpu"}


def test_cli_extract_supports_time_slice_prefix_and_subfolder(tmp_path: Path) -> None:
    """验证 `yolo_studio.cli` 的 `extract` 子命令支持起止时间裁剪、前缀、JPG 质量与子目录分建。"""
    import cv2
    import numpy as np

    from yolo_studio.cli import main as cli_main

    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (120, 90))
    assert writer.isOpened()
    for i in range(30):
        frame = np.full((90, 120, 3), (i * 8) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    out_dir = tmp_path / "cli_out"
    code = cli_main(
        [
            "extract",
            str(video_path),
            "-o",
            str(out_dir),
            "--mode",
            "interval_seconds",
            "--interval",
            "0.5",
            "--start",
            "0.5",
            "--end",
            "2.0",
            "--dedup",
            "0",
            "--quality",
            "90",
            "--prefix",
            "camA_",
            "--subfolder",
        ]
    )
    assert code == 0
    sub_files = sorted((out_dir / "clip").glob("camA_clip_*.jpg"))
    assert len(sub_files) == 3  # 0.5s(帧5)、1.0s(帧10)、1.5s(帧15)
