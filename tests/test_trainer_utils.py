"""线程路由输出捕获与训练辅助函数单元测试。

职责：
    验证 `core.log_capture` 的 `\\r` 覆盖语义、线程隔离与 `ultralytics` 日志转发，
    以及 `core.trainer` 中断点续训解析与推理输出文件名消歧等纯函数行为。

运行环境与协作边界：
    通过 `pytest` 运行，依赖 `torch`（`core.trainer` 模块导入时加载），不启动真实训练、
    不访问网络，也不加载任何 YOLO 权重。
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
