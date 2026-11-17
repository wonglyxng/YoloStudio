"""视频抽帧引擎单元测试。

职责：
    验证 `probe_video` 与 `extract_frames` 在按秒间隔抽帧、目标帧数均匀抽帧、
    最长边缩放、中文路径输出、静态帧相似去重、同名视频输出消歧、
    大步长 seek 读取以及取消行为上的表现。

运行环境与协作边界：
    通过 `pytest` 在临时目录（`tmp_path`）中运行，依赖 `opencv-python` 与 `numpy`
    生成合成 MP4 测试视频，直接覆盖 `yolo_studio.core.frame_extractor`。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from yolo_studio.core.frame_extractor import (
    ExtractionConfig,
    ExtractionMode,
    extract_frames,
    probe_video,
)


def _create_synthetic_video(path: Path, num_frames: int = 30, fps: float = 10.0) -> Path:
    """生成包含水平移动矩形目标的合成 MP4 测试视频。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (160, 120))
    assert writer.isOpened()
    for i in range(num_frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        # 逐帧移动绿色矩形位置，使相邻帧在视觉上存在明显差异
        x = (i * 4) % 120
        cv2.rectangle(frame, (x, 20), (x + 30, 80), (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


def _create_static_video(path: Path, num_frames: int = 20, fps: float = 10.0) -> Path:
    """生成画面完全静止的合成 MP4 测试视频，用于验证去重逻辑。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (160, 120))
    assert writer.isOpened()
    static_frame = np.full((120, 160, 3), 120, dtype=np.uint8)
    for _ in range(num_frames):
        writer.write(static_frame)
    writer.release()
    return path


def test_probe_and_extract_by_interval(tmp_path: Path) -> None:
    """验证视频元信息探测及按秒间隔抽帧（含中文输出目录路径）。"""
    video = _create_synthetic_video(tmp_path / "demo.mp4", num_frames=30, fps=10.0)
    info = probe_video(video)
    assert info.width == 160
    assert info.height == 120
    assert info.frame_count == 30

    out_dir = tmp_path / "帧输出"
    cfg = ExtractionConfig(
        output_dir=out_dir,
        mode=ExtractionMode.INTERVAL_SECONDS,
        interval_seconds=1.0,
        dedup_threshold=0.0,
        image_format="jpg",
    )
    res = extract_frames([video], cfg)
    assert not res.errors
    assert res.videos_processed == 1
    assert res.frames_saved == 3
    assert all(p.is_file() for p in res.saved_files)


def test_extract_target_count_and_resize(tmp_path: Path) -> None:
    """验证按目标帧数均匀抽帧及 `max_dimension` 最长边等比缩放。"""
    video = _create_synthetic_video(tmp_path / "sample.mp4", num_frames=40, fps=20.0)
    out_dir = tmp_path / "frames_png"
    cfg = ExtractionConfig(
        output_dir=out_dir,
        mode=ExtractionMode.TARGET_COUNT,
        target_count=5,
        max_dimension=80,
        image_format="png",
    )
    res = extract_frames([video], cfg)
    assert res.frames_saved == 5
    img = cv2.imdecode(np.fromfile(str(res.saved_files[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    assert max(img.shape[:2]) == 80


def test_dedup_skips_static_frames(tmp_path: Path) -> None:
    """验证相似帧去重阈值能够跳过完全静止视频中的重复抽取帧。"""
    video = _create_static_video(tmp_path / "static.mp4", num_frames=20, fps=10.0)
    out_dir = tmp_path / "dedup_out"
    cfg = ExtractionConfig(
        output_dir=out_dir,
        mode=ExtractionMode.EVERY_N_FRAMES,
        every_n_frames=5,
        dedup_threshold=1.0,
    )
    res = extract_frames([video], cfg)
    assert res.frames_saved == 1
    assert res.frames_skipped_similar == 3


def test_same_name_videos_do_not_overwrite_frames(tmp_path: Path) -> None:
    """验证不同目录下同名视频抽帧时文件名自动消歧，帧图片不会互相覆盖。"""
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()
    v1 = _create_synthetic_video(tmp_path / "cam1" / "rec.mp4", 20, 10.0)
    v2 = _create_synthetic_video(tmp_path / "cam2" / "rec.mp4", 20, 10.0)
    cfg = ExtractionConfig(
        output_dir=tmp_path / "out",
        mode=ExtractionMode.EVERY_N_FRAMES,
        every_n_frames=10,
    )
    res = extract_frames([v1, v2], cfg)
    assert res.frames_saved == 4
    assert len({p.name for p in res.saved_files}) == 4
    assert {p.name.rsplit("_", 1)[0] for p in res.saved_files} == {"cam1_rec", "cam2_rec"}


def test_large_step_seek_and_cancel(tmp_path: Path) -> None:
    """验证超过 seek 阈值的大步长抽帧结果正确，并验证取消后不计入已处理视频。"""
    from threading import Event

    video = _create_synthetic_video(tmp_path / "long.mp4", num_frames=700, fps=25.0)
    cfg = ExtractionConfig(output_dir=tmp_path / "seek", mode=ExtractionMode.EVERY_N_FRAMES, every_n_frames=320)
    res = extract_frames([video], cfg)
    assert res.frames_saved == 3  # 第 0、320、640 帧

    cancel = Event()
    cancel.set()
    res2 = extract_frames([video], ExtractionConfig(output_dir=tmp_path / "cancel"), cancel_event=cancel)
    assert res2.cancelled and res2.videos_processed == 0
