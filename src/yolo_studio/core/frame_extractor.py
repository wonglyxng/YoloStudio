"""录制视频批量抽帧引擎。

职责：
    扫描视频文件与目录、读取视频元信息、按指定采样策略抽取图像帧，
    并执行起止时间裁剪、相似帧去重与分辨率缩放后落盘保存。

运行环境：
    Python >= 3.11，依赖 `opencv-python` 与 `numpy`；通过字节编码写入以兼容中文路径。

协作对象与职责边界：
    为 `cli._cmd_extract`、`gui.workers.FrameExtractWorker` 与 `gui.tab_extract.ExtractTab`
    提供底层抽帧能力；仅负责视频读取与图片保存，不涉及标注解析、模型推理或 UI 状态管理。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".ts", ".m4v", ".webm"}


class ExtractionMode(str, Enum):
    """抽帧采样策略枚举。"""

    INTERVAL_SECONDS = "interval_seconds"
    EVERY_N_FRAMES = "every_n_frames"
    TARGET_COUNT = "target_count"


@dataclass
class VideoInfo:
    """单个视频文件的元数据摘要。

    单位与回退规则：
        `fps` 单位为帧/秒（视频头未提供有效正值时回退为 `30.0`）；
        `width`/`height` 为像素尺寸；`duration_sec` 单位为秒。
    """

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    duration_sec: float


@dataclass
class ExtractionConfig:
    """单次批量抽帧任务的运行配置。

    关键字段说明：
        `interval_seconds`：按秒抽帧间隔（秒，运行时最小夹紧为 `0.01`）。
        `every_n_frames` / `target_count`：帧步长或每视频目标抽取帧数（均至少为 `1`）。
        `start_sec` / `end_sec`：截取起止时间（秒）；`end_sec` 为 `None` 或 `<= 0` 表示抽到结尾。
        `dedup_threshold`：相似帧灰度缩略图平均绝对像素差阈值（`[0, 255]`），`<= 0` 关闭去重。
        `max_dimension`：输出最长边像素上限，`<= 0` 保持原尺寸。
        `image_format` / `jpeg_quality`：输出格式（`jpg` 或 `png`）及 JPG 压缩质量（`1..100`）。
        `create_video_subfolder`：是否按视频文件名 `stem` 在 `output_dir` 下分建子目录。
    """

    output_dir: Path
    mode: ExtractionMode = ExtractionMode.INTERVAL_SECONDS
    interval_seconds: float = 1.0
    every_n_frames: int = 30
    target_count: int = 100
    start_sec: float = 0.0
    end_sec: float | None = None
    dedup_threshold: float = 0.0
    max_dimension: int = 0
    image_format: str = "jpg"
    jpeg_quality: int = 95
    prefix: str = ""
    create_video_subfolder: bool = False


@dataclass
class ExtractionProgress:
    """抽帧过程中向调用方派发的进度载荷。

    `video_index` 从 1 开始计数；`last_saved_path` 仅在当前回调伴随新帧落盘时非空，
    纯步进刷新时为 `None`。
    """

    video_index: int
    total_videos: int
    video_name: str
    current_frame: int
    total_frames: int
    saved_total: int
    skipped_similar: int
    last_saved_path: Path | None = None


@dataclass
class ExtractionResult:
    """单次抽帧运行完成或取消后的汇总结果。

    `errors` 收集单视频打开失败、时间段无效或无有效输入等问题；
    `cancelled` 为 `True` 表示中途响应了 `cancel_event`。
    """

    output_dir: Path
    videos_processed: int
    frames_saved: int
    frames_skipped_similar: int
    saved_files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False


def collect_videos(inputs: list[Path]) -> list[Path]:
    """将文件或目录路径展开为有序且不重复的视频文件路径列表。

    只读扫描：忽略不存在的路径；目录按路径字典序递归扫描，
    并依据 `Path.resolve()` 去重以过滤重复传入的同一视频。
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                    resolved = child.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        found.append(child)
    return found


def probe_video(video_path: Path) -> VideoInfo:
    """使用 OpenCV 读取单个视频的基础元信息。

    无法打开文件时抛出 `ValueError`；无论成功或异常均会在 `finally` 中释放 `VideoCapture`。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = frame_count / fps if frame_count > 0 else 0.0
        return VideoInfo(
            path=video_path,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_sec=duration_sec,
        )
    finally:
        cap.release()


def _resize_if_needed(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    if max_dimension <= 0:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _write_image_unicode_safe(
    output_path: Path, frame: np.ndarray, image_format: str, jpeg_quality: int
) -> None:
    """使用 `cv2.imencode` 编码后经 `Path.write_bytes` 落盘，规避 OpenCV 中文路径写入失败问题。"""
    ext = ".png" if image_format.lower() == "png" else ".jpg"
    params: list[int] = []
    if ext == ".jpg":
        quality = max(1, min(100, int(jpeg_quality)))
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, encoded = cv2.imencode(ext, frame, params)
    if not success:
        raise RuntimeError(f"图像编码失败: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded.tobytes())


def _compute_frame_diff(prev_gray_small: np.ndarray | None, frame: np.ndarray) -> tuple[float, np.ndarray]:
    """在 160x90 灰度缩略图上计算归一化平均绝对像素差（取值范围 [0, 255]）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    if prev_gray_small is None:
        return 255.0, small
    diff = float(np.mean(cv2.absdiff(prev_gray_small, small)))
    return diff, small


def _compute_step_frames(info: VideoInfo, config: ExtractionConfig, start_frame: int, end_frame: int) -> float:
    span = max(1, end_frame - start_frame)
    if config.mode == ExtractionMode.INTERVAL_SECONDS:
        interval = max(0.01, float(config.interval_seconds))
        return max(1.0, info.fps * interval)
    if config.mode == ExtractionMode.EVERY_N_FRAMES:
        return float(max(1, int(config.every_n_frames)))
    if config.mode == ExtractionMode.TARGET_COUNT:
        target = max(1, int(config.target_count))
        if target >= span:
            return 1.0
        return span / float(target)
    return 1.0


# 距下一个采样点超过该帧数时才使用 `CAP_PROP_POS_FRAMES` 跳转；更近时用 `grab()` 顺序跳帧。
# 原因：H.264/H.265 等长 GOP 编码每次 seek 都要从前一关键帧重新解码，小步长下频繁 seek
# 反而比顺序读取慢，且部分后端 seek 不精确；`grab()` 只解码不做 retrieve/色彩转换，开销更低。
SEEK_THRESHOLD_FRAMES = 300


def _plan_video_name_keys(videos: list[Path]) -> dict[Path, str]:
    """为每个视频规划输出命名键（用作帧文件名前缀与子目录名），保证不同视频之间不重名。

    文件 `stem`（不区分大小写）唯一时直接使用；多个视频同名（如 `cam1/rec.mp4` 与 `cam2/rec.mp4`）
    时改用 `上级目录名_stem`，仍冲突再追加序号，防止帧文件互相覆盖。
    """
    stem_counts: dict[str, int] = {}
    for v in videos:
        stem_counts[v.stem.lower()] = stem_counts.get(v.stem.lower(), 0) + 1
    used: set[str] = set()
    keys: dict[Path, str] = {}
    for v in videos:
        base = v.stem if stem_counts[v.stem.lower()] == 1 else f"{v.parent.name}_{v.stem}"
        key = base
        idx = 1
        while key.lower() in used:
            key = f"{base}_{idx}"
            idx += 1
        used.add(key.lower())
        keys[v] = key
    return keys


def extract_frames(
    video_paths: list[Path],
    config: ExtractionConfig,
    progress_callback: Callable[[ExtractionProgress], None] | None = None,
    cancel_event: Event | None = None,
) -> ExtractionResult:
    """按配置从多个视频中批量抽取帧图片并写入输出目录。

    副作用与资源管理：
        自动创建 `config.output_dir` 及子目录并写出图片文件，存在同名文件时直接覆盖；
        多个视频同名时由 `_plan_video_name_keys` 消歧，帧文件不会跨视频互相覆盖。
        每个视频的 `cv2.VideoCapture` 在内部 `finally` 块释放，防止句柄泄漏。

    读取策略：
        距下一个采样点不超过 `SEEK_THRESHOLD_FRAMES` 帧时用 `grab()` 顺序跳过，否则直接 seek。

    回调、取消与失败行为：
        保存新帧时立即触发 `progress_callback`，跳过的帧每 30 帧触发一次进度更新；
        外部通过 `cancel_event` 请求取消后将在下个检查点停止并返回已保存结果
        （`result.cancelled = True`，已落盘文件不回滚，被中断的视频不计入 `videos_processed`）；
        未找到任何有效视频或单视频打开失败/时间段非法时将错误信息记入 `result.errors`，
        单视频失败不会中断其余视频处理。
    """
    videos = collect_videos(video_paths)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    result = ExtractionResult(
        output_dir=config.output_dir,
        videos_processed=0,
        frames_saved=0,
        frames_skipped_similar=0,
    )
    if not videos:
        result.errors.append("未找到有效的视频文件。")
        return result

    ext = ".png" if config.image_format.lower() == "png" else ".jpg"
    total_videos = len(videos)
    name_keys = _plan_video_name_keys(videos)

    def _cancelled() -> bool:
        return bool(cancel_event and cancel_event.is_set())

    for vid_idx, video_path in enumerate(videos, start=1):
        if _cancelled():
            result.cancelled = True
            break

        try:
            info = probe_video(video_path)
        except Exception as exc:
            result.errors.append(str(exc))
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            result.errors.append(f"无法打开视频: {video_path}")
            continue

        try:
            total_frames = info.frame_count if info.frame_count > 0 else 0
            start_frame = max(0, int(round(max(0.0, config.start_sec) * info.fps)))
            if config.end_sec is not None and config.end_sec > 0:
                end_frame = int(round(config.end_sec * info.fps))
                if total_frames > 0:
                    end_frame = min(total_frames, end_frame)
            else:
                end_frame = total_frames if total_frames > 0 else 10**9

            if end_frame <= start_frame and total_frames > 0:
                result.errors.append(f"视频时间范围无效: {video_path.name}")
                continue

            step = _compute_step_frames(info, config, start_frame, end_frame)
            name_key = name_keys[video_path]
            target_dir = (
                config.output_dir / name_key if config.create_video_subfolder else config.output_dir
            )
            name_prefix = f"{config.prefix}{name_key}" if config.prefix else name_key

            def _emit_progress(frame_idx: int, last_saved: Path | None) -> None:
                if progress_callback is None:
                    return
                progress_callback(
                    ExtractionProgress(
                        video_index=vid_idx,
                        total_videos=total_videos,
                        video_name=video_path.name,
                        current_frame=frame_idx + 1,
                        total_frames=total_frames if total_frames > 0 else frame_idx + 1,
                        saved_total=result.frames_saved,
                        skipped_similar=result.frames_skipped_similar,
                        last_saved_path=last_saved,
                    )
                )

            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # frame_idx 始终表示下一次 grab/read 将得到的帧序号
            frame_idx = start_frame
            next_sample_at = float(start_frame)
            saved_for_video = 0
            prev_small: np.ndarray | None = None
            interrupted = False

            while frame_idx < end_frame:
                if _cancelled():
                    interrupted = True
                    break

                target = max(frame_idx, int(math.ceil(next_sample_at - 1e-6)))
                if target >= end_frame:
                    break

                if target - frame_idx > SEEK_THRESHOLD_FRAMES:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    frame_idx = target
                else:
                    stream_ended = False
                    while frame_idx < target:
                        if _cancelled():
                            interrupted = True
                            break
                        if not cap.grab():
                            stream_ended = True
                            break
                        frame_idx += 1
                        if frame_idx % 30 == 0:
                            _emit_progress(frame_idx, None)
                    if interrupted or stream_ended:
                        break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                while next_sample_at <= frame_idx + 1e-6:
                    next_sample_at += step

                is_duplicate = False
                if config.dedup_threshold > 0:
                    diff, current_small = _compute_frame_diff(prev_small, frame)
                    if prev_small is not None and diff < config.dedup_threshold:
                        is_duplicate = True
                        result.frames_skipped_similar += 1
                    else:
                        prev_small = current_small

                if not is_duplicate:
                    processed = _resize_if_needed(frame, config.max_dimension)
                    saved_for_video += 1
                    out_path = target_dir / f"{name_prefix}_{saved_for_video:06d}{ext}"
                    _write_image_unicode_safe(
                        out_path, processed, config.image_format, config.jpeg_quality
                    )
                    result.frames_saved += 1
                    result.saved_files.append(out_path)
                    _emit_progress(frame_idx, out_path)
                else:
                    _emit_progress(frame_idx, None)

                frame_idx += 1

            if interrupted:
                result.cancelled = True
                break
            result.videos_processed += 1
        finally:
            cap.release()

    return result
