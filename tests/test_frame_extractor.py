"""视频抽帧引擎单元测试。

职责：
    验证 `probe_video`、`compute_video_hash`、`compile_naming_rule`、`preview_naming_rule`
    与 `extract_frames` 在按秒间隔抽帧、目标帧数均匀抽帧、默认哈希+毫秒命名、自定义前缀/模板命名、
    时间戳异常回退、非法命名规则拦截、最长边缩放、中文路径输出、静态帧相似去重、
    同名视频输出消歧、大步长 seek 读取以及取消行为上的表现。

运行环境与协作边界：
    通过 `pytest` 在临时目录（`tmp_path`）中运行，依赖 `opencv-python` 与 `numpy`
    生成合成 MP4 测试视频，直接覆盖 `yolo_studio.core.frame_extractor`。
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import pytest

from yolo_studio.core.frame_extractor import (
    ExtractionConfig,
    ExtractionMode,
    VideoInfo,
    _resolve_frame_timestamp_ms,
    allocate_total_frame_quotas,
    compile_naming_rule,
    compute_video_hash,
    estimate_batch_video_frames,
    estimate_video_frames,
    extract_frames,
    format_duration_hms,
    preview_naming_rule,
    probe_video,
    read_video_preview_frame,
)


def _create_synthetic_video(
    path: Path, num_frames: int = 30, fps: float = 10.0, color: tuple[int, int, int] = (0, 255, 0)
) -> Path:
    """生成包含水平移动矩形目标的合成 MP4 测试视频。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (160, 120))
    assert writer.isOpened()
    for i in range(num_frames):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        # 逐帧移动矩形位置，使相邻帧在视觉上存在明显差异
        x = (i * 4) % 120
        cv2.rectangle(frame, (x, 20), (x + 30, 80), color, -1)
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
    expected_hash = hashlib.sha256(video.read_bytes()).hexdigest()[:32]
    assert [p.name for p in res.saved_files] == [
        f"vc{expected_hash}_f0000000000.jpg",
        f"vc{expected_hash}_f0000001000.jpg",
        f"vc{expected_hash}_f0000002000.jpg",
    ]


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
    """验证不同目录下同名视频在默认哈希命名与自定义 `{video}`/前缀命名下均不会互相覆盖。"""
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()
    v1 = _create_synthetic_video(tmp_path / "cam1" / "rec.mp4", 20, 10.0, color=(0, 255, 0))
    v2 = _create_synthetic_video(tmp_path / "cam2" / "rec.mp4", 20, 10.0, color=(255, 0, 0))

    # 1) 未指定命名规则：按视频内容 SHA-256 前 32 位天然区分
    default_cfg = ExtractionConfig(
        output_dir=tmp_path / "out_default",
        mode=ExtractionMode.EVERY_N_FRAMES,
        every_n_frames=10,
    )
    res_default = extract_frames([v1, v2], default_cfg)
    assert not res_default.errors
    assert res_default.frames_saved == 4
    assert len({p.name for p in res_default.saved_files}) == 4
    h1 = compute_video_hash(v1)
    h2 = compute_video_hash(v2)
    assert h1 is not None and h2 is not None and len(h1) == 32 and h1 != h2
    assert {p.name.split("_")[0] for p in res_default.saved_files} == {f"vc{h1}", f"vc{h2}"}

    # 2) 指定自定义模板含 {video} 时，同名视频自动消歧为 cam1_rec / cam2_rec
    named_cfg = ExtractionConfig(
        output_dir=tmp_path / "out_named",
        mode=ExtractionMode.EVERY_N_FRAMES,
        every_n_frames=10,
        naming_rule="{video}_{seq:06d}",
    )
    res_named = extract_frames([v1, v2], named_cfg)
    assert not res_named.errors
    assert res_named.frames_saved == 4
    assert len({p.name for p in res_named.saved_files}) == 4
    assert {p.name.rsplit("_", 1)[0] for p in res_named.saved_files} == {"cam1_rec", "cam2_rec"}


def test_custom_naming_rules_and_conflict_guard(tmp_path: Path) -> None:
    """验证普通前缀、兼容 prefix、自定义占位符模板以及单次运行冲突保护与跨次幂等覆盖。"""
    video = _create_synthetic_video(tmp_path / "clip.mp4", num_frames=30, fps=10.0)
    full_hash = hashlib.sha256(video.read_bytes()).hexdigest()
    short_hash = full_hash[:32]

    # 1) 普通文本命名规则视作前缀，输出 {前缀}{视频名}_{6位序号}
    res_plain = extract_frames(
        [video],
        ExtractionConfig(
            output_dir=tmp_path / "plain",
            mode=ExtractionMode.EVERY_N_FRAMES,
            every_n_frames=10,
            naming_rule="sceneA_",
        ),
    )
    assert [p.name for p in res_plain.saved_files] == [
        "sceneA_clip_000001.jpg",
        "sceneA_clip_000002.jpg",
        "sceneA_clip_000003.jpg",
    ]

    # 2) 兼容旧 prefix 字段（naming_rule 留空时生效）
    res_legacy = extract_frames(
        [video],
        ExtractionConfig(
            output_dir=tmp_path / "legacy",
            mode=ExtractionMode.EVERY_N_FRAMES,
            every_n_frames=15,
            prefix="cam1_",
        ),
    )
    assert [p.name for p in res_legacy.saved_files] == [
        "cam1_clip_000001.jpg",
        "cam1_clip_000002.jpg",
    ]

    # 3) 自定义占位符模板（覆盖 video / hash / hash_full / ms / frame / seq / index / prefix）
    res_tpl = extract_frames(
        [video],
        ExtractionConfig(
            output_dir=tmp_path / "tpl",
            mode=ExtractionMode.EVERY_N_FRAMES,
            every_n_frames=10,
            prefix="p_",
            naming_rule="{prefix}{video}_{index:03d}_fr{frame:04d}_f{ms:010d}",
            image_format="png",
        ),
    )
    assert [p.name for p in res_tpl.saved_files] == [
        "p_clip_001_fr0000_f0000000000.png",
        "p_clip_002_fr0010_f0000001000.png",
        "p_clip_003_fr0020_f0000002000.png",
    ]

    # 4) 模板显式引用 {hash} / {hash_full}
    res_hash_tpl = extract_frames(
        [video],
        ExtractionConfig(
            output_dir=tmp_path / "hash_tpl",
            mode=ExtractionMode.EVERY_N_FRAMES,
            every_n_frames=20,
            naming_rule="h_{hash:.8}_{seq:02d}",
        ),
    )
    assert [p.name for p in res_hash_tpl.saved_files] == [
        f"h_{short_hash[:8]}_01.jpg",
        f"h_{short_hash[:8]}_02.jpg",
    ]
    assert compile_naming_rule("{hash_full}_{seq}").needs_video_hash is True
    assert compile_naming_rule("{video}_{seq:06d}").needs_video_hash is False

    # 5) 自定义模板不含任何变化字段时，单次运行内自动消歧不覆盖；跨次运行可幂等覆盖
    fixed_dir = tmp_path / "fixed"
    fixed_cfg = ExtractionConfig(
        output_dir=fixed_dir,
        mode=ExtractionMode.EVERY_N_FRAMES,
        every_n_frames=10,
        naming_rule="{video}",
    )
    res_fixed_1 = extract_frames([video], fixed_cfg)
    assert [p.name for p in res_fixed_1.saved_files] == ["clip.jpg", "clip_1.jpg", "clip_2.jpg"]
    res_fixed_2 = extract_frames([video], fixed_cfg)
    assert [p.name for p in res_fixed_2.saved_files] == ["clip.jpg", "clip_1.jpg", "clip_2.jpg"]


def test_naming_rule_validation_and_timestamp_fallback(tmp_path: Path) -> None:
    """验证非法命名规则被严格拦截，且帧时间戳缺失/异常时安全回退至帧率估算。"""
    # 1) 预览输出
    assert preview_naming_rule("", image_format="jpg") == "vc<32位视频SHA256>_f0000001500.jpg"
    assert preview_naming_rule("", image_format="png") == "vc<32位视频SHA256>_f0000001500.png"
    assert preview_naming_rule("cam1_", image_format="jpg") == "cam1_demo_000001.jpg"
    assert preview_naming_rule("{video}_f{ms:010d}", image_format="png") == "demo_f0000001500.png"

    # 2) 非法规则抛出 ValueError
    invalid_rules = [
        "{video",
        "video}",
        "{unknown}_{seq}",
        "{video.__class__}",
        "{video[0]}",
        "{video!r}",
        "{video:06d}",
        "dir/name_{seq}",
        "bad:name_{seq}",
        "..",
    ]
    for bad in invalid_rules:
        with pytest.raises(ValueError):
            compile_naming_rule(bad)

    # 3) 传入 extract_frames 时写入 errors 且不落盘
    video = _create_synthetic_video(tmp_path / "v.mp4", num_frames=10, fps=10.0)
    res_bad = extract_frames(
        [video],
        ExtractionConfig(output_dir=tmp_path / "bad_out", naming_rule="{not_allowed}"),
    )
    assert res_bad.errors
    assert res_bad.frames_saved == 0
    assert res_bad.videos_processed == 0

    # 4) CAP_PROP_POS_MSEC 正常与异常回退逻辑
    class _FakeCap:
        def __init__(self, raw_ms: float) -> None:
            self._raw_ms = raw_ms

        def get(self, _prop: int) -> float:
            return self._raw_ms

    assert _resolve_frame_timestamp_ms(_FakeCap(0.0), frame_idx=0, fps=25.0) == 0
    assert _resolve_frame_timestamp_ms(_FakeCap(1234.6), frame_idx=30, fps=25.0) == 1235
    assert _resolve_frame_timestamp_ms(_FakeCap(0.0), frame_idx=10, fps=25.0) == 400
    assert _resolve_frame_timestamp_ms(_FakeCap(-1.0), frame_idx=10, fps=25.0) == 400
    assert _resolve_frame_timestamp_ms(_FakeCap(math.nan), frame_idx=10, fps=25.0) == 400


def test_large_step_seek_and_cancel(tmp_path: Path) -> None:
    """验证超过 seek 阈值的大步长抽帧结果正确，并验证取消后不计入已处理视频。"""
    video = _create_synthetic_video(tmp_path / "long.mp4", num_frames=700, fps=25.0)
    cfg = ExtractionConfig(output_dir=tmp_path / "seek", mode=ExtractionMode.EVERY_N_FRAMES, every_n_frames=320)
    res = extract_frames([video], cfg)
    assert res.frames_saved == 3  # 第 0、320、640 帧
    expected_hash = hashlib.sha256(video.read_bytes()).hexdigest()[:32]
    assert [p.name for p in res.saved_files] == [
        f"vc{expected_hash}_f0000000000.jpg",
        f"vc{expected_hash}_f0000012800.jpg",
        f"vc{expected_hash}_f0000025600.jpg",
    ]

    cancel = Event()
    cancel.set()
    assert compute_video_hash(video, cancel_event=cancel) is None
    res2 = extract_frames([video], ExtractionConfig(output_dir=tmp_path / "cancel"), cancel_event=cancel)
    assert res2.cancelled and res2.videos_processed == 0


def test_format_duration_estimate_frames_preview_and_subfolder_rule(tmp_path: Path) -> None:
    """验证可读时长格式化、预估抽帧数与实际一致性、只读预览抓帧与子目录命名示例。"""
    # 1) 时长格式化边界与小时进制
    assert format_duration_hms(0.0) == "00:00"
    assert format_duration_hms(-5.0) == "00:00"
    assert format_duration_hms(math.nan) == "00:00"
    assert format_duration_hms(math.inf) == "00:00"
    assert format_duration_hms(411.1) == "06:51"
    assert format_duration_hms(3661.0) == "1:01:01"

    # 2) 三种抽帧策略与起止裁剪预估帧数与实际 extract_frames(dedup=0) 严格一致
    video = _create_synthetic_video(tmp_path / "est.mp4", num_frames=45, fps=15.0)
    info = probe_video(video)
    assert info.frame_count == 45

    configs = [
        ExtractionConfig(
            output_dir=tmp_path / "e_sec",
            mode=ExtractionMode.INTERVAL_SECONDS,
            interval_seconds=0.5,
            start_sec=0.4,
            end_sec=2.6,
            dedup_threshold=0.0,
        ),
        ExtractionConfig(
            output_dir=tmp_path / "e_step",
            mode=ExtractionMode.EVERY_N_FRAMES,
            every_n_frames=7,
            start_sec=0.0,
            end_sec=None,
            dedup_threshold=0.0,
        ),
        ExtractionConfig(
            output_dir=tmp_path / "e_cnt",
            mode=ExtractionMode.TARGET_COUNT,
            target_count=8,
            start_sec=0.2,
            end_sec=2.2,
            dedup_threshold=0.0,
        ),
    ]
    for cfg in configs:
        estimated = estimate_video_frames(info, cfg)
        real = extract_frames([video], cfg)
        assert estimated == real.frames_saved

    # 无效帧数或空时间范围返回 0
    assert estimate_video_frames(
        VideoInfo(path=video, fps=15.0, frame_count=0, width=160, height=120, duration_sec=0.0),
        configs[0],
    ) == 0
    assert estimate_video_frames(
        info,
        ExtractionConfig(output_dir=tmp_path / "bad_span", start_sec=2.0, end_sec=1.0),
    ) == 0

    # 3) 只读预览抓帧（首帧、指定时间、超界时间夹紧与不存在文件）
    f0 = read_video_preview_frame(video, timestamp_sec=0.0, max_dimension=80)
    assert f0 is not None
    img0, sec0 = f0
    assert max(img0.shape[:2]) == 80
    assert sec0 >= 0.0

    f_mid = read_video_preview_frame(video, timestamp_sec=1.5, max_dimension=160)
    assert f_mid is not None
    img_mid, sec_mid = f_mid
    assert img_mid.shape[:2] == (120, 160)
    assert sec_mid >= 1.0

    f_over = read_video_preview_frame(video, timestamp_sec=999.0, max_dimension=160)
    assert f_over is not None
    assert read_video_preview_frame(tmp_path / "missing.mp4") is None

    # 4) 命名规则示例支持分建子目录前缀
    assert preview_naming_rule("", image_format="jpg", create_video_subfolder=True) == (
        "demo/vc<32位视频SHA256>_f0000001500.jpg"
    )
    assert preview_naming_rule("{video}_{seq:04d}", image_format="png", create_video_subfolder=True) == (
        "demo/demo_0001.png"
    )


def test_random_total_count_stratified_sampling_and_quotas(tmp_path: Path) -> None:
    """验证合计总帧数分层随机抽帧的时长配额分配、容量回流、桶内覆盖及默认自动随机性。"""
    # 1) 配额分配：按有效时长权重分配 + 同长不同帧率公平分配 + 超短视频触顶回流
    v_short = VideoInfo(path=tmp_path / "s.mp4", fps=10.0, frame_count=50, width=160, height=120, duration_sec=5.0)
    v_long = VideoInfo(path=tmp_path / "l.mp4", fps=20.0, frame_count=300, width=160, height=120, duration_sec=15.0)
    cfg_40 = ExtractionConfig(
        output_dir=tmp_path / "q1", mode=ExtractionMode.RANDOM_TOTAL_COUNT, target_count=40
    )
    assert allocate_total_frame_quotas([v_short, v_long], cfg_40) == [10, 30]
    assert estimate_batch_video_frames([v_short, v_long], cfg_40) == [10, 30]

    # 同为 10 秒但帧率分别为 10fps 与 30fps，按时长权重应平分 20 帧
    v_fps10 = VideoInfo(path=tmp_path / "f10.mp4", fps=10.0, frame_count=100, width=160, height=120, duration_sec=10.0)
    v_fps30 = VideoInfo(path=tmp_path / "f30.mp4", fps=30.0, frame_count=300, width=160, height=120, duration_sec=10.0)
    assert allocate_total_frame_quotas(
        [v_fps10, v_fps30],
        ExtractionConfig(output_dir=tmp_path / "q2", mode=ExtractionMode.RANDOM_TOTAL_COUNT, target_count=20),
    ) == [10, 10]

    # 同有效时长 1s，但容量分别为 5 帧与 100 帧；目标 30 帧时前者触顶 5 帧，余量回流给后者 -> [5, 25]
    v_cap5 = VideoInfo(path=tmp_path / "c5.mp4", fps=5.0, frame_count=5, width=160, height=120, duration_sec=1.0)
    v_cap100 = VideoInfo(path=tmp_path / "c100.mp4", fps=100.0, frame_count=100, width=160, height=120, duration_sec=1.0)
    assert allocate_total_frame_quotas(
        [v_cap5, v_cap100],
        ExtractionConfig(output_dir=tmp_path / "q3", mode=ExtractionMode.RANDOM_TOTAL_COUNT, target_count=30),
    ) == [5, 25]

    # 目标小于视频个数，以及目标超出全部总容量时的边界钳制
    v_2s = VideoInfo(path=tmp_path / "t2.mp4", fps=10.0, frame_count=20, width=160, height=120, duration_sec=2.0)
    assert allocate_total_frame_quotas(
        [v_2s, v_2s, v_2s],
        ExtractionConfig(output_dir=tmp_path / "q4", mode=ExtractionMode.RANDOM_TOTAL_COUNT, target_count=2),
    ) == [1, 1, 0]
    assert allocate_total_frame_quotas(
        [v_short, v_long],
        ExtractionConfig(output_dir=tmp_path / "q5", mode=ExtractionMode.RANDOM_TOTAL_COUNT, target_count=9999),
    ) == [50, 300]

    # 2) 真实分层随机抽帧：结合起止裁剪，验证总帧数精确、各帧分别落在对应分层桶内
    va = _create_synthetic_video(tmp_path / "va.mp4", num_frames=40, fps=10.0)
    vb = _create_synthetic_video(tmp_path / "vb.mp4", num_frames=120, fps=20.0)
    cfg_run = ExtractionConfig(
        output_dir=tmp_path / "rand_out",
        mode=ExtractionMode.RANDOM_TOTAL_COUNT,
        target_count=10,
        start_sec=1.0,
        end_sec=3.0,
        dedup_threshold=0.0,
        naming_rule="{video}_{frame:06d}",
        random_seed=1234,
    )
    est_list = estimate_batch_video_frames([probe_video(va), probe_video(vb)], cfg_run)
    assert est_list == [5, 5]

    res1 = extract_frames([va, vb], cfg_run)
    assert not res1.errors
    assert res1.videos_processed == 2
    assert res1.frames_saved == 10

    va_frames = sorted(int(p.stem.split("_")[-1]) for p in res1.saved_files if p.name.startswith("va_"))
    vb_frames = sorted(int(p.stem.split("_")[-1]) for p in res1.saved_files if p.name.startswith("vb_"))
    assert len(va_frames) == 5 and len(vb_frames) == 5
    for idx, frame_no in enumerate(va_frames):
        lo, hi = 10 + (idx * 20) // 5, 10 + ((idx + 1) * 20) // 5
        assert lo <= frame_no < hi
    for idx, frame_no in enumerate(vb_frames):
        lo, hi = 20 + (idx * 40) // 5, 20 + ((idx + 1) * 40) // 5
        assert lo <= frame_no < hi

    # 相同 seed 可复现
    cfg_run_dup = ExtractionConfig(
        output_dir=tmp_path / "rand_out_dup",
        mode=ExtractionMode.RANDOM_TOTAL_COUNT,
        target_count=10,
        start_sec=1.0,
        end_sec=3.0,
        dedup_threshold=0.0,
        naming_rule="{video}_{frame:06d}",
        random_seed=1234,
    )
    res_dup = extract_frames([va, vb], cfg_run_dup)
    assert [p.name for p in res_dup.saved_files] == [p.name for p in res1.saved_files]

    # 3) 目标总帧数 < 视频数时，配额为 0 的视频无错跳过但计入已处理视频数
    res_one = extract_frames(
        [va, vb],
        ExtractionConfig(
            output_dir=tmp_path / "rand_one",
            mode=ExtractionMode.RANDOM_TOTAL_COUNT,
            target_count=1,
            dedup_threshold=0.0,
            random_seed=7,
        ),
    )
    assert not res_one.errors
    assert res_one.videos_processed == 2
    assert res_one.frames_saved == 1

    # 4) 默认 random_seed=None 每次自动随机，多次抽样在大采样空间上产生差异化结果
    observed_sets: set[tuple[str, ...]] = set()
    for run_i in range(5):
        res_auto = extract_frames(
            [vb],
            ExtractionConfig(
                output_dir=tmp_path / f"rand_auto_{run_i}",
                mode=ExtractionMode.RANDOM_TOTAL_COUNT,
                target_count=6,
                dedup_threshold=0.0,
                naming_rule="{video}_{frame:06d}",
            ),
        )
        assert res_auto.frames_saved == 6
        observed_sets.add(tuple(p.name for p in res_auto.saved_files))
    assert len(observed_sets) >= 2
