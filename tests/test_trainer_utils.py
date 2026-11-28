"""线程路由输出捕获、训练超参数解析/日志与 CLI 入口单元测试。

职责：
    验证 `core.log_capture` 的 `\\r` 覆盖语义、重入 `print` 防递归、线程隔离与 `ultralytics` 日志转发，
    `core.trainer` 中断点续训解析、超参数预设、附加参数解析、双阶段训练超参数日志输出、推理输出消歧与硬件探测缓存，
    以及 `yolo_studio.cli` 抽帧与训练子命令参数解析、预设覆盖及端到端执行。

运行环境与协作边界：
    通过 `pytest` 运行；`core.trainer` 仅在调用 `detect_compute_device` 时按需加载 `torch`，
    测试通过模拟轻量 `YOLO` 对象校验训练回调，不启动真实模型训练、不访问外部网络，也不加载任何 YOLO 权重。
"""

from __future__ import annotations

import logging
import sys
import threading
import types
from pathlib import Path

import pytest

from yolo_studio.core.log_capture import _LineAssembler, capture_thread_output
from yolo_studio.core.trainer import (
    TRAIN_HYP_PRESETS,
    TrainConfig,
    _resolve_resume_checkpoint,
    _unique_pred_path,
    apply_hyp_preset,
    build_train_kwargs,
    parse_extra_train_args,
    train_yolo26,
)


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
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    handler = logging.StreamHandler(sys.stdout)
    logger.handlers = [handler]
    logger.propagate = False
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
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

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

    # 未指定命名规则与前缀时，默认采用 vc{sha256前32位}_f{10位毫秒时长}
    import hashlib

    expected_hash = hashlib.sha256(video_path.read_bytes()).hexdigest()[:32]
    default_out = tmp_path / "cli_default_out"
    code_default = cli_main(
        [
            "extract",
            str(video_path),
            "-o",
            str(default_out),
            "--mode",
            "every_n_frames",
            "--step",
            "10",
            "--dedup",
            "0",
        ]
    )
    assert code_default == 0
    assert sorted(p.name for p in default_out.glob("*.jpg")) == [
        f"vc{expected_hash}_f0000000000.jpg",
        f"vc{expected_hash}_f0000001000.jpg",
        f"vc{expected_hash}_f0000002000.jpg",
    ]

    # 指定自定义 --naming-rule 模板时以模板为准；非法模板返回非零退出码
    tpl_out = tmp_path / "cli_tpl_out"
    code_tpl = cli_main(
        [
            "extract",
            str(video_path),
            "-o",
            str(tpl_out),
            "--mode",
            "every_n_frames",
            "--step",
            "15",
            "--dedup",
            "0",
            "--naming-rule",
            "{video}_t{ms:010d}",
        ]
    )
    assert code_tpl == 0
    assert sorted(p.name for p in tpl_out.glob("*.jpg")) == [
        "clip_t0000000000.jpg",
        "clip_t0000001500.jpg",
    ]

    code_bad = cli_main(
        [
            "extract",
            str(video_path),
            "-o",
            str(tmp_path / "cli_bad_out"),
            "--naming-rule",
            "{unknown_field}",
        ]
    )
    assert code_bad == 1


def test_capture_thread_output_reentrant_print_no_recursion() -> None:
    """验证把 `print` 本身作为 `capture_thread_output` 回调时不会触发无限递归。"""
    with capture_thread_output(print):
        print("重入输出安全测试")


def test_parse_extra_train_args_formats_and_errors() -> None:
    """验证附加超参数解析器兼容 JSON 对象（含尾随逗号）、无花括号键值片段与 key=value 语法，并拦截非法输入。"""
    assert parse_extra_train_args("   ") == {}

    # 1) 标准 JSON 及带尾随逗号的块（如从配置文件末尾截取的 `},`）
    parsed_json = parse_extra_train_args('{"box": 7.5, "plots": false, "freeze": [0, 1],},')
    assert parsed_json == {"box": 7.5, "plots": False, "freeze": [0, 1]}

    # 2) 无外层花括号的 JSON 键值片段
    snippet = '"mosaic": 0.0,\n"scale": 0.15,\n"optimizer": "AdamW",'
    assert parse_extra_train_args(snippet) == {
        "mosaic": 0.0,
        "scale": 0.15,
        "optimizer": "AdamW",
    }

    # 3) key=value 混合类型表达式（含列表内部逗号）
    kv = parse_extra_train_args("box=7.5, weight_decay=0.0005; single_cls=true, freeze=[0, 1, 2], time=null")
    assert kv == {
        "box": 7.5,
        "weight_decay": 0.0005,
        "single_cls": True,
        "freeze": [0, 1, 2],
        "time": None,
    }

    # 4) 非法语法与非法键名均抛出中文 ValueError
    with pytest.raises(ValueError, match="分隔符"):
        parse_extra_train_args("invalid_token_only")
    with pytest.raises(ValueError, match="未闭合"):
        parse_extra_train_args("freeze=[0, 1")
    with pytest.raises(ValueError, match="超参数名称"):
        parse_extra_train_args('{"bad-key": 1}')


def test_apply_hyp_preset_and_build_train_kwargs(tmp_path: Path) -> None:
    """验证 `game_ui` 预设字段映射及 `extra_args` 对同名参数的最高优先级覆盖。"""
    data_yaml = tmp_path / "data.yaml"
    cfg = TrainConfig(
        data_yaml=data_yaml,
        epochs=160,
        imgsz=960,
        batch=4,
        patience=160,
        workers=0,
        amp=False,
        plots=False,
        extra_args={"box": 7.5, "mosaic": 0.05},
    )
    apply_hyp_preset(cfg, "game_ui")
    kwargs = build_train_kwargs(cfg)

    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == 0.001
    assert kwargs["lrf"] == 0.05
    assert kwargs["nbs"] == 16
    assert kwargs["scale"] == 0.15
    assert kwargs["translate"] == 0.03
    assert kwargs["hsv_h"] == 0.01
    assert kwargs["hsv_s"] == 0.2
    assert kwargs["hsv_v"] == 0.2
    assert kwargs["amp"] is False
    assert kwargs["plots"] is False
    # extra_args 覆盖同名 mosaic 并附加自定义 box
    assert kwargs["mosaic"] == 0.05
    assert kwargs["box"] == 7.5

    with pytest.raises(ValueError, match="未知的训练超参数预设"):
        apply_hyp_preset(cfg, "not_exists")


def test_train_yolo26_emits_submitted_and_effective_hyperparams_to_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证训练启动前输出提交的 `arguments` 配置块，且 `on_train_start` 输出合并推导后的实际生效超参数。"""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [target]\n", encoding="utf-8")

    class _FakeYOLO:
        def __init__(self, model_source: str) -> None:
            self.model_source = model_source
            self.callbacks: dict[str, list] = {}
            self.trainer: object | None = None

        def add_callback(self, event: str, cb: object) -> None:
            self.callbacks.setdefault(event, []).append(cb)

        def train(self, **kwargs: object) -> object:
            save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"pt")

            # 模拟 Ultralytics `optimizer='auto'` 自动选定 AdamW 并重算 lr0、计算梯度累积步数后的 Trainer 状态
            effective_args = dict(kwargs)
            effective_args["optimizer"] = "AdamW"
            effective_args["lr0"] = 0.001667
            effective_args["momentum"] = 0.9
            effective_args["weight_decay"] = 0.0005

            trainer_ns = types.SimpleNamespace(
                args=types.SimpleNamespace(**effective_args),
                amp=False,
                accumulate=4,
                save_dir=save_dir,
                epoch=0,
                epochs=int(kwargs["epochs"]),  # type: ignore[arg-type]
                metrics={"metrics/mAP50(B)": 0.88},
                tloss=None,
            )
            self.trainer = trainer_ns
            for cb in self.callbacks.get("on_train_start", []):
                cb(trainer_ns)
            for cb in self.callbacks.get("on_fit_epoch_end", []):
                cb(trainer_ns)
            return types.SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.88})

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    logs: list[str] = []
    cfg = TrainConfig(
        data_yaml=data_yaml,
        epochs=10,
        batch=4,
        nbs=16,
        optimizer="auto",
        mosaic=0.0,
        project_dir=tmp_path / "runs",
        run_name="exp_log",
        extra_args={"box": 8.0},
    )
    res = train_yolo26(cfg, log_callback=logs.append)
    assert res.best_weights is not None

    joined = "\n".join(logs)
    assert "=== 训练超参数配置 (提交至 Ultralytics) ===" in joined
    assert '"arguments": {' in joined
    assert '"mosaic": 0.0' in joined
    assert '"box": 8.0' in joined
    assert '当前 optimizer="auto"' in joined

    assert "=== 实际生效训练超参数 (Trainer 初始化完成) ===" in joined
    assert "nbs=16 (accumulate=4)" in joined
    assert "optimizer=AdamW, lr0=0.001667" in joined
    assert "[附加参数]   box=8.0" in joined


def test_cli_train_preset_and_overrides_build_expected_config(tmp_path: Path) -> None:
    """验证 CLI `train` 子命令支持 `--preset`、细分超参数覆盖与 `--extra-arg`，并保持预设名与核心表同步。"""
    from yolo_studio.cli import _build_cli_train_config, build_parser, main as cli_main

    parser = build_parser()
    # 核对 CLI `--preset` 可选值与核心预设字典完全一致
    train_sub = next(
        sub_parser
        for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict) and "train" in action.choices
        for name, sub_parser in action.choices.items()
        if name == "train"
    )
    preset_action = next(a for a in train_sub._actions if "--preset" in a.option_strings)
    assert set(preset_action.choices or ()) == set(TRAIN_HYP_PRESETS.keys())

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [ui]\n", encoding="utf-8")
    parsed = parser.parse_args(
        [
            "train",
            "-d",
            str(data_yaml),
            "--preset",
            "game_ui",
            "--epochs",
            "160",
            "--patience",
            "160",
            "--imgsz",
            "960",
            "--batch",
            "4",
            "--workers",
            "0",
            "--lr0",
            "0.0008",
            "--no-amp",
            "--no-plots",
            "--extra-arg",
            "box=7.5, freeze=[0, 1]",
        ]
    )
    cfg = _build_cli_train_config(parsed)
    assert isinstance(cfg, TrainConfig)
    assert cfg.optimizer == "AdamW"
    assert cfg.lr0 == 0.0008  # 命令行显式值覆盖 game_ui 预设中的 0.001
    assert cfg.lrf == 0.05
    assert cfg.nbs == 16
    assert cfg.mosaic == 0.0
    assert cfg.scale == 0.15
    assert cfg.amp is False
    assert cfg.plots is False
    assert cfg.workers == 0
    assert cfg.patience == 160
    assert cfg.extra_args == {"box": 7.5, "freeze": [0, 1]}

    # 非法 --extra-arg 返回退出码 2
    bad_code = cli_main(["train", "-d", str(data_yaml), "--extra-arg", "broken_entry"])
    assert bad_code == 2


def test_parse_train_arguments_text_and_split_known_extra() -> None:
    """验证训练参数 JSON / 日志片段导入解析与界面字段/附加参数分流保护逻辑。"""
    from yolo_studio.core.trainer import (
        parse_train_arguments_text,
        split_known_and_extra_train_args,
    )

    sample = """
    === 训练超参数配置 (提交至 Ultralytics) ===
    "arguments": {
      "data": "C:\\\\other\\\\data.yaml",
      "project": "C:\\\\other\\\\runs",
      "name": "gpu-training",
      "epochs": 160,
      "optimizer": "AdamW",
      "lr0": 0.001,
      "exist_ok": false,
      "pretrained": true,
      "save": true,
      "box": 7.5
    },
    [提示] 示例提示行
    """
    parsed = parse_train_arguments_text(sample)
    assert parsed["epochs"] == 160
    assert parsed["optimizer"] == "AdamW"
    assert parsed["box"] == 7.5

    # 默认不导入外部路径，且过滤与默认行为一致的 exist_ok/pretrained/save
    known, extra = split_known_and_extra_train_args(parsed, include_paths=False)
    assert "data" not in known and "project" not in known and "name" not in known
    assert known["epochs"] == 160
    assert known["optimizer"] == "AdamW"
    assert extra == {"box": 7.5}

    # 显式开启 include_paths 时纳入路径与实验名
    known_with_paths, _ = split_known_and_extra_train_args(parsed, include_paths=True)
    assert known_with_paths["name"] == "gpu-training"
    assert known_with_paths["data"] == "C:\\other\\data.yaml"

    # 外层包一层 {"arguments": {...}} 同样支持解包
    wrapped = parse_train_arguments_text('{"arguments": {"mosaic": 0.0, "scale": 0.15}}')
    assert wrapped == {"mosaic": 0.0, "scale": 0.15}

    with pytest.raises(ValueError, match="请输入"):
        parse_train_arguments_text("   \n=== 仅标题 ===\n")


def test_train_yolo26_final_eval_keeps_completed_epoch_and_marks_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证跑满与早停场景下，Ultralytics `final_eval()` 的临时 `epoch += 1` 不会导致完成轮次多加 1。"""
    import threading

    from yolo_studio.core.trainer import EpochMetrics

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [target]\n", encoding="utf-8")

    class _FakeLossTensor:
        def __init__(self, vals: list[float]) -> None:
            self._vals = vals

        def detach(self) -> "_FakeLossTensor":
            return self

        def cpu(self) -> "_FakeLossTensor":
            return self

        def tolist(self) -> list[float]:
            return list(self._vals)

    def _run_case(*, total_epochs: int, actual_epochs: int, cancel_during_final_eval: bool) -> tuple[list[EpochMetrics], bool]:
        cancel_ev = threading.Event()

        class _FakeYOLO:
            def __init__(self, _model_source: str) -> None:
                self.callbacks: dict[str, list] = {}
                self.trainer: object | None = None

            def add_callback(self, event: str, cb: object) -> None:
                self.callbacks.setdefault(event, []).append(cb)

            def train(self, **kwargs: object) -> object:
                save_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
                (save_dir / "weights").mkdir(parents=True, exist_ok=True)
                (save_dir / "weights" / "best.pt").write_bytes(b"pt")

                trainer_ns = types.SimpleNamespace(
                    args=types.SimpleNamespace(**kwargs),
                    save_dir=save_dir,
                    epoch=0,
                    epochs=total_epochs,
                    metrics={},
                    tloss=None,
                    stop=False,
                )
                self.trainer = trainer_ns
                for cb in self.callbacks.get("on_train_start", []):
                    cb(trainer_ns)

                # 模拟正常训练轮次：每轮成对触发 on_train_epoch_end -> on_fit_epoch_end
                for idx in range(actual_epochs):
                    trainer_ns.epoch = idx
                    trainer_ns.tloss = _FakeLossTensor([0.4 - idx * 0.05, 0.25 - idx * 0.02])
                    trainer_ns.metrics = {
                        "metrics/precision(B)": 0.80 + idx * 0.02,
                        "metrics/recall(B)": 0.75 + idx * 0.02,
                        "metrics/mAP50(B)": 0.82 + idx * 0.02,
                        "metrics/mAP50-95(B)": 0.60 + idx * 0.02,
                    }
                    for cb in self.callbacks.get("on_train_epoch_end", []):
                        cb(trainer_ns)
                    for cb in self.callbacks.get("on_fit_epoch_end", []):
                        cb(trainer_ns)

                # 模拟 Ultralytics BaseTrainer.final_eval()：跳过 on_train_epoch_end，
                # 将 epoch 临时加 1 后再次触发 on_fit_epoch_end 上报 best.pt 最终评估指标
                if cancel_during_final_eval:
                    cancel_ev.set()
                best_metrics = {
                    "metrics/precision(B)": 0.945,
                    "metrics/recall(B)": 0.912,
                    "metrics/mAP50(B)": 0.961,
                    "metrics/mAP50-95(B)": 0.785,
                }
                trainer_ns.metrics = dict(best_metrics)
                trainer_ns.epoch += 1
                for cb in self.callbacks.get("on_fit_epoch_end", []):
                    cb(trainer_ns)
                trainer_ns.epoch -= 1
                return types.SimpleNamespace(results_dict=best_metrics)

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYOLO  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

        received: list[EpochMetrics] = []
        cfg = TrainConfig(
            data_yaml=data_yaml,
            epochs=total_epochs,
            project_dir=tmp_path / "runs",
            run_name=f"case_{total_epochs}_{actual_epochs}",
        )
        res = train_yolo26(cfg, epoch_callback=received.append, cancel_event=cancel_ev)
        return received, res.stopped_early

    # 1) 跑满全部 3 轮：常规回调为 1, 2, 3；final_eval 保持 epoch=3 且 is_final_eval=True，迟到取消不计为提前停训
    full_events, full_stopped = _run_case(total_epochs=3, actual_epochs=3, cancel_during_final_eval=True)
    assert [(e.epoch, e.total_epochs, e.is_final_eval) for e in full_events] == [
        (1, 3, False),
        (2, 3, False),
        (3, 3, False),
        (3, 3, True),
    ]
    assert full_events[-1].precision == pytest.approx(0.945)
    assert full_events[-1].recall == pytest.approx(0.912)
    assert full_events[-1].map50 == pytest.approx(0.961)
    assert full_events[-1].map50_95 == pytest.approx(0.785)
    assert full_events[-1].box_loss == 0.0 and full_events[-1].cls_loss == 0.0
    assert full_stopped is False

    # 2) 早停在第 2 / 10 轮：final_eval 同样保持实际完成轮次 epoch=2（不会变成 3 / 10）
    early_events, _ = _run_case(total_epochs=10, actual_epochs=2, cancel_during_final_eval=False)
    assert [(e.epoch, e.total_epochs, e.is_final_eval) for e in early_events] == [
        (1, 10, False),
        (2, 10, False),
        (2, 10, True),
    ]
