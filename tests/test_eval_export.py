"""权重元数据探测、模型导出配置/IO 预览、CLI `export` 子命令与第 6 页（`EvalTab`）单元测试。

职责：
    1. 验证 `.pt` 权重元数据只读解析（训练尺寸方形/矩形、有序类别清单、检测头分支能力及异常兜底）；
    2. 验证 `ExportConfig` 参数规范化、`build_export_kwargs` 按格式过滤（ONNX / TorchScript / TensorRT）、
       `describe_export_io` 输入输出张量估算与风险告警，以及 `export_model_with_config` 端到端行为；
    3. 验证 CLI `export` 子命令默认沿用权重训练尺寸、`--inspect-only`、全量导出选项与非法参数退出码 `2`；
    4. 在离线 `offscreen` Qt 环境下验证 `EvalTab` 自动探测权重元数据并同步验证/导出尺寸、
       独立导出尺寸（方形/矩形/一键恢复）、格式与 NMS 三态防呆联动、过期异步请求隔离及 `ExportWorker` 传参。

运行环境与协作边界：
    通过 `pytest` 离线运行，依赖 `PyQt6`；使用模拟 checkpoint、模拟 `YOLO` 及轻量假 Worker，
    不触发真实模型导出或外部网络访问；若本地存在预置 `yolo26s.pt` 权重，额外执行一次只读实测。
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QObject, Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import QApplication, QMessageBox

import yolo_studio.core.trainer as trainer_module
import yolo_studio.gui.tab_eval as tab_eval_module
from yolo_studio.cli import _build_cli_export_config, build_parser, main as cli_main
from yolo_studio.core.trainer import (
    DeviceInfo,
    ExportConfig,
    WeightsMetadata,
    align_imgsz_to_stride,
    build_export_kwargs,
    describe_export_io,
    export_model,
    export_model_with_config,
    format_class_order,
    inspect_weights_metadata,
    normalize_export_imgsz,
    parse_export_nms_mode,
)
from yolo_studio.gui.tab_eval import EvalTab


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """初始化模块级无头 `QApplication`；与内嵌工作台测试共用进程时提前开启 OpenGL 上下文共享。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if existing is not None:
        return existing
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication([])


def _make_device_info(*, has_gpu: bool = True) -> DeviceInfo:
    if has_gpu:
        return DeviceInfo(
            cuda_available=True,
            device_count=1,
            device_name="Mock RTX 4060",
            vram_gb=8.0,
            recommended_device="0",
            probing=False,
            gpu_devices=(("0", "Mock RTX 4060", 8.0),),
        )
    return DeviceInfo(
        cuda_available=False,
        device_count=0,
        device_name="Mock CPU",
        vram_gb=0.0,
        recommended_device="cpu",
        probing=False,
        gpu_devices=(),
    )


class _StubWeightsInspectWorker(QObject):
    """替代后台 `WeightsInspectWorker` 的同步信号桩，支持测试中手动控制完成顺序。"""

    finished_metadata = Signal(int, object)
    failed = Signal(int, str)
    finished = Signal()

    def __init__(self, weights_path: Path, request_id: int = 0) -> None:
        super().__init__()
        self.weights_path = weights_path
        self.request_id = request_id
        self.started = False
        self.running = False

    def start(self) -> None:
        self.started = True
        self.running = True

    def isRunning(self) -> bool:
        return self.running

    def complete_with(self, meta: WeightsMetadata) -> None:
        self.running = False
        self.finished_metadata.emit(self.request_id, meta)
        self.finished.emit()

    def fail_with(self, err: str) -> None:
        self.running = False
        self.failed.emit(self.request_id, err)
        self.finished.emit()


class _StubExportWorker(QObject):
    """替代后台 `ExportWorker` 的轻量信号桩，用于记录传入的 `ExportConfig`。"""

    log_line = Signal(str)
    finished_path = Signal(str)
    failed = Signal(str)

    def __init__(self, config: ExportConfig) -> None:
        super().__init__()
        self.config = config
        self.started = False
        self.running = False

    def start(self) -> None:
        self.started = True
        self.running = True

    def isRunning(self) -> bool:
        return self.running


def test_export_size_nms_and_class_formatting_helpers() -> None:
    """验证尺寸规范化、步长对齐、NMS 模式解析与有序类别格式化边界。"""
    assert normalize_export_imgsz(960) == (960, 960)
    assert normalize_export_imgsz("960") == (960, 960)
    assert normalize_export_imgsz("544,960") == (544, 960)
    assert normalize_export_imgsz("544x960") == (544, 960)
    assert normalize_export_imgsz([544, 960]) == (544, 960)
    assert align_imgsz_to_stride((540, 960), max_stride=32) == (544, 960)

    with pytest.raises(ValueError, match="导出尺寸"):
        normalize_export_imgsz("bad_size")
    with pytest.raises(ValueError, match="有效范围"):
        normalize_export_imgsz(16)
    with pytest.raises(ValueError, match="导出尺寸"):
        normalize_export_imgsz(True)  # type: ignore[arg-type]

    assert parse_export_nms_mode("raw") is None
    assert parse_export_nms_mode(None) is None
    assert parse_export_nms_mode("end2end") is False
    assert parse_export_nms_mode(False) is False
    assert parse_export_nms_mode("embedded") is True
    assert parse_export_nms_mode(True) is True
    with pytest.raises(ValueError, match="无效的 NMS 模式"):
        parse_export_nms_mode("unknown_mode")

    names = ((0, "hero"), (1, "enemy"), (2, "boss"), (3, "item"))
    assert format_class_order(names) == "0: hero, 1: enemy, 2: boss, 3: item"
    assert format_class_order(names, max_items=2) == "0: hero, 1: enemy, ...（共 4 类）"
    assert format_class_order(()) == "未记录"


def test_inspect_weights_metadata_parses_checkpoint_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 `inspect_weights_metadata` 能解析方形/矩形训练尺寸、排序后的类别字典与端到端头能力。"""
    pt_file = tmp_path / "best.pt"
    pt_file.write_bytes(b"fake-pt")

    fake_head = types.SimpleNamespace(one2one_cv2=object(), cv2=object())
    fake_model = types.SimpleNamespace(
        # 故意打乱字典键顺序，校验输出按编号升序排列
        names={2: "boss", 0: "player", 1: "monster"},
        args={"imgsz": 640, "task": "detect"},
        stride=[8, 16, 32],
        yaml={"channels": 3},
        model=[fake_head],
    )
    fake_ckpt = {
        "ema": fake_model,
        "train_args": {"imgsz": 960, "task": "detect"},
        "epoch": 88,
    }

    fake_tasks_mod = types.ModuleType("ultralytics.nn.tasks")
    fake_tasks_mod.torch_safe_load = lambda _p: (fake_ckpt, str(pt_file))  # type: ignore[attr-defined]
    fake_nn_mod = types.ModuleType("ultralytics.nn")
    fake_nn_mod.tasks = fake_tasks_mod  # type: ignore[attr-defined]
    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.nn = fake_nn_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setitem(sys.modules, "ultralytics.nn", fake_nn_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.tasks", fake_tasks_mod)

    meta = inspect_weights_metadata(pt_file)
    assert meta.train_imgsz == (960, 960)
    assert meta.task == "detect"
    assert meta.class_names == ((0, "player"), (1, "monster"), (2, "boss"))
    assert meta.num_classes == 3
    assert meta.strides == (8, 16, 32)
    assert meta.supports_end2end is True
    assert meta.one2many_available is True
    assert meta.epoch == 88

    # 测试矩形训练尺寸 + 列表类别 + 仅保留 one2one 分支
    rect_head = types.SimpleNamespace(one2one_cv2=object(), cv2=None)
    rect_model = types.SimpleNamespace(
        names=["button", "dialog"],
        args={"imgsz": [544, 960]},
        stride=[8, 16, 32],
        yaml={"channels": 3},
        model=[rect_head],
    )
    fake_tasks_mod.torch_safe_load = lambda _p: ({"model": rect_model, "epoch": -1}, str(pt_file))  # type: ignore[attr-defined]
    meta_rect = inspect_weights_metadata(pt_file)
    assert meta_rect.train_imgsz == (544, 960)
    assert meta_rect.class_names == ((0, "button"), (1, "dialog"))
    assert meta_rect.supports_end2end is True
    assert meta_rect.one2many_available is False
    assert meta_rect.epoch is None

    with pytest.raises(FileNotFoundError):
        inspect_weights_metadata(tmp_path / "missing.pt")


def test_inspect_real_bundled_yolo26s_weights_if_present() -> None:
    """若本地存在预置 `yolo26s.pt` 权重，则在独立子进程中执行真实只读元数据解析校验。

    设计原因：
        通过隔离子进程导入真实 `torch` / `ultralytics`，避免污染当前 pytest 进程的
        `logging.getLogger("ultralytics")` 处理器状态及影响其他轻量单元测试。
    """
    import json
    import subprocess

    candidate = Path("build/portable_cache/weights/yolo26s.pt")
    if not candidate.is_file():
        pytest.skip("本地未找到预置 yolo26s.pt 权重，跳过真实权重只读测试")

    src_dir = str((Path(__file__).resolve().parents[1] / "src").resolve())
    script = (
        f"import json, sys; from pathlib import Path; sys.path.insert(0, {src_dir!r}); "
        "from yolo_studio.core.trainer import inspect_weights_metadata; "
        f"m = inspect_weights_metadata(Path({str(candidate.resolve())!r})); "
        "print(json.dumps({'train_imgsz': list(m.train_imgsz) if m.train_imgsz else None, "
        "'num_classes': m.num_classes, 'first': list(m.class_names[0]) if m.class_names else None, "
        "'supports_end2end': m.supports_end2end, 'one2many_available': m.one2many_available}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["train_imgsz"] == [640, 640]
    assert payload["num_classes"] == 80
    assert payload["first"] == [0, "person"]
    assert payload["supports_end2end"] is True
    assert payload["one2many_available"] is True


def test_build_export_kwargs_and_describe_io_formats_and_warnings(tmp_path: Path) -> None:
    """验证各导出格式的参数过滤、FP16/quantize 映射、IO 形状计算及不一致告警。"""
    pt_file = tmp_path / "best.pt"
    pt_file.write_bytes(b"pt")
    meta = WeightsMetadata(
        weights_path=pt_file,
        task="detect",
        train_imgsz=(960, 960),
        class_names=((0, "a"), (1, "b"), (2, "c")),
        strides=(8, 16, 32),
        channels=3,
        supports_end2end=True,
        one2many_available=True,
    )

    # 1) ONNX 未显式传 imgsz 时自动沿用训练尺寸 960，固定 CPU 并忽略 half，支持 opset 与 simplify
    cfg_onnx = ExportConfig(
        weights_path=pt_file,
        export_format="onnx",
        imgsz=None,
        batch=1,
        dynamic=False,
        nms=None,
        opset=17,
        simplify=True,
        half=True,
        device="0",
    )
    kwargs, warns = build_export_kwargs(cfg_onnx, metadata=meta, use_quantize_arg=True)
    assert kwargs["format"] == "onnx"
    assert kwargs["imgsz"] == 960
    assert kwargs["device"] == "cpu"
    assert kwargs["opset"] == 17
    assert kwargs["simplify"] is True
    assert "half" not in kwargs and "quantize" not in kwargs
    assert any("不支持 FP16" in w for w in warns)

    io_raw, io_warns = describe_export_io(cfg_onnx, metadata=meta)
    assert "(1, 3, 960, 960)" in io_raw
    assert "(1, 7, 18900)" in io_raw  # 4 + 3 类，960x960 共 18900 候选框
    assert not io_warns

    # 2) 导出尺寸故意改成 640 时给出与训练尺寸 960 不一致告警；切换 end2end 输出 (B, 300, 6)
    cfg_e2e = ExportConfig(
        weights_path=pt_file,
        export_format="onnx",
        imgsz=640,
        batch=2,
        dynamic=False,
        nms=False,
        max_det=200,
    )
    io_e2e, e2e_warns = describe_export_io(cfg_e2e, metadata=meta)
    assert "(2, 3, 640, 640)" in io_e2e
    assert "(2, 200, 6)" in io_e2e
    assert any("960×960 不一致" in w for w in e2e_warns)

    # 3) TorchScript 自动剔除不支持的 opset/simplify，GPU 开启 FP16 映射为 quantize=16
    cfg_ts = ExportConfig(
        weights_path=pt_file,
        export_format="torchscript",
        imgsz=(544, 960),
        batch=1,
        dynamic=True,
        nms=True,
        max_det=100,
        conf=0.3,
        iou=0.45,
        opset=17,
        simplify=False,
        half=True,
        device="0",
    )
    ts_kwargs, ts_warns = build_export_kwargs(cfg_ts, metadata=meta, use_quantize_arg=True)
    assert ts_kwargs["format"] == "torchscript"
    assert ts_kwargs["imgsz"] == [544, 960]
    assert ts_kwargs["device"] == "0"
    assert ts_kwargs["quantize"] == 16
    assert ts_kwargs["nms"] is True
    assert ts_kwargs["max_det"] == 100
    assert ts_kwargs["conf"] == pytest.approx(0.3)
    assert ts_kwargs["iou"] == pytest.approx(0.45)
    assert "opset" not in ts_kwargs
    assert "simplify" not in ts_kwargs
    assert any("不支持 opset" in w for w in ts_warns)

    # 4) TensorRT 别名规范化、workspace 透传与动态 batch=1 提示
    cfg_trt = ExportConfig(
        weights_path=pt_file,
        export_format="tensorrt",
        imgsz=960,
        batch=1,
        dynamic=True,
        half=True,
        workspace=4.0,
        device="0",
        extra_args={"verbose": True},
    )
    trt_kwargs, _ = build_export_kwargs(cfg_trt, metadata=meta, use_quantize_arg=False)
    assert trt_kwargs["format"] == "engine"
    assert trt_kwargs["half"] is True
    assert trt_kwargs["workspace"] == 4.0
    assert trt_kwargs["verbose"] is True
    _, trt_io_warns = describe_export_io(cfg_trt, metadata=meta)
    assert any("最大批大小" in w for w in trt_io_warns)

    # 5) embedded NMS 且 opset < 11 拦截
    with pytest.raises(ValueError, match="opset >= 11"):
        build_export_kwargs(
            ExportConfig(weights_path=pt_file, export_format="onnx", nms=True, opset=10)
        )


def test_export_model_with_config_and_cli_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """验证 `export_model_with_config` 完整日志/传参流程及 CLI `export` 子命令（含 `--inspect-only` 与错参拦截）。"""
    pt_file = tmp_path / "best.pt"
    pt_file.write_bytes(b"pt")
    meta = WeightsMetadata(
        weights_path=pt_file,
        task="detect",
        train_imgsz=(960, 960),
        class_names=((0, "ui_btn"), (1, "hp_bar")),
        strides=(8, 16, 32),
        channels=3,
        supports_end2end=True,
        one2many_available=True,
    )
    monkeypatch.setattr(trainer_module, "inspect_weights_metadata", lambda _p: meta)
    monkeypatch.setattr(trainer_module, "_ultralytics_supports_quantize_arg", lambda: True)

    captured_calls: list[dict[str, object]] = []

    class _FakeYOLO:
        def __init__(self, source: str) -> None:
            self.source = source

        def export(self, **kwargs: object) -> str:
            captured_calls.append(dict(kwargs))
            out = pt_file.with_suffix(".onnx")
            out.write_bytes(b"onnx")
            return str(out)

    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    # 1) 核心 API 不传 imgsz 自动继承权重 960
    logs: list[str] = []
    exported = export_model(
        pt_file,
        export_format="onnx",
        batch=2,
        dynamic=True,
        nms=False,
        max_det=150,
        opset=17,
        log_callback=logs.append,
    )
    assert exported == pt_file.with_suffix(".onnx")
    assert captured_calls[-1]["imgsz"] == 960
    assert captured_calls[-1]["batch"] == 2
    assert captured_calls[-1]["dynamic"] is True
    assert captured_calls[-1]["nms"] is False
    assert captured_calls[-1]["max_det"] == 150
    assert captured_calls[-1]["opset"] == 17
    joined = "\n".join(logs)
    assert "0: ui_btn, 1: hp_bar" in joined
    assert "已自动沿用权重训练尺寸: 960×960" in joined

    # 2) CLI --inspect-only 仅打印元数据与 IO 形状，不触发实际导出
    before_count = len(captured_calls)
    code_inspect = cli_main(["export", "-w", str(pt_file), "--inspect-only", "--nms", "end2end"])
    assert code_inspect == 0
    assert len(captured_calls) == before_count
    out_text = capsys.readouterr().out
    assert "训练输入尺寸: 960×960" in out_text
    assert "0: ui_btn, 1: hp_bar" in out_text
    assert "(1, 300, 6)" in out_text

    # 3) CLI 完整导出参数解析与执行（矩形 imgsz、opset、no-simplify、extra-arg）
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "export",
            "-w",
            str(pt_file),
            "-f",
            "onnx",
            "--imgsz",
            "544,960",
            "--batch",
            "4",
            "--dynamic",
            "--nms",
            "embedded",
            "--max-det",
            "120",
            "--conf",
            "0.3",
            "--iou",
            "0.45",
            "--opset",
            "18",
            "--no-simplify",
            "--extra-arg",
            "agnostic_nms=true",
        ]
    )
    cli_cfg = _build_cli_export_config(parsed)
    assert isinstance(cli_cfg, ExportConfig)
    assert cli_cfg.imgsz == (544, 960)
    assert cli_cfg.batch == 4
    assert cli_cfg.dynamic is True
    assert cli_cfg.nms is True
    assert cli_cfg.max_det == 120
    assert cli_cfg.conf == pytest.approx(0.3)
    assert cli_cfg.iou == pytest.approx(0.45)
    assert cli_cfg.opset == 18
    assert cli_cfg.simplify is False
    assert cli_cfg.extra_args == {"agnostic_nms": True}

    code_run = cli_main(
        [
            "export",
            "-w",
            str(pt_file),
            "-f",
            "onnx",
            "--imgsz",
            "544x960",
            "--opset",
            "17",
            "--nms",
            "end2end",
        ]
    )
    assert code_run == 0
    assert captured_calls[-1]["imgsz"] == [544, 960]
    assert captured_calls[-1]["nms"] is False
    assert captured_calls[-1]["opset"] == 17

    # 4) CLI 非法参数返回退出码 2；export_model_with_config 对缺失权重抛出 FileNotFoundError
    assert cli_main(["export", "-w", str(pt_file), "--imgsz", "not_a_size"]) == 2
    assert cli_main(["export", "-w", str(pt_file), "--nms", "embedded", "--opset", "9"]) == 2
    assert cli_main(["export", "-w", str(pt_file), "--extra-arg", "broken_token"]) == 2
    with pytest.raises(FileNotFoundError):
        export_model_with_config(ExportConfig(weights_path=tmp_path / "missing.pt"))


def test_eval_tab_auto_reads_weights_syncs_imgsz_and_exports_independent_config(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证第 6 页自动探测权重元数据、同步训练尺寸、忽略过期探测结果并按独立导出参数启动 `ExportWorker`。"""
    pt_a = tmp_path / "model_960.pt"
    pt_b = tmp_path / "model_rect.pt"
    pt_a.write_bytes(b"pt-a")
    pt_b.write_bytes(b"pt-b")

    inspect_workers: list[_StubWeightsInspectWorker] = []
    export_workers: list[_StubExportWorker] = []

    def _inspect_factory(path: Path, request_id: int = 0) -> _StubWeightsInspectWorker:
        w = _StubWeightsInspectWorker(path, request_id=request_id)
        inspect_workers.append(w)
        return w

    def _export_factory(cfg: ExportConfig) -> _StubExportWorker:
        w = _StubExportWorker(cfg)
        export_workers.append(w)
        return w

    monkeypatch.setattr(tab_eval_module, "WeightsInspectWorker", _inspect_factory)
    monkeypatch.setattr(tab_eval_module, "ExportWorker", _export_factory)

    warnings_shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda _p, _t, msg: warnings_shown.append(str(msg))),
    )

    tab = EvalTab(workspace_root=tmp_path, device_info=_make_device_info(has_gpu=True))
    try:
        # 1) 传入权重 A（训练尺寸 960x960，3 个类别）：应立即触发探测并同步验证与导出尺寸
        tab.set_weights_and_yaml(str(pt_a), "")
        qt_app.processEvents()
        assert len(inspect_workers) == 1
        meta_a = WeightsMetadata(
            weights_path=pt_a,
            task="detect",
            train_imgsz=(960, 960),
            class_names=((0, "btn"), (1, "enemy"), (2, "npc")),
            supports_end2end=True,
            one2many_available=True,
        )
        inspect_workers[0].complete_with(meta_a)
        qt_app.processEvents()

        assert tab.imgsz_spin.value() == 960
        assert tab.export_h_spin.value() == 960
        assert tab.export_w_spin.value() == 960
        assert tab.export_square_chk.isChecked() is True
        assert tab.export_w_spin.isEnabled() is False
        assert tab.view_classes_btn.isEnabled() is True
        assert tab.sync_train_imgsz_btn.isEnabled() is True
        assert "960×960" in tab.weights_info_label.text()
        assert "0: btn, 1: enemy, 2: npc" in tab.weights_info_label.text()
        assert "(1, 3, 960, 960)" in tab.export_io_label.text()
        assert "(1, 7, 18900)" in tab.export_io_label.text()

        # 2) 用户独立调整导出尺寸为 640：验证 ImgSz 保持 960 不受影响，IO 预览立即弹出不一致提示；点击「用训练尺寸」恢复
        tab.export_h_spin.setValue(640)
        qt_app.processEvents()
        assert tab.imgsz_spin.value() == 960
        assert tab.export_w_spin.value() == 640
        assert "960×960 不一致" in tab.export_io_label.text()

        tab.sync_train_imgsz_btn.click()
        qt_app.processEvents()
        assert tab.export_h_spin.value() == 960
        assert tab.export_w_spin.value() == 960
        assert "不一致" not in tab.export_io_label.text()

        # 3) 切换格式与 NMS 模式防呆联动，并取消方形联动设置矩形导出尺寸 544x960
        tab.export_square_chk.setChecked(False)
        tab.export_h_spin.setValue(544)
        tab.export_w_spin.setValue(960)
        tab.export_batch_spin.setValue(2)
        tab.dynamic_chk.setChecked(True)
        opset_17_idx = tab.export_opset_combo.findData(17)
        assert opset_17_idx >= 0
        tab.export_opset_combo.setCurrentIndex(opset_17_idx)
        tab.simplify_chk.setChecked(False)
        tab.nms_mode_combo.setCurrentIndex(1)  # end2end
        tab.export_max_det_spin.setValue(200)
        qt_app.processEvents()

        assert tab.export_max_det_spin.isEnabled() is True
        assert "(batch, 200, 6)" in tab.export_io_label.text()

        # 切到 TorchScript 时 opset/simplify 自动置灰，切回 ONNX 恢复可用
        tab.format_combo.setCurrentIndex(1)  # torchscript
        qt_app.processEvents()
        assert tab.export_opset_combo.isEnabled() is False
        assert tab.simplify_chk.isEnabled() is False
        assert tab.half_chk.isEnabled() is True
        tab.format_combo.setCurrentIndex(0)  # onnx
        qt_app.processEvents()
        assert tab.export_opset_combo.isEnabled() is True
        assert tab.simplify_chk.isEnabled() is True
        assert tab.half_chk.isEnabled() is False

        # 点击导出按钮：校验传入 ExportWorker 的 ExportConfig 包含所有独立参数
        tab.export_btn.click()
        qt_app.processEvents()
        assert len(export_workers) == 1
        sent_cfg = export_workers[0].config
        assert sent_cfg.weights_path == pt_a
        assert sent_cfg.export_format == "onnx"
        assert sent_cfg.imgsz == (544, 960)
        assert sent_cfg.batch == 2
        assert sent_cfg.dynamic is True
        assert sent_cfg.nms is False
        assert sent_cfg.max_det == 200
        assert sent_cfg.opset == 17
        assert sent_cfg.simplify is False
        assert sent_cfg.half is False
        export_workers[0].running = False
        tab._set_busy(False)

        # 4) 过期探测结果竞态防护：连续切换权重 A -> B，先回来的 A 旧回调不得覆盖 B 的矩形训练尺寸
        tab.weights_picker.setText(str(pt_a))
        tab._trigger_weights_inspect()  # 同文件缓存命中不新开；换用新文件 B 测试
        tab.set_weights_and_yaml(str(pt_b), "")
        qt_app.processEvents()
        worker_b = inspect_workers[-1]
        assert worker_b.weights_path == pt_b

        # 再人为发一次 A 的切换并切回 B，构造一前一后两个在途请求
        tab.set_weights_and_yaml(str(pt_a), "")
        stale_worker = inspect_workers[-1]
        tab.set_weights_and_yaml(str(pt_b), "")
        latest_worker = inspect_workers[-1]

        meta_b = WeightsMetadata(
            weights_path=pt_b,
            task="detect",
            train_imgsz=(544, 960),
            class_names=((0, "only_cls"),),
            supports_end2end=False,
        )
        latest_worker.complete_with(meta_b)
        stale_worker.complete_with(meta_a)  # 过时结果后到达，必须被忽略
        qt_app.processEvents()

        assert tab.current_weights_metadata() == meta_b
        assert tab.export_square_chk.isChecked() is False
        assert tab.export_h_spin.value() == 544
        assert tab.export_w_spin.value() == 960
        assert "0: only_cls" in tab.weights_info_label.text()

        # 5) TensorRT 在 CPU 模式下自动置灰「导出模型」按钮，直接触发导出防御逻辑也应弹出清晰提示拦截
        tab.format_combo.setCurrentIndex(2)  # engine
        tab.device_ctrl.gpu_chk.setChecked(False)
        qt_app.processEvents()
        assert tab.export_btn.isEnabled() is False
        assert "GPU" in tab.export_btn.toolTip()
        # 忙碌态结束后仍应保持禁用，不会被误恢复
        tab._set_busy(True)
        tab._set_busy(False)
        assert tab.export_btn.isEnabled() is False

        before_export_cnt = len(export_workers)
        tab._on_export()
        qt_app.processEvents()
        assert len(export_workers) == before_export_cnt
        assert any("TensorRT" in msg for msg in warnings_shown)

        # 重新启用 GPU 后导出按钮恢复可用
        tab.device_ctrl.gpu_chk.setChecked(True)
        qt_app.processEvents()
        assert tab.export_btn.isEnabled() is True
    finally:
        tab.close()
