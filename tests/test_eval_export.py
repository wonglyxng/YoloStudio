"""权重/导出模型元数据探测、验证与推理参数对齐、导出配置/IO 预览、CLI `export` 子命令与第 6 页（`EvalTab`）单元测试。

职责：
    1. 验证 `.pt` 权重与已导出部署模型（`.onnx` / `.torchscript` / `.engine`）元数据只读解析
       （训练/导出尺寸方形/矩形、有序类别清单、动态属性、检测头分支/固化模式及异常兜底）；
    2. 验证 `build_predict_kwargs` / `build_val_kwargs`、`predict_source` / `validate_model`
       对检测头（`one2many` / `end2end`）、输入填充（`rect=True/False`）、紧凑防重叠半透明绘制及已导出模型的传参与日志输出；
    3. 验证 `ExportConfig` 参数规范化、`build_export_kwargs` 按格式过滤（ONNX / TorchScript / TensorRT）、
       `describe_export_io` 输入输出张量估算与风险告警，以及 `export_model_with_config` 端到端行为；
    4. 验证 CLI `export` 子命令默认沿用权重训练尺寸、`--inspect-only`、全量导出选项与非法参数退出码 `2`；
    5. 在离线 `offscreen` Qt 环境下验证 `EvalTab` 自动探测元数据并同步尺寸、推理检测头/填充与导出对齐、
       预览样式控件与原图放大查看、直接加载 `.onnx` 实测联动、一键切回 `.pt` 保留自定义导出尺寸、过期异步请求隔离及 Worker 传参。

运行环境与协作边界：
    通过 `pytest` 离线运行，依赖 `PyQt6`；主要测试使用模拟 checkpoint/元数据、模拟 `YOLO` 及轻量假 Worker，
    不访问外部网络；若本地存在预置 `yolo26s.pt` / `yolo26n.pt` 权重，额外在隔离子进程中执行真实只读/临时导出实测。
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QObject, Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import QApplication, QMessageBox

import yolo_studio.core.trainer as trainer_module
import yolo_studio.gui.tab_eval as tab_eval_module
from yolo_studio.cli import _build_cli_export_config, build_parser, main as cli_main
from yolo_studio.core.trainer import (
    ONNX_DEPLOY_META_JSON_KEY,
    ONNX_DEPLOY_SCHEMA_VERSION,
    DeviceInfo,
    ExportConfig,
    PredictRenderConfig,
    WeightsMetadata,
    _layout_non_overlapping_labels,
    _rect_intersect_area,
    align_imgsz_to_stride,
    build_export_kwargs,
    build_onnx_deploy_metadata,
    build_predict_kwargs,
    build_val_kwargs,
    describe_export_io,
    export_model,
    export_model_with_config,
    format_class_order,
    format_predict_render_summary,
    format_weights_metadata_log,
    inspect_weights_metadata,
    normalize_export_imgsz,
    normalize_padding_value,
    normalize_predict_render_config,
    parse_export_nms_mode,
    parse_export_preprocess_mode,
    parse_predict_line_width_mode,
    parse_predict_render_style,
    predict_source,
    render_predict_result,
    validate_model,
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
            # 提供真实可读的 ONNX 容器，覆盖提交参数到部署元数据的写回。
            import onnx

            onnx.save(onnx.ModelProto(), str(out))
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

    with pytest.raises(ValueError, match="写入失败"):
        with monkeypatch.context() as scoped:

            def reject_write(*args, **kwargs):
                raise ValueError("写入失败")

            scoped.setattr(trainer_module, "write_onnx_deploy_metadata", reject_write)
            export_model(pt_file, export_format="onnx")

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


def test_build_predict_and_val_kwargs_pt_and_exported_models(tmp_path: Path) -> None:
    """验证 `build_predict_kwargs` / `build_val_kwargs` 对 `.pt`、静态/动态 `.onnx` 及 `.engine` 的参数组装与拦截。"""
    pt_file = tmp_path / "best.pt"
    onnx_file = tmp_path / "best.onnx"
    engine_file = tmp_path / "best.engine"
    src_img = tmp_path / "sample.jpg"
    data_yaml = tmp_path / "data.yaml"
    for p in (pt_file, onnx_file, engine_file, src_img, data_yaml):
        p.write_bytes(b"stub")

    pt_dual = WeightsMetadata(
        weights_path=pt_file,
        train_imgsz=(960, 960),
        supports_end2end=True,
        one2many_available=True,
    )
    pt_raw_only = WeightsMetadata(
        weights_path=pt_file,
        train_imgsz=(640, 640),
        supports_end2end=False,
        one2many_available=True,
    )
    onnx_static_e2e = WeightsMetadata(
        weights_path=onnx_file,
        train_imgsz=(576, 960),
        supports_end2end=True,
        one2many_available=False,
        model_format="onnx",
        is_exported=True,
        export_nms_mode="end2end",
        export_dynamic=False,
        export_batch=1,
    )
    onnx_dyn_emb = WeightsMetadata(
        weights_path=onnx_file,
        train_imgsz=(960, 960),
        supports_end2end=False,
        one2many_available=True,
        model_format="onnx",
        is_exported=True,
        export_nms_mode="embedded",
        export_dynamic=True,
        export_batch=1,
    )

    # 1) PT 默认 one2many + rect=True；切换 end2end + 矩形 + rect=False
    kw_pt_def, w_pt_def = build_predict_kwargs(
        pt_file, src_img, imgsz=960, rect=True, nms=None, metadata=pt_dual
    )
    assert kw_pt_def["imgsz"] == 960
    assert kw_pt_def["rect"] is True
    assert "nms" not in kw_pt_def
    assert not w_pt_def

    kw_pt_e2e, w_pt_e2e = build_predict_kwargs(
        pt_file,
        src_img,
        conf=0.6,
        imgsz=(576, 960),
        rect=False,
        nms=False,
        max_det=150,
        metadata=pt_dual,
    )
    assert kw_pt_e2e["imgsz"] == [576, 960]
    assert kw_pt_e2e["rect"] is False
    assert kw_pt_e2e["nms"] is False
    assert kw_pt_e2e["max_det"] == 150
    assert kw_pt_e2e["conf"] == pytest.approx(0.6)
    assert not w_pt_e2e

    # 单分支 PT 选 end2end 给出回退提示
    _, w_fallback = build_predict_kwargs(pt_file, src_img, nms=False, metadata=pt_raw_only)
    assert any("one2one" in w for w in w_fallback)

    # 2) 静态 ONNX 自动对齐导出尺寸 [576, 960] 并强制 rect=False，且忽略额外 nms 覆盖
    kw_onnx_st, w_onnx_st = build_predict_kwargs(
        onnx_file, src_img, imgsz=640, rect=True, nms=False, metadata=onnx_static_e2e
    )
    assert kw_onnx_st["imgsz"] == [576, 960]
    assert kw_onnx_st["rect"] is False
    assert "nms" not in kw_onnx_st
    assert any("固定使用导出尺寸" in w for w in w_onnx_st)
    assert any("忽略额外指定" in w for w in w_onnx_st)

    # 3) 旧动态 ONNX 保留传入尺寸与 rect，新部署声明则决定参考画布和步长填充。
    kw_onnx_dyn, w_onnx_dyn = build_predict_kwargs(
        onnx_file, src_img, imgsz=800, rect=True, metadata=onnx_dyn_emb
    )
    assert kw_onnx_dyn["imgsz"] == 800
    assert kw_onnx_dyn["rect"] is True
    assert any("图内嵌入 NMS" in w for w in w_onnx_dyn)
    declared = replace(
        onnx_dyn_emb, deploy_schema=ONNX_DEPLOY_SCHEMA_VERSION, train_imgsz=(576, 960), preprocess_auto_rect=True
    )
    declared_kwargs, _ = build_predict_kwargs(onnx_file, src_img, imgsz=640, rect=False, metadata=declared)
    assert declared_kwargs["imgsz"] == [576, 960]
    assert declared_kwargs["rect"] is True

    # 4) 验证参数：矩形尺寸自动关闭 rect=False；TensorRT CPU 下推理/验证均拦截
    kw_val_rect, w_val_rect = build_val_kwargs(
        pt_file, data_yaml, imgsz=(544, 960), nms=False, rect=True, metadata=pt_dual
    )
    assert kw_val_rect["imgsz"] == [544, 960]
    assert kw_val_rect["nms"] is False
    assert kw_val_rect["rect"] is False
    assert any("rect=False" in w for w in w_val_rect)

    with pytest.raises(ValueError, match="CUDA GPU"):
        build_predict_kwargs(engine_file, src_img, device="cpu")
    with pytest.raises(ValueError, match="CUDA GPU"):
        build_val_kwargs(engine_file, data_yaml, device="cpu")


def test_predict_source_validate_and_exported_metadata_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 `predict_source` / `validate_model` 完整透传检测头与填充参数、输出日志及导出模型元数据解析。"""
    import numpy as np

    pt_file = tmp_path / "best.pt"
    onnx_file = tmp_path / "best.onnx"
    empty_onnx = tmp_path / "empty.onnx"
    img_file = tmp_path / "frame.jpg"
    data_yaml = tmp_path / "data.yaml"
    pt_file.write_bytes(b"pt")
    onnx_file.write_bytes(b"onnx-stub")
    empty_onnx.write_bytes(b"")
    img_file.write_bytes(b"jpg")
    data_yaml.write_text("names: [boss]\n", encoding="utf-8")

    # 1) 模拟 BaseBackend.read_metadata 校验 .onnx 元数据（end2end / embedded / raw 动态）解析
    meta_cases = [
        (
            {
                "task": "detect",
                "imgsz": "[576, 960]",
                "stride": "32",
                "batch": "1",
                "channels": "3",
                "names": "{0: 'tree', 1: 'player'}",
                "end2end": "True",
                "args": "{'nms': False, 'dynamic': False, 'batch': 1}",
            },
            "end2end",
            False,
            (576, 960),
        ),
        (
            {
                "task": "detect",
                "imgsz": "[960, 960]",
                "stride": "32",
                "batch": "2",
                "channels": "3",
                "names": "{0: 'tree'}",
                "end2end": "False",
                "args": "{'nms': True, 'dynamic': False, 'batch': 2}",
            },
            "embedded",
            False,
            (960, 960),
        ),
        (
            {
                "task": "detect",
                "imgsz": "[640, 640]",
                "stride": "32",
                "batch": "1",
                "channels": "3",
                "names": "{0: 'tree'}",
                "end2end": "False",
                "args": "{'nms': None, 'dynamic': True, 'batch': 1}",
            },
            "raw",
            True,
            (640, 640),
        ),
    ]

    fake_base_mod = types.ModuleType("ultralytics.nn.backends.base")
    fake_backends_mod = types.ModuleType("ultralytics.nn.backends")
    fake_backends_mod.base = fake_base_mod  # type: ignore[attr-defined]
    fake_nn_mod = types.ModuleType("ultralytics.nn")
    fake_nn_mod.backends = fake_backends_mod  # type: ignore[attr-defined]
    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.nn = fake_nn_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setitem(sys.modules, "ultralytics.nn", fake_nn_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.backends", fake_backends_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.backends.base", fake_base_mod)

    for raw_meta, exp_mode, exp_dyn, exp_sz in meta_cases:
        fake_base_mod.BaseBackend = types.SimpleNamespace(read_metadata=staticmethod(lambda _p, m=raw_meta: dict(m)))  # type: ignore[attr-defined]
        parsed = inspect_weights_metadata(onnx_file)
        assert parsed.is_exported is True
        assert parsed.model_format == "onnx"
        assert parsed.export_nms_mode == exp_mode
        assert parsed.export_dynamic is exp_dyn
        assert parsed.train_imgsz == exp_sz
        log_text = format_weights_metadata_log(parsed)
        assert "导出模型元数据" in log_text

    with pytest.raises(ValueError, match="为空"):
        inspect_weights_metadata(empty_onnx)

    # export_model_with_config 拒绝将已导出 .onnx 作为源权重再导出
    with pytest.raises(ValueError, match="仅支持 PyTorch"):
        export_model_with_config(ExportConfig(weights_path=onnx_file))

    # 2) 验证 predict_source 与 validate_model 透传参数及日志块输出
    captured_pred: list[dict[str, object]] = []
    captured_val: list[dict[str, object]] = []

    class _FakeResult:
        def __init__(self, path: str) -> None:
            self.path = path

        def plot(self) -> np.ndarray:
            return np.zeros((32, 32, 3), dtype=np.uint8)

    class _FakeYOLO:
        def __init__(self, source: str) -> None:
            self.source = source

        def predict(self, **kwargs: object) -> list[_FakeResult]:
            captured_pred.append(dict(kwargs))
            return [_FakeResult(str(img_file))]

        def val(self, **kwargs: object) -> object:
            captured_val.append(dict(kwargs))
            return types.SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.91})

    fake_ultra.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    monkeypatch.setattr(
        trainer_module,
        "inspect_weights_metadata",
        lambda p: WeightsMetadata(
            weights_path=p,
            train_imgsz=(960, 960),
            supports_end2end=True,
            one2many_available=True,
        ),
    )

    pred_logs: list[str] = []
    outputs = predict_source(
        pt_file,
        img_file,
        tmp_path / "pred_out",
        conf=0.60,
        iou=0.45,
        imgsz=(576, 960),
        device="cpu",
        nms=False,
        rect=False,
        max_det=180,
        log_callback=pred_logs.append,
    )
    assert len(outputs) == 1 and outputs[0].is_file()
    assert captured_pred[-1]["imgsz"] == [576, 960]
    assert captured_pred[-1]["rect"] is False
    assert captured_pred[-1]["nms"] is False
    assert captured_pred[-1]["max_det"] == 180
    assert any("推理效果测试配置" in line for line in pred_logs)
    assert any("one2one" in line for line in pred_logs)

    val_logs: list[str] = []
    metrics = validate_model(
        pt_file,
        data_yaml,
        imgsz=960,
        device="cpu",
        nms=False,
        log_callback=val_logs.append,
    )
    assert metrics["metrics/mAP50(B)"] == pytest.approx(0.91)
    assert captured_val[-1]["nms"] is False
    assert captured_val[-1]["imgsz"] == 960
    assert any("模型验证配置" in line for line in val_logs)


def test_real_exported_onnx_metadata_and_predict_if_present(tmp_path: Path) -> None:
    """若本地存在预置 `yolo26n.pt`，在独立子进程中导出微型 ONNX 并实测元数据只读解析与 ONNX 推理。"""
    import json
    import subprocess

    candidate = Path("build/portable_cache/weights/yolo26n.pt")
    if not candidate.is_file():
        pytest.skip("本地未找到预置 yolo26n.pt 权重，跳过真实 ONNX 导出与推理实测")

    src_dir = str((Path(__file__).resolve().parents[1] / "src").resolve())
    script = (
        f"import json, shutil, sys; from pathlib import Path; import cv2, numpy as np; "
        f"sys.path.insert(0, {src_dir!r}); "
        "from yolo_studio.core.trainer import export_model, inspect_weights_metadata, predict_source; "
        f"work = Path({str(tmp_path.resolve())!r}); "
        f"pt = work / 'yolo26n.pt'; shutil.copy({str(candidate.resolve())!r}, pt); "
        "onnx_p = export_model(pt, export_format='onnx', imgsz=(192, 320), nms=False, simplify=False, device='cpu'); "
        "meta = inspect_weights_metadata(onnx_p); "
        "img_p = work / 'blank.jpg'; cv2.imwrite(str(img_p), np.full((180, 320, 3), 120, dtype=np.uint8)); "
        "outs = predict_source(onnx_p, img_p, work / 'out', conf=0.25, imgsz=640, device='cpu'); "
        "print(json.dumps({'is_exported': meta.is_exported, 'fmt': meta.model_format, "
        "'imgsz': list(meta.train_imgsz) if meta.train_imgsz else None, "
        "'nms_mode': meta.export_nms_mode, 'dynamic': meta.export_dynamic, "
        "'num_classes': meta.num_classes, 'out_count': len(outs)}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["is_exported"] is True
    assert payload["fmt"] == "onnx"
    assert payload["imgsz"] == [192, 320]
    assert payload["nms_mode"] == "end2end"
    assert payload["dynamic"] is False
    assert payload["num_classes"] == 80
    assert payload["out_count"] == 1


def test_eval_tab_predict_head_padding_sync_export_and_onnx_switching(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证第 6 页检测头/填充模式联动、对齐导出参数、实测导出 ONNX 及一键切回 `.pt` 保留导出尺寸。"""
    pt_file = tmp_path / "best.pt"
    onnx_file = tmp_path / "best.onnx"
    engine_file = tmp_path / "best.engine"
    img_file = tmp_path / "demo.jpg"
    yaml_file = tmp_path / "data.yaml"
    for p in (pt_file, onnx_file, engine_file, img_file, yaml_file):
        p.write_bytes(b"stub")

    inspect_workers: list[_StubWeightsInspectWorker] = []
    pred_workers: list[object] = []
    val_workers: list[object] = []

    class _StubPredictWorker(QObject):
        log_line = Signal(str)
        finished_outputs = Signal(object)
        failed = Signal(str)

        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = dict(kwargs)
            self.running = False

        def start(self) -> None:
            self.running = True

        def isRunning(self) -> bool:
            return self.running

    class _StubValidateWorker(QObject):
        log_line = Signal(str)
        finished_metrics = Signal(object)
        failed = Signal(str)

        def __init__(
            self,
            weights_path: Path,
            data_yaml: Path,
            imgsz: int | tuple[int, int],
            device: str,
            *,
            nms: bool | None = None,
            rect: bool | None = None,
        ) -> None:
            super().__init__()
            self.weights_path = weights_path
            self.data_yaml = data_yaml
            self.imgsz = imgsz
            self.device = device
            self.nms = nms
            self.rect = rect
            self.running = False

        def start(self) -> None:
            self.running = True

        def isRunning(self) -> bool:
            return self.running

    def _inspect_factory(path: Path, request_id: int = 0) -> _StubWeightsInspectWorker:
        w = _StubWeightsInspectWorker(path, request_id=request_id)
        inspect_workers.append(w)
        return w

    def _pred_factory(**kwargs: object) -> _StubPredictWorker:
        w = _StubPredictWorker(**kwargs)
        pred_workers.append(w)
        return w

    def _val_factory(*args: object, **kwargs: object) -> _StubValidateWorker:
        w = _StubValidateWorker(*args, **kwargs)  # type: ignore[arg-type]
        val_workers.append(w)
        return w

    monkeypatch.setattr(tab_eval_module, "WeightsInspectWorker", _inspect_factory)
    monkeypatch.setattr(tab_eval_module, "PredictWorker", _pred_factory)
    monkeypatch.setattr(tab_eval_module, "ValidateWorker", _val_factory)

    warnings_shown: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda _p, _t, msg: warnings_shown.append(str(msg))),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *_a: None))

    tab = EvalTab(workspace_root=tmp_path, device_info=_make_device_info(has_gpu=True))
    try:
        # 1) 加载双头 .pt（960x960）
        tab.set_weights_and_yaml(str(pt_file), str(yaml_file))
        tab.source_edit.setText(str(img_file))
        qt_app.processEvents()
        meta_pt = WeightsMetadata(
            weights_path=pt_file,
            task="detect",
            train_imgsz=(960, 960),
            class_names=((0, "monster"), (1, "boss")),
            supports_end2end=True,
            one2many_available=True,
        )
        inspect_workers[-1].complete_with(meta_pt)
        qt_app.processEvents()

        assert tab.eval_head_combo.isEnabled() is True
        assert tab.iou_spin.isEnabled() is True
        # 默认 PT 预览是 rect=True，下方导出是静态 960x960 -> 摘要应给出填充方式差异提示
        assert "存在差异" in tab.pred_summary_label.text()

        # 2) 下方导出设为 end2end + 矩形 576x960 + MaxDet=180，点击「对齐导出参数」
        tab.export_square_chk.setChecked(False)
        tab.export_h_spin.setValue(576)
        tab.export_w_spin.setValue(960)
        tab.nms_mode_combo.setCurrentIndex(1)  # end2end
        tab.export_max_det_spin.setValue(180)
        qt_app.processEvents()

        assert "检测头" in tab.pred_summary_label.text()
        assert "输入尺寸" in tab.pred_summary_label.text()

        tab.sync_export_pred_btn.click()
        qt_app.processEvents()

        assert tab.eval_head_combo.currentData() == "end2end"
        assert tab.pred_pad_combo.currentData() == "export"
        assert tab.iou_spin.isEnabled() is False
        assert "存在差异" not in tab.pred_summary_label.text()

        # 3) 运行推理预览与验证：验证 Worker 精确拿到对齐后的矩形尺寸、rect=False、nms=False、max_det=180
        tab.predict_btn.click()
        qt_app.processEvents()
        assert len(pred_workers) == 1
        sent_pred = pred_workers[-1].kwargs  # type: ignore[attr-defined]
        assert sent_pred["imgsz"] == (576, 960)
        assert sent_pred["rect"] is False
        assert sent_pred["nms"] is False
        assert sent_pred["max_det"] == 180
        pred_workers[-1].running = False  # type: ignore[attr-defined]
        tab._set_busy(False)

        tab.val_btn.click()
        qt_app.processEvents()
        assert len(val_workers) == 1
        assert val_workers[-1].nms is False  # type: ignore[attr-defined]
        val_workers[-1].running = False  # type: ignore[attr-defined]
        tab._set_busy(False)

        # 4) 模拟导出 ONNX 成功 -> 出现「测刚导出模型」快捷按钮 -> 点击切换至 ONNX
        tab._on_export_done(str(onnx_file))
        qt_app.processEvents()
        assert tab.switch_model_btn.isVisibleTo(tab) is True
        assert tab.switch_model_btn.text() == "测刚导出模型"

        tab.switch_model_btn.click()
        qt_app.processEvents()
        assert Path(tab.weights_picker.text()) == onnx_file
        meta_onnx = WeightsMetadata(
            weights_path=onnx_file,
            task="detect",
            train_imgsz=(576, 960),
            class_names=((0, "monster"), (1, "boss")),
            supports_end2end=True,
            one2many_available=False,
            model_format="onnx",
            is_exported=True,
            export_nms_mode="end2end",
            export_dynamic=False,
            export_batch=1,
        )
        inspect_workers[-1].complete_with(meta_onnx)
        qt_app.processEvents()

        # 加载静态 end2end ONNX 后：检测头、ImgSz、填充、IoU、对齐按钮、导出按钮全部锁定防呆
        assert "ONNX 导出模型" in tab.weights_info_label.text()
        assert tab.eval_head_combo.isEnabled() is False
        assert tab.imgsz_spin.isEnabled() is False
        assert tab.pred_pad_combo.isEnabled() is False
        assert tab.iou_spin.isEnabled() is False
        assert tab.sync_export_pred_btn.isEnabled() is False
        assert tab.export_btn.isEnabled() is False
        assert tab.switch_model_btn.text() == "切回 .pt 权重"

        # 对静态 ONNX 点击推理：传给 PredictWorker 的是 onnx 路径、固定尺寸 (576, 960)、rect=False、nms=None
        tab.predict_btn.click()
        qt_app.processEvents()
        sent_onnx = pred_workers[-1].kwargs  # type: ignore[attr-defined]
        assert sent_onnx["weights_path"] == onnx_file
        assert sent_onnx["imgsz"] == (576, 960)
        assert sent_onnx["rect"] is False
        assert sent_onnx["nms"] is None
        pred_workers[-1].running = False  # type: ignore[attr-defined]
        tab._set_busy(False)

        # 5) 点击「切回 .pt 权重」：回到 best.pt，保留用户刚才调好的 576x960 自定义导出尺寸，且恢复导出按钮
        tab.switch_model_btn.click()
        qt_app.processEvents()
        assert Path(tab.weights_picker.text()) == pt_file
        inspect_workers[-1].complete_with(meta_pt)
        qt_app.processEvents()

        assert tab.export_square_chk.isChecked() is False
        assert tab.export_h_spin.value() == 576
        assert tab.export_w_spin.value() == 960
        assert tab.export_btn.isEnabled() is True
        assert tab.eval_head_combo.isEnabled() is True
        assert tab.switch_model_btn.text() == "测刚导出模型"

        # 6) 导出设为 embedded (conf=0.35, iou=0.55) 并对齐：检测头回切 one2many，且同步 Conf/IoU
        tab.nms_mode_combo.setCurrentIndex(2)  # embedded
        tab.export_nms_conf_spin.setValue(0.35)
        tab.export_nms_iou_spin.setValue(0.55)
        qt_app.processEvents()
        tab.sync_export_pred_btn.click()
        qt_app.processEvents()

        assert tab.eval_head_combo.currentData() == "one2many"
        assert tab.iou_spin.isEnabled() is True
        assert tab.conf_spin.value() == pytest.approx(0.35)
        assert tab.iou_spin.value() == pytest.approx(0.55)

        # 7) 加载 .engine 模型并切至 CPU：验证与推理按钮均置灰防误点，直接调用也弹窗拦截
        tab.set_weights_and_yaml(str(engine_file), str(yaml_file))
        qt_app.processEvents()
        inspect_workers[-1].complete_with(
            WeightsMetadata(
                weights_path=engine_file,
                train_imgsz=(960, 960),
                model_format="engine",
                is_exported=True,
                export_nms_mode="raw",
                export_dynamic=False,
            )
        )
        tab.device_ctrl.gpu_chk.setChecked(False)
        qt_app.processEvents()
        assert tab.val_btn.isEnabled() is False
        assert tab.predict_btn.isEnabled() is False
        assert tab.export_btn.isEnabled() is False

        before_pred = len(pred_workers)
        tab._on_predict()
        assert len(pred_workers) == before_pred
        assert any("TensorRT" in msg for msg in warnings_shown)
    finally:
        tab.close()


def test_onnx_deploy_metadata_builder_cli_and_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 ONNX 部署预处理元数据构造、参数校验、CLI 选项解析与只读元数据还原。"""
    import json

    pt_file = tmp_path / "best.pt"
    onnx_file = tmp_path / "best.onnx"
    pt_file.write_bytes(b"pt")
    onnx_file.write_bytes(b"onnx")

    # 1) 预处理模式与补边灰度校验
    assert parse_export_preprocess_mode(None) == "letterbox_center"
    assert parse_export_preprocess_mode("top_left") == "letterbox_top_left"
    assert parse_export_preprocess_mode("scale_fill") == "stretch"
    with pytest.raises(ValueError, match="无效的预处理策略"):
        parse_export_preprocess_mode("invalid_mode")

    assert normalize_padding_value(0) == 0
    assert normalize_padding_value(255) == 255
    with pytest.raises(ValueError, match="0 ~ 255"):
        normalize_padding_value(256)
    with pytest.raises(ValueError, match="整数"):
        normalize_padding_value(True)  # type: ignore[arg-type]

    meta_pt = WeightsMetadata(
        weights_path=pt_file,
        task="detect",
        train_imgsz=(960, 960),
        class_names=((0, "stump"), (1, "boss")),
        strides=(8, 16, 32),
        channels=3,
        supports_end2end=True,
        one2many_available=True,
    )

    # 2) 左上靠边 + 黑边 0 + 矩形 576x960 + end2end
    cfg_tl = ExportConfig(
        weights_path=pt_file,
        export_format="onnx",
        imgsz=(576, 960),
        batch=1,
        dynamic=False,
        nms=False,
        max_det=200,
        preprocess_mode="letterbox_top_left",
        padding_value=0,
    )
    payload_tl, props_tl = build_onnx_deploy_metadata(cfg_tl, metadata=meta_pt)
    assert payload_tl["schema"] == ONNX_DEPLOY_SCHEMA_VERSION
    assert payload_tl["classes"] == ["stump", "boss"]
    assert payload_tl["input"]["height"] == 576 and payload_tl["input"]["width"] == 960  # type: ignore[index]
    assert payload_tl["preprocess"]["resize_mode"] == "letterbox"  # type: ignore[index]
    assert payload_tl["preprocess"]["padding_align"] == "top_left"  # type: ignore[index]
    assert payload_tl["preprocess"]["padding_value"] == 0  # type: ignore[index]
    assert payload_tl["preprocess"]["auto_rect"] is False  # type: ignore[index]
    assert payload_tl["postprocess"]["output_mode"] == "end2end"  # type: ignore[index]
    assert payload_tl["postprocess"]["nms_required"] is False  # type: ignore[index]
    assert payload_tl["postprocess"]["box_format"] == "xyxy"  # type: ignore[index]
    assert props_tl["preprocess_padding_align"] == "top_left"
    assert props_tl["preprocess_padding_value"] == "0"
    assert json.loads(props_tl[ONNX_DEPLOY_META_JSON_KEY]) == payload_tl

    io_tl, warns_tl = describe_export_io(cfg_tl, metadata=meta_pt)
    assert "左上靠边 LetterBox (补边=0)" in io_tl
    assert any("偏离 YOLO 默认" in w for w in warns_tl)

    # 3) 直接拉伸 + 动态 + embedded NMS
    cfg_st = ExportConfig(
        weights_path=pt_file,
        export_format="onnx",
        imgsz=640,
        dynamic=True,
        nms=True,
        conf=0.30,
        iou=0.50,
        preprocess_mode="stretch",
        padding_value=114,
    )
    payload_st, props_st = build_onnx_deploy_metadata(cfg_st, metadata=meta_pt)
    assert payload_st["preprocess"]["resize_mode"] == "stretch"  # type: ignore[index]
    assert payload_st["preprocess"]["padding_align"] == "none"  # type: ignore[index]
    assert payload_st["preprocess"]["padding_value"] is None  # type: ignore[index]
    assert payload_st["preprocess"]["auto_rect"] is False  # type: ignore[index]
    assert payload_st["postprocess"]["output_mode"] == "embedded"  # type: ignore[index]
    assert payload_st["postprocess"]["embedded_conf"] == pytest.approx(0.30)  # type: ignore[index]
    assert props_st["preprocess_padding_value"] == ""

    # 4) 非 ONNX 格式设自定义预处理时给出忽略提醒，不传进 export_kwargs
    cfg_ts = ExportConfig(
        weights_path=pt_file,
        export_format="torchscript",
        preprocess_mode="stretch",
    )
    ts_kw, ts_warns = build_export_kwargs(cfg_ts, metadata=meta_pt)
    assert "preprocess_mode" not in ts_kw
    assert any("仅对 ONNX 格式生效" in w for w in ts_warns)

    # 5) CLI export 解析 --preprocess / --pad-value / --no-preprocess-auto-rect 及非法值拦截
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "export",
            "-w",
            str(pt_file),
            "-f",
            "onnx",
            "--dynamic",
            "--preprocess",
            "letterbox_top_left",
            "--pad-value",
            "0",
            "--no-preprocess-auto-rect",
        ]
    )
    cli_cfg = _build_cli_export_config(parsed)
    assert isinstance(cli_cfg, ExportConfig)
    assert cli_cfg.preprocess_mode == "letterbox_top_left"
    assert cli_cfg.padding_value == 0
    assert cli_cfg.preprocess_auto_rect is False
    assert cli_main(["export", "-w", str(pt_file), "--pad-value", "300"]) == 2

    # 6) inspect_weights_metadata 既支持从 JSON 总包解析，也支持仅扁平键回退解析
    fake_base_mod = types.ModuleType("ultralytics.nn.backends.base")
    fake_backends_mod = types.ModuleType("ultralytics.nn.backends")
    fake_backends_mod.base = fake_base_mod  # type: ignore[attr-defined]
    fake_nn_mod = types.ModuleType("ultralytics.nn")
    fake_nn_mod.backends = fake_backends_mod  # type: ignore[attr-defined]
    fake_ultra = types.ModuleType("ultralytics")
    fake_ultra.nn = fake_nn_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)
    monkeypatch.setitem(sys.modules, "ultralytics.nn", fake_nn_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.backends", fake_backends_mod)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.backends.base", fake_base_mod)

    fake_base_mod.BaseBackend = types.SimpleNamespace(  # type: ignore[attr-defined]
        read_metadata=staticmethod(lambda _p: {"imgsz": "[576, 960]", **props_tl})
    )
    meta_from_json = inspect_weights_metadata(onnx_file)
    assert meta_from_json.deploy_schema == ONNX_DEPLOY_SCHEMA_VERSION
    assert meta_from_json.export_nms_mode == payload_tl["postprocess"]["output_mode"]
    assert meta_from_json.preprocess_mode == "letterbox_top_left"
    assert meta_from_json.preprocess_padding_value == 0
    assert meta_from_json.preprocess_auto_rect is False
    assert meta_from_json.class_names == ((0, "stump"), (1, "boss"))
    assert "预处理契约: 左上靠边 LetterBox (补边=0)" in format_weights_metadata_log(meta_from_json)

    flat_only = {k: v for k, v in props_st.items() if k != ONNX_DEPLOY_META_JSON_KEY}
    fake_base_mod.BaseBackend = types.SimpleNamespace(  # type: ignore[attr-defined]
        read_metadata=staticmethod(lambda _p: {"imgsz": "[640, 640]", **flat_only})
    )
    meta_from_flat = inspect_weights_metadata(onnx_file)
    assert meta_from_flat.preprocess_mode == "stretch"
    assert meta_from_flat.preprocess_padding_value is None


def test_real_exported_onnx_deploy_metadata_roundtrip_with_ort_if_present(tmp_path: Path) -> None:
    """若本地存在 `yolo26n.pt`，真实导出带自定义预处理元数据的 ONNX 并用 ONNX Runtime 与推理引擎端到端校验。"""
    import json
    import subprocess

    candidate = Path("build/portable_cache/weights/yolo26n.pt")
    if not candidate.is_file():
        pytest.skip("本地未找到预置 yolo26n.pt 权重，跳过真实 ONNX 元数据与推理闭环测试")

    src_dir = str((Path(__file__).resolve().parents[1] / "src").resolve())
    script = (
        f"import json, shutil, sys; from pathlib import Path; import cv2, numpy as np, onnxruntime as ort; "
        f"sys.path.insert(0, {src_dir!r}); "
        "from yolo_studio.core.trainer import export_model, inspect_weights_metadata, predict_source, ONNX_DEPLOY_META_JSON_KEY; "
        f"work = Path({str(tmp_path.resolve())!r}); "
        f"pt = work / 'yolo26n.pt'; shutil.copy({str(candidate.resolve())!r}, pt); "
        "img_p = work / 'blank.jpg'; cv2.imwrite(str(img_p), np.full((180, 320, 3), 120, dtype=np.uint8)); "
        # 1) 左上靠边 + 补边 0 导出 + ORT 读取 + 推理
        "onnx_tl = export_model(pt, export_format='onnx', imgsz=(192, 320), nms=False, simplify=False, "
        "preprocess_mode='letterbox_top_left', padding_value=0, device='cpu'); "
        "sess_tl = ort.InferenceSession(str(onnx_tl), providers=['CPUExecutionProvider']); "
        "ort_map_tl = sess_tl.get_modelmeta().custom_metadata_map; "
        "deploy_tl = json.loads(ort_map_tl[ONNX_DEPLOY_META_JSON_KEY]); "
        "meta_tl = inspect_weights_metadata(onnx_tl); "
        "outs_tl = predict_source(onnx_tl, img_p, work / 'out_tl', conf=0.25, device='cpu'); "
        # 2) 直接拉伸 stretch 导出 + ORT 读取 + 推理
        "pt2 = work / 'yolo26n_st.pt'; shutil.copy(pt, pt2); "
        "onnx_st = export_model(pt2, export_format='onnx', imgsz=320, nms=None, simplify=False, "
        "preprocess_mode='stretch', device='cpu'); "
        "sess_st = ort.InferenceSession(str(onnx_st), providers=['CPUExecutionProvider']); "
        "deploy_st = json.loads(sess_st.get_modelmeta().custom_metadata_map[ONNX_DEPLOY_META_JSON_KEY]); "
        "outs_st = predict_source(onnx_st, img_p, work / 'out_st', conf=0.25, device='cpu'); "
        "print(json.dumps({"
        "'tl_schema': deploy_tl['schema'], 'tl_align': deploy_tl['preprocess']['padding_align'], "
        "'tl_pad': deploy_tl['preprocess']['padding_value'], 'tl_hw': [deploy_tl['input']['height'], deploy_tl['input']['width']], "
        "'tl_out_mode': deploy_tl['postprocess']['output_mode'], 'tl_cls_len': len(deploy_tl['classes']), "
        "'tl_has_ultra_names': 'names' in ort_map_tl, 'tl_meta_mode': meta_tl.preprocess_mode, 'tl_meta_pad': meta_tl.preprocess_padding_value, "
        "'tl_outs': len(outs_tl), 'st_resize': deploy_st['preprocess']['resize_mode'], 'st_pad': deploy_st['preprocess']['padding_value'], "
        "'st_nms_req': deploy_st['postprocess']['nms_required'], 'st_outs': len(outs_st)}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=90,
    )
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["tl_schema"] == ONNX_DEPLOY_SCHEMA_VERSION
    assert res["tl_align"] == "top_left"
    assert res["tl_pad"] == 0
    assert res["tl_hw"] == [192, 320]
    assert res["tl_out_mode"] == "end2end"
    assert res["tl_cls_len"] == 80
    assert res["tl_has_ultra_names"] is True
    assert res["tl_meta_mode"] == "letterbox_top_left"
    assert res["tl_meta_pad"] == 0
    assert res["tl_outs"] == 1
    assert res["st_resize"] == "stretch"
    assert res["st_pad"] is None
    assert res["st_nms_req"] is True
    assert res["st_outs"] == 1


def test_eval_tab_export_preprocess_controls_and_predict_alignment(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证第 6 页导出预处理策略防呆、与推理对齐及加载带 `stretch` 元数据 ONNX 的自动锁定。"""
    pt_file = tmp_path / "best.pt"
    onnx_file = tmp_path / "best_stretch.onnx"
    img_file = tmp_path / "demo.jpg"
    for p in (pt_file, onnx_file, img_file):
        p.write_bytes(b"stub")

    inspect_workers: list[_StubWeightsInspectWorker] = []
    pred_workers: list[object] = []

    class _StubPredictWorker(QObject):
        log_line = Signal(str)
        finished_outputs = Signal(object)
        failed = Signal(str)

        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = dict(kwargs)
            self.running = False

        def start(self) -> None:
            self.running = True

        def isRunning(self) -> bool:
            return self.running

    monkeypatch.setattr(
        tab_eval_module,
        "WeightsInspectWorker",
        lambda path, request_id=0: (
            inspect_workers.append(_StubWeightsInspectWorker(path, request_id=request_id))
            or inspect_workers[-1]
        ),
    )
    monkeypatch.setattr(
        tab_eval_module,
        "PredictWorker",
        lambda **kwargs: (pred_workers.append(_StubPredictWorker(**kwargs)) or pred_workers[-1]),
    )

    tab = EvalTab(workspace_root=tmp_path, device_info=_make_device_info(has_gpu=True))
    try:
        tab.set_weights_and_yaml(str(pt_file), "")
        tab.source_edit.setText(str(img_file))
        qt_app.processEvents()
        inspect_workers[-1].complete_with(
            WeightsMetadata(
                weights_path=pt_file,
                task="detect",
                train_imgsz=(960, 960),
                class_names=((0, "stump"),),
                supports_end2end=True,
                one2many_available=True,
            )
        )
        qt_app.processEvents()

        # 1) ONNX 默认启用预处理与补边值；切 stretch 置灰补边值；切 TorchScript 置灰两者
        assert tab.export_preprocess_combo.isEnabled() is True
        assert tab.export_pad_value_spin.isEnabled() is True

        tab.export_preprocess_combo.setCurrentIndex(2)  # stretch
        qt_app.processEvents()
        assert tab.export_pad_value_spin.isEnabled() is False
        assert "偏离 YOLO 默认" in tab.export_io_label.text()

        tab.format_combo.setCurrentIndex(1)  # torchscript
        qt_app.processEvents()
        assert tab.export_preprocess_combo.isEnabled() is False
        assert tab.export_pad_value_spin.isEnabled() is False

        tab.format_combo.setCurrentIndex(0)  # onnx
        tab.export_preprocess_combo.setCurrentIndex(1)  # letterbox_top_left
        tab.export_pad_value_spin.setValue(0)
        qt_app.processEvents()
        assert tab.export_preprocess_combo.isEnabled() is True
        assert tab.export_pad_value_spin.isEnabled() is True

        exp_cfg = tab.build_export_config(allow_missing_weights=False)
        assert exp_cfg.preprocess_mode == "letterbox_top_left"
        assert exp_cfg.padding_value == 0

        # 2) 点击「对齐导出参数」后，推理参数摘要不再提示预处理差异，且 PredictWorker 收到 top_left + pad=0
        assert "预处理" in tab.pred_summary_label.text()
        tab.sync_export_pred_btn.click()
        qt_app.processEvents()
        assert "存在差异" not in tab.pred_summary_label.text()

        tab.predict_btn.click()
        qt_app.processEvents()
        sent_pt = pred_workers[-1].kwargs  # type: ignore[attr-defined]
        assert sent_pt["preprocess_mode"] == "letterbox_top_left"
        assert sent_pt["padding_value"] == 0
        pred_workers[-1].running = False  # type: ignore[attr-defined]
        tab._set_busy(False)

        # 3) 加载带 stretch 预处理契约的动态 ONNX：锁定 pred_pad_combo，PredictWorker 自动按 stretch 执行
        tab.set_weights_and_yaml(str(onnx_file), "")
        qt_app.processEvents()
        inspect_workers[-1].complete_with(
            WeightsMetadata(
                weights_path=onnx_file,
                task="detect",
                train_imgsz=(960, 960),
                class_names=((0, "stump"),),
                model_format="onnx",
                is_exported=True,
                export_nms_mode="end2end",
                export_dynamic=True,
                deploy_schema=ONNX_DEPLOY_SCHEMA_VERSION,
                preprocess_mode="stretch",
                preprocess_padding_value=None,
                preprocess_auto_rect=False,
            )
        )
        qt_app.processEvents()
        assert "直接拉伸 Stretch" in tab.weights_info_label.text()
        assert tab.pred_pad_combo.isEnabled() is False

        tab.predict_btn.click()
        qt_app.processEvents()
        sent_onnx = pred_workers[-1].kwargs  # type: ignore[attr-defined]
        assert sent_onnx["preprocess_mode"] == "stretch"
        assert sent_onnx["padding_value"] is None
        assert sent_onnx["rect"] is False
    finally:
        tab.close()


def test_predict_render_compact_non_overlapping_translucent_and_modes(tmp_path: Path) -> None:
    """验证紧凑防重叠半透明推理绘制引擎、置信度小数格式、样式切换与视频参数透传。"""
    import numpy as np

    # 1) 配置校验与摘要格式化
    assert parse_predict_render_style(None) == "compact"
    assert parse_predict_render_style("ultralytics") == "ultralytics"
    assert parse_predict_line_width_mode("thin") == "thin"
    assert parse_predict_line_width_mode("bold") == "bold"
    with pytest.raises(ValueError, match="无效的推理绘制样式"):
        parse_predict_render_style("unknown")
    with pytest.raises(ValueError, match="无效的框线挡位"):
        parse_predict_line_width_mode("huge")
    with pytest.raises(ValueError, match="label_alpha"):
        normalize_predict_render_config(PredictRenderConfig(label_alpha=1.5))

    assert "紧凑防重叠" in format_predict_render_summary(PredictRenderConfig())
    assert "仅画框" in format_predict_render_summary(
        PredictRenderConfig(show_labels=False, show_conf=False)
    )

    # 2) 二维防重叠排版：12 个水平紧邻小框（标签宽远大于框宽）开启避让后应 0 重叠且不越界
    img_w, img_h = 1024, 598
    dense_boxes = [(120 + i * 52, 260, 120 + i * 52 + 40, 320) for i in range(12)]
    dense_sizes = [(86, 18) for _ in dense_boxes]
    dense_scores = [0.95 - i * 0.01 for i in range(12)]

    placed_no_avoid = _layout_non_overlapping_labels(
        dense_boxes, dense_sizes, dense_scores, img_w=img_w, img_h=img_h, avoid_overlap=False
    )
    overlap_before = sum(
        _rect_intersect_area(placed_no_avoid[i], placed_no_avoid[j])
        for i in range(len(placed_no_avoid))
        for j in range(i + 1, len(placed_no_avoid))
    )
    assert overlap_before > 0

    placed_avoid = _layout_non_overlapping_labels(
        dense_boxes, dense_sizes, dense_scores, img_w=img_w, img_h=img_h, avoid_overlap=True
    )
    assert len(placed_avoid) == len(dense_boxes)
    for rx1, ry1, rx2, ry2 in placed_avoid:
        assert 0 <= rx1 < rx2 <= img_w
        assert 0 <= ry1 < ry2 <= img_h
    overlap_after = sum(
        _rect_intersect_area(placed_avoid[i], placed_avoid[j])
        for i in range(len(placed_avoid))
        for j in range(i + 1, len(placed_avoid))
    )
    assert overlap_after == 0

    # 3) 半透明验证：在亮底图上画单个检测框，确认标签区域像素介于纯底色与原图背景之间而非纯色覆盖
    canvas_bg = np.full((360, 640, 3), 220, dtype=np.uint8)
    one_box = types.SimpleNamespace(
        xyxy=np.array([[180.0, 140.0, 230.0, 210.0]], dtype=np.float32),
        cls=np.array([0.0], dtype=np.float32),
        conf=np.array([0.94], dtype=np.float32),
        is_track=False,
    )
    res_one = types.SimpleNamespace(orig_img=canvas_bg, boxes=one_box, names={0: "player"})
    img_semi = render_predict_result(
        res_one, PredictRenderConfig(style="compact", label_alpha=0.55)
    )
    img_opaque = render_predict_result(
        res_one, PredictRenderConfig(style="compact", label_alpha=1.0)
    )
    img_box_only = render_predict_result(
        res_one, PredictRenderConfig(style="compact", show_labels=False, show_conf=False)
    )
    assert img_semi.shape == canvas_bg.shape
    assert not np.array_equal(img_semi, img_opaque)
    # 仅画框模式不应在框上方标签带写入背景色块
    top_strip_box_only = img_box_only[110:136, 180:270]
    assert np.all(top_strip_box_only == 220)
    # 有标签模式在框上方标签带应已绘制像素，且半透明比不透明更接近底图 220
    top_strip_semi = img_semi[110:136, 180:270].astype(np.float32)
    top_strip_opaque = img_opaque[110:136, 180:270].astype(np.float32)
    assert float(np.abs(top_strip_semi - 220.0).mean()) > 0.5
    assert float(np.abs(top_strip_semi - 220.0).mean()) < float(
        np.abs(top_strip_opaque - 220.0).mean()
    )

    # 4) 覆盖中文类别名、4 种挡位及仅置信度/仅类别模式，确保不乱码不抛异常
    cn_boxes = types.SimpleNamespace(
        xyxy=np.array([[b[0], b[1], b[2], b[3]] for b in dense_boxes[:6]], dtype=np.float32),
        cls=np.array([i % 2 for i in range(6)], dtype=np.float32),
        conf=np.array([0.94, 0.87, 0.91, 0.76, 0.88, 0.95], dtype=np.float32),
        is_track=False,
    )
    res_cn = types.SimpleNamespace(
        orig_img=np.full((img_h, img_w, 3), 100, dtype=np.uint8),
        boxes=cn_boxes,
        names={0: "怪物", 1: "玩家"},
    )
    for lw_mode in ("auto", "thin", "medium", "bold"):
        for show_l, show_c in ((True, True), (True, False), (False, True), (False, False)):
            out_img = render_predict_result(
                res_cn,
                PredictRenderConfig(
                    style="compact",
                    line_width_mode=lw_mode,
                    show_labels=show_l,
                    show_conf=show_c,
                ),
            )
            assert out_img.shape == (img_h, img_w, 3)

    # 5) Ultralytics 默认模式与带 masks 非纯检测模式均正确透传 plot 参数；无 orig_img 桩对象兼容无参 plot
    plot_calls: list[dict[str, object]] = []

    class _RecordingResult:
        def __init__(self, *, with_masks: bool = False) -> None:
            self.orig_img = np.zeros((320, 480, 3), dtype=np.uint8)
            self.boxes = one_box
            self.names = {0: "player"}
            self.masks = object() if with_masks else None

        def plot(self, **kwargs: object) -> np.ndarray:
            plot_calls.append(dict(kwargs))
            return np.zeros((320, 480, 3), dtype=np.uint8)

    render_predict_result(
        _RecordingResult(),
        PredictRenderConfig(style="ultralytics", line_width_mode="thin", show_labels=False, show_conf=True),
    )
    assert plot_calls[-1] == {"labels": False, "conf": True, "line_width": 1}

    render_predict_result(
        _RecordingResult(with_masks=True),
        PredictRenderConfig(style="compact", line_width_mode="medium", show_labels=True, show_conf=False),
    )
    assert plot_calls[-1] == {"labels": True, "conf": False, "line_width": 2}

    stub_no_args = types.SimpleNamespace(plot=lambda: np.zeros((24, 24, 3), dtype=np.uint8))
    assert render_predict_result(stub_no_args).shape == (24, 24, 3)

    # 6) 视频推理参数：compact 由引擎逐帧写出 (save=False)，ultralytics 透传 save=True 与样式参数
    pt_file = tmp_path / "m.pt"
    vid_file = tmp_path / "demo.mp4"
    pt_file.write_bytes(b"pt")
    vid_file.write_bytes(b"mp4")
    kw_vid_compact, _ = build_predict_kwargs(
        pt_file,
        vid_file,
        is_video=True,
        output_dir=tmp_path / "out",
        render_config=PredictRenderConfig(style="compact"),
    )
    assert kw_vid_compact["save"] is False

    kw_vid_ultra, _ = build_predict_kwargs(
        pt_file,
        vid_file,
        is_video=True,
        output_dir=tmp_path / "out",
        render_config=PredictRenderConfig(
            style="ultralytics", line_width_mode="thin", show_labels=True, show_conf=False
        ),
    )
    assert kw_vid_ultra["save"] is True
    assert kw_vid_ultra["show_labels"] is True
    assert kw_vid_ultra["show_conf"] is False
    assert kw_vid_ultra["line_width"] == 1


def test_eval_tab_predict_render_controls_and_zoom_preview(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证第 6 页推理预览样式控件、Worker `render_config` 透传与原图放大查看按钮联动。"""
    import cv2
    import numpy as np

    pt_file = tmp_path / "best.pt"
    img_file = tmp_path / "demo.jpg"
    pred_img = tmp_path / "demo_pred.jpg"
    pt_file.write_bytes(b"pt")
    cv2.imwrite(str(img_file), np.full((240, 320, 3), 120, dtype=np.uint8))
    cv2.imwrite(str(pred_img), np.full((240, 320, 3), 150, dtype=np.uint8))

    pred_workers: list[object] = []

    class _StubPredictWorker(QObject):
        log_line = Signal(str)
        finished_outputs = Signal(object)
        failed = Signal(str)

        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = dict(kwargs)
            self.running = False

        def start(self) -> None:
            self.running = True

        def isRunning(self) -> bool:
            return self.running

    monkeypatch.setattr(
        tab_eval_module,
        "WeightsInspectWorker",
        lambda path, request_id=0: _StubWeightsInspectWorker(path, request_id=request_id),
    )
    monkeypatch.setattr(
        tab_eval_module,
        "PredictWorker",
        lambda **kwargs: (pred_workers.append(_StubPredictWorker(**kwargs)) or pred_workers[-1]),
    )

    tab = EvalTab(workspace_root=tmp_path, device_info=_make_device_info(has_gpu=True))
    try:
        # 1) 默认样式：紧凑防重叠 + 自动 + 类别 + 置信度，尚未生成预览图时放大按钮置灰
        cfg_default = tab.build_predict_render_config()
        assert cfg_default.style == "compact"
        assert cfg_default.line_width_mode == "auto"
        assert cfg_default.show_labels is True
        assert cfg_default.show_conf is True
        assert cfg_default.avoid_overlap is True
        assert 0.0 < cfg_default.label_alpha < 1.0
        assert "紧凑防重叠" in tab.pred_summary_label.text()
        assert tab.zoom_preview_btn.isEnabled() is False

        # 2) 修改预览样式控件并点击推理预览，确认 PredictWorker 准确拿到最新 render_config
        tab.set_weights_and_yaml(str(pt_file), "")
        tab.source_edit.setText(str(img_file))
        tab.pred_render_style_combo.setCurrentIndex(1)  # 切换至 Ultralytics 默认
        tab.pred_line_width_combo.setCurrentIndex(1)  # 切换至极细挡位
        tab.pred_show_labels_chk.setChecked(True)
        tab.pred_show_conf_chk.setChecked(False)
        qt_app.processEvents()

        assert "Ultralytics 默认" in tab.pred_summary_label.text()
        assert "仅类别" in tab.pred_summary_label.text()

        tab.predict_btn.click()
        qt_app.processEvents()
        assert len(pred_workers) == 1
        sent_render = pred_workers[-1].kwargs["render_config"]  # type: ignore[attr-defined]
        assert isinstance(sent_render, PredictRenderConfig)
        assert sent_render.style == "ultralytics"
        assert sent_render.line_width_mode == "thin"
        assert sent_render.show_labels is True
        assert sent_render.show_conf is False

        # 3) 模拟推理完成加载结果图：放大按钮启用，信息栏展示文件名与尺寸，真实缩放弹窗可正常适应与缩放
        tab._on_predict_done([pred_img])
        qt_app.processEvents()
        assert tab.zoom_preview_btn.isEnabled() is True
        assert "demo_pred.jpg" in tab.preview_info_label.text()
        assert "320×240" in tab.preview_info_label.text()

        pix = tab.preview_label.source_pixmap()
        assert pix is not None and not pix.isNull()
        real_zoom_dlg = tab_eval_module._PreviewZoomDialog(pix, image_path=pred_img, parent=tab)
        try:
            real_zoom_dlg._set_zoom(2.0)
            assert real_zoom_dlg.scale_label.text() == "200%"
            real_zoom_dlg._set_zoom(1.0)
            assert real_zoom_dlg.scale_label.text() == "100%"
            real_zoom_dlg._fit_to_window()
            assert real_zoom_dlg.scale_label.text().endswith("%")
        finally:
            real_zoom_dlg.close()

        opened_dialogs: list[object] = []

        class _StubZoomDialog:
            def __init__(self, pix, image_path=None, parent=None) -> None:
                self.pix = pix
                self.image_path = image_path
                opened_dialogs.append(self)

            def exec(self) -> int:
                return 0

        monkeypatch.setattr(tab_eval_module, "_PreviewZoomDialog", _StubZoomDialog)
        tab.zoom_preview_btn.click()
        tab.preview_label.double_clicked.emit()
        qt_app.processEvents()
        assert len(opened_dialogs) == 2
        assert opened_dialogs[0].image_path == pred_img  # type: ignore[attr-defined]
    finally:
        tab.close()
