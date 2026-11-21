"""「5. YOLO26 训练」标签页（`TrainTab`）实时训练指标重置与训练时长单元测试。

职责：
    在离线 `offscreen` Qt 环境下验证训练时长格式化边界、重新开始训练时实时指标与进度条的立即重置、
    单调时钟驱动的训练时长实时刷新与停表定格、旧训练线程信号隔离，以及校验失败/GPU 占用拦截、
    `extra_args` 覆盖 `epochs`、断点续训与关闭窗口时的边界行为。

运行环境与协作边界：
    通过 `pytest` 运行，依赖 `PyQt6`；仅实例化 `TrainTab` 并通过轻量假 `TrainWorker` 驱动信号，
    不启动真实模型训练，不加载内嵌 X-AnyLabeling 或访问外部网络。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QObject, Qt, pyqtSignal as Signal
from PyQt6.QtWidgets import QApplication, QMessageBox

import yolo_studio.gui.tab_train as tab_train_module
from yolo_studio.core.trainer import DeviceInfo, EpochMetrics, TrainConfig, TrainResult
from yolo_studio.gui.common import GpuTaskGuard
from yolo_studio.gui.tab_train import TrainTab, format_elapsed_duration


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """初始化模块级无头 `QApplication`；与内嵌工作台测试共用进程时需提前开启 OpenGL 上下文共享。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if existing is not None:
        return existing
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    return QApplication([])


def _make_cpu_device_info() -> DeviceInfo:
    """构造无 CUDA 探测开销的固定 CPU 硬件快照，避免单测触发 `import torch`。"""
    return DeviceInfo(
        cuda_available=False,
        device_count=0,
        device_name="Mock CPU",
        vram_gb=0.0,
        recommended_device="cpu",
        probing=False,
        gpu_devices=(),
    )


class _StubTrainWorker(QObject):
    """替代真实后台 `TrainWorker` 的同步信号桩，便于在测试中精确控制每个训练阶段。"""

    epoch_updated = Signal(object)
    log_line = Signal(str)
    finished_result = Signal(object)
    failed = Signal(str)

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.config = config
        self.started = False
        self.running = False
        self.cancelled = False

    def start(self) -> None:
        self.started = True
        self.running = True

    def isRunning(self) -> bool:
        return self.running

    def cancel(self) -> None:
        self.cancelled = True


def test_format_elapsed_duration_boundaries() -> None:
    """验证训练时长格式化函数对零值、负数、非有限值、小时进位及超过 99 小时的处理。"""
    assert format_elapsed_duration(0.0) == "00:00:00"
    assert format_elapsed_duration(0.99) == "00:00:00"
    assert format_elapsed_duration(-12.5) == "00:00:00"
    assert format_elapsed_duration(float("nan")) == "00:00:00"
    assert format_elapsed_duration(float("inf")) == "00:00:00"
    assert format_elapsed_duration(1.0) == "00:00:01"
    assert format_elapsed_duration(65.4) == "00:01:05"
    assert format_elapsed_duration(3661.9) == "01:01:01"
    assert format_elapsed_duration(100 * 3600 + 2 * 60 + 3) == "100:02:03"


def test_restart_training_resets_metrics_immediately_and_tracks_elapsed_time(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证首次训练结束后再次开始训练，全部实时指标立即重置、训练时长实时走动并在结束/失败时停表。"""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [target]\n", encoding="utf-8")

    now = [1000.0]
    monkeypatch.setattr(tab_train_module.time, "monotonic", lambda: now[0])

    created_workers: list[_StubTrainWorker] = []

    def _worker_factory(cfg: TrainConfig) -> _StubTrainWorker:
        w = _StubTrainWorker(cfg)
        created_workers.append(w)
        return w

    monkeypatch.setattr(tab_train_module, "TrainWorker", _worker_factory)

    critical_errors: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda _parent, _title, text: critical_errors.append(str(text))),
    )

    tab = TrainTab(workspace_root=tmp_path, device_info=_make_cpu_device_info())
    try:
        tab.set_data_yaml(str(data_yaml))
        tab.epochs_spin.setValue(10)

        # 初始空闲状态
        assert tab.val_epoch.text() == "0 / 0"
        assert tab.val_elapsed.text() == "-"
        assert tab.val_box_loss.text() == "-"
        assert tab._elapsed_timer.isActive() is False

        # 第 1 次启动训练：立即显示 0 / 10 与 00:00:00 并启动计时器
        tab._on_start()
        assert len(created_workers) == 1
        worker1 = created_workers[0]
        assert worker1.started is True
        assert tab.val_epoch.text() == "0 / 10"
        assert tab.val_elapsed.text() == "00:00:00"
        assert tab._elapsed_timer.isActive() is True

        # 第 1 次训练推进 125 秒并完成全部 10 轮
        now[0] = 1125.4
        worker1.epoch_updated.emit(
            EpochMetrics(
                epoch=10,
                total_epochs=10,
                box_loss=0.3125,
                cls_loss=0.2048,
                precision=0.9123,
                recall=0.8765,
                map50=0.9345,
                map50_95=0.7123,
            )
        )
        qt_app.processEvents()
        assert tab.progress_bar.value() == 100
        assert tab.val_epoch.text() == "10 / 10"
        assert tab.val_elapsed.text() == "00:02:05"
        assert tab.val_box_loss.text() == "0.3125"
        assert tab.val_cls_loss.text() == "0.2048"
        assert tab.val_precision.text() == "0.9123"
        assert tab.val_recall.text() == "0.8765"
        assert tab.val_map50.text() == "0.9345"
        assert tab.val_map50_95.text() == "0.7123"

        best_pt = tmp_path / "runs" / "exp1" / "weights" / "best.pt"
        best_pt.parent.mkdir(parents=True, exist_ok=True)
        best_pt.write_bytes(b"pt")
        now[0] = 1130.2
        worker1.running = False
        worker1.finished_result.emit(
            TrainResult(save_dir=best_pt.parents[1], best_weights=best_pt, last_weights=None)
        )
        qt_app.processEvents()

        # 第 1 次训练完成后定格时长并停表，日志输出总训练时长
        assert tab._elapsed_timer.isActive() is False
        assert tab.val_elapsed.text() == "00:02:10"
        assert "训练时长: 00:02:10" in tab.log_box.toPlainText()
        assert tab.send_eval_btn.isEnabled() is True
        assert tab._last_best_pt == str(best_pt)

        # 即使时间继续流逝，停表后卡片数值保持定格
        now[0] = 1500.0
        assert tab.val_elapsed.text() == "00:02:10"

        # 第 2 次重新开始训练（修改总轮数为 25）：在首轮回调到来前全部指标必须立刻重置
        tab.epochs_spin.setValue(25)
        tab._on_start()
        assert len(created_workers) == 2
        worker2 = created_workers[1]

        assert tab.progress_bar.value() == 0
        assert tab.val_epoch.text() == "0 / 25"
        assert tab.val_elapsed.text() == "00:00:00"
        assert tab.val_box_loss.text() == "-"
        assert tab.val_cls_loss.text() == "-"
        assert tab.val_precision.text() == "-"
        assert tab.val_recall.text() == "-"
        assert tab.val_map50.text() == "-"
        assert tab.val_map50_95.text() == "-"
        assert tab.send_eval_btn.isEnabled() is False
        assert tab._last_best_pt == ""
        assert tab._elapsed_timer.isActive() is True

        # 旧 worker 残留信号已断开，发射也不会污染第 2 次训练界面
        worker1.epoch_updated.emit(EpochMetrics(epoch=9, total_epochs=10, box_loss=9.9999))
        qt_app.processEvents()
        assert tab.val_epoch.text() == "0 / 25"
        assert tab.val_box_loss.text() == "-"

        # 未完成第 1 轮前，随单调时钟推进实时刷新训练时长
        now[0] = 1504.3
        tab._refresh_elapsed_time()
        assert tab.val_elapsed.text() == "00:00:04"

        # 模拟第 2 次训练失败：停表并定格最终时长
        now[0] = 1512.0
        worker2.running = False
        worker2.failed.emit("CUDA OOM")
        qt_app.processEvents()

        assert tab._elapsed_timer.isActive() is False
        assert tab.val_elapsed.text() == "00:00:12"
        assert "训练时长: 00:00:12" in tab.log_box.toPlainText()
        assert "[错误] CUDA OOM" in tab.log_box.toPlainText()
        assert critical_errors == ["CUDA OOM"]
        assert tab.start_btn.isEnabled() is True
    finally:
        tab.close()


def test_start_validation_guard_extra_epochs_resume_and_close_stops_timer(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证校验失败/互斥拦截不清空旧指标、`extra_args` 与 `resume` 的 Epoch 初始分母，以及关闭停表。"""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [target]\n", encoding="utf-8")

    now = [2000.0]
    monkeypatch.setattr(tab_train_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(tab_train_module, "TrainWorker", _StubTrainWorker)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *_args, **_kwargs: None))

    guard = GpuTaskGuard()
    tab = TrainTab(workspace_root=tmp_path, device_info=_make_cpu_device_info(), gpu_guard=guard)
    try:
        # 模拟已有历史训练指标
        tab._on_epoch_updated(EpochMetrics(epoch=5, total_epochs=5, box_loss=0.5, map50=0.8))
        tab.val_elapsed.setText("00:01:00")

        # 1) 表单校验失败（data.yaml 不存在）不应清除上次指标
        tab.set_data_yaml(str(tmp_path / "missing.yaml"))
        tab._on_start()
        assert tab.val_epoch.text() == "5 / 5"
        assert tab.val_elapsed.text() == "00:01:00"
        assert tab.val_box_loss.text() == "0.5000"

        # 2) GPU 互斥拦截不应清除上次指标或改写结果目录
        tab.set_data_yaml(str(data_yaml))
        prev_save_dir = tab._last_save_dir
        monkeypatch.setattr(guard, "try_acquire", lambda _parent, _name: False)
        tab._on_start()
        assert tab.val_epoch.text() == "5 / 5"
        assert tab.val_elapsed.text() == "00:01:00"
        assert tab._last_save_dir == prev_save_dir

        # 3) 恢复互斥放行，测试 extra_args 覆盖 epochs 时启动瞬间分母取覆盖值
        monkeypatch.setattr(guard, "try_acquire", lambda _parent, _name: True)
        tab.epochs_spin.setValue(100)
        tab.extra_args_edit.setPlainText("epochs=30")
        tab._on_start()
        assert tab.val_epoch.text() == "0 / 30"
        assert tab.val_elapsed.text() == "00:00:00"
        assert tab.val_box_loss.text() == "-"
        assert tab._elapsed_timer.isActive() is True

        # 结束上一次假任务，测试 resume=True 时 Epoch 暂重置为 0 / 0
        tab._on_finished(TrainResult(save_dir=tmp_path, best_weights=None, last_weights=None))
        tab._on_epoch_updated(EpochMetrics(epoch=30, total_epochs=30, box_loss=0.25))
        tab.resume_chk.setChecked(True)
        tab._on_start()
        assert tab.val_epoch.text() == "0 / 0"
        assert tab.val_elapsed.text() == "00:00:00"
        assert tab.val_box_loss.text() == "-"
        assert tab._elapsed_timer.isActive() is True

        # 4) 关闭标签页时自动停表
        tab.close()
        assert tab._elapsed_timer.isActive() is False
    finally:
        tab.close()


def test_final_eval_displays_best_metrics_without_overwriting_last_epoch(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 `best.pt` 最终评估在独立摘要卡展示且不覆盖最后一轮卡片，重启重置及 DDP 兜底均正常。"""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: [target]\n", encoding="utf-8")

    created_workers: list[_StubTrainWorker] = []

    def _worker_factory(cfg: TrainConfig) -> _StubTrainWorker:
        w = _StubTrainWorker(cfg)
        created_workers.append(w)
        return w

    monkeypatch.setattr(tab_train_module, "TrainWorker", _worker_factory)

    tab = TrainTab(workspace_root=tmp_path, device_info=_make_cpu_device_info())
    try:
        tab.set_data_yaml(str(data_yaml))
        tab.epochs_spin.setValue(10)

        # 初始空闲态：best.pt 最终评估卡片为占位符
        assert tab.val_best_precision.text() == "-"
        assert tab.val_best_recall.text() == "-"
        assert tab.val_best_map50.text() == "-"
        assert tab.val_best_map50_95.text() == "-"

        # 1) 启动训练 -> 第 10 轮结束 -> 收到 best.pt 最终评估回调
        tab._on_start()
        worker1 = created_workers[0]
        worker1.epoch_updated.emit(
            EpochMetrics(
                epoch=10,
                total_epochs=10,
                box_loss=0.3125,
                cls_loss=0.2048,
                precision=0.9012,
                recall=0.8650,
                map50=0.9210,
                map50_95=0.7015,
                is_final_eval=False,
            )
        )
        worker1.epoch_updated.emit(
            EpochMetrics(
                epoch=10,
                total_epochs=10,
                precision=0.9180,
                recall=0.8840,
                map50=0.9385,
                map50_95=0.7240,
                is_final_eval=True,
            )
        )
        qt_app.processEvents()

        # 上方最后一轮卡片完好保留，Epoch 保持 10 / 10
        assert tab.val_epoch.text() == "10 / 10"
        assert tab.val_box_loss.text() == "0.3125"
        assert tab.val_cls_loss.text() == "0.2048"
        assert tab.val_precision.text() == "0.9012"
        assert tab.val_recall.text() == "0.8650"
        assert tab.val_map50.text() == "0.9210"
        assert tab.val_map50_95.text() == "0.7015"

        # 下方 best.pt 最终评估卡片正确展示最佳权重指标
        assert tab.val_best_precision.text() == "0.9180"
        assert tab.val_best_recall.text() == "0.8840"
        assert tab.val_best_map50.text() == "0.9385"
        assert tab.val_best_map50_95.text() == "0.7240"

        best_pt = tmp_path / "runs" / "exp1" / "weights" / "best.pt"
        best_pt.parent.mkdir(parents=True, exist_ok=True)
        best_pt.write_bytes(b"pt")
        worker1.running = False
        worker1.finished_result.emit(
            TrainResult(save_dir=best_pt.parents[1], best_weights=best_pt, last_weights=None)
        )
        qt_app.processEvents()
        assert "best.pt 最终评估: P=0.9180, R=0.8840, mAP50=0.9385, mAP50-95=0.7240" in tab.log_box.toPlainText()

        # 2) 重新开始训练：best.pt 最终评估卡片与上方卡片一并立即重置
        tab._on_start()
        worker2 = created_workers[1]
        assert tab.val_epoch.text() == "0 / 10"
        assert tab.val_best_precision.text() == "-"
        assert tab.val_best_recall.text() == "-"
        assert tab.val_best_map50.text() == "-"
        assert tab.val_best_map50_95.text() == "-"

        # 3) 模拟多卡 DDP 等未走主进程 final_eval 回调的场景：早停在第 4 轮，从 TrainResult.metrics 兜底回填
        worker2.epoch_updated.emit(
            EpochMetrics(
                epoch=4,
                total_epochs=10,
                box_loss=0.4500,
                cls_loss=0.3200,
                precision=0.8200,
                recall=0.7900,
                map50=0.8400,
                map50_95=0.6100,
                is_final_eval=False,
            )
        )
        worker2.running = False
        worker2.finished_result.emit(
            TrainResult(
                save_dir=best_pt.parents[1],
                best_weights=best_pt,
                last_weights=None,
                metrics={
                    "metrics/precision(B)": 0.8950,
                    "metrics/recall(B)": 0.8710,
                    "metrics/mAP50(B)": 0.9120,
                    "metrics/mAP50-95(B)": 0.6840,
                },
                stopped_early=True,
            )
        )
        qt_app.processEvents()

        # 早停轮次与最后一轮指标不被覆盖，best.pt 卡片自动兜底填充
        assert tab.val_epoch.text() == "4 / 10"
        assert tab.val_precision.text() == "0.8200"
        assert tab.val_map50.text() == "0.8400"
        assert tab.val_best_precision.text() == "0.8950"
        assert tab.val_best_recall.text() == "0.8710"
        assert tab.val_best_map50.text() == "0.9120"
        assert tab.val_best_map50_95.text() == "0.6840"
    finally:
        tab.close()
