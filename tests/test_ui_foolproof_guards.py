"""各工作流标签页防呆联动与前置校验单元测试。

职责：
    在离线 `offscreen` Qt 环境下系统性验证五个配置标签页的防呆交互逻辑：
    1. 「1. 视频抽帧」：选 `png` 置灰 `JPG质量`、起止时间段非法拦截、去重 0 值提示与列表增删按钮空态保护；
    2. 「3. LLM 标注清洗」：已有 `.json` + 本地 SAM 模式下动态展示 GPU 开关、审核中禁用历史报告加载、
       LLM 凭据/模型名/Base URL 前置校验（含本地/局域网地址豁免）与分流子目录开关联动按钮文案；
    3. 「4. 数据集准备」：`Train` 比例上限允许 `1.00` 且不再误报归一化提示；
    4. 「5. YOLO26 训练」：断点续训 (`resume`) 置灰全部被 Ultralytics 忽略的超参数并保留可覆盖项、
       关闭 `val` 置灰 `Patience`、`Batch` 跳过 `0` 且 CPU 模式限制 `Batch >= 1` 并在切回 GPU 后恢复 `-1`；
    5. 「6. 验证、推理与导出」：TensorRT 在 CPU 模式下置灰导出按钮，ONNX/TensorRT 开启动态输入时更新 Batch 提示。

运行环境与协作边界：
    通过 `pytest` 离线运行，依赖 `PyQt6`；测试中通过 `monkeypatch` 替换弹窗与后台线程，
    不访问外部网络，也不修改宿主机持久化 `QSettings`。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt6.QtCore import QCoreApplication, QSettings, Qt
from PyQt6.QtWidgets import QApplication, QListWidgetItem, QMessageBox

import yolo_studio.gui.tab_extract as tab_extract_mod
import yolo_studio.gui.tab_review as tab_review_mod
from yolo_studio.core.mllm_reviewer import BoxRefineMode, MllmReviewResult
from yolo_studio.core.trainer import DeviceInfo
from yolo_studio.gui.tab_dataset import DatasetTab
from yolo_studio.gui.tab_eval import EvalTab
from yolo_studio.gui.tab_extract import ExtractTab
from yolo_studio.gui.tab_review import MllmReviewTab
from yolo_studio.gui.tab_train import TrainTab


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    """初始化模块级无头 `QApplication`。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QApplication.instance()
    if existing is None:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        return QApplication([])
    return existing  # type: ignore[return-value]


def _gpu_device_info() -> DeviceInfo:
    return DeviceInfo(
        cuda_available=True,
        device_count=1,
        device_name="NVIDIA GeForce RTX 4060",
        vram_gb=8.0,
        recommended_device="0",
        gpu_devices=(("0", "NVIDIA GeForce RTX 4060", 8.0),),
    )


def test_extract_tab_png_disables_quality_time_range_and_list_buttons(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证视频抽帧页 `png` 置灰 JPG 质量、起止时间非法拦截、去重 0 值显示及列表按钮空态防呆。"""
    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, msg: warnings.append(str(msg)))

    started_workers: list[object] = []

    class _FakeExtractWorker:
        def __init__(self, *_args, **_kwargs) -> None:
            started_workers.append(self)

    monkeypatch.setattr(tab_extract_mod, "FrameExtractWorker", _FakeExtractWorker)

    tab = ExtractTab(workspace_root=tmp_path)
    try:
        # 1) 默认 jpg 格式启用 JPG 质量，切换 png 后标签与微调框均置灰禁用
        assert tab.format_combo.currentText() == "jpg"
        assert tab.quality_spin.isEnabled() is True
        assert tab.quality_label.isEnabled() is True

        tab.format_combo.setCurrentText("png")
        qt_app.processEvents()
        assert tab.quality_spin.isEnabled() is False
        assert tab.quality_label.isEnabled() is False
        assert "无损" in tab.quality_spin.toolTip()

        tab.format_combo.setCurrentText("jpg")
        qt_app.processEvents()
        assert tab.quality_spin.isEnabled() is True
        assert tab.quality_label.isEnabled() is True

        # 2) 相似帧去重阈值 0 显示特殊文案
        tab.dedup_spin.setValue(0.0)
        assert tab.dedup_spin.text() == "0.00 (不去重)"

        # 3) 空列表时移除选中与清空列表置灰；添加条目后清空启用、选中后移除启用
        assert tab.remove_btn.isEnabled() is False
        assert tab.clear_btn.isEnabled() is False

        fake_video = tmp_path / "demo.mp4"
        fake_video.write_bytes(b"")
        item = QListWidgetItem("demo.mp4")
        item.setData(Qt.ItemDataRole.UserRole, str(fake_video))
        tab.video_list.addItem(item)
        tab._video_paths.append(fake_video)
        tab._refresh_list_action_states()

        assert tab.clear_btn.isEnabled() is True
        assert tab.remove_btn.isEnabled() is False

        tab.video_list.setCurrentRow(0)
        qt_app.processEvents()
        assert tab.remove_btn.isEnabled() is True

        # 4) 截取结束时间 <= 开始时间（且 > 0）时提前拦截，不启动后台线程
        tab.start_sec_spin.setValue(10.0)
        tab.end_sec_spin.setValue(5.0)
        tab._on_start()
        assert len(started_workers) == 0
        assert any("结束时间必须大于开始时间" in m for m in warnings)

        # 移除选中条目后列表再次为空，按钮恢复置灰
        tab._on_remove_selected()
        qt_app.processEvents()
        assert len(tab._video_paths) == 0
        assert tab.remove_btn.isEnabled() is False
        assert tab.clear_btn.isEnabled() is False
    finally:
        tab.close()


def test_review_tab_sam_device_visibility_precheck_and_busy_report_lock(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证清洗页已有 `.json`+SAM 动态展示 GPU 开关、LLM 前置校验、分流按钮文案与运行期禁用历史报告。"""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, _t, msg: warnings.append(str(msg)))

    started_configs: list[object] = []

    class _FakeSignal:
        def connect(self, _slot) -> None:
            pass

    class _FakeReviewWorker:
        def __init__(self, cfg) -> None:
            self.cfg = cfg
            self.progress = _FakeSignal()
            self.log_line = _FakeSignal()
            self.finished_result = _FakeSignal()
            self.failed = _FakeSignal()
            self._running = False

        def start(self) -> None:
            self._running = True
            started_configs.append(self.cfg)

        def isRunning(self) -> bool:
            return self._running

    monkeypatch.setattr(tab_review_mod, "MllmReviewWorker", _FakeReviewWorker)

    in_dir = tmp_path / "images"
    in_dir.mkdir()
    out_dir = tmp_path / "out"

    tab = MllmReviewTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        # 1) 已有 .json + OFF：GPU 开关禁用且 SAM 设备栏隐藏
        tab.source_mode_combo.setCurrentIndex(1)  # EXISTING_JSON
        qt_app.processEvents()
        assert tab.yolo_device_ctrl.isEnabled() is False
        assert tab.sam_device_holder.isHidden() is True

        # 2) 已有 .json + SAM_LOCAL：GPU 开关迁移至 SAM 行下并启用，切 CPU 立即反映到配置
        sam_idx = tab.refine_mode_combo.findData(BoxRefineMode.SAM_LOCAL)
        tab.refine_mode_combo.setCurrentIndex(sam_idx)
        qt_app.processEvents()
        assert tab.sam_device_holder.isHidden() is False
        assert tab.yolo_device_ctrl.isEnabled() is True
        assert tab._sam_device_row.indexOf(tab.yolo_device_ctrl) >= 0

        tab.yolo_device_ctrl.gpu_chk.setChecked(False)
        assert tab._build_current_config().yolo_device == "cpu"
        tab.yolo_device_ctrl.gpu_chk.setChecked(True)
        assert tab._build_current_config().yolo_device == "0"

        # 切回基准模型推理模式时，GPU 开关归位到基准推理参数行，SAM 设备栏重新隐藏
        tab.source_mode_combo.setCurrentIndex(0)  # YOLO_PREDICT
        qt_app.processEvents()
        assert tab.sam_device_holder.isHidden() is True
        assert tab._yolo_row.indexOf(tab.yolo_device_ctrl) >= 0
        assert tab.yolo_device_ctrl.isEnabled() is True

        # 3) 分流子目录开关同步更新工作台打开按钮文案
        assert tab.open_need_review_btn.text() == "在工作台打开待复核目录"
        tab.triage_dirs_chk.setChecked(False)
        qt_app.processEvents()
        assert tab.open_need_review_btn.text() == "在工作台打开输出目录"
        tab.triage_dirs_chk.setChecked(True)
        qt_app.processEvents()
        assert tab.open_need_review_btn.text() == "在工作台打开待复核目录"

        # 4) LLM 前置校验：缺少 Gemini Key / 模型名 / OpenAI 云端 Key 均提前拦截，不启动后台线程
        tab.source_mode_combo.setCurrentIndex(1)  # EXISTING_JSON，省去伪造 .pt
        tab.set_images_dir(str(in_dir))
        tab.output_picker.setText(str(out_dir))

        warnings.clear()
        tab._on_start()
        assert len(started_configs) == 0
        assert any("GEMINI_API_KEY" in m for m in warnings)

        tab.api_key_edit.setText("dummy-key")
        tab.model_edit.clear()
        warnings.clear()
        tab._on_start()
        assert len(started_configs) == 0
        assert any("模型名称" in m for m in warnings)

        # OpenAI 兼容模式：云端地址缺 Key 拦截，局域网/本地地址免 Key 放行
        tab.provider_combo.setCurrentIndex(1)
        tab.api_key_edit.clear()
        tab.base_url_edit.setText("https://dashscope.aliyuncs.com/compatible-mode/v1")
        tab.model_edit.setText("qwen-vl-max-latest")
        warnings.clear()
        tab._on_start()
        assert len(started_configs) == 0
        assert any("OPENAI_API_KEY" in m for m in warnings)

        tab.base_url_edit.setText("http://192.168.1.50:11434/v1")
        qt_app.processEvents()
        assert "本地/局域网" in tab.api_key_status_label.text()
        tab._on_start()
        assert len(started_configs) == 1

        # 5) 审核运行期间禁用「加载历史报告...」，完成或失败后自动恢复
        assert tab.load_report_btn.isEnabled() is False
        tab._on_finished(
            MllmReviewResult(
                output_dir=out_dir,
                classes=["person"],
                total_images=0,
                auto_passed_count=0,
                llm_fixed_count=0,
                need_review_count=0,
                error_count=0,
            )
        )
        assert tab.load_report_btn.isEnabled() is True

        tab._on_start()
        assert tab.load_report_btn.isEnabled() is False
        monkeypatch.setattr(QMessageBox, "critical", lambda *_a: None)
        tab._on_failed("simulated error")
        assert tab.load_report_btn.isEnabled() is True
    finally:
        tab.close()


def test_dataset_tab_allows_train_ratio_1_00_without_normalization_warning(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证数据集准备页 `Train` 上限放宽至 `1.00`，全量训练不产生多余归一化提示。"""
    tab = DatasetTab(workspace_root=tmp_path)
    try:
        assert tab.train_spin.maximum() == pytest.approx(1.0)
        tab.train_spin.setValue(1.0)
        tab.val_spin.setValue(0.0)
        tab.test_spin.setValue(0.0)
        qt_app.processEvents()
        assert "归一化" not in tab.ratio_hint.text()
        assert "Val 为 0" in tab.ratio_hint.text()
    finally:
        tab.close()


def test_train_tab_resume_val_patience_and_batch_guards(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证训练页断点续训防呆、`val=False` 置灰 `Patience`、`Batch` 跳 0 与 CPU 模式禁 `-1`。"""
    settings = QSettings(str(tmp_path / "train_guards.ini"), QSettings.Format.IniFormat)
    tab = TrainTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        # 1) Batch 跳过无效值 0：从 1 向下直达 -1（显示“自动”），从 -1 向上直达 1，手设 0 自动纠回 1
        tab.batch_spin.setValue(1)
        tab.batch_spin.stepBy(-1)
        assert tab.batch_spin.value() == -1
        assert tab.batch_spin.text() == "自动"
        tab.batch_spin.stepBy(1)
        assert tab.batch_spin.value() == 1
        tab.batch_spin.setValue(0)
        assert tab.batch_spin.value() == 1

        # 2) GPU 下设 Batch=-1，切 CPU 后自动钳制 >= 1 并切至安全值 4；持久化保留 -1 意图；切回 GPU 恢复 -1
        tab.batch_spin.setValue(-1)
        tab.device_ctrl.gpu_chk.setChecked(False)
        qt_app.processEvents()
        assert tab.batch_spin.minimum() == 1
        assert tab.batch_spin.value() == 4
        assert "CPU 模式不支持 AutoBatch" in tab.batch_spin.toolTip()

        tab.save_settings(settings, force=True)
        settings.sync()

        tab.device_ctrl.gpu_chk.setChecked(True)
        qt_app.processEvents()
        assert tab.batch_spin.minimum() == -1
        assert tab.batch_spin.value() == -1

        # 3) 关闭训练期验证 (val=False) 置灰 Patience，重新勾选恢复启用
        assert tab.patience_spin.isEnabled() is True
        tab.val_chk.setChecked(False)
        qt_app.processEvents()
        assert tab.patience_spin.isEnabled() is False
        assert "val=False" in tab.patience_spin.toolTip()
        tab.val_chk.setChecked(True)
        qt_app.processEvents()
        assert tab.patience_spin.isEnabled() is True

        # 4) 勾选断点续训 (resume) 展示提示条、置灰全部忽略项、保留可覆盖运行项（含 CloseMosaic）
        tab.mosaic_spin.setValue(0.0)
        qt_app.processEvents()
        assert tab.close_mosaic_spin.isEnabled() is False

        tab.resume_chk.setChecked(True)
        qt_app.processEvents()
        assert tab.resume_hint_label.isHidden() is False
        for ignored_w in (
            tab.model_combo,
            tab.preset_combo,
            tab.import_json_btn,
            tab.epochs_spin,
            tab.nbs_spin,
            tab.optimizer_combo,
            tab.lr0_spin,
            tab.lrf_spin,
            tab.warmup_spin,
            tab.cos_lr_chk,
            tab.seed_spin,
            tab.deterministic_chk,
            tab.amp_chk,
            tab.mosaic_spin,
            tab.mixup_spin,
            tab.hsv_h_spin,
        ):
            assert ignored_w.isEnabled() is False
        for allowed_w in (
            tab.custom_model_edit,
            tab.imgsz_combo,
            tab.batch_spin,
            tab.workers_spin,
            tab.patience_spin,
            tab.cache_chk,
            tab.val_chk,
            tab.plots_chk,
            tab.close_mosaic_spin,
            tab.run_name_edit,
        ):
            assert allowed_w.isEnabled() is True

        # 续训期间切 CPU 再切回 GPU，取消续训后应完整恢复 AMP 勾选状态与常规防呆规则
        tab.device_ctrl.gpu_chk.setChecked(False)
        qt_app.processEvents()
        tab.device_ctrl.gpu_chk.setChecked(True)
        qt_app.processEvents()
        tab.resume_chk.setChecked(False)
        qt_app.processEvents()

        assert tab.resume_hint_label.isHidden() is True
        assert tab.model_combo.isEnabled() is True
        assert tab.epochs_spin.isEnabled() is True
        assert tab.amp_chk.isEnabled() is True
        assert tab.amp_chk.isChecked() is True
        assert tab.close_mosaic_spin.isEnabled() is False  # Mosaic 仍为 0.0
    finally:
        tab.close()

    # 验证在 CPU 下保存的 -1 自动探测意图：恢复时保持 CPU 安全值 4，重新勾选 GPU 后自动恢复为 -1
    tab_restored = TrainTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        tab_restored.restore_settings(settings)
        assert tab_restored.device_ctrl.device() == "cpu"
        assert tab_restored.batch_spin.minimum() == 1
        assert tab_restored.batch_spin.value() == 4
        tab_restored.device_ctrl.gpu_chk.setChecked(True)
        qt_app.processEvents()
        assert tab_restored.batch_spin.minimum() == -1
        assert tab_restored.batch_spin.value() == -1
    finally:
        tab_restored.close()


def test_eval_tab_dynamic_batch_tooltip_and_engine_cpu_button_guard(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证验证/导出页 ONNX 与 TensorRT 动态输入 Batch 提示及 TensorRT CPU 禁用导出按钮。"""
    tab = EvalTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        # ONNX + dynamic
        tab.format_combo.setCurrentIndex(0)
        tab.dynamic_chk.setChecked(True)
        qt_app.processEvents()
        assert "参考" in tab.export_batch_spin.toolTip()

        # TensorRT + dynamic
        tab.format_combo.setCurrentIndex(2)
        qt_app.processEvents()
        assert "最大批大小" in tab.export_batch_spin.toolTip()
        assert tab.export_btn.isEnabled() is True

        # 切 CPU -> TensorRT 导出按钮自动置灰；切回 ONNX -> 重新启用
        tab.device_ctrl.gpu_chk.setChecked(False)
        qt_app.processEvents()
        assert tab.export_btn.isEnabled() is False

        tab.format_combo.setCurrentIndex(0)  # onnx
        qt_app.processEvents()
        assert tab.export_btn.isEnabled() is True
    finally:
        tab.close()
