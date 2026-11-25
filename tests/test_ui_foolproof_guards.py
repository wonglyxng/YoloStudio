"""各工作流标签页防呆联动与前置校验单元测试。

职责：
    在离线 `offscreen` Qt 环境下系统性验证五个配置标签页的防呆交互逻辑：
    1. 「1. 视频抽帧」：选 `png` 置灰 `JPG质量`、起止时间段非法拦截、去重 0 值提示与列表增删按钮空态保护；
    2. 「3. LLM 智能标注」：已有 `.json` + 本地 SAM 模式下动态展示 GPU 开关、运行中禁用历史报告加载、
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
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QCoreApplication, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QListWidgetItem, QMessageBox

import yolo_studio.gui.tab_extract as tab_extract_mod
import yolo_studio.gui.tab_review as tab_review_mod
from yolo_studio.core.frame_extractor import ExtractionMode, ExtractionProgress, ExtractionResult
from yolo_studio.core.mllm_reviewer import (
    GEMINI_MODELS,
    OPENAI_MODELS,
    BoxRefineMode,
    MllmProvider,
    MllmReviewResult,
    ReasoningEffort,
)
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

        # 3) 空列表时移除选中、清空列表与开始抽帧均置灰；添加条目后清空与开始启用、选中后移除启用
        assert tab.remove_btn.isEnabled() is False
        assert tab.clear_btn.isEnabled() is False
        assert tab.start_btn.isEnabled() is False

        fake_video = tmp_path / "demo.mp4"
        fake_video.write_bytes(b"")
        item = QListWidgetItem("demo.mp4")
        item.setData(Qt.ItemDataRole.UserRole, str(fake_video))
        tab.video_list.addItem(item)
        tab._video_paths.append(fake_video)
        tab._refresh_list_action_states()

        assert tab.clear_btn.isEnabled() is True
        assert tab.start_btn.isEnabled() is True
        assert tab.remove_btn.isEnabled() is False

        tab.video_list.setCurrentRow(0)
        qt_app.processEvents()
        assert tab.remove_btn.isEnabled() is True

        # 3.5) 切换抽帧策略时同步切换策略参数栈页、行内参数标签及三个微调框启用状态
        assert tab.strategy_param_stack.currentIndex() == 0
        assert tab.strategy_param_label.text() == "间隔:"
        assert tab.interval_spin.isEnabled() is True
        assert tab.frame_step_spin.isEnabled() is False
        assert tab.target_count_spin.isEnabled() is False

        tab.mode_combo.setCurrentIndex(tab.mode_combo.findData(ExtractionMode.EVERY_N_FRAMES))
        qt_app.processEvents()
        assert tab.strategy_param_stack.currentIndex() == 1
        assert tab.strategy_param_label.text() == "步长:"
        assert tab.frame_step_spin.isEnabled() is True
        assert tab.interval_spin.isEnabled() is False

        tab.mode_combo.setCurrentIndex(tab.mode_combo.findData(ExtractionMode.TARGET_COUNT))
        qt_app.processEvents()
        assert tab.strategy_param_stack.currentIndex() == 2
        assert tab.strategy_param_label.text() == "目标:"
        assert tab.target_count_spin.isEnabled() is True
        tab.mode_combo.setCurrentIndex(tab.mode_combo.findData(ExtractionMode.INTERVAL_SECONDS))
        qt_app.processEvents()

        # 4) 截取结束时间 <= 开始时间（且 > 0）时提前拦截，不启动后台线程
        tab.start_sec_spin.setValue(10.0)
        tab.end_sec_spin.setValue(5.0)
        tab._on_start()
        assert len(started_workers) == 0
        assert any("结束时间必须大于开始时间" in m for m in warnings)

        # 5) 命名规则实时预览（含默认规则、格式联动、子目录联动、预设模板、普通前缀与非法规则前置拦截）
        tab.start_sec_spin.setValue(0.0)
        tab.end_sec_spin.setValue(0.0)
        assert "vc<32位视频SHA256>_f0000001500.jpg" in tab.naming_preview_label.text()

        tab.format_combo.setCurrentText("png")
        qt_app.processEvents()
        assert "vc<32位视频SHA256>_f0000001500.png" in tab.naming_preview_label.text()

        tab.subfolder_chk.setChecked(True)
        qt_app.processEvents()
        assert "demo/vc<32位视频SHA256>_f0000001500.png" in tab.naming_preview_label.text()
        tab.subfolder_chk.setChecked(False)
        qt_app.processEvents()

        preset_menu = tab.naming_preset_btn.menu()
        assert preset_menu is not None and len(preset_menu.actions()) >= 2
        preset_menu.actions()[1].trigger()
        qt_app.processEvents()
        assert tab.naming_rule_edit.text() == "{video}_{seq:06d}"
        assert "demo_000001.png" in tab.naming_preview_label.text()

        tab.naming_rule_edit.setText("cam1_")
        qt_app.processEvents()
        assert "cam1_demo_000001.png" in tab.naming_preview_label.text()

        tab.naming_rule_edit.setText("{video}_f{ms:010d}")
        qt_app.processEvents()
        assert "demo_f0000001500.png" in tab.naming_preview_label.text()

        warnings.clear()
        tab.naming_rule_edit.setText("{invalid_key}")
        qt_app.processEvents()
        assert "不支持的占位符" in tab.naming_preview_label.text()
        tab._on_start()
        assert len(started_workers) == 0
        assert any("不支持的占位符" in m for m in warnings)

        # 移除选中条目后列表再次为空，移除/清空/开始抽帧按钮均恢复置灰
        tab._on_remove_selected()
        qt_app.processEvents()
        assert len(tab._video_paths) == 0
        assert tab.remove_btn.isEnabled() is False
        assert tab.clear_btn.isEnabled() is False
        assert tab.start_btn.isEnabled() is False
    finally:
        tab.close()


def test_extract_tab_multiselect_preview_overall_progress_and_completion_flow(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证抽帧页多选与 Delete 删除、选中视频异步预览、多视频单调总进度及非打断式完成流转。"""
    def _write_mp4(path: Path, frames: int = 12) -> Path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 120))
        assert writer.isOpened()
        for i in range(frames):
            img = np.full((120, 160, 3), (i * 15, 180, 60), dtype=np.uint8)
            writer.write(img)
        writer.release()
        return path

    v1 = _write_mp4(tmp_path / "clip1.mp4", frames=15)
    v2 = _write_mp4(tmp_path / "clip2.mp4", frames=15)
    v3 = _write_mp4(tmp_path / "clip3.mp4", frames=15)

    class _NoopSignal:
        def connect(self, _slot) -> None:
            pass

    class _FakeExtractWorker:
        def __init__(self, paths, cfg) -> None:
            self.paths = paths
            self.cfg = cfg
            self.progress = _NoopSignal()
            self.finished_result = _NoopSignal()
            self.failed = _NoopSignal()

        def start(self) -> None:
            pass

        def isRunning(self) -> bool:
            return False

    monkeypatch.setattr(tab_extract_mod, "FrameExtractWorker", _FakeExtractWorker)

    tab = ExtractTab(workspace_root=tmp_path)
    synced_dirs: list[str] = []
    sent_dirs: list[str] = []
    tab.frames_dir_synced.connect(synced_dirs.append)
    tab.send_frames_dir.connect(sent_dirs.append)

    try:
        tab.resize(1280, 760)
        tab.show()
        tab._add_video_paths([v1, v2, v3])
        deadline = time.monotonic() + 5.0
        while tab.probe_worker and tab.probe_worker.isRunning() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.02)
        qt_app.processEvents()

        assert tab.video_list.count() == 3
        assert "共 3 个" in tab.video_group.title()

        # 选中视频后异步解码预览帧并写入画布（非空 pixmap）
        deadline = time.monotonic() + 5.0
        while not tab.preview_label.has_preview_image() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.02)
        qt_app.processEvents()
        assert tab.preview_label.has_preview_image() is True
        assert tab.preview_label.pixmap() is not None and not tab.preview_label.pixmap().isNull()

        # 多选前两项后按 Delete 键批量移除
        tab.video_list.item(0).setSelected(True)
        tab.video_list.item(1).setSelected(True)
        QTest.keyClick(tab.video_list, Qt.Key.Key_Delete)
        qt_app.processEvents()
        assert tab.video_list.count() == 1
        assert len(tab._video_paths) == 1

        # 多视频进度条按全局帧数权重单调推进，第 2 个视频不会归零
        tab._add_video_paths([v1])
        deadline = time.monotonic() + 5.0
        while tab.probe_worker and tab.probe_worker.isRunning() and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.02)
        qt_app.processEvents()
        assert tab.video_list.count() == 2

        tab._on_start()
        tab._on_progress(
            ExtractionProgress(
                video_index=1,
                total_videos=2,
                video_name="clip3.mp4",
                current_frame=15,
                total_frames=15,
                saved_total=3,
                skipped_similar=1,
            )
        )
        pct_after_v1 = tab.progress_bar.value()
        assert pct_after_v1 == 50

        tab._on_progress(
            ExtractionProgress(
                video_index=2,
                total_videos=2,
                video_name="clip1.mp4",
                current_frame=3,
                total_frames=15,
                saved_total=4,
                skipped_similar=1,
            )
        )
        assert tab.progress_bar.value() >= pct_after_v1
        assert tab.progress_bar.value() == 60

        # 默认完成后仅静默同步并高亮发送按钮，不强行切页，且多行完成状态完整展示不裁切
        out_dir = tmp_path / "out_frames"
        tab._extract_started_at = 0.0
        tab._on_finished(
            ExtractionResult(
                output_dir=out_dir,
                videos_processed=2,
                frames_saved=8,
                frames_skipped_similar=2,
            )
        )
        qt_app.processEvents()
        assert synced_dirs == [str(out_dir)]
        assert sent_dirs == []
        assert tab.send_next_btn.objectName() == "accentActionBtn"
        assert tab._card_output.value_label.text() == "00:00"
        assert tab.status_label.height() >= tab.status_label.heightForWidth(tab.status_label.width())

        # 勾选「抽帧完成后自动进入标注工作台」后，无错完成会发射 send_frames_dir
        tab.auto_open_workbench_chk.setChecked(True)
        tab._on_finished(
            ExtractionResult(
                output_dir=out_dir,
                videos_processed=2,
                frames_saved=8,
                frames_skipped_similar=2,
            )
        )
        assert len(synced_dirs) == 2
        assert sent_dirs == [str(out_dir)]
    finally:
        if tab.probe_worker and tab.probe_worker.isRunning():
            tab.probe_worker.cancel()
            tab.probe_worker.wait(2000)
        tab.close()


def test_review_tab_sam_device_visibility_precheck_and_busy_report_lock(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证智能标注页已有 `.json`+SAM 动态展示 GPU 开关、LLM 前置校验、分流按钮文案与运行期禁用历史报告。"""
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
        compat_idx = tab.provider_combo.findData(MllmProvider.OPENAI_COMPAT)
        assert compat_idx >= 0
        tab.provider_combo.setCurrentIndex(compat_idx)
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

        # 6) 零样本直接标注模式防呆联动：隐藏权重/推理/已有标注行、禁用不适用收框与策略选项、必填类别校验与 SAM GPU 联动
        from PyQt6.QtGui import QStandardItemModel
        from yolo_studio.core.mllm_reviewer import ReviewSourceMode

        crop_idx = tab.refine_mode_combo.findData(BoxRefineMode.MLLM_CROP)
        tab.refine_mode_combo.setCurrentIndex(crop_idx)
        qt_app.processEvents()
        assert tab.refine_mode_combo.currentData() == BoxRefineMode.MLLM_CROP

        zs_idx = tab.source_mode_combo.findData(ReviewSourceMode.LLM_ZERO_SHOT)
        assert zs_idx >= 0
        tab.source_mode_combo.setCurrentIndex(zs_idx)
        qt_app.processEvents()

        assert tab.weights_picker.isHidden() is True
        assert tab.yolo_params_widget.isHidden() is True
        assert tab.ann_picker.isHidden() is True
        assert tab.refine_mode_combo.currentData() == BoxRefineMode.OFF
        assert tab.check_missing_chk.isEnabled() is False
        assert tab.add_missing_chk.isEnabled() is False
        assert tab.crops_chk.isEnabled() is False
        assert tab.high_conf_spin.isEnabled() is False
        assert "必填" in tab.classes_edit.placeholderText()

        refine_model = tab.refine_mode_combo.model()
        assert isinstance(refine_model, QStandardItemModel)
        assert refine_model.item(tab.refine_mode_combo.findData(BoxRefineMode.OFF)).isEnabled() is True
        assert refine_model.item(tab.refine_mode_combo.findData(BoxRefineMode.SAM_LOCAL)).isEnabled() is True
        assert refine_model.item(tab.refine_mode_combo.findData(BoxRefineMode.FLAG_LOOSE)).isEnabled() is False
        assert refine_model.item(tab.refine_mode_combo.findData(BoxRefineMode.MLLM_CROP)).isEnabled() is False

        # 未填目标类别清单时拦截启动
        tab.classes_edit.clear()
        warnings.clear()
        prev_started = len(started_configs)
        tab._on_start()
        assert len(started_configs) == prev_started
        assert any("目标类别清单" in m for m in warnings)

        # 零样本 + 本地 SAM：GPU 开关迁移到 SAM 行下方启用，且自动勾选并锁定“全部保留框收紧”
        tab.refine_mode_combo.setCurrentIndex(sam_idx)
        qt_app.processEvents()
        assert tab.sam_device_holder.isHidden() is False
        assert tab.yolo_device_ctrl.isEnabled() is True
        assert tab.sam_refine_all_chk.isChecked() is True
        assert tab.sam_refine_all_chk.isEnabled() is False

        tab.classes_edit.setText("person, helmet")
        tab._on_start()
        assert len(started_configs) == prev_started + 1
        assert started_configs[-1].source_mode == ReviewSourceMode.LLM_ZERO_SHOT
        assert started_configs[-1].box_refine_mode == BoxRefineMode.SAM_LOCAL
        assert started_configs[-1].allowed_classes == ["person", "helmet"]

        # 切回基准模型模式后，全部策略与收框下拉项恢复可用
        tab._on_failed("reset")
        tab.source_mode_combo.setCurrentIndex(0)
        qt_app.processEvents()
        assert tab.check_missing_chk.isEnabled() is True
        assert tab.sam_refine_all_chk.isEnabled() is True
        assert refine_model.item(tab.refine_mode_combo.findData(BoxRefineMode.FLAG_LOOSE)).isEnabled() is True
        assert refine_model.item(tab.refine_mode_combo.findData(BoxRefineMode.MLLM_CROP)).isEnabled() is True

        # 7) SAM 3 文本自动标注模式：默认纯本地禁用 LLM 配置且免 API Key 启动；勾选 LLM 二次质检后恢复校验
        sam3_idx = tab.source_mode_combo.findData(ReviewSourceMode.SAM3_TEXT)
        assert sam3_idx >= 0
        tab.source_mode_combo.setCurrentIndex(sam3_idx)
        qt_app.processEvents()

        assert tab.weights_picker.isHidden() is False
        assert tab.yolo_params_widget.isHidden() is False
        assert tab.sam3_llm_review_chk.isHidden() is False
        assert tab.ann_picker.isHidden() is True
        assert tab.llm_group.isEnabled() is False
        assert tab.start_btn.text() == "开始 SAM 3 自动标注"
        assert "自动通过" in tab.high_conf_label.text()
        assert tab.high_conf_spin.isEnabled() is True
        assert tab.refine_mode_combo.isEnabled() is False
        assert tab.guideline_edit.isEnabled() is False

        # 训练回填 best.pt 在 SAM 3 模式下不覆写用户指定的 sam3.pt，但切回基准模型模式后可自动恢复 best.pt
        fake_sam3_pt = tmp_path / "sam3.pt"
        fake_sam3_pt.write_bytes(b"sam3")
        fake_best_pt = tmp_path / "best.pt"
        fake_best_pt.write_bytes(b"best")
        tab.weights_picker.setText(str(fake_sam3_pt))
        tab.set_weights_path(str(fake_best_pt))
        assert Path(tab.weights_picker.text()) == fake_sam3_pt

        # 清空 API Key 并切到 Gemini（无环境变量），纯本地 SAM 3 无需 Key 即可直接启动
        tab.provider_combo.setCurrentIndex(0)
        tab.api_key_edit.clear()
        tab.classes_edit.setText("人物=person, 安全帽=safety helmet")
        prev_started = len(started_configs)
        tab._on_start()
        assert len(started_configs) == prev_started + 1
        assert started_configs[-1].source_mode == ReviewSourceMode.SAM3_TEXT
        assert started_configs[-1].sam3_enable_llm_review is False

        # 勾选「启用 LLM 二次质检」后激活分组 2，缺失 Gemini Key 将被拦截
        tab._on_failed("reset")
        tab.sam3_llm_review_chk.setChecked(True)
        qt_app.processEvents()
        assert tab.llm_group.isEnabled() is True
        assert tab.start_btn.text() == "开始 LLM 智能标注"
        assert tab.guideline_edit.isEnabled() is True
        warnings.clear()
        prev_started = len(started_configs)
        tab._on_start()
        assert len(started_configs) == prev_started
        assert any("GEMINI_API_KEY" in m for m in warnings)

        # 切回基准模型模式时自动恢复此前记忆的 best.pt，再切回 SAM 3 时恢复 sam3.pt
        tab.source_mode_combo.setCurrentIndex(0)
        qt_app.processEvents()
        assert Path(tab.weights_picker.text()) == fake_best_pt
        tab.source_mode_combo.setCurrentIndex(sam3_idx)
        qt_app.processEvents()
        assert Path(tab.weights_picker.text()) == fake_sam3_pt
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


def test_review_tab_official_openai_and_reasoning_effort_persistence(
    qt_app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证独立 OpenAI 官方提供方表单联动、Key 校验及思考/推理级别配置组装与持久化往返。"""
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

        def start(self) -> None:
            started_configs.append(self.cfg)

        def isRunning(self) -> bool:
            return False

    monkeypatch.setattr(tab_review_mod, "MllmReviewWorker", _FakeReviewWorker)

    in_dir = tmp_path / "imgs"
    in_dir.mkdir()
    settings = QSettings(str(tmp_path / "review_openai.ini"), QSettings.Format.IniFormat)

    tab1 = MllmReviewTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        assert tab1.model_edit.text() == GEMINI_MODELS[0][0]
        assert tab1.preset_combo.count() == len(GEMINI_MODELS)
        tab1.source_mode_combo.setCurrentIndex(tab1.source_mode_combo.findData(tab_review_mod.ReviewSourceMode.EXISTING_JSON))
        tab1.set_images_dir(str(in_dir))
        tab1.output_picker.setText(str(tmp_path / "out"))

        # 切换到 OpenAI 官方：隐藏 Base URL，填充官方模型预设，缺 OPENAI_API_KEY 拦截启动
        openai_idx = tab1.provider_combo.findData(MllmProvider.OPENAI)
        assert openai_idx >= 0
        tab1.provider_combo.setCurrentIndex(openai_idx)
        qt_app.processEvents()

        assert tab1.base_url_edit.isHidden() is True
        assert tab1.base_url_edit.text() == ""
        assert tab1.model_edit.text() == OPENAI_MODELS[0][0]
        assert tab1.preset_combo.count() == len(OPENAI_MODELS)
        assert "OPENAI_API_KEY" in tab1.api_key_status_label.text()

        warnings.clear()
        tab1._on_start()
        assert len(started_configs) == 0
        assert any("OpenAI API Key" in w for w in warnings)

        # 填写 Key 并选择 HIGH 推理级别后无需 Base URL 即可启动，且构建配置正确
        tab1.api_key_edit.setText("sk-test-official")
        high_idx = tab1.reasoning_effort_combo.findData(ReasoningEffort.HIGH)
        assert high_idx >= 0
        tab1.reasoning_effort_combo.setCurrentIndex(high_idx)
        qt_app.processEvents()
        assert "high" in tab1.reasoning_effort_hint_label.text()

        tab1._on_start()
        assert len(started_configs) == 1
        assert started_configs[0].provider == MllmProvider.OPENAI
        assert started_configs[0].reasoning_effort == ReasoningEffort.HIGH

        tab1.save_settings(settings)
        settings.sync()
    finally:
        tab1.close()

    # 重建页面恢复配置：OpenAI 官方、模型与 HIGH 推理级别完整还原，且 Base URL 保持隐藏
    tab2 = MllmReviewTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        tab2.restore_settings(settings)
        qt_app.processEvents()
        assert tab2.provider_combo.currentData() == MllmProvider.OPENAI
        assert tab2.base_url_edit.isHidden() is True
        assert tab2.model_edit.text() == OPENAI_MODELS[0][0]
        assert tab2.reasoning_effort_combo.currentData() == ReasoningEffort.HIGH
    finally:
        tab2.close()


def test_combobox_wheel_guard_blocks_accidental_change_and_forwards_scroll(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证全局下拉框禁滚轮守卫阻止未展开下拉框滚轮切换并透传外层滚动区，同时保留弹窗列表滚动。"""
    from PyQt6.QtCore import QPoint, QPointF
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QComboBox, QScrollArea, QVBoxLayout, QWidget

    from yolo_studio.gui.common import install_combobox_wheel_guard
    from yolo_studio.gui.main_window import MainWindow

    assert install_combobox_wheel_guard(qt_app) is True

    def _wheel_down(widget: QWidget) -> QWheelEvent:
        pos = QPointF(6.0, 6.0)
        gpos = QPointF(widget.mapToGlobal(QPoint(6, 6)))
        return QWheelEvent(
            pos,
            gpos,
            QPoint(0, 0),
            QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.resize(320, 140)
    container = QWidget()
    lay = QVBoxLayout(container)
    plain_combo = QComboBox()
    plain_combo.addItems(["A", "B", "C"])
    edit_combo = QComboBox()
    edit_combo.setEditable(True)
    edit_combo.addItems(["X", "Y", "Z"])
    long_combo = QComboBox()
    long_combo.addItems([f"item_{i}" for i in range(40)])
    long_combo.setMaxVisibleItems(5)
    lay.addWidget(plain_combo)
    lay.addWidget(edit_combo)
    lay.addWidget(long_combo)
    spacer = QWidget()
    spacer.setMinimumHeight(420)
    lay.addWidget(spacer)
    scroll.setWidget(container)
    scroll.show()
    qt_app.processEvents()

    try:
        # 1) 普通下拉框与可编辑下拉框本体均不响应滚轮切项，且滚轮透传使外层 QScrollArea 滚动
        assert scroll.verticalScrollBar().value() == 0
        QApplication.sendEvent(plain_combo, _wheel_down(plain_combo))
        qt_app.processEvents()
        assert plain_combo.currentIndex() == 0
        after_first = scroll.verticalScrollBar().value()
        assert after_first > 0

        assert edit_combo.lineEdit() is not None
        QApplication.sendEvent(edit_combo.lineEdit(), _wheel_down(edit_combo.lineEdit()))
        qt_app.processEvents()
        assert edit_combo.currentIndex() == 0
        assert scroll.verticalScrollBar().value() > after_first

        # 2) 展开下拉列表弹出窗口后，列表 viewport 仍可正常用滚轮滚动
        long_combo.showPopup()
        qt_app.processEvents()
        popup_vp = long_combo.view().viewport()
        QApplication.sendEvent(popup_vp, _wheel_down(popup_vp))
        qt_app.processEvents()
        assert long_combo.view().verticalScrollBar().value() > 0
        assert long_combo.currentIndex() == 0
        long_combo.hidePopup()
    finally:
        scroll.close()

    # 3) 验证 MainWindow 各标签页的下拉框在滚轮下也不会意外变更当前选项
    settings = QSettings(str(tmp_path / "wheel_main.ini"), QSettings.Format.IniFormat)
    win = MainWindow(workspace_root=tmp_path, lazy_init=True, settings=settings)
    try:
        win.show()
        qt_app.processEvents()
        combos = [c for c in win.findChildren(QComboBox) if c.count() >= 2]
        assert len(combos) >= 8
        for combo in combos:
            before_idx = combo.currentIndex()
            QApplication.sendEvent(combo, _wheel_down(combo))
            if combo.isEditable() and combo.lineEdit() is not None:
                QApplication.sendEvent(combo.lineEdit(), _wheel_down(combo.lineEdit()))
            assert combo.currentIndex() == before_idx
    finally:
        win.close()


def test_review_tab_reasoning_effort_model_foolproof_guards(
    qt_app: QApplication, tmp_path: Path
) -> None:
    """验证 Gemini 4 与 OpenAI 模型的思考/推理级别防呆：禁用不支持项、切模型平滑回退及历史非法配置静默纠偏。"""
    import json

    from PyQt6.QtGui import QStandardItemModel

    tab = MllmReviewTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        model = tab.reasoning_effort_combo.model()
        assert isinstance(model, QStandardItemModel)

        def _item_enabled(effort: ReasoningEffort) -> bool:
            idx = tab.reasoning_effort_combo.findData(effort)
            assert idx >= 0
            item = model.item(idx)
            assert item is not None
            return item.isEnabled()

        # 1) 默认 gemini-4-hydrogen：OFF 置灰禁用，LOW/MEDIUM/HIGH/XHIGH 启用
        assert tab.model_edit.text() == "gemini-4-hydrogen"
        assert _item_enabled(ReasoningEffort.AUTO) is True
        assert _item_enabled(ReasoningEffort.OFF) is False
        assert _item_enabled(ReasoningEffort.LOW) is True
        assert _item_enabled(ReasoningEffort.MEDIUM) is True
        assert _item_enabled(ReasoningEffort.HIGH) is True
        assert _item_enabled(ReasoningEffort.XHIGH) is True

        # 2) 在 gemini-4-neon 下选中 XHIGH，随后切到 gemini-4-carbon / gemini-4-argon：
        #    XHIGH 与 OFF 置灰，当前值自动回退到 HIGH，且提示栏说明不支持 xhigh
        def _preset_index(model_id: str) -> int:
            return next(
                i
                for i in range(tab.preset_combo.count())
                if (tab.preset_combo.itemData(i) or ("", ""))[0] == model_id
            )

        tab.preset_combo.setCurrentIndex(_preset_index("gemini-4-neon"))
        tab.reasoning_effort_combo.setCurrentIndex(tab.reasoning_effort_combo.findData(ReasoningEffort.XHIGH))
        qt_app.processEvents()
        assert tab.reasoning_effort_combo.currentData() == ReasoningEffort.XHIGH

        tab.preset_combo.setCurrentIndex(_preset_index("gemini-4-carbon"))
        qt_app.processEvents()
        assert _item_enabled(ReasoningEffort.OFF) is False
        assert _item_enabled(ReasoningEffort.XHIGH) is False
        assert tab.reasoning_effort_combo.currentData() == ReasoningEffort.HIGH
        assert "xhigh" in tab.reasoning_effort_hint_label.text()
        assert "不支持" in tab.reasoning_effort_hint_label.text()

        # 同样验证 gemini-4-argon 禁用 XHIGH
        tab.preset_combo.setCurrentIndex(_preset_index("gemini-4-argon"))
        qt_app.processEvents()
        assert _item_enabled(ReasoningEffort.XHIGH) is False

        # 切回 gemini-4-hydrogen 后 XHIGH 恢复启用
        tab.preset_combo.setCurrentIndex(_preset_index("gemini-4-hydrogen"))
        qt_app.processEvents()
        assert _item_enabled(ReasoningEffort.XHIGH) is True

        # 3) 切到 OpenAI 官方（gpt-6.2-sol / gpt-6.2-astra）：OFF 置灰，LOW..XHIGH 启用；
        #    手输 gpt-4o 后自动锁回 AUTO 并禁用下拉框
        tab.provider_combo.setCurrentIndex(tab.provider_combo.findData(MllmProvider.OPENAI))
        qt_app.processEvents()
        assert _item_enabled(ReasoningEffort.OFF) is False
        assert _item_enabled(ReasoningEffort.XHIGH) is True
        tab.reasoning_effort_combo.setCurrentIndex(tab.reasoning_effort_combo.findData(ReasoningEffort.HIGH))
        qt_app.processEvents()

        tab.model_edit.setText("gpt-4o")
        qt_app.processEvents()
        assert tab.reasoning_effort_combo.currentData() == ReasoningEffort.AUTO
        assert tab.reasoning_effort_combo.isEnabled() is False
        assert "非思考直出" in tab.reasoning_effort_hint_label.text()

        # 4) 切到 OpenAI 兼容的硅基流动预设（Qwen2.5-VL）同样锁定为默认；切到 DashScope 后恢复可用
        tab.provider_combo.setCurrentIndex(tab.provider_combo.findData(MllmProvider.OPENAI_COMPAT))
        qt_app.processEvents()
        assert tab.reasoning_effort_combo.isEnabled() is True  # 默认第 1 项为 DashScope qwen-vl-max-latest
        sf_idx = next(
            i
            for i in range(tab.preset_combo.count())
            if "Qwen2.5-VL" in str((tab.preset_combo.itemData(i) or ("", ""))[0])
        )
        tab.preset_combo.setCurrentIndex(sf_idx)
        qt_app.processEvents()
        assert tab.reasoning_effort_combo.currentData() == ReasoningEffort.AUTO
        assert tab.reasoning_effort_combo.isEnabled() is False
    finally:
        tab.close()

    # 5) 恢复历史非法组合（gemini-4-carbon + xhigh）时静默纠偏为 HIGH 且不误置脏标记
    settings = QSettings(str(tmp_path / "legacy_illegal_effort.ini"), QSettings.Format.IniFormat)
    settings.setValue(
        tab_review_mod._REVIEW_SETTINGS_KEY,
        json.dumps(
            {
                "version": 1,
                "provider": MllmProvider.GEMINI.value,
                "model_name": "gemini-4-carbon",
                "reasoning_effort": ReasoningEffort.XHIGH.value,
            }
        ),
    )
    settings.sync()

    tab_restore = MllmReviewTab(workspace_root=tmp_path, device_info=_gpu_device_info())
    try:
        tab_restore.restore_settings(settings)
        qt_app.processEvents()
        assert tab_restore.model_edit.text() == "gemini-4-carbon"
        assert tab_restore.reasoning_effort_combo.currentData() == ReasoningEffort.HIGH
        assert tab_restore._settings_dirty is False
    finally:
        tab_restore.close()


