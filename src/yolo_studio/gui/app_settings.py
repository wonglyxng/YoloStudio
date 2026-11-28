"""桌面端全局共享设置模型、持久化存储与凭据校验工具。

职责：
    集中维护跨标签页共享的三类应用配置，避免不同页面各自维护一份或互相穿透读取控件：
    1. **AI 大模型配置**（模型接口、常用预设、Base URL、模型名称、思考/推理级别与仅驻留内存的 API Key），
       同时服务「2. 标注工作台」单框局部 AI 精修与「3. 自动智能标注」批量任务；
    2. **本地视觉模型配置**（默认本地 SAM 2.1 收框/点选权重、SAM 3 权重路径与 GPU 加速默认偏好），
       让工作台与自动智能标注共享同一套 SAM 模型；
    3. **通用偏好设置**（标注工作台自动保存/标签显示/保持视野默认状态、启动记住上次目录等）。

运行环境与协作边界：
    运行于 PyQt6 主线程，通过外部注入的 `QSettings` 持久化至 `app/global_settings_json`；
    出于凭据安全约束，手动输入的明文 `api_key` 仅在当前进程内存中共享，绝不写入磁盘配置。
    首次加载若尚未存在全局设置键，会自动从历史 `review_tab/state_json` 无损迁移已有 AI 与 SAM 配置。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import ipaddress
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from PyQt6.QtCore import QObject, QSettings, pyqtSignal as Signal

from yolo_studio.core.mllm_reviewer import (
    GEMINI_MODELS,
    OPENAI_MODELS,
    MllmProvider,
    MllmReviewConfig,
    ReasoningEffort,
    resolve_effective_reasoning_effort,
)

APP_SETTINGS_KEY = "app/global_settings_json"
LEGACY_REVIEW_SETTINGS_KEY = "review_tab/state_json"

_PROVIDER_DISPLAY_NAMES: dict[MllmProvider, str] = {
    MllmProvider.GEMINI: "Google Gemini",
    MllmProvider.OPENAI: "OpenAI 官方",
    MllmProvider.OPENAI_COMPAT: "OpenAI 兼容",
}


def is_local_or_lan_url(base_url: str) -> bool:
    """判断接口 Base URL 是否指向本机或局域网服务（如 Ollama / vLLM，通常免填 API Key）。"""
    raw = base_url.strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return False


@dataclass
class AppPreferences:
    """桌面端全局共享设置快照。

    字段分组与生命周期：
        - AI 大模型：`provider` / `base_url` / `model_name` / `custom_model` / `reasoning_effort` 均持久化；
          `api_key` 仅驻留当前会话内存，保存至 `QSettings` 时强制置空。
        - 本地视觉模型：`sam_model_name` 同时驱动工作台 SAM 点选/收框与智能标注页 SAM 收紧；
          `sam3_weights_path` 留空时由 `resolve_sam3_weights()` 自动发现 `weights/sam3.pt`。
        - 通用偏好：`workbench_*`、`remember_last_dir` 与 `default_prefer_gpu` 控制各页初始默认行为。
    """

    # AI 大模型全局配置
    provider: MllmProvider = MllmProvider.GEMINI
    base_url: str = ""
    model_name: str = GEMINI_MODELS[0][0]
    custom_model: bool = False
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO
    api_key: str = ""

    # 本地视觉模型配置
    sam_model_name: str = "sam2.1_t.pt"
    sam3_weights_path: str = ""

    # 通用偏好设置
    workbench_auto_save: bool = True
    workbench_show_labels: bool = True
    workbench_keep_view: bool = False
    remember_last_dir: bool = True
    default_prefer_gpu: bool = True

    def clone(self) -> AppPreferences:
        """返回一份独立的配置副本，避免调用方原地修改污染存储内部快照。"""
        return replace(self)


def normalize_llm_preferences(prefs: AppPreferences) -> AppPreferences:
    """按当前内置预设与模型能力矩阵规范化 AI 配置（含旧版下线模型迁移与思考级别平滑回退）。"""
    normalized = prefs.clone()
    provider = normalized.provider
    model = normalized.model_name.strip()
    if provider == MllmProvider.OPENAI_COMPAT:
        normalized.base_url = normalized.base_url.strip()
        normalized.model_name = model
    else:
        normalized.base_url = ""
        builtin_models = {
            model_id
            for model_id, _ in (GEMINI_MODELS if provider == MllmProvider.GEMINI else OPENAI_MODELS)
        }
        default_model = GEMINI_MODELS[0][0] if provider == MllmProvider.GEMINI else OPENAI_MODELS[0][0]
        if model in builtin_models:
            normalized.model_name = model
            normalized.custom_model = False
        elif model and normalized.custom_model:
            normalized.model_name = model
        else:
            normalized.model_name = default_model
            normalized.custom_model = False

    effective_effort, _ = resolve_effective_reasoning_effort(
        provider, normalized.model_name, normalized.reasoning_effort
    )
    normalized.reasoning_effort = effective_effort
    normalized.sam_model_name = normalized.sam_model_name.strip() or "sam2.1_t.pt"
    normalized.sam3_weights_path = normalized.sam3_weights_path.strip()
    return normalized


def validate_llm_credentials(prefs: AppPreferences) -> tuple[bool, str]:
    """统一校验当前全局 AI 配置是否满足调用远程或本地 LLM 的前置条件。

    返回 `(是否通过, 失败时的中文引导说明)`；通过时第二项为空字符串。
    """
    model_name = prefs.model_name.strip()
    if not model_name:
        return False, "请先在右上角「⚙ 设置 → AI 大模型」中填写模型名称。"

    manual_key = prefs.api_key.strip()
    if prefs.provider == MllmProvider.GEMINI:
        if not (manual_key or os.environ.get("GEMINI_API_KEY", "").strip()):
            return (
                False,
                "未检测到 Gemini API Key：请在右上角「⚙ 设置 → AI 大模型」中填写 API Key，"
                "或配置环境变量 GEMINI_API_KEY。",
            )
        return True, ""

    if prefs.provider == MllmProvider.OPENAI:
        if not (manual_key or os.environ.get("OPENAI_API_KEY", "").strip()):
            return (
                False,
                "未检测到 OpenAI API Key：请在右上角「⚙ 设置 → AI 大模型」中填写 API Key，"
                "或配置环境变量 OPENAI_API_KEY。",
            )
        return True, ""

    base_url = prefs.base_url.strip()
    if not base_url:
        return False, "请先在右上角「⚙ 设置 → AI 大模型」中填写 OpenAI 兼容接口的 Base URL。"
    if (
        not manual_key
        and not os.environ.get("OPENAI_API_KEY", "").strip()
        and not is_local_or_lan_url(base_url)
    ):
        return (
            False,
            "未检测到 API Key 或环境变量 OPENAI_API_KEY。\n"
            "请在右上角「⚙ 设置 → AI 大模型」中填写 API Key；如使用本地服务请确认 Base URL 为本地或局域网地址。",
        )
    return True, ""


def describe_api_key_status(prefs: AppPreferences) -> tuple[str, str]:
    """返回当前全局 AI 凭据就绪状态的 `(中文摘要, 文字颜色)`，供设置弹窗与智能标注页复用。"""
    if prefs.api_key.strip():
        return "已使用手动填写的 API Key（仅本次会话生效，不落盘明文）", "#1a7f37"

    if prefs.provider == MllmProvider.GEMINI:
        if os.environ.get("GEMINI_API_KEY", "").strip():
            return "已检测到环境变量 GEMINI_API_KEY，API Key 可保持留空", "#1a7f37"
        return "未检测到环境变量 GEMINI_API_KEY，请在设置中填写或配置环境变量", "#9a6700"

    has_openai_env = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if prefs.provider == MllmProvider.OPENAI:
        if has_openai_env:
            return "已检测到环境变量 OPENAI_API_KEY，API Key 可保持留空", "#1a7f37"
        return "未检测到环境变量 OPENAI_API_KEY，请在设置中填写或配置环境变量", "#9a6700"

    if has_openai_env:
        return "已检测到环境变量 OPENAI_API_KEY，API Key 可保持留空", "#1a7f37"
    if is_local_or_lan_url(prefs.base_url):
        return "当前为本地/局域网服务地址（如 Ollama / vLLM），API Key 可留空直接连接", "#0969da"
    return "未检测到环境变量 OPENAI_API_KEY，请在设置中填写或配置环境变量", "#9a6700"


def format_llm_summary(prefs: AppPreferences) -> str:
    """将当前全局 AI 配置格式化为单行紧凑摘要，供智能标注页与工作台提示展示。"""
    prov_name = _PROVIDER_DISPLAY_NAMES.get(prefs.provider, prefs.provider.value)
    model = prefs.model_name.strip() or "(未填写模型)"
    effort_text = "模型默认" if prefs.reasoning_effort == ReasoningEffort.AUTO else prefs.reasoning_effort.value
    if prefs.provider == MllmProvider.OPENAI_COMPAT and prefs.base_url.strip():
        parsed = urlparse(
            prefs.base_url.strip() if "://" in prefs.base_url else f"http://{prefs.base_url.strip()}"
        )
        host = parsed.netloc or prefs.base_url.strip()
        return f"{prov_name} ({host}) · {model} · 推理: {effort_text}"
    return f"{prov_name} · {model} · 推理: {effort_text}"


def build_llm_probe_config(prefs: AppPreferences, workspace_root: Path) -> MllmReviewConfig:
    """按全局 AI 设置构建用于连通性测试或单框精修的基础 `MllmReviewConfig`。"""
    return MllmReviewConfig(
        images_dir=workspace_root,
        output_dir=workspace_root / "data" / "smart_labeled",
        provider=prefs.provider,
        model_name=prefs.model_name.strip(),
        api_key=prefs.api_key.strip(),
        base_url=prefs.base_url.strip(),
        reasoning_effort=prefs.reasoning_effort,
        sam_model_name=prefs.sam_model_name.strip() or "sam2.1_t.pt",
    )


class AppSettingsStore(QObject):
    """桌面端全局共享配置存储中心（持有内存态、变更信号与 `QSettings` 持久化）。

    生命周期与线程约束：
        仅在 PyQt6 主线程读写；`preferences` 属性始终返回当前快照副本，
        修改必须通过 `set_preferences` / `update` / `restore` / `reset_defaults` 进行，
        以确保订阅标签页同步收到 `changed` 及细粒度分域信号。
    """

    changed = Signal()
    llm_changed = Signal()
    sam_changed = Signal()
    general_changed = Signal()

    def __init__(
        self,
        workspace_root: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace_root = workspace_root
        self._prefs = AppPreferences()
        self._dirty = False

    @property
    def preferences(self) -> AppPreferences:
        """返回当前全局设置的只读快照副本。"""
        return self._prefs.clone()

    @property
    def is_dirty(self) -> bool:
        """当前内存态是否包含尚未写入 `QSettings` 的用户改动。"""
        return self._dirty

    def set_preferences(self, new_prefs: AppPreferences, *, mark_dirty: bool = True) -> bool:
        """用新的 `AppPreferences` 更新存储，并在发生实际变化时发射细粒度与总变更信号。"""
        old = self._prefs
        updated = new_prefs.clone()
        updated.api_key = updated.api_key.strip()
        updated.base_url = updated.base_url.strip()
        updated.model_name = updated.model_name.strip()
        updated.sam_model_name = updated.sam_model_name.strip() or "sam2.1_t.pt"
        updated.sam3_weights_path = updated.sam3_weights_path.strip()

        llm_diff = (
            old.provider != updated.provider
            or old.base_url != updated.base_url
            or old.model_name != updated.model_name
            or old.custom_model != updated.custom_model
            or old.reasoning_effort != updated.reasoning_effort
            or old.api_key != updated.api_key
        )
        sam_diff = (
            old.sam_model_name != updated.sam_model_name
            or old.sam3_weights_path != updated.sam3_weights_path
        )
        gen_diff = (
            old.workbench_auto_save != updated.workbench_auto_save
            or old.workbench_show_labels != updated.workbench_show_labels
            or old.workbench_keep_view != updated.workbench_keep_view
            or old.remember_last_dir != updated.remember_last_dir
            or old.default_prefer_gpu != updated.default_prefer_gpu
        )
        if not (llm_diff or sam_diff or gen_diff):
            return False

        self._prefs = updated
        # 明文 API Key 不落盘，仅当持久化字段变化时置位脏标记
        persisted_diff = (
            old.provider != updated.provider
            or old.base_url != updated.base_url
            or old.model_name != updated.model_name
            or old.custom_model != updated.custom_model
            or old.reasoning_effort != updated.reasoning_effort
            or sam_diff
            or gen_diff
        )
        if mark_dirty and persisted_diff:
            self._dirty = True
        if llm_diff:
            self.llm_changed.emit()
        if sam_diff:
            self.sam_changed.emit()
        if gen_diff:
            self.general_changed.emit()
        self.changed.emit()
        return True

    def update(self, *, mark_dirty: bool = True, **fields: object) -> bool:
        """按关键字增量更新部分配置字段并触发对应变更信号。"""
        updated = replace(self._prefs, **fields)  # type: ignore[arg-type]
        return self.set_preferences(updated, mark_dirty=mark_dirty)

    def reset_defaults(self, *, keep_api_key: bool = False) -> bool:
        """将全部全局设置恢复为出厂默认值。"""
        fresh = AppPreferences()
        if keep_api_key:
            fresh.api_key = self._prefs.api_key
        return self.set_preferences(fresh, mark_dirty=True)

    def save(self, settings: QSettings, *, force: bool = False) -> None:
        """将全局设置持久化写入 `QSettings`（强制剥离明文 `api_key`，绝不落盘保存密钥）。"""
        if not force and not self._dirty:
            return
        p = self._prefs
        payload: dict[str, object] = {
            "version": 1,
            "provider": p.provider.value,
            "base_url": p.base_url.strip(),
            "model_name": p.model_name.strip(),
            "custom_model": bool(p.custom_model),
            "reasoning_effort": p.reasoning_effort.value,
            "sam_model_name": p.sam_model_name.strip() or "sam2.1_t.pt",
            "sam3_weights_path": p.sam3_weights_path.strip(),
            "workbench_auto_save": bool(p.workbench_auto_save),
            "workbench_show_labels": bool(p.workbench_show_labels),
            "workbench_keep_view": bool(p.workbench_keep_view),
            "remember_last_dir": bool(p.remember_last_dir),
            "default_prefer_gpu": bool(p.default_prefer_gpu),
        }
        settings.setValue(APP_SETTINGS_KEY, json.dumps(payload, ensure_ascii=False))
        self._dirty = False

    def restore(self, settings: QSettings) -> bool:
        """从 `QSettings` 恢复全局设置；若尚无新版键则尝试从旧版 `review_tab/state_json` 无损迁移。

        返回值：
            成功解析并恢复（或迁移）配置时返回 `True`，无配置或数据损坏时保持默认并返回 `False`。
            恢复过程不置位 `_dirty`，防止空打开窗口覆写用户配置。
        """
        raw = settings.value(APP_SETTINGS_KEY, "")
        migrated_from_legacy = False
        if not raw or not isinstance(raw, str):
            raw = settings.value(LEGACY_REVIEW_SETTINGS_KEY, "")
            migrated_from_legacy = True
        if not raw or not isinstance(raw, str):
            return False
        try:
            state = json.loads(raw)
        except Exception:
            return False
        if not isinstance(state, dict):
            return False

        prefs = AppPreferences()
        prov_val = str(state.get("provider") or "")
        for prov in MllmProvider:
            if prov.value == prov_val:
                prefs.provider = prov
                break

        saved_url = str(state.get("base_url") or "").strip()
        saved_model = str(state.get("model_name") or "").strip()
        if migrated_from_legacy:
            schema_version = int(state.get("version") or 1)
            explicit_custom = bool(state.get("custom_model", False)) if schema_version >= 2 else False
        else:
            explicit_custom = bool(state.get("custom_model", False))

        prefs.base_url = saved_url
        prefs.model_name = saved_model
        prefs.custom_model = explicit_custom

        effort_val = str(state.get("reasoning_effort") or "")
        for effort in ReasoningEffort:
            if effort.value == effort_val:
                prefs.reasoning_effort = effort
                break

        sam_model = str(state.get("sam_model_name") or "").strip()
        if sam_model:
            prefs.sam_model_name = sam_model
        if not migrated_from_legacy:
            prefs.sam3_weights_path = str(state.get("sam3_weights_path") or "").strip()
            if "workbench_auto_save" in state:
                prefs.workbench_auto_save = bool(state["workbench_auto_save"])
            if "workbench_show_labels" in state:
                prefs.workbench_show_labels = bool(state["workbench_show_labels"])
            if "workbench_keep_view" in state:
                prefs.workbench_keep_view = bool(state["workbench_keep_view"])
            if "remember_last_dir" in state:
                prefs.remember_last_dir = bool(state["remember_last_dir"])
            if "default_prefer_gpu" in state:
                prefs.default_prefer_gpu = bool(state["default_prefer_gpu"])

        normalized = normalize_llm_preferences(prefs)
        # 绝不从磁盘恢复明文 API Key
        normalized.api_key = ""
        self.set_preferences(normalized, mark_dirty=False)
        self._dirty = False
        return True
