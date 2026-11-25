"""基于多模态大语言模型（LLM）的智能标注、质检精修与漏标发现引擎。

职责：
    串联“候选框发现（基准 YOLO 模型低阈值推理、读取已有 X-AnyLabeling `.json`、LLM 零样本直接标注，
    或本地 SAM 3 文本概念自动标注）→ 带编号总览图与局部裁剪图渲染（零样本模式直接发送干净原图）→
    多模态大模型结构化识别与质检精修（纯本地 SAM 3 模式可跳过 LLM）→ 决策合并去重与可选本地 SAM 像素级收边 →
    结果分流落盘（`auto_passed` / `llm_fixed` / `need_review`）与前后对比图生成”。

运行环境：
    Python >= 3.11，依赖 `opencv-python`、`numpy`、`Pillow`、`pydantic`；
    按模式额外需要 `ultralytics`（`yolo_predict`）、`google-genai`（Gemini）或 `openai`（OpenAI 兼容接口）。
    调用远程接口需通过配置或环境变量 `GEMINI_API_KEY` / `OPENAI_API_KEY` 提供有效凭据。

协作对象与职责边界：
    为 `cli._cmd_review`、`gui.workers.MllmReviewWorker` 与 `gui.tab_review.MllmReviewTab`
    提供核心管线，复用 `dataset_converter` 的图片扫描与标注文件匹配函数。
    仅负责智能标注、X-AnyLabeling `.json` 产出与汇总报告写出，不执行数据集划分或模型训练。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Literal, Protocol

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from yolo_studio.core.dataset_converter import (
    find_images,
    resolve_label_file,
    stem_collisions,
)

# 分流桶目录名；`need_review` 同时被 GUI 数据集页用作默认排除的未复核子目录
BUCKET_AUTO_PASSED = "auto_passed"
BUCKET_LLM_FIXED = "llm_fixed"
BUCKET_NEED_REVIEW = "need_review"
TRIAGE_BUCKETS = (BUCKET_AUTO_PASSED, BUCKET_LLM_FIXED, BUCKET_NEED_REVIEW)


class ReviewSourceMode(str, Enum):
    """LLM 智能标注的候选标注来源模式。"""

    YOLO_PREDICT = "yolo_predict"
    EXISTING_JSON = "existing_json"
    LLM_ZERO_SHOT = "llm_zero_shot"
    SAM3_TEXT = "sam3_text"


SOURCE_MODE_ITEMS: list[tuple[ReviewSourceMode, str]] = [
    (ReviewSourceMode.YOLO_PREDICT, "基准模型预标 + LLM 质检精修（推荐）"),
    (ReviewSourceMode.EXISTING_JSON, "已有标注 (.json) + LLM 质检精修"),
    (ReviewSourceMode.LLM_ZERO_SHOT, "LLM 零样本直接标注（冷启动，无需模型）"),
    (ReviewSourceMode.SAM3_TEXT, "SAM 3 文本自动标注（本地免费 / 可选 LLM 质检）"),
]

SOURCE_MODE_HINTS: dict[ReviewSourceMode, str] = {
    ReviewSourceMode.YOLO_PREDICT: (
        "先用基准 .pt 低阈值推理拉高召回，再由 LLM 剔除误检、改类、补漏与收框；"
        "尚无模型时可切换「LLM 零样本直接标注」或「SAM 3 文本自动标注」冷启动。"
    ),
    ReviewSourceMode.EXISTING_JSON: (
        "读取已有 X-AnyLabeling .json 标注，由 LLM 质检、补漏与收框，并将疑难样本分流到 need_review。"
    ),
    ReviewSourceMode.LLM_ZERO_SHOT: (
        "无需预训练模型或已有标注，必填「目标类别清单」由 LLM 直接全图定位目标（推荐配合「本地 SAM 像素级收紧」自动贴边）；"
        "检出有目标的图片统一进入 need_review 供工作台快速过目确认。"
    ),
    ReviewSourceMode.SAM3_TEXT: (
        "加载本地 sam3.pt，按「目标类别清单」（支持“类别名=英文提示词”如 头盔=safety helmet）直接输出像素级贴边框；"
        "默认纯本地零 API 费用，勾选「启用 LLM 二次质检」即可叠加复杂业务规则删误检、改类与补漏。"
    ),
}


def parse_sam3_class_prompts(raw_entries: list[str]) -> tuple[list[str], list[str]]:
    """解析 SAM 3 目标类别与文本提示词清单，支持 `类别名` 或 `类别名=英文提示词` 语法。

    示例：
        `["person", "头盔=safety helmet"]` → `(["person", "头盔"], ["person", "safety helmet"])`。
        保留首次出现的类别顺序，忽略空白项及重复类别名。
    """
    class_names: list[str] = []
    text_prompts: list[str] = []
    seen: set[str] = set()
    for raw in raw_entries:
        item = raw.strip()
        if not item:
            continue
        normalized = item.replace("＝", "=")
        if "=" in normalized:
            cls_part, prompt_part = normalized.split("=", 1)
            cls_name = cls_part.strip()
            prompt = prompt_part.strip() or cls_name
        else:
            cls_name = item
            prompt = item
        if not cls_name or cls_name in seen:
            continue
        seen.add(cls_name)
        class_names.append(cls_name)
        text_prompts.append(prompt)
    return class_names, text_prompts


class MllmProvider(str, Enum):
    """支持的多模态大模型接口提供方。"""

    GEMINI = "gemini"
    OPENAI = "openai"
    OPENAI_COMPAT = "openai_compat"


class ReasoningEffort(str, Enum):
    """多模态大模型的统一思考/推理深度级别。

    说明：
        `AUTO` 不向服务端显式传入思考级别，由目标模型按自身默认策略执行；
        其余级别会按接口与模型家族映射为 Gemini `thinking_level` 或 OpenAI `reasoning_effort`，
        并在遇到不兼容的 400 响应时自动单调降级至受支持档位或省略参数。
    """

    AUTO = "auto"
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


REASONING_EFFORT_ITEMS: list[tuple[ReasoningEffort, str]] = [
    (ReasoningEffort.AUTO, "模型默认（不显式指定，推荐）"),
    (ReasoningEffort.OFF, "关闭 / 极简（关闭思考或 minimal，最快）"),
    (ReasoningEffort.LOW, "低（low，轻量思考、速度优先）"),
    (ReasoningEffort.MEDIUM, "中（medium，速度与精度均衡）"),
    (ReasoningEffort.HIGH, "高（high，复杂场景深度推理）"),
    (ReasoningEffort.XHIGH, "极高（xhigh，仅旗舰推理模型支持）"),
]

REASONING_EFFORT_HINTS: dict[ReasoningEffort, str] = {
    ReasoningEffort.AUTO: "不显式传入思考级别参数，遵循所选模型服务端默认推理深度。",
    ReasoningEffort.OFF: "尽量关闭长思考或降至极简（Gemini 映射 minimal/low，OpenAI 推理模型映射 none/low）。",
    ReasoningEffort.LOW: "显式设为 low：轻量思考以降低延迟和 Token 消耗，适合常规清晰画面。",
    ReasoningEffort.MEDIUM: "显式设为 medium：兼顾标注准确率与响应速度，适合多数复杂检测场景。",
    ReasoningEffort.HIGH: "显式设为 high：最大化深度思考，适合密集遮挡、易混类别与复杂规则判定。",
    ReasoningEffort.XHIGH: "显式设为 xhigh：仅旗舰推理模型支持；若当前模型不支持则自动降级为 high。",
}


class BoxRefineMode(str, Enum):
    """已有候选框的边框紧贴标准化修正模式。"""

    OFF = "off"
    FLAG_LOOSE = "flag_loose"
    MLLM_CROP = "mllm_crop"
    SAM_LOCAL = "sam_local"


BOX_REFINE_MODE_ITEMS: list[tuple[BoxRefineMode, str]] = [
    (BoxRefineMode.OFF, "关闭（仅改错类 / 删误检 / 补漏）"),
    (BoxRefineMode.FLAG_LOOSE, "仅标记过松框 → 转入待人工复核"),
    (BoxRefineMode.MLLM_CROP, "LLM 局部放大图精修（懂业务规则，零额外模型）"),
    (BoxRefineMode.SAM_LOCAL, "本地 SAM 像素级收紧（轮廓最贴边 + 安全门禁）"),
]

BOX_REFINE_MODE_HINTS: dict[BoxRefineMode, str] = {
    BoxRefineMode.OFF: "保持原有框坐标不变，仅执行删误检、改错类与漏标检查，速度最快。",
    BoxRefineMode.FLAG_LOOSE: "不改坐标，仅由 LLM 识别偏松/截断框并分流到 need_review 供人工精修。",
    BoxRefineMode.MLLM_CROP: "强制附带局部放大图，由 LLM 在特写中输出更紧贴坐标（含安全门禁防缩坏，略增 Token）。",
    BoxRefineMode.SAM_LOCAL: "本地加载 SAM 零额外 API 费用像素级收边（首次自动下载权重），超门禁自动回退原框并转待复核。",
}

SAM_REFINE_MODELS: list[tuple[str, str]] = [
    ("sam2.1_t.pt", "SAM 2.1 Tiny (速度快，显存占用最低，推荐)"),
    ("sam2.1_s.pt", "SAM 2.1 Small (速度与贴边精度均衡)"),
    ("sam2.1_b.pt", "SAM 2.1 Base (更高边缘精度)"),
    ("mobile_sam.pt", "MobileSAM (超轻量快速分割)"),
]

# 本地 SAM 3 自动发现默认权重名顺序及对应的魔搭（ModelScope）官方公开镜像仓库
SAM3_DEFAULT_WEIGHT_NAMES: tuple[str, ...] = ("sam3.pt", "sam3.1_multiplex.pt")
SAM3_MODELSCOPE_REPOS: dict[str, str] = {
    "sam3.pt": "facebook/sam3",
    "sam3.1_multiplex.pt": "facebook/sam3.1",
}


def resolve_sam3_weights(
    weights_path: Path | None = None,
    workspace_root: Path | None = None,
) -> Path | None:
    """解析本地 SAM 3 / SAM 3.1 权重文件绝对路径；未命中时返回 `None`。

    查找规则：
        - 若显式传入 `weights_path`：仅当该文件存在且非空时返回其绝对路径，不静默回退其他文件；
        - 若 `weights_path` 为 `None`：按目录优先级依次检索环境变量 `YOLO_STUDIO_WEIGHTS_DIR`、
          `workspace_root / "weights"`、`Path.cwd() / "weights"`、应用代码根 `weights/`
          及独立便携包根 `<便携包根>/weights/`；在每个目录内按 `SAM3_DEFAULT_WEIGHT_NAMES`
          顺序（优先 `sam3.pt`，其次 `sam3.1_multiplex.pt`）查找非空文件。
          目录优先级高于文件名优先级，保证工作区权重不被仓库默认权重覆盖。
    """
    if weights_path is not None:
        return weights_path.resolve() if (weights_path.is_file() and weights_path.stat().st_size > 0) else None

    from yolo_studio.core.trainer import WEIGHTS_DIR_ENV_VAR

    code_root = Path(__file__).resolve().parents[3]
    search_dirs: list[Path] = []
    env_weights = os.environ.get(WEIGHTS_DIR_ENV_VAR, "").strip()
    if env_weights:
        search_dirs.append(Path(env_weights).expanduser())
    if workspace_root is not None:
        search_dirs.append(workspace_root / "weights")
    search_dirs.extend(
        [
            Path.cwd() / "weights",
            code_root / "weights",
            code_root.parent / "weights",
        ]
    )

    seen: set[Path] = set()
    for folder in search_dirs:
        try:
            resolved_dir = folder.resolve()
        except OSError:
            continue
        if resolved_dir in seen:
            continue
        seen.add(resolved_dir)
        for name in SAM3_DEFAULT_WEIGHT_NAMES:
            candidate = resolved_dir / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate.resolve()
    return None


def download_sam3_from_modelscope(
    dest_dir: Path,
    filename: str = "sam3.pt",
    repo_id: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_event: Event | None = None,
) -> Path:
    """从魔搭社区（ModelScope）公开镜像下载 SAM 3 / SAM 3.1 `.pt` 权重并校验文件大小与 SHA-256。

    断点续传与原子落盘：
        先写入 `<filename>.part` 临时文件并支持 HTTP Range 续传；全部接收并完成体积与 SHA-256 校验后
        原子重命名为目标 `.pt` 文件。取消下载或校验失败时抛出 `RuntimeError`。
    """
    import httpx

    resolved_repo = (repo_id or "").strip() or SAM3_MODELSCOPE_REPOS.get(filename, "facebook/sam3")
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    part = dest_dir / f"{filename}.part"

    with httpx.Client(timeout=httpx.Timeout(30.0, read=60.0), follow_redirects=True) as client:
        meta_resp = client.get(f"https://modelscope.cn/api/v1/models/{resolved_repo}/repo/files")
        meta_resp.raise_for_status()
        meta_json = meta_resp.json()
        files = meta_json.get("Data", {}).get("Files", []) if isinstance(meta_json.get("Data"), dict) else []
        finfo = next((f for f in files if f.get("Name") == filename), None)
        if not finfo:
            raise FileNotFoundError(f"魔搭仓库 {resolved_repo} 中未找到权重文件 {filename}。")

        expected_size = int(finfo.get("Size") or 0)
        expected_sha256 = str(finfo.get("Sha256") or "").strip().lower()
        if target.is_file() and expected_size > 0 and target.stat().st_size == expected_size:
            if progress_callback:
                progress_callback(expected_size, expected_size)
            return target.resolve()

        url = f"https://modelscope.cn/api/v1/models/{resolved_repo}/repo?Revision=master&FilePath={filename}"
        downloaded = part.stat().st_size if part.is_file() else 0
        if expected_size > 0 and downloaded > expected_size:
            part.unlink()
            downloaded = 0

        for attempt in range(1, 6):
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("已取消 SAM 3 权重下载。")
            downloaded = part.stat().st_size if part.is_file() else 0
            if expected_size > 0 and downloaded >= expected_size:
                break
            headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
            mode = "ab" if downloaded > 0 else "wb"
            try:
                with client.stream("GET", url, headers=headers) as resp:
                    if resp.status_code not in (200, 206):
                        raise RuntimeError(f"魔搭下载响应异常: HTTP {resp.status_code}")
                    if resp.status_code == 200 and downloaded > 0:
                        downloaded = 0
                        mode = "wb"
                    with open(part, mode) as fh:
                        for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                            if cancel_event and cancel_event.is_set():
                                raise RuntimeError("已取消 SAM 3 权重下载。")
                            if not chunk:
                                continue
                            fh.write(chunk)
                            downloaded += len(chunk)
                            if progress_callback and expected_size > 0:
                                progress_callback(downloaded, expected_size)
            except RuntimeError:
                raise
            except Exception:
                if attempt >= 5:
                    raise
                time.sleep(2.0)

    actual_size = part.stat().st_size if part.is_file() else 0
    if expected_size > 0 and actual_size != expected_size:
        raise RuntimeError(f"SAM 3 权重下载不完整：已下载 {actual_size} 字节，预期 {expected_size} 字节。")

    if expected_sha256:
        digest = hashlib.sha256()
        with open(part, "rb") as fh:
            while True:
                buf = fh.read(8 * 1024 * 1024)
                if not buf:
                    break
                digest.update(buf)
        got_sha = digest.hexdigest().lower()
        if got_sha != expected_sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"SAM 3 权重 SHA-256 校验失败：期望 {expected_sha256}，实际 {got_sha}。")

    part.replace(target)
    return target.resolve()


OPENAI_OFFICIAL_BASE_URL = "https://api.openai.com/v1"

OPENAI_MODELS: list[tuple[str, str]] = [
    ("gpt-6.2-sol", "GPT-6.2 Sol（高效均衡、高性价比，推荐）"),
    ("gpt-6.2-astra", "GPT-6.2 Astra（旗舰前沿、复杂场景最高精度）"),
]

OPENAI_PRESETS: list[tuple[str, str, str]] = [
    ("通义千问 DashScope (Qwen-VL)", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max-latest"),
    ("硅基流动 SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-VL-72B-Instruct"),
    ("本地 Ollama", "http://localhost:11434/v1", "qwen2.5vl:7b"),
    ("本地 vLLM", "http://localhost:8000/v1", "Qwen/Qwen2.5-VL-7B-Instruct"),
]

GEMINI_MODELS: list[tuple[str, str]] = [
    ("gemini-4-hydrogen", "Gemini 4 Hydrogen（快速高效、成本最低，推荐）"),
    ("gemini-4-neon", "Gemini 4 Neon（速度与精度均衡）"),
    ("gemini-4-argon", "Gemini 4 Argon（低 Token 开销高效推理）"),
    ("gemini-4-carbon", "Gemini 4 Carbon（旗舰重型、复杂场景最高精度）"),
]


# 常见模型输出同义词映射：OpenAI 兼容接口仅开 json_object 时可能产生非枚举评价词，前置归并为合法枚举
_LOOSE_QUALITY_ALIASES = frozenset(
    {"oversized", "too_large", "too_big", "too_loose", "wide", "loose_fit"}
)
_PARTIAL_QUALITY_ALIASES = frozenset(
    {"clipped", "cropped", "truncated", "cut_off", "cutoff", "incomplete", "too_small", "too_tight", "undersized"}
)
_ACTION_ALIASES = {
    "retain": "keep",
    "remove": "delete",
    "drop": "delete",
    "discard": "delete",
    "rename": "relabel",
    "change_label": "relabel",
}


def _coerce_box_2d_ints(value: object) -> object:
    """将归一化坐标列表中的浮点数或数字字符串四舍五入转为整数，非预期结构保留原值交由 Pydantic 拦截。"""
    if not isinstance(value, (list, tuple)):
        return value
    coerced: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return value
        if isinstance(item, int):
            coerced.append(item)
        elif isinstance(item, float):
            coerced.append(int(round(item)))
        elif isinstance(item, str):
            try:
                coerced.append(int(round(float(item.strip()))))
            except ValueError:
                return value
        else:
            return value
    return coerced


class BoxDecision(BaseModel):
    """针对单个已有候选框的质检与边框紧贴评估决策。"""

    box_id: int = Field(description="Candidate box ID shown in the overview image.")
    action: Literal["keep", "relabel", "delete"] = Field(
        description="keep if label and object are valid; relabel if wrong class; delete if false positive/background."
    )
    new_label: str | None = Field(
        default=None,
        description="Corrected class name from the allowed class list when action is relabel.",
    )
    box_quality: Literal["tight", "loose", "partial"] = Field(
        default="tight",
        description="tight if box fits object tightly; loose if too much background is included; partial if clipped.",
    )
    refined_box_2d: list[int] | None = Field(
        default=None,
        description=(
            "Optional tight [ymin, xmin, ymax, xmax] normalized to 0-1000 inside the zoomed crop image "
            "for this box_id when box_quality is loose or partial."
        ),
    )
    reason: str = Field(default="", description="Brief explanation in Chinese or English.")

    @field_validator("box_id", mode="before")
    @classmethod
    def _normalize_box_id(cls, value: object) -> object:
        """清理模型在 `box_id` 字符串中附带的方括号或 `#` 前缀（如 `"[0]"`、`"#0"`）。"""
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1].strip()
            cleaned = cleaned.lstrip("#").strip()
            return cleaned
        return value

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: object) -> object:
        """规范化大小写与常见动作同义词，明显非法取值仍交由 `Literal` 校验拦截。"""
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            return _ACTION_ALIASES.get(normalized, normalized)
        return value

    @field_validator("box_quality", mode="before")
    @classmethod
    def _normalize_box_quality(cls, value: object) -> str:
        """将自由文本评价词或 `null` 前置映射为 `tight`/`loose`/`partial`，避免整图质检因次要评价词报错中断。"""
        if not isinstance(value, str):
            return "tight"
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"tight", "loose", "partial"}:
            return normalized
        if normalized in _LOOSE_QUALITY_ALIASES:
            return "loose"
        if normalized in _PARTIAL_QUALITY_ALIASES:
            return "partial"
        return "tight"

    @field_validator("refined_box_2d", mode="before")
    @classmethod
    def _normalize_refined_box_2d(cls, value: object) -> object:
        """把空列表视为 `None`，并将局部精修坐标中的浮点数四舍五入转为整数。"""
        if value is None or value == []:
            return None
        return _coerce_box_2d_ints(value)

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, value: object) -> str:
        """将缺失或 `null` 原因安全转换为空字符串。"""
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)


class MissingBox(BaseModel):
    """LLM 检出的漏标目标框（`box_2d` 为 `[ymin, xmin, ymax, xmax]`，归一化至 `0..1000`）。"""

    label: str = Field(description="Class name from the allowed class list.")
    box_2d: list[int] = Field(
        description="2D bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000."
    )
    reason: str = Field(default="", description="Why this object was missed.")

    @field_validator("box_2d", mode="before")
    @classmethod
    def _normalize_box_2d(cls, value: object) -> object:
        """将漏标框坐标中的浮点数或数字字符串四舍五入转为整数。"""
        return _coerce_box_2d_ints(value)

    @field_validator("reason", mode="before")
    @classmethod
    def _normalize_reason(cls, value: object) -> str:
        """将缺失或 `null` 原因安全转换为空字符串。"""
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)


class MllmReviewResponse(BaseModel):
    """期望 LLM 返回的结构化质检响应模式。"""

    box_reviews: list[BoxDecision] = Field(default_factory=list)
    has_missing: bool = Field(
        default=False,
        description="True if there is any unannotated target object belonging to the allowed classes.",
    )
    missing_note: str = Field(
        default="",
        description="Description of any missed objects or locations when has_missing is True.",
    )
    added_boxes: list[MissingBox] = Field(default_factory=list)

    @field_validator("box_reviews", "added_boxes", mode="before")
    @classmethod
    def _normalize_list_fields(cls, value: object) -> object:
        """兼容模型显式返回 `null` 的列表字段并归一为空列表。"""
        return [] if value is None else value

    @field_validator("has_missing", mode="before")
    @classmethod
    def _normalize_has_missing(cls, value: object) -> object:
        """兼容模型显式返回 `null` 的漏标布尔字段并归一为 `False`。"""
        return False if value is None else value

    @field_validator("missing_note", mode="before")
    @classmethod
    def _normalize_missing_note(cls, value: object) -> str:
        """将缺失或 `null` 漏标说明安全转换为空字符串。"""
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)


@dataclass
class CandidateBox:
    """内部使用的像素坐标系边界框（`x1, y1, x2, y2` 单位为原图像素）。"""

    box_id: int
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None
    # 状态取值：original | kept | relabeled | relabel_invalid | deleted | llm_added | refined | refine_suspect；
    # relabel_invalid 表示模型要求改类但给出的新类别不在允许清单中，保留原类别并强制进入人工复核；
    # refine_suspect 表示判定过松或自动收框幅度超出安全门禁，保留原框并强制进入人工复核
    status: str = "original"
    reason: str = ""
    original_label: str = ""
    refined: bool = False
    original_xyxy: tuple[float, float, float, float] | None = None


@dataclass
class ImageReviewRecord:
    """单张图片智能标注前后的处理结果与统计记录。"""

    image_path: Path
    output_image_path: Path | None
    output_json_path: Path | None
    triage_bucket: str  # 分流桶取值：auto_passed | llm_fixed | need_review
    before_boxes: list[CandidateBox] = field(default_factory=list)
    after_boxes: list[CandidateBox] = field(default_factory=list)
    kept_count: int = 0
    relabeled_count: int = 0
    deleted_count: int = 0
    added_count: int = 0
    refined_count: int = 0
    loose_flagged_count: int = 0
    has_missing: bool = False
    missing_note: str = ""
    skipped_mllm: bool = False
    error: str = ""


@dataclass
class MllmReviewConfig:
    """LLM 智能标注管线的运行配置。

    关键字段说明：
        `provider` / `reasoning_effort`：多模态大模型接口提供方（`gemini` / `openai` / `openai_compat`）
            与统一思考/推理深度级别（默认 `auto` 遵循模型默认，不额外下发推理参数）。
        `yolo_conf` / `yolo_iou` / `high_conf_skip`：置信度与 IoU 阈值，取值范围 `(0, 1]`；
            仅当已有候选框非空、所有框均具有不低于 `high_conf_skip` 的置信度、`check_missing=False`
            且未启用要求全框收紧的边框修正时跳过 LLM。
        `box_refine_mode`：边框紧贴标准化修正模式（`off` / `flag_loose` / `mllm_crop` / `sam_local`）。
        `sam_model_name` / `sam_refine_all_kept`：本地 SAM 收框使用的权重与触发范围（默认仅收紧过松框）。
        `refine_min_area_ratio` / `refine_min_iou`：自动收框安全门禁下限，超限时保留原框并转 `need_review`。
        `max_crops_per_image`：每张图最多附带的局部放大裁剪数；`max_image_dim` 为送检图最长边像素限制。
        `max_workers`：并发处理线程池工作线程数（至少为 `1`）。
        `max_images`：本次最多处理的图片数量上限（`0` 表示处理目录中全部图片，`> 0` 仅按顺序抽取前 `N` 张用于快速试跑验证）。
    """

    images_dir: Path
    output_dir: Path
    source_mode: ReviewSourceMode = ReviewSourceMode.YOLO_PREDICT
    annotations_dir: Path | None = None
    weights_path: Path | None = None
    yolo_conf: float = 0.12
    yolo_iou: float = 0.60
    yolo_imgsz: int = 640
    yolo_device: str = "0"

    provider: MllmProvider = MllmProvider.GEMINI
    model_name: str = "gemini-4-hydrogen"
    api_key: str = ""
    base_url: str = ""
    reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO

    allowed_classes: list[str] = field(default_factory=list)
    custom_guideline: str = ""
    high_conf_skip: float = 0.85
    check_missing: bool = True
    add_missing_boxes: bool = True
    include_crops: bool = True
    max_crops_per_image: int = 8
    max_image_dim: int = 1280
    max_workers: int = 2
    max_images: int = 0
    triage_subfolders: bool = True
    recursive: bool = False
    box_refine_mode: BoxRefineMode = BoxRefineMode.OFF
    sam_model_name: str = "sam2.1_t.pt"
    sam_refine_all_kept: bool = False
    sam3_enable_llm_review: bool = False
    refine_min_area_ratio: float = 0.15
    refine_min_iou: float = 0.25


@dataclass
class MllmReviewResult:
    """单次 LLM 智能标注任务完成或取消后的汇总结果。"""

    output_dir: Path
    classes: list[str]
    total_images: int
    auto_passed_count: int
    llm_fixed_count: int
    need_review_count: int
    error_count: int
    records: list[ImageReviewRecord] = field(default_factory=list)
    report_path: Path | None = None
    classes_txt_path: Path | None = None
    cancelled: bool = False


class MllmClient(Protocol):
    """单张图片多模态审核客户端抽象协议。"""

    def review_image(
        self,
        overview_jpeg: bytes,
        crops: list[tuple[int, str, bytes]],
        system_prompt: str,
        user_prompt: str,
    ) -> MllmReviewResponse: ...


class BoxRefinerProtocol(Protocol):
    """本地像素级边框紧贴修正器抽象协议（输入原图与待修正框列表，返回 `box_id -> 紧贴 xyxy` 映射）。"""

    def refine_boxes(
        self, image_bgr: np.ndarray, boxes: list[CandidateBox]
    ) -> dict[int, tuple[float, float, float, float] | None]: ...


class Sam3DetectorProtocol(Protocol):
    """SAM 3 文本概念自动标注检测器抽象协议（输入图片路径、输出类别名与文本提示词，返回像素候选框）。"""

    def detect_image(
        self,
        image_path: Path,
        class_names: list[str],
        text_prompts: list[str],
    ) -> list[CandidateBox]: ...


def _read_image_unicode_safe(image_path: Path) -> np.ndarray:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return img


def _encode_jpeg_bytes(frame: np.ndarray, max_dim: int = 1280, quality: int = 90) -> bytes:
    h, w = frame.shape[:2]
    longest = max(h, w)
    resized = frame
    if max_dim > 0 and longest > max_dim:
        scale = max_dim / float(longest)
        resized = cv2.resize(
            frame,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def _extract_json_payload(raw_text: str) -> str:
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_review_response_text(raw_text: str) -> MllmReviewResponse:
    """清理 Markdown 代码围栏并将模型返回的 JSON 文本反序列化为 `MllmReviewResponse`。"""
    cleaned = _extract_json_payload(raw_text)
    return MllmReviewResponse.model_validate_json(cleaned)


def _normalize_gemini_model_id(model_name: str) -> str:
    """规范化 Gemini 模型名称：去除 `models/` 前缀并转为小写。"""
    return model_name.strip().lower().removeprefix("models/")


def _openai_model_is_o_series(model_name: str) -> bool:
    """判断规范化后的 OpenAI 模型名是否属于 `o` 系列推理模型（如 `o1` / `o3` / `o4-mini`）。"""
    model_id = _normalize_openai_model_id(model_name)
    return bool(model_id and re.match(r"^o[1-9](?:$|[-._])", model_id))


def _openai_model_is_non_reasoning(model_name: str) -> bool:
    """事前判断模型是否属于已知非推理直出模型（如 `gpt-4o` / `gpt-4.1` / `gpt-3.5`）。"""
    model_id = _normalize_openai_model_id(model_name)
    if not model_id:
        return False
    return bool(
        re.match(r"^(?:gpt-4o|gpt-4\.1|gpt-4-|gpt-4$|gpt-3\.5|chatgpt-4o)(?:$|[-._])", model_id)
    )


def _compat_model_is_non_reasoning(model_name: str) -> bool:
    """判断第三方 OpenAI 兼容模型名是否属于已知非思考视觉架构（如 `Qwen2.5-VL` / `Qwen2-VL` / `LLaVA`）。"""
    model_id = _normalize_openai_model_id(model_name)
    if not model_id:
        return False
    return bool(
        re.search(r"(?:^|[/:_-])(?:qwen2\.?5[-_]?vl|qwen2[-_]?vl|llava)(?:$|[^a-z0-9])", model_id)
    )


def supported_reasoning_efforts(
    provider: MllmProvider, model_name: str
) -> tuple[ReasoningEffort, ...]:
    """返回指定模型提供方与模型名称在界面及管线中受支持的统一思考/推理级别有序元组。"""
    if provider == MllmProvider.GEMINI:
        m = _normalize_gemini_model_id(model_name)
        if "gemini-4-hydrogen" in m or "gemini-4-neon" in m:
            return (
                ReasoningEffort.AUTO,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
                ReasoningEffort.XHIGH,
            )
        if "gemini-4-carbon" in m or "gemini-4-argon" in m:
            return (
                ReasoningEffort.AUTO,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            )
        if m.startswith("gemini-4"):
            # 用户手输未完成前缀（如 `gemini-4-`）时暂保留 `xhigh`，避免输入中途误清除
            return (
                ReasoningEffort.AUTO,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
                ReasoningEffort.XHIGH,
            )
        if "gemini-3.8" in m:
            return (
                ReasoningEffort.AUTO,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            )
        if "flash-lite" in m or m.startswith("gemini-3."):
            return (
                ReasoningEffort.AUTO,
                ReasoningEffort.OFF,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
            )
        return (
            ReasoningEffort.AUTO,
            ReasoningEffort.OFF,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        )

    if _openai_model_is_non_reasoning(model_name) or _compat_model_is_non_reasoning(model_name):
        return (ReasoningEffort.AUTO,)

    model_id = _normalize_openai_model_id(model_name)
    gpt_match = re.match(r"^gpt-(\d+)(?:$|[-._])", model_id)
    if gpt_match and int(gpt_match.group(1)) >= 6:
        return (
            ReasoningEffort.AUTO,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        )
    if _openai_model_is_o_series(model_name):
        return (
            ReasoningEffort.AUTO,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
        )
    if gpt_match and int(gpt_match.group(1)) == 5:
        if re.match(r"^gpt-5\.4(?:$|[-._])", model_id):
            return (
                ReasoningEffort.AUTO,
                ReasoningEffort.OFF,
                ReasoningEffort.LOW,
                ReasoningEffort.MEDIUM,
                ReasoningEffort.HIGH,
                ReasoningEffort.XHIGH,
            )
        return (
            ReasoningEffort.AUTO,
            ReasoningEffort.OFF,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
        )
    return (
        ReasoningEffort.AUTO,
        ReasoningEffort.OFF,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    )


def resolve_effective_reasoning_effort(
    provider: MllmProvider,
    model_name: str,
    effort: ReasoningEffort,
) -> tuple[ReasoningEffort, str | None]:
    """依据模型能力矩阵将请求的思考级别校正为最近似受支持级别，并返回可选中文提示。"""
    supported = supported_reasoning_efforts(provider, model_name)
    if effort in supported:
        return effort, None

    label_display = model_name.strip() or provider.value
    if supported == (ReasoningEffort.AUTO,):
        return (
            ReasoningEffort.AUTO,
            f"模型 '{label_display}' 为非推理直出模型，不支持思考级别调节，已切回模型默认",
        )
    if effort == ReasoningEffort.XHIGH and ReasoningEffort.HIGH in supported:
        return (
            ReasoningEffort.HIGH,
            f"模型 '{label_display}' 不支持 xhigh，已自动降级为最高可用档 high",
        )
    if effort == ReasoningEffort.OFF and ReasoningEffort.LOW in supported:
        return (
            ReasoningEffort.LOW,
            f"模型 '{label_display}' 不支持关闭思考/极简，已自动切换为最低可用档 low",
        )
    return (
        ReasoningEffort.AUTO,
        f"模型 '{label_display}' 不支持思考级别 {effort.value}，已切回模型默认",
    )


# 各级别单次请求读取超时（秒）：全部严格高于官方默认最大超时（600s），避免深度思考或高并发排队时被客户端提前掐断
MLLM_CONNECT_TIMEOUT_SECONDS: float = 30.0
MLLM_PROBE_TIMEOUT_SECONDS: float = 60.0
MLLM_REQUEST_TIMEOUTS_SECONDS: dict[ReasoningEffort, float] = {
    ReasoningEffort.OFF: 660.0,
    ReasoningEffort.LOW: 720.0,
    ReasoningEffort.AUTO: 900.0,
    ReasoningEffort.MEDIUM: 900.0,
    ReasoningEffort.HIGH: 1200.0,
    ReasoningEffort.XHIGH: 1500.0,
}


def resolve_mllm_request_timeout(
    provider: MllmProvider,
    model_name: str,
    effort: ReasoningEffort,
) -> float:
    """按接口、模型能力矩阵与思考级别返回单次请求读取超时（单位：秒，均高于官方 600s 上限）。"""
    effective_effort, _ = resolve_effective_reasoning_effort(provider, model_name, effort)
    return MLLM_REQUEST_TIMEOUTS_SECONDS.get(
        effective_effort, MLLM_REQUEST_TIMEOUTS_SECONDS[ReasoningEffort.AUTO]
    )


def resolve_mllm_max_retries(effort: ReasoningEffort) -> int:
    """按实际生效的思考级别返回 SDK 自动重试次数。

    设计原因：
        `HIGH` / `XHIGH` 深度推理单次耗时可能达十余分钟且消耗较多思考 Token，若超时后仍由 SDK 自动重试
        会导致双倍耗费与更长卡顿，因此返回 `0`；其余级别保留 `1` 次重试以吸收偶发网络抖动或 429/5xx。
    """
    if effort in {ReasoningEffort.HIGH, ReasoningEffort.XHIGH}:
        return 0
    return 1


def _format_exception_detail(exc: BaseException) -> str:
    """提取异常类型、摘要文本及最多 3 层 `__cause__` 底层原因，供错误落盘与运行日志诊断使用。"""
    base_msg = str(exc).strip()
    exc_type = type(exc).__name__

    causes: list[str] = []
    seen_ids: set[int] = {id(exc)}
    cur = exc.__cause__
    for _ in range(3):
        if cur is None or id(cur) in seen_ids:
            break
        seen_ids.add(id(cur))
        c_type = type(cur).__name__
        c_msg = str(cur).strip()
        fragment = f"{c_type}: {c_msg}" if c_msg else c_type
        if fragment and fragment not in causes and fragment != base_msg:
            causes.append(fragment)
        cur = cur.__cause__

    if not base_msg:
        summary = exc_type
    elif exc_type in {"RuntimeError", "ValueError", "FileNotFoundError"} and not causes:
        # 常规业务或校验异常本身已携带完整语义，无底层链时保持原文不变
        return base_msg
    elif exc_type in base_msg:
        summary = base_msg
    else:
        summary = f"{exc_type}: {base_msg}"

    if causes:
        return f"{summary}（底层原因: {' -> '.join(causes)}）"
    return summary


def _gemini_model_supports_minimal(model_name: str) -> bool:
    """判断 Gemini 模型是否支持 `thinking_level='minimal'`。"""
    return ReasoningEffort.OFF in supported_reasoning_efforts(MllmProvider.GEMINI, model_name)


def _build_gemini_thinking_candidates(
    effort: ReasoningEffort, model_name: str
) -> tuple[str | None, ...]:
    """将统一思考级别先经能力矩阵校正，再映射为 Gemini `thinking_level` 有序降级候选链（`None` 表示省略参数）。"""
    effective, _ = resolve_effective_reasoning_effort(MllmProvider.GEMINI, model_name, effort)
    if effective == ReasoningEffort.AUTO:
        return (None,)
    if effective == ReasoningEffort.OFF:
        return ("minimal", "low", None)
    if effective == ReasoningEffort.LOW:
        return ("low", None)
    if effective == ReasoningEffort.MEDIUM:
        return ("medium", None)
    if effective == ReasoningEffort.HIGH:
        return ("high", None)
    return ("xhigh", "high", None)


def _is_gemini_thinking_bad_request(exc: BaseException) -> bool:
    """判断 Gemini 异常是否因模型拒绝指定 `thinking_level` / `generation_config` 的 400 参数错误所致。"""
    code = getattr(exc, "code", None)
    status_code = getattr(exc, "status_code", None)
    status = str(getattr(exc, "status", None) or "").strip().lower()
    msg = f"{status} {exc}".lower()
    is_bad_request = (
        code == 400
        or status_code == 400
        or "invalid_argument" in msg
        or bool(re.search(r"\b400\b", msg))
    )
    if not is_bad_request:
        return False
    return any(tok in msg for tok in ("thinking_level", "thinking", "generation_config"))


def _close_client_safely(client: object | None) -> None:
    """安全关闭底层 HTTP/SDK 客户端以立即中断在途请求（幂等，忽略关闭异常）。"""
    if client is None:
        return
    close_fn = getattr(client, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception:
            pass


class GeminiMllmClient:
    """基于官方 `google-genai` Interactions API 的 Google Gemini 多模态审核客户端。

    凭据、超时与状态：
        初始化时优先使用传入的 `api_key`，为空时读取环境变量 `GEMINI_API_KEY`，均缺失则抛出 `ValueError`；
        `timeout` 单位为秒，未显式传入正数时按模型与思考级别经 `resolve_mllm_request_timeout` 计算，
        并换算为毫秒写入 `types.HttpOptions(timeout=...)`（作为单次请求 HTTP 超时传给 `genai.Client`）。
        内部持有无状态交互会话客户端（每次调用设置 `store=False`），并按 `reasoning_effort`
        维护单调降级索引 `_thinking_idx`，遇到不支持特定 `thinking_level` 的 400 响应时自动降级。
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-4-hydrogen",
        *,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        timeout: float | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        resolved_key = api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError("未提供 Gemini API Key（可在界面填写或设置环境变量 GEMINI_API_KEY）。")
        self.model_name = model_name.strip() or "gemini-4-hydrogen"
        self.reasoning_effort = ReasoningEffort(reasoning_effort)
        effective_timeout = (
            float(timeout)
            if (timeout is not None and timeout > 0)
            else resolve_mllm_request_timeout(
                MllmProvider.GEMINI, self.model_name, self.reasoning_effort
            )
        )
        self.request_timeout: float = effective_timeout
        timeout_ms = max(1000, int(round(effective_timeout * 1000)))
        self.client = genai.Client(
            api_key=resolved_key,
            http_options=types.HttpOptions(timeout=timeout_ms),
        )
        self._thinking_candidates = _build_gemini_thinking_candidates(self.reasoning_effort, self.model_name)
        self._thinking_idx = 0

    def review_image(
        self,
        overview_jpeg: bytes,
        crops: list[tuple[int, str, bytes]],
        system_prompt: str,
        user_prompt: str,
    ) -> MllmReviewResponse:
        """发送编号总览图及候选框裁剪图至 Gemini 并解析结构化审核结果；遇思考级别 400 拒绝时自动降级重试。"""
        contents: list[dict[str, object]] = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image",
                "data": base64.b64encode(overview_jpeg).decode("utf-8"),
                "mime_type": "image/jpeg",
            },
        ]
        for box_id, label, crop_bytes in crops:
            contents.append(
                {
                    "type": "text",
                    "text": f"Zoomed crop for candidate box_id={box_id} (current label='{label}'):",
                }
            )
            contents.append(
                {
                    "type": "image",
                    "data": base64.b64encode(crop_bytes).decode("utf-8"),
                    "mime_type": "image/jpeg",
                }
            )

        while True:
            sent_idx = self._thinking_idx
            sent_level = self._thinking_candidates[sent_idx]
            req_kwargs: dict[str, object] = {
                "model": self.model_name,
                "system_instruction": system_prompt,
                "input": contents,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": MllmReviewResponse.model_json_schema(),
                },
                "store": False,
            }
            if sent_level is not None:
                req_kwargs["generation_config"] = {"thinking_level": sent_level}
            try:
                interaction = self.client.interactions.create(**req_kwargs)
                break
            except Exception as exc:
                if sent_level is not None and _is_gemini_thinking_bad_request(exc):
                    self._thinking_idx = max(self._thinking_idx, sent_idx + 1)
                    continue
                raise

        output_text = getattr(interaction, "output_text", None) or ""
        if not output_text:
            raise RuntimeError("Gemini 返回为空，未包含文本输出。")
        return parse_review_response_text(output_text)

    def close(self) -> None:
        """关闭底层 `genai.Client` HTTP 连接池以立即中断在途请求（可重复调用）。"""
        _close_client_safely(self.client)


def _extract_openai_bad_request_info(exc: BaseException) -> tuple[str, str]:
    """提取 OpenAI 兼容接口 400 异常中的报错参数名与小写描述文本，用于精确判定不兼容参数。"""
    param = str(getattr(exc, "param", None) or "").strip().lower()
    parts: list[str] = []
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err_obj = body.get("error")
        payload = err_obj if isinstance(err_obj, dict) else body
        if not param:
            param = str(payload.get("param") or "").strip().lower()
        for key in ("message", "detail", "code"):
            val = payload.get(key)
            if val:
                parts.append(str(val))
    elif body is not None:
        parts.append(str(body))
    parts.append(str(exc))
    return param, " ".join(parts).lower()


def _is_temperature_bad_request(exc: BaseException) -> bool:
    """判断 400 异常是否因模型拒绝自定义 `temperature` 参数（仅支持默认值）所致。"""
    param, text = _extract_openai_bad_request_info(exc)
    return param == "temperature" or bool(re.search(r"(?<![a-z0-9_])temperature(?![a-z0-9_])", text))


def _is_max_tokens_bad_request(exc: BaseException) -> bool:
    """判断 400 异常是否因模型或接口拒绝 `max_tokens` 参数所致。"""
    param, text = _extract_openai_bad_request_info(exc)
    return param == "max_tokens" or bool(re.search(r"(?<![a-z0-9_])max_tokens(?![a-z0-9_])", text))


def _is_response_format_bad_request(exc: BaseException) -> bool:
    """判断 400 异常是否因后端不支持 `response_format` / `json_object` 结构化输出所致。"""
    param, text = _extract_openai_bad_request_info(exc)
    if param == "response_format" or param.startswith("response_format."):
        return True
    return bool(
        re.search(r"(?<![a-z0-9_])(?:response_format|json_object|json[\s_-]*mode)(?![a-z0-9_])", text)
    )


def _normalize_openai_model_id(model_name: str) -> str:
    """规范化 OpenAI 兼容模型名称：提取 `openai/` 中转前缀或 `ft:` 微调前缀下的基础模型 ID（小写）。"""
    model_id = model_name.strip().lower()
    if "/" in model_id:
        prefix, tail = model_id.rsplit("/", 1)
        if prefix == "openai" or prefix.endswith("/openai"):
            model_id = tail
    if model_id.startswith("ft:"):
        parts = model_id.split(":")
        if len(parts) >= 2 and parts[1]:
            model_id = parts[1]
    return model_id


def _openai_model_omits_custom_sampling(model_name: str) -> bool:
    """事前判断模型是否属于默认不支持自定义 `temperature` 及 `max_tokens` 的 OpenAI 新系列（`o` 系列与 `gpt-5+`）。"""
    model_id = _normalize_openai_model_id(model_name)
    if not model_id:
        return False
    if re.match(r"^o[1-9](?:$|[-._])", model_id):
        return True
    gpt_match = re.match(r"^gpt-(\d+)(?:$|[-._])", model_id)
    return bool(gpt_match and int(gpt_match.group(1)) >= 5)


def _build_openai_reasoning_candidates(
    effort: ReasoningEffort, model_name: str
) -> tuple[str | None, ...]:
    """将统一思考级别先经能力矩阵校正，再映射为 OpenAI `reasoning_effort` 有序降级候选链（`None` 表示省略参数）。"""
    effective, _ = resolve_effective_reasoning_effort(MllmProvider.OPENAI, model_name, effort)
    if effective == ReasoningEffort.AUTO:
        return (None,)
    if effective == ReasoningEffort.OFF:
        return ("none", "minimal", "low", None)
    if effective == ReasoningEffort.LOW:
        return ("low", None)
    if effective == ReasoningEffort.MEDIUM:
        return ("medium", None)
    if effective == ReasoningEffort.HIGH:
        return ("high", None)
    return ("xhigh", "high", None)


def _is_reasoning_effort_bad_request(exc: BaseException) -> bool:
    """判断 OpenAI 官方或兼容接口的 400 异常是否因拒绝 `reasoning_effort` 参数或其取值所致。"""
    param, text = _extract_openai_bad_request_info(exc)
    if param in {"reasoning_effort", "reasoning", "reasoning.effort"}:
        return True
    return bool(
        re.search(r"(?<![a-z0-9_])reasoning[_.\s-]*effort(?![a-z0-9_])", text)
        or re.search(r"unsupported[^\n]*reasoning", text)
    )


class OpenAICompatMllmClient:
    """OpenAI 官方与兼容接口多模态审核客户端，适配 OpenAI 官方、通义千问、硅基流动、Ollama 与 vLLM。

    凭据、超时与重试归属：
        - 当 `allow_anonymous_key=False`（OpenAI 官方）时：仅读取 `api_key` 或环境变量 `OPENAI_API_KEY`，
          均缺失时抛出 `ValueError`，绝不回退为 `"ollama"`；`base_url` 为空时默认使用 `OPENAI_OFFICIAL_BASE_URL`。
        - 当 `allow_anonymous_key=True`（OpenAI 兼容接口）时：依次回退 `api_key`、`OPENAI_API_KEY` 与 `"ollama"`，
          `base_url` 为空时默认使用 DashScope 兼容端点。
        - 请求超时（`request_timeout`，单位秒）：`timeout` 为 `None` 或非正数时按思考级别解析
          （均大于官方 600s 上限），连接超时限制为 `MLLM_CONNECT_TIMEOUT_SECONDS`（30s）；
          `max_retries` 未显式指定时对 `HIGH`/`XHIGH` 设为 `0` 以防深度推理超时重复扣费，其余设为 `1`。

    参数事前预判与自动降级状态：
        维护 `_temperature_supported`、`_json_mode_supported` 与 `_reasoning_idx` 能力状态：
        初始化时按模型名对 OpenAI `o` 系列及 `gpt-5+` 预置 `_temperature_supported = False`，对已知非推理模型
        事前省略 `reasoning_effort`；运行时仅在对应参数触发 `openai.BadRequestError` 时单调降级。
        每轮请求前先对状态取快照，多线程并发下依据本轮实际发送快照判定降级，不会互相误关且保证有限次终止。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float | None = None,
        *,
        reasoning_effort: ReasoningEffort = ReasoningEffort.AUTO,
        allow_anonymous_key: bool = True,
        provider_label: str = "OpenAI 兼容接口",
        max_retries: int | None = None,
    ) -> None:
        from openai import BadRequestError, OpenAI, Timeout

        self._provider_label = provider_label
        self.model_name = model_name.strip()
        if not self.model_name:
            hint = OPENAI_MODELS[0][0] if not allow_anonymous_key else "qwen-vl-max-latest"
            raise ValueError(f"请指定{provider_label}的模型名称（如 {hint}）。")

        explicit_or_env = api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not allow_anonymous_key and not explicit_or_env:
            raise ValueError("未提供 OpenAI API Key（可在界面填写或设置环境变量 OPENAI_API_KEY）。")
        resolved_key = explicit_or_env or "ollama"
        default_url = (
            OPENAI_OFFICIAL_BASE_URL
            if not allow_anonymous_key
            else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        resolved_url = base_url.strip() or default_url

        provider = MllmProvider.OPENAI_COMPAT if allow_anonymous_key else MllmProvider.OPENAI
        self.reasoning_effort = ReasoningEffort(reasoning_effort)
        effective_effort, _ = resolve_effective_reasoning_effort(
            provider, self.model_name, self.reasoning_effort
        )
        effective_timeout = (
            float(timeout)
            if (timeout is not None and timeout > 0)
            else resolve_mllm_request_timeout(provider, self.model_name, self.reasoning_effort)
        )
        effective_retries = (
            max(0, int(max_retries))
            if max_retries is not None
            else resolve_mllm_max_retries(effective_effort)
        )
        self.request_timeout: float = effective_timeout
        self.max_retries: int = effective_retries
        openai_timeout = Timeout(
            effective_timeout,
            connect=min(MLLM_CONNECT_TIMEOUT_SECONDS, effective_timeout),
        )

        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=openai_timeout,
            max_retries=effective_retries,
        )
        self._bad_request_error = BadRequestError
        self._temperature_supported = not _openai_model_omits_custom_sampling(self.model_name)
        self._json_mode_supported = True
        self._reasoning_candidates = _build_openai_reasoning_candidates(self.reasoning_effort, self.model_name)
        self._reasoning_idx = 0

    def review_image(
        self,
        overview_jpeg: bytes,
        crops: list[tuple[int, str, bytes]],
        system_prompt: str,
        user_prompt: str,
    ) -> MllmReviewResponse:
        """发送多模态对话请求并解析 JSON 审核结果；若后端拒绝可选参数则按快照单调降级重试。"""
        user_content: list[dict[str, object]] = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(overview_jpeg).decode("utf-8")
                },
            },
        ]
        for box_id, label, crop_bytes in crops:
            user_content.append(
                {
                    "type": "text",
                    "text": f"Zoomed crop for candidate box_id={box_id} (current label='{label}'):",
                }
            )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(crop_bytes).decode("utf-8")
                    },
                }
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        while True:
            # 先快照本轮实际发送的可选参数，防止并发线程修改标志位后漏判当前请求触发的 400
            sent_temperature = self._temperature_supported
            sent_json_mode = self._json_mode_supported
            sent_reasoning_idx = self._reasoning_idx
            sent_reasoning_val = self._reasoning_candidates[sent_reasoning_idx]
            req_kwargs: dict[str, object] = {
                "model": self.model_name,
                "messages": messages,
            }
            if sent_temperature:
                req_kwargs["temperature"] = 0.1
            if sent_json_mode:
                req_kwargs["response_format"] = {"type": "json_object"}
            if sent_reasoning_val is not None:
                req_kwargs["reasoning_effort"] = sent_reasoning_val
            try:
                # 保留 type: ignore 指令供静态类型检查器识别兼容接口动态构造的参数字典
                resp = self.client.chat.completions.create(**req_kwargs)  # type: ignore[arg-type]
                break
            except self._bad_request_error as exc:
                downgraded = False
                if sent_temperature and _is_temperature_bad_request(exc):
                    self._temperature_supported = False
                    downgraded = True
                if sent_reasoning_val is not None and _is_reasoning_effort_bad_request(exc):
                    self._reasoning_idx = max(self._reasoning_idx, sent_reasoning_idx + 1)
                    downgraded = True
                if sent_json_mode and _is_response_format_bad_request(exc):
                    self._json_mode_supported = False
                    downgraded = True
                if not downgraded:
                    if sent_reasoning_val is not None:
                        # 遇笼统 400 时优先省略 reasoning_effort，避免误关 json_mode 结构化输出
                        self._reasoning_idx = max(self._reasoning_idx, len(self._reasoning_candidates) - 1)
                        downgraded = True
                    elif sent_json_mode:
                        # 兼容部分第三方 OpenAI 兼容服务端在不支持 response_format 时仅返回笼统 400 文案
                        self._json_mode_supported = False
                        downgraded = True
                if not downgraded:
                    raise

        content = (resp.choices[0].message.content or "") if resp.choices else ""
        if not content.strip():
            raise RuntimeError(f"{self._provider_label}返回为空，未包含文本输出。")
        return parse_review_response_text(content)

    def close(self) -> None:
        """关闭底层 OpenAI HTTP 客户端以立即中断在途网络请求（可重复调用）。"""
        _close_client_safely(self.client)


def build_mllm_client(config: MllmReviewConfig) -> MllmClient:
    """根据配置中的 `provider` 创建对应的 LLM 标注质检客户端实例。"""
    if config.provider == MllmProvider.GEMINI:
        return GeminiMllmClient(
            api_key=config.api_key,
            model_name=config.model_name,
            reasoning_effort=config.reasoning_effort,
        )
    if config.provider == MllmProvider.OPENAI:
        return OpenAICompatMllmClient(
            api_key=config.api_key,
            base_url=OPENAI_OFFICIAL_BASE_URL,
            model_name=config.model_name.strip() or OPENAI_MODELS[0][0],
            reasoning_effort=config.reasoning_effort,
            allow_anonymous_key=False,
            provider_label="OpenAI 官方接口",
        )
    return OpenAICompatMllmClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model_name,
        reasoning_effort=config.reasoning_effort,
        allow_anonymous_key=True,
        provider_label="OpenAI 兼容接口",
    )


def test_mllm_connection(config: MllmReviewConfig) -> tuple[bool, str]:
    """向配置的 LLM 接口发送极简纯文本探针请求，快速校验 API Key、Base URL 与模型名称是否连通。

    返回值与失败行为：
        不抛出网络或鉴权异常，统一返回 `(是否连通, 中文结果摘要)`；
        探针不上传任何图片以将延迟和费用降至最低。
    """
    t0 = time.perf_counter()
    try:
        if config.provider == MllmProvider.GEMINI:
            from google import genai
            from google.genai import types

            resolved_key = config.api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()
            if not resolved_key:
                return False, "未提供 Gemini API Key（请在界面填写或设置环境变量 GEMINI_API_KEY）。"
            model_name = config.model_name.strip() or "gemini-4-hydrogen"
            client = genai.Client(
                api_key=resolved_key,
                http_options=types.HttpOptions(timeout=int(round(MLLM_PROBE_TIMEOUT_SECONDS * 1000))),
            )
            interaction = client.interactions.create(
                model=model_name,
                input=[{"type": "text", "text": "Reply OK."}],
                store=False,
            )
            reply = (getattr(interaction, "output_text", None) or "").strip()
            elapsed = time.perf_counter() - t0
            preview = reply[:40].replace("\n", " ") if reply else "(空响应)"
            return True, f"Gemini 接口连通正常（模型: {model_name}，耗时 {elapsed:.2f}s，回复: {preview}）"

        from openai import BadRequestError, OpenAI, Timeout

        is_official_openai = config.provider == MllmProvider.OPENAI
        explicit_or_env = config.api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if is_official_openai:
            if not explicit_or_env:
                return False, "未提供 OpenAI API Key（请在界面填写或设置环境变量 OPENAI_API_KEY）。"
            resolved_key = explicit_or_env
            resolved_url = OPENAI_OFFICIAL_BASE_URL
            provider_label = "OpenAI 官方接口"
            model_placeholder = OPENAI_MODELS[0][0]
        else:
            resolved_key = explicit_or_env or "ollama"
            resolved_url = config.base_url.strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            provider_label = "OpenAI 兼容接口"
            model_placeholder = "qwen-vl-max-latest"

        model_name = config.model_name.strip()
        if not model_name:
            return False, f"请先填写模型名称（如 {model_placeholder}）。"
        client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=Timeout(MLLM_PROBE_TIMEOUT_SECONDS, connect=20.0),
            max_retries=0,
        )
        omit_legacy_sampling = _openai_model_omits_custom_sampling(model_name)
        send_temperature = not omit_legacy_sampling
        send_max_tokens = not omit_legacy_sampling
        while True:
            probe_kwargs: dict[str, object] = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Reply OK."}],
            }
            if send_max_tokens:
                probe_kwargs["max_tokens"] = 12
            if send_temperature:
                probe_kwargs["temperature"] = 0.0
            try:
                # 保留 type: ignore 指令供静态类型检查器识别动态构造的探针参数字典
                resp = client.chat.completions.create(**probe_kwargs)  # type: ignore[arg-type]
                break
            except BadRequestError as exc:
                downgraded = False
                if send_temperature and _is_temperature_bad_request(exc):
                    send_temperature = False
                    downgraded = True
                if send_max_tokens and _is_max_tokens_bad_request(exc):
                    # 部分新模型拒绝 max_tokens；直接省略限制而非改传极小 max_completion_tokens，避免思考模型返回空文本
                    send_max_tokens = False
                    downgraded = True
                if not downgraded:
                    raise
        reply = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        elapsed = time.perf_counter() - t0
        preview = reply[:40].replace("\n", " ") if reply else "(空响应)"
        return True, f"{provider_label}连通正常（模型: {model_name}，耗时 {elapsed:.2f}s，回复: {preview}）"
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return False, f"接口连接失败（耗时 {elapsed:.2f}s）: {_format_exception_detail(exc)}"


class UltralyticsSamBoxRefiner:
    """基于 `ultralytics.SAM` 的本地像素级边框紧贴修正器（Box Prompt → 掩码最小外接矩形）。

    线程安全与运行环境：
        内部持有 `threading.Lock` 串行化 GPU/CPU 模型推理，保证多并发 LLM 标注线程共享同一 SAM 实例时
        不发生 CUDA 上下文争抢；若本地不存在权重，Ultralytics 会按模型名称自动下载。
    """

    def __init__(self, model_name: str = "sam2.1_t.pt", device: str = "0") -> None:
        from ultralytics import SAM

        from yolo_studio.core.trainer import ensure_torch_cpu_threads, resolve_pretrained_weights

        ensure_torch_cpu_threads()
        resolved = resolve_pretrained_weights(model_name.strip() or "sam2.1_t.pt")
        self.model_source = resolved
        self.device = device
        self._lock = Lock()
        self._model = SAM(resolved)

    def refine_boxes(
        self, image_bgr: np.ndarray, boxes: list[CandidateBox]
    ) -> dict[int, tuple[float, float, float, float] | None]:
        """对 `boxes` 执行 Box-Prompt 掩码分割并在原框容差范围内提取最小紧贴外接矩形。"""
        if not boxes:
            return {}
        img_h, img_w = image_bgr.shape[:2]
        bboxes_arg = [[float(b.x1), float(b.y1), float(b.x2), float(b.y2)] for b in boxes]
        with self._lock:
            results = self._model.predict(
                source=image_bgr,
                bboxes=bboxes_arg,
                device=self.device,
                verbose=False,
            )

        refined_map: dict[int, tuple[float, float, float, float] | None] = {b.box_id: None for b in boxes}
        if not results:
            return refined_map
        masks = getattr(results[0], "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return refined_map

        mask_arr = masks.data.detach().cpu().numpy()
        for idx, box in enumerate(boxes):
            if idx >= len(mask_arr):
                break
            mask = mask_arr[idx]
            mh, mw = mask.shape[:2]
            if mh != img_h or mw != img_w:
                mask = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            binary = mask > 0.5

            # 限制在原框向外最多 5% 容差范围内取紧贴轮廓，防止分割泄漏到远处无关物体
            bw = max(1.0, box.x2 - box.x1)
            bh = max(1.0, box.y2 - box.y1)
            rx1 = max(0, int(np.floor(box.x1 - bw * 0.05)))
            ry1 = max(0, int(np.floor(box.y1 - bh * 0.05)))
            rx2 = min(img_w, int(np.ceil(box.x2 + bw * 0.05)))
            ry2 = min(img_h, int(np.ceil(box.y2 + bh * 0.05)))
            roi = binary[ry1:ry2, rx1:rx2]
            ys, xs = np.nonzero(roi)
            if len(xs) == 0 or len(ys) == 0:
                continue
            nx1 = float(rx1 + int(xs.min()))
            ny1 = float(ry1 + int(ys.min()))
            nx2 = float(rx1 + int(xs.max()) + 1)
            ny2 = float(ry1 + int(ys.max()) + 1)
            if (nx2 - nx1) > 2.0 and (ny2 - ny1) > 2.0:
                refined_map[box.box_id] = (nx1, ny1, min(float(img_w), nx2), min(float(img_h), ny2))
        return refined_map


class UltralyticsSam3Detector:
    """基于 `ultralytics.models.sam.SAM3SemanticPredictor` 的本地 SAM 3 文本概念自动标注检测器。

    运行环境与前置依赖：
        依赖 `ultralytics >= 8.4`（提供 `SAM3SemanticPredictor`）、`clip` 分词器包
        （`pip install git+https://github.com/ultralytics/CLIP.git`）以及本地 `sam3.pt` 权重文件
        （Meta 受限权重，需预先从 HuggingFace 下载至本地）。
        GPU 设备下通过 `quantize=16` 启用 FP16 推理降低显存占用，CPU 下由底层自动回退至 FP32。
        内部持 `threading.Lock` 串行化推理调用，并使用 Unicode 安全读图以兼容 Windows 中文路径。
    """

    def __init__(
        self,
        weights_path: Path,
        *,
        conf: float = 0.25,
        iou: float = 0.60,
        imgsz: int = 640,
        device: str = "0",
    ) -> None:
        if not weights_path or not weights_path.is_file():
            raise FileNotFoundError(
                f"SAM 3 权重文件不存在: {weights_path}（sam3.pt 为 Meta 受限权重，请先下载至本地后再选择）。"
            )
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError as exc:
            raise RuntimeError(
                "当前 ultralytics 版本不支持 SAM3SemanticPredictor，请升级 ultralytics 至最新版本。"
            ) from exc

        from yolo_studio.core.trainer import ensure_torch_cpu_threads

        ensure_torch_cpu_threads()
        overrides: dict[str, object] = {
            "model": str(weights_path),
            "conf": float(conf),
            "iou": float(iou),
            "imgsz": int(imgsz),
            "device": str(device),
            "task": "segment",
            "mode": "predict",
            "quantize": 16,
            "save": False,
            "verbose": False,
        }
        self._lock = Lock()
        self._predictor = SAM3SemanticPredictor(overrides=overrides)

    def detect_image(
        self,
        image_path: Path,
        class_names: list[str],
        text_prompts: list[str],
    ) -> list[CandidateBox]:
        """对单张图片按 `text_prompts` 执行 SAM 3 语义检测分割，并将类别索引映射回 `class_names`。"""
        if not text_prompts:
            return []
        img_bgr = _read_image_unicode_safe(image_path)
        img_h, img_w = img_bgr.shape[:2]

        with self._lock:
            self._predictor.features = None
            try:
                results = self._predictor(source=img_bgr, text=text_prompts)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    f"加载 SAM 3 文本分词器或视觉依赖失败: {exc}。"
                    "请在当前 Python 环境安装依赖：pip install --no-deps git+https://github.com/ultralytics/CLIP.git && pip install ftfy regex timm"
                ) from exc

        if not results:
            return []
        res0 = results[0]
        r_boxes = getattr(res0, "boxes", None)
        if r_boxes is None or len(r_boxes) == 0:
            return []

        xyxy_list = r_boxes.xyxy.detach().cpu().tolist()
        cls_list = r_boxes.cls.detach().cpu().tolist()
        conf_list = r_boxes.conf.detach().cpu().tolist()

        masks_obj = getattr(res0, "masks", None)
        mask_arr = (
            masks_obj.data.detach().cpu().numpy()
            if (masks_obj is not None and getattr(masks_obj, "data", None) is not None)
            else None
        )

        boxes: list[CandidateBox] = []
        for idx, (xyxy, cls_id, score) in enumerate(zip(xyxy_list, cls_list, conf_list)):
            cid = int(cls_id)
            if 0 <= cid < len(class_names):
                label = class_names[cid]
            else:
                label = f"class_{cid}"

            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            # 若存在高分辨率像素掩码，优先取掩码最小外接矩形获得更紧贴像素边界的目标框
            if mask_arr is not None and idx < len(mask_arr):
                mask = mask_arr[idx]
                mh, mw = mask.shape[:2]
                if mh != img_h or mw != img_w:
                    mask = cv2.resize(mask.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                ys, xs = np.nonzero(mask > 0.5)
                if len(xs) > 0 and len(ys) > 0:
                    mx1 = float(int(xs.min()))
                    my1 = float(int(ys.min()))
                    mx2 = min(float(img_w), float(int(xs.max()) + 1))
                    my2 = min(float(img_h), float(int(ys.max()) + 1))
                    if (mx2 - mx1) > 2.0 and (my2 - my1) > 2.0:
                        x1, y1, x2, y2 = mx1, my1, mx2, my2

            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))
            if (x2 - x1) <= 2.0 or (y2 - y1) <= 2.0:
                continue

            boxes.append(
                CandidateBox(
                    box_id=len(boxes),
                    label=str(label),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(score),
                    original_label=str(label),
                )
            )
        return boxes


def _render_overview_and_crops(
    image_bgr: np.ndarray,
    boxes: list[CandidateBox],
    include_crops: bool,
    max_crops: int,
    high_conf_skip: float,
    max_dim: int,
    *,
    force_crops: bool = False,
) -> tuple[bytes, list[tuple[int, str, bytes]], dict[int, tuple[int, int, int, int]]]:
    """绘制带编号候选框总览图，并提取外扩 20% 的局部放大裁剪图及其在原图上的像素裁剪区域映射。"""
    h, w = image_bgr.shape[:2]
    canvas = image_bgr.copy()
    palette = [
        (0, 210, 0),
        (0, 140, 255),
        (255, 80, 80),
        (220, 0, 220),
        (0, 220, 220),
        (255, 180, 0),
    ]
    thickness = max(2, int(round(max(h, w) / 450.0)))
    font_scale = max(0.55, max(h, w) / 1400.0)

    for box in boxes:
        color = palette[box.box_id % len(palette)]
        p1 = (int(round(box.x1)), int(round(box.y1)))
        p2 = (int(round(box.x2)), int(round(box.y2)))
        cv2.rectangle(canvas, p1, p2, color, thickness)

        conf_str = f" {box.confidence:.2f}" if box.confidence is not None else ""
        tag = f"[{box.box_id}] {box.label}{conf_str}"
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        ty = max(th + 6, p1[1])
        cv2.rectangle(canvas, (p1[0], ty - th - 6), (p1[0] + tw + 6, ty + baseline - 2), color, -1)
        cv2.putText(
            canvas,
            tag,
            (p1[0] + 3, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    overview_jpeg = _encode_jpeg_bytes(canvas, max_dim=max_dim, quality=90)

    crops: list[tuple[int, str, bytes]] = []
    crop_regions: dict[int, tuple[int, int, int, int]] = {}
    if (include_crops or force_crops) and boxes:
        # 优先按置信度从低到高排序，确保低置信度可疑框优先获得局部裁剪放大图；
        # 当 `force_crops=True`（LLM 局部精修模式）时放宽上限以尽量覆盖全部候选框
        limit = max(max_crops, len(boxes)) if force_crops else max_crops
        prioritized = sorted(
            boxes,
            key=lambda b: b.confidence if b.confidence is not None else 0.5,
        )
        for box in prioritized:
            if len(crops) >= limit:
                break
            if not force_crops and box.confidence is not None and box.confidence >= high_conf_skip and len(boxes) > limit:
                continue
            bw = max(1.0, box.x2 - box.x1)
            bh = max(1.0, box.y2 - box.y1)
            mx = bw * 0.20
            my = bh * 0.20
            cx1 = max(0, int(round(box.x1 - mx)))
            cy1 = max(0, int(round(box.y1 - my)))
            cx2 = min(w, int(round(box.x2 + mx)))
            cy2 = min(h, int(round(box.y2 + my)))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crop_bgr = image_bgr[cy1:cy2, cx1:cx2]
            crops.append((box.box_id, box.label, _encode_jpeg_bytes(crop_bgr, max_dim=512, quality=92)))
            crop_regions[box.box_id] = (cx1, cy1, cx2, cy2)

    return overview_jpeg, crops, crop_regions


def _overview_scale(img_w: int, img_h: int, max_dim: int) -> float:
    """返回总览图相对原图的缩放系数（与 `_encode_jpeg_bytes` 的缩放规则一致，最大为 `1.0`）。"""
    longest = max(img_w, img_h)
    if max_dim > 0 and longest > max_dim:
        return max_dim / float(longest)
    return 1.0


def _build_prompts(
    boxes: list[CandidateBox],
    allowed_classes: list[str],
    config: MllmReviewConfig,
    coord_scale: float = 1.0,
) -> tuple[str, str]:
    """构造系统提示词与用户提示词，并按 `config.box_refine_mode` 附加边框贴合度评估与局部精修要求。"""
    classes_str = ", ".join(allowed_classes) if allowed_classes else "(any relevant object)"
    refine_mode = config.box_refine_mode

    refine_rule_sys = ""
    refine_rule_user = ""
    if refine_mode == BoxRefineMode.OFF:
        refine_rule_user = (
            "Set `box_quality` to 'tight' and `refined_box_2d` to null for every item in `box_reviews`.\n"
        )
    else:
        refine_rule_sys = (
            "\n3. Evaluate bounding box tightness (`box_quality`) for every kept/relabeled box — "
            "`box_quality` MUST ONLY be one of 'tight', 'loose', or 'partial' (for 'delete', set `box_quality` to 'tight' and `refined_box_2d` to null):\n"
            "   - 'tight': box closely hugs the target object boundary according to YOLO standards.\n"
            "   - 'loose': box is noticeably too large and includes excessive background margin.\n"
            "   - 'partial': box cuts off a significant part of the target object."
        )
        if refine_mode == BoxRefineMode.MLLM_CROP:
            refine_rule_sys += (
                "\n4. When box_quality is 'loose' or 'partial', provide `refined_box_2d` as "
                "[ymin, xmin, ymax, xmax] integers normalized to 0-1000 INSIDE THAT BOX'S ZOOMED CROP IMAGE "
                "(not the full overview image) tightly enclosing the target object."
            )
            refine_rule_user = (
                "For each candidate box where box_quality is 'loose' or 'partial', return `refined_box_2d` "
                "in [ymin, xmin, ymax, xmax] (0-1000) relative to its corresponding zoomed crop image.\n"
            )
        else:
            refine_rule_user = (
                "Also set `box_quality` ('tight', 'loose', or 'partial') for each kept/relabeled box; "
                "for 'delete' set `box_quality` to 'tight', and leave `refined_box_2d` as null for all boxes.\n"
            )

    system_prompt = (
        "You are a strict, accurate computer vision data quality reviewer. "
        "Your job is to inspect candidate bounding boxes drawn on an image and output ONLY valid JSON "
        "matching the requested schema.\n"
        f"Allowed target classes: [{classes_str}].\n"
        "Rules:\n"
        "1. For each numbered candidate box [box_id], decide:\n"
        "   - 'keep': the box genuinely encloses an object of its labeled class.\n"
        "   - 'relabel': the box encloses a valid target object, but its class is wrong; set new_label to an EXACT name from Allowed target classes.\n"
        "   - 'delete': false positive / background noise / severely wrong non-object region.\n"
        "2. Check if there are any unannotated target objects from Allowed target classes that are completely missing a box."
        f"{refine_rule_sys}"
    )
    if config.custom_guideline.strip():
        system_prompt += f"\nDomain Annotation Guidelines:\n{config.custom_guideline.strip()}"

    box_lines = []
    for b in boxes:
        conf_part = f", conf={b.confidence:.2f}" if b.confidence is not None else ""
        s = coord_scale
        box_lines.append(
            f"- box_id={b.box_id}, label='{b.label}'{conf_part}, "
            f"xyxy=[{int(b.x1 * s)}, {int(b.y1 * s)}, {int(b.x2 * s)}, {int(b.y2 * s)}]"
        )
    boxes_summary = "\n".join(box_lines) if box_lines else "(No candidate boxes were predicted on this image.)"

    missing_instr = (
        "Also inspect the entire image for ANY missed targets belonging to Allowed target classes that have NO box yet. "
        "If any are missing, set has_missing=true, explain briefly in missing_note, and "
        + (
            "return each missed object in added_boxes with label (from Allowed target classes) and box_2d as [ymin, xmin, ymax, xmax] integers normalized to 0-1000."
            if config.add_missing_boxes
            else "leave added_boxes as an empty list []."
        )
        if config.check_missing
        else "Do not search for missing objects; set has_missing=false and added_boxes=[]."
    )

    user_prompt = (
        f"Review the candidate bounding boxes shown on the first image (numbered [0], [1], ...).\n"
        f"Candidate xyxy coordinates are in pixels of the first image as sent to you.\n"
        f"Candidate boxes:\n{boxes_summary}\n\n"
        f"{refine_rule_user}"
        f"{missing_instr}\n\n"
        "Return strict JSON with keys: box_reviews (list of {box_id (int), "
        "action ('keep'|'relabel'|'delete'), new_label (str|null), "
        "box_quality ('tight'|'loose'|'partial'), refined_box_2d ([ymin,xmin,ymax,xmax]|null), reason (str)}), "
        "has_missing (bool), missing_note (str), added_boxes (list of {label, box_2d, reason})."
    )
    return system_prompt, user_prompt


def _build_zero_shot_prompts(
    allowed_classes: list[str],
    config: MllmReviewConfig,
) -> tuple[str, str]:
    """构造 LLM 零样本直接标注模式的系统提示词与用户提示词。"""
    classes_str = ", ".join(allowed_classes)
    system_prompt = (
        "You are a strict, accurate computer vision object detection annotator. "
        "Your job is to inspect the provided image from scratch and output ONLY valid JSON "
        "matching the requested schema.\n"
        f"Allowed target classes: [{classes_str}].\n"
        "Rules:\n"
        "1. Detect every clearly visible target object belonging to Allowed target classes and list them in `added_boxes`.\n"
        "2. For each detected object in `added_boxes`:\n"
        "   - `label` MUST be an EXACT name from Allowed target classes (never invent new classes).\n"
        "   - `box_2d` MUST be [ymin, xmin, ymax, xmax] integers normalized to 0-1000 tightly enclosing the visible target.\n"
        "   - Avoid duplicate overlapping boxes for the same object.\n"
        "   - `reason` gives a brief description (Chinese or English).\n"
        "3. Always return `box_reviews` as an empty list [].\n"
        "4. If no target objects exist in the image, return `added_boxes=[]` and `has_missing=false`.\n"
        "5. Set `has_missing=true` and describe in `missing_note` ONLY if the scene is too crowded, occluded, or ambiguous "
        "to accurately box every target; otherwise set `has_missing=false`."
    )
    if config.custom_guideline.strip():
        system_prompt += f"\nDomain Annotation Guidelines:\n{config.custom_guideline.strip()}"

    user_prompt = (
        "Detect and tightly bound ALL target objects in this image belonging to Allowed target classes.\n"
        "Return strict JSON with keys: box_reviews (always []), "
        "has_missing (bool), missing_note (str), "
        "added_boxes (list of {label, box_2d, reason} where box_2d is [ymin, xmin, ymax, xmax] normalized to 0-1000)."
    )
    return system_prompt, user_prompt


def _normalize_class_name(raw_name: str | None, allowed_classes: list[str]) -> str | None:
    if not raw_name:
        return None
    cleaned = raw_name.strip()
    if not allowed_classes:
        return cleaned or None
    if cleaned in allowed_classes:
        return cleaned
    lower_map = {c.lower(): c for c in allowed_classes}
    return lower_map.get(cleaned.lower())


def _box_iou(a: CandidateBox, b: CandidateBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a.x2 - a.x1) * (a.y2 - a.y1))
    area_b = max(1.0, (b.x2 - b.x1) * (b.y2 - b.y1))
    return inter / (area_a + area_b - inter)


def _box_2d_to_pixel_xyxy(
    box_2d: list[int], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    if len(box_2d) != 4:
        return None
    ymin, xmin, ymax, xmax = box_2d
    x1 = max(0.0, min(float(img_w), (float(xmin) / 1000.0) * img_w))
    y1 = max(0.0, min(float(img_h), (float(ymin) / 1000.0) * img_h))
    x2 = max(0.0, min(float(img_w), (float(xmax) / 1000.0) * img_w))
    y2 = max(0.0, min(float(img_h), (float(ymax) / 1000.0) * img_h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 2.0 or (y2 - y1) <= 2.0:
        return None
    return x1, y1, x2, y2


def map_crop_box_2d_to_image_xyxy(
    box_2d: list[int],
    crop_region: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """将局部裁剪图坐标系（`0..1000` 的 `[ymin, xmin, ymax, xmax]`）精确映射回原图像素坐标 `(x1, y1, x2, y2)`。"""
    if len(box_2d) != 4:
        return None
    cx1, cy1, cx2, cy2 = crop_region
    cw = max(1, cx2 - cx1)
    ch = max(1, cy2 - cy1)
    ymin, xmin, ymax, xmax = box_2d
    x1 = cx1 + (float(xmin) / 1000.0) * cw
    y1 = cy1 + (float(ymin) / 1000.0) * ch
    x2 = cx1 + (float(xmax) / 1000.0) * cw
    y2 = cy1 + (float(ymax) / 1000.0) * ch
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 2.0 or (y2 - y1) <= 2.0:
        return None
    return x1, y1, x2, y2


def evaluate_box_refinement(
    orig_box: CandidateBox,
    new_xyxy: tuple[float, float, float, float],
    *,
    min_area_ratio: float = 0.15,
    max_area_ratio: float = 1.60,
    min_iou: float = 0.25,
) -> Literal["accepted", "unchanged", "suspect"]:
    """对自动收紧后的新边界框执行安全门禁评估，防止缩过头或偏移到无关目标。

    判定结果：
        - `"unchanged"`：新旧框 IoU >= 0.96 且四边最大像素偏移 <= 2.0px，认为原框已足够紧贴；
        - `"suspect"`：面积比超出 `[min_area_ratio, max_area_ratio]` 或 IoU < `min_iou`，
          认为疑似误抠局部/偏移异常，调用方应保留原框并分流人工复核；
        - `"accepted"`：收紧幅度合理，可安全采用新框坐标。
    """
    nx1, ny1, nx2, ny2 = new_xyxy
    old_area = max(1.0, (orig_box.x2 - orig_box.x1) * (orig_box.y2 - orig_box.y1))
    new_area = max(0.0, (nx2 - nx1) * (ny2 - ny1))
    if new_area <= 4.0:
        return "suspect"

    probe = CandidateBox(box_id=orig_box.box_id, label=orig_box.label, x1=nx1, y1=ny1, x2=nx2, y2=ny2)
    iou = _box_iou(orig_box, probe)
    max_edge_delta = max(
        abs(nx1 - orig_box.x1),
        abs(ny1 - orig_box.y1),
        abs(nx2 - orig_box.x2),
        abs(ny2 - orig_box.y2),
    )
    if iou >= 0.96 and max_edge_delta <= 2.0:
        return "unchanged"

    area_ratio = new_area / old_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio or iou < min_iou:
        return "suspect"
    return "accepted"


def _apply_candidate_refinement(
    target_box: CandidateBox,
    new_xyxy: tuple[float, float, float, float] | None,
    *,
    source_tag: str,
    min_area_ratio: float,
    min_iou: float,
    mark_missing_as_suspect: bool = True,
) -> None:
    """将单框精修坐标经安全门禁后原地更新至 `target_box`；超门禁时保留原框并标为 `refine_suspect`。"""
    if new_xyxy is None:
        if mark_missing_as_suspect and target_box.status != "relabel_invalid":
            target_box.status = "refine_suspect"
            target_box.reason = f"{source_tag}未检出有效紧贴轮廓，需人工确认; {target_box.reason}".strip("; ")
        return

    verdict = evaluate_box_refinement(
        target_box, new_xyxy, min_area_ratio=min_area_ratio, min_iou=min_iou
    )
    if verdict == "accepted":
        target_box.original_xyxy = (target_box.x1, target_box.y1, target_box.x2, target_box.y2)
        target_box.x1, target_box.y1, target_box.x2, target_box.y2 = new_xyxy
        target_box.refined = True
        if target_box.status == "kept":
            target_box.status = "refined"
        target_box.reason = f"{source_tag}已收紧边框; {target_box.reason}".strip("; ")
    elif verdict == "suspect":
        if target_box.status != "relabel_invalid":
            target_box.status = "refine_suspect"
        target_box.reason = f"{source_tag}收框幅度超安全门禁已保留原框; {target_box.reason}".strip("; ")


def apply_sam_box_refinement(
    image_bgr: np.ndarray,
    after_boxes: list[CandidateBox],
    response: MllmReviewResponse | None,
    refiner: BoxRefinerProtocol,
    *,
    refine_all_kept: bool = False,
    include_llm_added: bool = False,
    min_area_ratio: float = 0.15,
    min_iou: float = 0.25,
) -> None:
    """在原地对满足条件的有效框执行本地 SAM 像素级收框并套用安全门禁。

    触发范围：
        默认跳过 `llm_added` 与 `relabel_invalid`，仅收紧判定过松（或 `refine_all_kept=True` 时的全部保留/改类）框；
        在 `LLM_ZERO_SHOT` 零样本模式下传入 `include_llm_added=True`，对 LLM 直接定位出的粗框统一执行 SAM 收边。
    """
    decision_map = {d.box_id: d for d in response.box_reviews} if response is not None else {}
    targets: list[CandidateBox] = []
    explicit_loose_ids: set[int] = set()
    for b in after_boxes:
        if b.status == "relabel_invalid":
            continue
        if b.status == "llm_added":
            if include_llm_added:
                targets.append(b)
            continue
        dec = decision_map.get(b.box_id)
        is_loose = dec is not None and dec.box_quality in {"loose", "partial"}
        if is_loose:
            explicit_loose_ids.add(b.box_id)
        if refine_all_kept or is_loose:
            targets.append(b)

    if not targets:
        return

    refined_map = refiner.refine_boxes(image_bgr, targets)
    for b in targets:
        _apply_candidate_refinement(
            b,
            refined_map.get(b.box_id),
            source_tag="SAM",
            min_area_ratio=min_area_ratio,
            min_iou=min_iou,
            mark_missing_as_suspect=b.box_id in explicit_loose_ids,
        )


def apply_review_decisions(
    before_boxes: list[CandidateBox],
    response: MllmReviewResponse,
    allowed_classes: list[str],
    img_w: int,
    img_h: int,
    add_missing_boxes: bool = True,
    *,
    box_refine_mode: BoxRefineMode = BoxRefineMode.OFF,
    crop_regions: dict[int, tuple[int, int, int, int]] | None = None,
    refine_min_area_ratio: float = 0.15,
    refine_min_iou: float = 0.25,
) -> tuple[list[CandidateBox], int, int, int, int]:
    """将 LLM 质检与边框紧贴决策合并到候选框集合中，并过滤重复补充框。

    返回兼容的 5 元组 `(after_boxes, kept, relabeled, deleted, added)`；
    未出现在 `box_reviews` 中的候选框按保留（`kept`）处理，
    要求改为非允许类别的框计入 `kept` 但状态为 `relabel_invalid`，
    启用 `FLAG_LOOSE` 或 `MLLM_CROP` 时同步将过松或超安全门禁的框标记为 `refine_suspect`（调用方据此分流到 `need_review`），
    新增补充框若与同类别现有框 IoU > 0.65 则视为重复并跳过。
    """
    decision_map = {d.box_id: d for d in response.box_reviews}
    regions = crop_regions or {}
    after_boxes: list[CandidateBox] = []
    kept = relabeled = deleted = added = 0
    next_id = max((b.box_id for b in before_boxes), default=-1) + 1

    for box in before_boxes:
        decision = decision_map.get(box.box_id)
        out_box: CandidateBox | None = None
        if decision is None or decision.action == "keep":
            kept += 1
            out_box = CandidateBox(
                box_id=box.box_id,
                label=box.label,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                confidence=box.confidence,
                status="kept",
                reason=decision.reason if decision else "",
                original_label=box.label,
            )
        elif decision.action == "relabel":
            resolved = _normalize_class_name(decision.new_label, allowed_classes)
            if resolved and resolved != box.label:
                relabeled += 1
                out_box = CandidateBox(
                    box_id=box.box_id,
                    label=resolved,
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    confidence=box.confidence,
                    status="relabeled",
                    reason=decision.reason,
                    original_label=box.label,
                )
            else:
                kept += 1
                invalid = bool(decision.new_label) and resolved is None
                out_box = CandidateBox(
                    box_id=box.box_id,
                    label=box.label,
                    x1=box.x1,
                    y1=box.y1,
                    x2=box.x2,
                    y2=box.y2,
                    confidence=box.confidence,
                    status="relabel_invalid" if invalid else "kept",
                    reason=(
                        f"模型建议改为非法类别 '{decision.new_label}'; {decision.reason}"
                        if invalid
                        else decision.reason
                    ),
                    original_label=box.label,
                )
        elif decision.action == "delete":
            deleted += 1

        if out_box is not None:
            if (
                decision is not None
                and decision.box_quality in {"loose", "partial"}
                and out_box.status != "relabel_invalid"
            ):
                if box_refine_mode == BoxRefineMode.FLAG_LOOSE:
                    out_box.status = "refine_suspect"
                    q_label = "边框过松含过多背景" if decision.box_quality == "loose" else "边框截断不完整"
                    out_box.reason = f"{q_label}; {out_box.reason}".strip("; ")
                elif box_refine_mode == BoxRefineMode.MLLM_CROP:
                    mapped_xyxy: tuple[float, float, float, float] | None = None
                    if decision.refined_box_2d and box.box_id in regions:
                        mapped_xyxy = map_crop_box_2d_to_image_xyxy(
                            decision.refined_box_2d, regions[box.box_id], img_w, img_h
                        )
                    _apply_candidate_refinement(
                        out_box,
                        mapped_xyxy,
                        source_tag="LLM局部精修",
                        min_area_ratio=refine_min_area_ratio,
                        min_iou=refine_min_iou,
                        mark_missing_as_suspect=True,
                    )
            after_boxes.append(out_box)

    if add_missing_boxes:
        for item in response.added_boxes:
            resolved_cls = _normalize_class_name(item.label, allowed_classes)
            if not resolved_cls:
                continue
            coords = _box_2d_to_pixel_xyxy(item.box_2d, img_w, img_h)
            if coords is None:
                continue
            cand = CandidateBox(
                box_id=next_id,
                label=resolved_cls,
                x1=coords[0],
                y1=coords[1],
                x2=coords[2],
                y2=coords[3],
                confidence=None,
                status="llm_added",
                reason=item.reason,
                original_label="",
            )
            # 过滤与同类别现有框高度重叠（IoU > 0.65）的重复补充框
            if any(
                existing.label == cand.label and _box_iou(existing, cand) > 0.65
                for existing in after_boxes
            ):
                continue
            next_id += 1
            added += 1
            after_boxes.append(cand)

    return after_boxes, kept, relabeled, deleted, added


def _write_xanylabeling_json(
    json_path: Path,
    image_filename: str,
    img_w: int,
    img_h: int,
    boxes: list[CandidateBox],
    has_missing: bool,
    missing_note: str,
) -> None:
    shapes = []
    for b in boxes:
        shapes.append(
            {
                "label": b.label,
                "score": round(b.confidence, 4) if b.confidence is not None else None,
                "points": [[round(b.x1, 2), round(b.y1, 2)], [round(b.x2, 2), round(b.y2, 2)]],
                "group_id": None,
                "description": b.status if not b.reason else f"{b.status}: {b.reason}",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {"llm_added": b.status == "llm_added", "box_refined": b.refined},
            }
        )
    payload = {
        "version": "2.4.4",
        "flags": {
            "has_missing_hint": has_missing,
            "missing_note": missing_note,
        },
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": int(img_h),
        "imageWidth": int(img_w),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_existing_json_boxes(json_path: Path) -> list[CandidateBox]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    boxes: list[CandidateBox] = []
    for idx, shape in enumerate(data.get("shapes", [])):
        label = str(shape.get("label", "")).strip()
        points = shape.get("points") or []
        if not label or len(points) < 2:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        conf = shape.get("score")
        boxes.append(
            CandidateBox(
                box_id=idx,
                label=label,
                x1=min(xs),
                y1=min(ys),
                x2=max(xs),
                y2=max(ys),
                confidence=float(conf) if conf is not None else None,
                original_label=label,
            )
        )
    return boxes


def _discover_yolo_candidates(
    images: list[Path],
    config: MllmReviewConfig,
    log_callback: Callable[[str], None] | None,
    cancel_event: Event | None = None,
) -> tuple[dict[Path, list[CandidateBox]], list[str]]:
    """使用基准模型逐张低阈值推理获取候选框；收到取消信号时提前返回已推理部分。"""
    from ultralytics import YOLO

    from yolo_studio.core.trainer import ensure_torch_cpu_threads

    if not config.weights_path or not config.weights_path.is_file():
        raise FileNotFoundError(f"基准模型权重文件不存在: {config.weights_path}")

    ensure_torch_cpu_threads()
    if log_callback:
        log_callback(
            f"正在使用基准模型 {config.weights_path.name} 对 {len(images)} 张图片执行低阈值推理 (conf={config.yolo_conf})..."
        )
    model = YOLO(str(config.weights_path))
    model_names = [str(v) for _, v in sorted(model.names.items())] if isinstance(model.names, dict) else []

    candidates_map: dict[Path, list[CandidateBox]] = {}
    for img_path in images:
        if cancel_event and cancel_event.is_set():
            break
        results = model.predict(
            source=str(img_path),
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            imgsz=config.yolo_imgsz,
            device=config.yolo_device,
            verbose=False,
        )
        boxes: list[CandidateBox] = []
        if results and results[0].boxes is not None:
            r_boxes = results[0].boxes
            xyxy_list = r_boxes.xyxy.cpu().tolist()
            cls_list = r_boxes.cls.cpu().tolist()
            conf_list = r_boxes.conf.cpu().tolist()
            for idx, (xyxy, cls_id, score) in enumerate(zip(xyxy_list, cls_list, conf_list)):
                cid = int(cls_id)
                label = (
                    model.names.get(cid, f"class_{cid}")
                    if isinstance(model.names, dict)
                    else f"class_{cid}"
                )
                boxes.append(
                    CandidateBox(
                        box_id=idx,
                        label=str(label),
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=float(score),
                        original_label=str(label),
                    )
                )
        candidates_map[img_path] = boxes

    return candidates_map, model_names


def _discover_sam3_candidates(
    images: list[Path],
    class_names: list[str],
    text_prompts: list[str],
    config: MllmReviewConfig,
    log_callback: Callable[[str], None] | None,
    cancel_event: Event | None = None,
    *,
    detector_override: Sam3DetectorProtocol | None = None,
) -> tuple[dict[Path, list[CandidateBox]], dict[Path, str]]:
    """使用本地 SAM 3 按文本概念提示逐张推理获取像素级贴边候选框；单图异常不中断整批。"""
    if detector_override is not None:
        detector: Sam3DetectorProtocol = detector_override
    else:
        resolved_weights = resolve_sam3_weights(config.weights_path)
        if resolved_weights is None:
            raise FileNotFoundError(
                f"SAM 3 权重文件不存在: {config.weights_path or 'weights/sam3.pt'}（可从魔搭 facebook/sam3 下载 sam3.pt 放入 weights/ 目录后自动识别）。"
            )
        if log_callback:
            log_callback(
                f"正在加载本地 SAM 3 模型 {resolved_weights.name} 对 {len(images)} 张图片执行文本自动标注 "
                f"(提示词: {text_prompts}, conf={config.yolo_conf}, device={config.yolo_device})..."
            )
        detector = UltralyticsSam3Detector(
            resolved_weights,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            imgsz=config.yolo_imgsz,
            device=config.yolo_device,
        )

    candidates_map: dict[Path, list[CandidateBox]] = {}
    load_errors: dict[Path, str] = {}
    for img_path in images:
        if cancel_event and cancel_event.is_set():
            break
        try:
            candidates_map[img_path] = detector.detect_image(img_path, class_names, text_prompts)
        except Exception as exc:
            candidates_map[img_path] = []
            load_errors[img_path] = f"SAM 3 推理失败: {exc}"
    return candidates_map, load_errors


def _discover_json_candidates(
    images: list[Path],
    config: MllmReviewConfig,
) -> tuple[dict[Path, list[CandidateBox]], list[str], dict[Path, str]]:
    """读取已有 X-AnyLabeling `.json` 作为候选框。

    返回 `(候选框映射, 类别发现顺序, 读取失败映射)`；单个标注文件损坏不会中断整批，
    而是记录到读取失败映射中，由调用方将该图分流至 `need_review`。
    同名（同 `stem`）图片不使用平铺标注回退匹配，避免跨子目录误配。
    """
    ann_dir = config.annotations_dir if config.annotations_dir else config.images_dir
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")

    colliding = stem_collisions(images)
    candidates_map: dict[Path, list[CandidateBox]] = {}
    discovered_classes: list[str] = []
    seen_cls: set[str] = set()
    load_errors: dict[Path, str] = {}

    for img_path in images:
        json_file = resolve_label_file(
            img_path,
            config.images_dir,
            ann_dir,
            ".json",
            allow_flat_fallback=img_path.stem not in colliding,
        )
        boxes: list[CandidateBox] = []
        if json_file:
            try:
                boxes = _load_existing_json_boxes(json_file)
            except (OSError, UnicodeDecodeError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
                load_errors[img_path] = f"读取已有标注失败 {json_file.name}: {exc}"
        for b in boxes:
            if b.label not in seen_cls:
                seen_cls.add(b.label)
                discovered_classes.append(b.label)
        candidates_map[img_path] = boxes

    return candidates_map, discovered_classes, load_errors


def _plan_output_names(images: list[Path], images_dir: Path) -> dict[Path, str]:
    """为每张输入图片规划输出文件名，保证同一输出目录内图片与 `.json` 均不重名。

    文件 `stem`（不区分大小写，兼容 Windows）唯一时沿用原文件名；
    否则用 `相对子目录__stem__扩展名` 拼接消歧（例如 `cam1__frame_001__jpg.jpg`），
    仍冲突时追加序号。规划在提交并发任务前一次性完成，因此各线程写出的文件互不覆盖。
    """
    stem_counts = Counter(p.stem.lower() for p in images)
    used_stems: set[str] = set()
    names: dict[Path, str] = {}
    for p in images:
        if stem_counts[p.stem.lower()] == 1:
            stem = p.stem
        else:
            try:
                rel_parent_parts = list(p.parent.relative_to(images_dir).parts)
            except ValueError:
                rel_parent_parts = []
            stem = "__".join(rel_parent_parts + [p.stem, p.suffix.lstrip(".").lower()])
        candidate = stem
        idx = 1
        while candidate.lower() in used_stems:
            candidate = f"{stem}_{idx}"
            idx += 1
        used_stems.add(candidate.lower())
        names[p] = f"{candidate}{p.suffix}"
    return names


def _remove_stale_bucket_outputs(output_dir: Path, out_name: str, keep_bucket: str) -> None:
    """删除同名输出在其他分流桶中的旧图片与 `.json`，防止重复标注后同一图片同时留在多个桶中。

    仅作用于 `output_dir/{auto_passed,llm_fixed,need_review}` 下与 `out_name` 同名的文件；
    若本次输入恰好来自其中某个桶（例如复检 `need_review`），该输入会在新结果写出后被移除，
    效果等同于把图片移动到新的分流桶。
    """
    for bucket in TRIAGE_BUCKETS:
        if bucket == keep_bucket:
            continue
        stale_img = output_dir / bucket / out_name
        for stale in (stale_img, stale_img.with_suffix(".json")):
            if stale.is_file():
                stale.unlink()


def _serialize_candidate_box(b: CandidateBox) -> dict[str, object]:
    """将 `CandidateBox` 序列化为可写入 `review_report.json` 的纯 JSON 字典。"""
    return {
        "box_id": b.box_id,
        "label": b.label,
        "x1": round(b.x1, 2),
        "y1": round(b.y1, 2),
        "x2": round(b.x2, 2),
        "y2": round(b.y2, 2),
        "confidence": round(b.confidence, 4) if b.confidence is not None else None,
        "status": b.status,
        "reason": b.reason,
        "original_label": b.original_label,
        "refined": b.refined,
        "original_xyxy": [round(v, 2) for v in b.original_xyxy] if b.original_xyxy else None,
    }


def _deserialize_candidate_box(raw: dict[str, object], fallback_id: int = 0) -> CandidateBox:
    """将 `review_report.json` 中记录的框字典还原为 `CandidateBox`。"""
    orig_raw = raw.get("original_xyxy")
    orig_xyxy: tuple[float, float, float, float] | None = None
    if isinstance(orig_raw, list) and len(orig_raw) == 4:
        orig_xyxy = (float(orig_raw[0]), float(orig_raw[1]), float(orig_raw[2]), float(orig_raw[3]))
    conf_raw = raw.get("confidence")
    return CandidateBox(
        box_id=int(raw.get("box_id", fallback_id)),  # type: ignore[arg-type]
        label=str(raw.get("label", "")),
        x1=float(raw.get("x1", 0.0)),  # type: ignore[arg-type]
        y1=float(raw.get("y1", 0.0)),  # type: ignore[arg-type]
        x2=float(raw.get("x2", 0.0)),  # type: ignore[arg-type]
        y2=float(raw.get("y2", 0.0)),  # type: ignore[arg-type]
        confidence=float(conf_raw) if conf_raw is not None else None,  # type: ignore[arg-type]
        status=str(raw.get("status", "kept")),
        reason=str(raw.get("reason", "")),
        original_label=str(raw.get("original_label", "")),
        refined=bool(raw.get("refined", False)),
        original_xyxy=orig_xyxy,
    )


def collect_review_images(
    images_dir: Path, recursive: bool = False, output_dir: Path | None = None
) -> list[Path]:
    """返回本次智能标注的输入图片列表（只读）；递归时跳过位于输入目录内部的输出目录。

    设计原因：
        递归扫描且 `output_dir` 位于 `images_dir` 内部（如 `raw_frames/smart_labeled/`）时，
        上一轮写出的 `auto_passed` / `llm_fixed` / `need_review` 结果会被重新当作输入处理，
        造成重复标注与结果互相覆盖，因此排除该输出子树。
        `output_dir` 与 `images_dir` 相同、或输入目录位于输出目录内部（如复检 `out/need_review`
        并输出回 `out`）时不做排除，保持原有复检行为。非递归扫描只看顶层，无需排除。
    路径无法解析时按未排除处理，不抛出异常。
    """
    images = find_images(images_dir, recursive=recursive)
    if not recursive or output_dir is None:
        return images
    try:
        in_root = images_dir.resolve()
        out_root = output_dir.resolve()
    except OSError:
        return images
    if out_root == in_root or not out_root.is_relative_to(in_root):
        return images
    return [p for p in images if not p.resolve().is_relative_to(out_root)]


def count_review_images(
    images_dir: Path, recursive: bool = False, output_dir: Path | None = None
) -> int:
    """安全统计待标注目录内的受支持图片数量（口径同 `collect_review_images`）；目录不存在或无权限访问时返回 `0`。"""
    if not images_dir or not images_dir.is_dir():
        return 0
    try:
        return len(collect_review_images(images_dir, recursive=recursive, output_dir=output_dir))
    except OSError:
        return 0


def _is_redundant_error_note(error: str, note: str) -> bool:
    """判断 `missing_note` 是否仅是对 `error` 的重复包装（兼容历史报告中的 `处理出错: <error>`）。"""
    err = error.strip()
    txt = note.strip()
    return bool(err and txt and err in txt)


def format_record_detail_summary(record: ImageReviewRecord) -> str:
    """将单张标注记录的逐框删改/收框/补漏理由整理为界面可读的简明中文摘要。"""
    parts: list[str] = []
    if record.error:
        parts.append(f"处理异常：{record.error}")

    # 收集被删除框的理由（新版记录在 before_boxes 标记 status="deleted"，旧报告按 box_id 差集回退）
    after_ids = {b.box_id for b in record.after_boxes}
    for b in record.before_boxes:
        if b.status == "deleted" or (b.box_id not in after_ids and record.deleted_count > 0):
            reason = f"（{b.reason}）" if b.reason else ""
            parts.append(f"删误检 #{b.box_id}[{b.label}]{reason}")

    for b in record.after_boxes:
        reason = f"（{b.reason}）" if b.reason else ""
        if b.status == "relabeled":
            arrow = f"{b.original_label}→{b.label}" if b.original_label else b.label
            tight = "+收框" if b.refined else ""
            parts.append(f"改类{tight} #{b.box_id}[{arrow}]{reason}")
        elif b.status == "relabel_invalid":
            parts.append(f"非法改类待修 #{b.box_id}[{b.label}]{reason}")
        elif b.status == "refine_suspect":
            parts.append(f"过松/超门禁待修 #{b.box_id}[{b.label}]{reason}")
        elif b.status == "llm_added":
            prefix = "新增标注" if not record.before_boxes else "补漏"
            tight = "+收框" if b.refined else ""
            parts.append(f"{prefix}{tight}[{b.label}]{reason}")
        elif b.refined or b.status == "refined":
            parts.append(f"已收框 #{b.box_id}[{b.label}]{reason}")

    if record.missing_note and not _is_redundant_error_note(record.error, record.missing_note):
        parts.append(f"漏标提示：{record.missing_note}")
    elif record.has_missing and not record.error and not any(b.status == "llm_added" for b in record.after_boxes):
        parts.append("检测到疑似漏标，请人工核查")

    if not parts:
        if record.skipped_mllm:
            if not record.before_boxes and record.after_boxes:
                return f"SAM 3 自动标注检出 {len(record.after_boxes)} 个高置信度目标，自动通过"
            return f"高置信度/空框跳过 LLM（共 {len(record.after_boxes)} 框）"
        return f"全部 {len(record.after_boxes)} 个目标核验通过，无删改或漏标"
    return "；".join(parts)


def load_review_report(report_or_dir: Path) -> MllmReviewResult:
    """从已有智能标注输出目录或 `review_report.json` 加载历史标注记录以供界面复核与对比预览。

    兼容策略：
        优先读取报告中保存的 `before_boxes` / `after_boxes`（保留完整删改、精修虚线对比信息）；
        若为旧版报告缺失框明细，则尽力从 `output_json` 或源图片同目录 `.json` 回退还原。
        文件不存在或 JSON 格式损坏时抛出 `FileNotFoundError` / `ValueError`。
    """
    report_path = report_or_dir / "review_report.json" if report_or_dir.is_dir() else report_or_dir
    if not report_path.is_file():
        raise FileNotFoundError(f"未找到标注报告文件: {report_path}")

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"无法解析标注报告 {report_path.name}: {exc}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"标注报告格式无效（缺少 items 列表）: {report_path}")

    out_dir = report_path.parent
    classes_raw = payload.get("classes") or []
    classes = [str(c).strip() for c in classes_raw if str(c).strip()]
    cls_file = out_dir / "classes.txt"
    if not classes and cls_file.is_file():
        classes = [
            line.strip() for line in cls_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    records: list[ImageReviewRecord] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            continue
        src_path = Path(str(item.get("source") or (out_dir / str(item.get("image", "")))))
        out_img_str = str(item.get("output_image") or "").strip()
        out_json_str = str(item.get("output_json") or "").strip()
        out_img = Path(out_img_str) if out_img_str else None
        out_json = Path(out_json_str) if out_json_str else None

        before_raw = item.get("before_boxes")
        after_raw = item.get("after_boxes")
        before_boxes: list[CandidateBox] = (
            [_deserialize_candidate_box(b, idx) for idx, b in enumerate(before_raw) if isinstance(b, dict)]
            if isinstance(before_raw, list)
            else []
        )
        after_boxes: list[CandidateBox] = (
            [_deserialize_candidate_box(b, idx) for idx, b in enumerate(after_raw) if isinstance(b, dict)]
            if isinstance(after_raw, list)
            else []
        )
        if not after_boxes and out_json and out_json.is_file():
            try:
                after_boxes = _load_existing_json_boxes(out_json)
            except Exception:
                pass
        if not before_boxes and src_path.with_suffix(".json").is_file():
            try:
                before_boxes = _load_existing_json_boxes(src_path.with_suffix(".json"))
            except Exception:
                pass

        records.append(
            ImageReviewRecord(
                image_path=src_path,
                output_image_path=out_img,
                output_json_path=out_json,
                triage_bucket=str(item.get("bucket") or BUCKET_NEED_REVIEW),
                before_boxes=before_boxes,
                after_boxes=after_boxes,
                kept_count=int(item.get("kept", 0) or 0),
                relabeled_count=int(item.get("relabeled", 0) or 0),
                deleted_count=int(item.get("deleted", 0) or 0),
                added_count=int(item.get("added", 0) or 0),
                refined_count=int(item.get("refined", 0) or 0),
                loose_flagged_count=int(item.get("loose_flagged", 0) or 0),
                has_missing=bool(item.get("has_missing", False)),
                missing_note=str(item.get("missing_note") or ""),
                error=str(item.get("error") or ""),
            )
        )

    auto_passed = sum(1 for r in records if r.triage_bucket == BUCKET_AUTO_PASSED)
    llm_fixed = sum(1 for r in records if r.triage_bucket == BUCKET_LLM_FIXED)
    need_review = sum(1 for r in records if r.triage_bucket == BUCKET_NEED_REVIEW)
    error_count = sum(1 for r in records if r.error)
    return MllmReviewResult(
        output_dir=out_dir,
        classes=classes,
        total_images=len(records),
        auto_passed_count=auto_passed,
        llm_fixed_count=llm_fixed,
        need_review_count=need_review,
        error_count=error_count,
        records=records,
        report_path=report_path,
        classes_txt_path=cls_file if cls_file.is_file() else None,
        cancelled=bool(payload.get("cancelled", False)),
    )


def run_mllm_review(
    config: MllmReviewConfig,
    client_override: MllmClient | None = None,
    progress_callback: Callable[[int, int, ImageReviewRecord], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    cancel_event: Event | None = None,
    *,
    refiner_override: BoxRefinerProtocol | None = None,
    sam3_detector_override: Sam3DetectorProtocol | None = None,
) -> MllmReviewResult:
    """执行候选框获取（YOLO 预标 / 已有 `.json` / LLM 零样本 / 本地 SAM 3 文本自动标注）、可选 LLM 并发质检与分流落盘。

    前置条件与失败方式：
        `images_dir` 不存在抛出 `FileNotFoundError`，无图片抛出 `ValueError`；
        在 `YOLO_PREDICT` 模式下需存在有效基准 `weights_path`，在 `EXISTING_JSON` 模式下需存在标注目录，
        在 `LLM_ZERO_SHOT` 模式下必须提供非空 `allowed_classes`，
        在 `SAM3_TEXT` 模式下必须提供非空 `allowed_classes` 与有效的本地 `sam3.pt`（或测试桩 `sam3_detector_override`）。
        当 `SAM3_TEXT` 未开启 `sam3_enable_llm_review` 时以纯本地模式运行，不初始化也不调用 LLM 客户端；
        所有检出框置信度均 >= `high_conf_skip` 且非空时归入 `auto_passed`，存在低置信度框或 0 框时归入 `need_review`。
        单张图片处理异常不会中止整批流程，而是写入 `need_review` 并记录 `error`，保证 `review_report.json` 必落盘。
    """
    if not config.images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {config.images_dir}")

    images = collect_review_images(
        config.images_dir, recursive=config.recursive, output_dir=config.output_dir
    )
    if config.recursive and log_callback:
        skipped = len(find_images(config.images_dir, recursive=True)) - len(images)
        if skipped > 0:
            log_callback(f"输出目录位于输入目录内部，已跳过其中 {skipped} 张上一轮结果图片，避免重复处理。")
    if not images:
        raise ValueError(f"未在目录中找到任何图片: {config.images_dir}")

    total_discovered = len(images)
    if config.max_images > 0 and total_discovered > config.max_images:
        images = images[: config.max_images]
        if log_callback:
            log_callback(
                f"已启用小批量试跑上限：目录共 {total_discovered} 张图，本次仅处理前 {len(images)} 张。"
            )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    is_zero_shot = config.source_mode == ReviewSourceMode.LLM_ZERO_SHOT
    is_sam3_text = config.source_mode == ReviewSourceMode.SAM3_TEXT
    sam3_pure_local = is_sam3_text and not config.sam3_enable_llm_review
    load_errors: dict[Path, str] = {}
    if config.source_mode == ReviewSourceMode.YOLO_PREDICT:
        candidates_map, discovered_classes = _discover_yolo_candidates(
            images, config, log_callback, cancel_event
        )
        allowed_classes = [c.strip() for c in config.allowed_classes if c.strip()] or discovered_classes
    elif config.source_mode == ReviewSourceMode.EXISTING_JSON:
        candidates_map, discovered_classes, load_errors = _discover_json_candidates(images, config)
        allowed_classes = [c.strip() for c in config.allowed_classes if c.strip()] or discovered_classes
    elif is_sam3_text:
        sam3_classes, sam3_prompts = parse_sam3_class_prompts(config.allowed_classes)
        if not sam3_classes:
            raise ValueError(
                "SAM 3 文本自动标注模式必须指定「目标类别清单」（支持“类别名=英文提示词”，如 头盔=safety helmet）。"
            )
        allowed_classes = sam3_classes
        candidates_map, load_errors = _discover_sam3_candidates(
            images,
            sam3_classes,
            sam3_prompts,
            config,
            log_callback,
            cancel_event,
            detector_override=sam3_detector_override,
        )
    else:
        candidates_map = {img_path: [] for img_path in images}
        allowed_classes = [c.strip() for c in config.allowed_classes if c.strip()]
        if not allowed_classes:
            raise ValueError(
                "LLM 零样本直接标注模式必须指定「目标类别清单」（请至少提供一个类别名称或导入 classes.txt）。"
            )

    effective_refine_mode = config.box_refine_mode
    if sam3_pure_local and effective_refine_mode != BoxRefineMode.OFF:
        if log_callback:
            log_callback(
                f"提示：纯本地 SAM 3 模式已输出像素级掩码贴边框，已自动忽略二次边框修正模式 '{effective_refine_mode.value}'。"
            )
        effective_refine_mode = BoxRefineMode.OFF
    elif is_zero_shot and effective_refine_mode in {BoxRefineMode.FLAG_LOOSE, BoxRefineMode.MLLM_CROP}:
        if log_callback:
            log_callback(
                f"提示：零样本直接标注模式仅支持关闭或本地 SAM 像素级收紧，已自动忽略边框修正模式 '{effective_refine_mode.value}'。"
            )
        effective_refine_mode = BoxRefineMode.OFF

    effective_effort = ReasoningEffort.AUTO
    if not sam3_pure_local:
        effective_effort, effort_notice = resolve_effective_reasoning_effort(
            config.provider, config.model_name, config.reasoning_effort
        )
        if effort_notice and log_callback:
            log_callback(f"提示：{effort_notice}。")

    already_cancelled = bool(cancel_event and cancel_event.is_set())
    client: MllmClient | None = None
    if not sam3_pure_local and not already_cancelled:
        client = client_override or build_mllm_client(config)

    sam_refiner: BoxRefinerProtocol | None = None
    if effective_refine_mode == BoxRefineMode.SAM_LOCAL and not already_cancelled:
        if refiner_override is not None:
            sam_refiner = refiner_override
        else:
            if log_callback:
                log_callback(f"正在加载本地 SAM 边框紧贴模型: {config.sam_model_name} (device={config.yolo_device})...")
            sam_refiner = UltralyticsSamBoxRefiner(
                model_name=config.sam_model_name,
                device=config.yolo_device,
            )

    output_names = _plan_output_names(images, config.images_dir)

    if log_callback and not already_cancelled:
        refine_desc = effective_refine_mode.value
        effort_desc = effective_effort.value
        if sam3_pure_local:
            log_callback(
                f"开始纯本地 SAM 3 自动标注分流（共 {len(images)} 张图，类别: {allowed_classes}，"
                f"自动通过阈值 ≥ {config.high_conf_skip:.2f}）..."
            )
        else:
            raw_timeout = getattr(client, "request_timeout", None)
            req_timeout = (
                float(raw_timeout)
                if isinstance(raw_timeout, (int, float)) and raw_timeout > 0
                else resolve_mllm_request_timeout(
                    config.provider, config.model_name, config.reasoning_effort
                )
            )
            timeout_desc = f"，单次超时: {int(round(req_timeout))}s"
            if is_sam3_text:
                log_callback(
                    f"开始 SAM 3 预标 + LLM 二次质检（共 {len(images)} 张图，类别: {allowed_classes}，"
                    f"推理级别: {effort_desc}{timeout_desc}，边框修正: {refine_desc}，并发={config.max_workers}）..."
                )
            else:
                mode_title = "LLM 零样本直接标注" if is_zero_shot else "LLM 智能标注"
                log_callback(
                    f"开始 {mode_title}（共 {len(images)} 张图，类别: {allowed_classes}，"
                    f"推理级别: {effort_desc}{timeout_desc}，边框修正: {refine_desc}，并发={config.max_workers}）..."
                )

    def _bucket_dir(bucket: str) -> Path:
        return (config.output_dir / bucket) if config.triage_subfolders else config.output_dir

    def _copy_image(img_path: Path, bucket: str) -> Path:
        target_dir = _bucket_dir(bucket)
        target_dir.mkdir(parents=True, exist_ok=True)
        out_img_path = target_dir / output_names[img_path]
        if out_img_path.resolve() != img_path.resolve():
            shutil.copy2(img_path, out_img_path)
        return out_img_path

    def _write_error_record(
        img_path: Path, before_boxes: list[CandidateBox], exc: BaseException
    ) -> ImageReviewRecord:
        """把出错图片连同原始候选框写入 `need_review`；写出过程再失败时仅记录错误，不向外抛出。"""
        message = _format_exception_detail(exc)
        out_img_path: Path | None = None
        out_json_path: Path | None = None
        try:
            out_img_path = _copy_image(img_path, BUCKET_NEED_REVIEW)
            try:
                with Image.open(img_path) as pil_img:
                    iw, ih = pil_img.size
            except Exception:
                iw, ih = 0, 0
            out_json_path = out_img_path.with_suffix(".json")
            _write_xanylabeling_json(
                out_json_path,
                out_img_path.name,
                img_w=iw,
                img_h=ih,
                boxes=before_boxes,
                has_missing=True,
                missing_note=f"LLM 处理异常: {message}",
            )
            if config.triage_subfolders:
                _remove_stale_bucket_outputs(config.output_dir, out_img_path.name, BUCKET_NEED_REVIEW)
        except Exception as write_exc:
            message = f"{message}；错误结果写出失败: {write_exc}"
            out_json_path = out_json_path if out_json_path and out_json_path.is_file() else None
        return ImageReviewRecord(
            image_path=img_path,
            output_image_path=out_img_path,
            output_json_path=out_json_path,
            triage_bucket=BUCKET_NEED_REVIEW,
            before_boxes=before_boxes,
            after_boxes=before_boxes,
            has_missing=True,
            missing_note="",
            error=message,
        )

    def _process_one(img_path: Path) -> ImageReviewRecord | None:
        if cancel_event and cancel_event.is_set():
            return None
        before_boxes = candidates_map.get(img_path, [])
        try:
            if img_path in load_errors:
                raise RuntimeError(load_errors[img_path])

            with Image.open(img_path) as pil_img:
                img_w, img_h = pil_img.size

            img_bgr: np.ndarray | None = None
            response: MllmReviewResponse | None = None
            if sam3_pure_local:
                raw_sam3_boxes = before_boxes
                before_boxes = []
                after_boxes = []
                has_low_conf = False
                for b in raw_sam3_boxes:
                    conf_val = b.confidence if b.confidence is not None else 0.0
                    is_low = conf_val < config.high_conf_skip
                    if is_low:
                        has_low_conf = True
                    after_boxes.append(
                        CandidateBox(
                            box_id=b.box_id,
                            label=b.label,
                            x1=b.x1,
                            y1=b.y1,
                            x2=b.x2,
                            y2=b.y2,
                            confidence=b.confidence,
                            status="kept",
                            reason=(
                                f"SAM 3 低置信度 (conf={conf_val:.2f} < {config.high_conf_skip:.2f})，建议复核"
                                if is_low
                                else f"SAM 3 自动标注 (conf={conf_val:.2f})"
                            ),
                            original_label=b.label,
                        )
                    )
                kept, relabeled, deleted, added = 0, 0, 0, len(after_boxes)
                if not after_boxes:
                    has_missing = True
                    missing_note = "SAM 3 未检出目标，请人工确认是否为负样本或调低 Conf 阈值"
                elif has_low_conf:
                    has_missing = False
                    missing_note = f"含低置信度目标（< {config.high_conf_skip:.2f}），建议人工复核"
                else:
                    has_missing = False
                    missing_note = ""
                skipped_mllm = True
            elif is_zero_shot:
                assert client is not None
                img_bgr = _read_image_unicode_safe(img_path)
                overview_jpeg = _encode_jpeg_bytes(img_bgr, max_dim=config.max_image_dim, quality=90)
                system_prompt, user_prompt = _build_zero_shot_prompts(allowed_classes, config)
                response = client.review_image(overview_jpeg, [], system_prompt, user_prompt)
                if cancel_event and cancel_event.is_set():
                    return None
                after_boxes, kept, relabeled, deleted, added = apply_review_decisions(
                    [],
                    response,
                    allowed_classes,
                    img_w=img_w,
                    img_h=img_h,
                    add_missing_boxes=True,
                    box_refine_mode=BoxRefineMode.OFF,
                )
                has_missing = response.has_missing
                missing_note = response.missing_note
                skipped_mllm = False
            else:
                # 当未开启漏标巡检且无需借助 LLM 判定框紧贴度时，高置信度图或空框图可直接跳过 LLM
                needs_mllm_for_refine = bool(before_boxes) and (
                    effective_refine_mode in {BoxRefineMode.FLAG_LOOSE, BoxRefineMode.MLLM_CROP}
                    or (effective_refine_mode == BoxRefineMode.SAM_LOCAL and not config.sam_refine_all_kept)
                )
                all_high_conf = bool(before_boxes) and all(
                    (b.confidence is not None and b.confidence >= config.high_conf_skip)
                    for b in before_boxes
                )
                skip_reason = ""
                if not config.check_missing and not needs_mllm_for_refine:
                    if all_high_conf:
                        skip_reason = "高置信度跳过 LLM"
                    elif not before_boxes:
                        skip_reason = "无候选框且未开启漏标检查，跳过 LLM"

                if skip_reason:
                    after_boxes = [
                        CandidateBox(
                            box_id=b.box_id,
                            label=b.label,
                            x1=b.x1,
                            y1=b.y1,
                            x2=b.x2,
                            y2=b.y2,
                            confidence=b.confidence,
                            status="kept",
                            reason=skip_reason,
                            original_label=b.label,
                        )
                        for b in before_boxes
                    ]
                    kept, relabeled, deleted, added = len(after_boxes), 0, 0, 0
                    has_missing = False
                    missing_note = ""
                    skipped_mllm = True
                else:
                    assert client is not None
                    img_bgr = _read_image_unicode_safe(img_path)
                    force_crops = effective_refine_mode == BoxRefineMode.MLLM_CROP
                    overview_jpeg, crops, crop_regions = _render_overview_and_crops(
                        img_bgr,
                        before_boxes,
                        include_crops=config.include_crops,
                        max_crops=config.max_crops_per_image,
                        high_conf_skip=config.high_conf_skip,
                        max_dim=config.max_image_dim,
                        force_crops=force_crops,
                    )
                    system_prompt, user_prompt = _build_prompts(
                        before_boxes,
                        allowed_classes,
                        config,
                        coord_scale=_overview_scale(img_w, img_h, config.max_image_dim),
                    )
                    response = client.review_image(overview_jpeg, crops, system_prompt, user_prompt)
                    if cancel_event and cancel_event.is_set():
                        return None
                    after_boxes, kept, relabeled, deleted, added = apply_review_decisions(
                        before_boxes,
                        response,
                        allowed_classes,
                        img_w=img_w,
                        img_h=img_h,
                        add_missing_boxes=config.add_missing_boxes,
                        box_refine_mode=effective_refine_mode,
                        crop_regions=crop_regions,
                        refine_min_area_ratio=config.refine_min_area_ratio,
                        refine_min_iou=config.refine_min_iou,
                    )
                    has_missing = response.has_missing
                    missing_note = response.missing_note
                    skipped_mllm = False

            # 在不修改上游候选缓存的前提下，把删框理由写入记录快照，便于界面展示与历史报告回看
            record_before_boxes = before_boxes
            if response is not None and deleted > 0:
                del_reasons = {
                    d.box_id: d.reason
                    for d in response.box_reviews
                    if d.action == "delete"
                }
                if del_reasons:
                    record_before_boxes = [
                        CandidateBox(
                            box_id=b.box_id,
                            label=b.label,
                            x1=b.x1,
                            y1=b.y1,
                            x2=b.x2,
                            y2=b.y2,
                            confidence=b.confidence,
                            status="deleted" if b.box_id in del_reasons else b.status,
                            reason=del_reasons.get(b.box_id, b.reason),
                            original_label=b.original_label,
                            refined=b.refined,
                            original_xyxy=b.original_xyxy,
                        )
                        for b in before_boxes
                    ]

            if effective_refine_mode == BoxRefineMode.SAM_LOCAL and sam_refiner is not None and after_boxes:
                if cancel_event and cancel_event.is_set():
                    return None
                if img_bgr is None:
                    img_bgr = _read_image_unicode_safe(img_path)
                apply_sam_box_refinement(
                    img_bgr,
                    after_boxes,
                    response,
                    sam_refiner,
                    refine_all_kept=True if is_zero_shot else config.sam_refine_all_kept,
                    include_llm_added=is_zero_shot,
                    min_area_ratio=config.refine_min_area_ratio,
                    min_iou=config.refine_min_iou,
                )

            if cancel_event and cancel_event.is_set():
                return None

            refined_count = sum(1 for b in after_boxes if b.refined)
            loose_flagged_count = sum(1 for b in after_boxes if b.status == "refine_suspect")
            has_invalid_relabel = any(b.status == "relabel_invalid" for b in after_boxes)

            if sam3_pure_local:
                all_sam3_high = bool(after_boxes) and all(
                    (b.confidence is not None and b.confidence >= config.high_conf_skip)
                    for b in after_boxes
                )
                bucket = BUCKET_AUTO_PASSED if all_sam3_high else BUCKET_NEED_REVIEW
            elif has_missing or added > 0 or has_invalid_relabel or loose_flagged_count > 0:
                bucket = BUCKET_NEED_REVIEW
            elif relabeled > 0 or deleted > 0 or refined_count > 0:
                bucket = BUCKET_LLM_FIXED
            else:
                bucket = BUCKET_AUTO_PASSED

            out_img_path = _copy_image(img_path, bucket)
            out_json_path: Path | None = out_img_path.with_suffix(".json")
            if after_boxes or config.check_missing or is_zero_shot or sam3_pure_local:
                _write_xanylabeling_json(
                    out_json_path,
                    out_img_path.name,
                    img_w=img_w,
                    img_h=img_h,
                    boxes=after_boxes,
                    has_missing=has_missing,
                    missing_note=missing_note,
                )
            else:
                if out_json_path.is_file() and out_json_path.resolve() != img_path.with_suffix(".json").resolve():
                    out_json_path.unlink()
                out_json_path = None
            if config.triage_subfolders:
                _remove_stale_bucket_outputs(config.output_dir, out_img_path.name, bucket)

            return ImageReviewRecord(
                image_path=img_path,
                output_image_path=out_img_path,
                output_json_path=out_json_path,
                triage_bucket=bucket,
                before_boxes=record_before_boxes,
                after_boxes=after_boxes,
                kept_count=kept,
                relabeled_count=relabeled,
                deleted_count=deleted,
                added_count=added,
                refined_count=refined_count,
                loose_flagged_count=loose_flagged_count,
                has_missing=has_missing,
                missing_note=missing_note,
                skipped_mllm=skipped_mllm,
            )
        except Exception as exc:
            # 用户取消时主动关闭客户端会触发 socket 断开异常，此时按取消收尾而非记为单图出错
            if cancel_event and cancel_event.is_set():
                return None
            return _write_error_record(img_path, before_boxes, exc)

    records: list[ImageReviewRecord] = []
    total = len(images)
    cancelled = already_cancelled

    workers = max(1, int(config.max_workers))
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures: dict[Future[ImageReviewRecord | None], Path] = {}
        for img_path in ([] if already_cancelled else images):
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            futures[pool.submit(_process_one, img_path)] = img_path

        pending: set[Future[ImageReviewRecord | None]] = set(futures.keys())
        completed_count = 0
        cancel_notified = False
        cancel_deadline: float | None = None

        while pending:
            if cancel_event and cancel_event.is_set():
                cancelled = True
                for pending_fut in list(pending):
                    if pending_fut.cancel():
                        pending.discard(pending_fut)
                if not cancel_notified:
                    cancel_notified = True
                    _close_client_safely(client)
                    cancel_deadline = time.monotonic() + 1.5
                    if log_callback:
                        log_callback("已收到停止指令，正在中断在途请求并保存已完成结果...")
                if not pending:
                    break
                remaining = (cancel_deadline - time.monotonic()) if cancel_deadline is not None else 0.0
                if remaining <= 0:
                    break
                done, pending = wait(
                    pending,
                    timeout=min(0.1, remaining),
                    return_when=FIRST_COMPLETED,
                )
            else:
                done, pending = wait(
                    pending,
                    timeout=0.1 if cancel_event is not None else None,
                    return_when=FIRST_COMPLETED,
                )

            for fut in done:
                if fut.cancelled():
                    continue
                try:
                    rec = fut.result()
                except Exception as exc:
                    if cancel_event and cancel_event.is_set():
                        cancelled = True
                        continue
                    src = futures[fut]
                    rec = ImageReviewRecord(
                        image_path=src,
                        output_image_path=None,
                        output_json_path=None,
                        triage_bucket=BUCKET_NEED_REVIEW,
                        has_missing=True,
                        missing_note="",
                        error=_format_exception_detail(exc),
                    )
                if rec is None:
                    cancelled = True
                    continue
                records.append(rec)
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total, rec)
                if log_callback:
                    if rec.error:
                        log_callback(
                            f"[{completed_count}/{total}] {rec.image_path.name} -> 处理出错 ({rec.triage_bucket}): {rec.error}"
                        )
                    else:
                        refine_part = f", 收框:{rec.refined_count}" if rec.refined_count else ""
                        loose_part = f", 过松/超门禁:{rec.loose_flagged_count}" if rec.loose_flagged_count else ""
                        if is_zero_shot or sam3_pure_local:
                            log_callback(
                                f"[{completed_count}/{total}] {rec.image_path.name} -> {rec.triage_bucket} "
                                f"(检出:{rec.added_count}{refine_part}{loose_part})"
                            )
                        else:
                            log_callback(
                                f"[{completed_count}/{total}] {rec.image_path.name} -> {rec.triage_bucket} "
                                f"(保留:{rec.kept_count}, 改类:{rec.relabeled_count}, 删误检:{rec.deleted_count}"
                                f"{refine_part}{loose_part}, 补漏:{rec.added_count})"
                            )
    finally:
        if cancel_event and cancel_event.is_set():
            cancelled = True
            _close_client_safely(client)
        elif client_override is None:
            _close_client_safely(client)
        pool.shutdown(wait=not cancelled, cancel_futures=True)

    records.sort(key=lambda r: str(r.image_path))

    auto_passed = sum(1 for r in records if r.triage_bucket == BUCKET_AUTO_PASSED)
    llm_fixed = sum(1 for r in records if r.triage_bucket == BUCKET_LLM_FIXED)
    need_review = sum(1 for r in records if r.triage_bucket == BUCKET_NEED_REVIEW)
    error_count = sum(1 for r in records if r.error)

    classes_txt_path = config.output_dir / "classes.txt"
    if allowed_classes:
        classes_txt_path.write_text("\n".join(allowed_classes) + "\n", encoding="utf-8")

    report_payload = {
        "total_images": len(records),
        "auto_passed": auto_passed,
        "llm_fixed": llm_fixed,
        "need_review": need_review,
        "errors": error_count,
        "cancelled": cancelled,
        "classes": allowed_classes,
        "source_mode": config.source_mode.value,
        "provider": config.provider.value,
        "reasoning_effort": ReasoningEffort.AUTO.value if sam3_pure_local else effective_effort.value,
        "sam3_enable_llm_review": bool(config.sam3_enable_llm_review) if is_sam3_text else False,
        "box_refine_mode": effective_refine_mode.value,
        "items": [
            {
                "image": r.image_path.name,
                "source": str(r.image_path),
                "bucket": r.triage_bucket,
                "kept": r.kept_count,
                "relabeled": r.relabeled_count,
                "deleted": r.deleted_count,
                "added": r.added_count,
                "refined": r.refined_count,
                "loose_flagged": r.loose_flagged_count,
                "has_missing": r.has_missing,
                "missing_note": r.missing_note,
                "error": r.error,
                "output_image": str(r.output_image_path) if r.output_image_path else "",
                "output_json": str(r.output_json_path) if r.output_json_path else "",
                "before_boxes": [_serialize_candidate_box(b) for b in r.before_boxes],
                "after_boxes": [_serialize_candidate_box(b) for b in r.after_boxes],
            }
            for r in records
        ],
    }
    report_path = config.output_dir / "review_report.json"
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return MllmReviewResult(
        output_dir=config.output_dir,
        classes=allowed_classes,
        total_images=len(records),
        auto_passed_count=auto_passed,
        llm_fixed_count=llm_fixed,
        need_review_count=need_review,
        error_count=error_count,
        records=records,
        report_path=report_path,
        classes_txt_path=classes_txt_path if allowed_classes else None,
        cancelled=cancelled,
    )


def _draw_dashed_rect(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_len: int = 8,
) -> None:
    """在图像上绘制虚线矩形，用于对比展示自动收紧前的原始松垮边界框位置。"""
    x1, y1 = p1
    x2, y2 = p2
    step = max(4, dash_len * 2)
    for x in range(x1, x2, step):
        cv2.line(canvas, (x, y1), (min(x + dash_len, x2), y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x, y2), (min(x + dash_len, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, step):
        cv2.line(canvas, (x1, y), (x1, min(y + dash_len, y2)), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y), (x2, min(y + dash_len, y2)), color, thickness, cv2.LINE_AA)


def _draw_labeled_boxes(
    base_bgr: np.ndarray, boxes: list[CandidateBox], banner_text: str, use_status_colors: bool
) -> np.ndarray:
    canvas = base_bgr.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(max(h, w) / 420.0)))
    font_scale = max(0.5, max(h, w) / 1400.0)
    status_colors = {
        "original": (200, 180, 0),
        "kept": (0, 200, 0),
        "relabeled": (0, 140, 255),
        "relabel_invalid": (60, 60, 255),
        "llm_added": (0, 220, 255),
        "refined": (255, 210, 40),
        "refine_suspect": (200, 40, 220),
    }

    for b in boxes:
        if use_status_colors and b.status == "llm_added" and b.refined:
            color = status_colors["refined"]
        else:
            color = status_colors.get(b.status, (0, 200, 0)) if use_status_colors else (200, 180, 0)
        # 在 After 侧用虚线叠加原框位置，直观呈现收紧前后背景裁除幅度
        if use_status_colors and b.refined and b.original_xyxy is not None:
            op1 = (int(round(b.original_xyxy[0])), int(round(b.original_xyxy[1])))
            op2 = (int(round(b.original_xyxy[2])), int(round(b.original_xyxy[3])))
            _draw_dashed_rect(canvas, op1, op2, (160, 160, 160), max(1, thickness - 1))

        p1 = (int(round(b.x1)), int(round(b.y1)))
        p2 = (int(round(b.x2)), int(round(b.y2)))
        cv2.rectangle(canvas, p1, p2, color, thickness)
        if b.status == "relabeled" and b.original_label:
            tag = f"{b.original_label}->{b.label}" + (" [Tight]" if b.refined else "")
        elif b.status == "llm_added":
            tag = f"+{b.label} (LLM)" + (" [Tight]" if b.refined else "")
        elif b.status == "refined":
            tag = f"{b.label} [Tight]"
        elif b.status == "refine_suspect":
            tag = f"{b.label} [Loose?]"
        else:
            tag = b.label
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        ty = max(th + 6, p1[1])
        cv2.rectangle(canvas, (p1[0], ty - th - 6), (p1[0] + tw + 6, ty + baseline - 2), color, -1)
        cv2.putText(canvas, tag, (p1[0] + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)

    bar_h = max(32, int(round(h * 0.06)))
    bar = np.full((bar_h, w, 3), 30, dtype=np.uint8)
    cv2.putText(bar, banner_text, (10, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, font_scale), (240, 240, 240), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def render_review_comparison(record: ImageReviewRecord, max_width: int = 1200) -> bytes:
    """渲染单张标注记录的“处理前 vs 处理后”左右拼接对比 JPEG 字节流。"""
    src_path = record.output_image_path if (record.output_image_path and record.output_image_path.is_file()) else record.image_path
    img_bgr = _read_image_unicode_safe(src_path)
    left = _draw_labeled_boxes(img_bgr, record.before_boxes, f"Before ({len(record.before_boxes)})", use_status_colors=False)
    right = _draw_labeled_boxes(img_bgr, record.after_boxes, f"After ({len(record.after_boxes)})", use_status_colors=True)
    combined = np.hstack([left, right])
    return _encode_jpeg_bytes(combined, max_dim=max_width, quality=90)
