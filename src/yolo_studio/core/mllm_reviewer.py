"""基于 MLLM 的自动标注审核、错标纠正与漏标发现引擎。

职责：
    串联“候选框发现（基准 YOLO 模型低阈值推理或读取已有 X-AnyLabeling `.json`）→
    带编号总览图与局部裁剪图渲染 → 多模态大模型结构化审核 → 决策合并去重 →
    结果分流落盘（`auto_passed` / `llm_fixed` / `need_review`）与前后对比图生成”。

运行环境：
    Python >= 3.11，依赖 `opencv-python`、`numpy`、`Pillow`、`pydantic`；
    按模式额外需要 `ultralytics`（`yolo_predict`）、`google-genai`（Gemini）或 `openai`（OpenAI 兼容接口）。
    调用远程接口需通过配置或环境变量 `GEMINI_API_KEY` / `OPENAI_API_KEY` 提供有效凭据。

协作对象与职责边界：
    为 `cli._cmd_review`、`gui.workers.MllmReviewWorker` 与 `gui.tab_review.MllmReviewTab`
    提供核心管线，复用 `dataset_converter.IMAGE_EXTENSIONS`。
    仅负责审核清洗、X-AnyLabeling `.json` 产出与汇总报告写出，不执行数据集划分或模型训练。
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event
from typing import Callable, Literal, Protocol

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from yolo_studio.core.dataset_converter import IMAGE_EXTENSIONS


class ReviewSourceMode(str, Enum):
    """MLLM 审核的候选标注来源模式。"""

    YOLO_PREDICT = "yolo_predict"
    EXISTING_JSON = "existing_json"


class MllmProvider(str, Enum):
    """支持的多模态大模型接口提供方。"""

    GEMINI = "gemini"
    OPENAI_COMPAT = "openai_compat"


OPENAI_PRESETS: list[tuple[str, str, str]] = [
    ("通义千问 DashScope (Qwen-VL)", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max-latest"),
    ("硅基流动 SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-VL-72B-Instruct"),
    ("本地 Ollama", "http://localhost:11434/v1", "qwen2.5vl:7b"),
    ("本地 vLLM", "http://localhost:8000/v1", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("OpenAI 官方", "https://api.openai.com/v1", "gpt-4o"),
]

GEMINI_MODELS: list[tuple[str, str]] = [
    ("gemini-3.8-flash", "Gemini 3.8 Flash（速度快、精度均衡，推荐）"),
    ("gemini-4-argon", "Gemini 4 Argon（复杂场景最高精度）"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite（超高吞吐、最低成本）"),
]


class BoxDecision(BaseModel):
    """针对单个已有候选框的审核决策。"""

    box_id: int = Field(description="Candidate box ID shown in the overview image.")
    action: Literal["keep", "relabel", "delete"] = Field(
        description="keep if label and object are valid; relabel if wrong class; delete if false positive/background."
    )
    new_label: str | None = Field(
        default=None,
        description="Corrected class name from the allowed class list when action is relabel.",
    )
    reason: str = Field(default="", description="Brief explanation in Chinese or English.")


class MissingBox(BaseModel):
    """MLLM 检出的漏标目标框（`box_2d` 为 `[ymin, xmin, ymax, xmax]`，归一化至 `0..1000`）。"""

    label: str = Field(description="Class name from the allowed class list.")
    box_2d: list[int] = Field(
        description="2D bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000."
    )
    reason: str = Field(default="", description="Why this object was missed.")


class MllmReviewResponse(BaseModel):
    """期望 MLLM 返回的结构化审核响应模式。"""

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
    status: str = "original"  # 状态取值：original | kept | relabeled | deleted | llm_added
    reason: str = ""
    original_label: str = ""


@dataclass
class ImageReviewRecord:
    """单张图片纠错前后的处理结果与统计记录。"""

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
    has_missing: bool = False
    missing_note: str = ""
    skipped_mllm: bool = False
    error: str = ""


@dataclass
class MllmReviewConfig:
    """MLLM 自动标注纠错清洗管线的运行配置。

    关键字段说明：
        `yolo_conf` / `yolo_iou` / `high_conf_skip`：置信度与 IoU 阈值，取值范围 `(0, 1]`；
        仅当已有候选框非空、所有框均具有不低于 `high_conf_skip` 的置信度且 `check_missing=False` 时跳过 MLLM。
        `max_crops_per_image`：每张图最多附带的局部放大裁剪数；`max_image_dim` 为送审图最长边像素限制。
        `max_workers`：并发审核线程池工作线程数（至少为 `1`）。
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
    model_name: str = "gemini-3.8-flash"
    api_key: str = ""
    base_url: str = ""

    allowed_classes: list[str] = field(default_factory=list)
    custom_guideline: str = ""
    high_conf_skip: float = 0.85
    check_missing: bool = True
    add_missing_boxes: bool = True
    include_crops: bool = True
    max_crops_per_image: int = 8
    max_image_dim: int = 1280
    max_workers: int = 2
    triage_subfolders: bool = True
    recursive: bool = False


@dataclass
class MllmReviewResult:
    """单次 MLLM 审核清洗任务完成或取消后的汇总结果。"""

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


class GeminiMllmClient:
    """基于官方 `google-genai` Interactions API 的 Google Gemini 多模态审核客户端。

    凭据与状态：
        初始化时优先使用传入的 `api_key`，为空时读取环境变量 `GEMINI_API_KEY`，均缺失则抛出 `ValueError`；
        内部持有无状态交互会话客户端（每次调用设置 `store=False`）。
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.8-flash") -> None:
        from google import genai

        resolved_key = api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError("未提供 Gemini API Key（可在界面填写或设置环境变量 GEMINI_API_KEY）。")
        self.client = genai.Client(api_key=resolved_key)
        self.model_name = model_name.strip() or "gemini-3.8-flash"

    def review_image(
        self,
        overview_jpeg: bytes,
        crops: list[tuple[int, str, bytes]],
        system_prompt: str,
        user_prompt: str,
    ) -> MllmReviewResponse:
        """发送编号总览图及候选框裁剪图至 Gemini 并解析结构化审核结果；响应文本为空时抛出 `RuntimeError`。"""
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

        interaction = self.client.interactions.create(
            model=self.model_name,
            system_instruction=system_prompt,
            input=contents,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": MllmReviewResponse.model_json_schema(),
            },
            store=False,
        )
        output_text = getattr(interaction, "output_text", None) or ""
        if not output_text:
            raise RuntimeError("Gemini 返回为空，未包含文本输出。")
        return parse_review_response_text(output_text)


class OpenAICompatMllmClient:
    """OpenAI 兼容多模态审核客户端，适配通义千问 Qwen-VL、硅基流动、本地 Ollama、vLLM 与 OpenAI。

    凭据与兼容性：
        初始化时依次回退 `api_key`、`OPENAI_API_KEY` 与 `"ollama"`（用于本地免鉴权服务），
        `base_url` 为空时默认使用 DashScope 兼容端点，`model_name` 为空抛出 `ValueError`。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float = 90.0,
    ) -> None:
        from openai import OpenAI

        resolved_key = api_key.strip() or os.environ.get("OPENAI_API_KEY", "").strip() or "ollama"
        resolved_url = base_url.strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=resolved_key, base_url=resolved_url, timeout=timeout)
        self.model_name = model_name.strip()
        if not self.model_name:
            raise ValueError("请指定 OpenAI 兼容接口的模型名称（如 qwen-vl-max-latest）。")

    def review_image(
        self,
        overview_jpeg: bytes,
        crops: list[tuple[int, str, bytes]],
        system_prompt: str,
        user_prompt: str,
    ) -> MllmReviewResponse:
        """发送多模态对话请求并解析 JSON 审核结果；若后端不支持 `json_object` 格式则自动回退重试。"""
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

        try:
            # 保留 type: ignore 指令供静态类型检查器识别兼容接口动态构造的列表消息
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception:
            # 部分本地 Ollama / vLLM 后端不支持 response_format={"type": "json_object"}，降级为纯文本约束重试
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.1,
            )

        content = resp.choices[0].message.content or ""
        return parse_review_response_text(content)


def build_mllm_client(config: MllmReviewConfig) -> MllmClient:
    """根据配置中的 `provider` 创建对应的 MLLM 审核客户端实例。"""
    if config.provider == MllmProvider.GEMINI:
        return GeminiMllmClient(api_key=config.api_key, model_name=config.model_name)
    return OpenAICompatMllmClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model_name,
    )


def _render_overview_and_crops(
    image_bgr: np.ndarray,
    boxes: list[CandidateBox],
    include_crops: bool,
    max_crops: int,
    high_conf_skip: float,
    max_dim: int,
) -> tuple[bytes, list[tuple[int, str, bytes]]]:
    """在图片上绘制带编号的候选框总览图，并为低置信度候选框提取外扩 20% 的局部放大裁剪图。"""
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
    if include_crops and boxes:
        # 优先按置信度从低到高排序，确保低置信度可疑框优先获得局部裁剪放大图
        prioritized = sorted(
            boxes,
            key=lambda b: b.confidence if b.confidence is not None else 0.5,
        )
        for box in prioritized:
            if len(crops) >= max_crops:
                break
            if box.confidence is not None and box.confidence >= high_conf_skip and len(boxes) > max_crops:
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

    return overview_jpeg, crops


def _build_prompts(
    boxes: list[CandidateBox],
    allowed_classes: list[str],
    config: MllmReviewConfig,
) -> tuple[str, str]:
    classes_str = ", ".join(allowed_classes) if allowed_classes else "(any relevant object)"
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
    )
    if config.custom_guideline.strip():
        system_prompt += f"\nDomain Annotation Guidelines:\n{config.custom_guideline.strip()}"

    box_lines = []
    for b in boxes:
        conf_part = f", conf={b.confidence:.2f}" if b.confidence is not None else ""
        box_lines.append(
            f"- box_id={b.box_id}, label='{b.label}'{conf_part}, xyxy=[{int(b.x1)}, {int(b.y1)}, {int(b.x2)}, {int(b.y2)}]"
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
        f"Candidate boxes:\n{boxes_summary}\n\n"
        f"{missing_instr}\n\n"
        "Return strict JSON with keys: box_reviews (list of {box_id, action, new_label, reason}), "
        "has_missing (bool), missing_note (str), added_boxes (list of {label, box_2d, reason})."
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


def apply_review_decisions(
    before_boxes: list[CandidateBox],
    response: MllmReviewResponse,
    allowed_classes: list[str],
    img_w: int,
    img_h: int,
    add_missing_boxes: bool = True,
) -> tuple[list[CandidateBox], int, int, int, int]:
    """将 MLLM 审核决策合并到候选框集合中，并过滤重复补充框。

    返回 `(after_boxes, kept, relabeled, deleted, added)`；
    未出现在 `box_reviews` 中的候选框按保留（`kept`）处理，
    新增补充框若与同类别现有框 IoU > 0.65 则视为重复并跳过。
    """
    decision_map = {d.box_id: d for d in response.box_reviews}
    after_boxes: list[CandidateBox] = []
    kept = relabeled = deleted = added = 0
    next_id = max((b.box_id for b in before_boxes), default=-1) + 1

    for box in before_boxes:
        decision = decision_map.get(box.box_id)
        if decision is None or decision.action == "keep":
            kept += 1
            after_boxes.append(
                CandidateBox(
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
            )
        elif decision.action == "relabel":
            resolved = _normalize_class_name(decision.new_label, allowed_classes)
            if resolved and resolved != box.label:
                relabeled += 1
                after_boxes.append(
                    CandidateBox(
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
                )
            else:
                kept += 1
                after_boxes.append(
                    CandidateBox(
                        box_id=box.box_id,
                        label=box.label,
                        x1=box.x1,
                        y1=box.y1,
                        x2=box.x2,
                        y2=box.y2,
                        confidence=box.confidence,
                        status="kept",
                        reason=decision.reason,
                        original_label=box.label,
                    )
                )
        elif decision.action == "delete":
            deleted += 1

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
                "flags": {"llm_added": b.status == "llm_added"},
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


def _find_images(images_dir: Path, recursive: bool) -> list[Path]:
    it = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _discover_yolo_candidates(
    images: list[Path],
    config: MllmReviewConfig,
    log_callback: Callable[[str], None] | None,
) -> tuple[dict[Path, list[CandidateBox]], list[str]]:
    from ultralytics import YOLO

    if not config.weights_path or not config.weights_path.is_file():
        raise FileNotFoundError(f"基准模型权重文件不存在: {config.weights_path}")

    if log_callback:
        log_callback(
            f"正在使用基准模型 {config.weights_path.name} 对 {len(images)} 张图片执行低阈值推理 (conf={config.yolo_conf})..."
        )
    model = YOLO(str(config.weights_path))
    model_names = [str(v) for _, v in sorted(model.names.items())] if isinstance(model.names, dict) else []

    candidates_map: dict[Path, list[CandidateBox]] = {}
    for img_path in images:
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


def _discover_json_candidates(
    images: list[Path],
    config: MllmReviewConfig,
) -> tuple[dict[Path, list[CandidateBox]], list[str]]:
    ann_dir = config.annotations_dir if config.annotations_dir else config.images_dir
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {ann_dir}")

    candidates_map: dict[Path, list[CandidateBox]] = {}
    discovered_classes: list[str] = []
    seen_cls: set[str] = set()

    for img_path in images:
        try:
            rel_parent = img_path.parent.relative_to(config.images_dir)
        except ValueError:
            rel_parent = Path(".")

        candidates = [
            ann_dir / rel_parent / f"{img_path.stem}.json",
            ann_dir / f"{img_path.stem}.json",
            img_path.with_suffix(".json"),
        ]
        json_file = next((c for c in candidates if c.is_file()), None)
        boxes = _load_existing_json_boxes(json_file) if json_file else []
        for b in boxes:
            if b.label not in seen_cls:
                seen_cls.add(b.label)
                discovered_classes.append(b.label)
        candidates_map[img_path] = boxes

    return candidates_map, discovered_classes


def run_mllm_review(
    config: MllmReviewConfig,
    client_override: MllmClient | None = None,
    progress_callback: Callable[[int, int, ImageReviewRecord], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    cancel_event: Event | None = None,
) -> MllmReviewResult:
    """执行基准候选框获取、MLLM 并发审核、结果分流与 X-AnyLabeling `.json` 导出。

    前置条件与失败方式：
        `images_dir` 不存在抛出 `FileNotFoundError`，无图片抛出 `ValueError`；
        在 `YOLO_PREDICT` 模式下需存在有效 `weights_path`，在 `EXISTING_JSON` 模式下需存在标注目录。
        单张图片审核异常不会中止整批流程，而是将该图连同原始候选框写入 `need_review` 并记录 `error`。

    并发、取消与输出产物：
        使用 `ThreadPoolExecutor(max_workers=max(1, config.max_workers))` 并发处理各图；
        若通过 `cancel_event` 请求停止，将停止提交新任务并尝试取消尚未开始的任务。
        会在 `config.output_dir` 下写出分流图片与 `.json`（视 `triage_subfolders` 决定是否建子目录）、
        `classes.txt`（当类别非空时）及 `review_report.json`。
    """
    if not config.images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {config.images_dir}")

    images = _find_images(config.images_dir, recursive=config.recursive)
    if not images:
        raise ValueError(f"未在目录中找到任何图片: {config.images_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.source_mode == ReviewSourceMode.YOLO_PREDICT:
        candidates_map, discovered_classes = _discover_yolo_candidates(images, config, log_callback)
    else:
        candidates_map, discovered_classes = _discover_json_candidates(images, config)

    allowed_classes = [c.strip() for c in config.allowed_classes if c.strip()] or discovered_classes
    client: MllmClient = client_override or build_mllm_client(config)

    if log_callback:
        log_callback(
            f"开始 MLLM 纠错审核（共 {len(images)} 张图，类别: {allowed_classes}，并发={config.max_workers}）..."
        )

    def _process_one(img_path: Path) -> ImageReviewRecord | None:
        if cancel_event and cancel_event.is_set():
            return None
        before_boxes = candidates_map.get(img_path, [])
        try:
            with Image.open(img_path) as pil_img:
                img_w, img_h = pil_img.size

            # 当已有框全部达到高置信度阈值且未开启漏标巡检时，直接跳过 MLLM 调用
            all_high_conf = bool(before_boxes) and all(
                (b.confidence is not None and b.confidence >= config.high_conf_skip)
                for b in before_boxes
            )
            if all_high_conf and not config.check_missing:
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
                        reason="高置信度跳过 MLLM",
                        original_label=b.label,
                    )
                    for b in before_boxes
                ]
                kept, relabeled, deleted, added = len(after_boxes), 0, 0, 0
                has_missing = False
                missing_note = ""
                skipped_mllm = True
            else:
                img_bgr = _read_image_unicode_safe(img_path)
                overview_jpeg, crops = _render_overview_and_crops(
                    img_bgr,
                    before_boxes,
                    include_crops=config.include_crops,
                    max_crops=config.max_crops_per_image,
                    high_conf_skip=config.high_conf_skip,
                    max_dim=config.max_image_dim,
                )
                system_prompt, user_prompt = _build_prompts(before_boxes, allowed_classes, config)
                response = client.review_image(overview_jpeg, crops, system_prompt, user_prompt)
                after_boxes, kept, relabeled, deleted, added = apply_review_decisions(
                    before_boxes,
                    response,
                    allowed_classes,
                    img_w=img_w,
                    img_h=img_h,
                    add_missing_boxes=config.add_missing_boxes,
                )
                has_missing = response.has_missing
                missing_note = response.missing_note
                skipped_mllm = False

            if has_missing or added > 0:
                bucket = "need_review"
            elif relabeled > 0 or deleted > 0:
                bucket = "llm_fixed"
            else:
                bucket = "auto_passed"

            target_dir = (config.output_dir / bucket) if config.triage_subfolders else config.output_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = target_dir / img_path.name
            if out_img_path.resolve() != img_path.resolve():
                shutil.copy2(img_path, out_img_path)
            out_json_path = out_img_path.with_suffix(".json")
            _write_xanylabeling_json(
                out_json_path,
                out_img_path.name,
                img_w=img_w,
                img_h=img_h,
                boxes=after_boxes,
                has_missing=has_missing,
                missing_note=missing_note,
            )

            return ImageReviewRecord(
                image_path=img_path,
                output_image_path=out_img_path,
                output_json_path=out_json_path,
                triage_bucket=bucket,
                before_boxes=before_boxes,
                after_boxes=after_boxes,
                kept_count=kept,
                relabeled_count=relabeled,
                deleted_count=deleted,
                added_count=added,
                has_missing=has_missing,
                missing_note=missing_note,
                skipped_mllm=skipped_mllm,
            )
        except Exception as exc:
            # 单图审核异常时保留图片与原始框并分流至 need_review，防止样本丢失
            target_dir = (
                (config.output_dir / "need_review") if config.triage_subfolders else config.output_dir
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            out_img_path = target_dir / img_path.name
            if out_img_path.resolve() != img_path.resolve():
                shutil.copy2(img_path, out_img_path)
            out_json_path = out_img_path.with_suffix(".json")
            try:
                with Image.open(img_path) as pil_img:
                    iw, ih = pil_img.size
            except Exception:
                iw, ih = 0, 0
            _write_xanylabeling_json(
                out_json_path,
                out_img_path.name,
                img_w=iw,
                img_h=ih,
                boxes=before_boxes,
                has_missing=True,
                missing_note=f"MLLM 审核异常: {exc}",
            )
            return ImageReviewRecord(
                image_path=img_path,
                output_image_path=out_img_path,
                output_json_path=out_json_path,
                triage_bucket="need_review",
                before_boxes=before_boxes,
                after_boxes=before_boxes,
                has_missing=True,
                missing_note=f"审核出错: {exc}",
                error=str(exc),
            )

    records: list[ImageReviewRecord] = []
    total = len(images)
    cancelled = False

    workers = max(1, int(config.max_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for img_path in images:
            if cancel_event and cancel_event.is_set():
                cancelled = True
                break
            futures[pool.submit(_process_one, img_path)] = img_path

        completed_count = 0
        for fut in as_completed(futures):
            if cancel_event and cancel_event.is_set():
                cancelled = True
                for pending in futures:
                    pending.cancel()
            if fut.cancelled():
                continue
            rec = fut.result()
            if rec is None:
                cancelled = True
                continue
            records.append(rec)
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total, rec)
            if log_callback:
                log_callback(
                    f"[{completed_count}/{total}] {rec.image_path.name} -> {rec.triage_bucket} "
                    f"(保留:{rec.kept_count}, 改类:{rec.relabeled_count}, 删误检:{rec.deleted_count}, 补漏:{rec.added_count})"
                )

    records.sort(key=lambda r: r.image_path.name)

    auto_passed = sum(1 for r in records if r.triage_bucket == "auto_passed")
    llm_fixed = sum(1 for r in records if r.triage_bucket == "llm_fixed")
    need_review = sum(1 for r in records if r.triage_bucket == "need_review")
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
        "classes": allowed_classes,
        "items": [
            {
                "image": r.image_path.name,
                "bucket": r.triage_bucket,
                "kept": r.kept_count,
                "relabeled": r.relabeled_count,
                "deleted": r.deleted_count,
                "added": r.added_count,
                "has_missing": r.has_missing,
                "missing_note": r.missing_note,
                "error": r.error,
                "output_json": str(r.output_json_path) if r.output_json_path else "",
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
        "llm_added": (255, 220, 0),
    }

    for b in boxes:
        color = status_colors.get(b.status, (0, 200, 0)) if use_status_colors else (200, 180, 0)
        p1 = (int(round(b.x1)), int(round(b.y1)))
        p2 = (int(round(b.x2)), int(round(b.y2)))
        cv2.rectangle(canvas, p1, p2, color, thickness)
        if b.status == "relabeled" and b.original_label:
            tag = f"{b.original_label}->{b.label}"
        elif b.status == "llm_added":
            tag = f"+{b.label} (LLM)"
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
    """渲染单张审核记录的“纠错前 vs 纠错后”左右拼接对比 JPEG 字节流。"""
    src_path = record.output_image_path if (record.output_image_path and record.output_image_path.is_file()) else record.image_path
    img_bgr = _read_image_unicode_safe(src_path)
    left = _draw_labeled_boxes(img_bgr, record.before_boxes, f"Before ({len(record.before_boxes)})", use_status_colors=False)
    right = _draw_labeled_boxes(img_bgr, record.after_boxes, f"After ({len(record.after_boxes)})", use_status_colors=True)
    combined = np.hstack([left, right])
    return _encode_jpeg_bytes(combined, max_dim=max_width, quality=90)
