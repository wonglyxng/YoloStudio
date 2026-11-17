"""MLLM 自动标注审核与纠错清洗管线单元测试。

职责：
    使用确定性桩客户端 `_FakeMllmClient` 离线验证 JSON 围栏解析、保留/改类/删误检/补漏与
    高重叠去重合并、三分类目录分流、对比图渲染及与 `dataset_converter` 的闭环衔接；
    并覆盖重复审核跨桶去重、异常与损坏标注兜底写报告、非法改类强制复核、
    递归同名输出消歧、空结果跳过调用以及 OpenAI 兼容客户端仅在 400 时降级等回归场景。

运行环境与协作边界：
    通过 `pytest` 在临时目录（`tmp_path`）中离线运行，不发起真实网络 API 调用
    （OpenAI 兼容客户端测试替换了底层 `chat.completions` 对象）；
    协作覆盖 `yolo_studio.core.mllm_reviewer` 与 `yolo_studio.core.dataset_converter`。
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from yolo_studio.core.dataset_converter import (
    AnnotationFormat,
    DatasetBuildConfig,
    build_yolo_dataset,
)
from yolo_studio.core.mllm_reviewer import (
    BoxDecision,
    CandidateBox,
    MissingBox,
    MllmProvider,
    MllmReviewConfig,
    MllmReviewResponse,
    ReviewSourceMode,
    apply_review_decisions,
    parse_review_response_text,
    render_review_comparison,
    run_mllm_review,
)


class _FakeMllmClient:
    """按调用顺序返回预设响应的确定性测试桩客户端（不访问外部网络）。"""

    def __init__(self, responses: list[MllmReviewResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[int, int]] = []  # 记录每次调用的 (总览图字节长度, 局部裁剪数量)

    def review_image(
        self,
        overview_jpeg: bytes,
        crops: list[tuple[int, str, bytes]],
        system_prompt: str,
        user_prompt: str,
    ) -> MllmReviewResponse:
        """记录输入载荷规模并按索引依次返回预设的 `MllmReviewResponse`。"""
        idx = len(self.calls)
        self.calls.append((len(overview_jpeg), len(crops)))
        return self._responses[idx]


def _write_img(path: Path, size: tuple[int, int] = (400, 200)) -> None:
    """在指定路径生成测试图片。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(60, 100, 140)).save(path)


def test_parse_review_response_with_markdown_fence() -> None:
    """验证带有 Markdown 代码围栏的 MLLM 输出可被正确清理并解析为结构化模型。"""
    raw = """```json
    {
      "box_reviews": [
        {"box_id": 0, "action": "relabel", "new_label": "helmet", "reason": "misclassified"}
      ],
      "has_missing": true,
      "missing_note": "one person missed",
      "added_boxes": [
        {"label": "person", "box_2d": [100, 200, 500, 600], "reason": "missed"}
      ]
    }
    ```"""
    parsed = parse_review_response_text(raw)
    assert len(parsed.box_reviews) == 1
    assert parsed.box_reviews[0].action == "relabel"
    assert parsed.has_missing is True
    assert parsed.added_boxes[0].box_2d == [100, 200, 500, 600]


def test_apply_review_decisions_and_deduplicate_added_box() -> None:
    """验证 keep/relabel/delete/add 决策合并，以及与既有框高度重叠的补充框会被去重跳过。"""
    before = [
        CandidateBox(box_id=0, label="person", x1=40, y1=20, x2=160, y2=180, confidence=0.80),
        CandidateBox(box_id=1, label="person", x1=200, y1=40, x2=280, y2=120, confidence=0.35),
        CandidateBox(box_id=2, label="helmet", x1=300, y1=10, x2=340, y2=40, confidence=0.15),
    ]
    response = MllmReviewResponse(
        box_reviews=[
            BoxDecision(box_id=0, action="keep", reason="ok"),
            BoxDecision(box_id=1, action="relabel", new_label="HELMET", reason="actually helmet"),
            BoxDecision(box_id=2, action="delete", reason="background noise"),
        ],
        has_missing=True,
        missing_note="added real missing box + duplicate overlapping existing person",
        added_boxes=[
            # 与 0 号框完全重合的重复补充框（400x200 图像上 x1=40,y1=20,x2=160,y2=180 对应 [100,100,900,400]）
            MissingBox(label="person", box_2d=[100, 100, 900, 400], reason="dup"),
            # 右侧新增的真实漏标目标
            MissingBox(label="person", box_2d=[200, 700, 800, 950], reason="missed person"),
        ],
    )
    after, kept, relabeled, deleted, added = apply_review_decisions(
        before,
        response,
        allowed_classes=["person", "helmet"],
        img_w=400,
        img_h=200,
        add_missing_boxes=True,
    )
    assert (kept, relabeled, deleted, added) == (1, 1, 1, 1)
    assert len(after) == 3
    assert after[1].label == "helmet"
    assert after[1].status == "relabeled"
    assert after[2].status == "llm_added"
    assert abs(after[2].x1 - 280.0) < 1e-3
    assert abs(after[2].y1 - 40.0) < 1e-3


def test_run_mllm_review_existing_json_triage_and_dataset_roundtrip(tmp_path: Path) -> None:
    """验证已有 `.json` 输入模式下的三分类分流、对比图渲染与数据集递归导入闭环。"""
    in_dir = tmp_path / "raw_frames"
    out_dir = tmp_path / "reviewed_frames"

    for idx in range(3):
        img_path = in_dir / f"img_{idx}.jpg"
        _write_img(img_path, (400, 200))
        payload = {
            "version": "2.4.4",
            "shapes": [
                {
                    "label": "person",
                    "score": 0.90 if idx == 0 else 0.40,
                    "points": [[40.0, 20.0], [160.0, 180.0]],
                    "shape_type": "rectangle",
                }
            ],
            "imagePath": img_path.name,
            "imageHeight": 200,
            "imageWidth": 400,
        }
        img_path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    # img_0：虽然置信度 0.90 >= 0.85，但默认开启 check_missing=True，仍会送审；无误且无漏标 -> auto_passed
    # img_1：存在类别修改 -> llm_fixed
    # img_2：存在漏标提示并补充新框 -> need_review
    fake_client = _FakeMllmClient(
        [
            MllmReviewResponse(
                box_reviews=[BoxDecision(box_id=0, action="keep")],
                has_missing=False,
            ),
            MllmReviewResponse(
                box_reviews=[BoxDecision(box_id=0, action="relabel", new_label="helmet", reason="fix")],
                has_missing=False,
            ),
            MllmReviewResponse(
                box_reviews=[BoxDecision(box_id=0, action="keep")],
                has_missing=True,
                missing_note="missed helmet on right",
                added_boxes=[MissingBox(label="helmet", box_2d=[200, 600, 500, 800])],
            ),
        ]
    )

    cfg = MllmReviewConfig(
        images_dir=in_dir,
        output_dir=out_dir,
        source_mode=ReviewSourceMode.EXISTING_JSON,
        provider=MllmProvider.GEMINI,
        allowed_classes=["person", "helmet"],
        max_workers=1,
        triage_subfolders=True,
    )
    res = run_mllm_review(cfg, client_override=fake_client)

    assert res.total_images == 3
    assert res.auto_passed_count == 1
    assert res.llm_fixed_count == 1
    assert res.need_review_count == 1
    assert (out_dir / "auto_passed" / "img_0.json").is_file()
    assert (out_dir / "llm_fixed" / "img_1.json").is_file()
    assert (out_dir / "need_review" / "img_2.json").is_file()
    assert res.report_path and res.report_path.is_file()

    # 验证纠错前后左右对比图能够正常编码为 JPEG
    preview_bytes = render_review_comparison(res.records[2])
    assert preview_bytes[:2] == b"\xff\xd8"  # JPEG 文件头魔数

    # 验证清洗分流后的子目录可通过 recursive=True 无缝导入 DatasetConverter
    ds_res = build_yolo_dataset(
        DatasetBuildConfig(
            images_dir=out_dir,
            output_dir=tmp_path / "yolo_ds",
            annotation_format=AnnotationFormat.XANYLABELING_JSON,
            classes_file=res.classes_txt_path,
            train_ratio=0.67,
            val_ratio=0.33,
            recursive=True,
        )
    )
    assert ds_res.train_count + ds_res.val_count == 3
    assert ds_res.total_boxes == 4  # 总框数为 1 + 1 + 2 = 4 个


class _RaisingClient:
    """每次调用都抛出异常的测试桩客户端。"""

    def review_image(self, *_args, **_kwargs) -> MllmReviewResponse:
        raise RuntimeError("模拟接口故障")


def _person_json(img_path: Path, score: float = 0.5) -> None:
    payload = {
        "shapes": [
            {
                "label": "person",
                "score": score,
                "points": [[40.0, 20.0], [160.0, 180.0]],
                "shape_type": "rectangle",
            }
        ],
        "imageWidth": 400,
        "imageHeight": 200,
    }
    img_path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")


def _json_cfg(in_dir: Path, out_dir: Path, **kwargs) -> MllmReviewConfig:
    return MllmReviewConfig(
        images_dir=in_dir,
        output_dir=out_dir,
        source_mode=ReviewSourceMode.EXISTING_JSON,
        allowed_classes=["person", "helmet"],
        max_workers=1,
        **kwargs,
    )


def test_rerun_moves_image_between_buckets_without_duplicates(tmp_path: Path) -> None:
    """验证重复审核到同一输出目录时，同名图片只保留在最新分流桶中。"""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img = in_dir / "a.jpg"
    _write_img(img)
    _person_json(img)

    first = MllmReviewResponse(box_reviews=[BoxDecision(box_id=0, action="keep")], has_missing=True)
    run_mllm_review(_json_cfg(in_dir, out_dir), client_override=_FakeMllmClient([first]))
    assert (out_dir / "need_review" / "a.jpg").is_file()

    second = MllmReviewResponse(box_reviews=[BoxDecision(box_id=0, action="keep")], has_missing=False)
    run_mllm_review(_json_cfg(in_dir, out_dir), client_override=_FakeMllmClient([second]))
    assert (out_dir / "auto_passed" / "a.jpg").is_file()
    assert not (out_dir / "need_review" / "a.jpg").exists()
    assert not (out_dir / "need_review" / "a.json").exists()


def test_errors_and_corrupt_json_still_write_report(tmp_path: Path) -> None:
    """验证接口异常与已有标注损坏都会分流到 need_review 且报告照常写出。"""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    good = in_dir / "good.jpg"
    _write_img(good)
    _person_json(good)
    broken = in_dir / "broken.jpg"
    _write_img(broken)
    broken.with_suffix(".json").write_text("{oops", encoding="utf-8")

    res = run_mllm_review(_json_cfg(in_dir, out_dir), client_override=_RaisingClient())
    assert res.total_images == 2
    assert res.error_count == 2
    assert res.need_review_count == 2
    assert res.report_path and res.report_path.is_file()
    report = json.loads(res.report_path.read_text(encoding="utf-8"))
    broken_item = next(i for i in report["items"] if i["image"] == "broken.jpg")
    assert "读取已有标注失败" in broken_item["error"]


def test_invalid_relabel_goes_to_need_review() -> None:
    """验证模型要求改为非允许类别时框被保留为 relabel_invalid，供调用方分流人工复核。"""
    before = [CandidateBox(box_id=0, label="person", x1=10, y1=10, x2=100, y2=100, confidence=0.5)]
    response = MllmReviewResponse(
        box_reviews=[BoxDecision(box_id=0, action="relabel", new_label="dog", reason="looks like dog")]
    )
    after, kept, relabeled, _deleted, _added = apply_review_decisions(
        before, response, ["person", "helmet"], img_w=400, img_h=200
    )
    assert (kept, relabeled) == (1, 0)
    assert after[0].status == "relabel_invalid"
    assert after[0].label == "person"


def test_recursive_same_name_outputs_do_not_overwrite(tmp_path: Path) -> None:
    """验证递归模式下不同子目录的同名图片输出到同一分流桶时不会互相覆盖。"""
    in_dir = tmp_path / "in"
    for sub in ("cam1", "cam2"):
        img = in_dir / sub / "frame.jpg"
        _write_img(img)
        _person_json(img)
    responses = [
        MllmReviewResponse(box_reviews=[BoxDecision(box_id=0, action="keep")]) for _ in range(2)
    ]
    out_dir = tmp_path / "out"
    res = run_mllm_review(
        _json_cfg(in_dir, out_dir, recursive=True), client_override=_FakeMllmClient(responses)
    )
    outputs = {r.output_image_path.name for r in res.records if r.output_image_path}
    assert outputs == {"cam1__frame__jpg.jpg", "cam2__frame__jpg.jpg"}
    assert len(list((out_dir / "auto_passed").glob("*.json"))) == 2


def test_empty_result_without_missing_check_skips_mllm_and_json(tmp_path: Path) -> None:
    """验证无候选框且未开启漏标检查时跳过 MLLM 调用，且不写出未经确认的空 `.json`。"""
    in_dir = tmp_path / "in"
    _write_img(in_dir / "empty.jpg")
    client = _FakeMllmClient([])
    res = run_mllm_review(
        _json_cfg(in_dir, tmp_path / "out", check_missing=False), client_override=client
    )
    assert client.calls == []
    rec = res.records[0]
    assert rec.skipped_mllm and rec.output_json_path is None
    assert rec.output_image_path and rec.output_image_path.is_file()


def test_openai_client_falls_back_only_on_bad_request() -> None:
    """验证 OpenAI 兼容客户端仅在 400 错误时降级为纯文本模式并记住结论，其他错误直接抛出。"""
    import httpx
    import openai

    from yolo_studio.core.mllm_reviewer import OpenAICompatMllmClient

    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    class _Msg:
        content = ok_json

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def __init__(self, error: Exception) -> None:
            self.error = error
            self.calls: list[bool] = []

        def create(self, **kwargs):
            self.calls.append("response_format" in kwargs)
            if "response_format" in kwargs:
                raise self.error
            return _Resp()

    client = OpenAICompatMllmClient(api_key="k", base_url="http://localhost/v1", model_name="m")
    bad = openai.BadRequestError("unsupported", response=httpx.Response(400, request=request), body=None)
    completions = _Completions(bad)
    client.client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()
    client.review_image(b"x", [], "s", "u")
    client.review_image(b"x", [], "s", "u")
    assert completions.calls == [True, False, False]

    auth_client = OpenAICompatMllmClient(api_key="k", base_url="http://localhost/v1", model_name="m")
    auth_err = openai.AuthenticationError("denied", response=httpx.Response(401, request=request), body=None)
    auth_completions = _Completions(auth_err)
    auth_client.client = type("C", (), {"chat": type("Chat", (), {"completions": auth_completions})()})()
    try:
        auth_client.review_image(b"x", [], "s", "u")
    except openai.AuthenticationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("鉴权错误不应被降级重试吞掉")
    assert auth_completions.calls == [True]
