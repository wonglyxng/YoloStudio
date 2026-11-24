"""LLM 智能标注与质检精修管线单元测试。

职责：
    使用确定性桩客户端 `_FakeMllmClient` 离线验证 JSON 围栏解析、保留/改类/删误检/补漏与
    高重叠去重合并、三分类目录分流、对比图渲染及与 `dataset_converter` 的闭环衔接；
    并覆盖重复标注跨桶去重、异常与损坏标注兜底写报告、非法改类强制复核、
    递归同名输出消歧、空结果跳过调用、OpenAI 兼容参数精确降级以及错误摘要去重等回归场景。

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
    """验证带有 Markdown 代码围栏的 LLM 输出可被正确清理并解析为结构化模型。"""
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
    out_dir = tmp_path / "smart_labeled"

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

    # 验证处理前后左右对比图能够正常编码为 JPEG
    preview_bytes = render_review_comparison(res.records[2])
    assert preview_bytes[:2] == b"\xff\xd8"  # JPEG 文件头魔数

    # 验证智能标注分流后的子目录可通过 recursive=True 无缝导入 DatasetConverter
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
    """验证无候选框且未开启漏标检查时跳过 LLM 调用，且不写出未经确认的空 `.json`。"""
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


def test_box_refine_flag_loose_routes_to_need_review(tmp_path: Path) -> None:
    """验证 `FLAG_LOOSE` 模式仅标记过松框为 `refine_suspect` 并分流 `need_review`，不改动原坐标。"""
    from yolo_studio.core.mllm_reviewer import BoxRefineMode

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img = in_dir / "loose.jpg"
    _write_img(img, (400, 200))
    _person_json(img)

    resp = MllmReviewResponse(
        box_reviews=[BoxDecision(box_id=0, action="keep", box_quality="loose", reason="too much background")]
    )
    res = run_mllm_review(
        _json_cfg(in_dir, out_dir, box_refine_mode=BoxRefineMode.FLAG_LOOSE),
        client_override=_FakeMllmClient([resp]),
    )
    assert res.need_review_count == 1
    rec = res.records[0]
    assert rec.loose_flagged_count == 1
    assert rec.refined_count == 0
    assert rec.after_boxes[0].status == "refine_suspect"
    assert (rec.after_boxes[0].x1, rec.after_boxes[0].y1) == (40.0, 20.0)


def test_box_refine_mllm_crop_maps_coords_and_enforces_safety_gate(tmp_path: Path) -> None:
    """验证 `MLLM_CROP` 模式将局部裁剪坐标精确映射回原图，并对超安全门禁的极端缩框兜底保护。"""
    from yolo_studio.core.mllm_reviewer import BoxRefineMode

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"

    # img1：合理收紧 -> 通过门禁并进入 llm_fixed，记录 original_xyxy 供虚线对比图渲染
    img1 = in_dir / "a_tighten.jpg"
    _write_img(img1, (400, 200))
    _person_json(img1)  # 原框 [40, 20, 160, 180] (宽120, 高160)；外扩 20% 裁剪区域约为 (16, 0, 184, 200)

    # img2：极端缩成极小点 -> 触发安全门禁保留原框并分流 need_review
    img2 = in_dir / "b_overshrink.jpg"
    _write_img(img2, (400, 200))
    _person_json(img2)

    resp_good = MllmReviewResponse(
        box_reviews=[
            BoxDecision(
                box_id=0,
                action="keep",
                box_quality="loose",
                # 在局部裁剪图内合理向内收紧约 15%
                refined_box_2d=[200, 220, 820, 780],
                reason="tightened to person body",
            )
        ]
    )
    resp_bad = MllmReviewResponse(
        box_reviews=[
            BoxDecision(
                box_id=0,
                action="keep",
                box_quality="loose",
                # 极端缩小到面积占比 < 5%，应被安全门禁拦截
                refined_box_2d=[450, 450, 520, 520],
                reason="bad tiny box",
            )
        ]
    )

    fake_client = _FakeMllmClient([resp_good, resp_bad])
    res = run_mllm_review(
        _json_cfg(in_dir, out_dir, include_crops=False, box_refine_mode=BoxRefineMode.MLLM_CROP),
        client_override=fake_client,
    )
    # 即使 include_crops=False，MLLM_CROP 模式仍强制生成局部裁剪图
    assert all(crop_cnt == 1 for _, crop_cnt in fake_client.calls)
    assert res.llm_fixed_count == 1
    assert res.need_review_count == 1

    rec_good = next(r for r in res.records if r.image_path.name == "a_tighten.jpg")
    assert rec_good.refined_count == 1
    b_good = rec_good.after_boxes[0]
    assert b_good.refined is True
    assert b_good.status == "refined"
    assert b_good.original_xyxy == (40.0, 20.0, 160.0, 180.0)
    assert b_good.x1 > 40.0 and b_good.x2 < 160.0
    # 对比图（含虚线原框与实线紧贴框）正常编码
    assert render_review_comparison(rec_good)[:2] == b"\xff\xd8"

    rec_bad = next(r for r in res.records if r.image_path.name == "b_overshrink.jpg")
    assert rec_bad.refined_count == 0
    assert rec_bad.loose_flagged_count == 1
    assert rec_bad.after_boxes[0].status == "refine_suspect"
    assert (rec_bad.after_boxes[0].x1, rec_bad.after_boxes[0].x2) == (40.0, 160.0)


def test_box_refine_sam_local_and_load_history_report_roundtrip(tmp_path: Path) -> None:
    """验证 `SAM_LOCAL` 离线桩收紧（仅过松框 vs 全部保留框）以及历史审核报告 `load_review_report` 完整往返。"""
    from yolo_studio.core.mllm_reviewer import BoxRefineMode, load_review_report

    class _FakeSamRefiner:
        def __init__(self) -> None:
            self.called_ids: list[list[int]] = []

        def refine_boxes(self, image_bgr, boxes: list[CandidateBox]):
            self.called_ids.append([b.box_id for b in boxes])
            return {b.box_id: (b.x1 + 12.0, b.y1 + 10.0, b.x2 - 12.0, b.y2 - 10.0) for b in boxes}

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img = in_dir / "sam_case.jpg"
    _write_img(img, (400, 200))
    _person_json(img, score=0.95)

    # 1) sam_refine_all_kept=False 且 LLM 判定 tight -> 不触发 SAM，直接 auto_passed
    refiner1 = _FakeSamRefiner()
    res1 = run_mllm_review(
        _json_cfg(in_dir, out_dir, box_refine_mode=BoxRefineMode.SAM_LOCAL, sam_refine_all_kept=False),
        client_override=_FakeMllmClient(
            [MllmReviewResponse(box_reviews=[BoxDecision(box_id=0, action="keep", box_quality="tight")])]
        ),
        refiner_override=refiner1,
    )
    assert refiner1.called_ids == []
    assert res1.auto_passed_count == 1

    # 2) sam_refine_all_kept=True 且 check_missing=False + 高置信度跳过 LLM -> 仍自动触发 SAM 收紧并分流 llm_fixed
    refiner2 = _FakeSamRefiner()
    res2 = run_mllm_review(
        _json_cfg(
            in_dir,
            out_dir,
            check_missing=False,
            box_refine_mode=BoxRefineMode.SAM_LOCAL,
            sam_refine_all_kept=True,
        ),
        client_override=_FakeMllmClient([]),
        refiner_override=refiner2,
    )
    assert refiner2.called_ids == [[0]]
    assert res2.llm_fixed_count == 1
    assert res2.records[0].refined_count == 1
    assert res2.records[0].after_boxes[0].x1 == 52.0

    # 3) 从 review_report.json 加载历史报告，精修计数与虚线原框信息完整保留
    loaded = load_review_report(out_dir)
    assert loaded.total_images == 1
    assert loaded.llm_fixed_count == 1
    assert loaded.records[0].refined_count == 1
    assert loaded.records[0].after_boxes[0].refined is True
    assert loaded.records[0].after_boxes[0].original_xyxy == (40.0, 20.0, 160.0, 180.0)


def test_mllm_connection_check_handles_missing_credentials(monkeypatch) -> None:
    """验证 `test_mllm_connection` 缺失 Key 或模型名时安全返回 `(False, 中文诊断)` 而不抛出异常。"""
    from yolo_studio.core.mllm_reviewer import test_mllm_connection

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ok_g, msg_g = test_mllm_connection(
        MllmReviewConfig(images_dir=Path("."), output_dir=Path("."), provider=MllmProvider.GEMINI, api_key="")
    )
    assert ok_g is False
    assert "GEMINI_API_KEY" in msg_g

    ok_o, msg_o = test_mllm_connection(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.OPENAI_COMPAT,
            model_name="",
        )
    )
    assert ok_o is False
    assert "模型名称" in msg_o


def test_max_images_trial_limit_count_and_delete_reason_summary(tmp_path: Path) -> None:
    """验证 `max_images` 小批量试跑上限、`count_review_images` 以及删除理由写入报告与详情摘要。"""
    from yolo_studio.core.mllm_reviewer import (
        count_review_images,
        format_record_detail_summary,
        load_review_report,
    )

    in_dir = tmp_path / "imgs"
    out_dir = tmp_path / "out"
    for name in ("01.jpg", "02.jpg", "03.jpg", "04.jpg"):
        img = in_dir / name
        _write_img(img, (400, 200))
        payload = {
            "shapes": [
                {
                    "label": "hero",
                    "score": 0.4,
                    "points": [[20.0, 20.0], [100.0, 100.0]],
                    "shape_type": "rectangle",
                }
            ],
            "imageWidth": 400,
            "imageHeight": 200,
        }
        img.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    assert count_review_images(in_dir) == 4
    assert count_review_images(tmp_path / "not_exist") == 0

    client = _FakeMllmClient(
        [
            MllmReviewResponse(
                box_reviews=[BoxDecision(box_id=0, action="delete", reason="地面阴影误检")],
                has_missing=False,
            ),
            MllmReviewResponse(
                box_reviews=[BoxDecision(box_id=0, action="keep", reason="目标清晰")],
                has_missing=False,
            ),
        ]
    )
    logs: list[str] = []
    res = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.EXISTING_JSON,
            allowed_classes=["hero"],
            max_workers=1,
            max_images=2,
            check_missing=True,
        ),
        client_override=client,
        log_callback=logs.append,
    )

    assert res.total_images == 2
    assert any("小批量试跑上限" in line for line in logs)
    del_rec = res.records[0]
    assert del_rec.deleted_count == 1
    assert del_rec.before_boxes[0].status == "deleted"
    assert del_rec.before_boxes[0].reason == "地面阴影误检"
    summary = format_record_detail_summary(del_rec)
    assert "删误检" in summary and "地面阴影误检" in summary

    # 历史报告往返加载后删框理由依然保留
    loaded = load_review_report(out_dir)
    assert loaded.total_images == 2
    assert "地面阴影误检" in format_record_detail_summary(loaded.records[0])


def test_collect_review_images_skips_output_inside_recursive_input(tmp_path: Path) -> None:
    """验证递归扫描时跳过位于输入目录内部的输出目录，而输出与输入相同或输入位于输出内部时不排除。"""
    from yolo_studio.core.mllm_reviewer import collect_review_images, count_review_images

    root = tmp_path / "frames"
    _write_img(root / "video-001" / "a.jpg", (64, 48))
    _write_img(root / "video-002" / "b.jpg", (64, 48))
    _write_img(root / "reviewed" / "auto_passed" / "a.jpg", (64, 48))

    out_inside = root / "reviewed"
    names = {p.relative_to(root).as_posix() for p in collect_review_images(root, True, out_inside)}
    assert names == {"video-001/a.jpg", "video-002/b.jpg"}
    assert count_review_images(root, recursive=True, output_dir=out_inside) == 2
    assert count_review_images(root, recursive=True) == 3
    assert count_review_images(root, recursive=False, output_dir=out_inside) == 0
    # 输出与输入相同、或复审 out/need_review 输出回 out 时保持原行为
    assert len(collect_review_images(root, True, root)) == 3
    assert len(collect_review_images(out_inside, True, root)) == 1


def test_cli_review_default_output_and_source_mode_hints() -> None:
    """验证 CLI `review` 默认输出路径已统一为 `data/smart_labeled`，且模式与收框提示词均不含混用的 `MLLM` 字样。"""
    from yolo_studio.cli import build_parser
    from yolo_studio.core.mllm_reviewer import (
        BOX_REFINE_MODE_HINTS,
        BOX_REFINE_MODE_ITEMS,
        SOURCE_MODE_HINTS,
        SOURCE_MODE_ITEMS,
    )

    parser = build_parser()
    args = parser.parse_args(["review", "-i", "data/raw_frames"])
    assert args.output == "data/smart_labeled"

    zs_args = parser.parse_args(
        [
            "review",
            "-i",
            "data/raw_frames",
            "--source-mode",
            "llm_zero_shot",
            "--classes",
            "person,helmet",
            "--box-refine",
            "sam_local",
        ]
    )
    assert zs_args.source_mode == "llm_zero_shot"
    assert zs_args.box_refine == "sam_local"
    assert zs_args.classes == "person,helmet"

    sam3_args = parser.parse_args(
        [
            "review",
            "-i",
            "data/raw_frames",
            "--source-mode",
            "sam3_text",
            "-w",
            "sam3.pt",
            "--classes",
            "安全帽=safety helmet,person",
            "--sam3-llm-review",
        ]
    )
    assert sam3_args.source_mode == "sam3_text"
    assert sam3_args.weights == "sam3.pt"
    assert sam3_args.sam3_llm_review is True
    assert sam3_args.classes == "安全帽=safety helmet,person"

    for _, label in SOURCE_MODE_ITEMS:
        assert "MLLM" not in label
    for hint in SOURCE_MODE_HINTS.values():
        assert "MLLM" not in hint
    for _, label in BOX_REFINE_MODE_ITEMS:
        assert "MLLM" not in label
    for hint in BOX_REFINE_MODE_HINTS.values():
        assert "MLLM" not in hint


def test_zero_shot_mode_requires_classes_and_handles_objects_and_negatives(tmp_path: Path) -> None:
    """验证 `LLM_ZERO_SHOT` 模式必填类别校验、零样本全图检出/去重与确认无目标负样本 `.json` 写出。"""
    import pytest

    from yolo_studio.core.mllm_reviewer import BoxRefineMode, format_record_detail_summary

    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    _write_img(in_dir / "01_has_obj.jpg", (400, 200))
    _write_img(in_dir / "02_empty_bg.jpg", (400, 200))

    # 1) 缺失 allowed_classes 直接抛出 ValueError
    with pytest.raises(ValueError, match="目标类别清单"):
        run_mllm_review(
            MllmReviewConfig(
                images_dir=in_dir,
                output_dir=out_dir,
                source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
                allowed_classes=[],
                max_workers=1,
            ),
            client_override=_FakeMllmClient([]),
        )

    # 2) 零样本正常识别：01 检出有效目标 + 过滤白名单外类别 + 过滤重复重叠框 -> need_review；
    #    02 确认无目标背景图 -> auto_passed 且写出空 shapes=[] .json；
    #    且传入不适用的 MLLM_CROP 自动降级并记录提示
    class _PromptCaptureClient:
        def __init__(self, responses: list[MllmReviewResponse]) -> None:
            self._responses = list(responses)
            self.calls: list[tuple[int, str, str]] = []

        def review_image(
            self,
            overview_jpeg: bytes,
            crops: list[tuple[int, str, bytes]],
            system_prompt: str,
            user_prompt: str,
        ) -> MllmReviewResponse:
            idx = len(self.calls)
            self.calls.append((len(crops), system_prompt, user_prompt))
            return self._responses[idx]

    client = _PromptCaptureClient(
        [
            MllmReviewResponse(
                added_boxes=[
                    MissingBox(label="PERSON", box_2d=[100, 100, 900, 400], reason="前景人物"),
                    # 同类完全重叠重复框，应被去重
                    MissingBox(label="person", box_2d=[105, 105, 895, 395], reason="重复框"),
                    # 不在白名单内的非法类别，应被过滤
                    MissingBox(label="cat", box_2d=[100, 500, 500, 700], reason="无关类别"),
                ],
                has_missing=False,
            ),
            MllmReviewResponse(added_boxes=[], has_missing=False),
        ]
    )
    logs: list[str] = []
    res = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["person", "helmet"],
            check_missing=False,  # 零样本模式下应忽略该开关，始终执行全图识别并写出负样本 .json
            add_missing_boxes=False,
            box_refine_mode=BoxRefineMode.MLLM_CROP,
            max_workers=1,
        ),
        client_override=client,
        log_callback=logs.append,
    )

    assert len(client.calls) == 2
    assert all(crop_cnt == 0 for crop_cnt, _, _ in client.calls)
    assert all("from scratch" in sys_p for _, sys_p, _ in client.calls)
    assert any("已自动忽略边框修正模式" in line for line in logs)

    assert res.total_images == 2
    assert res.need_review_count == 1
    assert res.auto_passed_count == 1

    rec_obj = next(r for r in res.records if r.image_path.name == "01_has_obj.jpg")
    assert rec_obj.triage_bucket == "need_review"
    assert rec_obj.added_count == 1
    assert len(rec_obj.after_boxes) == 1
    assert rec_obj.after_boxes[0].label == "person"
    assert rec_obj.after_boxes[0].status == "llm_added"
    assert "新增标注[person]" in format_record_detail_summary(rec_obj)

    rec_bg = next(r for r in res.records if r.image_path.name == "02_empty_bg.jpg")
    assert rec_bg.triage_bucket == "auto_passed"
    assert rec_bg.output_json_path is not None and rec_bg.output_json_path.is_file()
    bg_payload = json.loads(rec_bg.output_json_path.read_text(encoding="utf-8"))
    assert bg_payload["shapes"] == []

    assert res.report_path is not None
    report = json.loads(res.report_path.read_text(encoding="utf-8"))
    assert report["source_mode"] == "llm_zero_shot"
    assert report["box_refine_mode"] == "off"


def test_zero_shot_mode_with_sam_local_refinement_and_safety_gate(tmp_path: Path) -> None:
    """验证 `LLM_ZERO_SHOT` + `SAM_LOCAL` 联动对 LLM 粗框执行像素收紧，并保留安全门禁兜底保护。"""
    from yolo_studio.core.mllm_reviewer import BoxRefineMode, format_record_detail_summary

    class _MixedSamRefiner:
        def __init__(self) -> None:
            self.called_count = 0

        def refine_boxes(self, image_bgr, boxes: list[CandidateBox]):
            self.called_count += 1
            out: dict[int, tuple[float, float, float, float] | None] = {}
            for b in boxes:
                if b.label == "person":
                    # 合理向内收紧，应通过安全门禁
                    out[b.box_id] = (b.x1 + 10.0, b.y1 + 8.0, b.x2 - 10.0, b.y2 - 8.0)
                else:
                    # 极端缩成小噪点，应触发安全门禁保留原框并标记 refine_suspect
                    out[b.box_id] = (b.x1 + 1.0, b.y1 + 1.0, b.x1 + 4.0, b.y1 + 4.0)
            return out

    in_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    _write_img(in_dir / "scene.jpg", (400, 200))

    client = _FakeMllmClient(
        [
            MllmReviewResponse(
                added_boxes=[
                    # 原图坐标 (40, 20, 160, 180)
                    MissingBox(label="person", box_2d=[100, 100, 900, 400], reason="主体人物"),
                    # 原图坐标 (240, 40, 360, 160)
                    MissingBox(label="helmet", box_2d=[200, 600, 800, 900], reason="安全帽"),
                ],
                has_missing=False,
            )
        ]
    )
    refiner = _MixedSamRefiner()
    res = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["person", "helmet"],
            box_refine_mode=BoxRefineMode.SAM_LOCAL,
            sam_refine_all_kept=False,  # 零样本模式下应自动对全部检出框启用 SAM 收边
            max_workers=1,
        ),
        client_override=client,
        refiner_override=refiner,
    )

    assert refiner.called_count == 1
    assert res.need_review_count == 1
    rec = res.records[0]
    assert rec.added_count == 2
    assert rec.refined_count == 1
    assert rec.loose_flagged_count == 1

    person_box = next(b for b in rec.after_boxes if b.label == "person")
    assert person_box.status == "llm_added"
    assert person_box.refined is True
    assert person_box.original_xyxy == (40.0, 20.0, 160.0, 180.0)
    assert (person_box.x1, person_box.y1, person_box.x2, person_box.y2) == (50.0, 28.0, 150.0, 172.0)

    helmet_box = next(b for b in rec.after_boxes if b.label == "helmet")
    assert helmet_box.status == "refine_suspect"
    assert helmet_box.refined is False
    assert (helmet_box.x1, helmet_box.y1, helmet_box.x2, helmet_box.y2) == (240.0, 40.0, 360.0, 160.0)

    assert rec.output_json_path is not None and rec.output_json_path.is_file()
    saved_json = json.loads(rec.output_json_path.read_text(encoding="utf-8"))
    person_shape = next(s for s in saved_json["shapes"] if s["label"] == "person")
    assert person_shape["flags"] == {"llm_added": True, "box_refined": True}

    summary = format_record_detail_summary(rec)
    assert "新增标注+收框[person]" in summary
    assert "过松/超门禁待修" in summary
    assert render_review_comparison(rec)[:2] == b"\xff\xd8"


def test_sam3_text_pure_local_and_with_llm_review(tmp_path: Path, monkeypatch) -> None:
    """验证 `SAM3_TEXT` 提示词映射、纯本地零 API 费用分流，以及叠加 LLM 二次质检的完整闭环。"""
    import pytest

    from yolo_studio.core.mllm_reviewer import (
        BoxRefineMode,
        format_record_detail_summary,
        parse_sam3_class_prompts,
    )

    # 1) 类别与英文提示词映射解析（支持半角/全角等号与去重）
    cls_names, prompts = parse_sam3_class_prompts(
        [" 人物 = person ", "安全帽＝safety helmet", "car", "人物=duplicate", " "]
    )
    assert cls_names == ["人物", "安全帽", "car"]
    assert prompts == ["person", "safety helmet", "car"]

    in_dir = tmp_path / "raw"
    out_local = tmp_path / "out_local"
    _write_img(in_dir / "01_high.jpg", (400, 200))
    _write_img(in_dir / "02_low.jpg", (400, 200))
    _write_img(in_dir / "03_empty.jpg", (400, 200))
    _write_img(in_dir / "04_err.jpg", (400, 200))

    # 2) 缺失类别或缺失本地 sam3.pt 明确报错
    with pytest.raises(ValueError, match="目标类别清单"):
        run_mllm_review(
            MllmReviewConfig(
                images_dir=in_dir,
                output_dir=out_local,
                source_mode=ReviewSourceMode.SAM3_TEXT,
                allowed_classes=[],
            )
        )
    with pytest.raises(FileNotFoundError, match="SAM 3 权重文件不存在"):
        run_mllm_review(
            MllmReviewConfig(
                images_dir=in_dir,
                output_dir=out_local,
                source_mode=ReviewSourceMode.SAM3_TEXT,
                weights_path=tmp_path / "missing_sam3.pt",
                allowed_classes=["person"],
            )
        )

    class _FakeSam3Detector:
        def __init__(self) -> None:
            self.seen_prompts: list[tuple[list[str], list[str]]] = []

        def detect_image(
            self,
            image_path: Path,
            class_names: list[str],
            text_prompts: list[str],
        ) -> list[CandidateBox]:
            self.seen_prompts.append((list(class_names), list(text_prompts)))
            if image_path.name == "01_high.jpg":
                return [
                    CandidateBox(
                        box_id=0,
                        label=class_names[0],
                        x1=40.0,
                        y1=20.0,
                        x2=160.0,
                        y2=180.0,
                        confidence=0.92,
                        original_label=class_names[0],
                    )
                ]
            if image_path.name == "02_low.jpg":
                return [
                    CandidateBox(
                        box_id=0,
                        label=class_names[1],
                        x1=200.0,
                        y1=30.0,
                        x2=320.0,
                        y2=150.0,
                        confidence=0.42,
                        original_label=class_names[1],
                    )
                ]
            if image_path.name == "04_err.jpg":
                raise RuntimeError("模拟单图解码失败")
            return []

    # 3) 纯本地模式：清空环境变量 Key 且不传 client_override，仍可直接成功运行
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    detector = _FakeSam3Detector()
    res_local = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_local,
            source_mode=ReviewSourceMode.SAM3_TEXT,
            allowed_classes=["人物=person", "安全帽=safety helmet"],
            sam3_enable_llm_review=False,
            high_conf_skip=0.85,
            box_refine_mode=BoxRefineMode.SAM_LOCAL,  # 纯本地模式应自动忽略二次收框
            max_workers=1,
        ),
        sam3_detector_override=detector,
    )

    assert detector.seen_prompts[0] == (["人物", "安全帽"], ["person", "safety helmet"])
    assert res_local.classes == ["人物", "安全帽"]
    assert res_local.auto_passed_count == 1
    assert res_local.need_review_count == 3
    assert res_local.error_count == 1
    assert res_local.classes_txt_path is not None
    assert res_local.classes_txt_path.read_text(encoding="utf-8").splitlines() == ["人物", "安全帽"]

    rec_high = next(r for r in res_local.records if r.image_path.name == "01_high.jpg")
    assert rec_high.triage_bucket == "auto_passed"
    assert rec_high.added_count == 1
    assert rec_high.before_boxes == []
    assert "SAM 3 自动标注检出 1 个高置信度目标" in format_record_detail_summary(rec_high)
    assert render_review_comparison(rec_high)[:2] == b"\xff\xd8"

    rec_low = next(r for r in res_local.records if r.image_path.name == "02_low.jpg")
    assert rec_low.triage_bucket == "need_review"
    assert "低置信度" in rec_low.missing_note

    rec_empty = next(r for r in res_local.records if r.image_path.name == "03_empty.jpg")
    assert rec_empty.triage_bucket == "need_review"
    assert rec_empty.output_json_path is not None and rec_empty.output_json_path.is_file()
    assert json.loads(rec_empty.output_json_path.read_text(encoding="utf-8"))["shapes"] == []

    report_local = json.loads(res_local.report_path.read_text(encoding="utf-8"))
    assert report_local["source_mode"] == "sam3_text"
    assert report_local["sam3_enable_llm_review"] is False
    assert report_local["box_refine_mode"] == "off"

    # 4) SAM 3 预标 + LLM 二次质检：对 SAM 3 候选框执行改类/删误检，正确分流至 llm_fixed
    out_hybrid = tmp_path / "out_hybrid"
    hybrid_dir = tmp_path / "hybrid_in"
    _write_img(hybrid_dir / "01_high.jpg", (400, 200))
    _write_img(hybrid_dir / "02_low.jpg", (400, 200))
    fake_llm = _FakeMllmClient(
        [
            MllmReviewResponse(
                box_reviews=[BoxDecision(box_id=0, action="keep", reason="ok")],
                has_missing=False,
            ),
            MllmReviewResponse(
                box_reviews=[BoxDecision(box_id=0, action="relabel", new_label="人物", reason="误标为安全帽")],
                has_missing=False,
            ),
        ]
    )
    res_hybrid = run_mllm_review(
        MllmReviewConfig(
            images_dir=hybrid_dir,
            output_dir=out_hybrid,
            source_mode=ReviewSourceMode.SAM3_TEXT,
            allowed_classes=["人物=person", "安全帽=safety helmet"],
            sam3_enable_llm_review=True,
            max_workers=1,
        ),
        client_override=fake_llm,
        sam3_detector_override=_FakeSam3Detector(),
    )
    assert len(fake_llm.calls) == 2
    assert res_hybrid.auto_passed_count == 1
    assert res_hybrid.llm_fixed_count == 1
    rec_fixed = next(r for r in res_hybrid.records if r.image_path.name == "02_low.jpg")
    assert rec_fixed.relabeled_count == 1
    assert rec_fixed.after_boxes[0].label == "人物"
    report_hybrid = json.loads(res_hybrid.report_path.read_text(encoding="utf-8"))
    assert report_hybrid["source_mode"] == "sam3_text"
    assert report_hybrid["sam3_enable_llm_review"] is True

    # 5) 验证 resolve_sam3_weights 自动发现 workspace_root/weights 下的 sam3.pt / sam3.1_multiplex.pt，
    #    且显式传入不存在路径时返回 None（不静默回退）
    from yolo_studio.core.mllm_reviewer import resolve_sam3_weights

    ws = tmp_path / "ws_auto"
    ws_weights = ws / "weights"
    ws_weights.mkdir(parents=True)
    sam31 = ws_weights / "sam3.1_multiplex.pt"
    sam31.write_bytes(b"fake-sam31")
    assert resolve_sam3_weights(None, workspace_root=ws) == sam31.resolve()

    sam30 = ws_weights / "sam3.pt"
    sam30.write_bytes(b"fake-sam30")
    assert resolve_sam3_weights(None, workspace_root=ws) == sam30.resolve()
    assert resolve_sam3_weights(tmp_path / "definitely_missing.pt", workspace_root=ws) is None


def test_openai_client_downgrades_temperature_without_disabling_json_mode() -> None:
    """验证 OpenAI 兼容接口仅拒绝 `temperature` 时只省略 `temperature` 并保留 `json_object` 模式。"""
    import httpx
    import openai

    from yolo_studio.core.mllm_reviewer import OpenAICompatMllmClient

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    class _Msg:
        content = ok_json

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise openai.BadRequestError(
                    "Error code: 400 - Unsupported value: 'temperature' does not support 0.1 with this model. "
                    "Only the default (1) value is supported.",
                    response=httpx.Response(400, request=request),
                    body={
                        "message": "Unsupported value: 'temperature' does not support 0.1 with this model.",
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "unsupported_value",
                    },
                )
            return _Resp()

    client = OpenAICompatMllmClient(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model_name="custom-reasoning-alias",
    )
    completions = _Completions()
    client.client = type("C", (), {"chat": type("Chat", (), {"completions": completions})()})()

    client.review_image(b"x", [], "s", "u")
    client.review_image(b"x", [], "s", "u")

    # 第 1 轮先带 temperature+response_format 失败，随即省略 temperature 并保留 response_format 重试成功；
    # 第 2 轮直接省略 temperature 且保留 response_format 一次成功
    assert len(completions.calls) == 3
    assert "temperature" in completions.calls[0]
    assert completions.calls[0].get("response_format") == {"type": "json_object"}
    assert "temperature" not in completions.calls[1]
    assert completions.calls[1].get("response_format") == {"type": "json_object"}
    assert "temperature" not in completions.calls[2]
    assert completions.calls[2].get("response_format") == {"type": "json_object"}


def test_openai_client_downgrades_both_temperature_and_json_mode_and_raises_unrelated_400() -> None:
    """验证依次拒绝 `temperature` 与 `response_format`、并发降级竞争及无关 400 时的行为。"""
    import httpx
    import openai

    from yolo_studio.core.mllm_reviewer import OpenAICompatMllmClient

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    class _Msg:
        content = ok_json

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    # 1) 先后拒绝 temperature 与 response_format，最终降级至纯模型+消息参数
    class _RejectBothCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise openai.BadRequestError(
                    "Unsupported parameter: temperature",
                    response=httpx.Response(400, request=request),
                    body={"error": {"param": "temperature", "message": "Unsupported value"}},
                )
            if "response_format" in kwargs:
                raise openai.BadRequestError(
                    "response_format json_object is not supported",
                    response=httpx.Response(400, request=request),
                    body={"error": {"param": "response_format", "message": "not supported"}},
                )
            return _Resp()

    client = OpenAICompatMllmClient(api_key="k", base_url="http://localhost/v1", model_name="m")
    both = _RejectBothCompletions()
    client.client = type("C", (), {"chat": type("Chat", (), {"completions": both})()})()
    client.review_image(b"x", [], "s", "u")
    client.review_image(b"x", [], "s", "u")

    assert [
        ("temperature" in c, "response_format" in c) for c in both.calls
    ] == [(True, True), (False, True), (False, False), (False, False)]

    # 2) 并发竞争场景：本轮已携带 temperature=0.1 发出，但返回前另一线程已将 _temperature_supported 置为 False；
    #    本轮必须仍识别到自己发出过 temperature，重试时保留 json_mode 不误关
    race_client = OpenAICompatMllmClient(api_key="k", base_url="http://localhost/v1", model_name="m")

    class _RaceCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "temperature" in kwargs:
                race_client._temperature_supported = False
                raise openai.BadRequestError(
                    "Unsupported value: 'temperature'",
                    response=httpx.Response(400, request=request),
                    body={"param": "temperature"},
                )
            return _Resp()

    race_comp = _RaceCompletions()
    race_client.client = type("C", (), {"chat": type("Chat", (), {"completions": race_comp})()})()
    race_client.review_image(b"x", [], "s", "u")
    assert race_client._json_mode_supported is True
    assert len(race_comp.calls) == 2
    assert race_comp.calls[1].get("response_format") == {"type": "json_object"}

    # 3) 降级穷尽后，与参数降级无关的 400（如上下文超限）必须直接向上抛出
    class _ContextErrorCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            raise openai.BadRequestError(
                "This model's maximum context length is 128000 tokens.",
                response=httpx.Response(400, request=request),
                body={"error": {"code": "context_length_exceeded", "param": "messages"}},
            )

    err_client = OpenAICompatMllmClient(api_key="k", base_url="http://localhost/v1", model_name="m")
    err_comp = _ContextErrorCompletions()
    err_client.client = type("C", (), {"chat": type("Chat", (), {"completions": err_comp})()})()
    try:
        err_client.review_image(b"x", [], "s", "u")
    except openai.BadRequestError:
        pass
    else:  # pragma: no cover
        raise AssertionError("无关 400 错误应向上抛出")
    assert len(err_comp.calls) == 2


def test_mllm_connection_downgrades_max_tokens_and_temperature(monkeypatch) -> None:
    """验证 `test_mllm_connection` 在遇到不支持 `max_tokens` 与 `temperature` 时依次自动降级并在无关 400 时失败。"""
    import httpx
    import openai

    from yolo_studio.core.mllm_reviewer import test_mllm_connection

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    class _Msg:
        content = "OK"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    recorded_calls: list[dict[str, object]] = []
    mode = {"fail_unrelated": False}

    class _FakeCompletions:
        def create(self, **kwargs):
            recorded_calls.append(dict(kwargs))
            if mode["fail_unrelated"]:
                raise openai.BadRequestError(
                    "The model `missing-model` does not exist",
                    response=httpx.Response(400, request=request),
                    body={"error": {"code": "model_not_found", "param": "model"}},
                )
            if "max_tokens" in kwargs:
                raise openai.BadRequestError(
                    "Unsupported parameter: 'max_tokens' is not supported with this model. "
                    "Use 'max_completion_tokens' instead.",
                    response=httpx.Response(400, request=request),
                    body={"param": "max_tokens"},
                )
            if "temperature" in kwargs:
                raise openai.BadRequestError(
                    "Unsupported value: 'temperature' does not support 0.0 with this model.",
                    response=httpx.Response(400, request=request),
                    body={"param": "temperature"},
                )
            return _Resp()

    class _FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    ok, msg = test_mllm_connection(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.OPENAI_COMPAT,
            base_url="https://api.openai.com/v1",
            model_name="custom-reasoning-alias",
            api_key="test-key",
        )
    )
    assert ok is True
    assert "连通正常" in msg
    assert [
        ("max_tokens" in c, "temperature" in c, "max_completion_tokens" in c)
        for c in recorded_calls
    ] == [(True, True, False), (False, True, False), (False, False, False)]

    recorded_calls.clear()
    mode["fail_unrelated"] = True
    ok_bad, msg_bad = test_mllm_connection(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.OPENAI_COMPAT,
            base_url="https://api.openai.com/v1",
            model_name="missing-model",
            api_key="test-key",
        )
    )
    assert ok_bad is False
    assert "model_not_found" in msg_bad or "does not exist" in msg_bad
    assert len(recorded_calls) == 1


def test_error_record_summary_does_not_duplicate_error_in_missing_note(tmp_path: Path) -> None:
    """验证单图异常记录的详情摘要不再把同一报错作为漏标提示重复拼接，同时 JSON 文件仍保留异常注释。"""
    from yolo_studio.core.mllm_reviewer import ImageReviewRecord, format_record_detail_summary

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _write_img(in_dir / "err.png", (320, 180))

    class _FailClient:
        def review_image(self, *_args):
            raise RuntimeError("Error code: 400 - Unsupported value: temperature")

    res = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
        ),
        client_override=_FailClient(),
    )
    assert res.error_count == 1
    rec = res.records[0]
    assert rec.error == "Error code: 400 - Unsupported value: temperature"
    assert rec.missing_note == ""
    summary = format_record_detail_summary(rec)
    assert summary.count("Unsupported value: temperature") == 1
    assert "漏标提示" not in summary
    assert "检测到疑似漏标" not in summary

    assert rec.output_json_path is not None and rec.output_json_path.is_file()
    json_payload = json.loads(rec.output_json_path.read_text(encoding="utf-8"))
    assert "LLM 处理异常:" in json_payload["flags"]["missing_note"]

    # 兼容加载旧版历史报告（missing_note 仍含 "处理出错: <error>"）时同样不重复显示
    legacy_rec = ImageReviewRecord(
        image_path=in_dir / "err.png",
        output_image_path=rec.output_image_path,
        output_json_path=rec.output_json_path,
        triage_bucket="need_review",
        has_missing=True,
        missing_note="处理出错: Error code: 400 - Unsupported value: temperature",
        error="Error code: 400 - Unsupported value: temperature",
    )
    legacy_summary = format_record_detail_summary(legacy_rec)
    assert legacy_summary.count("Unsupported value: temperature") == 1
    assert "漏标提示" not in legacy_summary
    assert "检测到疑似漏标" not in legacy_summary


def test_openai_nextgen_models_omit_temperature_and_max_tokens_upfront(monkeypatch) -> None:
    """验证 OpenAI 新系列（`gpt-5+` / `o` 系列，含中转前缀与微调名，不限 Base URL）首次请求即省略采样与长度限制参数。"""
    import openai

    from yolo_studio.core.mllm_reviewer import OpenAICompatMllmClient, test_mllm_connection

    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    class _Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:
        def __init__(self, content: str) -> None:
            self.message = _Msg(content)

    class _Resp:
        def __init__(self, content: str) -> None:
            self.choices = [_Choice(content)]

    nextgen_cases = [
        ("https://api.openai.com/v1", "gpt-6.2-astra"),
        ("https://api.openai.com/v1", "gpt-5-mini"),
        ("https://proxy.example.com/v1", "openai/gpt-6.2-astra"),
        ("https://api.openai.com/v1", "o3-mini"),
        ("https://proxy.example.com/v1", "o4-mini"),
        ("https://api.openai.com/v1", "ft:gpt-5:org::id"),
    ]
    for base_url, model_name in nextgen_cases:
        calls: list[dict[str, object]] = []

        class _SingleOkCompletions:
            def create(self, **kwargs):
                calls.append(dict(kwargs))
                return _Resp(ok_json)

        client = OpenAICompatMllmClient(api_key="k", base_url=base_url, model_name=model_name)
        client.client = type("C", (), {"chat": type("Chat", (), {"completions": _SingleOkCompletions()})()})()
        client.review_image(b"x", [], "s", "u")
        assert len(calls) == 1, f"{model_name} 应单次成功，不应先触发 400"
        assert "temperature" not in calls[0], f"{model_name} 首次请求不应传 temperature"
        assert calls[0].get("response_format") == {"type": "json_object"}

    legacy_cases = [
        ("https://api.openai.com/v1", "gpt-4o"),
        ("https://api.openai.com/v1", "gpt-4.1"),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-vl-max-latest"),
        ("https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-VL-72B-Instruct"),
    ]
    for base_url, model_name in legacy_cases:
        calls = []

        class _LegacyCompletions:
            def create(self, **kwargs):
                calls.append(dict(kwargs))
                return _Resp(ok_json)

        client = OpenAICompatMllmClient(api_key="k", base_url=base_url, model_name=model_name)
        client.client = type("C", (), {"chat": type("Chat", (), {"completions": _LegacyCompletions()})()})()
        client.review_image(b"x", [], "s", "u")
        assert len(calls) == 1
        assert calls[0].get("temperature") == 0.1

    probe_calls: list[dict[str, object]] = []

    class _ProbeCompletions:
        def create(self, **kwargs):
            probe_calls.append(dict(kwargs))
            return _Resp("OK")

    class _FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = type("Chat", (), {"completions": _ProbeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    for base_url, model_name in (
        ("https://api.openai.com/v1", "gpt-6.2-astra"),
        ("https://proxy.example.com/v1", "openai/o3"),
    ):
        probe_calls.clear()
        ok, msg = test_mllm_connection(
            MllmReviewConfig(
                images_dir=Path("."),
                output_dir=Path("."),
                provider=MllmProvider.OPENAI_COMPAT,
                base_url=base_url,
                model_name=model_name,
                api_key="test-key",
            )
        )
        assert ok is True
        assert "连通正常" in msg
        assert len(probe_calls) == 1
        assert "temperature" not in probe_calls[0]
        assert "max_tokens" not in probe_calls[0]

    probe_calls.clear()
    ok_4o, _ = test_mllm_connection(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.OPENAI_COMPAT,
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o",
            api_key="test-key",
        )
    )
    assert ok_4o is True
    assert len(probe_calls) == 1
    assert probe_calls[0].get("temperature") == 0.0
    assert probe_calls[0].get("max_tokens") == 12
