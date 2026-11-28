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


def test_parse_review_response_normalizes_nonstandard_box_quality_and_null_fields() -> None:
    """验证结构化响应解析对非标准 `box_quality`、`null` 字段、浮点坐标及 `box_id` 字符串格式的前置宽容规范化。"""
    import pytest
    from pydantic import ValidationError

    # 覆盖线上实际出现过的全部非标准评价词及常见边界形态
    nonstandard_qualities = [
        "good",
        "fair",
        "poor",
        "wrong_class",
        "invalid",
        "mixed",
        "duplicate",
        "non_object",
        "overlapping_instances",
        "oversized",
        "cropped",
        None,
    ]
    box_reviews_payload = [
        {
            "box_id": f"[{idx}]" if idx == 1 else (f"#{idx}" if idx == 2 else idx),
            "action": "KEEP" if idx % 3 == 0 else ("rename" if idx % 3 == 1 else "discard"),
            "new_label": "monster" if idx % 3 == 1 else None,
            "box_quality": quality,
            "refined_box_2d": [100.4, 200.6, 800.2, 900.8] if quality == "oversized" else [],
            "reason": None if idx % 2 == 0 else "check",
        }
        for idx, quality in enumerate(nonstandard_qualities)
    ]
    raw_payload = json.dumps(
        {
            "box_reviews": box_reviews_payload,
            "has_missing": None,
            "missing_note": None,
            "added_boxes": [
                {"label": "loot", "box_2d": [120.2, 340.7, 450.9, 660.1], "reason": None}
            ],
        },
        ensure_ascii=False,
    )

    parsed = parse_review_response_text(raw_payload)
    assert len(parsed.box_reviews) == len(nonstandard_qualities)
    assert parsed.box_reviews[1].box_id == 1
    assert parsed.box_reviews[2].box_id == 2
    assert parsed.box_reviews[1].action == "relabel"
    assert parsed.box_reviews[2].action == "delete"
    assert parsed.has_missing is False
    assert parsed.missing_note == ""
    assert parsed.added_boxes[0].box_2d == [120, 341, 451, 660]
    assert parsed.added_boxes[0].reason == ""

    qualities_out = [item.box_quality for item in parsed.box_reviews]
    oversized_idx = nonstandard_qualities.index("oversized")
    cropped_idx = nonstandard_qualities.index("cropped")
    assert qualities_out[oversized_idx] == "loose"
    assert parsed.box_reviews[oversized_idx].refined_box_2d == [100, 201, 800, 901]
    assert qualities_out[cropped_idx] == "partial"
    for idx, q_out in enumerate(qualities_out):
        if idx not in (oversized_idx, cropped_idx):
            assert q_out == "tight"
            assert parsed.box_reviews[idx].refined_box_2d is None

    # 校验 null 列表字段同样安全回退为空列表
    parsed_null_lists = parse_review_response_text(
        '{"box_reviews": null, "has_missing": true, "missing_note": null, "added_boxes": null}'
    )
    assert parsed_null_lists.box_reviews == []
    assert parsed_null_lists.added_boxes == []
    assert parsed_null_lists.has_missing is True
    assert parsed_null_lists.missing_note == ""

    # 校验真正非法的 action 仍会被严格拦截
    with pytest.raises(ValidationError):
        parse_review_response_text(
            '{"box_reviews": [{"box_id": 0, "action": "unknown_action"}]}'
        )

    # 校验 JSON Schema 仍完整保留枚举定义，确保 Gemini Interactions API 契约不受影响
    schema = MllmReviewResponse.model_json_schema()
    box_props = schema["$defs"]["BoxDecision"]["properties"]
    assert box_props["box_quality"]["enum"] == ["tight", "loose", "partial"]
    assert box_props["action"]["enum"] == ["keep", "relabel", "delete"]


def test_build_prompts_specifies_box_quality_enum_in_all_refine_modes() -> None:
    """验证四种边框修正模式下提示词均明确写出 `box_quality` 允许枚举值及关闭/删框时的默认取值。"""
    from yolo_studio.core.mllm_reviewer import BoxRefineMode, _build_prompts

    boxes = [CandidateBox(box_id=0, label="player", x1=10.0, y1=20.0, x2=100.0, y2=120.0, confidence=0.75)]
    for mode in (
        BoxRefineMode.OFF,
        BoxRefineMode.FLAG_LOOSE,
        BoxRefineMode.MLLM_CROP,
        BoxRefineMode.SAM_LOCAL,
    ):
        cfg = MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            allowed_classes=["player", "monster"],
            box_refine_mode=mode,
        )
        sys_prompt, user_prompt = _build_prompts(boxes, ["player", "monster"], cfg)
        assert "'tight'|'loose'|'partial'" in user_prompt
        if mode == BoxRefineMode.OFF:
            assert "Set `box_quality` to 'tight'" in user_prompt
        else:
            assert "MUST ONLY be one of 'tight', 'loose', or 'partial'" in sys_prompt
            assert "for 'delete'" in sys_prompt


def test_openai_official_provider_separated_from_compat(monkeypatch) -> None:
    """验证 OpenAI 官方作为独立提供方：移除出兼容预设、强制校验 Key、固定官方端点并正确连接。"""
    import openai
    import pytest

    from yolo_studio.core.mllm_reviewer import (
        GEMINI_MODELS,
        OPENAI_MODELS,
        OPENAI_OFFICIAL_BASE_URL,
        OPENAI_PRESETS,
        REASONING_EFFORT_HINTS,
        REASONING_EFFORT_ITEMS,
        build_mllm_client,
        test_mllm_connection,
    )

    # 1) OPENAI_MODELS 仅含 GPT-6.2，GEMINI_MODELS 仅含 Gemini 4；OPENAI_PRESETS 不再包含 OpenAI 官方；预设与推理说明不含 MLLM 字样
    assert [m for m, _ in OPENAI_MODELS] == ["gpt-6.2-sol", "gpt-6.2-astra"]
    assert all(m.startswith("gpt-6.2-") for m, _ in OPENAI_MODELS)
    assert [m for m, _ in GEMINI_MODELS] == [
        "gemini-4-hydrogen",
        "gemini-4-neon",
        "gemini-4-argon",
        "gemini-4-carbon",
    ]
    assert all(m.startswith("gemini-4-") for m, _ in GEMINI_MODELS)
    assert all("api.openai.com" not in url and "OpenAI 官方" not in name for name, url, _ in OPENAI_PRESETS)
    for _, label in OPENAI_MODELS + GEMINI_MODELS:
        assert "MLLM" not in label
    for _, label in REASONING_EFFORT_ITEMS:
        assert "MLLM" not in label
    for hint in REASONING_EFFORT_HINTS.values():
        assert "MLLM" not in hint

    # 2) OpenAI 官方缺失 Key 时 build_mllm_client 与 test_mllm_connection 均严格拒绝，不回退 ollama
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg_no_key = MllmReviewConfig(
        images_dir=Path("."),
        output_dir=Path("."),
        provider=MllmProvider.OPENAI,
        model_name="gpt-6.2-sol",
        base_url="http://localhost:11434/v1",  # 即便残留兼容地址也不应被官方模式采纳
        api_key="",
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_mllm_client(cfg_no_key)

    ok_missing, msg_missing = test_mllm_connection(cfg_no_key)
    assert ok_missing is False
    assert "OPENAI_API_KEY" in msg_missing

    # 3) 提供 Key 后，build_mllm_client 与 test_mllm_connection 固定连向官方 Base URL
    captured_inits: list[dict[str, object]] = []

    class _FakeResp:
        class _Choice:
            class _Msg:
                content = "OK"

            message = _Msg()

        choices = [_Choice()]

    class _FakeCompletions:
        def create(self, **_kwargs):
            return _FakeResp()

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured_inits.append(dict(kwargs))
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    cfg_ok = MllmReviewConfig(
        images_dir=Path("."),
        output_dir=Path("."),
        provider=MllmProvider.OPENAI,
        model_name="gpt-6.2-sol",
        base_url="http://localhost:11434/v1",
        api_key="sk-official-test",
    )
    client = build_mllm_client(cfg_ok)
    assert client.model_name == "gpt-6.2-sol"
    assert captured_inits[-1]["api_key"] == "sk-official-test"
    assert captured_inits[-1]["base_url"] == OPENAI_OFFICIAL_BASE_URL

    ok_conn, msg_conn = test_mllm_connection(cfg_ok)
    assert ok_conn is True
    assert "OpenAI 官方接口连通正常" in msg_conn
    assert captured_inits[-1]["base_url"] == OPENAI_OFFICIAL_BASE_URL


def test_gemini_thinking_level_mapping_and_bad_request_downgrade(monkeypatch) -> None:
    """验证 Gemini 思考级别按模型映射并在遇 400 拒绝时自动单调降级。"""
    from google import genai

    from yolo_studio.core.mllm_reviewer import GeminiMllmClient, ReasoningEffort

    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'
    calls_log: list[dict[str, object]] = []

    class _FakeInteractions:
        def __init__(self, reject_levels: set[str] | None = None, fail_auth: bool = False) -> None:
            self.reject_levels = reject_levels or set()
            self.fail_auth = fail_auth

        def create(self, **kwargs):
            calls_log.append(dict(kwargs))
            if self.fail_auth:
                raise RuntimeError("401 UNAUTHENTICATED: invalid api key")
            gen_cfg = kwargs.get("generation_config") or {}
            level = gen_cfg.get("thinking_level") if isinstance(gen_cfg, dict) else None
            if level in self.reject_levels:
                err = RuntimeError(f"400 INVALID_ARGUMENT: thinking_level '{level}' is not supported")
                setattr(err, "code", 400)
                raise err
            return type("Resp", (), {"output_text": ok_json})()

    monkeypatch.setattr(genai, "Client", lambda **_kw: type("C", (), {"interactions": _FakeInteractions()})())

    # 1) AUTO 不传 generation_config；flash-lite + OFF 传 minimal；3.8-flash + OFF 事前直传 low
    c_auto = GeminiMllmClient(api_key="k", model_name="gemini-3.8-flash", reasoning_effort=ReasoningEffort.AUTO)
    c_auto.review_image(b"x", [], "s", "u")
    assert "generation_config" not in calls_log[-1]

    c_lite_off = GeminiMllmClient(
        api_key="k", model_name="gemini-3.5-flash-lite", reasoning_effort=ReasoningEffort.OFF
    )
    c_lite_off.review_image(b"x", [], "s", "u")
    assert calls_log[-1]["generation_config"] == {"thinking_level": "minimal"}

    c_flash_off = GeminiMllmClient(
        api_key="k", model_name="gemini-3.8-flash", reasoning_effort=ReasoningEffort.OFF
    )
    c_flash_off.review_image(b"x", [], "s", "u")
    assert calls_log[-1]["generation_config"] == {"thinking_level": "low"}

    # 2) gemini-4-neon 支持 xhigh：遇 400 依次从 xhigh -> high -> 省略，并在后续调用记住最终可用档位；
    #    gemini-4-carbon / gemini-4-argon 不支持 xhigh：事前直降为 high，不先发 xhigh 撞 400；
    #    gemini-4-hydrogen 支持 xhigh：直接下发 xhigh
    calls_log.clear()
    c_xhigh = GeminiMllmClient(api_key="k", model_name="gemini-4-neon", reasoning_effort=ReasoningEffort.XHIGH)
    c_xhigh.client = type("C", (), {"interactions": _FakeInteractions(reject_levels={"xhigh", "high"})})()
    c_xhigh.review_image(b"x", [], "s", "u")
    c_xhigh.review_image(b"x", [], "s", "u")
    assert [
        (c.get("generation_config") or {}).get("thinking_level") if isinstance(c.get("generation_config"), dict) else None
        for c in calls_log
    ] == ["xhigh", "high", None, None]

    calls_log.clear()
    for no_xhigh_model in ("gemini-4-carbon", "gemini-4-argon"):
        c_no_x = GeminiMllmClient(api_key="k", model_name=no_xhigh_model, reasoning_effort=ReasoningEffort.XHIGH)
        c_no_x.client = type("C", (), {"interactions": _FakeInteractions()})()
        c_no_x.review_image(b"x", [], "s", "u")
        assert calls_log[-1]["generation_config"] == {"thinking_level": "high"}

    c_hydro = GeminiMllmClient(api_key="k", model_name="gemini-4-hydrogen", reasoning_effort=ReasoningEffort.XHIGH)
    c_hydro.client = type("C", (), {"interactions": _FakeInteractions()})()
    c_hydro.review_image(b"x", [], "s", "u")
    assert calls_log[-1]["generation_config"] == {"thinking_level": "xhigh"}

    # 3) 非 400 鉴权异常不得被吞掉
    c_auth = GeminiMllmClient(api_key="k", model_name="gemini-3.8-flash", reasoning_effort=ReasoningEffort.HIGH)
    c_auth.client = type("C", (), {"interactions": _FakeInteractions(fail_auth=True)})()
    import pytest

    with pytest.raises(RuntimeError, match="401 UNAUTHENTICATED"):
        c_auth.review_image(b"x", [], "s", "u")


def test_openai_reasoning_effort_mapping_and_bad_request_downgrade() -> None:
    """验证 OpenAI 推理级别事前预判、400 单调降级且不误关 `json_object` 模式，以及 CLI 参数解析。"""
    import httpx
    import openai

    from yolo_studio.cli import build_parser
    from yolo_studio.core.mllm_reviewer import OpenAICompatMllmClient, ReasoningEffort

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    class _Msg:
        content = ok_json

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    # 1) gpt-4o / gpt-4.1 即便指定 HIGH 也事前省略 reasoning_effort；o4-mini + OFF 事前直传 low
    for non_r_model in ("gpt-4o", "gpt-4.1"):
        calls: list[dict[str, object]] = []
        client = OpenAICompatMllmClient(
            api_key="k",
            base_url="https://api.openai.com/v1",
            model_name=non_r_model,
            reasoning_effort=ReasoningEffort.HIGH,
            allow_anonymous_key=False,
        )
        client.client = type(
            "C",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Comp", (), {"create": lambda _s, **kw: (calls.append(dict(kw)), _Resp())[1]}
                        )()
                    },
                )()
            },
        )()
        client.review_image(b"x", [], "s", "u")
        assert "reasoning_effort" not in calls[0]

    o_calls: list[dict[str, object]] = []
    for min_low_model in ("o4-mini", "gpt-6.2-sol", "gpt-6.2-astra"):
        o_calls.clear()
        o_client = OpenAICompatMllmClient(
            api_key="k",
            base_url="https://api.openai.com/v1",
            model_name=min_low_model,
            reasoning_effort=ReasoningEffort.OFF,
            allow_anonymous_key=False,
        )
        o_client.client = type(
            "C",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Comp", (), {"create": lambda _s, **kw: (o_calls.append(dict(kw)), _Resp())[1]}
                        )()
                    },
                )()
            },
        )()
        o_client.review_image(b"x", [], "s", "u")
        assert o_calls[0]["reasoning_effort"] == "low"

    # 2) gpt-5.4 + OFF 遇服务端拒绝 none 和 minimal，降级至 low 成功且全程保留 response_format
    seq_calls: list[dict[str, object]] = []

    class _RejectNoneMinimal:
        def create(self, **kwargs):
            seq_calls.append(dict(kwargs))
            if kwargs.get("reasoning_effort") in {"none", "minimal"}:
                raise openai.BadRequestError(
                    "Unsupported value for reasoning_effort",
                    response=httpx.Response(400, request=request),
                    body={"error": {"param": "reasoning_effort", "message": "unsupported value"}},
                )
            return _Resp()

    gpt5_client = OpenAICompatMllmClient(
        api_key="k",
        base_url="https://api.openai.com/v1",
        model_name="gpt-5.4",
        reasoning_effort=ReasoningEffort.OFF,
        allow_anonymous_key=False,
    )
    gpt5_client.client = type("C", (), {"chat": type("Chat", (), {"completions": _RejectNoneMinimal()})()})()
    gpt5_client.review_image(b"x", [], "s", "u")
    gpt5_client.review_image(b"x", [], "s", "u")
    assert [c.get("reasoning_effort") for c in seq_calls] == ["none", "minimal", "low", "low"]
    assert all(c.get("response_format") == {"type": "json_object"} for c in seq_calls)

    # 3) 第三方兼容接口对 reasoning_effort 返回笼统 400 时，先放弃 reasoning_effort 而不是先关 json_mode
    compat_calls: list[dict[str, object]] = []

    class _VagueRejectReasoning:
        def create(self, **kwargs):
            compat_calls.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise openai.BadRequestError(
                    "invalid payload",
                    response=httpx.Response(400, request=request),
                    body={"message": "invalid payload"},
                )
            return _Resp()

    compat_client = OpenAICompatMllmClient(
        api_key="k",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-vl-max-latest",
        reasoning_effort=ReasoningEffort.HIGH,
    )
    compat_client.client = type("C", (), {"chat": type("Chat", (), {"completions": _VagueRejectReasoning()})()})()
    compat_client.review_image(b"x", [], "s", "u")
    assert len(compat_calls) == 2
    assert "reasoning_effort" in compat_calls[0]
    assert "reasoning_effort" not in compat_calls[1]
    assert compat_calls[1].get("response_format") == {"type": "json_object"}

    # 4) CLI review 子命令默认模型及支持 --provider openai 与 --reasoning-effort
    parser = build_parser()
    default_parsed = parser.parse_args(["review", "-i", "data/raw"])
    assert default_parsed.model == "gemini-4-hydrogen"
    parsed = parser.parse_args(
        [
            "review",
            "-i",
            "data/raw",
            "--provider",
            "openai",
            "--model",
            "gpt-6.2-sol",
            "--reasoning-effort",
            "medium",
        ]
    )
    assert parsed.provider == "openai"
    assert parsed.model == "gpt-6.2-sol"
    assert parsed.reasoning_effort == "medium"


def test_reasoning_effort_capability_matrix_and_guard(tmp_path: Path) -> None:
    """验证各模型思考级别能力矩阵、事前回退提示以及 `run_mllm_review` 日志与报告落地一致性。"""
    from yolo_studio.core.mllm_reviewer import (
        ReasoningEffort,
        resolve_effective_reasoning_effort,
        supported_reasoning_efforts,
    )

    # 1) Gemini 系列能力矩阵
    for m in ("gemini-4-carbon", "gemini-4-argon"):
        assert supported_reasoning_efforts(MllmProvider.GEMINI, m) == (
            ReasoningEffort.AUTO,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
        )
        eff_x, msg_x = resolve_effective_reasoning_effort(MllmProvider.GEMINI, m, ReasoningEffort.XHIGH)
        assert eff_x == ReasoningEffort.HIGH
        assert msg_x and "xhigh" in msg_x
        eff_off, msg_off = resolve_effective_reasoning_effort(MllmProvider.GEMINI, m, ReasoningEffort.OFF)
        assert eff_off == ReasoningEffort.LOW
        assert msg_off and "low" in msg_off

    for m in ("gemini-4-hydrogen", "gemini-4-neon"):
        assert supported_reasoning_efforts(MllmProvider.GEMINI, m) == (
            ReasoningEffort.AUTO,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        )
        assert resolve_effective_reasoning_effort(MllmProvider.GEMINI, m, ReasoningEffort.XHIGH) == (
            ReasoningEffort.XHIGH,
            None,
        )

    assert supported_reasoning_efforts(MllmProvider.GEMINI, "gemini-3.5-flash-lite") == (
        ReasoningEffort.AUTO,
        ReasoningEffort.OFF,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
    )

    # 2) OpenAI 官方与兼容系列能力矩阵
    for m in ("gpt-6.2-sol", "gpt-6.2-astra"):
        assert supported_reasoning_efforts(MllmProvider.OPENAI, m) == (
            ReasoningEffort.AUTO,
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
        )
        eff_off, msg_off = resolve_effective_reasoning_effort(MllmProvider.OPENAI, m, ReasoningEffort.OFF)
        assert eff_off == ReasoningEffort.LOW
        assert msg_off and "low" in msg_off

    assert supported_reasoning_efforts(MllmProvider.OPENAI, "o4-mini") == (
        ReasoningEffort.AUTO,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
    )
    assert supported_reasoning_efforts(MllmProvider.OPENAI, "gpt-5.4") == (
        ReasoningEffort.AUTO,
        ReasoningEffort.OFF,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    )
    assert supported_reasoning_efforts(MllmProvider.OPENAI, "gpt-5-mini") == (
        ReasoningEffort.AUTO,
        ReasoningEffort.OFF,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
    )
    for non_r_provider, non_r_model in (
        (MllmProvider.OPENAI, "gpt-4o"),
        (MllmProvider.OPENAI_COMPAT, "Qwen/Qwen2.5-VL-72B-Instruct"),
    ):
        assert supported_reasoning_efforts(non_r_provider, non_r_model) == (ReasoningEffort.AUTO,)
        eff_nr, msg_nr = resolve_effective_reasoning_effort(non_r_provider, non_r_model, ReasoningEffort.HIGH)
        assert eff_nr == ReasoningEffort.AUTO
        assert msg_nr and "非推理直出" in msg_nr

    assert supported_reasoning_efforts(MllmProvider.OPENAI_COMPAT, "qwen-vl-max-latest") == (
        ReasoningEffort.AUTO,
        ReasoningEffort.OFF,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
        ReasoningEffort.HIGH,
        ReasoningEffort.XHIGH,
    )

    # 3) run_mllm_review 执行 gemini-4-carbon + XHIGH 时输出中文降级提示且报告记录实际生效级别 high
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _write_img(in_dir / "sample.jpg", (200, 120))
    _person_json(in_dir / "sample.jpg")

    logs: list[str] = []
    res = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.EXISTING_JSON,
            provider=MllmProvider.GEMINI,
            model_name="gemini-4-carbon",
            reasoning_effort=ReasoningEffort.XHIGH,
            allowed_classes=["person"],
            max_workers=1,
        ),
        client_override=_FakeMllmClient(
            [MllmReviewResponse(box_reviews=[BoxDecision(box_id=0, action="keep")], has_missing=False)]
        ),
        log_callback=logs.append,
    )
    assert any("不支持 xhigh" in line for line in logs)
    assert any("推理级别: high" in line for line in logs)
    assert res.report_path is not None
    report = json.loads(res.report_path.read_text(encoding="utf-8"))
    assert report["reasoning_effort"] == "high"


def test_mllm_timeouts_exceed_official_max_and_retries_follow_effort(monkeypatch) -> None:
    """验证各提供方与各思考级别超时均大于官方 600s 上限，且深度推理禁用自动重试以免重复扣费。"""
    import openai
    from google import genai

    from yolo_studio.core.mllm_reviewer import (
        MLLM_CONNECT_TIMEOUT_SECONDS,
        MLLM_PROBE_TIMEOUT_SECONDS,
        MLLM_REQUEST_TIMEOUTS_SECONDS,
        GeminiMllmClient,
        OpenAICompatMllmClient,
        ReasoningEffort,
        build_mllm_client,
        resolve_mllm_max_retries,
        resolve_mllm_request_timeout,
        test_mllm_connection,
    )

    # 1) 全矩阵超时解析均严格大于官方默认最大超时（600.0s），且随思考级别单调递增
    official_max_timeout = 600.0
    for effort, val in MLLM_REQUEST_TIMEOUTS_SECONDS.items():
        assert val > official_max_timeout, f"{effort.value} 超时 {val}s 应大于官方上限 {official_max_timeout}s"

    for provider, model_name in (
        (MllmProvider.GEMINI, "gemini-4-neon"),
        (MllmProvider.GEMINI, "gemini-3.5-flash-lite"),
        (MllmProvider.OPENAI, "gpt-6.2-astra"),
        (MllmProvider.OPENAI, "gpt-4o"),
        (MllmProvider.OPENAI_COMPAT, "qwen-vl-max-latest"),
        (MllmProvider.OPENAI_COMPAT, "Qwen/Qwen2.5-VL-72B-Instruct"),
    ):
        for effort in ReasoningEffort:
            resolved = resolve_mllm_request_timeout(provider, model_name, effort)
            assert resolved > official_max_timeout

    # 支持完整 OFF..XHIGH 的模型按档位严格递增；不支持 OFF 的 gpt-6.2-astra 会先折算至 LOW (720s)
    assert (
        resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-5.4", ReasoningEffort.OFF)
        < resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-5.4", ReasoningEffort.LOW)
        < resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-5.4", ReasoningEffort.AUTO)
        == resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-5.4", ReasoningEffort.MEDIUM)
        < resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-5.4", ReasoningEffort.HIGH)
        < resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-5.4", ReasoningEffort.XHIGH)
    )
    assert (
        resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-6.2-astra", ReasoningEffort.OFF)
        == resolve_mllm_request_timeout(MllmProvider.OPENAI, "gpt-6.2-astra", ReasoningEffort.LOW)
        == 720.0
    )
    assert resolve_mllm_max_retries(ReasoningEffort.HIGH) == 0
    assert resolve_mllm_max_retries(ReasoningEffort.XHIGH) == 0
    for normal_effort in (
        ReasoningEffort.AUTO,
        ReasoningEffort.OFF,
        ReasoningEffort.LOW,
        ReasoningEffort.MEDIUM,
    ):
        assert resolve_mllm_max_retries(normal_effort) == 1

    # 2) GeminiMllmClient 按模型能力矩阵折算毫秒传入 HttpOptions(timeout=...)，且支持显式覆盖
    gemini_inits: list[dict[str, object]] = []

    class _FakeGenaiClient:
        def __init__(self, **kwargs) -> None:
            gemini_inits.append(dict(kwargs))
            self.interactions = object()

    monkeypatch.setattr(genai, "Client", _FakeGenaiClient)

    g_carbon = GeminiMllmClient(
        api_key="k",
        model_name="gemini-4-carbon",
        reasoning_effort=ReasoningEffort.XHIGH,  # carbon 最高支持 high -> 1200s -> 1_200_000ms
    )
    assert g_carbon.request_timeout == 1200.0
    assert gemini_inits[-1]["http_options"].timeout == 1_200_000

    g_neon = GeminiMllmClient(
        api_key="k",
        model_name="gemini-4-neon",
        reasoning_effort=ReasoningEffort.XHIGH,
    )
    assert g_neon.request_timeout == 1500.0
    assert gemini_inits[-1]["http_options"].timeout == 1_500_000

    g_override = GeminiMllmClient(
        api_key="k",
        model_name="gemini-4-hydrogen",
        reasoning_effort=ReasoningEffort.LOW,
        timeout=1800.0,
    )
    assert g_override.request_timeout == 1800.0
    assert gemini_inits[-1]["http_options"].timeout == 1_800_000

    # 3) OpenAICompatMllmClient（官方与兼容接口）传入 openai.Timeout 与 max_retries
    openai_inits: list[dict[str, object]] = []

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            openai_inits.append(dict(kwargs))
            self.chat = type("Chat", (), {"completions": object()})()

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    oa_high = build_mllm_client(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.OPENAI,
            model_name="gpt-6.2-astra",
            api_key="sk-test",
            reasoning_effort=ReasoningEffort.HIGH,
        )
    )
    assert isinstance(oa_high, OpenAICompatMllmClient)
    assert oa_high.request_timeout == 1200.0
    assert oa_high.max_retries == 0
    last_oa_kw = openai_inits[-1]
    assert isinstance(last_oa_kw["timeout"], openai.Timeout)
    assert last_oa_kw["timeout"].read == 1200.0
    assert last_oa_kw["timeout"].connect == MLLM_CONNECT_TIMEOUT_SECONDS
    assert last_oa_kw["max_retries"] == 0

    compat_low = build_mllm_client(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.OPENAI_COMPAT,
            base_url="http://localhost:11434/v1",
            model_name="qwen2.5vl:7b",
            reasoning_effort=ReasoningEffort.HIGH,  # 非推理模型折算回 AUTO (900s)
        )
    )
    assert isinstance(compat_low, OpenAICompatMllmClient)
    assert compat_low.request_timeout == 900.0
    assert compat_low.max_retries == 1
    assert openai_inits[-1]["timeout"].read == 900.0
    assert openai_inits[-1]["max_retries"] == 1

    explicit_oa = OpenAICompatMllmClient(
        api_key="k",
        base_url="http://localhost:11434/v1",
        model_name="qwen-vl-max-latest",
        timeout=2000.0,
        max_retries=2,
    )
    assert explicit_oa.request_timeout == 2000.0
    assert explicit_oa.max_retries == 2
    assert openai_inits[-1]["timeout"].read == 2000.0
    assert openai_inits[-1]["max_retries"] == 2

    # 4) 连通性探针超时为 60s（OpenAI 探针 max_retries=0），失败时透出底层异常类型与 __cause__
    import httpx

    gemini_inits.clear()
    openai_inits.clear()

    class _FailGenaiInteractions:
        def create(self, **_kwargs):
            raise TimeoutError("probe read timed out")

    class _FailGenaiClient:
        def __init__(self, **kwargs) -> None:
            gemini_inits.append(dict(kwargs))
            self.interactions = _FailGenaiInteractions()

    monkeypatch.setattr(genai, "Client", _FailGenaiClient)
    ok_g, msg_g = test_mllm_connection(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.GEMINI,
            model_name="gemini-4-hydrogen",
            api_key="k",
        )
    )
    assert ok_g is False
    assert gemini_inits[-1]["http_options"].timeout == int(round(MLLM_PROBE_TIMEOUT_SECONDS * 1000))
    assert "TimeoutError" in msg_g

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    class _ConnErrCompletions:
        def create(self, **_kwargs):
            root_cause = OSError("TLS handshake reset by peer")
            err = openai.APIConnectionError(request=req)
            err.__cause__ = root_cause
            raise err

    class _FailOpenAI:
        def __init__(self, **kwargs) -> None:
            openai_inits.append(dict(kwargs))
            self.chat = type("Chat", (), {"completions": _ConnErrCompletions()})()

    monkeypatch.setattr(openai, "OpenAI", _FailOpenAI)
    ok_o, msg_o = test_mllm_connection(
        MllmReviewConfig(
            images_dir=Path("."),
            output_dir=Path("."),
            provider=MllmProvider.OPENAI,
            model_name="gpt-6.2-astra",
            api_key="sk-test",
        )
    )
    assert ok_o is False
    assert openai_inits[-1]["timeout"].read == MLLM_PROBE_TIMEOUT_SECONDS
    assert openai_inits[-1]["max_retries"] == 0
    assert "APIConnectionError" in msg_o
    assert "TLS handshake reset by peer" in msg_o


def test_run_mllm_review_logs_timeout_and_exception_root_cause(tmp_path: Path) -> None:
    """验证 `run_mllm_review` 启动日志含单次超时配置，且失败记录与运行日志完整透出底层原因。"""
    import httpx
    import openai

    from yolo_studio.core.mllm_reviewer import ReasoningEffort

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _write_img(in_dir / "a_timeout.png", (320, 180))
    _write_img(in_dir / "b_conn.png", (320, 180))

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    class _TimeoutAndConnClient:
        request_timeout = 1200.0

        def review_image(self, *_args):
            if not hasattr(self, "_called"):
                self._called = True
                err = openai.APITimeoutError(request=req)
                err.__cause__ = TimeoutError("The read operation timed out")
                raise err
            err = openai.APIConnectionError(request=req)
            err.__cause__ = ConnectionResetError("Connection reset by peer")
            raise err

    logs: list[str] = []
    res = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            provider=MllmProvider.OPENAI,
            model_name="gpt-6.2-astra",
            reasoning_effort=ReasoningEffort.HIGH,
            allowed_classes=["player"],
            max_workers=1,
        ),
        client_override=_TimeoutAndConnClient(),
        log_callback=logs.append,
    )

    assert res.error_count == 2
    assert any("单次超时: 1200s" in line for line in logs)
    assert any("处理出错" in line and "APITimeoutError" in line and "The read operation timed out" in line for line in logs)
    assert any("处理出错" in line and "APIConnectionError" in line and "Connection reset by peer" in line for line in logs)

    report = json.loads(res.report_path.read_text(encoding="utf-8"))
    errors_by_name = {item["image"]: item["error"] for item in report["items"]}
    assert "APITimeoutError" in errors_by_name["a_timeout.png"]
    assert "The read operation timed out" in errors_by_name["a_timeout.png"]
    assert "APIConnectionError" in errors_by_name["b_conn.png"]
    assert "Connection reset by peer" in errors_by_name["b_conn.png"]


def test_run_mllm_review_cancel_aborts_inflight_openai_socket_promptly(tmp_path: Path) -> None:
    """验证置位 `cancel_event` 时通过关闭底层客户端立即打断阻塞中的网络请求，不死等超长超时也不误报错误。"""
    import socket
    import threading
    import time

    from yolo_studio.core.mllm_reviewer import OpenAICompatMllmClient, ReasoningEffort

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    for name in ("01.png", "02.png", "03.png"):
        _write_img(in_dir / name, (160, 120))

    # 启动仅接收请求但不回包的本地 TCP 黑洞服务端，模拟深度推理长阻塞等待
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(5)

    connected_ev = threading.Event()
    accepted_conns: list[socket.socket] = []

    def _accept_loop() -> None:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            accepted_conns.append(conn)
            try:
                conn.recv(1024)
            except OSError:
                pass
            connected_ev.set()

    threading.Thread(target=_accept_loop, daemon=True).start()

    cancel_ev = threading.Event()
    client = OpenAICompatMllmClient(
        api_key="sk-test",
        base_url=f"http://127.0.0.1:{port}/v1",
        model_name="gpt-6.2-astra",
        reasoning_effort=ReasoningEffort.HIGH,
        allow_anonymous_key=False,
    )
    assert client.request_timeout == 1200.0

    def _trigger_cancel_when_connected() -> None:
        if connected_ev.wait(timeout=5.0):
            cancel_ev.set()

    threading.Thread(target=_trigger_cancel_when_connected, daemon=True).start()

    logs: list[str] = []
    t0 = time.perf_counter()
    try:
        res = run_mllm_review(
            MllmReviewConfig(
                images_dir=in_dir,
                output_dir=out_dir,
                source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
                provider=MllmProvider.OPENAI,
                model_name="gpt-6.2-astra",
                reasoning_effort=ReasoningEffort.HIGH,
                allowed_classes=["player"],
                max_workers=1,
            ),
            client_override=client,
            log_callback=logs.append,
            cancel_event=cancel_ev,
        )
    finally:
        for conn in accepted_conns:
            try:
                conn.close()
            except OSError:
                pass
        try:
            srv.close()
        except OSError:
            pass

    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0, f"取消响应耗时过长（{elapsed:.2f}s），应立即中断在途连接"
    assert res.cancelled is True
    assert res.error_count == 0
    assert any("已收到停止指令" in line for line in logs)
    assert res.report_path is not None and res.report_path.is_file()
    report = json.loads(res.report_path.read_text(encoding="utf-8"))
    assert report["cancelled"] is True


def test_run_mllm_review_cancel_exits_promptly_even_with_non_closeable_client(tmp_path: Path) -> None:
    """验证即使客户端未实现 `close()` 仍处于阻塞中，`run_mllm_review` 也能跳过剩余图片并在排空窗口后迅速退出。"""
    import threading
    import time

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    for name in ("01.png", "02.png", "03.png"):
        _write_img(in_dir / name, (160, 120))

    cancel_ev = threading.Event()
    unblock_ev = threading.Event()
    second_entered = threading.Event()
    calls: list[int] = []

    class _BlockingNoCloseClient:
        def review_image(self, *_args) -> MllmReviewResponse:
            idx = len(calls)
            calls.append(idx)
            if idx == 0:
                return MllmReviewResponse(
                    added_boxes=[MissingBox(label="player", box_2d=[100, 100, 500, 500])],
                    has_missing=False,
                )
            second_entered.set()
            unblock_ev.wait(timeout=30.0)
            return MllmReviewResponse(added_boxes=[], has_missing=False)

    def _trigger_cancel() -> None:
        if second_entered.wait(timeout=5.0):
            cancel_ev.set()

    threading.Thread(target=_trigger_cancel, daemon=True).start()

    t0 = time.perf_counter()
    try:
        res = run_mllm_review(
            MllmReviewConfig(
                images_dir=in_dir,
                output_dir=out_dir,
                source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
                allowed_classes=["player"],
                max_workers=1,
            ),
            client_override=_BlockingNoCloseClient(),
            cancel_event=cancel_ev,
        )
    finally:
        unblock_ev.set()

    elapsed = time.perf_counter() - t0
    assert elapsed < 3.5, f"无 close() 客户端取消退出耗时过长（{elapsed:.2f}s）"
    assert res.cancelled is True
    assert len(res.records) == 1
    assert res.records[0].added_count == 1
    assert calls == [0, 1]


def test_run_mllm_review_records_and_logs_per_image_elapsed_time(tmp_path: Path) -> None:
    """验证正常与失败图片均记录单图耗时并在日志与 `review_report.json` 往返中保留，同时兼容无该字段的旧报告。"""
    from yolo_studio.core.mllm_reviewer import load_review_report

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    _write_img(in_dir / "01_ok.png", (200, 120))
    _write_img(in_dir / "02_err.png", (200, 120))

    class _MixedClient:
        def __init__(self) -> None:
            self.idx = 0

        def review_image(self, *_args) -> MllmReviewResponse:
            self.idx += 1
            if self.idx == 1:
                return MllmReviewResponse(
                    added_boxes=[MissingBox(label="player", box_2d=[100, 100, 500, 500])],
                    has_missing=False,
                )
            raise RuntimeError("模拟单图失败")

    logs: list[str] = []
    res = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
        ),
        client_override=_MixedClient(),
        log_callback=logs.append,
    )

    assert len(res.records) == 2
    for rec in res.records:
        assert rec.elapsed_sec is not None and rec.elapsed_sec >= 0.0

    item_logs = [line for line in logs if line.startswith("[1/2]") or line.startswith("[2/2]")]
    assert len(item_logs) == 2
    assert all("耗时:" in line and line.endswith("s)") or "耗时:" in line for line in item_logs)
    assert any("01_ok.png" in line and "耗时:" in line for line in item_logs)
    assert any("02_err.png" in line and "处理出错" in line and "耗时:" in line for line in item_logs)

    loaded = load_review_report(out_dir)
    assert all(r.elapsed_sec is not None and r.elapsed_sec >= 0.0 for r in loaded.records)

    # 兼容不含 elapsed_sec 的旧版报告
    assert res.report_path is not None
    raw_report = json.loads(res.report_path.read_text(encoding="utf-8"))
    for item in raw_report["items"]:
        item.pop("elapsed_sec", None)
    res.report_path.write_text(json.dumps(raw_report, ensure_ascii=False, indent=2), encoding="utf-8")
    legacy_loaded = load_review_report(out_dir)
    assert all(r.elapsed_sec is None for r in legacy_loaded.records)


def test_timestamped_log_formatting_and_all_tabs_emit_timestamps(tmp_path: Path) -> None:
    """验证统一时间戳格式化函数及 4 个带日志标签页输出均以 `[YYYY-MM-DD HH:MM:SS]` 开头。"""
    import os
    import re
    from datetime import datetime

    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication, QPlainTextEdit, QTextEdit

    from yolo_studio.core.trainer import DeviceInfo
    from yolo_studio.gui.common import append_timestamped_log, format_timestamped_log
    from yolo_studio.gui.tab_dataset import DatasetTab
    from yolo_studio.gui.tab_eval import EvalTab
    from yolo_studio.gui.tab_review import MllmReviewTab
    from yolo_studio.gui.tab_train import TrainTab

    fixed_dt = datetime(2026, 11, 25, 19, 45, 30)
    assert format_timestamped_log("", now=fixed_dt) == ""
    assert format_timestamped_log("单行日志", now=fixed_dt) == "[2026-11-25 19:45:30] 单行日志"
    assert (
        format_timestamped_log("\n=== 阶段标题 ===\n  \n内容行", now=fixed_dt)
        == "\n[2026-11-25 19:45:30] === 阶段标题 ===\n\n[2026-11-25 19:45:30] 内容行"
    )
    already = "[2026-01-01 00:00:00] 已有时间戳"
    assert format_timestamped_log(already, now=fixed_dt) == already

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication([])

    plain = QPlainTextEdit()
    rich = QTextEdit()
    append_timestamped_log(plain, "plain 行", now=fixed_dt)
    append_timestamped_log(rich, "rich 行", now=fixed_dt)
    assert plain.toPlainText().strip() == "[2026-11-25 19:45:30] plain 行"
    assert rich.toPlainText().strip() == "[2026-11-25 19:45:30] rich 行"

    cpu_info = DeviceInfo(
        cuda_available=False,
        device_count=0,
        device_name="Mock CPU",
        vram_gb=0.0,
        recommended_device="cpu",
        probing=False,
        gpu_devices=(),
    )
    ts_line_re = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] .+")

    review_tab = MllmReviewTab(tmp_path, device_info=cpu_info)
    review_tab._append_log("自动标注测试行")
    assert ts_line_re.match(review_tab.log_box.toPlainText().strip())

    dataset_tab = DatasetTab(tmp_path)
    dataset_tab._append_log("数据集测试行")
    assert ts_line_re.match(dataset_tab.log_box.toPlainText().strip())

    train_tab = TrainTab(tmp_path, device_info=cpu_info)
    train_tab._on_log_line("训练测试行")
    assert ts_line_re.match(train_tab.log_box.toPlainText().strip())

    eval_tab = EvalTab(tmp_path, device_info=cpu_info)
    eval_tab._append_log("评估导出测试行")
    assert ts_line_re.match(eval_tab.log_box.toPlainText().strip())


def test_transient_disconnect_classifier_excludes_timeouts_and_status_errors() -> None:
    """验证瞬时断连识别函数精确命中连接/协议断开，并严格排除超时与 HTTP 状态错误。"""
    import httpx
    import httpx2
    import openai

    from yolo_studio.core.mllm_reviewer import _is_transient_disconnect_error

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    # 1) 真实连接/协议断开返回 True
    conn_err = openai.APIConnectionError(request=req)
    conn_err.__cause__ = httpx2.RemoteProtocolError("Server disconnected without sending a response.")
    assert _is_transient_disconnect_error(conn_err) is True
    assert _is_transient_disconnect_error(ConnectionResetError("reset")) is True
    assert _is_transient_disconnect_error(httpx2.ReadError("read failed")) is True

    # 2) APITimeoutError 虽然继承自 APIConnectionError，但必须被识别为超时而不触发重试
    timeout_err = openai.APITimeoutError(request=req)
    timeout_err.__cause__ = TimeoutError("timed out")
    assert _is_transient_disconnect_error(timeout_err) is False
    assert _is_transient_disconnect_error(TimeoutError("plain timeout")) is False
    assert _is_transient_disconnect_error(httpx2.ReadTimeout("read timeout")) is False

    # 3) HTTP 4xx/5xx 与普通业务异常均返回 False
    bad_req = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    auth_err = openai.AuthenticationError("denied", response=httpx.Response(401, request=req), body=None)
    assert _is_transient_disconnect_error(bad_req) is False
    assert _is_transient_disconnect_error(auth_err) is False
    assert _is_transient_disconnect_error(RuntimeError("boom")) is False


def test_gemini_streaming_accumulation_and_short_disconnect_reconnect_retry(monkeypatch) -> None:
    """验证 Gemini 流式文本拼接、`stream` 400 自动降级、短耗时断连新建客户端重连与超时不重试。"""
    import httpx2
    import pytest
    from google import genai

    from yolo_studio.core.mllm_reviewer import GeminiMllmClient, ReasoningEffort

    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    def _make_stream(chunks: list[str], *, closed_flag: list[bool] | None = None):
        events = [
            type("Ev", (), {"event_type": "interaction.created"})(),
            type(
                "Ev",
                (),
                {
                    "event_type": "step.delta",
                    "delta": type("D", (), {"type": "thought_signature", "text": None})(),
                },
            )(),
        ]
        for piece in chunks:
            events.append(
                type(
                    "Ev",
                    (),
                    {
                        "event_type": "step.delta",
                        "delta": type("D", (), {"type": "text", "text": piece})(),
                    },
                )()
            )
        events.append(
            type(
                "Ev",
                (),
                {
                    "event_type": "interaction.completed",
                    "interaction": type("I", (), {"status": "completed", "output_text": None})(),
                },
            )()
        )

        class _FakeStream:
            def __iter__(self):
                return iter(events)

            def close(self) -> None:
                if closed_flag is not None:
                    closed_flag.append(True)

        return _FakeStream()

    # 1) 默认传 stream=True，多段 delta 文本正确拼接且流被自动关闭
    created_clients: list[object] = []
    calls_log: list[dict[str, object]] = []
    stream_closed: list[bool] = []

    class _StreamOkInteractions:
        def __init__(self, cid: int) -> None:
            self.cid = cid

        def create(self, **kwargs):
            calls_log.append({"cid": self.cid, **kwargs})
            return _make_stream([ok_json[:24], ok_json[24:]], closed_flag=stream_closed)

    class _FakeClient:
        def __init__(self, **_kwargs) -> None:
            cid = len(created_clients)
            self.cid = cid
            self.closed = False
            self.interactions = _StreamOkInteractions(cid)
            created_clients.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(genai, "Client", _FakeClient)
    g_cli = GeminiMllmClient(api_key="k", model_name="gemini-4-neon", reasoning_effort=ReasoningEffort.HIGH)
    parsed = g_cli.review_image(b"x", [], "s", "u")
    assert parsed.has_missing is False
    assert calls_log[0].get("stream") is True
    assert stream_closed == [True]

    # 2) 短耗时 RemoteProtocolError 触发新建 SDK 客户端并最终成功，新建临时客户端用完即关
    created_clients.clear()
    calls_log.clear()
    mode = {"behavior": "disconnect_once"}

    class _RetryInteractions:
        def __init__(self, cid: int) -> None:
            self.cid = cid

        def create(self, **kwargs):
            calls_log.append({"cid": self.cid, **kwargs})
            if mode["behavior"] == "disconnect_once" and len(calls_log) == 1:
                raise httpx2.RemoteProtocolError("Server disconnected without sending a response.")
            if mode["behavior"] == "reject_stream" and kwargs.get("stream"):
                err = RuntimeError("400 INVALID_ARGUMENT: stream is not supported")
                setattr(err, "code", 400)
                raise err
            if mode["behavior"] == "non_stream_ok":
                return type("Resp", (), {"output_text": ok_json})()
            return _make_stream([ok_json])

    class _FakeRetryClient:
        def __init__(self, **_kwargs) -> None:
            cid = len(created_clients)
            self.cid = cid
            self.closed = False
            self.interactions = _RetryInteractions(cid)
            created_clients.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(genai, "Client", _FakeRetryClient)
    g_retry = GeminiMllmClient(
        api_key="k",
        model_name="gemini-4-neon",
        transient_disconnect_retries=2,
        transient_retry_backoff_base=0.0,
    )
    g_retry.review_image(b"x", [], "s", "u")
    assert len(calls_log) == 2
    assert calls_log[0]["cid"] == 0
    assert calls_log[1]["cid"] == 1
    assert created_clients[1].closed is True

    # 3) 400 明确拒绝 stream 时降级为非流式并记住结论
    calls_log.clear()
    mode["behavior"] = "reject_stream"
    g_retry.review_image(b"x", [], "s", "u")
    mode["behavior"] = "non_stream_ok"
    g_retry.review_image(b"x", [], "s", "u")
    assert [bool(c.get("stream")) for c in calls_log] == [True, False, False]

    # 4) 超过耗时阈值的长请求断连不重试，避免重复扣费
    calls_log.clear()
    mode["behavior"] = "disconnect_once"
    g_long = GeminiMllmClient(
        api_key="k",
        model_name="gemini-4-neon",
        transient_disconnect_retries=2,
        transient_disconnect_max_elapsed=0.0,
        transient_retry_backoff_base=0.0,
    )
    with pytest.raises(httpx2.RemoteProtocolError):
        g_long.review_image(b"x", [], "s", "u")
    assert len(calls_log) == 1


def test_openai_streaming_accumulation_and_short_disconnect_reconnect_retry(monkeypatch) -> None:
    """验证 OpenAI 流式分片拼接、`stream` 400 自动降级、短耗时断连重连、超时不重试及 `close()` 秒断退避。"""
    import httpx
    import httpx2
    import openai
    import pytest

    from yolo_studio.core.mllm_reviewer import OpenAICompatMllmClient, ReasoningEffort

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    def _make_oa_stream(parts: list[str], *, closed_flag: list[bool] | None = None):
        chunks = [
            type(
                "Chunk",
                (),
                {"choices": [type("C", (), {"delta": type("D", (), {"content": p})()})()]},
            )()
            for p in parts
        ]
        # 附加一个 choices=[] 的 usage 尾包验证空分片安全跳过
        chunks.append(type("Chunk", (), {"choices": []})())

        class _FakeStream:
            def __iter__(self):
                return iter(chunks)

            def close(self) -> None:
                if closed_flag is not None:
                    closed_flag.append(True)

        return _FakeStream()

    created_clients: list[object] = []
    calls_log: list[dict[str, object]] = []
    stream_closed: list[bool] = []
    mode = {"behavior": "stream_ok"}

    class _Completions:
        def __init__(self, cid: int) -> None:
            self.cid = cid

        def create(self, **kwargs):
            calls_log.append({"cid": self.cid, **kwargs})
            if mode["behavior"] == "disconnect_once" and len(calls_log) == 1:
                err = openai.APIConnectionError(request=req)
                err.__cause__ = httpx2.RemoteProtocolError("Server disconnected without sending a response.")
                raise err
            if mode["behavior"] == "timeout":
                err = openai.APITimeoutError(request=req)
                err.__cause__ = TimeoutError("read timed out")
                raise err
            if mode["behavior"] == "reject_stream" and kwargs.get("stream"):
                raise openai.BadRequestError(
                    "Unsupported parameter: stream",
                    response=httpx.Response(400, request=req),
                    body={"error": {"param": "stream", "message": "stream not supported"}},
                )
            if not kwargs.get("stream"):
                return type(
                    "Resp",
                    (),
                    {"choices": [type("C", (), {"message": type("M", (), {"content": ok_json})()})()]},
                )()
            return _make_oa_stream([ok_json[:20], ok_json[20:]], closed_flag=stream_closed)

    class _FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            cid = len(created_clients)
            self.cid = cid
            self.closed = False
            self.chat = type("Chat", (), {"completions": _Completions(cid)})()
            created_clients.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    # 1) 默认流式拼接并关闭流对象
    cli = OpenAICompatMllmClient(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model_name="gpt-6.2-astra",
        reasoning_effort=ReasoningEffort.HIGH,
        allow_anonymous_key=False,
        transient_disconnect_retries=2,
        transient_retry_backoff_base=0.0,
    )
    parsed = cli.review_image(b"x", [], "s", "u")
    assert parsed.has_missing is False
    assert calls_log[0].get("stream") is True
    assert stream_closed == [True]

    # 2) 短耗时 APIConnectionError 触发新建客户端实例重连并成功
    calls_log.clear()
    mode["behavior"] = "disconnect_once"
    cli.review_image(b"x", [], "s", "u")
    assert len(calls_log) == 2
    assert calls_log[0]["cid"] == 0
    assert calls_log[1]["cid"] == 1
    assert created_clients[1].closed is True

    # 3) 400 拒绝 stream 时降级并保留 json_object
    calls_log.clear()
    mode["behavior"] = "reject_stream"
    cli.review_image(b"x", [], "s", "u")
    cli.review_image(b"x", [], "s", "u")
    assert [bool(c.get("stream")) for c in calls_log] == [True, False, False]
    assert all(c.get("response_format") == {"type": "json_object"} for c in calls_log)

    # 4) APITimeoutError 不触发瞬时断连重试
    calls_log.clear()
    mode["behavior"] = "timeout"
    with pytest.raises(openai.APITimeoutError):
        cli.review_image(b"x", [], "s", "u")
    assert len(calls_log) == 1

    # 5) 超过 max_elapsed 时不重试；close() 置位后立即中断退避等待
    calls_log.clear()
    mode["behavior"] = "disconnect_once"
    cli_long = OpenAICompatMllmClient(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model_name="gpt-6.2-astra",
        allow_anonymous_key=False,
        transient_disconnect_retries=2,
        transient_disconnect_max_elapsed=0.0,
        transient_retry_backoff_base=0.0,
    )
    with pytest.raises(openai.APIConnectionError):
        cli_long.review_image(b"x", [], "s", "u")
    assert len(calls_log) == 1

    import threading
    import time

    calls_log.clear()
    cli_cancel = OpenAICompatMllmClient(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model_name="gpt-6.2-astra",
        allow_anonymous_key=False,
        transient_disconnect_retries=2,
        transient_retry_backoff_base=10.0,
    )
    threading.Timer(0.15, cli_cancel.close).start()
    t0 = time.perf_counter()
    with pytest.raises(openai.APIConnectionError):
        cli_cancel.review_image(b"x", [], "s", "u")
    assert time.perf_counter() - t0 < 2.0
    assert len(calls_log) == 1


def test_rate_limit_and_server_error_backoff_retry_and_shared_cooldown(monkeypatch) -> None:
    """验证 429/5xx 分类、Retry-After 尊重、硬配额 0 重试、应用层退避重试及共享冷却与关闭打断。"""
    import threading
    import time

    import httpx
    import openai
    import pytest
    from google import genai

    from yolo_studio.core.mllm_reviewer import (
        GeminiMllmClient,
        OpenAICompatMllmClient,
        ReasoningEffort,
        _classify_circuit_breaker_error,
        _extract_http_status_code,
        _extract_retry_after_seconds,
        _is_hard_quota_or_billing_exhausted,
        _is_retryable_http_error,
    )

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    ok_json = '{"box_reviews": [], "has_missing": false, "missing_note": "", "added_boxes": []}'

    # 1) 状态码提取、Retry-After 解析与错误归类边界
    rl_err = openai.RateLimitError(
        "Rate limit reached",
        response=httpx.Response(429, headers={"Retry-After": "1.5"}, request=req),
        body={"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached"}},
    )
    rl_ms_err = openai.RateLimitError(
        "Rate limit reached",
        response=httpx.Response(429, headers={"retry-after-ms": "250"}, request=req),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    quota_err = openai.RateLimitError(
        "You exceeded your current quota, please check your plan and billing details.",
        response=httpx.Response(429, request=req),
        body={"error": {"code": "insufficient_quota", "type": "insufficient_quota"}},
    )
    srv_err = openai.InternalServerError(
        "Service Unavailable",
        response=httpx.Response(503, request=req),
        body={"error": {"message": "overloaded"}},
    )
    auth_err = openai.AuthenticationError(
        "Invalid API key",
        response=httpx.Response(401, request=req),
        body={"error": {"code": "invalid_api_key"}},
    )
    not_found_err = openai.NotFoundError(
        "Model not found",
        response=httpx.Response(404, request=req),
        body={"error": {"code": "model_not_found"}},
    )

    assert _extract_http_status_code(rl_err) == 429
    assert _extract_retry_after_seconds(rl_err) == pytest.approx(1.5)
    assert _extract_retry_after_seconds(rl_ms_err) == pytest.approx(0.25)
    assert _is_retryable_http_error(rl_err) is True
    assert _is_retryable_http_error(srv_err) is True
    assert _is_hard_quota_or_billing_exhausted(quota_err) is True
    assert _is_retryable_http_error(quota_err) is False
    assert _is_retryable_http_error(auth_err) is False

    assert _classify_circuit_breaker_error(rl_err)[0] == "transient"
    assert _classify_circuit_breaker_error(srv_err)[0] == "transient"
    assert _classify_circuit_breaker_error(quota_err)[0] == "fatal"
    assert _classify_circuit_breaker_error(auth_err)[0] == "fatal"
    assert _classify_circuit_breaker_error(not_found_err)[0] == "fatal"
    assert _classify_circuit_breaker_error(RuntimeError("局部 JSON 解析失败"))[0] == "other"

    # 2) OpenAI 在 HIGH（SDK max_retries=0）下遇 429 + 503 仍在应用层重试成功，并写入共享冷却；
    #    遇硬配额 429 或 401 时 0 重试立即抛出
    oa_calls: list[dict[str, object]] = []
    oa_script: list[object] = []

    class _OaCompletions:
        def create(self, **kwargs):
            oa_calls.append(dict(kwargs))
            action = oa_script.pop(0)
            if isinstance(action, Exception):
                raise action
            return type(
                "Resp",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": ok_json})()})()]},
            )()

    class _FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = type("Chat", (), {"completions": _OaCompletions()})()

        def close(self) -> None:
            pass

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    oa_cli = OpenAICompatMllmClient(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model_name="gpt-6.2-astra",
        reasoning_effort=ReasoningEffort.HIGH,
        allow_anonymous_key=False,
        rate_limit_retries=4,
        rate_limit_backoff_base=0.0,
        rate_limit_backoff_max=5.0,
    )
    assert oa_cli.max_retries == 0
    oa_cli._stream_supported = False

    oa_script.extend(
        [
            openai.RateLimitError(
                "429",
                response=httpx.Response(429, headers={"Retry-After": "0.05"}, request=req),
                body={"error": {"code": "rate_limit_exceeded"}},
            ),
            srv_err,
            "ok",
        ]
    )
    before_cd = time.monotonic()
    parsed_oa = oa_cli.review_image(b"x", [], "s", "u")
    assert parsed_oa.has_missing is False
    assert len(oa_calls) == 3
    assert oa_cli._cooldown_until >= before_cd

    # 硬配额 429 与 401 不重试
    oa_calls.clear()
    oa_script.append(quota_err)
    with pytest.raises(openai.RateLimitError):
        oa_cli.review_image(b"x", [], "s", "u")
    assert len(oa_calls) == 1

    oa_calls.clear()
    oa_script.append(auth_err)
    with pytest.raises(openai.AuthenticationError):
        oa_cli.review_image(b"x", [], "s", "u")
    assert len(oa_calls) == 1

    # 共享冷却可被 close() 立即打断，且不会发出新的 HTTP 请求
    oa_cli._register_shared_cooldown(10.0)
    oa_calls.clear()
    threading.Timer(0.12, oa_cli.close).start()
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="请求已取消"):
        oa_cli.review_image(b"x", [], "s", "u")
    assert time.perf_counter() - t0 < 2.0
    assert len(oa_calls) == 0

    # 3) Gemini 在 HIGH 下遇 429 RESOURCE_EXHAUSTED / 503 UNAVAILABLE 同样在应用层重试成功，遇 403/硬配额不重试
    g_calls: list[dict[str, object]] = []
    g_script: list[object] = []

    class _FakeGenaiClient:
        def __init__(self, **_kwargs) -> None:
            self.interactions = self

        def create(self, **kwargs):
            g_calls.append(dict(kwargs))
            action = g_script.pop(0)
            if isinstance(action, Exception):
                raise action
            return type("Resp", (), {"output_text": ok_json})()

        def close(self) -> None:
            pass

    monkeypatch.setattr(genai, "Client", _FakeGenaiClient)
    g_cli = GeminiMllmClient(
        api_key="k",
        model_name="gemini-4-neon",
        reasoning_effort=ReasoningEffort.HIGH,
        rate_limit_retries=3,
        rate_limit_backoff_base=0.0,
    )
    g_cli._stream_supported = False

    g_429 = RuntimeError("429 RESOURCE_EXHAUSTED: rate limit exceeded")
    setattr(g_429, "code", 429)
    g_503 = RuntimeError("503 UNAVAILABLE: model overloaded")
    setattr(g_503, "code", 503)
    g_403 = RuntimeError("403 PERMISSION_DENIED: api key invalid")
    setattr(g_403, "code", 403)

    g_script.extend([g_429, g_503, "ok"])
    parsed_g = g_cli.review_image(b"x", [], "s", "u")
    assert parsed_g.has_missing is False
    assert len(g_calls) == 3

    g_calls.clear()
    g_script.append(g_403)
    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        g_cli.review_image(b"x", [], "s", "u")
    assert len(g_calls) == 1


def test_run_mllm_review_resume_skips_succeeded_and_retries_errors_with_atomic_report(
    tmp_path: Path,
) -> None:
    """验证实时原子落盘无残留临时文件，以及断点续传复用已成功记录、仅补跑失败与未处理图片。"""
    from threading import Event

    from yolo_studio.core.mllm_reviewer import load_review_report

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    img_names = ["01_ok.jpg", "02_ok.jpg", "03_err.jpg", "04_pending.jpg"]
    for name in img_names:
        _write_img(in_dir / name, (320, 180))

    # 第 1 轮：01/02 成功，03 局部报错，第 3 张完成后触发取消跳过 04；验证实时落盘与无残留 .tmp
    cancel_ev = Event()
    round1_calls: list[int] = []

    class _Round1Client:
        def review_image(self, *_args) -> MllmReviewResponse:
            idx = len(round1_calls)
            round1_calls.append(idx)
            if idx == 2:
                raise RuntimeError("模拟单图返回损坏 JSON")
            return MllmReviewResponse(
                box_reviews=[],
                has_missing=False,
                added_boxes=[MissingBox(label="player", box_2d=[100, 100, 500, 500], reason="ok")],
            )

    progress_r1: list[tuple[int, int, str]] = []

    def _on_prog_r1(cur: int, total: int, rec) -> None:
        progress_r1.append((cur, total, rec.image_path.name))
        if cur == 3:
            cancel_ev.set()

    res1 = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
            resume_from_existing=True,
        ),
        client_override=_Round1Client(),
        progress_callback=_on_prog_r1,
        cancel_event=cancel_ev,
    )
    assert res1.cancelled is True
    assert res1.total_images == 3
    assert res1.error_count == 1
    assert res1.resumed_skipped_count == 0
    assert len(round1_calls) == 3
    assert res1.report_path is not None and res1.report_path.is_file()
    assert list(out_dir.glob("*.tmp")) == []

    # 第 2 轮启用断点续传：跳过 01/02，仅处理 03（上次报错）和 04（上次未跑）
    round2_calls: list[int] = []

    class _AlwaysOkClient:
        def __init__(self, counter: list[int]) -> None:
            self._counter = counter

        def review_image(self, *_args) -> MllmReviewResponse:
            self._counter.append(len(self._counter))
            return MllmReviewResponse(
                box_reviews=[],
                has_missing=False,
                added_boxes=[MissingBox(label="player", box_2d=[120, 120, 480, 480], reason="resumed")],
            )

    progress_r2: list[tuple[int, int, str]] = []
    res2 = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
            resume_from_existing=True,
        ),
        client_override=_AlwaysOkClient(round2_calls),
        progress_callback=lambda c, t, r: progress_r2.append((c, t, r.image_path.name)),
    )
    assert res2.cancelled is False
    assert res2.total_images == 4
    assert res2.error_count == 0
    assert res2.resumed_skipped_count == 2
    assert len(round2_calls) == 2
    assert len(progress_r2) == 4
    assert [name for _, _, name in progress_r2[:2]] == ["01_ok.jpg", "02_ok.jpg"]

    loaded_r2 = load_review_report(res2.report_path)
    assert loaded_r2.resumed_skipped_count == 2
    assert loaded_r2.aborted_reason == ""

    # 第 3 轮再次续传：全部 4 张已成功且输出完整 -> 0 次模型调用
    round3_calls: list[int] = []
    res3 = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
            resume_from_existing=True,
        ),
        client_override=_AlwaysOkClient(round3_calls),
    )
    assert res3.resumed_skipped_count == 4
    assert len(round3_calls) == 0

    # 删除其中 1 张输出图后再续传：仅自动补跑缺失的那 1 张
    rec_01 = next(r for r in res3.records if r.image_path.name == "01_ok.jpg")
    assert rec_01.output_image_path is not None
    rec_01.output_image_path.unlink()
    round4_calls: list[int] = []
    res4 = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_dir,
            output_dir=out_dir,
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
            resume_from_existing=True,
        ),
        client_override=_AlwaysOkClient(round4_calls),
    )
    assert res4.resumed_skipped_count == 3
    assert len(round4_calls) == 1


def test_run_mllm_review_circuit_breaker_on_fatal_and_consecutive_transient_errors(
    tmp_path: Path,
) -> None:
    """验证致命错误立即熔断、连续系统性故障自动熔断、成功清零计数、局部错误不误触发及 CLI 参数解析。"""
    import httpx
    import openai

    from yolo_studio.cli import build_parser
    from yolo_studio.core.mllm_reviewer import load_review_report

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    # 场景 A：6 张图第 1 张遇到 401 鉴权失败 / 429 硬配额耗尽 -> 立即熔断，仅调用 1 次并落盘 aborted_reason
    in_fatal = tmp_path / "in_fatal"
    for i in range(6):
        _write_img(in_fatal / f"img_{i:02d}.jpg", (240, 160))

    for tag, fatal_exc in (
        (
            "auth401",
            openai.AuthenticationError(
                "Invalid API key",
                response=httpx.Response(401, request=req),
                body={"error": {"code": "invalid_api_key"}},
            ),
        ),
        (
            "quota429",
            openai.RateLimitError(
                "Exceeded quota",
                response=httpx.Response(429, request=req),
                body={"error": {"code": "insufficient_quota"}},
            ),
        ),
    ):
        call_count = 0

        class _FatalClient:
            def __init__(self, exc: Exception) -> None:
                self._exc = exc

            def review_image(self, *_args):
                nonlocal call_count
                call_count += 1
                raise self._exc

        logs: list[str] = []
        res_fatal = run_mllm_review(
            MllmReviewConfig(
                images_dir=in_fatal,
                output_dir=tmp_path / f"out_{tag}",
                source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
                allowed_classes=["player"],
                max_workers=1,
            ),
            client_override=_FatalClient(fatal_exc),
            log_callback=logs.append,
        )
        assert call_count == 1
        assert res_fatal.cancelled is True
        assert res_fatal.total_images == 1
        assert res_fatal.error_count == 1
        assert res_fatal.aborted_reason != ""
        assert any("[熔断停机]" in line for line in logs)
        reloaded = load_review_report(res_fatal.report_path)
        assert reloaded.aborted_reason == res_fatal.aborted_reason

    # 场景 B：8 张图，max_consecutive_errors=3；前 2 次网络断连 -> 第 3 张成功（清零）-> 第 4~6 张连续 3 次断连触发熔断，第 7~8 张不调用
    in_trans = tmp_path / "in_trans"
    for i in range(8):
        _write_img(in_trans / f"t_{i:02d}.jpg", (240, 160))

    trans_calls = 0

    class _FlakyClient:
        def review_image(self, *_args) -> MllmReviewResponse:
            nonlocal trans_calls
            idx = trans_calls
            trans_calls += 1
            if idx == 2:
                return MllmReviewResponse(box_reviews=[], has_missing=False, added_boxes=[])
            raise openai.APIConnectionError(request=req)

    res_trans = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_trans,
            output_dir=tmp_path / "out_trans",
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
            max_consecutive_errors=3,
        ),
        client_override=_FlakyClient(),
    )
    assert trans_calls == 6
    assert res_trans.cancelled is True
    assert res_trans.total_images == 6
    assert res_trans.error_count == 5
    assert "连续 3 张" in res_trans.aborted_reason

    # 场景 C：连续 4 张普通 RuntimeError（单图局部错误）不应误触发全局熔断
    in_local = tmp_path / "in_local"
    for i in range(4):
        _write_img(in_local / f"loc_{i:02d}.jpg", (240, 160))

    local_calls = 0

    class _LocalErrClient:
        def review_image(self, *_args):
            nonlocal local_calls
            local_calls += 1
            raise RuntimeError("模型输出缺少有效 JSON")

    res_local = run_mllm_review(
        MllmReviewConfig(
            images_dir=in_local,
            output_dir=tmp_path / "out_local",
            source_mode=ReviewSourceMode.LLM_ZERO_SHOT,
            allowed_classes=["player"],
            max_workers=1,
            max_consecutive_errors=2,
        ),
        client_override=_LocalErrClient(),
    )
    assert local_calls == 4
    assert res_local.cancelled is False
    assert res_local.aborted_reason == ""

    # 场景 D：CLI 参数 --resume / --no-resume 与 --max-consecutive-errors 解析验证
    parser = build_parser()
    args_default = parser.parse_args(["review", "-i", "in_dir"])
    assert args_default.resume is False
    assert args_default.max_consecutive_errors == 5
    args_custom = parser.parse_args(["review", "-i", "in_dir", "--resume", "--max-consecutive-errors", "3"])
    assert args_custom.resume is True
    assert args_custom.max_consecutive_errors == 3
    args_off = parser.parse_args(["review", "-i", "in_dir", "--no-resume", "--max-consecutive-errors", "0"])
    assert args_off.resume is False
    assert args_off.max_consecutive_errors == 0


def test_sam_point_prompt_mask_and_refiner_shape() -> None:
    """验证本地 SAM 点选掩码孤立噪点连通域过滤及单目标多点 `(1, K, 2)` 提示张量约定。"""
    import threading
    import numpy as np

    from yolo_studio.core.mllm_reviewer import (
        UltralyticsSamBoxRefiner,
        _mask_to_point_prompt_xyxy,
    )

    # 1) 二值掩码含主目标块 [40:100, 60:140] 与远处孤立噪点 [5:12, 5:12]，前景点落在主目标上时应自动过滤噪点
    mask = np.zeros((200, 300), dtype=bool)
    mask[40:100, 60:140] = True
    mask[5:12, 5:12] = True
    xyxy = _mask_to_point_prompt_xyxy(mask, points=[(100.0, 70.0), (10.0, 10.0)], labels=[1, 0], img_w=300, img_h=200)
    assert xyxy == (60.0, 40.0, 140.0, 100.0)

    # 2) 空掩码返回 None
    assert _mask_to_point_prompt_xyxy(np.zeros((200, 300), dtype=bool), [(100.0, 70.0)], [1], 300, 200) is None

    # 3) UltralyticsSamBoxRefiner.segment_by_points 按单目标多点 3D/2D 形状传给底层 SAM.predict
    predict_calls: list[dict[str, object]] = []

    class _FakeMaskTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return self._arr

    class _FakeSamModel:
        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            m = np.zeros((1, 200, 300), dtype=np.float32)
            m[0, 50:120, 80:160] = 1.0
            return [type("Res", (), {"masks": type("M", (), {"data": _FakeMaskTensor(m)})()})()]

    refiner = object.__new__(UltralyticsSamBoxRefiner)
    refiner.device = "cpu"
    refiner._lock = threading.Lock()
    refiner._model = _FakeSamModel()

    img_bgr = np.zeros((200, 300, 3), dtype=np.uint8)
    # 缺少前景点直接返回 None，不触发推理
    assert refiner.segment_by_points(img_bgr, [(100.0, 80.0)], [0]) is None
    assert len(predict_calls) == 0

    out = refiner.segment_by_points(img_bgr, [(110.0, 80.0), (170.0, 80.0)], [1, 0])
    assert out == (80.0, 50.0, 160.0, 120.0)
    assert len(predict_calls) == 1
    assert predict_calls[0]["points"] == [[[110.0, 80.0], [170.0, 80.0]]]
    assert predict_calls[0]["labels"] == [[1, 0]]


def test_fix_single_box_with_mllm_relabel_refine_delete_and_keep(tmp_path: Path) -> None:
    """验证单框局部 AI 精修仅上传 1 张外扩局部图、透传业务规则，并正确处理改类/收框/扩框/删除与保留。"""
    import numpy as np
    import pytest

    from yolo_studio.core.mllm_reviewer import (
        BoxDecision,
        CandidateBox,
        MllmReviewConfig,
        MllmReviewResponse,
        fix_single_box_with_mllm,
    )

    img_bgr = np.full((600, 800, 3), 120, dtype=np.uint8)
    orig_box = CandidateBox(box_id=0, label="monster", x1=200.0, y1=150.0, x2=300.0, y2=250.0)
    cfg = MllmReviewConfig(
        images_dir=tmp_path,
        output_dir=tmp_path / "out",
        allowed_classes=["player", "monster", "loot"],
        custom_guideline="只框主体，不含武器及特效",
    )

    seen_calls: list[tuple[bytes, list, str, str]] = []
    scripted: list[MllmReviewResponse] = []

    class _StubClient:
        def review_image(self, overview_jpeg, crops, system_prompt, user_prompt):
            seen_calls.append((overview_jpeg, list(crops), system_prompt, user_prompt))
            return scripted.pop(0)

    # 1) 改类 + 缩框：外扩 45%（mx=max(45,48)=48）后裁剪区为 [152, 102, 348, 298]，尺寸 196×196
    scripted.append(
        MllmReviewResponse(
            box_reviews=[
                BoxDecision(
                    box_id=0,
                    action="relabel",
                    new_label="player",
                    box_quality="loose",
                    refined_box_2d=[250, 250, 750, 750],
                    reason="主体为玩家且去除了右侧武器",
                )
            ]
        )
    )
    res1 = fix_single_box_with_mllm(img_bgr, orig_box, cfg, client_override=_StubClient())
    assert len(seen_calls) == 1
    assert seen_calls[0][1] == []  # 仅发 1 张局部图，无额外 crops 爆炸
    assert "只框主体，不含武器及特效" in seen_calls[0][2]
    assert res1.action == "relabel"
    assert res1.label == "player"
    assert res1.relabeled is True
    assert res1.refined is True
    assert res1.x1 == pytest.approx(152.0 + 0.25 * 196.0, abs=1.0)
    assert res1.x2 == pytest.approx(152.0 + 0.75 * 196.0, abs=1.0)
    assert "武器" in res1.reason

    # 2) 截断偏小框向外扩全（测试能超出原框范围补全主体）
    scripted.append(
        MllmReviewResponse(
            box_reviews=[
                BoxDecision(
                    box_id=0,
                    action="keep",
                    box_quality="partial",
                    refined_box_2d=[100, 100, 900, 900],
                    reason="原框截断了下半身已向外补全",
                )
            ]
        )
    )
    res2 = fix_single_box_with_mllm(img_bgr, orig_box, cfg, client_override=_StubClient())
    assert res2.action == "keep"
    assert res2.refined is True
    assert res2.x1 < orig_box.x1 and res2.x2 > orig_box.x2
    assert res2.y1 < orig_box.y1 and res2.y2 > orig_box.y2

    # 3) 判定误检删除
    scripted.append(
        MllmReviewResponse(
            box_reviews=[
                BoxDecision(box_id=0, action="delete", reason="纯技能光效非目标")
            ]
        )
    )
    res3 = fix_single_box_with_mllm(img_bgr, orig_box, cfg, client_override=_StubClient())
    assert res3.action == "delete"
    assert res3.refined is False
    assert "光效" in res3.reason

    # 4) 核验通过（原框已紧贴不改动）
    scripted.append(
        MllmReviewResponse(
            box_reviews=[
                BoxDecision(box_id=0, action="keep", box_quality="tight", refined_box_2d=None, reason="框已紧贴主体")
            ]
        )
    )
    res4 = fix_single_box_with_mllm(img_bgr, orig_box, cfg, client_override=_StubClient())
    assert res4.action == "keep"
    assert res4.refined is False
    assert res4.relabeled is False
    assert (res4.x1, res4.y1, res4.x2, res4.y2) == (200.0, 150.0, 300.0, 250.0)


def test_sam_refiner_feature_cache_hit_lru_eviction_and_clear() -> None:
    """验证 UltralyticsSamBoxRefiner 图像特征 LRU 缓存命中、同图多次点选/收框免重编码、低分框不偏移及清理。"""
    from collections import OrderedDict
    import threading

    import numpy as np
    import torch

    from yolo_studio.core.mllm_reviewer import CandidateBox, UltralyticsSamBoxRefiner

    class _FakePredictor:
        def __init__(self) -> None:
            self.encode_calls = 0
            self.decode_calls = 0
            self.reset_calls = 0
            self.non_overlap_masks = False
            self.args = type("Args", (), {"conf": 0.25})()
            self.model = type("InnerModel", (), {"mask_threshold": 0.0})()

        def setup_source(self, _image_bgr: np.ndarray) -> None:
            pass

        def preprocess(self, imgs: list[np.ndarray]) -> torch.Tensor:
            h, w = imgs[0].shape[:2]
            return torch.zeros((1, 3, h, w), dtype=torch.float32)

        def get_im_features(self, im: torch.Tensor) -> dict[str, torch.Tensor]:
            self.encode_calls += 1
            return {"image_embed": torch.full((1, 4, 8, 8), float(self.encode_calls))}

        def reset_image(self) -> None:
            self.reset_calls += 1

        def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
            return (dst_shape, src_shape, bboxes, points, labels, masks)

        def _inference_features(self, features, dst_shape, src_shape, bboxes, points, labels, _masks, multimask_output=False):
            self.decode_calls += 1
            h, w = src_shape
            if bboxes is not None:
                # 第 0 个框低置信度(0.10 <= 0.25)应被跳过，第 1 个框高置信度(0.95)应精确落到 box_id=1 不发生索引错位
                m = torch.full((len(bboxes), h, w), -1.0, dtype=torch.float32)
                scores = torch.full((len(bboxes),), 0.95, dtype=torch.float32)
                if len(bboxes) >= 1:
                    scores[0] = 0.10
                    m[0, 10:30, 10:30] = 2.0
                if len(bboxes) >= 2:
                    m[1, 60:110, 90:150] = 2.0
                return m, scores

            m = torch.full((1, h, w), -1.0, dtype=torch.float32)
            m[0, 40:100, 70:140] = 2.0
            return m, torch.tensor([0.92], dtype=torch.float32)

    fake_pred = _FakePredictor()
    refiner = object.__new__(UltralyticsSamBoxRefiner)
    refiner.model_source = "sam2.1_t.pt"
    refiner.device = "cpu"
    refiner._lock = threading.Lock()
    refiner._model = type("FakeSamWrapper", (), {"predictor": fake_pred})()
    refiner._max_cached_images = 2
    refiner._feature_cache = OrderedDict()
    refiner._cache_hits = 0
    refiner._cache_misses = 0
    refiner._last_cache_hit = False

    img_a = np.full((160, 200, 3), 30, dtype=np.uint8)
    img_b = np.full((160, 200, 3), 90, dtype=np.uint8)
    img_c = np.full((160, 200, 3), 180, dtype=np.uint8)

    # 1) 首次预编码图 A -> 编码 1 次；重复预编码图 A -> 直接命中不重复编码
    assert refiner.precompute_image(img_a, image_key="img_a") is True
    assert fake_pred.encode_calls == 1
    assert refiner.has_cached_image("img_a") is True
    assert refiner.precompute_image(img_a, image_key="img_a") is False
    assert fake_pred.encode_calls == 1

    # 2) 同图连续点选分割与多框收紧全部复用缓存特征（encode_calls 保持 1）
    pt_xyxy = refiner.segment_by_points(img_a, [(100.0, 70.0)], [1], image_key="img_a")
    assert pt_xyxy == (70.0, 40.0, 140.0, 100.0)
    assert refiner.last_cache_hit is True
    assert fake_pred.encode_calls == 1

    cands = [
        CandidateBox(box_id=0, label="a", x1=8.0, y1=8.0, x2=35.0, y2=35.0, confidence=0.5),
        CandidateBox(box_id=1, label="b", x1=85.0, y1=55.0, x2=155.0, y2=115.0, confidence=0.8),
    ]
    refined = refiner.refine_boxes(img_a, cands, image_key="img_a")
    assert refined[0] is None  # 低分掩码过滤
    assert refined[1] == (90.0, 60.0, 150.0, 110.0)  # 第 1 框坐标无错位
    assert refiner.last_cache_hit is True
    assert fake_pred.encode_calls == 1
    assert fake_pred.decode_calls == 2
    assert refiner.cache_stats()["hits"] == 3
    assert refiner.cache_stats()["misses"] == 1

    # 3) 容量=2 时顺序编码 B 与 C，最久未使用的 A 应被 LRU 淘汰
    assert refiner.precompute_image(img_b, image_key="img_b") is True
    assert refiner.precompute_image(img_c, image_key="img_c") is True
    assert fake_pred.encode_calls == 3
    assert refiner.has_cached_image("img_a") is False
    assert refiner.has_cached_image("img_b") is True
    assert refiner.has_cached_image("img_c") is True

    # 4) 未显式传 image_key 时按像素内容指纹自动命中，不同像素数组不误命中
    assert refiner.segment_by_points(img_a, [(100.0, 70.0)], [1]) == (70.0, 40.0, 140.0, 100.0)
    assert refiner.last_cache_hit is False
    assert fake_pred.encode_calls == 4
    assert refiner.segment_by_points(img_a, [(105.0, 72.0)], [1]) == (70.0, 40.0, 140.0, 100.0)
    assert refiner.last_cache_hit is True
    assert fake_pred.encode_calls == 4

    # 5) 清理缓存后释放特征条目并重置底层预测器引用
    prev_resets = fake_pred.reset_calls
    refiner.clear_image_cache()
    assert refiner.cache_stats()["cached_images"] == 0
    assert fake_pred.reset_calls > prev_resets










