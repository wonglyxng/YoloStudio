# 004 ONNX 部署元数据契约说明

本文档定义 **YoloStudio** 导出 `.onnx` 部署模型时写入 ONNX 原生 `metadata_props` 的自描述部署契约（协议版本 `yolostudio.deploy.v1`），供下游推理项目（如 `GameAssist`）在仅加载单个 `.onnx` 文件时自动获取完整的预处理、后处理与类别信息，避免额外附带外部配置文件或前后端硬编码不一致。

---

## 1. 设计原则与兼容性边界

- **单文件自描述交付**：所有预处理策略、颜色通道顺序、归一化口径、输出头格式、坐标空间与类别顺序均保存在 `.onnx` 自身的 `metadata_props` 中，交付物仍然只有唯一的 `.onnx` 文件。
- **不破坏 Ultralytics 原有字段**：导出完成后仅追加/更新 `yolostudio_deploy_meta` 及带 `deploy_*` / `preprocess_*` / `postprocess_*` 前缀的键，完整保留 Ultralytics 原生写入的 `task`、`imgsz`、`stride`、`batch`、`channels`、`names`、`end2end`、`args` 等键，100% 兼容 Ultralytics、ONNX Runtime、Netron 与 X-AnyLabeling。
- **跨语言直接解析**：提供标准 RFC 8259 UTF-8 JSON 总包（`yolostudio_deploy_meta`，类别为标准字符串数组）以及等价的扁平字符串快捷键，Python / C# / C++ / Rust 等任意 ONNX Runtime 客户端均可一步读取，无需解析 Python 单引号字典字符串。

---

## 2. 结构化 JSON 总包（推荐优先读取）

读取键名：`yolostudio_deploy_meta`

示例（960×960 静态 `end2end` ONNX + 标准居中 LetterBox）：

```json
{
  "schema": "yolostudio.deploy.v1",
  "task": "detect",
  "input": {
    "name": "images",
    "layout": "NCHW",
    "dtype": "float32",
    "batch": 1,
    "channels": 3,
    "height": 960,
    "width": 960,
    "dynamic": false,
    "stride": 32,
    "color_format": "RGB",
    "value_range": [0.0, 1.0],
    "mean": [0.0, 0.0, 0.0],
    "std": [255.0, 255.0, 255.0]
  },
  "preprocess": {
    "mode": "letterbox_center",
    "resize_mode": "letterbox",
    "padding_align": "center",
    "padding_value": 114,
    "auto_rect": false,
    "interpolation": "linear",
    "scale_up": true
  },
  "postprocess": {
    "output_name": "output0",
    "output_mode": "end2end",
    "nms_required": false,
    "box_format": "xyxy",
    "box_coord_space": "input_pixels",
    "max_det": 300,
    "embedded_conf": null,
    "embedded_iou": null
  },
  "classes": ["monster", "boss"]
}
```

### 字段语义说明

#### `input`（输入张量契约）

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `name` | `string` | 模型主输入节点名（来自 ONNX 图，通常为 `"images"`） |
| `layout` | `string` | 固定 `"NCHW"` |
| `dtype` | `string` | 输入元素数据类型（`"float32"` 或 `"float16"`，从 ONNX 图读取） |
| `batch` | `int` | 导出批大小（动态模型为参考批大小） |
| `channels` | `int` | 输入通道数（通常为 `3`） |
| `height` / `width` | `int` | 对齐步长后的导出高×宽（静态模型为强制输入分辨率；动态模型为参考分辨率） |
| `dynamic` | `bool` | 是否支持动态批大小与动态分辨率（`height`/`width` 需为 `stride` 倍数） |
| `stride` | `int` | 检测头最大步长（通常为 `32`） |
| `color_format` | `string` | 固定 `"RGB"`（OpenCV 默认读入为 BGR，须先转 RGB） |
| `value_range` | `[float, float]` | 像素值范围 `[0.0, 1.0]` |
| `mean` / `std` | `float[3]` | 标准化参数 `(pixel - mean) / std`，YOLO 标准为 `mean=[0,0,0]`、`std=[255,255,255]` |

#### `preprocess`（图像缩放与填充契约）

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| `mode` | `"letterbox_center"` \| `"letterbox_top_left"` \| `"stretch"` | 预处理模式总键名 |
| `resize_mode` | `"letterbox"` \| `"stretch"` | `letterbox`=保持原图宽高比等比缩放并补边；`stretch`=直接拉伸至目标高×宽（不保持宽高比） |
| `padding_align` | `"center"` \| `"top_left"` \| `"none"` | `center`=四周居中补边（YOLO 默认）；`top_left`=图像贴左上角、仅右下补边；`none`=`stretch` 无补边 |
| `padding_value` | `int (0..255)` \| `null` | `letterbox` 补边像素灰度值（默认 `114`，三通道同值）；`stretch` 时为 `null` |
| `auto_rect` | `bool` | 仅当 `dynamic=true` 且 `resize_mode="letterbox"` 时可为 `true`，表示允许推理端将短边仅补齐至 `stride` 的最小倍数（如 `576×960`）；静态模型与 `stretch` 恒为 `false` |
| `interpolation` | `"linear"` | 缩放插值方式（OpenCV `INTER_LINEAR` 双线性插值） |
| `scale_up` | `bool` | 是否允许将小于目标尺寸的原图等比放大（默认 `true`） |

#### `postprocess`（输出格式与坐标还原契约）

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| `output_name` | `string` | 主输出节点名（通常为 `"output0"`） |
| `output_mode` | `"raw"` \| `"end2end"` \| `"embedded"` | `raw`=原始 `one2many` 输出 `[B, 4+nc, anchors]`；`end2end`=YOLO26 原生 `one2one` 输出 `[B, max_det, 6]`；`embedded`=图内嵌入 NMS 输出 `[B, max_det, 6]` |
| `nms_required` | `bool` | 下游推理端是否需要自行执行外部 NMS（仅 `raw` 为 `true`） |
| `box_format` | `"xywh"` \| `"xyxy"` | `raw` 输出中心点宽高 `[cx, cy, w, h]`；`end2end` 与 `embedded` 输出角点 `[x1, y1, x2, y2, score, class_id]` |
| `box_coord_space` | `"input_pixels"` | 输出框坐标位于**预处理后的模型输入图像像素坐标系** |
| `max_det` | `int` \| `null` | `end2end` / `embedded` 模式最大单图输出框数；`raw` 为 `null` |
| `embedded_conf` / `embedded_iou` | `float` \| `null` | 仅 `embedded` 模式记录图内固化的过滤/去重阈值 |

#### `classes`

- 按类别 ID `0, 1, 2, ...` 升序排列的类别名称字符串数组。

---

## 3. 扁平快捷键清单（免 JSON 解析备选）

如果推理端不希望引入 JSON 解析器，可直接从 `custom_metadata_map` 读取以下扁平字符串键：

| 键名 | 示例值 |
| :--- | :--- |
| `deploy_schema` | `yolostudio.deploy.v1` |
| `preprocess_mode` | `letterbox_center` / `letterbox_top_left` / `stretch` |
| `preprocess_resize_mode` | `letterbox` / `stretch` |
| `preprocess_padding_align` | `center` / `top_left` / `none` |
| `preprocess_padding_value` | `114`（`stretch` 时为空字符串 `""`） |
| `preprocess_auto_rect` | `false` / `true` |
| `preprocess_color_format` | `RGB` |
| `preprocess_input_layout` | `NCHW` |
| `preprocess_normalize` | `0_1` |
| `preprocess_interpolation` | `linear` |
| `postprocess_output_mode` | `raw` / `end2end` / `embedded` |
| `postprocess_nms_required` | `true` / `false` |
| `postprocess_box_format` | `xywh` / `xyxy` |
| `postprocess_box_coord_space` | `input_pixels` |

---

## 4. 下游推理端（如 `GameAssist`）接入与坐标还原参考

### 读取元数据（Python ONNX Runtime 示例）

```python
import json
import onnxruntime as ort

session = ort.InferenceSession("best.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
meta_map = session.get_modelmeta().custom_metadata_map
deploy = json.loads(meta_map["yolostudio_deploy_meta"])

in_h, in_w = deploy["input"]["height"], deploy["input"]["width"]
pre = deploy["preprocess"]   # resize_mode, padding_align, padding_value, auto_rect
post = deploy["postprocess"] # output_mode, nms_required, box_format
classes = deploy["classes"]
```

### 按契约执行预处理与框坐标还原公式

设原图高宽为 `(orig_h, orig_w)`，目标网络输入高宽为 `(in_h, in_w)`：

1. **`letterbox` + `center`（推荐默认）**：
   - 缩放比 `r = min(in_h / orig_h, in_w / orig_w)`
   - 缩放后无补边尺寸 `new_w = max(round(orig_w * r), 1)`、`new_h = max(round(orig_h * r), 1)`
   - 若 `auto_rect=true`（仅动态 ONNX），取 `dw = (in_w - new_w) % stride`、`dh = (in_h - new_h) % stride`；否则 `dw = in_w - new_w`、`dh = in_h - new_h`
   - `pad_left = round(dw / 2 - 0.1)`，`pad_top = round(dh / 2 - 0.1)`，四周用 `padding_value` 补边
   - 坐标还原：`x_orig = (x_in - pad_left) / (new_w / orig_w)`，`y_orig = (y_in - pad_top) / (new_h / orig_h)`，再裁剪至 `[0, orig_w]` / `[0, orig_h]`
2. **`letterbox` + `top_left`**：
   - `r`、`new_w`、`new_h` 同上，图像贴左上角（`pad_left = 0`、`pad_top = 0`），仅在右侧和底部用 `padding_value` 补边
   - 坐标还原：`x_orig = x_in / (new_w / orig_w)`，`y_orig = y_in / (new_h / orig_h)`，再裁剪至原图范围
3. **`stretch`（直接拉伸）**：
   - 直接 `resize` 至 `(in_w, in_h)`，无补边
   - 坐标还原：`x_orig = x_in * (orig_w / in_w)`，`y_orig = y_in * (orig_h / in_h)`，再裁剪至原图范围

---

## 5. YoloStudio 导出与验证方式

- **桌面端 GUI**：在 **「6. 验证、推理与导出」→「3. 模型导出」** 选择 `ONNX` 格式后，在 **「预处理」** 行选择 `居中填充` / `左上靠边填充` / `直接拉伸` 并按需调整 `补边值 (0~255)`，点击「导出模型」即自动写入 `.onnx`。导出完成后点击「测刚导出模型」，YoloStudio 会自动读取该 ONNX 内的预处理契约并按相同预处理执行推理预览。
- **命令行 CLI**：
  ```powershell
  # 默认居中 LetterBox (补边=114)
  .venv\Scripts\python.exe cli.py export -w "runs\detect\yolo26_train\weights\best.pt" -f onnx --imgsz 960 --nms raw

  # 手动指定左上靠边填充 + 黑边 0（或直接拉伸 --preprocess stretch）
  .venv\Scripts\python.exe cli.py export -w "runs\detect\yolo26_train\weights\best.pt" -f onnx --imgsz 576,960 --nms end2end --preprocess letterbox_top_left --pad-value 0
  ```
