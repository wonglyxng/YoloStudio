"""标注目录级类别批量操作：统计类别使用情况、跨文件重命名/合并类别并自动备份原文件。

职责：
    为标注工作台「类别重命名 / 合并」提供目录级落盘能力，避免只改当前图导致同一目标在数据集中
    出现新旧两个类别名。直接在原始 `.json` 字典上替换 `shapes[*].label`，矩形与非矩形形状一并处理，
    其余字段（顶层键、形状扩展字段、`flags` 等）原样保留，做到无损改写。

运行环境与协作边界：
    纯 Python 核心模块，不依赖 PyQt6；由 `gui.tab_labeling` 在主线程同步调用。
    改写前先把原文件复制到调用方给定的备份目录并写入 `manifest.json`（记录新旧类别名与文件清单），
    单个文件采用“临时文件 + `os.replace`”原子替换；读写失败的文件计入失败清单，不中断其余文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable


@dataclass
class LabelUsage:
    """某个类别在一批 `.json` 文件中的使用情况。

    `files` 为包含该类别（任意形状类型）的标注文件路径列表，保持输入顺序；
    `shape_count` 为这些文件中该类别形状的总数。
    """

    files: list[Path] = field(default_factory=list)
    shape_count: int = 0


@dataclass
class LabelRenameResult:
    """跨文件重命名/合并类别的执行结果。

    `changed_files` 为实际改写成功的文件；`changed_shapes` 为被改名的形状总数；
    `failed` 为读写失败的 `(路径, 中文原因)`；`backup_dir` 为本次备份目录（无文件需要改写时为 `None`）。
    """

    changed_files: list[Path] = field(default_factory=list)
    changed_shapes: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)
    backup_dir: Path | None = None


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    """读取 `.json` 并返回字典；文件不存在、损坏或非字典结构时返回 `None`。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _count_label(data: dict[str, Any], label: str) -> int:
    return sum(
        1
        for shape in (data.get("shapes") or [])
        if isinstance(shape, dict) and str(shape.get("label") or "").strip() == label
    )


def scan_label_usage(json_paths: Iterable[Path], label: str) -> LabelUsage:
    """统计类别 `label` 在给定 `.json` 文件中的使用情况（只读，不修改任何文件）。

    类别名比较前去除首尾空白；不存在或无法解析的文件直接跳过。
    """
    target = label.strip()
    usage = LabelUsage()
    if not target:
        return usage
    for path in json_paths:
        data = _load_json_dict(Path(path))
        if data is None:
            continue
        cnt = _count_label(data, target)
        if cnt > 0:
            usage.files.append(Path(path))
            usage.shape_count += cnt
    return usage


def make_label_backup_dir(workspace_root: Path, *, now: datetime | None = None) -> Path:
    """返回本次类别批量改写的备份目录路径 `<workspace>/data/backups/label_rename_<时间戳>`（不创建）。

    时间戳使用本机本地时间，精确到秒；同一秒内重复调用时追加序号避免覆盖已有备份。
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = workspace_root / "data" / "backups"
    candidate = base / f"label_rename_{stamp}"
    seq = 2
    while candidate.exists():
        candidate = base / f"label_rename_{stamp}_{seq}"
        seq += 1
    return candidate


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """以 UTF-8、`indent=2` 原子写入 JSON（与 `annotation_io.save_image_annotation` 格式一致）。"""
    text = json.dumps(data, ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _backup_relative_name(path: Path, base_dir: Path | None) -> Path:
    """计算文件在备份目录中的相对路径：位于 `base_dir` 下时保留子目录结构，否则仅用文件名。"""
    if base_dir is not None:
        try:
            return path.resolve().relative_to(base_dir.resolve())
        except ValueError:
            pass
    return Path(path.name)


def rename_label_in_json_files(
    json_paths: Iterable[Path],
    old_label: str,
    new_label: str,
    *,
    backup_dir: Path | None,
    base_dir: Path | None = None,
) -> LabelRenameResult:
    """把一批 `.json` 中类别 `old_label` 的形状改名为 `new_label`，先备份再原子改写。

    参数：
        json_paths: 待处理的标注文件；不含 `old_label` 的文件不会被备份或改写。
        old_label / new_label: 新旧类别名（去除首尾空白后比较）；`new_label` 已存在于文件中时即为“合并”。
        backup_dir: 备份目录，传 `None` 表示不备份（仅供测试或调用方自行备份时使用）。
        base_dir: 计算备份相对路径的根目录，通常为当前标注目录，用于保留子目录结构。

    失败方式：
        新旧类别名为空或相同时直接返回空结果；单个文件读写失败计入 `failed` 并继续处理其余文件；
        备份失败的文件不会被改写，保证每个被改写的文件都有备份。
    """
    old = old_label.strip()
    new = new_label.strip()
    result = LabelRenameResult()
    if not old or not new or old == new:
        return result

    pending: list[tuple[Path, dict[str, Any], int]] = []
    for raw_path in json_paths:
        path = Path(raw_path)
        data = _load_json_dict(path)
        if data is None:
            if path.is_file():
                result.failed.append((path, "文件无法解析为标注 JSON"))
            continue
        cnt = _count_label(data, old)
        if cnt > 0:
            pending.append((path, data, cnt))

    if not pending:
        return result

    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        result.backup_dir = backup_dir

    for path, data, cnt in pending:
        if backup_dir is not None:
            try:
                dst = backup_dir / _backup_relative_name(path, base_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
            except OSError as exc:
                result.failed.append((path, f"备份失败，未改写：{exc}"))
                continue
        for shape in data.get("shapes") or []:
            if isinstance(shape, dict) and str(shape.get("label") or "").strip() == old:
                shape["label"] = new
        try:
            _atomic_write_json(path, data)
        except OSError as exc:
            result.failed.append((path, f"写入失败：{exc}"))
            continue
        result.changed_files.append(path)
        result.changed_shapes += cnt

    if backup_dir is not None:
        manifest = {
            "operation": "rename_label",
            "old_label": old,
            "new_label": new,
            "base_dir": str(base_dir) if base_dir is not None else "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "files": [str(p) for p in result.changed_files],
            "restore_hint": "将本目录中的 .json 按相对路径复制回 base_dir 即可还原",
        }
        try:
            (backup_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            # 清单写入失败不影响已完成的备份与改写，原文件副本仍在备份目录中
            pass
    return result
