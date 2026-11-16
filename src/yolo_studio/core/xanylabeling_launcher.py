"""X-AnyLabeling 外部标注工具发现与启动辅助模块。

职责：
    在系统 `PATH` 与常见 Windows 安装目录中查找 X-AnyLabeling 可执行文件，
    并在指定的抽帧/待复核图片目录上非阻塞拉起 X-AnyLabeling 进程。

运行环境：
    Python >= 3.11，桌面 GUI 会话环境；预设路径面向 Windows，同时兼容 `PATH` 中的跨平台命令名。

协作对象与职责边界：
    由 `gui.tab_dataset.DatasetTab` 与 `gui.tab_review.MllmReviewTab` 调用。
    仅负责查找路径与启动独立子进程，不监控、等待或关闭子进程，也不读写标注文件。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


COMMON_WINDOWS_LOCATIONS = [
    Path("C:/Program Files/X-AnyLabeling/X-AnyLabeling.exe"),
    Path("C:/Program Files (x86)/X-AnyLabeling/X-AnyLabeling.exe"),
    Path.home() / "Desktop" / "X-AnyLabeling.exe",
    Path.home() / "Downloads" / "X-AnyLabeling.exe",
    Path("D:/X-AnyLabeling/X-AnyLabeling.exe"),
    Path("D:/Tools/X-AnyLabeling/X-AnyLabeling.exe"),
]


def find_xanylabeling_executable() -> Path | None:
    """依次检索 `PATH` 与常见安装目录中的 X-AnyLabeling 可执行文件，未找到返回 `None`。"""
    for cmd in ("X-AnyLabeling", "x-anylabeling", "xanylabeling"):
        found = shutil.which(cmd)
        if found:
            return Path(found)

    for candidate in COMMON_WINDOWS_LOCATIONS:
        if candidate.is_file():
            return candidate
    return None


def launch_xanylabeling(
    frames_dir: Path,
    executable_path: Path | None = None,
    classes_file: Path | None = None,
) -> subprocess.Popen[bytes]:
    """在目标图片目录上以非阻塞子进程方式启动 X-AnyLabeling。

    调用条件与失败方式：
        要求 `frames_dir` 为已存在目录且能解析出有效的 `executable_path`，
        任一条件不满足时抛出 `FileNotFoundError`。若提供有效的 `classes_file`，
        会附加 `--labels` 参数。调用方自行决定是否管理返回的 `Popen` 句柄。
    """
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {frames_dir}")

    exe = executable_path or find_xanylabeling_executable()
    if exe is None or not exe.exists():
        raise FileNotFoundError(
            "未找到 X-AnyLabeling 可执行程序，请先在界面中指定 X-AnyLabeling.exe 路径。"
        )

    args = [str(exe), str(frames_dir)]
    if classes_file and classes_file.is_file():
        args.extend(["--labels", str(classes_file)])

    return subprocess.Popen(args)
