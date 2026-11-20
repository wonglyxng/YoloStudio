"""YOLO26 视频抽帧与训练工具包（YoloStudio）。

定义软件包顶层边界并导出单一版本源 `_version.__version__`；导入本模块不会启动 GUI、CLI 或后台线程。
"""

from yolo_studio._version import __version__

__all__ = ["__version__"]
