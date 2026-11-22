"""YoloStudio 项目单一版本源模块。

职责与协作边界：
    作为全仓库唯一版本号定义文件，统一供 `pyproject.toml`（setuptools 动态版本）、
    桌面主窗口标题栏、启动欢迎图、CLI `--version` 选项及 `scripts/build_portable.py`
    （构建产物命名与 `YoloStudio.exe` Windows PE 文件版本属性）共享读取。
    发布新版本时仅需修改本文件的 `__version__`（或通过 `build_portable.ps1 --bump` 自动递增）。
"""

__version__ = "0.3.7"
