"""YoloStudio Windows 独立便携整合包（方案 A）自动化构建与自检工具。

职责与真实入口：
    将免安装的独立 Python 3.12 运行时、第三方依赖库、YoloStudio 应用代码、内嵌 X-AnyLabeling 源码、
    原生无控制台启动器 `YoloStudio.exe`、命令行入口 `YoloStudio-CLI.cmd`、预置模型权重与开源协议文件
    组装为无外部 Python 依赖即可在目标 Windows x64 电脑解压运行的便携目录（及可选 `.7z`/`.zip` 归档包）。
    可通过 `python scripts/build_portable.py` 或包装脚本 `scripts/build_portable.ps1` 直接调用
    （完整流程说明见 `docs/003_独立便携整合包构建与发布说明.md`）。

输入来源与构建模式：
    1. `--mode clean`（推荐正式发布）：从指定独立 Python 目录/归档、`uv` 或 `python-build-standalone`
       官方发行包创建纯净 `runtime/`，按 `--variant gpu|cpu` 安装依赖并消除 OpenCV 同名包冲突。
    2. `--mode from-venv`（本机离线快速打包）：基于现有 `.venv` 与其 `pyvenv.cfg` 指向的宿主基础 Python
       组装独立 `runtime/`，自动剥离 `pyvenv.cfg`、可编辑安装 `.pth`、`direct_url.json` 与测试专用包。

输出及产物位置：
    默认输出目录为 `dist/YoloStudio-v<版本号>-win64-<gpu|cpu>/`；
    启用 `--archive 7z` 或 `--archive zip` 时在同级 `dist/` 下生成对应的单文件压缩包。

环境依赖与失败方式：
    仅依赖 Python >= 3.11 标准库运行本构建脚本；编译 `YoloStudio.exe` 依赖 Windows 10/11 系统内置的
    `.NET Framework` 编译器 `csc.exe`；`7z` 压缩依赖 `7z.exe`（缺失时抛出 `FileNotFoundError` 或改用 `zip`）。
    任一构建或便携性冒烟校验步骤失败时抛出异常并以非零状态码退出。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MSVC_RUNTIME_DLL_NAMES: tuple[str, ...] = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
    "vcomp140.dll",
)

# 从宿主基础 Python 复制到便携 runtime 时跳过的顶层冗余项
BASE_PYTHON_EXCLUDED_TOP: frozenset[str] = frozenset(
    {
        "pyvenv.cfg",
        "doc",
        "tcl",
        "scripts",
        "__pycache__",
    }
)

# 从宿主基础 Python 的 Lib 标准库目录复制时默认跳过的子目录（避免混入测试套件与闲置 Tk 组件）
BASE_LIB_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        "test",
        "tests",
        "idlelib",
        "tkinter",
        "turtledemo",
        "__pycache__",
    }
)

# 便携包清洗阶段从 site-packages 中剔除的纯开发测试包前缀
DEV_ONLY_PACKAGE_PREFIXES: tuple[str, ...] = (
    "pytest",
    "_pytest",
    "pluggy",
    "iniconfig",
    "py.py",
)

# 复制 third_party/X-AnyLabeling 时仅保留运行与许可证必需项，剔除宣传大图、示例与单元测试
XAL_ALLOWED_ITEMS: tuple[str, ...] = (
    "anylabeling",
    "LICENSE",
    "pyproject.toml",
    "README.md",
    "README_zh-CN.md",
)

CORE_PIP_REQUIREMENTS: tuple[str, ...] = (
    "ultralytics>=8.4.0",
    "PyQt6>=6.6.0",
    "PyQt6-WebEngine>=6.6.0",
    "Pillow>=10.0.0",
    "PyYAML>=6.0.1",
    "numpy>=2.0.0",
    "google-genai>=2.30.1",
    "openai>=1.50.0",
    "pydantic>=2.0.0",
)

PBS_LATEST_API_URL = (
    "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
)


@dataclass
class BuildConfig:
    """独立便携整合包单次构建参数配置。

    关键字段说明：
        `mode`：`"clean"` 表示从纯净独立 Python 重新安装依赖；`"from-venv"` 表示克隆现有虚拟环境。
        `variant`：`"gpu"`（含 CUDA PyTorch 与 `onnxruntime-gpu`）或 `"cpu"`（轻量纯 CPU 推理/标注版）。
        `output_root`：构建产物根目录（默认 `<仓库根>/dist`），最终包位于其下的 `<bundle_name>/`。
        `archive_format`：`"none"`、`"7z"` 或 `"zip"`。
    """

    repo_root: Path = REPO_ROOT
    mode: str = "from-venv"
    variant: str = "gpu"
    output_root: Path = REPO_ROOT / "dist"
    cache_dir: Path = REPO_ROOT / "build" / "portable_cache"
    venv_dir: Path = REPO_ROOT / ".venv"
    standalone_python_dir: Path | None = None
    standalone_python_archive: Path | None = None
    python_version: str = "3.12"
    torch_index_url: str = ""
    pip_index_url: str = ""
    extra_weights: list[Path] = field(default_factory=list)
    download_weights: list[str] = field(default_factory=list)
    fix_opencv: bool = False
    archive_format: str = "none"
    skip_smoke_test: bool = False
    skip_heavy_imports: bool = False
    overwrite: bool = True


def read_project_version(repo_root: Path) -> str:
    """从仓库 `pyproject.toml` 解析项目版本号；未匹配到时抛出 `RuntimeError`。"""
    pyproject_file = repo_root / "pyproject.toml"
    content = pyproject_file.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        raise RuntimeError(f"无法从 {pyproject_file} 解析 version 字段")
    return match.group(1)


def parse_pyvenv_home(venv_dir: Path) -> Path:
    """读取虚拟环境 `pyvenv.cfg` 中的 `home` 字段并返回宿主基础 Python 根目录。

    失败方式：
        `pyvenv.cfg` 缺失、缺少 `home` 键或对应目录不存在 `python.exe` 时抛出 `FileNotFoundError` / `ValueError`。
    """
    cfg_file = venv_dir / "pyvenv.cfg"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"未找到虚拟环境配置文件: {cfg_file}")

    home_value: str | None = None
    for raw_line in cfg_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        if key.strip().lower() == "home":
            home_value = val.strip()
            break

    if not home_value:
        raise ValueError(f"pyvenv.cfg 中缺少 home 配置项: {cfg_file}")

    home_dir = Path(home_value).resolve()
    if not (home_dir / "python.exe").is_file():
        raise FileNotFoundError(f"宿主基础 Python 目录缺少 python.exe: {home_dir}")
    return home_dir


def _copy_tree_filtered(
    src_dir: Path,
    dst_dir: Path,
    exclude_names_lower: frozenset[str] = frozenset({"__pycache__"}),
    exclude_suffixes_lower: tuple[str, ...] = (".pyc", ".pyo"),
) -> None:
    """递归复制目录树，按不区分大小写的名称集合与后缀元组过滤无关缓存文件。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        name_lower = item.name.lower()
        if name_lower in exclude_names_lower:
            continue
        target = dst_dir / item.name
        if item.is_dir():
            _copy_tree_filtered(item, target, exclude_names_lower, exclude_suffixes_lower)
        elif item.is_file():
            if name_lower.endswith(exclude_suffixes_lower):
                continue
            shutil.copy2(item, target)


def clone_base_python_runtime(
    base_python_dir: Path,
    runtime_dir: Path,
    keep_site_packages: bool = False,
) -> None:
    """将基础 Python 核心文件与标准库复制到 `runtime_dir`。

    参数说明：
        `keep_site_packages`：`from-venv` 模式下设为 `False` 以排除宿主全局杂包；
        `clean` 模式下从纯净 `python-build-standalone` 归档初始化时设为 `True`，保留其预置的 `pip`。
        任何模式下均跳过 `pyvenv.cfg` 与受限 `._pth` 文件，并保留 `Lib/ensurepip` 供包管理器自举兜底。
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lib_excluded = (
        BASE_LIB_EXCLUDED_DIRS
        if keep_site_packages
        else (BASE_LIB_EXCLUDED_DIRS | {"site-packages"})
    )

    for entry in base_python_dir.iterdir():
        name_lower = entry.name.lower()
        if name_lower in BASE_PYTHON_EXCLUDED_TOP:
            continue
        if name_lower.endswith("._pth"):
            # 不复制嵌入版受限路径锁定文件，统一使用标准 Lib/site-packages + .pth 机制
            continue
        dst = runtime_dir / entry.name
        if entry.is_dir():
            if name_lower == "lib":
                dst.mkdir(parents=True, exist_ok=True)
                for lib_item in entry.iterdir():
                    if lib_item.name.lower() in lib_excluded:
                        continue
                    lib_dst = dst / lib_item.name
                    if lib_item.is_dir():
                        _copy_tree_filtered(lib_item, lib_dst)
                    elif lib_item.is_file():
                        shutil.copy2(lib_item, lib_dst)
            else:
                _copy_tree_filtered(entry, dst)
        elif entry.is_file():
            shutil.copy2(entry, dst)

    (runtime_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)


def _safe_extract_archive(archive_path: Path, extract_to: Path) -> Path:
    """安全解压 `.tar.gz` / `.tgz` / `.zip` 独立 Python 归档，校验路径越界并返回含 `python.exe` 的根目录。"""
    extract_to.mkdir(parents=True, exist_ok=True)
    root_resolved = extract_to.resolve()
    name_lower = archive_path.name.lower()

    if name_lower.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path, "r:*") as tar:
            for member in tar.getmembers():
                target = (extract_to / member.name).resolve()
                if root_resolved != target and root_resolved not in target.parents:
                    raise ValueError(f"检测到越界归档条目: {member.name}")
            try:
                tar.extractall(extract_to, filter="data")
            except TypeError:
                tar.extractall(extract_to)
    elif name_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member_name in zf.namelist():
                target = (extract_to / member_name).resolve()
                if root_resolved != target and root_resolved not in target.parents:
                    raise ValueError(f"检测到越界归档条目: {member_name}")
            zf.extractall(extract_to)
    else:
        raise ValueError(f"不支持的独立 Python 归档格式: {archive_path}")

    for candidate in [extract_to, *sorted(extract_to.rglob("*"))]:
        if candidate.is_dir() and (candidate / "python.exe").is_file():
            return candidate
    raise FileNotFoundError(f"解压后的归档中未找到 python.exe: {archive_path}")


def _download_python_build_standalone(python_version: str, cache_dir: Path) -> Path:
    """从 `python-build-standalone` 最新 Release 下载匹配的 Windows x64 独立 Python 归档并缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"cpython-{python_version}."
    suffix = "-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"

    existing = sorted(cache_dir.glob(f"{prefix}*{suffix}"))
    if existing:
        print(f"[缓存] 复用已下载的独立 Python 归档: {existing[-1]}")
        return existing[-1]

    print(f"[网络] 正在查询 python-build-standalone 最新 {python_version} Windows x64 发行包...")
    req = urllib.request.Request(
        PBS_LATEST_API_URL,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "YoloStudio-Packager"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        release_data = json.loads(resp.read().decode("utf-8"))

    matched_asset: dict[str, object] | None = None
    for asset in release_data.get("assets", []):
        name = str(asset.get("name", ""))
        if name.startswith(prefix) and name.endswith(suffix):
            matched_asset = asset
            break

    if not matched_asset:
        raise RuntimeError(
            f"未在最新 Release 中找到匹配 {prefix}*{suffix} 的独立 Python 归档，"
            "可通过 --standalone-python-archive 手动指定离线包。"
        )

    asset_name = str(matched_asset["name"])
    download_url = str(matched_asset["browser_download_url"])
    target_file = cache_dir / asset_name
    tmp_file = cache_dir / f"{asset_name}.part"
    print(f"[下载] {download_url} -> {target_file}")
    with urllib.request.urlopen(download_url, timeout=180) as src, tmp_file.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp_file.replace(target_file)
    return target_file


def prepare_clean_runtime(config: BuildConfig, runtime_dir: Path) -> None:
    """创建纯净独立 Python 运行时并按 `config.variant` 安装全量运行依赖。"""
    if config.standalone_python_dir is not None:
        src_py = config.standalone_python_dir.resolve()
        if not (src_py / "python.exe").is_file():
            raise FileNotFoundError(f"指定的 --standalone-python-dir 缺少 python.exe: {src_py}")
        clone_base_python_runtime(src_py, runtime_dir)
    else:
        archive = config.standalone_python_archive
        if archive is None:
            uv_bin = shutil.which("uv")
            if uv_bin:
                uv_py_root = config.cache_dir / "uv_python"
                uv_py_root.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [uv_bin, "python", "install", config.python_version, "--install-dir", str(uv_py_root)],
                    check=True,
                )
                candidates = [
                    p
                    for p in sorted(uv_py_root.iterdir())
                    if p.is_dir() and (p / "python.exe").is_file() and config.python_version in p.name
                ]
                if candidates:
                    clone_base_python_runtime(candidates[-1], runtime_dir, keep_site_packages=True)
                else:
                    archive = _download_python_build_standalone(config.python_version, config.cache_dir)
            else:
                archive = _download_python_build_standalone(config.python_version, config.cache_dir)

        if archive is not None:
            extract_stage = config.cache_dir / "extracted_py"
            if extract_stage.exists():
                shutil.rmtree(extract_stage)
            extracted_root = _safe_extract_archive(archive.resolve(), extract_stage)
            clone_base_python_runtime(extracted_root, runtime_dir, keep_site_packages=True)
            shutil.rmtree(extract_stage, ignore_errors=True)

    py_exe = runtime_dir / "python.exe"
    pip_env = _isolated_subprocess_env(runtime_dir)

    # 确保 pip 可用
    check_pip = subprocess.run([str(py_exe), "-m", "pip", "--version"], env=pip_env, capture_output=True)
    if check_pip.returncode != 0:
        subprocess.run([str(py_exe), "-m", "ensurepip", "--upgrade"], env=pip_env, check=True)

    pip_common = [str(py_exe), "-m", "pip", "install", "--no-warn-script-location"]
    if config.pip_index_url:
        pip_common.extend(["-i", config.pip_index_url])

    torch_index = config.torch_index_url.strip()
    if not torch_index:
        torch_index = (
            "https://download.pytorch.org/whl/cu126"
            if config.variant == "gpu"
            else "https://download.pytorch.org/whl/cpu"
        )

    print(f"[依赖] 安装 PyTorch ({config.variant}): {torch_index}")
    subprocess.run(
        [str(py_exe), "-m", "pip", "install", "--no-warn-script-location", "torch", "torchvision", "--index-url", torch_index],
        env=pip_env,
        check=True,
    )

    xal_extra = "gpu" if config.variant == "gpu" else "cpu"
    xal_path = config.repo_root / "third_party" / "X-AnyLabeling"
    print(f"[依赖] 安装 X-AnyLabeling[{xal_extra}] 运行时依赖...")
    subprocess.run(
        [*pip_common, f"{xal_path}[{xal_extra}]"],
        env=pip_env,
        check=True,
    )

    print("[依赖] 安装 YoloStudio 核心依赖...")
    subprocess.run([*pip_common, *CORE_PIP_REQUIREMENTS], env=pip_env, check=True)

    _enforce_single_opencv_headless(py_exe, pip_env, config.pip_index_url)


def _enforce_single_opencv_headless(py_exe: Path, env: dict[str, str], pip_index_url: str = "") -> None:
    """卸载冲突的 `opencv-python` 并以 `--no-deps` 重装 `opencv-contrib-python-headless`。"""
    print("[依赖] 规范化 OpenCV 包，消除 opencv-python 与 contrib-headless 覆盖冲突...")
    subprocess.run(
        [str(py_exe), "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python-headless"],
        env=env,
        check=True,
    )
    install_cmd = [
        str(py_exe),
        "-m",
        "pip",
        "install",
        "--no-warn-script-location",
        "--no-deps",
        "opencv-contrib-python-headless>=4.7.0.72",
    ]
    if pip_index_url:
        install_cmd.extend(["-i", pip_index_url])
    subprocess.run(install_cmd, env=env, check=True)


def prepare_from_venv_runtime(config: BuildConfig, runtime_dir: Path) -> None:
    """从现有虚拟环境及其宿主基础 Python 离线克隆组装独立 `runtime/`。"""
    venv_dir = config.venv_dir.resolve()
    venv_sp = venv_dir / "Lib" / "site-packages"
    if not venv_sp.is_dir():
        raise FileNotFoundError(f"虚拟环境缺少 site-packages 目录: {venv_sp}")

    base_py = (
        config.standalone_python_dir.resolve()
        if config.standalone_python_dir is not None
        else parse_pyvenv_home(venv_dir)
    )
    print(f"[运行时] 复制基础 Python 解释器: {base_py} -> {runtime_dir}")
    clone_base_python_runtime(base_py, runtime_dir)

    target_sp = runtime_dir / "Lib" / "site-packages"
    print(f"[运行时] 复制第三方依赖库: {venv_sp} -> {target_sp}（体积较大，请稍候）...")
    _copy_tree_filtered(venv_sp, target_sp)

    if config.fix_opencv:
        py_exe = runtime_dir / "python.exe"
        _enforce_single_opencv_headless(py_exe, _isolated_subprocess_env(runtime_dir), config.pip_index_url)


def sanitize_site_packages(site_packages_dir: Path, remove_bundled_anylabeling_copy: bool = True) -> list[str]:
    """清理 `site-packages` 中的本地绝对路径残留、可编辑安装钩子及开发期测试专用包。

    处理内容：
        1. 删除 `__editable__*.pth`、`__editable__*.py`、`*.egg-link`；
        2. 清理 `easy-install.pth` 中包含盘符或外部绝对路径的行；
        3. 删除 `*.dist-info/direct_url.json`，防止暴露开发机构建目录；
        4. 删除 `pytest`、`_pytest`、`pluggy`、`iniconfig` 等纯测试包；
        5. 当 `remove_bundled_anylabeling_copy=True` 时，若 `clean` 模式将 `anylabeling` 与 `yolo_studio`
           安装进了 `site-packages`，移除重复包体以统一由 `app/` 目录通过相对 `.pth` 提供唯一源码版本。
    返回被移除或净化的路径摘要列表。
    """
    removed: list[str] = []
    if not site_packages_dir.is_dir():
        return removed

    for entry in list(site_packages_dir.iterdir()):
        name = entry.name
        name_lower = name.lower()

        if (
            name_lower.startswith("__editable__")
            or name_lower.endswith(".egg-link")
            or any(
                name_lower == prefix
                or name_lower.startswith(f"{prefix}-")
                or name_lower.startswith(f"{prefix}_")
                for prefix in DEV_ONLY_PACKAGE_PREFIXES
            )
        ):
            _remove_path(entry)
            removed.append(name)
            continue

        if remove_bundled_anylabeling_copy and name_lower in {"anylabeling", "yolo_studio"}:
            _remove_path(entry)
            removed.append(name)
            continue

        if entry.is_dir() and name_lower.endswith(".dist-info"):
            direct_url = entry / "direct_url.json"
            if direct_url.is_file():
                direct_url.unlink()
                removed.append(f"{name}/direct_url.json")

    easy_install = site_packages_dir / "easy-install.pth"
    if easy_install.is_file():
        kept_lines: list[str] = []
        for raw in easy_install.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw.strip()
            if not stripped or re.match(r"^[a-zA-Z]:[\\/]", stripped) or stripped.startswith("/"):
                removed.append(f"easy-install.pth:{stripped}")
                continue
            kept_lines.append(raw)
        if kept_lines:
            easy_install.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        else:
            easy_install.unlink()
            removed.append("easy-install.pth")

    return removed


def write_portable_pth(site_packages_dir: Path) -> Path:
    """在 `runtime/Lib/site-packages` 写入纯相对路径 `.pth` 文件，挂载 `app/src` 与内嵌 `X-AnyLabeling`。

    设计原因：
        CPython `site.addsitedir` 以 `site-packages` 自身目录为基准解析 `.pth` 中的相对路径；
        采用此方式不仅免去了硬编码绝对路径，还能保证 Windows 下 PyTorch `DataLoader(workers>0)`
        通过 `spawn` 启动的子进程在导入业务模块前自动挂载 `yolo_studio` 与 `anylabeling`。
    """
    site_packages_dir.mkdir(parents=True, exist_ok=True)
    pth_file = site_packages_dir / "yolo_studio_portable.pth"
    pth_content = "\n".join(
        [
            "# YoloStudio 独立便携包相对模块搜索路径（由 build_portable.py 自动生成）",
            "../../../app/src",
            "../../../app/third_party/X-AnyLabeling",
            "",
        ]
    )
    pth_file.write_text(pth_content, encoding="utf-8")
    return pth_file


def bundle_msvc_runtime_dlls(runtime_dir: Path) -> list[str]:
    """收集并补齐 MSVC C/C++/OpenMP 运行时 DLL 到 `runtime/` 根目录，保障全新 Windows 裸机可直接加载扩展。"""
    search_dirs: list[Path] = [runtime_dir]
    sys_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if sys_root:
        search_dirs.append(Path(sys_root) / "System32")
    search_dirs.append(Path(sys.executable).resolve().parent)
    search_dirs.append(Path(sys.base_prefix).resolve())
    for raw_part in os.environ.get("PATH", "").split(os.pathsep):
        if raw_part.strip():
            search_dirs.append(Path(raw_part.strip()))

    copied: list[str] = []
    for dll_name in MSVC_RUNTIME_DLL_NAMES:
        target = runtime_dir / dll_name
        if target.is_file():
            continue
        for folder in search_dirs:
            candidate = folder / dll_name
            if candidate.is_file():
                shutil.copy2(candidate, target)
                copied.append(dll_name)
                break
    return copied


def copy_app_and_docs(repo_root: Path, bundle_root: Path) -> None:
    """复制应用入口、`src/yolo_studio`、精简后的 `third_party/X-AnyLabeling`、文档与开源协议声明。"""
    app_dir = bundle_root / "app"
    app_dir.mkdir(parents=True, exist_ok=True)

    for top_script in ("main.py", "cli.py", "pyproject.toml"):
        src_file = repo_root / top_script
        if src_file.is_file():
            shutil.copy2(src_file, app_dir / top_script)

    _copy_tree_filtered(repo_root / "src" / "yolo_studio", app_dir / "src" / "yolo_studio")

    xal_src = repo_root / "third_party" / "X-AnyLabeling"
    xal_dst = app_dir / "third_party" / "X-AnyLabeling"
    xal_dst.mkdir(parents=True, exist_ok=True)
    for item_name in XAL_ALLOWED_ITEMS:
        src_item = xal_src / item_name
        if not src_item.exists():
            continue
        dst_item = xal_dst / item_name
        if src_item.is_dir():
            _copy_tree_filtered(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

    for root_doc in ("LICENSE", "THIRD_PARTY_NOTICES.md", "README.md", "项目文档索引.md"):
        src_doc = repo_root / root_doc
        if src_doc.is_file():
            shutil.copy2(src_doc, bundle_root / root_doc)

    docs_src = repo_root / "docs"
    if docs_src.is_dir():
        _copy_tree_filtered(docs_src, bundle_root / "docs")


def prepare_weights_and_workspace(config: BuildConfig, bundle_root: Path) -> list[str]:
    """初始化 `workspace/` 与 `weights/` 目录，收集本地已有或指定预置的 `.pt` 模型权重。"""
    (bundle_root / "workspace").mkdir(parents=True, exist_ok=True)
    weights_dir = bundle_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    staged_weights: list[str] = []
    candidate_files: list[Path] = list(config.extra_weights)
    for default_name in ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"):
        for folder in (config.repo_root, config.repo_root / "weights"):
            candidate = folder / default_name
            if candidate.is_file() and candidate not in candidate_files:
                candidate_files.append(candidate)

    for weight_path in candidate_files:
        if weight_path.is_file() and weight_path.stat().st_size > 0:
            dst = weights_dir / weight_path.name
            shutil.copy2(weight_path, dst)
            staged_weights.append(weight_path.name)

    if config.download_weights:
        py_exe = bundle_root / "runtime" / "python.exe"
        env = _isolated_subprocess_env(bundle_root / "runtime", bundle_root)
        dl_code = """
import shutil
import sys
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset

dst = Path(sys.argv[1]).resolve()
downloaded = Path(attempt_download_asset(dst)).resolve()
if downloaded.is_file() and downloaded != dst:
    shutil.copy2(downloaded, dst)
if not dst.is_file() or dst.stat().st_size == 0:
    raise RuntimeError(f"未能下载预训练模型权重: {dst.name}")
print(f"downloaded_ok={dst.name} ({dst.stat().st_size} bytes)")
"""
        for model_name in config.download_weights:
            target = weights_dir / Path(model_name).name
            if target.is_file() and target.stat().st_size > 0:
                staged_weights.append(target.name)
                continue
            print(f"[权重] 下载预置模型权重至 weights/: {model_name}")
            subprocess.run(
                [str(py_exe), "-c", dl_code, str(target)],
                cwd=str(weights_dir),
                env=env,
                check=True,
            )
            if target.is_file():
                staged_weights.append(target.name)

    return staged_weights


def find_csharp_compiler() -> Path:
    """在系统 `PATH` 与 Windows `.NET Framework` 标准目录中查找 `csc.exe`，未找到抛出 `FileNotFoundError`。"""
    from_path = shutil.which("csc") or shutil.which("csc.exe")
    if from_path:
        return Path(from_path).resolve()

    win_dir = Path(os.environ.get("WINDIR") or os.environ.get("SystemRoot") or r"C:\Windows")
    candidates = [
        win_dir / "Microsoft.NET" / "Framework64" / "v4.0.30319" / "csc.exe",
        win_dir / "Microsoft.NET" / "Framework" / "v4.0.30319" / "csc.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 Windows 内置 C# 编译器 csc.exe，无法编译无黑框启动器 YoloStudio.exe。")


def compile_windows_launcher(
    launcher_cs: Path,
    output_exe: Path,
    icon_file: Path | None = None,
) -> Path:
    """调用系统内置 `csc.exe` 将 `YoloStudioLauncher.cs` 编译为原生无黑框 GUI 启动器 `YoloStudio.exe`。"""
    if not launcher_cs.is_file():
        raise FileNotFoundError(f"启动器源码不存在: {launcher_cs}")

    csc_path = find_csharp_compiler()
    output_exe.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(csc_path),
        "/nologo",
        "/target:winexe",
        "/optimize+",
        "/platform:anycpu",
        "/r:System.Windows.Forms.dll",
        f"/out:{output_exe}",
    ]
    if icon_file is not None and icon_file.is_file():
        cmd.append(f"/win32icon:{icon_file}")
    cmd.append(str(launcher_cs))

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0 or not output_exe.is_file():
        detail = (res.stdout or "") + "\n" + (res.stderr or "")
        raise RuntimeError(f"编译 YoloStudio.exe 失败 (退出码 {res.returncode}):\n{detail.strip()}")
    return output_exe


def write_cli_and_quickstart_files(bundle_root: Path, version: str, variant: str) -> None:
    """在便携包根目录生成命令行入口 `YoloStudio-CLI.cmd` 与终端用户快速启动说明文本。

    批处理兼容性说明：
        全程使用标签跳转而非多行 `if (...)` 圆括号块，避免便携包解压在 `Program Files (x86)`
        等含圆括号目录或回显含括号文本时触发 `cmd.exe` 语法截断报错。
    """
    cli_cmd = bundle_root / "YoloStudio-CLI.cmd"
    cli_script = "\r\n".join(
        [
            "@echo off",
            "setlocal",
            "chcp 65001 >nul",
            'set "BUNDLE_ROOT=%~dp0"',
            'if "%BUNDLE_ROOT:~-1%"=="\\" set "BUNDLE_ROOT=%BUNDLE_ROOT:~0,-1%"',
            'set "PYTHONHOME="',
            'set "PYTHONPATH="',
            'set "VIRTUAL_ENV="',
            'set "CONDA_PREFIX="',
            'set "CONDA_DEFAULT_ENV="',
            'set "PYTHONNOUSERSITE=1"',
            'set "PYTHONUTF8=1"',
            'set "PYTHONIOENCODING=utf-8"',
            'if "%YOLO_STUDIO_WORKSPACE%"=="" set "YOLO_STUDIO_WORKSPACE=%BUNDLE_ROOT%\\workspace"',
            'if "%YOLO_STUDIO_WEIGHTS_DIR%"=="" set "YOLO_STUDIO_WEIGHTS_DIR=%BUNDLE_ROOT%\\weights"',
            'set "PATH=%BUNDLE_ROOT%\\runtime;%BUNDLE_ROOT%\\runtime\\DLLs;%BUNDLE_ROOT%\\runtime\\Library\\bin;%PATH%"',
            "",
            'if exist "%BUNDLE_ROOT%\\runtime\\python.exe" goto :runtime_ok',
            "echo [错误] 未找到内置运行环境 runtime\\python.exe，请先完整解压压缩包后再运行。",
            "pause",
            "exit /b 1",
            "",
            ":runtime_ok",
            'if not "%~1"=="" goto :run_with_args',
            "echo ================================================================",
            f"echo   YoloStudio v{version} [{variant.upper()}] 独立便携命令行环境",
            "echo ================================================================",
            "echo 示例命令：",
            'echo   YoloStudio-CLI.cmd --help',
            'echo   YoloStudio-CLI.cmd extract "D:\\videos\\demo.mp4" -o "%YOLO_STUDIO_WORKSPACE%\\data\\raw_frames"',
            "echo.",
            '"%BUNDLE_ROOT%\\runtime\\python.exe" "%BUNDLE_ROOT%\\app\\cli.py" --help',
            "echo.",
            'if not exist "%YOLO_STUDIO_WORKSPACE%" mkdir "%YOLO_STUDIO_WORKSPACE%" 2>nul',
            'cmd /k "cd /d \"%YOLO_STUDIO_WORKSPACE%\""',
            "exit /b 0",
            "",
            ":run_with_args",
            '"%BUNDLE_ROOT%\\runtime\\python.exe" "%BUNDLE_ROOT%\\app\\cli.py" %*',
            "exit /b %ERRORLEVEL%",
            "",
        ]
    )
    cli_cmd.write_bytes(cli_script.encode("utf-8"))

    quickstart = bundle_root / "使用说明_请先解压后双击YoloStudio运行.txt"
    quickstart_text = "\r\n".join(
        [
            f"YoloStudio v{version} ({variant.upper()} 独立便携整合包)",
            "============================================================",
            "",
            "【1. 启动方式（无需安装 Python 或 CUDA Toolkit）】",
            "  1) 请先将整个压缩包【完整解压】到任意本地磁盘目录（支持中文路径，建议预留 10GB 以上空间）。",
            "  2) 双击根目录下的 YoloStudio.exe 即可启动桌面图形界面。",
            "  3) 如需使用批处理命令行，请调用 YoloStudio-CLI.cmd（如：YoloStudio-CLI.cmd extract --help）。",
            "  4) 如遇极少数环境异常需查看底层终端输出，可在命令行运行：YoloStudio.exe --console",
            "",
            "【2. 目录说明】",
            "  - YoloStudio.exe      : 桌面图形界面启动器（无黑框）",
            "  - YoloStudio-CLI.cmd  : 无头批处理命令行入口",
            "  - workspace/          : 默认用户数据工作区（抽帧图片、构建的数据集与训练结果默认保存在此）",
            "  - weights/            : 预训练模型权重存放目录（可将离线下载的 yolo26n.pt / yolo26s.pt 放入此处）",
            "  - runtime/            : 内置独立 Python 3.12 运行时与深度学习依赖库（请勿随意删除）",
            "  - app/                : YoloStudio 与内嵌 X-AnyLabeling 应用源码（后续小版本补丁直接覆盖此目录即可）",
            "  - docs/               : 详细图文工作流与参数使用说明",
            "",
            "【3. 显卡加速说明】",
            "  - GPU 版内置了 CUDA 与 cuDNN 运行库；若使用 NVIDIA 显卡加速，仅需电脑已安装常规 NVIDIA 显卡驱动。",
            "  - 未检测到可用 NVIDIA GPU 的电脑会自动回退至 CPU 模式运行。",
            "",
            "【4. 开源协议】",
            "  - 本软件整体遵循 GNU General Public License v3.0 (GPL-3.0-only)，详见根目录 LICENSE 与 THIRD_PARTY_NOTICES.md。",
            "",
        ]
    )
    quickstart.write_bytes(quickstart_text.encode("utf-8"))


def verify_portable_bundle(bundle_root: Path, skip_heavy_imports: bool = False) -> None:
    """对组装完成的便携整合包执行静态路径泄漏扫描与隔离环境运行冒烟验证。"""
    required_files = [
        bundle_root / "YoloStudio.exe",
        bundle_root / "YoloStudio-CLI.cmd",
        bundle_root / "runtime" / "python.exe",
        bundle_root / "app" / "main.py",
        bundle_root / "app" / "cli.py",
        bundle_root / "LICENSE",
        bundle_root / "THIRD_PARTY_NOTICES.md",
    ]
    for req in required_files:
        if not req.is_file():
            raise FileNotFoundError(f"便携包完整性校验失败，缺少关键文件: {req}")

    if (bundle_root / "runtime" / "pyvenv.cfg").exists():
        raise RuntimeError("便携包 runtime/ 中仍残留 pyvenv.cfg，会导致在目标电脑上寻找构建机路径失败。")

    sp_dir = bundle_root / "runtime" / "Lib" / "site-packages"
    for pth_file in sp_dir.glob("*.pth"):
        for line in pth_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("import "):
                continue
            if re.match(r"^[a-zA-Z]:[\\/]", stripped):
                raise RuntimeError(f"检测到 .pth 文件含构建机绝对路径 ({pth_file.name}): {stripped}")

    py_exe = bundle_root / "runtime" / "python.exe"
    env = _isolated_subprocess_env(bundle_root / "runtime", bundle_root)
    env["QT_QPA_PLATFORM"] = "offscreen"

    print("[自检] 验证隔离环境下 YoloStudio-CLI.cmd --help 启动...")
    if os.name == "nt":
        cli_cmd_args: str | list[str] = f'cmd.exe /d /s /c ""{bundle_root / "YoloStudio-CLI.cmd"}" --help"'
    else:
        cli_cmd_args = [str(py_exe), str(bundle_root / "app" / "cli.py"), "--help"]
    cli_res = subprocess.run(
        cli_cmd_args,
        cwd=str(bundle_root / "workspace"),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if cli_res.returncode != 0:
        raise RuntimeError(f"便携包 CLI --help 自检失败:\n{cli_res.stdout}\n{cli_res.stderr}")

    heavy_flag = "0" if skip_heavy_imports else "1"
    probe_code = """
import sys
from pathlib import Path
import cv2, yaml, PIL, pydantic, PyQt6.QtWidgets
import yolo_studio, anylabeling

bundle = Path(sys.argv[1]).resolve()
ys_path = Path(yolo_studio.__file__).resolve()
al_path = Path(anylabeling.__file__).resolve()
assert bundle in ys_path.parents, f"yolo_studio 未从便携包内加载: {ys_path}"
assert bundle in al_path.parents, f"anylabeling 未从便携包内加载: {al_path}"

if sys.argv[2] == "1":
    import torch, onnxruntime, ultralytics
    print(f"probe_ok torch={torch.__version__} ort={onnxruntime.__version__} ultra={ultralytics.__version__}")
else:
    print("probe_ok light")
"""
    print("[自检] 验证相对 .pth 模块挂载路径隔离与核心运行库导入...")
    probe_res = subprocess.run(
        [str(py_exe), "-c", probe_code, str(bundle_root), heavy_flag],
        cwd=str(bundle_root / "workspace"),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if probe_res.returncode != 0:
        raise RuntimeError(f"便携包模块隔离与导入自检失败:\n{probe_res.stdout}\n{probe_res.stderr}")
    print(f"[自检] 通过: {probe_res.stdout.strip()}")


def create_bundle_archive(bundle_root: Path, archive_format: str) -> Path | None:
    """按指定格式（`7z` 或 `zip`）将便携整合包目录打包为单文件压缩归档，保留顶层目录名。"""
    fmt = archive_format.strip().lower()
    if fmt in {"", "none"}:
        return None

    parent_dir = bundle_root.parent
    if fmt == "7z":
        seven_zip = shutil.which("7z") or shutil.which("7z.exe")
        if not seven_zip:
            default_7z = Path(r"C:\Program Files\7-Zip\7z.exe")
            if default_7z.is_file():
                seven_zip = str(default_7z)
        if not seven_zip:
            raise FileNotFoundError("未找到 7z.exe，请安装 7-Zip 或改用 --archive zip / --archive none。")

        archive_path = parent_dir / f"{bundle_root.name}.7z"
        if archive_path.exists():
            archive_path.unlink()
        print(f"[归档] 正在使用 7-Zip 多线程压缩生成: {archive_path} ...")
        subprocess.run(
            [seven_zip, "a", "-t7z", "-mx=5", "-mmt=on", str(archive_path), bundle_root.name],
            cwd=str(parent_dir),
            check=True,
        )
        return archive_path

    if fmt == "zip":
        archive_path = parent_dir / f"{bundle_root.name}.zip"
        if archive_path.exists():
            archive_path.unlink()
        print(f"[归档] 正在生成 ZIP 压缩包: {archive_path} ...")
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True
        ) as zf:
            for item in sorted(bundle_root.rglob("*")):
                rel = item.relative_to(parent_dir)
                if item.is_file():
                    zf.write(item, rel.as_posix())
        return archive_path

    raise ValueError(f"不支持的归档格式: {archive_format}（可选: none / 7z / zip）")


def build_portable_bundle(config: BuildConfig) -> tuple[Path, Path | None]:
    """执行完整的独立便携整合包构建流水线，返回 `(便携包目录路径, 压缩包路径或None)`。"""
    version = read_project_version(config.repo_root)
    bundle_name = f"YoloStudio-v{version}-win64-{config.variant}"
    bundle_root = (config.output_root / bundle_name).resolve()

    if bundle_root.exists():
        if not config.overwrite:
            raise FileExistsError(f"输出目录已存在（未启用覆盖）: {bundle_root}")
        print(f"[清理] 移除旧构建目录: {bundle_root}")
        shutil.rmtree(bundle_root)

    bundle_root.mkdir(parents=True, exist_ok=True)
    runtime_dir = bundle_root / "runtime"

    if config.mode == "clean":
        prepare_clean_runtime(config, runtime_dir)
    elif config.mode == "from-venv":
        prepare_from_venv_runtime(config, runtime_dir)
    else:
        raise ValueError(f"未知构建模式: {config.mode}")

    sp_dir = runtime_dir / "Lib" / "site-packages"
    sanitized = sanitize_site_packages(sp_dir, remove_bundled_anylabeling_copy=True)
    if sanitized:
        print(f"[净化] 已清理 {len(sanitized)} 项本地路径/开发测试元数据。")

    write_portable_pth(sp_dir)
    bundled_dlls = bundle_msvc_runtime_dlls(runtime_dir)
    if bundled_dlls:
        print(f"[运行时] 补齐 MSVC 运行库 DLL: {', '.join(bundled_dlls)}")

    print("[应用] 同步 YoloStudio 源码、精简版内嵌 X-AnyLabeling、文档与协议声明...")
    copy_app_and_docs(config.repo_root, bundle_root)

    staged_weights = prepare_weights_and_workspace(config, bundle_root)
    if staged_weights:
        print(f"[权重] 已内置模型权重: {', '.join(staged_weights)}")

    launcher_cs = config.repo_root / "packaging" / "launcher" / "YoloStudioLauncher.cs"
    icon_path = (
        config.repo_root
        / "third_party"
        / "X-AnyLabeling"
        / "anylabeling"
        / "resources"
        / "images"
        / "icon.ico"
    )
    print("[启动器] 编译无黑框原生桌面启动器 YoloStudio.exe...")
    compile_windows_launcher(launcher_cs, bundle_root / "YoloStudio.exe", icon_file=icon_path)
    write_cli_and_quickstart_files(bundle_root, version=version, variant=config.variant)

    if not config.skip_smoke_test:
        verify_portable_bundle(bundle_root, skip_heavy_imports=config.skip_heavy_imports)

    archive_path = create_bundle_archive(bundle_root, config.archive_format)
    print(f"[完成] 独立便携整合包目录已就绪: {bundle_root}")
    if archive_path is not None:
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"[完成] 发布压缩包已生成: {archive_path} ({size_mb:.1f} MiB)")
    return bundle_root, archive_path


def _isolated_subprocess_env(runtime_dir: Path, bundle_root: Path | None = None) -> dict[str, str]:
    """构造剔除了宿主 Python/Conda 干扰的干净子进程环境变量字典。"""
    env = dict(os.environ)
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "_OLD_VIRTUAL_PATH",
    ):
        env.pop(key, None)

    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if bundle_root is not None:
        env["YOLO_STUDIO_WORKSPACE"] = str(bundle_root / "workspace")
        env["YOLO_STUDIO_WEIGHTS_DIR"] = str(bundle_root / "weights")

    prepend = [
        str(runtime_dir),
        str(runtime_dir / "DLLs"),
        str(runtime_dir / "Library" / "bin"),
        str(runtime_dir / "Scripts"),
    ]
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*prepend, existing_path] if existing_path else prepend)
    return env


def _remove_path(target: Path) -> None:
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="构建 YoloStudio Windows 独立便携整合包（无需目标电脑安装 Python）。"
    )
    parser.add_argument(
        "--mode",
        choices=("from-venv", "clean"),
        default="from-venv",
        help="构建模式：from-venv 离线克隆当前虚拟环境（默认，速度快）；clean 从独立 Python 全新安装依赖（推荐正式发版）。",
    )
    parser.add_argument(
        "--variant",
        choices=("gpu", "cpu"),
        default="gpu",
        help="硬件变体标识：gpu（默认，含 CUDA PyTorch 与 onnxruntime-gpu）或 cpu（轻量版）。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "dist",
        help="构建输出父目录（默认: dist/）。",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=REPO_ROOT / ".venv",
        help="from-venv 模式下读取的源虚拟环境目录（默认: .venv）。",
    )
    parser.add_argument(
        "--standalone-python-dir",
        type=Path,
        default=None,
        help="可选：手动指定本地已解压的独立 Python 根目录（含 python.exe）。",
    )
    parser.add_argument(
        "--standalone-python-archive",
        type=Path,
        default=None,
        help="可选：clean 模式下指定离线下载好的 python-build-standalone (.tar.gz / .zip) 归档文件。",
    )
    parser.add_argument(
        "--python-version",
        default="3.12",
        help="clean 模式下获取的独立 Python 主次版本（默认: 3.12）。",
    )
    parser.add_argument(
        "--torch-index-url",
        default="",
        help="clean 模式下安装 PyTorch 的自定义源地址（留空则 gpu 用 cu126，cpu 用官方 cpu 源）。",
    )
    parser.add_argument(
        "--pip-index-url",
        default="",
        help="clean 模式下常规 pip 包镜像源地址（如清华源 https://pypi.tuna.tsinghua.edu.cn/simple）。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="*",
        default=[],
        help="额外拷入便携包 weights/ 目录的本地 .pt 权重文件路径列表。",
    )
    parser.add_argument(
        "--download-weights",
        nargs="*",
        default=[],
        help="构建阶段自动下载并内置到 weights/ 的官方预训练权重名（例如: yolo26n.pt yolo26s.pt）。",
    )
    parser.add_argument(
        "--fix-opencv",
        action="store_true",
        help="from-venv 模式下联网重装纯 opencv-contrib-python-headless 以消除双包覆盖（clean 模式默认已开启）。",
    )
    parser.add_argument(
        "--archive",
        choices=("none", "7z", "zip"),
        default="none",
        help="构建完成后是否自动打包为单文件压缩包（默认: none，可选 7z 或 zip）。",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="跳过构建后的子进程隔离运行冒烟自检。",
    )
    parser.add_argument(
        "--skip-heavy-imports",
        action="store_true",
        help="冒烟自检时跳过 torch / onnxruntime / ultralytics 重型库导入以缩短测试耗时。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行主入口：解析参数并驱动便携整合包构建与验证。"""
    args = build_arg_parser().parse_args(argv)
    config = BuildConfig(
        repo_root=REPO_ROOT,
        mode=args.mode,
        variant=args.variant,
        output_root=args.output_root,
        venv_dir=args.venv_dir,
        standalone_python_dir=args.standalone_python_dir,
        standalone_python_archive=args.standalone_python_archive,
        python_version=args.python_version,
        torch_index_url=args.torch_index_url,
        pip_index_url=args.pip_index_url,
        extra_weights=list(args.weights),
        download_weights=list(args.download_weights),
        fix_opencv=bool(args.fix_opencv),
        archive_format=args.archive,
        skip_smoke_test=bool(args.skip_smoke_test),
        skip_heavy_imports=bool(args.skip_heavy_imports),
    )
    build_portable_bundle(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
