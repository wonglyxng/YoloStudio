"""独立便携整合包构建逻辑、工作区自动回退与离线权重解析单元测试。

职责：
    验证 `yolo_studio.app.resolve_workspace_root` 与 `report_startup_crash` 在环境变量覆盖、
    便携包 `app/` 目录隔离及只读目录回退下的行为；验证 `core.trainer.resolve_pretrained_weights`
    对预置 `.pt` 权重的检索优先级；验证 `scripts/build_portable.py` 的 `pyvenv.cfg` 解析、
    `site-packages` 绝对路径净化、相对 `.pth` 挂载、源码瘦身复制以及 `YoloStudio.exe` 原生启动器编译。

运行环境与协作边界：
    通过 `pytest` 运行，不访问外部网络，不拷贝数 GB 真实虚拟环境，也不弹出桌面提示框。
"""

from __future__ import annotations

import importlib.util
import os
import site
import sys
from pathlib import Path

import pytest

from yolo_studio.app import (
    WORKSPACE_ENV_VAR,
    report_startup_crash,
    resolve_workspace_root,
)
from yolo_studio.core.trainer import WEIGHTS_DIR_ENV_VAR, resolve_pretrained_weights

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_build_portable_module():
    """按文件路径动态加载 `scripts/build_portable.py` 模块供单元测试直接调用。"""
    script_path = REPO_ROOT / "scripts" / "build_portable.py"
    spec = importlib.util.spec_from_file_location("build_portable", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_portable"] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_workspace_root_env_and_portable_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证环境变量优先，便携包 `app/` 目录自动路由至同级 `workspace/`，常规目录保持原位。"""
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)

    # 常规源码目录：直接使用该目录本身
    dev_root = tmp_path / "repo_dev"
    dev_root.mkdir()
    assert resolve_workspace_root(dev_root) == dev_root.resolve()

    # 独立便携包布局：<bundle>/app + <bundle>/runtime/python.exe -> 自动导向 <bundle>/workspace
    bundle_root = tmp_path / "YoloStudio-Bundle"
    app_dir = bundle_root / "app"
    runtime_dir = bundle_root / "runtime"
    app_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "python.exe").write_bytes(b"MZ")

    resolved_ws = resolve_workspace_root(app_dir)
    assert resolved_ws == (bundle_root / "workspace").resolve()
    assert resolved_ws.is_dir()

    # 显式设置环境变量时覆盖默认规则
    custom_ws = tmp_path / "explicit_workspace"
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(custom_ws))
    assert resolve_workspace_root(app_dir) == custom_ws.resolve()
    assert custom_ws.is_dir()


def test_resolve_workspace_root_falls_back_when_readonly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证候选目录不可写时自动回退至用户个人目录下的可写 `YoloStudio` 工作区。"""
    import yolo_studio.app as app_mod

    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    readonly_dir = (tmp_path / "ProgramFiles_YoloStudio").resolve()
    fake_home = tmp_path / "fake_user_home"
    (fake_home / "Documents").mkdir(parents=True)

    real_check = app_mod._ensure_writable_directory

    def _mock_writable(directory: Path) -> bool:
        if directory.resolve() == readonly_dir:
            return False
        return real_check(directory)

    monkeypatch.setattr(app_mod, "_ensure_writable_directory", _mock_writable)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    fallback = resolve_workspace_root(readonly_dir)
    assert fallback == (fake_home / "Documents" / "YoloStudio").resolve()
    assert fallback.is_dir()


def test_report_startup_crash_writes_log_without_dialog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证启动崩溃记录器在禁用弹窗时将异常堆栈写入 `<workspace>/logs/startup_crash.log`。"""
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    try:
        raise RuntimeError("模拟无控制台启动异常")
    except RuntimeError as exc:
        log_path = report_startup_crash(exc, workspace_root=tmp_path, show_dialog=False)

    assert log_path is not None and log_path.is_file()
    text = log_path.read_text(encoding="utf-8")
    assert "RuntimeError: 模拟无控制台启动异常" in text


def test_resolve_pretrained_weights_priority_and_empty_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证裸模型名优先命中本地预置权重目录，忽略 0 字节空文件，缺失时回退原模型名。"""
    env_weights = tmp_path / "env_weights"
    ws_root = tmp_path / "ws"
    ws_weights = ws_root / "weights"
    env_weights.mkdir(parents=True)
    ws_weights.mkdir(parents=True)

    monkeypatch.setenv(WEIGHTS_DIR_ENV_VAR, str(env_weights))
    monkeypatch.chdir(tmp_path)

    # 0 字节损坏文件不应被当作有效预训练权重
    (env_weights / "yolo26s.pt").write_bytes(b"")
    assert resolve_pretrained_weights("yolo26s.pt", workspace_root=ws_root) == "yolo26s.pt"

    # 工作区 weights 存在有效文件时命中
    ws_pt = ws_weights / "yolo26s.pt"
    ws_pt.write_bytes(b"valid_ws_weights")
    assert resolve_pretrained_weights("yolo26s.pt", workspace_root=ws_root) == str(ws_pt.resolve())

    # 环境变量 YOLO_STUDIO_WEIGHTS_DIR 存在有效文件时优先级高于工作区
    env_pt = env_weights / "yolo26s.pt"
    env_pt.write_bytes(b"valid_env_weights")
    assert resolve_pretrained_weights("yolo26s.pt", workspace_root=ws_root) == str(env_pt.resolve())


def test_build_portable_sanitize_and_relative_pth(tmp_path: Path) -> None:
    """验证 `site-packages` 便携化净化逻辑及相对路径 `.pth` 的实际解析效果。"""
    bp = _load_build_portable_module()

    bundle_root = tmp_path / "bundle"
    sp_dir = bundle_root / "runtime" / "Lib" / "site-packages"
    app_src = bundle_root / "app" / "src"
    app_xal = bundle_root / "app" / "third_party" / "X-AnyLabeling"
    sp_dir.mkdir(parents=True)
    app_src.mkdir(parents=True)
    app_xal.mkdir(parents=True)

    # 构造可编辑安装残留、绝对路径 direct_url.json、测试专用包及正常运行时包
    (sp_dir / "__editable__.x_anylabeling_cvhub-4.3.3.pth").write_text(
        r"C:\work\pyspace\YoloStudio\third_party\X-AnyLabeling", encoding="utf-8"
    )
    (sp_dir / "easy-install.pth").write_text(
        "C:\\dev\\local_pkg\n./relative_pkg\n", encoding="utf-8"
    )
    dist_info = sp_dir / "x_anylabeling_cvhub-4.3.3.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text("Name: x-anylabeling-cvhub\nVersion: 4.3.3\n", encoding="utf-8")
    (dist_info / "direct_url.json").write_text('{"url": "file:///C:/work/repo"}', encoding="utf-8")

    (sp_dir / "pytest").mkdir()
    (sp_dir / "pytest-9.2.0.dist-info").mkdir()
    (sp_dir / "numpy").mkdir()

    bp.sanitize_site_packages(sp_dir)
    pth_path = bp.write_portable_pth(sp_dir)

    assert not (sp_dir / "__editable__.x_anylabeling_cvhub-4.3.3.pth").exists()
    assert not (dist_info / "direct_url.json").exists()
    assert (dist_info / "METADATA").is_file()
    assert not (sp_dir / "pytest").exists()
    assert not (sp_dir / "pytest-9.2.0.dist-info").exists()
    assert (sp_dir / "numpy").is_dir()
    assert (sp_dir / "easy-install.pth").read_text(encoding="utf-8").strip() == "./relative_pkg"
    assert pth_path.is_file()

    known: set[str] = set()
    site.addsitedir(str(sp_dir), known)
    normalized_known = {os.path.normcase(os.path.abspath(p)) for p in known}
    assert os.path.normcase(str(app_src.resolve())) in normalized_known
    assert os.path.normcase(str(app_xal.resolve())) in normalized_known


def test_parse_pyvenv_home_and_slim_app_copy(tmp_path: Path) -> None:
    """验证 `parse_pyvenv_home` 解析与 `copy_app_and_docs` 对大型宣传素材目录的剔除。"""
    bp = _load_build_portable_module()

    fake_home = tmp_path / "base_py"
    fake_home.mkdir()
    (fake_home / "python.exe").write_bytes(b"MZ")
    fake_venv = tmp_path / "venv"
    fake_venv.mkdir()
    (fake_venv / "pyvenv.cfg").write_text(
        f"home = {fake_home}\nversion = 3.12.2\n", encoding="utf-8"
    )
    assert bp.parse_pyvenv_home(fake_venv) == fake_home.resolve()

    bundle_out = tmp_path / "out_bundle"
    bp.copy_app_and_docs(REPO_ROOT, bundle_out)
    assert (bundle_out / "app" / "main.py").is_file()
    assert (bundle_out / "app" / "src" / "yolo_studio" / "app.py").is_file()
    assert (bundle_out / "app" / "src" / "yolo_studio" / "resources" / "logo.ico").is_file()
    assert (bundle_out / "app" / "src" / "yolo_studio" / "resources" / "logo.png").is_file()
    assert (bundle_out / "app" / "third_party" / "X-AnyLabeling" / "anylabeling" / "app_info.py").is_file()
    assert (bundle_out / "LICENSE").is_file()
    assert (bundle_out / "THIRD_PARTY_NOTICES.md").is_file()
    # 确保 70+ MB 的非运行素材与测试套件未被混入便携包
    assert not (bundle_out / "app" / "third_party" / "X-AnyLabeling" / "assets").exists()
    assert not (bundle_out / "app" / "third_party" / "X-AnyLabeling" / "examples").exists()
    assert not (bundle_out / "app" / "third_party" / "X-AnyLabeling" / "tests").exists()


def test_clone_base_python_runtime_preserves_pip_in_clean_mode_and_ensurepip(tmp_path: Path) -> None:
    """验证 `clean` 模式保留归档预置的 `site-packages/pip`，且两种模式均保留 `Lib/ensurepip`。"""
    bp = _load_build_portable_module()

    src_py = tmp_path / "standalone_py"
    (src_py / "Lib" / "site-packages" / "pip").mkdir(parents=True)
    (src_py / "Lib" / "site-packages" / "pip" / "__init__.py").write_text("# pip\n", encoding="utf-8")
    (src_py / "Lib" / "ensurepip").mkdir(parents=True)
    (src_py / "Lib" / "ensurepip" / "__init__.py").write_text("# ensurepip\n", encoding="utf-8")
    (src_py / "python.exe").write_bytes(b"MZ")

    clean_rt = tmp_path / "clean_runtime"
    bp.clone_base_python_runtime(src_py, clean_rt, keep_site_packages=True)
    assert (clean_rt / "Lib" / "site-packages" / "pip" / "__init__.py").is_file()
    assert (clean_rt / "Lib" / "ensurepip" / "__init__.py").is_file()

    venv_rt = tmp_path / "venv_runtime"
    bp.clone_base_python_runtime(src_py, venv_rt, keep_site_packages=False)
    assert not (venv_rt / "Lib" / "site-packages" / "pip").exists()
    assert (venv_rt / "Lib" / "ensurepip" / "__init__.py").is_file()


@pytest.mark.skipif(os.name != "nt", reason="仅在 Windows 上编译原生 GUI 启动器 YoloStudio.exe")
def test_compile_windows_launcher_and_cli_cmd_with_parens_path(tmp_path: Path) -> None:
    """验证 `csc.exe` 编译生成有效 PE `.exe`，且 `YoloStudio-CLI.cmd` 在含空格与圆括号路径下可正常运行。"""
    import subprocess

    bp = _load_build_portable_module()

    launcher_cs = REPO_ROOT / "packaging" / "launcher" / "YoloStudioLauncher.cs"
    icon_file = bp.resolve_launcher_icon_path(REPO_ROOT)
    assert icon_file == REPO_ROOT / "src" / "yolo_studio" / "resources" / "logo.ico"
    bundle_dir = tmp_path / "Program Files (x86)" / "YoloStudio 中文目录"
    out_exe = bundle_dir / "YoloStudio.exe"
    compiled = bp.compile_windows_launcher(launcher_cs, out_exe, icon_file=icon_file)
    assert compiled.is_file()
    header = compiled.read_bytes()[:2]
    assert header == b"MZ"
    assert compiled.stat().st_size > 4096

    # 模拟最小 CLI 入口脚本并用宿主 python 验证含括号+中文+空格目录下的 YoloStudio-CLI.cmd
    (bundle_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "app").mkdir(parents=True, exist_ok=True)
    bp.write_cli_and_quickstart_files(bundle_dir, version="0.2.0", variant="gpu")
    # 将当前测试 python 最小复制或制作假 cli.py 验证批处理参数透传无语法错误
    (bundle_dir / "app" / "cli.py").write_text(
        "import sys; print('CLI_ARGS_OK:' + ','.join(sys.argv[1:]))\n", encoding="utf-8"
    )
    base_py = Path(sys.base_prefix)
    for name in ("python.exe", "python3.dll"):
        src = base_py / name
        if src.is_file():
            (bundle_dir / "runtime" / name).write_bytes(src.read_bytes())
    for dll in base_py.glob("python3*.dll"):
        (bundle_dir / "runtime" / dll.name).write_bytes(dll.read_bytes())
    for dll in base_py.glob("vcruntime*.dll"):
        (bundle_dir / "runtime" / dll.name).write_bytes(dll.read_bytes())

    # 使用目录联接（Junction，无需管理员或开发者模式特权）挂载基础标准库以完成真实解释器启动测试
    junction_dst = bundle_dir / "runtime" / "Lib"
    subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(junction_dst), str(base_py / "Lib")],
        check=True,
        capture_output=True,
    )
    try:
        cmd_str = f'cmd.exe /d /s /c ""{bundle_dir / "YoloStudio-CLI.cmd"}" extract --help"'
        res = subprocess.run(
            cmd_str,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
        assert "CLI_ARGS_OK:extract,--help" in res.stdout
    finally:
        # 先移除目录联接本身，避免清理临时目录时跟随进入宿主标准库
        subprocess.run(["cmd.exe", "/c", "rmdir", str(junction_dst)], capture_output=True)


def test_strip_source_to_pyc_only_preserves_whitelist_and_imports(tmp_path: Path) -> None:
    """验证 `strip_source_to_pyc_only` 将常规模块转为同级 `.pyc` 并删除 `.py`，同时保留白名单文件且可正常导入。"""
    import subprocess

    bp = _load_build_portable_module()

    pkg_root = tmp_path / "pkg_root"
    mod_dir = pkg_root / "demo_pkg"
    cfg_dir = mod_dir / "configs"
    cfg_dir.mkdir(parents=True)
    (mod_dir / "__init__.py").write_text("# comment\n", encoding="utf-8")
    (mod_dir / "logic.py").write_text("VALUE = 42\n", encoding="utf-8")
    (cfg_dir / "model_cfg.py").write_text("CFG = 'ok'\n", encoding="utf-8")

    count = bp.strip_source_to_pyc_only(
        mod_dir,
        Path(sys.executable),
        keep_py_rel_markers=("configs/",),
        keep_empty_init_py=True,
    )
    assert count == 1
    assert not (mod_dir / "logic.py").exists()
    assert (mod_dir / "logic.pyc").is_file()
    assert (mod_dir / "__init__.py").read_text(encoding="utf-8") == ""
    assert (cfg_dir / "model_cfg.py").is_file()
    assert not any(mod_dir.rglob("__pycache__"))

    env = dict(os.environ)
    env["PYTHONPATH"] = str(pkg_root)
    res = subprocess.run(
        [sys.executable, "-c", "import demo_pkg.logic as m; print(m.VALUE)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "42"


@pytest.mark.skipif(os.name != "nt", reason="仅在 Windows 上测试原生启动器防篡改校验")
def test_integrity_manifest_tamper_detection_and_runtime_cleanup(tmp_path: Path) -> None:
    """验证清理 `runtime/Scripts` 及编译进 `YoloStudio.exe` 的 SHA-256 防篡改校验（完好放行、改动拦截）。"""
    import subprocess

    bp = _load_build_portable_module()

    bundle = tmp_path / "protected_bundle"
    (bundle / "runtime" / "Scripts").mkdir(parents=True)
    (bundle / "runtime" / "Scripts" / "yolo.exe").write_bytes(b"fake_wrapper")
    (bundle / "runtime" / "share" / "man").mkdir(parents=True)
    (bundle / "runtime" / "Lib" / "site-packages").mkdir(parents=True)
    bp.write_portable_pth(bundle / "runtime" / "Lib" / "site-packages")

    pkg_dir = bundle / "app" / "src" / "yolo_studio"
    pkg_dir.mkdir(parents=True)
    (bundle / "app" / "main.py").write_text("print('main')\n", encoding="utf-8")
    (bundle / "app" / "cli.py").write_text("print('cli')\n", encoding="utf-8")
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    fake_pyd = pkg_dir / "app.cp312-win_amd64.pyd"
    fake_pyd.write_bytes(b"MZ_FAKE_BINARY_PAYLOAD")

    removed = bp.cleanup_runtime_artifacts(bundle / "runtime")
    assert "Scripts" in removed
    assert not (bundle / "runtime" / "Scripts").exists()

    manifest_cs = tmp_path / "IntegrityManifest.g.cs"
    count = bp.generate_integrity_manifest_cs(bundle, manifest_cs, version="0.3.0", include_integrity=True)
    assert count == 5

    launcher_cs = REPO_ROOT / "packaging" / "launcher" / "YoloStudioLauncher.cs"
    out_exe = bundle / "YoloStudio.exe"
    bp.compile_windows_launcher(launcher_cs, out_exe, manifest_cs=manifest_cs)

    # 校验 Windows PE 文件版本属性已与单一版本源自动对齐
    ver_probe = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"$v=(Get-Item -LiteralPath '{out_exe}').VersionInfo; Write-Output \"$($v.FileVersion)|$($v.ProductVersion)\"",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert ver_probe.stdout.strip() == "0.3.0.0|0.3.0"

    # 1) 文件完好时校验放行（返回码 0）
    ok_res = subprocess.run([str(out_exe), "--verify-only"], capture_output=True)
    assert ok_res.returncode == 0

    # 2) 篡改入口 main.py 任意字节后立即拦截（返回码 2）
    orig_main = (bundle / "app" / "main.py").read_text(encoding="utf-8")
    (bundle / "app" / "main.py").write_text(orig_main + "# tampered\n", encoding="utf-8")
    tamper_main = subprocess.run([str(out_exe), "--verify-only"], capture_output=True)
    assert tamper_main.returncode == 2
    (bundle / "app" / "main.py").write_text(orig_main, encoding="utf-8")

    # 3) 篡改核心 .pyd 二进制任意字节后立即拦截（返回码 2）
    fake_pyd.write_bytes(b"MZ_MODIFIED_PAYLOAD!!!")
    tamper_pyd = subprocess.run([str(out_exe), "--verify-only"], capture_output=True)
    assert tamper_pyd.returncode == 2


def test_single_source_version_read_write_bump_and_cli_version(tmp_path: Path) -> None:
    """验证单一版本源读写、`patch`/`minor`/`major` 递增与 CLI `--version` 输出一致。"""
    import subprocess
    import yolo_studio

    bp = _load_build_portable_module()

    repo_ver = bp.read_project_version(REPO_ROOT)
    assert yolo_studio.__version__ == repo_ver

    # 隔离临时目录验证版本递增与格式校验
    fake_repo = tmp_path / "repo"
    ver_file = fake_repo / "src" / "yolo_studio" / "_version.py"
    ver_file.parent.mkdir(parents=True)
    ver_file.write_text('__version__ = "0.3.0"\n', encoding="utf-8")

    assert bp.bump_project_version(fake_repo, "patch") == "0.3.1"
    assert bp.bump_project_version(fake_repo, "minor") == "0.4.0"
    assert bp.bump_project_version(fake_repo, "major") == "1.0.0"
    assert bp.read_project_version(fake_repo) == "1.0.0"
    assert bp._to_windows_four_part_version("1.2.3-rc1") == "1.2.3.0"

    with pytest.raises(ValueError):
        bp.write_project_version(fake_repo, "bad_version")

    cli_res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "cli.py"), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"YoloStudio v{repo_ver}" in (cli_res.stdout + cli_res.stderr)


@pytest.mark.skipif(os.name != "nt", reason="仅在 Windows 上测试补丁包覆盖升级后的原生启动器完整性校验")
def test_create_patch_archive_overlays_cleanly_and_passes_integrity_check(tmp_path: Path) -> None:
    """验证 `create_patch_archive` 仅打包轻量增量组件，覆盖旧版目录后保留用户数据且通过防篡改校验。"""
    import subprocess
    import zipfile

    bp = _load_build_portable_module()

    # 1) 模拟新构建完成的受保护便携包（含大体量 runtime 假库）
    new_bundle = tmp_path / "dist" / "YoloStudio-v0.3.1-win64-cpu"
    sp_dir = new_bundle / "runtime" / "Lib" / "site-packages"
    sp_dir.mkdir(parents=True)
    bp.write_portable_pth(sp_dir)
    (sp_dir / "huge_torch_placeholder.bin").write_bytes(b"X" * 1024)
    (new_bundle / "app" / "src" / "yolo_studio").mkdir(parents=True)
    (new_bundle / "app" / "main.py").write_text("print('v0.3.1')\n", encoding="utf-8")
    (new_bundle / "app" / "cli.py").write_text("print('cli')\n", encoding="utf-8")
    (new_bundle / "app" / "src" / "yolo_studio" / "app.cp312-win_amd64.pyd").write_bytes(b"NEW_PYD")
    (new_bundle / "docs").mkdir(parents=True)
    (new_bundle / "docs" / "001.md").write_text("# doc\n", encoding="utf-8")
    (new_bundle / "LICENSE").write_text("GPL", encoding="utf-8")

    manifest_cs = tmp_path / "IntegrityManifest.g.cs"
    bp.generate_integrity_manifest_cs(new_bundle, manifest_cs, version="0.3.1", include_integrity=True)
    bp.compile_windows_launcher(
        REPO_ROOT / "packaging" / "launcher" / "YoloStudioLauncher.cs",
        new_bundle / "YoloStudio.exe",
        manifest_cs=manifest_cs,
    )

    patch_zip = bp.create_patch_archive(new_bundle, version="0.3.1", archive_format="zip")
    assert patch_zip.is_file()
    with zipfile.ZipFile(patch_zip, "r") as zf:
        names = set(zf.namelist())
        assert "YoloStudio.exe" in names
        assert "app/main.py" in names
        assert "app/src/yolo_studio/app.cp312-win_amd64.pyd" in names
        assert "runtime/Lib/site-packages/yolo_studio_portable.pth" in names
        assert "升级说明_覆盖解压到旧版根目录即可.txt" in names
        assert not any("huge_torch_placeholder" in n for n in names)

    # 2) 模拟旧版 v0.3.0 目录（含用户历史 workspace 数据），覆盖解压补丁后校验完好
    old_bundle = tmp_path / "old_user_install"
    (old_bundle / "workspace" / "runs").mkdir(parents=True)
    user_data = old_bundle / "workspace" / "runs" / "best.pt"
    user_data.write_bytes(b"USER_TRAINED_MODEL")
    (old_bundle / "app" / "main.py").parent.mkdir(parents=True, exist_ok=True)
    (old_bundle / "app" / "main.py").write_text("print('old v0.3.0')\n", encoding="utf-8")

    with zipfile.ZipFile(patch_zip, "r") as zf:
        zf.extractall(old_bundle)

    assert user_data.read_bytes() == b"USER_TRAINED_MODEL"
    assert "v0.3.1" in (old_bundle / "app" / "main.py").read_text(encoding="utf-8")
    verify_res = subprocess.run([str(old_bundle / "YoloStudio.exe"), "--verify-only"], capture_output=True)
    assert verify_res.returncode == 0


def test_rewrite_connect_calls_for_cython_and_slot_bridge_behavior() -> None:
    """验证 AST 重写严格保持 `__future__` 置顶并包裹 `.connect()`，且桥接器能截断多余信号参数。"""
    bp = _load_build_portable_module()

    # 1) 无 .connect() 调用的模块不做改动
    plain_src = '"""doc"""\nfrom __future__ import annotations\nx = 1\n'
    out_plain, changed_plain = bp.rewrite_connect_calls_for_cython(plain_src)
    assert changed_plain is False
    assert out_plain == plain_src

    # 2) 含多种 .connect() 的模块应正确注入导入于 __future__ 之后并仅包裹首参
    gui_src = (
        '"""模块文档"""\n'
        "from __future__ import annotations\n"
        "def wire(btn, sig, conn_type):\n"
        "    btn.clicked.connect(self._on_browse)\n"
        "    btn.clicked.connect(lambda: 123)\n"
        "    sig.connect(self._on_ready, conn_type)\n"
    )
    rewritten, changed = bp.rewrite_connect_calls_for_cython(gui_src)
    assert changed is True
    lines = [line.strip() for line in rewritten.splitlines() if line.strip()]
    assert lines[1] == "from __future__ import annotations"
    assert lines[2] == "from yolo_studio._qt_slot_bridge import adapt_slot as _ys_adapt_slot"
    assert "_ys_adapt_slot(self._on_browse)" in rewritten
    assert "_ys_adapt_slot(lambda: 123)" in rewritten
    assert "sig.connect(_ys_adapt_slot(self._on_ready), conn_type)" in rewritten
    compile(rewritten, "<rewritten_gui>", "exec")

    # 3) 验证桥接模块 adapt_slot 对非 PyFunction 可调用对象（模拟 Cython cyfunction）的参数截断与透传
    ns: dict[str, object] = {}
    exec(bp._QT_SLOT_BRIDGE_SOURCE, ns)
    adapt_slot = ns["adapt_slot"]
    assert callable(adapt_slot)

    # 非可调用对象（如 sqlite3.connect("a.db") 的字符串参数）原样透传
    assert adapt_slot("demo.db") == "demo.db"

    calls: list[tuple[object, ...]] = []

    class _ZeroArgCallable:
        def __call__(self) -> str:
            calls.append(())
            return "zero_ok"

    class _TwoArgCallable:
        def __call__(self, a: int, b: str = "def") -> tuple[int, str]:
            calls.append((a, b))
            return (a, b)

    class _VarArgsCallable:
        def __call__(self, *args: object) -> tuple[object, ...]:
            calls.append(args)
            return args

    wrapped_zero = adapt_slot(_ZeroArgCallable())
    assert wrapped_zero(False) == "zero_ok"
    assert wrapped_zero(False, 99) == "zero_ok"

    wrapped_two = adapt_slot(_TwoArgCallable())
    assert wrapped_two(7, "hi", "extra") == (7, "hi")
    assert wrapped_two(8) == (8, "def")

    var_obj = _VarArgsCallable()
    assert adapt_slot(var_obj) is var_obj


def test_install_runtime_excepthook_logs_without_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证全局运行时异常钩子在禁用弹窗时将未捕获异常追加写入 `logs/runtime_error.log`。"""
    from yolo_studio.app import install_runtime_excepthook

    orig_hook = sys.excepthook
    try:
        install_runtime_excepthook(workspace_root=tmp_path, show_dialog=False)
        try:
            raise ValueError("模拟槽函数异常")
        except ValueError:
            exc_type, exc_val, exc_tb = sys.exc_info()
            assert exc_type is not None
            sys.excepthook(exc_type, exc_val, exc_tb)
    finally:
        sys.excepthook = orig_hook

    log_file = tmp_path / "logs" / "runtime_error.log"
    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "ValueError: 模拟槽函数异常" in content


def test_fast_copy_tree_filters_pycache_and_preserves_when_empty_exclusions(tmp_path: Path) -> None:
    """验证 `_fast_copy_tree` 默认排除缓存字节码，且在底座复用空排除集合下完整保留 `.pyc`。"""
    bp = _load_build_portable_module()

    src = tmp_path / "src_tree"
    (src / "pkg" / "__pycache__").mkdir(parents=True)
    (src / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    (src / "pkg" / "stale.pyc").write_bytes(b"PYC_TOP")
    (src / "pkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"PYC_CACHE")

    # 1) 默认模式：排除 __pycache__、.pyc 与 .pyo
    dst_filtered = tmp_path / "dst_filtered"
    bp._fast_copy_tree(src, dst_filtered)
    assert (dst_filtered / "pkg" / "mod.py").is_file()
    assert not (dst_filtered / "pkg" / "stale.pyc").exists()
    assert not (dst_filtered / "pkg" / "__pycache__").exists()

    # 2) 底座复用模式：空排除集合时连同预编译 .pyc 完整保留
    dst_full = tmp_path / "dst_full"
    bp._fast_copy_tree(src, dst_full, exclude_names_lower=frozenset(), exclude_suffixes_lower=())
    assert (dst_full / "pkg" / "mod.py").is_file()
    assert (dst_full / "pkg" / "stale.pyc").read_bytes() == b"PYC_TOP"
    assert (dst_full / "pkg" / "__pycache__" / "mod.cpython-312.pyc").read_bytes() == b"PYC_CACHE"


def test_runtime_fingerprint_ignores_version_changes_and_tracks_dependencies(tmp_path: Path) -> None:
    """验证底座依赖指纹与项目版本号完全解耦，但对依赖、变体及 venv 三方包变化保持敏感。"""
    bp = _load_build_portable_module()

    fake_repo = tmp_path / "repo"
    (fake_repo / "src" / "yolo_studio").mkdir(parents=True)
    (fake_repo / "third_party" / "X-AnyLabeling").mkdir(parents=True)
    ver_file = fake_repo / "src" / "yolo_studio" / "_version.py"
    ver_file.write_text('__version__ = "0.3.3"\n', encoding="utf-8")
    (fake_repo / "requirements.txt").write_text("# comment\nnumpy>=2.0.0\n", encoding="utf-8")
    (fake_repo / "pyproject.toml").write_text(
        '[project]\nname = "yolo-studio"\nrequires-python = ">=3.11"\ndependencies = ["numpy>=2.0.0"]\n',
        encoding="utf-8",
    )
    (fake_repo / "third_party" / "X-AnyLabeling" / "pyproject.toml").write_text(
        '[project]\nname = "x-anylabeling"\ndependencies = ["PyQt6>=6.6.0"]\n',
        encoding="utf-8",
    )

    fake_venv = tmp_path / "venv"
    sp = fake_venv / "Lib" / "site-packages"
    sp.mkdir(parents=True)
    (fake_venv / "pyvenv.cfg").write_text("home = C:\\Python312\nversion = 3.12.2\n", encoding="utf-8")
    (sp / "numpy-2.1.0.dist-info").mkdir()
    (sp / "numpy-2.1.0.dist-info" / "METADATA").write_text("Name: numpy\nVersion: 2.1.0\n", encoding="utf-8")
    (sp / "yolo_studio-0.3.3.dist-info").mkdir()
    (sp / "pytest-9.0.0.dist-info").mkdir()

    cfg = bp.BuildConfig(repo_root=fake_repo, mode="from-venv", variant="gpu", venv_dir=fake_venv)
    fp_base = bp.compute_runtime_fingerprint(cfg)

    # 1) 仅修改版本号、自身 editable dist-info 或开发测试包 pytest，指纹应保持完全不变
    ver_file.write_text('__version__ = "0.3.4"\n', encoding="utf-8")
    (sp / "yolo_studio-0.3.3.dist-info").rename(sp / "yolo_studio-0.3.4.dist-info")
    (sp / "pytest-9.0.0.dist-info").rename(sp / "pytest-9.1.0.dist-info")
    assert bp.compute_runtime_fingerprint(cfg)["digest"] == fp_base["digest"]

    # 2) 变更硬件变体（gpu -> cpu），指纹必须变化
    cfg_cpu = bp.BuildConfig(repo_root=fake_repo, mode="from-venv", variant="cpu", venv_dir=fake_venv)
    assert bp.compute_runtime_fingerprint(cfg_cpu)["digest"] != fp_base["digest"]

    # 3) 变更 requirements.txt 实质依赖，指纹必须变化
    (fake_repo / "requirements.txt").write_text("numpy>=2.1.0\nPillow>=10.0.0\n", encoding="utf-8")
    fp_after_req = bp.compute_runtime_fingerprint(cfg)
    assert fp_after_req["digest"] != fp_base["digest"]

    # 4) 升级 venv 中真实运行时第三方包版本，指纹必须变化
    (sp / "numpy-2.1.0.dist-info" / "METADATA").write_text("Name: numpy\nVersion: 2.2.0\n", encoding="utf-8")
    assert bp.compute_runtime_fingerprint(cfg)["digest"] != fp_after_req["digest"]


def test_find_reusable_runtime_priority_and_validation(tmp_path: Path) -> None:
    """验证 `find_reusable_runtime` 的优先级（原地 > 缓存 > 历史构建）与无效目录拒绝规则。"""
    bp = _load_build_portable_module()

    def _make_runtime(rt_dir: Path, stamp: dict[str, object] | None, *, with_scripts: bool = False) -> None:
        (rt_dir / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
        (rt_dir / "python.exe").write_bytes(b"MZ")
        if with_scripts:
            (rt_dir / "Scripts").mkdir(exist_ok=True)
        if stamp is not None:
            bp.write_runtime_stamp(rt_dir, stamp)

    out_root = tmp_path / "dist"
    cache_dir = tmp_path / "cache"
    cfg = bp.BuildConfig(
        repo_root=REPO_ROOT,
        mode="clean",
        variant="cpu",
        output_root=out_root,
        cache_dir=cache_dir,
    )
    fp = bp.compute_runtime_fingerprint(cfg)
    stale_fp = {"schema_version": 1, "digest": "0" * 64, "details": {}}
    target_bundle = out_root / "YoloStudio-v0.3.4-win64-cpu"

    # 仅存在指纹过期与含残留 Scripts 的历史目录时应返回 None
    _make_runtime(out_root / "YoloStudio-v0.3.1-win64-cpu" / "runtime", stale_fp)
    _make_runtime(out_root / "YoloStudio-v0.3.2-win64-cpu" / "runtime", fp, with_scripts=True)
    assert bp.find_reusable_runtime(cfg, target_bundle, expected_fingerprint=fp) is None

    # 存在合法同变体历史目录时命中历史目录
    hist_rt = out_root / "YoloStudio-v0.3.3-win64-cpu" / "runtime"
    _make_runtime(hist_rt, fp)
    found = bp.find_reusable_runtime(cfg, target_bundle, expected_fingerprint=fp)
    assert found is not None and found[0] == hist_rt.resolve()

    # 存在底座缓存时优先级高于历史目录
    cache_rt = cache_dir / "runtimes" / "win64-cpu-clean"
    _make_runtime(cache_rt, fp)
    found_cache = bp.find_reusable_runtime(cfg, target_bundle, expected_fingerprint=fp)
    assert found_cache is not None and found_cache[0] == cache_rt.resolve()

    # 目标输出目录自身已含合法底座时优先级最高（原地零拷贝复用）
    inplace_rt = target_bundle / "runtime"
    _make_runtime(inplace_rt, fp)
    found_inplace = bp.find_reusable_runtime(cfg, target_bundle, expected_fingerprint=fp)
    assert found_inplace is not None and found_inplace[0] == inplace_rt.resolve()


def test_compression_presets_and_arg_parser(tmp_path: Path) -> None:
    """验证 `fast`/`normal`/`max` 压缩档位校验、ZIP 归档输出与新 CLI 参数解析。"""
    import zipfile

    bp = _load_build_portable_module()

    bundle = tmp_path / "YoloStudio-v0.3.3-win64-cpu"
    (bundle / "app").mkdir(parents=True)
    (bundle / "app" / "hello.txt").write_text("hello yolo studio\n" * 100, encoding="utf-8")

    for level in ("fast", "normal", "max"):
        archive = bp.create_bundle_archive(bundle, "zip", compression=level)
        assert archive is not None and archive.is_file()
        with zipfile.ZipFile(archive, "r") as zf:
            assert "YoloStudio-v0.3.3-win64-cpu/app/hello.txt" in zf.namelist()

    with pytest.raises(ValueError, match="不支持的压缩档位"):
        bp.create_bundle_archive(bundle, "zip", compression="ultra")

    parsed = bp.build_arg_parser().parse_args(
        [
            "--variant",
            "cpu",
            "--compression",
            "fast",
            "--reuse-runtime",
            "--patch-only",
            "--runtime-source",
            str(tmp_path),
        ]
    )
    assert parsed.compression == "fast"
    assert parsed.reuse_runtime is True
    assert parsed.patch_only is True
    assert parsed.runtime_source == tmp_path


def _create_minimal_fake_repo(repo_dir: Path, version: str = "0.3.9") -> Path:
    """在临时目录中构造可供完整打包/补丁打包离线单测使用的精简模拟仓库。"""
    import shutil

    (repo_dir / "src" / "yolo_studio").mkdir(parents=True)
    (repo_dir / "src" / "yolo_studio" / "_version.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (repo_dir / "src" / "yolo_studio" / "__init__.py").write_text(
        "from yolo_studio._version import __version__\n", encoding="utf-8"
    )
    (repo_dir / "main.py").write_text("print('main')\n", encoding="utf-8")
    (repo_dir / "cli.py").write_text("print('cli')\n", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text(
        '[project]\nname = "yolo-studio"\ndependencies = ["numpy>=2.0.0"]\n', encoding="utf-8"
    )
    (repo_dir / "requirements.txt").write_text("numpy>=2.0.0\n", encoding="utf-8")
    (repo_dir / "LICENSE").write_text("GPL-3.0-only\n", encoding="utf-8")
    (repo_dir / "THIRD_PARTY_NOTICES.md").write_text("# Notices\n", encoding="utf-8")
    (repo_dir / "docs").mkdir(parents=True)
    (repo_dir / "docs" / "001.md").write_text("# Doc\n", encoding="utf-8")
    (repo_dir / "third_party" / "X-AnyLabeling" / "anylabeling").mkdir(parents=True)
    (repo_dir / "third_party" / "X-AnyLabeling" / "anylabeling" / "__init__.py").write_text(
        "", encoding="utf-8"
    )
    (repo_dir / "third_party" / "X-AnyLabeling" / "pyproject.toml").write_text(
        '[project]\nname = "x-anylabeling"\ndependencies = []\n', encoding="utf-8"
    )
    launcher_dst = repo_dir / "packaging" / "launcher"
    launcher_dst.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "packaging" / "launcher" / "YoloStudioLauncher.cs",
        launcher_dst / "YoloStudioLauncher.cs",
    )
    return repo_dir


@pytest.mark.skipif(os.name != "nt", reason="仅在 Windows 上测试目录联接挂载与原生启动器补丁构建")
def test_build_patch_only_mounts_junction_readonly_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 `build_patch_only` 通过目录联接只读挂载 `runtime/`、对源目录零写入并正确输出补丁包。"""
    import zipfile

    bp = _load_build_portable_module()

    fake_repo = _create_minimal_fake_repo(tmp_path / "fake_repo", version="0.3.9")
    out_root = tmp_path / "dist"
    existing_rt = out_root / "YoloStudio-v0.3.8-win64-cpu" / "runtime"
    sp = existing_rt / "Lib" / "site-packages"
    sp.mkdir(parents=True)
    (existing_rt / "python.exe").write_bytes(b"MZ")
    (sp / "heavy_pkg.bin").write_bytes(b"DO_NOT_INCLUDE_IN_PATCH")
    bp.write_portable_pth(sp)

    before_snapshot = {
        p.relative_to(existing_rt).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(existing_rt.rglob("*"))
        if p.is_file()
    }

    observed_mounts: list[tuple[Path, bool]] = []

    def _capture_precompile(bundle_root: Path, protect: str = "none", include_runtime: bool = True) -> None:
        # 验证补丁模式确实是通过目录联接挂载 runtime 且不扫描 runtime/Lib
        assert include_runtime is False
        rt_mount = bundle_root / "runtime"
        observed_mounts.append((bundle_root, bp._is_directory_junction(rt_mount)))

    monkeypatch.setattr(bp, "precompile_bytecode", _capture_precompile)

    cfg = bp.BuildConfig(
        repo_root=fake_repo,
        mode="clean",
        variant="cpu",
        protect="none",
        output_root=out_root,
        archive_format="zip",
        compression="fast",
        patch_only=True,
        skip_smoke_test=True,
    )
    patch_zip = bp.build_patch_only(cfg)

    assert len(observed_mounts) == 1
    stage_dir_used, was_junction = observed_mounts[0]
    assert was_junction is True
    # 验证临时工作区与目录联接均已安全卸载并清理
    assert not stage_dir_used.exists()

    assert patch_zip == (out_root / "YoloStudio-v0.3.9-win64-patch.zip").resolve()
    assert patch_zip.is_file()
    # 确认没有额外构建完整目录
    assert not (out_root / "YoloStudio-v0.3.9-win64-cpu").exists()

    # 确认源 runtime 严格只读零变动
    after_snapshot = {
        p.relative_to(existing_rt).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(existing_rt.rglob("*"))
        if p.is_file()
    }
    assert after_snapshot == before_snapshot

    with zipfile.ZipFile(patch_zip, "r") as zf:
        names = set(zf.namelist())
        assert "YoloStudio.exe" in names
        assert "YoloStudio-CLI.cmd" in names
        assert "app/main.py" in names
        assert "runtime/Lib/site-packages/yolo_studio_portable.pth" in names
        assert "升级说明_覆盖解压到旧版根目录即可.txt" in names
        assert not any("heavy_pkg.bin" in n for n in names)


@pytest.mark.skipif(os.name != "nt", reason="仅在 Windows 上测试带启动器编译的底座复用构建流水线")
def test_reuse_runtime_cross_dir_and_inplace_skips_reinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """验证 `--reuse-runtime` 在跨版本克隆与同版本原地重建时均跳过重装与全量字节码编译，且不留旧应用脏文件。"""
    bp = _load_build_portable_module()

    fake_repo = _create_minimal_fake_repo(tmp_path / "fake_repo", version="0.3.9")
    out_root = tmp_path / "dist"
    cache_dir = tmp_path / "cache"

    cfg = bp.BuildConfig(
        repo_root=fake_repo,
        mode="clean",
        variant="cpu",
        protect="none",
        output_root=out_root,
        cache_dir=cache_dir,
        reuse_runtime=True,
        skip_smoke_test=True,
    )
    fp = bp.compute_runtime_fingerprint(cfg)

    # 构造上一个版本的带印章且已预编译 .pyc 的历史底座
    old_rt = out_root / "YoloStudio-v0.3.8-win64-cpu" / "runtime"
    (old_rt / "Lib" / "site-packages" / "__pycache__").mkdir(parents=True)
    (old_rt / "python.exe").write_bytes(b"MZ")
    (old_rt / "Lib" / "site-packages" / "__pycache__" / "precompiled.cpython-312.pyc").write_bytes(b"PYC")
    bp.write_portable_pth(old_rt / "Lib" / "site-packages")
    bp.write_runtime_stamp(old_rt, fp)

    def _forbid_prepare(*args: object, **kwargs: object) -> None:
        raise AssertionError("命中可复用底座时不应重新执行运行时安装")

    precompile_calls: list[tuple[str, bool]] = []

    def _record_precompile(bundle_root: Path, protect: str = "none", include_runtime: bool = True) -> None:
        precompile_calls.append((protect, include_runtime))

    monkeypatch.setattr(bp, "prepare_clean_runtime", _forbid_prepare)
    monkeypatch.setattr(bp, "prepare_from_venv_runtime", _forbid_prepare)
    monkeypatch.setattr(bp, "precompile_bytecode", _record_precompile)

    # 1) 跨版本复用（v0.3.8 -> v0.3.9）
    bundle_dir, archive = bp.build_portable_bundle(cfg)
    assert archive is None
    assert bundle_dir == (out_root / "YoloStudio-v0.3.9-win64-cpu").resolve()
    assert (bundle_dir / "YoloStudio.exe").is_file()
    assert (
        bundle_dir / "runtime" / "Lib" / "site-packages" / "__pycache__" / "precompiled.cpython-312.pyc"
    ).read_bytes() == b"PYC"
    assert bp.read_runtime_stamp(bundle_dir / "runtime")["digest"] == fp["digest"]
    assert precompile_calls == [("none", False)]

    # 2) 原地复用（重复构建 v0.3.9）：应保留 runtime 与 weights，清除 app 下陈旧残留文件
    stale_app_file = bundle_dir / "app" / "stale_old_module.py"
    stale_app_file.write_text("# stale\n", encoding="utf-8")
    keep_weight = bundle_dir / "weights" / "custom.pt"
    keep_weight.write_bytes(b"WEIGHT_KEEP")
    sentinel_rt = bundle_dir / "runtime" / "inplace_sentinel.txt"
    sentinel_rt.write_text("alive", encoding="utf-8")

    bp.build_portable_bundle(cfg)
    assert sentinel_rt.is_file()
    assert keep_weight.read_bytes() == b"WEIGHT_KEEP"
    assert not stale_app_file.exists()
    assert precompile_calls == [("none", False), ("none", False)]


def test_app_logo_assets_resolution_and_splash_render(tmp_path: Path) -> None:
    """验证自有 LOGO 资产规格、图标解析优先级（自有优先、缺失回退）及高分屏启动图渲染。"""
    from PIL import Image
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication

    from yolo_studio.app import (
        APP_SPLASH_SUBTITLE,
        _build_app_icon,
        _find_app_icon_path,
        _find_app_logo_path,
        _render_startup_splash_pixmap,
    )

    bp = _load_build_portable_module()

    own_ico = REPO_ROOT / "src" / "yolo_studio" / "resources" / "logo.ico"
    own_png = REPO_ROOT / "src" / "yolo_studio" / "resources" / "logo.png"
    assert own_ico.is_file()
    assert own_png.is_file()

    # 1) 校验 PNG 主图与多尺寸 ICO 帧完整性
    with Image.open(own_png) as png_img:
        assert png_img.mode == "RGBA"
        assert png_img.size == (512, 512)
        # 四角为透明背景，说明圆角图标外无白底残留
        assert png_img.getpixel((0, 0))[3] == 0
        assert png_img.getpixel((511, 511))[3] == 0

    with Image.open(own_ico) as ico_img:
        sizes = set(ico_img.info.get("sizes", set()))
        expected_sizes = {(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)}
        assert expected_sizes.issubset(sizes)

    # 2) 校验正常仓库环境优先命中 YoloStudio 自有图标
    pkg_dir = REPO_ROOT / "src" / "yolo_studio"
    assert _find_app_icon_path(REPO_ROOT, REPO_ROOT, package_dir=pkg_dir) == own_ico
    assert _find_app_logo_path(REPO_ROOT, REPO_ROOT, package_dir=pkg_dir) == own_png
    assert bp.resolve_launcher_icon_path(REPO_ROOT) == own_ico

    # 3) 校验缺失自有图标时回退至 X-AnyLabeling 图标；均缺失时返回 None
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    assert _find_app_icon_path(empty_repo, empty_repo) is None
    assert _find_app_logo_path(empty_repo, empty_repo) is None
    assert bp.resolve_launcher_icon_path(empty_repo) is None

    fallback_ico = (
        empty_repo
        / "third_party"
        / "X-AnyLabeling"
        / "anylabeling"
        / "resources"
        / "images"
        / "icon.ico"
    )
    fallback_ico.parent.mkdir(parents=True)
    fallback_ico.write_bytes(own_ico.read_bytes())
    assert _find_app_icon_path(empty_repo, empty_repo) == fallback_ico
    assert bp.resolve_launcher_icon_path(empty_repo) == fallback_ico

    # 4) 校验离线 Qt 环境下 QIcon 构建与 1x/2x 高分屏启动欢迎图绘制
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        app = QApplication([])

    qicon = _build_app_icon(own_ico, logo_path=own_png)
    assert qicon is not None and not qicon.isNull()
    avail = {(s.width(), s.height()) for s in qicon.availableSizes()}
    assert (16, 16) in avail and (256, 256) in avail and (512, 512) in avail

    assert APP_SPLASH_SUBTITLE == "一站式 YOLO26 抽帧、标注清洗与训练导出工作台"
    pm_1x = _render_startup_splash_pixmap(1.0, own_ico, logo_path=own_png)
    pm_2x = _render_startup_splash_pixmap(2.0, own_ico, logo_path=own_png)
    assert not pm_1x.isNull() and pm_1x.width() == 508 and pm_1x.height() == 210
    assert not pm_2x.isNull() and pm_2x.width() == 1016 and pm_2x.height() == 420
    assert pm_2x.devicePixelRatio() == 2.0
