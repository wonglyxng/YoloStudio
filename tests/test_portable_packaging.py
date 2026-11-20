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
    icon_file = (
        REPO_ROOT
        / "third_party"
        / "X-AnyLabeling"
        / "anylabeling"
        / "resources"
        / "images"
        / "icon.ico"
    )
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
    count = bp.generate_integrity_manifest_cs(bundle, manifest_cs)
    assert count == 5

    launcher_cs = REPO_ROOT / "packaging" / "launcher" / "YoloStudioLauncher.cs"
    out_exe = bundle / "YoloStudio.exe"
    bp.compile_windows_launcher(launcher_cs, out_exe, manifest_cs=manifest_cs)

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
