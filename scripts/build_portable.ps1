<#
.SYNOPSIS
    YoloStudio Windows 独立便携整合包一键构建 PowerShell 入口脚本。

.DESCRIPTION
    职责：
        定位本机可用的 Python >= 3.11 解释器（优先使用仓库 `.venv\Scripts\python.exe`，
        不存在时回退至系统 `PATH` 中的 `python`），并将全部命令行参数委托转发给
        `scripts/build_portable.py` 执行便携运行时组装、原生启动器编译、隔离自检与归档压缩。
    运行环境：
        Windows PowerShell 5.1 或 PowerShell 7+（`pwsh`）。
    协作对象与职责边界：
        仅负责宿主驱动解释器探测与退出码透传，具体构建逻辑与文件操作均由 `scripts/build_portable.py` 承担
        （完整参数与示例见 `docs/003_独立便携整合包构建与发布说明.md`）。
#>

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildPy  = Join-Path $PSScriptRoot "build_portable.py"
$VenvPy   = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $VenvPy) {
    $DriverPython = $VenvPy
} else {
    $FoundPy = Get-Command python -ErrorAction SilentlyContinue
    if (-not $FoundPy) {
        Write-Error "未找到可用的构建驱动解释器（既无 .venv\Scripts\python.exe，PATH 中也无 python）。"
        exit 1
    }
    $DriverPython = $FoundPy.Source
}

& $DriverPython $BuildPy @args
exit $LASTEXITCODE
