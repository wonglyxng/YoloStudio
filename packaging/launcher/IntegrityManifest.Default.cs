// YoloStudio 启动器默认完整性清单占位文件（用于 `--protect none` 未启用保护模式）。
//
// 职责：
//     提供空的 `IntegrityManifest.Entries` 字典，保证在未生成编译期 SHA-256 清单时
//     `YoloStudioLauncher.cs` 仍可独立通过 `csc.exe` 编译运行。
//
// 协作对象与职责边界：
//     在启用 `--protect pyd` 或 `--protect pyc` 构建保护包时，
//     `scripts/build_portable.py` 会生成包含真实 SHA-256 指纹的临时 `IntegrityManifest.g.cs`
//     替代本文件参与编译。

using System.Collections.Generic;

namespace YoloStudio.Launcher
{
    /// <summary>
    /// 编译期嵌入的核心程序文件 SHA-256 完整性清单（默认为空，表示跳过校验）。
    /// </summary>
    internal static class IntegrityManifest
    {
        internal static readonly Dictionary<string, string> Entries =
            new Dictionary<string, string>();
    }
}
