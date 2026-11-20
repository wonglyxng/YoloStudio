// YoloStudio 启动器默认元数据与完整性清单占位文件（供直接独立编译 `YoloStudioLauncher.cs` 时使用）。
//
// 职责：
//     提供默认程序集版本占位与空的 `IntegrityManifest.Entries` 字典；
//     在正常通过 `scripts/build_portable.py` 打包时，构建脚本会按 `src/yolo_studio/_version.py`
//     动态生成包含准确版本号与防篡改指纹的临时清单源码替代本文件参与编译。

using System.Collections.Generic;
using System.Reflection;

[assembly: AssemblyVersion("0.0.0.0")]
[assembly: AssemblyFileVersion("0.0.0.0")]
[assembly: AssemblyInformationalVersion("0.0.0-dev")]

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
