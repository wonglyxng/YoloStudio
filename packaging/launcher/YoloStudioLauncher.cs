// YoloStudio 独立便携整合包原生 Windows GUI 启动器源码。
//
// 职责：
//     编译生成便携包根目录下的无控制台黑框可执行程序 `YoloStudio.exe`，
//     以启动器自身所在目录为基准按纯相对路径定位内置独立 Python 运行时（`runtime\pythonw.exe`）
//     与应用入口（`app\main.py`），装配进程级环境隔离变量后拉起桌面主界面。
//
// 运行环境：
//     构建期由 Windows 10/11 系统内置 `.NET Framework 4.x` 编译器 `csc.exe` 编译（零外部 SDK 依赖）；
//     运行期面向 Windows 10/11 x64 桌面环境，无需目标电脑预装 Python、CUDA Toolkit 或开发环境。
//
// 协作对象与职责边界：
//     由 `scripts/build_portable.py` 在打包阶段调用 `csc.exe` 编译并嵌入应用图标；
//     运行期仅负责校验解压完整性、隔离宿主 Python/Conda 环境变量并启动 `app\main.py`，
//     子进程成功拉起后启动器自身立即退出，不驻留后台或接管业务状态。

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

[assembly: AssemblyTitle("YoloStudio Launcher")]
[assembly: AssemblyDescription("YOLO26 视频抽帧、内嵌标注、MLLM 纠错与训练一体化便携启动器")]
[assembly: AssemblyProduct("YoloStudio")]
[assembly: AssemblyVersion("0.2.0.0")]
[assembly: AssemblyFileVersion("0.2.0.0")]

namespace YoloStudio.Launcher
{
    /// <summary>
    /// 便携包桌面启动入口类：负责路径校验、环境变量净化与子进程创建。
    /// </summary>
    internal static class Program
    {
        [DllImport("user32.dll")]
        private static extern bool SetProcessDPIAware();

        /// <summary>
        /// 启动器主入口；默认使用 `pythonw.exe` 静默启动 GUI，传入 `--console` 时改用 `python.exe` 显示终端便于排障。
        /// 正常拉起子进程返回 `0`；文件缺失或进程创建失败时弹出中文提示框并返回 `1`。
        /// </summary>
        [STAThread]
        private static int Main(string[] args)
        {
            try
            {
                SetProcessDPIAware();
            }
            catch
            {
                // 旧系统不支持 DPI API 时忽略，交由子进程 Qt6 自行管理高分屏缩放
            }

            string bundleRoot = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(
                Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar
            );

            bool useConsole = false;
            List<string> forwardedArgs = new List<string>();
            if (args != null)
            {
                foreach (string arg in args)
                {
                    if (string.Equals(arg, "--console", StringComparison.OrdinalIgnoreCase))
                    {
                        useConsole = true;
                    }
                    else
                    {
                        forwardedArgs.Add(arg);
                    }
                }
            }

            string runtimeDir = Path.Combine(bundleRoot, "runtime");
            string pythonExeName = useConsole ? "python.exe" : "pythonw.exe";
            string pythonPath = Path.Combine(runtimeDir, pythonExeName);
            if (!File.Exists(pythonPath) && !useConsole)
            {
                // 若精简环境未提供 pythonw.exe，回退查找 python.exe 并隐藏窗口启动
                string fallbackPython = Path.Combine(runtimeDir, "python.exe");
                if (File.Exists(fallbackPython))
                {
                    pythonPath = fallbackPython;
                }
            }

            string appDir = Path.Combine(bundleRoot, "app");
            string entryScript = Path.Combine(appDir, "main.py");

            if (!File.Exists(pythonPath) || !File.Exists(entryScript))
            {
                string missingMessage =
                    "未找到完整的内置运行环境或应用入口文件：\n\n" +
                    "  期望解释器: " + pythonPath + "\n" +
                    "  期望入口脚本: " + entryScript + "\n\n" +
                    "常见原因：您可能直接在压缩软件（如 WinRAR / 7-Zip / 资源管理器预览）中双击了 YoloStudio.exe。\n" +
                    "请先将整个压缩包【完整解压】到本地磁盘文件夹后，再双击 YoloStudio.exe 启动。";
                MessageBox.Show(
                    missingMessage,
                    "YoloStudio 启动提示",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning
                );
                return 1;
            }

            string workspaceDir = Path.Combine(bundleRoot, "workspace");
            string weightsDir = Path.Combine(bundleRoot, "weights");
            string workingDir = TryPrepareDirectory(workspaceDir) ? workspaceDir : bundleRoot;
            TryPrepareDirectory(weightsDir);

            StringBuilder cmdArgs = new StringBuilder();
            cmdArgs.Append(QuoteArgument(entryScript));
            foreach (string item in forwardedArgs)
            {
                cmdArgs.Append(' ');
                cmdArgs.Append(QuoteArgument(item));
            }

            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = cmdArgs.ToString(),
                WorkingDirectory = workingDir,
                UseShellExecute = false,
                CreateNoWindow = !useConsole
            };

            ConfigureIsolatedEnvironment(psi, runtimeDir, workspaceDir, weightsDir);

            try
            {
                Process child = Process.Start(psi);
                if (child == null)
                {
                    MessageBox.Show(
                        "操作系统未能创建 Python 运行进程，请检查杀毒软件拦截记录或目录访问权限。",
                        "YoloStudio 启动错误",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error
                    );
                    return 1;
                }
                return 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "启动内置运行环境失败：\n\n" + ex.Message + "\n\n路径: " + pythonPath,
                    "YoloStudio 启动错误",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return 1;
            }
        }

        /// <summary>
        /// 尝试创建目录，成功返回 `true`；位于只读介质或无权限时返回 `false`，交由 Python 端自动回退至用户目录。
        /// </summary>
        private static bool TryPrepareDirectory(string path)
        {
            try
            {
                if (!Directory.Exists(path))
                {
                    Directory.CreateDirectory(path);
                }
                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// 净化宿主遗留的 Python/Conda 环境变量，并注入便携包专属路径与 UTF-8 编码配置。
        /// </summary>
        private static void ConfigureIsolatedEnvironment(
            ProcessStartInfo psi,
            string runtimeDir,
            string workspaceDir,
            string weightsDir
        )
        {
            string[] pollutedVars = new string[]
            {
                "PYTHONHOME",
                "PYTHONPATH",
                "VIRTUAL_ENV",
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "_OLD_VIRTUAL_PATH"
            };
            foreach (string key in pollutedVars)
            {
                if (psi.EnvironmentVariables.ContainsKey(key))
                {
                    psi.EnvironmentVariables.Remove(key);
                }
            }

            psi.EnvironmentVariables["PYTHONNOUSERSITE"] = "1";
            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

            if (string.IsNullOrWhiteSpace(psi.EnvironmentVariables["YOLO_STUDIO_WORKSPACE"]))
            {
                psi.EnvironmentVariables["YOLO_STUDIO_WORKSPACE"] = workspaceDir;
            }
            if (string.IsNullOrWhiteSpace(psi.EnvironmentVariables["YOLO_STUDIO_WEIGHTS_DIR"]))
            {
                psi.EnvironmentVariables["YOLO_STUDIO_WEIGHTS_DIR"] = weightsDir;
            }

            List<string> prependPaths = new List<string> { runtimeDir };
            string dllsDir = Path.Combine(runtimeDir, "DLLs");
            if (Directory.Exists(dllsDir))
            {
                prependPaths.Add(dllsDir);
            }
            string libraryBinDir = Path.Combine(Path.Combine(runtimeDir, "Library"), "bin");
            if (Directory.Exists(libraryBinDir))
            {
                prependPaths.Add(libraryBinDir);
            }
            string scriptsDir = Path.Combine(runtimeDir, "Scripts");
            if (Directory.Exists(scriptsDir))
            {
                prependPaths.Add(scriptsDir);
            }

            string existingPath = psi.EnvironmentVariables["PATH"] ?? string.Empty;
            string prefix = string.Join(Path.PathSeparator.ToString(), prependPaths.ToArray());
            psi.EnvironmentVariables["PATH"] = string.IsNullOrEmpty(existingPath)
                ? prefix
                : (prefix + Path.PathSeparator + existingPath);
        }

        /// <summary>
        /// 按 Windows `CommandLineToArgvW` 规则安全转义命令行参数，兼容含空格、中文、反斜杠与引号的路径。
        /// </summary>
        private static string QuoteArgument(string arg)
        {
            if (string.IsNullOrEmpty(arg))
            {
                return "\"\"";
            }

            bool needsQuotes = false;
            foreach (char c in arg)
            {
                if (char.IsWhiteSpace(c) || c == '"')
                {
                    needsQuotes = true;
                    break;
                }
            }
            if (!needsQuotes)
            {
                return arg;
            }

            StringBuilder sb = new StringBuilder();
            sb.Append('"');
            int backslashes = 0;
            foreach (char c in arg)
            {
                if (c == '\\')
                {
                    backslashes++;
                }
                else if (c == '"')
                {
                    sb.Append('\\', backslashes * 2 + 1);
                    sb.Append('"');
                    backslashes = 0;
                }
                else
                {
                    if (backslashes > 0)
                    {
                        sb.Append('\\', backslashes);
                        backslashes = 0;
                    }
                    sb.Append(c);
                }
            }
            if (backslashes > 0)
            {
                sb.Append('\\', backslashes * 2);
            }
            sb.Append('"');
            return sb.ToString();
        }
    }
}
