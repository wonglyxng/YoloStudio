"""按线程路由的标准输出/日志捕获工具。

职责：
    把训练、验证、推理、导出等后台任务在**当前线程**中产生的 `print`/TQDM 进度输出
    以及 `ultralytics` 日志记录按行转发到调用方回调，同时不影响其他线程与主线程的正常输出。

运行环境：
    Python >= 3.11，仅依赖标准库。

设计原因与协作边界：
    `contextlib.redirect_stdout` 会替换进程级 `sys.stdout`，多个后台任务并发时互相串扰，
    且先退出的任务会提前恢复 `sys.stdout`；同时 Ultralytics 在导入时就把日志
    `StreamHandler` 绑定到当时的 `sys.stdout` 对象，事后重定向无法捕获其日志。
    本模块改为：进程内一次性把 `sys.stdout`/`sys.stderr` 换成 `_RoutingStream` 代理，
    代理按 `threading.get_ident()` 查找已注册的路由，未注册线程原样写往原始流；
    并在每次捕获开始时把 `ultralytics` 日志器中指向原始控制台的 `StreamHandler` 改指代理。
    由 `core.trainer` 与 `core.mllm_reviewer` 使用；不负责线程创建，也不持久化日志。
"""

from __future__ import annotations

import io
import logging
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

_ROUTES_LOCK = threading.Lock()
# 线程标识 -> 行组装器；仅在 `capture_thread_output` 作用域内存在，退出时恢复为外层路由或移除
_ROUTES: dict[int, "_LineAssembler"] = {}


class _LineAssembler:
    """把任意分片的文本组装成完整行并回调，`\\r` 按终端语义视为“覆盖当前行”。

    状态：`_buffer` 保存当前尚未换行的行内容。遇到 `\\r`（且不是 `\\r\\n`）时丢弃已缓存内容，
    因此 TQDM 的逐次刷新不会产生大量重复行，只有最终以 `\\n` 结束的那一版会被回调。
    仅由所属线程调用，无需加锁。
    """

    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback
        self._buffer = ""
        # 标记当前是否正处于 `_callback` 调用栈内；若回调自身再次写向 stdout/stderr（例如 `callback=print`），
        # `_RoutingStream.write` 据此绕过再次捕获并直接写往原始流，防止无限递归
        self._emitting = False

    def feed(self, text: str) -> None:
        """追加文本并对每个完整行调用回调（空白行被忽略）。"""
        if not text:
            return
        data = self._buffer + text.replace("\r\n", "\n")
        lines = data.split("\n")
        self._buffer = lines.pop()
        for line in lines:
            self._emit(line)
        # 对尚未换行的尾部按 `\r` 覆盖语义只保留最后一段
        if "\r" in self._buffer:
            self._buffer = self._buffer.rsplit("\r", 1)[1]

    def _emit(self, line: str) -> None:
        if "\r" in line:
            line = line.rsplit("\r", 1)[1]
        stripped = line.strip()
        if stripped:
            self._emitting = True
            try:
                self._callback(stripped)
            finally:
                self._emitting = False

    def flush(self) -> None:
        """刷出尾部未换行的残留内容并清空缓冲。"""
        if self._buffer:
            self._emit(self._buffer)
        self._buffer = ""


class _RoutingStream(io.TextIOBase):
    """替换 `sys.stdout`/`sys.stderr` 的代理流：已注册线程写入对应回调，其余线程写往原始流。

    `original` 为安装代理时的原始流（`pythonw` 下可能为 `None`，此时未注册线程的输出被丢弃）。
    """

    def __init__(self, original: io.TextIOBase | None) -> None:
        super().__init__()
        self.original = original

    def write(self, s: str) -> int:
        route = _ROUTES.get(threading.get_ident())
        if route is not None and not route._emitting:
            route.feed(s)
            return len(s)
        if self.original is not None:
            try:
                return self.original.write(s)
            except Exception:
                return len(s)
        return len(s)

    def flush(self) -> None:
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        if self.original is not None:
            return self.original.fileno()
        raise io.UnsupportedOperation("fileno")

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self.original, "encoding", None) or "utf-8"


def _install_routing_streams() -> None:
    """幂等安装代理流；若 `sys.stdout` 已被他人（如 pytest）替换为非代理对象则重新包装当前对象。"""
    if not isinstance(sys.stdout, _RoutingStream):
        sys.stdout = _RoutingStream(sys.stdout)
    if not isinstance(sys.stderr, _RoutingStream):
        sys.stderr = _RoutingStream(sys.stderr)


def _retarget_ultralytics_handlers() -> None:
    """把 `ultralytics` 日志器中直接写控制台的 `StreamHandler` 改指当前代理流。

    只处理精确类型为 `logging.StreamHandler` 的处理器（不触碰 `FileHandler` 等子类）。
    """
    logger = logging.getLogger("ultralytics")
    for handler in logger.handlers:
        if type(handler) is logging.StreamHandler and not isinstance(handler.stream, _RoutingStream):
            target = sys.stderr if handler.stream is getattr(sys.stderr, "original", None) else sys.stdout
            handler.setStream(target)


@contextmanager
def capture_thread_output(callback: Callable[[str], None] | None) -> Iterator[None]:
    """在当前线程内把标准输出、标准错误与 `ultralytics` 日志按行转发给 `callback`。

    调用约束：
        必须在产生输出的同一线程中进入与退出；可嵌套，退出时恢复外层路由。
        `callback` 为 `None` 时不做任何处理。退出时会刷出尾部残留行。
        `callback` 在产生输出的线程中同步调用，GUI 调用方应通过 Qt 信号转发到主线程。
    """
    if callback is None:
        yield
        return
    _install_routing_streams()
    _retarget_ultralytics_handlers()
    tid = threading.get_ident()
    assembler = _LineAssembler(callback)
    with _ROUTES_LOCK:
        previous = _ROUTES.get(tid)
        _ROUTES[tid] = assembler
    try:
        yield
    finally:
        assembler.flush()
        with _ROUTES_LOCK:
            if previous is not None:
                _ROUTES[tid] = previous
            else:
                _ROUTES.pop(tid, None)
