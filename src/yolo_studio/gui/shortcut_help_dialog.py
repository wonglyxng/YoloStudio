"""标注工作台快捷键速查面板。

职责：
    集中展示工作台全部快捷键与鼠标操作，按“导航 / 模式 / 编辑框 / SAM 与 AI / 视图”分组，
    解决单键快捷键只散落在按钮提示里、难以被发现的问题。由 `?` 或 `F1` 唤出，`Esc` / `?` / `F1` 关闭。

运行环境与协作边界：
    运行于 PyQt6 主线程，纯展示、无副作用；快捷键清单以模块常量 `WORKBENCH_SHORTCUT_GROUPS` 维护，
    新增或修改工作台快捷键时需同步更新此处（测试会核对关键条目存在）。
"""

from __future__ import annotations

import html

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

# (分组名, [(按键, 说明), ...])；按键文本仅用于展示
WORKBENCH_SHORTCUT_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "图片导航与保存",
        (
            ("A / D", "上一张 / 下一张图片"),
            ("F", "确认通过当前图并跳到下一张（0 框保存为空图负样本）"),
            ("I / Shift+I", "标记或编辑当前图问题备注 / 跳到下一张问题图"),
            ("Ctrl+S", "立即保存当前图片标注"),
            ("Ctrl+Z / Ctrl+Y", "撤销 / 重做"),
        ),
    ),
    (
        "标注模式",
        (
            ("E", "编辑模式：选中、移动、拖拽控制点"),
            ("R / W", "画框模式：拖拽成框，或两次点击起点与终点"),
            ("Q", "SAM 点选模式：点击目标自动生成贴边框"),
            ("H", "抓手模式：左键拖动画布"),
            ("Esc", "取消草稿 → 返回编辑模式 → 取消选中"),
        ),
    ),
    (
        "编辑目标框",
        (
            ("Tab / Shift+Tab", "在当前图的框之间循环切换选中"),
            ("Ctrl+A", "全选当前图所有可见目标框"),
            ("Ctrl / Shift + 点选框", "多选 / 取消选中多个框（拖任意选中框可整组平移）"),
            ("Shift / Ctrl + 拖空白", "橡皮筋框选多个目标框"),
            ("方向键 / Shift+方向键", "选中框（含多选整组）微调 1 px / 10 px"),
            ("Del / Backspace", "删除选中框"),
            ("Ctrl+D", "复制选中框"),
            ("Ctrl+C / Ctrl+V", "复制框 / 粘贴到当前图（可跨图）"),
            ("Shift+V", "复用上一张图的全部框"),
            ("1 ~ 9", "切换当前类别；选中框时直接改类"),
            ("C", "打开可搜索类别选择器（改选中框类别或切换当前类别）"),
            ("V", "隐藏 / 显示当前选中框"),
            ("K", "锁定 / 解锁当前选中框（防误拖、防误删）"),
            ("双击类别", "把选中框改为该类别"),
        ),
    ),
    (
        "SAM 与 AI",
        (
            ("左键 / 右键（点选模式）", "添加前景点 / 排除背景点"),
            ("Backspace（点选模式）", "撤回上一个提示点"),
            ("Enter（点选模式）", "确认落框并继续标下一个目标"),
            ("G", "对选中框做局部放大 AI 精修"),
        ),
    ),
    (
        "视图",
        (
            ("滚轮", "以鼠标为中心缩放"),
            ("空格+左键 / 右键拖 / 中键拖", "平移画布"),
            ("0 / Ctrl+0", "适应窗口"),
            ("Ctrl+1", "100% 原始像素"),
            ("+ / - （或 Ctrl+= / Ctrl+-）", "放大 / 缩小"),
            ("L", "显示 / 隐藏框标签"),
            ("O", "显示 / 隐藏上一帧框淡色参考（洋葱皮）"),
            ("X", "常显十字准星"),
            ("B", "暗部 / 对比度临时增强"),
            ("M", "光标局部放大镜（3.5× 像素级贴边瞄准）"),
            ("? / F1", "打开 / 关闭本面板"),
        ),
    ),
)


def _render_groups_html() -> str:
    """把快捷键分组渲染为两栏 HTML 表格，便于一屏看完。"""
    cells: list[str] = []
    for title, items in WORKBENCH_SHORTCUT_GROUPS:
        rows = "".join(
            "<tr>"
            f"<td style='padding:3px 10px 3px 0; white-space:nowrap;'>"
            f"<span style='background:#f6f8fa; border:1px solid #d0d7de; border-radius:4px; "
            f"padding:1px 5px; font-family:Consolas,monospace; color:#24292f;'>{html.escape(key)}</span></td>"
            f"<td style='padding:3px 0; color:#24292f;'>{html.escape(desc)}</td>"
            "</tr>"
            for key, desc in items
        )
        cells.append(
            f"<div style='font-weight:600; color:#0550ae; margin:6px 0 4px 0;'>{html.escape(title)}</div>"
            f"<table cellspacing='0' cellpadding='0'>{rows}</table>"
        )
    half = (len(cells) + 1) // 2
    left = "".join(cells[:half])
    right = "".join(cells[half:])
    return (
        "<table width='100%' cellspacing='0' cellpadding='0'><tr>"
        f"<td valign='top' width='50%' style='padding-right:16px;'>{left}</td>"
        f"<td valign='top' width='50%'>{right}</td>"
        "</tr></table>"
    )


class ShortcutHelpDialog(QDialog):
    """工作台快捷键速查面板（非模态，`Esc` / `?` / `F1` 关闭）。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("快捷键速查 — 标注工作台")
        self.setModal(False)
        self.resize(780, 520)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        tip = QLabel("单键快捷键在输入框获得焦点时不生效；按 ? 或 F1 随时打开本面板。")
        tip.setStyleSheet("color: #57606a; font-size: 12px;")
        lay.addWidget(tip)

        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setStyleSheet("QTextBrowser { border: 1px solid #d8dee4; border-radius: 6px; padding: 6px; }")
        self.browser.setHtml(_render_groups_html())
        lay.addWidget(self.browser, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.close_btn)
        lay.addLayout(btn_row)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Question, Qt.Key.Key_F1, Qt.Key.Key_Slash):
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)
