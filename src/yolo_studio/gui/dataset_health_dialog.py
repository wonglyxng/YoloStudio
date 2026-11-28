"""标注工作台「数据集健康体检」对话框。

职责：
    展示 `core.dataset_health.analyze_dataset_health` 产出的离线体检报告：顶部 KPI 卡片摘要、
    按类型可筛选的异常问题清单表、类别样本分布与不平衡预警表；支持双击或点击按钮直接跳转
    到工作台对应图片并选中聚焦问题目标框，支持在修复后点击「重新体检」实时刷新。

运行环境与协作边界：
    运行于 PyQt6 主线程，采用非模态展示以便与工作台画布并排逐个修复；通过 `jump_to_target`
    信号通知工作台切图与选框，不直接修改任何 `.json` 文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, pyqtSignal as Signal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from yolo_studio.core.annotation_io import label_color_rgb
from yolo_studio.core.dataset_health import (
    HEALTH_ISSUE_LABELS,
    DatasetHealthReport,
    HealthIssue,
    HealthIssueKind,
)


def _swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
    pix = QPixmap(12, 12)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(*rgb))
    p.drawRoundedRect(1, 1, 10, 10, 3, 3)
    p.end()
    return QIcon(pix)


class DatasetHealthDialog(QDialog):
    """数据集标注健康体检非模态面板（发射 `jump_to_target(image_path, box_index)` 联动工作台）。"""

    jump_to_target = Signal(object, int)  # (Path, box_index；-1 表示仅切图)

    def __init__(
        self,
        report: DatasetHealthReport,
        *,
        rescan_provider: Callable[[], DatasetHealthReport] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._report = report
        self._rescan_provider = rescan_provider
        self._visible_issues: list[HealthIssue] = []

        self.setWindowTitle("数据集标注健康体检 — 标注工作台")
        self.setModal(False)
        self.resize(840, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        # 顶部 KPI 卡片行
        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(8)
        self.kpi_images_val = self._add_kpi_card(kpi_row, "图片覆盖")
        self.kpi_boxes_val = self._add_kpi_card(kpi_row, "总目标框")
        self.kpi_issues_val = self._add_kpi_card(kpi_row, "检出问题")
        self.kpi_balance_val = self._add_kpi_card(kpi_row, "类别均衡比")
        root.addLayout(kpi_row)

        self.tabs = QTabWidget()

        # 页签 1：问题清单
        issue_page = QWidget()
        i_lay = QVBoxLayout(issue_page)
        i_lay.setContentsMargins(6, 6, 6, 6)
        i_lay.setSpacing(6)

        i_bar = QHBoxLayout()
        i_bar.setSpacing(6)
        i_bar.addWidget(QLabel("异常筛选:"))
        self.issue_filter_combo = QComboBox()
        self.issue_filter_combo.addItem("全部异常", "all")
        for kind in HealthIssueKind:
            self.issue_filter_combo.addItem(HEALTH_ISSUE_LABELS[kind], kind.value)
        self.issue_filter_combo.currentIndexChanged.connect(self._populate_issue_table)
        i_bar.addWidget(self.issue_filter_combo)
        self.issue_hint_label = QLabel("双击任意行立即跳转到对应图片并聚焦问题框")
        self.issue_hint_label.setStyleSheet("color: #57606a; font-size: 12px;")
        i_bar.addWidget(self.issue_hint_label, 1)
        self.jump_issue_btn = QPushButton("定位到工作台")
        self.jump_issue_btn.clicked.connect(self._on_jump_selected_issue)
        i_bar.addWidget(self.jump_issue_btn)
        i_lay.addLayout(i_bar)

        self.issue_table = QTableWidget(0, 4)
        self.issue_table.setHorizontalHeaderLabels(["问题类型", "图片文件", "类别", "说明"])
        self.issue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.issue_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.issue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.issue_table.verticalHeader().setVisible(False)
        hdr = self.issue_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.issue_table.setColumnWidth(1, 210)
        self.issue_table.cellDoubleClicked.connect(lambda _r, _c: self._on_jump_selected_issue())
        i_lay.addWidget(self.issue_table, 1)
        self.tabs.addTab(issue_page, "问题清单")

        # 页签 2：类别分布
        cls_page = QWidget()
        c_lay = QVBoxLayout(cls_page)
        c_lay.setContentsMargins(6, 6, 6, 6)
        c_lay.setSpacing(6)
        self.zero_class_label = QLabel("")
        self.zero_class_label.setWordWrap(True)
        self.zero_class_label.setStyleSheet("color: #9a6700; font-size: 12px;")
        self.zero_class_label.setVisible(False)
        c_lay.addWidget(self.zero_class_label)

        self.class_table = QTableWidget(0, 5)
        self.class_table.setHorizontalHeaderLabels(["类别", "框数", "图片数", "占比", "状态"])
        self.class_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.class_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.class_table.verticalHeader().setVisible(False)
        chdr = self.class_table.horizontalHeader()
        chdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4):
            chdr.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.class_table.cellDoubleClicked.connect(self._on_jump_class_sample)
        c_lay.addWidget(self.class_table, 1)
        self.tabs.addTab(cls_page, "类别分布")

        root.addWidget(self.tabs, 1)

        btn_row = QHBoxLayout()
        self.rescan_btn = QPushButton("重新体检")
        self.rescan_btn.setVisible(rescan_provider is not None)
        self.rescan_btn.clicked.connect(self._on_rescan)
        btn_row.addWidget(self.rescan_btn)
        btn_row.addStretch(1)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.close_btn)
        root.addLayout(btn_row)

        self.update_report(report)

    @staticmethod
    def _add_kpi_card(parent_lay: QHBoxLayout, title: str) -> QLabel:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 6px; }"
            "QLabel { border: none; }"
        )
        cl = QVBoxLayout(card)
        cl.setContentsMargins(10, 6, 10, 6)
        cl.setSpacing(2)
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("color: #57606a; font-size: 11px;")
        v_lbl = QLabel("-")
        v_lbl.setStyleSheet("color: #24292f; font-size: 14px; font-weight: 600;")
        cl.addWidget(t_lbl)
        cl.addWidget(v_lbl)
        parent_lay.addWidget(card, 1)
        return v_lbl

    def report(self) -> DatasetHealthReport:
        """返回当前展示的体检报告。"""
        return self._report

    def update_report(self, report: DatasetHealthReport) -> None:
        """用最新体检报告刷新所有 KPI 卡片、问题表与类别分布表。"""
        self._report = report
        r = report
        self.kpi_images_val.setText(
            f"已标 {r.labeled_images} · 空 {r.empty_images} / {r.total_images}"
        )
        self.kpi_boxes_val.setText(f"{r.total_boxes} 框（{len(r.class_stats)} 类）")
        issue_color = "#cf222e" if r.issues else "#1a7f37"
        self.kpi_issues_val.setStyleSheet(f"color: {issue_color}; font-size: 14px; font-weight: 600;")
        self.kpi_issues_val.setText(f"{len(r.issues)} 项" if r.issues else "0 项（健康）")
        if r.imbalance_ratio > 0:
            bal_color = "#9a6700" if r.imbalance_ratio >= 10.0 else "#24292f"
            self.kpi_balance_val.setStyleSheet(f"color: {bal_color}; font-size: 14px; font-weight: 600;")
            self.kpi_balance_val.setText(f"{r.imbalance_ratio:.1f} : 1")
        else:
            self.kpi_balance_val.setStyleSheet("color: #57606a; font-size: 14px; font-weight: 600;")
            self.kpi_balance_val.setText("-")

        self.tabs.setTabText(0, f"问题清单 ({len(r.issues)})")
        self.tabs.setTabText(1, f"类别分布 ({len(r.class_stats)})")
        self._populate_issue_table()
        self._populate_class_table()

    def _populate_issue_table(self) -> None:
        filter_key = str(self.issue_filter_combo.currentData() or "all")
        filtered = [
            it
            for it in self._report.issues
            if filter_key == "all" or it.kind.value == filter_key
        ]
        self._visible_issues = filtered
        self.issue_table.setRowCount(len(filtered))
        for row, it in enumerate(filtered):
            kind_item = QTableWidgetItem(HEALTH_ISSUE_LABELS.get(it.kind, it.kind.value))
            if it.kind in (HealthIssueKind.CORRUPT_JSON, HealthIssueKind.DUPLICATE_BOX):
                kind_item.setForeground(QColor("#cf222e"))
            elif it.kind in (HealthIssueKind.IMAGE_ISSUE_FLAG, HealthIssueKind.UNREGISTERED_LABEL):
                kind_item.setForeground(QColor("#9a6700"))
            img_item = QTableWidgetItem(it.image_path.name)
            img_item.setToolTip(str(it.image_path))
            lbl_item = QTableWidgetItem(it.label or "-")
            if it.label:
                lbl_item.setIcon(_swatch_icon(label_color_rgb(it.label)))
            desc_item = QTableWidgetItem(it.summary)
            self.issue_table.setItem(row, 0, kind_item)
            self.issue_table.setItem(row, 1, img_item)
            self.issue_table.setItem(row, 2, lbl_item)
            self.issue_table.setItem(row, 3, desc_item)
        self.jump_issue_btn.setEnabled(bool(filtered))

    def _populate_class_table(self) -> None:
        stats = self._report.class_stats
        self.class_table.setRowCount(len(stats))
        for row, st in enumerate(stats):
            name_item = QTableWidgetItem(_swatch_icon(label_color_rgb(st.label)), st.label)
            cnt_item = QTableWidgetItem(str(st.box_count))
            cnt_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            img_item = QTableWidgetItem(str(st.image_count))
            img_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            pct_item = QTableWidgetItem(f"{st.share_ratio * 100:.1f}%")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            tags: list[str] = []
            if st.is_unregistered:
                tags.append("未登记")
            if st.is_underrepresented:
                tags.append("样本偏少")
            status_item = QTableWidgetItem(" · ".join(tags) if tags else "正常")
            if tags:
                status_item.setForeground(QColor("#9a6700"))
            else:
                status_item.setForeground(QColor("#1a7f37"))
            self.class_table.setItem(row, 0, name_item)
            self.class_table.setItem(row, 1, cnt_item)
            self.class_table.setItem(row, 2, img_item)
            self.class_table.setItem(row, 3, pct_item)
            self.class_table.setItem(row, 4, status_item)

        if self._report.zero_count_classes:
            zeros = "、".join(self._report.zero_count_classes)
            self.zero_class_label.setText(f"提示：类别清单中有 {len(self._report.zero_count_classes)} 个零样本类别：{zeros}")
            self.zero_class_label.setVisible(True)
        else:
            self.zero_class_label.setVisible(False)

    def _on_jump_selected_issue(self) -> None:
        row = self.issue_table.currentRow()
        if not (0 <= row < len(self._visible_issues)):
            if self._visible_issues:
                row = 0
            else:
                return
        it = self._visible_issues[row]
        self.jump_to_target.emit(it.image_path, -1 if it.box_index is None else int(it.box_index))

    def _on_jump_class_sample(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._report.class_stats):
            sample = self._report.class_stats[row].sample_image
            if sample is not None:
                self.jump_to_target.emit(sample, -1)

    def _on_rescan(self) -> None:
        if self._rescan_provider is not None:
            self.update_report(self._rescan_provider())
