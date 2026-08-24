"""Scan-filter dialog: per-file UI for excluding individual scans.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Opened from a right-click menu on any file row in Pre-Analysis or
Analysis. Lists every scan present in the file with timing and signal
stats; the user toggles include checkboxes (with bulk and rate-threshold
helpers) and the result lands in the global ScanFilterRegistry. Both tabs
subscribe to ``filters_changed`` so the spectrum / fit immediately picks
up the new filter without an explicit refresh.

Depends on: gui.scan_filter (bunches_per_scan, get_registry,
read_bunches_per_channel, summarize_scans).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from gui.scan_filter import (
    bunches_per_scan, get_registry, read_bunches_per_channel,
    summarize_scans,
)


def _read_scan_payload(asdf_path, pmt_gate):
    """Open an ASDF cheaply and pull just what summarize_scans needs.

    Returns ``(rows, n_total_scans, error)`` where ``rows`` is the
    list of per-scan dicts (each with the keys documented in
    ``summarize_scans``), ``n_total_scans`` is the maximum scan index
    present plus 1 (so the dialog can disambiguate "scan 0..N
    appeared in the file" from "the file goes up to scan M"), and
    ``error`` is a human-readable string when something went wrong
    (the dialog short-circuits on non-empty error).
    """
    try:
        import asdf
        with asdf.open(asdf_path) as af:
            raw = np.asarray(af.tree["raw"])
            scanning_ranges = af.tree.get("ScanningRanges")
            step_size = af.tree.get("StepSize")
            bpc = int(af.tree.get("BunchesPerChannel", 1) or 1)
    except Exception as exc:
        return [], 0, f"Could not read {os.path.basename(asdf_path)}: {exc}"

    bps = bunches_per_scan(scanning_ranges, step_size, bpc)
    if bps <= 0:
        return [], 0, (
            f"{os.path.basename(asdf_path)}: missing ScanningRanges / "
            f"StepSize -- can't derive scans for this file.")

    # Standard column order from clstools' Load_Run.
    ts = raw[:, 0]
    bunch = raw[:, 2].astype(np.int64)
    channel = raw[:, 3].astype(int)

    rows = summarize_scans(
        bunch, ts, channel,
        pmt_gate=pmt_gate,
        scanning_ranges=scanning_ranges,
        step_size=step_size,
        bunches_per_channel=bpc)

    n_total = int(bunch.max() // bps) + 1 if len(bunch) else 0
    return rows, n_total, ""


class ScanFilterDialog(QDialog):
    """Modal per-file scan filter editor."""

    # Column indices as named constants for the bulk-action helpers.
    # Their order is the column layout the table is populated with.
    COL_SCAN, COL_START, COL_DUR, COL_NEV, COL_NPMT, COL_RATE, COL_INCL = \
        range(7)

    def __init__(self, asdf_path, pmt_gate=(3, 4), parent=None):
        super().__init__(parent)
        from gui.dialog_style import style_dialog
        self.setWindowTitle(f"Filter scans — {os.path.basename(asdf_path)}")
        self.resize(760, 560)
        # Same visual language as the tabs and the other pop-out dialogs.
        style_dialog(self)
        self._asdf_path = asdf_path
        self._pmt_gate = list(pmt_gate or [3, 4])
        self._registry = get_registry()

        self._rows, self._n_total, err = _read_scan_payload(
            asdf_path, self._pmt_gate)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Header summary ─────────────────────────────────
        header = QLabel(
            f"<b>{os.path.basename(asdf_path)}</b>"
            f"<br><span style='color:#888'>{asdf_path}</span>")
        header.setTextFormat(Qt.TextFormat.RichText)
        header.setWordWrap(True)
        layout.addWidget(header)

        if err:
            msg = QLabel(
                f"<span style='color:#c33'>{err}</span>"
                f"<br>The dialog can't show scans for this file. Filtering "
                f"is unavailable here.")
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setWordWrap(True)
            layout.addWidget(msg)
            layout.addStretch()
            close_bb = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Close)
            close_bb.rejected.connect(self.reject)
            layout.addWidget(close_bb)
            return

        # Stat line: helps the user see how aggressively they're cutting.
        # #666 is near-invisible against the app's #2d2d30 dialog background;
        # the palette's muted grey reads without shouting.
        self._stat_label = QLabel("")
        self._stat_label.setStyleSheet("color:#9a9aa0; font-size:11px;")
        layout.addWidget(self._stat_label)

        # ── Bulk controls ─────────────────────────────────
        # Two rows of bulk actions so the buttons don't crowd each
        # other on narrow windows. Row 1: Check/Uncheck All + the
        # selection-scoped variants the user reaches via Shift-click.
        # Row 2: the rate-threshold action.
        bulk = QHBoxLayout()
        bulk.setSpacing(6)
        check_all = QPushButton("Check All")
        uncheck_all = QPushButton("Uncheck All")
        check_all.clicked.connect(self._check_all)
        uncheck_all.clicked.connect(self._uncheck_all)
        bulk.addWidget(check_all)
        bulk.addWidget(uncheck_all)
        bulk.addSpacing(12)
        self._check_selected_btn = QPushButton("Check Selected")
        self._check_selected_btn.setToolTip(
            "Mark every currently selected row as included. Click a row "
            "to select; Shift-click for a range; Ctrl-click to toggle "
            "individual rows.")
        self._check_selected_btn.clicked.connect(
            lambda: self._set_selected(True))
        bulk.addWidget(self._check_selected_btn)
        self._uncheck_selected_btn = QPushButton("Uncheck Selected")
        self._uncheck_selected_btn.setToolTip(
            "Mark every currently selected row as excluded. Click a "
            "row to select; Shift-click for a range.")
        self._uncheck_selected_btn.clicked.connect(
            lambda: self._set_selected(False))
        bulk.addWidget(self._uncheck_selected_btn)
        bulk.addStretch()
        layout.addLayout(bulk)

        bulk2 = QHBoxLayout()
        bulk2.setSpacing(6)
        bulk2.addWidget(QLabel("Exclude rate <"))
        self._rate_threshold = QDoubleSpinBox()
        self._rate_threshold.setRange(0.0, 1e6)
        self._rate_threshold.setDecimals(3)
        self._rate_threshold.setSingleStep(0.1)
        self._rate_threshold.setValue(0.0)
        self._rate_threshold.setFixedWidth(90)
        self._rate_threshold.setToolTip(
            "Threshold for the bulk rate cut, in PMT events per second.")
        bulk2.addWidget(self._rate_threshold)
        bulk2.addWidget(QLabel("Hz"))
        rate_apply = QPushButton("Apply")
        rate_apply.setToolTip(
            "Uncheck every scan whose PMT rate is below the threshold.\n"
            "Existing unchecks are preserved (this is additive).")
        rate_apply.clicked.connect(self._apply_rate_threshold)
        bulk2.addWidget(rate_apply)
        bulk2.addStretch()
        layout.addLayout(bulk2)

        # ── Scan table ─────────────────────────────────────
        self._table = QTableWidget(len(self._rows), 7)
        self._table.setHorizontalHeaderLabels([
            "Scan", "Start (s)", "Dur (s)", "Events", "PMT",
            "Rate (Hz)", "Include",
        ])
        # Multi-row selection so the Check/Uncheck Selected buttons
        # can operate on contiguous ranges (Shift-click) and ad-hoc
        # picks (Ctrl-click). Whole-row selection means a click anywhere
        # in a row selects that scan, not just the clicked cell.
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        # Ensure a comfortable row height so checkboxes don't crowd
        # adjacent rows.
        self._table.verticalHeader().setDefaultSectionSize(22)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(False)
        layout.addWidget(self._table, 1)

        self._build_table()

        # ── OK / Cancel ────────────────────────────────────
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._refresh_stat_line()

    # ── Table population ────────────────────────────────────────────

    def _build_table(self):
        """Populate the scan rows from ``self._rows`` and seed the
        include checkboxes from the registry's current excluded set."""
        excluded = self._registry.get(self._asdf_path)
        # Use the file's TSstart as t=0 for human-readable column.
        if self._rows:
            t0 = float(self._rows[0]["ts_start"])
        else:
            t0 = 0.0
        self._t0 = t0
        self._checkboxes = []

        for i, row in enumerate(self._rows):
            scan_idx = int(row["scan"])
            self._set_text(i, self.COL_SCAN, str(scan_idx),
                           Qt.AlignmentFlag.AlignCenter)
            self._set_text(i, self.COL_START,
                           f"{row['ts_start'] - t0:.1f}",
                           Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)
            self._set_text(i, self.COL_DUR,
                           f"{row['duration']:.1f}",
                           Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)
            self._set_text(i, self.COL_NEV,
                           f"{row['n_events']}",
                           Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)
            self._set_text(i, self.COL_NPMT,
                           f"{row['n_pmt']}",
                           Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)
            self._set_text(i, self.COL_RATE,
                           f"{row['rate_hz']:.3f}",
                           Qt.AlignmentFlag.AlignRight
                           | Qt.AlignmentFlag.AlignVCenter)

            # Center the include checkbox in its cell. Wrapping it in a
            # QWidget + horizontal layout is the standard Qt trick.
            cb = QCheckBox()
            cb.setChecked(scan_idx not in excluded)
            cb.toggled.connect(self._refresh_stat_line)
            holder = QWidget()
            hlay = QHBoxLayout(holder)
            hlay.setContentsMargins(0, 0, 0, 0)
            hlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hlay.addWidget(cb)
            self._table.setCellWidget(i, self.COL_INCL, holder)
            self._checkboxes.append((scan_idx, cb))

    def _set_text(self, row, col, text, align=None):
        item = QTableWidgetItem(text)
        if align is not None:
            item.setTextAlignment(align)
        # Items are display-only; flags strip editing/selecting at the
        # widget level (NoSelection / NoEdit) but a defensive flag
        # clear here keeps the cells inert if those policies change.
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, col, item)

    # ── Bulk actions ────────────────────────────────────────────────

    def _check_all(self):
        for _, cb in self._checkboxes:
            cb.setChecked(True)
        self._refresh_stat_line()

    def _uncheck_all(self):
        for _, cb in self._checkboxes:
            cb.setChecked(False)
        self._refresh_stat_line()

    def _set_selected(self, checked):
        """Apply a single check-state to every selected row.

        Reads the table's selected rows via ``selectionModel()`` --
        with SelectRows + ExtendedSelection, that returns one entry
        per selected row regardless of which column the user clicked.
        Empty selection is a no-op so the buttons don't silently wipe
        the whole table when nothing is highlighted.
        """
        rows = sorted({i.row() for i in
                       self._table.selectionModel().selectedRows()})
        if not rows:
            # Falling back to "everything" would surprise the user;
            # better to nudge them via the stat line.
            self._stat_label.setText(
                "<span style='color:#c93'>Select one or more rows first "
                "(click + Shift-click for a range), then apply.</span>")
            return
        for row in rows:
            if 0 <= row < len(self._checkboxes):
                _, cb = self._checkboxes[row]
                cb.setChecked(checked)
        self._refresh_stat_line()

    def _apply_rate_threshold(self):
        """Uncheck scans whose PMT rate is below the spinbox value.

        Additive: an already-unchecked scan stays unchecked. A scan
        that currently has rate >= threshold is left in its current
        state (we don't re-check anything; the user might have
        manually excluded a high-rate scan for their own reason).
        """
        threshold = float(self._rate_threshold.value())
        for (scan_idx, cb), row in zip(self._checkboxes, self._rows):
            if float(row["rate_hz"]) < threshold and cb.isChecked():
                cb.setChecked(False)
        self._refresh_stat_line()

    def _refresh_stat_line(self):
        n = len(self._checkboxes)
        n_excluded = sum(1 for _, cb in self._checkboxes
                         if not cb.isChecked())
        n_included = n - n_excluded
        if n == 0:
            self._stat_label.setText("No scans found in this file.")
            return
        # Show counts plus the rate range to give the user a sense of
        # the cut they're considering. min/max ignored when n_included
        # is 0 (rare but possible if user unchecks everything).
        rates = [r["rate_hz"] for (_, cb), r
                 in zip(self._checkboxes, self._rows) if cb.isChecked()]
        if rates:
            rmin, rmax = min(rates), max(rates)
            rate_str = (f"  &nbsp;rate: {rmin:.3f}–{rmax:.3f} Hz"
                        f" (median {sorted(rates)[len(rates)//2]:.3f})")
        else:
            rate_str = ""
        self._stat_label.setText(
            f"Including <b>{n_included}</b> of {n} scans "
            f"<span style='color:#888'>(excluding {n_excluded})</span>"
            f"{rate_str}")
        self._stat_label.setTextFormat(Qt.TextFormat.RichText)

    # ── Persist ─────────────────────────────────────────────────────

    def _on_accept(self):
        # Build the excluded set from the unchecked rows; only scans
        # actually present in the file can be excluded (you can't
        # toggle a scan you can't see). The registry's set() handles
        # the empty-set-clears-entry case so a file with everything
        # checked drops out of the YAML cleanly.
        excluded = {scan_idx for scan_idx, cb in self._checkboxes
                    if not cb.isChecked()}
        self._registry.set(self._asdf_path, excluded)
        self.accept()
