"""Tests for WP8 UI fixes (2026-06-02 code review).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers the timestamp-unit gate rescale (#21 ts-unit-change-shifts-gate): when
the display unit changes, PreAnalysisTab._on_ts_unit_changed must rescale the
lo/hi/binsize spinboxes by old_div/new_div so the ABSOLUTE time window (value x
divisor seconds) is preserved, instead of silently moving the gate to a
different physical slice. Driven on a lightweight stub so no QApplication is
needed.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_wp8_ui_fixes -v

Depends on: gui.preanalysis_tab.PreAnalysisTab (_on_ts_unit_changed,
_ts_unit_divisor).
"""

import unittest


class _FakeSpin:
    def __init__(self, value):
        self._v = value

    def value(self):
        return self._v

    def setValue(self, v):
        self._v = v

    def blockSignals(self, _b):
        pass


class _FakeCombo:
    def __init__(self, text):
        self._t = text

    def currentText(self):
        return self._t


class _StubTab:
    def __init__(self, unit, lo, hi, binsize, prev_div):
        self._ts_unit = _FakeCombo(unit)
        self._ts_lo = _FakeSpin(lo)
        self._ts_hi = _FakeSpin(hi)
        self._ts_binsize = _FakeSpin(binsize)
        self._ts_prev_divisor = prev_div
        self.replotted = False

    def _schedule_replot(self):
        self.replotted = True

    def _ts_unit_divisor(self):
        # Reuse the real seconds-per-unit mapping against the stub's combo.
        from gui.preanalysis_tab import PreAnalysisTab
        return PreAnalysisTab._ts_unit_divisor(self)


class TimestampUnitGateRescaleTests(unittest.TestCase):
    def _run(self, stub):
        from gui.preanalysis_tab import PreAnalysisTab
        PreAnalysisTab._on_ts_unit_changed(stub)

    def test_seconds_to_minutes_preserves_absolute_window(self):
        # Gate [60, 120] s, binsize 1 s. Switching to Minutes (div 60) must
        # rescale to [1, 2] min, 1/60 min -> same absolute seconds.
        stub = _StubTab("Minutes", 60.0, 120.0, 1.0, prev_div=1.0)
        self._run(stub)
        self.assertAlmostEqual(stub._ts_lo.value(), 1.0)
        self.assertAlmostEqual(stub._ts_hi.value(), 2.0)
        self.assertAlmostEqual(stub._ts_binsize.value(), 1.0 / 60.0)
        # Absolute window unchanged: value * new_div == original seconds.
        self.assertAlmostEqual(stub._ts_lo.value() * 60.0, 60.0)
        self.assertAlmostEqual(stub._ts_hi.value() * 60.0, 120.0)
        self.assertEqual(stub._ts_prev_divisor, 60.0)
        self.assertTrue(stub.replotted)

    def test_minutes_to_hours_preserves_absolute_window(self):
        # [2, 4] min with prev_div=60 -> Hours (div 3600): [1/30, 1/15] h.
        stub = _StubTab("Hours", 2.0, 4.0, 1.0, prev_div=60.0)
        self._run(stub)
        self.assertAlmostEqual(stub._ts_lo.value() * 3600.0, 2.0 * 60.0)
        self.assertAlmostEqual(stub._ts_hi.value() * 3600.0, 4.0 * 60.0)

    def test_no_change_when_divisor_same(self):
        stub = _StubTab("Seconds", 10.0, 20.0, 1.0, prev_div=1.0)
        self._run(stub)
        self.assertEqual(stub._ts_lo.value(), 10.0)  # unchanged
        self.assertEqual(stub._ts_hi.value(), 20.0)
        self.assertTrue(stub.replotted)


if __name__ == "__main__":
    unittest.main()
