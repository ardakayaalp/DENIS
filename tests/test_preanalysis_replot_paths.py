"""Which redraw a Pre-Analysis input asks for.

Typing in the Z / A boxes was seconds of lag per keystroke (Arda,
2026-09-24). Two reasons, both fixed here:

* every change asked for a FULL replot, which redraws the calibration
  and cooler panels -- 1.3 s of a 2.0 s replot, measured on his T04
  session -- and those are drawn from the recorded VOLTAGES, upstream
  of the Doppler conversion, so a change of isotope cannot move a
  pixel of them;
* the spin boxes tracked every keystroke, so "70" was two changes.

The frequency-axis inputs (Z, A, mass, harmonic, energy levels) now
schedule a spectrum recompute instead. Not the gate-drag blit: that
one reuses a cached line whose x-data is still valid, and here the
x-axis is exactly what moved.

Run from the project root:
    .venv/Scripts/python.exe -m pytest \\
        tests/test_preanalysis_replot_paths.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])


class _Recorder:
    """Stands in for the three redraw routines and remembers which
    one ran."""

    def __init__(self, project):
        self.calls = []
        project._replot = lambda spectrum_only=False: self.calls.append(
            "spectrum" if spectrum_only else "full")
        project._replot_spectrum_fast = lambda: self.calls.append("blit")
        project._replot_calibrations = lambda: self.calls.append(
            "calibrations")


class ReplotPathTests(unittest.TestCase):

    def setUp(self):
        from gui.preanalysis_tab import PreAnalysisTab
        self.p = PreAnalysisTab()
        self.addCleanup(self.p.deleteLater)
        self.p._replot_timer.stop()
        self.rec = _Recorder(self.p)

    def _fire(self):
        """Run what the debounce timer would have run."""
        self.p._on_replot_timer()

    # -- what each input asks for -----------------------------------
    def test_changing_A_asks_for_a_spectrum_recompute(self):
        self.p._on_a_changed(72)
        self._fire()
        self.assertEqual(self.rec.calls, ["spectrum"])

    def test_changing_Z_asks_for_a_spectrum_recompute(self):
        self.p._on_z_changed(32)
        self._fire()
        self.assertEqual(self.rec.calls, ["spectrum"])

    def test_the_calibration_panels_are_not_redrawn(self):
        """They are drawn from the recorded voltages, upstream of the
        Doppler conversion: a change of isotope cannot move them."""
        self.p._on_a_changed(72)
        self._fire()
        self.assertNotIn("calibrations", self.rec.calls)

    def test_the_mass_the_harmonic_and_the_levels_go_the_same_way(self):
        for spin, value in ((self.p._mass_spin, 71.92),
                            (self.p._harmonic, 3),
                            (self.p._e_lower, 1.5),
                            (self.p._e_upper, 2.5)):
            self.rec.calls.clear()
            self.p._pending_spectrum_recompute = False
            self.p._replot_timer.stop()
            spin.setValue(value)
            self._fire()
            self.assertEqual(self.rec.calls, ["spectrum"],
                             f"{spin.objectName() or spin} -> "
                             f"{self.rec.calls}")

    def test_a_gate_drag_still_blits(self):
        self.p._schedule_replot_light()
        self._fire()
        self.assertEqual(self.rec.calls, ["blit"])

    def test_anything_else_still_redraws_everything(self):
        self.p._schedule_replot()
        self._fire()
        self.assertEqual(self.rec.calls, ["full"])

    # -- how the three coalesce in one debounce window --------------
    def test_a_queued_full_replot_wins(self):
        """It covers the spectrum too, so downgrading would lose the
        rest of the redraw."""
        self.p._schedule_replot()
        self.p._schedule_replot_spectrum()
        self._fire()
        self.assertEqual(self.rec.calls, ["full"])

    def test_a_queued_blit_is_upgraded(self):
        """The blit reuses a cached line whose x-data just moved."""
        self.p._schedule_replot_light()
        self.p._schedule_replot_spectrum()
        self._fire()
        self.assertEqual(self.rec.calls, ["spectrum"])

    def test_a_full_replot_asked_for_afterwards_still_wins(self):
        self.p._schedule_replot_spectrum()
        self.p._schedule_replot()
        self._fire()
        self.assertEqual(self.rec.calls, ["full"])

    def test_a_gate_drag_afterwards_does_not_downgrade_it(self):
        self.p._schedule_replot_spectrum()
        self.p._schedule_replot_light()
        self._fire()
        self.assertEqual(self.rec.calls, ["spectrum"])

    def test_the_flag_does_not_survive_its_replot(self):
        self.p._schedule_replot_spectrum()
        self._fire()
        self.rec.calls.clear()
        self.p._schedule_replot()
        self._fire()
        self.assertEqual(self.rec.calls, ["full"])

    def test_many_changes_coalesce_into_one_redraw(self):
        for a in (70, 71, 72, 73):
            self.p._on_a_changed(a)
        self._fire()
        self.assertEqual(self.rec.calls, ["spectrum"])


class ReplotsDuringALoadTests(unittest.TestCase):
    """A restore sets dozens of widgets, each of which would schedule a
    replot. The event loop is pumped during a load (so the app does not
    look hung to Windows), so those would actually fire -- once per
    file, seconds each. The restore ends with a replot of its own."""

    def setUp(self):
        from gui.preanalysis_tab import PreAnalysisTab
        self.p = PreAnalysisTab()
        self.addCleanup(self.p.deleteLater)
        self.p._replot_timer.stop()
        self.rec = _Recorder(self.p)
        import gui.load_progress as lp
        self.lp = lp
        self.addCleanup(setattr, lp, "_active", None)

    def _loading(self, on):
        self.lp._active = object() if on else None

    def test_a_full_replot_stands_down(self):
        self._loading(True)
        self.p._schedule_replot()
        self.assertFalse(self.p._replot_timer.isActive())

    def test_a_spectrum_replot_stands_down(self):
        self._loading(True)
        self.p._on_a_changed(72)
        self.assertFalse(self.p._replot_timer.isActive())

    def test_a_gate_drag_replot_stands_down(self):
        self._loading(True)
        self.p._schedule_replot_light()
        self.assertFalse(self.p._replot_timer.isActive())

    def test_the_restores_own_replot_still_runs(self):
        """_restore_from_dict calls _replot() directly at the end --
        suppressing the SCHEDULER must not suppress that."""
        self._loading(True)
        self.p._replot(spectrum_only=False)
        self.assertEqual(self.rec.calls, ["full"])

    def test_scheduling_works_again_afterwards(self):
        self._loading(True)
        self.p._schedule_replot()
        self._loading(False)
        self.p._schedule_replot()
        self.assertTrue(self.p._replot_timer.isActive())


class KeyboardTrackingTests(unittest.TestCase):
    """Typing "70" must be one change, not one per digit."""

    def setUp(self):
        from gui.preanalysis_tab import PreAnalysisTab
        self.p = PreAnalysisTab()
        self.addCleanup(self.p.deleteLater)

    def test_the_frequency_axis_inputs_settle_before_they_fire(self):
        for name in ("_z_spin", "_a_spin", "_mass_spin", "_harmonic",
                     "_e_lower", "_e_upper"):
            self.assertFalse(getattr(self.p, name).keyboardTracking(),
                             name)

    def test_stepping_still_fires_immediately(self):
        """Keyboard tracking is about typing; an arrow click is a
        finished value."""
        seen = []
        self.p._a_spin.valueChanged.connect(seen.append)
        self.p._a_spin.stepUp()
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
