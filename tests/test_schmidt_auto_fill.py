"""Estimate tab: SchmidtWidget auto-fills the magnetic moment field.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

When the Schmidt single-particle estimate is enabled, the widget emits a
non-zero moment and writes it into the isotope panel's mu field, so an
isotope marked "Use Schmidt moment" never reaches the estimation
pipeline with mu=0 (which would divide by zero in the reference-scaling
step). Also covers from_dict population of a saved mu=0 spec, leaving a
user-supplied mu untouched, and routing the auto-fill through the shared
QUndoStack so Ctrl-Z reverts it while project loads do not dirty it.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_schmidt_auto_fill -v

Depends on: gui.estimate_tab (SchmidtWidget, IsotopePanel); PySide6 for
QApplication, QWidget, and QUndoStack.
"""

import unittest

from PySide6.QtWidgets import QApplication


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


class SchmidtAutoFillTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_enabling_schmidt_emits_a_non_zero_moment(self):
        from gui.estimate_tab import SchmidtWidget

        captured = []
        w = SchmidtWidget()
        w.moment_changed.connect(captured.append)

        # Pick a non-trivial single-particle config first so the
        # default orbital doesn't matter.
        w.single_radio.setChecked(True)
        idx = w.sp_orbital.findText("1f7/2")
        if idx >= 0:
            w.sp_orbital.setCurrentIndex(idx)
        idx = w.sp_type.findText("proton")
        if idx >= 0:
            w.sp_type.setCurrentIndex(idx)
        # Toggling on should emit at least one moment.
        w.enable_check.setChecked(True)
        self.assertTrue(captured)
        self.assertNotEqual(captured[-1], 0.0)

    def test_disabled_widget_does_not_emit(self):
        from gui.estimate_tab import SchmidtWidget

        captured = []
        w = SchmidtWidget()
        w.moment_changed.connect(captured.append)

        # Default state is disabled; changing spec must NOT emit.
        idx = w.sp_orbital.findText("1f7/2")
        if idx >= 0:
            w.sp_orbital.setCurrentIndex(idx)
        self.assertEqual(captured, [])

    def test_spec_change_after_enable_re_emits(self):
        from gui.estimate_tab import SchmidtWidget

        captured = []
        w = SchmidtWidget()
        w.moment_changed.connect(captured.append)
        w.enable_check.setChecked(True)
        n_initial = len(captured)
        # Change orbital -- expect a new emission.
        cur = w.sp_orbital.currentIndex()
        other = (cur + 1) % w.sp_orbital.count()
        w.sp_orbital.setCurrentIndex(other)
        self.assertGreater(len(captured), n_initial)

    def test_isotope_panel_mu_field_autofills_on_schmidt_enable(self):
        from gui.estimate_tab import IsotopePanel

        panel = IsotopePanel(index=0)
        # Pre-condition: μ=0, Schmidt off.
        self.assertEqual(panel.iso_mu.value(), 0.0)
        self.assertFalse(panel.schmidt.enable_check.isChecked())

        # Enable Schmidt with a sensible spec -- the panel's μ field
        # must end up non-zero so the estimation pipeline can scale
        # the reference moment.
        idx = panel.schmidt.sp_type.findText("proton")
        if idx >= 0:
            panel.schmidt.sp_type.setCurrentIndex(idx)
        idx = panel.schmidt.sp_orbital.findText("1f7/2")
        if idx >= 0:
            panel.schmidt.sp_orbital.setCurrentIndex(idx)
        panel.schmidt.enable_check.setChecked(True)
        self.assertNotEqual(panel.iso_mu.value(), 0.0)

    def test_from_dict_with_saved_zero_mu_populates(self):
        """Pre-fix YAML configs saved μ=0 next to an enabled Schmidt
        spec. On load, the parent's from_dict must populate μ so a
        re-run of the estimation succeeds without re-toggling the
        Schmidt widget."""
        from gui.estimate_tab import IsotopePanel

        panel = IsotopePanel(index=0)
        cfg = {
            "I": 3.5, "mu": 0.0, "Q": 0.0, "A": 73,
            "schmidt_valence": {"type": "proton", "orbital": "1f7/2"},
        }
        panel.from_dict(cfg)
        self.assertNotEqual(panel.iso_mu.value(), 0.0)

    def test_from_dict_with_custom_mu_does_not_overwrite(self):
        """A user-saved non-zero μ alongside an enabled Schmidt spec
        is left alone: respecting explicit user intent."""
        from gui.estimate_tab import IsotopePanel

        panel = IsotopePanel(index=0)
        cfg = {
            "I": 3.5, "mu": 1.234, "Q": 0.0, "A": 73,
            "schmidt_valence": {"type": "proton", "orbital": "1f7/2"},
        }
        panel.from_dict(cfg)
        self.assertAlmostEqual(panel.iso_mu.value(), 1.234)


class SchmidtAutoFillUndoTests(unittest.TestCase):
    """User complaint: ticking Schmidt overwrites μ but Ctrl-Z then
    doesn't revert it. The auto-fill must record on the same
    QUndoStack every other spinbox edit uses."""

    def setUp(self):
        _ensure_app()

    def test_auto_fill_pushes_to_undo_stack(self):
        from PySide6.QtGui import QUndoStack
        from PySide6.QtWidgets import QWidget
        from gui.estimate_tab import IsotopePanel

        # Wrap the panel in a parent that exposes ``_undo_stack``
        # like the real main window does, so ``self.window()`` in
        # ``_apply_schmidt_moment`` finds the stack.
        container = QWidget()
        container._undo_stack = QUndoStack()
        panel = IsotopePanel(index=0, parent=container)
        panel.iso_mu.setValue(1.234)

        # Tick Schmidt with a sensible spec.
        idx = panel.schmidt.sp_orbital.findText("1f7/2")
        if idx >= 0:
            panel.schmidt.sp_orbital.setCurrentIndex(idx)
        idx = panel.schmidt.sp_type.findText("proton")
        if idx >= 0:
            panel.schmidt.sp_type.setCurrentIndex(idx)
        panel.schmidt.enable_check.setChecked(True)

        schmidt_value = panel.iso_mu.value()
        self.assertNotEqual(schmidt_value, 1.234)
        # An undo command was pushed.
        self.assertGreaterEqual(container._undo_stack.count(), 1)
        self.assertTrue(container._undo_stack.canUndo())

        # Undo reverts to the pre-Schmidt value.
        container._undo_stack.undo()
        self.assertAlmostEqual(panel.iso_mu.value(), 1.234)

    def test_from_dict_does_not_dirty_undo_stack(self):
        """Loading a project must not leave fake 'Change value'
        entries on the undo stack -- only user-driven changes do."""
        from PySide6.QtGui import QUndoStack
        from PySide6.QtWidgets import QWidget
        from gui.estimate_tab import IsotopePanel

        container = QWidget()
        container._undo_stack = QUndoStack()
        panel = IsotopePanel(index=0, parent=container)

        cfg = {
            "I": 3.5, "mu": 0.0, "Q": 0.0, "A": 73,
            "schmidt_valence": {"type": "proton", "orbital": "1f7/2"},
        }
        # Pre-condition: empty stack.
        self.assertEqual(container._undo_stack.count(), 0)
        panel.from_dict(cfg)
        # Stack must still be empty (the auto-fill went through the
        # _bypass_undo path).
        self.assertEqual(container._undo_stack.count(), 0)
        self.assertNotEqual(panel.iso_mu.value(), 0.0)


if __name__ == "__main__":
    unittest.main()
