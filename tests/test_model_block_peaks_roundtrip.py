"""A saved HFS model comes back with its own hyperfine lines.

The peak rows are generated from I, Jl and Ju. from_dict restores
those spins with signals blocked, so nothing rebuilt the rows: the
table still described the spin the block was CREATED with (the
Ge-like default, I = 7/2), and every saved intensity was dropped
because no label matched. A 171Yb model saved with three free
intensities reloaded with 21 Racah defaults -- a fit that then used
the wrong line strengths without saying so (found 2026-09-25 while
generating a Yb calibration session).

Run from the project root:
    .venv/Scripts/python.exe -m pytest \\
        tests/test_model_block_peaks_roundtrip.py -q
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui.analysis.blocks import ModelBlock  # noqa: E402


class PeakRoundTripTests(unittest.TestCase):

    def _block(self, I=None, Jl=None, Ju=None, racah=True):
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        for row in b._param_rows:
            if row["name"] == "I" and I is not None:
                row["value"].setValue(I)
            elif row["name"] == "Jl" and Jl is not None:
                row["value"].setValue(Jl)
            elif row["name"] == "Ju" and Ju is not None:
                row["value"].setValue(Ju)
        b._racah_check.setChecked(racah)
        if I is not None or Jl is not None or Ju is not None:
            b._rebuild_peak_table()
        return b

    @staticmethod
    def _labels(block):
        return [block._peaks_table.item(i, 0).text()
                for i in range(block._peaks_table.rowCount())
                if block._peaks_table.item(i, 0)]

    @staticmethod
    def _amps(block):
        return {block._peaks_table.item(i, 0).text():
                round(block._peaks_table.cellWidget(i, 1).value(), 6)
                for i in range(block._peaks_table.rowCount())
                if block._peaks_table.item(i, 0)}

    def test_a_half_spin_model_has_three_lines(self):
        """171Yb+: I = 1/2, S1/2 -> P1/2."""
        b = self._block(I=0.5, Jl=0.5, Ju=0.5, racah=False)
        self.assertEqual(sorted(self._labels(b)),
                         ["0to1", "1to0", "1to1"])

    def test_the_lines_survive_a_save_and_load(self):
        a = self._block(I=0.5, Jl=0.5, Ju=0.5, racah=False)
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        b.from_dict(a.to_dict())
        self.assertEqual(sorted(self._labels(b)),
                         ["0to1", "1to0", "1to1"])

    def test_the_intensities_survive(self):
        a = self._block(I=0.5, Jl=0.5, Ju=0.5, racah=False)
        for i in range(a._peaks_table.rowCount()):
            a._peaks_table.cellWidget(i, 1).setValue(0.25 * (i + 1))
        saved = self._amps(a)
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        b.from_dict(a.to_dict())
        self.assertEqual(self._amps(b), saved)

    def test_a_fixed_intensity_stays_fixed(self):
        """The strongest line is usually pinned to 1."""
        from PySide6.QtWidgets import QCheckBox
        a = self._block(I=0.5, Jl=0.5, Ju=0.5, racah=False)
        a._peaks_table.cellWidget(0, 1).setValue(1.0)
        a._peaks_table.cellWidget(0, 2).findChild(QCheckBox).setChecked(False)
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        b.from_dict(a.to_dict())
        self.assertFalse(
            b._peaks_table.cellWidget(0, 2).findChild(QCheckBox).isChecked())

    def test_a_five_halves_model_too(self):
        """173Yb+: I = 5/2 gives four lines, not three."""
        a = self._block(I=2.5, Jl=0.5, Ju=0.5, racah=False)
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        b.from_dict(a.to_dict())
        self.assertEqual(sorted(self._labels(b)),
                         ["2to2", "2to3", "3to2", "3to3"])

    def test_the_fit_config_carries_the_right_lines(self):
        """get_model_config is what reaches the fitter."""
        a = self._block(I=0.5, Jl=0.5, Ju=0.5, racah=False)
        for i in range(a._peaks_table.rowCount()):
            a._peaks_table.cellWidget(i, 1).setValue(0.5)
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        b.from_dict(a.to_dict())
        cfg = b.get_model_config()
        self.assertEqual(sorted(cfg["peak_amplitudes"]),
                         ["0to1", "1to0", "1to1"])

    def test_an_unchanged_spin_is_left_alone(self):
        """The rebuild imports satlas2, so it only runs when the saved
        labels differ from the ones already in the table."""
        a = self._block(racah=False)
        before = self._labels(a)
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        b.from_dict(a.to_dict())
        self.assertEqual(self._labels(b), before)

    def test_a_save_without_amplitudes_still_loads(self):
        a = self._block(I=0.5, Jl=0.5, Ju=0.5, racah=False)
        d = a.to_dict()
        d.pop("peak_amplitudes")
        b = ModelBlock()
        self.addCleanup(b.deleteLater)
        b.from_dict(d)          # must not raise


if __name__ == "__main__":
    unittest.main()
