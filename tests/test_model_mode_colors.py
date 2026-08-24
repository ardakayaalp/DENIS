"""The Mode cell's colour must agree with its text after a project load.

Date:    2026-07-14
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

The ModelBlock parameter table colour-codes each row's constraint Mode --
green for Free, red for Fixed -- and the colour is what the eye actually
reads; nobody scans fourteen rows of small text. The tint is driven by
``currentTextChanged``, but ``from_dict`` has to set the combo with
signals blocked (the handler applies constraints and can raise dialogs),
so nothing repainted the cells on load: a parameter saved as Fixed came
back showing the word "Fixed" on the green Free background it happened to
be constructed with. The colour said the opposite of the truth.

The bug only bit the rows whose *default* differs from the saved value,
which is why it looked so arbitrary in practice -- Al/Au/Bl/Bu (default
Free) went wrong while I/Jl/Ju (locked Fixed) stayed right.

Run from the project root with the project's interpreter:

    .venv/Scripts/python.exe -m unittest tests.test_model_mode_colors -v

Depends on: gui.analysis.blocks (ModelBlock, _MODE_COLORS); PySide6.
"""

import re
import unittest

from PySide6.QtWidgets import QApplication

_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _cell_colour(combo):
    """The background colour actually painted on a Mode combo."""
    m = re.search(r"background-color:\s*(#[0-9a-fA-F]{6})", combo.styleSheet())
    return m.group(1) if m else None


class ModeColourRoundTripTests(unittest.TestCase):

    def setUp(self):
        _ensure_app()
        from gui.analysis.blocks import ModelBlock, _MODE_COLORS
        self.ModelBlock = ModelBlock
        self.colours = _MODE_COLORS

    def _reloaded(self, fixed=(), free=()):
        """A ModelBlock restored from a project with these modes saved."""
        d = self.ModelBlock("M1").to_dict()
        for name in fixed:
            if name in d["params"]:
                d["params"][name]["vary"] = False      # -> Fixed
                d["params"][name]["expr"] = ""
        for name in free:
            if name in d["params"]:
                d["params"][name]["vary"] = True       # -> Free
                d["params"][name]["expr"] = ""
        mb = self.ModelBlock("M1")
        mb.from_dict(d)
        return mb

    def _assert_consistent(self, mb):
        wrong = []
        for row in mb._param_rows:
            text = row["mode"].currentText()
            want = self.colours.get(text)
            got = _cell_colour(row["mode"])
            if want is not None and got != want:
                wrong.append(f"{row['name']}: text={text} colour={got} "
                             f"expected={want}")
        self.assertEqual(wrong, [], "Mode cells whose colour contradicts "
                                    "their text:\n  " + "\n  ".join(wrong))

    def test_a_param_saved_as_fixed_reloads_red_not_green(self):
        """The reported bug, exactly: Al/Au/Bl/Bu default to Free, so before
        the fix they reloaded saying "Fixed" on a green background."""
        mb = self._reloaded(fixed=("Al", "Au", "Bl", "Bu"))
        for name in ("Al", "Au", "Bl", "Bu"):
            row = next(r for r in mb._param_rows if r["name"] == name)
            self.assertEqual(row["mode"].currentText(), "Fixed")
            self.assertEqual(_cell_colour(row["mode"]),
                             self.colours["Fixed"], f"{name} should be red")

    def test_a_param_saved_as_free_reloads_green(self):
        mb = self._reloaded(free=("centroid", "scale", "FWHMG", "FWHML"))
        for name in ("centroid", "scale", "FWHMG", "FWHML"):
            row = next(r for r in mb._param_rows if r["name"] == name)
            self.assertEqual(row["mode"].currentText(), "Free")
            self.assertEqual(_cell_colour(row["mode"]),
                             self.colours["Free"], f"{name} should be green")

    def test_every_row_agrees_with_its_own_text(self):
        """The general invariant, over a mixed project."""
        mb = self._reloaded(
            fixed=("Al", "Au", "Bl", "Bu", "Cl", "Cu"),
            free=("centroid", "scale", "FWHMG", "FWHML", "Bkg_p0"))
        self._assert_consistent(mb)

    def test_flipping_free_to_fixed_and_back_stays_consistent(self):
        """A second load must not leave the previous load's colour behind."""
        mb = self._reloaded(fixed=("Al", "Au"))
        d = mb.to_dict()
        for name in ("Al", "Au"):
            d["params"][name]["vary"] = True        # now Free
        mb.from_dict(d)
        self._assert_consistent(mb)
        for name in ("Al", "Au"):
            row = next(r for r in mb._param_rows if r["name"] == name)
            self.assertEqual(_cell_colour(row["mode"]), self.colours["Free"])

    def test_locked_physics_params_stay_red(self):
        """I / Jl / Ju are metadata, not fitted: always Fixed, always red."""
        mb = self._reloaded(free=("centroid",))
        for name in ("I", "Jl", "Ju"):
            row = next((r for r in mb._param_rows if r["name"] == name), None)
            if row is None:
                continue
            self.assertEqual(row["mode"].currentText(), "Fixed")
            self.assertEqual(_cell_colour(row["mode"]), self.colours["Fixed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
