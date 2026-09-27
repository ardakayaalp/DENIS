"""The window that shows a session file being loaded.

Restoring a save opens every run named in it -- half a minute on a
campaign, during which DENIS used to look hung. The window says what
it is on.

Two properties matter more than the pixels:

* it must not re-enter the event loop, or a queued timer would fire
  and a click would reach a half-rebuilt UI. It paints itself with
  repaint() and lays itself out by hand, because the layout request it
  would normally queue is never serviced;
* reporting must cost nothing when no window is up, since the loaders
  call it unconditionally.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_load_progress.py -q
"""
import os
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui import load_progress as lp  # noqa: E402

PHASES = [("registries", "Scan filters", 1),
          ("preanalysis", "Pre-Analysis", 45),
          ("analysis", "Analysis", 45),
          ("layout", "Window layout", 2)]


class ReportingWithoutAWindowTests(unittest.TestCase):
    """The loaders call report() whether or not anyone is watching."""

    def test_it_is_a_no_op(self):
        self.assertIsNone(lp.active())
        lp.report("run_7947.asdf", 3, 40)      # must not raise
        lp.report()

    def test_active_is_none(self):
        self.assertIsNone(lp.active())

    def test_nothing_is_loading(self):
        self.assertFalse(lp.is_loading())


class WindowTests(unittest.TestCase):

    def setUp(self):
        self.parent = QWidget()
        self.addCleanup(self.parent.deleteLater)
        self.dlg = lp.LoadProgressDialog.begin(
            self.parent, "T04.yaml", PHASES)
        self.addCleanup(self.dlg.finish)

    def test_it_becomes_the_reporting_target(self):
        self.assertIs(lp.active(), self.dlg)

    def test_finishing_clears_it(self):
        self.dlg.finish()
        self.assertIsNone(lp.active())

    def test_it_says_a_load_is_in_progress(self):
        self.assertTrue(lp.is_loading())
        self.dlg.finish()
        self.assertFalse(lp.is_loading())

    def test_it_names_the_file(self):
        self.assertIn("T04.yaml", self.dlg._title.text())

    def test_it_is_modal(self):
        """So a click cannot reach the half-rebuilt window behind it."""
        self.assertEqual(self.dlg.windowModality(),
                         Qt.WindowModality.ApplicationModal)

    def test_it_cannot_be_closed(self):
        """There is nothing to cancel back to: a half-applied restore
        is a worse state than a slow one."""
        self.assertFalse(
            self.dlg.windowFlags() & Qt.WindowType.WindowCloseButtonHint)

    def test_the_phase_label_follows_the_phase(self):
        self.dlg.phase("analysis")
        self.assertEqual(self.dlg._phase_label.text(), "Analysis")

    def test_the_bar_moves_within_the_phase(self):
        self.dlg.phase("preanalysis")          # spans 1/93 .. 46/93
        start = self.dlg._bar.value()
        self.dlg.report("run_1.asdf", 1, 2)
        half = self.dlg._bar.value()
        self.dlg.report("run_2.asdf", 2, 2)
        end = self.dlg._bar.value()
        self.assertLess(start, half)
        self.assertLess(half, end)
        self.assertLessEqual(end, lp._SCALE)

    def test_the_phases_partition_the_bar(self):
        """A file with no Pre-Analysis section must not leave a hole in
        the middle of the bar."""
        dlg = lp.LoadProgressDialog.begin(
            self.parent, "x.yaml",
            [("a", "A", 1), ("b", "B", 1)])
        self.addCleanup(dlg.finish)
        dlg.phase("a")
        dlg.report("", 1, 1)
        self.assertEqual(dlg._bar.value(), lp._SCALE // 2)
        dlg.phase("b")
        dlg.report("", 1, 1)
        self.assertEqual(dlg._bar.value(), lp._SCALE)

    def test_an_unknown_phase_does_not_throw(self):
        self.dlg.phase("something else")
        self.assertEqual(self.dlg._phase_label.text(), "something else")

    def test_a_report_without_numbers_only_changes_the_text(self):
        self.dlg.phase("analysis")
        value = self.dlg._bar.value()
        self.dlg.report("run_7947.asdf")
        self.assertEqual(self.dlg._bar.value(), value)
        self.assertIn("7947", self.dlg._detail.text())

    def test_a_long_name_is_elided_not_clipped(self):
        """The label has a fixed height and no chance to re-lay-out,
        so it has to shorten the text itself."""
        self.dlg.report("run_" + "x" * 400 + ".asdf")
        self.assertLess(len(self.dlg._detail.text()), 200)
        self.assertIn("…", self.dlg._detail.text())

    def test_the_value_is_not_throttled_even_when_painting_is(self):
        """Painting twice in a millisecond is waste; losing the
        position is a bug."""
        self.dlg.phase("analysis")
        self.dlg.report("a", 1, 100)
        first = self.dlg._bar.value()
        self.dlg.report("b", 99, 100)
        self.assertGreater(self.dlg._bar.value(), first)

    def test_the_module_level_report_reaches_it(self):
        self.dlg.phase("analysis")
        lp.report("run_9999.asdf")
        self.assertIn("9999", self.dlg._detail.text())

    def test_the_context_manager_closes_it(self):
        with lp.LoadProgressDialog.begin(self.parent, "y.yaml",
                                         PHASES) as d:
            self.assertIs(lp.active(), d)
        self.assertIsNone(lp.active())


class PumpingTests(unittest.TestCase):
    """The window must keep the application alive while it works.

    An app that does not pump its message queue for half a minute is
    "Not Responding" to Windows, which offers to close it -- and doing
    that kills DENIS mid-restore. That is a crash from the outside,
    with nothing in the log but the banner (2026-09-25).
    """

    def setUp(self):
        self.parent = QWidget()
        self.addCleanup(self.parent.deleteLater)
        self.dlg = lp.LoadProgressDialog.begin(
            self.parent, "T01.yaml", PHASES)
        self.addCleanup(self.dlg.finish)

    def test_a_report_pumps_the_event_loop(self):
        from PySide6.QtCore import QTimer
        fired = []
        QTimer.singleShot(0, lambda: fired.append(True))
        self.dlg.phase("analysis")          # forces a paint
        self.assertTrue(fired, "a queued timer did not run: the app is "
                              "frozen for the whole load")

    def test_a_deferred_delete_is_left_alone(self):
        """The restore deletes every project widget with deleteLater.
        Destroying them mid-load is a use-after-free through
        _apply_zoom, which touches every widget in the process."""
        doomed = QWidget()
        doomed.setObjectName("doomed")
        doomed.deleteLater()
        self.dlg.phase("analysis")
        try:
            name = doomed.objectName()
        except RuntimeError:
            name = None
        self.assertEqual(name, "doomed")

    def test_it_pumps_with_user_input_excluded(self):
        """The flag is the protection: a click or keystroke from the OS
        must not reach a half-rebuilt UI. (Qt only filters SPONTANEOUS
        input that way, so this checks the call, not a synthetic
        posted event -- which Qt delivers regardless.)"""
        from PySide6.QtCore import QCoreApplication, QEventLoop
        flags = []
        real = QCoreApplication.processEvents
        QCoreApplication.processEvents = staticmethod(
            lambda *a, **k: flags.append(a[0] if a else None))
        try:
            self.dlg.phase("analysis")
        finally:
            QCoreApplication.processEvents = real
        self.assertTrue(flags, "no pump at all")
        self.assertTrue(
            all(f == QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                for f in flags), flags)

    def test_it_is_modal_on_top_of_that(self):
        self.assertEqual(self.dlg.windowModality(),
                         Qt.WindowModality.ApplicationModal)


class SilentModeTests(unittest.TestCase):
    """The window can be switched off without losing the pumping.

    The pumping is what stops Windows declaring the app dead and
    offering to close it; the window is the part that can interact
    with the rest of the desktop. When a load is crashing, being able
    to take the window out of the picture without going back to a
    frozen app is the difference between a diagnosis and a guess.
    """

    def setUp(self):
        self.parent = QWidget()
        self.addCleanup(self.parent.deleteLater)

    def test_the_setting_picks_the_silent_one(self):
        p = lp.begin(self.parent, "T01.yaml", PHASES, show=False)
        self.addCleanup(p.finish)
        self.assertIsInstance(p, lp.SilentProgress)
        self.assertNotIsInstance(p, lp.LoadProgressDialog)

    def test_the_default_is_the_window(self):
        p = lp.begin(self.parent, "T01.yaml", PHASES, show=True)
        self.addCleanup(p.finish)
        self.assertIsInstance(p, lp.LoadProgressDialog)

    def test_it_still_counts_as_loading(self):
        """So the Pre-Analysis replots stand down either way."""
        p = lp.begin(self.parent, "T01.yaml", PHASES, show=False)
        self.addCleanup(p.finish)
        self.assertTrue(lp.is_loading())
        p.finish()
        self.assertFalse(lp.is_loading())

    def test_it_still_pumps(self):
        from PySide6.QtCore import QTimer
        p = lp.begin(self.parent, "T01.yaml", PHASES, show=False)
        self.addCleanup(p.finish)
        fired = []
        QTimer.singleShot(0, lambda: fired.append(True))
        p.phase("analysis")
        self.assertTrue(fired, "silent mode must still keep the app alive")

    def test_it_creates_no_window(self):
        from PySide6.QtWidgets import QApplication
        before = len(QApplication.topLevelWidgets())
        p = lp.begin(self.parent, "T01.yaml", PHASES, show=False)
        self.addCleanup(p.finish)
        p.phase("analysis")
        p.report("run_7947.asdf", 1, 2)
        self.assertEqual(len(QApplication.topLevelWidgets()), before)

    def test_reporting_through_the_module_works(self):
        p = lp.begin(self.parent, "T01.yaml", PHASES, show=False)
        self.addCleanup(p.finish)
        lp.report("run_7947.asdf", 1, 2)      # must not raise

    def test_a_broken_settings_file_still_gives_a_window(self):
        import gui.shared_widgets as sw
        real = sw._load_settings
        sw._load_settings = lambda: (_ for _ in ()).throw(OSError("gone"))
        self.addCleanup(setattr, sw, "_load_settings", real)
        p = lp.begin(self.parent, "T01.yaml", PHASES)
        self.addCleanup(p.finish)
        self.assertIsInstance(p, lp.LoadProgressDialog)


class LoadIntegrationTests(unittest.TestCase):
    """A real load drives it."""

    def setUp(self):
        import tempfile
        import shutil
        from PySide6.QtWidgets import QMessageBox
        # Force the window on. Without this the test reads the real
        # settings file and passes or fails depending on whether the
        # developer happens to have switched the window off.
        import gui.shared_widgets as sw
        real_settings = sw._load_settings
        sw._load_settings = lambda: {lp.SETTING: True}
        self.addCleanup(setattr, sw, "_load_settings", real_settings)
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name in ("information", "warning", "critical"):
            real = getattr(QMessageBox, name)
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
            self.addCleanup(setattr, QMessageBox, name, real)
        self.seen = []
        real_report = lp.LoadProgressDialog.report
        real_phase = lp.LoadProgressDialog.phase

        def report(dlg, text="", done=None, total=None):
            self.seen.append(("report", dlg._phase_label.text(), text))
            return real_report(dlg, text, done, total)

        def phase(dlg, key, detail=""):
            self.seen.append(("phase", key, detail))
            return real_phase(dlg, key, detail)

        lp.LoadProgressDialog.report = report
        lp.LoadProgressDialog.phase = phase
        self.addCleanup(setattr, lp.LoadProgressDialog, "report",
                        real_report)
        self.addCleanup(setattr, lp.LoadProgressDialog, "phase",
                        real_phase)

    @classmethod
    def setUpClass(cls):
        """One window for the class.

        Forcing a MainWindow to be destroyed and building another is a
        use-after-free: MainWindow._apply_zoom walks
        QApplication.allWidgets() and touches every one of them, so a
        widget torn down outside the event loop is a dangling pointer
        in that list (heap corruption, seen 2026-09-24).
        """
        from gui.main_window import MainWindow
        cls.window = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.window.deleteLater()

    def _load(self, data):
        import yaml
        path = os.path.join(self.dir, "session.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh)
        self.window._load_from_path(path)
        return self.window

    def test_a_load_walks_the_phases_in_order(self):
        self._load({"analysis": {"projects": [
            {"project_name": "74Ge", "blocks": []},
            {"project_name": "70Ge", "blocks": []}]}})
        phases = [k for kind, k, _d in self.seen if kind == "phase"]
        self.assertEqual(phases, ["registries", "analysis", "layout"])

    def test_a_missing_section_is_not_a_phase(self):
        """No Pre-Analysis in the file, no Pre-Analysis in the bar."""
        self._load({"analysis": {"projects": []}})
        phases = [k for kind, k, _d in self.seen if kind == "phase"]
        self.assertNotIn("preanalysis", phases)

    def test_every_project_is_named_as_it_is_rebuilt(self):
        self._load({"analysis": {"projects": [
            {"project_name": "74Ge", "blocks": []},
            {"project_name": "70Ge", "blocks": []}]}})
        named = [t for kind, _p, t in self.seen if kind == "report"]
        self.assertIn("74Ge", named)
        self.assertIn("70Ge", named)

    def test_the_window_is_gone_afterwards(self):
        self._load({"analysis": {"projects": []}})
        self.assertIsNone(lp.active())

    def test_the_setting_is_honoured_by_a_real_load(self):
        """Switched off, a load must still work -- and make no window."""
        import gui.shared_widgets as sw
        sw._load_settings = lambda: {lp.SETTING: False}
        seen = []
        real = lp.SilentProgress.phase
        lp.SilentProgress.phase = lambda self, key, detail="": (
            seen.append(key), real(self, key, detail))[1]
        self.addCleanup(setattr, lp.SilentProgress, "phase", real)
        self._load({"analysis": {"projects": [
            {"project_name": "74Ge", "blocks": []}]}})
        self.assertIn("analysis", seen)
        self.assertIsNone(lp.active())


class PaintingFrozenTests(unittest.TestCase):
    """A restore rebuilds every tab, including three matplotlib
    canvases per Pre-Analysis project, while the event loop is pumped.
    Painting a half-built canvas is not something matplotlib or the
    Agg backend promise to survive, so the main window does not paint
    at all until the restore is finished."""

    def setUp(self):
        import tempfile
        import shutil
        from PySide6.QtWidgets import QMessageBox
        import gui.shared_widgets as sw
        real_settings = sw._load_settings
        sw._load_settings = lambda: {lp.SETTING: True}
        self.addCleanup(setattr, sw, "_load_settings", real_settings)
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)
        for name in ("information", "warning", "critical"):
            real = getattr(QMessageBox, name)
            setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
            self.addCleanup(setattr, QMessageBox, name, real)

    def _window(self):
        from gui.main_window import MainWindow
        w = MainWindow()
        self.addCleanup(w.deleteLater)
        return w

    def _load(self, w, data):
        import yaml
        path = os.path.join(self.dir, "s.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh)
        w._load_from_path(path)

    def test_updates_are_off_while_the_tabs_are_rebuilt(self):
        w = self._window()
        seen = []
        real = lp.LoadProgressDialog.report

        def spy(dlg, text="", done=None, total=None):
            seen.append(w.updatesEnabled())
            return real(dlg, text, done, total)

        lp.LoadProgressDialog.report = spy
        self.addCleanup(setattr, lp.LoadProgressDialog, "report", real)
        self._load(w, {"analysis": {"projects": [
            {"project_name": "74Ge", "blocks": []}]}})
        self.assertTrue(seen, "no progress reported at all")
        self.assertTrue(all(v is False for v in seen), seen)

    def test_painting_comes_back_afterwards(self):
        w = self._window()
        self._load(w, {"analysis": {"projects": []}})
        self.assertTrue(w.updatesEnabled())

    def test_painting_comes_back_even_when_the_load_fails(self):
        w = self._window()
        path = os.path.join(self.dir, "broken.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("just a string, not a mapping\n")
        w._load_from_path(path)
        self.assertTrue(w.updatesEnabled())


class DataStackPreloadTests(unittest.TestCase):
    """pandas pulls in pyarrow, which loads ~60 MB of native libraries
    (arrow.dll alone is 22 MB). Left to the lazy import that happened
    deep inside a session restore, with a dozen run files already in
    memory, and on Arda's machine it died there with an access
    violation during the DLL load (2026-09-25, stack in his session
    log). It is loaded while the process is still small instead."""

    @classmethod
    def setUpClass(cls):
        from gui.main_window import MainWindow
        cls.window = MainWindow()

    @classmethod
    def tearDownClass(cls):
        cls.window.deleteLater()

    def test_the_window_can_preload_it(self):
        self.window.preload_data_stack()
        self.assertIn("pandas", sys.modules)

    def test_it_is_safe_to_call_twice(self):
        self.window.preload_data_stack()
        self.window.preload_data_stack()
        self.assertIn("pandas", sys.modules)

    def test_a_load_imports_it_before_reading_anything(self):
        """A session opened in the first seconds, before the idle
        preload has run, must not leave it to the Pre-Analysis
        restore."""
        import inspect
        src = inspect.getsource(type(self.window)._load_from_path)
        before = src.index("import pandas")
        after = src.index("_restore_all")
        self.assertLess(before, after,
                        "pandas must be imported before the restore")

    def test_main_schedules_the_preload(self):
        import inspect
        from gui import main_window
        src = inspect.getsource(main_window.main)
        self.assertIn("preload_data_stack", src)
        self.assertIn("singleShot", src.split("preload_data_stack")[0][-80:])


if __name__ == "__main__":
    unittest.main()
