"""Tools ▸ ASDF Viewer, and "View ASDF…" on the run lists.

Arda (2026-09-26): "add a raw asdf viewer ... user drops asdf files and
the viewer shows the raw asdf entries, the header and the event tables.
Also add right click options on the file column in preanalysis tab and
analysis tab on the source block ... the tool ui opens with that run
file already added."

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_asdf_viewer.py -q
"""
import datetime as dt
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMenu  # noqa: E402

_APP = QApplication.instance() or QApplication([])

from gui.asdf_viewer import (  # noqa: E402
    AsdfViewer, MAX_LISTED_VALUES, column_stats, open_in_viewer, read_asdf)

EVENTS = np.array([[1.7e9, -900.0, 1, 3, 39.7, 2.9912],
                   [1.7e9, -899.0, 2, 4, 40.3, 2.9913],
                   [1.7e9, -898.0, 3, 3, 41.0, 2.9912]])
COLUMNS = ["timestamp", "voltage", "bunch_number", "channel", "time",
           "cooler"]


def write_run(folder, run=7507, name=None):
    import asdf
    tree = {"Run": run, "CoolerVoltage": 2.99128455,
            "Date": dt.datetime(2026, 5, 8, 13, 29, 36),
            "Experiment": "I340-V_CLS", "LaserSetpoint": 13522.68,
            "ScanningRanges": [[-900.0, -800.0], [600.0, 1000.0]],
            "CalSet": np.linspace(-900, 1000, 5),
            "CalReadback": np.linspace(-899.5, 1000.5, 5),
            "raw": EVENTS, "raw_header": COLUMNS}
    path = os.path.join(folder, name or f"run_{run}.asdf")
    asdf.AsdfFile(tree).write_to(path)
    return path


class ReadTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.path = write_run(cls.dir)
        cls.info = read_asdf(cls.path)

    def test_the_header_is_every_non_array_entry(self):
        keys = [k for k, _ in self.info["header"]]
        for k in ("Run", "CoolerVoltage", "Date", "Experiment",
                  "LaserSetpoint", "ScanningRanges", "raw_header"):
            self.assertIn(k, keys)
        for k in ("raw", "CalSet", "asdf_library", "history"):
            self.assertNotIn(k, keys)

    def test_values_are_as_stored(self):
        h = dict(self.info["header"])
        self.assertEqual(h["CoolerVoltage"], 2.99128455)   # not x10 000
        self.assertEqual(h["Run"], 7507)

    def test_the_event_table_comes_first_with_named_columns(self):
        raw = self.info["tables"][0]
        self.assertEqual(raw["name"], "raw")
        self.assertEqual(raw["columns"], COLUMNS)
        np.testing.assert_array_equal(raw["data"], EVENTS)

    def test_equal_length_1d_arrays_share_a_table(self):
        cal = self.info["tables"][1]
        self.assertEqual(sorted(cal["columns"]), ["CalReadback", "CalSet"])
        self.assertEqual(cal["data"].shape, (5, 2))

    def test_the_tree_keeps_the_asdf_entries(self):
        self.assertIn("asdf_library", self.info["tree"])

    def test_a_2d_array_without_names_gets_numbered_columns(self):
        import asdf
        p = os.path.join(self.dir, "plain.asdf")
        asdf.AsdfFile({"grid": np.zeros((2, 3))}).write_to(p)
        tab = read_asdf(p)["tables"][0]
        self.assertEqual(tab["columns"], ["grid[0]", "grid[1]", "grid[2]"])

    def test_a_split_shows_its_parent(self):
        from gui.analysis.vasdf import write_vasdf
        v = os.path.join(self.dir, "run_7507_low.vasdf")
        try:
            write_vasdf(v, parent_path=self.path, source_id="s", label="low",
                        lo=-900.0, hi=-850.0)
        except TypeError:
            self.skipTest("write_vasdf signature differs")
        info = read_asdf(v)
        self.assertEqual(os.path.normpath(info["parent_path"]),
                         os.path.normpath(self.path))
        self.assertIsNotNone(info["split"])
        self.assertEqual(info["tables"][0]["data"].shape, EVENTS.shape)


class StatsTests(unittest.TestCase):

    def test_min_max_mean_and_few_distinct_values(self):
        stats = column_stats(EVENTS)
        lo, hi, mean, distinct = stats[COLUMNS.index("channel")]
        self.assertEqual((lo, hi), (3.0, 4.0))
        self.assertAlmostEqual(mean, 10 / 3)
        self.assertEqual(distinct, [3.0, 4.0])

    def test_many_distinct_values_are_counted(self):
        col = np.arange(MAX_LISTED_VALUES + 5, dtype=float)[:, None]
        self.assertEqual(column_stats(col)[0][3], MAX_LISTED_VALUES + 5)


class ViewerTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.a = write_run(self.dir, 7507)
        self.b = write_run(self.dir, 7939)
        self.v = AsdfViewer()
        self.addCleanup(self.v.deleteLater)
        self.addCleanup(self.v.stop_loading)

    def add(self, paths):
        self.v.add_files(paths)
        self.assertTrue(self.v.wait_until_loaded(), "loading timed out")

    def test_adding_fills_every_view(self):
        self.add([self.a])
        self.assertEqual(self.v.model.rowCount(), 3)
        self.assertEqual(self.v.model.columnCount(), 6)
        self.assertEqual(self.v.model.headerData(
            3, Qt.Orientation.Horizontal), "channel")
        self.assertGreater(self.v.header_table.rowCount(), 5)
        self.assertGreater(self.v.tree.topLevelItemCount(), 5)
        self.assertEqual(self.v.stats_table.rowCount(), 6)

    def test_the_list_names_the_run(self):
        self.add([self.a, self.b])
        self.assertIn("7939", self.v.file_list.item(1).text())

    def test_a_file_opened_twice_is_listed_once(self):
        self.add([self.a])
        self.add([self.b, self.a])
        self.assertEqual(self.v.file_list.count(), 2)
        self.assertEqual(self.v.current_info()["path"],
                         os.path.normpath(self.a))

    def test_the_cooler_note_says_how_denis_reads_it(self):
        self.add([self.a])
        t = self.v.header_table
        row = next(r for r in range(t.rowCount())
                   if t.item(r, 0).text() == "CoolerVoltage")
        self.assertIn("29,912.85", t.item(row, 3).text())

    def test_switching_tables(self):
        self.add([self.a])
        self.v.table_combo.setCurrentIndex(1)
        self.assertEqual(self.v.model.rowCount(), 5)

    def test_an_unreadable_file_is_listed_with_the_reason(self):
        bad = os.path.join(self.dir, "broken.asdf")
        with open(bad, "w") as fh:
            fh.write("not asdf")
        self.add([bad])
        self.assertIn(os.path.normpath(bad), self.v.failed())
        self.assertIn("⚠", self.v.file_list.item(0).text())

    def test_remove_and_clear(self):
        self.add([self.a, self.b])
        self.v.remove_current()
        self.assertEqual(self.v.file_list.count(), 1)
        self.v.clear()
        self.assertEqual(self.v.file_list.count(), 0)
        self.assertIs(self.v.stack.currentWidget(), self.v.empty_label)

    def test_dropping_files(self):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(self.a),
                      QUrl.fromLocalFile(os.path.join(self.dir, "x.txt"))])
        ev = QDropEvent(QPointF(5, 5), Qt.DropAction.CopyAction, mime,
                        Qt.MouseButton.LeftButton,
                        Qt.KeyboardModifier.NoModifier)
        self.v.dropEvent(ev)
        self.v.wait_until_loaded()
        self.assertEqual(self.v.paths(), [os.path.normpath(self.a)])

    def test_export_csv(self):
        self.add([self.a])
        out = os.path.join(self.dir, "events.csv")
        self.v.export_current_table(out)
        lines = open(out, encoding="utf-8").read().splitlines()
        self.assertEqual(lines[0], ",".join(COLUMNS))
        self.assertEqual(len(lines), 4)


class RightClickTests(unittest.TestCase):
    """Both run lists open the viewer with the run already added."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = write_run(self.dir)

    def _patch_menu(self, on_exec):
        """Swap QMenu for a subclass whose exec() answers at once.

        Assigning QMenu.exec on the Qt class does not reach the C++
        call, and the real menu would sit waiting for a click. Both
        menus look QMenu up in a module namespace at call time, so the
        subclass goes there.
        """
        import PySide6.QtWidgets as qtw
        import gui.analysis.blocks as blocks

        class FakeMenu(QMenu):
            def exec(self, *a, **k):
                return on_exec(self)
        for mod in (qtw, blocks):
            self.addCleanup(setattr, mod, "QMenu", getattr(mod, "QMenu"))
            mod.QMenu = FakeMenu

    def _choose(self, text):
        def pick(menu):
            for act in menu.actions():
                if act.text().startswith(text):
                    act.trigger()
                    return act
            raise AssertionError(f"no '{text}' in the menu: "
                                 f"{[x.text() for x in menu.actions()]}")
        self._patch_menu(pick)

    def test_open_in_viewer_reuses_the_main_window_viewer(self):
        calls = []

        class Top:
            def window(self):
                return self

            def open_asdf_viewer(self, paths):
                calls.append(paths)
        open_in_viewer([self.path], anchor=Top())
        self.assertEqual(calls, [[self.path]])

    def test_the_main_window_keeps_one_viewer(self):
        from gui.main_window import MainWindow
        w = MainWindow.__new__(MainWindow)
        # only what open_asdf_viewer needs
        from PySide6.QtWidgets import QMainWindow
        QMainWindow.__init__(w)
        self.addCleanup(w.deleteLater)
        w._asdf_viewer = None
        v1 = MainWindow.open_asdf_viewer(w, [self.path])
        v2 = MainWindow.open_asdf_viewer(w, [self.path])
        self.addCleanup(v1.hide)
        self.assertIs(v1, v2)
        self.assertEqual(v1.file_list.count(), 1)

    def test_preanalysis_file_entry(self):
        from gui.preanalysis_tab import FileEntry
        from PySide6.QtGui import QContextMenuEvent
        from PySide6.QtCore import QPoint
        seen = []
        import gui.asdf_viewer as av
        orig = av.open_in_viewer
        av.open_in_viewer = lambda paths, anchor=None: seen.append(paths)
        self.addCleanup(setattr, av, "open_in_viewer", orig)
        e = FileEntry(self.path, "7507", 29912.8, "2026-05-08", 13522.68,
                      167.2)
        self.addCleanup(e.deleteLater)
        self._choose("View ASDF")
        e.contextMenuEvent(QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse, QPoint(1, 1), QPoint(1, 1)))
        self.assertEqual(seen, [[self.path]])

    def test_source_block_file_row(self):
        from gui.analysis.blocks import SourceBlock
        from PySide6.QtCore import QPoint
        seen = []
        import gui.asdf_viewer as av
        orig = av.open_in_viewer
        av.open_in_viewer = lambda paths, anchor=None: seen.append(paths)
        self.addCleanup(setattr, av, "open_in_viewer", orig)
        sb = SourceBlock()
        self.addCleanup(sb.deleteLater)
        entry = {"path": self.path, "run_number": "7507"}
        self._choose("View ASDF")
        sb._show_file_row_menu(entry, sb, QPoint(1, 1))
        self.assertEqual(seen, [[self.path]])

    def test_a_merged_row_cannot_be_viewed(self):
        from gui.analysis.blocks import SourceBlock
        from PySide6.QtCore import QPoint
        sb = SourceBlock()
        self.addCleanup(sb.deleteLater)
        states = []

        def look(menu):
            states.extend(act.isEnabled() for act in menu.actions()
                          if act.text().startswith("View ASDF"))
            return None
        self._patch_menu(look)
        sb._show_file_row_menu({"path": "", "is_merged": True}, sb,
                               QPoint(1, 1))
        self.assertEqual(states, [False])


class BackgroundLoadingTests(unittest.TestCase):
    """The window never waits for a file (Arda, 2026-09-27: "can we
    make the loading multicore process so it loads faster?")."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.paths = [write_run(self.dir, 7000 + i) for i in range(4)]
        self.v = AsdfViewer()
        self.addCleanup(self.v.deleteLater)
        self.addCleanup(self.v.stop_loading)

    def test_files_are_listed_at_once_and_filled_in_later(self):
        self.v.add_files(self.paths)
        self.assertEqual(self.v.file_list.count(), 4)
        self.assertTrue(self.v.is_loading())
        self.assertTrue(self.v.wait_until_loaded())
        self.assertFalse(self.v.is_loading())
        self.assertIn("Run 7003", self.v.file_list.item(3).text())
        self.assertEqual(self.v.model.rowCount(), 3)   # last one selected

    def test_a_few_files_do_not_start_processes(self):
        from gui.asdf_viewer import AsdfLoader
        jobs = [(p, p, None) for p in self.paths]
        self.assertFalse(AsdfLoader(jobs).use_processes)

    def test_many_files_use_worker_processes(self):
        import gui.asdf_viewer as av
        from gui.asdf_viewer import AsdfLoader
        old = av.PROCESS_MIN_FILES
        av.PROCESS_MIN_FILES = 2
        self.addCleanup(setattr, av, "PROCESS_MIN_FILES", old)
        jobs = [(os.path.normpath(p), os.path.normpath(p), None)
                for p in self.paths]
        self.assertTrue(AsdfLoader(jobs).use_processes)
        self.v.add_files(self.paths)
        self.assertTrue(self.v.wait_until_loaded(120000))
        self.assertEqual(len(self.v._files), 4)
        runs = sorted(dict(i["header"])["Run"]
                      for i in self.v._files.values())
        self.assertEqual(runs, [7000, 7001, 7002, 7003])

    def test_the_worker_needs_neither_qt_nor_the_gui(self):
        """So a worker process starts by importing numpy and asdf only."""
        import subprocess
        import sys
        code = ("import sys; import cls_estimations.asdf_contents; "
                "bad = [m for m in sys.modules if m.startswith(('PySide6', "
                "'gui'))]; print(bad); sys.exit(1 if bad else 0)")
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.dirname(
                               os.path.abspath(__file__))))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_clearing_while_loading_drops_the_late_results(self):
        self.v.add_files(self.paths)
        self.v.clear()
        self.v.wait_until_loaded()
        self.assertEqual(self.v.file_list.count(), 0)
        self.assertEqual(self.v._files, {})


class TooltipTests(unittest.TestCase):
    """What each column and header entry is, and in which unit."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.v = AsdfViewer()
        cls.v.add_files([write_run(cls.dir)])
        cls.v.wait_until_loaded()

    @classmethod
    def tearDownClass(cls):
        cls.v.deleteLater()

    def _header_tip(self, col):
        return self.v.model.headerData(
            COLUMNS.index(col), Qt.Orientation.Horizontal,
            Qt.ItemDataRole.ToolTipRole)

    def test_every_event_column_is_explained(self):
        for col in COLUMNS:
            with self.subTest(column=col):
                self.assertTrue(self._header_tip(col))

    def test_units_are_stated(self):
        self.assertIn("\u00b5s", self._header_tip("time"))
        self.assertIn("1970-01-01", self._header_tip("timestamp"))
        self.assertIn("10,000", self._header_tip("cooler"))
        self.assertIn("in V", self._header_tip("voltage"))

    def test_a_timestamp_cell_shows_its_date(self):
        tip = self.v.model.data(self.v.model.index(0, 0),
                                Qt.ItemDataRole.ToolTipRole)
        self.assertIn("UTC", tip)
        self.assertRegex(tip, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_a_cooler_cell_shows_volts(self):
        tip = self.v.model.data(self.v.model.index(0, 5),
                                Qt.ItemDataRole.ToolTipRole)
        self.assertEqual(tip, "29,912.00 V")

    def test_header_keys_are_explained(self):
        t = self.v.header_table
        tips = {t.item(r, 0).text(): t.item(r, 0).toolTip()
                for r in range(t.rowCount())}
        self.assertIn("10,000", tips["CoolerVoltage"])
        self.assertIn("cm", tips["LaserSetpoint"])
        self.assertIn("Column names", tips["raw_header"])

    def test_every_field_of_a_real_run_is_documented(self):
        """The keys a CLS run file carries (run_7507, 2026-05-08)."""
        from gui.asdf_viewer import FIELD_DOCS, COLUMN_DOCS
        real = ["BunchesPerChannel", "CalReadback", "CalSet",
                "CoolerVoltage", "Date", "DwellTime", "Experiment",
                "LaserSetpoint", "MassAMU", "MassSetPoint", "Run",
                "ScanningRanges", "SortingUtilities", "StepSize"]
        for key in real:
            with self.subTest(key=key):
                self.assertIn(key, FIELD_DOCS)
        self.assertEqual(sorted(COLUMN_DOCS), sorted(COLUMNS))


if __name__ == "__main__":
    unittest.main()
