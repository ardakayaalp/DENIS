"""Keeping a second copy of a native library out of the process.

A DLL's base name is process-global on Windows: whoever loads
``arrow.dll`` first wins, and every later user of that name gets it,
whatever it was built against. Anaconda ships its own Arrow in
``anaconda3\\Library\\bin``, that directory is on the machine PATH, and
DENIS's pyarrow comes from its own virtual environment. With
Anaconda's copy in first, ``import pyarrow`` dies at
``pyarrow/__init__.py`` line 71 -- which is exactly where DENIS died,
every time, when it was started from the desktop shortcut and a
session was loaded (2026-09-25). Started from a shell it did not,
because that PATH is different, which is why it took so long to find.

Two defences, both tested here: the directories are taken out of this
process's PATH at startup, and pyarrow is loaded early so that even if
a copy is reachable some other way, ours owns the name first.

Run from the project root:
    .venv/Scripts/python.exe -m pytest tests/test_shadowing_dlls.py -q
"""
import os
import tempfile
import unittest

from gui.main_window import SHADOWING_DLLS, strip_shadowing_dll_dirs


class StripShadowingDirsTests(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def _dir(self, name, *files):
        d = os.path.join(self.root, name)
        os.makedirs(d, exist_ok=True)
        for f in files:
            with open(os.path.join(d, f), "wb") as fh:
                fh.write(b"not really a dll")
        return d

    def test_a_directory_with_a_shadowing_dll_is_dropped(self):
        bad = self._dir("conda_library_bin", "arrow.dll")
        good = self._dir("windows_system32", "kernel32.dll")
        clean, dropped = strip_shadowing_dll_dirs(
            os.pathsep.join([good, bad]))
        self.assertEqual(dropped, [bad])
        self.assertEqual(clean, good)

    def test_everything_else_is_left_alone(self):
        dirs = [self._dir(f"d{i}", "something.dll") for i in range(4)]
        clean, dropped = strip_shadowing_dll_dirs(os.pathsep.join(dirs))
        self.assertEqual(dropped, [])
        self.assertEqual(clean.split(os.pathsep), dirs)

    def test_our_own_installation_is_never_dropped(self):
        """The venv's pyarrow ships the arrow.dll we WANT."""
        ours = self._dir("app/.venv/Lib/site-packages/pyarrow",
                         "arrow.dll")
        clean, dropped = strip_shadowing_dll_dirs(
            ours, keep_under=os.path.join(self.root, "app"))
        self.assertEqual(dropped, [])
        self.assertEqual(clean, ours)

    def test_the_order_of_what_is_kept_survives(self):
        a = self._dir("a")
        bad = self._dir("bad", "arrow.dll")
        b = self._dir("b")
        clean, _dropped = strip_shadowing_dll_dirs(
            os.pathsep.join([a, bad, b]))
        self.assertEqual(clean.split(os.pathsep), [a, b])

    def test_blank_and_missing_entries_are_dropped_quietly(self):
        a = self._dir("a")
        gone = os.path.join(self.root, "does_not_exist")
        clean, dropped = strip_shadowing_dll_dirs(
            os.pathsep.join(["", a, gone, "  "]))
        self.assertEqual(dropped, [])
        self.assertEqual(clean.split(os.pathsep), [a, gone])

    def test_an_empty_path_is_not_an_error(self):
        self.assertEqual(strip_shadowing_dll_dirs(""), ("", []))
        self.assertEqual(strip_shadowing_dll_dirs(None), ("", []))

    def test_arrow_is_on_the_list(self):
        self.assertIn("arrow.dll", SHADOWING_DLLS)

    def test_main_sanitises_before_importing_anything(self):
        """It has to run before the first native import, or another
        library has already resolved the name."""
        import inspect
        from gui import main_window
        src = inspect.getsource(main_window.main)
        cut = src.index("strip_shadowing_dll_dirs")
        self.assertNotIn("import pandas", src[:cut])
        self.assertNotIn("QApplication(", src[:cut])


class PreloadOrderTests(unittest.TestCase):
    """The second defence: ours owns the name first."""

    def test_the_preload_takes_pyarrow_explicitly(self):
        import inspect
        from gui.main_window import MainWindow
        src = inspect.getsource(MainWindow.preload_data_stack)
        self.assertIn("import pyarrow", src)
        self.assertLess(src.index("import pyarrow"),
                        src.index("import pandas"))


if __name__ == "__main__":
    unittest.main()
