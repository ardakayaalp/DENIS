"""Tests for the WP6 session-logging module (2026-06-02 code review).

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Covers gui.session_log: old-log pruning, the get_logger naming and the
log_timing context manager, and install() creating a session file that
captures a logged WARNING. The install() test snapshots and restores all
global state it touches (sys.stdout/stderr, the exception hooks, the "denis"
and "py.warnings" loggers, and the module's install flag) so it cannot leak
a tee writing to a deleted temp file into the rest of the suite.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_session_log -v

Depends on: gui.session_log.
"""

import io
import logging
import os
import sys
import tempfile
import threading
import unittest

from gui import session_log


class PruneTests(unittest.TestCase):
    def test_keeps_newest_n_and_leaves_non_logs(self):
        d = tempfile.mkdtemp()
        paths = []
        for i in range(5):
            p = os.path.join(d, f"denis_2026-06-{i + 1:02d}_000000.log")
            with open(p, "w") as f:
                f.write("x")
            os.utime(p, (1000 + i, 1000 + i))  # strictly ascending mtime
            paths.append(p)
        other = os.path.join(d, "notes.txt")
        with open(other, "w") as f:
            f.write("x")

        session_log._prune_old_logs(d, keep=2)

        remaining = {f for f in os.listdir(d) if f.endswith(".log")}
        self.assertEqual(len(remaining), 2)
        self.assertIn(os.path.basename(paths[4]), remaining)  # newest
        self.assertIn(os.path.basename(paths[3]), remaining)
        self.assertTrue(os.path.isfile(other))  # non-log left alone


class LoggerHelpersTests(unittest.TestCase):
    """get_logger naming + log_timing, without calling install()."""

    def setUp(self):
        self.denis = logging.getLogger("denis")
        self._lvl = self.denis.level
        self.buf = io.StringIO()
        self.h = logging.StreamHandler(self.buf)
        self.denis.addHandler(self.h)
        self.denis.setLevel(logging.INFO)

    def tearDown(self):
        self.denis.removeHandler(self.h)
        self.denis.setLevel(self._lvl)

    def test_get_logger_names(self):
        self.assertEqual(session_log.get_logger().name, "denis")
        self.assertEqual(session_log.get_logger("foo").name, "denis.foo")

    def test_log_timing_ok(self):
        with session_log.log_timing("unit-op"):
            pass
        self.h.flush()
        out = self.buf.getvalue()
        self.assertIn("unit-op", out)
        self.assertIn("OK", out)

    def test_log_timing_failed_marks_and_reraises(self):
        with self.assertRaises(ValueError):
            with session_log.log_timing("boom-op"):
                raise ValueError("boom")
        self.h.flush()
        self.assertIn("FAILED", self.buf.getvalue())


class InstallTests(unittest.TestCase):
    """install() side effects, fully snapshotted/restored."""

    def setUp(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._excepthook = sys.excepthook
        self._thook = threading.excepthook
        self._installed = session_log._installed
        self._logpath = session_log._log_path
        self.denis = logging.getLogger("denis")
        self.pyw = logging.getLogger("py.warnings")
        self._denis_handlers = list(self.denis.handlers)
        self._pyw_handlers = list(self.pyw.handlers)
        self._denis_lvl = self.denis.level
        self._denis_prop = self.denis.propagate

    def tearDown(self):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        sys.excepthook = self._excepthook
        threading.excepthook = self._thook
        for h in list(self.denis.handlers):
            if h not in self._denis_handlers:
                try:
                    h.close()
                except Exception:
                    pass
        self.denis.handlers[:] = self._denis_handlers
        self.pyw.handlers[:] = self._pyw_handlers
        self.denis.setLevel(self._denis_lvl)
        self.denis.propagate = self._denis_prop
        logging.captureWarnings(False)
        session_log._installed = self._installed
        session_log._log_path = self._logpath

    def test_install_creates_file_and_captures_warning(self):
        d = tempfile.mkdtemp()
        session_log._installed = False
        session_log._log_path = None
        path = session_log.install(d, verbose=False)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))
        session_log.get_logger("itest").warning("captured-warning-xyz")
        for h in self.denis.handlers:
            h.flush()
        txt = open(path, encoding="utf-8").read()
        self.assertIn("captured-warning-xyz", txt)
        self.assertIn("WARNING", txt)

    def test_install_is_idempotent(self):
        d = tempfile.mkdtemp()
        session_log._installed = False
        session_log._log_path = None
        p1 = session_log.install(d)
        # Second call (already installed) must no-op and return the same path.
        p2 = session_log.install(tempfile.mkdtemp())
        self.assertEqual(p1, p2)


if __name__ == "__main__":
    unittest.main()
