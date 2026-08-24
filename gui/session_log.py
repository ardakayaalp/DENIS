"""Session logging and observability for DENIS.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

A single, always-on observability sink so that a freeze or a silent crash
leaves a trace even when the app is launched by double-click (pythonw, no
console). One call to :func:`install` at the top of ``main()`` sets up, under
``<app>/logs``:

* a dated, pruned session log file (oldest files trimmed, so the folder does
  not grow without bound);
* a tee of ``stdout``/``stderr`` to that file (the verbatim banner and any
  ``print`` still land in the log) and to the console when one exists;
* a structured ``logging`` logger ("denis") writing the same file with
  timestamps and levels, for warnings/errors/timings;
* Python ``warnings`` redirected into the logger;
* ``sys.excepthook`` and ``threading.excepthook`` so uncaught exceptions on
  the GUI thread and worker threads are recorded.

:func:`install_qt_handler` additionally routes Qt's own messages (the usual
breadcrumbs before a paint crash or a cross-thread-timer freeze) into the log;
call it before ``QApplication`` is constructed. :func:`log_timing` wraps a
block to record its wall-clock duration.

Depends on: standard library only (PySide6.QtCore is imported lazily inside
:func:`install_qt_handler`).
"""

import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime

_LOGGER_NAME = "denis"
_installed = False
_log_path = None


def get_logger(name=None):
    """Return the shared "denis" logger, or a named child of it.

    Always safe to call: if :func:`install` has not run (e.g. under tests),
    the logger simply has no handlers and messages go nowhere.
    """
    base = logging.getLogger(_LOGGER_NAME)
    return base if name is None else base.getChild(name)


def _prune_old_logs(logs_dir, keep):
    """Delete all but the ``keep`` newest ``denis_*.log`` files."""
    try:
        files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir)
                 if f.startswith("denis_") and f.endswith(".log")]
    except OSError:
        return
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for p in files[keep:]:
        try:
            os.remove(p)
        except OSError:
            pass


class _TeeStream:
    """Write to the original stream (if any) and the session log file.

    ``original`` is None when launched via pythonw.exe (no console);
    ``log_fh`` is None only if the log file could not be opened. Failures on
    either sink are swallowed so logging can never crash the app.
    """

    def __init__(self, original, log_fh):
        self._orig = original
        self._log = log_fh

    def write(self, text):
        if not text:
            return
        for s in (self._orig, self._log):
            if s is not None:
                try:
                    s.write(text)
                except Exception:
                    pass

    def flush(self):
        for s in (self._orig, self._log):
            if s is not None:
                try:
                    s.flush()
                except Exception:
                    pass

    def fileno(self):
        if self._orig is None:
            raise OSError("no underlying stream")
        return self._orig.fileno()

    def isatty(self):
        return False


def install(app_root, *, verbose=False, keep=20):
    """Install the session log unconditionally. Idempotent.

    Returns the log file path, or None if the file could not be created (in
    which case stdout/stderr are still tee'd to the console only). ``verbose``
    selects DEBUG level (the repurposed "save session log" Setting); ``keep``
    is how many past session logs to retain.
    """
    global _installed, _log_path
    if _installed:
        return _log_path

    logs_dir = os.path.join(app_root, "logs")
    log_fh = None
    try:
        os.makedirs(logs_dir, exist_ok=True)
        _prune_old_logs(logs_dir, keep)
        _log_path = os.path.join(
            logs_dir, f"denis_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log")
        log_fh = open(_log_path, "w", encoding="utf-8", buffering=1)
    except Exception:
        log_fh = None
        _log_path = None

    # 1. Tee stdout/stderr -> console (if present) + log file.
    sys.stdout = _TeeStream(sys.__stdout__, log_fh)
    sys.stderr = _TeeStream(sys.__stderr__, log_fh)

    # 2. Structured logger writing the same file with timestamps/levels.
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if log_fh is not None and not logger.handlers:
        handler = logging.StreamHandler(log_fh)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        logger.addHandler(handler)

    # 3. Uncaught exceptions -> log, then chain the previous behaviour.
    _orig_excepthook = sys.excepthook

    def _crash_hook(exc_type, exc_value, exc_tb):
        try:
            logger.critical("Uncaught exception",
                            exc_info=(exc_type, exc_value, exc_tb))
        except Exception:
            pass
        _orig_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_hook

    def _thread_hook(args):
        try:
            logger.error(
                "Uncaught exception in thread %r",
                getattr(args.thread, "name", "?"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
        except Exception:
            pass

    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass

    # 4. Route Python warnings through the logger.
    try:
        logging.captureWarnings(True)
        pyw = logging.getLogger("py.warnings")
        if logger.handlers and not pyw.handlers:
            pyw.addHandler(logger.handlers[0])
    except Exception:
        pass

    _installed = True
    return _log_path


def install_qt_handler():
    """Route Qt's own messages into the session log. Call before QApplication.

    Qt writes from C++ to native stderr, which bypasses the Python tee, so
    without this the warnings that precede freezes/paint crashes are lost.
    """
    try:
        from PySide6.QtCore import qInstallMessageHandler, QtMsgType
    except Exception:
        return
    logger = get_logger("qt")
    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode, context, message):
        try:
            loc = ""
            f = getattr(context, "file", None)
            if f:
                loc = f" ({f}:{getattr(context, 'line', 0)})"
            logger.log(level_map.get(mode, logging.INFO), "%s%s", message, loc)
        except Exception:
            pass

    try:
        qInstallMessageHandler(handler)
    except Exception:
        pass


@contextmanager
def log_timing(label):
    """Log the wall-clock duration of a block at INFO (WARNING if it raised).

    Usage::

        with log_timing("fit run %s" % name):
            ...
    """
    logger = get_logger("timing")
    t0 = time.perf_counter()
    ok = True
    try:
        yield
    except Exception:
        ok = False
        raise
    finally:
        dt = time.perf_counter() - t0
        logger.log(logging.INFO if ok else logging.WARNING,
                   "%s %s in %.2fs", label, "OK" if ok else "FAILED", dt)
