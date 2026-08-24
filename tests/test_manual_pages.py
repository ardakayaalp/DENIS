"""In-UI manual: page generation, rendering pipelines, and viewer navigation.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Asserts that every contents-tree node produces non-empty HTML, that node
ids are unique, that the equation and diagram render pipelines write
their image files, and that the viewer constructs, navigates with
consistent back/forward history, and scales wide images to fit the pane
without clipping, upscaling, or horizontal overflow. Un-authored nodes
fall back to an auto-generated "planned content" page built from the
survey outline, so the suite guards both authored pages and the
planned-page generator as the structure evolves.

Run from the project root:

    .venv/Scripts/python.exe -m unittest tests.test_manual_pages -v

Depends on: gui.manual.* (structure, content, equations, diagrams,
viewer.ManualWindow); PySide6 for QApplication/QCoreApplication.
"""

import os
import unittest

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication


def _image_display_widths(browser):
    """Display widths of every embedded image in a content browser."""
    doc = browser.document()
    widths = []
    block = doc.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            frag = it.fragment()
            if frag.isValid() and frag.charFormat().isImageFormat():
                fmt = frag.charFormat().toImageFormat()
                widths.append((round(fmt.width()),
                               browser._native_size(fmt.name())[0]))
            it += 1
        block = block.next()
    return widths


_APP = None


def _ensure_app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _all_ids():
    from gui.manual import structure
    ids = []
    for sec in structure.TOC:
        ids.append(sec["id"])
        ids.extend(c["id"] for c in sec.get("children", []))
    return ids


class ManualContentTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_every_node_builds_nonempty_html(self):
        from gui.manual import content as content_mod
        for pid in _all_ids():
            html = content_mod.get_page_html(pid)
            self.assertTrue(html and len(html) > 10,
                            f"empty page for {pid!r}")
            # The planned-page fallback flags missing PLAN entries; none expected.
            self.assertNotIn("Page not found", html, f"unknown id {pid!r}")

    def test_ids_are_unique(self):
        ids = _all_ids()
        self.assertEqual(len(ids), len(set(ids)), "duplicate node ids in TOC")

    def test_sample_pages_authored(self):
        from gui.manual import content as content_mod
        for pid in ("introduction", "physics-doppler", "est-beam"):
            self.assertTrue(content_mod.is_authored(pid),
                            f"expected {pid!r} to be authored")

    def test_curated_equations_render(self):
        from gui.manual import equations
        for name in equations.CURATED:
            path = equations.render(name)
            self.assertTrue(os.path.exists(path),
                            f"equation {name!r} did not render")

    def test_diagrams_render(self):
        from gui.manual import diagrams
        for name in diagrams.DIAGRAMS:
            path = diagrams.render(name)
            self.assertTrue(os.path.exists(path),
                            f"diagram {name!r} did not render")


class ManualViewerTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_window_constructs_and_navigates(self):
        from gui.manual.viewer import ManualWindow
        w = ManualWindow()
        # Tree exposes every node.
        self.assertEqual(len(w._id_to_item), len(_all_ids()))
        # Navigation builds history; back/forward stay consistent.
        for pid in ("physics-doppler", "est-beam", "introduction", "an-fitter"):
            w.navigate(pid)
        self.assertEqual(len(w._history), 5)  # home page + 4 navigations
        w.go_back()
        self.assertTrue(w.act_fwd.isEnabled())
        w.go_forward()
        self.assertFalse(w.act_fwd.isEnabled())

    def test_wide_images_scale_to_pane_width(self):
        """A figure wider than the pane must be scaled down to fit (never
        clipped), and images are never upscaled past native resolution."""
        from gui.manual.viewer import ManualWindow
        w = ManualWindow()
        w.resize(560, 700)
        w.show()
        QCoreApplication.processEvents()
        w.navigate("introduction")  # has the wide four-tab-map diagram
        QCoreApplication.processEvents()
        avail = w.browser.viewport().width() - w.browser._WIDTH_MARGIN
        widths = _image_display_widths(w.browser)
        self.assertTrue(widths, "introduction page should embed a diagram")
        for display_w, native_w in widths:
            self.assertLessEqual(display_w, avail + 1,
                                 "image overflows the pane (clipping)")
            self.assertLessEqual(display_w, native_w + 1,
                                 "image upscaled past native resolution")
        # At least one image was actually wider than the pane and got shrunk.
        self.assertTrue(any(native_w > avail for _, native_w in widths))

    def test_authored_pages_do_not_overflow_horizontally(self):
        """No authored page should be wider than its pane (which would force a
        horizontal scrollbar) — guards code blocks, wide tables, and images."""
        from gui.manual.viewer import ManualWindow
        from gui.manual import content as content_mod
        w = ManualWindow()
        w.resize(1040, 760)
        w.show()
        QCoreApplication.processEvents()
        offenders = []
        for pid in [p for p in _all_ids() if content_mod.is_authored(p)]:
            w.navigate(pid)
            QCoreApplication.processEvents()
            doc = w.browser.document()
            overflow = doc.idealWidth() - w.browser.viewport().width()
            if overflow > 2:
                offenders.append((pid, round(overflow)))
        self.assertFalse(offenders, f"pages overflow horizontally: {offenders}")


if __name__ == "__main__":
    unittest.main()
