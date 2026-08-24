"""HTML page-building helpers for authored manual pages.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Provides the primitives authored pages call to assemble an HTML body (headings,
equations, figures, diagrams, code blocks, callouts, links, tables, and related-
topic boxes), keeping styling and cross-linking consistent. The HTML targets
Qt's rich-text subset (QTextBrowser), so layout relies on tables and inline
styles rather than modern CSS. Internal links use the ``page:<id>`` scheme
intercepted by the viewer's ``anchorClicked`` handler; external links use
``http(s)://`` and open in the system browser.

Depends on: gui.manual.equations, gui.manual.diagrams, gui.manual.structure.
"""

import html as _html
import os

from gui.manual import equations as _equations
from gui.manual import diagrams as _diagrams
from gui.manual.structure import PLAN

# Committed screenshots/figures live here (separate from the generated cache).
FIGS_DIR = os.path.join(os.path.dirname(__file__), "figs")

# id -> display title, for link text and "Related topics" blocks.
TITLES = {pid: meta.get("title", pid) for pid, meta in PLAN.items()}


def title_of(page_id):
    return TITLES.get(page_id, page_id)


def img_url(path):
    """Filesystem path -> file:/// URL with forward slashes (Qt-friendly)."""
    return "file:///" + os.path.abspath(path).replace("\\", "/")


# ── Inline / block primitives ────────────────────────────────────

def equation(name, caption=None):
    """Centered rendered-equation block; optional small caption beneath."""
    path = _equations.render(name)
    cap = (f'<div align="center"><span style="color:#9aa0a6; font-size:9pt;">'
           f'{_html.escape(caption)}</span></div>') if caption else ""
    return (f'<div align="center" style="margin:10px 0;">'
            f'<img src="{img_url(path)}"></div>{cap}')


def figure(path, caption=None):
    """Image figure with optional caption."""
    cap = (f'<div align="center"><span style="color:#9aa0a6; font-size:9pt;">'
           f'{caption}</span></div>') if caption else ""
    return (f'<div align="center" style="margin:12px 0;">'
            f'<img src="{img_url(path)}"></div>{cap}')


def diagram(name, caption=None):
    """Render a registered schematic diagram and embed it as a figure."""
    return figure(_diagrams.render(name), caption=caption)


def code(src, caption=None):
    """Monospace code/snippet block (content is HTML-escaped)."""
    body = _html.escape(src.rstrip("\n"))
    cap = (f'<div style="color:#9aa0a6; font-size:9pt; margin-bottom:2px;">'
           f'{_html.escape(caption)}</div>') if caption else ""
    return (
        f'{cap}<table width="100%" cellpadding="8" cellspacing="0" '
        f'bgcolor="#1e1f22"><tr><td>'
        f'<pre style="margin:0; color:#dcdcdc; font-family:Consolas,'
        f'\'DejaVu Sans Mono\',monospace; font-size:9.5pt; '
        f'white-space:pre-wrap;">{body}</pre>'
        f'</td></tr></table>'
    )


def link(page_id, text=None):
    """Internal hyperlink to another manual page."""
    return f'<a href="page:{page_id}">{_html.escape(text or title_of(page_id))}</a>'


def extlink(url, text):
    return f'<a href="{url}">{_html.escape(text)}</a>'


def _callout(bg, border, label, body_html):
    return (
        f'<table width="100%" cellpadding="8" cellspacing="0" bgcolor="{bg}" '
        f'style="margin:10px 0;"><tr>'
        f'<td style="border-left:3px solid {border};">'
        f'<b style="color:{border};">{label}</b><br>{body_html}</td></tr></table>'
    )


def note(body_html):
    return _callout("#26303a", "#4ea1d5", "Note", body_html)


def tip(body_html):
    return _callout("#26352a", "#5cb85c", "Tip", body_html)


def warning(body_html):
    return _callout("#3a3326", "#e0b85a", "Caution", body_html)


def screenshot(name, caption=None):
    """Embed a committed screenshot ``figs/<name>`` if it exists, otherwise
    show a labelled placeholder slot. Lets pages reference screenshots that are
    added later without breaking."""
    path = os.path.join(FIGS_DIR, name)
    if os.path.exists(path):
        return figure(path, caption)
    return screenshot_slot(name, caption or name)


def screenshot_slot(slot_id, caption):
    """Placeholder box for a screenshot to be captured later."""
    return (
        f'<table width="100%" cellpadding="14" cellspacing="0" bgcolor="#2b2b2b" '
        f'style="margin:10px 0; border:1px dashed #555;"><tr>'
        f'<td align="center">'
        f'<span style="color:#888; font-size:10pt;">[ screenshot: '
        f'{_html.escape(caption)} ]</span><br>'
        f'<span style="color:#666; font-size:8pt;">slot &lt;{_html.escape(slot_id)}&gt; '
        f'&mdash; to be captured</span></td></tr></table>'
    )


def related(page_ids):
    """A 'Related topics' block of internal links."""
    if not page_ids:
        return ""
    items = "".join(f"<li>{link(p)}</li>" for p in page_ids)
    return (f'<table width="100%" cellpadding="8" cellspacing="0" bgcolor="#26303a" '
            f'style="margin:14px 0;"><tr><td style="border-left:3px solid #4ea1d5;">'
            f'<b style="color:#4ea1d5;">Related topics</b>'
            f'<ul style="margin:4px 0 0 0;">{items}</ul></td></tr></table>')


def code_ref(text):
    """Inline reference to a source location, e.g. cls_estimations/doppler.py:47."""
    return (f'<span style="color:#9aa0a6; font-family:Consolas,monospace; '
            f'font-size:9pt;">{_html.escape(text)}</span>')
