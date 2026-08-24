"""Render structured page "blocks" into manual HTML.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Lets authored pages be expressed as data (a list of block dicts) rather than
hand-written HTML f-strings, which keeps content consistent, avoids HTML/escaping
and f-string-backslash pitfalls, and lets content be produced/validated as JSON.
Resolves each block dict to an HTML fragment via the gui.manual.render helpers.

Depends on: gui.manual.render.

Block types (each a dict with a ``type`` key):
    {"type": "h2"|"h3", "text": "..."}
    {"type": "p", "text": "...inline html + [[id]] / [[id|label]] links..."}
    {"type": "ul"|"ol", "items": ["...", ...]}
    {"type": "equation", "name": "<catalog-name>", "caption": "..."}
    {"type": "code", "code": "...", "caption": "...", "lang": "python"}
    {"type": "diagram", "name": "<diagram-name>", "caption": "..."}
    {"type": "screenshot", "file": "name.png", "caption": "..."}
    {"type": "note"|"tip"|"warning", "text": "..."}
    {"type": "table", "headers": ["..."], "rows": [["...", ...], ...]}
    {"type": "related", "ids": ["id", ...]}

Inline cross-links use ``[[id]]`` or ``[[id|label]]`` tokens, resolved to the
viewer's ``page:`` link scheme. Other inline HTML (<b>, <i>, <code>, <sub>,
<sup>, entities) is passed through unchanged.
"""

import html as _html
import re

from gui.manual import render as R

_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")


def _inline(text):
    """Resolve [[id|label]] link tokens; leave other inline HTML untouched."""
    if text is None:
        return ""

    def repl(m):
        pid = m.group(1).strip()
        label = (m.group(2) or "").strip() or None
        return R.link(pid, label)

    return _LINK_RE.sub(repl, str(text))


def _table(headers, rows):
    head = "".join(f"<td><b>{_inline(h)}</b></td>" for h in (headers or []))
    head_row = (f'<tr bgcolor="#2b3036">{head}</tr>' if headers else "")
    body = "".join(
        "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
        for row in (rows or [])
    )
    return (f'<table cellpadding="5" cellspacing="0" '
            f'style="margin:8px 0;">{head_row}{body}</table>')


def _render_block(b):
    t = b.get("type")
    if t in ("h2", "h3"):
        return f"<{t}>{_inline(b.get('text'))}</{t}>"
    if t == "p":
        return f"<p>{_inline(b.get('text'))}</p>"
    if t in ("ul", "ol"):
        items = "".join(f"<li>{_inline(x)}</li>" for x in b.get("items", []))
        return f"<{t}>{items}</{t}>"
    if t == "equation":
        return R.equation(b["name"], b.get("caption"))
    if t == "code":
        return R.code(b.get("code", ""), b.get("caption"))
    if t == "diagram":
        return R.diagram(b["name"], b.get("caption"))
    if t == "screenshot":
        return R.screenshot(b["file"], b.get("caption"))
    if t == "note":
        return R.note(_inline(b.get("text")))
    if t == "tip":
        return R.tip(_inline(b.get("text")))
    if t == "warning":
        return R.warning(_inline(b.get("text")))
    if t == "table":
        return _table(b.get("headers"), b.get("rows"))
    if t == "related":
        return R.related(b.get("ids", []))
    # Unknown block type: render nothing rather than crash.
    return ""


def render_blocks(page_id, blocks):
    """Render a full page: an <h1> title (from the TOC) followed by blocks."""
    parts = [f"<h1>{_html.escape(R.title_of(page_id))}</h1>"]
    for b in blocks or []:
        parts.append(_render_block(b))
    return "\n".join(p for p in parts if p)


def make_pages(data):
    """Build a {id: callable->html} registry from {id: [blocks]} data."""
    return {pid: (lambda pid=pid: render_blocks(pid, data[pid])) for pid in data}
