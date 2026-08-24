"""Authored manual page registry with a planned-page fallback.

Date:    2026-06-02
Version: 1.0.0
Author:  Arda Kayaalp <arda.kayaalp@kuleuven.be>

Aggregates the authored pages from the per-section content modules (each exports
a ``PAGES`` dict mapping a node id to a zero-arg callable returning an HTML body)
and exposes ``get_page_html``. Any TOC id without an authored page renders a
"planned content" page built from ``structure.PLAN``, so the manual is fully
navigable and every node previews its outlined content until it is written.

Depends on: gui.manual.render, gui.manual.structure, and the content sub-modules
(generated, getting_started, physics, estimate).
"""

import html as _html

from gui.manual import render as R
from gui.manual.structure import PLAN, TOC

# Merge order: `generated` (data-driven pages) is applied first so the
# hand-authored modules listed after it can override any of its page ids.
from gui.manual.content import generated, getting_started, physics, estimate

_AUTHORED = {}
for _mod in (generated, getting_started, physics, estimate):
    _AUTHORED.update(_mod.PAGES)


# Map each node id to its child ids (for section overview pages).
_CHILDREN = {}
for _sec in TOC:
    _CHILDREN[_sec["id"]] = [c["id"] for c in _sec.get("children", [])]


def is_authored(page_id):
    return page_id in _AUTHORED


def _planned_page(page_id):
    meta = PLAN.get(page_id)
    if meta is None:
        return (f"<h1>Page not found</h1><p>No content or plan is registered "
                f"for <code>{_html.escape(page_id)}</code>.</p>")

    title = meta.get("title", page_id)
    parts = [f"<h1>{_html.escape(title)}</h1>"]
    parts.append(R.note(
        "This page is <b>planned but not yet written</b>. The outline below was "
        "generated from a code-grounded survey; we fill it in section by "
        "section, verifying each claim against the source."))

    if meta.get("summary"):
        parts.append(f"<p>{_html.escape(meta['summary'])}</p>")

    # Section overview pages: list their subsections as links.
    kids = _CHILDREN.get(page_id)
    if kids:
        parts.append("<h2>In this section</h2><ul>")
        parts.extend(f"<li>{R.link(k)}</li>" for k in kids)
        parts.append("</ul>")

    if meta.get("keyTopics"):
        parts.append("<h2>This page will cover</h2><ul>")
        parts.extend(f"<li>{_html.escape(t)}</li>" for t in meta["keyTopics"])
        parts.append("</ul>")

    if meta.get("equations"):
        names = ", ".join(f"<code>{_html.escape(e)}</code>" for e in meta["equations"])
        parts.append(f"<h2>Equations</h2><p>{names}</p>")

    if meta.get("codeSnippets"):
        parts.append("<h2>Code snippets planned</h2><ul>")
        parts.extend(f"<li>{_html.escape(c)}</li>" for c in meta["codeSnippets"])
        parts.append("</ul>")

    if meta.get("figures"):
        parts.append("<h2>Figures planned</h2>")
        for i, fig in enumerate(meta["figures"]):
            parts.append(R.screenshot_slot(f"{page_id}-{i}", fig))

    if meta.get("crossLinks"):
        parts.append(R.related(meta["crossLinks"]))

    return "\n".join(parts)


def get_page_html(page_id):
    """Return the HTML body for a node id (authored if available, else planned)."""
    if page_id in _AUTHORED:
        try:
            return _AUTHORED[page_id]()
        except Exception as exc:  # never let a content bug crash the viewer
            return (f"<h1>{_html.escape(R.title_of(page_id))}</h1>"
                    f"<p style='color:#e06c5a;'>Error rendering this page: "
                    f"{_html.escape(str(exc))}</p>")
    return _planned_page(page_id)
