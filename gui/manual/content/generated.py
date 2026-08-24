"""Auto-generated: workflow-authored pages loaded from generated_pages.json."""
import json
import os

from gui.manual.builder import make_pages

_PATH = os.path.join(os.path.dirname(__file__), 'generated_pages.json')
with open(_PATH, encoding='utf-8') as _f:
    PAGES_DATA = json.load(_f)

PAGES = make_pages(PAGES_DATA)
