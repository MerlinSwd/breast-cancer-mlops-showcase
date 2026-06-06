from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

project = "Breast Cancer MLOps Showcase"
author = "Merlin"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = project
html_static_path: list[str] = []

mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'default'
});
"""
