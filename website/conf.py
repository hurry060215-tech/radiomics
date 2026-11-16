import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "ov-radiomics"
author = "radiomics contributors"
extensions = [
    "myst_nb",
    "sphinxcontrib.mermaid",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]
autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = False
autodoc_typehints = "description"
html_theme = "pydata_sphinx_theme"
html_title = "Radiomics · 教程与文档"
documentation_version = os.environ.get("RADIOMICS_DOC_VERSION", "latest")
html_baseurl = "https://hurry060215-tech.github.io/radiomics/" + documentation_version + "/"
language = "zh_CN"
nb_execution_mode = "off"
exclude_patterns = ["_build"]
html_theme_options = {
    "github_url": "https://github.com/hurry060215-tech/radiomics",
    "navigation_depth": 3,
    "show_toc_level": 2,
    "header_links_before_dropdown": 5,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_end": ["build-revision"],
}
html_context = {
    "default_mode": "light",
    "build_revision": os.environ.get("RADIOMICS_BUILD_REVISION", "unversioned"),
}
templates_path = ["_templates"]
myst_heading_anchors = 3
html_static_path = ["_static"]
html_css_files = ["custom.css"]
