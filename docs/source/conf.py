import os
import sys
from sphinx.application import Sphinx
from sphinx.ext import autodoc

project_path = os.path.abspath("../..")
sys.path.insert(0, project_path)

project = "treExplainer"
copyright = "2024, Samet Çopur"
author = "Samet Çopur"
version = "1.0.1"
release = "1.0.1"
autodoc_member_order = "bysource"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    "myst_parser",
]

autodoc_typehints = "description"

autodoc_default_options = {
    "show-inheritance": False,
}

python_use_unqualified_type_names = True

autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter
