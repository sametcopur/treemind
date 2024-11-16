import os
import sys
from sphinx.ext import autodoc
import shutil

project_path = os.path.abspath("../..")

build_dir = os.path.abspath("../../build")  # Adjust path if needed
if os.path.exists(build_dir):
    print(f"Cleaning build directory: {build_dir}")
    shutil.rmtree(build_dir)

sys.path.insert(0, project_path)

project = "treemind"
copyright = "2024, Samet Çopur"
author = "Samet Çopur"
version = "0.0.1"
release = "0.0.1"
autodoc_member_order = "bysource"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
]

autodoc_typehints = "description"

python_use_unqualified_type_names = True

autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_css_files = [
    'custom.css',
]


class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter
