import os
import sys
from sphinx.ext import autodoc

# Proje yolunu kontrol etmek için çıktılar ekleyelim
print(f"Current working directory (cwd): {os.getcwd()}")  # Çalışma dizini
print(f"sys.path: {sys.path}")  # Python'un yükleme yolları

# Projenizin kök dizinini ekleyin
project_path = os.path.abspath("../..")
sys.path.insert(0, project_path)
print(f"Project path added to sys.path: {project_path}")
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

autodoc_mock_imports = ['treemind']

class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter
