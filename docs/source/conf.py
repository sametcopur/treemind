import os
import sys
from sphinx.ext import autodoc
from pathlib import Path

project_path = os.path.abspath("../..")

sys.path.insert(0, project_path)

def process_explainer_files(directory: str):
    """
    Process the explainer files in the specified directory.
    1. Delete `explainer.pxd` and `explainer.pyx` if they exist.
    2. Rename `explainer.pyi` to `explainer.py`.

    :param directory: The directory containing the explainer files.
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Directory {directory} does not exist or is not a directory.")
        return

    # Define file paths
    pxd_file = dir_path / "explainer.pxd"
    pyx_file = dir_path / "explainer.pyx"
    pyi_file = dir_path / "explainer.pyi"
    py_file = dir_path / "explainer.py"

    # Remove .pxd file if it exists
    if pxd_file.exists():
        os.remove(pxd_file)
        print(f"Deleted {pxd_file}")

    # Remove .pyx file if it exists
    if pyx_file.exists():
        os.remove(pyx_file)
        print(f"Deleted {pyx_file}")

    # Rename .pyi file to .py
    if pyi_file.exists():
        if py_file.exists():
            os.remove(py_file)  # Remove existing .py file if it exists
            print(f"Removed existing {py_file}")
        os.rename(pyi_file, py_file)
        print(f"Renamed {pyi_file} to {py_file}")
    else:
        print(f"{pyi_file} does not exist, no .py file created.")


# COMMENT LOCALLY
DIRECTORY = "../../treemind/algorithm"
process_explainer_files(DIRECTORY)
print("File processing completed.")

import treemind

project = "treemind"
copyright = "2024, Samet Çopur"
author = "Samet Çopur"
version = "0.1.1"
release = "0.1.1"

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


class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter
