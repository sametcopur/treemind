from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np
import sys

if sys.platform == "win32":
    extra_compile_args = [
        "/O2",
        "/fp:fast",
        "/Ot",
        "/Ox",
        "/Oi",
        "/GT",
    ]
    extra_link_args = ["/OPT:REF", "/OPT:ICF"]

elif sys.platform == "linux":
    extra_compile_args = [
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-ftree-vectorize",
        "-fstrict-aliasing",
        "-fstack-protector-strong",
    ]
    extra_link_args = []

elif sys.platform == "darwin":
    extra_compile_args = [
        "-Wall",
        "-O3",
        "-fno-math-errno",       
        "-fno-signed-zeros",    
        "-fno-trapping-math", 
        "-funroll-loops",
        "-ftree-vectorize",
        "-fstrict-aliasing",
        "-fstack-protector-strong",
        "-Wno-unreachable-code-fallthrough",
    ]
    extra_link_args = []

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


extensions = [
    Extension(
        name="treemind.algorithm.explainer",
        sources=["treemind/algorithm/explainer.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="treemind.algorithm.rule",
        sources=["treemind/algorithm/rule.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="treemind.algorithm.utils",
        sources=["treemind/algorithm/utils.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="treemind.algorithm.perp",
        sources=["treemind/algorithm/perp.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="treemind.algorithm.lgb",
        sources=["treemind/algorithm/lgb.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="treemind.algorithm.sk",
        sources=["treemind/algorithm/sk.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="treemind.algorithm.xgb",
        sources=["treemind/algorithm/xgb.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="treemind.algorithm.cb",
        sources=["treemind/algorithm/cb.pyx"],
        include_dirs=[np.get_include(), "treemind/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
]

setup(
    name="treemind",
    version="0.2.0",
    description="treemind",
    packages=find_packages(include=["treemind", "treemind.*"]),
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "nonecheck": False,
            "overflowcheck": False,
            "infer_types": True,
        },
    ),
)

