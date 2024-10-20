from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# Compiler and linker flags for different platforms
if sys.platform == "win32":
    extra_compile_args = [
        "/O2",
        "/fp:fast",
        "/arch:AVX2",
        "/GL",
        "/Ot",
        "/Ox",
        "/favor:INTEL64",
        "/Oi",
        "/GT",
        "/Gw",
        "/GS-",
        "/Qpar",
        "/Qpar-report:1",
    ]
    extra_link_args = ["/LTCG", "/OPT:REF", "/OPT:ICF"]

elif sys.platform == "linux":
    extra_compile_args = [
        "-O3",
        "-ffast-math",
        "-march=native",
        "-funroll-loops",
        "-finline-functions",
        "-ftree-vectorize",
        "-fprefetch-loop-arrays",
        "-fstrict-aliasing",
        "-fstack-protector-strong",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    extra_link_args = ["-Wl,--gc-sections", "-Wl,--as-needed"]
elif sys.platform == "darwin":  # macOS
    extra_compile_args = [
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-finline-functions",
        "-ftree-vectorize",
        "-fstrict-aliasing",
        "-fstack-protector-strong",
        "-ffunction-sections",
        "-fdata-sections",
        "-Wno-unreachable-code-fallthrough",
    ]
    extra_link_args = ["-Wl,-dead_strip"]

define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


extensions = [
    Extension(
        name="tree_explainer.algorithm.explainer",
        sources=["tree_explainer/algorithm/explainer.pyx"],
        include_dirs=[np.get_include(), "tree_explainer/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="tree_explainer.algorithm.rule",
        sources=["tree_explainer/algorithm/rule.pyx"],
        include_dirs=[np.get_include(), "tree_explainer/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="tree_explainer.algorithm.utils",
        sources=["tree_explainer/algorithm/utils.pyx"],
        include_dirs=[np.get_include(), "tree_explainer/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="tree_explainer.algorithm.lgb",
        sources=["tree_explainer/algorithm/lgb.pyx"],
        include_dirs=[np.get_include(), "tree_explainer/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
    Extension(
        name="tree_explainer.algorithm.xgb",
        sources=["tree_explainer/algorithm/xgb.pyx"],
        include_dirs=[np.get_include(), "tree_explainer/algorithm"],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
    ),
]

setup(
    name="qpboost",
    version="0.1",
    description="QPBoost Solver",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
