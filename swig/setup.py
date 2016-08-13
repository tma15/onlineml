#!/usr/bin/env python
from distutils.core import setup, Extension


example_module = Extension('_onlineml',
    sources=['onlineml_wrap.cxx'],
    include_dirs=["../"],
    extra_compile_args=["-O3"],
    language="c++",
)

setup (name = '@PACKAGE@',
    version = '@VERSION@',
    author      = "SWIG Docs",
    description = """online machine learning algorithms""",
    ext_modules = [example_module],
    py_modules = ["onlineml"],
)
