#!/usr/bin/env python
from distutils.core import setup, Extension


example_module = Extension('_onlineml',
    sources=['onlineml_wrap.cxx'],
    include_dirs=["../src/include"],
    extra_compile_args=["-O3"],
    language="c++",
)

setup (name = 'onlineml',
    version = '0.1',
    author      = "SWIG Docs",
    description = """online machine learning algorithms""",
    ext_modules = [example_module],
    py_modules = ["onlineml"],
)
