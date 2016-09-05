#!/usr/bin/env python
from distutils.core import setup, Extension

__version__ = "0.1"

example_module = Extension('_onlineml',
    sources=['onlineml_wrap.cxx'],
    extra_compile_args=["-O3"],
    language="c++",
)
setup(
    name = 'onlineml',
    version = __version__,
    author      = "Takuya Makino",
    description = """online machine learning algorithms""",
    ext_modules = [example_module],
)
