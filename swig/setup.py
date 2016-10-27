import sys
import subprocess
from distutils.core import setup, Extension

def cmd(string):
    out = subprocess.check_output(string.split(), universal_newlines=True)
    return out.strip()

__version__ = cmd("train_onlineml_model -v")

example_module = Extension('_onlineml',
    sources=['onlineml_wrap.cxx'],
    extra_compile_args=["-O9"],
    language="c++",
)
setup(
    name = 'onlineml',
    version = __version__,
    author      = "Takuya Makino",
    description = """online machine learning algorithms""",
    ext_modules = [example_module],
)
