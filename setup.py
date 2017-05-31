from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("find_pattern.pyx"),
    include_dirs=[numpy.get_include()]
)
