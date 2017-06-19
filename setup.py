from distutils.core import setup
from Cython.Build import cythonize
import numpy

# ext_modules = [Extension('func1', ['util/func1_pc.py', 'util/funct2_pc.py'],)]

setup(
    ext_modules = cythonize("cython_function/*.pyx"),
    include_dirs=[numpy.get_include()]
)
