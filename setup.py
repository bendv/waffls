#!/usr/bin/env python

from __future__ import print_function
import os
import numpy as np
from setuptools import setup
from distutils.extension import Extension
from distutils.command.sdist import sdist as _sdist

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("waffls._dswe", ["waffls/_dswe.pyx"]),
        Extension("waffls._pseudocluster", ["waffls/_pseudocluster.pyx"])
        ]
    cmdclass.update({
        'build_ext': build_ext
        })
else:
    ext_modules += [
        Extension("waffls._dswe", ["waffls/_dswe.c"]),
        Extension("waffls._pseudocluster", ["waffls/_pseudocluster.c"])
        ]


# Make sure the compiled Cython files in the distribution are up-to-date
class sdist(_sdist):
    def run(self):
        from Cython.Build import cythonize
        cythonize(['waffls/*.pyx'])
        _sdist.run(self)
cmdclass['sdist'] = sdist


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

exec(read('waffls/__version__.py'))

setup(
    name = 'waffls',
    version = __version__,
    packages = ['waffls',],
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    include_dirs = [np.get_include()],
    license = 'MIT',
    long_description = read('README.md'),
    long_description_content_type='text/markdown',
    install_requires = [
        'rasterio', 
        'numpy', 
        'scipy', 
        'scikit-learn', 
        'datetime', 
        ],
    author = 'Ben DeVries',
    author_email = 'devries.br@gmail.com',
    url = 'https://github.com/bendv/waffls'
)


