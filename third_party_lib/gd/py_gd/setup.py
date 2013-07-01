#!/usr/bin/env python

"""
setup.py script for the py_gd package
"""

from setuptools import setup
#from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy #for the include dirs...

ext_modules=[ Extension("py_gd.py_gd",
                        ["py_gd/py_gd.pyx"],
                        include_dirs = ["../static_libs/include/",
                                        numpy.get_include(),
                                        ],
                        libraries=["jpeg"],
                        extra_objects=["../static_libs/lib/libgd.a",
                                       "../static_libs/lib/libpng.a"],
                         )]
setup(
    name = "py_gd",
    version='0.1.1',
    description = "python wrappers around libgd graphics lib",
    #long_description=read('README'),
    author = "Christopher H. Barker",
    author_email = "chris.barker@Nnoaa.gov",
    #url="",
    license = "Public Domain",
    keywords = "graphics cython drawing",
    ext_modules = cythonize(ext_modules), 
    packages = ["py_gd",],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Utilities",
        "License :: Public Domain",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 2 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Multimedia :: Graphics",
    ],
)


