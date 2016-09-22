#!/usr/bin/env python

"""
setup.py for geometry package

It's now built with the main gnome setup.py, but kept this here for easier testing in place...
only useful now for "develop" mode
"""

#from distutils.core import setup
from setuptools import setup # to support "develop" mode
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy # for the includes for the Cython code


ext_modules = [Extension("cy_point_in_polygon",
                         sources=["cy_point_in_polygon.pyx",
                                  "c_point_in_polygon.c"],
                         include_dirs=[numpy.get_include()]),
               ]

setup(
    name = "geometry",
    version = "0.03",
    author = "Chris Barker",
    author_email = "Chris.Barker@noaa.gov",
    description = ("some geometry for GNOME"),
#    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        ],
    cmdclass = {'build_ext': build_ext},
#    packages = ["geometry"],
    ext_modules = ext_modules,
#    modules = ["BBox.py",]
    scripts = [],
    )
