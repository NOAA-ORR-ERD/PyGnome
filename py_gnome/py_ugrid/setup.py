#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup # to support "develop" mode
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy # for the includes for the Cython code


ext_modules = [Extension("py_ugrid.geometry.cy_point_in_polygon",
                         sources=["py_ugrid/geometry/cy_point_in_polygon.pyx",
                                  "py_ugrid/geometry/c_point_in_polygon.c"],
                         include_dirs=[numpy.get_include()]),
               ]

setup(
    name = "py_ugrid",
    version = "0.02",
    author = "Chris Barker, Eli Ateljevich, Rusty Chris Holleman",
    author_email = "Chris.Barker@noaa.gov",
    description = ("A package for working with unstructured grids, and the data on them"),
    license = "BSD",
    keywords = "unstructured numpy models",
#    url = "http://packages.python.org/an_example_pypi_project",
#    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        ],
    cmdclass = {'build_ext': build_ext},
    packages = ["py_ugrid","tests"],
    ext_modules = ext_modules,
    scripts = [],
    )
