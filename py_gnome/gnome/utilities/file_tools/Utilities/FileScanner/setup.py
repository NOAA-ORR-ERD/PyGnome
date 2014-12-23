#!/usr/bin/env python
from setuptools import setup, Extension
# This setup is suitable for "python setup.py develop".  

#from distutils.core import setup, Extension


import numpy # so it can find the headers
module1 = Extension('gnome.utilities.file_tools.Utilities.FileScanner',
                    sources = ['Source/FileScan_module.c'],
                    include_dirs = [numpy.get_include(),],
                    )

description = """This is general purpose module for reading floating point numbers out of
ascii text files
"""

setup (name = 'FileScan',
       version = '0.10',
       description = description,
       ext_modules = [module1],
       )

