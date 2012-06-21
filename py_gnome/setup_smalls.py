#!/usr/bin/env python

CPP_CODE_DIR = "../lib_gnome"

import numpy as np
import os
import sys
from sysconfig import get_config_var

from setuptools import setup
#from distutils.core import setup

from Cython.Distutils import build_ext 
from distutils.extension import Extension

# the cython extensions to build -- each should corespond to a *.pyx file
extension_names = [
                   'wind_mover',
                   ]

cpp_files = [ 'MemUtils.cpp',
              'Mover_c.cpp',
              'Replacements.cpp',
              'ClassID_c.cpp',
              'TimeValuesIO.cpp',
              'MYRANDOM.cpp',
              'GEOMETRY.cpp',
              'OSSMTimeValue_c.cpp',
              'TimeValue_c.cpp',
              'RectUtils.cpp',
              'WindMover_c.cpp',
              'CompFunctions.cpp',
              'CMYLIST.cpp',
              'GEOMETR2.cpp',
              'StringFunctions.cpp',
              'OUTILS.cpp',
              ]

files = [os.path.join(CPP_CODE_DIR , file) for file in cpp_files]

extra_includes="."

compile_args=None

## setting the "pyGNOME" define so that conditional compliation in the cpp files is done right.
macros = [('pyGNOME', 1),]

## Build the extension objects
extensions = []
for mod_name in extension_names:
    cy_file = os.path.join("cyGNOME", mod_name+".pyx")
    extensions.append(  Extension('gnome.' + mod_name,
                                  [cy_file] + files, 
                                  language="c++",
                                  define_macros = macros,
                                  extra_compile_args=compile_args,
                                  extra_link_args = ['-Wl,../third_party_lib/libnetcdf.a',],
                                  include_dirs=[CPP_CODE_DIR,
                                                np.get_include(),
                                                'cyGNOME',
                                                extra_includes,
                                                ],
                                  )
                        )


setup(name='python gnome',
      version='beta', 
      requires=['numpy'],
      cmdclass={'build_ext': build_ext },
      packages=['gnome','gnome.utilities',],
      ext_modules=extensions
     )

