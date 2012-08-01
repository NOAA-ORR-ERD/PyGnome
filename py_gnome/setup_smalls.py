#!/usr/bin/env python

## NOTE: this works with "distribute" package, but not with setuptools.
from setuptools import setup # to support "develop" mode: 
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np
import os
import sys


CPP_CODE_DIR = "../lib_gnome"


#from sysconfig import get_config_var

# the cython extensions to build -- each should corespond to a *.pyx file
extension_names = [
                   'wind_mover',
                   'cats_mover',
                   'netcdf_mover',
                   ]

cpp_files = [ 
              'RectGridVel_c.cpp',
              'MemUtils.cpp',
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
              'CATSMover_c.cpp',
              'CurrentMover_c.cpp',
              'ShioTimeValue_c.cpp',
              'ShioHeight.cpp',
              'TriGridVel_c.cpp',
              'DagTree.cpp',
              'DagTreeIO.cpp',
              'ShioCurrent1.cpp',
              'ShioCurrent2.cpp',
              'NetCDFMover_c.cpp',
              'TriGridVel3D_c.cpp',
              ]

files = [os.path.join(CPP_CODE_DIR , file) for file in cpp_files]

extra_includes="."
compile_args=None
macros = [('pyGNOME', 1),]
link_args = []

if sys.platform == "darwin":
    link_args = ['-Wl,../third_party_lib/libnetcdf.a',]
elif sys.platform == "win32":
    compile_args = ['/W0',]
    link_args = ['../third_party_lib/netcdf3.6.3.lib', \
                 '/DEFAULTLIB:MSVCRT.lib',
                 '/NODEFAULTLIB:LIBCMT.lib',
                 ]
    macros += [('CYTHON_CCOMPLEX', 0),]

## setting the "pyGNOME" define so that conditional compliation in the cpp files is done right.
macros = [('pyGNOME', 1),]

## Build the extension objects
extensions = []
for mod_name in extension_names:
    cy_file = os.path.join("cygnome", mod_name+".pyx")
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

## non-standard extensions:
basic_types_ext = Extension('gnome.basic_types',
                            ['cygnome/basic_types.pyx'], 
                            language="c++",
                            define_macros = macros,
                            include_dirs=[CPP_CODE_DIR],
                            )

extensions.append(basic_types_ext)


setup(name='pyGnome',
      version='alpha', 
      requires=['numpy'],
      cmdclass={'build_ext': build_ext },
      packages=['gnome',
                'gnome.utilities',
                ],
      ext_modules=extensions
     )


