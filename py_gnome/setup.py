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
extension_names = ['model',
                   'cats_mover',
                   'random_mover',
                   'wind_mover',
                   ]

cpp_files = [ 'MemUtils.cpp',
              'Mover_c.cpp',
              'Replacements.cpp',
              'CMapLayer_c.cpp',
              'ClassID_c.cpp',
              'ComponentMover_c.cpp',
              'OUTILS.cpp',
              'TimeValuesIO.cpp',
              'NetCDFMover_c.cpp',
              'RectGridVel_c.cpp',
              'Model_c.cpp',
              'PtCurMover_c.cpp',
              'CompoundMap_c.cpp',
              'NetCDFMoverCurv_c.cpp',
              'TriGridVel3D_c.cpp',
              'TideCurCycleMover_c.cpp',
              'MakeTriangles.cpp',
              'NetCDFMoverTri_c.cpp',
              'CATSMover3D_c.cpp',
              'MakeDagTree.cpp',
              'CompoundMover_c.cpp',
              'TriCurMover_c.cpp',
              'Random3D_c.cpp',
              'PtCurMap_c.cpp', 
              'LEList_c.cpp',
              'OLEList_c.cpp',
              'OSSMWeatherer_c.cpp',
              'Weatherer_c.cpp',
              'MYRANDOM.cpp',
              'Map_c.cpp',
              'CATSMover_c.cpp',
              'GEOMETRY.cpp',
              'ShioCurrent1.cpp',
              'ShioCurrent2.cpp',
              'ShioHeight.cpp',
              'OSSMTimeValue_c.cpp',
              'TimeValue_c.cpp',
              'VectMap_c.cpp',
              'DagTreeIO.cpp',
              'RectUtils.cpp',
              'ShioTimeValue_c.cpp',
              'Random_c.cpp',
              'WindMover_c.cpp',
              'CurrentMover_c.cpp',
              'CompFunctions.cpp',
              'CMYLIST.cpp',
              'GEOMETR2.cpp',
              'TriGridVel_c.cpp',
              'DagTree.cpp',
              'StringFunctions.cpp',
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

