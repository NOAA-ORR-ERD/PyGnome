#!/usr/bin/env python

"""

The master setup.py file

you should be able to run :

python setup.py develop

to build and install the whole thing in development mode

(it will only work right with distribute, not setuptools)

All the shared C++ code is compiled with  basic_types.pyx

It needs to be imported before any other extensions (which happens in the gnome.__init__.py file)

"""

## NOTE: this works with "distribute" package, but not with setuptools.
from setuptools import setup # to support "develop" mode: 
from distutils.extension import Extension
from Cython.Distutils import build_ext
from subprocess import call

import numpy as np
import os
import sys

if "clean" in "".join(sys.argv[1:]):
    target = 'clean'
else:
    target = 'build'
    
CPP_CODE_DIR = "../lib_gnome"

# the cython extensions to build -- each should correspond to a *.pyx file
extension_names = [
                   'cy_wind_mover',
# CATS mover broken at the moment                   
#                   'cy_cats_mover',
#                   'cy_netcdf_mover',
                   'cy_ossm_time',
                   ]

cpp_files = [ 
              'RectGridVeL_c.cpp',
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


cpp_files = [os.path.join(CPP_CODE_DIR , file) for file in cpp_files]

## setting the "pyGNOME" define so that conditional compilation in the cpp files is done right.
macros = [('pyGNOME', 1),]

## Build the extension objects
extensions = []
lib= []
libdirs= []

extra_includes="."
compile_args = []
link_args = []

if sys.platform == "darwin":
    # build cy_basic_types along with lib_gnome so we can use distutils for building everything
    # and putting it in the correct place for linking.
    # cy_basic_types needs to be imported before any other extensions - this is being done
    # in the gnome/cy_gnome/__init__.py
    basic_types_ext = Extension('gnome.cy_gnome.cy_basic_types',
                                ['cygnome/cy_basic_types.pyx'] + cpp_files, 
                                language="c++",
                                define_macros = macros,
                                extra_compile_args=compile_args,
   	                            extra_link_args= ['-Wl,../third_party_lib/libnetcdf.a'],
                                include_dirs=[CPP_CODE_DIR,
                                              np.get_include(),
                                              'cyGNOME',
                                              extra_includes,
                                              ],
                                )
    
    extensions.append(basic_types_ext)

elif sys.platform == "win32":
    
    
    compile_args = ['/W0',]
    link_args = ['/DEFAULTLIB:MSVCRT.lib',
                 '/NODEFAULTLIB:LIBCMT.lib',
                ] 
    # let's build C++ here
    sys.path.append(".\gnome\DLL")   # need this for linking to work properly
    proj = '..\project_files\lib_gnomeDLL\lib_gnomeDLL.vcproj'
    config = '/p:configuration=debug' 
    platform = '/p:platform=Win32'
    call(['msbuild',proj,'/t:'+target,config,platform])

    lib += ['lib_gnomeDLL']
    libdirs += ['gnome/cy_gnome']
    macros += [('CYTHON_CCOMPLEX', 0),]
    extension_names += ['cy_basic_types']

#
### the "master" extension -- of the extra stuff, so the whole C++ lib will be there for the others
#

# TODO: the extensions below look for the shared object lib_gnome in 
# './build/lib.macosx-10.3-fat-2.7/gnome' and './gnome'
# Ideally, we should build lib_gnome first and move it to wherever we wish to link from .. currently
# the build_ext and develop will find and link to the object in different places.

for mod_name in extension_names:
   cy_file = os.path.join("cygnome", mod_name+".pyx")
   extensions.append(  Extension('gnome.cy_gnome.' + mod_name,
                                 [cy_file], 
                                 language="c++",
                                 define_macros = macros,
                                 extra_compile_args=compile_args,
                                 libraries = lib,
                                 library_dirs = libdirs,
                                 include_dirs=[CPP_CODE_DIR,
                                               np.get_include(),
                                               'cyGNOME',
                                               extra_includes,
                                               ],
                                 )
                       )


setup(name='pyGnome',
      version='alpha', 
      requires=['numpy'],
      cmdclass={'build_ext': build_ext },
      packages=['gnome',
    #            'gnome.utilities',
                ],
      ext_modules=extensions
     )

