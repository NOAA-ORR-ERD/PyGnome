#!/usr/bin/env python

"""

The master setup.py file for py_gnome

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
import subprocess 

import numpy as np
import os
import sys

if "clean" in "".join(sys.argv[1:]):
    target = 'clean'
else:
    target = 'build'

if "cleanall" in "".join(sys.argv[1:]):
    target = 'clean'
    print "Deleting cython files .."
    os.system('rm -v gnome/cy_gnome/cy_*.so')
    os.system('rm -v gnome/cy_gnome/cy_*.pyd')
    os.system('rm -v gnome/cy_gnome/cy_*.cpp')
    os.system('rm -rv build')
    os.system('rm -rv pyGnome.egg-info')
    sys.argv[1] = 'clean'   # this is what distutils understands

# only for windows
if "debug" in "".join(sys.argv[2:]):
    config = 'debug'
else:
    config = 'release'    # only used by windows

sys.argv.count(config) != 0 and sys.argv.remove(config)

# for the mac -- forcing 32 bit only builds
if sys.platform == 'darwin':
    #Setting this should force only 32 bit intel build
    os.environ['ARCHFLAGS'] = "-arch i386"


CPP_CODE_DIR = "../lib_gnome"

# the cython extensions to build -- each should correspond to a *.pyx file
extension_names = ['cy_mover',
                   'cy_helpers',
                   'cy_wind_mover',
                   'cy_cats_mover',
                   #'cy_gridcurrent_mover',
                   #'cy_gridwind_mover',
                   'cy_ossm_time',
                   'cy_random_mover',
                   'cy_land_check',
                   'cy_shio_time',
                   ]

cpp_files = [ 
              'RectGridVeL_c.cpp',
              'MemUtils.cpp',
              'Mover_c.cpp',
              'Replacements.cpp',
              'ClassID_c.cpp',
              'Random_c.cpp',
              'TimeValuesIO.cpp',
              'GEOMETRY.cpp',
              'OSSMTimeValue_c.cpp',
              'TimeValue_c.cpp',
              'RectUtils.cpp',
              'WindMover_c.cpp',
              'CompFunctions.cpp',
              #'CMYLIST.cpp',
              #'GEOMETR2.cpp',
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
              'GridCurrentMover_c.cpp',
              'GridWindMover_c.cpp',
              'TimeGridVel_c.cpp',
              'TimeGridWind_c.cpp',
              'MakeTriangles.cpp',
              'MakeDagTree.cpp',
              ]


cpp_files = [os.path.join(CPP_CODE_DIR , file) for file in cpp_files]

## setting the "pyGNOME" define so that conditional compilation in the cpp files is done right.
macros = [('pyGNOME', 1),]

## Build the extension objects
extensions = []
lib= []
libdirs= []

compile_args = []
link_args = []

# List of include directories for cython code - append to this list as needed for each platform
l_include_dirs = [CPP_CODE_DIR,
                    np.get_include(),
                    '.']


if sys.platform == "darwin":
    # build cy_basic_types along with lib_gnome so we can use distutils for building everything
    # and putting it in the correct place for linking.
    # cy_basic_types needs to be imported before any other extensions - this is being done
    # in the gnome/cy_gnome/__init__.py
    basic_types_ext = Extension('gnome.cy_gnome.cy_basic_types',
                                ['gnome/cy_gnome/cy_basic_types.pyx'] + cpp_files, 
                                language="c++",
                                define_macros = macros,
                                extra_compile_args=compile_args,
   	                            extra_link_args= ['-Wl,../third_party_lib/libnetcdf.a'],
                                include_dirs=l_include_dirs,
                                )
    
    extensions.append(basic_types_ext)

elif sys.platform == "win32":
   
    # see if msbuild exists
    # using subprocess.PIPE to supress output - not quite sure how/why it works?
    found_msbuild = subprocess.call("where msbuild",
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    if found_msbuild == 0:
        msbuild = 'msbuild'
    else:
        msbuild = r'c:\Windows\Microsoft.NET\Framework\v3.5\msbuild.exe'
        found_msbuild = os.path.exists( msbuild)
        if not found_msbuild:
            print "Could not find msbuild in system path"
            print "Retry after adding msbuild to system path"
            print "or try building from Visual studio prompt"
            sys.exit(0)

    compile_args = ['/W0','/MD']
   
    # NOTE: This used to work with the runtime libraries
    #       that were part of lib_gnomeDLL, however, that is currently broken.
    #       The updates below was a try at forcing the same libs as the working version
    #       in CythonProj/ .. still working on this issue
    #link_args = ['DEFAULTLIB:MSVCRT.lib',
    #             '/NODEFAULTLIB:LIBCMT.lib'
    #             ]

    link_args = [
                 #'/VERBOSE:LIB',               # shows library search path
                 #'/DEFAULTLIB:kernel32.lib',
                 #'/DEFAULTLIB:user32.lib',
                 #'/DEFAULTLIB:gdi32.lib', 
                 #'/DEFAULTLIB:winspool.lib', 
                 #'/DEFAULTLIB:comdlg32.lib', 
                 #'/DEFAULTLIB:advapi32.lib', 
                 #'/DEFAULTLIB:shell32.lib',
                 #'/DEFAULTLIB:ole32.lib', 
                 #'/DEFAULTLIB:oleaut32.lib', 
                 #'/DEFAULTLIB:uuid.lib',
                 #'/DEFAULTLIB:odbc32.lib',
                 #'/DEFAULTLIB:odbccp32.lib',
                 #'/DEFAULTLIB:libcpmt.lib',
                 #'/DEFAULTLIB:LIBCMT.lib',
                 ]
    # let's build C++ here
    sys.path.append(".\gnome\DLL")   # need this for linking to work properly
    proj = '..\project_files\lib_gnomeDLL\lib_gnomeDLL.sln'
    platform = '/p:platform=Win32'
    subprocess.call([msbuild,proj,'/t:'+target,'/p:configuration='+config,platform])

    lib += ['lib_gnomeDLL']
    libdirs += ['gnome/cy_gnome']
    macros += [('CYTHON_CCOMPLEX', 0),]
    extension_names += ['cy_basic_types']
    l_include_dirs += [r'..\third_party_lib\vs2008']

#
### the "master" extension -- of the extra stuff, so the whole C++ lib will be there for the others
#

# TODO: the extensions below look for the shared object lib_gnome in 
# './build/lib.macosx-10.3-fat-2.7/gnome' and './gnome'
# Ideally, we should build lib_gnome first and move it to wherever we wish to link from .. currently
# the build_ext and develop will find and link to the object in different places.

for mod_name in extension_names:
   cy_file = os.path.join("gnome/cy_gnome", mod_name+".pyx")
   extensions.append(  Extension('gnome.cy_gnome.' + mod_name,
                                 [cy_file], 
                                 language="c++",
                                 define_macros = macros,
                                 extra_compile_args=compile_args,
   	                             extra_link_args=link_args,
                                 libraries = lib,
                                 library_dirs = libdirs,
                                 include_dirs=l_include_dirs,
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
