#!/usr/bin/env python

"""

The master setup.py file for py_gnome

you should be able to run :

python setup.py develop

to build and install the whole thing in development mode

(it will only work right with distribute, not setuptools)

All the shared C++ code is compiled with  basic_types.pyx

It needs to be imported before any other extensions
(which happens in the gnome.__init__.py file)

"""

## NOTE: this works with "distribute" package, but not with setuptools.
import os
import glob
import shutil
import sys

# to support "develop" mode:
from setuptools import setup, find_packages

from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

if "clean" in "".join(sys.argv[1:]):
    target = 'clean'
else:
    target = 'build'

if "cleanall" in "".join(sys.argv[1:]):
    target = 'clean'

    rm_files = ['gnome/cy_gnome/*.so',
                'gnome/cy_gnome/cy_*.pyd',
                'gnome/cy_gnome/cy_*.cpp',
                'gnome/utilities/geometry/cy_*.pyd',
                'gnome/utilities/geometry/cy_*.so',
                'gnome/utilities/geometry/cy_*.c',
                ]

    for files_ in rm_files:
        for file_ in glob.glob(files_):
            print "Deleting auto-generated files: {0}".format(file_)
            os.remove(file_)


    rm_dir = ['pyGnome.egg-info', 'build']
    for dir_ in rm_dir:
        print "Deleting auto-generated directory: {0}".format(dir_)
        try:
            shutil.rmtree(dir_)
        except OSError as e:
            print e

    # this is what distutils understands
    sys.argv[1] = 'clean'


# only for windows
if "debug" in "".join(sys.argv[2:]):
    config = 'debug'
else:
    config = 'release'    # only used by windows


if sys.argv.count(config) != 0:
    sys.argv.remove(config)


# for the mac -- decide whether we are 32 bit build
if sys.platform == 'darwin' and sys.maxint <= 2147483647:
    #Setting this should force only 32 bit intel build
    os.environ['ARCHFLAGS'] = "-arch i386"


# the cython extensions to build -- each should correspond to a *.pyx file
extension_names = [
                   'cy_helpers',
                   'cy_ossm_time',
                   'cy_shio_time',
                   'cy_mover',
                   'cy_wind_mover',
                   'cy_cats_mover',
                   'cy_gridcurrent_mover',
                   'cy_gridwind_mover',
                   'cy_random_mover',
                   'cy_random_vertical_mover',
                   'cy_land_check',
                   'cy_grid_map',
                   ]

cpp_files = ['RectGridVeL_c.cpp',
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
             #'NetCDFMover_c.cpp',
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
             'GridMap_c.cpp',
             'GridMapUtils.cpp',
             'RandomVertical_c.cpp',
             ]


cpp_code_dir = os.path.join('..', 'lib_gnome')
cpp_files = [os.path.join(cpp_code_dir, f) for f in cpp_files]

## setting the "pyGNOME" define so that conditional compilation
## in the cpp files is done right.
macros = [('pyGNOME', 1), ]

## Build the extension objects
compile_args = []
extensions = []

lib = []
libdirs = []
static_lib_files = []
link_args = []

# List of include directories for cython code.
# append to this list as needed for each platform
include_dirs = [cpp_code_dir, np.get_include(), '.']


# build cy_basic_types along with lib_gnome so we can use distutils
# for building everything
# and putting it in the correct place for linking.
# cy_basic_types needs to be imported before any other extensions.
# This is being done in the gnome/cy_gnome/__init__.py

if sys.platform == "darwin":
    if os.environ.has_key('ARCHFLAGS'):
        print 'setting darwin ARCHFLAGS to our architecture.'
        architecture = os.environ['ARCHFLAGS'].split()[1]
    else:
        architecture = 'x86_64'

    include_dirs.append('../third_party/%s/include' % architecture)
    third_party_lib_dir = '../third_party/%s/lib' % architecture

    static_lib_names = ('hdf5', 'hdf5_hl', 'netcdf', 'netcdf_c++4')
    static_lib_files = [os.path.join(third_party_lib_dir, 'lib%s.a' % l)
                        for l in static_lib_names]

    basic_types_ext = Extension(r'gnome.cy_gnome.cy_basic_types',
            ['gnome/cy_gnome/cy_basic_types.pyx'] + cpp_files,
            language='c++',
            define_macros=macros,
            extra_compile_args=compile_args,
            extra_link_args=['-lz', '-lcurl'],
            extra_objects=static_lib_files,
            include_dirs=include_dirs,
            )

    extensions.append(basic_types_ext)
    static_lib_files = []


elif sys.platform == "linux2":

    ## for some reason I have to create build/temp.linux-i686-2.7
    ## else the compile fails saying temp.linux-i686-2.7 is not found
    ## required for develop or install mode
    build_temp = './build/temp.linux-i686-2.7' 
    if 'clean' not in sys.argv[1] and not os.path.exists('./build/temp.linux-i686-2.7'):
        os.makedirs(build_temp)    
   
    ## Not sure calling setup twice is the way to go - but do this for now
    setup(name='pyGnome', # not required since ext defines this
          cmdclass={'build_ext': build_ext},
          ext_modules=[Extension('gnome.cy_gnome.libgnome',
                                 cpp_files,
                                 language='c++',
                                 define_macros=macros,
                                 libraries=['netcdf'],
                                 include_dirs=[ cpp_code_dir],
                                 )])
  
    ## in install mode, it compiles and builds libgnome inside lib.linux-i686-2.7/gnome/cy_gnome
    ## this should be moved to build/temp.linux-i686-2.7 so cython files build and link properly
    if 'install' in sys.argv[1]:
        bdir = glob.glob(os.path.join('build/*/gnome/cy_gnome','libgnome.so'))
        if len(bdir) > 1:
            raise Exception("Found more than one libgnome.so library during install mode in 'build/*/gnome/cy_gnome'")
        if len(bdir) == 0:
            raise Exception("Did not find libgnome.so library during install mode in 'build/*/gnome/cy_gnome'")
        
        libpath = os.path.dirname( bdir[0])

    else:
        libpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'gnome', 'cy_gnome')
    
    ## Need this for finding lib during linking and at runtime
    ## using -rpath to define runtime path. Use $ORIGIN to define libgnome.so relative to cy_*.so
    os.environ['LDFLAGS'] = "-L{0} -Wl,-rpath='$ORIGIN'".format(libpath)

    ## End building C++ shared object
    lib = ['gnome']
    basic_types_ext = Extension(r'gnome.cy_gnome.cy_basic_types',
                                ['gnome/cy_gnome/cy_basic_types.pyx'],
                                language='c++',
                                define_macros=macros,
                                extra_compile_args=compile_args,
                                libraries=lib,
                                include_dirs=include_dirs,
                                )

    extensions.append(basic_types_ext)

elif sys.platform == "win32":
    # Distutils normally only works with VS2008.
    # this is to trick it into seeing VS2010 or VS2012
    # We will prefer VS2012, then VS2010
    if 'VS110COMNTOOLS' in os.environ:
        os.environ['VS90COMNTOOLS'] = os.environ['VS110COMNTOOLS']
    elif 'VS100COMNTOOLS' in os.environ:
        os.environ['VS90COMNTOOLS'] = os.environ['VS100COMNTOOLS']

    # build our compile arguments
    macros.append(('_EXPORTS', 1))
    macros.append(('_CRT_SECURE_NO_WARNINGS', 1))
    compile_args = ['/EHsc']
    include_dirs.append(os.path.join('..', 'third_party_lib', 'netcdf-4.3'))
    include_dirs.append(os.path.join('..', 'third_party_lib', 'win32_headers')) #Zelenke

    # build our linking arguments
    netcdf_dir = os.path.join('..', 'third_party_lib', 'netcdf-4.3',
                              'win32', 'i386')
    libdirs.append(netcdf_dir)
    link_args.append('/MANIFEST')

    static_lib_names = ('netcdf',)
    static_lib_files = [os.path.join(netcdf_dir, '%s.lib' % l)
                        for l in static_lib_names]

    basic_types_ext = Extension(r'gnome.cy_gnome.cy_basic_types',
            [r'gnome\cy_gnome\cy_basic_types.pyx'] + cpp_files,
            language='c++',
            define_macros=macros,
            extra_compile_args=compile_args,
            library_dirs=libdirs,
            extra_link_args=link_args,
            extra_objects=static_lib_files,
            include_dirs=include_dirs,
            )

    extensions.append(basic_types_ext)

    # we will reference this library when building all other extensions
    static_lib_files = [os.path.join('build', 'temp.win32-2.7', 'Release',
                                     'gnome', 'cy_gnome',
                                     'cy_basic_types.lib')]
    libdirs = []


#
### the "master" extension -- of the extra stuff,
### so the whole C++ lib will be there for the others
#

# TODO: the extensions below look for the shared object lib_gnome in
# './build/lib.macosx-10.3-fat-2.7/gnome' and './gnome'
# Ideally, we should build lib_gnome first and move it
# to wherever we wish to link from .. currently the build_ext and develop
# will find and link to the object in different places.

for mod_name in extension_names:
    cy_file = os.path.join("gnome/cy_gnome", mod_name + ".pyx")
    extensions.append(Extension('gnome.cy_gnome.' + mod_name,
                                 [cy_file],
                                 language="c++",
                                 define_macros=macros,
                                 extra_compile_args=compile_args,
                                 extra_link_args=link_args,
                                 libraries=lib,
                                 library_dirs=libdirs,
                                 extra_objects=static_lib_files,
                                 include_dirs=include_dirs,
                                 )
                       )

# and platfrom-independent cython extensions:
# well...not entirely platform-independant.  We need to pass the link_args
extensions.append( Extension("gnome.utilities.geometry.cy_point_in_polygon",
                             sources=["gnome/utilities/geometry/cy_point_in_polygon.pyx",
                                      "gnome/utilities/geometry/c_point_in_polygon.c"],
                             include_dirs=[np.get_include()],
                             extra_link_args=link_args,
                            )
                  )

setup(name='pyGnome',
      version='alpha',
      requires=['numpy'],
      cmdclass={'build_ext': build_ext},
      packages=find_packages(exclude=['gnome.deprecated']),
      ext_modules=extensions
     )
