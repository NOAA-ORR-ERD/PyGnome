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
import sys
import sysconfig
import glob
import shutil

# to support "develop" mode:
from setuptools import setup, find_packages

from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np


def target_dir(name):
    '''Returns the name of a distutils build directory'''
    f = '{dirname}.{platform}-{version[0]}.{version[1]}'
    return f.format(dirname=name,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)


def target_path(name='temp'):
    '''returns the full build path'''
    return os.path.join('build', target_dir(name))


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


## setup our environment and architecture
## These should be properties that are used by all Extensions
libfile = ''
if sys.maxint <= 2147483647:
    architecture = 'i386'
else:
    architecture = 'x86_64'

if sys.platform == 'darwin':
    # for the mac -- decide whether we are 32 bit build
    if architecture == 'i386':
        #Setting this should force only 32 bit intel build
        os.environ['ARCHFLAGS'] = "-arch i386"
    else:
        os.environ['ARCHFLAGS'] = "-arch x86_64"
    libfile = 'lib{0}.a'  # OSX static library filename format
elif sys.platform == "win32":
    # Distutils normally only works with VS2008.
    # this is to trick it into seeing VS2010 or VS2012
    # We will prefer VS2012, then VS2010
    if 'VS110COMNTOOLS' in os.environ:
        os.environ['VS90COMNTOOLS'] = os.environ['VS110COMNTOOLS']
    elif 'VS100COMNTOOLS' in os.environ:
        os.environ['VS90COMNTOOLS'] = os.environ['VS100COMNTOOLS']

    libfile = '{0}.lib'  # windows static library filename format


##
## setup our third party libraries environment - for Win32/Mac OSX
## Linux does not use the libraries in third_party_lib. It links against
## netcdf shared objects installed by apt-get
##
if sys.platform is "darwin" or "win32":
    third_party_dir = os.path.join('..', 'third_party_lib')
    
    # the netCDF environment
    netcdf_base = os.path.join(third_party_dir, 'netcdf-4.3',
                              sys.platform, architecture)
    netcdf_libs = os.path.join(netcdf_base, 'lib')
    netcdf_inc = os.path.join(netcdf_base, 'include')
    
    if sys.platform == 'win32':
        netcdf_names = ('netcdf',)
    else:
        netcdf_names = ('hdf5', 'hdf5_hl', 'netcdf', 'netcdf_c++4')
    
    netcdf_lib_files = [os.path.join(netcdf_libs, libfile.format(l))
                        for l in netcdf_names]


# the cython extensions to build -- each should correspond to a *.pyx file
extension_names = ['cy_mover',
                   'cy_helpers',
                   'cy_wind_mover',
                   'cy_cats_mover',
                   'cy_gridcurrent_mover',
                   'cy_gridwind_mover',
                   'cy_ossm_time',
                   'cy_random_mover',
                   'cy_random_vertical_mover',
                   'cy_rise_velocity_mover',
                   'cy_land_check',
                   'cy_grid_map',
                   'cy_shio_time',
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
             'RiseVelocity_c.cpp',
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
link_args = []

# List of include directories for cython code.
# append to this list as needed for each platform
include_dirs = [cpp_code_dir,
                np.get_include(),
                netcdf_inc,
                '.']
static_lib_files = netcdf_lib_files

# build cy_basic_types along with lib_gnome so we can use distutils
# for building everything
# and putting it in the correct place for linking.
# cy_basic_types needs to be imported before any other extensions.
# This is being done in the gnome/cy_gnome/__init__.py

# JS NOTE: 'darwin' and 'win32' statically link against netcdf library.
#          On linux, we link against the dynamic netcdf libraries (shared 
#          objects) since netcdf, hdf5 can be installed with a package manager.
#          We also don't have the static builds for these.
#          Also, the static_lib_files only need to be linked against 
#          lib_gnome in the following Extension.

if sys.platform == "darwin":

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

elif sys.platform == "win32":
    # build our compile arguments
    macros.append(('_EXPORTS', 1))
    macros.append(('_CRT_SECURE_NO_WARNINGS', 1))

    compile_args = ['/EHsc']

    link_args.append('/MANIFEST')

    include_dirs.append(os.path.join(third_party_dir, 'win32_headers'))

    # build our linking arguments
    libdirs.append(netcdf_libs)

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
    static_lib_files = [os.path.join(target_path(),
                                     'Release', 'gnome', 'cy_gnome',
                                     'cy_basic_types.lib')]
    libdirs = []

elif sys.platform == "linux2":

    ## for some reason I have to create build/temp.linux-i686-2.7
    ## else the compile fails saying temp.linux-i686-2.7 is not found
    ## required for develop or install mode
    build_temp = target_path()
    if 'clean' not in sys.argv[1] and not os.path.exists(build_temp):
        os.makedirs(build_temp)

    ## Not sure calling setup twice is the way to go - but do this for now
    ## NOTE: This is also linking against the netcdf library (*.so), not
    ## the static netcdf. We didn't build a NETCDF static library.
    setup(name='pyGnome',  # not required since ext defines this
          cmdclass={'build_ext': build_ext},
          ext_modules=[Extension('gnome.cy_gnome.libgnome',
                                 cpp_files,
                                 language='c++',
                                 define_macros=macros,
                                 libraries=['netcdf'],
                                 include_dirs=[cpp_code_dir],
                                 )])

    ## In install mode, it compiles and builds libgnome inside
    ## lib.linux-i686-2.7/gnome/cy_gnome
    ## This should be moved to build/temp.linux-i686-2.7 so cython files
    ## build and link properly
    if 'install' in sys.argv[1]:
        bdir = glob.glob(os.path.join('build/*/gnome/cy_gnome', 'libgnome.so'))
        if len(bdir) > 1:
            raise Exception("Found more than one libgnome.so library" \
                            " during install mode in 'build/*/gnome/cy_gnome'")
        if len(bdir) == 0:
            raise Exception("Did not find libgnome.so library during install" \
                            " mode in 'build/*/gnome/cy_gnome'")

        libpath = os.path.dirname(bdir[0])

    else:
        libpath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'gnome', 'cy_gnome')

    ## Need this for finding lib during linking and at runtime
    ## using -rpath to define runtime path. Use $ORIGIN to define libgnome.so
    ## relative to cy_*.so
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


#
### All other lib_gnome-based cython extensions.
### These depend on the successful build of cy_basic_types
#

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
                                #extra_objects=static_lib_files,
                                include_dirs=include_dirs,
                                )
                       )

# and platfrom-independent cython extensions:
# well...not entirely platform-independant.  We need to pass the link_args
poly_cypath = os.path.join('gnome', 'utilities', 'geometry')
sources = [os.path.join(poly_cypath, 'cy_point_in_polygon.pyx'),
           os.path.join(poly_cypath, 'c_point_in_polygon.c')]
extensions.append(Extension("gnome.utilities.geometry.cy_point_in_polygon",
                            sources=sources,
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
