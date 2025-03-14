## THIS IS THE OLD SETUP.PY -- not longer used
## Still here, just in case we want to reference it in the future


#!/usr/bin/env python
"""
The master setup.py file for py_gnome

you should be able to run :
    python setup.py develop

To build and install the whole thing in development mode
All the shared C++ code is compiled with  basic_types.pyx

It needs to be imported before any other extensions
(which happens in the gnome.__init__.py file)
"""

import datetime
import importlib
import glob
import os
from pathlib import Path
import platform
import shutil
import sys
import sysconfig


# to support "develop" mode:
import setuptools
from setuptools import setup

from distutils.command.clean import clean

from distutils.extension import Extension
from Cython.Distutils import build_ext

from git import Repo

import numpy as np

# could run setup from anywhere (though not really??)
SETUP_PATH = os.path.dirname(os.path.abspath(__file__))

# the extension used for compiled modules
# does this not work ??
# comp_modules_ext = sysconfig.get_config_var('EXT_SUFFIX')
# if comp_modules_ext is None:
#     comp_modules_ext = '.lib' if 'win' in sys.platform else ".so"
if sys.version_info.major < 3 or sys.version_info.minor < 9:
    raise NotImplementedError("PyGNOME can only be built with Python 3.9 or above")
else:
    py_impl = ['.'] + [c.lower() for c in platform.python_implementation() if c.isupper()]
    py_impl = py_impl + sysconfig.get_python_version().split('.') + ['-'] + [sysconfig.get_platform().replace('-','_')]
    win_comp_modules_ext = ''.join(py_impl) + '.lib'

# cd to SETUP_PATH, run develop or install, then cd back
CWD = os.getcwd()
os.chdir(SETUP_PATH)

try:
    repo = Repo('../.')
    branch_name = repo.active_branch.name
    last_update = next(repo.iter_commits()).committed_datetime.isoformat(),
except:  # anything goes wrong -- we want to keep moving
    branch_name = 'no-branch'
    last_update = datetime.datetime.now().isoformat()


def target_dir(name):
    '''Returns the name of a distutils build directory'''
    return ('{dirname}.{platform}-cpython-{version[0]}{version[1]}'
            .format(dirname=name,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info))


def target_path(name='temp'):
    '''returns the full build path'''
    return os.path.join('build', target_dir(name))


class cleanall(clean):
    description = ("cleans files generated by 'develop' mode and files "
                   "autogenerated by cython")

    def run(self):
        # call base class clean
        clean.run(self)

        self.clean_python_files()
        self.clean_cython_files()

        rm_dir = ['pyGnome.egg-info', 'build']
        for dir_ in rm_dir:
            print("Deleting auto-generated directory: {0}".format(dir_))
            try:
                shutil.rmtree(dir_)
            except OSError as err:
                if err.errno != 2:  # ignore the not-found error
                    raise

    def clean_python_files(self):
        # clean any byte-compiled python files
        paths = [os.path.join(SETUP_PATH, 'gnome'),
                 os.path.join(SETUP_PATH, 'scripts'),
                 os.path.join(SETUP_PATH, 'tests')]
        exts = ['*.pyc']

        self.clean_files(paths, exts)

    def clean_cython_files(self):
        # clean remaining cython/cpp files
        paths = [os.path.join(SETUP_PATH, 'gnome', 'cy_gnome'),
                 os.path.join(SETUP_PATH, 'gnome', 'utilities', 'geometry')]
        exts = ['*.so', 'cy_*.pyd', 'cy_*.cpp', 'cy_*.c']

        self.clean_files(paths, exts)

    def clean_files(self, paths, exts):
        for path in paths:
            # first, any local files directly under the path
            for ext in exts:
                for f in glob.glob(os.path.join(path, ext)):
                    self.delete_file(f)

            # next, walk any sub-directories
            for root, dirs, _files in os.walk(path, topdown=False):
                for d in dirs:
                    for ext in exts:
                        for f in glob.glob(os.path.join(root, d, ext)):
                            self.delete_file(f)

    def delete_file(self, filepath):
        print("Deleting auto-generated file: {0}".format(filepath))
        try:
            if os.path.isdir(filepath):
                shutil.rmtree(filepath)
            else:
                os.remove(filepath)
        except OSError as err:
            print(("Failed to remove {0}. Error: {1}"
                  .format(filepath, err)))


# setup our environment and architecture
# These should be properties that are used by all Extensions
libfile = ''

# fixme: is this only for the mac?  And aren't there better ways to get the
#        architecture?
if sys.maxsize <= 2 ** 32:
    architecture = 'i386'
else:
    architecture = 'x86_64'

if sys.platform == 'darwin':
    libfile = 'lib{0}.a'  # OSX static library filename format

elif sys.platform == "win32":
    libfile = '{0}.lib'  # windows static library filename format

# setup our third party libraries environment - for Win32/Mac OSX
# fixme: on Windows, should use the conda libs.

# Linux and OS-X should be using conda (or system) libs for netcdf

def get_netcdf_libs():
    """
    Find the netcdf4 libaries:

    1) if present rely on nc-config
    2) search for a user env var
    3) try to look directly for conda libs
    4) fall back to the versions distributed with the py_gnome code
    """
    import subprocess

    # check for nc-config
    try:
        result = subprocess.check_output(["nc-config", "--libs"]).split()
        lib_dir = result[0]
        libs = result[1:]
        include_dir = subprocess.check_output(["nc-config", "--includedir"])
        include_dir = include_dir.decode("ASCII").strip()
        lib_dir = lib_dir.decode("ASCII")
        libs = [l.decode("ASCII") for l in libs]
        return lib_dir, libs, include_dir
    except OSError:
        raise NotImplementedError("this setup.py needs nc-config "
                                  "to find netcdf libs")

def get_conda_includes_win():
    """
    If this is being run within a conda environment, use CONDA_PREFIX
    and get the include, bin, and lib dirs off that. If no CONDA_PREFIX is found,
    revert to the old third_party_libs
    """
    prefix = os.environ['CONDA_PREFIX']
    lib_dir = os.path.join(prefix, 'Library', 'lib')
    inc_dir = os.path.join(prefix, 'Library', 'include')
    bin_dir = os.path.join(prefix, 'Library', 'bin')
    return lib_dir, inc_dir, bin_dir

# maybe this will work with Windows, too" at least with conda?
if sys.platform.startswith("linux") or sys.platform == "darwin":
    netcdf_base, netcdf_libs, netcdf_inc = get_netcdf_libs()
    netcdf_lib_files = []
elif sys.platform in ("win32"):  # but for now, still using our shipped libs.
    third_party_dir = os.path.join('..', 'third_party_lib')
    if 'CONDA_PREFIX' in os.environ:
        netcdf_libs, netcdf_inc, netcdf_dll = get_conda_includes_win()
        if sys.platform == 'win32':
            netcdf_names = ('netcdf',)
        else:
            netcdf_names = ('hdf5', 'hdf5_hl', 'netcdf', 'netcdf_c++4')
        netcdf_lib_files = [os.path.join(netcdf_libs, libfile.format(l))
                            for l in netcdf_names]
    else:
        # the netCDF environment
        netcdf_base = os.path.join(third_party_dir, 'netcdf-4.3',
                                sys.platform, architecture)
        netcdf_libs = os.path.join(netcdf_base, 'lib')
        netcdf_inc = os.path.join(netcdf_base, 'include')

        if sys.platform == 'win32':  # oddly, 64 bit Windows is still win32
            # also copy the netcdf *.dlls to cy_gnome directory
            # On windows the dlls have the same names for those used by python's
            # netCDF4 module and PyGnome modules. For PyGnome, we had the latest
            # netcdf dlls from UCARR site but this was giving DLL import errors.
            # Netcdf dlls that come with Python's netCDF4 module (C. Gohlke's site)
            # were different from the netcdf4 DLLs we got from UCARR.
            # For now, third_party_lib contains the DLLs installed in site-packages
            # from C. Gohlke's site.
            #
            # Alternatively, we could also look for python netCDF4 package and copy
            # DLLs from site-packages. This way the DLLs used and loaded by PyGnome
            # are the same as the DLL used and expected by netCDF4. PyGnome loads
            # the DLL with cy_basic_types.pyd and it also imports netCDF4 when
            # netcdf_outputters module is imported - this was causing the previous
            # conflict. The DLL loaded in memory should be consistent - that's the
            # best understanding of current issue!
            # STILL WORKING ON A MORE PERMANENT SOLUTION
            win_dlls = os.path.join(netcdf_base, 'bin')
            dlls_path = os.path.join(os.getcwd(), win_dlls)

            for dll in glob.glob(os.path.join(dlls_path, '*.dll')):
                dlls_dst = os.path.join(os.getcwd(), 'gnome/cy_gnome/')

                dll_name = os.path.split(dll)[1]
                if sys.argv[1] == 'cleanall' or sys.argv[1] == 'clean':
                    rm_dll = os.path.join(dlls_dst, dll_name)
                    if os.path.exists(rm_dll):
                        os.remove(rm_dll)
                        print("deleted: " + rm_dll)
                else:
                    # Note: weird permissions/file locking thing on Windows --
                    #       couldn't delete or overwrite the dll...
                    #       so only copy if it's not there already
                    if not os.path.isfile(os.path.join(dlls_dst, dll_name)):
                        print("copy: " + dll + " to: " + dlls_dst)
                        shutil.copy(dll, dlls_dst)
            netcdf_names = ('netcdf',)
        else:
            netcdf_names = ('hdf5', 'hdf5_hl', 'netcdf', 'netcdf_c++4')


        netcdf_lib_files = [os.path.join(netcdf_libs, libfile.format(l))
                            for l in netcdf_names]


# the cython extensions to build -- each should correspond to a *.pyx file
extension_names = ['cy_mover',
                   'cy_helpers',
                   'cy_wind_mover',
                   'cy_current_mover',
                   'cy_cats_mover',
                   'cy_component_mover',
                   'cy_gridcurrent_mover',
                   'cy_gridwind_mover',
                   'cy_ice_mover',
                   'cy_ice_wind_mover',
                   'cy_currentcycle_mover',
                   'cy_ossm_time',
                   'cy_random_mover',
                   'cy_random_mover_3d',
                   'cy_rise_velocity_mover',
                   'cy_land_check',
                   'cy_grid_map',
                   'cy_shio_time',
                   'cy_grid',
                   'cy_grid_rect',
                   'cy_grid_curv',
                   'cy_weatherers'
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
             # 'CMYLIST.cpp',
             # 'GEOMETR2.cpp',
             'StringFunctions.cpp',
             'OUTILS.cpp',
             # 'NetCDFMover_c.cpp',
             'CATSMover_c.cpp',
             'CurrentMover_c.cpp',
             'ComponentMover_c.cpp',
             'ShioTimeValue_c.cpp',
             'ShioHeight.cpp',
             'TriGridVel_c.cpp',
             'DagTree.cpp',
             'DagTreeIO.cpp',
             'ShioCurrent1.cpp',
             'ShioCurrent2.cpp',
             'GridCurrentMover_c.cpp',
             'GridWindMover_c.cpp',
             'IceMover_c.cpp',
             'IceWindMover_c.cpp',
             'CurrentCycleMover_c.cpp',
             'TimeGridVel_c.cpp',
             'TimeGridWind_c.cpp',
             'MakeTriangles.cpp',
             'MakeDagTree.cpp',
             'GridMap_c.cpp',
             'GridMapUtils.cpp',
             'RandomVertical_c.cpp',
             'RiseVelocity_c.cpp',
             'Weatherers_c.cpp',
             ]


cpp_code_dir = os.path.join('.', 'lib_gnome')
cpp_files = [os.path.join(cpp_code_dir, f) for f in cpp_files]


# setting the "pyGNOME" define so that conditional compilation
# in the cpp files is done right.
macros = [('pyGNOME', 1), ]

# Build the extension objects

# suppressing certain warnings
compile_args = [
    "-Wno-unused-function",  # unused function - cython creates a lot
    "-stdlib=libc++",  # to use the "correct" C++ libs
]

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

# JS NOTE: 'darwin' and 'win32' statically link against netcdf library.
#          On linux, we link against the dynamic netcdf libraries (shared
#          objects) since netcdf, hdf5 can be installed with a package manager.
#          We also don't have the static builds for these.
#          Also, the static_lib_files only need to be linked against
#          lib_gnome in the following Extension.
# CHB NOTE: one of these days, we need to figure out how to build against
#           conda netcdf...

if sys.platform == "darwin":
    # On the mac, the libgnome code is linked into the cy_basic_types
    #    extension -- which makes it available to all the other extensions
    #.   as long as it's imported first.
    # This is being done in the gnome/cy_gnome/__init__.py
    basic_types_ext = Extension('gnome.cy_gnome.cy_basic_types',
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
    # On windows, the
    # build our compile arguments
    macros.append(('_EXPORTS', 1))
    macros.append(('_CRT_SECURE_NO_WARNINGS', 1))

    compile_args = ['/EHsc']

    link_args.append('/MANIFEST')

    include_dirs.append(os.path.join(third_party_dir, 'win32_headers'))

    # build our linking arguments
    libdirs.append(netcdf_libs)

    basic_types_ext = Extension('gnome.cy_gnome.cy_basic_types',
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
    # fixme: Where is this getting built??
    #        That filename needs to be dynamically determined with code somewhat like:
    # if sys.version_info.major > 2:
    #     suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    #     libname = 'gnome' + suffix
    # else:
    #     libname = 'gnome.lib'

    #setuptools > 55 no longer puts built .lib in Release. These are now in lib_gnome
    static_lib_files = None
    if setuptools.__version__ > '55':
        static_lib_files = [os.path.join(target_path(),
                                        'lib_gnome',
                                        'cy_basic_types'+ win_comp_modules_ext)]
    else:
        static_lib_files = [os.path.join(target_path(),
                                     'Release', 'gnome', 'cy_gnome',
                                     'cy_basic_types'+ win_comp_modules_ext)]
    libdirs = []

elif sys.platform.startswith("linux"):
    # print("in linux stanza (line 416): include dirs")
    # print(include_dirs)
    # for some reason I have to create build/temp.linux-i686-2.7
    # else the compile fails saying temp.linux-i686-2.7 is not found
    # required for develop or install mode
    build_temp = target_path()
    if 'clean' not in sys.argv[1:] and not os.path.exists(build_temp):
        os.makedirs(build_temp)

    # Not sure calling setup twice is the way to go - but do this for now
    #    it should be straightforward to simply build a shared lib.
    #    but if it ain't broke ...
    # NOTE: This is also linking against the netcdf library (*.so), not
    # the static netcdf. We didn't build a NETCDF static library.
    setup(name='pyGnome',  # not required since ext defines this
          cmdclass={'build_ext': build_ext,
                    'cleanall': cleanall},
          ext_modules=[Extension('gnome.cy_gnome.libgnome',
                                 cpp_files,
                                 language='c++',
                                 define_macros=macros,
                                 libraries=['netcdf'],
                                 include_dirs=include_dirs,
                                 )])

    # In install mode, it compiles and builds libgnome inside
    # lib.linux-i686-2.7/gnome/cy_gnome
    # This should be moved to build/temp.linux-i686-2.7 so cython files
    # build and link properly
    # get the lib name -- py3 adds a bunch of platform cruft
    if sys.version_info.major > 2:
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
        libname = 'gnome' + suffix
    else:
        libname = 'gnome.so'

    if 'install' in sys.argv[1]:
        bdir = glob.glob(os.path.join('build/*/gnome/cy_gnome', "lib" + libname))
        if len(bdir) > 1:
            raise Exception("Found more than one libgnome library "
                            "during install mode in 'build/*/gnome/cy_gnome'")
        if len(bdir) == 0:
            raise Exception("Did not find libgnome library "
                            "during install mode in 'build/*/gnome/cy_gnome'")

        libpath = os.path.dirname(bdir[0])

    else:
        libpath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'gnome',
                               'cy_gnome')

    # Need this for finding lib during linking and at runtime
    # using -rpath to define runtime path. Use $ORIGIN to define libgnome.so
    # relative to cy_*.so
    os.environ['LDFLAGS'] = "-L{0} -Wl,-rpath='$ORIGIN'".format(libpath)

    # End building C++ shared object
    compile_args = ["-Wl,-rpath,'$ORIGIN'"]

    libname = libname[:-3] if libname.endswith(".so") else libname

    lib = [libname]
    basic_types_ext = Extension('gnome.cy_gnome.cy_basic_types',
                                ['gnome/cy_gnome/cy_basic_types.pyx'],
                                language='c++',
                                define_macros=macros,
                                extra_compile_args=compile_args,
                                libraries=lib,
                                include_dirs=include_dirs,
                                )

    extensions.append(basic_types_ext)
    static_lib_files = []

#
# All other lib_gnome-based cython extensions.
# These depend on the successful build of cy_basic_types
#
for mod_name in extension_names:
    cy_file = os.path.join("gnome/cy_gnome", mod_name + ".pyx")
    print("setting up cython extensions")
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

# and platform-independent cython extensions:
# well...not entirely platform-independent.  We need to pass the link_args
poly_cypath = os.path.join('gnome', 'utilities', 'geometry')
sources = [os.path.join(poly_cypath, 'cy_point_in_polygon.pyx'),
           os.path.join(poly_cypath, 'c_point_in_polygon.c')]

include_dirs = [np.get_include(), '../lib_gnome']
if sys.platform == "win32":
    include_dirs.append(os.path.join(third_party_dir, 'win32_headers'))

extensions.append(Extension("gnome.utilities.geometry.cy_point_in_polygon",
                            sources=sources,
                            include_dirs=include_dirs,
                            extra_compile_args=compile_args,
                            extra_link_args=link_args,
                            ))

if sys.version_info.major == 2:
    # this doesn't work under Python3
    extensions.append(Extension("gnome.utilities.file_tools.filescanner",
                                sources=[os.path.join('gnome',
                                                      'utilities',
                                                      'file_tools',
                                                      'filescanner.pyx')],
                                extra_compile_args=compile_args,
                                include_dirs=include_dirs,
                                language="c",
                                ))

for e in extensions:
    e.cython_directives = {'language_level': "3"}  # all are Python-3

setup(name='pyGnome',
      version=get_version(),
      ext_modules=extensions,
      packages=find_packages(),
      package_dir={'gnome': 'gnome'},
      package_data={'gnome': ['data/yeardata/*',
                              'outputters/sample.b64',
                              'weatherers/platforms.json',
                              'outputters/erma_data_package/*',
                              ]},
      # you are not going to be able to "pip install" this anyway
      # -- no need for requirements
      requires=[],   # want other packages here?
      cmdclass={'build_ext': build_ext,
                'cleanall': cleanall},
      url="https://github.com/NOAA-ORR-ERD/PyGnome"
      )

# Change current working directory back to what user originally had
os.chdir(CWD)
