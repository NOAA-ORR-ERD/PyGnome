CPP_CODE_DIR = "../lib_gnome"
import numpy as np
import os
import sys
from sysconfig import get_config_var

from setuptools import setup
#from Cython.Distutils import build_ext 
#from distutils.core import setup
from distutils.extension import Extension

files = ['MemUtils.cpp', 'Mover_c.cpp', 'Replacements.cpp']
files += ['CMapLayer_c.cpp', 'ClassID_c.cpp', 'ComponentMover_c.cpp']
files += ['OUTILS.cpp', 'TimeValuesIO.cpp',]
files += ['NetCDFMover_c.cpp', 'RectGridVel_c.cpp', 'Model_c.cpp',]
files += ['PtCurMover_c.cpp', 'CompoundMap_c.cpp', 'NetCDFMoverCurv_c.cpp',]
files += ['TriGridVel3D_c.cpp', 'TideCurCycleMover_c.cpp', 'MakeTriangles.cpp',]
files += ['NetCDFMoverTri_c.cpp', 'CATSMover3D_c.cpp', 'MakeDagTree.cpp']
files += ['CompoundMover_c.cpp', 'TriCurMover_c.cpp', 'Random3D_c.cpp',]
files += ['PtCurMap_c.cpp', 'LEList_c.cpp', 'OLEList_c.cpp',]
files += ['OSSMWeatherer_c.cpp', 'Weatherer_c.cpp', 'MYRANDOM.cpp',]
files += ['Map_c.cpp', 'CATSMover_c.cpp', 'GEOMETRY.cpp']
files += ['ShioCurrent1.cpp', 'ShioCurrent2.cpp', 'ShioHeight.cpp',]
files += ['OSSMTimeValue_c.cpp', 'TimeValue_c.cpp', 'VectMap_c.cpp']
files += ['DagTreeIO.cpp', 'RectUtils.cpp', 'ShioTimeValue_c.cpp']
files += ['Random_c.cpp', 'WindMover_c.cpp', 'CurrentMover_c.cpp']
files += ['CompFunctions.cpp', 'CMYLIST.cpp', 'GEOMETR2.cpp']
files += ['TriGridVel_c.cpp', 'DagTree.cpp', 'StringFunctions.cpp']

temp_list = ['cyGNOME/c_gnome.pyx']
for file in files:
    temp_list.append(os.path.join(CPP_CODE_DIR ,file))
files = temp_list

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
setup(name='python gnome',
      version='beta', 
      requires=['numpy'],
      packages=['gnome','gnome.utilities',],
      ext_modules=[Extension('gnome.c_gnome',
                             files, 
                             language="c++",
			     define_macros = macros,
                             extra_compile_args=compile_args,
			     extra_link_args=link_args,
			     include_dirs=[CPP_CODE_DIR,
                                           np.get_include(),
                                           'cyGNOME',
                                           extra_includes,
                                           ],
                             )
                   ]


     )

