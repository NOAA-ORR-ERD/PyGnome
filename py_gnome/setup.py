CPP_CODE_DIR = "cygnome/codeFiles/"
import numpy as np
import os 
from sysconfig import get_config_var

from setuptools import setup
from Cython.Distutils import build_ext 
#from distutils.core import setup
from distutils.extension import Extension

files = ['MemUtils/MemUtils.cpp', 'Mover/Mover_c.cpp']
files += ['Random/Random_c.cpp', 'WindMover/WindMover_c.cpp']
files += ['CompFunctions.cpp', 'CMyList/CMYLIST.cpp']
files += ['OSSMTimeValue/OSSMTimeValue_c.cpp', 'TimeValue/TimeValue_c.cpp']
files += ['GEOMETRY.cpp']

temp_list = ['cyGNOME/c_gnome.cpp']
for file in files:
    temp_list.append(os.path.join(CPP_CODE_DIR ,file))
files = temp_list

compile_args=["-I.", "-fpascal-strings", "-fasm-blocks"]

if get_config_var('UNIVERSALSDK') != None:
    pass
else:
    print 'UNIVERSALSDK not set. aborting.'
    exit(-1)

setup(name='python gnome',
      version='beta', 
      requires=['numpy'],
      cmdclass={'build_ext': build_ext },
      packages=['gnome',],
      ext_modules=[Extension('gnome.c_gnome',
                             files, 
                             language="c++",
                             include_dirs=[CPP_CODE_DIR ,
                                           np.get_include(),
                                           'cyGNOME',
                                           get_config_var('UNIVERSALSDK')+'/Developer/Headers/FlatCarbon',
                                           ],
                             )
                   ]


     )

