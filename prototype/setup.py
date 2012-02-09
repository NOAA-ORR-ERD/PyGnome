""" """
import numpy as np
from os import getcwd
from sysconfig import get_config_var
from Cython.Distutils import build_ext 
from distutils.core import setup
from distutils.extension import Extension

files = ['MemUtils/MemUtils.cpp', 'Mover/Mover_c.cpp']
files += ['Random/Random_c.cpp', 'WindMover/WindMover_c.cpp']
files += ['CompFunctions.cpp', 'CMyList/CMYLIST.cpp']
files += ['OSSMTimeValue/OSSMTimeValue_c.cpp', 'TimeValue/TimeValue_c.cpp']
files += ['GEOMETRY.cpp']

temp_list = ['cyGNOME/c_gnome.pyx']
for file in files:
    temp_list += ["cyGNOME/codeFiles/"+file]
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
      packages=['pyGNOME',],
      ext_modules=[Extension('c_gnome',
				files, 
				language="c++",
                             	include_dirs=["cyGNOME/codeFiles",
                                           	np.get_include(),
                                           	'cyGNOME',
                                           	get_config_var('UNIVERSALSDK')+'/Developer/Headers/FlatCarbon',
                                           ],
                             )
		]


     )

