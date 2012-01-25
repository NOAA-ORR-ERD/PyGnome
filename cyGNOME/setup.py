from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os, fnmatch
 
import numpy as np
 
#for root, dirs, _files in os.walk("codeFiles/"):
#	for file in _files:
#		if fnmatch.fnmatch(file, "*.cpp") or fnmatch.fnmatch(file, "*.CPP"):
#			files += [os.path.join(root, file.lower())]

files = ['MemUtils/MemUtils.cpp', 'Mover/Mover_c.cpp']
files += ['Map/Map_c.cpp', 'GEOMETRY.cpp']
files += ['Random/Random_c.cpp', 'WindMover/WindMover_c.cpp']
files += ['CompFunctions.cpp', 'CMyList/CMYLIST.cpp']
files += ['OSSMTimeValue/OSSMTimeValue_c.cpp', 'TimeValue/TimeValue_c.cpp']

tempList = ['cyGNOME.pyx']
for file in files:
	tempList += ["codeFiles/"+file]
files = tempList

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("cyGNOME",
                             files,
                             language="c++",
                             extra_compile_args=["-I.",
                                                 #"-idirafter /Users/alex.hadjilambris/Desktop/tCython/tLEs",
                                                 "-fpascal-strings", "-fasm-blocks"],
                             include_dirs=["codeFiles",
                                           np.get_include(),
                                           ".",
                                           "/Developer/SDKs/MacOSX10.4u.sdk/Developer/Headers/FlatCarbon",
                                           #"/Users/alex.hadjilambris/Desktop/class\ separation/GNOMEUniversal",
                                           ],
                             )
                   ]
#    "/usr/local/include/", "/Developer/Headers/FlatCarbon/"])]
    )

