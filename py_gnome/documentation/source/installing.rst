##########
Installing
##########

As PyGNOME is under active development, it is usually best to install it from the source code. This also provides access to the code for examples, and to examine it in order to better understand it and potentially customize it.

The source can be found on gitHub here:

https://github.com/NOAA-ORR-ERD/PyGnome

You can either "clone" the repository with git (recommended), or download a zip file of the source code.

Installing from Source:
#######################

PyGNOME consists of compiled C++ code (``libgnome``), compiled Cython code (``*.pyx`` files), and compiled Python extensions.

Currently, the only option is to install from the source code, which requires an appropriate compiler. At this time, NOAA is not maintaining pre-built binaries.

PyGNOME depends on a number of third party packages -- the complete list can be found in the various ``conda_requirements.txt`` files.

There are many dependencies that can be hard to build, so the easiest way is to use the conda package manager, but you can do it by hand as well -- see below.


.. include:: ../../../Installing.rst

.. include:: ../../../Install_without_conda.rst
