*************************************
Installation without conda / Anaconda
*************************************

**WARNING:** These notes are out of date and not tested. All our development work is done using conda and conda-forge -- there are a few too many dependencies not well supported by "pure" pip / PyPi packages at this point.

Note that current PyGNOME is Python 3 (>= 3.8) -- some of these notes are for Python2. But if you do get this working, please consider supplying a Pull Request with updated notes.

Overview:
=========

PyGNOME has a lot of dependencies -- you can find the full list by looking at the ``conda_requirements_*.txt`` file.

Most of them are well maintained 3rd party packages, that can be installed via pip, or by following the package author's instructions.

But a few are maintained by the GNOME team at NOAA. These are all best installed by downloading the source code from the NOAA-ORR-ERD gitHub organization, and building and installing them from that source. These are:

pynucos: https://github.com/NOAA-ORR-ERD/PyNUCOS

gridded: https://github.com/NOAA-ORR-ERD/gridded

cell_tree2d (required by gridded): https://github.com/NOAA-ORR-ERD/cell_tree2d

py_gd: https://github.com/NOAA-ORR-ERD/py_gd


Building everything by hand / with pip
======================================

Since this is development work, it might be good to create and run this
in a virtual environment of some sort: virtualenv, pipenv, etc.

A virtual environment is **not required** to run PyGnome.
But you may be working in an environment (on a corporate network, for example) that restricts your access to the system files on your computer.
In such a case, you may require a virtualenv in order to freely install python packages in python's site-packages dir. (site-packages is the standard place where python installers will put packages after building them)

You may also want to consider using conda environments -- see above.

There is C++/Cython code that must be built - you will need the correct C/C++ compiler and recent setuptools, etc. See "Installing With Anaconda" for more detail (or online for docs on "building C extensions to Python")

Python
------

Most people use Python itself from Python.org:

https://www.python.org/downloads/


Linux (Tested in 64-bit, CentOS)
--------------------------------

For Linux use appropriate package manager (yum on CentOS, apt-get on Ubuntu) to
download/install binary dependencies.


Binary Dependencies
...................

1. setuptools is required.
    ``> sudo apt-get install python-setuptools``
    \` \`

2. To compile Python extensions, you need the development libs for Python:

    > sudo apt-get install python-dev

3. netCDF4 python module requires NetCDF libraries:

   libhdf5-serial-dev

   libnetcdf-dev

4. The following python packages, documented in PyGnome's
   requirements.txt, may need to be manually installed.

Binaries for

`Numpy <http://packages.ubuntu.com/raring/python/python-numpy>`__ and

`Cython <http://packages.ubuntu.com/raring/python/cython>`__
can be installed using apt-get.

Current binaries for these are sufficiently new: (Numpy >=1.16) and (Cython >= 0.29).

If you use virtualenv and apt-get to install these system site packages.
Remember to use the ``--system-site-packages`` option when creating a
new virtualenv so the system packages are available in the virtualenv.

Alternatively, pip install should also work. The following builds the
latest packages in your virtualenv once the above dependencies are met::

    > pip install numpy
    > pip install cython
    > pip install netCDF4

The remaining dependencies are python packages and can be installed using::

   pip install -r requirements.txt

(NOTE: we do not regularly test the requirements.txt file -- it may be incomplete -- PR's accepted)

See `Build PyGnome <#build-pygnome>`__ section below.


Windows
-------

For compiling python extensions on Windows with python3 it is best to use the Microsoft the Visual Studio 2019 (or later) Build Tools.
They should be available here:

https://visualstudio.microsoft.com/downloads/

The free "Community" version should be fine.

Once installed, you will want to use one of the  "Visual Studio Developer Command Prompts" provided to actually build PyGNOME -- it sets up the compiler for you.

Only 64 bit Windows is supported by PyGNOME

Binary Dependencies
...................

Download and install the newest Windows executable distribution of

`Python 3.9 <http://www.python.org/download/>`_

A number of the packages that GNOME depends on have very complex and
brittle build processes, and depend on third-party libraries that can be
a challenge to build.

Fortunately, many of them are available as binary wheels on PyPi

Another option is to use a Python scientific distribution, such as
`Anaconda <https://www.anaconda.com/products/distribution>`_ -- in which case, stop and read the ``Installing.rst`` file.


Dependencies can be installed using the command::

    > pip install -r requirements.txt

See `Build PyGnome <#build-pygnome>`__ section below.


Build PyGnome
=============

1. Clone the PyGnome repository::

    > git clone https://github.com/NOAA-ORR-ERD/PyGnome.git

2. pip install all of GNOME's python package dependencies::

    > cd PyGnome/py_gnome
    > pip install -r requirements.txt

3. Install the adios_db pacakge -- it is under active development along  with py_gnome, so it's best to install that from source as well:

   https://github.com/NOAA-ORR-ERD/adios_oil_database/tree/production/adios_db

4. build the ``py_gnome`` module in develop or install mode:

   ``> python setup.py build``

   or

   ``> python setup.py install``

   or

   ``> python setup.py develop``

The other option you may need is ``cleanall``, which should clean the development environment -- good to do after puling new code from git::

   > python setup.py cleanall


5. If this successfully completes, then run the unit tests::

    > pytest --runslow tests/unit_tests

Once all of the ``py_gnome`` unit tests pass, PyGnome is now built and
ready to be put to use. You can use the ``gnome`` module inside your
python scripts to set up a variety of modeling scenarios.

There are example full scripts in the ``py_gnome/scripts`` directory.

Documentation of PyGNOME can be found at:

https://gnome.orr.noaa.gov/doc/pygnome/index.html



