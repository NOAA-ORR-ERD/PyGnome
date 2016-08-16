Installation in Development Mode
================================

Since this is development work, it might be good to create and run this
in a virtual environment. `Virtualenv <http://www.virtualenv.org/en/latest/>`__ and `Virtual envwrapper <http://virtualenvwrapper.readthedocs.org/en/latest/>`__ eases
management of virtual environments.

A virtualenv is **not required** to run PyGnome. But you may be working
in an environment (on a corporate network, for example) that restricts
your access to the system files on your computer. In such a case, you
may require a virtualenv in order to freely install python packages in
python's site-packages area. (site-packages is the standard place where
python installers will put packages after building them)

You may also want to consider using conda environments.

There is C++/Cython code that must be built - you will need the corect C compiler and recent setuptools, etc.

python.org
==========

The following has been tested against `Python
2.7 <https://www.python.org/downloads/>`__

Linux (Tested in 32-bit, Ubuntu raring 13.04)
---------------------------------------------

For Linux use appropriate package manager (apt-get on ubuntu) to
download/install binary dependencies.

Binary Dependencies
...................

1. setuptools is required.
    ``> sudo apt-get install python-setuptools``
    \` \`
2. `Pillow <http://pillow.readthedocs.org/en/latest/installation.html>`__
   has binary dependencies. Visit the docs to get list of dependencies
   for your system. Pillow requires Python's development libraries::

    > sudo apt-get install python-dev

This did not build symlinks to libraries for me in /usr/lib, so had to manually create them::

    > sudo ln -s /usr/lib/``\ uname
   -i``-linux-gnu/libfreetype.so /usr/lib/``      ``> sudo ln -s /usr/lib/``\ uname
   -i``-linux-gnu/libjpeg.so /usr/lib/``      ``> sudo ln -s /usr/lib/``\ uname
   -i``-linux-gnu/libz.so /usr/lib/```` \`

3. netCDF4 python module requires NetCDF libraries: libhdf5-serial-dev,
   libnetcdf-dev

4. The following python packages, documented in PyGnome's
   requirements.txt, may need to be manually installed.

Binaries for

`Numpy <http://packages.ubuntu.com/raring/python/python-numpy>`__ and
`Cython <http://packages.ubuntu.com/raring/python/cython>`__
can be installed using apt-get.

Current binaries for these are sufficiently new: (Numpy >=1.11.0) and (Cython >= 0.24.1).

If you use virtualenv and apt-get to install these system site packages.
Remember to use the ``--system-site-packages`` option when creating a
new virtualenv so the system packages are available in the virtualenv.

Alternatively, pip install should also work. The following builds the
latest packages in your virtualenv once the above dependencies are met::

    > pip install numpy
    > pip install cython
    > pip install netCDF4
    > pip install Pillow

The remaining dependencies are python packages and can be installed using::

   pip install -r requirements.txt

See `Build PyGnome <#build-pygnome>`__ section below.


Windows 7 (64-bit, using VS-2008 express edition)
-------------------------------------------------

For compiling python extensions on Windows, you need the correct version of teh MS compiler:  "Visual Studio 2008". Microsoft has made a versin of this compiler al properly set up for python extensions:

Microsoft Visual C++ Compiler for Python 2.7:

https://www.microsoft.com/en-us/download/details.aspx?id=44266

This compiler should work for both 32 bit and 64 bit Windows.

Binary Dependencies
...................

Download and install the newest Windows executable distribution of
`Python 2.7 <http://www.python.org/download/>`__ (*note: we are not
quite ready for Python 3.0*) Make sure the distribution is named
consistently with the Python environment you desire. For example,
binaries ending in *win64-py2.7.exe are for Python 2.7.* (64-bit)

A number of the packages that GNOME depends on have very complex and
brittle build processes, and depend on third-party libraries that can be
a challenge to build.

Fortunately, `Chris Gohlke's
website <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`__ contains
pre-compiled binary distributions for many of these packages.

*(The full list of dependent packages and their minimum versions can be
found* \_in the file GNOME2/py\_\_\ *gnome/requirements.txt)*

There are also more binary wheels available every day -- it's worth checking PyPi

Another option is to use a Python scientific distribution, such as
`Anaconda <https://store.continuum.io/cshop/anaconda/>`__ or `Enthought
Canopy <https://www.enthought.com/products/canopy/>`__

Here are the binary packages required:

1. `setuptools <http://www.lfd.uci.edu/~gohlke/pythonlibs/#setuptools>`__
2. `pip <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pip>`__

At this point, we should test that pip is installed correctly.
On command line invoke the following pip commands.
These should show usage information for 'pip', and then a list of
installed packages::

    >  pip

    Usage:   
      pip <command> [options]

    Commands:
      install                     Install packages.
      download                    Download packages.
      uninstall                   Uninstall packages.

    > pip list
    alabaster (0.7.9)
    appnope (0.1.0)
    awesome-slugify (1.6.5)
    ...

3. `numpy-MKL <http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy>`__
4. `Cython <http://www.lfd.uci.edu/~gohlke/pythonlibs/#cython>`__
5. `Pillow <https://pypi.python.org/pypi/Pillow/2.8.1>`__
6. 64-bit 1.0.6 version of
   `netCDF4 <http://www.lfd.uci.edu/~gohlke/pythonlibs/#netcdf4>`__
7. `lxml <http://www.lfd.uci.edu/~gohlke/pythonlibs/#lxml>`__ - required
   for webgnome
8. `python-cjson <http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-cjson>`__
   - required for webgnome

The remaining dependencies are python packages and can be installed using the command::

    > pip install -r requirements.txt

See `Build PyGnome <#build-pygnome>`__ section below.

Windows 7 (64-bit, using VS-2008 express edition)
.................................................

Building GNOME for 64 bit Windows is similar to the 32 bit Windows
build, and has similar binary dependencies. There are however some extra
steps you need to perform in order to build py\_gnome.


Build PyGnome
-------------

1. Clone the PyGnome repository::

    > git clone https://github.com/NOAA-ORR-ERD/PyGnome.git

2. pip install all of GNOME's python package dependencies::

    > cd PyGnome/py_gnome
    > pip install -r requirements.txt

3. Install the Oil Library package. The OilLibary package is under active development along  with py_gnome, so it's best to install that from source as well:

   https://github.com/NOAA-ORR-ERD/OilLibrary

4. build the ``py_gnome`` module in develop mode first as install mode may
   still need some testing/work.
    
   The other option you may need is ``cleanall``, which should clean the development environment -- good to do after puling new code from git.

5. If this successfully completes, then run the unit tests::

    > py.test --runslow tests/unit_tests

Once all of the ``py_gnome`` unit tests pass, PyGnome is now built and
ready to be put to use. You can use the ``gnome`` module inside your
python scripts to set up a variety of modeling scenarios.

There are example full scripts in the ``py_gnome/scripts`` directory.
