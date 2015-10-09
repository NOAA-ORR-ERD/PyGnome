# Installation in Development Mode #

Since this is development work, it might be good to create and run this in a virtual environment.
[Virtual env](http://www.virtualenv.org/en/latest/) and 
[Virtual env wrapper](http://virtualenvwrapper.readthedocs.org/en/latest/) eases management of virtual environments.

A virtualenv is **not required** to run PyGnome.
But you may be working in an environment (on a corporate network, for example)
that restricts your access to the system files on your computer.  In such a
case, you may require a virtualenv in order to freely install python packages
in python's site-packages area. (site-packages is the standard place where
python installers will put packages after building them)

There is C++/Cython code that must be built - **setuptools must be >= 2.1.**  

## python.org ##

The following has been tested against
[Python 2.7](https://www.python.org/downloads/)

## Linux (Tested in 32-bit, Ubuntu raring 13.04) ##

For Linux use appropriate package manager (apt-get on ubuntu)
to download/install binary dependencies.


### Binary Dependencies ###
1. setuptools and distutils are required. distutils should come installed with Python.  
   `> sudo apt-get install python-setuptools`  
   ` `
2. [Pillow](http://pillow.readthedocs.org/en/latest/installation.html) has binary dependencies.
   Visit the docs to get list of dependencies for your system.
   Pillow requires Python's development libraries  
   `> sudo apt-get install python-dev`  
   ` `
   This did not build symlinks to libraries for me in /usr/lib, so had to
   manually create them:  
   ``> sudo ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib/``  
   ``> sudo ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib/``  
   ``> sudo ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib/``  
   ` `
3. netCDF4 python module requires NetCDF libraries:
   libhdf5-serial-dev, libnetcdf-dev
4. The following python packages, documented in PyGnome's requirements.txt,
   may need to be manually installed.  
   Binaries for [Numpy](http://packages.ubuntu.com/raring/python/python-numpy)
   and [Cython](http://packages.ubuntu.com/raring/python/cython)
   can be installed using apt-get.  
   Current binaries for these are sufficiently new:
   (Numpy >=1.7.0) and (Cython >= 0.17.1).  
   
If you use virtualenv and apt-get to install these system site packages.
Remember to use the `--system-site-packages` option when creating a new
virtualenv so the system packages are available in the virtualenv.

Alternatively, pip install should also work. 
The following builds the latest packages in your virtualenv once the above dependencies are met.

> `> pip install numpy`  
> `> pip install cython`  
> `> pip install netCDF4`  
> `> pip install Pillow`  

The remaining dependencies are python packages and can be installed using pip install -r requirements.txt  
See [Build PyGnome](#build-pygnome) section below.

### Windows 7 (32-bit, using VS-2008 express edition) ###

For compiling python extensions on Windows, you can use
MS Visual Studio 2008 if it is available to you.
But Microsoft isn't really supporting that version anymore,
so it is probably best to use 
[Microsoft Visual C++ Compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266),
which is freely downloadable.

This compiler should work for both 32 bit and 64 bit Windows.

#### Binary Dependencies ####

Download and install the newest Windows executable distribution of [Python 2.7](http://www.python.org/download/)
(*note: we are not quite ready for Python 3.0*)
Make sure the distribution is named consistently with the Python environment you desire.
For example, binaries ending in *win32-py2.7.exe are for Python 2.7.* (32-bit)

A number of the packages that GNOME depends on have very complex and brittle build processes, and depend on third-party libraries that can be a challenge to build.

Fortunately, [Chris Gohlke's website](http://www.lfd.uci.edu/~gohlke/pythonlibs/) contains pre-compiled binary distributions for many of these packages.  

_(The full list of dependent packages and their minimum versions can be found_
_in the file GNOME2/py___gnome/requirements.txt)_

Another option is to use a Python scientific distribution, such as
[Anaconda](https://store.continuum.io/cshop/anaconda/)
or [Enthought Canopy](https://www.enthought.com/products/canopy/)



Here are the binary packages required:

1. [setuptools](http://www.lfd.uci.edu/~gohlke/pythonlibs/#setuptools)
2. [pip](http://www.lfd.uci.edu/~gohlke/pythonlibs/#pip)  
   At this point, we should test that pip is installed correctly.  
   On command line invoke the following pip commands.  
   These should show usage information for 'pip', and then a list of installed packages.  
   ` > pip`  
   ` Usage:`  
   `   pip <command> [options]`  
   ` `  
   `    Commands:`  
   `      install                     Install packages.`  
   `      ...`  
   ` `  
   `    > pip list`  
   `    pip (1.4.1)`  
   `    setuptools (1.1.4)`  
   ` `
3. [numpy-MKL](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
4. [Cython](http://www.lfd.uci.edu/~gohlke/pythonlibs/#cython)
5. [Pillow](https://pypi.python.org/pypi/Pillow/2.8.1)
6. 32-bit 1.0.6 version of [netCDF4](http://www.lfd.uci.edu/~gohlke/pythonlibs/#netcdf4)
7. [lxml](http://www.lfd.uci.edu/~gohlke/pythonlibs/#lxml) - required for webgnome
8. [python-cjson](http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-cjson) - required for webgnome

The remaining dependencies are python packages and can be installed using the command:  
>`pip install -r requirements.txt`

See [Build PyGnome](#build-pygnome) section below.

### Windows 7 (64-bit, using VS-2008 express edition) ###

Building GNOME for 64 bit Windows is similar to the 32 bit Windows build, and has similar
binary dependencies.
There are however some extra steps you need to perform in order to build py_gnome.

### Build PyGnome ###

1. Clone the PyGnome repository.  
   `> git clone https://github.com/NOAA-ORR-ERD/PyGnome.git`
   ` `  
2. pip install all of GNOME's python package dependencies.  
   `> cd PyGnome/py_gnome`  
   `> pip install -r requirements.txt`  
   ` `  
3. build the py_gnome module in develop mode first as install mode may
   still need some testing/work.  
   _Note: using 'developall' argument will automatically build the_
   _oil___library in develop mode._  
   This is required for PyGnome and is currently part of this repo so easiest
   to automatically build it.  
   Other options are to clean the development environment (cleandev) and to
   rebuild the oil library database (remake_oil_db).
   NOTE: 'developall' is not recognized on Linux (only Mac & Windows).
   If building on Linux, then do python setup.py develop for
   py_gnome/. and /oil_library/. separately.
   `> python setup.py developall`  
   ` `  
4. If this successfully completes, then run unit_tests
   `> py.test --runslow tests/unit_tests/`  

Once all of the py_gnome unit tests pass, PyGnome is now built and ready to be put to use.
You can use the `gnome` module inside your python scripts to set up a variety of modelling
scenarios.

There are example full scripts in the ``py_gnome/scripts`` directory.

