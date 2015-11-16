
# Building / Installing GNOME with the Anaconda python distribution #

[Anaconda](https://store.continuum.io/cshop/anaconda/) is a Python distribution
that has most of the difficult-to-build packages that py_gnome needs already
built in. Thus it's a nice target for running GNOME on your own system.

## Windows:

You want the Windows 64 bit Python 2.7 version
Installing with the defaults works fine.
You probably want to let it set the PATH for you -- that's a pain to do by hand.

## OS-X:

Anaconda provides a 64 bit version -- this should work well with py_gnome.
Either the graphical installer or the command line one is fine --
use the graphical one if you are not all that comfortable with the *nix
command line.

## Linux:

The Linux 64bit-python2.7 is the one to use.

*NOTE: using Anaconda on Linux is NOT tested...by us anyway.  And may not work precisely according to these install instructions.*

## conda

[conda](http://conda.pydata.org/docs/intro.html) is the package manager that Anaconda is built on. So when working with Anaconda, you will want to use the conda package manager for installing conda packages. Pip still works, but is not preferred.

## Setting up

Install: [Anaconda](https://www.continuum.io/downloads)

or alternatively: [Miniconda](http://conda.pydata.org/miniconda.html)

Anaconda will give you a wide variety of extra stuff, including development tools, that are very useful for scientific computing. It is a big install, though.

Miniconda is a minimal anaconda installation (basically only Python and conda).
This is a smaller and faster install, but then you will need to install
the packages you need. If you are only planning on using this for py_gnome, miniconda is fine.

Once you have either Anaconda or Miniconda installed, the rest of the instructions should be the same.

## Update your (new) system

Once you have Anaconda or miniconda installed, you should start by getting everything up to date, sometimes pacakges have been updated since the installer was built::

```
> conda update anaconda
```

or if you have Miniconda...

```
> conda update conda
```



## Setting up anaconda.org

[anaconda.org](http://anaconda.org) is a web service where people can host conda packages
for download. The way this is done is through anaconda  "channels", which can be thought of
simply as places on anaconda.org where collections of packages are bundled together
by the people hosting them.

Many of the dependencies that py_gnome requires come out of the box with
Anaconda, but a few don't.

So we have set up [our own anaconda channel](https://anaconda.org/noaa-orr-erd)
where we put various packages needed for py_gnome. But you don't need to access the web site to use it...conda can find everything you need:

To install the anaconda client:

```
> conda install anaconda
```

and to add the NOAA-ORR-ERD binstar channel to Anaconda:

```
> conda config --add channels https://conda.anaconda.org/NOAA-ORR-ERD
```

Now conda will know to go look in our anaconda channel for the packages you need.

## Download GNOME

At this point you will need some files from the PyGnome sources.
If you have not downloaded it yet, it is available here:

[https://github.com/NOAA-ORR-ERD/PyGnome](https://github.com/NOAA-ORR-ERD/PyGnome)

## Dependencies

The Anaconda dependencies for PyGnome are listed in the file
`conda_packages.txt` in the top directory of the project.

To install all the packages pygnome needs:

```
> cd pygnome  # or wherever you put the PyGnome project
> conda install --file conda_packages.txt
```

To get the whole setup, this file has a full dump of a conda environment
with all the dependencies.

conda support "environments" -- walled of setups for a compete set of packages of a poaorticualr version. If you are using python and conda for multipiel projet that may require different versions of various pacakges, you can set up a conda environment for py_gnome with:

```
> conda create -n py_gnome --file conda_packages.txt`
```

Note that if you are using python primarily for py_gnome, you don't need to
create a separate environment.


## Compilers

To build py_gnome, you will need a C/C++ compiler.  The procedure for getting
compile tools varies with the platform you are on.

### OS-X

The system compiler for OS-X is XCode. It can be installed from the App Store.

_Note: it is a HUGE download._

After installing XCode, you still need to install the "Command Line Tools".
Xcode includes a new "Downloads" preference pane to install optional components
such as command line tools, and previous iOS Simulators.

To install the XCode command line tools:
 - Start Xcode from the launchpad
 - Click the "Xcode" dropdown menu button in the top left of the screen near the Apple logo
 - Click "Preferences", then click "Downloads".
 - Command Line Tools should be one of the downloadable items, and there should be an install button for that item.  Click to install.

Once the command line tools are installed, you should be able to build py_gnome as above.


### Windows

For compiling python extensions on Windows using Anaconda, you can use MS Visual Studio 2008 if it is available to you.
But Microsoft isn't really supporting that version anymore, so it is probably best to use
[Microsoft Visual C++ Compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266),
which is freely downloadable.

_Note: if you are building on windows, the python package setuptools needs to be at version 6 or higher to properly query the compiler environment._ -- a recent conda install will have this version.


### Linux

Linux uses the GNU gcc compiler. If it is not already installed on your system, use your system package manager to get it.

 - apt for Ubuntu and Linux Mint
 - rpm for Red Hat
 - dpkg for Debian
 - yum for CentOS
 - ??? for other distros

## Building py_gnome

Ok, at this point we should at last have all the necessary third-party environments in place.

Right now, it is probably best to build py_gnome from source. And it is probably best to build a "develop" target for your py_gnome package if you plan on developing or debugging the py_gnome source code.

Building the "develop" target allows changes in the package python code (or source control updates), to take place immediately.

Of course if you plan on simply using the package, you may certainly build with the "install" target.  Just keep in mind that any updates to the project will need to be rebuilt and re-installed in order for changes to take effect.


### OS-X Note:

Anaconda does some strange things with system libraries and linking on OS-X,
so we have a high level script that will build and re-link the libs for you.

So to build py_gnome on OS-X:

```
> cd py_gnome
> ./build_anaconda.sh
```

### Other platforms

As far as I know, the same linking issues don't exist, so you can build directly.
There are a number of options for building:

```
> python setup.py developall
```

builds everything; both gnome and the oil_library modules. There are also these options:

```
> python setup.py develop
```

>builds and installs just the gnome module development target

```
> python setup.py cleandev
```

cleans files generated by the build as well as files auto-generated by cython. It is a good idea t run `cleandev` after updating from the gitHub repo -- particularly if strange errors are occuring.

A sub-module of py_gnome that is also built with these steps is the oil_library.
When initially built using developall, it builds a database of oil properties for py_gnome (and others) to use.

If the oil database schema has been updated either manually or through a `git pull` you will need to re-build the oil database with this command:

```
> python setup.py remake_oil_db
```

## Testing py_gnome

We have an extensive set of unit and functional tests to make sure that py_gnome is working properly.

To run the tests:

```
> cd PyGnome/py_gnome/tests/unit_tests
> py.test
```

and if those pass, you can run:

```
> py.test --runslow
```

which will run some more tests, some of which take a while to run.

Note that the tests will try to auto-download some data files.  If you are not on the internet, this will fail.  And of course if you have a slow connection, these files could take a while to download. Once the tests are run once, the downloaded files are cached for future test runs.






