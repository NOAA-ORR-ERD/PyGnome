Building / Installing GNOME with the conda / Anaconda python distribution
=========================================================================

`Anaconda <https://store.continuum.io/cshop/anaconda/>`__ is a Python
distribution that has most of the difficult-to-build packages that
``py_gnome`` needs already built in. Thus it's a nice target for running
GNOME on your own system. "conda" is the pacakg manager used to manage the system.

py_gnome CAN be used with any Python distribution, but you will need to find or build a number of pacakges that are not easy to manage. If you are familiar with complex python packaging, then you can probably make it work. But conda makes it much easier, and that's what we use ourselves, and support.

Anaconda vs miniconda:
----------------------

`Anaconda <https://store.continuum.io/cshop/anaconda/>`__ provides a fairly complete python system for computational programming -- it is a large install, but comes with a lot of nice stuff pre-packaged that all works together.

`miniconda <http://conda.pydata.org/miniconda.html>`__ is a much smaller install -- it provides only Python and the conda package mangement system. You can install miniconda, and then install only the packges you need to run ``py_gnome``.

Either will work fine with ``py_gnome``.

**NOTES:**

Be sure to get the python2 version of Anaconda. py_gnoe is currently only python 2 compatible.

Anaconda (and miniconda?) can be installed in either single-user or multi-user mode:

https://docs.continuum.io/anaconda/install

We (and Continuum) recommend single-user mode -- that way, administrator privileges are not required for either initial installation or maintaining the system.

Windows:
........

You want the Windows 64 bit Python 2.7 version. Installing with the
defaults works fine. You probably want to let it set the PATH for you --
that's a pain to do by hand.


OS-X:
.....

Anaconda provides a 64 bit version -- this should work well with
``py_gnome``. Either the graphical installer or the command line one is
fine -- use the graphical one if you are not all that comfortable with
the \*nix command line.

Linux:
......

The Linux 64bit-python2.7 is the one to use.

We do not support 32 bit on any platform.

conda
-----

`conda <http://conda.pydata.org/docs/intro.html>`__ is the package
manager that Anaconda is built on. So when working with Anaconda, you
use the conda package manager for installing conda packages. ``pip``
can also be used with conda, but it's best to use use conda if you can.

We have made sure that every package you need is available for conda.

Setting up
..........

Install: `Anaconda <https://www.continuum.io/downloads>`__

or alternatively: `Miniconda <http://conda.pydata.org/miniconda.html>`__

Once you have either Anaconda or Miniconda installed, the rest of the
instructions should be the same.

Update your (new) system
........................

Once you have Anaconda or miniconda installed, you should start by
getting everything up to date, sometimes packages have been updated
since the installer was built.

Enter the following on the command-line::

    > conda update conda

will update the conda package manager itself (and its dependencies)

Setting up anaconda.org channels
................................

`anaconda.org <http://anaconda.org>`__ is a web service where people can
host conda packages for download. The way this is done is through
anaconda "channels", which can be thought of simply as places on
anaconda.org where collections of packages are bundled together by the
people hosting them.

Many of the dependencies that ``py_gnome`` requires come out of the box
with Anaconda (or the conda "defaults" channel), but a few don't.

The "conda-forge" project:

https://conda-forge.github.io/

Is a community project to build a wide variety of packages for conda --
it supports most of what PyGNOME needs.

However, there a few NOAA-specific packages that are not (yet) on conda-forge,
so we have set up
`our own anaconda channel <https://anaconda.org/noaa-orr-erd>`__
where we put various packages needed for ``py_gnome``.

Adding extra channels to conda:
...............................

Add the NOAA-ORR-ERD channel::

    > conda config --add channels NOAA-ORR-ERD

Add the conda-forge channel::

    > conda config --add channels conda-forge

When you add a channel to conda, it puts it at the top of the list.
So now when you install a package, conda will first look in conda-forge,
then NOAA-ORR-ERD, and then in the default channel.
This order should work well for PyGNOME.
Be sure to add the channels in the order we specify.
You can see what channels you have with::

    > conda config --get channels

It should return something like this::

    --add channels 'defaults'   # lowest priority
    --add channels 'NOAA-ORR-ERD'
    --add channels 'conda-forge'   # highest priority

In that order -- the order is important

conda environments
------------------

The conda system supports isolated "environments" that can be used to
maintain different versions of various packages. For more information
see:

http://conda.pydata.org/docs/using/envs.html

IF you are only going to use Python / Anaconda for PyGNOME, then you
can ignore this. However, if you are using Anaconda for other projects
that might depend on specific versions of specific libraries
(like numpy, scipy, etc), then you may want create an environment
for PyGNOME::

    conda create --name gnome python=2

This will create an environment called "gnome" with Python2 and the core
pieces you need to run conda. To use that environment, you activate it
with::

    source activate gnome

or on Windows::

    activate gnome

and when you are done, you can deactivate it with::

    source deactivate

(or just ``deactivate`` on Windows)

After activating the environment, you can proceed with these instructions,
and all the packages ``py_gnome`` needs will be installed into that environment
and kept separate from your main Anaconda install.

You will need to activate the environment any time you want to work with
``py_gnome`` in the future

**NOTE:** Again, if you are only using Python / conda for GNOME, it is not necessary to deal with the complications of environments.


Download GNOME
--------------

At this point you will need the files from the ``py_gnome`` sources. If you
have not downloaded it yet, it is available here:

https://github.com/NOAA-ORR-ERD/PyGnome

You can either download a zip file of all the sources and unpack it, or
you can "clone" the git repository. If you clone the repository, you will
be able to update the code with the latest version with a simple command,
rather than having to re-download the whole package.

Downloading a single release
----------------------------

zip and tar archives of the PyGnome source code can be found here:

https://github.com/NOAA-ORR-ERD/PyGnome/releases

This will get you the entire source archive of a given release, which is a fine way to work with PyGnome. However, if you want to be able to quickly include changes as we update the code, you may want to work with a git "clone" of the source code instead.

Cloning the PyGNOME git repository
----------------------------------


git
...

You will need a git client:

Linux:
  it should be available from your package manager::

    $ apt_get install git
    or
    $ yum install git

OS-X:
  git comes with the XCode command line tools:

  http://osxdaily.com/2014/02/12/install-command-line-tools-mac-os-x/

Windows:
  the "official" git for Windows installer is a good bet:

  https://git-for-windows.github.io/

Once you have the client, it's as easy as::

  $ git clone https://github.com/NOAA-ORR-ERD/PyGnome.git

This will create a PyGnome directory with all the code in it.


Dependencies
------------

The conda packages required by ``py_gnome`` are listed in the file
``conda_requirements.txt`` in the top directory of the project.

To install all the packages ``py_gnome`` needs:

::

    > cd PyGnome  # or wherever you put the PyGnome project
    > conda install --file conda_requirements.txt


This should install all the packages required by ``py_gnome``.

(make sure you are in the correct conda environment, and you have the
conda-forge and NOAA-ORR-ERD channels enabled.)

If installing conda_requirements.txt fails:
...........................................

If you get an error about a particular package not being able to be installed, then conda will not install ANY of the packages in the file. We try hard to make sure everything is available on one of the channels we recommend. If however, a package of that particular version is missing, you can try:

Edit the conda_requirements.txt file and comment out the offending package by putting a "#" at the start of the line::

    ...
    scipy>=0.17
    py_gd>=0.1.5
    # libgd>=2.2.2
    gsw>=3.0.3
    ...

That will disable that particular package, and hopefully everything else will install.

YOu can then try installing the offending package without a version specification::

    > conda install libgd

And it may work for you.


The Oil Library
---------------

If you want to use py_gnome with "real oil", rather than inert particles, you will need NOAA's OilLibrary package:

https://github.com/NOAA-ORR-ERD/OilLibrary

This is under active development along with ``py_gnome``, so you are best off downloading the sources from gitHub and installing it from source -- similar to ``py_gnome``. Though the lated releases of each should be compatible.

cloning the repository ::

  $ git clone https://github.com/NOAA-ORR-ERD/OilLibrary.git

Installing the package::

  $ cd OilLibrary/
  $ python setup.py install

(you may get a lot of INFO and WARNNG messages as the oil library database is built)

Testing the oil_library install::

  $ py.test

(you may need to ``conda install pytest`` to get that command)

you should see something like::

  ================================= 87 passed in 0.88 seconds ===============================

when done.

Compilers
---------

To build ``py_gnome``, you will need a C/C++ compiler. The procedure for
getting the compiler tools varies with the platform you are on.

OS-X
....

The system compiler for OS-X is XCode. It can be installed from the App
Store.

*Note: it is a HUGE download.*

[you may be able to install only the command line tools -- Apple keeps changing its mind]

After installing XCode, you still need to install the "Command Line
Tools".  XCode includes a new "Downloads" preference pane to install
optional components such as command line tools, and previous iOS
Simulators.

**NOTE:** This may be slightly different on different versions of OS-X
and XCode -- google is your friend.

To install the XCode command line tools: - Start XCode from the
launchpad - Click the "XCode" dropdown menu button in the top left of
the screen near the Apple logo - Click "Preferences", then click
"Downloads". - Command Line Tools should be one of the downloadable
items, and there should be an install button for that item. Click to
install.

Once the command line tools are installed, you should be able to build
``py_gnome`` as described below.

Windows
.......

For compiling python extensions on Windows with python2.7 it is best to use the

`Microsoft Visual C++ Compiler for Python
2.7 <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`__,

which is freely downloadable.

Linux
.....

Linux uses the GNU gcc compiler. If it is not already installed on your
system, use your system package manager to get it.

-  apt for Ubuntu and Linux Mint
-  rpm for Red Hat
-  dpkg for Debian
-  yum for CentOS
-  ??? for other distros

Building ``py_gnome``
.....................

At this point you should at last have all the necessary third-party
tools in place.

Right now, it is probably best to build ``py_gnome`` from source. And it is
probably best to build a "develop" target for your ``py_gnome`` package if
you plan on developing or debugging the ``py_gnome`` source code. (or updating the source code from gitHub)

Building the "develop" target allows changes in the package python code
(or source control updates), to take place immediately.

Of course if you plan on simply using the package, you may certainly
build with the "install" target. Just keep in mind that any updates to
the project will need to be rebuilt and re-installed in order for
changes to take effect.

OS-X Note:
..........

Anaconda does some strange things with system libraries and linking on
OS-X, so we have a high level script that will build and re-link the
libs for you.

So to build ``py_gnome`` on OS-X::

    $ cd py_gnome
    $ ./build_anaconda.sh develop

or:

    $ ./build_anaconda.sh install


Other platforms
...............

As far as we know, the linking issues encountered on OS-X don't exist
for other platforms, so you can build directly. There are a number of
options for building::

    > python setup.py develop

builds and installs the gnome module development target

::

    > python setup.py cleanall

cleans files generated by the build as well as files auto-generated by
cython. It is a good idea to run ``cleanall`` after updating from the
gitHub repo -- particularly if strange errors are occurring.

You will need to re-run ``develop`` or ``install`` after running ``cleanall``

Testing ``py_gnome``
--------------------

We have an extensive set of unit and functional tests to make sure that
``py_gnome`` is working properly.

To run the tests::

    > cd PyGnome/py_gnome/tests/unit_tests
    > py.test

and if those pass, you can run::

    > py.test --runslow

which will run some more tests, some of which take a while to run.

Note that the tests will try to auto-download some data files. If you
are not on the internet, this will fail. And of course if you have a
slow connection, these files could take a while to download. Once the
tests are run once, the downloaded files are cached for future test
runs.

Running scripts
---------------

There are some example scripts in the ``scripts`` directory. You should be able to run these scripts in the same way as any Python script (with an IDE such as Spyder or PyCharm, or at the command line).

To run a script on the command line::

    cd py_gnome/scripts
    cd script_boston

as an example -- there are quite a few.

If you are using a conda environment:

    source activate gnome

or on Windows::

    activate gnome

Run the script::

    python script_boston.py

Each of the scripts exercised different features of py_gnome -- they are hopefully well commented to see how they work.



