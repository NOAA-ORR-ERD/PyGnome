
==================================================================
Building / Installing GNOME with the Anaconda python distribution
==================================================================

Anaconda (https://store.continuum.io/cshop/anaconda/) is a Python distribution that comes with most of the difficult to build and install packages that py_gnome needs. Thus it's a nice target for running GNOME on your own system.

conda
=====

conda (http://conda.pydata.org/docs/intro.html) is the package manager that Anaconda is built on. So when working with Anaconda, you are really using the conda package manager, and installing conda packages.

dependencies
============

Dependencies are listed in the requirements.txt file (and the conda build meta.yaml) file.

But to get the whole setup, the ``conda_requirements.txt`` file has a full dump of a conda environment with all the dependencies. You can set up a conda environment for py_gnome with::

  conda create -n py_gnome --file conda_packages.txt 

Many of the dependencies that py_gnome requires come out of the box with Anaconda. But a few don't so we've set up a "channel" on binstar, to supply the extra packages. Once that channel is setup, conda should be able to find everything it needs.

Setting up
===========

Once you have Anaconda installed, you should start by getting everything up to date::

  conda update anaconda

And install the binstar client::

  conda install binstar

(maybe only for building / uploading packages to binstar?)

Setting up binstar
-------------------

To add the NOAA-ORR-ERD binstar channel to Anaconda::

  conda config --add channels http://conda.binstar.org/NOAA-ORR-ERD

We also rely on the channel of one of the core conda developers:

  conda config --add channels http://conda.binstar.org/asmeurer


Installing GNOME, etc.
----------------------

It should now be as easy as::
 
  conda install py_gnome

This will install the latest version that we have up on binstar


Building the oil_library and py_gnome
======================================

If you want to work with the bleeding edge version of py_gnome, and/or contribute to development, then you need to set up your conda environment with all the dependencies, then build py_gnome and the oil library yourself:

Getting all the dependencies
----------------------------

conda can install a list of dependencies from a file, so you can simply do:



Building with conda should be straightforward. In the oil_library directory::

  conda build oil_library_conda_recipe/


     








