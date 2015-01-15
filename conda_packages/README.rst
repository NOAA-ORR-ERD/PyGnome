Third Party Conda Packages
===========================

This dir has code, etc, that helps us set up Anaconda for use with gnome.

There are a number of packages that py_gnome needs that are not available
(at the moment) from conda, but are on PyPI. For teh most part, you can use:
``pip install`` to install PyPI packages in an Anaconda python environemnt,
but then the conda package manager does know about them, so things get a bit
confused.

So this system help you setup up a binsar repo with conda packages for various
dependencies needed for PyGNOME.

If you jsut need to run py_gnome, you should be abel to simpley add the:

NOAA-ORR-ERD

binstar channel to your conda configurationa nd be done. You only need this
stuff if you want to update dependency pacakges, etc.

Building, etc.
-------------

The ``conda-pip.py`` script will generate and build conda packages from PyPI,
and then upload them to binstar. The sinmiplest way to run it is::

  ./conda-pip.py all

and it should do it all :-)

You can also have it do a single pacakge::

  ./conda-pip.py transaction

Note that the script is set tp upload to the NOAA-ORR-ERD binstar organization channel, which you will need permisiions to oupload to.

You can change this by editing the ``BINSTAR_USER`` variable in the script.






