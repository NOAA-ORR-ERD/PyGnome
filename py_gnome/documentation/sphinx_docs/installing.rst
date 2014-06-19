Installing
==========

PyGnome consists of compiled C++ code (`libgnome`), compiled Cython code (`*.pyx` files), and compiled python extensions.
It can be installed either from source, in which case you'll need an appropriate compiler, or from binaries provided by NOAA

Dependencies
------------
pyGNOME depends on a number of third party packages -- the complete list can be found in the `requirements.txt` file.
This file can also be processed by the `pip` package installer, which (if all goes well) will auto install all the dependencies.


Binary Installation
-------------------

(to be filled in when we have binaries to provide!)

Building
--------

Ideally, it is as simple as::

    $ python setup.py build
    $ python setup.py install

or::

    $ python setup.py develop

(develop mode installs links to the code, rather than copying the code into python's site-packages.
 It is helpful if you want to update the code, and have the new version run right away.)

(MORE HERE!!)

Mac OS-X
........

Mac-specific instructions here

Windows
.......

Windows-specific instructions here

Linux
.....

Linux-specific instructions here







