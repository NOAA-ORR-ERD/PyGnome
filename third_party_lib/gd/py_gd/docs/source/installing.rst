Installing
=====================

py_gd is written in Cython (http://www.cython.org) and depends on the gd library (http://libgd.bitbucket.org/)

You need libgd compiled (with support for image formats you want..)


Dependencies
---------------------

Cython (www.cython.org): required to re-build the wrappers.

numpy (www.numpy.org): required at build tiem for the headers, etc, and optionally at run time.

pyGNOME depends on libgd (http://libgd.bitbucket.org/) and whatever other libs libgd itself depends on.


Binary Installation
--------------------
(to be filled in when we have binaries to provide!)

Building
---------------------

Ideally, it is as simple as::

    $ python setup.py build
    $ python setup.py install
or::
    $ python setup.py develop

(develop mode installs links to the code, rather than copying the code into python's site-packages -- it is helpful if you want to be updating the code, and have the new version run right away.)

(MORE HERE!!)

Mac OS-X
............

Mac-specifc intructions here

Windows
............

Windows-specifc intructions here

Linux
............

Linux-specifc intructions here







