.. include:: ./links.rst

Introduction
============

PyGNOME is a set of python bindings (and utilities) to the General NOAA
Operational Modeling Environment (GNOME). It is used by NOAA to build the
new |webgnome| interface to the model and to do an assortment of batch processing
and testing. It can be used to write your own scripts to set up and run scenarios.
It can also be modified to include alternative algorithms or customized 
oil fate and transport processes.

History
-------

GNOME began development in the late 1990s, as the successor to NOAA's original
oil spill model, the On Scene Spill Model (OSSM). It was built using an object
oriented approach, written in C++ , with a dual platform GUI, originally for
Windows32 and MacOS. The newer Python bindings in PyGNOME are a combination of wrappers around
the same computational code used in the desktop GUI version and new code
written in a combination of Python, Cython and C++.










