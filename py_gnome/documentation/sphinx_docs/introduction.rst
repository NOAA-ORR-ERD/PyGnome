Introduction
=====================

pyGNOME is a set of python bindings (and utilities) to the General NOAA Operational Modeling Environment (GNOME). It is used by NOAA to build the webGNOME interface to the model, and to do an assortment of batch processing and testing. It can be used to write your own customized models using the GNOME code base.

History
----------------------

GNOME began development in the late 1990s, as the successor to NOAA's original oil spill model, the On Scene Spill Model (OSSM). It was built using an object oriented approach, written in C++ , with a dual platform GUI, originally for Windows32 and MacOS. The GUI has been ported to Mac OS-X, and a new web interface is underway. The Python bindings are a combination of wrappers around the same computational code used in the desktop GUI version, and new code written in a combination of Python, Cython and C++.











