.. include:: ./links.rst

############
Introduction
############

PyGNOME is general particle tracking code written primarily in Python, with some bindings to C++ code for certain functionality. It the core engine of the General NOAA Operational Modeling Environment (GNOME) Suite of tools.

It is used by NOAA as the engine behind |webgnome|  -- an publicly availabe Web-based interface customized for oil spill modeling.

PyGNOME can be used to write your own scripts for oil spill and general particle tracking applications.

PyGNOME is a flexible particle tracking framework that can be customized with alternative algorithms for oil fate and transport processes, or virtually any oceanic particle tracking application. It includes a full set of algorithms to support oil spill modeling, but is also used for a variety of applicaitons, including soluble chemicals in water, tracking of marine debris, Larvae transport, etc.

History
-------

GNOME began development in the late 1990s, as the successor to NOAA's original oil spill model, the On Scene Spill Model (OSSM).
It was built using an object oriented approach, written in C++ , with a dual platform GUI, originally for
Windows32 and MacOS.
The latest version is written in Python with new code, as well as  wrappers around the same computational code used in the desktop GUI version.
The current version is written in a combination of Python, Cython and C++.










