Introduction
============

pyGNOME is a set of python bindings (and utilities) to the General NOAA
Operational Modeling Environment (GNOME). It is used by NOAA to build the
webGNOME interface to the model, and to do an assortment of batch processing
and testing. It can be used to write your own customized models using the GNOME
code base.

History
-------

GNOME began development in the late 1990s, as the successor to NOAA's original
oil spill model, the On Scene Spill Model (OSSM). It was built using an object
oriented approach, written in C++ , with a dual platform GUI, originally for
Windows32 and MacOS. The GUI has been ported to Mac OS-X, and a new web
interface is underway. The Python bindings are a combination of wrappers around
the same computational code used in the desktop GUI version, and new code
written in a combination of Python, Cython and C++.

What It Does
------

It is fundamentally a Lagrangian element (particle tracking) model -- the oil or
other substance is represented as Lagrangian elements (LEs), or particles, in 
the model, with their movement and properties tracked over time.

During the model these elements are acted upon by different natural processes.
GNOME uses "Mover" objects to encompass any processes that affect the position
of an element in 3D space. "Weatherer" objects encompass most other effects.

GNOME defines the environmental conditions of the model using
'Environment Objects', which are designed to represent data flexibly from a 
large variety of sources. Simple scalars, time series data, or full 4D
environment data in netCDF format are some examples of what GNOME can intake.

"Map" objects handle all collision-related effects such as particle beaching.
Generally if there is a situation where particles can collide with barriers
in the environment, the Map would have purview.

"Spill" objects encapsulate the complexities of 'Where, What, When, and How'
for releasing oil into the model.

Lastly, "Outputter" objects handle all aspects of exporting data from the
model in whatever format they are designed to handle.







