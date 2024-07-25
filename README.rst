.. image:: graphics/new_gnome_icon/GNOME_logo_450px-wide.png
   :alt: GNOME Logo
   :align: center

#######
PyGNOME
#######


Introduction
============

**GNOME** (General NOAA Operational Modeling Environment) is a modeling tool
developed by the National Oceanic and Atmospheric Administration (**NOAA**),
Office of Response and Restoration (**ORR**), Emergency Response Division.

It is designed to support oil and other hazardous material spills in the
coastal environment, and is also a full featured, flexible particle tracking
system, that can be used for other oceanographic transport applications,
such as fish larvae, marine debris, etc.

PyGNOME is a python package that encapsulates GNOME's functionality.

Disclaimer:
-----------

**This code is under active development**

* It should not be considered an officially endorsed NOAA product.
* Output produced by this code should not be considered endorsed by NOAA.

Documentation
=============

`Project Documentation <https://gnome.orr.noaa.gov/doc/pygnome/index.html>`_

`FAQ <https://github.com/NOAA-ORR-ERD/GNOME2/wiki/FAQ---Troubleshoot>`_


Installation
============

We have put some effort into making this package reasonably easy,
or at least possible, to install on a few different computing platforms:

 - OS-X
 - Windows
 - Linux (tested on CentOS 7)

This package contains modules written in C/C++, and they must be
compiled for this package to function, and we primarily use the Anaconda
distribution of Python for installation.  Anaconda is built primarily for
scientific, engineering, and math applications.
It is now the only supported way to get set up to use ``PyGNOME``,
and it is used in our development and testing process.

`Install using Anaconda <./Installing.rst>`_


The WebGNOME Interface:
=======================

Scripting is the most featurefull way to access PyGNOME's capabilities.
However we have developed a system that allows a user to create and run
PyGNOME models from a web browser.

There is a publicly available instance of WebGNOME at:

https://gnome.orr.noaa.gov

If you want to run your own instance of WebGNOME, the code is in the
following projects:

- `WebGnomeAPI <https://github.com/NOAA-ORR-ERD/WebGnomeAPI>`_:
  A web server that implements the PyGNOME interface
- `WebGnomeClient <https://github.com/NOAA-ORR-ERD/WebGnomeClient>`_:
  A Web application for setting up and running PyGNOME models

**Fair Warning:**

The WebGNOME system is under active development, and by its very nature
does not expose the full capabilities of PyGNOME.
