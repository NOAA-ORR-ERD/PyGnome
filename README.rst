
.. image:: graphics/new_gnome_icon/GNOME_logo_450px-wide.png
   :alt: GNOME Logo
   :align: center

#######
PyGnome
#######

Documentation
=============

`Project Documentation <https://gnome.orr.noaa.gov/doc/pygnome/index.html>`_

`FAQ <https://github.com/NOAA-ORR-ERD/GNOME2/wiki/FAQ---Troubleshoot>`_

Introduction
============

**GNOME** (General NOAA Operational Modeling Environment) is a modeling tool
developed by the National Oceanic and Atmospheric Administration (**NOAA**),
Office of Response and Restoration (**ORR**), Emergency Response Division.

It is designed to support oil and other hazardous material spills in the coastal environment.

And this is a python package that encapsulates GNOME's functionality.

Disclaimer:
-----------

**This code is under active development**

* It should not be considered an officially endorsed NOAA product.
* Output produced by this code should not be considered endorsed by NOAA.

For the "operational" version, please see our main web site:

http://response.restoration.noaa.gov/gnome


Installation
============

We have put some effort into making this package reasonably easy,
or at least possible, to install in a number of ways on a few different computing platforms:

 - OS-X
 - Windows
 - Linux (tested on CentOS 7)

The "tricky" part is installing the dependencies: details in the following:

`Install using conda <./Installing.rst>`_

The conda package manager is built primarily for scientific, engineering,
and math applications it is the easiest way to get set up to use ``py_gnome``, and it what is used in our development and testing process.

If you don't want to / can't use conda -- here are some notes on that:

`Installing Without Conda <./Install_without_conda.rst>`_


The WebGNOME Interface:
=======================

Scripting is of course the most featureful way to access PyGnome's capabilities.
However we do have a few projects in development that allow a user to
create and run PyGnome models from a web browser.

There is a publicly available instance of WebGNOME at:

https://gnome.orr.noaa.gov

If you are curious, you can check out the following projects:

- `WebGnomeAPI <https://github.com/NOAA-ORR-ERD/WebGnomeAPI>`_:
  A web server that implements the PyGnome interface
- `WebGnomeClient <https://github.com/NOAA-ORR-ERD/WebGnomeClient>`_:
  A Web application for setting up and running PyGnome models

Fair Warning:

These projects are under active development, and by their very nature do not implement the full capabilities of PyGnome.

